//! FusedOpRegistry — kernel-side payload. Phase 7.6 step 1 (skeleton).
//!
//! Architecture v1.0 splits the fused-op registry across two crates:
//! - graph-side metadata in `fuel-graph::registry` (id, name, family,
//!   pattern, decompose, backward, shape/dtype rules);
//! - kernel-side payload here in `fuel-storage::fused` ([`BackendImpl`],
//!   [`CostEstimate`], [`PrecisionGuarantee`], [`KernelRevisionHash`]).
//!
//! The split exists because [`KernelRef`] lives in fuel-storage and
//! fuel-graph cannot depend on fuel-storage (the dependency arrow goes
//! the other way). Joining the two halves is by [`fuel_graph::registry::FusedOpId`]
//! at runtime: the optimizer reads the metadata-side entry to reason
//! about decomposition and shape, then asks the kernel-side
//! [`FusedKernelRegistry`] for the per-backend [`BackendImpl`] when it
//! needs to pre-resolve a `KernelRef`.
//!
//! ## Status (step 1)
//!
//! Types only. No callers; no behavior change. Subsequent steps:
//! - Step 3: register the SoftmaxLastDim CPU `BackendImpl` and dispatch
//!   `Op::Fused(SOFTMAX_LAST_DIM, _)` through it from the executor.
//! - Step 6-9: extend per-backend coverage, populate `PrecisionGuarantee`
//!   and `cost`, and migrate the binding-table lookup off the executor's
//!   hot path.

use crate::kernel::{KernelCaps, KernelRef};
use fuel_core_types::{DType, Shape, backend::BackendCapabilities, probe::BackendId};
use fuel_graph::registry::{FusedOpId, FusedOpParams};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Per-backend kernel implementation for one fused op. The optimizer
/// reads this to (1) pre-resolve [`KernelRef`] for nodes it places on
/// this backend, (2) score routes against [`CostEstimate`], and
/// (3) admit candidates against the per-route tolerance budget via
/// [`PrecisionGuarantee`].
///
/// Function-pointer composition (no trait-object indirection) — the
/// registry stores [`BackendImpl`] values inline, the executor calls
/// the function pointer directly.
///
/// A single `(FusedOpId, BackendId)` decision point may have multiple
/// registered `BackendImpl`s — one per dtype tuple, and potentially
/// multiple alternatives at the same dtype (e.g. cuBLAS bias-epilogue
/// vs CUTLASS bias-epilogue for `(FUSED_LINEAR, [BF16,BF16,BF16,BF16],
/// Cuda)`). The route picker filters by [`Self::dtypes`] and ranks the
/// remaining alternatives by cost + precision.
#[derive(Copy, Clone)]
pub struct BackendImpl {
    /// Dispatch wrapper for this backend's kernel for this fused op.
    /// Same `KernelRef` signature as primitive-op kernels.
    pub kernel: KernelRef,
    /// Dtype tuple this kernel is registered for. Convention mirrors
    /// the binding-table key shape: typically `[input1, input2, ...,
    /// output]` per the existing `register(table, op, &[dtype; N],
    /// backend, kernel)` call sites. Using `&'static [DType]` keeps
    /// `BackendImpl` `Copy` and lets registrations declare dtypes
    /// inline as `&[F32, F32, F32, F32]` literals.
    pub dtypes: &'static [DType],
    /// Cost-estimate function. Given the input shapes, the
    /// per-instance fused-op params, and the backend's capabilities,
    /// returns a [`CostEstimate`] used for placement and route ranking.
    pub cost: fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate,
    /// Numerical precision properties of this kernel.
    pub precision: PrecisionGuarantee,
    /// Layout / capability flags (e.g. strided_input).
    pub caps: KernelCaps,
    /// Revision hash — opaque identifier for the source-version of this
    /// kernel. Persisted optimization caches use this to detect kernel
    /// drift and invalidate stale cache entries (see
    /// `docs/architecture/11-persistence.md`).
    pub revision: KernelRevisionHash,
}

/// Coarse cost model for one kernel invocation. Layer-1 of the
/// architecture's cost-model tower (FLOP counts + bandwidth + launch
/// overhead). Layer-2 — empirical refinement from per-deployment
/// telemetry — composes on top of these without changing this shape.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CostEstimate {
    /// Compute pressure — number of floating-point operations.
    pub flops: u64,
    /// Bandwidth pressure — bytes moved through device memory hierarchy
    /// for this kernel's inputs + outputs (excluding cache hits).
    pub bytes_moved: u64,
    /// Fixed launch overhead. CPU kernels measure this in tens of ns;
    /// GPU launches in low microseconds.
    pub kernel_overhead_ns: u32,
}

/// What this kernel guarantees about its numerical behavior.
///
/// Replaces the binary-flag OracleGrade concept that pre-architecture
/// drafts used; per architecture v1.0 every kernel registration carries
/// a structured precision statement so the optimizer can reason about
/// tolerance budgets and pick comparators for calibration.
///
/// `bit_stable_on_same_hardware` is the strongest property. The
/// always-built backend (fuel-cpu-backend by convention) commits to
/// providing at least one `bit_stable_on_same_hardware: true` kernel
/// per primitive op as the architecture v1.0 coverage commitment;
/// step 7 enforces this via a CI lint.
///
/// The Optional fields encode the bound's flavor (ULP / relative /
/// absolute). Multiple may be present; the optimizer takes the
/// intersection (a budget is admissible only if every populated bound
/// satisfies it). Absent bounds mean "this kernel makes no claim
/// about that flavor."
#[derive(Copy, Clone, Debug)]
pub struct PrecisionGuarantee {
    /// True iff this kernel produces bit-identical output for
    /// bit-identical inputs on the same hardware (no nondeterminism
    /// from kernel scheduling, atomic ordering, etc.).
    pub bit_stable_on_same_hardware: bool,
    /// Maximum unit-in-last-place error vs the IEEE-754 correctly-
    /// rounded result. Tighter than max_relative for low-magnitude
    /// values; many vendor math libraries quote ULPs.
    pub max_ulp: Option<u32>,
    /// Maximum relative error: `|out - ref| / max(|ref|, eps)`.
    pub max_relative: Option<f64>,
    /// Maximum absolute error: `|out - ref|`.
    pub max_absolute: Option<f64>,
    /// Free-text qualifier — for audited claims, this carries the
    /// audit reason (vendor citation, scheduler-dependence note,
    /// implementation hint). For the [`UNAUDITED`] sentinel, this
    /// is a fixed placeholder string the coverage lint matches
    /// against to distinguish unaudited placeholders from audited
    /// claims that happen to have no static bound. **If you change
    /// `UNAUDITED.notes`, the lint's detector picks up the new value
    /// automatically — it reads `PrecisionGuarantee::UNAUDITED.notes`
    /// at test time, not a hardcoded string.**
    ///
    /// [`UNAUDITED`]: PrecisionGuarantee::UNAUDITED
    pub notes: &'static str,
}

impl PrecisionGuarantee {
    /// "Reference" guarantee: bit-stable on same hardware, no error
    /// claim above that. Used by reference-grade CPU kernels that
    /// commit to deterministic IEEE-754 evaluation.
    pub const REFERENCE: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: true,
        max_ulp: Some(0),
        max_relative: Some(0.0),
        max_absolute: Some(0.0),
        notes: "Reference IEEE-754 evaluation; bit-identical re-run.",
    };

    /// Phase 7.6 step 7b: shared precision guarantee for the
    /// always-built CPU backend's **primitive** kernels — elementwise
    /// unary/binary, reductions, matmul, casts, comparisons, indexing,
    /// scalar ops, transcendentals (Exp/Log/Sin/Cos/Tanh/Sigmoid/
    /// Gelu/Silu), etc. All claim:
    ///
    /// - `bit_stable_on_same_hardware: true` (deterministic iteration
    ///   order; no atomic FP adds; sequential nested loops)
    /// - F32 accumulator for BF16/F16 inputs, native F32 / F64 / U32
    ///   for matching dtypes
    /// - No specific ULP / relative / absolute bound claimed at this
    ///   layer — the empirical calibration framework in step 8
    ///   populates per-op-per-shape bounds via reference comparisons
    ///
    /// This is the default value bulk-applied by
    /// [`crate::kernel::KernelBindingTable::fill_unset_cpu_precision`]
    /// at the end of `register_cpu_kernels`. Sites that need a
    /// different claim must call
    /// [`crate::kernel::KernelBindingTable::register_with_precision`]
    /// explicitly with the weaker guarantee — those don't get
    /// bulk-overwritten because the fill only touches UNAUDITED
    /// entries.
    pub const PRIMITIVE_DETERMINISTIC_CPU: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: true,
        max_ulp: None,
        max_relative: None,
        max_absolute: None,
        notes: "fuel-cpu-backend primitive kernel family: deterministic \
                nested-loop iteration order; F32 accumulator for \
                half-precision inputs; bit-identical same-hardware \
                re-run. Per-op-per-shape ULP / relative bounds land \
                with the step-8 empirical calibration framework.",
    };

    /// Default placeholder for kernels whose precision hasn't been
    /// audited yet. Identified by the coverage lint via notes-
    /// equality against `PrecisionGuarantee::UNAUDITED.notes` —
    /// the only way a registration ends up with this specific
    /// notes string is by using the literal `UNAUDITED` const
    /// (the default in unannotated `register(...)` / `register_with_caps(...)`
    /// calls).
    ///
    /// **For kernels audited and concluded "no static bound
    /// applies"** (e.g. Vulkan subgroup reductions where FADD
    /// order is scheduler-determined per dispatch), use
    /// [`none`] instead. It returns a structurally-similar
    /// PrecisionGuarantee but with the audit reason in `notes`,
    /// captured at the registration site. The lint accepts any
    /// notes value other than UNAUDITED's specific text.
    ///
    /// [`none`]: PrecisionGuarantee::none
    pub const UNAUDITED: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: false,
        max_ulp: None,
        max_relative: None,
        max_absolute: None,
        notes: "PrecisionGuarantee::UNAUDITED — default placeholder for \
                kernels whose precision hasn't been audited yet. Coverage \
                lints flag these. Sites must either annotate via \
                `register_with_precision` with a real claim or use \
                `PrecisionGuarantee::none(reason)` if the audit \
                concluded no static bound applies.",
    };

    /// Constructor for the "audited, no static guarantee could be
    /// established" conclusion. Used by Vulkan subgroup reductions
    /// (where FADD order is scheduler-determined per dispatch) and
    /// similar kernels whose numerical character can't be
    /// summarized by static ULP / relative / absolute bounds.
    ///
    /// The `reason` is captured in `notes`, so the audit reasoning
    /// lives at the registration site — visible in code review,
    /// not buried in a separate lint allowlist file. Any `reason`
    /// other than `UNAUDITED.notes` makes the resulting value pass
    /// the coverage lint.
    ///
    /// Distinguished from [`UNAUDITED`] by `notes` content alone
    /// (the value-field shape is identical). The lint uses
    /// `precision.notes == PrecisionGuarantee::UNAUDITED.notes` as
    /// the unaudited detector — robust to UNAUDITED's notes text
    /// drifting because it reads from the const itself.
    ///
    /// [`UNAUDITED`]: PrecisionGuarantee::UNAUDITED
    pub const fn none(reason: &'static str) -> PrecisionGuarantee {
        PrecisionGuarantee {
            bit_stable_on_same_hardware: false,
            max_ulp: None,
            max_relative: None,
            max_absolute: None,
            notes: reason,
        }
    }
}

/// Opaque revision hash of a registered kernel. Persisted optimization
/// caches read this to detect kernel drift between cache build and
/// cache load (see `docs/architecture/11-persistence.md`). Computed
/// from kernel source + version metadata at registration time; step 9
/// fills in the actual hashing function alongside the binding-table
/// planning-time refactor.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelRevisionHash(pub u64);

impl KernelRevisionHash {
    /// Sentinel meaning "no revision tracked yet." Used by step-1-shipped
    /// `BackendImpl` registrations until step 9 wires real hashing.
    pub const UNTRACKED: KernelRevisionHash = KernelRevisionHash(0);
}

/// Inline capacity for per-fused-op backend lists. SmallVec at 4 fits
/// CPU + CUDA + Vulkan + Metal without spilling to heap; the typical
/// fused op has 1-3 backends with kernels.
type BackendImplList = SmallVec<[(BackendId, BackendImpl); 4]>;

/// Kernel-side registry: `FusedOpId` → list of per-backend
/// [`BackendImpl`]s. Joined to `fuel-graph`'s metadata-side
/// [`fuel_graph::registry::FusedOpRegistry`] by id at runtime.
///
/// Built at process startup, frozen thereafter (architecture v1.0:
/// no runtime extensibility).
#[derive(Default)]
pub struct FusedKernelRegistry {
    by_id: HashMap<FusedOpId, BackendImplList>,
}

impl FusedKernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a [`BackendImpl`] for this `(id, backend)` pair. Always
    /// appends — multiple impls per `(id, backend)` are allowed (per-
    /// dtype registrations, alternative algorithms at the same dtype,
    /// etc.). The route picker filters by [`BackendImpl::dtypes`] and
    /// ranks remaining alternatives at lookup time.
    pub fn register(&mut self, id: FusedOpId, backend: BackendId, impl_: BackendImpl) {
        self.by_id.entry(id).or_default().push((backend, impl_));
    }

    /// Look up the first [`BackendImpl`] registered for `(id, backend)`
    /// regardless of dtypes. Convenience for callers that already know
    /// a single impl exists for the pair. Prefer
    /// [`Self::lookup_by_dtypes`] when multiple per-dtype impls may
    /// share a backend.
    pub fn lookup(&self, id: FusedOpId, backend: BackendId) -> Option<BackendImpl> {
        self.by_id
            .get(&id)
            .and_then(|impls| impls.iter().find(|(b, _)| *b == backend))
            .map(|(_, impl_)| *impl_)
    }

    /// Look up the [`BackendImpl`] registered for `(id, backend,
    /// dtypes)`. Returns `None` when no kernel matches all three; the
    /// optimizer's fallback in that case is to lower the fused op via
    /// the metadata-side `decompose` and run it as primitives on the
    /// backend.
    ///
    /// When multiple impls match (alternatives at the same decision
    /// point — e.g. cuBLAS vs CUTLASS bf16 matmul), the first
    /// registration wins. Step 9's route picker replaces this with a
    /// cost-aware selector.
    pub fn lookup_by_dtypes(
        &self,
        id: FusedOpId,
        backend: BackendId,
        dtypes: &[DType],
    ) -> Option<BackendImpl> {
        self.by_id
            .get(&id)
            .and_then(|impls| {
                impls
                    .iter()
                    .find(|(b, i)| *b == backend && i.dtypes == dtypes)
            })
            .map(|(_, impl_)| *impl_)
    }

    /// All `(BackendId, BackendImpl)` pairs registered for this id.
    /// The optimizer reads this when deciding placement: a fused op
    /// with a CUDA-only kernel limits its placement candidates.
    pub fn impls_for(&self, id: FusedOpId) -> &[(BackendId, BackendImpl)] {
        self.by_id.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of distinct `FusedOpId`s with at least one registered
    /// backend impl.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the registry has any registered impls.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// Register a [`BackendImpl`] into a [`FusedKernelRegistry`] via the
/// step-6 macro shape. Hides the boilerplate of constructing the
/// `BackendImpl` struct so registration sites read as a flat list of
/// `(id, backend, dtypes, kernel, cost, precision)` tuples.
///
/// Caps default to [`KernelCaps::empty()`] and revision to
/// [`KernelRevisionHash::UNTRACKED`]; pass `caps = ...` and
/// `revision = ...` to override.
///
/// # Examples
///
/// ```ignore
/// register_fused!(
///     registry,
///     FusedOps::FUSED_LINEAR,
///     BackendId::Cpu,
///     &[F32, F32, F32, F32],
///     fused_linear_f32_cpu_wrapper,
///     cost = cost_fused_linear_cpu,
///     precision = PrecisionGuarantee::REFERENCE,
/// );
/// ```
#[macro_export]
macro_rules! register_fused {
    (
        $registry:expr,
        $id:expr,
        $backend:expr,
        $dtypes:expr,
        $kernel:expr,
        cost = $cost:expr,
        precision = $precision:expr
        $(, caps = $caps:expr)?
        $(, revision = $revision:expr)?
        $(,)?
    ) => {{
        #[allow(unused_mut, unused_assignments)]
        let mut caps = $crate::kernel::KernelCaps::empty();
        $( caps = $caps; )?
        #[allow(unused_mut, unused_assignments)]
        let mut revision = $crate::fused::KernelRevisionHash::UNTRACKED;
        $( revision = $revision; )?
        $registry.register(
            $id,
            $backend,
            $crate::fused::BackendImpl {
                kernel: $kernel,
                dtypes: $dtypes,
                cost: $cost,
                precision: $precision,
                caps,
                revision,
            },
        );
    }};
}

/// Process-wide default kernel registry: every backend's fused-op
/// `BackendImpl`s registered at startup. Built once on first access
/// via [`std::sync::OnceLock`]; immutable thereafter (architecture
/// v1.0: no runtime extensibility).
///
/// Today's coverage:
/// - `FUSED_LINEAR` × `Cpu` × `{F32, F64, BF16, F16}` — four
///   bit-stable per-dtype impls registered via [`register_default_kernels`].
///
/// Backend-side crates (fuel-cuda-backend, fuel-vulkan-backend) extend
/// this set by either (a) registering during their own startup, or
/// (b) the step-9 binding-table refactor where the route picker pulls
/// from this registry. Today's executors continue to lookup via the
/// per-dtype [`crate::dispatch::KernelBindingTable`]; this registry is
/// the architecture-target shape for CUTLASS / cuBLAS alternative
/// registrations and step-9's pre-resolved `KernelRef` pipeline.
pub fn default_kernel_registry() -> &'static FusedKernelRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<FusedKernelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut r = FusedKernelRegistry::new();
        register_default_kernels(&mut r);
        r
    })
}

/// Populate a [`FusedKernelRegistry`] with the always-built kernels
/// (today: `fuel-cpu-backend` FusedLinear F32/F64/BF16/F16). Called
/// from [`default_kernel_registry`]'s OnceLock initializer; exposed as
/// a free function so backend crates can compose against a custom
/// registry in tests.
pub fn register_default_kernels(r: &mut FusedKernelRegistry) {
    crate::dispatch::register_default_fused_kernels(r);
}

/// Static cost model for `(FUSED_LINEAR, Cpu)` kernels — the conservative
/// FLOP + bandwidth model per architecture v1.0 §"Layer-1 cost model."
/// Identical for F32/F64/BF16/F16 (the kernels accumulate in F32
/// regardless of input dtype); the empirical Layer-2 refinement
/// framework will tighten per-dtype later.
///
/// Shapes: `[a, b, bias]` where `a = [..., M, K]`, `b = [..., K, N]`,
/// `bias = [N]`.
pub fn cost_fused_linear_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 3, "FusedLinear cost: expected 3 input shapes");
    let a_dims = shapes[0].dims();
    let b_dims = shapes[1].dims();
    let rank = a_dims.len();
    if rank < 2 || b_dims.len() < 2 {
        return CostEstimate::default();
    }
    let m = a_dims[rank - 2] as u64;
    let k = a_dims[rank - 1] as u64;
    let n = b_dims[b_dims.len() - 1] as u64;
    let batch: u64 = a_dims[..rank - 2].iter().map(|d| *d as u64).product::<u64>().max(1);
    // FMA counts as 2 FLOPs. Matmul: 2·M·N·K per batch. Bias-add: M·N per batch.
    let mm_flops = 2u64 * m * n * k * batch;
    let bias_flops = m * n * batch;
    // Conservative bandwidth: assume 4 B/elem (F32-equivalent). Empirical
    // refinement adjusts per-dtype; the static layer is intentionally coarse.
    let elems = batch * (m * k + k * n + m * n) + n;
    let bytes_moved = elems * 4;
    CostEstimate {
        flops: mm_flops + bias_flops,
        bytes_moved,
        kernel_overhead_ns: 50, // CPU launch overhead, ballpark
    }
}

/// Precision guarantee shared by every `(FUSED_LINEAR, Cpu, *)` impl
/// in fuel-cpu-backend: deterministic per-element accumulation in F32
/// (BF16/F16 inputs upcast and narrow at the end), bit-identical on
/// same hardware. Bounded error claims (max_ulp / max_relative) depend
/// on the contracting dim K and so stay `None` at this layer — the
/// empirical calibration framework derives them per-shape.
pub const FUSED_LINEAR_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend FusedLinear: deterministic matmul + bias-add; \
            BF16/F16 accumulate in F32 and narrow at end; same-hardware re-run \
            bit-identical.",
};

/// Static cost model for `(FUSED_SOFTMAX_CROSS_ENTROPY, Cpu)` — row-by-row
/// stable log-softmax + gather. FLOPs scale with `n_rows × vocab`
/// (two passes per row: one for max+sum_exp, one for the loss
/// accumulator); memory is `n_rows × vocab` reads plus the targets and
/// the scalar/vector output.
///
/// Shapes: `[logits, targets]` where `logits = [..., V]` (F32) and
/// `targets = [...]` (I64); the per-row reduction count comes from
/// the logits shape's leading dims.
pub fn cost_fused_softmax_cross_entropy_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(
        shapes.len(), 2,
        "FusedSoftmaxCrossEntropy cost: expected 2 input shapes (logits, targets)",
    );
    let logits_dims = shapes[0].dims();
    if logits_dims.is_empty() {
        return CostEstimate::default();
    }
    let vocab = *logits_dims.last().unwrap() as u64;
    let n_rows: u64 = logits_dims[..logits_dims.len() - 1]
        .iter()
        .map(|d| *d as u64)
        .product::<u64>()
        .max(1);
    // Two passes over vocab per row: pass 1 finds max (V compares),
    // pass 2 sums exp(x - max) (V FMAs ≈ 2V FLOPs). Plus one log
    // call. The transcendental `exp` counts as ~10 FLOPs by convention.
    let per_row_flops = vocab + 2 * vocab + 10 * vocab + 10;
    let total_flops = n_rows * per_row_flops;
    // Bandwidth: logits (n_rows × vocab × 4) + targets (n_rows × 8) +
    // output (4 bytes for Mean/Sum, ~n_rows × 4 for None — average to
    // the worst case here for the conservative layer-1 estimate).
    let bytes_moved = n_rows * vocab * 4 + n_rows * 8 + n_rows * 4;
    CostEstimate {
        flops: total_flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(FUSED_SOFTMAX_CROSS_ENTROPY, Cpu, *)` —
/// stable log-sum-exp accumulated in F64, gather + NLL in F64,
/// narrow to F32 at end. Bit-identical re-run on same hardware. The
/// kernel's iteration order is fixed, so the only sources of
/// non-determinism (reduction-order changes, denormal flushing) are
/// excluded by construction.
pub const FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend FusedSoftmaxCrossEntropy: stable log-sum-exp \
            in F64 accumulator; gather + NLL in F64; narrow to F32 at end; \
            same-hardware re-run bit-identical.",
};

/// Static cost model for `(CAUSAL_CONV1D, Cpu)` — depthwise 1-D conv.
/// FLOPs scale with `batch × channels × seq_out × kernel`; bandwidth
/// is `batch × channels × seq_in` (x reads) + `channels × kernel`
/// (weight reads) + `channels` (bias) + `batch × channels × seq_out`
/// (output writes). SiLU adds one transcendental per output element.
///
/// Shapes: `[x, weight, bias]` where `x = [batch, channels, seq_in]`,
/// `weight = [channels, 1, kernel]`, `bias = [channels]`.
pub fn cost_causal_conv1d_cpu(
    shapes: &[Shape],
    params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(
        shapes.len(), 3,
        "CausalConv1d cost: expected 3 input shapes (x, weight, bias)",
    );
    let use_silu = match params {
        FusedOpParams::CausalConv1d { use_silu } => *use_silu,
        _ => return CostEstimate::default(),
    };
    let x_dims = shapes[0].dims();
    let w_dims = shapes[1].dims();
    if x_dims.len() != 3 || w_dims.len() != 3 {
        return CostEstimate::default();
    }
    let batch = x_dims[0] as u64;
    let channels = x_dims[1] as u64;
    let seq_in = x_dims[2] as u64;
    let kernel = w_dims[2] as u64;
    if seq_in < kernel - 1 {
        return CostEstimate::default();
    }
    let seq_out = seq_in - (kernel - 1);
    // 2·kernel FLOPs per output element (FMA = 2). SiLU adds ~10
    // FLOPs per element (transcendental convention).
    let per_out_flops = 2 * kernel + if use_silu { 10 } else { 0 };
    let flops = batch * channels * seq_out * per_out_flops;
    let bytes_moved = batch * channels * seq_in * 4
        + channels * kernel * 4
        + channels * 4
        + batch * channels * seq_out * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(CAUSAL_CONV1D, Cpu, *)` — textbook
/// nested-loop depthwise conv with F32 accumulator; SiLU computed
/// element-wise via `x / (1 + exp(-x))`. Iteration order fixed;
/// bit-identical re-run on same hardware.
pub const CAUSAL_CONV1D_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend CausalConv1d: textbook depthwise nested-loop \
            with F32 accumulator; optional SiLU via x/(1+exp(-x)); \
            same-hardware re-run bit-identical.",
};

/// Static cost model for `(SELECTIVE_SCAN, Cpu)` — Mamba's selective
/// state-space scan. FLOPs scale with `batch · seqlen · dim · dstate`
/// (the inner triple-nested loop). Bandwidth is dominated by reading
/// u/delta/b/c per timestep + writing y.
///
/// Shapes: `[u, delta, a, b, c]`. Per-step cost: ~4·dstate FMAs (exp +
/// mul + add for h update; mul + accumulate for y).
pub fn cost_selective_scan_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(
        shapes.len(), 5,
        "SelectiveScan cost: expected 5 input shapes (u, delta, a, b, c)",
    );
    let u_dims = shapes[0].dims();
    let a_dims = shapes[2].dims();
    if u_dims.len() != 3 || a_dims.len() != 2 {
        return CostEstimate::default();
    }
    let batch = u_dims[0] as u64;
    let seqlen = u_dims[1] as u64;
    let dim = u_dims[2] as u64;
    let dstate = a_dims[1] as u64;
    // ~8 FLOPs per (b, t, i, j) iteration: exp (10), mul (1), mul (1),
    // add (1), mul (1), add (1). Conservative average ~16/iter once
    // exp is amortized.
    let per_iter_flops = 16;
    let flops = batch * seqlen * dim * dstate * per_iter_flops;
    // u + delta + b + c (read per step) + a (read once) + out (written once).
    let bytes_moved = 2 * batch * seqlen * dim * 4
        + 2 * batch * seqlen * dstate * 4
        + dim * dstate * 4
        + batch * seqlen * dim * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(SELECTIVE_SCAN, Cpu, *)` — recurrence
/// accumulated in F64 (h is `Vec<f64>` internally), final y narrowed
/// to F32. Iteration order fixed; bit-identical re-run on same
/// hardware. The F64 accumulator buys ~7 extra decimal digits of
/// headroom vs naive F32 over the per-step state update.
pub const SELECTIVE_SCAN_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend SelectiveScan: state recurrence in F64 \
            accumulator; narrow to F32 on y store; deterministic \
            iteration order; same-hardware re-run bit-identical.",
};

/// Static cost model for `(CONV2D, Cpu)` kernels — FLOP + bandwidth
/// per architecture v1.0 §"Layer-1 cost model."
///
/// Shapes: `[x, weight, (bias)]` where `x = [N, Cin, H, W]`,
/// `weight = [Cout, Cin/groups, Kh, Kw]`, optional `bias = [Cout]`.
/// The per-instance `FusedOpParams::Conv2D { stride, padding, groups }`
/// supply the rest; output spatial dims are recomputed from input
/// shape + params via the standard `(H + 2·pad - Kh) / stride + 1`.
pub fn cost_conv2d_cpu(
    shapes: &[Shape],
    params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert!(
        shapes.len() == 2 || shapes.len() == 3,
        "Conv2D cost: expected 2 or 3 input shapes",
    );
    let (stride, padding, groups) = match params {
        FusedOpParams::Conv2D { stride, padding, groups } => (*stride, *padding, *groups),
        _ => return CostEstimate::default(),
    };
    let x_dims = shapes[0].dims();
    let w_dims = shapes[1].dims();
    if x_dims.len() != 4 || w_dims.len() != 4 || groups == 0 {
        return CostEstimate::default();
    }
    let (n, cin, h_in, w_in) = (
        x_dims[0] as u64, x_dims[1] as u64,
        x_dims[2] as u64, x_dims[3] as u64,
    );
    let (cout, cin_per_g, kh, kw) = (
        w_dims[0] as u64, w_dims[1] as u64,
        w_dims[2] as u64, w_dims[3] as u64,
    );
    let (sh, sw) = (stride.0 as u64, stride.1 as u64);
    let (ph, pw) = (padding.0 as u64, padding.1 as u64);
    if sh == 0 || sw == 0 || kh > h_in + 2 * ph || kw > w_in + 2 * pw {
        return CostEstimate::default();
    }
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    // FMA counts as 2 FLOPs per (Cin/g · Kh · Kw) inner-product step,
    // summed across N · Cout · Hout · Wout output positions. Bias-add
    // contributes one FLOP per output element.
    let conv_flops = 2u64 * n * cout * h_out * w_out * cin_per_g * kh * kw;
    let bias_flops = if shapes.len() == 3 { n * cout * h_out * w_out } else { 0 };
    let elems_in   = n * cin * h_in * w_in;
    let elems_w    = cout * cin_per_g * kh * kw;
    let elems_out  = n * cout * h_out * w_out;
    let elems_bias = if shapes.len() == 3 { cout } else { 0 };
    let bytes_moved = (elems_in + elems_w + elems_out + elems_bias) * 4;
    CostEstimate {
        flops: conv_flops + bias_flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee shared by every `(CONV2D, Cpu, *)` impl in
/// fuel-cpu-backend: textbook nested-loop accumulation in F32 (F64
/// kernels accumulate in F64; BF16/F16 upcast each multiply to F32
/// and narrow at end), bit-identical re-run on same hardware.
/// Same shape as [`FUSED_LINEAR_CPU_PRECISION`] — the conv kernel's
/// numerical character is governed by the same "F32 accumulator,
/// deterministic iteration order" properties.
pub const CONV2D_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend Conv2D: textbook nested-loop \
            cross-correlation; F32 accumulator (F64 for F64 input); \
            BF16/F16 multiply in F32 and narrow at end; same-hardware \
            re-run bit-identical.",
};

// =============================================================================
// Phase 7.6 step 6 — cost functions + PrecisionGuarantees for the
// remaining 8 ops that have CPU byte-level wrappers in
// `fuel-storage::dispatch`. The four backward helpers
// (SoftmaxLastDimBackward / LayerNormLastDimBackward /
// RmsNormLastDimBackward / ReduceMaxToBackward) are not covered
// here — their CPU dispatch flows through the `GraphBackend` trait
// methods in `fuel-graph-executor`, not the byte-level binding
// table. Wrapper conversion + step-6 registration for those is a
// follow-up.
// =============================================================================

/// Shared precision guarantee for the norm/softmax family — softmax,
/// rms_norm, layer_norm. All three CPU kernels use deterministic
/// elementwise + scalar-reduction patterns with F32 accumulator
/// (F64 for F64 input; BF16/F16 multiply in F32 and narrow at end).
/// Same shape as [`FUSED_LINEAR_CPU_PRECISION`] — the numerical
/// character is governed by the same "F32 accumulator, deterministic
/// iteration order" properties.
pub const NORM_FAMILY_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend Softmax/RmsNorm/LayerNorm: deterministic \
            elementwise + per-row reduction; F32 accumulator (F64 for \
            F64 input); BF16/F16 multiply in F32 and narrow at end; \
            same-hardware re-run bit-identical.",
};

/// Cost model for `(SoftmaxLastDim | RmsNormLastDim | LayerNormLastDim,
/// Cpu)` kernels — outer-product structure: `outer × last_dim`
/// elementwise pass + scalar reduction. FLOPs scale linearly with
/// element count, bandwidth dominates for typical token-dim sizes.
///
/// Phase 7.6 step 8 — branches on `FusedOpParams` to use a per-op
/// FLOP/element count: softmax ≈ 5, rms_norm ≈ 4, layer_norm ≈ 7
/// (plus their backwards, which do comparable per-element work).
/// Replaces the step-6 midpoint-of-5 estimate. Both forward and
/// backward variants of each norm use the same FLOP/element count
/// because they touch the data the same way per row.
///
/// Shapes: `[x]` (forward, 1 input) or `[x_or_y, g]` (backward,
/// 2 inputs). Total element count comes from input[0] either way.
pub fn cost_norm_family_cpu(
    shapes: &[Shape],
    params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert!(
        !shapes.is_empty(),
        "Norm-family cost: expected ≥1 input shape",
    );
    let dims = shapes[0].dims();
    if dims.is_empty() {
        return CostEstimate::default();
    }
    let elems: u64 = dims.iter().map(|&d| d as u64).product();
    let flops_per_elem: u64 = match params {
        FusedOpParams::SoftmaxLastDim
        | FusedOpParams::SoftmaxLastDimBackward
            => 5, // max-sub + exp + sum + divide
        FusedOpParams::RmsNormLastDim { .. }
        | FusedOpParams::RmsNormLastDimBackward { .. }
            => 4, // sqr + sum + sqrt + divide
        FusedOpParams::LayerNormLastDim { .. }
        | FusedOpParams::LayerNormLastDimBackward { .. }
            => 7, // mean-sub + sqr + sum + sqrt + divide + center
        _ => 5,   // fallback if a future caller mis-dispatches
    };
    // Backwards have 2 inputs + 1 output (3 element-count touches);
    // forwards have 1 input + 1 output (2 element-count touches).
    let bandwidth_factor: u64 = if shapes.len() >= 2 { 3 } else { 2 };
    CostEstimate {
        flops: flops_per_elem * elems,
        bytes_moved: bandwidth_factor * elems * 4,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(ROPE, Cpu, *)` impls. Rotary position
/// embedding is a rotation in 2D subspaces — multiplies + adds with
/// the cos/sin tables; F32 accumulator semantics match the norm
/// family.
pub const ROPE_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend Rope: per-head rotation `out = x·cos + \
            rotated(x)·sin` with deterministic iteration order. \
            BF16/F16 multiply in F32 and narrow at end; same-hardware \
            re-run bit-identical.",
};

/// Cost model for `(ROPE, Cpu)` — per-element rotation costs 4 FMAs
/// (2 multiplies + 2 adds in two 2D planes).
///
/// Shapes: `[x, cos, sin]` where `x = [..., seq, head_dim]`,
/// `cos = sin = [seq, head_dim]`.
pub fn cost_rope_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 3, "Rope cost: expected 3 input shapes (x, cos, sin)");
    let dims = shapes[0].dims();
    if dims.is_empty() {
        return CostEstimate::default();
    }
    let elems: u64 = dims.iter().map(|&d| d as u64).product();
    // 4 FLOPs per element (2 FMA pairs across the two rotation planes).
    let flops = 4 * elems;
    // Read x + cos/sin tables (small), write x_out.
    let cs_elems: u64 = shapes[1].dims().iter().map(|&d| d as u64).product::<u64>() * 2;
    let bytes_moved = (2 * elems + cs_elems) * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(CONV_TRANSPOSE2D, Cpu, *)` impls.
/// Same structure as Conv2D — textbook nested-loop accumulation in
/// F32 (F64 for F64 input).
pub const CONV_TRANSPOSE2D_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend ConvTranspose2D: textbook scatter-with-stride \
            + nested loops; F32 accumulator (F64 for F64); BF16/F16 \
            multiply in F32 and narrow at end; same-hardware bit-identical.",
};

/// Cost model for `(CONV_TRANSPOSE2D, Cpu)` — FLOP count parallels
/// Conv2D: `2·N·Cout·Hout·Wout·(Cin/g)·Kh·Kw` (each output position
/// accumulates contributions from `Cin/g · Kh · Kw` input positions,
/// reading transposed).
///
/// Shapes: `[x, weight]` where `x = [N, Cin, H, W]`,
/// `weight = [Cin, Cout/groups, Kh, Kw]` (note transposed channel
/// order vs Conv2D).
pub fn cost_conv_transpose2d_cpu(
    shapes: &[Shape],
    params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 2, "ConvTranspose2D cost: expected 2 input shapes");
    let (stride, padding, output_padding, dilation, groups) = match params {
        FusedOpParams::ConvTranspose2D {
            stride, padding, output_padding, dilation, groups,
        } => (*stride, *padding, *output_padding, *dilation, *groups),
        _ => return CostEstimate::default(),
    };
    let x_dims = shapes[0].dims();
    let w_dims = shapes[1].dims();
    if x_dims.len() != 4 || w_dims.len() != 4 || groups == 0 {
        return CostEstimate::default();
    }
    let (n, cin, h_in, w_in) = (
        x_dims[0] as u64, x_dims[1] as u64,
        x_dims[2] as u64, x_dims[3] as u64,
    );
    let (_, cout_per_g, kh, kw) = (
        w_dims[0] as u64, w_dims[1] as u64,
        w_dims[2] as u64, w_dims[3] as u64,
    );
    let cout = cout_per_g * groups as u64;
    // Output spatial dims: `(H-1)·stride - 2·pad + dilation·(K-1) + out_pad + 1`.
    let h_out = h_in.saturating_sub(1) * stride.0 as u64
        + dilation.0 as u64 * kh.saturating_sub(1)
        + output_padding.0 as u64
        + 1;
    let h_out = h_out.saturating_sub(2 * padding.0 as u64);
    let w_out = w_in.saturating_sub(1) * stride.1 as u64
        + dilation.1 as u64 * kw.saturating_sub(1)
        + output_padding.1 as u64
        + 1;
    let w_out = w_out.saturating_sub(2 * padding.1 as u64);
    let cin_per_g = cin / groups as u64;
    let flops = 2u64 * n * cout * h_out * w_out * cin_per_g * kh * kw;
    let elems_in   = n * cin * h_in * w_in;
    let elems_w    = cin * cout_per_g * kh * kw;
    let elems_out  = n * cout * h_out * w_out;
    let bytes_moved = (elems_in + elems_w + elems_out) * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(FLASH_ATTN | PAGED_ATTN, Cpu, *)` impls.
/// The CPU naive-attention reference is bit-stable per-hardware
/// because both kernels iterate in fixed order. However the tiled
/// flash-attention GPU kernel produces different numerics than the
/// naive reference (different reduction tree); that's a GPU-side
/// PrecisionGuarantee concern — the CPU side stays bit-stable.
pub const ATTN_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend FlashAttn/PagedAttn: naive scaled-dot- \
            product reference; F32 accumulator; deterministic \
            iteration order; bit-identical re-run on same hardware. \
            GPU kernels produce different numerics (tiled softmax) — \
            their PrecisionGuarantee is declared separately when they \
            register.",
};

/// Cost model for `(FLASH_ATTN, Cpu)` and `(PAGED_ATTN, Cpu)` kernels
/// — scaled-dot-product attention is `O(B·Hq·Sq·Sk·D)` multiplies
/// for the QK matmul + same for the PV matmul, plus softmax (small).
///
/// For FlashAttn shapes are `[q, k, v, (alibi)]` with
/// `q = [B, Hq, Sq, D]`, `k = v = [B, Hkv, Sk, D]`.
/// For PagedAttn the cache shapes complicate the calculation; we
/// approximate using q's `Sq` × the maximum context length implied
/// by k_cache's `num_blocks · block_size`. Both share this function
/// because the dominant FLOP term has the same shape.
pub fn cost_attn_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    if shapes.is_empty() {
        return CostEstimate::default();
    }
    let q_dims = shapes[0].dims();
    if q_dims.len() != 4 {
        return CostEstimate::default();
    }
    let (b, hq, sq, d) = (
        q_dims[0] as u64, q_dims[1] as u64,
        q_dims[2] as u64, q_dims[3] as u64,
    );
    // Approximate K-len from input[1] (k or k_cache). For FlashAttn
    // k.shape[-2] is Sk; for PagedAttn k_cache.shape is `[num_blocks,
    // block_size, Hkv, D]` so the effective Sk per query is bounded
    // by `num_blocks · block_size`.
    let sk: u64 = if shapes.len() >= 2 {
        let k_dims = shapes[1].dims();
        if k_dims.len() == 4 {
            // FlashAttn shape: [B, Hkv, Sk, D] → Sk at index 2.
            // PagedAttn shape: [num_blocks, block_size, Hkv, D] →
            // total slots = dim[0] · dim[1]; conservative upper bound.
            (k_dims[2] as u64).max(k_dims[0] as u64 * k_dims[1] as u64)
        } else {
            sq
        }
    } else {
        sq
    };
    // QK matmul: 2·B·Hq·Sq·Sk·D FLOPs.
    // PV matmul: 2·B·Hq·Sq·Sk·D FLOPs.
    // Softmax: ~5·B·Hq·Sq·Sk FLOPs (small relative to the matmuls).
    let mm_flops = 4u64 * b * hq * sq * sk * d;
    let sm_flops = 5u64 * b * hq * sq * sk;
    // Bandwidth: q, k, v reads + output write; approximate.
    let elems_qkv = b * hq * sq * d + 2 * b * hq * sk * d;
    let elems_out = b * hq * sq * d;
    let bytes_moved = (elems_qkv + elems_out) * 4;
    CostEstimate {
        flops: mm_flops + sm_flops,
        bytes_moved,
        // Attention has higher launch overhead than elementwise.
        kernel_overhead_ns: 200,
    }
}

/// Precision guarantee for `(QMATMUL, Cpu, *)` impls. Quantized
/// matmul dequantizes inline; the dequant arithmetic is exact for
/// the quantization scheme's block format, and the F32 matmul
/// accumulator stays deterministic. Bit-identical on same hardware.
pub const QMATMUL_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend QMatMul: per-block dequant + F32 matmul \
            accumulate; deterministic iteration order; bit-identical \
            re-run on same hardware. Inherent precision loss from \
            the quantization itself is a property of the QuantType, \
            not the kernel — captured separately when the calibration \
            framework lands.",
};

/// Precision guarantee for `(REDUCE_MAX_TO_BACKWARD, Cpu, *)` impls.
/// The kernel recomputes the forward max, builds a tie-mask,
/// counts ties via reduce_sum, divides upstream by the count, then
/// broadcasts back and gates by the mask. All deterministic on
/// same hardware (no atomics, no parallel reductions); F32
/// accumulator for half-precision dtypes via the shared
/// `reduce_max_to_backward_impl` adapter.
pub const REDUCE_MAX_TO_BACKWARD_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend ReduceMaxTo backward: recomputed-max + \
            tie-mask + scale + gate; deterministic iteration order; \
            F32 accumulator for half-precision; bit-identical same-\
            hardware re-run.",
};

/// Precision guarantee for `(INPLACE_AFFINE, Cpu)`. Single-pass
/// `x = mul · x + add` per element with no parallel accumulation;
/// deterministic iteration order; bit-identical same-hardware re-run.
/// The half-precision dtypes (bf16/f16) pivot through f32 for the
/// arithmetic, identical to the non-inplace Affine family.
pub const INPLACE_AFFINE_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend in-place affine: single-pass `mul · x + \
            add` per element; deterministic iteration order; F32 pivot \
            for half-precision dtypes; bit-identical same-hardware re-run.",
};

/// Cost model for `(INPLACE_AFFINE, Cpu)`. Single pass: 1 FMA per
/// element. Bandwidth is one read + one write per element (in place,
/// so the buffer is touched once for read + once for write through
/// the same cache line).
pub fn cost_inplace_affine_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 1, "InplaceAffine cost: expected 1 input shape");
    let n: u64 = shapes[0].dims().iter().map(|&d| d as u64).product();
    // mul + add per element = 1 FMA. Treat as 1 FLOP for simplicity
    // (matches Affine's accounting; the fused mul-add is one tier-2
    // float op).
    let flops = n;
    // In-place: the buffer is read once + written once. Use
    // dtype-agnostic 4 bytes/elem default — the cost model's caller
    // scales by the actual dtype size if it cares.
    let bytes_moved = n * 4 * 2;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

/// Precision guarantee for `(POWI_BACKWARD, Cpu)`. Per-element
/// `grad_x = exp · x^(exp-1) · upstream` with the same floating-point
/// determinism guarantees as the forward PowI primitive: deterministic
/// iteration order, no parallel reductions, F32 compute for half-
/// precision dtypes via the shared `powi_backward_half_kernel`.
pub const POWI_BACKWARD_CPU_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-cpu-backend PowI backward: single-pass `exp · \
            x.powi(exp-1) · upstream` per element; deterministic \
            iteration order; F32 compute for half-precision; bit-\
            identical same-hardware re-run.",
};

/// Cost model for `(POWI_BACKWARD, Cpu)` — single pass over the input:
/// per-element `powi(exp-1)` (≈ log2(|exp|) FMAs) + 2 multiplies. Bandwidth
/// dominated by reading `x` + `upstream` and writing `grad_x`.
pub fn cost_powi_backward_cpu(
    shapes: &[Shape],
    params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 2, "PowIBackward cost: expected 2 input shapes");
    let in_count: u64 = shapes[0].dims().iter().map(|&d| d as u64).product();
    let exp_abs = match params {
        FusedOpParams::PowIBackward { exp } => (*exp).unsigned_abs().max(1) as u64,
        _ => 1,
    };
    // power-by-squaring: ceil(log2(|exp|)) multiplies for the powi,
    // +2 multiplies for the coefficient and upstream factors.
    let powi_muls = 64 - exp_abs.leading_zeros() as u64;
    let flops = in_count * (powi_muls + 2);
    let bytes_moved = in_count * 4 /* x */ + in_count * 4 /* upstream */ + in_count * 4 /* grad_x */;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 60,
    }
}

/// Cost model for `(REDUCE_MAX_TO_BACKWARD, Cpu)` — five passes over
/// the input: forward max, mask build, count, scale, broadcast +
/// gate. Roughly `5 · in_count` FLOPs, with bandwidth dominated by
/// 5× the input element count.
pub fn cost_reduce_max_to_backward_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 2, "ReduceMaxToBackward cost: expected 2 input shapes");
    let in_count: u64 = shapes[0].dims().iter().map(|&d| d as u64).product();
    let out_count: u64 = shapes[1].dims().iter().map(|&d| d as u64).product();
    let flops = 5 * in_count + 2 * out_count;
    let bytes_moved = (5 * in_count + 3 * out_count) * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 80,
    }
}

/// Cost model for `(QMATMUL, Cpu)` — same FLOP count as a regular
/// matmul (the dequant adds a small per-block overhead that's
/// dwarfed by the FMA count for any useful K).
///
/// Shapes: `[a, w_q_bytes]` where `a = [..., M, K]`. `N` comes from
/// the FusedOpParams::QMatMul payload.
pub fn cost_qmatmul_cpu(
    shapes: &[Shape],
    params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 2, "QMatMul cost: expected 2 input shapes");
    let (k, n) = match params {
        FusedOpParams::QMatMul { k, n, .. } => (*k as u64, *n as u64),
        _ => return CostEstimate::default(),
    };
    let a_dims = shapes[0].dims();
    let rank = a_dims.len();
    if rank < 2 {
        return CostEstimate::default();
    }
    let m = a_dims[rank - 2] as u64;
    let batch: u64 = a_dims[..rank - 2].iter().map(|&d| d as u64).product::<u64>().max(1);
    // 2·M·N·K FLOPs per batch (FMA).
    let flops = 2 * batch * m * n * k;
    // Bandwidth: read A + W_Q (counted as bytes via shapes[1]
    // elem_count of U32 = 4 bytes), write C.
    let a_elems = batch * m * k;
    let w_u32_elems: u64 = shapes[1].dims().iter().map(|&d| d as u64).product();
    let c_elems = batch * m * n;
    let bytes_moved = (a_elems + c_elems) * 4 + w_u32_elems * 4;
    CostEstimate {
        flops,
        bytes_moved,
        kernel_overhead_ns: 50,
    }
}

// =============================================================================
// Vulkan PrecisionGuarantee constants (Phase 7.6 step 9c follow-up,
// 2026-05-23). Per-kernel precision claims so the optimizer's
// tolerance-budget pass can admit Vulkan alternatives. The constants
// group Vulkan kernels by their numerical character: pointwise IEEE-
// 754 ops (bit-stable, ~1 ULP), half-precision pointwise (bit-stable
// at lower mantissa), transcendentals (vendor lib ~2-4 ULP),
// reductions (NOT bit-stable due to subgroup composition), matmul
// (deterministic accumulation), tensor-core matmul (FMA chains),
// byte-level (memcpy, bit-identical), cast, and qmatmul.
// =============================================================================

/// Vulkan f32 / f64 pointwise ops — Add, Sub, Mul, Div, Maximum,
/// Minimum, Neg, Sqr, Sqrt, Relu, Step, Affine, Clamp.
///
/// IEEE-754 conformant per the Vulkan SPIR-V spec: direct hardware
/// FADD / FMUL / FMA per thread, no atomic accumulators, no
/// cross-thread communication. Bit-stable on same hardware (kernel
/// scheduling doesn't affect per-thread outputs). ULP is the
/// standard IEEE-754 round-to-nearest tolerance (≤0.5 ULP for FADD/
/// FMUL/FDIV, ≤1 ULP for FMA chains).
///
/// Not REFERENCE-grade because we don't claim ULP=0 across all
/// implementations — Vulkan compilers may reorder commutative ops
/// or fuse multiply-adds in ways that differ from a strict
/// IEEE-754-strict reference. Within those bounds, every conforming
/// Vulkan implementation produces results within 1 ULP.
pub const VULKAN_FLOAT_POINTWISE_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: Some(1),
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend f32/f64 pointwise: direct hardware FADD/\
            FMUL/FMA per thread; no atomics; bit-stable on same hardware. \
            Within 1 ULP of IEEE-754 round-to-nearest; small variation \
            across implementations due to commutative-reorder + FMA fusion.",
};

/// Vulkan f16 / bf16 pointwise ops — same op surface as
/// [`VULKAN_FLOAT_POINTWISE_PRECISION`] at half precision. f16 uses
/// shaderFloat16 native ops (where available); bf16 packs two values
/// per u32 with software upcast/downcast through f32 for the arithmetic.
///
/// Bit-stable on same hardware (no atomics, deterministic per-thread).
/// ULP claim is 1 in the dtype's mantissa space — meaningful for f16
/// (10-bit mantissa) and bf16 (7-bit mantissa) precision budgeting.
/// Absolute error scales with magnitude × 2^(-mantissa_bits).
pub const VULKAN_HALF_POINTWISE_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: Some(1),
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend f16/bf16 pointwise: shaderFloat16 native or \
            u32-packed bf16; bit-stable on same hardware; within 1 ULP in \
            the dtype's mantissa space (10-bit f16, 7-bit bf16).",
};

/// Vulkan transcendentals — Exp, Log, Sin, Cos, Tanh, Sigmoid, Silu,
/// Gelu (f32/f16/bf16/f64). f32/f16 go through GLSL.std.450's vendor-
/// library implementations (variable ULP per GPU/driver); f64 uses
/// the Horner-polynomial approximations in `unary_f64.slang`
/// (target ~1e-12 relative error).
///
/// Bit-stable on same hardware (driver-deterministic; the
/// transcendental implementation is fixed per build). ULP bound is
/// loose: Vulkan spec allows 3-4 ULP for GLSL.std.450 transcendentals;
/// real implementations are typically tighter (~1-2 ULP on NVIDIA's
/// libdevice).
pub const VULKAN_TRANSCENDENTAL_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: Some(4),
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend transcendentals: GLSL.std.450 vendor lib \
            (f32/f16/bf16) or Horner polynomial (f64); bit-stable on same \
            hardware; Vulkan spec allows ≤4 ULP, real GPUs typically tighter.",
};

// Vulkan reductions — SumReduce, MaxReduce, MinReduce, MeanReduce,
// SoftmaxLastDim, RmsNormLastDim, LayerNormLastDim, and any kernel
// that uses subgroup tree reductions — use `PrecisionGuarantee::
// UNAUDITED` (audited result of "no static bound to claim"). The
// reasoning: subgroup composition is scheduler-determined per
// dispatch, so reordering FADDs across subgroups produces different
// floating-point results; no static ULP / relative / absolute bound
// applies. Their (op, dtypes, Vulkan) tuples are listed in the
// `vulkan_dispatch_per_kernel_precision_and_cost_coverage` lint's
// `KNOWN_GAPS` allowlist so the lint accepts UNAUDITED for these
// specific kernels; new Vulkan reduction kernels must be added to
// KNOWN_GAPS at registration time. The architectural decision is
// thus visible at review time rather than buried in a per-kernel
// constant.

/// Vulkan matmul (f32 standard, no tensor cores) — tiled / reg-tile /
/// matvec kernels. Accumulation order is deterministic per kernel
/// dispatch (each output element's reduction runs in a single thread
/// with no cross-thread atomic adds); the kernel is bit-stable on
/// same hardware.
///
/// ULP claim left unspecified; the per-element error is bounded by
/// `K · ULP(2 · max_abs_product)` from the K-deep FMA chain.
pub const VULKAN_MATMUL_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend f32 matmul: tiled / reg-tile / matvec \
            kernels with per-thread sequential FMA accumulation; bit-stable \
            on same hardware; error bounded by O(K · ULP) from K-deep \
            reduction.",
};

/// Vulkan tensor-core mixed-precision matmul (f32 × bf16 → f32, via
/// `matmul_coop` cooperative kernels on Ampere+ tensor cores).
///
/// Tensor-core FMAs accumulate in extended precision (f32 internal,
/// IEEE-754 conformant FMA chain); the cross-tile reduction happens
/// with full f32 precision. Bit-stable on same hardware per the
/// Ampere/Ada spec (the tensor cores are deterministic given the
/// same inputs + scheduling). ULP claim is wider than pure f32
/// because the bf16 inputs lose 16 mantissa bits before the FMA.
pub const VULKAN_MATMUL_TENSORCORE_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend tensor-core matmul (f32 × bf16 → f32): \
            cooperative-matrix FMAs with f32 internal accumulator; bit-stable \
            on same hardware; per-tile error bounded by O(K · ULP_bf16 + \
            tile_K · ULP_f32).",
};

/// Vulkan byte-level (memcpy-style) ops — Triu, Tril, Flip, Roll,
/// WriteSlice, IndexSelect, Concat, Copy (D2H/H2D). Pure data movement
/// with no FP arithmetic; output bytes are bit-identical to input
/// bytes selected/permuted according to the op.
///
/// Bit-stable on same hardware, ULP/relative/absolute = 0 (byte-level
/// equivalence). The strongest precision claim, equivalent to
/// REFERENCE but earned through "no FP math" rather than "explicit
/// IEEE-754 evaluation."
pub const VULKAN_BYTE_LEVEL_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: Some(0),
    max_relative: Some(0.0),
    max_absolute: Some(0.0),
    notes: "fuel-vulkan-backend byte-level ops (Triu/Tril/Flip/Roll/\
            WriteSlice/IndexSelect/Concat/Copy): pure data movement, no FP \
            math; bit-identical to input bytes.",
};

/// Vulkan dtype cast — f32↔f16, f32↔bf16, F8E4M3↔{f32,f16,bf16}.
/// Pure conversion with no accumulation; output is the IEEE-754
/// round-to-nearest representation of the input in the target dtype.
///
/// Bit-stable on same hardware (no atomics, no cross-thread comm).
/// ULP=0 vs the IEEE-754 round-to-nearest reference for the target
/// dtype (the cast itself is exact within the target's representable
/// range; any precision loss is inherent to the dtype, not the
/// implementation).
pub const VULKAN_CAST_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: Some(0),
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend Cast: pure dtype conversion via SPIR-V \
            OpFConvert / OpConvertFToS / OpConvertSToF; bit-stable on same \
            hardware; ULP=0 vs IEEE-754 round-to-nearest in target dtype \
            (loss is inherent to the target dtype, not the implementation).",
};

/// Vulkan QMatMul (Q4_0 / Q4_K_M / Q8_0 weights × F32 activation).
/// Per-block dequant + per-output-element FMA accumulation; same
/// deterministic structure as the CPU QMatMul kernel (bit-stable on
/// same hardware).
///
/// The inherent precision loss from quantization is a property of
/// the QuantType (captured separately when the calibration framework
/// lands); the kernel itself contributes only the matmul accumulation
/// error.
pub const VULKAN_QMATMUL_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
    bit_stable_on_same_hardware: true,
    max_ulp: None,
    max_relative: None,
    max_absolute: None,
    notes: "fuel-vulkan-backend QMatMul (Q4_0/Q4_K_M/Q8_0 × F32): per-block \
            dequant + per-output-element FMA; bit-stable on same hardware. \
            Quantization precision loss is a QuantType property, not the \
            kernel's contribution.",
};

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::{DType, Layout, Result};
    use std::sync::Arc;
    use std::sync::RwLock;

    fn dummy_kernel(
        _inputs: &[Arc<RwLock<crate::Storage>>],
        _outputs: &mut [Arc<RwLock<crate::Storage>>],
        _layouts: &[Layout],
        _params: &crate::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn dummy_cost(_s: &[Shape], _p: &FusedOpParams, _c: &BackendCapabilities) -> CostEstimate {
        CostEstimate::default()
    }

    const DUMMY_DTYPES: &[DType] = &[DType::F32];

    fn make_impl() -> BackendImpl {
        BackendImpl {
            kernel: dummy_kernel,
            dtypes: DUMMY_DTYPES,
            cost: dummy_cost,
            precision: PrecisionGuarantee::UNAUDITED,
            caps: KernelCaps::empty(),
            revision: KernelRevisionHash::UNTRACKED,
        }
    }

    /// Smoke: empty registry has no impls.
    #[test]
    fn fused_kernel_registry_empty() {
        let r = FusedKernelRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.lookup(FusedOpId(1), BackendId::Cpu).is_none());
        assert!(r.impls_for(FusedOpId(1)).is_empty());
        // Suppress unused-warning for DType import on no-feature builds.
        let _ = DType::F32;
    }

    /// Smoke: register and look up a single impl.
    #[test]
    fn fused_kernel_registry_register_and_lookup() {
        let mut r = FusedKernelRegistry::new();
        r.register(FusedOpId(1), BackendId::Cpu, make_impl());
        assert!(!r.is_empty());
        assert_eq!(r.len(), 1);
        let got = r.lookup(FusedOpId(1), BackendId::Cpu);
        assert!(got.is_some());
        let impls = r.impls_for(FusedOpId(1));
        assert_eq!(impls.len(), 1);
        assert_eq!(impls[0].0, BackendId::Cpu);
    }

    /// Re-registering at the same `(id, backend)` appends — multiple
    /// alternatives at the same decision point are a feature (e.g.
    /// cuBLAS vs CUTLASS bf16 matmul at the same dtype). The route
    /// picker filters by dtypes via `lookup_by_dtypes` and ranks
    /// remaining alternatives.
    #[test]
    fn fused_kernel_registry_register_appends_alternatives() {
        let mut r = FusedKernelRegistry::new();
        r.register(FusedOpId(1), BackendId::Cpu, make_impl());
        r.register(FusedOpId(1), BackendId::Cpu, make_impl());
        assert_eq!(r.len(), 1, "one id");
        assert_eq!(
            r.impls_for(FusedOpId(1)).len(),
            2,
            "two alternatives registered",
        );
    }

    /// PrecisionGuarantee::REFERENCE has the strongest properties.
    #[test]
    fn precision_guarantee_reference_is_strict() {
        let p = PrecisionGuarantee::REFERENCE;
        assert!(p.bit_stable_on_same_hardware);
        assert_eq!(p.max_ulp, Some(0));
    }

    /// CostEstimate::default is all-zero.
    #[test]
    fn cost_estimate_default_zero() {
        let c = CostEstimate::default();
        assert_eq!(c.flops, 0);
        assert_eq!(c.bytes_moved, 0);
        assert_eq!(c.kernel_overhead_ns, 0);
    }

    /// Step 6 — `default_kernel_registry` populates the four CPU
    /// FusedLinear impls under `FUSED_LINEAR × Cpu × (F32|F64|BF16|F16)`.
    #[test]
    fn default_kernel_registry_has_fused_linear_cpu_quartet() {
        use fuel_graph::registry::FusedOps;

        let r = default_kernel_registry();
        let impls = r.impls_for(FusedOps::FUSED_LINEAR);
        assert_eq!(
            impls.len(),
            4,
            "expected 4 FusedLinear × Cpu × dtype registrations, got {}",
            impls.len(),
        );
        for (backend, _) in impls {
            assert_eq!(*backend, BackendId::Cpu);
        }
        // Per-dtype lookup hits each registration.
        for dtype in [DType::F32, DType::F64, DType::BF16, DType::F16] {
            let want = [dtype; 4];
            let got = r.lookup_by_dtypes(
                FusedOps::FUSED_LINEAR, BackendId::Cpu, &want,
            );
            assert!(
                got.is_some(),
                "lookup_by_dtypes missed FusedLinear × Cpu × {dtype:?}",
            );
            let impl_ = got.unwrap();
            assert!(
                impl_.precision.bit_stable_on_same_hardware,
                "CPU FusedLinear should be bit-stable on same hardware",
            );
        }
    }

    /// Step 6 (Conv2D extension) — `default_kernel_registry` populates
    /// eight CPU Conv2D impls: four dtypes × {no-bias, with-bias}.
    /// The route picker filters by the dtype tuple length to dispatch
    /// the right variant per call site.
    #[test]
    fn default_kernel_registry_has_conv2d_cpu_octet() {
        use fuel_graph::registry::FusedOps;

        let r = default_kernel_registry();
        let impls = r.impls_for(FusedOps::CONV2D);
        assert_eq!(
            impls.len(),
            8,
            "expected 8 Conv2D × Cpu × dtype × (no-bias|with-bias) \
             registrations, got {}",
            impls.len(),
        );
        for (backend, _) in impls {
            assert_eq!(*backend, BackendId::Cpu);
        }
        // No-bias (rank-3) and with-bias (rank-4) lookups both hit per dtype.
        for dtype in [DType::F32, DType::F64, DType::BF16, DType::F16] {
            let no_bias = [dtype; 3];
            let with_bias = [dtype; 4];
            assert!(
                r.lookup_by_dtypes(FusedOps::CONV2D, BackendId::Cpu, &no_bias).is_some(),
                "lookup_by_dtypes missed Conv2D × Cpu × {dtype:?} (no-bias)",
            );
            let with_bias_impl = r.lookup_by_dtypes(
                FusedOps::CONV2D, BackendId::Cpu, &with_bias,
            );
            assert!(
                with_bias_impl.is_some(),
                "lookup_by_dtypes missed Conv2D × Cpu × {dtype:?} (with-bias)",
            );
            let impl_ = with_bias_impl.unwrap();
            assert!(
                impl_.precision.bit_stable_on_same_hardware,
                "CPU Conv2D should be bit-stable on same hardware",
            );
        }
    }

    /// Phase 7.6 step 7 — the architecture v1.0 §05 "always-built
    /// backend bit-stable coverage commitment" lint.
    ///
    /// Every fused op registered in
    /// `fuel_graph::registry::default_registry()` MUST have at least
    /// one CPU `BackendImpl` in
    /// `fuel_storage::fused::default_kernel_registry()` with
    /// `precision.bit_stable_on_same_hardware == true`. This is the
    /// architecture's correctness anchor: the always-built backend
    /// gives every downstream consumer a deterministic
    /// implementation to fall back on, so cross-backend equivalence
    /// tests have a fixed reference.
    ///
    /// The lint runs as a unit test so violations surface in CI
    /// rather than at runtime. Adding a new `FusedOpEntry` without
    /// a matching CPU registration fails this test immediately.
    ///
    /// As of step-6 follow-up (2026-05-11), the previously-empty-
    /// covered backward helpers (SoftmaxLastDimBackward,
    /// LayerNormLastDimBackward, RmsNormLastDimBackward,
    /// ReduceMaxToBackward) gained byte-level CPU wrappers +
    /// binding-table registrations + BackendImpls. The `KNOWN_GAPS`
    /// allowlist is therefore **empty** — every registered fused
    /// op has bit-stable CPU coverage.
    ///
    /// Note on primitive Ops: this lint covers the **fused-op**
    /// registry only. Primitive ops (Op::Add, Op::MatMul, etc.)
    /// dispatch through `KernelBindingTable` and don't carry a
    /// `PrecisionGuarantee` field today. Extending the architecture
    /// commitment to primitives is step 7b — pending a binding-table
    /// refactor or a parallel PrecisionGuarantee side-table per
    /// OpKind.
    #[test]
    fn precision_guarantee_lint_bit_stable_cpu_coverage() {
        use fuel_graph::registry::{default_registry, FusedOpId};

        // The allowlist is empty now that every registered fused op
        // has bit-stable CPU coverage. If a future commit introduces
        // a new entry without immediate CPU coverage, add it here
        // with a documented reason AND file a follow-up to remove
        // the line.
        const KNOWN_GAPS: &[(FusedOpId, &str)] = &[];

        let meta = default_registry();
        let kernels = default_kernel_registry();

        let mut failures: Vec<String> = Vec::new();
        let mut covered = 0usize;
        let mut allowlisted = 0usize;

        for entry in meta.entries_iter() {
            let id = entry.id;
            let name = entry.name;

            if let Some((_, reason)) = KNOWN_GAPS.iter().find(|(g, _)| *g == id) {
                allowlisted += 1;
                // Sanity: a gap entry should have no CPU coverage.
                // If a CPU registration shows up for an
                // allowlisted id, the gap entry should be removed.
                let has_cpu = kernels.impls_for(id).iter().any(|(b, _)| *b == BackendId::Cpu);
                if has_cpu {
                    failures.push(format!(
                        "FusedOpId {id:?} ({name}) is on the KNOWN_GAPS allowlist \
                         but DOES have a CPU registration now. Reason given was: \
                         {reason:?}. Remove the allowlist entry to enable the \
                         bit-stable lint.",
                    ));
                }
                continue;
            }

            let impls = kernels.impls_for(id);
            let has_bit_stable_cpu = impls.iter().any(|(backend, impl_)| {
                *backend == BackendId::Cpu && impl_.precision.bit_stable_on_same_hardware
            });
            if has_bit_stable_cpu {
                covered += 1;
            } else {
                failures.push(format!(
                    "FusedOpId {id:?} ({name}) has no bit-stable CPU BackendImpl. \
                     Architecture v1.0 §05 requires the always-built backend \
                     (fuel-cpu-backend) to provide at least one \
                     `bit_stable_on_same_hardware: true` kernel per registered \
                     fused op. Either add a CPU registration in \
                     fuel-storage::dispatch::register_default_fused_kernels with \
                     a matching PrecisionGuarantee, or add a line to KNOWN_GAPS \
                     above with a documented reason.",
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "Architecture v1.0 bit-stable CPU coverage lint failed:\n{}",
            failures.join("\n"),
        );
        // Sanity: the lint should be exercising real ops. If both
        // counts are zero, something is wrong with the registry
        // population.
        assert!(
            covered > 0 || allowlisted > 0,
            "lint covered 0 ops — registry appears empty"
        );
        // Stronger commitment: with KNOWN_GAPS empty, every entry
        // should be covered.
        assert_eq!(
            allowlisted, 0,
            "KNOWN_GAPS allowlist is empty by design; saw {allowlisted} allowlisted",
        );
        // Sanity: we should see all 19 registered fused ops covered.
        // 14 from the original Phase 7.6 lineup + PowIBackward
        // (autograd switched from emitting the primitive decomposition
        // to emitting Op::Fused(POWI_BACKWARD, _)) + INPLACE_AFFINE
        // (Phase 3 of the in-place ops infrastructure) +
        // FUSED_SOFTMAX_CROSS_ENTROPY + CAUSAL_CONV1D + SELECTIVE_SCAN
        // (the CPU OpKind coverage plan's first three fused-op additions).
        assert_eq!(
            covered, 19,
            "expected 19 fused ops covered by bit-stable CPU impls, got {covered}",
        );
    }

    /// Negative-path check for the bit-stable lint logic: a
    /// `FusedKernelRegistry` with no CPU impl for a registered id
    /// must fail the bit-stable check. Verifies the lint's failure
    /// detection works rather than just relying on the positive
    /// path being green.
    ///
    /// Doesn't touch `default_kernel_registry()` — runs the check
    /// against a freshly-constructed empty registry so the failure
    /// is deterministic regardless of global state.
    #[test]
    fn precision_guarantee_lint_detects_missing_cpu() {
        use fuel_graph::registry::FusedOpId;
        let empty = FusedKernelRegistry::new();
        let id = FusedOpId(1);
        // Same predicate the production lint uses.
        let has_bit_stable_cpu = empty.impls_for(id).iter().any(|(backend, impl_)| {
            *backend == BackendId::Cpu && impl_.precision.bit_stable_on_same_hardware
        });
        assert!(!has_bit_stable_cpu,
            "empty registry must not report bit-stable CPU coverage");
    }

    /// Negative-path check: a CPU registration without
    /// `bit_stable_on_same_hardware: true` must NOT satisfy the
    /// lint. Captures the difference between "any CPU impl" and "a
    /// bit-stable CPU impl" — the architecture commits to the
    /// stronger property.
    #[test]
    fn precision_guarantee_lint_rejects_non_bit_stable_cpu() {
        use fuel_graph::registry::FusedOpId;
        let mut r = FusedKernelRegistry::new();
        let id = FusedOpId(42);
        let weak = BackendImpl {
            kernel: dummy_kernel,
            dtypes: DUMMY_DTYPES,
            cost: dummy_cost,
            // UNAUDITED has bit_stable_on_same_hardware: false.
            precision: PrecisionGuarantee::UNAUDITED,
            caps: KernelCaps::empty(),
            revision: KernelRevisionHash::UNTRACKED,
        };
        r.register(id, BackendId::Cpu, weak);
        let has_bit_stable_cpu = r.impls_for(id).iter().any(|(backend, impl_)| {
            *backend == BackendId::Cpu && impl_.precision.bit_stable_on_same_hardware
        });
        assert!(!has_bit_stable_cpu,
            "an UNAUDITED-precision CPU impl must not satisfy the bit-stable lint");
    }

    /// Step 6 + backward-helper follow-up — coverage assertion for
    /// the 12 ops that gained BackendImpls (8 forwards from step 6
    /// + 4 backward helpers from the follow-up). Each entry should
    /// have at least the expected CPU impl count after
    /// `default_kernel_registry()` populates.
    #[test]
    fn default_kernel_registry_step6_coverage() {
        use fuel_graph::registry::FusedOps;

        let r = default_kernel_registry();
        // (id, expected_impl_count) — derived from the dtype-tuple
        // shapes registered. SoftmaxLastDim/RmsNormLastDim/
        // LayerNormLastDim/Rope = 4 dtypes each. ConvTranspose2D /
        // FlashAttn / PagedAttn = 4 dtypes × 2 shapes = 8 each.
        // QMatMul = 1 (F32 only). The 4 backward helpers = 4 dtypes
        // each.
        for (id, want) in [
            (FusedOps::SOFTMAX_LAST_DIM,             4usize),
            (FusedOps::RMS_NORM_LAST_DIM,            4),
            (FusedOps::LAYER_NORM_LAST_DIM,          4),
            (FusedOps::ROPE,                         4),
            (FusedOps::CONV_TRANSPOSE2D,             8),
            (FusedOps::FLASH_ATTN,                   8),
            (FusedOps::PAGED_ATTN,                   8),
            (FusedOps::QMATMUL,                      1),
            (FusedOps::SOFTMAX_LAST_DIM_BACKWARD,    4),
            (FusedOps::LAYER_NORM_LAST_DIM_BACKWARD, 4),
            (FusedOps::RMS_NORM_LAST_DIM_BACKWARD,   4),
            (FusedOps::REDUCE_MAX_TO_BACKWARD,       4),
        ] {
            let impls = r.impls_for(id);
            assert_eq!(
                impls.len(), want,
                "expected {want} CPU impls for FusedOpId {id:?}, got {}",
                impls.len(),
            );
            for (backend, impl_) in impls {
                assert_eq!(*backend, BackendId::Cpu);
                assert!(
                    impl_.precision.bit_stable_on_same_hardware,
                    "CPU impl for {id:?} should be bit-stable on same hardware",
                );
            }
        }
    }

    /// Conv2D cost — FLOPs scale with `2·N·Cout·Hout·Wave·(Cin/g)·Kh·Kw`
    /// + bias-add FLOPs (when bias is present).
    #[test]
    fn cost_conv2d_cpu_flops_scale() {
        use crate::fused::cost_conv2d_cpu;
        use fuel_core_types::DeviceLocation;
        use std::collections::HashSet;

        // x: [N=2, Cin=4, H=8, W=8], w: [Cout=6, Cin/g=4, Kh=3, Kw=3],
        // stride=(1,1), padding=(1,1), groups=1.
        // Hout = Wout = (8 + 2 - 3) + 1 = 8.
        // Conv FLOPs (FMA = 2): 2·N·Cout·Hout·Wout·Cin_per_g·Kh·Kw
        //                    = 2·2·6·8·8·4·3·3 = 55296.
        // Bias FLOPs: N·Cout·Hout·Wout = 2·6·8·8 = 768.
        let x = Shape::from_dims(&[2, 4, 8, 8]);
        let w = Shape::from_dims(&[6, 4, 3, 3]);
        let bias = Shape::from_dims(&[6]);
        let caps = BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 1,
            access_granularity_bits: 8,
            transfer_paths: vec![],
        };
        let params = FusedOpParams::Conv2D {
            stride:  (1, 1),
            padding: (1, 1),
            groups:  1,
        };
        let c = cost_conv2d_cpu(&[x, w, bias], &params, &caps);
        assert_eq!(c.flops, 55296 + 768);
        assert!(c.bytes_moved > 0);
    }

    /// Cost model — FLOPs scale with 2·M·N·K + M·N, batch multiplies.
    #[test]
    fn cost_fused_linear_cpu_flops_scale() {
        use fuel_core_types::DeviceLocation;
        use std::collections::HashSet;

        let a = Shape::from_dims(&[2, 4, 8]);  // batch=2, M=4, K=8
        let b = Shape::from_dims(&[2, 8, 16]); // batch=2, K=8, N=16
        let bias = Shape::from_dims(&[16]);
        let caps = BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 1,
            access_granularity_bits: 8,
            transfer_paths: vec![],
        };
        let c = cost_fused_linear_cpu(
            &[a, b, bias],
            &FusedOpParams::FusedLinear,
            &caps,
        );
        // 2 · 2 · 4 · 16 · 8 = 2048 matmul FLOPs + 2 · 4 · 16 = 128 bias FLOPs
        assert_eq!(c.flops, 2048 + 128);
        assert!(c.bytes_moved > 0);
    }
}
