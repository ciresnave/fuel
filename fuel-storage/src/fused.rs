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
    /// Free-text qualifier — implementation hints, vendor citation,
    /// known caveats. Surfaces in error messages; not load-bearing.
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

    /// Conservative all-unknown defaults. Used as a placeholder during
    /// the migration when a real PrecisionGuarantee hasn't been audited
    /// yet. Step 7 replaces every use of this with a real claim and
    /// adds a CI lint that fails when a registered kernel still uses
    /// `UNKNOWN`.
    pub const UNKNOWN: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: false,
        max_ulp: None,
        max_relative: None,
        max_absolute: None,
        notes: "PrecisionGuarantee::UNKNOWN — populate via step 7.",
    };
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
/// Shapes: `[x]` where `x` is rank-≥1; the reduction is along the
/// last axis.
pub fn cost_norm_family_cpu(
    shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    debug_assert_eq!(shapes.len(), 1, "Norm-family cost: expected 1 input shape");
    let dims = shapes[0].dims();
    if dims.is_empty() {
        return CostEstimate::default();
    }
    let elems: u64 = dims.iter().map(|&d| d as u64).product();
    // ~5 FLOPs per element on average for softmax (max-subtract + exp
    // + sum + divide); ~3 for rms_norm (sqr + sum + sqrt + divide);
    // ~7 for layer_norm (mean-sub + sqr + sum + sqrt + divide). Use
    // 5 as a midpoint; cost is advisory.
    let flops = 5 * elems;
    // 2 reads + 1 write of every element.
    let bytes_moved = 3 * elems * 4;
    CostEstimate {
        flops,
        bytes_moved,
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
            precision: PrecisionGuarantee::UNKNOWN,
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
        // Sanity: we should see all 14 registered fused ops covered.
        assert_eq!(
            covered, 14,
            "expected 14 fused ops covered by bit-stable CPU impls, got {covered}",
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
            // UNKNOWN has bit_stable_on_same_hardware: false.
            precision: PrecisionGuarantee::UNKNOWN,
            caps: KernelCaps::empty(),
            revision: KernelRevisionHash::UNTRACKED,
        };
        r.register(id, BackendId::Cpu, weak);
        let has_bit_stable_cpu = r.impls_for(id).iter().any(|(backend, impl_)| {
            *backend == BackendId::Cpu && impl_.precision.bit_stable_on_same_hardware
        });
        assert!(!has_bit_stable_cpu,
            "an UNKNOWN-precision CPU impl must not satisfy the bit-stable lint");
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
