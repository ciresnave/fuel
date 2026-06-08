//! The Judge — Phase 6b's empirical profiler.
//!
//! For every `(backend, device)` pair the probe report knows about,
//! the Judge walks a matrix of `(op_kind, dtype, size_class)` and
//! measures two things: wall-clock latency and numerical precision
//! relative to the reference backend. Output is a persistable
//! [`ProfileReport`] that the (future) ranked dispatch table indexes
//! at realize time.
//!
//! ## Submodules
//!
//! - [`cache`] — process-wide [`DispatchTable`] cache. Hosts
//!   `cached()`, `populate_dispatch_table()`, `invalidate()`. Was
//!   `fuel_core::dispatch` before the 2026-05-31 dispatch-move; the
//!   name was a misnomer (it caches the Judge's output, not the
//!   binding-table dispatch path), so it now lives under the Judge
//!   module that owns its lifecycle. The cache surface is re-exported
//!   at this module's top level so callers can write
//!   `fuel_core::judge::cached()` directly.
//!
//! # Scope of this revision
//!
//! Ships the types, the runner skeleton, and per-family probe
//! coverage spanning the inference op surface in F32:
//!
//! - **Ops**: see [`PROFILED_OPS`] for the canonical list. Currently:
//!   - dense linear algebra: [`OpKind::MatMul`]
//!   - elementwise binary: Add / Sub / Mul / Div / Maximum / Minimum
//!   - elementwise unary: Neg / Sqr / Sqrt / Exp / Log / Sin / Cos /
//!     Tanh / Sigmoid / Silu / Gelu / Relu / Step
//!   - per-axis reductions: SumReduce / MaxReduce / MinReduce /
//!     MeanReduce (probed as last-dim reduction over `[rows, 64]`)
//!   - reduce-to-broadcast-target: ReduceSumTo / ReduceMaxTo
//!   - parametric one-input: Affine / Clamp / PowI
//!   - 28 OpKind variants total, 84 (op, size) cells per backend
//!     class.
//! - **Dtypes**: f32 for now. f64 / bf16 / f16 are a mechanical
//!   extension once the dispatch table surfaces the dtype axis.
//! - **Backends**: every backend in the [`crate::factories`] registry.
//!   A backend that doesn't implement an op is skipped per cell with
//!   a stderr note (the realize call is wrapped in `catch_unwind`),
//!   not fatal to the run — what makes the Judge safe to expand
//!   ahead of every backend's coverage.
//!
//! Op kinds *not yet profiled* (build_input_graph would `panic!` if
//! the size_plan returned a non-empty ladder for them; the size_plan's
//! catch-all `_ => Vec::new()` arm silently skips):
//!
//! - composition / fused ops: SoftmaxLastDim / RmsNormLastDim /
//!   LayerNormLastDim / Rope / FlashAttn / PagedAttn / FusedLinear.
//!   These are slated for Phase 7.6 fused-op profiling (per ROADMAP),
//!   not the primitive-OpKind sweep this module covers.
//! - shape-rearranging: Cast / IndexSelect / Gather / Concat /
//!   IndexAdd / ScatterAdd / ArgMaxDim / ArgMinDim.
//! - convolutions: Conv2D / ConvTranspose2D.
//! - quantized: QMatMul.
//!
//! Each of those is a separable add: extend [`PROFILED_OPS`], add a
//! `size_plan` arm with a representative size ladder, add a
//! `build_input_graph` arm that constructs the input graph.
//!
//! # Equivalence-class discipline
//!
//! The Judge profiles one device per equivalence class. A rig with
//! four identical RTX 4090s runs one CUDA profile and the dispatch
//! table shares the result across all four ordinals.
//!
//! # Why latency *and* precision
//!
//! Two ops with identical latency can produce different answers — a
//! fast backend that drifts 1e-3 rel vs reference is fine for most
//! ML workloads but catastrophic for a correctness-critical path
//! like a cross-device equivalence test. Carrying both axes lets a
//! dispatch table serve both "fastest" and "most accurate"
//! criteria without re-profiling.

pub mod cache;
pub use cache::*;

use crate::probe::ProbeReport;
use fuel_core_types::probe::{BackendId, DeviceDescriptor};
use fuel_core_types::{DType, Result, Shape};
use std::panic::AssertUnwindSafe;
use std::time::Instant;

// Re-export the dispatch types (moved to fuel-core-types so
// `fuel-graph-router`'s Router can consume them without depending
// on `fuel-core`). All existing callers `use fuel_core::judge::*`
// keep working — `ProfileReport::save` / `ProfileReport::load` are
// inherent methods on the moved type.
pub use fuel_core_types::dispatch::{
    OpKind, ProfileEntry, ProfileReport, SizeClass, PROFILE_REPORT_VERSION,
};

/// Default filename for the persisted profile report.
pub const PROFILE_REPORT_FILENAME: &str = "judge.json";

/// Op kinds the Judge currently profiles. Each entry must have a
/// matching `size_plan` arm (returning a non-empty ladder) and a
/// matching `build_input_graph` arm. New families land here as they
/// ship; the order here is the order entries appear in the persisted
/// report (deterministic for diff-friendly output).
const PROFILED_OPS: &[OpKind] = &[
    OpKind::MatMul,
    OpKind::AddElementwise,
    // --- elementwise unary fanout ---
    OpKind::NegElementwise,
    OpKind::SqrElementwise,
    OpKind::SqrtElementwise,
    OpKind::ExpElementwise,
    OpKind::LogElementwise,
    OpKind::SinElementwise,
    OpKind::CosElementwise,
    OpKind::TanhElementwise,
    OpKind::SigmoidElementwise,
    OpKind::SiluElementwise,
    OpKind::GeluElementwise,
    OpKind::ReluElementwise,
    OpKind::StepElementwise,
    OpKind::RecipElementwise,
    OpKind::AbsElementwise,
    OpKind::FloorElementwise,
    OpKind::CeilElementwise,
    OpKind::RoundElementwise,
    OpKind::SignElementwise,
    OpKind::ErfElementwise,
    OpKind::GeluErfElementwise,
    OpKind::RsqrtElementwise,
    // --- elementwise binary additions ---
    OpKind::PowElementwise,
    OpKind::RemElementwise,
    // --- elementwise binary fanout ---
    OpKind::SubElementwise,
    OpKind::MulElementwise,
    OpKind::DivElementwise,
    OpKind::MaximumElementwise,
    OpKind::MinimumElementwise,
    // --- reductions along one dim ---
    OpKind::SumReduce,
    OpKind::MaxReduce,
    OpKind::MinReduce,
    OpKind::MeanReduce,
    // --- reduce-to-broadcast-target ---
    OpKind::ReduceSumTo,
    OpKind::ReduceMaxTo,
    // --- scalar / clamp / powi ---
    OpKind::Affine,
    OpKind::ClampElementwise,
    OpKind::PowIElementwise,
];

pub fn default_report_path() -> Option<std::path::PathBuf> {
    crate::probe::default_report_path()
        .and_then(|p| p.parent().map(|parent| parent.join(PROFILE_REPORT_FILENAME)))
}

/// How many measurement iterations per (op, dtype, size, backend)
/// cell. Median of this many runs is recorded.
pub const DEFAULT_ITERATIONS: u32 = 7;

/// How many warmup iterations before the timed run. Discards cold-
/// cache and JIT-compile latency; matters most on GPU backends
/// where the first launch includes kernel-cache population.
pub const DEFAULT_WARMUP: u32 = 3;

/// The runner. Consumes a [`ProbeReport`] (so it knows which devices
/// exist and which equivalence classes to share across), builds the
/// full measurement matrix, and produces a [`ProfileReport`].
pub struct Judge {
    pub iterations: u32,
    pub warmup:     u32,
    /// Optional shrunk size ladder. `None` = use the full default
    /// ladder (three sizes per op, up to 1024×1024 matmul / 2²⁰
    /// elementwise). Tests supply a shrunk ladder to stay fast.
    pub size_plan_override: Option<Vec<(OpKind, OpSize)>>,
}

impl Default for Judge {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_ITERATIONS,
            warmup:     DEFAULT_WARMUP,
            size_plan_override: None,
        }
    }
}

impl Judge {
    /// Representative size classes for the Judge's op matrix. Each
    /// entry is (op, [input element counts]) — the Judge picks the
    /// first entry whose op matches and iterates the sizes.
    fn size_plan(&self, op: OpKind) -> Vec<OpSize> {
        if let Some(over) = &self.size_plan_override {
            return over.iter()
                .filter(|(k, _)| *k == op)
                .map(|(_, s)| *s)
                .collect();
        }
        match op {
            OpKind::MatMul => vec![
                OpSize::MatMul { m: 64,  n: 64,  k: 64  },
                OpSize::MatMul { m: 256, n: 256, k: 256 },
                OpSize::MatMul { m: 1024, n: 1024, k: 1024 },
            ],
            // Element-wise binary + unary share one ladder. Three sizes
            // span the regime where launch overhead dominates (1 KiB),
            // L2 still fits the working set (256 KiB), and DRAM
            // bandwidth dominates (4 MiB).
            OpKind::AddElementwise
            | OpKind::SubElementwise
            | OpKind::MulElementwise
            | OpKind::DivElementwise
            | OpKind::MaximumElementwise
            | OpKind::MinimumElementwise
            | OpKind::PowElementwise
            | OpKind::RemElementwise
            | OpKind::NegElementwise
            | OpKind::SqrElementwise
            | OpKind::SqrtElementwise
            | OpKind::ExpElementwise
            | OpKind::LogElementwise
            | OpKind::SinElementwise
            | OpKind::CosElementwise
            | OpKind::TanhElementwise
            | OpKind::SigmoidElementwise
            | OpKind::SiluElementwise
            | OpKind::GeluElementwise
            | OpKind::ReluElementwise
            | OpKind::StepElementwise
            | OpKind::RecipElementwise
            | OpKind::AbsElementwise
            | OpKind::FloorElementwise
            | OpKind::CeilElementwise
            | OpKind::RoundElementwise
            | OpKind::SignElementwise
            | OpKind::ErfElementwise
            | OpKind::GeluErfElementwise
            | OpKind::RsqrtElementwise
            | OpKind::Affine
            | OpKind::ClampElementwise
            | OpKind::PowIElementwise => vec![
                OpSize::Elementwise(1 << 10),
                OpSize::Elementwise(1 << 16),
                OpSize::Elementwise(1 << 20),
            ],
            // Per-axis reductions: probe last-dim reductions over a
            // `[rows, cols]` shape with cols=64 (typical hidden-dim
            // chunk). Total elements 1 KiB / 64 KiB / 1 MiB to align
            // with the elementwise size ladder for cross-family
            // size_class comparison in the dispatch table.
            OpKind::SumReduce
            | OpKind::MaxReduce
            | OpKind::MinReduce
            | OpKind::MeanReduce => vec![
                OpSize::Reduce { rows: 1 << 4,  cols: 64 },
                OpSize::Reduce { rows: 1 << 10, cols: 64 },
                OpSize::Reduce { rows: 1 << 14, cols: 64 },
            ],
            // Reduce-to-broadcast-target — reduces a `[rows, cols]`
            // input to a `[1, cols]` output (sum/max along leading
            // dim). The canonical autograd-backward shape; broader
            // patterns (multi-axis, mid-rank insertion) follow once
            // the dispatch tables can carry them.
            OpKind::ReduceSumTo
            | OpKind::ReduceMaxTo => vec![
                OpSize::ReduceTo { rows: 1 << 4,  cols: 64 },
                OpSize::ReduceTo { rows: 1 << 10, cols: 64 },
                OpSize::ReduceTo { rows: 1 << 14, cols: 64 },
            ],
            // OpKind is `#[non_exhaustive]` — future variants land
            // here until the Judge gets a measurement strategy for
            // them. Empty plan = "Judge skips this op silently."
            _ => Vec::new(),
        }
    }

    /// Run the full profile matrix for every equivalence class in
    /// `probe`. Skips backends that aren't compiled in; logs (stderr)
    /// any backend that's present but hasn't been wired into the
    /// Judge yet (e.g. Vulkan in this revision).
    ///
    /// **Reference-retirement note (2026-06-07)**: the loop order is
    /// inverted relative to the pre-retirement shape — outer over
    /// `(op, size)`, inner over equivalence classes. This lets Judge
    /// collect all backends' outputs at a given cell, compute pairwise
    /// consensus across them, and assign each entry's `max_rel_error`
    /// relative to the consensus cluster (no privileged "reference
    /// output" required). When only one backend is available at a
    /// cell, consensus is trivial and `max_rel_error = 0.0` (no peers
    /// to compare against — the caller treats the lone backend as
    /// authoritative).
    pub fn run(&self, probe: &ProbeReport) -> ProfileReport {
        let mut entries = Vec::new();
        let classes = probe.equivalence_classes();
        // Stable iteration order over equivalence classes — sort
        // class keys so the profile report is deterministic across
        // runs (HashMap iteration order is not guaranteed across
        // process invocations).
        let mut class_keys: Vec<_> = classes.keys().collect();
        class_keys.sort_by_key(|k| (k.backend as u32, k.device_id, k.vendor_id));

        for &op in PROFILED_OPS {
            for sz in self.size_plan(op) {
                let size_class = SizeClass::from_elem_count(sz.total_elements());

                // First pass: per equivalence-class representative,
                // measure latency + capture the kernel's output for
                // consensus comparison.
                let mut cell_runs: Vec<CellRun> = Vec::with_capacity(class_keys.len());
                for key in &class_keys {
                    let devs = &classes[key];
                    let rep = devs[0];
                    if let Some(run) = self.measure_on_device_capturing(op, DType::F32, &sz, rep) {
                        cell_runs.push(run);
                    }
                }

                // Second pass: compute pairwise consensus across the
                // backends that produced output for this cell.
                let consensus = compute_pairwise_consensus(&cell_runs);

                // Third pass: emit one ProfileEntry per device,
                // replicated across each equivalence class. The
                // `max_rel_error` is derived from consensus position.
                for (i, run) in cell_runs.iter().enumerate() {
                    let rel_err =
                        max_rel_err_vs_consensus(&cell_runs, &consensus, i);
                    // Find the equivalence class this run belongs to
                    // (its representative's backend matches `run.backend`).
                    let class_devs = classes.iter().find_map(|(k, devs)| {
                        if k.backend == run.backend
                            && devs[0].device_index == run.device_index
                        {
                            Some(devs)
                        } else {
                            None
                        }
                    });
                    if let Some(devs) = class_devs {
                        for d in devs {
                            entries.push(ProfileEntry {
                                op,
                                dtype: DType::F32,
                                size_class,
                                backend: run.backend,
                                device_index: d.device_index,
                                latency_ns: run.latency_ns,
                                iterations: run.iterations,
                                max_rel_error: rel_err,
                            });
                        }
                    }
                }
            }
        }

        ProfileReport { version: PROFILE_REPORT_VERSION, entries }
    }

    /// Measure one (op, dtype, size) cell on one representative
    /// device, capturing the kernel's output alongside the timing
    /// data so the caller can compute pairwise consensus across
    /// backends. Returns `None` if the backend isn't compiled in,
    /// its constructor errors out, or the realize itself fails.
    ///
    /// **Reference-retirement note (2026-06-07)**: replaces the
    /// pre-retirement `measure_on_device` which compared against
    /// `tensor.realize_f32_reference()` per-cell. The new flow
    /// defers correctness comparison to [`Self::run`]'s consensus
    /// pass.
    fn measure_on_device_capturing(
        &self,
        op: OpKind,
        dtype: DType,
        size: &OpSize,
        device: &DeviceDescriptor,
    ) -> Option<CellRun> {
        assert_eq!(dtype, DType::F32, "judge: only f32 wired for now");

        // Walk the factory registry instead of naming each backend.
        // A backend that isn't compiled in simply doesn't appear in
        // the registry; one whose constructor fails is logged and
        // skipped per-device.
        let factory = match crate::factories::factory_for(device.backend) {
            Some(f) => f,
            None => {
                eprintln!(
                    "judge: skipping {}:{} — backend not compiled in",
                    device.backend, device.device_index,
                );
                return None;
            }
        };
        let mut realizer = match factory.try_make_realizer(device.device_index) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "judge: skipping {}:{} — factory failed: {e}",
                    device.backend, device.device_index,
                );
                return None;
            }
        };

        let cell = self.time_op_capturing(op, size, device, realizer.as_mut());
        drop(realizer);
        cell
    }

    /// Build the op's input graph, realize it for warmup + timed
    /// runs, measure latency, capture the first iteration's output.
    /// Returns the captured output alongside the timing data; the
    /// caller computes correctness via consensus across backends
    /// (see [`Self::run`]).
    ///
    /// All realize calls are wrapped in `catch_unwind` — a backend
    /// that doesn't yet implement the op (or panics on a corner-
    /// case input) is logged and skipped, not fatal to the run.
    /// This is what makes the Judge safe to expand across the full
    /// op surface ahead of every backend's coverage.
    fn time_op_capturing(
        &self,
        op: OpKind,
        size: &OpSize,
        device: &DeviceDescriptor,
        realizer: &mut dyn crate::factories::LazyRealizer,
    ) -> Option<CellRun> {
        let tensor = match std::panic::catch_unwind(AssertUnwindSafe(|| {
            build_input_graph(op, size)
        })) {
            Ok(t) => t,
            Err(_) => {
                eprintln!(
                    "judge: skipping {op}@{size:?} — build_input_graph panicked",
                );
                return None;
            }
        };

        // Warmup — discard timings; stabilize kernel caches, warm
        // up any BLAS internal state, fault-in heap. If any warmup
        // call panics (e.g. backend doesn't support this op), skip
        // the entire (op, backend) cell.
        for _ in 0..self.warmup {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                realizer.realize_f32(&tensor)
            }));
            if r.is_err() {
                eprintln!(
                    "judge: skipping {op}@{size:?} on {}:{} — backend realize panicked",
                    device.backend, device.device_index,
                );
                return None;
            }
        }

        // Timed iterations — record each and take the median so
        // one pre-empted scheduler tick doesn't skew the number.
        // Capture the first iteration's output for consensus
        // comparison; subsequent iterations discard the output
        // (correctness is deterministic for a given backend + input).
        let mut timings_ns = Vec::with_capacity(self.iterations as usize);
        let mut captured_output: Option<Vec<f32>> = None;
        for _ in 0..self.iterations {
            let t0 = Instant::now();
            let out = match std::panic::catch_unwind(AssertUnwindSafe(|| {
                realizer.realize_f32(&tensor)
            })) {
                Ok(v) => v,
                Err(_) => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} — backend realize panicked mid-run",
                        device.backend, device.device_index,
                    );
                    return None;
                }
            };
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            timings_ns.push(elapsed_ns);
            if captured_output.is_none() {
                captured_output = Some(out);
            }
        }
        timings_ns.sort_unstable();
        let median = timings_ns[timings_ns.len() / 2];
        let output = captured_output?;

        Some(CellRun {
            backend: device.backend,
            device_index: device.device_index,
            output,
            latency_ns: median,
            iterations: self.iterations,
        })
    }
}

/// Per-op size descriptor. Exposed (not private) so tests can supply
/// a shrunk `size_plan_override`; not in the persisted profile
/// report shape — the report uses the bucketed [`SizeClass`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpSize {
    /// Dense matmul `[m, k] @ [k, n] -> [m, n]`.
    MatMul { m: usize, n: usize, k: usize },
    /// Flat 1-D shape used by unary, binary, scalar/affine, clamp, powi
    /// kernels. Probe builds inputs of shape `[n]`.
    Elementwise(usize),
    /// Reduction along the last dim of a 2-D shape `[rows, cols]`.
    /// Total element count = rows × cols. Used for `SumReduce` /
    /// `MaxReduce` / `MinReduce` / `MeanReduce`.
    Reduce { rows: usize, cols: usize },
    /// Reduce-to-broadcast-target from `[rows, cols]` → `[1, cols]`
    /// (sum-reduce-along-rows). Used for `ReduceSumTo` / `ReduceMaxTo`.
    ReduceTo { rows: usize, cols: usize },
}

impl OpSize {
    fn total_elements(&self) -> usize {
        match *self {
            OpSize::MatMul { m, n, k: _ } => m * n,
            OpSize::Elementwise(n) => n,
            OpSize::Reduce { rows, cols } => rows * cols,
            OpSize::ReduceTo { rows, cols } => rows * cols,
        }
    }
}

/// Build a 1-node graph for the given (op, size) that takes constant
/// inputs. The inputs are deterministic so precision comparisons
/// across backends are meaningful.
fn build_input_graph(op: OpKind, size: &OpSize) -> crate::lazy::LazyTensor {
    use crate::lazy::LazyTensor;
    match (op, *size) {
        (OpKind::MatMul, OpSize::MatMul { m, n, k }) => {
            let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
            let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
            let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]), &crate::Device::cpu());
            let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
            a.matmul(&b).unwrap()
        }
        (op, OpSize::Elementwise(n)) if is_binary_elementwise(op) => {
            let (a_data, b_data) = binary_inputs(op, n);
            let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[n]), &crate::Device::cpu());
            let b = a.const_f32_like(b_data, Shape::from_dims(&[n]));
            apply_binary(op, &a, &b)
        }
        // -------- elementwise unary fanout --------
        //
        // Domain notes:
        // - sqrt/log require strictly positive inputs; inputs are in
        //   `[0.5, 2.5]` after the +1.5 offset.
        // - exp/sigmoid saturate quickly; inputs are bounded
        //   `[-1, 1]` (sin output) so exp stays in `[0.36, 2.72]` and
        //   sigmoid stays away from saturation (more measurable
        //   precision differences across backends).
        // - silu/gelu/relu/tanh/neg/sqr/step/sin/cos all accept any f32.
        // -------- per-axis reductions --------
        //
        // Reduce along the last dim of `[rows, cols]`, producing
        // `[rows]`. Bucketing on total element count means the
        // dispatch table groups `Reduce(rows=1024, cols=64)` and
        // `Elementwise(65536)` in the same size class — fine, the
        // size_class is the per-op axis, not a cross-op identity.
        (op, OpSize::Reduce { rows, cols }) if is_reduction(op) => {
            let n = rows * cols;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.7e-3).sin()).collect();
            let a = LazyTensor::from_f32(
                data,
                Shape::from_dims(&[rows, cols]),
                &crate::Device::cpu(),
            );
            apply_reduction(op, &a)
        }
        // -------- reduce-to-broadcast-target --------
        //
        // Reduce `[rows, cols]` to `[1, cols]`.
        (op, OpSize::ReduceTo { rows, cols }) if is_reduce_to(op) => {
            let n = rows * cols;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.7e-3).sin()).collect();
            let a = LazyTensor::from_f32(
                data,
                Shape::from_dims(&[rows, cols]),
                &crate::Device::cpu(),
            );
            let target = Shape::from_dims(&[1, cols]);
            match op {
                OpKind::ReduceSumTo => a.reduce_sum_to(target),
                OpKind::ReduceMaxTo => a.reduce_max_to(target),
                _ => unreachable!(),
            }
        }
        // -------- scalar / clamp / powi (one-input non-unary) --------
        //
        // Affine here is the canonical MulScalar form (mul=2.0); the
        // dispatch key OpKind::Affine covers AddScalar too. Clamp uses
        // bounds [-0.5, 0.5] so roughly half the [-1, 1] sin input
        // gets clipped, exercising both branches of the kernel. PowI
        // uses exp=3 so the kernel iterates rather than degenerating
        // to sqr/identity.
        (op, OpSize::Elementwise(n)) if is_scalar_op(op) => {
            let data: Vec<f32> = unary_input(n);
            let a = LazyTensor::from_f32(data, Shape::from_dims(&[n]), &crate::Device::cpu());
            match op {
                OpKind::Affine           => a.mul_scalar(2.0),
                OpKind::ClampElementwise => a.clamp(-0.5, 0.5),
                OpKind::PowIElementwise  => a.powi(3),
                _ => unreachable!(),
            }
        }
        (op, OpSize::Elementwise(n)) if is_unary_elementwise(op) => {
            let raw = unary_input(n);
            // Sqrt/Log require strictly positive inputs; Recip can't be
            // measured against zero (1/0 = inf saturates `max_rel_err`).
            // The +1.5 shift puts unary_input's [-1, 1] range into
            // [0.5, 2.5] — safe for all three.
            let needs_nonzero = matches!(
                op,
                OpKind::SqrtElementwise
                | OpKind::LogElementwise
                | OpKind::RecipElementwise
                | OpKind::RsqrtElementwise,
            );
            let data: Vec<f32> = if needs_nonzero {
                raw.into_iter().map(|x| x + 1.5).collect()
            } else {
                raw
            };
            let a = LazyTensor::from_f32(data, Shape::from_dims(&[n]), &crate::Device::cpu());
            apply_unary(op, &a)
        }
        (op, sz) => panic!("build_input_graph: op {op:?} size {sz:?} not implemented"),
    }
}

/// Whether `op` is a one-input scalar / clamp / powi op the Judge
/// profiles using the `Elementwise(n)` size ladder. These differ
/// from the unary ops only in carrying static parameters on the op
/// itself (scalar coefficient, clamp bounds, integer exponent).
fn is_scalar_op(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Affine
        | OpKind::ClampElementwise
        | OpKind::PowIElementwise,
    )
}

/// Whether `op` is a reduce-to-broadcast-target op the Judge profiles
/// using the `ReduceTo { rows, cols }` size ladder.
fn is_reduce_to(op: OpKind) -> bool {
    matches!(op, OpKind::ReduceSumTo | OpKind::ReduceMaxTo)
}

/// Whether `op` is a per-axis reduction the Judge profiles using the
/// `Reduce { rows, cols }` size ladder.
fn is_reduction(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::SumReduce
        | OpKind::MaxReduce
        | OpKind::MinReduce
        | OpKind::MeanReduce,
    )
}

/// Dispatch one per-axis reduction along the last dim. The reduced
/// dim is removed from the output shape.
fn apply_reduction(op: OpKind, a: &crate::lazy::LazyTensor) -> crate::lazy::LazyTensor {
    // Reduce the last dim. Input is rank-2 `[rows, cols]`; output is
    // rank-1 `[rows]`.
    let last_dim = a.rank() - 1;
    match op {
        OpKind::SumReduce  => a.sum_dim(last_dim).unwrap(),
        OpKind::MaxReduce  => a.max_dim(last_dim).unwrap(),
        OpKind::MinReduce  => a.min_dim(last_dim).unwrap(),
        OpKind::MeanReduce => a.mean_dim(last_dim).unwrap(),
        _ => unreachable!("apply_reduction called on non-reduction OpKind {op:?}"),
    }
}

/// Whether `op` is one of the elementwise binary ops the Judge profiles
/// using the `Elementwise(n)` size ladder.
fn is_binary_elementwise(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise
        | OpKind::DivElementwise
        | OpKind::MaximumElementwise
        | OpKind::MinimumElementwise
        | OpKind::PowElementwise
        | OpKind::RemElementwise,
    )
}

/// Deterministic two-input data for binary ops. `a` is `sin(i*2.1e-3)`
/// and `b` is `cos(i*1.9e-3)` for most ops; for Div, `b` is shifted
/// away from zero (`+ 1.5`, range `[0.5, 2.5]`) to avoid division by
/// near-zero values that would saturate `max_rel_err` to infinity on
/// any tiny precision difference.
fn binary_inputs(op: OpKind, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a: Vec<f32> = (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect();
    let mut b: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.9e-3).cos()).collect();
    if matches!(op, OpKind::DivElementwise) {
        for x in &mut b { *x += 1.5; }
    }
    if matches!(op, OpKind::PowElementwise) {
        // Both inputs must be positive: pow(neg, non-int) = NaN under
        // IEEE-754 and saturates max_rel_err. Shift both to [0.5, 2.5].
        for x in &mut a { *x += 1.5; }
        for x in &mut b { *x += 1.5; }
    }
    if matches!(op, OpKind::RemElementwise) {
        // Divisor must be away from zero (`a / b` blows up otherwise).
        for x in &mut b { *x += 1.5; }
    }
    (a, b)
}

/// Dispatch one elementwise binary op against `(a, b)`.
fn apply_binary(
    op: OpKind,
    a: &crate::lazy::LazyTensor,
    b: &crate::lazy::LazyTensor,
) -> crate::lazy::LazyTensor {
    match op {
        OpKind::AddElementwise     => a.add(b).unwrap(),
        OpKind::SubElementwise     => a.sub(b).unwrap(),
        OpKind::MulElementwise     => a.mul(b).unwrap(),
        OpKind::DivElementwise     => a.div(b).unwrap(),
        OpKind::MaximumElementwise => a.maximum(b).unwrap(),
        OpKind::MinimumElementwise => a.minimum(b).unwrap(),
        // Pow/Rem return Result; expect() in Judge's measurement
        // path is fine — inputs are constructed locally here, so a
        // shape/dtype mismatch is a programming bug in the test
        // harness, not a runtime user input.
        OpKind::PowElementwise     => a.pow(b).expect("judge: pow shape/dtype invariant"),
        OpKind::RemElementwise     => a.rem(b).expect("judge: rem shape/dtype invariant"),
        _ => unreachable!("apply_binary called on non-binary OpKind {op:?}"),
    }
}

/// Whether `op` is one of the elementwise unary ops the Judge profiles
/// using the `Elementwise(n)` size ladder.
fn is_unary_elementwise(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::NegElementwise
        | OpKind::SqrElementwise
        | OpKind::SqrtElementwise
        | OpKind::ExpElementwise
        | OpKind::LogElementwise
        | OpKind::SinElementwise
        | OpKind::CosElementwise
        | OpKind::TanhElementwise
        | OpKind::SigmoidElementwise
        | OpKind::SiluElementwise
        | OpKind::GeluElementwise
        | OpKind::ReluElementwise
        | OpKind::StepElementwise
        | OpKind::RecipElementwise
        | OpKind::AbsElementwise
        | OpKind::FloorElementwise
        | OpKind::CeilElementwise
        | OpKind::RoundElementwise
        | OpKind::SignElementwise
        | OpKind::ErfElementwise
        | OpKind::GeluErfElementwise
        | OpKind::RsqrtElementwise,
    )
}

/// Deterministic unary-op input. `sin(i * 2.1e-3)` produces values in
/// `[-1, 1]`, varied enough that argmax/argmin/relu/step all see a
/// realistic mix of positive and negative values.
fn unary_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect()
}

/// Dispatch one elementwise unary op against `a`.
fn apply_unary(op: OpKind, a: &crate::lazy::LazyTensor) -> crate::lazy::LazyTensor {
    match op {
        OpKind::NegElementwise     => a.neg(),
        OpKind::SqrElementwise     => a.sqr(),
        OpKind::SqrtElementwise    => a.sqrt(),
        OpKind::ExpElementwise     => a.exp(),
        OpKind::LogElementwise     => a.log(),
        OpKind::TanhElementwise    => a.tanh(),
        OpKind::SigmoidElementwise => a.sigmoid(),
        OpKind::SiluElementwise    => a.silu(),
        OpKind::GeluElementwise    => a.gelu(),
        OpKind::ReluElementwise    => a.relu(),
        OpKind::SinElementwise     => a.sin(),
        OpKind::CosElementwise     => a.cos(),
        OpKind::StepElementwise    => a.step(),
        OpKind::RecipElementwise   => a.recip(),
        OpKind::AbsElementwise     => a.abs(),
        OpKind::FloorElementwise   => a.floor(),
        OpKind::CeilElementwise    => a.ceil(),
        OpKind::RoundElementwise   => a.round(),
        OpKind::SignElementwise    => a.sign(),
        OpKind::ErfElementwise     => a.erf(),
        OpKind::GeluErfElementwise => a.gelu_erf(),
        OpKind::RsqrtElementwise   => a.rsqrt(),
        _ => unreachable!("apply_unary called on non-unary OpKind {op:?}"),
    }
}

/// Max element-wise relative error between two same-length f32
/// vectors. Denominator guards against divide-by-zero on mutually-
/// zero pairs. Returns 0.0 when the lengths don't match (caller is
/// responsible for length sanity — this function is in the hot
/// path and avoids panics).
fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut worst = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return f32::INFINITY;
        }
        let denom = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        let rel = (x - y).abs() / denom;
        if rel > worst { worst = rel; }
    }
    worst
}

// =============================================================
// Pairwise consensus — Reference-retirement replacement (2026-06-07)
// =============================================================
//
// Pre-retirement, Judge compared every backend's output against
// `tensor.realize_f32_reference()` to populate `ProfileEntry::max_rel_error`.
// That privileged the `fuel-reference-backend` crate as THE oracle —
// the architectural decision in v0.2 of `05-backend-contract.md`
// demoted Reference but the code lagged.
//
// Post-retirement, Judge runs every available backend's kernel for
// each (op, dtype, size) cell, clusters the outputs by mutual
// `rel_err < CONSENSUS_EPSILON`, and computes each backend's
// `max_rel_error` against the largest cluster (the "consensus
// group"). A backend in the consensus group reports its max
// within-cluster rel_err (typically near 0); a backend outside
// the consensus reports its rel_err against the cluster — that's
// the outlier signal callers want.

/// Generous epsilon for cross-backend consensus clustering.
/// Different backends produce numerically-different bits for the
/// same logical operation (IEEE rounding rules + accumulation
/// order differ across BLAS impls, GPU shader cores, etc.); the
/// cluster definition has to accommodate this. `1e-3` is loose
/// enough that floating-point accumulation reordering won't push
/// honest implementations out of consensus while still catching
/// kernels that are silently wrong.
///
/// Per-op tightening can land later via per-op
/// `PrecisionGuarantee::max_relative` if a particular op family
/// warrants tighter clustering. The 1e-3 default matches the
/// `assert!(e.max_rel_error < 5e-3)` thresholds used in Judge's
/// own tests below — both numbers reflect "this is the noise
/// floor for honest cross-backend implementations of f32 math."
const CONSENSUS_EPSILON: f32 = 1e-3;

/// One backend's measurement for one (op, dtype, size) cell —
/// kernel output captured alongside latency for downstream
/// consensus comparison.
#[derive(Debug)]
struct CellRun {
    backend: BackendId,
    device_index: u32,
    output: Vec<f32>,
    latency_ns: u64,
    iterations: u32,
}

/// Compute the largest mutually-close cluster across `runs`. Returns
/// indices into `runs` (sorted ascending) of the consensus members.
///
/// Algorithm: for each run i, build the set of runs that pairwise
/// agree with i (and with each other). The largest such set is the
/// consensus. Ties broken by lowest first-index — deterministic.
///
/// Edge cases:
/// - Empty `runs`: empty consensus.
/// - Single run: trivial consensus of [0]. The lone backend is
///   authoritative by absence of peers.
/// - Two runs agreeing within `CONSENSUS_EPSILON`: consensus is
///   both; each reports its rel_err against the other (small).
/// - Two runs disagreeing: consensus is [0] (first wins ties by
///   index); the other gets the disagreement as its rel_err.
///   Callers should treat 2-backend cells with disagreement as
///   "human review needed" — neither answer is independently
///   validated.
/// - Three runs with one outlier: consensus is the two that agree;
///   the outlier reports its rel_err against the cluster.
fn compute_pairwise_consensus(runs: &[CellRun]) -> Vec<usize> {
    let n = runs.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Pairwise rel_err matrix: agree[i][j] = (max_rel_err < CONSENSUS_EPSILON).
    let mut agree = vec![vec![false; n]; n];
    for i in 0..n {
        agree[i][i] = true;
        for j in (i + 1)..n {
            let close = max_rel_err(&runs[i].output, &runs[j].output) < CONSENSUS_EPSILON;
            agree[i][j] = close;
            agree[j][i] = close;
        }
    }

    // For each i, expand a cluster greedily: start with {i}, add
    // each j ∈ neighbors that pairwise-agrees with all current
    // members. Greedy — but for small N (typical: 2-4 backends),
    // exhaustive search would also be cheap. Greedy is correct
    // when the threshold defines a true equivalence-like relation;
    // for f32 cross-backend agreement that's a reasonable assumption.
    let mut best: Vec<usize> = vec![0];
    for i in 0..n {
        let mut cluster = vec![i];
        for j in 0..n {
            if j == i {
                continue;
            }
            // j joins iff agrees with every existing member.
            if cluster.iter().all(|&k| agree[j][k]) {
                cluster.push(j);
            }
        }
        cluster.sort_unstable();
        // Larger cluster wins; ties broken by lower starting index.
        if cluster.len() > best.len() {
            best = cluster;
        }
    }
    best
}

/// Compute `runs[self_idx]`'s max relative error against the
/// consensus cluster. When the consensus contains only `self_idx`
/// (single backend or every backend disagreed), returns `0.0` —
/// there's no peer to measure drift against.
fn max_rel_err_vs_consensus(
    runs: &[CellRun],
    consensus: &[usize],
    self_idx: usize,
) -> f32 {
    let peers: Vec<usize> = consensus
        .iter()
        .copied()
        .filter(|&i| i != self_idx)
        .collect();
    if peers.is_empty() {
        return 0.0;
    }
    let self_out = &runs[self_idx].output;
    let mut worst = 0.0_f32;
    for i in peers {
        let e = max_rel_err(self_out, &runs[i].output);
        if e > worst {
            worst = e;
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_class_matches_log2_floor() {
        assert_eq!(SizeClass::from_elem_count(1).0, 0);
        assert_eq!(SizeClass::from_elem_count(2).0, 1);
        assert_eq!(SizeClass::from_elem_count(3).0, 1);
        assert_eq!(SizeClass::from_elem_count(4).0, 2);
        assert_eq!(SizeClass::from_elem_count(1 << 16).0, 16);
        assert_eq!(SizeClass::from_elem_count(65_537).0, 16);
    }

    #[test]
    fn size_class_zero_input_clamps_to_one() {
        assert_eq!(SizeClass::from_elem_count(0).0, 0);
    }

    #[test]
    fn op_kind_display_is_stable() {
        assert_eq!(OpKind::MatMul.to_string(), "matmul");
        assert_eq!(OpKind::AddElementwise.to_string(), "add");
    }

    #[test]
    fn judge_profiles_cpu_on_small_matmul() {
        // Post-Reference-retirement (2026-06-07): with Reference
        // gone, the local single-backend (CPU) is alone in its
        // (op, dtype, size) cell. Consensus is trivially `[0]`;
        // `max_rel_error` is `0.0` (no peers to measure drift
        // against). The test verifies the Judge still produces
        // entries with sane timing data when only one backend is
        // available — the single-backend edge case in the
        // consensus algorithm.
        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                (OpKind::AddElementwise, OpSize::Elementwise(1 << 10)),
            ]),
        };
        let report = judge.run(&probe);
        assert_eq!(report.version, PROFILE_REPORT_VERSION);
        assert!(report.entries.iter().any(|e| e.backend == BackendId::Cpu));
        // Single-backend cells: rel_err is 0.0 by convention (no
        // peers means no comparison reference). Latency is real.
        for e in report.entries.iter().filter(|e| e.backend == BackendId::Cpu) {
            assert_eq!(e.max_rel_error, 0.0,
                "single-backend cell should report 0.0 rel_err: {e:?}");
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_unary_elementwise_ops() {
        // Exercise every elementwise-unary OpKind on cpu + reference at
        // a tiny size. Confirms the size_plan / build_input_graph /
        // apply_unary wiring is complete and that no unary kernel
        // diverges wildly from reference on the cpu backend.
        let probe = ProbeReport::probe_all();
        let unary = [
            OpKind::NegElementwise, OpKind::SqrElementwise, OpKind::SqrtElementwise,
            OpKind::ExpElementwise, OpKind::LogElementwise, OpKind::SinElementwise,
            OpKind::CosElementwise, OpKind::TanhElementwise, OpKind::SigmoidElementwise,
            OpKind::SiluElementwise, OpKind::GeluElementwise, OpKind::ReluElementwise,
            OpKind::StepElementwise, OpKind::RecipElementwise, OpKind::AbsElementwise,
            OpKind::FloorElementwise, OpKind::CeilElementwise,
            OpKind::RoundElementwise, OpKind::SignElementwise,
            OpKind::ErfElementwise,   OpKind::GeluErfElementwise,
            OpKind::RsqrtElementwise,
        ];
        let plan: Vec<_> = unary.iter()
            .map(|&op| (op, OpSize::Elementwise(1 << 8)))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
        };
        let report = judge.run(&probe);
        for &op in &unary {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert_eq!(cpu_entries.len(), 1,
                "expected one cpu entry for {op}, got {}", cpu_entries.len());
            let e = cpu_entries[0];
            // Elementwise unary ops are bit-stable on cpu vs reference
            // (no accumulation order to worry about). Allow a
            // generous bound so transcendental approximations
            // (tanh/sigmoid/silu/gelu) on different math libs still
            // pass — but a runaway divergence (>1e-3) flags a bug.
            assert!(e.max_rel_error < 1e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_binary_elementwise_ops() {
        let probe = ProbeReport::probe_all();
        let binary = [
            OpKind::AddElementwise, OpKind::SubElementwise, OpKind::MulElementwise,
            OpKind::DivElementwise, OpKind::MaximumElementwise, OpKind::MinimumElementwise,
            OpKind::PowElementwise, OpKind::RemElementwise,
        ];
        let plan: Vec<_> = binary.iter()
            .map(|&op| (op, OpSize::Elementwise(1 << 8)))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
        };
        let report = judge.run(&probe);
        for &op in &binary {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert_eq!(cpu_entries.len(), 1, "expected one cpu entry for {op}");
            let e = cpu_entries[0];
            assert!(e.max_rel_error < 1e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_reductions() {
        let probe = ProbeReport::probe_all();
        let reduce = [
            OpKind::SumReduce, OpKind::MaxReduce,
            OpKind::MinReduce, OpKind::MeanReduce,
        ];
        let plan: Vec<_> = reduce.iter()
            .map(|&op| (op, OpSize::Reduce { rows: 16, cols: 16 }))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
        };
        let report = judge.run(&probe);
        for &op in &reduce {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert_eq!(cpu_entries.len(), 1, "expected one cpu entry for {op}");
            let e = cpu_entries[0];
            // Sum-style reductions accumulate in different order on
            // backend vs reference; allow up to 5e-3.
            assert!(e.max_rel_error < 5e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_reduce_to() {
        let probe = ProbeReport::probe_all();
        let reduce_to = [OpKind::ReduceSumTo, OpKind::ReduceMaxTo];
        let plan: Vec<_> = reduce_to.iter()
            .map(|&op| (op, OpSize::ReduceTo { rows: 16, cols: 16 }))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
        };
        let report = judge.run(&probe);
        for &op in &reduce_to {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert_eq!(cpu_entries.len(), 1, "expected one cpu entry for {op}");
            let e = cpu_entries[0];
            assert!(e.max_rel_error < 5e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn judge_profiles_all_scalar_ops() {
        let probe = ProbeReport::probe_all();
        let scalar = [
            OpKind::Affine, OpKind::ClampElementwise, OpKind::PowIElementwise,
        ];
        let plan: Vec<_> = scalar.iter()
            .map(|&op| (op, OpSize::Elementwise(1 << 8)))
            .collect();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(plan),
        };
        let report = judge.run(&probe);
        for &op in &scalar {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert_eq!(cpu_entries.len(), 1, "expected one cpu entry for {op}");
            let e = cpu_entries[0];
            assert!(e.max_rel_error < 1e-3,
                "cpu vs reference disagreement on {op}: rel_err={}",
                e.max_rel_error);
            assert!(e.latency_ns > 0);
        }
    }

    #[test]
    fn dispatch_table_built_from_expanded_report_serves_multiple_kinds() {
        // Confirms the DispatchTable's O(1) lookup path handles the
        // expanded OpKind coverage. The route picker can now pick
        // among many more (op, size_class) cells than the original
        // matmul + add report supported.
        use fuel_core_types::dispatch::{Criterion, DispatchTable};

        let probe = ProbeReport::probe_all();
        let judge = Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                (OpKind::AddElementwise, OpSize::Elementwise(1 << 8)),
                (OpKind::ReluElementwise, OpSize::Elementwise(1 << 8)),
                (OpKind::SumReduce, OpSize::Reduce { rows: 16, cols: 16 }),
                (OpKind::ReduceSumTo, OpSize::ReduceTo { rows: 16, cols: 16 }),
                (OpKind::Affine, OpSize::Elementwise(1 << 8)),
            ]),
        };
        let report = judge.run(&probe);
        let table = DispatchTable::build(&report);

        // Every kind in the plan should yield a Pick at every
        // criterion. SizeClass(8) covers the elementwise plan
        // (256 elems = 2^8) and the reductions (16*16 = 2^8); the
        // matmul cell is at SizeClass(10) (32*32 = 2^10).
        let elementwise_kinds = [
            OpKind::AddElementwise, OpKind::ReluElementwise,
            OpKind::SumReduce, OpKind::ReduceSumTo, OpKind::Affine,
        ];
        for &op in &elementwise_kinds {
            for &crit in &[Criterion::Fastest, Criterion::MostAccurate, Criterion::Balanced] {
                let pick = table.pick(op, DType::F32, SizeClass(8), crit);
                assert!(pick.is_some(),
                    "DispatchTable missing pick for {op}@2^8 / {crit:?}");
            }
        }
        let matmul_pick = table.pick(OpKind::MatMul, DType::F32, SizeClass(10), Criterion::Fastest);
        assert!(matmul_pick.is_some(), "DispatchTable missing matmul pick");

        // The table's keys() should include every (op, dtype,
        // size_class) tuple exercised by the plan.
        let keys = table.keys();
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::ReluElementwise && sc.0 == 8));
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::SumReduce && sc.0 == 8));
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::ReduceSumTo && sc.0 == 8));
        assert!(keys.iter().any(|(op, _, sc)| *op == OpKind::Affine && sc.0 == 8));
    }

    #[test]
    fn profile_report_save_load_roundtrip() {
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![ProfileEntry {
                op: OpKind::MatMul,
                dtype: DType::F32,
                size_class: SizeClass(8),
                backend: BackendId::Cpu,
                device_index: 0,
                latency_ns: 12_345,
                iterations: 7,
                max_rel_error: 1e-7,
            }],
        };
        let tmp = std::env::temp_dir().join(format!(
            "fuel-judge-test-{}.json", std::process::id()
        ));
        report.save(&tmp).expect("save");
        let loaded = ProfileReport::load(&tmp).expect("load").expect("file exists");
        assert_eq!(loaded, report);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn max_rel_err_matches_known_values() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        // f32 rounding in both `2.001` literals and the subtraction
        // means the exact numeric answer is not `0.001 / 2.001`. Pin
        // the answer loosely: should be within a few ULP of that
        // ratio and close to 5e-4.
        let got = max_rel_err(&[1.0, 2.0], &[1.0, 2.001]);
        assert!(
            (got - 5.0e-4).abs() < 1e-6,
            "expected ~0.0005, got {got}",
        );
        // Length mismatch → infinity.
        assert!(max_rel_err(&[1.0], &[1.0, 1.0]).is_infinite());
        // NaN in either operand → infinity.
        assert!(max_rel_err(&[f32::NAN], &[0.0]).is_infinite());
    }
}
