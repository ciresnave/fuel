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
use fuel_correctness_fixtures::{
    validate_against_fixture, CorrectnessDrift, CorrectnessFixture, FixtureFile,
};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Instant;
use std::sync::{Arc, RwLock};

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
    /// Optional pre-validated correctness fixtures keyed by
    /// `(op, dtype, size_class)`. When a fixture covers a profiling
    /// cell, the Judge validates every CellRun's output against it
    /// via [`validate_against_fixture`] and assigns each
    /// `ProfileEntry::max_rel_error` from the verdict, bypassing the
    /// inline pairwise-consensus path for that cell. Cells without a
    /// matching fixture fall back to the consensus path (today's
    /// default behavior on multi-backend systems).
    ///
    /// One key may map to multiple fixtures when capture used
    /// different `input_seed`s; the Judge applies the first whose
    /// fixture-shape and the cell's regenerated input agree. If no
    /// fixture matches the cell's input seed (mismatch logged), the
    /// fallback consensus path is used for the cell.
    pub fixtures: Option<HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>>>,
}

impl Default for Judge {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_ITERATIONS,
            warmup:     DEFAULT_WARMUP,
            size_plan_override: None,
            fixtures: None,
        }
    }
}

impl Judge {
    /// Construct a Judge with the default measurement parameters and
    /// the fixture set loaded from `path`. The path may point at a
    /// single `*.json` file or a directory; in the directory case
    /// every `*.json` under the root is loaded recursively (the
    /// `fuel-correctness-fixtures` capture binary writes
    /// `v1/<dtype>/<op>.json`, so a single root covers an entire
    /// fixture distribution).
    ///
    /// Returns an `io::Error` wrapped as a `fuel-core` `Error` if the
    /// path can't be read or any file fails to parse as a
    /// [`FixtureFile`].
    ///
    /// **Use**: pass a fixture root once at Judge construction time;
    /// every cell the Judge profiles thereafter validates against the
    /// fixture (if one exists for that cell) instead of inline
    /// consensus. Cells with no matching fixture fall back to the
    /// consensus path.
    pub fn with_fixtures_from(path: &Path) -> Result<Self> {
        let fixtures = load_fixtures_recursive(path)?;
        let mut judge = Self::default();
        judge.fixtures = Some(fixtures);
        Ok(judge)
    }

    /// Look up a fixture matching `(op, dtype, size_class)` and the
    /// expected element count of the cell's output. Returns `None`
    /// when no fixture set is configured, when no fixture is keyed
    /// for the cell, or when every keyed fixture has a mismatched
    /// `output_element_count` (which would indicate the fixture was
    /// captured against a different shape than the current size_plan
    /// emits — log + skip).
    fn lookup_fixture(
        &self,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
        expected_elem_count: usize,
    ) -> Option<&CorrectnessFixture> {
        let map = self.fixtures.as_ref()?;
        let bucket = map.get(&(op, dtype, size_class))?;
        for f in bucket {
            if f.output_element_count == expected_elem_count {
                return Some(f);
            }
        }
        // The bucket exists but no fixture matches the cell's shape.
        // The capture used a different input convention; skip + fall
        // back to consensus rather than emit a misleading rel_err.
        eprintln!(
            "judge: fixture bucket for ({op:?}, {dtype:?}, size_class={}) has no entry \
             matching expected_elem_count={expected_elem_count}; falling back to consensus",
            size_class.0,
        );
        None
    }

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
    ///
    /// **Per-alternative measurement (2026-06-08)**: for each
    /// equivalence-class representative the Judge now walks
    /// `KernelBindingTable::lookup_alternatives(op_kind, dtypes,
    /// backend)` and emits one `CellRun` per registered alternative.
    /// The first alternative is timed via the standard realizer path
    /// (matching pre-v2 behavior, since `lookup_with_caps` returns
    /// the first alternative — what the picker would dispatch).
    /// Subsequent alternatives at the same `(op, dtypes, backend)`
    /// key are timed via a direct kernel-pointer call so AOCL/MKL
    /// siblings get distinct latency numbers.
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
                // consensus comparison. The realizer path measures
                // the first alternative at the binding-table cell
                // (`lookup_with_caps`'s first-registered convention).
                let mut cell_runs: Vec<CellRun> = Vec::with_capacity(class_keys.len());
                for key in &class_keys {
                    let devs = &classes[key];
                    let rep = devs[0];
                    if let Some(mut run) = self.measure_on_device_capturing(op, DType::F32, &sz, rep) {
                        // Tag the realizer-measured run with the first
                        // alternative's `kernel_source` from the
                        // binding table — that's the kernel
                        // `lookup_with_caps` returned, which is what
                        // the realizer just dispatched.
                        run.kernel_source =
                            primary_kernel_source(op, rep.backend).to_string();
                        let primary_source = run.kernel_source.clone();
                        cell_runs.push(run);

                        // Per-alternative measurement: walk SUBSEQUENT
                        // alternatives at the same `(op_kind, dtypes,
                        // backend)` binding-table cell and time each
                        // via a direct kernel-pointer call. The
                        // primary alternative is already covered by
                        // the realizer measurement above.
                        if let Some(extra_runs) =
                            self.measure_extra_alternatives(op, &sz, rep, &primary_source)
                        {
                            for extra in extra_runs {
                                cell_runs.push(extra);
                            }
                        }
                    }
                }

                // Second pass: pick a correctness verdict for each
                // CellRun. The fixture fast-path is preferred when a
                // fixture exists for the cell — that's the
                // pre-validated multi-backend output captured on a
                // reference rig, so single-backend systems get the
                // same correctness signal without needing peers
                // locally. When no fixture is present, fall back to
                // pairwise consensus across the CellRuns at this
                // cell.
                let expected_elem_count = cell_runs.first()
                    .map(|r| r.output.len())
                    .unwrap_or(0);
                let fixture =
                    self.lookup_fixture(op, DType::F32, size_class, expected_elem_count);
                let consensus = if fixture.is_none() {
                    compute_pairwise_consensus(&cell_runs)
                } else {
                    Vec::new() // unused on the fixture path
                };

                // Third pass: emit one ProfileEntry per (run, device).
                // The first alternative's CellRun replicates across
                // each equivalence-class device; per-alternative
                // CellRuns from the binding-table direct path
                // record on the representative's device_index only
                // (the cross-device sharing convention applies to
                // backend-level equivalence, not within-cell kernel
                // sibling differentiation — siblings are
                // hardware-agnostic at the binding-table layer).
                for (i, run) in cell_runs.iter().enumerate() {
                    let rel_err = if let Some(f) = fixture {
                        max_rel_err_vs_fixture(&run.output, f)
                    } else {
                        max_rel_err_vs_consensus(&cell_runs, &consensus, i)
                    };
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
                                kernel_source: run.kernel_source.clone(),
                            });
                        }
                    }
                }
            }
        }

        ProfileReport { version: PROFILE_REPORT_VERSION, entries }
    }

    /// Walk binding-table alternatives at `(op_kind, dtypes, backend)`
    /// beyond the first (which the realizer already timed) and run a
    /// direct kernel-pointer measurement on each. Returns `None` when
    /// only one alternative is registered (no extra work) or when the
    /// op family is not yet wired into the direct-call path.
    ///
    /// **Why direct call instead of realizer**: the realizer dispatches
    /// through `lookup_with_caps`, which returns the FIRST registered
    /// alternative. To measure AOCL when MKL is first-registered, we
    /// have to bypass the picker entirely and invoke the specific
    /// `BindingEntry::kernel` function pointer with hand-built inputs.
    ///
    /// **Scope**: v1 supports the subset of [`PROFILED_OPS`] where
    /// input/output Storage + Layout + OpParams can be built without
    /// going through the lazy-tensor graph. Today: matmul, elementwise
    /// unary/binary, reductions, reduce-to, affine/clamp/powi. Other
    /// op families return `None` and only the realizer-measured first
    /// alternative is recorded for the cell.
    fn measure_extra_alternatives(
        &self,
        op: OpKind,
        size: &OpSize,
        device: &DeviceDescriptor,
        primary_kernel_source: &str,
    ) -> Option<Vec<CellRun>> {
        // Direct kernel-pointer calls are only meaningful on the CPU
        // backend today — CUDA / Vulkan storage handles are backend-
        // specific and the realizer-internal allocator hierarchy
        // doesn't expose a stand-alone "build CUDA storage from f32
        // slice" entry point at the binding-table layer.
        if device.backend != BackendId::Cpu {
            return None;
        }

        let alternatives = direct_call_alternatives(op, primary_kernel_source);
        if alternatives.is_empty() {
            return None;
        }

        // Build the inputs once — every alternative consumes the same
        // input data, so this is shared across the per-alt timing loop.
        let prepared = match prepare_direct_call_inputs(op, size) {
            Some(p) => p,
            None => return None,
        };

        let mut extra: Vec<CellRun> = Vec::with_capacity(alternatives.len());
        for alt in alternatives {
            if let Some(run) = self.time_alternative_direct(
                op, size, device, &prepared, &alt,
            ) {
                extra.push(run);
            }
        }
        if extra.is_empty() { None } else { Some(extra) }
    }

    /// Direct-call timing path. Bypasses the realizer; calls the
    /// supplied `BindingEntry::kernel` function pointer directly with
    /// the prepared inputs. `iterations` median + `warmup` discard
    /// applied identically to the realizer path.
    fn time_alternative_direct(
        &self,
        op: OpKind,
        size: &OpSize,
        device: &DeviceDescriptor,
        prepared: &PreparedDirectCall,
        alt: &DirectCallAlternative,
    ) -> Option<CellRun> {
        use fuel_dispatch::kernel::OpParams as _; // ensure visible in scope

        // Warmup — discard timings. A backend whose kernel panics
        // mid-warmup is treated as if the alternative didn't exist
        // (skip this CellRun); the primary realizer-measured run
        // still represents the cell.
        let kernel = alt.kernel;
        for _ in 0..self.warmup {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut outs = vec![Arc::clone(&prepared.output)];
                kernel(&prepared.inputs, &mut outs, &prepared.layouts, &prepared.op_params)
            }));
            if r.is_err() {
                eprintln!(
                    "judge: skipping {op}@{size:?} on {}:{} kernel_source={:?} — kernel panicked in warmup",
                    device.backend, device.device_index, alt.kernel_source,
                );
                return None;
            }
            // Match `time_op_capturing`'s call-twice tolerance for
            // panic-vs-return-Err: an alternative that returns Err is
            // a real signal (the kernel rejected the inputs) — skip
            // rather than abort.
            if let Ok(Err(_)) = r {
                eprintln!(
                    "judge: skipping {op}@{size:?} on {}:{} kernel_source={:?} — kernel returned Err in warmup",
                    device.backend, device.device_index, alt.kernel_source,
                );
                return None;
            }
        }

        let mut timings_ns = Vec::with_capacity(self.iterations as usize);
        let mut captured_output: Option<Vec<f32>> = None;
        for _ in 0..self.iterations {
            let t0 = Instant::now();
            let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut outs = vec![Arc::clone(&prepared.output)];
                kernel(&prepared.inputs, &mut outs, &prepared.layouts, &prepared.op_params)
            }));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            match res {
                Ok(Ok(())) => {}
                _ => {
                    eprintln!(
                        "judge: skipping {op}@{size:?} on {}:{} kernel_source={:?} — direct call failed mid-run",
                        device.backend, device.device_index, alt.kernel_source,
                    );
                    return None;
                }
            }
            timings_ns.push(elapsed_ns);
            if captured_output.is_none() {
                captured_output = Some(read_output_f32(&prepared.output));
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
            kernel_source: alt.kernel_source.to_string(),
        })
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
            // Filled in by `Judge::run` after this returns — the
            // binding-table lookup needs the BackendId which the
            // caller has but this function's narrow scope doesn't.
            kernel_source: String::new(),
        })
    }
}

// =============================================================
// Direct-kernel-call infrastructure for per-alternative timing
// (Phase 1.1 of the backend-extensions refactor, 2026-06-08)
// =============================================================

/// One alternative the direct-call path will time. Holds the
/// kernel function pointer and the `kernel_source` tag the resulting
/// CellRun carries.
struct DirectCallAlternative {
    kernel: fuel_dispatch::kernel::KernelRef,
    kernel_source: &'static str,
}

/// Shared inputs/outputs/layouts/op_params for one (op, size) cell.
/// Built once per cell and reused across every alternative's
/// direct-call measurement so the kernels compare on identical
/// inputs (a prerequisite for cross-alternative consensus to mean
/// anything).
struct PreparedDirectCall {
    inputs:    Vec<Arc<RwLock<fuel_storage::Storage>>>,
    output:    Arc<RwLock<fuel_storage::Storage>>,
    layouts:   Vec<fuel_core_types::Layout>,
    op_params: fuel_dispatch::kernel::OpParams,
}

/// Look up the `kernel_source` of the FIRST alternative at the
/// binding-table cell — that's what `lookup_with_caps` returns and
/// what the realizer just dispatched. Returns `""` when no
/// alternative is registered (the realizer would fall back to a
/// different code path).
fn primary_kernel_source(op: OpKind, backend: BackendId) -> &'static str {
    let dtypes = match canonical_binding_dtypes_for(op) {
        Some(d) => d,
        None => return "",
    };
    let table = fuel_dispatch::dispatch::global_bindings();
    let alts = table.lookup_alternatives(op, &dtypes, backend);
    alts.first().map(|e| e.kernel_source).unwrap_or("")
}

/// Collect every binding-table alternative at the cell EXCEPT the
/// primary (already timed via the realizer). Returns a fresh `Vec`
/// of [`DirectCallAlternative`] — empty when the cell has only one
/// alternative or the op isn't direct-call-eligible yet.
fn direct_call_alternatives(
    op: OpKind,
    primary_kernel_source: &str,
) -> Vec<DirectCallAlternative> {
    let dtypes = match canonical_binding_dtypes_for(op) {
        Some(d) => d,
        None => return Vec::new(),
    };
    let table = fuel_dispatch::dispatch::global_bindings();
    let alts = table.lookup_alternatives(op, &dtypes, BackendId::Cpu);
    if alts.len() < 2 {
        return Vec::new();
    }
    // Skip the primary (first alternative — which the realizer already
    // measured). `primary_kernel_source` matches its tag; subsequent
    // alternatives with distinct kernel_sources are the work.
    let mut extras = Vec::with_capacity(alts.len().saturating_sub(1));
    let mut skipped_primary = false;
    for alt in alts.iter() {
        if !skipped_primary && alt.kernel_source == primary_kernel_source {
            skipped_primary = true;
            continue;
        }
        extras.push(DirectCallAlternative {
            kernel: alt.kernel,
            kernel_source: alt.kernel_source,
        });
    }
    extras
}

/// Map an `OpKind` to its binding-table dtype list (inputs + outputs)
/// for the F32 single-dtype Judge profile. Returns `None` for op
/// families the direct-call path doesn't yet support — caller falls
/// back to first-alternative-only measurement at the cell.
fn canonical_binding_dtypes_for(op: OpKind) -> Option<Vec<DType>> {
    // Most elementwise ops follow `[input..., output]`. Reductions
    // and reduce-to follow `[input, output]`. MatMul is 3 inputs
    // (no — 2 inputs + 1 output): `[lhs, rhs, out] = [F32, F32, F32]`.
    let f32 = DType::F32;
    Some(match op {
        OpKind::MatMul => vec![f32, f32, f32],
        // Binary elementwise: 2 inputs + 1 output.
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise
        | OpKind::DivElementwise
        | OpKind::MaximumElementwise
        | OpKind::MinimumElementwise
        | OpKind::PowElementwise
        | OpKind::RemElementwise => vec![f32, f32, f32],
        // Unary elementwise: 1 input + 1 output.
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
        | OpKind::RsqrtElementwise
        | OpKind::Affine
        | OpKind::ClampElementwise
        | OpKind::PowIElementwise => vec![f32, f32],
        // Reductions: 1 input + 1 output.
        OpKind::SumReduce
        | OpKind::MaxReduce
        | OpKind::MinReduce
        | OpKind::MeanReduce
        | OpKind::ReduceSumTo
        | OpKind::ReduceMaxTo => vec![f32, f32],
        // All others: not yet wired into the direct-call path. The
        // realizer-measured first alternative is the only entry for
        // the cell until this list expands.
        _ => return None,
    })
}

/// Build the inputs/outputs/layouts/op_params for direct-call timing
/// of one (op, size) cell. Returns `None` for ops not yet supported.
/// Mirrors the input domains used by `build_input_graph` so the
/// direct-call CellRuns produce numerically-comparable output to the
/// realizer-measured primary alternative (consensus needs aligned
/// inputs across CellRuns).
fn prepare_direct_call_inputs(op: OpKind, size: &OpSize) -> Option<PreparedDirectCall> {
    use fuel_core_types::Layout;
    use fuel_cpu_backend::CpuStorageBytes;
    use fuel_dispatch::kernel::OpParams;
    use fuel_storage::{BackendStorage, Storage};

    let make_storage = |bytes: CpuStorageBytes| {
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cpu(bytes),
            DType::F32,
        )))
    };

    match (op, *size) {
        (OpKind::MatMul, OpSize::MatMul { m, n, k }) => {
            let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
            let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
            let lhs = make_storage(CpuStorageBytes::from_slice(&a_data));
            let rhs = make_storage(CpuStorageBytes::from_slice(&b_data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                m * n * std::mem::size_of::<f32>(),
            ));
            let layouts = vec![
                Layout::contiguous(Shape::from_dims(&[m, k])),
                Layout::contiguous(Shape::from_dims(&[k, n])),
                Layout::contiguous(Shape::from_dims(&[m, n])),
            ];
            let op_params = OpParams::Matmul {
                lhs_batch_dims: Vec::new(),
                rhs_batch_dims: Vec::new(),
                m, n, k,
            };
            Some(PreparedDirectCall {
                inputs: vec![lhs, rhs],
                output: out,
                layouts,
                op_params,
            })
        }
        (op, OpSize::Elementwise(n)) if is_binary_elementwise(op) => {
            let (a_data, b_data) = binary_inputs(op, n);
            let lhs = make_storage(CpuStorageBytes::from_slice(&a_data));
            let rhs = make_storage(CpuStorageBytes::from_slice(&b_data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                n * std::mem::size_of::<f32>(),
            ));
            let shape = Shape::from_dims(&[n]);
            let layouts = vec![
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape),
            ];
            Some(PreparedDirectCall {
                inputs: vec![lhs, rhs],
                output: out,
                layouts,
                op_params: OpParams::None,
            })
        }
        (op, OpSize::Elementwise(n)) if is_unary_elementwise(op) => {
            let raw = unary_input(n);
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
            let inp = make_storage(CpuStorageBytes::from_slice(&data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                n * std::mem::size_of::<f32>(),
            ));
            let shape = Shape::from_dims(&[n]);
            let layouts = vec![
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape),
            ];
            Some(PreparedDirectCall {
                inputs: vec![inp],
                output: out,
                layouts,
                op_params: OpParams::None,
            })
        }
        (op, OpSize::Elementwise(n)) if is_scalar_op(op) => {
            let data = unary_input(n);
            let inp = make_storage(CpuStorageBytes::from_slice(&data));
            let out = make_storage(CpuStorageBytes::from_zero_bytes(
                n * std::mem::size_of::<f32>(),
            ));
            let shape = Shape::from_dims(&[n]);
            let layouts = vec![
                Layout::contiguous(shape.clone()),
                Layout::contiguous(shape),
            ];
            let op_params = match op {
                OpKind::Affine           => OpParams::Affine { mul: 2.0, add: 0.0 },
                OpKind::ClampElementwise => OpParams::Clamp { min: -0.5, max: 0.5 },
                OpKind::PowIElementwise  => OpParams::PowI { exp: 3 },
                _ => unreachable!(),
            };
            Some(PreparedDirectCall {
                inputs: vec![inp],
                output: out,
                layouts,
                op_params,
            })
        }
        // Reductions + ReduceTo not yet wired into the direct-call
        // path: the OpParams shape needs the input layout, and the
        // output shape is op-dependent. Falls back to first-alt-only
        // measurement until the binding-table key conventions are
        // double-checked end-to-end here.
        _ => None,
    }
}

/// Read an output `Storage`'s F32 bytes back out as a `Vec<f32>` for
/// consensus comparison.
fn read_output_f32(out: &Arc<RwLock<fuel_storage::Storage>>) -> Vec<f32> {
    use fuel_storage::BackendStorage;
    let g = out.read().expect("output storage lock");
    match &g.inner {
        BackendStorage::Cpu(c) => c
            .as_slice::<f32>()
            .expect("cpu storage as f32 slice")
            .to_vec(),
        // Direct-call path is CPU-only today; other backends are
        // guarded by `measure_extra_alternatives` returning None
        // before we reach this point. A non-CPU storage here is a
        // bug, but we'd rather skip the entry than panic in the
        // Judge.
        #[allow(unreachable_patterns)]
        _ => Vec::new(),
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

/// One backend's measurement for one (op, dtype, size, kernel_source)
/// cell — kernel output captured alongside latency for downstream
/// consensus comparison.
///
/// **Per-alternative measurement (2026-06-08)**: `kernel_source`
/// distinguishes sibling kernels at one `(op, dtypes, backend)`
/// binding-table key (e.g. AOCL vs MKL vs portable-cpu at
/// `(MatMul, [F32×3], BackendId::Cpu)`). Each alternative produces
/// its own CellRun; the pairwise consensus then sees the full
/// cross-kernel-source population, treating AOCL/MKL/portable as
/// peers alongside CUDA/Vulkan.
#[derive(Debug)]
struct CellRun {
    backend: BackendId,
    device_index: u32,
    output: Vec<f32>,
    latency_ns: u64,
    iterations: u32,
    /// Diagnostic tag identifying the kernel sibling at the
    /// binding-table cell. `""` for cells with a single alternative
    /// or when the kernel-source tag is unknown.
    kernel_source: String,
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

/// Reinterpret an f32 vector as the little-endian bytes that
/// [`validate_against_fixture`] expects, run validation, and lift
/// the verdict into a single rel_err number for the ProfileEntry.
///
/// - `Ok(())` → `0.0` (output matches fixture within tolerance).
/// - `Err(OutOfTolerance { rel_err, .. })` → that `rel_err`.
/// - `Err(LengthMismatch | NonFinite)` → `f32::INFINITY` — these
///   are categorical failures (wrong shape, or NaN/Inf where finite
///   was expected). The dispatch picker will rank such an
///   alternative behind any finite-rel_err peer; the kernel-source
///   tag on the ProfileEntry tells us which alternative blew up.
fn max_rel_err_vs_fixture(out: &[f32], fixture: &CorrectnessFixture) -> f32 {
    let bytes: Vec<u8> = out.iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    match validate_against_fixture(fixture, &bytes) {
        Ok(()) => 0.0,
        Err(CorrectnessDrift::OutOfTolerance { rel_err, .. }) => rel_err as f32,
        Err(CorrectnessDrift::LengthMismatch { .. }) => f32::INFINITY,
        Err(CorrectnessDrift::NonFinite { .. }) => f32::INFINITY,
    }
}

/// Load every `*.json` file under `root` and flatten the contained
/// [`FixtureFile`] entries into a map keyed by
/// `(op, dtype, size_class)`. `root` may itself be a file (one
/// fixture file loaded) or a directory (recursive walk).
///
/// Errors propagate the first I/O failure or deserialization error
/// encountered. A directory with no `*.json` files yields an empty
/// map (the Judge then behaves as if no fixtures were configured).
fn load_fixtures_recursive(
    root: &Path,
) -> Result<HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>>> {
    let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> = HashMap::new();
    let mut visit = |p: &Path| -> Result<()> {
        let raw = std::fs::read_to_string(p)
            .map_err(|e| crate::error::Error::Msg(format!(
                "judge: failed to read fixture file {}: {e}", p.display(),
            )))?;
        let file: FixtureFile = serde_json::from_str(&raw)
            .map_err(|e| crate::error::Error::Msg(format!(
                "judge: failed to parse fixture file {} as FixtureFile: {e}",
                p.display(),
            )))?;
        for f in file.fixtures {
            let key = (f.op, f.dtype, f.size_class);
            map.entry(key).or_default().push(f);
        }
        Ok(())
    };

    fn walk(
        dir: &Path,
        visit: &mut dyn FnMut(&Path) -> Result<()>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::error::Error::Msg(format!(
                "judge: failed to read fixture dir {}: {e}", dir.display(),
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| crate::error::Error::Msg(format!(
                "judge: failed reading dir entry in {}: {e}", dir.display(),
            )))?;
            let p = entry.path();
            if p.is_dir() {
                walk(&p, visit)?;
            } else if p.extension().and_then(|s| s.to_str()) == Some("json") {
                visit(&p)?;
            }
        }
        Ok(())
    }

    if root.is_file() {
        visit(root)?;
    } else if root.is_dir() {
        walk(root, &mut visit)?;
    } else {
        return Err(crate::error::Error::Msg(format!(
            "judge: fixture path {} does not exist", root.display(),
        )));
    }
    Ok(map)
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
            fixtures: None,
        };
        let report = judge.run(&probe);
        assert_eq!(report.version, PROFILE_REPORT_VERSION);
        assert!(report.entries.iter().any(|e| e.backend == BackendId::Cpu));
        // Post-v2 (per-alternative measurement): one CPU cell may
        // produce multiple entries (one per kernel_source — AOCL,
        // MKL, portable-cpu). When peers exist the rel_err can be
        // non-zero but should stay below the bit-stable floor;
        // when only one alternative is registered rel_err is 0.0
        // (no peers → no drift reference).
        for e in report.entries.iter().filter(|e| e.backend == BackendId::Cpu) {
            assert!(e.max_rel_error < 1e-3,
                "cpu cell rel_err should stay tight ({e:?})");
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
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &unary {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            // Post-v2 (per-alternative measurement): ≥1 — when only the
            // portable CPU kernel is registered the count is 1; with
            // AOCL/MKL feature-gated alternatives it grows. Pick the
            // first entry (the realizer-primary measurement) for the
            // numerical-divergence check.
            assert!(!cpu_entries.is_empty(),
                "expected ≥1 cpu entry for {op}, got 0");
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
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &binary {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
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
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &reduce {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
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
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &reduce_to {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
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
            fixtures: None,
        };
        let report = judge.run(&probe);
        for &op in &scalar {
            let cpu_entries: Vec<_> = report.entries.iter()
                .filter(|e| e.op == op && e.backend == BackendId::Cpu)
                .collect();
            assert!(!cpu_entries.is_empty(), "expected >=1 cpu entry for {op}, got 0");
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
            fixtures: None,
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
                kernel_source: String::new(),
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

    /// Fixture-fast-path integration: when a fixture exists for the
    /// cell, the Judge derives `max_rel_error` from
    /// `validate_against_fixture` instead of running inline pairwise
    /// consensus. This test constructs an intentionally-divergent
    /// fixture for the AddElementwise cell and asserts the divergence
    /// shows up as the entries' `max_rel_error`.
    ///
    /// We deliberately *don't* use a fixture matching the actual cpu
    /// kernel's output here — the assertion is on the wire-up, not on
    /// any specific numerical value the cpu produces. Real fixtures
    /// land via the capture pipeline; this test verifies the path.
    #[test]
    fn judge_uses_fixture_when_provided() {
        use fuel_correctness_fixtures::{
            CorrectnessFixture, ToleranceBand,
        };

        let probe = ProbeReport::probe_all();
        let elem_count = 1 << 10;
        // Plant an obviously-wrong fixture: all zeros expected. The
        // cpu add kernel produces sin+cos = non-zero, so
        // validate_against_fixture's OutOfTolerance.rel_err lands in
        // every entry's max_rel_error (well above any plausible
        // cross-backend drift floor).
        let zeros: Vec<u8> = vec![0u8; elem_count * 4];
        let bogus_fixture = CorrectnessFixture {
            op: OpKind::AddElementwise,
            dtype: DType::F32,
            size_class: SizeClass::from_elem_count(elem_count),
            input_seed: 0,
            input_hash: 0,
            expected_output: zeros,
            output_element_count: elem_count,
            // Tight tolerance — guarantees the all-zeros fixture
            // disagrees with any honest add(sin, cos) output.
            tolerance: ToleranceBand {
                max_relative: 1e-9,
                max_absolute: 1e-12,
            },
        };
        let mut map: HashMap<(OpKind, DType, SizeClass), Vec<CorrectnessFixture>> =
            HashMap::new();
        map.insert(
            (OpKind::AddElementwise, DType::F32, SizeClass::from_elem_count(elem_count)),
            vec![bogus_fixture],
        );

        let judge = Judge {
            iterations: 3,
            warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::AddElementwise, OpSize::Elementwise(elem_count)),
            ]),
            fixtures: Some(map),
        };
        let report = judge.run(&probe);
        let cpu_entries: Vec<_> = report.entries.iter()
            .filter(|e| e.op == OpKind::AddElementwise && e.backend == BackendId::Cpu)
            .collect();
        assert!(!cpu_entries.is_empty(),
            "expected ≥1 cpu entry for AddElementwise, got 0");
        // The bogus all-zeros fixture is wildly wrong vs the honest
        // sin+cos output, so every cpu entry's max_rel_error should
        // be very large (≥ 1.0 — relative error vs ~zero expected is
        // unbounded; floor at 1.0 is conservative). Without the
        // fixture fast-path this value would be ~0.0 (lone CPU
        // backend → trivial consensus → no peers → 0.0).
        for e in &cpu_entries {
            assert!(e.max_rel_error >= 1.0,
                "expected fixture-derived rel_err ≥1.0 (vs all-zeros fixture), got {} for entry {e:?}",
                e.max_rel_error);
        }
    }

    /// Loader: `with_fixtures_from(file)` reads a JSON fixture file
    /// and produces a Judge whose `fixtures` map has the entry keyed
    /// correctly.
    #[test]
    fn judge_with_fixtures_from_loads_json_file() {
        use fuel_correctness_fixtures::{
            CorrectnessFixture, FixtureFile, ToleranceBand,
            FIXTURE_FILE_VERSION,
        };

        let fixture = CorrectnessFixture {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(10),
            input_seed: 42,
            input_hash: 0xdeadbeef,
            expected_output: vec![0u8; 16],
            output_element_count: 4,
            tolerance: ToleranceBand::F32_DEFAULT,
        };
        let file = FixtureFile {
            version: FIXTURE_FILE_VERSION,
            fixtures: vec![fixture],
        };
        let dir = std::env::temp_dir().join(format!(
            "fuel-judge-fixture-load-{}", std::process::id(),
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("matmul_f32.json");
        std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();

        let judge = Judge::with_fixtures_from(&dir).expect("load");
        let map = judge.fixtures.as_ref().expect("fixtures populated");
        let key = (OpKind::MatMul, DType::F32, SizeClass(10));
        let bucket = map.get(&key).expect("matmul fixture in bucket");
        assert_eq!(bucket.len(), 1);
        assert_eq!(bucket[0].input_seed, 42);

        // Loading from a single file path works too.
        let judge2 = Judge::with_fixtures_from(&path).expect("load file");
        let map2 = judge2.fixtures.as_ref().expect("fixtures populated");
        assert!(map2.contains_key(&key));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
