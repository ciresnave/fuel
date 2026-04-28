//! The Judge — Phase 6b's empirical profiler.
//!
//! For every `(backend, device)` pair the probe report knows about,
//! the Judge walks a matrix of `(op_kind, dtype, size_class)` and
//! measures two things: wall-clock latency and numerical precision
//! relative to the reference backend. Output is a persistable
//! [`ProfileReport`] that the (future) ranked dispatch table indexes
//! at realize time.
//!
//! # Scope of this revision
//!
//! Ships the types, the runner skeleton, and profiling for two
//! op kinds and three dtypes so the end-to-end pipeline is wired:
//!
//! - **Ops**: [`OpKind::MatMul`] and [`OpKind::AddElementwise`]. MatMul
//!   is the headline op; AddElementwise exercises the cheapest path
//!   to make sure the Judge doesn't accidentally skew its timing on
//!   tiny work.
//! - **Dtypes**: f32 for now. f64 / bf16 / f16 are a mechanical
//!   extension once the dispatch table surfaces the dtype axis.
//! - **Backends**: CPU (fast path via `fuel-graph-cpu`), reference,
//!   and CUDA (gated on the `cuda` feature). Vulkan is identified in
//!   the probe but not yet profiled here — needs a
//!   `LazyTensor::realize_f32_vulkan` helper equivalent to the CUDA
//!   one. Skipped with a stderr note so the runner is explicit about
//!   the gap.
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

use crate::probe::ProbeReport;
use fuel_core_types::probe::{BackendId, DeviceDescriptor};
use fuel_core_types::{DType, Result, Shape};
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
            OpKind::AddElementwise => vec![
                OpSize::Elementwise(1 << 10),
                OpSize::Elementwise(1 << 16),
                OpSize::Elementwise(1 << 20),
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
    pub fn run(&self, probe: &ProbeReport) -> ProfileReport {
        let mut entries = Vec::new();

        // Equivalence-class deduplication: measure one representative
        // per class, replicate the entry across every device_index in
        // the class.
        let classes = probe.equivalence_classes();
        for (_key, devs) in &classes {
            // Pick the first device as the representative. Deterministic
            // because `HashMap::get` returns a consistent ordering within
            // a single process run (and we don't compare across runs —
            // profile entries carry their own device_index so dispatch
            // lookups are by full identity, not by index).
            let rep = devs[0];
            for &op in &[OpKind::MatMul, OpKind::AddElementwise] {
                for sz in self.size_plan(op) {
                    if let Some(entry) = self.measure_on_device(op, DType::F32, &sz, rep) {
                        // Replicate the (latency, error) across every
                        // device in the class — only the device_index
                        // field differs.
                        for d in devs {
                            entries.push(ProfileEntry {
                                device_index: d.device_index,
                                ..entry.clone()
                            });
                        }
                    }
                }
            }
        }

        ProfileReport { version: PROFILE_REPORT_VERSION, entries }
    }

    /// Measure one (op, dtype, size) cell on one representative
    /// device. Returns `None` if the backend isn't wired into the
    /// Judge yet (Vulkan, Metal).
    fn measure_on_device(
        &self,
        op: OpKind,
        dtype: DType,
        size: &OpSize,
        device: &DeviceDescriptor,
    ) -> Option<ProfileEntry> {
        assert_eq!(dtype, DType::F32, "judge: only f32 wired for now");

        let size_class = SizeClass::from_elem_count(size.total_elements());

        // Backend-specific realize closure. Returns the realized f32
        // vector so downstream precision measurement can compare it
        // against the reference.
        #[cfg(feature = "cuda")]
        let mut cuda_executor: Option<fuel_graph_executor::GraphExecutor<fuel_graph_cuda::CudaBackend>> =
            None;
        #[cfg(feature = "vulkan")]
        let mut vulkan_executor: Option<fuel_graph_executor::GraphExecutor<fuel_graph_vulkan::VulkanBackend>> =
            None;
        #[cfg(feature = "aocl")]
        let mut aocl_executor: Option<fuel_graph_executor::GraphExecutor<fuel_aocl_cpu_backend::AoclBackend>> =
            None;
        #[cfg(feature = "onemkl")]
        let mut mkl_executor: Option<fuel_graph_executor::GraphExecutor<fuel_mkl_cpu_backend::MklBackend>> =
            None;

        let realize: Box<dyn FnMut(&crate::lazy::LazyTensor) -> Vec<f32>> = match device.backend {
            BackendId::Reference => Box::new(|t| t.realize_f32_reference()),
            BackendId::Cpu       => Box::new(|t| t.realize_f32()),
            #[cfg(feature = "onemkl")]
            BackendId::Mkl => {
                let backend = match fuel_mkl_cpu_backend::MklBackend::try_new() {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "judge: skipping mkl:{} — MklBackend::try_new failed: {e}",
                            device.device_index,
                        );
                        return None;
                    }
                };
                mkl_executor = Some(fuel_graph_executor::GraphExecutor::new(backend));
                let exe = mkl_executor.as_mut().unwrap();
                let exe_ptr: *mut fuel_graph_executor::GraphExecutor<fuel_mkl_cpu_backend::MklBackend>
                    = exe;
                // SAFETY: same shape as AOCL/CUDA/Vulkan arms — exe
                // lives in the Option on the stack frame that outlives
                // the closure's lifetime.
                Box::new(move |t| t.realize_f32_mkl(unsafe { &mut *exe_ptr }))
            }
            #[cfg(not(feature = "onemkl"))]
            BackendId::Mkl => {
                eprintln!(
                    "judge: skipping mkl:{} — onemkl feature not enabled",
                    device.device_index,
                );
                return None;
            }
            #[cfg(feature = "aocl")]
            BackendId::Aocl => {
                let backend = match fuel_aocl_cpu_backend::AoclBackend::try_new() {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "judge: skipping aocl:{} — AoclBackend::try_new failed: {e}",
                            device.device_index,
                        );
                        return None;
                    }
                };
                aocl_executor = Some(fuel_graph_executor::GraphExecutor::new(backend));
                let exe = aocl_executor.as_mut().unwrap();
                let exe_ptr: *mut fuel_graph_executor::GraphExecutor<fuel_aocl_cpu_backend::AoclBackend>
                    = exe;
                // SAFETY: same shape as the CUDA / Vulkan arms — exe
                // lives in the Option on the stack frame that outlives
                // the closure's lifetime.
                Box::new(move |t| t.realize_f32_aocl(unsafe { &mut *exe_ptr }))
            }
            #[cfg(not(feature = "aocl"))]
            BackendId::Aocl => {
                eprintln!(
                    "judge: skipping aocl:{} — aocl feature not enabled",
                    device.device_index,
                );
                return None;
            }
            #[cfg(feature = "cuda")]
            BackendId::Cuda => {
                // One executor per device for the whole measurement
                // run; avoids re-creating CUDA contexts per iteration.
                let dev = match fuel_graph_cuda::CudaDevice::new(device.device_index as usize) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!(
                            "judge: skipping cuda:{} — CudaDevice::new failed: {e}",
                            device.device_index,
                        );
                        return None;
                    }
                };
                cuda_executor = Some(fuel_graph_executor::GraphExecutor::new(
                    fuel_graph_cuda::CudaBackend::new(dev),
                ));
                let exe = cuda_executor.as_mut().unwrap();
                let exe_ptr: *mut fuel_graph_executor::GraphExecutor<fuel_graph_cuda::CudaBackend> = exe;
                // SAFETY: we hold the executor in an Option on the stack
                // that outlives the returned closure (lifetime-bounded
                // to this function's body). The raw pointer avoids the
                // borrow-checker's inability to see that.
                Box::new(move |t| t.realize_f32_cuda(unsafe { &mut *exe_ptr }))
            }
            #[cfg(feature = "vulkan")]
            BackendId::Vulkan => {
                let backend = match fuel_graph_vulkan::VulkanBackend::with_selection(
                    fuel_graph_vulkan::DeviceSelection::Index(device.device_index as usize),
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "judge: skipping vulkan:{} — VulkanBackend init failed: {e}",
                            device.device_index,
                        );
                        return None;
                    }
                };
                vulkan_executor = Some(fuel_graph_executor::GraphExecutor::new(backend));
                let exe = vulkan_executor.as_mut().unwrap();
                let exe_ptr: *mut fuel_graph_executor::GraphExecutor<fuel_graph_vulkan::VulkanBackend>
                    = exe;
                // SAFETY: same shape as the CUDA arm — exe lives in
                // the Option on the stack frame that outlives the
                // closure's lifetime.
                Box::new(move |t| t.realize_f32_vulkan(unsafe { &mut *exe_ptr }))
            }
            #[cfg(not(feature = "vulkan"))]
            BackendId::Vulkan => {
                eprintln!(
                    "judge: skipping vulkan:{} — vulkan feature not enabled",
                    device.device_index,
                );
                return None;
            }
            other => {
                eprintln!(
                    "judge: skipping backend {other} device:{} — not yet wired",
                    device.device_index,
                );
                return None;
            }
        };

        let entry = self.time_op(op, size, device, size_class, realize);
        // Drop the per-device executors here, tearing down their
        // contexts before we move on to the next device.
        #[cfg(feature = "cuda")]
        drop(cuda_executor);
        #[cfg(feature = "vulkan")]
        drop(vulkan_executor);
        #[cfg(feature = "aocl")]
        drop(aocl_executor);
        #[cfg(feature = "onemkl")]
        drop(mkl_executor);
        entry
    }

    /// Build the op's input graph, realize it for warmup+timed runs,
    /// measure latency, compare against reference for precision.
    fn time_op(
        &self,
        op: OpKind,
        size: &OpSize,
        device: &DeviceDescriptor,
        size_class: SizeClass,
        mut realize: Box<dyn FnMut(&crate::lazy::LazyTensor) -> Vec<f32>>,
    ) -> Option<ProfileEntry> {
        let tensor = build_input_graph(op, size);

        // Precision check — do this first while the reference
        // backend's output is fresh (avoids fighting the compiler
        // over closure borrows later).
        let reference_out = tensor.realize_f32_reference();

        // Warmup — discard timings; stabilize kernel caches, warm up
        // any BLAS internal state, fault-in heap.
        for _ in 0..self.warmup {
            let _ = realize(&tensor);
        }

        // Timed iterations — record each and take the median so one
        // pre-empted scheduler tick doesn't skew the number.
        let mut timings_ns = Vec::with_capacity(self.iterations as usize);
        let mut max_rel_error: f32 = 0.0;
        for _ in 0..self.iterations {
            let t0 = Instant::now();
            let out = realize(&tensor);
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            timings_ns.push(elapsed_ns);

            // Only compare on the first iteration to save work; precision
            // is deterministic for a given (backend, input).
            if timings_ns.len() == 1 {
                max_rel_error = max_rel_err(&out, &reference_out);
            }
        }
        timings_ns.sort_unstable();
        let median = timings_ns[timings_ns.len() / 2];

        Some(ProfileEntry {
            op,
            dtype: DType::F32,
            size_class,
            backend: device.backend,
            device_index: device.device_index,
            latency_ns: median,
            iterations: self.iterations,
            max_rel_error,
        })
    }
}

/// Per-op size descriptor. Exposed (not private) so tests can supply
/// a shrunk `size_plan_override`; not in the persisted profile
/// report shape — the report uses the bucketed [`SizeClass`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpSize {
    MatMul { m: usize, n: usize, k: usize },
    Elementwise(usize),
}

impl OpSize {
    fn total_elements(&self) -> usize {
        match *self {
            OpSize::MatMul { m, n, k: _ } => m * n,
            OpSize::Elementwise(n) => n,
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
            let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
            let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
            a.matmul(&b)
        }
        (OpKind::AddElementwise, OpSize::Elementwise(n)) => {
            let a_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 2.1e-3).sin()).collect();
            let b_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.9e-3).cos()).collect();
            let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[n]));
            let b = a.const_f32_like(b_data, Shape::from_dims(&[n]));
            a.add(&b)
        }
        (op, sz) => panic!("build_input_graph: op {op:?} size {sz:?} not implemented"),
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
    fn judge_profiles_cpu_and_reference_on_small_matmul() {
        // Minimal end-to-end: probe (captures cpu + reference), run
        // the Judge with a shrunk size plan + iteration count so the
        // test stays in the low-second range, verify both backends
        // produce an entry and their relative error is small.
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
        assert!(report.entries.iter().any(|e| e.backend == BackendId::Reference));
        // Reference backend vs itself: rel error exactly 0.
        for e in report.entries.iter().filter(|e| e.backend == BackendId::Reference) {
            assert_eq!(e.max_rel_error, 0.0,
                "reference backend disagrees with itself at {e:?}");
        }
        // CPU fast path vs reference: gemm's blocked sum order
        // differs from the reference textbook triple-loop, so
        // accumulated f32 drift on a 1024×1024 matmul can reach
        // ~1e-3 relative. 5e-3 is the cliff beyond which we'd
        // suspect an actual bug, not rounding.
        for e in report.entries.iter().filter(|e| e.backend == BackendId::Cpu) {
            assert!(e.max_rel_error < 5e-3,
                "cpu fast path diverges too far from reference: {e:?}");
            assert!(e.latency_ns > 0);
        }
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
