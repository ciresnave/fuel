//! Transfer-cost model for Phase 6c's DAG planner.
//!
//! When dispatch decisions span multiple backends, every cross-
//! backend edge incurs a transfer cost: an `Op::Copy` insertion
//! that pays bandwidth proportional to the tensor size. Phase 6b's
//! dispatch table ignores this cost — it picks each op's backend
//! purely on `latency_ns`. That's fine when every input/output
//! lands on the same backend, but it can give the wrong answer
//! when running an op on a "fast" backend would require a transfer
//! more expensive than the speedup itself.
//!
//! Phase 6c fixes this by adding a per-(src, dst) **bandwidth**
//! matrix. Cost of moving an N-byte tensor from `src` to `dst` is
//! `N * ns_per_byte(src, dst)`. The matrix is measured once at
//! probe time (it depends on PCIe topology + driver, not on the
//! workload) and persisted alongside the probe + Judge reports.
//!
//! # What's in this commit (Phase 6c-A)
//!
//! - [`BandwidthMatrix`] data type + serde.
//! - [`BandwidthMatrix::measure`] that exercises upload + download
//!   on each enabled backend and records the bandwidth.
//! - JSON persistence (save/load atomic write, schema versioning).
//!
//! What's **not** here yet, slated for Phase 6c-B:
//!
//! - Full DP planner that uses the matrix to pick per-node placement
//!   minimizing total cost (compute + transfer).
//! - D2D measurements for genuine multi-GPU rigs. Today every
//!   device measurement only covers the host↔device pair.
//! - Cross-backend (e.g. CUDA→Vulkan) D2D — that needs `Op::Copy`
//!   to support direct device transfers, which it currently doesn't
//!   (it routes through host).

use crate::probe::ProbeReport;
use fuel_core_types::probe::BackendId;
use fuel_core_types::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Schema version for persisted bandwidth reports. Bump when the
/// `BandwidthMatrix` shape changes.
pub const BANDWIDTH_REPORT_VERSION: u32 = 1;

/// Default filename for the persisted bandwidth report.
pub const BANDWIDTH_REPORT_FILENAME: &str = "bandwidth.json";

/// One-way cost in ns per byte from `src` backend to `dst` backend.
/// Includes any unavoidable per-call overhead (driver call, memory
/// alloc) amortized over the measurement buffer size — so this is
/// the *effective* bandwidth at the buffer size used for
/// measurement, not a peak-DMA number.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransferCost {
    pub src: BackendId,
    pub dst: BackendId,
    pub ns_per_byte: f64,
}

impl TransferCost {
    /// Cost (in ns) of transferring `bytes` from `src` to `dst`
    /// according to this entry.
    pub fn cost_ns(&self, bytes: usize) -> u64 {
        (self.ns_per_byte * bytes as f64).round() as u64
    }
}

/// Per-(src, dst) bandwidth lookup. Measured once at probe time.
///
/// Rows / columns of the matrix are [`BackendId`]s. The same-backend
/// entries are always present (they describe in-backend bandwidth,
/// usually a memcpy). Cross-backend entries are present only for
/// pairs that have a working transfer path; missing entries indicate
/// "no direct transfer — must round-trip through CPU".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandwidthMatrix {
    pub version: u32,
    /// Buffer size used for the measurement, in bytes. Stored so
    /// callers can interpret the ns_per_byte numbers in context —
    /// effective bandwidth depends on buffer size for small
    /// transfers (driver overhead dominates) and converges at
    /// large sizes.
    pub measurement_bytes: usize,
    pub entries: Vec<TransferCost>,
}

impl BandwidthMatrix {
    /// Look up the cost of a transfer. Returns `None` if no entry
    /// exists for the pair. Callers that need a fallback should
    /// route through CPU (host) — `cpu` is always reachable.
    pub fn lookup(&self, src: BackendId, dst: BackendId) -> Option<&TransferCost> {
        self.entries.iter().find(|e| e.src == src && e.dst == dst)
    }

    /// Measure transfer bandwidth for every backend in `probe`.
    /// One round-trip (upload + download) per backend; the upload
    /// time gives `(Cpu, backend)` cost and the download time gives
    /// `(backend, Cpu)`. Runs `iters` measurement rounds (default 5)
    /// and takes the median to filter scheduler jitter.
    ///
    /// `bytes` controls the measurement buffer size — defaults to
    /// 16 MB (`1 << 24`), large enough to amortize per-call driver
    /// overhead but small enough to fit on consumer GPUs.
    pub fn measure(probe: &ProbeReport) -> Self {
        Self::measure_with(probe, DEFAULT_MEASUREMENT_BYTES, DEFAULT_ITERATIONS)
    }

    /// Measurement with explicit buffer size + iteration count.
    pub fn measure_with(probe: &ProbeReport, bytes: usize, iters: u32) -> Self {
        let mut entries: Vec<TransferCost> = Vec::new();
        let mut seen_keys: HashMap<BackendId, bool> = HashMap::new();

        // CPU vs CPU is a memcpy — give it a representative number.
        // Single trivial entry; subsequent backends will append
        // (Cpu, X) and (X, Cpu) pairs as they're measured.
        if probe.devices.iter().any(|d| d.backend == BackendId::Cpu) {
            let cpu_cpu = measure_cpu_memcpy(bytes, iters);
            entries.push(TransferCost {
                src: BackendId::Cpu,
                dst: BackendId::Cpu,
                ns_per_byte: cpu_cpu,
            });
            seen_keys.insert(BackendId::Cpu, true);
        }

        // For each non-CPU backend, measure (Cpu → backend) upload
        // and (backend → Cpu) download. We measure the FIRST device
        // in each backend's class; the same-SKU equivalence class
        // shares the result.
        let mut measured_backends: HashMap<BackendId, ()> = HashMap::new();
        for d in &probe.devices {
            if matches!(d.backend, BackendId::Cpu) {
                continue;
            }
            if measured_backends.contains_key(&d.backend) {
                continue;
            }
            measured_backends.insert(d.backend, ());

            if let Some((h2d, d2h)) = measure_h2d_d2h(d, bytes, iters) {
                entries.push(TransferCost {
                    src: BackendId::Cpu,
                    dst: d.backend,
                    ns_per_byte: h2d,
                });
                entries.push(TransferCost {
                    src: d.backend,
                    dst: BackendId::Cpu,
                    ns_per_byte: d2h,
                });
            } else {
                eprintln!(
                    "transfer_cost: skipping {} measurement \
                     (backend instantiation failed or feature not enabled)",
                    d.backend
                );
            }
        }

        Self {
            version: BANDWIDTH_REPORT_VERSION,
            measurement_bytes: bytes,
            entries,
        }
    }

    /// Atomic JSON write (sibling `.tmp` + rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| fuel_core_types::Error::Msg(
                format!("transfer_cost: JSON encode failed: {e}")))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json)
            .map_err(|e| fuel_core_types::Error::Msg(
                format!("transfer_cost: write {tmp:?} failed: {e}")))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| fuel_core_types::Error::Msg(
                format!("transfer_cost: rename {tmp:?} → {path:?} failed: {e}")))?;
        Ok(())
    }

    /// Load a persisted bandwidth report. `Ok(None)` for missing file
    /// or schema-version mismatch (cache miss, re-measure).
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(fuel_core_types::Error::Msg(
                format!("transfer_cost: read {path:?} failed: {e}"))),
        };
        let report: Self = serde_json::from_slice(&bytes)
            .map_err(|e| fuel_core_types::Error::Msg(
                format!("transfer_cost: parse {path:?} failed: {e}")))?;
        if report.version != BANDWIDTH_REPORT_VERSION {
            return Ok(None);
        }
        Ok(Some(report))
    }
}

pub const DEFAULT_MEASUREMENT_BYTES: usize = 1 << 24;  // 16 MiB
pub const DEFAULT_ITERATIONS: u32 = 5;

/// CPU-to-CPU bandwidth — just a memcpy. Used as a baseline for
/// comparison and for the `(Cpu, Cpu)` and `(Reference, Reference)`
/// entries.
fn measure_cpu_memcpy(bytes: usize, iters: u32) -> f64 {
    let n_f32 = bytes / 4;
    let src: Vec<f32> = (0..n_f32).map(|i| i as f32 * 1e-3).collect();
    let mut timings = Vec::with_capacity(iters as usize);
    let mut dst: Vec<f32> = vec![0.0; n_f32];
    for _ in 0..iters {
        let t0 = Instant::now();
        dst.copy_from_slice(&src);
        timings.push(t0.elapsed().as_nanos() as u64);
    }
    timings.sort_unstable();
    let median = timings[timings.len() / 2] as f64;
    median / bytes as f64
}

/// Measure H2D + D2H for a specific (backend, device). Returns the
/// pair `(h2d_ns_per_byte, d2h_ns_per_byte)` or `None` if the
/// backend isn't enabled / can't be initialized.
fn measure_h2d_d2h(
    device: &fuel_core_types::probe::DeviceDescriptor,
    bytes: usize,
    iters: u32,
) -> Option<(f64, f64)> {
    let n_f32 = bytes / 4;
    let host_data: Vec<f32> = (0..n_f32).map(|i| (i as f32) * 1e-3).collect();
    let host_buf = fuel_core_types::HostBuffer::F32(host_data);
    let shape = fuel_core_types::Shape::from_dims(&[n_f32]);

    match device.backend {
        #[cfg(feature = "cuda")]
        BackendId::Cuda => {
            let dev = fuel_cuda_backend::CudaDevice::new(device.device_index as usize).ok()?;
            let backend = fuel_cuda_backend::CudaBackend::new(dev);
            measure_round_trip_via_backend(&backend, &host_buf, &shape, bytes, iters)
        }
        #[cfg(feature = "vulkan")]
        BackendId::Vulkan => {
            let backend = fuel_vulkan_backend::VulkanBackend::with_selection(
                fuel_vulkan_backend::DeviceSelection::Index(device.device_index as usize),
            ).ok()?;
            measure_round_trip_via_backend(&backend, &host_buf, &shape, bytes, iters)
        }
        _ => None,
    }
}

#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn measure_round_trip_via_backend<B: fuel_graph_executor::GraphBackend>(
    backend: &B,
    host_buf: &fuel_core_types::HostBuffer,
    shape: &fuel_core_types::Shape,
    bytes: usize,
    iters: u32,
) -> Option<(f64, f64)> {
    // Warmup — first upload may include JIT / kernel-cache costs.
    let warmup = backend.upload(host_buf, shape).ok()?;
    let _ = backend.download(&warmup).ok()?;
    drop(warmup);

    let mut h2d = Vec::with_capacity(iters as usize);
    let mut d2h = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t0 = Instant::now();
        let storage = backend.upload(host_buf, shape).ok()?;
        let elapsed_h2d = t0.elapsed().as_nanos() as u64;
        h2d.push(elapsed_h2d);

        let t1 = Instant::now();
        let _ = backend.download(&storage).ok()?;
        let elapsed_d2h = t1.elapsed().as_nanos() as u64;
        d2h.push(elapsed_d2h);
    }
    h2d.sort_unstable();
    d2h.sort_unstable();
    let h2d_med = h2d[h2d.len() / 2] as f64;
    let d2h_med = d2h[d2h.len() / 2] as f64;
    Some((h2d_med / bytes as f64, d2h_med / bytes as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::DeviceLocation;

    fn cpu_only_probe() -> ProbeReport {
        use fuel_core_types::probe::DeviceDescriptor;
        ProbeReport {
            version: crate::probe::PROBE_REPORT_VERSION,
            devices: vec![DeviceDescriptor {
                backend:            BackendId::Cpu,
                device_index:       0,
                hardware_sku:       "test cpu".to_string(),
                vendor_id:          0,
                device_id:          0,
                compute_capability: None,
                driver_version:     "test".to_string(),
                total_memory_bytes: 0,
                location:           DeviceLocation::Cpu,
            }],
        }
    }

    #[test]
    fn cpu_only_matrix_has_one_entry() {
        // 64 KiB so the test runs in microseconds.
        let m = BandwidthMatrix::measure_with(&cpu_only_probe(), 64 * 1024, 3);
        assert_eq!(m.entries.len(), 1);
        let cpu_self = m.lookup(BackendId::Cpu, BackendId::Cpu).unwrap();
        assert_eq!(cpu_self.src, BackendId::Cpu);
        assert_eq!(cpu_self.dst, BackendId::Cpu);
        // Sanity: ns_per_byte should be small but positive.
        assert!(cpu_self.ns_per_byte >= 0.0);
        assert!(cpu_self.ns_per_byte < 1000.0,
            "cpu memcpy should be well under 1µs/byte; got {}", cpu_self.ns_per_byte);
    }

    #[test]
    fn cost_ns_scales_linearly() {
        let c = TransferCost {
            src: BackendId::Cpu, dst: BackendId::Cuda,
            ns_per_byte: 0.1,
        };
        assert_eq!(c.cost_ns(0), 0);
        assert_eq!(c.cost_ns(100), 10);
        assert_eq!(c.cost_ns(1_000_000), 100_000);
    }

    #[test]
    fn save_load_roundtrip() {
        let report = BandwidthMatrix {
            version: BANDWIDTH_REPORT_VERSION,
            measurement_bytes: 1 << 20,
            entries: vec![
                TransferCost { src: BackendId::Cpu, dst: BackendId::Cpu, ns_per_byte: 0.05 },
                TransferCost { src: BackendId::Cpu, dst: BackendId::Cuda, ns_per_byte: 0.5 },
                TransferCost { src: BackendId::Cuda, dst: BackendId::Cpu, ns_per_byte: 0.6 },
            ],
        };
        let tmp = std::env::temp_dir().join(format!(
            "fuel-bandwidth-test-{}.json", std::process::id()
        ));
        report.save(&tmp).expect("save");
        let loaded = BandwidthMatrix::load(&tmp).expect("load").expect("file exists");
        assert_eq!(loaded, report);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_missing_file_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "fuel-bandwidth-test-nonexistent-{}.json", std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let loaded = BandwidthMatrix::load(&tmp).expect("missing file is not an error");
        assert!(loaded.is_none());
    }

    #[test]
    fn load_wrong_version_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "fuel-bandwidth-test-oldver-{}.json", std::process::id()
        ));
        let ancient = serde_json::json!({
            "version": 0,
            "measurement_bytes": 0,
            "entries": [],
        });
        std::fs::write(&tmp, serde_json::to_vec(&ancient).unwrap()).unwrap();
        let loaded = BandwidthMatrix::load(&tmp).expect("old version parses, not errors");
        assert!(loaded.is_none(), "old-version reports should be treated as cache miss");
        let _ = std::fs::remove_file(&tmp);
    }
}
