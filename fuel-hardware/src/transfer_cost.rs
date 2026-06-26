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
//! - [`BandwidthMatrix::measure`] that times H2D + D2H on each
//!   enabled backend and records the bandwidth. Since
//!   executor-unification Session 6 the GPU timing rides the Stage 1
//!   calibration probes below (byte-storage substrate APIs) instead
//!   of the retired legacy `GraphBackend` upload/download wrappers.
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
//!
//! # Stage 1 — transfer calibration for the load-time planner
//!
//! The second layer in this module ([`TransferEstimate`] +
//! [`TransferCalibration`]) is Stage 1 of the load-time incremental
//! planner (`docs/session-prompts/load-time-incremental-planner.md`).
//! It differs from the Phase 6c [`BandwidthMatrix`] in three ways:
//!
//! - **Keyed by `DeviceLocation` pairs**, matching `SystemTopology`'s
//!   transfer-path matrix, not by `BackendId` — the planner prices
//!   *device* boundary crossings.
//! - **Bandwidth + fixed latency** from a two-point linear fit over
//!   multiple transfer sizes, so small transfers aren't priced at the
//!   amortized-large-buffer rate.
//! - **Measured through the byte-storage substrate APIs**
//!   (`CudaStorageBytes::from_cpu_bytes`/`to_cpu_bytes`,
//!   `VulkanBackend::upload_bytes`/`download_bytes`) — the exact
//!   paths the pipelined executor pays — instead of the legacy
//!   `GraphBackend` upload/download wrappers.
//!
//! `SystemTopology` owns the lazy once-per-generation cache (its
//! `transfer_calibration`); this module owns the probe + fit math +
//! conservative fallbacks.

use crate::probe::ProbeReport;
use fuel_ir::backend::TransferPath;
use fuel_ir::probe::BackendId;
use fuel_ir::{DeviceLocation, Result};
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
    /// `iters` applies to the CPU memcpy baseline; the GPU H2D/D2H
    /// paths go through the Stage 1 calibration probe (which uses
    /// [`CALIBRATION_ITERS`] internally).
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

            if let Some((h2d, d2h)) = measure_h2d_d2h(d, bytes) {
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
            .map_err(|e| fuel_ir::Error::Msg(
                format!("transfer_cost: JSON encode failed: {e}")))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json)
            .map_err(|e| fuel_ir::Error::Msg(
                format!("transfer_cost: write {tmp:?} failed: {e}")))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| fuel_ir::Error::Msg(
                format!("transfer_cost: rename {tmp:?} → {path:?} failed: {e}")))?;
        Ok(())
    }

    /// Load a persisted bandwidth report. `Ok(None)` for missing file
    /// or schema-version mismatch (cache miss, re-measure).
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(fuel_ir::Error::Msg(
                format!("transfer_cost: read {path:?} failed: {e}"))),
        };
        let report: Self = serde_json::from_slice(&bytes)
            .map_err(|e| fuel_ir::Error::Msg(
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

/// Measure H2D + D2H for a specific (backend, device) through the
/// Stage 1 calibration probes ([`probe_cuda_device`] /
/// [`probe_vulkan_device`] — the byte-storage substrate APIs, i.e.
/// the exact paths the pipelined executor's `Op::Copy` pays). The
/// probed [`TransferEstimate`]s (linear latency + bandwidth fit) are
/// flattened to the matrix's single effective `ns_per_byte` at the
/// requested measurement buffer size.
///
/// Returns the pair `(h2d_ns_per_byte, d2h_ns_per_byte)` or `None`
/// if the backend isn't enabled / can't be initialized.
///
/// History: this used to time `upload`/`download` round-trips through
/// the legacy `GraphBackend` trait (`measure_round_trip_via_backend`)
/// — the last code-level legacy-executor reference in fuel-core.
/// Re-pointed onto the calibration substrate (executor-unification
/// Session 6); iteration count is the probe's [`CALIBRATION_ITERS`].
fn measure_h2d_d2h(
    device: &fuel_ir::probe::DeviceDescriptor,
    bytes: usize,
) -> Option<(f64, f64)> {
    let (h2d, d2h) = match device.backend {
        #[cfg(feature = "cuda")]
        BackendId::Cuda => probe_cuda_device(device.device_index as usize)?,
        #[cfg(feature = "vulkan")]
        BackendId::Vulkan => probe_vulkan_device(device.device_index as usize)?,
        _ => return None,
    };
    let per_byte =
        |e: TransferEstimate| e.estimate_ns(bytes as u64) as f64 / bytes.max(1) as f64;
    Some((per_byte(h2d), per_byte(d2h)))
}

// ===========================================================================
// Stage 1 — transfer calibration (load-time incremental planner)
// ===========================================================================

/// Transfer sizes the calibration probe times per path. Three sizes
/// spanning three orders of magnitude so the linear fit separates the
/// fixed per-call latency (dominates at 64 KiB) from the bandwidth
/// term (dominates at 64 MiB).
pub const CALIBRATION_SIZES: [usize; 3] = [64 * 1024, 4 << 20, 64 << 20];

/// Timed iterations per (path, size); the median filters scheduler
/// jitter. Kept small — the probe runs lazily on first planner
/// request and should stay in the tens-of-ms range per device.
pub const CALIBRATION_ITERS: u32 = 3;

/// Numeric cost model for one transfer path: a fixed per-call latency
/// plus a bandwidth term. `time(bytes) ≈ latency_ns + bytes /
/// bandwidth`. Stage 1 of the load-time incremental planner
/// (`docs/session-prompts/load-time-incremental-planner.md`); priced
/// by the Stage 2 cost composer and the Stage 3 placement DP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferEstimate {
    /// Effective sustained bandwidth in bytes per second. Never zero
    /// — constructors clamp to ≥ 1 so [`Self::estimate_ns`] can't
    /// divide by zero.
    pub bandwidth_bytes_per_sec: u64,
    /// Fixed per-transfer overhead in nanoseconds (driver call,
    /// staging-buffer setup, fence/sync), independent of size.
    pub latency_ns: u64,
}

impl TransferEstimate {
    /// Estimated wall-clock nanoseconds to move `bytes` over this
    /// path: `latency_ns + bytes * 1e9 / bandwidth`. Saturating
    /// arithmetic throughout — never panics, never overflows, and
    /// monotonically non-decreasing in `bytes`.
    pub fn estimate_ns(&self, bytes: u64) -> u64 {
        let bw = self.bandwidth_bytes_per_sec.max(1) as u128;
        let transfer = (bytes as u128).saturating_mul(1_000_000_000) / bw;
        let total = transfer.saturating_add(self.latency_ns as u128);
        u64::try_from(total).unwrap_or(u64::MAX)
    }

    /// Conservative static estimate for an unprobed path, by path
    /// class. "Conservative" means biased *against* moving data: the
    /// bandwidths sit below what healthy hardware delivers and the
    /// latencies above typical driver overhead, so a planner working
    /// from fallbacks only crosses a device boundary when the win is
    /// decisive. Rationale per class:
    ///
    /// - **SameDevice** — no bytes move; zero cost (`u64::MAX`
    ///   bandwidth, zero latency).
    /// - **Peer** — PCIe 4.0 x16 P2P sustains ~20–25 GB/s; NVLink
    ///   far more. Fallback 12 GB/s, 15 µs (P2P enablement +
    ///   cross-device sync round-trip).
    /// - **SharedMemory** — UMA/dma-buf zero-copy paths (Apple
    ///   Silicon, iGPUs) reach memory-bus speeds (> 50 GB/s).
    ///   Fallback 20 GB/s, 2 µs (mapping is cheap, not free).
    /// - **DeviceCopy** — pageable-host cudaMemcpy / staged
    ///   vkCmdCopyBuffer on PCIe 4.0 x16 measures ~6–13 GB/s
    ///   end-to-end including alloc + sync (what Fuel's storage
    ///   APIs actually pay). Fallback 8 GB/s, 30 µs.
    /// - **HostStaging** — two DeviceCopy hops through host RAM:
    ///   roughly half the bandwidth, double the latency.
    ///   Fallback 4 GB/s, 60 µs.
    pub fn fallback_for(path: TransferPath) -> TransferEstimate {
        match path {
            TransferPath::SameDevice => TransferEstimate {
                bandwidth_bytes_per_sec: u64::MAX,
                latency_ns: 0,
            },
            TransferPath::Peer => TransferEstimate {
                bandwidth_bytes_per_sec: 12_000_000_000,
                latency_ns: 15_000,
            },
            TransferPath::SharedMemory => TransferEstimate {
                bandwidth_bytes_per_sec: 20_000_000_000,
                latency_ns: 2_000,
            },
            TransferPath::DeviceCopy => TransferEstimate {
                bandwidth_bytes_per_sec: 8_000_000_000,
                latency_ns: 30_000,
            },
            TransferPath::HostStaging => TransferEstimate {
                bandwidth_bytes_per_sec: 4_000_000_000,
                latency_ns: 60_000,
            },
        }
    }

    /// Serial composition of two transfer stages (e.g. D2H then H2D
    /// for a host-staged D2D move): latencies add, bandwidths combine
    /// harmonically (`1/bw = 1/bw₁ + 1/bw₂` — the bytes traverse both
    /// links back to back).
    pub fn compose_staged(first: TransferEstimate, second: TransferEstimate) -> TransferEstimate {
        let b1 = first.bandwidth_bytes_per_sec.max(1) as u128;
        let b2 = second.bandwidth_bytes_per_sec.max(1) as u128;
        let bw = (b1 * b2) / (b1 + b2);
        TransferEstimate {
            bandwidth_bytes_per_sec: u64::try_from(bw).unwrap_or(u64::MAX).max(1),
            latency_ns: first.latency_ns.saturating_add(second.latency_ns),
        }
    }
}

/// Two-point linear fit of `(bytes, ns)` measurement points to a
/// [`TransferEstimate`]. Uses the smallest- and largest-size points:
/// `bandwidth = Δbytes / Δtime`, `latency = t_lo - bytes_lo /
/// bandwidth` (clamped ≥ 0). Degenerate inputs degrade gracefully
/// instead of erroring:
///
/// - empty input → `None`;
/// - a single distinct size → bandwidth from that point, zero
///   latency;
/// - non-increasing time (timer noise: the large transfer measured
///   no slower than the small one) → bandwidth from the large point
///   alone, the small point's full time as latency.
pub fn fit_transfer_estimate(points: &[(u64, u64)]) -> Option<TransferEstimate> {
    let lo = points.iter().copied().min_by_key(|p| p.0)?;
    let hi = points.iter().copied().max_by_key(|p| p.0)?;

    let bandwidth_from_point = |(bytes, ns): (u64, u64)| -> u64 {
        let bw = (bytes as u128).saturating_mul(1_000_000_000) / (ns.max(1) as u128);
        u64::try_from(bw).unwrap_or(u64::MAX).max(1)
    };

    if hi.0 == lo.0 {
        // One distinct size — can't separate latency from bandwidth;
        // attribute everything to bandwidth.
        return Some(TransferEstimate {
            bandwidth_bytes_per_sec: bandwidth_from_point(hi),
            latency_ns: 0,
        });
    }

    let dbytes = (hi.0 - lo.0) as u128;
    let dns = hi.1.saturating_sub(lo.1) as u128;
    if dns == 0 {
        // The big transfer wasn't measurably slower — noise. Take
        // the large point's amortized bandwidth; the small point's
        // time is then (over-)attributed to fixed latency, which is
        // the conservative direction.
        return Some(TransferEstimate {
            bandwidth_bytes_per_sec: bandwidth_from_point(hi),
            latency_ns: lo.1,
        });
    }

    let bw_u128 = dbytes.saturating_mul(1_000_000_000) / dns;
    let bandwidth = u64::try_from(bw_u128).unwrap_or(u64::MAX).max(1);
    let lo_transfer = (lo.0 as u128).saturating_mul(1_000_000_000) / bandwidth as u128;
    let latency = (lo.1 as u128).saturating_sub(lo_transfer);
    Some(TransferEstimate {
        bandwidth_bytes_per_sec: bandwidth,
        latency_ns: u64::try_from(latency).unwrap_or(u64::MAX),
    })
}

/// Probed per-path transfer estimates, keyed by `(src, dst)`
/// [`DeviceLocation`] pairs. Built once per topology generation by
/// [`TransferCalibration::calibrate`]; cached lazily on the
/// `SystemTopology` snapshot (its `transfer_calibration`).
///
/// Only paths the probe could actually exercise appear here (H2D +
/// D2H per reachable GPU device today). Everything else falls back
/// to [`TransferEstimate::fallback_for`] at lookup time — consumers
/// go through `SystemTopology::transfer_estimate`, which never
/// returns `None`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TransferCalibration {
    probed: HashMap<(DeviceLocation, DeviceLocation), TransferEstimate>,
}

impl TransferCalibration {
    /// Probe every device in `devices` that has a compiled-in,
    /// instantiable byte-storage path: H2D (`Cpu → dev`) and D2H
    /// (`dev → Cpu`) per GPU device, [`CALIBRATION_SIZES`] ×
    /// [`CALIBRATION_ITERS`] timed round-trips each, two-point fit.
    ///
    /// CPU-only hosts (or builds without the `cuda`/`vulkan`
    /// features) probe nothing and return an empty calibration —
    /// no probe, no error. A device whose backend fails to
    /// instantiate (driver missing at runtime) is silently skipped;
    /// its paths price via fallbacks.
    pub fn calibrate(devices: &[DeviceLocation]) -> TransferCalibration {
        let mut probed = HashMap::new();
        for &dev in devices {
            let pair = match dev {
                DeviceLocation::Cpu => None,
                #[cfg(feature = "cuda")]
                DeviceLocation::Cuda { gpu_id } => probe_cuda_device(gpu_id),
                #[cfg(feature = "vulkan")]
                DeviceLocation::Vulkan { gpu_id } => probe_vulkan_device(gpu_id),
                // Metal probe lands with the Metal byte-storage
                // surface; feature-gated backends not compiled in
                // also fall through to fallback pricing.
                _ => None,
            };
            if let Some((h2d, d2h)) = pair {
                probed.insert((DeviceLocation::Cpu, dev), h2d);
                probed.insert((dev, DeviceLocation::Cpu), d2h);
            }
        }
        TransferCalibration { probed }
    }

    /// Construct from explicit entries — for tests and future
    /// persisted-calibration loading.
    pub fn from_entries(
        entries: impl IntoIterator<Item = ((DeviceLocation, DeviceLocation), TransferEstimate)>,
    ) -> TransferCalibration {
        TransferCalibration {
            probed: entries.into_iter().collect(),
        }
    }

    /// The probed estimate for `(src, dst)`, if that path was
    /// measured this calibration. `None` means "not probed" — the
    /// caller falls back per path class.
    pub fn probed(&self, src: DeviceLocation, dst: DeviceLocation) -> Option<TransferEstimate> {
        self.probed.get(&(src, dst)).copied()
    }

    /// All probed paths. Diagnostics + tests.
    pub fn probed_paths(&self) -> &HashMap<(DeviceLocation, DeviceLocation), TransferEstimate> {
        &self.probed
    }

    /// True when nothing was probed (CPU-only host or no compiled-in
    /// GPU features).
    pub fn is_empty(&self) -> bool {
        self.probed.is_empty()
    }
}

/// Time H2D + D2H through a backend's byte-storage upload/download
/// closures at every calibration size; median per size; two-point
/// fit. Returns `None` if any transfer fails (backend unusable —
/// the caller skips the device and fallbacks apply).
#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn measure_calibration_points<S>(
    upload: impl Fn(&[u8]) -> Option<S>,
    download: impl Fn(&S) -> Option<Vec<u8>>,
) -> Option<(TransferEstimate, TransferEstimate)> {
    // Warmup round-trip at the smallest size — first calls pay
    // context init / allocator warm-up that isn't per-transfer cost.
    let warm = vec![0_u8; CALIBRATION_SIZES[0]];
    let s = upload(&warm)?;
    let _ = download(&s)?;
    drop(s);

    let mut h2d_points = Vec::with_capacity(CALIBRATION_SIZES.len());
    let mut d2h_points = Vec::with_capacity(CALIBRATION_SIZES.len());
    for &size in CALIBRATION_SIZES.iter() {
        let host = vec![1_u8; size];
        let mut h2d_ns = Vec::with_capacity(CALIBRATION_ITERS as usize);
        let mut d2h_ns = Vec::with_capacity(CALIBRATION_ITERS as usize);
        for _ in 0..CALIBRATION_ITERS {
            let t0 = Instant::now();
            let storage = upload(&host)?;
            h2d_ns.push(t0.elapsed().as_nanos() as u64);
            let t1 = Instant::now();
            let back = download(&storage)?;
            d2h_ns.push(t1.elapsed().as_nanos() as u64);
            debug_assert_eq!(back.len(), size);
        }
        h2d_ns.sort_unstable();
        d2h_ns.sort_unstable();
        h2d_points.push((size as u64, h2d_ns[h2d_ns.len() / 2]));
        d2h_points.push((size as u64, d2h_ns[d2h_ns.len() / 2]));
    }
    Some((
        fit_transfer_estimate(&h2d_points)?,
        fit_transfer_estimate(&d2h_points)?,
    ))
}

/// CUDA H2D/D2H probe through the Phase 7.5 byte-storage substrate
/// (`CudaStorageBytes::from_cpu_bytes` / `to_cpu_bytes`) — the same
/// path `Op::Copy` pays on the pipelined executor.
#[cfg(feature = "cuda")]
fn probe_cuda_device(gpu_id: usize) -> Option<(TransferEstimate, TransferEstimate)> {
    let dev = fuel_cuda_backend::CudaDevice::new(gpu_id).ok()?;
    measure_calibration_points(
        |bytes| fuel_cuda_backend::CudaStorageBytes::from_cpu_bytes(&dev, bytes).ok(),
        |storage| storage.to_cpu_bytes().ok(),
    )
}

/// Vulkan H2D/D2H probe through the Phase 7.5 byte-storage substrate
/// (`VulkanBackend::upload_bytes` / `download_bytes`).
#[cfg(feature = "vulkan")]
fn probe_vulkan_device(gpu_id: usize) -> Option<(TransferEstimate, TransferEstimate)> {
    let backend = fuel_vulkan_backend::VulkanBackend::with_selection(
        fuel_vulkan_backend::DeviceSelection::Index(gpu_id),
    )
    .ok()?;
    measure_calibration_points(
        |bytes| backend.upload_bytes(bytes).ok(),
        |storage| backend.download_bytes(storage).ok(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DeviceLocation;

    fn cpu_only_probe() -> ProbeReport {
        use fuel_ir::probe::DeviceDescriptor;
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

    // ===== Stage 1 — transfer calibration =====

    /// Synthetic perfectly-linear data: 2 bytes/ns (2 GB/s) bandwidth
    /// + 10 µs fixed latency. The two-point fit must recover both
    /// exactly.
    #[test]
    fn fit_recovers_synthetic_linear() {
        let bw_bytes_per_ns = 2_u64;
        let latency = 10_000_u64;
        let points: Vec<(u64, u64)> = CALIBRATION_SIZES
            .iter()
            .map(|&s| (s as u64, latency + s as u64 / bw_bytes_per_ns))
            .collect();
        let est = fit_transfer_estimate(&points).expect("3 points fit");
        assert_eq!(est.bandwidth_bytes_per_sec, 2_000_000_000);
        assert_eq!(est.latency_ns, latency);
    }

    #[test]
    fn fit_empty_returns_none() {
        assert!(fit_transfer_estimate(&[]).is_none());
    }

    /// One distinct size: bandwidth from that point, zero latency.
    #[test]
    fn fit_single_size_point() {
        let est = fit_transfer_estimate(&[(1024, 1000)]).expect("single point fits");
        assert_eq!(est.bandwidth_bytes_per_sec, 1_024_000_000);
        assert_eq!(est.latency_ns, 0);
    }

    /// Timer noise — the big transfer measured no slower (equal and
    /// inverted cases). Must not panic; bandwidth comes from the
    /// large point, the small point's time becomes latency.
    #[test]
    fn fit_nonincreasing_time_degrades_gracefully() {
        for hi_ns in [5_000_u64, 4_000] {
            let est = fit_transfer_estimate(&[(64_000, 5_000), (64_000_000, hi_ns)])
                .expect("degenerate points still fit");
            assert!(est.bandwidth_bytes_per_sec >= 1);
            assert_eq!(est.latency_ns, 5_000);
            // 64 MB / hi_ns amortized bandwidth.
            assert_eq!(
                est.bandwidth_bytes_per_sec,
                64_000_000_u64 * 1_000_000_000 / hi_ns,
            );
        }
    }

    /// estimate_ns is monotonic in bytes and saturates instead of
    /// overflowing at absurd sizes.
    #[test]
    fn estimate_ns_monotonic_and_saturating() {
        let est = TransferEstimate {
            bandwidth_bytes_per_sec: 8_000_000_000,
            latency_ns: 30_000,
        };
        let mut prev = 0_u64;
        for bytes in [0_u64, 1, 64 * 1024, 4 << 20, 64 << 20, 1 << 40] {
            let ns = est.estimate_ns(bytes);
            assert!(ns >= prev, "estimate_ns must be monotonic in bytes");
            prev = ns;
        }
        assert_eq!(est.estimate_ns(0), 30_000, "zero bytes still pays latency");
        // No panic / overflow at u64::MAX bytes.
        let _ = est.estimate_ns(u64::MAX);
        // Zero bandwidth is clamped, not divided by.
        let degenerate = TransferEstimate { bandwidth_bytes_per_sec: 0, latency_ns: 0 };
        let _ = degenerate.estimate_ns(1 << 30);
    }

    /// Fallbacks: every path class has a positive bandwidth, and the
    /// conservative ordering holds — slower path classes get lower
    /// bandwidth and higher latency.
    #[test]
    fn fallback_table_is_conservatively_ordered() {
        let same = TransferEstimate::fallback_for(TransferPath::SameDevice);
        let shared = TransferEstimate::fallback_for(TransferPath::SharedMemory);
        let peer = TransferEstimate::fallback_for(TransferPath::Peer);
        let copy = TransferEstimate::fallback_for(TransferPath::DeviceCopy);
        let staged = TransferEstimate::fallback_for(TransferPath::HostStaging);
        for e in [same, shared, peer, copy, staged] {
            assert!(e.bandwidth_bytes_per_sec >= 1);
        }
        assert!(same.estimate_ns(1 << 20) <= shared.estimate_ns(1 << 20));
        assert!(shared.bandwidth_bytes_per_sec > peer.bandwidth_bytes_per_sec);
        assert!(peer.bandwidth_bytes_per_sec > copy.bandwidth_bytes_per_sec);
        assert!(copy.bandwidth_bytes_per_sec > staged.bandwidth_bytes_per_sec);
        assert!(staged.latency_ns > copy.latency_ns);
        assert!(copy.latency_ns > peer.latency_ns);
        assert!(peer.latency_ns > shared.latency_ns);
        assert_eq!(same.latency_ns, 0);
        assert_eq!(same.estimate_ns(64 << 20), 0, "SameDevice moves nothing");
    }

    /// Serial staging composition: latencies add, bandwidths combine
    /// harmonically.
    #[test]
    fn compose_staged_adds_latency_halves_equal_bandwidth() {
        let leg = TransferEstimate {
            bandwidth_bytes_per_sec: 8_000_000_000,
            latency_ns: 30_000,
        };
        let staged = TransferEstimate::compose_staged(leg, leg);
        assert_eq!(staged.bandwidth_bytes_per_sec, 4_000_000_000);
        assert_eq!(staged.latency_ns, 60_000);
        // Asymmetric: 1/bw = 1/12e9 + 1/4e9 → 3e9.
        let fast = TransferEstimate { bandwidth_bytes_per_sec: 12_000_000_000, latency_ns: 1_000 };
        let slow = TransferEstimate { bandwidth_bytes_per_sec: 4_000_000_000, latency_ns: 2_000 };
        let mixed = TransferEstimate::compose_staged(fast, slow);
        assert_eq!(mixed.bandwidth_bytes_per_sec, 3_000_000_000);
        assert_eq!(mixed.latency_ns, 3_000);
    }

    /// CPU-only hosts: zero paths, no probe, no error.
    #[test]
    fn calibrate_cpu_only_is_empty() {
        assert!(TransferCalibration::calibrate(&[]).is_empty());
        assert!(TransferCalibration::calibrate(&[DeviceLocation::Cpu]).is_empty());
    }

    /// Probed-path lookup hits exact (src, dst) keys only.
    #[test]
    fn calibration_probed_lookup() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let h2d = TransferEstimate { bandwidth_bytes_per_sec: 10_000_000_000, latency_ns: 9_000 };
        let cal = TransferCalibration::from_entries([((DeviceLocation::Cpu, cuda0), h2d)]);
        assert_eq!(cal.probed(DeviceLocation::Cpu, cuda0), Some(h2d));
        assert_eq!(cal.probed(cuda0, DeviceLocation::Cpu), None, "reverse path not probed");
        assert!(!cal.is_empty());
        assert_eq!(cal.probed_paths().len(), 1);
    }
}
