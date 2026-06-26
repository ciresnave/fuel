//! OS-level system RAM queries for the CPU backend's
//! [`fuel_ir::backend::BackendRuntime`] impl.
//!
//! Architectural note: system RAM is shared with the entire OS, not
//! just Fuel's workload. The reported value reflects total
//! system-wide free / total memory, not per-process. Selectors that
//! consume this signal should weight it accordingly (e.g.
//! `available_bytes` dropping suddenly may reflect another process
//! starting, not Fuel allocating). Compare to GPU backends where
//! `vram_free` is typically per-device (still shared with other
//! processes but at least bounded to one device).
//!
//! ## Platform coverage (v1)
//!
//! - **Linux**: parses `/proc/meminfo` (no external dep). Reads
//!   `MemAvailable` (or `MemFree` on older kernels) and `MemTotal`.
//! - **Windows**: calls `GlobalMemoryStatusEx` via `windows-sys`.
//!   Reports `ullAvailPhys` and `ullTotalPhys`.
//! - **macOS / others**: returns `None`. The trait contract says
//!   `None` means "no signal available"; selectors fall back to
//!   static cost. A macOS impl using `host_statistics64` /
//!   `vm_statistics64` can land later.
//!
//! ## Caching
//!
//! Both queries are cached for `CACHE_TTL_NANOS` (~100ms) via an
//! atomic-timestamp + mutex-guarded cell pattern. The cache is
//! shared across all [`CpuBackendDevice`] instances (the device is
//! stateless; the singleton cache is correct). Selectors poll at
//! sub-realize granularity; caching keeps the hot path cheap.

use std::sync::Mutex;
use std::time::Instant;

/// Cached system-memory snapshot. Refreshed lazily when older than
/// [`CACHE_TTL`].
#[derive(Debug, Clone, Copy)]
struct Snapshot {
    available_bytes: Option<u64>,
    total_bytes: Option<u64>,
    captured_at: Instant,
}

/// Cache TTL — re-query the OS at most once per ~100ms.
const CACHE_TTL_MILLIS: u128 = 100;

/// Process-wide cache. Stateless backends share one snapshot;
/// `Mutex` serializes refresh attempts (selectors that hammer
/// `available_bytes` from multiple threads still pay only one
/// query per TTL window).
static SNAPSHOT: Mutex<Option<Snapshot>> = Mutex::new(None);

/// Bytes of system RAM the OS reports as currently available to new
/// allocations. `None` on platforms without an implemented query.
///
/// "Available" semantics:
///
/// - Linux: `MemAvailable` (Kernel ≥3.14) — estimate of memory
///   available for new allocations including reclaimable cache,
///   without swapping. Falls back to `MemFree + Buffers + Cached`
///   on older kernels.
/// - Windows: `ullAvailPhys` from `GlobalMemoryStatusEx` — physical
///   memory currently available.
pub fn available_bytes() -> Option<u64> {
    snapshot().available_bytes
}

/// Total physical RAM in bytes. `None` on platforms without an
/// implemented query.
pub fn total_bytes() -> Option<u64> {
    snapshot().total_bytes
}

/// Refresh-on-stale snapshot accessor. The mutex serializes
/// concurrent refresh attempts so a thundering herd still incurs
/// one OS query per TTL window. Reads of a still-fresh snapshot
/// take the lock just long enough to clone the cached value.
fn snapshot() -> Snapshot {
    let mut guard = SNAPSHOT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = guard.as_ref() {
        if s.captured_at.elapsed().as_millis() < CACHE_TTL_MILLIS {
            return *s;
        }
    }
    let fresh = capture();
    *guard = Some(fresh);
    fresh
}

/// Platform-dispatch entry point. Each `cfg`-gated impl produces
/// the same `Snapshot` shape; unknown platforms return `None` for
/// both fields.
fn capture() -> Snapshot {
    let (available, total) = capture_platform();
    Snapshot {
        available_bytes: available,
        total_bytes: total,
        captured_at: Instant::now(),
    }
}

// -----------------------------------------------------------------
// Platform implementations
// -----------------------------------------------------------------

#[cfg(target_os = "linux")]
fn capture_platform() -> (Option<u64>, Option<u64>) {
    // /proc/meminfo on Linux. Format is a series of
    //   FieldName:    1234 kB
    // lines. We want MemAvailable (preferred) and MemTotal.
    let contents = match std::fs::read_to_string("/proc/meminfo") {
        Ok(s) => s,
        Err(_) => return (None, None),
    };
    let mut available = None;
    let mut total = None;
    let mut mem_free = None;
    let mut buffers = None;
    let mut cached = None;
    for line in contents.lines() {
        if let Some((field, rest)) = line.split_once(':') {
            // Value is "  1234 kB" — strip whitespace + unit.
            let kb = rest.trim().strip_suffix("kB").unwrap_or(rest.trim());
            let kb: u64 = match kb.trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let bytes = kb.saturating_mul(1024);
            match field {
                "MemAvailable" => available = Some(bytes),
                "MemTotal" => total = Some(bytes),
                "MemFree" => mem_free = Some(bytes),
                "Buffers" => buffers = Some(bytes),
                "Cached" => cached = Some(bytes),
                _ => {}
            }
        }
    }
    // Older kernels (<3.14) lack MemAvailable; fall back to
    // MemFree + Buffers + Cached as a coarse approximation.
    let available = available.or_else(|| {
        match (mem_free, buffers, cached) {
            (Some(f), Some(b), Some(c)) => Some(f + b + c),
            (Some(f), _, _) => Some(f),
            _ => None,
        }
    });
    (available, total)
}

#[cfg(target_os = "windows")]
fn capture_platform() -> (Option<u64>, Option<u64>) {
    use windows_sys::Win32::System::SystemInformation::{
        GlobalMemoryStatusEx, MEMORYSTATUSEX,
    };
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        return (None, None);
    }
    (Some(status.ullAvailPhys), Some(status.ullTotalPhys))
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn capture_platform() -> (Option<u64>, Option<u64>) {
    // macOS / BSD / others: no impl yet. Returning None is correct
    // per the trait contract — selectors fall back to static cost.
    // A macOS impl using `host_statistics64` / `vm_statistics64`
    // (via the libc dep already in fuel-cpu-backend's optional deps)
    // can land as a follow-up commit.
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: on a platform with an implemented query, both
    /// fields are `Some` and total >= available. On unsupported
    /// platforms, both are `None`. Either is a valid honest report.
    #[test]
    fn snapshot_is_internally_consistent() {
        let s = snapshot();
        match (s.available_bytes, s.total_bytes) {
            (Some(a), Some(t)) => {
                assert!(
                    a <= t,
                    "available ({a}) must not exceed total ({t}) — \
                     OS query inconsistency",
                );
                assert!(t > 0, "total must be positive on a real system");
            }
            (None, None) => {
                // Unsupported platform — valid per the trait contract.
            }
            (Some(_), None) | (None, Some(_)) => {
                panic!("inconsistent snapshot: one field measured, the other not");
            }
        }
    }

    /// Cache hit returns the same snapshot for back-to-back calls
    /// within the TTL window.
    #[test]
    fn back_to_back_calls_hit_cache() {
        let a = snapshot();
        let b = snapshot();
        assert_eq!(a.captured_at, b.captured_at);
    }
}
