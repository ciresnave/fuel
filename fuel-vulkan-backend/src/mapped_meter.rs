// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host-visible *mapped* byte accounting — a process-wide peak counter over the
//! aperture-relevant quantity.
//!
//! Motivation (2026-07-31 host-aperture kernel bugcheck; see
//! `docs/postmortems/2026-07-31-gpu-host-aperture-crash.md`): the CPU host
//! aperture — the PCIe BAR window the CPU uses to address video memory — is a
//! separate, smaller, fragmentation-sensitive budget from total VRAM, and it is
//! invisible to ordinary VRAM budget queries (`cuMemGetInfo`, Vulkan
//! memory-budget both report VRAM). The aperture cares about CPU-**mapped**
//! bytes, not resident VRAM — so a resident-VRAM counter measures the wrong
//! quantity and a clean reading of it is a false negative that looks like a
//! result. This meter tracks the one aperture quantity we can attribute to
//! ourselves: the host-visible allocations we actually map.
//!
//! Scope: it counts bytes for host-visible memory *we* map — every host-visible
//! mapped staging allocation on the transfer paths: H2D (`upload_bytes`,
//! `write_bytes`, `upload_slice`) and D2H (`download_bytes`). Device-local tensor
//! VRAM is not host-visible and is not mapped by us, so it does not count here.
//!
//! **This is an UPPER-BOUND proxy, not exact BAR occupancy — read the number with
//! that caveat.** The BAR aperture is consumed specifically by memory that is BOTH
//! `DEVICE_LOCAL` and `HOST_VISIBLE` (the ReBAR-exposed VRAM window). *Plain*
//! host-visible memory (system RAM the GPU reads over PCIe) is mapped into the CPU
//! address space but does NOT consume the GPU's BAR aperture. This meter counts
//! all host-visible mapped bytes without distinguishing the two, so it is a
//! conservative superset of true aperture use — and after the allocator's
//! mitigation-2 change (host-visible staging now *prefers* a system-RAM type), much
//! of what it counts may be system RAM, not BAR. Counting it as literal aperture
//! would repeat the very "measured the wrong quantity" error the post-mortem warns
//! about; treat it as "our host-visible mapped footprint (an aperture upper bound)".
//! Precise BAR-only accounting would need per-allocation memory-type introspection
//! (does this type carry `DEVICE_LOCAL`?) — a future refinement. It also does not
//! count the D2H download pool's one-time reservation, only the active per-download
//! staging. Partial by design: it sees our footprint, not the machine's total.

use std::sync::atomic::{AtomicU64, Ordering};

/// A thread-safe current + peak byte meter. Pure — no device, no I/O — so it is
/// unit-testable without a GPU.
///
/// Call [`record_map`](Self::record_map) when a host-visible mapping is acquired
/// and [`record_unmap`](Self::record_unmap) when it is released; [`peak`](Self::peak)
/// is the high-water mark of [`current`](Self::current) since the last
/// [`reset_peak`](Self::reset_peak).
///
/// Both accumulators saturate rather than wrap: a byte counter that wrapped to a
/// small value would read as "aperture is fine" — the precise false-negative
/// this instrument exists to prevent — and an unmatched `record_unmap` must not
/// underflow to `~u64::MAX` and poison the peak forever.
#[derive(Debug)]
pub struct MappedByteMeter {
    current: AtomicU64,
    peak: AtomicU64,
}

impl MappedByteMeter {
    /// A fresh meter reading zero. `const` so a process-global `static` instance
    /// can be constructed without lazy initialization.
    pub const fn new() -> Self {
        Self {
            current: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }
    }

    /// Record `bytes` newly mapped. Saturating add on `current`, then bump the
    /// peak high-water mark if the new `current` exceeds it.
    pub fn record_map(&self, bytes: u64) {
        let mut cur = self.current.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_add(bytes);
            match self.current.compare_exchange_weak(
                cur,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.bump_peak(next);
                    break;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// Record `bytes` unmapped. Saturating sub on `current` (never underflow).
    /// The peak is a high-water mark and is deliberately left untouched.
    pub fn record_unmap(&self, bytes: u64) {
        let mut cur = self.current.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_sub(bytes);
            match self.current.compare_exchange_weak(
                cur,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Raise the peak high-water mark to `candidate` if it exceeds the current
    /// peak. Lock-free CAS loop; a concurrent writer that already pushed the peak
    /// higher wins (the `while candidate > peak` guard keeps it monotonic).
    fn bump_peak(&self, candidate: u64) {
        let mut peak = self.peak.load(Ordering::Relaxed);
        while candidate > peak {
            match self.peak.compare_exchange_weak(
                peak,
                candidate,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }

    /// Host-visible mapped bytes currently outstanding.
    pub fn current(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// Peak of `current` since construction or the last [`reset_peak`](Self::reset_peak).
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Reset the peak to the **current** value — not to zero. The bytes still
    /// mapped are still occupying the aperture, so a fresh measurement window
    /// starts from what is presently held, not from an empty slate.
    pub fn reset_peak(&self) {
        self.peak.store(self.current(), Ordering::Relaxed);
    }
}

impl Default for MappedByteMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-wide meter over every host-visible mapped byte this process requests
/// through the Vulkan backend. One instance per process, because the aperture is
/// a per-process/machine resource shared across all `VulkanBackend` instances —
/// per-backend meters would each see only a slice of the real pressure.
static HOST_MAPPED: MappedByteMeter = MappedByteMeter::new();

/// Record `bytes` of host-visible memory just mapped (call at the map site).
pub fn record_host_map(bytes: u64) {
    HOST_MAPPED.record_map(bytes);
}

/// Record `bytes` of host-visible memory just unmapped (call at the unmap / free site).
pub fn record_host_unmap(bytes: u64) {
    HOST_MAPPED.record_unmap(bytes);
}

/// Host-visible mapped bytes this process currently holds.
pub fn mapped_host_visible_bytes() -> u64 {
    HOST_MAPPED.current()
}

/// Peak host-visible mapped bytes since process start (or the last reset). This
/// is the aperture-footprint number to watch.
pub fn mapped_host_visible_peak_bytes() -> u64 {
    HOST_MAPPED.peak()
}

/// Reset the process-wide peak high-water mark to the current mapped total, to
/// open a fresh measurement window.
pub fn reset_host_mapped_peak() {
    HOST_MAPPED.reset_peak();
}

/// RAII guard that accounts a host-visible mapping in the process-wide meter for
/// exactly its own lifetime: [`record_host_map`] on construction,
/// [`record_host_unmap`] on drop.
///
/// Bind one to a mapped staging allocation (`let _g = MappedGuard::new(size);`)
/// and the unmap is recorded on **every** exit from the scope — the success path,
/// a `?` early return, or an unwind — so a mapping that errors part-way can never
/// ratchet the counter upward without a matching release. That leak mode (a failed
/// host mapping whose block is never freed, turning a recoverable failure into
/// progressive exhaustion) is one of the latent defects the 2026-07-31
/// host-aperture post-mortem called out; the guard makes it structurally
/// impossible for the *accounting* to drift the same way.
#[derive(Debug)]
pub struct MappedGuard {
    bytes: u64,
}

impl MappedGuard {
    /// Record `bytes` mapped now; the matching unmap fires on drop.
    pub fn new(bytes: u64) -> Self {
        record_host_map(bytes);
        Self { bytes }
    }
}

impl Drop for MappedGuard {
    fn drop(&mut self) {
        record_host_unmap(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Tests that touch the process-global `HOST_MAPPED` meter run in the same
    /// test binary and would race on it under Rust's default parallel harness.
    /// Serialize them so each sees a stable baseline for its delta assertions.
    static GLOBAL_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn records_current_and_tracks_peak() {
        let m = MappedByteMeter::new();
        assert_eq!(m.current(), 0);
        assert_eq!(m.peak(), 0);

        m.record_map(100);
        assert_eq!(m.current(), 100);
        assert_eq!(m.peak(), 100);

        m.record_map(200);
        assert_eq!(m.current(), 300);
        assert_eq!(m.peak(), 300);
    }

    #[test]
    fn peak_is_a_high_water_mark_that_survives_unmap() {
        let m = MappedByteMeter::new();
        m.record_map(500);
        m.record_unmap(400);
        // Current drops, but the peak remembers the high point.
        assert_eq!(m.current(), 100);
        assert_eq!(m.peak(), 500);
    }

    #[test]
    fn unmap_saturates_and_never_underflows() {
        let m = MappedByteMeter::new();
        m.record_map(100);
        // Unmap more than is outstanding (an accounting bug upstream must NOT
        // wrap current to ~u64::MAX and read as catastrophic aperture use).
        m.record_unmap(250);
        assert_eq!(m.current(), 0);
        // Peak still reflects the real high point, unpoisoned.
        assert_eq!(m.peak(), 100);
    }

    #[test]
    fn map_saturates_and_never_overflows() {
        let m = MappedByteMeter::new();
        m.record_map(u64::MAX);
        assert_eq!(m.current(), u64::MAX);
        m.record_map(1);
        // Saturates at the ceiling rather than wrapping to a small value.
        assert_eq!(m.current(), u64::MAX);
        assert_eq!(m.peak(), u64::MAX);
    }

    #[test]
    fn reset_peak_drops_to_current_not_zero() {
        let m = MappedByteMeter::new();
        m.record_map(300);
        m.record_unmap(100);
        assert_eq!(m.current(), 200);
        assert_eq!(m.peak(), 300);

        m.reset_peak();
        // Peak rebased to what is still mapped, not to an empty slate.
        assert_eq!(m.peak(), 200);
        assert_eq!(m.current(), 200);

        // A subsequent smaller spike does not lift the rebased peak below current…
        m.record_map(50);
        assert_eq!(m.peak(), 250);
    }

    #[test]
    fn concurrent_maps_sum_exactly() {
        let m = Arc::new(MappedByteMeter::new());
        let threads = 8u64;
        let per_thread = 10_000u64;
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let m = Arc::clone(&m);
                thread::spawn(move || {
                    for _ in 0..per_thread {
                        m.record_map(1);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let total = threads * per_thread;
        assert_eq!(m.current(), total);
        assert_eq!(m.peak(), total);
    }

    #[test]
    fn concurrent_map_unmap_balances_to_zero() {
        let m = Arc::new(MappedByteMeter::new());
        let threads = 8u64;
        let iters = 10_000u64;
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let m = Arc::clone(&m);
                thread::spawn(move || {
                    for _ in 0..iters {
                        m.record_map(64);
                        m.record_unmap(64);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Every map is matched by an unmap, so the net must be exactly zero…
        assert_eq!(m.current(), 0);
        // …but the peak observed at least one concurrently-held mapping.
        assert!(
            m.peak() >= 64,
            "peak should have observed live mappings, got {}",
            m.peak()
        );
    }

    #[test]
    fn process_global_accessors_are_wired() {
        let _lock = GLOBAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // The process-global is shared across the whole test binary, so assert on
        // deltas rather than absolute values, under the serialization lock.
        let before = mapped_host_visible_bytes();
        record_host_map(4096);
        assert_eq!(mapped_host_visible_bytes(), before + 4096);
        assert!(mapped_host_visible_peak_bytes() >= before + 4096);
        record_host_unmap(4096);
        assert_eq!(mapped_host_visible_bytes(), before);
    }

    #[test]
    fn mapped_guard_accounts_for_its_lifetime() {
        let _lock = GLOBAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let before = mapped_host_visible_bytes();
        {
            let _g = MappedGuard::new(8192);
            assert_eq!(mapped_host_visible_bytes(), before + 8192);
            assert!(mapped_host_visible_peak_bytes() >= before + 8192);
        }
        // Guard dropped at scope end → unmap recorded, back to baseline.
        assert_eq!(mapped_host_visible_bytes(), before);
    }

    #[test]
    fn mapped_guard_records_unmap_even_on_early_return() {
        let _lock = GLOBAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let before = mapped_host_visible_bytes();
        // A `?`-style early return leaves the scope without an explicit unmap;
        // the guard's Drop must still fire (this is the anti-ratchet property).
        fn scope_that_returns_early(bytes: u64) -> Result<(), ()> {
            let _g = MappedGuard::new(bytes);
            Err(())
        }
        let _ = scope_that_returns_early(4096);
        assert_eq!(mapped_host_visible_bytes(), before);
    }
}
