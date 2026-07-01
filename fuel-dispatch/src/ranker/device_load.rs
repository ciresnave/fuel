//! `DeviceLoadSelector` + the `load_tier` bucketing — Step E Phase C,
//! PR C2: the **live-load arm re-pick**.
//!
//! This is the concrete selector the [`super::runtime_selector`] module
//! sketch names (`DeviceLoadSelector` — "queue-depth + stream-utilization
//! probing") and that said "needs telemetry infrastructure that doesn't
//! exist yet". B1 shipped that infrastructure: the process-wide per-device
//! in-flight counter (`fuel-dispatch::dispatch::inflight_count`) exposed
//! through the Tier-2 [`BackendStreams::pending_work_count`] seam.
//!
//! # What C2 adds
//!
//! At each `Op::Branch` the runtime picker ranks the viable arms. C1 made
//! that ranking happen lazily at the streaming-walk frontier; C2 makes the
//! ranking **load-aware** by demoting arms whose device is busy *right
//! now*. The "right now" is the whole point of Step E: the executor picks
//! the path that drains the device queues fastest at the moment the
//! frontier reaches the branch.
//!
//! # The composition (where the load leg slots)
//!
//! Load is **not** a standalone production selector — it is folded into
//! [`super::ChainedSelector`] as a key leg between the VRAM guard and the
//! latency rank:
//!
//! ```text
//! (pressure_tier, load_tier, latency_ns, original_index)   ← pick the min
//!       │             │            │             │
//!       │             │            │             └ static-winner determinism
//!       │             │            └ Judge-measured / static composite
//!       │             └ THIS module: coarse bucket of pending_work_count / slot_capacity
//!       └ VRAM fit (WontFit skipped; Tight=1; Comfortable/Unknown=0)
//! ```
//!
//! **VRAM outranks load (critical).** `pressure_tier` stays the FIRST key
//! component, so load reorders arms ONLY within a VRAM fit tier — the
//! selector never picks a `WontFit`/OOM-adjacent device to balance load.
//! See [`super::ChainedSelector`]'s `select`.
//!
//! This module ships:
//!
//! - [`load_tier`] — the pure `(count, capacity) -> u8` bucketing the
//!   `ChainedSelector` key uses (and a standalone unit can test in
//!   isolation).
//! - [`DeviceLoadSelector`] — a standalone [`RuntimeSelector`] that ranks
//!   purely on `(load_tier, original_index)`. Not on the production path
//!   (that is `ChainedSelector`'s folded leg, per design §3.3(a)); it
//!   exists as the **unit-testable core** of the load decision, mirroring
//!   how `VramPressureSelector` is the standalone for the guard leg.
//! - [`LoadLookup`] / [`load_tier_for`] — read the live `pending_work_count`
//!   for a candidate's `(backend, device)` through the route picker's
//!   existing [`super::BackendRuntimeLookup`] (the handle the bridge hands
//!   out — `DeviceRuntimeHandle` — already implements `BackendStreams`, so
//!   the SAME lookup carries both the VRAM and the load signal; no second
//!   lookup is needed).
//!
//! # Honesty (no-signal = tier 0)
//!
//! A candidate whose handle is absent, is not a `BackendStreams` (CPU /
//! Reference — no queue concept), or reports `pending_work_count() == None`
//! gets **`load_tier` 0** — an honest "no load signal", exactly the way a
//! VRAM `Unknown` is treated. Tier 0 is NEVER a fabricated "idle": it means
//! "this leg has nothing to say", so an arm with no signal ties with a
//! genuinely-idle arm and the lower-index (static winner) breaks the tie.
//! With no signal anywhere every arm is tier 0 and the load leg vanishes
//! from the key — the degenerate-fallback that keeps C2 byte-identical to
//! pre-C2 dispatch (design §3.2).

use fuel_backend_contract::backend::BackendRuntime;
use fuel_ir::probe::BackendId;
use fuel_ir::DeviceLocation;

use super::{AlternativeSet, BackendRuntimeLookup, Candidate, RuntimeSelector};

/// Coarse load buckets for a device's live in-flight count, relative to
/// its advertised slot capacity. Coarse (3 buckets) on purpose — mirrors
/// the [`super::route_picker::free_bytes_bucket`] philosophy so jitter (a
/// single op submitted/drained between two reads) does not thrash the arm
/// pick or invalidate the `RouteCache` fingerprint, while a genuine
/// idle→saturated transition still reorders.
pub const LOAD_TIER_IDLE: u8 = 0;
/// Below ~one slot-capacity's worth of in-flight work, but not idle.
pub const LOAD_TIER_MODERATE: u8 = 1;
/// At or above the advertised slot capacity — the device's queues are
/// full; another arm on a less-loaded device drains faster.
pub const LOAD_TIER_SATURATED: u8 = 2;

/// Bucket a device's live in-flight `count` (from
/// [`BackendStreams::pending_work_count`]) against its advertised
/// `capacity` (from `slot_capacity`) into a coarse load tier.
///
/// Device-relative (`count / capacity`) so the tiering means the same
/// thing across a 1-stream CUDA device and a hypothetical N-queue one,
/// per design §9 open-Q1:
///
/// - `count == 0`                → [`LOAD_TIER_IDLE`] (nothing in flight).
/// - `count < capacity` (utilization in `(0, 1)`) → either idle or
///   moderate: utilization `< 0.25` is still [`LOAD_TIER_IDLE`] (a single
///   op on a multi-slot device isn't "loaded"), else
///   [`LOAD_TIER_MODERATE`].
/// - `count >= capacity` (utilization `>= 1.0`) → [`LOAD_TIER_SATURATED`].
///
/// `capacity == 0` is treated as 1 (a backend that opts into
/// `BackendStreams` has at least one slot; defending against a 0 avoids a
/// divide-by-zero and reads any non-zero count as saturated, which is the
/// honest pessimistic reading).
///
/// This is a pure function of two integers — no I/O, no panic — so it is
/// the unit-test seam for the whole load decision.
pub fn load_tier(count: u32, capacity: u32) -> u8 {
    if count == 0 {
        return LOAD_TIER_IDLE;
    }
    let capacity = capacity.max(1);
    if count >= capacity {
        return LOAD_TIER_SATURATED;
    }
    // count in (0, capacity): split idle vs moderate at 25% utilization so a
    // lone op on a wide device stays "idle". `count * 4 < capacity` is the
    // integer form of `count / capacity < 0.25`.
    if count.saturating_mul(4) < capacity {
        LOAD_TIER_IDLE
    } else {
        LOAD_TIER_MODERATE
    }
}

/// A lookup that hands the load-aware selector a [`BackendRuntime`] handle
/// for a `(backend, device)` — the SAME shape + the SAME instances as the
/// route picker's [`super::BackendRuntimeLookup`]. C2 reuses that lookup
/// rather than introducing a second one: the bridge's handle
/// (`DeviceRuntimeHandle`) already implements `BackendStreams`, so the
/// load signal is reachable through `as_backend_streams()` on the handle
/// the VRAM guard already queries (design §3.3 — one load source serves
/// all backends, the selector never holds executor internals).
pub type LoadLookup = BackendRuntimeLookup;

/// Read the live load tier for a candidate's `(backend, device)` through
/// `lookup`. Returns [`LOAD_TIER_IDLE`] (the honest no-signal tier) when:
///
/// - no lookup is configured,
/// - the lookup has no handle for the pair,
/// - the handle is not a `BackendStreams` (CPU / Reference — no queue
///   concept; `as_backend_streams()` is `None`), or
/// - the handle's `pending_work_count()` is `None` (a streaming backend
///   that genuinely can't report depth right now).
///
/// Never fabricates a non-zero load from a missing signal — see the module
/// honesty note.
pub fn load_tier_for(lookup: Option<&LoadLookup>, c: &Candidate) -> u8 {
    let Some(lookup) = lookup else {
        return LOAD_TIER_IDLE;
    };
    let Some(handle) = lookup(c.backend, c.device) else {
        return LOAD_TIER_IDLE;
    };
    load_tier_of_handle(handle.as_ref())
}

/// The load tier of one already-resolved runtime handle. Split out so
/// [`super::ChainedSelector`] (which already holds the boxed handle for
/// its VRAM `would_fit` query) can read load off the *same* handle without
/// a second lookup call.
pub(crate) fn load_tier_of_handle(handle: &dyn BackendRuntime) -> u8 {
    match handle.as_backend_streams() {
        Some(streams) => match streams.pending_work_count() {
            Some(count) => load_tier(count, streams.slot_capacity()),
            None => LOAD_TIER_IDLE,
        },
        None => LOAD_TIER_IDLE,
    }
}

/// Standalone load-only runtime selector — the **unit-testable core** of
/// the C2 load decision (design §3.3(b)). It ranks viable candidates by
/// `(load_tier, original_index)` and picks the minimum: the least-loaded
/// arm, ties broken toward the static winner (lowest index).
///
/// NOT the production path — production folds the load leg into
/// [`super::ChainedSelector`] so it composes with the VRAM guard + Judge
/// rank in one key (a standalone load selector can't see VRAM, so it could
/// pick a busy-but-fitting device's idle peer that won't fit — exactly the
/// composition problem `ChainedSelector` exists to solve). This selector
/// is the isolated proof that the `load_tier` leg picks the unloaded arm.
#[derive(Clone)]
pub struct DeviceLoadSelector {
    lookup: Option<LoadLookup>,
}

impl DeviceLoadSelector {
    /// Construct with a load lookup. `None` ⇒ every candidate is
    /// [`LOAD_TIER_IDLE`] ⇒ the pick is `set.winner()` (the no-signal
    /// degenerate, mirroring `ChainedSelector` with no signals).
    pub fn new(lookup: Option<LoadLookup>) -> Self {
        Self { lookup }
    }
}

impl std::fmt::Debug for DeviceLoadSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceLoadSelector")
            .field("lookup", &self.lookup.as_ref().map(|_| "<closure>"))
            .finish()
    }
}

impl RuntimeSelector for DeviceLoadSelector {
    fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
        let alts = set.alternatives();
        if alts.is_empty() {
            return None;
        }
        // Walk in original (cost-ranked / arm) order, tracking the minimum
        // (load_tier, idx) key. No VRAM guard here — that is the
        // production ChainedSelector's job; this standalone proves the
        // load leg alone picks the least-loaded arm.
        let mut best: Option<(u8, usize)> = None;
        for (i, c) in alts.iter().enumerate() {
            let key = (load_tier_for(self.lookup.as_ref(), c), i);
            if best.map_or(true, |b| key < b) {
                best = Some(key);
            }
        }
        match best {
            Some((_, i)) => alts.get(i),
            None => set.winner(),
        }
    }
}

/// A test-only [`fuel_backend_contract::backend::BackendStreams`] handle
/// reporting a fixed `(pending_work_count, slot_capacity)` — the simulated
/// live-load signal C2's tests inject in place of a real device. Lives
/// here (not behind `#[cfg(test)]`) so the route-picker + bridge tests can
/// build a fake load lookup without re-deriving the trait wiring.
pub struct MockBackendStreams {
    /// What `pending_work_count()` reports. `None` = "I can't measure" (a
    /// streaming backend with no depth available) ⇒ tier 0.
    pub pending: Option<u32>,
    /// What `slot_capacity()` reports.
    pub capacity: u32,
}

impl MockBackendStreams {
    /// A handle reporting `pending` in-flight ops against `capacity` slots.
    pub fn new(pending: Option<u32>, capacity: u32) -> Self {
        Self { pending, capacity }
    }
}

impl BackendRuntime for MockBackendStreams {
    // No VRAM signal — this mock carries only the load signal, so the VRAM
    // guard sees Unknown (tier 0) for it. The headline gate's two arms are
    // both VRAM-fit (Unknown), so the load leg is the only differentiator.
    fn available_bytes(&self) -> Option<u64> {
        None
    }
    fn total_bytes(&self) -> Option<u64> {
        None
    }
    fn as_backend_streams(
        &self,
    ) -> Option<&dyn fuel_backend_contract::backend::BackendStreams> {
        Some(self)
    }
}

impl fuel_backend_contract::backend::BackendStreams for MockBackendStreams {
    fn pending_work_count(&self) -> Option<u32> {
        self.pending
    }
    fn slot_capacity(&self) -> u32 {
        self.capacity
    }
    fn flush(&self) -> fuel_ir::Result<()> {
        Ok(())
    }
}

/// Build a load lookup that returns a [`MockBackendStreams`] per backend
/// from `(backend, pending, capacity)` entries; backends not listed
/// resolve to `None` (= no handle = tier 0). The shared shape the C2 tests
/// (here, route_picker, and the bridge) use to inject a fake live-load
/// signal. `device` is ignored — the entries key on `backend`, matching
/// how the route picker maps an arm to a single default device per
/// backend.
pub fn mock_load_lookup(entries: Vec<(BackendId, Option<u32>, u32)>) -> LoadLookup {
    use std::sync::Arc;
    Arc::new(move |b: BackendId, _d: DeviceLocation| {
        entries
            .iter()
            .find(|(eb, _, _)| *eb == b)
            .map(|&(_, pending, capacity)| {
                Box::new(MockBackendStreams::new(pending, capacity))
                    as super::BackendRuntimeHandle
            })
    })
}

/// A test-only runtime handle carrying BOTH a VRAM signal
/// (available/total, read by the VRAM guard) AND a live-load signal
/// (pending/capacity, read through `BackendStreams`) — the exact shape of
/// the production `DeviceRuntimeHandle`, which answers both `would_fit` and
/// `pending_work_count` off one handle. Used by the C2 headline-gate test
/// to build a 2-device branched graph whose arms are both VRAM-fit but
/// differ in live load.
pub struct MockCombinedRuntime {
    /// VRAM available bytes (the guard's `available_bytes`).
    pub available: Option<u64>,
    /// VRAM total bytes (the guard's `total_bytes`).
    pub total: Option<u64>,
    /// Live in-flight count (`pending_work_count`).
    pub pending: Option<u32>,
    /// Advertised slot capacity (`slot_capacity`).
    pub capacity: u32,
}

impl BackendRuntime for MockCombinedRuntime {
    fn available_bytes(&self) -> Option<u64> {
        self.available
    }
    fn total_bytes(&self) -> Option<u64> {
        self.total
    }
    fn as_backend_streams(
        &self,
    ) -> Option<&dyn fuel_backend_contract::backend::BackendStreams> {
        Some(self)
    }
}

impl fuel_backend_contract::backend::BackendStreams for MockCombinedRuntime {
    fn pending_work_count(&self) -> Option<u32> {
        self.pending
    }
    fn slot_capacity(&self) -> u32 {
        self.capacity
    }
    fn flush(&self) -> fuel_ir::Result<()> {
        Ok(())
    }
}

/// Build a combined VRAM+load lookup from
/// `(backend, available, total, pending, capacity)` entries — one
/// [`MockCombinedRuntime`] per backend, answering both the VRAM guard and
/// the load leg, exactly like the production `DeviceRuntimeHandle`.
/// Backends not listed resolve to `None`.
pub fn mock_combined_lookup(
    entries: Vec<(BackendId, Option<u64>, Option<u64>, Option<u32>, u32)>,
) -> LoadLookup {
    use std::sync::Arc;
    Arc::new(move |b: BackendId, _d: DeviceLocation| {
        entries
            .iter()
            .find(|(eb, _, _, _, _)| *eb == b)
            .map(|&(_, available, total, pending, capacity)| {
                Box::new(MockCombinedRuntime {
                    available,
                    total,
                    pending,
                    capacity,
                }) as super::BackendRuntimeHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_ir::{DeviceLocation, Layout, Result};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn make_candidate(backend: BackendId, device: DeviceLocation) -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend,
            device,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: 0,
                bytes_moved: 0,
                kernel_overhead_ns: 0,
            },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    // ===== load_tier bucketing =====

    /// Zero in flight is always idle, regardless of capacity.
    #[test]
    fn load_tier_zero_is_idle() {
        assert_eq!(load_tier(0, 1), LOAD_TIER_IDLE);
        assert_eq!(load_tier(0, 8), LOAD_TIER_IDLE);
    }

    /// At/above capacity is saturated.
    #[test]
    fn load_tier_at_or_above_capacity_is_saturated() {
        assert_eq!(load_tier(1, 1), LOAD_TIER_SATURATED);
        assert_eq!(load_tier(8, 8), LOAD_TIER_SATURATED);
        assert_eq!(load_tier(100, 1), LOAD_TIER_SATURATED);
        assert_eq!(load_tier(9, 8), LOAD_TIER_SATURATED);
    }

    /// Below 25% utilization on a wide device is still idle; at/above 25%
    /// (but below capacity) is moderate.
    #[test]
    fn load_tier_splits_idle_and_moderate_at_quarter() {
        // capacity 8: count 1 → 12.5% → idle; count 2 → 25% → moderate.
        assert_eq!(load_tier(1, 8), LOAD_TIER_IDLE);
        assert_eq!(load_tier(2, 8), LOAD_TIER_MODERATE);
        assert_eq!(load_tier(7, 8), LOAD_TIER_MODERATE);
    }

    /// The single-slot device (the B1 default `slot_capacity == 1`): any
    /// in-flight work is saturated, zero is idle — there is no moderate
    /// band, which is exactly right for a 1-stream device.
    #[test]
    fn load_tier_single_slot_is_binary() {
        assert_eq!(load_tier(0, 1), LOAD_TIER_IDLE);
        assert_eq!(load_tier(1, 1), LOAD_TIER_SATURATED);
        assert_eq!(load_tier(5, 1), LOAD_TIER_SATURATED);
    }

    /// `capacity == 0` defends against divide-by-zero and reads any
    /// non-zero count as saturated (the pessimistic honest reading).
    #[test]
    fn load_tier_zero_capacity_is_safe() {
        assert_eq!(load_tier(0, 0), LOAD_TIER_IDLE);
        assert_eq!(load_tier(1, 0), LOAD_TIER_SATURATED);
    }

    // ===== load_tier_for: reading through the lookup =====

    /// No lookup ⇒ tier 0 (no signal).
    #[test]
    fn load_tier_for_no_lookup_is_idle() {
        let c = make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 });
        assert_eq!(load_tier_for(None, &c), LOAD_TIER_IDLE);
    }

    /// A backend absent from the lookup ⇒ tier 0 (no handle).
    #[test]
    fn load_tier_for_absent_backend_is_idle() {
        let lookup = mock_load_lookup(vec![(BackendId::Cuda, Some(8), 1)]);
        let cpu = make_candidate(BackendId::Cpu, DeviceLocation::Cpu);
        assert_eq!(load_tier_for(Some(&lookup), &cpu), LOAD_TIER_IDLE);
    }

    /// `pending_work_count() == None` (a streaming backend that can't
    /// report depth) ⇒ tier 0, NOT a fabricated load.
    #[test]
    fn load_tier_for_none_count_is_idle() {
        let lookup = mock_load_lookup(vec![(BackendId::Cuda, None, 1)]);
        let cuda = make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 });
        assert_eq!(load_tier_for(Some(&lookup), &cuda), LOAD_TIER_IDLE);
    }

    /// A saturated device reads as saturated through the lookup.
    #[test]
    fn load_tier_for_reads_saturated() {
        let lookup = mock_load_lookup(vec![(BackendId::Cuda, Some(4), 1)]);
        let cuda = make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 });
        assert_eq!(load_tier_for(Some(&lookup), &cuda), LOAD_TIER_SATURATED);
    }

    // ===== DeviceLoadSelector =====

    /// No lookup ⇒ the standalone selector picks the winner (arm 0) — the
    /// no-signal degenerate.
    #[test]
    fn selector_no_signal_picks_winner() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }));
        set.push(make_candidate(BackendId::Vulkan, DeviceLocation::Vulkan { gpu_id: 0 }));
        let sel = DeviceLoadSelector::new(None);
        assert_eq!(sel.select(&set).expect("non-empty").backend, BackendId::Cuda);
    }

    /// The headline mechanic in isolation: arm-0 (CUDA) is SATURATED, arm-1
    /// (Vulkan) is idle ⇒ the load-only selector flips to the unloaded
    /// arm-1.
    #[test]
    fn selector_flips_to_unloaded_arm() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }));
        set.push(make_candidate(BackendId::Vulkan, DeviceLocation::Vulkan { gpu_id: 0 }));

        // CUDA saturated (4 in flight, 1 slot), Vulkan idle (0 in flight).
        let lookup = mock_load_lookup(vec![
            (BackendId::Cuda, Some(4), 1),
            (BackendId::Vulkan, Some(0), 1),
        ]);
        let sel = DeviceLoadSelector::new(Some(lookup));
        assert_eq!(
            sel.select(&set).expect("non-empty").backend,
            BackendId::Vulkan,
            "saturated arm-0 demoted below idle arm-1",
        );
    }

    /// Equal load ⇒ the static winner (arm 0) breaks the tie — determinism.
    #[test]
    fn selector_equal_load_keeps_winner() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }));
        set.push(make_candidate(BackendId::Vulkan, DeviceLocation::Vulkan { gpu_id: 0 }));

        // Both saturated → same tier → lower index wins.
        let lookup = mock_load_lookup(vec![
            (BackendId::Cuda, Some(4), 1),
            (BackendId::Vulkan, Some(4), 1),
        ]);
        let sel = DeviceLoadSelector::new(Some(lookup));
        assert_eq!(sel.select(&set).expect("non-empty").backend, BackendId::Cuda);
    }

    /// Empty set ⇒ None (trait contract).
    #[test]
    fn selector_empty_set_is_none() {
        let sel = DeviceLoadSelector::new(None);
        assert!(sel.select(&AlternativeSet::empty()).is_none());
    }

    /// Debug impl does not panic on the closure field.
    #[test]
    fn debug_does_not_panic() {
        let sel = DeviceLoadSelector::new(Some(mock_load_lookup(vec![])));
        assert!(format!("{sel:?}").contains("DeviceLoadSelector"));
    }
}
