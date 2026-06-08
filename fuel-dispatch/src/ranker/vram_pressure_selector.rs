//! `VramPressureSelector` — Phase 5.2 runtime selector that demotes
//! or skips candidates whose backend can't fit the projected output
//! at dispatch time.
//!
//! # Why this selector exists
//!
//! Picker 1 (the optimizer ranker) commits to a top-N at plan time
//! using static cost + Layer-2 Judge data. Neither layer sees the
//! current memory pressure on a device — that's a runtime signal.
//! When VRAM is near-saturated, the static winner may be a kernel
//! whose backend can no longer accommodate the projected output.
//! Picking that candidate would surface as an OOM at dispatch.
//!
//! `VramPressureSelector` reads each candidate's
//! [`BackendRuntime::would_fit`] for the projected output bytes and:
//!
//! - **[`FitStatus::WontFit`]** → skip the candidate entirely
//! - **[`FitStatus::Tight`]** → demote below Comfortable / Unknown
//!   (keep visible, but prefer Comfortable / Unknown alternatives
//!   if any exist)
//! - **[`FitStatus::Comfortable`]** / **[`FitStatus::Unknown`]** →
//!   leave in place. Unknown is honest "no signal"; we don't
//!   pretend it's pressure, so it sorts equal to Comfortable. The
//!   cost-rank winner breaks ties.
//!
//! If every candidate is `WontFit`, the selector falls back to the
//! static winner — the caller already lost (an OOM is coming), but
//! deferring to the winner gives the planner-level surface the
//! cleanest error site rather than silently picking nothing.
//!
//! # What needs to be passed at construction
//!
//! The trait's [`RuntimeSelector::select`] signature receives only
//! `&AlternativeSet`. Per-node context (the projected output bytes)
//! has to live on the selector. Phase 5.2 ships two extension
//! points:
//!
//! - `backend_runtime_lookup`: maps `(BackendId, DeviceLocation)` to
//!   a [`BackendRuntime`] handle. The runtime handle is borrowed for
//!   the duration of one `select` call — the lookup callback owns
//!   the lifecycle.
//! - `estimate_output_bytes`: maps a `Candidate` to a projected
//!   output-size estimate. The default
//!   ([`default_estimate_output_bytes`]) returns
//!   `c.static_cost.bytes_moved`, which is a deliberately
//!   pessimistic upper bound — `bytes_moved` includes reads +
//!   writes + intermediates, so it overcounts the true output
//!   allocation by ~3-10×. Callers with the live node-output
//!   shape resolved at plan time (Shape × DType bytes) should plug
//!   a tighter estimator in.
//!
//! Both are `Arc<dyn Fn>` so the selector stays cloneable + shareable
//! across executor threads.

use std::sync::Arc;

use fuel_core_types::backend::{BackendRuntime, FitStatus};
use fuel_core_types::probe::BackendId;
use fuel_core_types::DeviceLocation;

use super::{AlternativeSet, Candidate, RuntimeSelector};

/// Boxed backend-runtime handle. Returned by the
/// [`VramPressureSelector::backend_runtime_lookup`] callback for
/// each `(backend, device)` pair the selector needs to query.
pub type BackendRuntimeHandle = Box<dyn BackendRuntime + Send + Sync>;

/// Lookup callback shape — produces a [`BackendRuntimeHandle`] for a
/// given `(BackendId, DeviceLocation)`. `None` when the executor
/// hasn't materialized a backend for that pair (e.g. CUDA on a
/// host with no NVIDIA driver). The selector treats `None` as
/// [`FitStatus::Unknown`] — no signal, leave the candidate's
/// position untouched.
pub type BackendRuntimeLookup =
    Arc<dyn Fn(BackendId, DeviceLocation) -> Option<BackendRuntimeHandle> + Send + Sync>;

/// Output-size estimator callback — given a candidate, returns the
/// projected output bytes for the fit check. Callers that have the
/// exact post-realize output size from the graph plug a precise
/// estimator in; callers that don't can use [`default_estimate_output_bytes`]
/// as a deliberately pessimistic fallback.
pub type OutputBytesEstimator = Arc<dyn Fn(&Candidate) -> u64 + Send + Sync>;

/// Default output-size estimator: treats the candidate's
/// [`crate::fused::CostEstimate::bytes_moved`] as the projected
/// allocation. `bytes_moved` covers reads + writes + intermediates,
/// so this overcounts. That's deliberate — Phase 5.2's pressure
/// check is a safety net, and false-positives (calling a backend
/// `Tight` that's actually comfortable) demote rather than block.
/// Callers wanting tighter estimates can compute one from the
/// node's output `Shape` and `DType` at plan-build time and plug
/// their own estimator in.
pub fn default_estimate_output_bytes(c: &Candidate) -> u64 {
    c.static_cost.bytes_moved
}

/// Phase 5.2 runtime selector that demotes candidates whose backend
/// reports memory pressure for the projected output size.
///
/// See module docs for the full rationale + design tradeoffs.
///
/// # Construction
///
/// ```ignore
/// use std::sync::Arc;
/// use fuel_dispatch::ranker::{
///     VramPressureSelector, default_estimate_output_bytes,
/// };
///
/// let selector = VramPressureSelector::new(
///     /* backend_runtime_lookup */ Arc::new(|_b, _d| None),
///     /* estimate_output_bytes  */ Arc::new(default_estimate_output_bytes),
/// );
/// ```
#[derive(Clone)]
pub struct VramPressureSelector {
    backend_runtime_lookup: BackendRuntimeLookup,
    estimate_output_bytes: OutputBytesEstimator,
}

impl VramPressureSelector {
    /// Construct a new selector. Both callbacks are `Arc<dyn Fn>` so
    /// the selector is cheap to clone + share across threads.
    pub fn new(
        backend_runtime_lookup: BackendRuntimeLookup,
        estimate_output_bytes: OutputBytesEstimator,
    ) -> Self {
        Self {
            backend_runtime_lookup,
            estimate_output_bytes,
        }
    }

    /// Construct with the default [`default_estimate_output_bytes`]
    /// estimator. Use this when the caller only has a runtime-lookup
    /// callback and wants the conservative fallback estimate.
    pub fn with_default_estimator(backend_runtime_lookup: BackendRuntimeLookup) -> Self {
        Self::new(
            backend_runtime_lookup,
            Arc::new(default_estimate_output_bytes),
        )
    }

    /// Query the lookup callback and the candidate's fit status.
    /// `Unknown` is returned when:
    ///
    /// - the lookup callback returns `None` (no backend handle), or
    /// - the backend's `would_fit` returns `Unknown` (no memory
    ///   signal available).
    fn fit_status_for(&self, c: &Candidate) -> FitStatus {
        let size = (self.estimate_output_bytes)(c);
        match (self.backend_runtime_lookup)(c.backend, c.device) {
            Some(runtime) => runtime.would_fit(size),
            None => FitStatus::Unknown,
        }
    }
}

impl std::fmt::Debug for VramPressureSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramPressureSelector")
            .field("backend_runtime_lookup", &"<closure>")
            .field("estimate_output_bytes", &"<closure>")
            .finish()
    }
}

impl RuntimeSelector for VramPressureSelector {
    fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
        let alts = set.alternatives();
        if alts.is_empty() {
            return None;
        }

        // Classify each candidate. We then walk in original (cost-
        // ranked) order and pick the first comfortable / unknown /
        // tight survivor. Skipped candidates are never picked.
        //
        // Priority: {Comfortable, Unknown} > Tight. Unknown is
        // honest "no signal" — we don't pretend it's pressure, so
        // it sorts equal to Comfortable. We DON'T sort — we walk
        // in cost-rank order and remember the best status seen so
        // far. That way ties (including Comfortable-vs-Unknown
        // ties) break toward the static winner.
        let mut best_idx: Option<usize> = None;
        let mut best_status: Option<FitStatus> = None;

        // Score: lower is better. Comfortable = Unknown = 0, Tight = 1.
        fn score_of(s: FitStatus) -> u8 {
            match s {
                FitStatus::Comfortable | FitStatus::Unknown => 0,
                FitStatus::Tight => 1,
                FitStatus::WontFit => unreachable!("filtered above"),
            }
        }

        for (i, c) in alts.iter().enumerate() {
            let status = self.fit_status_for(c);
            if matches!(status, FitStatus::WontFit) {
                // Skip — under no circumstances pick this.
                continue;
            }

            let score = score_of(status);
            let prev_score = best_status.map(score_of);
            match prev_score {
                None => {
                    best_idx = Some(i);
                    best_status = Some(status);
                }
                Some(prev) if score < prev => {
                    best_idx = Some(i);
                    best_status = Some(status);
                }
                _ => {}
            }
        }

        // If every candidate was WontFit, fall back to the static
        // winner. The caller is in trouble either way — surfacing
        // the winner gives planner-level error reporting the cleanest
        // site (rather than silently returning None on a non-empty set,
        // which would violate the trait's contract).
        match best_idx {
            Some(i) => alts.get(i),
            None => set.winner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_core_types::backend::BackendRuntime;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DeviceLocation, Layout, Result};
    use fuel_storage::Storage;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn make_candidate(
        backend: BackendId,
        device: DeviceLocation,
        bytes_moved: u64,
    ) -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend,
            device,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: bytes_moved,
                bytes_moved,
                kernel_overhead_ns: 0,
            },
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    /// Mock BackendRuntime — returns configured (available, total).
    struct MockRuntime {
        available: Option<u64>,
        total: Option<u64>,
    }
    impl BackendRuntime for MockRuntime {
        fn available_bytes(&self) -> Option<u64> {
            self.available
        }
        fn total_bytes(&self) -> Option<u64> {
            self.total
        }
    }

    /// Build a lookup that returns the same `(available, total)` for
    /// the named backend, and `None` for every other backend.
    fn single_backend_lookup(
        target: BackendId,
        available: Option<u64>,
        total: Option<u64>,
    ) -> BackendRuntimeLookup {
        Arc::new(move |b, _d| {
            if b == target {
                Some(Box::new(MockRuntime { available, total })
                    as Box<dyn BackendRuntime + Send + Sync>)
            } else {
                None
            }
        })
    }

    /// Build a lookup that varies per backend.
    fn per_backend_lookup(
        cuda: (Option<u64>, Option<u64>),
        cpu: (Option<u64>, Option<u64>),
    ) -> BackendRuntimeLookup {
        Arc::new(move |b, _d| match b {
            BackendId::Cuda => Some(Box::new(MockRuntime {
                available: cuda.0,
                total: cuda.1,
            }) as Box<dyn BackendRuntime + Send + Sync>),
            BackendId::Cpu => Some(Box::new(MockRuntime {
                available: cpu.0,
                total: cpu.1,
            }) as Box<dyn BackendRuntime + Send + Sync>),
            _ => None,
        })
    }

    /// Helper to build a selector with the default-bytes estimator.
    fn make_selector(lookup: BackendRuntimeLookup) -> VramPressureSelector {
        VramPressureSelector::with_default_estimator(lookup)
    }

    /// Baseline: all candidates `Comfortable` → top winner picked.
    #[test]
    fn all_comfortable_picks_winner() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 200));

        // Both backends report 1MB free of 2MB total → comfortable.
        let lookup = per_backend_lookup(
            (Some(1_000_000), Some(2_000_000)),
            (Some(1_000_000), Some(2_000_000)),
        );
        let sel = make_selector(lookup);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(pick.backend, BackendId::Cuda, "winner is the cost-ranked first");
    }

    /// WontFit candidate gets skipped — picker falls through to the
    /// next admissible one.
    #[test]
    fn wont_fit_skips_candidate() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 1_000));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 1_000));

        // CUDA has only 500 bytes free (won't fit 1_000 estimate).
        // CPU has plenty of room.
        let lookup = per_backend_lookup(
            (Some(500), Some(10_000)),
            (Some(8_000), Some(10_000)),
        );
        let sel = make_selector(lookup);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cpu,
            "CUDA skipped due to WontFit; CPU is next admissible",
        );
    }

    /// Tight candidate gets demoted below Comfortable — even if it
    /// came first in cost rank.
    #[test]
    fn tight_demoted_below_comfortable() {
        let mut set = AlternativeSet::empty();
        // CUDA winner by cost (smaller bytes_moved).
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 100));

        // To trigger Tight: total > 0, size <= available, but
        // post-used / total > 0.85. With available = total
        // (nothing currently used), size > 0.85*total triggers Tight.
        // CUDA: available=100, total=100 → alloc 100 → 100% used
        // → 1.0 > 0.85 → Tight (and size <= available so not WontFit).
        // CPU: ample room → Comfortable.
        let lookup = per_backend_lookup(
            (Some(100), Some(100)),
            (Some(10_000), Some(10_000)),
        );
        let sel = make_selector(lookup);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cpu,
            "CUDA demoted (Tight) to favor Comfortable CPU even though CUDA was cost-winner",
        );
    }

    /// All candidates WontFit → fall back to the static winner (so
    /// the executor surfaces a clean OOM at dispatch rather than
    /// the selector silently returning None on a non-empty set).
    #[test]
    fn all_wont_fit_falls_back_to_winner() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 1_000));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 1_000));

        // Both backends report 1 byte free → both WontFit for the
        // 1_000-byte estimate.
        let lookup = per_backend_lookup(
            (Some(1), Some(10_000)),
            (Some(1), Some(10_000)),
        );
        let sel = make_selector(lookup);
        let pick = sel.select(&set).expect("non-empty (falls back to winner)");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "all WontFit → fall back to static winner",
        );
    }

    /// Unknown sorts equal to Comfortable — no penalty. The
    /// cost-winner that came first in the alt set keeps its
    /// position when its fit status is Unknown and the runner-up
    /// is Comfortable. We don't pretend "no signal" is pressure.
    #[test]
    fn unknown_ties_with_comfortable_preserves_winner() {
        let mut set = AlternativeSet::empty();
        // Cost winner is Cuda (cheaper). CPU is the runner-up.
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 100));

        // Lookup returns a runtime ONLY for CPU; CUDA is Unknown.
        // CPU is Comfortable; Unknown ties with Comfortable, so the
        // first-in-cost-rank survivor (CUDA) wins the tie.
        let lookup = single_backend_lookup(BackendId::Cpu, Some(10_000), Some(10_000));
        let sel = make_selector(lookup);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "Unknown ties with Comfortable; cost-rank winner preserved",
        );
    }

    /// Empty set → None (matches trait contract).
    #[test]
    fn empty_set_returns_none() {
        let set = AlternativeSet::empty();
        let lookup: BackendRuntimeLookup = Arc::new(|_, _| None);
        let sel = make_selector(lookup);
        assert!(sel.select(&set).is_none());
    }

    /// Custom output-bytes estimator is honored.
    #[test]
    fn custom_estimator_changes_fit_outcome() {
        let mut set = AlternativeSet::empty();
        // bytes_moved=100; backend has 50 free.
        set.push(make_candidate(BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 100));

        // With default estimator (uses bytes_moved=100): CUDA WontFit.
        // With custom estimator returning 10: CUDA Comfortable.
        let lookup = per_backend_lookup(
            (Some(50), Some(1_000)),
            (Some(50), Some(1_000)),
        );

        // Custom estimator returns 10 — well within 50.
        let custom: OutputBytesEstimator = Arc::new(|_c: &Candidate| 10u64);
        let sel = VramPressureSelector::new(lookup.clone(), custom);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "custom estimator returned size that fits → cost-winner stays",
        );

        // Default estimator → CUDA over-allocates (bytes_moved=100 vs
        // available=50) → WontFit → fallback to CPU. CPU also has
        // available=50 with bytes_moved=100 → also WontFit → all
        // WontFit → fall back to static winner (CUDA).
        let sel_default = VramPressureSelector::with_default_estimator(lookup);
        let pick = sel_default.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "all WontFit → static winner fallback",
        );
    }

    /// Debug impl prints the type name without panicking on the
    /// closure fields.
    #[test]
    fn debug_does_not_panic() {
        let lookup: BackendRuntimeLookup = Arc::new(|_, _| None);
        let sel = VramPressureSelector::with_default_estimator(lookup);
        let s = format!("{sel:?}");
        assert!(s.contains("VramPressureSelector"));
    }
}
