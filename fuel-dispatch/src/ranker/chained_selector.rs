//! `ChainedSelector` — the production Picker 2 composition:
//! VRAM-pressure GUARD → Judge-measured RANK → static-winner
//! fallback.
//!
//! Picker-arc step 3 (2026-06-10). Phase 5.2 shipped
//! [`super::VramPressureSelector`] and [`super::JudgeAwareSelector`]
//! as standalone selectors with zero production consumers. The two
//! cannot be composed through [`RuntimeSelector::select`] — `select`
//! returns a single pick, not a narrowed set — and `JudgeAwareSelector`
//! must be constructed per decision point while the executor takes
//! ONE selector per realize. This module ships the composition as
//! its own selector with explicit semantics, reading the per-node
//! `(op, dtype, size_class)` from the [`DecisionContext`] that
//! `compile_plan` stamps on every [`AlternativeSet`].
//!
//! # Semantics
//!
//! For each candidate, compute a sort key
//! `(pressure_tier, load_tier, latency_ns, original_index)` and pick
//! the minimum:
//!
//! 1. **Pressure tier (the guard).** Query the candidate backend's
//!    [`fuel_backend_contract::backend::BackendRuntime::would_fit`] for
//!    the estimated output bytes:
//!    - [`FitStatus::WontFit`] → the candidate is SKIPPED — never
//!      picked while any alternative survives.
//!    - [`FitStatus::Tight`] → tier 1 (demoted below all tier-0
//!      candidates — pressure reorders only under contention).
//!    - [`FitStatus::Comfortable`] / [`FitStatus::Unknown`] →
//!      tier 0. Unknown is honest "no signal", NOT pressure — it
//!      ties with Comfortable (post-remediation semantics).
//!    No runtime lookup configured, or no handle for the pair, also
//!    yields Unknown.
//! 2. **Load tier (the live-load re-pick — Step E Phase C / C2).** A
//!    coarse bucket of the candidate device's live in-flight count over
//!    its slot capacity ([`super::device_load::load_tier`]), read off the
//!    SAME runtime handle the guard queried, through its Tier-2
//!    [`fuel_backend_contract::backend::BackendStreams`] seam: idle (0),
//!    moderate (1), saturated (2). A busy device's arm sorts after an
//!    idle device's arm **of the same VRAM tier** — "which path drains
//!    the queues fastest right now". A handle with no `BackendStreams`
//!    (CPU / Reference), no handle at all, or a `None`
//!    `pending_work_count` ⇒ tier 0 — an honest no-signal, identical to
//!    a VRAM `Unknown`, NEVER a fabricated "idle".
//!
//!    **VRAM outranks load (critical).** `pressure_tier` is the FIRST key
//!    component, so the load leg reorders arms ONLY within a VRAM fit
//!    tier — it can never lift a `WontFit` (skipped outright) or a
//!    `Tight` (tier 1) arm above a `Comfortable` (tier 0) one to balance
//!    load. Load is a tie-break *under* the OOM-safety guard, not a peer
//!    of it.
//! 3. **Latency (the rank).** Within a (pressure, load) tier, candidates
//!    rank on one uniform nanosecond axis: the Judge's measured latency
//!    at `(ctx.op, ctx.principal_dtype, ctx.size_class, backend,
//!    kernel_source)` when a measurement exists, else
//!    [`composite_ns`] of the candidate's (plan-refined) static
//!    cost. Measured and unmeasured candidates compete on equal
//!    footing — see the 2026-06-08 adversarial finding against
//!    "measured trumps unmeasured". No judge, or no
//!    [`DecisionContext`] on the set, means every candidate uses
//!    its static composite.
//! 4. **Original index.** Ties resolve toward Picker 1's static
//!    rank, preserving determinism.
//!
//! If EVERY candidate is `WontFit`, fall back to `set.winner()` —
//! the realize is in trouble either way; surfacing the static winner
//! gives the executor the cleanest OOM error site instead of
//! violating the trait's "non-empty set MUST produce a pick"
//! contract (mirrors `VramPressureSelector`).
//!
//! # Degenerate-fallback guarantee
//!
//! With no signals at all (no judge data, all fit statuses
//! Unknown/Comfortable, no `BackendStreams` load), every key is
//! `(0, 0, composite_ns(static), i)`. `compile_plan` ranks sets
//! ascending by that same composite before truncation, so the minimum
//! key is index 0 — `set.winner()`, byte-identical to
//! [`super::WinnerSelector`] on every plan-produced set. The added
//! `load_tier` leg is a constant 0 when no device reports load, so it
//! drops out of the comparison entirely: C2 changes nothing unless
//! devices genuinely contend (design §3.2).

use std::sync::Arc;

use fuel_ir::backend::FitStatus;

use super::{
    composite_ns, default_backend_rates, default_estimate_output_bytes, AlternativeSet,
    BackendRuntimeLookup, Candidate, DecisionContext, JudgeOracle, OutputBytesEstimator,
    RuntimeSelector,
};

/// Production runtime selector: VRAM-pressure guard + Judge-measured
/// rank + static-winner fallback. See the module docs for the exact
/// key semantics.
///
/// Both signal sources are optional so the selector degrades
/// honestly:
///
/// - `judge: None` → the rank leg uses static composite costs only.
/// - `backend_runtime_lookup: None` → the guard leg sees every
///   candidate as [`FitStatus::Unknown`] (tier 0, no skips).
/// - Both `None` → exact [`super::WinnerSelector`] behavior on
///   plan-produced (cost-ranked) sets.
#[derive(Clone)]
pub struct ChainedSelector {
    judge: Option<Arc<dyn JudgeOracle>>,
    backend_runtime_lookup: Option<BackendRuntimeLookup>,
    estimate_output_bytes: OutputBytesEstimator,
}

impl ChainedSelector {
    /// Construct with explicit signal sources + output-bytes
    /// estimator. See [`super::default_estimate_output_bytes`] for
    /// the estimator contract (deliberately pessimistic default).
    pub fn new(
        judge: Option<Arc<dyn JudgeOracle>>,
        backend_runtime_lookup: Option<BackendRuntimeLookup>,
        estimate_output_bytes: OutputBytesEstimator,
    ) -> Self {
        Self {
            judge,
            backend_runtime_lookup,
            estimate_output_bytes,
        }
    }

    /// Construct with the conservative
    /// [`default_estimate_output_bytes`] estimator — the shape the
    /// production bridge uses until per-node output shapes are
    /// threaded through.
    pub fn with_default_estimator(
        judge: Option<Arc<dyn JudgeOracle>>,
        backend_runtime_lookup: Option<BackendRuntimeLookup>,
    ) -> Self {
        Self::new(
            judge,
            backend_runtime_lookup,
            Arc::new(default_estimate_output_bytes),
        )
    }

    /// Guard + load legs from ONE lookup: the candidate's VRAM fit status
    /// AND its live load tier, both read off the same runtime handle.
    ///
    /// - **fit status** — `would_fit(estimated_output_bytes)`; `Unknown`
    ///   when no lookup is configured, the lookup has no handle for the
    ///   `(backend, device)` pair, or the backend can't answer.
    /// - **load tier** — the candidate device's coarse live in-flight
    ///   bucket via the handle's Tier-2 `BackendStreams` seam
    ///   ([`super::device_load::load_tier_of_handle`]);
    ///   [`super::device_load::LOAD_TIER_IDLE`] (tier 0 — honest no-signal)
    ///   when there is no lookup / no handle / no `BackendStreams` (CPU,
    ///   Reference) / a `None` `pending_work_count`. NEVER a fabricated
    ///   "idle".
    ///
    /// Reading both off one handle (rather than a second lookup call for
    /// load) keeps the per-candidate cost a single lookup + the SAME
    /// `DeviceRuntimeHandle` the bridge hands out — which already
    /// implements `BackendStreams`, so the VRAM lookup IS the load lookup
    /// (design §3.3: one load source serves all backends).
    fn fit_and_load_for(&self, c: &Candidate) -> (FitStatus, u8) {
        let Some(lookup) = self.backend_runtime_lookup.as_ref() else {
            return (FitStatus::Unknown, super::device_load::LOAD_TIER_IDLE);
        };
        match lookup(c.backend, c.device) {
            Some(runtime) => {
                let fit = runtime.would_fit((self.estimate_output_bytes)(c));
                let load = super::device_load::load_tier_of_handle(runtime.as_ref());
                (fit, load)
            }
            None => (FitStatus::Unknown, super::device_load::LOAD_TIER_IDLE),
        }
    }

    /// Rank leg: one uniform nanosecond figure per candidate —
    /// measured latency when the Judge has the cell, static
    /// composite otherwise — plus the plan-time inbound-transfer
    /// term (Stage 2). Plan-produced sets are device-pruned so the
    /// term is uniform across the set in practice; adding it keeps
    /// the selector's scale consistent with the plan rank.
    fn latency_ns(&self, c: &Candidate, ctx: Option<&DecisionContext>) -> u64 {
        let kernel_ns = (|| {
            if let (Some(judge), Some(ctx)) = (self.judge.as_ref(), ctx) {
                if let Some(measured) = judge.measured_latency_ns(
                    ctx.op,
                    ctx.principal_dtype,
                    ctx.size_class,
                    c.backend,
                    c.kernel_source,
                ) {
                    return measured;
                }
            }
            let (cr, bw) = default_backend_rates(c.backend);
            composite_ns(&c.static_cost, cr, bw)
        })();
        kernel_ns.saturating_add(c.inbound_transfer_ns)
    }
}

impl std::fmt::Debug for ChainedSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainedSelector")
            .field(
                "judge",
                &self.judge.as_ref().map(|_| "<dyn JudgeOracle>"),
            )
            .field(
                "backend_runtime_lookup",
                &self.backend_runtime_lookup.as_ref().map(|_| "<closure>"),
            )
            .field("estimate_output_bytes", &"<closure>")
            .finish()
    }
}

impl RuntimeSelector for ChainedSelector {
    fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
        let alts = set.alternatives();
        if alts.is_empty() {
            return None;
        }
        let ctx = set.context();

        // Walk in original (cost-ranked) order, tracking the minimum
        // (pressure_tier, load_tier, ns, idx) key. WontFit candidates are
        // skipped outright. `load_tier` slots BELOW the VRAM guard and
        // ABOVE the latency rank, so load reorders arms only within a VRAM
        // fit tier (VRAM outranks load) and never overrides the static
        // rank by more than a tie-break under that guard (Step E / C2).
        let mut best: Option<(u8, u8, u64, usize)> = None;
        for (i, c) in alts.iter().enumerate() {
            let (fit, load_tier) = self.fit_and_load_for(c);
            let pressure_tier = match fit {
                FitStatus::WontFit => continue,
                FitStatus::Comfortable | FitStatus::Unknown => 0u8,
                FitStatus::Tight => 1u8,
            };
            let key = (pressure_tier, load_tier, self.latency_ns(c, ctx), i);
            if best.map_or(true, |b| key < b) {
                best = Some(key);
            }
        }

        match best {
            Some((_, _, _, i)) => alts.get(i),
            // Every candidate WontFit → static winner, so the
            // executor surfaces a clean OOM at the planner-blessed
            // pick instead of the selector breaking the non-empty-
            // set contract.
            None => set.winner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use crate::ranker::{HashMapJudge, WinnerSelector};
    use fuel_backend_contract::backend::BackendRuntime;
    use fuel_ir::dispatch::{OpKind, SizeClass};
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DType, DeviceLocation, Layout, Result};
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

    /// `composite_ns` of this candidate equals `cost_ns` (flops
    /// dominate; bytes_moved = cost so memory_ns = cost/4 < cost).
    fn make_candidate(
        backend: BackendId,
        device: DeviceLocation,
        cost_ns: u64,
    ) -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend,
            device,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: cost_ns,
                bytes_moved: cost_ns,
                kernel_overhead_ns: 0,
            },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    fn ctx() -> DecisionContext {
        DecisionContext {
            op: OpKind::MatMul,
            principal_dtype: DType::F32,
            size_class: SizeClass(16),
        }
    }

    /// Cost-ranked two-candidate set (what compile_plan produces):
    /// CUDA wins at 100ns, CPU runner-up at 200ns. Context stamped.
    fn ranked_set() -> AlternativeSet {
        let mut set = AlternativeSet::empty();
        // CUDA raw work is 3000 FLOPs, but at the GPU throughput prior
        // (30 FLOPs/ns) that composites to 100 ns — the value these
        // tests reason about. CPU's 200 FLOPs composites to 200 ns at
        // the 1 FLOP/ns baseline. So the static rank is still CUDA(100)
        // < CPU(200), and the measured-vs-unmeasured thresholds below
        // are relative to CUDA's 100 ns composite.
        set.push(make_candidate(
            BackendId::Cuda,
            DeviceLocation::Cuda { gpu_id: 0 },
            3000,
        ));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 200));
        set.set_context(ctx());
        set
    }

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

    /// Lookup with per-backend (available, total); backends not in
    /// the list resolve to `None` (= Unknown).
    fn lookup_for(
        entries: Vec<(BackendId, Option<u64>, Option<u64>)>,
    ) -> BackendRuntimeLookup {
        Arc::new(move |b, _d| {
            entries.iter().find(|(eb, _, _)| *eb == b).map(|&(_, a, t)| {
                Box::new(MockRuntime { available: a, total: t })
                    as Box<dyn BackendRuntime + Send + Sync>
            })
        })
    }

    // ===== Step E Phase C / C2: the load leg =====

    /// A runtime handle carrying BOTH a VRAM signal (available/total) AND
    /// a Tier-2 `BackendStreams` live-load signal (pending/capacity). The
    /// production `DeviceRuntimeHandle` is exactly this shape — one handle
    /// answering both `would_fit` and `pending_work_count` — so the
    /// ChainedSelector reads load off the same handle it queries for VRAM.
    struct MockRuntimeWithLoad {
        available: Option<u64>,
        total: Option<u64>,
        pending: Option<u32>,
        capacity: u32,
    }
    impl BackendRuntime for MockRuntimeWithLoad {
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
    impl fuel_backend_contract::backend::BackendStreams for MockRuntimeWithLoad {
        fn pending_work_count(&self) -> Option<u32> {
            self.pending
        }
        fn slot_capacity(&self) -> u32 {
            self.capacity
        }
        fn flush(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Per-backend `(available, total, pending, capacity)` lookup handing
    /// out the combined VRAM+load handle; absent backends ⇒ `None`.
    fn load_lookup_for(
        entries: Vec<(BackendId, Option<u64>, Option<u64>, Option<u32>, u32)>,
    ) -> BackendRuntimeLookup {
        Arc::new(move |b, _d| {
            entries
                .iter()
                .find(|(eb, _, _, _, _)| *eb == b)
                .map(|&(_, a, t, p, cap)| {
                    Box::new(MockRuntimeWithLoad {
                        available: a,
                        total: t,
                        pending: p,
                        capacity: cap,
                    }) as Box<dyn BackendRuntime + Send + Sync>
                })
        })
    }

    /// LOAD LEG: two arms, both VRAM-Comfortable, but the cost-winner
    /// (CUDA) is SATURATED while the runner-up (CPU) is idle. The load
    /// tier demotes the saturated arm below the idle one within the shared
    /// VRAM tier, flipping the pick to the unloaded CPU arm. This is the
    /// headline C2 mechanic at the selector level: "which path drains the
    /// queues fastest right now".
    #[test]
    fn load_demotes_saturated_arm_within_vram_tier() {
        let set = ranked_set();
        // CUDA: ample VRAM (Comfortable) but 4 ops in flight on 1 slot
        // (saturated). CPU: ample VRAM, idle (0 in flight).
        let lookup = load_lookup_for(vec![
            (BackendId::Cuda, Some(10_000), Some(10_000), Some(4), 1),
            (BackendId::Cpu, Some(10_000), Some(10_000), Some(0), 1),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cpu,
            "saturated cost-winner demoted below the idle arm by the load leg",
        );
    }

    /// VRAM OUTRANKS LOAD (the critical guard): the cost-winner (CUDA)
    /// WONT FIT, and the runner-up (CPU) is SATURATED. Load would prefer
    /// to stay off the saturated CPU — but VRAM is the FIRST key leg, so
    /// the WontFit CUDA is skipped outright and the picker takes the
    /// saturated-but-fitting CPU arm. Load NEVER lifts a WontFit device to
    /// balance load.
    #[test]
    fn vram_wont_fit_outranks_load() {
        let set = ranked_set();
        // CUDA: 1 byte free of 10_000 → WontFit (skipped). CPU: ample VRAM
        // but saturated (4 in flight / 1 slot).
        let lookup = load_lookup_for(vec![
            (BackendId::Cuda, Some(1), Some(10_000), Some(0), 1),
            (BackendId::Cpu, Some(8_000), Some(10_000), Some(4), 1),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cpu,
            "WontFit CUDA skipped; saturated-but-fitting CPU taken — VRAM \
             outranks load",
        );
    }

    /// VRAM OUTRANKS LOAD, the Tight case: the cost-winner (CUDA) is Tight
    /// (VRAM tier 1) but IDLE; the runner-up (CPU) is Comfortable (tier 0)
    /// but SATURATED. The Tight CUDA's idle load tier (0) does NOT lift it
    /// above the Comfortable CPU — pressure_tier (1 vs 0) is compared
    /// first, so CPU (Comfortable) wins despite being the more loaded
    /// device. Load reorders only WITHIN a fit tier.
    #[test]
    fn tight_idle_still_loses_to_comfortable_saturated() {
        let set = ranked_set();
        // CUDA: available==total==100, alloc 100 → 100% used → Tight; idle.
        // CPU: ample VRAM → Comfortable; saturated.
        let lookup = load_lookup_for(vec![
            (BackendId::Cuda, Some(100), Some(100), Some(0), 1),
            (BackendId::Cpu, Some(10_000), Some(10_000), Some(4), 1),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cpu,
            "Tight+idle CUDA stays demoted below Comfortable+saturated CPU; \
             VRAM tier compared before load tier",
        );
    }

    /// No-load degenerate: both arms VRAM-Comfortable and BOTH idle (0 in
    /// flight) ⇒ every load_tier is 0 ⇒ the key reduces to the pre-C2
    /// `(pressure, latency, idx)` ⇒ the static cost-winner (CUDA) is
    /// preserved. Byte-identical to pre-C2 dispatch.
    #[test]
    fn no_load_preserves_static_winner() {
        let set = ranked_set();
        let lookup = load_lookup_for(vec![
            (BackendId::Cuda, Some(10_000), Some(10_000), Some(0), 1),
            (BackendId::Cpu, Some(10_000), Some(10_000), Some(0), 1),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cuda,
            "no load anywhere ⇒ load leg is constant 0 ⇒ static winner stands",
        );
    }

    /// `pending_work_count() == None` (a streaming backend that can't
    /// report depth) is an honest no-signal = tier 0, NOT a fabricated
    /// load. The cost-winner is preserved even though it "has" a streams
    /// handle — because the handle reports `None`.
    #[test]
    fn none_count_is_no_signal_not_load() {
        let set = ranked_set();
        let lookup = load_lookup_for(vec![
            (BackendId::Cuda, Some(10_000), Some(10_000), None, 1),
            (BackendId::Cpu, Some(10_000), Some(10_000), None, 1),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cuda,
            "None pending_work_count ⇒ tier 0 (no signal) ⇒ static winner",
        );
    }

    /// No judge data + no runtime lookup → the pick equals
    /// WinnerSelector's on a cost-ranked set. The degenerate-
    /// fallback guarantee the production bridge relies on.
    #[test]
    fn no_signals_matches_winner_selector_exactly() {
        let set = ranked_set();
        let chained = ChainedSelector::with_default_estimator(None, None);
        let baseline = WinnerSelector.select(&set).expect("non-empty");
        let pick = chained.select(&set).expect("non-empty");
        assert_eq!(pick.backend, baseline.backend);
        assert_eq!(pick.backend, BackendId::Cuda, "static winner preserved");
    }

    /// Empty judge (no measurements) + Comfortable everywhere is
    /// also a no-signal case — still the static winner.
    #[test]
    fn empty_judge_and_comfortable_pressure_keeps_winner() {
        let set = ranked_set();
        let judge: Arc<dyn JudgeOracle> = Arc::new(HashMapJudge::new());
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(1_000_000), Some(2_000_000)),
            (BackendId::Cpu, Some(1_000_000), Some(2_000_000)),
        ]);
        let chained = ChainedSelector::with_default_estimator(Some(judge), Some(lookup));
        let pick = chained.select(&set).expect("non-empty");
        assert_eq!(pick.backend, BackendId::Cuda);
    }

    /// GUARD: a WontFit winner is skipped; the next admissible
    /// candidate is picked even though it loses on static cost.
    #[test]
    fn guard_skips_wont_fit_winner() {
        let set = ranked_set();
        // CUDA has 50 bytes free; default estimator projects
        // bytes_moved = 100 → WontFit. CPU has plenty.
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(50), Some(10_000)),
            (BackendId::Cpu, Some(8_000), Some(10_000)),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        let pick = chained.select(&set).expect("non-empty");
        assert_eq!(pick.backend, BackendId::Cpu, "WontFit winner skipped");
    }

    /// GUARD: Tight demotes below Comfortable even when the Tight
    /// candidate wins on latency — pressure reorders under
    /// contention.
    #[test]
    fn tight_demoted_below_comfortable() {
        let set = ranked_set();
        // CUDA: available == total == 100, alloc 100 → 100% used →
        // Tight (fits, but over the 0.85 threshold). CPU: ample.
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(100), Some(100)),
            (BackendId::Cpu, Some(10_000), Some(10_000)),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        let pick = chained.select(&set).expect("non-empty");
        assert_eq!(pick.backend, BackendId::Cpu, "Tight loses to Comfortable");
    }

    /// GUARD: Unknown ties with Comfortable — no handle for the
    /// winner's backend must NOT demote it.
    #[test]
    fn unknown_ties_with_comfortable_preserves_winner() {
        let set = ranked_set();
        // Only CPU has a handle (Comfortable); CUDA is Unknown.
        let lookup = lookup_for(vec![(BackendId::Cpu, Some(10_000), Some(10_000))]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        let pick = chained.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "Unknown is no-signal, not pressure",
        );
    }

    /// GUARD: every candidate WontFit → fall back to the static
    /// winner (clean OOM site, non-empty-set contract upheld).
    #[test]
    fn all_wont_fit_falls_back_to_winner() {
        let set = ranked_set();
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(1), Some(10_000)),
            (BackendId::Cpu, Some(1), Some(10_000)),
        ]);
        let chained = ChainedSelector::with_default_estimator(None, Some(lookup));
        let pick = chained.select(&set).expect("non-empty per contract");
        assert_eq!(pick.backend, BackendId::Cuda);
    }

    /// RANK: a Judge measurement on the runner-up flips the pick
    /// when it beats the winner's figure; a slow measurement does
    /// not.
    #[test]
    fn judge_measurement_flips_winner_both_directions() {
        let c = ctx();

        // CPU measured at 50ns < CUDA's unmeasured composite 100ns
        // → CPU wins.
        let mut judge = HashMapJudge::new();
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cpu, "", 50);
        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(judge) as Arc<dyn JudgeOracle>),
            None,
        );
        let set = ranked_set();
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cpu,
            "fast measurement beats slower unmeasured composite",
        );

        // CPU measured at 5000ns > CUDA's 100ns → CUDA stays.
        let mut judge = HashMapJudge::new();
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cpu, "", 5_000);
        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(judge) as Arc<dyn JudgeOracle>),
            None,
        );
        let set = ranked_set();
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cuda,
            "slow measurement does not trump a faster unmeasured candidate",
        );
    }

    /// RANK respects the guard: judge prefers a WontFit candidate,
    /// but the guard's skip wins — measured speed never overrides
    /// "the bytes don't fit".
    #[test]
    fn guard_overrides_judge_preference() {
        let c = ctx();
        let mut judge = HashMapJudge::new();
        // CUDA measured blazing fast...
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cuda, "", 1);
        // ...but CUDA can't fit the output.
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(50), Some(10_000)),
            (BackendId::Cpu, Some(8_000), Some(10_000)),
        ]);
        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(judge) as Arc<dyn JudgeOracle>),
            Some(lookup),
        );
        let set = ranked_set();
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cpu,
        );
    }

    /// Judge ranks WITHIN a pressure tier: two Comfortable
    /// candidates re-order by measurement while a Tight one stays
    /// demoted regardless of its measurement.
    #[test]
    fn judge_ranks_within_pressure_tier() {
        let c = ctx();
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(
            BackendId::Cuda,
            DeviceLocation::Cuda { gpu_id: 0 },
            100,
        ));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 200));
        set.push(make_candidate(
            BackendId::Vulkan,
            DeviceLocation::Vulkan { gpu_id: 0 },
            300,
        ));
        set.set_context(c);

        // CUDA is Tight; CPU + Vulkan Comfortable.
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(100), Some(100)),
            (BackendId::Cpu, Some(10_000), Some(10_000)),
            (BackendId::Vulkan, Some(10_000), Some(10_000)),
        ]);
        // Judge: CUDA fastest overall (doesn't matter — Tight),
        // Vulkan beats CPU within tier 0.
        let mut judge = HashMapJudge::new();
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cuda, "", 1);
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cpu, "", 90);
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Vulkan, "", 40);

        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(judge) as Arc<dyn JudgeOracle>),
            Some(lookup),
        );
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Vulkan,
            "judge re-ranks the Comfortable tier; Tight stays demoted",
        );
    }

    /// Sibling kernels at the same backend rank independently via
    /// kernel_source — the end-to-end reason kernel_source is on
    /// both Candidate and the Judge key.
    #[test]
    fn kernel_source_disambiguates_cpu_siblings() {
        let c = ctx();
        let mut set = AlternativeSet::empty();
        let mut aocl = make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 500);
        aocl.kernel_source = "aocl";
        let mut mkl = make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 500);
        mkl.kernel_source = "mkl";
        set.push(aocl);
        set.push(mkl);
        set.set_context(c);

        let mut judge = HashMapJudge::new();
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cpu, "aocl", 9_000);
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cpu, "mkl", 1_000);
        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(judge) as Arc<dyn JudgeOracle>),
            None,
        );
        assert_eq!(
            chained.select(&set).expect("non-empty").kernel_source,
            "mkl",
        );
    }

    /// No DecisionContext on the set → the judge leg is inert even
    /// when measurements exist; static order stands.
    #[test]
    fn missing_context_disables_judge_leg() {
        let c = ctx();
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(
            BackendId::Cuda,
            DeviceLocation::Cuda { gpu_id: 0 },
            100,
        ));
        set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 200));
        // NOTE: no set_context.

        let mut judge = HashMapJudge::new();
        judge.insert(c.op, c.principal_dtype, c.size_class, BackendId::Cpu, "", 1);
        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(judge) as Arc<dyn JudgeOracle>),
            None,
        );
        assert_eq!(
            chained.select(&set).expect("non-empty").backend,
            BackendId::Cuda,
            "no context → no judge key → static winner",
        );
    }

    /// Empty set → None per the trait contract.
    #[test]
    fn empty_set_returns_none() {
        let set = AlternativeSet::empty();
        let chained = ChainedSelector::with_default_estimator(None, None);
        assert!(chained.select(&set).is_none());
    }

    /// Debug impl doesn't panic on the closure / dyn fields.
    #[test]
    fn debug_does_not_panic() {
        let chained = ChainedSelector::with_default_estimator(
            Some(Arc::new(HashMapJudge::new()) as Arc<dyn JudgeOracle>),
            Some(Arc::new(|_, _| None)),
        );
        let s = format!("{chained:?}");
        assert!(s.contains("ChainedSelector"));
    }
}
