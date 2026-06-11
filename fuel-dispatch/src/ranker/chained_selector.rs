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
//! `(pressure_tier, latency_ns, original_index)` and pick the
//! minimum:
//!
//! 1. **Pressure tier (the guard).** Query the candidate backend's
//!    [`fuel_core_types::backend::BackendRuntime::would_fit`] for
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
//! 2. **Latency (the rank).** Within a tier, candidates rank on one
//!    uniform nanosecond axis: the Judge's measured latency at
//!    `(ctx.op, ctx.principal_dtype, ctx.size_class, backend,
//!    kernel_source)` when a measurement exists, else
//!    [`composite_ns`] of the candidate's (plan-refined) static
//!    cost. Measured and unmeasured candidates compete on equal
//!    footing — see the 2026-06-08 adversarial finding against
//!    "measured trumps unmeasured". No judge, or no
//!    [`DecisionContext`] on the set, means every candidate uses
//!    its static composite.
//! 3. **Original index.** Ties resolve toward Picker 1's static
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
//! Unknown/Comfortable), every key is `(0, composite_ns(static), i)`.
//! `compile_plan` ranks sets ascending by that same composite before
//! truncation, so the minimum key is index 0 — `set.winner()`,
//! byte-identical to [`super::WinnerSelector`] on every
//! plan-produced set.

use std::sync::Arc;

use fuel_core_types::backend::FitStatus;

use super::{
    composite_ns, default_estimate_output_bytes, AlternativeSet, BackendRuntimeLookup,
    Candidate, DecisionContext, JudgeOracle, OutputBytesEstimator, RuntimeSelector,
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

    /// Guard leg: the candidate's fit status. `Unknown` when no
    /// lookup is configured, the lookup has no handle for the
    /// `(backend, device)` pair, or the backend itself can't answer.
    fn fit_status_for(&self, c: &Candidate) -> FitStatus {
        let Some(lookup) = self.backend_runtime_lookup.as_ref() else {
            return FitStatus::Unknown;
        };
        match lookup(c.backend, c.device) {
            Some(runtime) => runtime.would_fit((self.estimate_output_bytes)(c)),
            None => FitStatus::Unknown,
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
            composite_ns(&c.static_cost)
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
        // (tier, ns, idx) key. WontFit candidates are skipped
        // outright.
        let mut best: Option<(u8, u64, usize)> = None;
        for (i, c) in alts.iter().enumerate() {
            let tier = match self.fit_status_for(c) {
                FitStatus::WontFit => continue,
                FitStatus::Comfortable | FitStatus::Unknown => 0u8,
                FitStatus::Tight => 1u8,
            };
            let key = (tier, self.latency_ns(c, ctx), i);
            if best.map_or(true, |b| key < b) {
                best = Some(key);
            }
        }

        match best {
            Some((_, _, i)) => alts.get(i),
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
    use fuel_core_types::backend::BackendRuntime;
    use fuel_core_types::dispatch::{OpKind, SizeClass};
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result};
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
        set.push(make_candidate(
            BackendId::Cuda,
            DeviceLocation::Cuda { gpu_id: 0 },
            100,
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
