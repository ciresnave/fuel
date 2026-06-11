//! `JudgeAwareSelector` — Phase 5.2 runtime selector that re-queries
//! the [`JudgeOracle`] at dispatch time and re-ranks candidates by
//! fresh empirical measurements.
//!
//! # Why this selector exists
//!
//! Picker 1 (the optimizer ranker) consumes Judge data once during
//! `compute_static_costs`, at plan-build time. If new measurements
//! land between plan build and dispatch (the Judge profiles
//! incrementally; a long-running process accumulates entries during
//! its first realizes), the plan's ranking can be stale. The static
//! cost was Layer-1 + whatever Judge entries existed at plan time;
//! a Layer-2 entry that materializes after the plan freezes is not
//! reflected.
//!
//! `JudgeAwareSelector` fixes this by re-querying the Judge at
//! select time for each candidate's `(op, dtype, size_class, backend)`
//! cell. Candidates with a fresh measurement get re-ranked by that
//! measurement; candidates without a measurement keep their static
//! rank position. The selector returns the new top-1.
//!
//! # Selector context
//!
//! The Judge lookup key is
//! `(OpKind, DType, SizeClass, BackendId)`. Only `BackendId` is on
//! the [`Candidate`]; the other three components are per-decision-
//! point context that the trait's `select(&AlternativeSet)` signature
//! deliberately does NOT carry. Phase 5.2 resolves this the same way
//! Phase 5.1 documented as the path forward: the selector is
//! constructed PER DECISION POINT with the op/dtype/size context
//! baked in. The caller (Phase 4's executor migration, when it lands)
//! produces one of these per kernel-bearing node from the plan's
//! enumeration metadata.
//!
//! This shape is deliberately not the future "shared selector for
//! whole realize" — that would require widening the trait. It's the
//! pragmatic Phase 5.2 shape that lets the selector exist before
//! the wider trait change has consensus.
//!
//! # What "kernel_source" does here
//!
//! Phase v0.4 of the backend contract carries `kernel_source` on
//! both [`Candidate`] and [`fuel_core_types::dispatch::ProfileEntry`].
//! The [`JudgeOracle`] trait's `measured_latency_ns` doesn't take
//! `kernel_source` today (the trait predates per-alternative
//! measurement). When a future Judge revision adds it, this selector
//! is the first consumer that will benefit — multiple candidates
//! with the same `(op, dtype, size, backend)` but different
//! `kernel_source` will get distinguished. For now the selector
//! re-queries by `(op, dtype, size_class, backend)` and accepts
//! that AOCL vs MKL candidates on `BackendId::Cpu` share a Judge
//! cell.

use std::sync::Arc;

use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::DType;

use super::{composite_ns, AlternativeSet, Candidate, JudgeOracle, RuntimeSelector};

/// Phase 5.2 runtime selector that re-queries the Judge at dispatch
/// time and re-ranks candidates by the freshest measured latency.
///
/// See module docs for the full rationale.
///
/// # Construction
///
/// ```ignore
/// use std::sync::Arc;
/// use fuel_core_types::dispatch::{OpKind, SizeClass};
/// use fuel_core_types::DType;
/// use fuel_dispatch::ranker::{HashMapJudge, JudgeAwareSelector};
///
/// let judge = Arc::new(HashMapJudge::new());
/// let selector = JudgeAwareSelector::new(
///     judge,
///     OpKind::MatMul,
///     DType::F32,
///     SizeClass::from_elem_count(1024 * 1024),
/// );
/// ```
#[derive(Clone)]
pub struct JudgeAwareSelector {
    judge: Arc<dyn JudgeOracle>,
    op: OpKind,
    dtype: DType,
    size_class: SizeClass,
}

impl JudgeAwareSelector {
    /// Construct a selector bound to one `(op, dtype, size_class)`
    /// decision point. The selector re-queries `judge` at each
    /// `select` call for each candidate's backend.
    pub fn new(
        judge: Arc<dyn JudgeOracle>,
        op: OpKind,
        dtype: DType,
        size_class: SizeClass,
    ) -> Self {
        Self {
            judge,
            op,
            dtype,
            size_class,
        }
    }

    /// The op this selector queries the Judge under.
    pub fn op(&self) -> OpKind {
        self.op
    }

    /// The dtype this selector queries the Judge under.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The size class this selector queries the Judge under.
    pub fn size_class(&self) -> SizeClass {
        self.size_class
    }

    /// Look up the candidate's measured latency, if any.
    ///
    /// Threads the candidate's `kernel_source` through the Judge so
    /// sibling kernels at the same `(backend, device)` slot (e.g.
    /// AOCL vs MKL vs portable-cpu under `BackendId::Cpu`) get their
    /// own measurement, instead of collapsing onto whichever
    /// registered last.
    fn measured_latency(&self, c: &Candidate) -> Option<u64> {
        self.judge.measured_latency_ns(
            self.op,
            self.dtype,
            self.size_class,
            c.backend,
            c.kernel_source,
        )
    }
}

impl std::fmt::Debug for JudgeAwareSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JudgeAwareSelector")
            .field("judge", &"<dyn JudgeOracle>")
            .field("op", &self.op)
            .field("dtype", &self.dtype)
            .field("size_class", &self.size_class)
            .finish()
    }
}

impl RuntimeSelector for JudgeAwareSelector {
    fn select<'a>(&self, set: &'a AlternativeSet) -> Option<&'a Candidate> {
        let alts = set.alternatives();
        if alts.is_empty() {
            return None;
        }

        // Rank every candidate by a single nanosecond figure:
        //
        // - Measured candidates use the Judge's measured latency.
        // - Unmeasured candidates use `composite_ns(static_cost)`,
        //   the same Layer-1 scalar Picker 1 already ranks by.
        //
        // This produces a uniform ordering across both groups: a
        // 100ms measured kernel correctly loses to a 50ns unmeasured
        // one when the Layer-1 estimate is trustworthy. The prior
        // shape ("any measured candidate trumps all unmeasured
        // candidates") was wrong — see the 2026-06-08 adversarial
        // verification of the Phase 5.2 selector. Treating the two
        // groups on equal footing is the principled fix.
        //
        // Tie-break by original index so ties resolve toward
        // Picker 1's static rank — preserving determinism and the
        // `set.winner()` invariant when every candidate scores the
        // same.
        let (best_idx, _) = alts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                // Stage 2: the inbound-transfer term adds serially
                // to both groups — the bytes must land before either
                // a measured or an estimated kernel can run. Plan-
                // produced sets are device-pruned so the term is
                // uniform in practice; adding it keeps the scale
                // consistent with the plan rank.
                let score = self
                    .measured_latency(c)
                    .unwrap_or_else(|| composite_ns(&c.static_cost))
                    .saturating_add(c.inbound_transfer_ns);
                (i, score)
            })
            .min_by_key(|&(idx, score)| (score, idx))
            .expect("non-empty after the early return");
        alts.get(best_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use crate::ranker::HashMapJudge;
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

    fn make_candidate(backend: BackendId, cost: u64) -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: cost,
                bytes_moved: cost,
                kernel_overhead_ns: 0,
            },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    /// Helper: build a selector against a freshly-populated judge.
    fn make_selector(judge: HashMapJudge) -> JudgeAwareSelector {
        JudgeAwareSelector::new(
            Arc::new(judge),
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
        )
    }

    /// No measurements → behaves like WinnerSelector.
    #[test]
    fn empty_judge_falls_back_to_winner() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));

        let sel = make_selector(HashMapJudge::new());
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "no measurements → static winner",
        );
    }

    /// Judge measurement on the loser → re-rank flips order.
    #[test]
    fn judge_measurement_flips_picker_order() {
        let mut set = AlternativeSet::empty();
        // Cost rank: CUDA (100) wins. CPU (200) loses.
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));

        // Judge says CUDA is actually 10ms while CPU is 1ms.
        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cuda,
            "",
            10_000_000,
        );
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "",
            1_000_000,
        );

        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cpu,
            "Judge says CPU is 10x faster → CPU wins despite static cost",
        );
    }

    /// Only one candidate measured → measured and unmeasured rank
    /// uniformly on the same nanosecond axis. A fast (low static
    /// cost) unmeasured candidate beats a slow measured one; a
    /// fast measurement beats a high static cost. The selector no
    /// longer auto-promotes every measured candidate above every
    /// unmeasured one.
    #[test]
    fn single_measurement_competes_uniformly_against_unmeasured() {
        // Case A: the measurement is slow → unmeasured static
        // winner should still win.
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));
        set.push(make_candidate(BackendId::Vulkan, 300));

        // CPU measured at 500_000 ns — slow compared to the
        // unmeasured CUDA (composite_ns = 100 ns).
        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "",
            500_000,
        );
        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "slow measurement does NOT trump a faster unmeasured static cost",
        );

        // Case B: the measurement is fast → the measured candidate
        // wins.
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 1_000_000));
        set.push(make_candidate(BackendId::Cpu, 2_000_000));
        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "",
            500,
        );
        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cpu,
            "fast measurement beats slower unmeasured static cost",
        );
    }

    /// All candidates measured → lowest latency wins.
    #[test]
    fn all_measured_picks_lowest_latency() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));
        set.push(make_candidate(BackendId::Vulkan, 300));

        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cuda,
            "",
            10_000,
        );
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "",
            5_000,
        );
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Vulkan,
            "",
            1_000,
        );

        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Vulkan,
            "lowest latency wins regardless of static cost order",
        );
    }

    /// Different op key in the Judge → no match → falls back to
    /// winner. Proves the selector queries by its constructed
    /// (op, dtype, size_class), not by candidate.
    #[test]
    fn judge_key_must_match_selector_context() {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));

        // Judge has entries — but under a DIFFERENT op.
        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::AddElementwise, // selector is configured for MatMul
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "",
            1_000,
        );

        // Selector configured for MatMul.
        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "Judge has no entry at our (op,dtype,size) → static winner",
        );
    }

    /// Tie in latency → earlier-original-index wins (stable).
    #[test]
    fn latency_tie_breaks_toward_static_winner() {
        let mut set = AlternativeSet::empty();
        // CUDA at original idx 0; CPU at idx 1; both measured at
        // the same latency.
        set.push(make_candidate(BackendId::Cuda, 100));
        set.push(make_candidate(BackendId::Cpu, 200));

        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::MatMul, DType::F32, SizeClass(16), BackendId::Cuda, "", 5_000);
        judge.insert(OpKind::MatMul, DType::F32, SizeClass(16), BackendId::Cpu, "", 5_000);

        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.backend,
            BackendId::Cuda,
            "tie on latency → earlier original index wins (static-rank tiebreak)",
        );
    }

    /// Empty set → None.
    #[test]
    fn empty_set_returns_none() {
        let set = AlternativeSet::empty();
        let sel = make_selector(HashMapJudge::new());
        assert!(sel.select(&set).is_none());
    }

    /// Debug impl doesn't panic on the Arc<dyn> field.
    #[test]
    fn debug_does_not_panic() {
        let sel = make_selector(HashMapJudge::new());
        let s = format!("{sel:?}");
        assert!(s.contains("JudgeAwareSelector"));
        assert!(s.contains("MatMul"));
    }

    /// Accessor methods round-trip the construction args.
    #[test]
    fn accessors_round_trip_construction() {
        let sel = JudgeAwareSelector::new(
            Arc::new(HashMapJudge::new()),
            OpKind::AddElementwise,
            DType::BF16,
            SizeClass(7),
        );
        assert_eq!(sel.op(), OpKind::AddElementwise);
        assert_eq!(sel.dtype(), DType::BF16);
        assert_eq!(sel.size_class(), SizeClass(7));
    }

    /// Two `BackendId::Cpu` candidates with distinct `kernel_source`
    /// tags receive DISTINCT measurements from the Judge. The trait
    /// fix that threads `kernel_source` into `measured_latency_ns`
    /// only matters if the selector propagates it correctly — this
    /// test pins that behavior.
    ///
    /// Without the fix, the second `judge.insert(..., "aocl", _)` and
    /// `judge.insert(..., "mkl", _)` would collide on the same key and
    /// last-write-wins would collapse the ranking. With the fix, each
    /// sibling kernel is judged on its own merits and the selector
    /// picks the cheaper of the two.
    #[test]
    fn distinct_kernel_sources_get_distinct_measurements() {
        let mut set = AlternativeSet::empty();
        // Two Cpu candidates: aocl and mkl. Identical backend, identical
        // device, identical static cost — only `kernel_source` differs.
        let mut c_aocl = make_candidate(BackendId::Cpu, 500);
        c_aocl.kernel_source = "aocl";
        let mut c_mkl = make_candidate(BackendId::Cpu, 500);
        c_mkl.kernel_source = "mkl";
        set.push(c_aocl);
        set.push(c_mkl);

        // Judge: aocl is slow (10ms), mkl is fast (1ms).
        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "aocl",
            10_000_000,
        );
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "mkl",
            1_000_000,
        );

        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(pick.backend, BackendId::Cpu);
        assert_eq!(
            pick.kernel_source, "mkl",
            "selector threads kernel_source into the Judge so siblings rank independently",
        );

        // Flip the measurements and confirm the OTHER sibling wins.
        let mut set = AlternativeSet::empty();
        let mut c_aocl = make_candidate(BackendId::Cpu, 500);
        c_aocl.kernel_source = "aocl";
        let mut c_mkl = make_candidate(BackendId::Cpu, 500);
        c_mkl.kernel_source = "mkl";
        set.push(c_aocl);
        set.push(c_mkl);

        let mut judge = HashMapJudge::new();
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "aocl",
            500_000,
        );
        judge.insert(
            OpKind::MatMul,
            DType::F32,
            SizeClass(16),
            BackendId::Cpu,
            "mkl",
            5_000_000,
        );
        let sel = make_selector(judge);
        let pick = sel.select(&set).expect("non-empty");
        assert_eq!(
            pick.kernel_source, "aocl",
            "flipped measurements → other sibling wins, proving siblings are not collapsed",
        );
    }
}
