//! `AlternativeSet` — bounded collection of [`Candidate`]s at one
//! graph decision point.
//!
//! Phase 1.1 of the picker-work arc. The set is what
//! [`apply_filter_chain`] mutates and what (in later phases)
//! `rank_by_cost` orders and `truncate_to_top_n` bounds. The newtype
//! exists so we have somewhere to hang the per-decision-point
//! invariants — top-N preservation, filter application, eventual
//! coupling resolution — without exposing a bare `Vec`.
//!
//! [`apply_filter_chain`]: super::apply_filter_chain

use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::DType;
use smallvec::SmallVec;

use super::candidate::Candidate;

/// Plan-time identity of the decision point an [`AlternativeSet`]
/// belongs to — the `(op, principal dtype, size class)` triple the
/// Judge keys its measurements on.
///
/// Stamped by `compile_plan` so dispatch-time selectors (Picker 2)
/// can re-query the [`super::JudgeOracle`] per candidate without
/// being constructed per node. The derivation matches
/// [`super::cost::compute_static_costs`]'s Layer-2 lookup exactly:
///
/// - `principal_dtype` — the first entry of the node's lookup
///   dtypes (first input's dtype by `build_lookup_dtypes`
///   convention).
/// - `size_class` — [`SizeClass::from_elem_count`] of the FIRST
///   input's shape (`SizeClass(0)` for nullary ops), matching the
///   Judge profiler's bucketing.
///
/// A set without a context (`AlternativeSet::context() == None`)
/// simply disables judge re-querying in context-aware selectors —
/// they fall back to the candidates' static costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionContext {
    /// Dispatch OpKind of the decision point.
    pub op: OpKind,
    /// Principal dtype (first lookup dtype) — the Judge's dtype axis.
    pub principal_dtype: DType,
    /// Size bucket of the first input — the Judge's size axis.
    pub size_class: SizeClass,
}

/// Default top-N preservation per architecture v1.0 §04
/// ("Default N=3 per decision point gives the runtime picker
/// meaningful flexibility ... without exploding storage or search
/// cost"). Configurable per-`AlternativeSet` via
/// [`AlternativeSet::with_max_n`].
pub const DEFAULT_MAX_N: usize = 3;

/// Bounded collection of [`Candidate`]s at one decision point. The
/// optimizer ranker constructs one of these per kernel-bearing graph
/// node, runs the filter chain, ranks survivors by composite cost,
/// and truncates to `max_n` for storage on the optimized graph.
///
/// Stored as a `SmallVec` because:
///
/// - Inline capacity 4 covers every decision point that exists today
///   (CPU + CUDA + Vulkan + one cuBLAS/CUTLASS sibling once that
///   registers).
/// - Spilling to heap only happens when a decision point genuinely
///   has more than 4 alternatives — rare, and the cost is amortized
///   over a one-time enumeration anyway.
#[derive(Clone, Debug)]
pub struct AlternativeSet {
    candidates: SmallVec<[Candidate; 4]>,
    max_n: usize,
    /// Decision-point identity for dispatch-time Judge re-queries.
    /// `None` until `compile_plan` stamps it (or for hand-built
    /// test sets) — context-aware selectors then skip the Judge leg.
    context: Option<DecisionContext>,
}

impl AlternativeSet {
    /// Build an empty set with [`DEFAULT_MAX_N`] (`= 3`). Tests + the
    /// candidate enumerator both use this; consumers needing a
    /// different `max_n` go through [`Self::with_max_n`].
    pub fn empty() -> Self {
        Self {
            candidates: SmallVec::new(),
            max_n: DEFAULT_MAX_N,
            context: None,
        }
    }

    /// Empty set with an explicit `max_n`. Useful for tests + future
    /// callers that want a per-node N (e.g. memory-constrained
    /// realizes that want top-1 only).
    pub fn with_max_n(max_n: usize) -> Self {
        Self {
            candidates: SmallVec::new(),
            max_n,
            context: None,
        }
    }

    /// Build a set from a pre-collected list of candidates. The
    /// enumerator path; truncation to `max_n` happens after the
    /// filter chain + cost rank, not here.
    pub fn from_candidates(candidates: Vec<Candidate>, max_n: usize) -> Self {
        Self {
            candidates: SmallVec::from_vec(candidates),
            max_n,
            context: None,
        }
    }

    /// Stamp the decision-point identity. `compile_plan` calls this
    /// once per kernel-bearing node; the context survives filtering,
    /// ranking, and truncation untouched (it describes the decision
    /// point, not the candidates).
    pub fn set_context(&mut self, ctx: DecisionContext) {
        self.context = Some(ctx);
    }

    /// The decision-point identity, if stamped. Dispatch-time
    /// selectors use this to key [`super::JudgeOracle`] lookups.
    pub fn context(&self) -> Option<&DecisionContext> {
        self.context.as_ref()
    }

    /// Append one candidate. Used by the enumerator as it walks
    /// `(backend, device)` combinations.
    pub fn push(&mut self, c: Candidate) {
        self.candidates.push(c);
    }

    /// How many alternatives remain. Drives the filter chain's
    /// hard-vs-soft decision and (in later phases) the executor's
    /// "we have N to choose from" telemetry.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Is the set empty? (i.e., no admissible alternative).
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// The configured top-N bound.
    pub fn max_n(&self) -> usize {
        self.max_n
    }

    /// Borrow the full candidate list. Read-only — the filter chain
    /// uses this to compute the keep-mask without mutating.
    pub fn alternatives(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Retain only the entries at `indices` (must be sorted ascending,
    /// distinct, and in-range). The filter-chain pipeline produces a
    /// `Vec<usize>` from each filter and calls this to apply it.
    ///
    /// Panics in debug builds if `indices` is mis-shaped (out-of-range
    /// or unsorted). Release builds skip the check — the producer is
    /// always internal to the ranker.
    pub fn retain_indices(&mut self, indices: &[usize]) {
        debug_assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "retain_indices: input must be sorted strictly ascending; got {indices:?}",
        );
        debug_assert!(
            indices.iter().all(|&i| i < self.candidates.len()),
            "retain_indices: out-of-range index in {indices:?} (len={})",
            self.candidates.len(),
        );
        // Build a new vec in-place via the standard retain pattern.
        // SmallVec doesn't have retain_indices, so we mark-and-sweep.
        let mut iter = indices.iter().copied().peekable();
        let mut idx = 0;
        self.candidates.retain(|_| {
            let keep = matches!(iter.peek(), Some(&i) if i == idx);
            if keep {
                iter.next();
            }
            idx += 1;
            keep
        });
    }

    /// Truncate to top-`max_n`. Caller is expected to have called
    /// [`Self::rank_by_composite_cost`] first; this is a pure
    /// suffix-drop.
    pub fn truncate_to_top_n(&mut self) {
        self.candidates.truncate(self.max_n);
    }

    /// Rank candidates by the per-path **cost vector**
    /// ([`super::cost_vector::CostVector`]) — Phase B PR-B1.
    ///
    /// The vector's axes are one central `time` metric, per-tier
    /// memory, and discrete precision/accuracy (see the cost_vector
    /// module docs). Ranking uses
    /// [`CostVector::total_order_key`], which keeps the **winner
    /// time-first**: the lowest-central-`time` candidate sorts to
    /// index 0 (arm-0), exactly preserving the old `composite_ns`
    /// winner, with ties broken **precision → accuracy → memory**
    /// (the constitution's order).
    ///
    /// Why time-first for the winner while [`CostVector::dominates`]
    /// exists: realize follows arm-0, so the *winner* must stay the
    /// lowest-time candidate to preserve realize behavior. Pareto
    /// dominance is the relation the *frontier retention* will use —
    /// that's PR-B2, which retires the top-N truncation for a
    /// per-device Pareto frontier + crowding cap. B1 keeps
    /// [`DEFAULT_MAX_N`] / [`Self::truncate_to_top_n`] untouched.
    ///
    /// Stable sort — equal-key candidates keep their relative order,
    /// which matters when registration order is the residual
    /// tie-breaker (and, post-Stage-2, when decision-device
    /// candidates are enumerated ahead of off-device ones).
    ///
    /// [`CostVector::total_order_key`]: super::cost_vector::CostVector::total_order_key
    /// [`CostVector::dominates`]: super::cost_vector::CostVector::dominates
    pub fn rank_by_cost(&mut self) {
        use super::cost_vector::CostVector;
        self.candidates
            .sort_by_key(|c| CostVector::from_candidate(c).total_order_key());
    }

    /// Sort candidates ascending by their composite static cost
    /// (Layer-1 score; see [`super::cost::composite_ns`]) plus the
    /// Stage-2 inbound-transfer term
    /// ([`Candidate::inbound_transfer_ns`]).
    ///
    /// As of Phase B PR-B1 this delegates to [`Self::rank_by_cost`]:
    /// the cost VECTOR is now the ranking primitive, and its
    /// `total_order_key` is time-first, so for a single-device,
    /// single-precision candidate set (the common case + the entire
    /// CPU `--lib` suite) the winner is unchanged — lowest central
    /// `time` is exactly the old `composite_ns` winner. Retained as a
    /// named alias so existing callers (`compile_plan`,
    /// `compile_run_view`) and tests don't churn.
    ///
    /// Phase 1.4 of the picker-work arc. Composite cost is
    /// `max(compute_ns, memory_ns) + overhead_ns`, treating
    /// compute and memory as parallel (roofline model). Layer-2
    /// Judge data refines `static_cost` before this call
    /// (`compute_static_costs`); the transfer term is added
    /// serially — the bytes must land before the kernel can run.
    pub fn rank_by_composite_cost(&mut self) {
        self.rank_by_cost();
    }

    /// Set the `static_cost` field of the candidate at `index`.
    /// Used by [`super::cost::compute_static_costs`] to populate
    /// the field after enumeration. Panics in debug builds if
    /// `index >= len`.
    pub fn set_static_cost(&mut self, index: usize, cost: crate::fused::CostEstimate) {
        debug_assert!(
            index < self.candidates.len(),
            "set_static_cost: index {index} out of range (len={})",
            self.candidates.len(),
        );
        self.candidates[index].static_cost = cost;
    }

    /// Set the Stage-2 inbound-transfer term of the candidate at
    /// `index`. Used by
    /// [`super::cost::apply_inbound_transfer_costs`]. Panics in
    /// debug builds if `index >= len`.
    pub fn set_inbound_transfer_ns(&mut self, index: usize, ns: u64) {
        debug_assert!(
            index < self.candidates.len(),
            "set_inbound_transfer_ns: index {index} out of range (len={})",
            self.candidates.len(),
        );
        self.candidates[index].inbound_transfer_ns = ns;
    }

    /// The current top candidate (first entry). After the full
    /// pipeline — filter → rank → truncate — this is the runtime
    /// selector's (Picker 2) default pick. `None` if the set is
    /// empty.
    pub fn winner(&self) -> Option<&Candidate> {
        self.candidates.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DeviceLocation, Layout, Result};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn dummy_candidate(flops: u64) -> Candidate {
        Candidate {
            kernel: noop,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate { flops, bytes_moved: 0, kernel_overhead_ns: 0 },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    /// Phase B PR-B1 behavior preservation: `rank_by_cost` on a
    /// single-precision, multi-candidate set yields the SAME winner
    /// (and full order) as ranking by the old `composite_ns +
    /// inbound_transfer` scalar. This is the safety contract — the
    /// CPU `--lib` suite is exactly this case.
    #[test]
    fn rank_by_cost_preserves_single_precision_winner() {
        use super::super::cost::composite_ns;

        let mk = |flops: u64, bytes: u64, overhead: u32, inbound: u64| Candidate {
            inbound_transfer_ns: inbound,
            static_cost: CostEstimate {
                flops,
                bytes_moved: bytes,
                kernel_overhead_ns: overhead,
            },
            ..dummy_candidate(0)
        };
        let cands = vec![
            mk(300, 0, 0, 0),
            mk(100, 0, 0, 50),   // 150 total
            mk(200, 800, 0, 0),  // max(200,200)=200
            mk(0, 4000, 10, 90), // 1000+10+90=1100
            mk(120, 0, 0, 0),
        ];

        // Expected order from the OLD scalar key.
        let mut expected = cands.clone();
        expected.sort_by_key(|c| {
            composite_ns(&c.static_cost).saturating_add(c.inbound_transfer_ns)
        });
        let expected_flops: Vec<u64> =
            expected.iter().map(|c| c.static_cost.flops).collect();

        // Order from the NEW cost-vector rank.
        let mut set = AlternativeSet::from_candidates(cands, DEFAULT_MAX_N);
        set.rank_by_cost();
        let got_flops: Vec<u64> =
            set.alternatives().iter().map(|c| c.static_cost.flops).collect();

        assert_eq!(
            got_flops, expected_flops,
            "single-precision cost-vector rank matches the old composite scalar order",
        );
    }

    /// Rank composes the inbound-transfer term serially with the
    /// composite kernel cost.
    #[test]
    fn rank_includes_inbound_transfer_term() {
        let mut s = AlternativeSet::from_candidates(
            vec![
                Candidate { inbound_transfer_ns: 5_000, ..dummy_candidate(100) },
                dummy_candidate(200),
            ],
            DEFAULT_MAX_N,
        );
        s.rank_by_composite_cost();
        assert_eq!(
            s.winner().unwrap().static_cost.flops,
            200,
            "200 ns total beats 100 ns kernel + 5 µs transfer",
        );
    }

    #[test]
    fn empty_has_no_winner() {
        let s = AlternativeSet::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.winner().is_none());
        assert_eq!(s.max_n(), DEFAULT_MAX_N);
    }

    #[test]
    fn push_grows_and_first_is_winner() {
        let mut s = AlternativeSet::empty();
        s.push(dummy_candidate(10));
        s.push(dummy_candidate(20));
        assert_eq!(s.len(), 2);
        assert_eq!(s.winner().unwrap().static_cost.flops, 10);
    }

    #[test]
    fn from_candidates_seeds_and_preserves_order() {
        let s = AlternativeSet::from_candidates(
            vec![dummy_candidate(1), dummy_candidate(2), dummy_candidate(3)],
            5,
        );
        assert_eq!(s.len(), 3);
        assert_eq!(s.max_n(), 5);
        let flops: Vec<u64> = s.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![1, 2, 3]);
    }

    #[test]
    fn truncate_respects_max_n() {
        let mut s = AlternativeSet::with_max_n(2);
        s.push(dummy_candidate(1));
        s.push(dummy_candidate(2));
        s.push(dummy_candidate(3));
        s.push(dummy_candidate(4));
        assert_eq!(s.len(), 4);
        s.truncate_to_top_n();
        assert_eq!(s.len(), 2);
        let flops: Vec<u64> = s.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![1, 2]);
    }

    #[test]
    fn truncate_no_op_when_under_max_n() {
        let mut s = AlternativeSet::empty();
        s.push(dummy_candidate(1));
        s.truncate_to_top_n();
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn retain_indices_keeps_selected_entries() {
        let mut s = AlternativeSet::from_candidates(
            (0..5).map(|i| dummy_candidate(i)).collect(),
            DEFAULT_MAX_N,
        );
        s.retain_indices(&[0, 2, 4]);
        let flops: Vec<u64> = s.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![0, 2, 4]);
    }

    #[test]
    fn retain_indices_empty_clears_set() {
        let mut s = AlternativeSet::from_candidates(
            vec![dummy_candidate(1), dummy_candidate(2)],
            DEFAULT_MAX_N,
        );
        s.retain_indices(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn retain_indices_keep_all_is_identity() {
        let mut s = AlternativeSet::from_candidates(
            vec![dummy_candidate(1), dummy_candidate(2), dummy_candidate(3)],
            DEFAULT_MAX_N,
        );
        s.retain_indices(&[0, 1, 2]);
        assert_eq!(s.len(), 3);
    }

    #[test]
    #[should_panic(expected = "sorted strictly ascending")]
    fn retain_indices_unsorted_panics_in_debug() {
        let mut s = AlternativeSet::from_candidates(
            vec![dummy_candidate(1), dummy_candidate(2)],
            DEFAULT_MAX_N,
        );
        s.retain_indices(&[1, 0]);
    }

    #[test]
    #[should_panic(expected = "out-of-range")]
    fn retain_indices_out_of_range_panics_in_debug() {
        let mut s = AlternativeSet::from_candidates(
            vec![dummy_candidate(1)],
            DEFAULT_MAX_N,
        );
        s.retain_indices(&[5]);
    }

    /// Context defaults to None, round-trips through `set_context`,
    /// and survives retain + truncate (it describes the decision
    /// point, not the candidates).
    #[test]
    fn context_round_trips_and_survives_mutation() {
        use fuel_core_types::dispatch::{OpKind, SizeClass};
        use fuel_core_types::DType;

        let mut s = AlternativeSet::from_candidates(
            vec![dummy_candidate(1), dummy_candidate(2), dummy_candidate(3)],
            2,
        );
        assert!(s.context().is_none(), "fresh sets carry no context");

        let ctx = DecisionContext {
            op: OpKind::MatMul,
            principal_dtype: DType::F32,
            size_class: SizeClass(16),
        };
        s.set_context(ctx);
        assert_eq!(s.context(), Some(&ctx));

        s.retain_indices(&[0, 2]);
        s.truncate_to_top_n();
        assert_eq!(
            s.context(),
            Some(&ctx),
            "context survives retain + truncate",
        );
    }
}
