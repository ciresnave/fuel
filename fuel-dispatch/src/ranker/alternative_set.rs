//! `AlternativeSet` — bounded collection of [`Candidate`]s at one
//! graph decision point.
//!
//! Phase 1.1 of the picker-work arc. The set is what
//! [`apply_filter_chain`] mutates and what `rank_by_cost` orders.
//! The newtype exists so we have somewhere to hang the
//! per-decision-point invariants — retention, filter application,
//! eventual coupling resolution — without exposing a bare `Vec`.
//!
//! # Retention: per-ending-device Pareto frontier + crowding cap
//!
//! Phase B PR-B2 of the "plan IS the graph" rebuild **retires the
//! fixed top-N** retention (`DEFAULT_MAX_N` / `truncate_to_top_n`) and
//! replaces it with [`AlternativeSet::retain_per_device_frontier`],
//! per `docs/architecture/04-optimization.md` §"Bounding the frontier:
//! Pareto per device + crowding cap" and decisions-log #8.
//!
//! The bound is **not** a single small N across all devices (the old
//! "default N=3" framing strands slow devices — the scalar-top-N
//! failure). Instead, retention:
//!
//! 1. buckets candidates by **ending device** ([`Candidate::device`]);
//! 2. within each device bucket keeps the **Pareto-optimal**
//!    candidates (non-dominated under [`CostVector::dominates`]),
//!    dropping dominated ones — lossless because a path dominated on
//!    the same ending device can never beat its dominator downstream;
//! 3. **never strands the last `(device, backend)`** path — a
//!    `(device, backend)` pair always keeps at least one candidate,
//!    even if globally dominated (the constitution forbids stranding a
//!    device);
//! 4. backstops each device bucket with an **NSGA-II crowding-distance
//!    cap** of `keep`/device ([`KEEP_PER_DEVICE`] ≈ 32 from the
//!    frontier prototype) — when a bucket's frontier exceeds `keep`,
//!    the most-crowded entries are dropped (boundary points get
//!    infinity, interior points the sum of normalized neighbor gaps
//!    per [`CostVector`] axis) until the bucket is ≤ `keep`.
//!
//! Invariants (asserted + tested): **≥1 path per device** survives,
//! the total retained is **≤ `keep` × (#devices)**, the last
//! `(device, backend)` is never stranded, and the **arm-0 winner**
//! (the time-first candidate `rank_by_cost` puts at index 0) is always
//! retained — so realize behavior is preserved.
//!
//! [`apply_filter_chain`]: super::apply_filter_chain
//! [`CostVector::dominates`]: super::cost_vector::CostVector::dominates

use fuel_ir::dispatch::{OpKind, SizeClass};
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation};
use smallvec::SmallVec;

use super::candidate::Candidate;
use super::cost_vector::CostVector;

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

/// Crowding-distance cap **per ending device** — the real retention
/// knob (Phase B PR-B2), per `docs/architecture/04-optimization.md`
/// §"The tunable: `keep` per device". The whole surviving frontier is
/// bounded by `keep × devices`, so cost scales with the bounded
/// frontier, not an unbounded N × M.
///
/// `32` matches the frontier prototype (`C:/Projects/frontier-prototype`),
/// which found `keep ≈ 32`/device reproduces the no-cap optimum on
/// every runtime query (fastest / most-precise / least-memory /
/// balanced) and bounds even adversarial continuous-axis cases. It is
/// emphatically **not** the old "default N=3 across all devices": that
/// stranded slow devices (the scalar-top-N failure the per-device
/// frontier exists to avoid).
pub const KEEP_PER_DEVICE: usize = 32;

/// Bounded collection of [`Candidate`]s at one decision point. The
/// optimizer ranker constructs one of these per kernel-bearing graph
/// node, runs the filter chain, ranks survivors by composite cost,
/// and retains the **per-ending-device Pareto frontier** (capped at
/// [`KEEP_PER_DEVICE`]/device by crowding distance) for storage on the
/// optimized graph — see [`Self::retain_per_device_frontier`].
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
    /// Decision-point identity for dispatch-time Judge re-queries.
    /// `None` until `compile_plan` stamps it (or for hand-built
    /// test sets) — context-aware selectors then skip the Judge leg.
    context: Option<DecisionContext>,
}

impl AlternativeSet {
    /// Build an empty set. Tests + the candidate enumerator both use
    /// this. Retention (the per-device Pareto frontier + crowding cap)
    /// is applied later via [`Self::retain_per_device_frontier`], not
    /// at construction.
    pub fn empty() -> Self {
        Self {
            candidates: SmallVec::new(),
            context: None,
        }
    }

    /// Build a set from a pre-collected list of candidates. The
    /// enumerator path; retention (per-device Pareto frontier +
    /// crowding cap) happens after the filter chain + cost rank via
    /// [`Self::retain_per_device_frontier`], not here.
    pub fn from_candidates(candidates: Vec<Candidate>) -> Self {
        Self {
            candidates: SmallVec::from_vec(candidates),
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

    /// Retain the **per-ending-device Pareto frontier**, backstopped by
    /// an NSGA-II crowding-distance cap of `keep`/device — Phase B
    /// PR-B2's retention policy, replacing the retired fixed top-N
    /// truncation.
    ///
    /// Production callers pass [`KEEP_PER_DEVICE`]. The set is expected
    /// to have been ranked ([`Self::rank_by_cost`]) so the arm-0 winner
    /// (lowest central `time`) is at index 0; this method preserves
    /// that winner and re-ranks the survivors so the winner stays at
    /// index 0 (realize follows arm-0, so its identity must not change).
    ///
    /// # Algorithm
    ///
    /// 1. **Bucket by ending device** ([`Candidate::device`]).
    /// 2. **Per-device Pareto frontier:** within each bucket keep the
    ///    candidates not [`dominated`](CostVector::dominates) by any
    ///    other candidate in the *same* bucket; drop the dominated.
    /// 3. **Never strand the last `(device, backend)`:** for every
    ///    distinct `(device, backend)` present in the input, if the
    ///    Pareto filter dropped all of its candidates, re-admit its
    ///    time-best one. A `(device, backend)` with a single candidate
    ///    therefore always survives, even if globally dominated.
    /// 4. **Crowding cap:** if a device bucket still exceeds `keep`,
    ///    drop the lowest-crowding-distance entries (NSGA-II: boundary
    ///    points per axis get `f64::INFINITY`, interior points the sum
    ///    of normalized neighbor gaps over the [`CostVector`] axes)
    ///    until the bucket is `≤ keep` — but never drop a candidate
    ///    that is the *sole* survivor of its `(device, backend)`, and
    ///    never drop the global arm-0 winner.
    ///
    /// # Invariants (asserted)
    ///
    /// - **≥1 path per device** survives (each device bucket is
    ///   non-empty after retention).
    /// - **Total ≤ `keep × devices`.**
    /// - The arm-0 winner is retained.
    ///
    /// A `keep` of 0 is treated as 1 (a device bucket can never be
    /// emptied — that would strand the device).
    pub fn retain_per_device_frontier(&mut self, keep: usize) {
        let keep = keep.max(1);
        let n = self.candidates.len();
        if n == 0 {
            return;
        }

        // Index 0 is the arm-0 winner (caller ranked); it must survive.
        let winner_device = self.candidates[0].device;

        // Precompute each candidate's cost vector once.
        let vectors: Vec<CostVector> = self
            .candidates
            .iter()
            .map(CostVector::from_candidate)
            .collect();

        // (1) Bucket candidate indices by ending device. Use a Vec of
        //     (device, indices) to keep deterministic ordering (devices
        //     appear in first-seen order, which keeps the winner's
        //     device first since index 0 is the winner).
        let mut device_buckets: Vec<(DeviceLocation, Vec<usize>)> = Vec::new();
        for i in 0..n {
            let dev = self.candidates[i].device;
            if let Some(entry) = device_buckets.iter_mut().find(|(d, _)| *d == dev) {
                entry.1.push(i);
            } else {
                device_buckets.push((dev, vec![i]));
            }
        }

        // The final keep-set of original indices.
        let mut keep_set: Vec<usize> = Vec::with_capacity(n);

        for (_dev, bucket) in &device_buckets {
            // (2) Per-device Pareto frontier: keep indices not dominated
            //     by any other index in the SAME bucket.
            let mut frontier: Vec<usize> = bucket
                .iter()
                .copied()
                .filter(|&i| {
                    !bucket
                        .iter()
                        .any(|&j| j != i && vectors[j].dominates(&vectors[i]))
                })
                .collect();

            // (3) Never strand the last (device, backend): for each
            //     distinct backend in this bucket, ensure ≥1 of its
            //     candidates is on the frontier; otherwise re-admit its
            //     time-best (lowest total_order_key) candidate.
            let mut backends_in_bucket: Vec<BackendId> = Vec::new();
            for &i in bucket {
                let b = self.candidates[i].backend;
                if !backends_in_bucket.contains(&b) {
                    backends_in_bucket.push(b);
                }
            }
            for &b in &backends_in_bucket {
                let on_frontier = frontier
                    .iter()
                    .any(|&i| self.candidates[i].backend == b);
                if !on_frontier {
                    if let Some(&best) = bucket
                        .iter()
                        .filter(|&&i| self.candidates[i].backend == b)
                        .min_by_key(|&&i| vectors[i].total_order_key())
                    {
                        frontier.push(best);
                    }
                }
            }

            // The global winner must never be dropped from its bucket.
            if self.candidates.first().is_some()
                && _dev == &winner_device
                && !frontier.contains(&0)
            {
                frontier.push(0);
            }

            // (4) Crowding cap: reduce this bucket's frontier to `keep`
            //     by NSGA-II crowding distance, protecting the sole
            //     survivor of each (device, backend) and the arm-0
            //     winner.
            if frontier.len() > keep {
                crowding_cap(
                    &mut frontier,
                    &vectors,
                    &self.candidates,
                    keep,
                    winner_device,
                );
            }

            keep_set.extend(frontier);
        }

        // Sort the keep-set ascending so `retain_indices`' contract
        // (sorted, distinct) holds; the subsequent re-rank restores the
        // winner-first order.
        keep_set.sort_unstable();
        keep_set.dedup();

        // Invariants — assert before we mutate (the producer is
        // internal, so a violation is a bug, not a user error).
        debug_assert!(
            keep_set.contains(&0),
            "retain_per_device_frontier: arm-0 winner must be retained",
        );
        debug_assert!(
            keep_set.len() <= keep.saturating_mul(device_buckets.len()),
            "retain_per_device_frontier: total {} exceeds keep×devices = {}×{}",
            keep_set.len(),
            keep,
            device_buckets.len(),
        );

        self.retain_indices(&keep_set);

        // Re-rank so the arm-0 winner is back at index 0 (retention may
        // have reordered nothing, but `retain_indices` preserves the
        // pre-existing order, which for a ranked set already had the
        // winner first; re-rank is defensive + cheap).
        self.rank_by_cost();
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
    /// dominance is the relation the *frontier retention*
    /// ([`Self::retain_per_device_frontier`], PR-B2) uses — that
    /// retired the top-N truncation for a per-device Pareto frontier +
    /// crowding cap.
    ///
    /// Stable sort — equal-key candidates keep their relative order,
    /// which matters when registration order is the residual
    /// tie-breaker (and, post-Stage-2, when decision-device
    /// candidates are enumerated ahead of off-device ones).
    ///
    /// [`CostVector::total_order_key`]: super::cost_vector::CostVector::total_order_key
    /// [`CostVector::dominates`]: super::cost_vector::CostVector::dominates
    pub fn rank_by_cost(&mut self) {
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
    /// pipeline — filter → rank → retain — this is the runtime
    /// selector's (Picker 2) default pick. `None` if the set is
    /// empty.
    pub fn winner(&self) -> Option<&Candidate> {
        self.candidates.first()
    }
}

/// Reduce `frontier` (original-index references into `candidates`) to
/// `keep` entries by **NSGA-II crowding distance** over the
/// [`CostVector`] axes, dropping the most-crowded first.
///
/// Crowding distance per axis: candidates are sorted on that axis, the
/// two boundary points get `f64::INFINITY` (so per-axis extremes — the
/// spread — always survive), and each interior point accrues the
/// normalized gap between its two neighbors. A candidate's total
/// crowding distance is the sum over all axes. Lowest distance =
/// most-crowded = dropped first.
///
/// Two entries are protected from being dropped: (a) the global arm-0
/// winner (index 0, when this is the winner's device bucket), and
/// (b) the sole survivor of any `(device, backend)` in this bucket —
/// dropping it would strand that backend (never-strand-last invariant).
fn crowding_cap(
    frontier: &mut Vec<usize>,
    vectors: &[CostVector],
    candidates: &[Candidate],
    keep: usize,
    winner_device: DeviceLocation,
) {
    // The crowding distance is computed over the CURRENT frontier
    // membership and recomputed as we drop, so the surviving spread is
    // re-measured each time (standard NSGA-II truncation).
    while frontier.len() > keep {
        let dist = crowding_distances(frontier, vectors);

        // Find the protected set: arm-0 winner (in its own device
        // bucket) + any backend with exactly one representative left.
        let winner_present = frontier.contains(&0);
        let mut backend_counts: std::collections::HashMap<BackendId, usize> =
            std::collections::HashMap::new();
        for &i in frontier.iter() {
            *backend_counts.entry(candidates[i].backend).or_insert(0) += 1;
        }

        // Drop the droppable entry with the lowest crowding distance.
        let mut victim: Option<(usize, f64)> = None; // (position in frontier, distance)
        for (pos, &i) in frontier.iter().enumerate() {
            let is_winner =
                i == 0 && winner_present && candidates[i].device == winner_device;
            let is_sole_backend =
                backend_counts.get(&candidates[i].backend).copied().unwrap_or(0) <= 1;
            if is_winner || is_sole_backend {
                continue;
            }
            let d = dist[pos];
            match victim {
                Some((_, best)) if best <= d => {}
                _ => victim = Some((pos, d)),
            }
        }

        match victim {
            Some((pos, _)) => {
                frontier.remove(pos);
            }
            // Everything remaining is protected (all distinct backends,
            // and the winner) — can't cap below the protected floor
            // without stranding a backend. Stop.
            None => break,
        }
    }
}

/// Per-entry NSGA-II crowding distance over the [`CostVector`] axes,
/// indexed parallel to `frontier`. Boundary points per axis get
/// `f64::INFINITY`; interior points accrue normalized neighbor gaps.
///
/// Axes (all treated as numeric for spread purposes): central `time`,
/// host-RAM bytes, device-VRAM bytes, `precision` digits, `accuracy`
/// rank. Orientation doesn't matter for crowding — only the spread
/// along each axis does.
fn crowding_distances(frontier: &[usize], vectors: &[CostVector]) -> Vec<f64> {
    let m = frontier.len();
    let mut dist = vec![0.0f64; m];
    if m <= 2 {
        // Every point is a boundary point — all maximally spread.
        return vec![f64::INFINITY; m];
    }

    // Each axis as an f64 extractor.
    let axes: [fn(&CostVector) -> f64; 5] = [
        |v| v.time as f64,
        |v| v.memory.host_ram_bytes as f64,
        |v| v.memory.device_vram_bytes as f64,
        |v| v.precision as f64,
        |v| v.accuracy.rank() as f64,
    ];

    for axis in axes {
        // Sort frontier positions by this axis value.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| {
            let va = axis(&vectors[frontier[a]]);
            let vb = axis(&vectors[frontier[b]]);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let min = axis(&vectors[frontier[order[0]]]);
        let max = axis(&vectors[frontier[order[m - 1]]]);
        let range = max - min;

        // Boundary points get infinity (preserve per-axis extremes).
        dist[order[0]] = f64::INFINITY;
        dist[order[m - 1]] = f64::INFINITY;

        if range <= 0.0 {
            // Degenerate axis (all equal) contributes nothing to the
            // interior; boundaries already pinned to infinity.
            continue;
        }

        for k in 1..(m - 1) {
            let prev = axis(&vectors[frontier[order[k - 1]]]);
            let next = axis(&vectors[frontier[order[k + 1]]]);
            dist[order[k]] += (next - prev) / range;
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DeviceLocation, Layout, Result};
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

    /// A candidate with explicit device + backend + cost knobs, for the
    /// PR-B2 per-device-frontier tests. `time` is set via `flops`
    /// (composite_ns ≈ flops here since bytes/overhead are 0).
    fn cand(
        device: DeviceLocation,
        backend: BackendId,
        flops: u64,
        bytes: u64,
        precision: PrecisionGuarantee,
    ) -> Candidate {
        Candidate {
            backend,
            device,
            precision,
            static_cost: CostEstimate { flops, bytes_moved: bytes, kernel_overhead_ns: 0 },
            ..dummy_candidate(0)
        }
    }

    fn cuda(id: usize) -> DeviceLocation {
        DeviceLocation::Cuda { gpu_id: id }
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
        let mut set = AlternativeSet::from_candidates(cands);
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
        let mut s = AlternativeSet::from_candidates(vec![
            Candidate { inbound_transfer_ns: 5_000, ..dummy_candidate(100) },
            dummy_candidate(200),
        ]);
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
        let s = AlternativeSet::from_candidates(vec![
            dummy_candidate(1),
            dummy_candidate(2),
            dummy_candidate(3),
        ]);
        assert_eq!(s.len(), 3);
        let flops: Vec<u64> = s.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![1, 2, 3]);
    }

    #[test]
    fn retain_indices_keeps_selected_entries() {
        let mut s = AlternativeSet::from_candidates(
            (0..5).map(|i| dummy_candidate(i)).collect(),
        );
        s.retain_indices(&[0, 2, 4]);
        let flops: Vec<u64> = s.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![0, 2, 4]);
    }

    #[test]
    fn retain_indices_empty_clears_set() {
        let mut s = AlternativeSet::from_candidates(vec![
            dummy_candidate(1),
            dummy_candidate(2),
        ]);
        s.retain_indices(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn retain_indices_keep_all_is_identity() {
        let mut s = AlternativeSet::from_candidates(vec![
            dummy_candidate(1),
            dummy_candidate(2),
            dummy_candidate(3),
        ]);
        s.retain_indices(&[0, 1, 2]);
        assert_eq!(s.len(), 3);
    }

    #[test]
    #[should_panic(expected = "sorted strictly ascending")]
    fn retain_indices_unsorted_panics_in_debug() {
        let mut s = AlternativeSet::from_candidates(vec![
            dummy_candidate(1),
            dummy_candidate(2),
        ]);
        s.retain_indices(&[1, 0]);
    }

    #[test]
    #[should_panic(expected = "out-of-range")]
    fn retain_indices_out_of_range_panics_in_debug() {
        let mut s = AlternativeSet::from_candidates(vec![dummy_candidate(1)]);
        s.retain_indices(&[5]);
    }

    /// Context defaults to None, round-trips through `set_context`,
    /// and survives retain (it describes the decision point, not the
    /// candidates).
    #[test]
    fn context_round_trips_and_survives_mutation() {
        use fuel_ir::dispatch::{OpKind, SizeClass};
        use fuel_ir::DType;

        let mut s = AlternativeSet::from_candidates(vec![
            dummy_candidate(1),
            dummy_candidate(2),
            dummy_candidate(3),
        ]);
        assert!(s.context().is_none(), "fresh sets carry no context");

        let ctx = DecisionContext {
            op: OpKind::MatMul,
            principal_dtype: DType::F32,
            size_class: SizeClass(16),
        };
        s.set_context(ctx);
        assert_eq!(s.context(), Some(&ctx));

        s.retain_indices(&[0, 2]);
        assert_eq!(
            s.context(),
            Some(&ctx),
            "context survives retain",
        );
    }

    // ===== Phase B PR-B2: per-ending-device Pareto frontier +
    //        crowding cap (retiring the fixed top-N). =====

    /// (a) A multi-device set retains the **per-device Pareto
    /// frontier**, NOT the global top-3: a slow-but-only-CUDA candidate
    /// survives even when several CPU candidates are all faster. The
    /// old `DEFAULT_MAX_N = 3` truncation on a global time sort would
    /// have stranded the CUDA device entirely.
    #[test]
    fn retains_per_device_frontier_not_global_top_n() {
        // 3 fast CPU candidates + 1 slow CUDA candidate. A global
        // top-3 keeps only the 3 CPU ones (CUDA stranded). The
        // per-device frontier must keep ≥1 CPU and the 1 CUDA.
        let cands = vec![
            cand(DeviceLocation::Cpu, BackendId::Cpu, 100, 0, PrecisionGuarantee::REFERENCE),
            cand(DeviceLocation::Cpu, BackendId::Cpu, 200, 0, PrecisionGuarantee::REFERENCE),
            cand(DeviceLocation::Cpu, BackendId::Cpu, 300, 0, PrecisionGuarantee::REFERENCE),
            cand(cuda(0), BackendId::Cuda, 10_000, 0, PrecisionGuarantee::REFERENCE),
        ];
        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        set.retain_per_device_frontier(KEEP_PER_DEVICE);

        let has_cuda = set
            .alternatives()
            .iter()
            .any(|c| matches!(c.device, DeviceLocation::Cuda { .. }));
        let has_cpu = set
            .alternatives()
            .iter()
            .any(|c| c.device == DeviceLocation::Cpu);
        assert!(has_cuda, "slow CUDA path must NOT be stranded by a global top-N");
        assert!(has_cpu, "CPU path survives");

        // Arm-0 winner is the fastest overall (CPU flops=100).
        assert_eq!(set.winner().unwrap().static_cost.flops, 100);
    }

    /// (b) Within a device, a dominated candidate is dropped. Two CPU
    /// candidates where one is strictly worse on every axis → only the
    /// dominator survives on that device.
    #[test]
    fn drops_within_device_dominated_candidate() {
        // dominator: fast + reference precision.
        // dominated: slower + lower precision (UNAUDITED) — strictly
        //            worse on time AND precision AND accuracy.
        let cands = vec![
            cand(DeviceLocation::Cpu, BackendId::Cpu, 100, 0, PrecisionGuarantee::REFERENCE),
            cand(DeviceLocation::Cpu, BackendId::Cpu, 500, 0, PrecisionGuarantee::UNAUDITED),
        ];
        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        set.retain_per_device_frontier(KEEP_PER_DEVICE);

        // Both are the SAME (device, backend), so never-strand doesn't
        // force the dominated one back — the dominator alone survives.
        assert_eq!(set.len(), 1, "dominated same-backend candidate dropped");
        assert_eq!(set.winner().unwrap().static_cost.flops, 100);
    }

    /// (c) A device frontier larger than `keep` is crowding-capped to
    /// exactly `keep`, keeping the spread (the per-axis boundary points
    /// — fastest + slowest along the time axis — survive).
    #[test]
    fn crowding_caps_oversized_frontier_keeping_spread() {
        // Build a single-device, single-backend Pareto front along the
        // time↔precision tradeoff: faster ⇒ lower precision, so every
        // candidate is mutually non-dominated (a genuine frontier).
        let keep = 4usize;
        let mut cands = Vec::new();
        let n = 10u64;
        for k in 0..n {
            // time INcreases with k AND precision INcreases with k (a
            // genuine speed↔precision tradeoff) → every pair is
            // mutually non-dominated, so the Pareto frontier == all 10.
            let rel = 10f64.powi(-(2 + k as i32)); // higher k → tighter
            let p = PrecisionGuarantee {
                bit_stable_on_same_hardware: true,
                max_ulp: None,
                max_relative: Some(rel),
                max_absolute: None,
                notes: "frontier",
            };
            cands.push(cand(DeviceLocation::Cpu, BackendId::Cpu, 100 + k * 100, 0, p));
        }
        let fastest_time_flops = 100; // k=0
        let slowest_time_flops = 100 + (n - 1) * 100; // k=9 → 1000

        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        // Sanity: all mutually non-dominated → frontier == all 10.
        set.retain_per_device_frontier(keep);

        assert_eq!(set.len(), keep, "capped to exactly keep");

        // Spread preserved: the time-axis extremes survive.
        let times: Vec<u64> =
            set.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert!(
            times.contains(&fastest_time_flops),
            "fastest (time-axis min) boundary survives crowding; got {times:?}",
        );
        assert!(
            times.contains(&slowest_time_flops),
            "slowest (time-axis max) boundary survives crowding; got {times:?}",
        );
    }

    /// (d) ≥1 per device survives + total ≤ keep × devices.
    #[test]
    fn at_least_one_per_device_and_total_bounded() {
        let keep = 3usize;
        // 3 devices, each with several candidates.
        let mut cands = Vec::new();
        for d in 0..3usize {
            let dev = if d == 0 { DeviceLocation::Cpu } else { cuda(d) };
            let backend = if d == 0 { BackendId::Cpu } else { BackendId::Cuda };
            for k in 0..6u64 {
                // mutually non-dominated within a device (time↔precision)
                let rel = 10f64.powi(-(2 + (6 - k) as i32));
                let p = PrecisionGuarantee {
                    bit_stable_on_same_hardware: true,
                    max_ulp: None,
                    max_relative: Some(rel),
                    max_absolute: None,
                    notes: "x",
                };
                cands.push(cand(dev, backend, 100 + k * 50 + d as u64, 0, p));
            }
        }
        let n_devices = 3;
        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        set.retain_per_device_frontier(keep);

        // ≥1 per device.
        for d in 0..3usize {
            let dev = if d == 0 { DeviceLocation::Cpu } else { cuda(d) };
            let count = set.alternatives().iter().filter(|c| c.device == dev).count();
            assert!(count >= 1, "device {dev:?} must keep ≥1 path");
        }
        // total ≤ keep × devices.
        assert!(
            set.len() <= keep * n_devices,
            "total {} ≤ keep×devices = {}",
            set.len(),
            keep * n_devices,
        );
    }

    /// Never-strand-last `(device, backend)`: a globally-dominated
    /// candidate that is the SOLE representative of its (device,
    /// backend) is retained anyway. Here a Vulkan candidate dominated
    /// on every axis by a CPU one survives because it is Vulkan's only
    /// path (and Vulkan is its own device bucket — the per-device
    /// frontier already keeps it; this also exercises a same-device
    /// distinct-backend case).
    #[test]
    fn never_strands_last_device_backend() {
        // CPU dominator (fast, reference) vs a Vulkan candidate that is
        // slower + unaudited. Different device ⇒ different bucket ⇒
        // Vulkan's bucket keeps it by the per-device frontier.
        let cands = vec![
            cand(DeviceLocation::Cpu, BackendId::Cpu, 100, 0, PrecisionGuarantee::REFERENCE),
            cand(
                DeviceLocation::Vulkan { gpu_id: 0 },
                BackendId::Vulkan,
                9_999,
                0,
                PrecisionGuarantee::UNAUDITED,
            ),
        ];
        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        set.retain_per_device_frontier(KEEP_PER_DEVICE);

        let has_vulkan = set
            .alternatives()
            .iter()
            .any(|c| matches!(c.device, DeviceLocation::Vulkan { .. }));
        assert!(has_vulkan, "last Vulkan (device, backend) must never be stranded");
    }

    /// A fully-dominated device bucket whose sole candidate is globally
    /// dominated on every axis is still retained — the per-device
    /// frontier never empties a device, and the `(device, backend)`
    /// re-admission backstops it. Here the Metal candidate is worse on
    /// every axis than the CPU one but is its (device, backend)'s only
    /// path.
    #[test]
    fn never_strands_globally_dominated_sole_path() {
        let cands = vec![
            cand(DeviceLocation::Cpu, BackendId::Cpu, 100, 0, PrecisionGuarantee::REFERENCE),
            cand(
                DeviceLocation::Metal { gpu_id: 0 },
                BackendId::Metal,
                9_999,
                999_999,
                PrecisionGuarantee::UNAUDITED,
            ),
        ];
        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        set.retain_per_device_frontier(KEEP_PER_DEVICE);

        let backends: Vec<BackendId> =
            set.alternatives().iter().map(|c| c.backend).collect();
        assert!(backends.contains(&BackendId::Cpu), "CPU dominator kept");
        assert!(
            backends.contains(&BackendId::Metal),
            "dominated-but-sole Metal (device, backend) never stranded; got {backends:?}",
        );
    }

    /// (e) The arm-0 winner is unchanged vs a plain B1 rank: retention
    /// keeps the same index-0 candidate the rank produced.
    #[test]
    fn arm0_winner_unchanged_by_retention() {
        let cands = vec![
            cand(DeviceLocation::Cpu, BackendId::Cpu, 300, 0, PrecisionGuarantee::REFERENCE),
            cand(cuda(0), BackendId::Cuda, 100, 0, PrecisionGuarantee::REFERENCE),
            cand(DeviceLocation::Cpu, BackendId::Cpu, 200, 0, PrecisionGuarantee::REFERENCE),
        ];
        // B1 winner: lowest time (flops=100 on CUDA).
        let mut b1 = AlternativeSet::from_candidates(cands.clone());
        b1.rank_by_cost();
        let b1_winner = (b1.winner().unwrap().backend, b1.winner().unwrap().device);

        let mut set = AlternativeSet::from_candidates(cands);
        set.rank_by_cost();
        set.retain_per_device_frontier(KEEP_PER_DEVICE);
        let b2_winner = (set.winner().unwrap().backend, set.winner().unwrap().device);

        assert_eq!(b1_winner, b2_winner, "arm-0 winner unchanged by B2 retention");
        assert_eq!(set.winner().unwrap().static_cost.flops, 100);
    }

    /// A single-device, single-candidate set is unchanged (the common
    /// CPU `--lib` case): retention keeps the one candidate.
    #[test]
    fn single_candidate_passthrough() {
        let mut set = AlternativeSet::from_candidates(vec![dummy_candidate(42)]);
        set.rank_by_cost();
        set.retain_per_device_frontier(KEEP_PER_DEVICE);
        assert_eq!(set.len(), 1);
        assert_eq!(set.winner().unwrap().static_cost.flops, 42);
    }
}
