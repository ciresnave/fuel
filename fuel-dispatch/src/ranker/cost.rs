//! Cost composition + ranking for the picker. Phase 1.4 of the
//! picker-work arc.
//!
//! Layer-1 static cost composition only. The scoring function
//! converts a `CostEstimate { flops, bytes_moved, kernel_overhead_ns }`
//! into a composite nanosecond figure the ranker sorts by. Layer-2
//! (Judge empirical refinement) lands in Phase 3 and either
//! refines `static_cost` before this composition happens or
//! provides its own rank.
//!
//! # The composition model
//!
//! For each candidate, the picker estimates wall-clock by combining
//! three terms:
//!
//! - **Compute time** = `flops / cap.peak_flops_per_s` (seconds)
//! - **Memory time** = `bytes_moved / cap.peak_bandwidth_bytes_per_s`
//! - **Overhead** = `kernel_overhead_ns / 1e9` (seconds)
//!
//! Returned as `u64` nanoseconds (saturating). The arithmetic
//! intentionally treats compute and memory as **parallel** — a
//! kernel that's memory-bound spends `max(compute, memory)` rather
//! than `compute + memory`. This is the classic roofline shape and
//! matches how every modern accelerator's pipelines actually behave.
//!
//! For Phase 1.4 the picker doesn't yet read live BackendCapabilities
//! (the populate-costs step is the caller's responsibility once it
//! has them). The standalone [`composite_ns`] takes a `CostEstimate`
//! already calibrated against capabilities and reduces it to a
//! sortable scalar.

use fuel_core_types::backend::BackendCapabilities;
use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, Shape};

use super::alternative_set::AlternativeSet;
use super::judge::JudgeOracle;
use crate::fused::CostEstimate;
use crate::kernel::{KernelBindingTable, OpParams};

/// Convert a `CostEstimate` into a sortable composite nanosecond
/// figure. Lower is better. Treats compute + memory as parallel
/// (roofline-style) and adds launch overhead serially.
///
/// Saturating arithmetic — extreme inputs clamp to `u64::MAX`
/// rather than overflowing. This is plan-time scoring, not a
/// realtime path; the saturation is purely defensive.
pub fn composite_ns(cost: &CostEstimate) -> u64 {
    // Treat 1 FLOP ≈ 1 ns and 1 byte ≈ 0.25 ns as Layer-1's neutral
    // throughput priors. Real backend capabilities (peak FLOPs/s,
    // bandwidth GB/s) will refine this once they thread through from
    // SystemTopology in Phase 1.5+; until then, the proportionality
    // is what matters for ordering, not the absolute numbers.
    let compute_ns = cost.flops;                // 1 FLOP/ns prior
    let memory_ns = cost.bytes_moved / 4;       // 4 GB/s prior
    let parallel_ns = compute_ns.max(memory_ns);
    parallel_ns.saturating_add(cost.kernel_overhead_ns as u64)
}

/// Closure type for "give me the BackendCapabilities for this
/// backend." The caller (Phase 1.5's compile_plan) reads from
/// `SystemTopology::capabilities(...)` or the CapabilityRegistry;
/// this module accepts a callback so it stays ignorant of where
/// the capabilities come from.
pub type CapabilitiesLookup<'a> = dyn Fn(BackendId) -> Option<&'a BackendCapabilities> + 'a;

/// Compute and populate `static_cost` on every candidate in `set`.
///
/// For each candidate, looks up the binding-table entry that
/// produced it (matched by `kernel` function pointer identity) and
/// invokes the entry's `CostFn(shapes, dtypes, op_params, caps)`
/// to compute the Layer-1 cost. Candidates whose backend has no
/// capabilities entry, or whose kernel pointer is no longer
/// findable (defensive — the table shouldn't change between
/// enumeration and costing within one plan), retain the default
/// zero-cost.
///
/// When `judge` is `Some`, runs the Layer-2 refinement after
/// Layer-1: for each candidate, queries the oracle for a measured
/// latency at this (op, dtype, size_class, backend) cell. If the
/// measurement exists, **replaces** the Layer-1 estimate with a
/// Layer-2-equivalent: zero FLOPs + zero bytes + `kernel_overhead_ns
/// = saturating_cast(latency_ns)`. The composite scoring then
/// returns that latency directly. Cells without measurements keep
/// the Layer-1 estimate (silent fallback — no measurement is NOT
/// the same as "this kernel is fast").
///
/// The size_class is derived from `shapes[0]`'s element count via
/// [`SizeClass::from_elem_count`]. Matches the Judge profiler's
/// bucketing convention. If `shapes` is empty (truly nullary op),
/// `SizeClass(0)` is used as a defensive default.
///
/// The caller supplies `shapes` (input operand shapes for the
/// decision point), the capabilities lookup closure, and optional
/// judge oracle.
pub fn compute_static_costs(
    set: &mut AlternativeSet,
    op_kind: OpKind,
    dtypes: &[DType],
    shapes: &[Shape],
    bindings: &KernelBindingTable,
    capabilities_for: &CapabilitiesLookup<'_>,
    judge: Option<&dyn JudgeOracle>,
) {
    // Layer-1: static cost via the binding-table's CostFn.
    for i in 0..set.len() {
        let (kernel_ptr, backend, op_params) = {
            let c = &set.alternatives()[i];
            (c.kernel as *const () as usize, c.backend, c.op_params.clone())
        };
        let entries = bindings.lookup_alternatives(op_kind, dtypes, backend);
        let Some(entry) = entries.iter().find(|e| {
            (e.kernel as *const () as usize) == kernel_ptr
        }) else {
            continue;
        };
        let Some(caps) = capabilities_for(backend) else {
            continue;
        };
        let cost = (entry.cost)(shapes, dtypes, &op_params, caps);
        set.set_static_cost(i, cost);
    }

    // Layer-2 refinement: if the Judge has data for this cell,
    // replace the Layer-1 estimate with a latency-equivalent shape
    // so composite_ns returns the measurement.
    let Some(judge) = judge else {
        return;
    };
    // Pick the principal dtype — by convention the first input's
    // dtype, which matches the Judge profiler's keying. For
    // mixed-dtype ops (Cast etc.) this picks the source dtype;
    // future refinement could thread the destination as a separate
    // axis if a measurable wins/losses pattern emerges.
    let principal_dtype = match dtypes.first() {
        Some(&dt) => dt,
        None => return,
    };
    let size_class = shapes
        .first()
        .map(|s| SizeClass::from_elem_count(s.elem_count()))
        .unwrap_or(SizeClass(0));
    for i in 0..set.len() {
        let backend = set.alternatives()[i].backend;
        let Some(latency_ns) = judge
            .measured_latency_ns(op_kind, principal_dtype, size_class, backend)
        else {
            continue;
        };
        // Convert latency to a CostEstimate that composite_ns will
        // return as-is. Saturate u64 → u32 for kernel_overhead_ns;
        // anything above u32::MAX ns (~4.3 seconds) is in
        // practice a degenerate case we still want correctly
        // ordered.
        let overhead = latency_ns.min(u32::MAX as u64) as u32;
        set.set_static_cost(
            i,
            CostEstimate {
                flops: 0,
                bytes_moved: 0,
                kernel_overhead_ns: overhead,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{unknown_cost, KernelBindingTable, KernelCaps};
    use crate::ranker::alternative_set::{AlternativeSet, DEFAULT_MAX_N};
    use crate::ranker::candidate::Candidate;
    use fuel_core_types::backend::{BackendCapabilities, SubstrateClass, TransferPath};
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result, Shape};
    use fuel_storage::Storage;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, RwLock};

    fn noop_a(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn noop_b(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn caps_for_test(backend_id: BackendId) -> BackendCapabilities {
        BackendCapabilities {
            backend_id,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 64,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
            storage_substrate: SubstrateClass::HostBytes,
        }
    }

    fn candidate_with_cost(kernel: crate::kernel::KernelRef, cost: CostEstimate) -> Candidate {
        Candidate {
            kernel,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: cost,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    #[test]
    fn composite_ns_zero_cost_is_zero() {
        assert_eq!(composite_ns(&CostEstimate::default()), 0);
    }

    #[test]
    fn composite_ns_flops_dominant() {
        // 1000 FLOPs, 0 bytes, 0 overhead → 1000 ns.
        let c = CostEstimate { flops: 1000, bytes_moved: 0, kernel_overhead_ns: 0 };
        assert_eq!(composite_ns(&c), 1000);
    }

    #[test]
    fn composite_ns_memory_dominant() {
        // 0 FLOPs, 4000 bytes, 0 overhead → 1000 ns (4 bytes/ns).
        let c = CostEstimate { flops: 0, bytes_moved: 4000, kernel_overhead_ns: 0 };
        assert_eq!(composite_ns(&c), 1000);
    }

    #[test]
    fn composite_ns_takes_max_of_compute_and_memory() {
        // Compute = 500 ns, memory = 1000 ns → max is 1000 (parallel).
        let c = CostEstimate { flops: 500, bytes_moved: 4000, kernel_overhead_ns: 0 };
        assert_eq!(composite_ns(&c), 1000);
    }

    #[test]
    fn composite_ns_overhead_serial_after_parallel() {
        // Parallel work = max(500, 800) = 800. Overhead = 200.
        // Total = 1000.
        let c = CostEstimate { flops: 500, bytes_moved: 3200, kernel_overhead_ns: 200 };
        assert_eq!(composite_ns(&c), 1000);
    }

    #[test]
    fn composite_ns_saturates_at_u64_max() {
        let c = CostEstimate {
            flops: u64::MAX,
            bytes_moved: u64::MAX,
            kernel_overhead_ns: u32::MAX,
        };
        // max() of two u64::MAX values is u64::MAX; saturating_add
        // pins to u64::MAX.
        assert_eq!(composite_ns(&c), u64::MAX);
    }

    #[test]
    fn rank_by_composite_cost_orders_ascending() {
        let mut set = AlternativeSet::from_candidates(
            vec![
                candidate_with_cost(noop_a, CostEstimate { flops: 300, bytes_moved: 0, kernel_overhead_ns: 0 }),
                candidate_with_cost(noop_b, CostEstimate { flops: 100, bytes_moved: 0, kernel_overhead_ns: 0 }),
                candidate_with_cost(noop_a, CostEstimate { flops: 200, bytes_moved: 0, kernel_overhead_ns: 0 }),
            ],
            DEFAULT_MAX_N,
        );
        set.rank_by_composite_cost();
        let costs: Vec<u64> = set.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(costs, vec![100, 200, 300]);
    }

    #[test]
    fn rank_then_truncate_yields_top_n() {
        let mut set = AlternativeSet::from_candidates(
            (0..5)
                .map(|i| candidate_with_cost(noop_a, CostEstimate {
                    flops: (5 - i) * 100,
                    bytes_moved: 0,
                    kernel_overhead_ns: 0,
                }))
                .collect(),
            2,
        );
        set.rank_by_composite_cost();
        set.truncate_to_top_n();
        assert_eq!(set.len(), 2);
        let costs: Vec<u64> = set.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(costs, vec![100, 200], "top-2 cheapest survive");
    }

    fn make_cost_fn(flops: u64, bytes: u64, overhead: u32) -> crate::kernel::CostFn {
        // Function pointers can't capture state, so we use distinct
        // functions per test scenario. For the populate test below,
        // a simple two-tier setup is enough.
        // Hack: define a closure-like family by using nested fn defs.
        let _ = (flops, bytes, overhead);
        |_, _, _, _| CostEstimate { flops: 1000, bytes_moved: 0, kernel_overhead_ns: 0 }
    }

    #[test]
    fn compute_static_costs_populates_via_binding_lookup() {
        fn cost_a(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 500, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise,
            &dtypes,
            BackendId::Cpu,
            noop_a,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cost_a,
        );

        let mut set = AlternativeSet::from_candidates(
            vec![candidate_with_cost(noop_a, CostEstimate::default())],
            DEFAULT_MAX_N,
        );

        let cpu_caps = caps_for_test(BackendId::Cpu);
        let lookup: HashMap<BackendId, BackendCapabilities> =
            [(BackendId::Cpu, cpu_caps)].into_iter().collect();
        let lookup_fn = |b: BackendId| lookup.get(&b);

        compute_static_costs(
            &mut set,
            OpKind::AddElementwise,
            &dtypes,
            &[Shape::from(vec![4])],
            &bindings,
            &lookup_fn,
            None,
        );

        assert_eq!(set.alternatives()[0].static_cost.flops, 500);
        let _ = make_cost_fn; // silence unused-warning
    }

    #[test]
    fn compute_static_costs_leaves_default_when_no_capabilities() {
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise,
            &dtypes,
            BackendId::Cpu,
            noop_a,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        let mut set = AlternativeSet::from_candidates(
            vec![candidate_with_cost(noop_a, CostEstimate::default())],
            DEFAULT_MAX_N,
        );
        // Empty lookup → no caps → cost not computed.
        let empty_lookup = |_: BackendId| -> Option<&BackendCapabilities> { None };
        compute_static_costs(
            &mut set,
            OpKind::AddElementwise,
            &dtypes,
            &[Shape::from(vec![4])],
            &bindings,
            &empty_lookup,
            None,
        );
        assert_eq!(set.alternatives()[0].static_cost, CostEstimate::default());
    }

    #[test]
    fn compute_static_costs_skips_candidates_without_matching_binding() {
        // Candidate's kernel pointer doesn't match any binding-table
        // entry (defensive case). Cost stays default.
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise,
            &dtypes,
            BackendId::Cpu,
            noop_a,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        let mut set = AlternativeSet::from_candidates(
            // noop_b isn't registered.
            vec![candidate_with_cost(noop_b, CostEstimate::default())],
            DEFAULT_MAX_N,
        );
        let cpu_caps = caps_for_test(BackendId::Cpu);
        let lookup: HashMap<BackendId, BackendCapabilities> =
            [(BackendId::Cpu, cpu_caps)].into_iter().collect();
        compute_static_costs(
            &mut set,
            OpKind::AddElementwise,
            &dtypes,
            &[Shape::from(vec![4])],
            &bindings,
            &|b| lookup.get(&b),
            None,
        );
        assert_eq!(set.alternatives()[0].static_cost, CostEstimate::default());
    }

    // ===== Phase 3: Layer-2 refinement via JudgeOracle =====

    #[test]
    fn judge_refinement_replaces_layer1_with_measured_latency() {
        use crate::ranker::judge::HashMapJudge;

        // Layer 1 says this kernel is 1000 ns; Judge measured 250 ns.
        // After refinement composite_ns should report the measurement.
        fn layer1(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 1000, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise,
            &dtypes,
            BackendId::Cpu,
            noop_a,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            layer1,
        );

        let mut set = AlternativeSet::from_candidates(
            vec![candidate_with_cost(noop_a, CostEstimate::default())],
            DEFAULT_MAX_N,
        );
        let cpu_caps = caps_for_test(BackendId::Cpu);
        let lookup: HashMap<BackendId, BackendCapabilities> =
            [(BackendId::Cpu, cpu_caps)].into_iter().collect();
        let shapes = [Shape::from(vec![4])];
        let sc = SizeClass::from_elem_count(4);

        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Cpu, 250);

        compute_static_costs(
            &mut set,
            OpKind::AddElementwise,
            &dtypes,
            &shapes,
            &bindings,
            &|b| lookup.get(&b),
            Some(&judge),
        );

        let c = &set.alternatives()[0];
        assert_eq!(c.static_cost.flops, 0, "Layer-2 zeroes FLOPs");
        assert_eq!(c.static_cost.bytes_moved, 0, "Layer-2 zeroes bytes");
        assert_eq!(c.static_cost.kernel_overhead_ns, 250, "Layer-2 stamps latency");
        assert_eq!(composite_ns(&c.static_cost), 250, "composite returns measurement");
    }

    #[test]
    fn judge_missing_measurement_leaves_layer1_intact() {
        // Cell isn't in the Judge map → Layer-1 estimate stays.
        fn layer1(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 1000, bytes_moved: 4000, kernel_overhead_ns: 50 }
        }
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise,
            &dtypes,
            BackendId::Cpu,
            noop_a,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            layer1,
        );
        let mut set = AlternativeSet::from_candidates(
            vec![candidate_with_cost(noop_a, CostEstimate::default())],
            DEFAULT_MAX_N,
        );
        let cpu_caps = caps_for_test(BackendId::Cpu);
        let lookup: HashMap<BackendId, BackendCapabilities> =
            [(BackendId::Cpu, cpu_caps)].into_iter().collect();
        let empty_judge = crate::ranker::judge::HashMapJudge::new();
        compute_static_costs(
            &mut set,
            OpKind::AddElementwise,
            &dtypes,
            &[Shape::from(vec![4])],
            &bindings,
            &|b| lookup.get(&b),
            Some(&empty_judge),
        );
        let c = &set.alternatives()[0];
        assert_eq!(c.static_cost.flops, 1000, "Layer-1 FLOPs survive");
        assert_eq!(c.static_cost.bytes_moved, 4000);
        assert_eq!(c.static_cost.kernel_overhead_ns, 50);
    }

    #[test]
    fn judge_refinement_per_backend_can_flip_winner() {
        // Two backends. Layer-1 says CPU cheap, Aocl expensive.
        // Judge measured opposite: Aocl 100ns, CPU 500ns.
        // After refinement + rank, Aocl wins.
        fn cpu_layer1(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 100, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        fn aocl_layer1(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 1000, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise, &dtypes, BackendId::Cpu, noop_a,
            KernelCaps::empty(), PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cpu_layer1,
        );
        bindings.register_full(
            OpKind::AddElementwise, &dtypes, BackendId::Aocl, noop_b,
            KernelCaps::empty(), PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            aocl_layer1,
        );
        let mut set = AlternativeSet::from_candidates(
            vec![
                Candidate { backend: BackendId::Cpu, ..candidate_with_cost(noop_a, CostEstimate::default()) },
                Candidate { backend: BackendId::Aocl, ..candidate_with_cost(noop_b, CostEstimate::default()) },
            ],
            DEFAULT_MAX_N,
        );
        let lookup: HashMap<BackendId, BackendCapabilities> = [
            (BackendId::Cpu, caps_for_test(BackendId::Cpu)),
            (BackendId::Aocl, caps_for_test(BackendId::Aocl)),
        ].into_iter().collect();
        let sc = SizeClass::from_elem_count(4);
        let mut judge = crate::ranker::judge::HashMapJudge::new();
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Cpu, 500);
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Aocl, 100);

        compute_static_costs(
            &mut set, OpKind::AddElementwise, &dtypes,
            &[Shape::from(vec![4])], &bindings, &|b| lookup.get(&b), Some(&judge),
        );
        set.rank_by_composite_cost();
        assert_eq!(
            set.winner().unwrap().backend,
            BackendId::Aocl,
            "Layer-2 measurement reverses Layer-1 verdict",
        );
    }

    #[test]
    fn judge_partial_coverage_mixes_layer1_and_layer2() {
        // Two backends. Judge measured ONE of them; the other keeps
        // Layer-1. Ranking has to handle the mixed-shape cost.
        fn cheap(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 50, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        fn expensive(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 10_000, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise, &dtypes, BackendId::Cpu, noop_a,
            KernelCaps::empty(), PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, cheap,
        );
        bindings.register_full(
            OpKind::AddElementwise, &dtypes, BackendId::Aocl, noop_b,
            KernelCaps::empty(), PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, expensive,
        );
        let mut set = AlternativeSet::from_candidates(
            vec![
                Candidate { backend: BackendId::Cpu, ..candidate_with_cost(noop_a, CostEstimate::default()) },
                Candidate { backend: BackendId::Aocl, ..candidate_with_cost(noop_b, CostEstimate::default()) },
            ],
            DEFAULT_MAX_N,
        );
        let lookup: HashMap<BackendId, BackendCapabilities> = [
            (BackendId::Cpu, caps_for_test(BackendId::Cpu)),
            (BackendId::Aocl, caps_for_test(BackendId::Aocl)),
        ].into_iter().collect();
        let sc = SizeClass::from_elem_count(4);
        let mut judge = crate::ranker::judge::HashMapJudge::new();
        // Only measure Aocl (Judge said it's 20ns — way better than
        // Layer-1's 10000-FLOP estimate).
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Aocl, 20);

        compute_static_costs(
            &mut set, OpKind::AddElementwise, &dtypes,
            &[Shape::from(vec![4])], &bindings, &|b| lookup.get(&b), Some(&judge),
        );
        set.rank_by_composite_cost();
        // CPU = Layer-1 cost = 50 ns; Aocl = Layer-2 = 20 ns → Aocl wins.
        assert_eq!(
            set.winner().unwrap().backend, BackendId::Aocl,
            "partial Judge coverage still influences ranking",
        );
    }

    #[test]
    fn judge_saturates_above_u32_max_ns() {
        fn layer1(_: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities) -> CostEstimate {
            CostEstimate { flops: 1, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut bindings = KernelBindingTable::new();
        let dtypes = [DType::F32, DType::F32, DType::F32];
        bindings.register_full(
            OpKind::AddElementwise, &dtypes, BackendId::Cpu, noop_a,
            KernelCaps::empty(), PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, layer1,
        );
        let mut set = AlternativeSet::from_candidates(
            vec![candidate_with_cost(noop_a, CostEstimate::default())],
            DEFAULT_MAX_N,
        );
        let lookup: HashMap<BackendId, BackendCapabilities> =
            [(BackendId::Cpu, caps_for_test(BackendId::Cpu))].into_iter().collect();
        let mut judge = crate::ranker::judge::HashMapJudge::new();
        // Latency exceeds u32::MAX ns (~4.3 s).
        judge.insert(
            OpKind::AddElementwise, DType::F32,
            SizeClass::from_elem_count(4), BackendId::Cpu,
            u64::MAX,
        );
        compute_static_costs(
            &mut set, OpKind::AddElementwise, &dtypes,
            &[Shape::from(vec![4])], &bindings, &|b| lookup.get(&b), Some(&judge),
        );
        assert_eq!(
            set.alternatives()[0].static_cost.kernel_overhead_ns,
            u32::MAX,
            "u64 → u32 saturating cast pins at u32::MAX",
        );
    }
}
