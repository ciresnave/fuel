//! Per-graph execution-plan store — planner Stage 4a.
//!
//! Program anchor: `docs/session-prompts/load-time-incremental-
//! planner.md` (Stage 4) + architecture `04-optimization.md`
//! §Load-time incremental planning (v0.4). The 2026-06-11 decision:
//! planning moves out of `realize()` — `realize()` reduces to
//! *ensure plan coverage of the requested roots, then dispatch*.
//! This module is the memoization half of that move: the planning
//! work `prepare()` used to redo per realize call is keyed and
//! reused, and decode-loop graph growth replans only the delta.
//!
//! ## Keying + invalidation
//!
//! Plans key on **(graph identity, pinned device)** and validate
//! against the **topology generation**:
//!
//! - *Graph identity* is the `Arc<RwLock<Graph>>` allocation
//!   (pointer + a stored `Weak` re-checked by `Arc::ptr_eq` so a
//!   reallocated address can never resurrect a dead graph's plans).
//! - *Pinned device* is the realize call's `DeviceLocation` — the
//!   same graph realized on CPU and CUDA holds two independent
//!   plans.
//! - *Topology generation*: a stored plan whose
//!   [`ExecutionPlan::generation`] no longer matches
//!   [`crate::dispatch::topology_generation`] is stale — the lookup
//!   treats it as a miss (full rebuild) and counts an invalidation.
//!   This is the same signal the executor's Phase-4.3 dispatch-chunk
//!   check and the bridge's `TopologyChanged` retry already key on.
//!
//! ## Hit / extension / miss
//!
//! Coverage is judged per realize against the realize's own topo
//! order: a node *needs* a plan entry exactly when
//! [`crate::plan::node_needs_plan_entry`] says so (the same gate
//! `compile_plan` uses — the two cannot drift). Three outcomes:
//!
//! - **Hit** — every needs-entry node in the order is covered by the
//!   stored plan. No planning work runs; the stored `Arc` is
//!   returned. Repeat realizes of an unchanged graph (decode-loop
//!   steps whose only new node is the realize-root `Op::Copy`
//!   splice, which needs no entry) land here.
//! - **Extension** — the graph grew (nodes appended for the next
//!   decode step): the build callback runs `compile_plan` with
//!   [`crate::plan::PlanOptions::reuse_plan`] = the stored base, so
//!   covered nodes clone their sets and only the delta enumerates /
//!   filters / ranks.
//! - **Miss** — no stored plan (or stale generation): full build.
//!
//! Graph *mutation behind the frontier* (a shrink — impossible for
//! Fuel's append-only graphs; indicates identity confusion) is a
//! typed error, never a panic.
//!
//! ## Commit horizon (v1)
//!
//! Once a realize's dispatch begins, the plan in use is immutable —
//! the executor holds its own `Arc<ExecutionPlan>` snapshot, so a
//! concurrent store update (a revised plan from another thread or a
//! later `Planner::warm`) never disturbs an in-flight realize; the
//! NEXT realize picks the revision up. True mid-realize ahead-of-
//! frontier swap (the planner revising chunks the executor hasn't
//! reached) is Stage 4b — explicitly deferred.
//!
//! ## Known staleness (documented, accepted for 4a)
//!
//! - Judge measurements landing between builds don't invalidate a
//!   stored plan (the Judge doesn't bump the topology generation).
//!   The production runtime selector (JudgeAware) re-queries live
//!   profile data at dispatch time, so picks still improve within
//!   the stored top-N; only the top-N membership itself is frozen
//!   until the next rebuild.
//! - Reused candidates keep their original inbound-transfer terms;
//!   a residency shift between realizes is repriced only for delta
//!   nodes. The bridge's residency-stitch passes run per realize
//!   against actual storage locations, so execution correctness
//!   never depends on plan-time pricing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use fuel_core_types::{DeviceLocation, Error, Result};
use fuel_graph::{Graph, NodeId};

use crate::plan::{node_needs_plan_entry, ExecutionPlan};

/// Per-graph store counters. Snapshot via [`PlanStore::stats`];
/// counters are cumulative for the graph's lifetime in the store.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlanStoreStats {
    /// Lookups fully served by a stored plan (zero planning work).
    pub hits: u64,
    /// Full builds with no reusable base (first realize on this
    /// (graph, device), or after a generation invalidation).
    pub misses: u64,
    /// Incremental builds — a base existed and only the delta was
    /// planned (`PlanOptions::reuse_plan` path).
    pub extensions: u64,
    /// Stored plans discarded because the topology generation moved.
    pub invalidations: u64,
    /// Total needs-entry nodes actually planned (misses count the
    /// full order's needs-entry nodes; extensions count only the
    /// delta). The decode-loop test asserts this grows by the delta,
    /// not the prefix.
    pub nodes_planned: u64,
}

/// One stored plan for a (graph, device) pairing.
struct StoredPlan {
    plan: Arc<ExecutionPlan>,
    /// `Graph::len()` observed when the plan was stored — the plan
    /// frontier. Graphs are append-only, so a later observation
    /// below this value means the "same" graph identity no longer
    /// refers to the graph that was planned (typed error).
    planned_len: usize,
}

/// All store state for one graph identity.
struct GraphEntry {
    /// Re-validated by `Arc::ptr_eq` on every lookup — a dead graph
    /// whose allocation address was reused can never serve plans.
    graph: Weak<RwLock<Graph>>,
    plans: HashMap<DeviceLocation, StoredPlan>,
    stats: PlanStoreStats,
}

/// Sweep threshold: when the store map exceeds this many graph
/// entries on insert, dead-`Weak` entries are retained out.
/// Decode patterns that build a fresh graph per step (the
/// kv-context family) churn entries; the sweep keeps the map
/// bounded by the number of LIVE graphs plus this slack.
const SWEEP_THRESHOLD: usize = 32;

/// The per-graph execution-plan store. One process-wide instance
/// ([`PlanStore::global`]) serves production; tests may build their
/// own with [`PlanStore::new`] for isolation.
pub struct PlanStore {
    inner: Mutex<HashMap<usize, GraphEntry>>,
}

impl PlanStore {
    /// Fresh, empty store.
    pub fn new() -> Self {
        Self { inner: Mutex::new(HashMap::new()) }
    }

    /// Process-wide store used by the production realize path
    /// (`fuel-core::pipelined_bridge`).
    pub fn global() -> &'static PlanStore {
        static GLOBAL: OnceLock<PlanStore> = OnceLock::new();
        GLOBAL.get_or_init(PlanStore::new)
    }

    /// Resolve a plan covering `order` for `(graph, device)`,
    /// building (fully or incrementally) via `build` only when the
    /// store can't serve the lookup.
    ///
    /// `order` is the realize's own topo order (the same nodes the
    /// build callback will hand `compile_plan`). `build` receives
    /// `Some(base)` on the extension path — callers thread it into
    /// [`crate::plan::PlanOptions::with_reuse_plan`] — and `None`
    /// on a full miss.
    ///
    /// Locking: the store mutex is never held across `build` or any
    /// graph lock; the graph read lock is taken only for the
    /// coverage walk.
    pub fn plan_for(
        &self,
        graph: &Arc<RwLock<Graph>>,
        device: DeviceLocation,
        order: &[NodeId],
        build: &mut dyn FnMut(Option<&Arc<ExecutionPlan>>) -> Result<ExecutionPlan>,
    ) -> Result<Arc<ExecutionPlan>> {
        let key = Arc::as_ptr(graph) as usize;
        let current_gen = crate::dispatch::topology_generation();

        // Phase 1 (store mutex, brief): fetch the candidate base.
        let (base, invalidated) = {
            let mut store = self
                .inner
                .lock()
                .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
            match store.get(&key) {
                Some(entry)
                    if entry
                        .graph
                        .upgrade()
                        .is_some_and(|g| Arc::ptr_eq(&g, graph)) =>
                {
                    match entry.plans.get(&device) {
                        Some(sp) if sp.plan.generation == current_gen => {
                            (Some((Arc::clone(&sp.plan), sp.planned_len)), false)
                        }
                        Some(_) => (None, true),
                        None => (None, false),
                    }
                }
                Some(_) => {
                    // Dead graph or a reallocated address (ABA):
                    // drop the stale entry; this lookup is a miss.
                    store.remove(&key);
                    (None, false)
                }
                None => (None, false),
            }
        };

        // Phase 2 (graph read lock): frontier sanity + coverage
        // delta against the base.
        let (graph_len, delta) = {
            let g = graph
                .read()
                .map_err(|_| Error::Msg("graph lock poisoned".into()).bt())?;
            let len = g.len();
            if let Some((_, planned_len)) = &base {
                if len < *planned_len {
                    return Err(Error::Msg(format!(
                        "plan store: graph shrank behind the plan frontier \
                         (len {len} < planned {planned_len}) — graphs are \
                         append-only, so a stored plan for this identity \
                         refers to a different graph state. This is a bug \
                         in graph-identity management, not a recoverable \
                         condition.",
                    ))
                    .bt());
                }
            }
            let mut delta = 0usize;
            for &id in order {
                if !node_needs_plan_entry(&g.node(id).op) {
                    continue;
                }
                let covered = base
                    .as_ref()
                    .is_some_and(|(plan, _)| plan.alternatives(id).is_some());
                if !covered {
                    delta += 1;
                }
            }
            (len, delta)
        };

        // Phase 3: hit fast-path — zero planning work.
        if let Some((plan, _)) = &base {
            if delta == 0 {
                let mut store = self
                    .inner
                    .lock()
                    .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
                if let Some(entry) = store.get_mut(&key) {
                    entry.stats.hits += 1;
                }
                return Ok(Arc::clone(plan));
            }
        }

        // Phase 4: build (full or incremental) with no locks held.
        let built = Arc::new(build(base.as_ref().map(|(p, _)| p))?);

        // Phase 5: store + counters.
        let mut store = self
            .inner
            .lock()
            .map_err(|_| Error::Msg("plan store mutex poisoned".into()).bt())?;
        if store.len() > SWEEP_THRESHOLD {
            store.retain(|_, e| e.graph.strong_count() > 0);
        }
        let entry = store.entry(key).or_insert_with(|| GraphEntry {
            graph: Arc::downgrade(graph),
            plans: HashMap::new(),
            stats: PlanStoreStats::default(),
        });
        if entry
            .graph
            .upgrade()
            .map_or(true, |g| !Arc::ptr_eq(&g, graph))
        {
            // The entry belonged to a different (dead) graph that
            // shared this address — reset it for the live one.
            entry.graph = Arc::downgrade(graph);
            entry.plans.clear();
            entry.stats = PlanStoreStats::default();
        }
        if invalidated {
            entry.stats.invalidations += 1;
        }
        if base.is_some() {
            entry.stats.extensions += 1;
        } else {
            entry.stats.misses += 1;
        }
        entry.stats.nodes_planned += delta as u64;
        entry.plans.insert(
            device,
            StoredPlan { plan: Arc::clone(&built), planned_len: graph_len },
        );
        Ok(built)
    }

    /// Snapshot the cumulative counters for `graph`. `None` when the
    /// store has never planned for this graph (or its entry was
    /// swept/reset).
    pub fn stats(&self, graph: &Arc<RwLock<Graph>>) -> Option<PlanStoreStats> {
        let key = Arc::as_ptr(graph) as usize;
        let store = self.inner.lock().ok()?;
        let entry = store.get(&key)?;
        entry
            .graph
            .upgrade()
            .is_some_and(|g| Arc::ptr_eq(&g, graph))
            .then_some(entry.stats)
    }
}

impl Default for PlanStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranker::{AlternativeSet, Candidate};
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, Shape};
    use fuel_graph::{topo_order_multi, Node, Op};

    fn push_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        })
    }

    fn noop_kernel(
        _i: &[Arc<RwLock<fuel_storage::Storage>>],
        _o: &mut [Arc<RwLock<fuel_storage::Storage>>],
        _l: &[fuel_core_types::Layout],
        _p: &crate::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate() -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: crate::kernel::KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: crate::fused::PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: Default::default(),
            inbound_transfer_ns: 0,
            op_params: crate::kernel::OpParams::None,
            coupling: Vec::new(),
            kernel_source: "test",
        }
    }

    /// Synthetic builder: covers every needs-entry node in `order`,
    /// reusing base sets where present (mirrors the
    /// `compile_plan` + `reuse_plan` contract without the binding
    /// table).
    fn build_covering(
        graph: &Arc<RwLock<Graph>>,
        order: &[NodeId],
        base: Option<&Arc<ExecutionPlan>>,
    ) -> Result<ExecutionPlan> {
        let g = graph.read().unwrap();
        let mut alternatives = HashMap::new();
        for &id in order {
            if !node_needs_plan_entry(&g.node(id).op) {
                continue;
            }
            if let Some(set) = base.and_then(|b| b.alternatives(id)) {
                alternatives.insert(id, set.clone());
                continue;
            }
            let mut set = AlternativeSet::empty();
            set.push(candidate());
            alternatives.insert(id, set);
        }
        Ok(ExecutionPlan {
            order: order.to_vec(),
            alternatives,
            generation: crate::dispatch::topology_generation(),
        })
    }

    /// Chain graph: Const → Neg → Neg → ... (n compute nodes).
    fn chain_graph(n: usize) -> (Arc<RwLock<Graph>>, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let tip = {
            let mut g = graph.write().unwrap();
            let mut prev = push_node(&mut g, Op::Const, vec![]);
            for _ in 0..n {
                prev = push_node(&mut g, Op::Neg, vec![prev]);
            }
            prev
        };
        (graph, tip)
    }

    fn order_for(graph: &Arc<RwLock<Graph>>, tip: NodeId) -> Vec<NodeId> {
        let g = graph.read().unwrap();
        topo_order_multi(&g, &[tip])
    }

    #[test]
    fn miss_then_hit_same_graph_device() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(3);
        let order = order_for(&graph, tip);

        let mut builds = 0usize;
        let p1 = store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                builds += 1;
                assert!(base.is_none(), "first lookup is a full miss");
                build_covering(&graph, &order, base)
            })
            .unwrap();
        assert_eq!(builds, 1);

        let p2 = store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |_| {
                panic!("second lookup must be a hit — build must not run")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&p1, &p2), "hit returns the stored Arc");

        let stats = store.stats(&graph).expect("entry exists");
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.extensions, 0);
        assert_eq!(stats.nodes_planned, 3, "three Neg nodes planned once");
    }

    #[test]
    fn growth_extends_with_base_and_plans_delta_only() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(4);
        let order = order_for(&graph, tip);
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                build_covering(&graph, &order, base)
            })
            .unwrap();

        // Decode step: append 2 more compute nodes.
        let new_tip = {
            let mut g = graph.write().unwrap();
            let a = push_node(&mut g, Op::Neg, vec![tip]);
            push_node(&mut g, Op::Sqr, vec![a])
        };
        let order2 = order_for(&graph, new_tip);

        let mut saw_base = false;
        let p2 = store
            .plan_for(&graph, DeviceLocation::Cpu, &order2, &mut |base| {
                saw_base = base.is_some();
                build_covering(&graph, &order2, base)
            })
            .unwrap();
        assert!(saw_base, "growth path threads the stored base for reuse");
        assert!(
            p2.alternatives(new_tip).is_some(),
            "extended plan covers the appended tip",
        );

        let stats = store.stats(&graph).unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.extensions, 1);
        assert_eq!(
            stats.nodes_planned,
            4 + 2,
            "extension planned the 2-node delta, not the 4-node prefix again",
        );
    }

    #[test]
    fn device_change_is_an_independent_miss() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(2);
        let order = order_for(&graph, tip);
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                build_covering(&graph, &order, base)
            })
            .unwrap();

        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut base_seen: Option<bool> = None;
        store
            .plan_for(&graph, cuda0, &order, &mut |base| {
                base_seen = Some(base.is_some());
                build_covering(&graph, &order, base)
            })
            .unwrap();
        assert_eq!(
            base_seen,
            Some(false),
            "a different device key never reuses another device's plan",
        );
        let stats = store.stats(&graph).unwrap();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.invalidations, 0);

        // Both keys now serve hits independently.
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |_| {
                panic!("CPU key must hit")
            })
            .unwrap();
        store
            .plan_for(&graph, cuda0, &order, &mut |_| panic!("CUDA key must hit"))
            .unwrap();
        assert_eq!(store.stats(&graph).unwrap().hits, 2);
    }

    #[test]
    fn generation_bump_invalidates_to_full_rebuild() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(2);
        let order = order_for(&graph, tip);
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                build_covering(&graph, &order, base)
            })
            .unwrap();

        crate::dispatch::bump_topology_generation();

        let mut base_seen: Option<bool> = None;
        store
            .plan_for(&graph, DeviceLocation::Cpu, &order, &mut |base| {
                base_seen = Some(base.is_some());
                build_covering(&graph, &order, base)
            })
            .unwrap();
        assert_eq!(
            base_seen,
            Some(false),
            "stale generation must NOT be offered as a reuse base — the \
             Phase-4.3 contract is a full rebuild against the fresh topology",
        );
        let stats = store.stats(&graph).unwrap();
        assert_eq!(stats.invalidations, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn distinct_graphs_have_independent_entries() {
        let store = PlanStore::new();
        let (g1, t1) = chain_graph(2);
        let (g2, t2) = chain_graph(2);
        let o1 = order_for(&g1, t1);
        let o2 = order_for(&g2, t2);
        store
            .plan_for(&g1, DeviceLocation::Cpu, &o1, &mut |b| {
                build_covering(&g1, &o1, b)
            })
            .unwrap();
        store
            .plan_for(&g2, DeviceLocation::Cpu, &o2, &mut |b| {
                build_covering(&g2, &o2, b)
            })
            .unwrap();
        assert_eq!(store.stats(&g1).unwrap().misses, 1);
        assert_eq!(store.stats(&g2).unwrap().misses, 1);
        assert_eq!(store.stats(&g1).unwrap().hits, 0);
    }

    #[test]
    fn stats_none_for_unplanned_graph() {
        let store = PlanStore::new();
        let (graph, _) = chain_graph(1);
        assert!(store.stats(&graph).is_none());
    }

    #[test]
    fn build_error_propagates_and_stores_nothing() {
        let store = PlanStore::new();
        let (graph, tip) = chain_graph(1);
        let order = order_for(&graph, tip);
        let err = store.plan_for(&graph, DeviceLocation::Cpu, &order, &mut |_| {
            Err(Error::Msg("synthetic build failure".into()).bt())
        });
        assert!(err.is_err());
        assert!(
            store.stats(&graph).is_none(),
            "failed builds must not leave a store entry behind",
        );
    }
}
