//! Load-time planning driver — planner Stage 4a.
//!
//! Program anchor: `docs/session-prompts/load-time-incremental-
//! planner.md` (Stage 4) + architecture `04-optimization.md`
//! §Load-time incremental planning (v0.4). The decision (2026-06-11):
//! planning starts as soon as Fuel is told to load a model, and
//! `realize()` reduces to ensure-plan-coverage + dispatch.
//!
//! [`Planner::warm`] is the explicit load-time entry: model loaders
//! (and user code that builds a graph ahead of realizing it) call it
//! after graph construction to populate the per-graph plan store
//! ([`fuel_dispatch::plan_store::PlanStore`]) so the first realize
//! finds its planning half already done and skips plan-build (a
//! store HIT — observable via [`fuel_dispatch::plan_store::PlanStore::stats`]).
//!
//! ## v1 scope + explicit 4b deferrals
//!
//! - **Synchronous-incremental**: `warm` plans on the calling thread.
//!   A background planning thread that chases graph-construction
//!   events (so kernels, weight page-in, and planning overlap) is
//!   Stage 4b.
//! - **Residency-blind**: `warm` plans with an empty residency view
//!   (no const cache exists yet at load time). On a single-device
//!   host the plan is identical to the realize-time one; on
//!   multi-device hosts inputs resident OFF the pinned device would
//!   price differently — those slots don't exist before the first
//!   realize, so the drift window is empty today. Residency-aware
//!   warm (threading an `InferenceContext`'s persistent slots) is
//!   4b.
//! - **Same-graph reuse only**: the store keys on graph identity, so
//!   decode patterns that build a fresh graph per step (the
//!   kv-context forward family) get their warm benefit per step —
//!   plan once before realize — but not across steps. Cross-graph
//!   structural-hash memoization is Stage 5.
//!
//! ## Commit horizon (v1)
//!
//! A realize that began dispatching holds an immutable
//! `Arc<ExecutionPlan>` snapshot; `warm` (or any concurrent
//! plan-store update) never disturbs it. Revisions land for the
//! NEXT realize. Mid-realize ahead-of-frontier swap is 4b.

use std::sync::{Arc, RwLock};

use fuel_graph::{Graph, NodeId};
use fuel_core_types::Result;
use fuel_dispatch::pipelined::StorageCache;

use crate::Device;

/// Load-time planning driver. Stateless — the plan state lives in
/// the process-wide [`fuel_dispatch::plan_store::PlanStore`].
pub struct Planner;

impl Planner {
    /// Populate the plan store for `(graph, device)` ahead of the
    /// first realize: plans every kernel-bearing node reachable from
    /// `targets` (enumeration + filter + cost rank + placement DP —
    /// the full `compile_plan` pipeline the realize path runs) and
    /// stores the result keyed by graph identity + device + topology
    /// generation.
    ///
    /// The subsequent `realize_*` call on the same graph + device
    /// reuses the stored plan (its realize-root `Op::Copy` splice
    /// needs no plan entry, so coverage holds) — plan-build is
    /// skipped entirely. Graph growth after `warm` (more nodes
    /// appended before realizing) downgrades gracefully to an
    /// incremental extension: only the delta is planned.
    ///
    /// Errors are the same typed plan-time failures realize would
    /// surface (`NoBackendForOp`, missing device context, lock
    /// poisoning). Callers treating warm as advisory may discard the
    /// error — the realize path is authoritative and will re-surface
    /// any genuine failure with full context.
    ///
    /// Empty `targets` is a no-op.
    pub fn warm(
        graph: &Arc<RwLock<Graph>>,
        targets: &[NodeId],
        device: &Device,
    ) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }
        crate::pipelined_bridge::build_execution_plan(
            graph,
            targets,
            device.location(),
            &StorageCache::new(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::{DType, Shape};
    use fuel_dispatch::plan_store::{PlanStore, PlanStoreStats};
    use fuel_graph::{Node, Op};
    use fuel_memory::{BackendStorage, Storage};

    fn push_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        })
    }

    fn cpu_storage_f32(vals: &[f32]) -> Arc<RwLock<Storage>> {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cpu(
                fuel_cpu_backend::byte_storage::CpuStorageBytes::from_bytes(bytes),
            ),
            DType::F32,
        )))
    }

    fn stats_for(graph: &Arc<RwLock<Graph>>) -> PlanStoreStats {
        PlanStore::global().stats(graph).unwrap_or_default()
    }

    /// Concurrency note shared by the tests below: the fuel-core
    /// suite's topology tests bump the global generation counter
    /// while running (`SystemTopology::refresh`), which legitimately
    /// invalidates store entries mid-test. Each test therefore
    /// retries its scenario on a FRESH graph whenever an
    /// invalidation (or unexpected miss) landed inside its window,
    /// and only asserts on a churn-free pass.
    const CHURN_RETRIES: usize = 16;

    /// Stage 4a item 4: a warm()-then-realize sequence serves the
    /// realize's planning half from the store — the first realize
    /// after warm skips plan-build (HIT), and plans zero additional
    /// nodes.
    #[test]
    fn warm_then_first_realize_skips_plan_build() {
        let device = crate::Device::cpu();
        for _ in 0..CHURN_RETRIES {
            let graph = Arc::new(RwLock::new(Graph::new()));
            let (c1, c2, add) = {
                let mut g = graph.write().unwrap();
                let c1 = push_node(&mut g, Op::Const, vec![]);
                let c2 = push_node(&mut g, Op::Const, vec![]);
                let add = push_node(&mut g, Op::Add, vec![c1, c2]);
                (c1, c2, add)
            };

            Planner::warm(&graph, &[add], &device).expect("warm plans the graph");
            let warmed = stats_for(&graph);
            assert_eq!(warmed.misses, 1, "warm itself is the store's miss");
            assert_eq!(warmed.hits, 0);

            let mut initial = fuel_dispatch::pipelined::StorageCache::new();
            initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
            initial.insert(c2, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));
            let out = crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                &graph, add, &device, initial,
            )
            .expect("realize after warm");
            assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);

            let after = stats_for(&graph);
            if after.invalidations > warmed.invalidations
                || after.misses > warmed.misses
            {
                // A concurrent topology-generation bump invalidated
                // the warmed plan mid-test; retry on a fresh graph.
                continue;
            }
            assert_eq!(
                after.hits,
                warmed.hits + 1,
                "first realize after warm() must HIT the store",
            );
            assert_eq!(
                after.nodes_planned, warmed.nodes_planned,
                "realize planned zero nodes — warm covered everything \
                 (the realize-root Op::Copy splice needs no plan entry)",
            );
            return;
        }
        panic!(
            "topology generation churned through every retry window — \
             the suite's churn test should settle well within this budget"
        );
    }

    /// Stage 4a item 1: repeat realizes on an unchanged graph reuse
    /// the stored plan (decode-loop shape) — the second realize is a
    /// HIT even though prepare() appended its own root splice.
    #[test]
    fn repeat_realize_hits_store() {
        let device = crate::Device::cpu();
        for _ in 0..CHURN_RETRIES {
            let graph = Arc::new(RwLock::new(Graph::new()));
            let (c1, neg) = {
                let mut g = graph.write().unwrap();
                let c1 = push_node(&mut g, Op::Const, vec![]);
                let neg = push_node(&mut g, Op::Neg, vec![c1]);
                (c1, neg)
            };
            let realize = |g: &Arc<RwLock<Graph>>| {
                let mut initial = fuel_dispatch::pipelined::StorageCache::new();
                initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
                crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                    g, neg, &device, initial,
                )
            };

            let out1 = realize(&graph).expect("first realize");
            assert_eq!(out1, vec![-1.0, -2.0, -3.0, -4.0]);
            let s1 = stats_for(&graph);

            let out2 = realize(&graph).expect("second realize");
            assert_eq!(out2, out1);
            let s2 = stats_for(&graph);

            if s2.invalidations > s1.invalidations || s2.misses > s1.misses {
                continue; // concurrent generation bump — retry fresh
            }
            assert_eq!(s2.hits, s1.hits + 1, "repeat realize HITs");
            assert_eq!(
                s2.nodes_planned, s1.nodes_planned,
                "repeat realize planned nothing new",
            );
            return;
        }
        panic!("topology generation churned through every retry window");
    }

    /// Stage 4a item 1 (growing frontier): a decode-loop-shaped
    /// graph that appends nodes between realizes extends the stored
    /// plan incrementally — only the appended delta is planned.
    #[test]
    fn decode_loop_growth_extends_delta_only() {
        let device = crate::Device::cpu();
        for _ in 0..CHURN_RETRIES {
            let graph = Arc::new(RwLock::new(Graph::new()));
            let (c1, tip) = {
                let mut g = graph.write().unwrap();
                let c1 = push_node(&mut g, Op::Const, vec![]);
                let mut tip = c1;
                for _ in 0..4 {
                    tip = push_node(&mut g, Op::Neg, vec![tip]);
                }
                (c1, tip)
            };
            let realize = |g: &Arc<RwLock<Graph>>, t: NodeId| {
                let mut initial = fuel_dispatch::pipelined::StorageCache::new();
                initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
                crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
                    g, t, &device, initial,
                )
            };

            realize(&graph, tip).expect("prefill realize");
            let s1 = stats_for(&graph);
            assert_eq!(s1.nodes_planned, 4, "prefill planned the 4-node chain");

            // "Next token": append 3 more compute nodes.
            let new_tip = {
                let mut g = graph.write().unwrap();
                let a = push_node(&mut g, Op::Neg, vec![tip]);
                let b = push_node(&mut g, Op::Sqr, vec![a]);
                push_node(&mut g, Op::Neg, vec![b])
            };
            realize(&graph, new_tip).expect("decode-step realize");
            let s2 = stats_for(&graph);

            if s2.invalidations > s1.invalidations || s2.misses > s1.misses {
                continue; // concurrent generation bump — retry fresh
            }
            assert_eq!(s2.extensions, s1.extensions + 1, "growth → extension");
            assert_eq!(
                s2.nodes_planned,
                s1.nodes_planned + 3,
                "extension planned exactly the 3-node delta, not the prefix",
            );
            return;
        }
        panic!("topology generation churned through every retry window");
    }

    /// Warm on empty targets is a no-op (no store entry, no error).
    #[test]
    fn warm_empty_targets_is_noop() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        Planner::warm(&graph, &[], &crate::Device::cpu()).expect("no-op");
        assert!(PlanStore::global().stats(&graph).is_none());
    }

    /// Stage 4a item 5 measurement: per-token plan overhead on a
    /// synthetic 100-node-per-step decode-loop graph, store-on vs
    /// store-off. Ignored — a manual measurement, not a gate; run
    /// with `cargo test -p fuel-core --lib --release -- --ignored
    /// measure_decode_loop_plan_overhead --nocapture`.
    #[test]
    #[ignore = "measurement, not a gate — run manually with --nocapture"]
    fn measure_decode_loop_plan_overhead() {
        use std::time::Instant;
        const NODES_PER_STEP: usize = 100;
        const STEPS: usize = 30;

        let build_step = |graph: &Arc<RwLock<Graph>>, prev: NodeId| -> NodeId {
            let mut g = graph.write().unwrap();
            let mut tip = prev;
            for i in 0..NODES_PER_STEP {
                let op = if i % 2 == 0 { Op::Neg } else { Op::Sqr };
                tip = push_node(&mut g, op, vec![tip]);
            }
            tip
        };
        let order_for = |graph: &Arc<RwLock<Graph>>, tip: NodeId| {
            let g = graph.read().unwrap();
            fuel_graph::topo_order_multi(&g, &[tip])
        };
        let cache = StorageCache::new();
        let loc = fuel_core_types::DeviceLocation::Cpu;

        // Store ON: private store, plan_for with reuse-extension.
        let store = PlanStore::new();
        let graph_on = Arc::new(RwLock::new(Graph::new()));
        let mut tip_on = {
            let mut g = graph_on.write().unwrap();
            push_node(&mut g, Op::Const, vec![])
        };
        let t_on = Instant::now();
        for _ in 0..STEPS {
            tip_on = build_step(&graph_on, tip_on);
            let order = order_for(&graph_on, tip_on);
            store
                .plan_for(&graph_on, loc, &order, &mut |base| {
                    crate::pipelined_bridge::compile_bridge_plan(
                        &graph_on, &order, loc, &cache, base,
                    )
                })
                .expect("store-on plan");
        }
        let on = t_on.elapsed();

        // Store OFF: full compile per step (pre-Stage-4 behavior).
        let graph_off = Arc::new(RwLock::new(Graph::new()));
        let mut tip_off = {
            let mut g = graph_off.write().unwrap();
            push_node(&mut g, Op::Const, vec![])
        };
        let t_off = Instant::now();
        for _ in 0..STEPS {
            tip_off = build_step(&graph_off, tip_off);
            let order = order_for(&graph_off, tip_off);
            crate::pipelined_bridge::compile_bridge_plan(
                &graph_off, &order, loc, &cache, None,
            )
            .expect("store-off plan");
        }
        let off = t_off.elapsed();

        let per_tok_on = on.as_secs_f64() * 1e6 / STEPS as f64;
        let per_tok_off = off.as_secs_f64() * 1e6 / STEPS as f64;
        println!(
            "decode-loop plan overhead ({NODES_PER_STEP} nodes/step, \
             {STEPS} steps): store-on {per_tok_on:.1} µs/token, \
             store-off {per_tok_off:.1} µs/token ({:.1}x)",
            per_tok_off / per_tok_on,
        );
        assert!(
            on <= off,
            "store-on planning must not be slower than full replans \
             (on {per_tok_on:.1} µs/token vs off {per_tok_off:.1} µs/token)",
        );
    }
}
