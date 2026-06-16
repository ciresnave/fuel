//! Load-time planning driver.
//!
//! Program anchor: `docs/session-prompts/plan-is-graph-rebuild.md`
//! (Phase D — load-time, input-independent graph build).
//!
//! [`Planner::warm`] is the explicit load-time entry: model loaders
//! (and user code that builds a graph ahead of realizing it) may call
//! it after graph construction to pre-optimize.
//!
//! ## Status post-PR-A3b-2
//!
//! The identity-keyed [`PlanStore`](fuel_dispatch) memoization layer
//! `warm` used to populate was deleted in PR-A3b-2 (the "plan IS the
//! graph" rebuild retires the separate `ExecutionPlan` source of truth
//! and the per-graph plan store). With the load-time-build model not
//! yet landed (Phase D), `warm` has nothing to memoize: the optimized
//! form lives **in the graph**, built once per realize by
//! `optimize_graph`, and there is no separate plan to stash and reuse.
//!
//! `warm` is therefore a **no-op** today, preserved as the seam Phase D
//! will reimplement (optimize the graph in place at load so the first
//! realize finds the multi-path form already written). The realize path
//! remains authoritative — it runs `optimize_graph` itself.

use std::sync::{Arc, RwLock};

use fuel_graph::{Graph, NodeId};
use fuel_core_types::Result;

use crate::Device;

/// Load-time planning driver. Stateless.
pub struct Planner;

impl Planner {
    /// Load-time pre-optimization hook for `(graph, device)`.
    ///
    /// **No-op as of PR-A3b-2**: the per-graph plan store this used to
    /// populate was deleted with the legacy `compile_plan`/`PlanStore`
    /// path. The "plan IS the graph" form is built once per realize by
    /// `optimize_graph` (in `fuel-core::pipelined_bridge`), so there is
    /// nothing to memoize ahead of the first realize until the
    /// load-time-build model (Phase D) lands. This hook is preserved as
    /// the seam Phase D will reimplement to optimize the graph in place
    /// at load.
    ///
    /// Always `Ok(())`. Callers (e.g. the decode loop in
    /// `fuel-core::lazy`) already treat `warm` as advisory and discard
    /// its result; the realize path runs the authoritative optimization.
    pub fn warm(
        _graph: &Arc<RwLock<Graph>>,
        _targets: &[NodeId],
        _device: &Device,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::{DType, Shape};
    use fuel_dispatch::pipelined::StorageCache;
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

    /// `warm` is a no-op (no store to populate post-A3b-2) but must
    /// stay error-free, and a realize after `warm` must still produce
    /// correct values — the realize path runs the authoritative
    /// optimization itself.
    #[test]
    fn warm_is_noop_and_realize_still_correct() {
        let device = crate::Device::cpu();
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (c1, c2, add) = {
            let mut g = graph.write().unwrap();
            let c1 = push_node(&mut g, Op::Const, vec![]);
            let c2 = push_node(&mut g, Op::Const, vec![]);
            let add = push_node(&mut g, Op::Add, vec![c1, c2]);
            (c1, c2, add)
        };

        Planner::warm(&graph, &[add], &device).expect("warm is a no-op Ok");

        let mut initial = StorageCache::new();
        initial.insert(c1, cpu_storage_f32(&[1.0, 2.0, 3.0, 4.0]));
        initial.insert(c2, cpu_storage_f32(&[10.0, 20.0, 30.0, 40.0]));
        let out = crate::pipelined_bridge::realize_one_as_with_initial::<f32>(
            &graph, add, &device, initial,
        )
        .expect("realize after warm");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Warm on empty targets is a no-op (no error).
    #[test]
    fn warm_empty_targets_is_noop() {
        let graph = Arc::new(RwLock::new(Graph::new()));
        Planner::warm(&graph, &[], &crate::Device::cpu()).expect("no-op");
    }
}
