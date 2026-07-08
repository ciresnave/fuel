//! Runtime-fused-op arm emission — the gated `Op::Branch` that makes an adopted
//! (JIT-synthesized / imported) fused op reachable, mirroring [`crate::decode_flash`].
//!
//! Same constitutional posture as the flash arm: **the optimizer emits/prunes arms;
//! backends never decide.** Arm 0 is the region's **existing primitive subgraph**
//! (the decompose / correctness oracle the graph already holds); arm 1 is the
//! synthesized `Op::Fused(runtime_id, Runtime)` kernel, offered **only** when a
//! kernel is bound for the target backend ([`fused_kernel_available`]) and pinned to
//! that backend. `finalize_branches` guarantees the merge reads arm 0, so a
//! finalized-but-unpicked graph — and every backend with no synthesized arm —
//! realizes on the primitive route, byte-identical. The route picker chooses arm 1
//! when it wins; the kernel resolves from the `FusedOpId`-keyed runtime sidecar.
//!
//! Unlike the flash arm (emitted at graph-build by the decode builder), a runtime
//! fused op is synthesized *after* load, so this arm is emitted during
//! **(re-)optimization** — a pathfinder scans the base map for regions matching an
//! adopted op and calls this per match. This module is the emitter; the pathfinder
//! that finds matches + splices reconverge points registers it into `optimize_graph`.

use fuel_graph::registry::{FusedOpId, FusedOpParams};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::Result;
use fuel_ir::probe::BackendId;

use crate::runtime_fused_kernels::fused_kernel_available;

/// A matched runtime-fused region the optimizer offers as an arm candidate. Arm 0
/// (`primitive_sink`, the region's existing output) is already in the graph; this
/// describes how to build arm 1 (the fused kernel) and where to splice the branch.
///
/// `inputs[0]` is the branch **diverge** point (the primitive chain and the fused
/// node both depart from it); the other inputs are shared external operands.
/// `primitive_sink` is arm 0's exit (same shape/dtype as the fused output).
/// `reconverge` is the sole consumer of `primitive_sink` (the merge).
#[derive(Clone, Debug)]
pub struct RuntimeFusedSpec {
    /// The adopted runtime fused op (`id >= RUNTIME_FUSED_BASE`).
    pub runtime_id: FusedOpId,
    /// The region's bound external inputs, in bind-index order (from `match_region`).
    pub inputs: Vec<NodeId>,
    /// Arm 0's exit — the region's existing primitive-subgraph output.
    pub primitive_sink: NodeId,
    /// The sole consumer of `primitive_sink` (the merge / reconverge point).
    pub reconverge: NodeId,
    /// The backend the synthesized kernel is bound for; arm 1 is pinned to it.
    pub backend: BackendId,
    /// The extracted scalar args for the fused op (the `extract:` slots); empty
    /// for a parameterless runtime op.
    pub scalars: Vec<f64>,
}

/// Offer a runtime-fused `Op::Branch` arm for an adopted op, gated on kernel
/// availability. Returns `Ok(Some(branch))` when the arm is emitted, `Ok(None)`
/// when the gate declines (no kernel for `backend` → no arm; the region stays on
/// arm 0's primitives). Never a realize-time decision — the capability gate and
/// the backend pin happen here, in the optimizer.
pub fn offer_runtime_fused_arm(
    graph: &mut Graph,
    spec: &RuntimeFusedSpec,
) -> Result<Option<NodeId>> {
    debug_assert!(spec.runtime_id.is_runtime(), "offer_runtime_fused_arm on a static id");

    // The capability gate: only offer the fused arm when a kernel is bound for
    // this backend. No kernel ⇒ no arm ⇒ the region stays on its primitives.
    if !fused_kernel_available(spec.runtime_id, spec.backend) {
        return Ok(None);
    }
    if spec.inputs.is_empty() {
        return Ok(None); // no diverge point; nothing to branch from.
    }

    // Arm 1 MUST equal arm 0's shape + dtype (finalize_branches' cast-to-uniform).
    let (shape, dtype) = {
        let n = graph.node(spec.primitive_sink);
        (n.shape.clone(), n.dtype)
    };
    let fused = graph.push(Node {
        op: Op::Fused(
            spec.runtime_id,
            FusedOpParams::Runtime { scalars: spec.scalars.clone() },
        ),
        inputs: spec.inputs.clone(),
        shape,
        dtype,
    });
    graph.set_target_backend(fused, spec.backend);

    // Emit the branch: arm 0 = the primitive subgraph (fallback/oracle), arm 1 =
    // the fused kernel. Diverge at inputs[0]; reconverge at the region's consumer.
    let mut builder = graph.open_branch(spec.inputs[0]);
    builder.add_arm(spec.primitive_sink); // arm 0 — the runnability fallback
    builder.add_arm(fused); // arm 1 — the synthesized kernel
    builder.finalize_branches(graph, spec.reconverge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_fused_kernels::adopt_runtime_fused;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_ir::{DType, Layout, Shape};
    use std::sync::{Arc, RwLock as StdRwLock};

    fn noop_kernel(
        _inputs: &[Arc<StdRwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<StdRwLock<fuel_memory::Storage>>],
        _layouts: &[Layout],
        _params: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }

    /// relu(add(a, b)) as a PatternNode region.
    fn relu_add_region() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    /// Build relu(add(a,b)) with a downstream consumer (neg) and return
    /// (graph, [a,b], relu_sink, neg_reconverge).
    fn graph_with_region() -> (Graph, Vec<NodeId>, NodeId, NodeId) {
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let leaf = |g: &mut Graph| {
            g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 })
        };
        let a = leaf(&mut g);
        let b = leaf(&mut g);
        let add = g.push(Node { op: Op::Add, inputs: vec![a, b], shape: s.clone(), dtype: DType::F32 });
        let relu = g.push(Node { op: Op::Relu, inputs: vec![add], shape: s.clone(), dtype: DType::F32 });
        let neg = g.push(Node { op: Op::Neg, inputs: vec![relu], shape: s.clone(), dtype: DType::F32 });
        (g, vec![a, b], relu, neg)
    }

    #[test]
    fn offers_a_gated_branch_when_a_kernel_is_bound() {
        let rid = adopt_runtime_fused(
            "test::arm::relu_add",
            relu_add_region(),
            noop_kernel as crate::kernel::KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("registrable region");

        let (mut g, inputs, relu, neg) = graph_with_region();
        let spec = RuntimeFusedSpec {
            runtime_id: rid,
            inputs,
            primitive_sink: relu,
            reconverge: neg,
            backend: BackendId::Cpu,
            scalars: vec![],
        };
        let branch = offer_runtime_fused_arm(&mut g, &spec).expect("valid branch").expect("emitted");
        assert!(matches!(g.node(branch).op, Op::Branch { .. }), "an Op::Branch decision point");
    }

    #[test]
    fn declines_when_no_kernel_for_the_backend() {
        let rid = adopt_runtime_fused(
            "test::arm::relu_add::cpu_only",
            relu_add_region(),
            noop_kernel as crate::kernel::KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("registrable region");

        let (mut g, inputs, relu, neg) = graph_with_region();
        // Kernel bound on Cpu, not Cuda → no arm for a Cuda request.
        let spec = RuntimeFusedSpec {
            runtime_id: rid,
            inputs,
            primitive_sink: relu,
            reconverge: neg,
            backend: BackendId::Cuda,
            scalars: vec![],
        };
        assert!(
            offer_runtime_fused_arm(&mut g, &spec).expect("no error").is_none(),
            "no kernel for Cuda ⇒ no fused arm (region stays primitive)",
        );
    }
}
