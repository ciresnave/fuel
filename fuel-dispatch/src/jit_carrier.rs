// SPDX-License-Identifier: MIT OR Apache-2.0
//! The JIT **carrier** — the trigger that turns a region no backend can run
//! into a [`JitRequest`].
//!
//! Fuel's JIT seam was complete end-to-end and **unrouted**: `JitRequest` /
//! [`crate::jit_adopt::adopt_from_response`] / `jit_cuda_load` /
//! `offer_runtime_fused_arm` all existed and were device-validated, but nothing
//! ever *constructed* a request from a real graph. `adopt_from_response` had no
//! non-test caller. This module is the missing half.
//!
//! ## What it decides
//!
//! Given a fused node and a target backend, ask the one question the JIT route
//! exists to answer:
//!
//! > This op has **no kernel on this device**. Can we hand its region to a
//! > synthesizer instead of shipping the tensors to a device that can run it?
//!
//! The alternative — lowering to primitives — requires **every** primitive in
//! the recipe to be bound on the target. Handing the region over requires
//! **none** of them. That asymmetry is the whole argument for the seam, and it
//! is why the trigger keys on the *fused* op's absence rather than on the
//! recipe's placeability.
//!
//! ## Why every decline is `None` rather than an error
//!
//! A carrier that errors would make "no kernel here" a failure, when it is the
//! ordinary case that the planner already handles by placing the node
//! elsewhere. Every reason to not synthesize is a typed `None` (G2 posture):
//! not fused, already runnable here, no recipe exposed, or an operand dtype
//! with no seam spelling. **Producing a request is the exception; declining is
//! the norm.**
//!
//! ## Measured context (2026-08-06)
//!
//! `Op::PagedAttn` is the motivating case and the reason this is not
//! speculative: **no GPU backend implements it**, so under CUDA the fused node
//! is host-placed (per-node `placement_of`) and the KV caches cross the bus
//! every token. Its recipe is 6-of-7 CUDA-resolvable — one registration away
//! from the primitive route working.
//!
//! **Known limitation, measured rather than assumed:** the real
//! `BaracudaSynthesizer` currently **declines** that region with
//! `JitError::MixedDtype`, because `StructureKey` carries one dtype slot and
//! `OperandKey` discards per-operand dtype at key derivation — so any *indexed*
//! region (float data + integer indices) is undescribable to it today. The
//! carrier is built and correct; its highest-value consumer is blocked upstream
//! on an identity-schema change (KISS RFC #29, sk4 per-operand dtype). Built
//! anyway because the trigger is the piece Fuel owns, and because the next
//! fused op with no backend implementation may not be indexed.

use baracuda_kernels_types::{ArchSku, OperandDesc};
use fuel_graph::{Graph, NodeId};
use fuel_ir::probe::BackendId;
use fuel_kernel_seam::{JitBudget, JitRequest};

use crate::jit_adopt::dtype_to_element_kind;
use crate::kernel::KernelBindingTable;

/// Build a [`JitRequest`] for `id` if — and only if — it is a fused node that
/// **cannot run on `backend`** and whose region can be described honestly.
///
/// Returns `None` (never an error) when any of these hold:
///
/// - `id` is not an `Op::Fused` node;
/// - a kernel for it **already exists** on `backend` — nothing to synthesize;
/// - the op does not expose its recipe as data
///   ([`fuel_graph::registry::recipe_for`]);
/// - any operand's `DType` has **no seam spelling**. Declining here is
///   deliberate: substituting a nearby `ElementKind` would ask the synthesizer
///   about a *different kernel* than the caller has. That is not hypothetical —
///   typing `PagedAttn`'s U32 index operands as I32 makes all six operands
///   uniform, which flips a `MixedDtype` decline into a false acceptance.
///
/// Operands are emitted in **bind order, then the output**, matching the seam's
/// `OpDef` convention (`n_inputs + 1` entries).
pub fn jit_request_for_unplaceable_fused(
    graph: &Graph,
    id: NodeId,
    backend: BackendId,
    table: &KernelBindingTable,
    arch: ArchSku,
    budget: JitBudget,
) -> Option<JitRequest> {
    let node = graph.node(id);

    // Only fused ops have a recipe to hand over.
    let fused_id = match &node.op {
        fuel_graph::Op::Fused(fid, _) => *fid,
        _ => return None,
    };

    // THE TRIGGER: synthesize only when this device cannot already run it.
    // `lookup` keys on (op, [input dtypes.., output dtype], backend) — the same
    // question the planner asks — so "runnable here" means exactly what it
    // means everywhere else.
    let mut key_dtypes: Vec<fuel_ir::DType> =
        node.inputs.iter().map(|&i| graph.node(i).dtype).collect();
    key_dtypes.push(node.dtype);
    if let Some(op_kind) =
        crate::runtime_fused_kernels::static_fused_id_to_binding_table_op_kind(fused_id)
        && table.lookup(op_kind, &key_dtypes, backend).is_ok()
    {
        return None; // already runnable here — the seam is for gaps
    }

    // The region, as data, without touching the graph.
    let region = fuel_graph::registry::recipe_for(graph, id)?;

    // Operands: inputs in bind order, then the output. Any dtype without a seam
    // spelling declines the whole request rather than being approximated.
    let mut operands: Vec<OperandDesc> = Vec::with_capacity(node.inputs.len() + 1);
    for &input in &node.inputs {
        operands.push(operand_desc(graph, input)?);
    }
    operands.push(operand_desc(graph, id)?);

    Some(JitRequest {
        region,
        operands,
        arch,
        budget,
    })
}

/// One node's shape/strides/dtype as an [`OperandDesc`]. `None` if the dtype has
/// no seam spelling — see the honest-typing note on the caller.
fn operand_desc(graph: &Graph, id: NodeId) -> Option<OperandDesc> {
    let node = graph.node(id);
    let dims = node.shape.dims();
    let ek = dtype_to_element_kind(node.dtype)?;

    // Row-major contiguous strides, innermost = 1.
    let mut strides = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as i64;
    }
    let shape: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    let align = node.dtype.size_in_bytes() as u32;
    Some(OperandDesc::new(dims.len(), &shape, &strides, ek, align))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_graph::registry::{FusedOpParams, FusedOps};
    use fuel_graph::{Node, Op};
    use fuel_ir::{DType, Shape};

    fn leaf(g: &mut Graph, dims: &[usize], dtype: DType) -> NodeId {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(dims),
            dtype,
        })
    }

    /// A well-formed `PagedAttn` node: F32 data operands, **U32 index operands**.
    fn paged_node(g: &mut Graph) -> (NodeId, FusedOpParams) {
        let q = leaf(g, &[1, 4, 1, 4], DType::F32);
        let kc = leaf(g, &[32, 4, 4, 4], DType::F32);
        let vc = leaf(g, &[32, 4, 4, 4], DType::F32);
        let bt = leaf(g, &[1, 8], DType::U32);
        let cl = leaf(g, &[1], DType::U32);
        let params = FusedOpParams::PagedAttn {
            softmax_scale: 0.5,
            block_size: 4,
            softcap: None,
        };
        let id = g.push(Node {
            op: Op::Fused(FusedOps::PAGED_ATTN, params.clone()),
            inputs: vec![q, kc, vc, bt, cl],
            shape: Shape::from_dims(&[1, 4, 1, 4]),
            dtype: DType::F32,
        });
        (id, params)
    }

    /// **The carrier produces a request for a fused op no backend can run** —
    /// the case the whole seam exists for.
    ///
    /// Asserts the request is *well-formed for the seam*, not merely non-`None`:
    /// `n_inputs + 1` operands in bind-then-output order, and — the load-bearing
    /// part — the index operands typed **U32**, not silently widened. Typing
    /// them as I32 would make all six operands uniform, which converts a real
    /// `MixedDtype` decline into a false acceptance. That is not a hypothetical
    /// failure mode; it is the one this test exists to prevent.
    #[test]
    fn builds_a_request_for_a_fused_op_with_no_kernel_here() {
        use baracuda_kernels_types::ElementKind;

        let mut g = Graph::new();
        let (id, _) = paged_node(&mut g);

        // An EMPTY table: nothing is runnable on Cuda, so the trigger fires.
        let table = KernelBindingTable::new();
        let req = jit_request_for_unplaceable_fused(
            &g,
            id,
            BackendId::Cuda,
            &table,
            ArchSku::Sm89,
            JitBudget {
                max_compile_ms: 1_000,
            },
        )
        .expect("a fused op with no kernel here yields a request");

        assert_eq!(
            req.operands.len(),
            6,
            "5 inputs + 1 output, bind order then output"
        );
        assert_eq!(req.operands[0].dtype, ElementKind::F32, "q is F32");
        assert_eq!(
            req.operands[3].dtype,
            ElementKind::U32,
            "block_table must be U32 — widening it to I32 makes every operand \
             uniform and turns a real MixedDtype decline into a false accept",
        );
        assert_eq!(
            req.operands[4].dtype,
            ElementKind::U32,
            "context_lens must be U32"
        );
        assert_eq!(req.operands[5].dtype, ElementKind::F32, "output is F32");
        assert!(
            matches!(req.region, fuel_graph::jit::PatternNode::Op { .. }),
            "the region must be a real Op tree, not a bare bind",
        );
    }

    /// **The trigger is a trigger, not a firehose: a non-fused node declines.**
    ///
    /// Without this, a carrier that returned `Some` for everything would satisfy
    /// the test above — it asserts the request is well-formed, not that it was
    /// warranted.
    #[test]
    fn declines_a_non_fused_node() {
        let mut g = Graph::new();
        let c = leaf(&mut g, &[4], DType::F32);
        let table = KernelBindingTable::new();
        assert!(
            jit_request_for_unplaceable_fused(
                &g,
                c,
                BackendId::Cuda,
                &table,
                ArchSku::Sm89,
                JitBudget {
                    max_compile_ms: 1_000
                },
            )
            .is_none(),
            "only fused ops have a region to hand over",
        );
    }

    /// **A dtype with no seam spelling declines the WHOLE request.**
    ///
    /// The alternative — substituting a nearby `ElementKind` — asks the
    /// synthesizer about a different kernel than the caller has. `I16` is real
    /// in Fuel and has no seam spelling, so it is the honest probe for this.
    #[test]
    fn declines_rather_than_approximating_an_unspellable_dtype() {
        let mut g = Graph::new();
        let q = leaf(&mut g, &[1, 4, 1, 4], DType::F32);
        let kc = leaf(&mut g, &[32, 4, 4, 4], DType::F32);
        let vc = leaf(&mut g, &[32, 4, 4, 4], DType::F32);
        // I16 has no ElementKind spelling (the seam has S8/U8/I32/I64/U32).
        let bt = leaf(&mut g, &[1, 8], DType::I16);
        let cl = leaf(&mut g, &[1], DType::U32);
        let id = g.push(Node {
            op: Op::Fused(
                FusedOps::PAGED_ATTN,
                FusedOpParams::PagedAttn {
                    softmax_scale: 0.5,
                    block_size: 4,
                    softcap: None,
                },
            ),
            inputs: vec![q, kc, vc, bt, cl],
            shape: Shape::from_dims(&[1, 4, 1, 4]),
            dtype: DType::F32,
        });
        let table = KernelBindingTable::new();
        assert!(
            jit_request_for_unplaceable_fused(
                &g,
                id,
                BackendId::Cuda,
                &table,
                ArchSku::Sm89,
                JitBudget {
                    max_compile_ms: 1_000
                },
            )
            .is_none(),
            "an unspellable operand dtype must decline, not approximate",
        );
    }
}
