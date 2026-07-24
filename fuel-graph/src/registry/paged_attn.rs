//! PagedAttn — paged-cache scaled-dot-product attention. Phase 7.6
//! step 4 (continued — ninth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[q, k_cache, v_cache, block_table, context_lens, optional
//! alibi_slopes]`.
//!   - `q`:            `[B, Hq, Sq, D]`
//!   - `k_cache`:      `[num_blocks, block_size, Hkv, D]`
//!   - `v_cache`:      `[num_blocks, block_size, Hkv, D]`
//!   - `block_table`:  `[B, max_num_blocks_per_seq]` (u32)
//!   - `context_lens`: `[B]` (u32)
//!   - `alibi_slopes`: `[Hq]` (optional)
//!
//! Output: same shape as `q` (`[B, Hq, Sq, D]`).
//!
//! ## Architectural note — decode-only, non-differentiable
//!
//! PagedAttn is decode-side only by construction: the paged KV cache
//! has variable-length sequences and no training pass writes through
//! it. No gradient rule (matches the legacy `Op::PagedAttn { .. }`
//! arm in `Tensor::backward`, which panics). The registry entry's
//! `BackwardKind::NotDifferentiable` reflects this.
//!
//! No primitive decomposition exposed at the registry layer — same
//! rationale as FlashAttn (the paged-block traversal is the point of
//! the kernel; a "decompose to materialized k_cache + materialized
//! attention" lowering would defeat the design).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for PagedAttn.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::PAGED_ATTN,
        name:       "PagedAttn",
        family:     FusedOpFamily::Attention,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output shape equals input 0 (`q`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert!(
        input_shapes.len() == 5 || input_shapes.len() == 6,
        "PagedAttn takes 5 or 6 inputs",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0 (`q`).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert!(
        input_dtypes.len() == 5 || input_dtypes.len() == 6,
        "PagedAttn takes 5 or 6 inputs",
    );
    input_dtypes[0]
}

// --- recipe builders (Increment C, C-T2) -------------------------------------
// Tiny constructors for the portable [`PatternNode`] recipe data. Every shape
// is a concrete integer at decompose time, so shape-changer nodes carry a BAKED
// absolute `target_shape` (the FSCE / selective_scan posture, distinct from a
// §6.20 rel attr) — the recipe mirrors the pre-C-T2 imperative body node-for-node.

fn r_op(op: OpTag, attrs: OpAttrs, operands: Vec<PatternNode>) -> PatternNode {
    PatternNode::Op { op, attrs, operands }
}
fn r_bind(i: u8) -> PatternNode {
    PatternNode::Bind { index: i }
}
/// A baked absolute `target_shape` attr for a shape-changer (Reshape/BroadcastTo).
fn r_shape(dims: &[usize]) -> OpAttrs {
    OpAttrs { target_shape: dims.iter().map(|&d| d as i64).collect(), ..OpAttrs::default() }
}
/// A baked scalar for a scalar-param op (MulScalar) — NOT an open slot.
fn r_scalar(v: f64) -> OpAttrs {
    OpAttrs { scalars: vec![v], ..OpAttrs::default() }
}
/// A single-axis attr (IndexSelect dim).
fn r_axis(a: i64) -> OpAttrs {
    OpAttrs { axis: Some(a), ..OpAttrs::default() }
}
/// An absolute permutation attr.
fn r_perm(p: Vec<u8>) -> OpAttrs {
    OpAttrs { perm: p, ..OpAttrs::default() }
}

/// Gather a paged KV cache into dense attention form as recipe DATA: `IndexSelect`
/// the physical blocks named by the (flattened) block table, reshape the
/// `[B·max_blk, block_size, Hkv, D]` result to `[B, kv_len, Hkv, D]`, permute to
/// `[B, Hkv, kv_len, D]`, then GQA-repeat heads up to `Hq`. `cache` is the bound
/// k/v-cache operand; `bt_flat` the shared flattened block-table subtree (emit's
/// identity-share dedups the two occurrences to one node).
#[allow(clippy::too_many_arguments)]
fn gather_kv_recipe(
    cache: PatternNode,
    bt_flat: PatternNode,
    b: usize,
    kv_len: usize,
    hkv: usize,
    hq: usize,
    d: usize,
) -> PatternNode {
    let sel = r_op(OpTag::IndexSelect, r_axis(0), vec![cache, bt_flat]);
    let seq = r_op(OpTag::Reshape, r_shape(&[b, kv_len, hkv, d]), vec![sel]);
    let perm = r_op(OpTag::Permute, r_perm(vec![0, 2, 1, 3]), vec![seq]);
    repeat_kv_heads_recipe(perm, b, hkv, hq, kv_len, d)
}

/// GQA head-repeat as recipe DATA (mirrors `flash_attn::repeat_kv_heads`): when
/// `Hq == Hkv` the operand passes through unchanged; otherwise Reshape → (equal-
/// rank) BroadcastTo → Reshape. The explicit Reshape keeps the BroadcastTo
/// operand at equal rank so emit's D4 auto-pad never fires (the recipe matches
/// the legacy exactly).
fn repeat_kv_heads_recipe(
    x: PatternNode,
    b: usize,
    hkv: usize,
    hq: usize,
    s: usize,
    d: usize,
) -> PatternNode {
    if hq == hkv {
        return x;
    }
    let g = hq / hkv;
    let r5 = r_op(OpTag::Reshape, r_shape(&[b, hkv, 1, s, d]), vec![x]);
    let bc = r_op(OpTag::BroadcastTo, r_shape(&[b, hkv, g, s, d]), vec![r5]);
    r_op(OpTag::Reshape, r_shape(&[b, hq, s, d]), vec![bc])
}

/// ALiBi bias `slope[h] · (j - i)` broadcast to `[B, Hq, Sq, Sk]` as recipe DATA,
/// the `q_pos_offset = 0`, F32 specialization of `flash_attn::alibi_bias` used by
/// paged decode (no bottom-right `AddScalar` shift; no F32→F32 identity `Cast`).
/// `alibi` is the bound slopes operand (`[Hq]`). Every BroadcastTo is preceded by
/// an equal-rank Reshape (no D4 auto-pad).
fn alibi_bias_recipe(alibi: PatternNode, b: usize, hq: usize, sq: usize, sk: usize) -> PatternNode {
    let grid = &[sq, sk];
    let full = &[b, hq, sq, sk];
    let row = r_op(OpTag::Reshape, r_shape(&[sq, 1]), vec![r_op(OpTag::Iota, r_shape(&[sq]), vec![])]);
    let col = r_op(OpTag::Reshape, r_shape(&[1, sk]), vec![r_op(OpTag::Iota, r_shape(&[sk]), vec![])]);
    let row_bc = r_op(OpTag::BroadcastTo, r_shape(grid), vec![row]);
    let col_bc = r_op(OpTag::BroadcastTo, r_shape(grid), vec![col]);
    let rel = r_op(OpTag::Sub, OpAttrs::default(), vec![col_bc, row_bc]); // j - i (offset 0)
    let rel_re = r_op(OpTag::Reshape, r_shape(&[1, 1, sq, sk]), vec![rel]);
    let rel_4d = r_op(OpTag::BroadcastTo, r_shape(full), vec![rel_re]);
    let slope_re = r_op(OpTag::Reshape, r_shape(&[1, hq, 1, 1]), vec![alibi]);
    let slope_4d = r_op(OpTag::BroadcastTo, r_shape(full), vec![slope_re]);
    r_op(OpTag::Mul, OpAttrs::default(), vec![slope_4d, rel_4d]) // dtype F32 → no Cast
}

/// PagedAttn's gather-+-SDPA primitive recipe as portable [`PatternNode`] DATA
/// (Increment C, C-T2) — the structure-preserving migration of the pre-C-T2
/// imperative `decompose` body onto the re-emit machinery. The paged-block
/// traversal is the *fast kernel's* design point, but the always-correct
/// primitive form gathers the physical blocks (`IndexSelect`), runs dense SDPA
/// (`MatMul`/softmax/`MatMul`), and masks the variable-length tail
/// (`Iota`/`Cast`/`Ge`/`MaskedFill`) — every step in the closed `Op` basis, so
/// per G2 this is a real decomposition, not a basis-gap self-return.
///
/// **Config-branch mechanism (mechanism 1):** `softcap` and the alibi presence
/// (5- vs 6-input) select structure via ordinary Rust `if`s; `scale`, the softcap
/// `cap`, `-inf`, and every extent are baked per call — ZERO open scalar slots.
///
/// **Nested-fused (mechanism 2a):** the softmax stays an `OpTag::Fused`
/// (`SOFTMAX_LAST_DIM`) node carried AS-IS, so the optimizer can re-cover it; the
/// C-T2 `tag_to_op`/`emit` carrier reconstructs it and reads its shape/dtype from
/// the softmax registry entry.
///
/// Bind space: `0 = q`, `1 = k_cache`, `2 = v_cache`, `3 = block_table`,
/// `4 = context_lens`, `5 = alibi_slopes` (present iff `has_alibi`) — the fused
/// node's input order.
///
/// The MaskedFill fill value is authored dtype-polymorphically (no `cast_dtype`);
/// emit resolves the `Scalar` to operand[0]'s dtype. For F32 attention this is
/// `Scalar::F32(-inf)`, byte-identical to the imperative body; a non-F32 config
/// resolves to that dtype's `-inf` (the A2 carrier's dtype-correct behavior — the
/// legacy always baked F32, an under-protective quirk this migration supersedes),
/// so parity is asserted at F32 (per the plan).
#[allow(clippy::too_many_arguments)]
fn recipe(
    b: usize,
    hq: usize,
    sq: usize,
    d: usize,
    hkv: usize,
    max_blk: usize,
    block_size: usize,
    scale: f32,
    softcap: Option<f32>,
    has_alibi: bool,
) -> PatternNode {
    let kv_len = max_blk * block_size;
    let scores_shape = &[b, hq, sq, kv_len];

    // --- 1. gather physical blocks via the (shared) flattened block table ---
    let bt_flat = r_op(OpTag::Reshape, r_shape(&[b * max_blk]), vec![r_bind(3)]);
    let k_att = gather_kv_recipe(r_bind(1), bt_flat.clone(), b, kv_len, hkv, hq, d);
    let v_att = gather_kv_recipe(r_bind(2), bt_flat, b, kv_len, hkv, hq, d);

    // --- 2. scores = scale · q·kᵀ  (+ softcap, + alibi) ---
    let kt = r_op(OpTag::Permute, r_perm(vec![0, 1, 3, 2]), vec![k_att]);
    let scores = r_op(OpTag::MatMul, OpAttrs::default(), vec![r_bind(0), kt]);
    let mut scaled = r_op(OpTag::MulScalar, r_scalar(scale as f64), vec![scores]);
    if let Some(cap) = softcap {
        let pre = r_op(OpTag::MulScalar, r_scalar(1.0 / cap as f64), vec![scaled]);
        let t = r_op(OpTag::Tanh, OpAttrs::default(), vec![pre]);
        scaled = r_op(OpTag::MulScalar, r_scalar(cap as f64), vec![t]);
    }
    if has_alibi {
        let bias = alibi_bias_recipe(r_bind(5), b, hq, sq, kv_len);
        scaled = r_op(OpTag::Add, OpAttrs::default(), vec![scaled, bias]);
    }

    // --- 3. variable-length mask: -inf where key_pos ≥ context_len ---
    let pos = r_op(OpTag::Iota, r_shape(&[kv_len]), vec![]);
    let pos_re = r_op(OpTag::Reshape, r_shape(&[1, 1, 1, kv_len]), vec![pos]);
    let pos_bc = r_op(OpTag::BroadcastTo, r_shape(scores_shape), vec![pos_re]);
    let cl_f = r_op(OpTag::Cast, cast_f32_attr(), vec![r_bind(4)]);
    let cl_re = r_op(OpTag::Reshape, r_shape(&[b, 1, 1, 1]), vec![cl_f]);
    let cl_bc = r_op(OpTag::BroadcastTo, r_shape(scores_shape), vec![cl_re]);
    let mask = r_op(OpTag::Ge, OpAttrs::default(), vec![pos_bc, cl_bc]);
    // Dtype-polymorphic -inf fill: emit resolves the Scalar to operand[0]'s dtype.
    let masked = r_op(OpTag::MaskedFill, r_scalar(f32::NEG_INFINITY as f64), vec![scaled, mask]);

    // --- 4. probs = softmax(masked) carried as a nested Op::Fused node; out = probs · v ---
    let probs = r_op(OpTag::Fused, fused_softmax_attr(), vec![masked]);
    r_op(OpTag::MatMul, OpAttrs::default(), vec![probs, v_att])
}

/// The `Cast(F32)` attr for the `context_lens` U32→F32 conversion.
fn cast_f32_attr() -> OpAttrs {
    OpAttrs { cast_dtype: Some(DType::F32.as_str().to_string()), ..OpAttrs::default() }
}

/// The nested-softmax selector attr (mechanism 2a): names the `SoftmaxLastDim`
/// registry entry the C-T2 carrier reconstructs to.
fn fused_softmax_attr() -> OpAttrs {
    OpAttrs { fused_op: Some("SoftmaxLastDim".to_string()), ..OpAttrs::default() }
}

/// Lower a fused PagedAttn node to its primitive gather-+-SDPA subgraph and
/// return the new root id. Since Increment C C-T2 a re-emit of [`recipe`]'s
/// portable data through the [`decompose_via_recipe`] bridge (structure-
/// preserving: the emitted base map is node-for-node identical to the pre-C-T2
/// imperative body — see the parity test in `tests`). The per-call recipe bakes
/// the concrete shapes and selects the softcap / alibi structure via ordinary
/// Rust control flow (the config-branch mechanism); the softmax rides as a nested
/// `Op::Fused` node (mechanism 2a). Inputs are
/// `[q, k_cache, v_cache, block_table, context_lens, [alibi_slopes]]`.
///
/// Per G2 this is total + never-panic: a wrong-params payload or a malformed node
/// (wrong input arity, wrong ranks) returns `id` (the driver's fixpoint signal),
/// and any bridge decline (validation, bind-arity, emit) returns `id` too.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (scale, block_size, softcap) = match params {
        FusedOpParams::PagedAttn { softmax_scale, block_size, softcap } => {
            (*softmax_scale, *block_size, *softcap)
        }
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };
    let (q_shape, kc_shape, bt_shape, n_inputs) = {
        let n = graph.node(id);
        // A well-formed PagedAttn node has exactly 5 or 6 inputs. Malformed →
        // fixpoint self-return (never panic).
        if n.inputs.len() != 5 && n.inputs.len() != 6 {
            return id;
        }
        (
            graph.node(n.inputs[0]).shape.clone(), // q
            graph.node(n.inputs[1]).shape.clone(), // k_cache
            graph.node(n.inputs[3]).shape.clone(), // block_table
            n.inputs.len(),
        )
    };
    let q_dims = q_shape.dims();
    let kc_dims = kc_shape.dims();
    let bt_dims = bt_shape.dims();
    // Defensive shape guards — malformed → fixpoint self-return.
    if q_dims.len() != 4 || kc_dims.len() != 4 || bt_dims.len() != 2 {
        return id;
    }
    let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let hkv = kc_dims[2]; // k_cache [num_blocks, block_size, Hkv, D]
    let max_blk = bt_dims[1]; // block_table [B, max_blk]
    let has_alibi = n_inputs == 6;

    let recipe_node = recipe(b, hq, sq, d, hkv, max_blk, block_size, scale, softcap, has_alibi);
    // No open scalar slots (scale / softcap / -inf are baked constants).
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Matcher stub — PagedAttn originates from explicit builders, not
/// user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};
    use fuel_ir::Scalar;

    /// FROZEN copy of the pre-C-T2 imperative `paged_attn::decompose` body,
    /// verbatim (the explicit-`Shape` `graph.push` spelling + `legacy_gather_kv`
    /// helper), before C-T2 replaced the live body with the data recipe. The
    /// C-T2 structure-preservation oracle: the migrated recipe re-emit must
    /// produce a graph structurally identical to this.
    fn frozen_legacy_decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
        let (q_id, kc_id, vc_id, bt_id, cl_id, alibi_id, q_shape, dtype) = {
            let n = graph.node(id);
            let q_shape = graph.node(n.inputs[0]).shape.clone();
            let alibi = if n.inputs.len() == 6 { Some(n.inputs[5]) } else { None };
            (
                n.inputs[0], n.inputs[1], n.inputs[2], n.inputs[3], n.inputs[4], alibi, q_shape, n.dtype,
            )
        };
        let (scale, block_size, softcap) = match params {
            FusedOpParams::PagedAttn { softmax_scale, block_size, softcap } => {
                (*softmax_scale, *block_size, *softcap)
            }
            _ => return id,
        };

        let q_dims = q_shape.dims();
        let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let hkv = graph.node(kc_id).shape.dims()[2];
        let max_blk = graph.node(bt_id).shape.dims()[1];
        let kv_len = max_blk * block_size;

        let bt_flat = graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&[b * max_blk])),
            inputs: vec![bt_id],
            shape: Shape::from_dims(&[b * max_blk]),
            dtype: DType::U32,
        });
        let k_att = legacy_gather_kv(graph, kc_id, bt_flat, b, max_blk, block_size, hkv, hq, d, kv_len, dtype);
        let v_att = legacy_gather_kv(graph, vc_id, bt_flat, b, max_blk, block_size, hkv, hq, d, kv_len, dtype);

        let scores_shape = Shape::from_dims(&[b, hq, sq, kv_len]);

        let kt = graph.push(Node {
            op: Op::Permute(vec![0, 1, 3, 2]),
            inputs: vec![k_att],
            shape: Shape::from_dims(&[b, hq, d, kv_len]),
            dtype,
        });
        let scores = graph.push(Node {
            op: Op::MatMul,
            inputs: vec![q_id, kt],
            shape: scores_shape.clone(),
            dtype,
        });
        let mut scaled = graph.push(Node {
            op: Op::MulScalar(scale as f64),
            inputs: vec![scores],
            shape: scores_shape.clone(),
            dtype,
        });
        if let Some(cap) = softcap {
            let pre = graph.push(Node {
                op: Op::MulScalar(1.0 / cap as f64),
                inputs: vec![scaled],
                shape: scores_shape.clone(),
                dtype,
            });
            let t = graph.push(Node {
                op: Op::Tanh,
                inputs: vec![pre],
                shape: scores_shape.clone(),
                dtype,
            });
            scaled = graph.push(Node {
                op: Op::MulScalar(cap as f64),
                inputs: vec![t],
                shape: scores_shape.clone(),
                dtype,
            });
        }
        if let Some(alibi) = alibi_id {
            let bias = crate::registry::flash_attn::alibi_bias(graph, alibi, b, hq, sq, kv_len, 0, dtype);
            scaled = graph.push(Node {
                op: Op::Add,
                inputs: vec![scaled, bias],
                shape: scores_shape.clone(),
                dtype,
            });
        }

        let pos = graph.push(Node {
            op: Op::Iota { len: kv_len },
            inputs: vec![],
            shape: Shape::from_dims(&[kv_len]),
            dtype: DType::F32,
        });
        let pos_re = graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&[1, 1, 1, kv_len])),
            inputs: vec![pos],
            shape: Shape::from_dims(&[1, 1, 1, kv_len]),
            dtype: DType::F32,
        });
        let pos_bc = graph.push(Node {
            op: Op::BroadcastTo(scores_shape.clone()),
            inputs: vec![pos_re],
            shape: scores_shape.clone(),
            dtype: DType::F32,
        });
        let cl_f = graph.push(Node {
            op: Op::Cast(DType::F32),
            inputs: vec![cl_id],
            shape: Shape::from_dims(&[b]),
            dtype: DType::F32,
        });
        let cl_re = graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&[b, 1, 1, 1])),
            inputs: vec![cl_f],
            shape: Shape::from_dims(&[b, 1, 1, 1]),
            dtype: DType::F32,
        });
        let cl_bc = graph.push(Node {
            op: Op::BroadcastTo(scores_shape.clone()),
            inputs: vec![cl_re],
            shape: scores_shape.clone(),
            dtype: DType::F32,
        });
        let mask = graph.push(Node {
            op: Op::Ge,
            inputs: vec![pos_bc, cl_bc],
            shape: scores_shape.clone(),
            dtype: DType::U8,
        });
        let masked = graph.push(Node {
            op: Op::MaskedFill { value: Scalar::F32(f32::NEG_INFINITY) },
            inputs: vec![scaled, mask],
            shape: scores_shape.clone(),
            dtype,
        });

        let probs = graph.push(Node {
            op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
            inputs: vec![masked],
            shape: scores_shape,
            dtype,
        });
        graph.push(Node {
            op: Op::MatMul,
            inputs: vec![probs, v_att],
            shape: q_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-C-T2 imperative `gather_kv` helper.
    #[allow(clippy::too_many_arguments)]
    fn legacy_gather_kv(
        graph: &mut Graph,
        cache: NodeId,
        bt_flat: NodeId,
        b: usize,
        max_blk: usize,
        block_size: usize,
        hkv: usize,
        hq: usize,
        d: usize,
        kv_len: usize,
        dtype: DType,
    ) -> NodeId {
        let sel = graph.push(Node {
            op: Op::IndexSelect { dim: 0 },
            inputs: vec![cache, bt_flat],
            shape: Shape::from_dims(&[b * max_blk, block_size, hkv, d]),
            dtype,
        });
        let seq = graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&[b, kv_len, hkv, d])),
            inputs: vec![sel],
            shape: Shape::from_dims(&[b, kv_len, hkv, d]),
            dtype,
        });
        let perm = graph.push(Node {
            op: Op::Permute(vec![0, 2, 1, 3]),
            inputs: vec![seq],
            shape: Shape::from_dims(&[b, hkv, kv_len, d]),
            dtype,
        });
        crate::registry::flash_attn::repeat_kv_heads(graph, perm, b, hkv, hq, kv_len, d, dtype)
    }

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, recursively-equal inputs). A shared leaf (same NodeId — a
    /// bound external input) matches by identity. Shape/dtype-sensitive at EVERY
    /// node, so it catches any structural drift the recipe migration introduces.
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return;
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(na.shape, nb.shape, "shape mismatch at {:?}: {:?} vs {:?}", na.op, na.shape, nb.shape);
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(na.inputs.len(), nb.inputs.len(), "arity mismatch at {:?}", na.op);
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }

    /// Build a fused PagedAttn node over the `[q, k_cache, v_cache, block_table,
    /// context_lens, [alibi]]` inputs, all F32/U32 `Op::Const` leaves. Returns
    /// the fused NodeId.
    #[allow(clippy::too_many_arguments)]
    fn paged_node(
        g: &mut Graph,
        b: usize,
        hq: usize,
        hkv: usize,
        sq: usize,
        d: usize,
        num_blocks: usize,
        block_size: usize,
        max_blk: usize,
        scale: f32,
        softcap: Option<f32>,
        has_alibi: bool,
    ) -> NodeId {
        let q_shape = Shape::from_dims(&[b, hq, sq, d]);
        let q = g.push(Node { op: Op::Const, inputs: vec![], shape: q_shape.clone(), dtype: DType::F32 });
        let cache_shape = Shape::from_dims(&[num_blocks, block_size, hkv, d]);
        let kc = g.push(Node { op: Op::Const, inputs: vec![], shape: cache_shape.clone(), dtype: DType::F32 });
        let vc = g.push(Node { op: Op::Const, inputs: vec![], shape: cache_shape, dtype: DType::F32 });
        let bt = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[b, max_blk]), dtype: DType::U32 });
        let cl = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[b]), dtype: DType::U32 });
        let mut inputs = vec![q, kc, vc, bt, cl];
        if has_alibi {
            let al = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[hq]), dtype: DType::F32 });
            inputs.push(al);
        }
        g.push(Node {
            op: Op::Fused(
                FusedOps::PAGED_ATTN,
                FusedOpParams::PagedAttn { softmax_scale: scale, block_size, softcap },
            ),
            inputs,
            shape: q_shape,
            dtype: DType::F32,
        })
    }

    /// C-T2: the recipe decompose fires for every softcap × alibi × GQA config,
    /// and its emitted base map is node-for-node identical to the FROZEN pre-C-T2
    /// imperative body — the structure-preservation contract. Asserted at F32
    /// (per the plan's MaskedFill-dtype note: the emit-resolved `-inf` Scalar
    /// equals the imperative `Scalar::F32(-inf)` only at F32).
    #[test]
    fn paged_attn_recipe_decompose_matches_frozen_legacy_for_every_config() {
        // b=1, sq=3, d=2, block_size=2, max_blk=2 → kv_len=4 (≠ sq).
        for softcap in [None, Some(30.0f32)] {
            for has_alibi in [false, true] {
                for (hq, hkv) in [(2usize, 2usize), (4usize, 2usize)] {
                    let mut g = Graph::new();
                    let fused = paged_node(
                        &mut g, 1, hq, hkv, 3, 2, /*num_blocks*/ 4, /*block_size*/ 2,
                        /*max_blk*/ 2, /*scale*/ 0.5, softcap, has_alibi,
                    );
                    let out_shape = g.node(fused).shape.clone();
                    let params = match &g.node(fused).op {
                        Op::Fused(_, p) => p.clone(),
                        _ => unreachable!(),
                    };

                    let new_root = decompose(&mut g, fused, &params);
                    assert_ne!(
                        new_root, fused,
                        "recipe decompose fires (softcap={softcap:?}, alibi={has_alibi}, hq={hq}, hkv={hkv})"
                    );
                    assert_eq!(
                        g.node(new_root).shape, out_shape,
                        "output shape matches shape_rule (softcap={softcap:?}, alibi={has_alibi})"
                    );
                    assert_eq!(g.node(new_root).dtype, DType::F32, "PagedAttn output dtype is F32");

                    let legacy_root = frozen_legacy_decompose(&mut g, fused, &params);
                    assert_structural_eq(&g, new_root, legacy_root);
                }
            }
        }
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn paged_attn_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = paged_node(&mut g, 1, 2, 2, 3, 2, 4, 2, 2, 0.5, None, false);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
