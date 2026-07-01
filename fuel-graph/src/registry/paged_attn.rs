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
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Scalar, Shape};

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

/// Decompose to materialized paged attention. The paged-block traversal is the
/// *fast kernel's* design point, but the always-correct primitive form is a
/// gather + SDPA: every step is in the closed `Op` basis (`IndexSelect` does
/// the block-table gather; `Iota`/`Cast`/`Ge`/`MaskedFill` build the
/// variable-length `context_lens` mask), so per G2 this is a real
/// decomposition, not a basis-gap self-return. Inputs are
/// `[q, k_cache, v_cache, block_table, context_lens, [alibi_slopes]]`.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (q_id, kc_id, vc_id, bt_id, cl_id, alibi_id, q_shape, dtype) = {
        let n = graph.node(id);
        let q_shape = graph.node(n.inputs[0]).shape.clone();
        let alibi = if n.inputs.len() == 6 { Some(n.inputs[5]) } else { None };
        (
            n.inputs[0], n.inputs[1], n.inputs[2], n.inputs[3], n.inputs[4], alibi, q_shape, n.dtype,
        )
    };
    let (scale, block_size, softcap) = match params {
        FusedOpParams::PagedAttn {
            softmax_scale,
            block_size,
            softcap,
        } => (*softmax_scale, *block_size, *softcap),
        // Wrong params for this id — can't decompose; return self.
        _ => return id,
    };

    let q_dims = q_shape.dims();
    let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let hkv = graph.node(kc_id).shape.dims()[2]; // k_cache [num_blocks, block_size, Hkv, D]
    let max_blk = graph.node(bt_id).shape.dims()[1]; // block_table [B, max_blk]
    let kv_len = max_blk * block_size;

    // --- 1. gather physical blocks via the block table -----------------
    // Flatten block_table → 1-D U32 index, IndexSelect the cache blocks, then
    // reshape/permute to [B, Hkv, kv_len, D] and GQA-repeat up to Hq.
    let bt_flat = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[b * max_blk])),
        inputs: vec![bt_id],
        shape: Shape::from_dims(&[b * max_blk]),
        dtype: DType::U32,
    });
    let k_att = gather_kv(graph, kc_id, bt_flat, b, max_blk, block_size, hkv, hq, d, kv_len, dtype);
    let v_att = gather_kv(graph, vc_id, bt_flat, b, max_blk, block_size, hkv, hq, d, kv_len, dtype);

    let scores_shape = Shape::from_dims(&[b, hq, sq, kv_len]);

    // --- 2. scores = scale · q·kᵀ  (+ softcap, + alibi) ----------------
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
        // slope[h] · (j - i) over the [Sq, kv_len] grid (same convention as
        // flash_attn's alibi).
        let bias = super::flash_attn::alibi_bias(graph, alibi, b, hq, sq, kv_len, dtype);
        scaled = graph.push(Node {
            op: Op::Add,
            inputs: vec![scaled, bias],
            shape: scores_shape.clone(),
            dtype,
        });
    }

    // --- 3. variable-length mask: -inf where key_pos ≥ context_len -----
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
    // context_lens [B] U32 → F32 → [B,1,1,1] → broadcast.
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
    // mask = (pos ≥ context_len)  → U8 (1 at invalid positions).
    let mask = graph.push(Node {
        op: Op::Ge,
        inputs: vec![pos_bc, cl_bc],
        shape: scores_shape.clone(),
        dtype: DType::U8,
    });
    let masked = graph.push(Node {
        op: Op::MaskedFill {
            value: Scalar::F32(f32::NEG_INFINITY),
        },
        inputs: vec![scaled, mask],
        shape: scores_shape.clone(),
        dtype,
    });

    // --- 4. probs = softmax(masked); out = probs · v -------------------
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

/// Gather a paged KV cache into dense attention form: `IndexSelect` the
/// physical blocks named by the (flattened) block table, reshape the
/// `[B·max_blk, block_size, Hkv, D]` result to `[B, kv_len, Hkv, D]`, permute
/// to `[B, Hkv, kv_len, D]`, then GQA-repeat heads up to `Hq`.
#[allow(clippy::too_many_arguments)]
fn gather_kv(
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
    super::flash_attn::repeat_kv_heads(graph, perm, b, hkv, hq, kv_len, d, dtype)
}

/// Matcher stub — PagedAttn originates from explicit builders, not
/// user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
