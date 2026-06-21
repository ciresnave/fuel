//! FlashAttn — multi-head scaled-dot-product attention with
//! FlashAttention-shaped kernel hooks. Phase 7.6 step 4 (continued —
//! eighth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   `decompose` to the materialized SDPA subgraph for the vanilla case
//!   (returns self for configs needing not-yet-present primitives, per G2),
//!   and a stubbed pattern).
//!
//! Inputs: `[q, k, v, optional alibi_slopes]`.
//!   - `q`: `[B, Hq, Sq, D]`
//!   - `k`: `[B, Hkv, Sk, D]`
//!   - `v`: `[B, Hkv, Sk, D]`
//!   - `alibi_slopes` (optional): `[Hq]`
//!
//! Output: same shape as `q` (`[B, Hq, Sq, D]`).
//!
//! ## Architectural note — no primitive decomposition (yet)
//!
//! Attention does have a primitive decomposition (`matmul → softmax →
//! matmul`, with masking + scaling), but FlashAttn's value is
//! specifically that it *avoids* materializing the `[B, Hq, Sq, Sk]`
//! attention matrix — a primitive lowering would defeat the purpose.
//! Backends without a flash-attention kernel route through
//! `GraphExecutor::cpu_fallback` to the reference naive-attention
//! implementation (which does decompose internally). A graph-level
//! `decompose` to a primitive subgraph would be a footgun: it would
//! either reproduce the very memory blowup FlashAttn exists to avoid,
//! or pretend the primitive form is equivalent when it isn't (the
//! tiled softmax in the kernel produces different numerics than the
//! naive form).
//!
//! Backward is not yet implemented (panic stub in `Tensor::backward`);
//! the FlashAttn-shaped backward is a separate algorithm (the
//! "recompute" variant in the FlashAttention paper) and lands when a
//! consumer needs differentiable attention. Today `BackwardKind::
//! NotDifferentiable` reflects runtime behavior.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for FlashAttn.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::FLASH_ATTN,
        name:       "FlashAttn",
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
        input_shapes.len() == 4 || input_shapes.len() == 5,
        "FlashAttn takes 4 or 5 inputs (q, k, v, [softmax_lse], [alibi])",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0 (`q`).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert!(
        input_dtypes.len() == 4 || input_dtypes.len() == 5,
        "FlashAttn takes 4 or 5 inputs",
    );
    input_dtypes[0]
}

/// Decompose FlashAttn to its primitive **materialized scaled-dot-product
/// attention** subgraph and return the new root. The general recipe:
///
/// ```text
///   k,v    = repeat-heads if GQA/MQA (Hq != Hkv)   # [B, Hq, Sk, D]
///   kT     = Permute([0,1,3,2])(k)                 # [B, Hq, D, Sk]
///   scores = MatMul(q, kT)                         # [B, Hq, Sq, Sk]
///   scaled = MulScalar(softmax_scale)(scores)
///          [ softcap: cap·tanh(scaled/cap) ]
///          [ alibi:   scaled += slope · (j - i)  via Iota positions ]
///          [ causal / sliding-window: additive -inf bands via Triu/Tril ]
///   probs  = SoftmaxLastDim(scaled)                # softmax over Sk
///   out    = MatMul(probs, v)                      # [B, Hq, Sq, D]
/// ```
///
/// This is the *math* FlashAttn computes; the fused kernel is a faster,
/// numerically-close implementation (online vs. materialized softmax).
/// Whether to keep the fused form or use this lowering is the optimizer's
/// cost-guided call — not `decompose`'s, which returns the recipe whenever
/// it *can* express the configuration.
///
/// Per G2 (2026-06-20) `decompose` is total and never panics. Today every
/// **static** config decomposes (vanilla, GQA/MQA, softcap, causal,
/// sliding-window, alibi). The one config still returning self is a
/// **dynamic `k_len`** (persistent-decode over a capacity KV-cache): its
/// decomposition needs a *symbolic* slice of K/V to the live length, which
/// belongs to the symbolic-extent subsystem ([`phase-d-symbolic-extents`])
/// — a cross-subsystem integration, not a missing primitive. It is wired
/// there, not papered over here.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (q_id, k_id, v_id, alibi_id, q_shape, k_shape, dtype) = {
        let n = graph.node(id);
        let q_shape = graph.node(n.inputs[0]).shape.clone();
        let k_shape = graph.node(n.inputs[1]).shape.clone();
        let alibi = if n.inputs.len() == 4 { Some(n.inputs[3]) } else { None };
        (n.inputs[0], n.inputs[1], n.inputs[2], alibi, q_shape, k_shape, n.dtype)
    };

    let (softmax_scale, causal, window_l, window_r, softcap, has_dyn_klen) = match params {
        FusedOpParams::FlashAttn {
            softmax_scale, causal, window_size_left, window_size_right, softcap, k_len,
        } => (
            *softmax_scale, *causal, *window_size_left, *window_size_right,
            *softcap, k_len.is_some(),
        ),
        // Wrong params for this id — can't decompose; return self.
        _ => return id,
    };

    // Dynamic k_len needs a symbolic slice of K/V (symbolic-extent subsystem).
    // It is decomposable, just not with the static primitives here — return
    // self for now; the symbolic-extent integration owns this path.
    if has_dyn_klen {
        return id;
    }

    let q_dims = q_shape.dims();
    let k_dims = k_shape.dims();
    let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (hkv, sk) = (k_dims[1], k_dims[2]);

    let rc = recompute_probs(
        graph, q_id, k_id, v_id, alibi_id, b, hq, sq, sk, d, hkv, softmax_scale, causal,
        window_l, window_r, softcap, dtype,
    );
    // out = probs · v  →  [B, Hq, Sq, D]
    graph.push(Node {
        op: Op::MatMul,
        inputs: vec![rc.probs, rc.v_rep],
        shape: q_shape,
        dtype,
    })
}

/// Recomputed attention internals shared by the forward decompose and the
/// backward ([`super::flash_attn_backward`]): the softmax probabilities and
/// the repeated K/V, plus — when softcap is active — the saved `tanh` node the
/// backward needs for the `1 - tanh²` derivative.
pub(crate) struct AttnRecompute {
    pub probs: NodeId,
    pub k_rep: NodeId,
    pub v_rep: NodeId,
    pub softcap_tanh: Option<NodeId>,
    pub scores_shape: Shape,
}

/// Build `probs = softmax( mask( alibi( softcap( scale·QKᵀ ) ) ) )` plus the
/// repeated K/V, all from primitive ops. Shared by the forward decompose and
/// the backward so their recompute of the score/probability state is
/// byte-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recompute_probs(
    graph: &mut Graph,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
    alibi_id: Option<NodeId>,
    b: usize,
    hq: usize,
    sq: usize,
    sk: usize,
    d: usize,
    hkv: usize,
    softmax_scale: f32,
    causal: bool,
    window_l: Option<usize>,
    window_r: Option<usize>,
    softcap: Option<f32>,
    dtype: DType,
) -> AttnRecompute {
    let scores_shape = Shape::from_dims(&[b, hq, sq, sk]);

    // --- GQA / MQA: repeat K and V heads from Hkv up to Hq. -------------
    let k_rep = repeat_kv_heads(graph, k_id, b, hkv, hq, sk, d, dtype);
    let v_rep = repeat_kv_heads(graph, v_id, b, hkv, hq, sk, d, dtype);

    // --- scores = scale · (q · kᵀ) -------------------------------------
    let kt_id = graph.push(Node {
        op: Op::Permute(vec![0, 1, 3, 2]),
        inputs: vec![k_rep],
        shape: Shape::from_dims(&[b, hq, d, sk]),
        dtype,
    });
    let scores_id = graph.push(Node {
        op: Op::MatMul,
        inputs: vec![q_id, kt_id],
        shape: scores_shape.clone(),
        dtype,
    });
    let mut scaled = graph.push(Node {
        op: Op::MulScalar(softmax_scale as f64),
        inputs: vec![scores_id],
        shape: scores_shape.clone(),
        dtype,
    });

    // --- softcap: cap · tanh(scaled / cap) -----------------------------
    let mut softcap_tanh = None;
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
        softcap_tanh = Some(t);
        scaled = graph.push(Node {
            op: Op::MulScalar(cap as f64),
            inputs: vec![t],
            shape: scores_shape.clone(),
            dtype,
        });
    }

    // --- alibi: scaled += slope[h] · (j - i) ---------------------------
    if let Some(alibi) = alibi_id {
        let bias = alibi_bias(graph, alibi, b, hq, sq, sk, dtype);
        scaled = graph.push(Node {
            op: Op::Add,
            inputs: vec![scaled, bias],
            shape: scores_shape.clone(),
            dtype,
        });
    }

    // --- causal / sliding-window: additive -inf bands ------------------
    let needs_mask = causal || window_r.is_some() || window_l.is_some();
    if needs_mask {
        let zeros = graph.push(Node {
            op: Op::MulScalar(0.0),
            inputs: vec![scaled],
            shape: scores_shape.clone(),
            dtype,
        });
        let neg_inf = graph.push(Node {
            op: Op::AddScalar(f64::NEG_INFINITY),
            inputs: vec![zeros],
            shape: scores_shape.clone(),
            dtype,
        });
        let add_band = |graph: &mut Graph, scaled: NodeId, op: Op| -> NodeId {
            let band = graph.push(Node {
                op,
                inputs: vec![neg_inf],
                shape: scores_shape.clone(),
                dtype,
            });
            graph.push(Node {
                op: Op::Add,
                inputs: vec![scaled, band],
                shape: scores_shape.clone(),
                dtype,
            })
        };
        if causal {
            scaled = add_band(graph, scaled, Op::Triu { diagonal: 1 });
        }
        if let Some(r) = window_r {
            scaled = add_band(graph, scaled, Op::Triu { diagonal: r as i64 + 1 });
        }
        if let Some(l) = window_l {
            scaled = add_band(graph, scaled, Op::Tril { diagonal: -(l as i64) - 1 });
        }
    }

    let probs = graph.push(Node {
        op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
        inputs: vec![scaled],
        shape: scores_shape.clone(),
        dtype,
    });
    AttnRecompute {
        probs,
        k_rep,
        v_rep,
        softcap_tanh,
        scores_shape,
    }
}

/// Repeat a `[B, Hkv, S, D]` K/V tensor's heads up to `Hq` (GQA/MQA) via
/// `Reshape → BroadcastTo → Reshape`. Identity when `Hq == Hkv`. Shared with
/// `paged_attn` (hkv-major / g-minor ordering).
pub(crate) fn repeat_kv_heads(
    graph: &mut Graph, x_id: NodeId,
    b: usize, hkv: usize, hq: usize, s: usize, d: usize, dtype: DType,
) -> NodeId {
    if hq == hkv {
        return x_id;
    }
    let g = hq / hkv;
    let r5 = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[b, hkv, 1, s, d])),
        inputs: vec![x_id], shape: Shape::from_dims(&[b, hkv, 1, s, d]), dtype,
    });
    let bc = graph.push(Node {
        op: Op::BroadcastTo(Shape::from_dims(&[b, hkv, g, s, d])),
        inputs: vec![r5], shape: Shape::from_dims(&[b, hkv, g, s, d]), dtype,
    });
    graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[b, hq, s, d])),
        inputs: vec![bc], shape: Shape::from_dims(&[b, hq, s, d]), dtype,
    })
}

/// Build the ALiBi bias `slope[h] · (j - i)` broadcast to `[B, Hq, Sq, Sk]`,
/// cast to `dtype`. Uses `Op::Iota` for the row/column position indices.
/// Shared with `paged_attn` (`Sk` = the paged `kv_len`).
pub(crate) fn alibi_bias(
    graph: &mut Graph, alibi_id: NodeId,
    b: usize, hq: usize, sq: usize, sk: usize, dtype: DType,
) -> NodeId {
    let f32 = DType::F32;
    let grid = Shape::from_dims(&[sq, sk]);
    let row_iota = graph.push(Node {
        op: Op::Iota { len: sq }, inputs: vec![], shape: Shape::from_dims(&[sq]), dtype: f32,
    });
    let row = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[sq, 1])),
        inputs: vec![row_iota], shape: Shape::from_dims(&[sq, 1]), dtype: f32,
    });
    let col_iota = graph.push(Node {
        op: Op::Iota { len: sk }, inputs: vec![], shape: Shape::from_dims(&[sk]), dtype: f32,
    });
    let col = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[1, sk])),
        inputs: vec![col_iota], shape: Shape::from_dims(&[1, sk]), dtype: f32,
    });
    let row_bc = graph.push(Node {
        op: Op::BroadcastTo(grid.clone()), inputs: vec![row], shape: grid.clone(), dtype: f32,
    });
    let col_bc = graph.push(Node {
        op: Op::BroadcastTo(grid.clone()), inputs: vec![col], shape: grid.clone(), dtype: f32,
    });
    let rel = graph.push(Node {
        op: Op::Sub, inputs: vec![col_bc, row_bc], shape: grid, dtype: f32,   // j - i
    });
    let full = Shape::from_dims(&[b, hq, sq, sk]);
    let rel_re = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[1, 1, sq, sk])),
        inputs: vec![rel], shape: Shape::from_dims(&[1, 1, sq, sk]), dtype: f32,
    });
    let rel_4d = graph.push(Node {
        op: Op::BroadcastTo(full.clone()),
        inputs: vec![rel_re], shape: full.clone(), dtype: f32,
    });
    let slope_re = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[1, hq, 1, 1])),
        inputs: vec![alibi_id], shape: Shape::from_dims(&[1, hq, 1, 1]), dtype: f32,
    });
    let slope_4d = graph.push(Node {
        op: Op::BroadcastTo(full.clone()),
        inputs: vec![slope_re], shape: full.clone(), dtype: f32,
    });
    let bias_f32 = graph.push(Node {
        op: Op::Mul, inputs: vec![slope_4d, rel_4d], shape: full.clone(), dtype: f32,
    });
    // Match the scores dtype. A F32→F32 cast is an identity the executor has
    // no kernel for, so emit the Cast only when the attention dtype differs;
    // for F32 attention the bias node is already the right dtype.
    if dtype == f32 {
        bias_f32
    } else {
        graph.push(Node { op: Op::Cast(dtype), inputs: vec![bias_f32], shape: full, dtype })
    }
}

/// Matcher stub — FlashAttn nodes originate from
/// `Tensor::flash_attn`-style builders, not from user-decomposed
/// `matmul + softmax + matmul` patterns. Recognizing the latter as
/// fusion-into-FlashAttn would require careful tolerance handling
/// (the tiled-softmax numerics aren't bit-identical to the naive
/// form) and isn't on the step-4 critical path.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
