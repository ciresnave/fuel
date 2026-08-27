// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! Backward is not yet implemented (panic stub in `NodeHandle::backward`);
//! the FlashAttn-shaped backward is a separate algorithm (the
//! "recompute" variant in the FlashAttention paper) and lands when a
//! consumer needs differentiable attention. Today `BackwardKind::
//! NotDifferentiable` reflects runtime behavior.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, DynScalar, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for FlashAttn.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::FLASH_ATTN,
        name: "FlashAttn",
        family: FusedOpFamily::Attention,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
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

// --- recipe builders (Increment C, C-T3) -------------------------------------
// Tiny constructors for the portable [`PatternNode`] recipe data. Every extent
// and diagonal is a concrete integer at decompose time (the concrete/None k_len
// arms only — the Sym arm never reaches here), so shape-changer nodes carry a
// BAKED absolute `target_shape` and the causal/window bands bake their concrete
// diagonal into `axis` (the FSCE / paged_attn posture) — the recipe mirrors the
// pre-C-T3 imperative body node-for-node. Reuses the C-T2 nested-fused carrier
// for the softmax (mechanism 2a).

fn r_op(op: OpTag, attrs: OpAttrs, operands: Vec<PatternNode>) -> PatternNode {
    PatternNode::Op {
        op,
        attrs,
        operands,
    }
}
fn r_bind(i: u8) -> PatternNode {
    PatternNode::Bind { index: i }
}
/// A baked absolute `target_shape` attr for a shape-changer (Reshape/BroadcastTo).
fn r_shape(dims: &[usize]) -> OpAttrs {
    OpAttrs {
        target_shape: dims.iter().map(|&d| d as i64).collect(),
        ..OpAttrs::default()
    }
}
/// A baked scalar for a scalar-param op (Mul/AddScalar) — NOT an open slot.
fn r_scalar(v: f64) -> OpAttrs {
    OpAttrs {
        scalars: vec![v],
        ..OpAttrs::default()
    }
}
/// An absolute permutation attr.
fn r_perm(p: Vec<u8>) -> OpAttrs {
    OpAttrs {
        perm: p,
        ..OpAttrs::default()
    }
}
/// A baked concrete diagonal for a Triu/Tril band (`tag_to_op` reads `axis`).
fn r_diag(diagonal: i64) -> OpAttrs {
    OpAttrs {
        axis: Some(diagonal),
        ..OpAttrs::default()
    }
}
/// A baked `Slice { dim, start, len }` attr (all concrete at decompose time).
fn r_slice(dim: i64, start: u64, len: u64) -> OpAttrs {
    OpAttrs {
        axis: Some(dim),
        slice_start: Some(start),
        slice_len: Some(len),
        ..OpAttrs::default()
    }
}
/// A `Cast(dtype)` attr (target rides `cast_dtype` as the stable name string).
fn cast_attr(dtype: DType) -> OpAttrs {
    OpAttrs {
        cast_dtype: Some(dtype.as_str().to_string()),
        ..OpAttrs::default()
    }
}
/// The nested-softmax selector attr (mechanism 2a): names the `SoftmaxLastDim`
/// registry entry the C-T2 carrier reconstructs to.
fn fused_softmax_attr() -> OpAttrs {
    OpAttrs {
        fused_op: Some("SoftmaxLastDim".to_string()),
        ..OpAttrs::default()
    }
}

/// GQA head-repeat as recipe DATA (mirrors [`repeat_kv_heads`]): when
/// `Hq == Hkv` the operand passes through unchanged; otherwise Reshape →
/// (equal-rank) BroadcastTo → Reshape. The explicit Reshape keeps the
/// BroadcastTo operand at equal rank so emit's D4 auto-pad never fires (the
/// recipe matches the legacy exactly).
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

/// ALiBi bias `slope[h] · (j - (i + q_pos_offset))` broadcast to `[B, Hq, Sq,
/// Sk]` as recipe DATA — the structure-preserving migration of
/// [`alibi_bias`]. The bottom-right decode alignment rides an `AddScalar`
/// shift emitted only when `q_pos_offset != 0` (byte-identical to the
/// imperative guard). The interior nodes are all F32 (Iota → F32); the final
/// F32→`dtype` `Cast` is emitted only when the attention dtype differs (an
/// F32→F32 cast is an executor-less identity), exactly as the imperative does.
/// Every BroadcastTo is preceded by an equal-rank Reshape (no D4 auto-pad).
fn alibi_bias_recipe(
    alibi: PatternNode,
    b: usize,
    hq: usize,
    sq: usize,
    sk: usize,
    q_pos_offset: usize,
    dtype: DType,
) -> PatternNode {
    let grid = &[sq, sk];
    let full = &[b, hq, sq, sk];
    let row = r_op(
        OpTag::Reshape,
        r_shape(&[sq, 1]),
        vec![r_op(OpTag::Iota, r_shape(&[sq]), vec![])],
    );
    let col = r_op(
        OpTag::Reshape,
        r_shape(&[1, sk]),
        vec![r_op(OpTag::Iota, r_shape(&[sk]), vec![])],
    );
    let row_bc = r_op(OpTag::BroadcastTo, r_shape(grid), vec![row]);
    let col_bc = r_op(OpTag::BroadcastTo, r_shape(grid), vec![col]);
    let mut rel = r_op(OpTag::Sub, OpAttrs::default(), vec![col_bc, row_bc]); // j - i
    if q_pos_offset != 0 {
        rel = r_op(
            OpTag::AddScalar,
            r_scalar(-(q_pos_offset as f64)),
            vec![rel],
        );
    }
    let rel_re = r_op(OpTag::Reshape, r_shape(&[1, 1, sq, sk]), vec![rel]);
    let rel_4d = r_op(OpTag::BroadcastTo, r_shape(full), vec![rel_re]);
    let slope_re = r_op(OpTag::Reshape, r_shape(&[1, hq, 1, 1]), vec![alibi]);
    let slope_4d = r_op(OpTag::BroadcastTo, r_shape(full), vec![slope_re]);
    let bias_f32 = r_op(OpTag::Mul, OpAttrs::default(), vec![slope_4d, rel_4d]);
    if dtype == DType::F32 {
        bias_f32
    } else {
        r_op(OpTag::Cast, cast_attr(dtype), vec![bias_f32])
    }
}

/// The pieces of the recomputed attention state a decompose recipe needs, all as
/// portable [`PatternNode`] DATA — the recipe-side mirror of the imperative
/// [`AttnRecompute`]. The forward decompose consumes only `probs` + `v_rep`; the
/// backward ([`super::flash_attn_backward`], C-T4) also consumes `k_rep` (for dQ)
/// and `softcap_tanh` (the `1 − tanh²` derivative). Because every field is a
/// slot-free subtree, emit's within-call identity-share collapses a field that
/// re-occurs (e.g. `k_rep` appears both inside `probs` via `kt` AND standalone in
/// dQ; `softcap_tanh` appears inside `probs` via `MulScalar(cap)` AND in the
/// backward's `1 − tanh²`) to ONE node — reproducing the imperative DAG sharing.
pub(crate) struct AttnRecomputeRecipe {
    pub probs: PatternNode,
    pub k_rep: PatternNode,
    pub v_rep: PatternNode,
    pub softcap_tanh: Option<PatternNode>,
}

/// Build `probs = softmax( mask( alibi( softcap( scale·QKᵀ ) ) ) )` plus the
/// GQA-repeated K/V (and, when softcap is active, the saved `tanh`), all as
/// portable recipe DATA (the structure-preserving mirror of [`recompute_probs`],
/// shared by the C-T3 forward and the C-T4 backward). `k_use`/`v_use` are the
/// (possibly-sliced) K/V recipe subtrees. Returns an [`AttnRecomputeRecipe`].
///
/// The mask bands share the SAME `neg_inf` node (`AddScalar(-inf)(MulScalar(0)(
/// S0))`, `S0` = the scaled scores at mask entry): every band is authored over a
/// clone of `neg_inf`, and emit's slot-free identity-share collapses the clones
/// (and the shared `S0`) to one node each — reproducing the imperative DAG
/// sharing. The concrete `q_pos_offset` shifts every band diagonal (`o+1`,
/// `o+r+1`, `o-l-1`) and the alibi rel, baked per call. ZERO open scalar slots
/// (scale, `1/cap`, `cap`, `0`, `-inf` are all baked pattern constants).
#[allow(clippy::too_many_arguments)]
pub(crate) fn recompute_probs_recipe(
    q: PatternNode,
    k_use: PatternNode,
    v_use: PatternNode,
    alibi: Option<PatternNode>,
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
    q_pos_offset: usize,
    dtype: DType,
) -> AttnRecomputeRecipe {
    // --- GQA / MQA: repeat K and V heads from Hkv up to Hq. -------------
    let k_rep = repeat_kv_heads_recipe(k_use, b, hkv, hq, sk, d);
    let v_rep = repeat_kv_heads_recipe(v_use, b, hkv, hq, sk, d);

    // --- scores = scale · (q · kᵀ) -------------------------------------
    // `k_rep` is cloned into `kt`; the standalone `k_rep` is returned for the
    // backward's dQ (emit's identity-share dedups the two structurally-equal
    // occurrences back to one node).
    let kt = r_op(
        OpTag::Permute,
        r_perm(vec![0, 1, 3, 2]),
        vec![k_rep.clone()],
    );
    let scores = r_op(OpTag::MatMul, OpAttrs::default(), vec![q, kt]);
    let mut scaled = r_op(
        OpTag::MulScalar,
        r_scalar(softmax_scale as f64),
        vec![scores],
    );

    // --- softcap: cap · tanh(scaled / cap) -----------------------------
    // The saved `tanh` (`t`) is cloned into `scaled` AND returned for the
    // backward's `1 − tanh²` derivative (identity-share dedups the two).
    let mut softcap_tanh = None;
    if let Some(cap) = softcap {
        let pre = r_op(OpTag::MulScalar, r_scalar(1.0 / cap as f64), vec![scaled]);
        let t = r_op(OpTag::Tanh, OpAttrs::default(), vec![pre]);
        softcap_tanh = Some(t.clone());
        scaled = r_op(OpTag::MulScalar, r_scalar(cap as f64), vec![t]);
    }

    // --- alibi: scaled += slope[h] · (j - (i + q_pos_offset)) ----------
    if let Some(al) = alibi {
        let bias = alibi_bias_recipe(al, b, hq, sq, sk, q_pos_offset, dtype);
        scaled = r_op(OpTag::Add, OpAttrs::default(), vec![scaled, bias]);
    }

    // --- causal / sliding-window: additive -inf bands ------------------
    let needs_mask = causal || window_r.is_some() || window_l.is_some();
    if needs_mask {
        // `S0` = scaled at mask entry; shared by `neg_inf` and the first band's
        // Add (emit dedups the clones). `neg_inf = -inf · 0 · S0`.
        let s0 = scaled;
        let zeros = r_op(OpTag::MulScalar, r_scalar(0.0), vec![s0.clone()]);
        let neg_inf = r_op(OpTag::AddScalar, r_scalar(f64::NEG_INFINITY), vec![zeros]);
        // `q_pos_offset` shifts every band's diagonal: query row `i` sits at
        // absolute position `i + offset`, so key column `j` is causal iff
        // `j ≤ i + offset`.
        let o = q_pos_offset as i64;
        let add_band = |running: PatternNode, band_op: OpTag, diagonal: i64| -> PatternNode {
            let band = r_op(band_op, r_diag(diagonal), vec![neg_inf.clone()]);
            r_op(OpTag::Add, OpAttrs::default(), vec![running, band])
        };
        let mut running = s0;
        if causal {
            running = add_band(running, OpTag::Triu, o + 1);
        }
        if let Some(r) = window_r {
            running = add_band(running, OpTag::Triu, o + r as i64 + 1);
        }
        if let Some(l) = window_l {
            running = add_band(running, OpTag::Tril, o - l as i64 - 1);
        }
        scaled = running;
    }

    // --- probs = softmax(scaled), carried as a nested Op::Fused node (2a) ---
    let probs = r_op(OpTag::Fused, fused_softmax_attr(), vec![scaled]);
    AttnRecomputeRecipe {
        probs,
        k_rep,
        v_rep,
        softcap_tanh,
    }
}

/// FlashAttn's materialized-SDPA primitive recipe as portable [`PatternNode`]
/// DATA (Increment C, C-T3) — the structure-preserving migration of the
/// pre-C-T3 imperative `decompose` body (the concrete/None `k_len` arms) onto
/// the re-emit machinery. `concrete_klen` selects the decode slice + bottom-
/// right alignment (`q_pos_offset = kl − Sq`); `has_alibi` / `softcap` /
/// `causal` / windows select structure via ordinary Rust `if`s. All extents +
/// diagonals are baked per call — ZERO open scalar slots.
///
/// Bind space: `0 = q`, `1 = k`, `2 = v`, `3 = alibi_slopes` (present iff
/// `has_alibi`) — the fused node's input order.
#[allow(clippy::too_many_arguments)]
fn recipe(
    b: usize,
    hq: usize,
    sq: usize,
    d: usize,
    hkv: usize,
    sk_full: usize,
    concrete_klen: Option<usize>,
    softmax_scale: f32,
    causal: bool,
    window_l: Option<usize>,
    window_r: Option<usize>,
    softcap: Option<f32>,
    has_alibi: bool,
    dtype: DType,
) -> PatternNode {
    // For a concrete k_len, Slice K/V to the live prefix and attend it
    // bottom-right-aligned (offset kl − Sq). None ⇒ full capacity, offset 0.
    let (k_use, v_use, sk, q_pos_offset) = match concrete_klen {
        None => (r_bind(1), r_bind(2), sk_full, 0usize),
        Some(kl) => {
            let k_sl = r_op(OpTag::Slice, r_slice(2, 0, kl as u64), vec![r_bind(1)]);
            let v_sl = r_op(OpTag::Slice, r_slice(2, 0, kl as u64), vec![r_bind(2)]);
            (k_sl, v_sl, kl, kl.saturating_sub(sq))
        }
    };
    let alibi = if has_alibi { Some(r_bind(3)) } else { None };
    let rc = recompute_probs_recipe(
        r_bind(0),
        k_use,
        v_use,
        alibi,
        b,
        hq,
        sq,
        sk,
        d,
        hkv,
        softmax_scale,
        causal,
        window_l,
        window_r,
        softcap,
        q_pos_offset,
        dtype,
    );
    // out = probs · v  →  [B, Hq, Sq, D]  (k_rep / softcap_tanh are backward-only)
    r_op(OpTag::MatMul, OpAttrs::default(), vec![rc.probs, rc.v_rep])
}

/// Decompose FlashAttn to its primitive **materialized scaled-dot-product
/// attention** subgraph and return the new root. Since Increment C C-T3 a
/// re-emit of [`recipe`]'s portable data through the [`decompose_via_recipe`]
/// bridge (structure-preserving: the emitted base map is node-for-node
/// identical to the pre-C-T3 imperative body — see the parity test in
/// `tests`). The general math:
///
/// ```text
///   k,v    = repeat-heads if GQA/MQA (Hq != Hkv)   # [B, Hq, Sk, D]
///   kT     = Permute([0,1,3,2])(k)                 # [B, Hq, D, Sk]
///   scores = MatMul(q, kT)                         # [B, Hq, Sq, Sk]
///   scaled = MulScalar(softmax_scale)(scores)
///          [ softcap: cap·tanh(scaled/cap) ]
///          [ alibi:   scaled += slope · (j - i)  via Iota positions ]
///          [ causal / sliding-window: additive -inf bands via Triu/Tril ]
///   probs  = SoftmaxLastDim(scaled)                # softmax over Sk (nested 2a)
///   out    = MatMul(probs, v)                      # [B, Hq, Sq, D]
/// ```
///
/// This is the *math* FlashAttn computes; the fused kernel is a faster,
/// numerically-close implementation (online vs. materialized softmax). Whether
/// to keep the fused form or use this lowering is the optimizer's cost-guided
/// call — not `decompose`'s, which returns the recipe whenever it *can* express
/// the configuration.
///
/// Per G2 (2026-06-20) `decompose` is total and never panics. It resolves
/// `k_len` — the live attended-prefix length backing a capacity-KV decode —
/// into one of three cases:
///
/// - **`None`** (vanilla / prefill): attend the full allocated `Sk`. Offset 0.
/// - **`Some(Concrete(kl))`** (per-step decode, or any build-time-known
///   length): a **static** config — `Slice` K/V to `kl` and run the SDPA
///   recipe **bottom-right-aligned** (`q_pos_offset = kl − Sq`, the standard
///   FA2 alignment, threaded into every causal / window / alibi band).
/// - **`Some(Sym(_))`** (persistent / plan-once decode): a genuine
///   **registry-layer basis gap**. Slicing K/V to a *symbolic* length needs a
///   primitive Fuel's basis lacks — `Op::Slice` carries a static `usize` len,
///   and there is no op that materializes a `DynScalar` into a length-mask
///   tensor *inside* a `decompose` fn (which sees only the static graph +
///   params, never the per-realize `SymEnv`). So `decompose` returns **self**
///   — the driver's never-crash fixpoint signal — BEFORE any recipe
///   construction — and the *symbolic* decode oracle is emitted one layer up by
///   the optimizer's decode-flash arm (`fuel_dispatch::decode_flash`), which
///   *does* hold the `SymEnv` and builds the `matmul → mask → softmax → matmul`
///   region over the capacity KV. See [`phase-d-symbolic-extents`]. Closing
///   this at the registry layer is a build-time basis extension (a
///   `DynScalar`-length `Slice`/mask op) — a future constitution decision, NOT
///   part of Increment C (a PERMANENT registry-layer gap this migration keeps).
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (softmax_scale, causal, window_l, window_r, softcap, k_len) = match params {
        FusedOpParams::FlashAttn {
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            k_len,
        } => (
            *softmax_scale,
            *causal,
            *window_size_left,
            *window_size_right,
            *softcap,
            *k_len,
        ),
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };

    // Resolve the attended length. A symbolic k_len is the one basis gap —
    // return self (fixpoint) BEFORE any recipe construction; the decode-flash
    // optimizer arm owns its oracle. Concrete/None arms proceed to the recipe.
    let concrete_klen: Option<usize> = match k_len {
        None => None,
        Some(DynScalar::Concrete(kl)) => Some(kl),
        Some(DynScalar::Sym(_)) => return id,
    };

    let (q_shape, k_shape, n_inputs, dtype) = {
        let n = graph.node(id);
        // A well-formed FlashAttn node has 3 (q,k,v) or 4 (+alibi) inputs.
        // Malformed → fixpoint self-return (never panic).
        if n.inputs.len() != 3 && n.inputs.len() != 4 {
            return id;
        }
        (
            graph.node(n.inputs[0]).shape.clone(), // q
            graph.node(n.inputs[1]).shape.clone(), // k
            n.inputs.len(),
            n.dtype,
        )
    };
    let q_dims = q_shape.dims();
    let k_dims = k_shape.dims();
    // Defensive shape guards — malformed → fixpoint self-return.
    if q_dims.len() != 4 || k_dims.len() != 4 {
        return id;
    }
    let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let hkv = k_dims[1];
    let sk_full = k_dims[2];
    let has_alibi = n_inputs == 4;

    let recipe_node = recipe(
        b,
        hq,
        sq,
        d,
        hkv,
        sk_full,
        concrete_klen,
        softmax_scale,
        causal,
        window_l,
        window_r,
        softcap,
        has_alibi,
        dtype,
    );
    // No open scalar slots (scale / softcap / -inf / diagonals are baked).
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Recomputed attention internals shared by the forward decompose and the
/// backward ([`super::flash_attn_backward`]): the softmax probabilities and
/// the repeated K/V, plus — when softcap is active — the saved `tanh` node the
/// backward needs for the `1 - tanh²` derivative.
// V-variant imperative path: `dV = Pᵀ·dO` is linear in P and needs only
// `probs`. `k_rep`/`v_rep`/`softcap_tanh`/`scores_shape` are consumed by the
// Q/K variants via `decompose_via_recipe` (flash_attn_backward.rs:256-297) and
// by tests; softcap's `1 - tanh²` derivative enters dQ/dK and cannot enter dV.
// `#[allow]`, not `#[expect]`: the fields ARE read in the test target, so an
// `expect` would be unfulfilled there — only the lib target sees them dead.
#[allow(dead_code)]
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
    q_pos_offset: usize,
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

    // --- alibi: scaled += slope[h] · (j - (i + q_pos_offset)) ----------
    if let Some(alibi) = alibi_id {
        let bias = alibi_bias(graph, alibi, b, hq, sq, sk, q_pos_offset, dtype);
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
        // `q_pos_offset` (= `k_len - Sq` for a bottom-right-aligned capacity-K
        // decode; 0 for the vanilla top-left case) shifts every band's
        // diagonal: query row `i` sits at absolute position `i + offset`, so a
        // key column `j` is causal iff `j ≤ i + offset`.
        let o = q_pos_offset as i64;
        if causal {
            scaled = add_band(graph, scaled, Op::Triu { diagonal: o + 1 });
        }
        if let Some(r) = window_r {
            scaled = add_band(
                graph,
                scaled,
                Op::Triu {
                    diagonal: o + r as i64 + 1,
                },
            );
        }
        if let Some(l) = window_l {
            scaled = add_band(
                graph,
                scaled,
                Op::Tril {
                    diagonal: o - l as i64 - 1,
                },
            );
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
    graph: &mut Graph,
    x_id: NodeId,
    b: usize,
    hkv: usize,
    hq: usize,
    s: usize,
    d: usize,
    dtype: DType,
) -> NodeId {
    if hq == hkv {
        return x_id;
    }
    let g = hq / hkv;
    let r5 = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[b, hkv, 1, s, d])),
        inputs: vec![x_id],
        shape: Shape::from_dims(&[b, hkv, 1, s, d]),
        dtype,
    });
    let bc = graph.push(Node {
        op: Op::BroadcastTo(Shape::from_dims(&[b, hkv, g, s, d])),
        inputs: vec![r5],
        shape: Shape::from_dims(&[b, hkv, g, s, d]),
        dtype,
    });
    graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[b, hq, s, d])),
        inputs: vec![bc],
        shape: Shape::from_dims(&[b, hq, s, d]),
        dtype,
    })
}

/// Build the ALiBi bias `slope[h] · (j - i)` broadcast to `[B, Hq, Sq, Sk]`,
/// cast to `dtype`. Uses `Op::Iota` for the row/column position indices.
/// Shared with `paged_attn` (`Sk` = the paged `kv_len`).
pub(crate) fn alibi_bias(
    graph: &mut Graph,
    alibi_id: NodeId,
    b: usize,
    hq: usize,
    sq: usize,
    sk: usize,
    q_pos_offset: usize,
    dtype: DType,
) -> NodeId {
    let f32 = DType::F32;
    let grid = Shape::from_dims(&[sq, sk]);
    let row_iota = graph.push(Node {
        op: Op::Iota { len: sq },
        inputs: vec![],
        shape: Shape::from_dims(&[sq]),
        dtype: f32,
    });
    let row = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[sq, 1])),
        inputs: vec![row_iota],
        shape: Shape::from_dims(&[sq, 1]),
        dtype: f32,
    });
    let col_iota = graph.push(Node {
        op: Op::Iota { len: sk },
        inputs: vec![],
        shape: Shape::from_dims(&[sk]),
        dtype: f32,
    });
    let col = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[1, sk])),
        inputs: vec![col_iota],
        shape: Shape::from_dims(&[1, sk]),
        dtype: f32,
    });
    let row_bc = graph.push(Node {
        op: Op::BroadcastTo(grid.clone()),
        inputs: vec![row],
        shape: grid.clone(),
        dtype: f32,
    });
    let col_bc = graph.push(Node {
        op: Op::BroadcastTo(grid.clone()),
        inputs: vec![col],
        shape: grid.clone(),
        dtype: f32,
    });
    let mut rel = graph.push(Node {
        op: Op::Sub,
        inputs: vec![col_bc, row_bc],
        shape: grid.clone(),
        dtype: f32, // j - i
    });
    // Bottom-right decode alignment: absolute query position is `i +
    // q_pos_offset`, so the relative distance is `j - (i + offset)`.
    if q_pos_offset != 0 {
        rel = graph.push(Node {
            op: Op::AddScalar(-(q_pos_offset as f64)),
            inputs: vec![rel],
            shape: grid,
            dtype: f32,
        });
    }
    let full = Shape::from_dims(&[b, hq, sq, sk]);
    let rel_re = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[1, 1, sq, sk])),
        inputs: vec![rel],
        shape: Shape::from_dims(&[1, 1, sq, sk]),
        dtype: f32,
    });
    let rel_4d = graph.push(Node {
        op: Op::BroadcastTo(full.clone()),
        inputs: vec![rel_re],
        shape: full.clone(),
        dtype: f32,
    });
    let slope_re = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[1, hq, 1, 1])),
        inputs: vec![alibi_id],
        shape: Shape::from_dims(&[1, hq, 1, 1]),
        dtype: f32,
    });
    let slope_4d = graph.push(Node {
        op: Op::BroadcastTo(full.clone()),
        inputs: vec![slope_re],
        shape: full.clone(),
        dtype: f32,
    });
    let bias_f32 = graph.push(Node {
        op: Op::Mul,
        inputs: vec![slope_4d, rel_4d],
        shape: full.clone(),
        dtype: f32,
    });
    // Match the scores dtype. A F32→F32 cast is an identity the executor has
    // no kernel for, so emit the Cast only when the attention dtype differs;
    // for F32 attention the bias node is already the right dtype.
    if dtype == f32 {
        bias_f32
    } else {
        graph.push(Node {
            op: Op::Cast(dtype),
            inputs: vec![bias_f32],
            shape: full,
            dtype,
        })
    }
}

/// Matcher stub — FlashAttn nodes originate from
/// `NodeHandle::flash_attn`-style builders, not from user-decomposed
/// `matmul + softmax + matmul` patterns. Recognizing the latter as
/// fusion-into-FlashAttn would require careful tolerance handling
/// (the tiled-softmax numerics aren't bit-identical to the naive
/// form) and isn't on the step-4 critical path.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FROZEN copy of the pre-C-T3 imperative `flash_attn::decompose` body,
    /// verbatim (the explicit-`Shape` `graph.push` spelling + the retained
    /// imperative `recompute_probs` helper), before C-T3 replaced the live body
    /// with the data recipe. The C-T3 structure-preservation oracle: the
    /// migrated recipe re-emit must produce a graph structurally identical to
    /// this. Only the concrete/None `k_len` arms are covered — the `Some(Sym)`
    /// arm self-returns in BOTH the frozen and the migrated body (the permanent
    /// registry-layer basis gap).
    fn frozen_legacy_decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
        let (q_id, k_id, v_id, alibi_id, q_shape, k_shape, dtype) = {
            let n = graph.node(id);
            let q_shape = graph.node(n.inputs[0]).shape.clone();
            let k_shape = graph.node(n.inputs[1]).shape.clone();
            let alibi = if n.inputs.len() == 4 {
                Some(n.inputs[3])
            } else {
                None
            };
            (
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                alibi,
                q_shape,
                k_shape,
                n.dtype,
            )
        };

        let (softmax_scale, causal, window_l, window_r, softcap, k_len) = match params {
            FusedOpParams::FlashAttn {
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                k_len,
            } => (
                *softmax_scale,
                *causal,
                *window_size_left,
                *window_size_right,
                *softcap,
                *k_len,
            ),
            _ => return id,
        };

        let concrete_klen: Option<usize> = match k_len {
            None => None,
            Some(DynScalar::Concrete(kl)) => Some(kl),
            Some(DynScalar::Sym(_)) => return id,
        };

        let q_dims = q_shape.dims();
        let k_dims = k_shape.dims();
        let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let hkv = k_dims[1];

        let (k_use, v_use, sk, q_pos_offset) = match concrete_klen {
            None => (k_id, v_id, k_dims[2], 0usize),
            Some(kl) => {
                let sliced = Shape::from_dims(&[b, hkv, kl, d]);
                let k_sl = graph.push(Node {
                    op: Op::Slice {
                        dim: 2,
                        start: 0,
                        len: kl,
                    },
                    inputs: vec![k_id],
                    shape: sliced.clone(),
                    dtype,
                });
                let v_sl = graph.push(Node {
                    op: Op::Slice {
                        dim: 2,
                        start: 0,
                        len: kl,
                    },
                    inputs: vec![v_id],
                    shape: sliced,
                    dtype,
                });
                (k_sl, v_sl, kl, kl.saturating_sub(sq))
            }
        };

        let rc = recompute_probs(
            graph,
            q_id,
            k_use,
            v_use,
            alibi_id,
            b,
            hq,
            sq,
            sk,
            d,
            hkv,
            softmax_scale,
            causal,
            window_l,
            window_r,
            softcap,
            q_pos_offset,
            dtype,
        );
        graph.push(Node {
            op: Op::MatMul,
            inputs: vec![rc.probs, rc.v_rep],
            shape: q_shape,
            dtype,
        })
    }

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, recursively-equal inputs). A shared leaf (same NodeId — a
    /// bound external input) matches by identity. Shape/dtype-sensitive at EVERY
    /// node, so it catches any structural drift the recipe migration introduces
    /// (a wrong band diagonal, a missing slice, an alibi offset, a lost share).
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return;
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(
            na.shape, nb.shape,
            "shape mismatch at {:?}: {:?} vs {:?}",
            na.op, na.shape, nb.shape
        );
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(
            na.inputs.len(),
            nb.inputs.len(),
            "arity mismatch at {:?}",
            na.op
        );
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }

    /// A representative FlashAttn config for the parity matrix.
    #[derive(Clone, Copy)]
    struct Cfg {
        hq: usize,
        hkv: usize,
        causal: bool,
        window_l: Option<usize>,
        window_r: Option<usize>,
        softcap: Option<f32>,
        alibi: bool,
        klen: Option<usize>,
    }

    /// Build a fused FLASH_ATTN node over `q [b,hq,sq,d]`, `k/v [b,hkv,Sk,d]`,
    /// optional `alibi [hq]`, all F32 `Op::Const` leaves. `Sk` is the capacity
    /// (a concrete `klen < Sk` exercises the decode slice). Returns the fused
    /// NodeId.
    fn flash_node(g: &mut Graph, b: usize, sq: usize, sk_cap: usize, d: usize, cfg: Cfg) -> NodeId {
        let q_shape = Shape::from_dims(&[b, cfg.hq, sq, d]);
        let q = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: q_shape.clone(),
            dtype: DType::F32,
        });
        let kv_shape = Shape::from_dims(&[b, cfg.hkv, sk_cap, d]);
        let k = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: kv_shape.clone(),
            dtype: DType::F32,
        });
        let v = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: kv_shape,
            dtype: DType::F32,
        });
        let mut inputs = vec![q, k, v];
        if cfg.alibi {
            let al = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[cfg.hq]),
                dtype: DType::F32,
            });
            inputs.push(al);
        }
        g.push(Node {
            op: Op::Fused(
                FusedOps::FLASH_ATTN,
                FusedOpParams::FlashAttn {
                    softmax_scale: 0.5,
                    causal: cfg.causal,
                    window_size_left: cfg.window_l,
                    window_size_right: cfg.window_r,
                    softcap: cfg.softcap,
                    k_len: cfg.klen.map(DynScalar::Concrete),
                },
            ),
            inputs,
            shape: q_shape,
            dtype: DType::F32,
        })
    }

    /// C-T3: ONE recipe form decomposes across a representative config matrix
    /// (plain / causal / windowed / GQA / softcap / alibi × None-and-concrete
    /// `k_len`), and its emitted base map is node-for-node identical to the
    /// FROZEN pre-C-T3 imperative body — the structure-preservation contract.
    /// Asserted at F32 (per the plan). The concrete-`k_len` arm exercises the
    /// decode Slice + `q_pos_offset` diagonal threading through every band + the
    /// alibi rel shift.
    #[test]
    fn flash_attn_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
        // b=1, sq=2, d=2, capacity Sk=4. klen=Some(3) < 4 fires the slice;
        // q_pos_offset = 3 - 2 = 1 (nonzero → exercises the band/alibi shift).
        let (b, sq, sk_cap, d) = (1usize, 2usize, 4usize, 2usize);
        // Mask combos cover causal + each window edge, independently and combined.
        let masks: [(bool, Option<usize>, Option<usize>); 6] = [
            (false, None, None),       // plain
            (true, None, None),        // causal
            (false, Some(2), Some(1)), // both windows
            (false, Some(2), None),    // left-only window
            (false, None, Some(1)),    // right-only window
            (true, Some(2), Some(1)),  // causal + windows (kitchen sink)
        ];
        let mut fired = 0usize;
        for (causal, window_l, window_r) in masks {
            for softcap in [None, Some(30.0f32)] {
                for alibi in [false, true] {
                    for (hq, hkv) in [(2usize, 2usize), (4usize, 2usize)] {
                        for klen in [None, Some(3usize)] {
                            let cfg = Cfg {
                                hq,
                                hkv,
                                causal,
                                window_l,
                                window_r,
                                softcap,
                                alibi,
                                klen,
                            };
                            let mut g = Graph::new();
                            let fused = flash_node(&mut g, b, sq, sk_cap, d, cfg);
                            let out_shape = g.node(fused).shape.clone();
                            let params = match &g.node(fused).op {
                                Op::Fused(_, p) => p.clone(),
                                _ => unreachable!(),
                            };

                            let new_root = decompose(&mut g, fused, &params);
                            assert_ne!(
                                new_root, fused,
                                "recipe decompose fires (causal={causal}, wl={window_l:?}, \
                                 wr={window_r:?}, softcap={softcap:?}, alibi={alibi}, \
                                 hq={hq}, hkv={hkv}, klen={klen:?})"
                            );
                            assert_eq!(g.node(new_root).shape, out_shape, "output = q shape");
                            assert_eq!(g.node(new_root).dtype, DType::F32);

                            let legacy_root = frozen_legacy_decompose(&mut g, fused, &params);
                            assert_structural_eq(&g, new_root, legacy_root);
                            fired += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            fired,
            6 * 2 * 2 * 2 * 2,
            "every config in the matrix was checked"
        );
    }

    /// Out-of-scope gap posture (permanent registry-layer basis gap): a
    /// `Some(Sym(_))` `k_len` self-returns BEFORE any recipe construction — no
    /// nodes emitted — exactly as the frozen legacy did. The symbolic decode
    /// oracle lives one layer up in `fuel_dispatch::decode_flash`.
    #[test]
    fn flash_attn_symbolic_klen_stays_a_fixpoint_gap() {
        let mut g = Graph::new();
        let cfg = Cfg {
            hq: 2,
            hkv: 2,
            causal: true,
            window_l: None,
            window_r: None,
            softcap: None,
            alibi: false,
            klen: None,
        };
        let fused = flash_node(&mut g, 1, 2, 4, 2, cfg);
        // Swap in a SYMBOLIC k_len (flash_node only builds concrete/None).
        let sym_params = FusedOpParams::FlashAttn {
            softmax_scale: 0.5,
            causal: true,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
            k_len: Some(DynScalar::Sym(fuel_ir::SymId(0))),
        };
        let before = g.len();
        let out = decompose(&mut g, fused, &sym_params);
        assert_eq!(
            out, fused,
            "symbolic k_len => registry-layer basis gap => fixpoint self-return"
        );
        assert_eq!(
            g.len(),
            before,
            "declined before any emission (no recipe built)"
        );
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn flash_attn_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let cfg = Cfg {
            hq: 2,
            hkv: 2,
            causal: false,
            window_l: None,
            window_r: None,
            softcap: None,
            alibi: false,
            klen: None,
        };
        let fused = flash_node(&mut g, 1, 2, 4, 2, cfg);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
