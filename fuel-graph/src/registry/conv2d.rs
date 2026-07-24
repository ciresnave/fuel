//! Conv2D — 2-D cross-correlation with stride / padding / groups.
//! Phase 7.6 step 4 (continued — sixth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   `decompose` that lowers the groups=1 case to the im2col recipe
//!   (`groups>1` self-returns pending CV2), stubbed pattern).
//!
//! ## Architectural note — migrates via an index-gather im2col recipe (NOT a basis gap)
//!
//! **Correction (2026-07-24):** an earlier version of this note claimed
//! Conv2D "has no clean decomposition into the current primitive set"
//! and needs a new `Op::Im2Col` (`Op::Unfold` / `Op::AsStrided`)
//! primitive. **That was wrong.** It weighed only ONE candidate
//! lowering — the `Op::Slice` + `Op::MatMul` + `Op::Concat` synthesis,
//! which does explode to `N·Hout·Wout` slice nodes and is correctly
//! rejected — and overlooked the **index-gather im2col idiom**, whose
//! node count is *constant* in the spatial size and which uses only
//! primitives already in the build-time-closed `Op` basis:
//!
//! - **im2col = `Op::IndexSelect`** (a strided gather) over the
//!   `Op::Pad`-padded, flattened spatial axis. One 1-D index of length
//!   `Kh·Kw·Hout·Wout` gathers the whole patch matrix for every
//!   `(n, c)`; overlapping windows simply repeat index values (a gather
//!   reads a source position as many times as needed).
//! - the window→flat-index map is built from **`Op::Iota`** +
//!   `MulScalar` / `Add` + **`Op::Cast(U32)`** — integer-valued, exact
//!   in F32 for padded-spatial extent `< 2^24`.
//! - the grouped / batched contraction against the reshaped weight is
//!   **`Op::MatMul`**'s batched (rank≥2, GQA-divisible batch-prefix)
//!   form; bias is a broadcast `Add`.
//!
//! This mirrors the CPU kernel itself, which is im2col + batched GEMM
//! (`fuel-cpu-backend/src/conv2d.rs`). So Conv2D migrates to a total
//! `PatternNode` recipe like any other Increment-C op — **no `Op`
//! variant is added and the build-time-closed primitive basis is
//! unchanged.** See the `10-decisions-log.md` 2026-07-24 addendum and
//! the `2026-07-24-incc-conv-im2col` plan for the full recipe shape.
//!
//! **Status (CV1, im2col-1):** [`decompose`] lowers the **groups=1**
//! case (with an optional bias tail) to that recipe — see [`recipe`].
//! A grouped conv (`groups>1`) is still a surfaced honest-miss that
//! self-returns (the G2 total + never-panic fixpoint, telemetry, never
//! a crash) pending slice CV2's batched/per-group path; backends
//! without a native Conv2D kernel route the still-fused node through
//! `GraphExecutor::cpu_fallback`.
//!
//! The matcher is also stubbed (returns `None`) — Conv2D nodes
//! originate from the `Tensor::conv2d` builder; user-decomposed forms
//! don't exist as a pattern to recognize.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for Conv2D.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::CONV2D,
        name:       "Conv2D",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // Conv2D's backward is real (dX via ConvTranspose2D, dW via a
        // transposed Conv2D, dB via reduce_sum_to) but is wired
        // through `Tensor::backward`'s `Op::Fused(CONV2D, _)` arm
        // directly — same pattern as the other 5 already-migrated ops.
        // The registry's `BackwardKind::Fused(id)` path is reserved
        // for backward HELPERS (SoftmaxLastDimBackward etc.) that get
        // their own FusedOpId; Conv2D's backward is structural, not a
        // helper.
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Output shape rule. Conv2D's output spatial dims follow the standard
/// formula `Hout = (Hin + 2·pad.0 - Kh) / stride.0 + 1` (and the same
/// for width). Dilation is always 1 today.
fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    debug_assert!(
        input_shapes.len() == 2 || input_shapes.len() == 3,
        "Conv2D takes 2 or 3 inputs (x, weight, [bias])",
    );
    let (stride, padding) = match params {
        FusedOpParams::Conv2D { stride, padding, .. } => (*stride, *padding),
        _ => panic!("conv2d::shape_rule got non-Conv2D params: {params:?}"),
    };
    let x_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert_eq!(x_dims.len(), 4, "Conv2D x must be rank 4");
    debug_assert_eq!(w_dims.len(), 4, "Conv2D weight must be rank 4");
    let (n, _cin, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (cout, _cin_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    Shape::from_dims(&[n, cout, h_out, w_out])
}

/// Dtype rule: Conv2D output dtype equals input 0 (x) dtype.
fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert!(
        input_dtypes.len() == 2 || input_dtypes.len() == 3,
        "Conv2D takes 2 or 3 inputs",
    );
    input_dtypes[0]
}

/// Conv2D's index-gather im2col recipe as a per-call-baked `PatternNode`
/// (Increment C im2col-1, CV1) — for the **groups=1** case, with an optional
/// bias tail. Everything is concrete at decompose time (all input extents are
/// read off the node), so — like `selective_scan`'s per-call `recipe(seqlen,…)`
/// — the recipe bakes ABSOLUTE `OpAttrs` (no shape-relative carriers needed).
///
/// Bind space: `0 = x [N,Cin,Hin,Win]`, `1 = weight [Cout,Cin,Kh,Kw]`,
/// `[2 = bias [Cout]]`. Emitted subgraph (all existing primitives):
///
/// ```text
///   xp   = Pad[(0,0),(0,0),(ph,ph),(pw,pw), Const 0](x)   # [N,Cin,Hpad,Wpad]
///   xf   = Reshape([N,Cin,Hpad*Wpad])(xp)
///   idx  = Cast(U32)( build_flat_index() )                # 1-D [Kh*Kw*Hout*Wout]
///   cols = IndexSelect(dim=2)(xf, idx)                    # [N,Cin,Kh*Kw*Hout*Wout]
///   P    = Reshape([N, Cin*Kh*Kw, Hout*Wout])(cols)       # (Cin,Kh,Kw) contraction order
///   Wb   = BroadcastTo([N,Cout,Cin*Kh*Kw])(Reshape([1,Cout,Cin*Kh*Kw])(weight))
///   Y    = MatMul(Wb, P)                                  # [N, Cout, Hout*Wout]
///   out  = Reshape([N, Cout, Hout, Wout])(Y)
///   [ + Add(BroadcastTo([N,Cout,Hout,Wout])(Reshape([1,Cout,1,1])(bias))) ]
/// ```
///
/// `build_flat_index()` gathers over the padded, flattened spatial axis:
/// `idx(ky,kx,oh,ow) = (oh·sh + ky)·Wpad + (ow·sw + kx)`. It is assembled as a
/// sum of four per-axis terms — `Op::Iota(axis_len)` reshaped to a rank-4
/// broadcastable form, scaled by its stride factor (`MulScalar`), and broadcast
/// to `[Kh,Kw,Hout,Wout]` — then flattened to 1-D and `Op::Cast(U32)`. Integer
/// values are exact in F32 for padded-spatial extent `< 2^24`. The padded input
/// carries zeros in the pad region, so a gather of an out-of-(unpadded)-bounds
/// window position reads a zero — matching the direct-conv "contributes 0"
/// convention exactly. The `(Cin,Kh,Kw)` reshape ordering of `P` lines the
/// contraction axis up with the weight's `[Cout, Cin·Kh·Kw]` reshape
/// (`fuel-conv`'s `((co·cin_per_g + ci)·k_h + ky)·k_w + kx` weight layout).
///
/// This mirrors the CPU kernel (im2col + batched GEMM) at the graph level and
/// adds NO `Op` variant — the build-time-closed primitive basis is unchanged.
fn recipe(
    n: usize, cin: usize,
    hpad: usize, wpad: usize,
    kh: usize, kw: usize,
    hout: usize, wout: usize,
    cout: usize,
    sh: usize, sw: usize,
    ph: usize, pw: usize,
    has_bias: bool,
) -> PatternNode {
    use OpTag as T;
    let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
    let bind = |i| PatternNode::Bind { index: i };
    let shape_attr = |dims: Vec<i64>| OpAttrs { target_shape: dims, ..OpAttrs::default() };

    let k = cin * kh * kw; // contraction dim  (Cin·Kh·Kw)
    let spatial = hout * wout; // output spatial (Hout·Wout)
    let l = kh * kw * hout * wout; // gathered positions per (n, cin)

    // ---- 1. Pad x -> [N,Cin,Hpad,Wpad], flatten spatial -> [N,Cin,Hpad*Wpad].
    let xp = op(
        T::Pad,
        OpAttrs {
            pad_amounts: vec![
                (0, 0),
                (0, 0),
                (ph as u64, ph as u64),
                (pw as u64, pw as u64),
            ],
            pad_mode:  Some(0), // PadMode::Constant
            pad_value: Some(0.0),
            ..OpAttrs::default()
        },
        vec![bind(0)],
    );
    let xf = op(
        T::Reshape,
        shape_attr(vec![n as i64, cin as i64, (hpad * wpad) as i64]),
        vec![xp],
    );

    // ---- 2. build the flat gather index [L].
    // idx(ky,kx,oh,ow) = ky·Wpad + kx + oh·(sh·Wpad) + ow·sw
    let idx4_shape = vec![kh as i64, kw as i64, hout as i64, wout as i64];
    // A per-axis term: Iota(axis_len) -> Reshape(rank-4 broadcastable)
    //   -> MulScalar(factor) -> BroadcastTo([Kh,Kw,Hout,Wout]).
    let term = |axis_len: usize, reshape_dims: Vec<i64>, factor: f64| -> PatternNode {
        let iota = op(
            T::Iota,
            OpAttrs { target_shape: vec![axis_len as i64], ..OpAttrs::default() },
            vec![],
        );
        let re = op(T::Reshape, shape_attr(reshape_dims), vec![iota]);
        let scaled = op(
            T::MulScalar,
            OpAttrs { scalars: vec![factor], ..OpAttrs::default() },
            vec![re],
        );
        op(T::BroadcastTo, shape_attr(idx4_shape.clone()), vec![scaled])
    };
    let term_ky = term(kh, vec![kh as i64, 1, 1, 1], wpad as f64);
    let term_kx = term(kw, vec![1, kw as i64, 1, 1], 1.0);
    let term_oh = term(hout, vec![1, 1, hout as i64, 1], (sh * wpad) as f64);
    let term_ow = term(wout, vec![1, 1, 1, wout as i64], sw as f64);
    let add = |a, b| op(T::Add, OpAttrs::default(), vec![a, b]);
    let idx4 = add(add(add(term_ky, term_kx), term_oh), term_ow);
    let idx1 = op(T::Reshape, shape_attr(vec![l as i64]), vec![idx4]);
    let idx = op(
        T::Cast,
        OpAttrs { cast_dtype: Some("u32".to_string()), ..OpAttrs::default() },
        vec![idx1],
    );

    // ---- 3. im2col gather -> [N,Cin,L]; reshape to patch matrix [N,K,Hout*Wout].
    let cols = op(
        T::IndexSelect,
        OpAttrs { axis: Some(2), ..OpAttrs::default() },
        vec![xf, idx],
    );
    let patch = op(
        T::Reshape,
        shape_attr(vec![n as i64, k as i64, spatial as i64]),
        vec![cols],
    );

    // ---- 4. weight -> [1,Cout,K] -> broadcast [N,Cout,K]; matmul -> [N,Cout,Hout*Wout].
    let wm = op(T::Reshape, shape_attr(vec![1, cout as i64, k as i64]), vec![bind(1)]);
    let wb = op(
        T::BroadcastTo,
        shape_attr(vec![n as i64, cout as i64, k as i64]),
        vec![wm],
    );
    let ymat = op(T::MatMul, OpAttrs::default(), vec![wb, patch]);

    // ---- 5. reshape to [N,Cout,Hout,Wout]; optional bias broadcast-Add tail.
    let out = op(
        T::Reshape,
        shape_attr(vec![n as i64, cout as i64, hout as i64, wout as i64]),
        vec![ymat],
    );
    if has_bias {
        let bm = op(
            T::Reshape,
            shape_attr(vec![1, cout as i64, 1, 1]),
            vec![bind(2)],
        );
        let bb = op(
            T::BroadcastTo,
            shape_attr(vec![n as i64, cout as i64, hout as i64, wout as i64]),
            vec![bm],
        );
        add(out, bb)
    } else {
        out
    }
}

/// Total decomposition of Conv2D via the index-gather im2col recipe (Increment
/// C im2col-1, CV1) — a re-emit of [`recipe`]'s portable data through the
/// [`decompose_via_recipe`] bridge. Conv2D is **NOT** a primitive-basis gap
/// (correcting the earlier "needs `Op::Im2Col`" claim; see the module note and
/// the `10-decisions-log.md` 2026-07-24 addendum): im2col is an overlapping-
/// window strided gather (`Op::IndexSelect`), fully expressible in the closed
/// `Op` basis.
///
/// **Scope (CV1): groups=1**, with or without bias. The stride/padding and the
/// concrete `Cin/Hin/Win/Cout/Kh/Kw` extents are read here and baked into the
/// per-call recipe. A grouped conv (`groups>1`) is a surfaced honest-miss that
/// stays fused pending slice CV2's batched/per-group path.
///
/// Per G2 (2026-06-20) this is total + never-panic: a wrong-params payload, a
/// `groups>1` node, a malformed input arity/shape, or any bridge decline
/// (bind-arity, rel-resolution, …) returns `id` — the driver's fixpoint signal,
/// a surfaced opaque-op gap, never a crash. The recipe is the *math* the kernel
/// computes; the fused CPU/CUDA Conv2D kernel stays the executed production path
/// (the optimizer chooses by cost), and the recipe realizes on any backend with
/// the primitive set for the base-map cover and the numeric verify oracle.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (stride, padding, groups) = match params {
        FusedOpParams::Conv2D { stride, padding, groups } => (*stride, *padding, *groups),
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };
    // CV1 scope: groups=1. Grouped conv is a surfaced honest-miss (CV2).
    if groups != 1 {
        return id;
    }

    // Read the input shapes in a short borrow. A well-formed groups=1 CONV2D
    // node has 2 inputs (x, weight) or 3 (x, weight, bias); malformed →
    // fixpoint self-return (never panic).
    let (x_shape, w_shape, n_inputs) = {
        let node = graph.node(id);
        if node.inputs.len() != 2 && node.inputs.len() != 3 {
            return id;
        }
        (
            graph.node(node.inputs[0]).shape.clone(),
            graph.node(node.inputs[1]).shape.clone(),
            node.inputs.len(),
        )
    };
    let x_dims = x_shape.dims();
    let w_dims = w_shape.dims();
    if x_dims.len() != 4 || w_dims.len() != 4 {
        return id;
    }
    let (n, cin, hin, win) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (cout, cin_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    // groups=1 ⇒ weight's Cin/group must equal x's Cin.
    if cin_per_g != cin {
        return id;
    }
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    if sh == 0 || sw == 0 {
        return id;
    }
    let hpad = hin + 2 * ph;
    let wpad = win + 2 * pw;
    // Guard the output-spatial arithmetic (kernel fits the padded input).
    if hpad < kh || wpad < kw {
        return id;
    }
    let hout = (hpad - kh) / sh + 1;
    let wout = (wpad - kw) / sw + 1;

    let has_bias = n_inputs == 3;
    let recipe_node = recipe(n, cin, hpad, wpad, kh, kw, hout, wout, cout, sh, sw, ph, pw, has_bias);
    // No open scalar slots — every MulScalar carries a baked stride factor.
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Matcher stub — Conv2D is always produced by the `Tensor::conv2d`
/// builder; there is no user-decomposed pattern to recognize as
/// `Op::Fused(CONV2D, _)`.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    fn mk_const(g: &mut Graph, dims: &[usize]) -> NodeId {
        g.push(Node {
            op:     Op::Const,
            inputs: vec![],
            shape:  Shape::from_dims(dims),
            dtype:  DType::F32,
        })
    }

    /// Build a fused CONV2D node over `x [N,Cin,Hin,Win]`,
    /// `weight [Cout, Cin/groups, Kh, Kw]`, optional `bias [Cout]`.
    #[allow(clippy::too_many_arguments)]
    fn fused_conv(
        g: &mut Graph,
        n: usize, cin: usize, hin: usize, win: usize,
        cout: usize, kh: usize, kw: usize,
        stride: (usize, usize), padding: (usize, usize), groups: usize,
        bias: bool,
    ) -> NodeId {
        let x = mk_const(g, &[n, cin, hin, win]);
        let w = mk_const(g, &[cout, cin / groups, kh, kw]);
        let mut inputs = vec![x, w];
        if bias {
            inputs.push(mk_const(g, &[cout]));
        }
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let hout = (hin + 2 * ph - kh) / sh + 1;
        let wout = (win + 2 * pw - kw) / sw + 1;
        g.push(Node {
            op: Op::Fused(
                FusedOps::CONV2D,
                FusedOpParams::Conv2D { stride, padding, groups },
            ),
            inputs,
            shape: Shape::from_dims(&[n, cout, hout, wout]),
            dtype: DType::F32,
        })
    }

    /// Collect every reachable node's `Op` from `root` (deduped by NodeId).
    fn reachable_ops(g: &Graph, root: NodeId) -> Vec<Op> {
        let mut stack = vec![root];
        let mut seen = std::collections::HashSet::new();
        let mut ops = Vec::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let node = g.node(id);
            ops.push(node.op.clone());
            for &inp in &node.inputs {
                stack.push(inp);
            }
        }
        ops
    }

    /// CV1 (im2col-1): a groups=1 CONV2D node (with and without bias)
    /// decomposes to the index-gather im2col + batched-matmul recipe —
    /// NO `Op::Fused(CONV2D)` survives, and the recipe is built from the
    /// existing `Pad`/`Iota`/`IndexSelect`/`MatMul` primitives. Born-red
    /// while `decompose` self-returns (`root == fused`).
    #[test]
    fn conv2d_groups1_decompose_lowers_to_im2col_matmul() {
        for bias in [false, true] {
            let mut g = Graph::new();
            // small mixed shape: N=1, Cin=2, 5x5, Cout=3, 2x2 kernel.
            let fused = fused_conv(&mut g, 1, 2, 5, 5, 3, 2, 2, (1, 1), (0, 0), 1, bias);
            let out_shape = g.node(fused).shape.clone();
            let params = match &g.node(fused).op {
                Op::Fused(_, p) => p.clone(),
                other => panic!("expected fused node, got {other:?}"),
            };

            let root = decompose(&mut g, fused, &params);
            assert_ne!(root, fused, "conv2d groups=1 must lower (bias={bias})");
            assert_eq!(
                g.node(root).shape, out_shape,
                "lowered root keeps [N,Cout,Hout,Wout] (bias={bias})",
            );
            assert_eq!(g.node(root).dtype, DType::F32);

            let ops = reachable_ops(&g, root);
            assert!(
                !ops.iter().any(|o| matches!(o, Op::Fused(fid, _) if *fid == FusedOps::CONV2D)),
                "no Op::Fused(CONV2D) remains after decompose (bias={bias})",
            );
            assert!(ops.iter().any(|o| matches!(o, Op::MatMul)), "recipe contains MatMul");
            assert!(
                ops.iter().any(|o| matches!(o, Op::IndexSelect { .. })),
                "recipe gathers patches via IndexSelect",
            );
            assert!(ops.iter().any(|o| matches!(o, Op::Iota { .. })), "index built from Iota");
            assert!(ops.iter().any(|o| matches!(o, Op::Pad { .. })), "input padded via Op::Pad");
            assert!(
                ops.iter().any(|o| matches!(o, Op::Cast(dt) if *dt == DType::U32)),
                "index cast to U32 for the gather",
            );
            if bias {
                assert!(ops.iter().any(|o| matches!(o, Op::Add)), "bias added via broadcast Add tail");
            }
        }
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint — never a
    /// crash — before any emission.
    #[test]
    fn conv2d_wrong_params_is_a_fixpoint() {
        let mut g = Graph::new();
        let fused = fused_conv(&mut g, 1, 2, 5, 5, 3, 2, 2, (1, 1), (0, 0), 1, false);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }

    /// CV1 scope is groups=1. A grouped conv is a surfaced honest-miss
    /// (the batched/per-group path lands in slice CV2) — it declines to a
    /// fixpoint, never a crash or a wrong (non-grouped) lowering.
    #[test]
    fn conv2d_groups_gt_1_is_a_fixpoint_for_now() {
        let mut g = Graph::new();
        let fused = fused_conv(&mut g, 1, 4, 5, 5, 4, 2, 2, (1, 1), (0, 0), 2, false);
        let before = g.len();
        let params = match &g.node(fused).op {
            Op::Fused(_, p) => p.clone(),
            other => panic!("expected fused node, got {other:?}"),
        };
        let out = decompose(&mut g, fused, &params);
        assert_eq!(out, fused, "groups>1 => fixpoint (CV1 handles groups=1 only)");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
