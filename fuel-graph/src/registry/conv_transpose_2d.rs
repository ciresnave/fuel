//! ConvTranspose2D — 2-D transposed (fractionally-strided)
//! convolution. Phase 7.6 step 4 (continued — seventh op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   `decompose` that lowers ANY `groups>=1` case to the col2im recipe
//!   (groups=1 rank-3; `groups>1` rank-4 batched over `[N, groups]`), stubbed
//!   pattern).
//!
//! ## Architectural note — migrates via a col2im scatter-add recipe (NOT a basis gap)
//!
//! **Correction (2026-07-24):** the earlier note claimed there is "no
//! `Op::Im2Col` (or `Op::Col2Im`) primitive that could express
//! ConvTranspose2D" and that the lowering "would produce astronomical
//! node counts." **That was wrong** — for the same reason as Conv2D
//! (see that registry entry's module note). Transposed convolution is
//! `Op::MatMul` then **col2im (overlap-add)**, which is the adjoint of
//! im2col's gather = a **scatter-add**: `Op::IndexAdd` / `Op::ScatterAdd`
//! into a zero-init base (a `MulScalar(0.0)` of a broadcast), then an
//! `Op::Slice` crop. `IndexAdd`'s `+=` semantics ARE the overlap-add;
//! `stride` / `output_padding` / `dilation` fold into the scatter-index
//! map (`Op::Iota` + arithmetic + `Op::Cast(U32)`) and the crop.
//! Constant node count, only build-time-closed-basis primitives — **no
//! `Op` variant is added.** So ConvTranspose2D migrates to a total
//! `PatternNode` recipe like any other Increment-C op. See the Conv2D
//! module note, the `10-decisions-log.md` 2026-07-24 addendum, and the
//! `2026-07-24-incc-conv-im2col` plan.
//!
//! **Status (CV3, im2col-3):** [`decompose`] lowers **any `groups>=1`**
//! (the builder's 2-input, no-bias form) to that recipe — the groups=1
//! rank-3 [`recipe`] and the `groups>1` (incl. depthwise `groups=Cin`)
//! rank-4 batched [`recipe_grouped`], which carries the group as an
//! explicit matmul batch axis (`[N, groups]`). The shared [`scatter_flat_index`]
//! (`Op::Iota`+arith+`Op::Cast(U32)`) and [`col2im_tail`] (zero base via
//! `MulScalar(0)`, `Op::IndexAdd` overlap-add, `Op::Slice` crop) are identical
//! across groupings. A wrong-params / non-2-input / malformed-shape /
//! indivisible-grouping payload is a surfaced honest-miss that self-returns (the
//! G2 total + never-panic fixpoint, telemetry, never a crash); backends without
//! a native ConvTranspose2D kernel realize the recipe. Parity is gated by a
//! real-backend numerical test vs the native CPU kernel (the col2im math
//! reorders the accumulation vs the kernel's `vec_dot`, so `rel<1e-5`
//! sabotage-calibrated, not bit-exact) — see
//! `fuel-core/tests/incc_conv_transpose_im2col_oracle.rs`.
//!
//! The matcher is stubbed for the same reason: ConvTranspose2D nodes
//! originate from `Tensor::conv_transpose2d` (and from `Conv2D`'s
//! backward `dX` formula); there is no user-decomposed form to
//! recognize as `Op::Fused(CONV_TRANSPOSE2D, _)`.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for ConvTranspose2D.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::CONV_TRANSPOSE2D,
        name: "ConvTranspose2D",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // ConvTranspose2D's backward isn't implemented today (per the
        // legacy `Op::ConvTranspose2D { .. }` arm in `Tensor::backward`
        // — it panics with a clear "needs the dilation-as-stride trick
        // + a real consumer" message). When higher-order gradients
        // are needed, that arm will switch to BackwardKind::Decompose
        // or wire a dedicated backward helper. For now NotDifferentiable
        // mirrors the actual runtime behavior.
        backward: BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Output shape rule. ConvTranspose2D's formula is the inverse of
/// Conv2D's:
///   `Hout = (H − 1)·s − 2·p + d·(Kh − 1) + out_pad + 1`
///   (and analogously for width).
fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    // 2 or 3, mirroring conv2d: the op's contract (`fused/conv-rope.fkc.md`
    // conv_transpose2d) declares an OPTIONAL bias operand and the registered
    // CPU wrapper seeds the output with it — the graph builder emits the
    // 2-input (no-bias) form today, but the rule fns must accept the
    // documented with-bias arity too (the FKC return cross-check probes it).
    debug_assert!(
        input_shapes.len() == 2 || input_shapes.len() == 3,
        "ConvTranspose2D takes 2 or 3 inputs (x, weight, [bias])",
    );
    let (stride, padding, output_padding, dilation, groups) = match params {
        FusedOpParams::ConvTranspose2D {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => (*stride, *padding, *output_padding, *dilation, *groups),
        _ => panic!("conv_transpose_2d::shape_rule got non-ConvTranspose2D params: {params:?}"),
    };
    let x_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert_eq!(x_dims.len(), 4, "ConvTranspose2D x must be rank 4");
    debug_assert_eq!(w_dims.len(), 4, "ConvTranspose2D weight must be rank 4");
    let (n, _cin, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    // Weight is `[Cin, Cout/groups, Kh, Kw]` for transposed conv.
    let (_cin_w, cout_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    let cout = cout_per_g * groups;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let (dh, dw) = dilation;
    let h_out = (h_in.saturating_sub(1)) * sh + dh * (kh.saturating_sub(1)) + oph + 1;
    let h_out = h_out.saturating_sub(2 * ph);
    let w_out = (w_in.saturating_sub(1)) * sw + dw * (kw.saturating_sub(1)) + opw + 1;
    let w_out = w_out.saturating_sub(2 * pw);
    Shape::from_dims(&[n, cout, h_out, w_out])
}

/// Dtype rule: output dtype equals input 0 (x) dtype.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    // 2 or 3 — see the arity note on `shape_rule` (optional bias operand).
    debug_assert!(
        input_dtypes.len() == 2 || input_dtypes.len() == 3,
        "ConvTranspose2D takes 2 or 3 inputs (x, weight, [bias])",
    );
    input_dtypes[0]
}

/// Build the 1-D `[Kh·Kw·Hin·Win]` `U32` scatter index over the FULL
/// (pre-crop) flattened output-spatial axis — the col2im column→flat-target
/// map, shared verbatim by the groups=1 [`recipe`] and the grouped
/// [`recipe_grouped`]. The index is the SAME tensor regardless of grouping
/// (grouping only reshapes the column stack and the weight; the per-column
/// scatter target is identical), so both recipes emit byte-identical index
/// sub-DAGs.
///
/// The column axis `l = ((ky·Kw + kx)·Hin + ih)·Win + iw` maps to the flat
/// full-buffer position `(ih·sh + ky·dh)·Wbuf + (iw·sw + kx·dw)`, assembled as a
/// sum of four per-axis terms — `Op::Iota(axis_len)` reshaped to a rank-4
/// broadcastable form, scaled by its stride/dilation factor (`MulScalar`),
/// broadcast to `[Kh,Kw,Hin,Win]` — then flattened to 1-D and `Op::Cast(U32)`.
/// Integer values are exact in F32 for the full-buffer extent `< 2^24`. This
/// mirrors the native CPU ConvTranspose2D kernel's `out = inp·stride + k·dilation`
/// scatter positions exactly (the crop, applied by [`col2im_tail`], subtracts
/// `padding` and drops `output_padding` overflow).
#[allow(clippy::too_many_arguments)]
fn scatter_flat_index(
    kh: usize,
    kw: usize,
    hin: usize,
    win: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    wbuf: usize,
) -> PatternNode {
    use OpTag as T;
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs: Box::new(attrs),
        operands,
    };
    let shape_attr = |dims: Vec<i64>| OpAttrs {
        target_shape: dims,
        ..OpAttrs::default()
    };

    let l = kh * kw * hin * win;
    let idx4_shape = vec![kh as i64, kw as i64, hin as i64, win as i64];
    // A per-axis term: Iota(axis_len) -> Reshape(rank-4 broadcastable)
    //   -> MulScalar(factor) -> BroadcastTo([Kh,Kw,Hin,Win]).
    let term = |axis_len: usize, reshape_dims: Vec<i64>, factor: f64| -> PatternNode {
        let iota = op(
            T::Iota,
            OpAttrs {
                target_shape: vec![axis_len as i64],
                ..OpAttrs::default()
            },
            vec![],
        );
        let re = op(T::Reshape, shape_attr(reshape_dims), vec![iota]);
        let scaled = op(
            T::MulScalar,
            OpAttrs {
                scalars: vec![factor],
                ..OpAttrs::default()
            },
            vec![re],
        );
        op(T::BroadcastTo, shape_attr(idx4_shape.clone()), vec![scaled])
    };
    let term_ky = term(kh, vec![kh as i64, 1, 1, 1], (dh * wbuf) as f64);
    let term_kx = term(kw, vec![1, kw as i64, 1, 1], dw as f64);
    let term_ih = term(hin, vec![1, 1, hin as i64, 1], (sh * wbuf) as f64);
    let term_iw = term(win, vec![1, 1, 1, win as i64], sw as f64);
    let add = |a, b| op(T::Add, OpAttrs::default(), vec![a, b]);
    let idx4 = add(add(add(term_ky, term_kx), term_ih), term_iw);
    let idx1 = op(T::Reshape, shape_attr(vec![l as i64]), vec![idx4]);
    op(
        T::Cast,
        OpAttrs {
            cast_dtype: Some("u32".to_string()),
            ..OpAttrs::default()
        },
        vec![idx1],
    )
}

/// The col2im tail — the overlap-add scatter + crop, shared by both the
/// groups=1 [`recipe`] and the grouped [`recipe_grouped`]. Given the reshaped
/// column stack `csrc [N, Cout, L]` (`L = Kh·Kw·Hin·Win`) and the [`scatter_flat_index`]
/// map `sidx [L]`:
///
/// ```text
///   base0 = MulScalar(0.0)(BroadcastTo([N,Cout,Hbuf*Wbuf])(Slice(csrc, dim=2, 0, 1)))
///   acc   = IndexAdd(dim=2)(base0, sidx, csrc)      # `+=` IS the overlap-add
///   acc4  = Reshape([N,Cout,Hbuf,Wbuf])(acc)
///   out   = Slice(dim=3, pw, Wout)(Slice(dim=2, ph, Hout)(acc4))
/// ```
///
/// The zero base is built without a data-carrying `Const` (a recipe has no
/// device handle): a length-1 slice of `csrc` broadcast to the full spatial
/// extent and multiplied by 0 — the `selective_scan` idiom (`0 · finite = 0`;
/// the column stack is a finite matmul result). `Op::IndexAdd`'s `base[.., idx[j], ..]
/// += src[.., j, ..]` accumulate semantics ARE the overlap-add: two columns that
/// scatter to the same buffer position sum. The final two `Op::Slice`s crop the
/// full buffer to `[N, Cout, Hout, Wout]` (dropping the `padding` border and any
/// `output_padding` overflow), matching the native kernel's `out -= padding`
/// bounds check.
#[allow(clippy::too_many_arguments)]
fn col2im_tail(
    csrc: PatternNode,
    sidx: PatternNode,
    n: usize,
    cout: usize,
    hbuf: usize,
    wbuf: usize,
    ph: usize,
    pw: usize,
    hout: usize,
    wout: usize,
) -> PatternNode {
    use OpTag as T;
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs: Box::new(attrs),
        operands,
    };
    let shape_attr = |dims: Vec<i64>| OpAttrs {
        target_shape: dims,
        ..OpAttrs::default()
    };
    let sbuf = (hbuf * wbuf) as i64;

    // Zero base [N, Cout, Hbuf*Wbuf] — no `Const` in a recipe (selective_scan idiom).
    let zero_col = op(
        T::Slice,
        OpAttrs {
            axis: Some(2),
            slice_start: Some(0),
            slice_len: Some(1),
            ..OpAttrs::default()
        },
        vec![csrc.clone()],
    );
    let base_b = op(
        T::BroadcastTo,
        shape_attr(vec![n as i64, cout as i64, sbuf]),
        vec![zero_col],
    );
    let base0 = op(
        T::MulScalar,
        OpAttrs {
            scalars: vec![0.0],
            ..OpAttrs::default()
        },
        vec![base_b],
    );

    // Overlap-add scatter: IndexAdd(dim=2)(base, index, src).
    let acc = op(
        T::IndexAdd,
        OpAttrs {
            axis: Some(2),
            ..OpAttrs::default()
        },
        vec![base0, sidx, csrc],
    );

    // Reshape to [N, Cout, Hbuf, Wbuf]; crop rows [ph, ph+Hout), cols [pw, pw+Wout).
    let acc4 = op(
        T::Reshape,
        shape_attr(vec![n as i64, cout as i64, hbuf as i64, wbuf as i64]),
        vec![acc],
    );
    let crop_h = op(
        T::Slice,
        OpAttrs {
            axis: Some(2),
            slice_start: Some(ph as u64),
            slice_len: Some(hout as u64),
            ..OpAttrs::default()
        },
        vec![acc4],
    );
    op(
        T::Slice,
        OpAttrs {
            axis: Some(3),
            slice_start: Some(pw as u64),
            slice_len: Some(wout as u64),
            ..OpAttrs::default()
        },
        vec![crop_h],
    )
}

/// ConvTranspose2D's col2im recipe for the **groups=1** case as a per-call-baked
/// `PatternNode` (Increment C im2col-3, CV3). Everything is concrete at decompose
/// time (all input extents are read off the node), so — like `selective_scan`'s
/// per-call `recipe(seqlen,…)` — the recipe bakes ABSOLUTE `OpAttrs`.
///
/// Bind space: `0 = x [N,Cin,Hin,Win]`, `1 = weight [Cin,Cout,Kh,Kw]`. Emitted
/// subgraph (all existing primitives):
///
/// ```text
///   xf   = Reshape([N,Cin,Hin*Win])(x)
///   Wp   = Permute([Cout,Kh,Kw,Cin])(weight)              # perm [1,2,3,0]
///   Wb   = BroadcastTo([N,Cout*Kh*Kw,Cin])(Reshape([1,Cout*Kh*Kw,Cin])(Wp))
///   cols = MatMul(Wb, xf)                                 # [N, Cout*Kh*Kw, Hin*Win]
///   csrc = Reshape([N,Cout, Kh*Kw*Hin*Win])(cols)         # column stack
///   ... col2im_tail (scatter + crop) -> [N,Cout,Hout,Wout]
/// ```
///
/// `cols[n,(co,ky,kx),(ih,iw)] = Σ_ci weight[ci,co,ky,kx]·x[n,ci,ih,iw]` — the
/// transposed-conv column contribution — via a `weightᵀ`-arranged batched matmul
/// (the `Permute` is materialized row-major by the executor's auto-Contiguize
/// before the reshape). Adds NO `Op` variant.
#[allow(clippy::too_many_arguments)]
fn recipe(
    n: usize,
    cin: usize,
    hin: usize,
    win: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    ph: usize,
    pw: usize,
    hbuf: usize,
    wbuf: usize,
    hout: usize,
    wout: usize,
) -> PatternNode {
    use OpTag as T;
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs: Box::new(attrs),
        operands,
    };
    let bind = |i| PatternNode::Bind { index: i };
    let shape_attr = |dims: Vec<i64>| OpAttrs {
        target_shape: dims,
        ..OpAttrs::default()
    };

    let m = cout * kh * kw; // matmul free dim (Cout·Kh·Kw)
    let s = hin * win; // input spatial (Hin·Win)
    let l = kh * kw * hin * win; // column axis

    // x -> [N, Cin, Hin*Win].
    let xf = op(
        T::Reshape,
        shape_attr(vec![n as i64, cin as i64, s as i64]),
        vec![bind(0)],
    );
    // weight [Cin,Cout,Kh,Kw] -> Permute [Cout,Kh,Kw,Cin] -> [1,M,Cin] -> [N,M,Cin].
    let wp = op(
        T::Permute,
        OpAttrs {
            perm: vec![1, 2, 3, 0],
            ..OpAttrs::default()
        },
        vec![bind(1)],
    );
    let wr = op(
        T::Reshape,
        shape_attr(vec![1, m as i64, cin as i64]),
        vec![wp],
    );
    let wb = op(
        T::BroadcastTo,
        shape_attr(vec![n as i64, m as i64, cin as i64]),
        vec![wr],
    );
    // cols [N, M, S]; reshape to the column stack [N, Cout, L].
    let cols = op(T::MatMul, OpAttrs::default(), vec![wb, xf]);
    let csrc = op(
        T::Reshape,
        shape_attr(vec![n as i64, cout as i64, l as i64]),
        vec![cols],
    );

    let sidx = scatter_flat_index(kh, kw, hin, win, sh, sw, dh, dw, wbuf);
    col2im_tail(csrc, sidx, n, cout, hbuf, wbuf, ph, pw, hout, wout)
}

/// ConvTranspose2D's col2im recipe for the **grouped** case (`groups>1`,
/// including depthwise `groups=Cin`) — Increment C im2col-3 (CV3). Carries the
/// group as an explicit batch axis: `Cin` splits into `(groups, Cin/g)`, `Cout`
/// into `(groups, Cout/g)`, and a **rank-4 batched `Op::MatMul` over `[N, groups]`**
/// stacks each group's columns against its own weight slice. Everything is
/// concrete at decompose time, so this bakes ABSOLUTE `OpAttrs`.
///
/// Bind space: `0 = x [N,Cin,Hin,Win]`, `1 = weight [Cin,Cout/g,Kh,Kw]`. Emitted
/// subgraph (all existing primitives):
///
/// ```text
///   xf   = Reshape([N, g, Cin/g, Hin*Win])(x)
///   Wp   = Permute([g,Cout/g,Kh,Kw,Cin/g])(Reshape([g,Cin/g,Cout/g,Kh,Kw])(weight))
///   Wb   = BroadcastTo([N,g,(Cout/g)*Kh*Kw,Cin/g])(Reshape([1,g,(Cout/g)*Kh*Kw,Cin/g])(Wp))
///   cols = MatMul(Wb, xf)                                 # [N, g, (Cout/g)*Kh*Kw, Hin*Win]
///   csrc = Reshape([N, Cout, Kh*Kw*Hin*Win])(cols)        # per-group column stack
///   ... col2im_tail (scatter + crop) -> [N,Cout,Hout,Wout]
/// ```
///
/// The `Cin = g·(Cin/g)` and `Cout = g·(Cout/g)` splits are plain row-major
/// reshapes (channel `c = group·per_g + in_group`), so the group index lines up
/// on both matmul operands and the `csrc` reshape re-merges `(g, Cout/g) → Cout`
/// and `(Kh,Kw,Hin,Win) → L` byte-identically to the groups=1 stack. The scatter
/// index and crop ([`col2im_tail`]) are identical to the groups=1 path. Adds NO
/// `Op` variant.
#[allow(clippy::too_many_arguments)]
fn recipe_grouped(
    n: usize,
    cin: usize,
    hin: usize,
    win: usize,
    cout: usize,
    groups: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    ph: usize,
    pw: usize,
    hbuf: usize,
    wbuf: usize,
    hout: usize,
    wout: usize,
) -> PatternNode {
    use OpTag as T;
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs: Box::new(attrs),
        operands,
    };
    let bind = |i| PatternNode::Bind { index: i };
    let shape_attr = |dims: Vec<i64>| OpAttrs {
        target_shape: dims,
        ..OpAttrs::default()
    };

    let cin_per_g = cin / groups;
    let cout_per_g = cout / groups;
    let m = cout_per_g * kh * kw; // per-group matmul free dim
    let s = hin * win; // input spatial
    let l = kh * kw * hin * win; // column axis

    // x -> [N, g, Cin/g, Hin*Win].
    let xf = op(
        T::Reshape,
        shape_attr(vec![n as i64, groups as i64, cin_per_g as i64, s as i64]),
        vec![bind(0)],
    );
    // weight [Cin,Cout/g,Kh,Kw] -> [g,Cin/g,Cout/g,Kh,Kw] -> Permute
    //   [g,Cout/g,Kh,Kw,Cin/g] -> [1,g,M,Cin/g] -> [N,g,M,Cin/g].
    let wr5 = op(
        T::Reshape,
        shape_attr(vec![
            groups as i64,
            cin_per_g as i64,
            cout_per_g as i64,
            kh as i64,
            kw as i64,
        ]),
        vec![bind(1)],
    );
    let wp = op(
        T::Permute,
        OpAttrs {
            perm: vec![0, 2, 3, 4, 1],
            ..OpAttrs::default()
        },
        vec![wr5],
    );
    let wr = op(
        T::Reshape,
        shape_attr(vec![1, groups as i64, m as i64, cin_per_g as i64]),
        vec![wp],
    );
    let wb = op(
        T::BroadcastTo,
        shape_attr(vec![n as i64, groups as i64, m as i64, cin_per_g as i64]),
        vec![wr],
    );
    // cols [N, g, M, S]; reshape to the merged column stack [N, Cout, L].
    let cols = op(T::MatMul, OpAttrs::default(), vec![wb, xf]);
    let csrc = op(
        T::Reshape,
        shape_attr(vec![n as i64, cout as i64, l as i64]),
        vec![cols],
    );

    let sidx = scatter_flat_index(kh, kw, hin, win, sh, sw, dh, dw, wbuf);
    col2im_tail(csrc, sidx, n, cout, hbuf, wbuf, ph, pw, hout, wout)
}

/// Total decomposition of ConvTranspose2D via the col2im scatter-add recipe
/// (Increment C im2col-3, CV3) — a re-emit of [`recipe`] / [`recipe_grouped`]'s
/// portable data through the [`decompose_via_recipe`] bridge. ConvTranspose2D is
/// **NOT** a primitive-basis gap (correcting the earlier "needs `Op::Col2Im`"
/// claim; see the module note and the `10-decisions-log.md` 2026-07-24 addendum):
/// col2im is the scatter-add adjoint of im2col's gather (`Op::IndexAdd`), fully
/// expressible in the closed `Op` basis.
///
/// **Scope: any `groups>=1`** (with the builder's 2-input, no-bias arity). The
/// stride/padding/output_padding/dilation/groups and the concrete
/// `Cin/Hin/Win/Cout/Kh/Kw` extents are read here and baked into the per-call
/// recipe; `groups==1` uses the rank-3 [`recipe`] and `groups>1` the rank-4
/// batched [`recipe_grouped`].
///
/// Per G2 (2026-06-20) this is total + never-panic: a wrong-params payload, a
/// non-2-input arity (the with-bias form is not yet in the recipe — a surfaced
/// honest miss), a malformed input shape, an indivisible grouping (`Cin` not a
/// multiple of `groups`, or the weight's `Cin` disagreeing), a zero
/// stride/dilation, or any bridge decline returns `id` — the driver's fixpoint
/// signal, a surfaced opaque-op gap, never a crash. The recipe is the *math* the
/// kernel computes; the fused CPU/CUDA ConvTranspose2D kernel stays the executed
/// production path (the optimizer chooses by cost), and the recipe realizes on
/// any backend with the primitive set for the base-map cover and the numeric
/// verify oracle.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (stride, padding, output_padding, dilation, groups) = match params {
        FusedOpParams::ConvTranspose2D {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        } => (*stride, *padding, *output_padding, *dilation, *groups),
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };
    if groups == 0 {
        return id;
    }

    // The builder emits the 2-input (x, weight) form; a with-bias 3-input form is
    // not yet in the recipe → surfaced honest-miss fixpoint (never a crash).
    let (x_shape, w_shape) = {
        let node = graph.node(id);
        if node.inputs.len() != 2 {
            return id;
        }
        (
            graph.node(node.inputs[0]).shape.clone(),
            graph.node(node.inputs[1]).shape.clone(),
        )
    };
    let x_dims = x_shape.dims();
    let w_dims = w_shape.dims();
    if x_dims.len() != 4 || w_dims.len() != 4 {
        return id;
    }
    let (n, cin, hin, win) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    // Weight is `[Cin, Cout/groups, Kh, Kw]` for transposed conv.
    let (cin_w, cout_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    // Grouping must be well-formed: weight's Cin equals x's Cin and Cin divisible
    // by groups (Cout = cout_per_g·groups is divisible by construction).
    if cin_w != cin || cin % groups != 0 {
        return id;
    }
    let cout = cout_per_g * groups;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let (dh, dw) = dilation;
    if sh == 0 || sw == 0 || dh == 0 || dw == 0 {
        return id;
    }
    // Cropped output dims (mirror `shape_rule`).
    let hout = ((hin.saturating_sub(1)) * sh + dh * (kh.saturating_sub(1)) + oph + 1)
        .saturating_sub(2 * ph);
    let wout = ((win.saturating_sub(1)) * sw + dw * (kw.saturating_sub(1)) + opw + 1)
        .saturating_sub(2 * pw);
    if hout == 0 || wout == 0 {
        return id;
    }
    // Full pre-crop scatter buffer: all raw scatter positions
    // (`(Hin-1)·sh + (Kh-1)·dh` max) plus the `output_padding` overflow, so the
    // crop window `[ph, ph+Hout)` always lands inside it.
    let hbuf = (hin.saturating_sub(1)) * sh + dh * (kh.saturating_sub(1)) + 1 + oph;
    let wbuf = (win.saturating_sub(1)) * sw + dw * (kw.saturating_sub(1)) + 1 + opw;
    if ph + hout > hbuf || pw + wout > wbuf {
        return id;
    }

    let recipe_node = if groups == 1 {
        recipe(
            n, cin, hin, win, cout, kh, kw, sh, sw, dh, dw, ph, pw, hbuf, wbuf, hout, wout,
        )
    } else {
        recipe_grouped(
            n, cin, hin, win, cout, groups, kh, kw, sh, sw, dh, dw, ph, pw, hbuf, wbuf, hout, wout,
        )
    };
    // No open scalar slots — every MulScalar carries a baked factor / zero.
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Matcher stub — see module preamble.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    fn mk_const(g: &mut Graph, dims: &[usize]) -> NodeId {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        })
    }

    /// Build a fused CONV_TRANSPOSE2D node over `x [N,Cin,Hin,Win]`,
    /// `weight [Cin, Cout/groups, Kh, Kw]` (2-input, no bias — what the
    /// builder emits).
    #[allow(clippy::too_many_arguments)]
    fn fused_conv_transpose(
        g: &mut Graph,
        n: usize,
        cin: usize,
        hin: usize,
        win: usize,
        cout_per_g: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> NodeId {
        let x = mk_const(g, &[n, cin, hin, win]);
        let w = mk_const(g, &[cin, cout_per_g, kh, kw]);
        let cout = cout_per_g * groups;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (oph, opw) = output_padding;
        let (dh, dw) = dilation;
        let hout = (hin - 1) * sh + dh * (kh - 1) + oph + 1 - 2 * ph;
        let wout = (win - 1) * sw + dw * (kw - 1) + opw + 1 - 2 * pw;
        g.push(Node {
            op: Op::Fused(
                FusedOps::CONV_TRANSPOSE2D,
                FusedOpParams::ConvTranspose2D {
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                },
            ),
            inputs: vec![x, w],
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

    /// Assert the lowered subgraph is the col2im recipe: no `Op::Fused`
    /// survives, and the scatter-adjoint primitives are present
    /// (`MatMul` for the column stack, `IndexAdd` for the overlap-add,
    /// `Iota`+`Cast(U32)` for the scatter index, `Slice` for the crop).
    fn assert_col2im_recipe(g: &Graph, root: NodeId, label: &str) {
        let ops = reachable_ops(g, root);
        assert!(
            !ops.iter()
                .any(|o| matches!(o, Op::Fused(fid, _) if *fid == FusedOps::CONV_TRANSPOSE2D)),
            "{label}: ConvTranspose2D must lower to primitives, not remain fused",
        );
        assert!(
            ops.iter().any(|o| matches!(o, Op::MatMul)),
            "{label}: recipe contains MatMul"
        );
        assert!(
            ops.iter().any(|o| matches!(o, Op::IndexAdd { .. })),
            "{label}: col2im overlap-add via IndexAdd",
        );
        assert!(
            ops.iter().any(|o| matches!(o, Op::Iota { .. })),
            "{label}: scatter index built from Iota"
        );
        assert!(
            ops.iter()
                .any(|o| matches!(o, Op::Cast(dt) if *dt == DType::U32)),
            "{label}: scatter index cast to U32",
        );
        assert!(
            ops.iter().any(|o| matches!(o, Op::Slice { .. })),
            "{label}: output cropped via Slice",
        );
    }

    /// CV3 (im2col-3): a groups=1 CONV_TRANSPOSE2D node decomposes to the
    /// col2im scatter-add recipe — NO `Op::Fused(CONV_TRANSPOSE2D)` survives.
    /// Born-red while `decompose` self-returns (`root == fused`).
    #[test]
    fn conv_transpose2d_groups1_decompose_lowers_to_col2im() {
        // (n, cin, hin, win, cout_per_g, kh, kw, stride, padding, out_pad, dilation)
        #[allow(clippy::type_complexity)]
        let cases: [(
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            (usize, usize),
            (usize, usize),
            (usize, usize),
            (usize, usize),
        ); 3] = [
            (1, 2, 4, 4, 3, 2, 2, (1, 1), (0, 0), (0, 0), (1, 1)), // plain
            (1, 3, 5, 4, 2, 3, 3, (2, 2), (1, 1), (1, 1), (1, 1)), // stride+pad+outpad
            (2, 2, 3, 3, 4, 2, 2, (2, 1), (0, 0), (0, 0), (2, 1)), // stride + dilation
        ];
        for (n, cin, hin, win, cpg, kh, kw, stride, padding, out_pad, dilation) in cases {
            let mut g = Graph::new();
            let fused = fused_conv_transpose(
                &mut g, n, cin, hin, win, cpg, kh, kw, stride, padding, out_pad, dilation, 1,
            );
            let out_shape = g.node(fused).shape.clone();
            let params = match &g.node(fused).op {
                Op::Fused(_, p) => p.clone(),
                other => panic!("expected fused node, got {other:?}"),
            };
            let root = decompose(&mut g, fused, &params);
            assert_ne!(
                root, fused,
                "groups=1 must lower ({stride:?}/{padding:?}/{out_pad:?}/{dilation:?})"
            );
            assert_eq!(
                g.node(root).shape,
                out_shape,
                "lowered root keeps [N,Cout,Hout,Wout]"
            );
            assert_eq!(g.node(root).dtype, DType::F32);
            assert_col2im_recipe(&g, root, "groups=1");
        }
    }

    /// CV3 (im2col-3): a groups>1 CONV_TRANSPOSE2D node (a grouped conv AND a
    /// depthwise transpose `groups=Cin`) decomposes to the batched col2im recipe.
    /// Born-red while `decompose` self-returns for `groups>1`.
    #[test]
    fn conv_transpose2d_groups_gt_1_decompose_lowers_to_col2im() {
        // (n, cin, hin, win, cout_per_g, kh, kw, stride, padding, out_pad, dilation, groups)
        #[allow(clippy::type_complexity)]
        let cases: [(
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            (usize, usize),
            (usize, usize),
            (usize, usize),
            (usize, usize),
            usize,
        ); 3] = [
            (1, 4, 4, 4, 3, 2, 2, (1, 1), (0, 0), (0, 0), (1, 1), 2), // groups=2
            (2, 4, 3, 3, 1, 2, 2, (2, 2), (0, 0), (1, 1), (1, 1), 4), // depthwise groups=Cin
            (1, 6, 5, 5, 2, 3, 3, (1, 1), (1, 1), (0, 0), (1, 1), 3), // grouped, multiplier
        ];
        for (n, cin, hin, win, cpg, kh, kw, stride, padding, out_pad, dilation, groups) in cases {
            let mut g = Graph::new();
            let fused = fused_conv_transpose(
                &mut g, n, cin, hin, win, cpg, kh, kw, stride, padding, out_pad, dilation, groups,
            );
            let out_shape = g.node(fused).shape.clone();
            let params = match &g.node(fused).op {
                Op::Fused(_, p) => p.clone(),
                other => panic!("expected fused node, got {other:?}"),
            };
            let root = decompose(&mut g, fused, &params);
            assert_ne!(root, fused, "groups={groups} must lower");
            assert_eq!(
                g.node(root).shape,
                out_shape,
                "lowered root keeps [N,Cout,Hout,Wout] (groups={groups})"
            );
            assert_eq!(g.node(root).dtype, DType::F32);
            assert_col2im_recipe(&g, root, "grouped");
        }
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint — never a
    /// crash — before any emission.
    #[test]
    fn conv_transpose2d_wrong_params_is_a_fixpoint() {
        let mut g = Graph::new();
        let fused = fused_conv_transpose(
            &mut g,
            1,
            2,
            4,
            4,
            3,
            2,
            2,
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
            1,
        );
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }

    /// Totality (G2): a malformed grouping the builder would never emit — `Cin`
    /// not divisible by `groups` — declines to a fixpoint, never a crash.
    #[test]
    fn conv_transpose2d_bad_groups_is_a_fixpoint() {
        let mut g = Graph::new();
        // cin=5, groups=2 ⇒ cin not divisible by groups — a malformed node.
        let x = mk_const(&mut g, &[1, 5, 4, 4]);
        let w = mk_const(&mut g, &[5, 3, 2, 2]);
        let params = FusedOpParams::ConvTranspose2D {
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            dilation: (1, 1),
            groups: 2,
        };
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::CONV_TRANSPOSE2D, params.clone()),
            inputs: vec![x, w],
            shape: Shape::from_dims(&[1, 6, 5, 5]),
            dtype: DType::F32,
        });
        let before = g.len();
        let out = decompose(&mut g, fused, &params);
        assert_eq!(out, fused, "Cin not divisible by groups => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
