//! ConvTranspose2D — 2-D transposed (fractionally-strided)
//! convolution. Phase 7.6 step 4 (continued — seventh op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   self-returning `decompose` pending the col2im-recipe migration,
//!   stubbed pattern).
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
//! Until that recipe lands (a follow-up slice on this branch), the
//! [`decompose`] below **self-returns** (the G2 total + never-panic
//! fixpoint, a surfaced honest-miss, never a crash); backends without a
//! native ConvTranspose2D kernel route through
//! `GraphExecutor::cpu_fallback`.
//!
//! The matcher is stubbed for the same reason: ConvTranspose2D nodes
//! originate from `Tensor::conv_transpose2d` (and from `Conv2D`'s
//! backward `dX` formula); there is no user-decomposed form to
//! recognize as `Op::Fused(CONV_TRANSPOSE2D, _)`.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for ConvTranspose2D.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::CONV_TRANSPOSE2D,
        name:       "ConvTranspose2D",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        // ConvTranspose2D's backward isn't implemented today (per the
        // legacy `Op::ConvTranspose2D { .. }` arm in `Tensor::backward`
        // — it panics with a clear "needs the dilation-as-stride trick
        // + a real consumer" message). When higher-order gradients
        // are needed, that arm will switch to BackwardKind::Decompose
        // or wire a dedicated backward helper. For now NotDifferentiable
        // mirrors the actual runtime behavior.
        backward:   BackwardKind::NotDifferentiable,
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
            stride, padding, output_padding, dilation, groups,
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
    let h_out = (h_in.saturating_sub(1)) * sh
        + dh * (kh.saturating_sub(1))
        + oph + 1;
    let h_out = h_out.saturating_sub(2 * ph);
    let w_out = (w_in.saturating_sub(1)) * sw
        + dw * (kw.saturating_sub(1))
        + opw + 1;
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

/// ConvTranspose2D migrates via a col2im (overlap-add) scatter-add recipe built
/// entirely from existing primitives — it is **NOT** a primitive-basis gap
/// (correcting the earlier "genuine G2 basis gap, needs `Op::Col2Im`/`Op::Im2Col`"
/// claim; see the module note and the `10-decisions-log.md` 2026-07-24 addendum).
/// Transposed conv = `Op::MatMul(weightᵀ-arranged, x)` producing a
/// `[N, Cout·Kh·Kw, Hin·Win]` column stack, then **col2im** = the adjoint of
/// im2col's gather = a scatter-add: `Op::IndexAdd` (`+=` IS the overlap-add) into
/// a zero-init base (`MulScalar(0.0)` of a broadcast), with the scatter-index map
/// built from `Op::Iota` + arithmetic + `Op::Cast(U32)` (`stride`/`output_padding`/
/// `dilation` fold into it), then an `Op::Slice` crop to `[N, Cout, Hout, Wout]`.
/// Node count is constant in the spatial size — the `Slice`+`ScatterAdd` soup the
/// superseded note weighed was not the only recipe.
///
/// Until that recipe migration lands (a follow-up slice on this branch), this
/// returns **self** — the G2 total + never-panic fixpoint (a surfaced
/// opaque-op honest-miss, never a crash); backends without a native kernel use
/// `GraphExecutor::cpu_fallback`.
pub fn decompose(_graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    id
}

/// Matcher stub — see module preamble.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
