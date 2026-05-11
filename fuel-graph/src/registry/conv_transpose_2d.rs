//! ConvTranspose2D — 2-D transposed (fractionally-strided)
//! convolution. Phase 7.6 step 4 (continued — seventh op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! ## Architectural note — no primitive decomposition (yet)
//!
//! Same gap as Conv2D — there is no `Op::Im2Col` (or `Op::Col2Im`)
//! primitive that could express ConvTranspose2D as a small primitive
//! subgraph. The textbook "scatter into a strided/padded buffer +
//! matmul + crop" lowering would produce astronomical node counts on
//! anything beyond trivial shapes. Backends without a native
//! ConvTranspose2D kernel route through `GraphExecutor::cpu_fallback`
//! instead. See the Conv2D registry entry's module docs for the full
//! discussion of this primitive-set gap.
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
use fuel_core_types::{DType, Shape};

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
    }
}

/// Output shape rule. ConvTranspose2D's formula is the inverse of
/// Conv2D's:
///   `Hout = (H − 1)·s − 2·p + d·(Kh − 1) + out_pad + 1`
///   (and analogously for width).
fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "ConvTranspose2D takes 2 inputs (x, weight)",
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
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "ConvTranspose2D takes 2 inputs",
    );
    input_dtypes[0]
}

/// See module preamble — ConvTranspose2D has no primitive
/// decomposition exposed at the registry layer until an
/// `Op::Im2Col` / `Op::Col2Im` primitive lands.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "conv_transpose_2d::decompose: ConvTranspose2D has no \
         primitive decomposition in the current Op set. See \
         `fuel-graph/src/registry/{{conv2d,conv_transpose_2d}}.rs` \
         module docs for the shared im2col-primitive gap.",
    );
}

/// Matcher stub — see module preamble.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
