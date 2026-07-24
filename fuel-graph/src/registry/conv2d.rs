//! Conv2D — 2-D cross-correlation with stride / padding / groups.
//! Phase 7.6 step 4 (continued — sixth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   self-returning `decompose` pending the im2col-recipe migration,
//!   stubbed pattern).
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
//! Until that recipe lands (a follow-up slice on this branch), the
//! [`decompose`] below **self-returns** — the G2 total + never-panic
//! fixpoint, a surfaced honest-miss (telemetry), never a crash — and
//! backends without a native Conv2D kernel route through
//! `GraphExecutor::cpu_fallback`.
//!
//! The matcher is also stubbed (returns `None`) — Conv2D nodes
//! originate from the `Tensor::conv2d` builder; user-decomposed forms
//! don't exist as a pattern to recognize.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};

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

/// Conv2D migrates via an index-gather im2col recipe built entirely from
/// existing primitives — it is **NOT** a primitive-basis gap (correcting the
/// earlier "genuine G2 basis gap, needs `Op::Im2Col`" claim; see the module
/// note and the `10-decisions-log.md` 2026-07-24 addendum). The clean lowering
/// is im2col + batched matmul: `Op::Pad` the input, flatten the spatial axis,
/// gather the overlapping `Kh×Kw` windows into a `[N, Cin·Kh·Kw, Hout·Wout]`
/// patch matrix with a single `Op::IndexSelect` (its 1-D index built from
/// `Op::Iota` + scalar arithmetic + `Op::Cast(U32)`, node count constant in the
/// spatial size), `Op::MatMul` against the reshaped weight `[Cout, Cin·Kh·Kw]`
/// (batched over groups), then `Op::Reshape` to `[N, Cout, Hout, Wout]` (bias
/// via a broadcast `Add`). The `Slice`+`Concat`+`MatMul` soup of `N·Hout·Wout`
/// slice nodes was NOT the only decomposition, as the superseded note assumed —
/// it is the one the index-gather idiom replaces.
///
/// Until that recipe migration lands (a follow-up slice on this branch), this
/// returns **self** — the G2 total + never-panic fixpoint (a surfaced
/// honest-miss, telemetry, never a crash). Backends without a native Conv2D
/// kernel route through `GraphExecutor::cpu_fallback`.
pub fn decompose(_graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    id
}

/// Matcher stub — Conv2D is always produced by the `Tensor::conv2d`
/// builder; there is no user-decomposed pattern to recognize as
/// `Op::Fused(CONV2D, _)`.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
