//! Conv2D — 2-D cross-correlation with stride / padding / groups.
//! Phase 7.6 step 4 (continued — sixth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! ## Architectural note — no primitive decomposition (yet)
//!
//! Unlike SoftmaxLastDim or RmsNormLastDim, Conv2D **has no clean
//! decomposition into the current primitive set**. The textbook
//! im2col-then-matmul lowering requires an `Op::Im2Col` (or equivalent
//! `Op::Unfold` / `Op::AsStrided`) primitive that hasn't been
//! introduced — every backend that supports Conv2D either has a native
//! kernel (CPU, CUDA cuDNN, AOCL) or does im2col internally inside its
//! own kernel (Vulkan, MKL) without ever surfacing it as a graph node.
//!
//! Synthesizing Conv2D from `Op::Slice` + `Op::MatMul` + `Op::Concat`
//! is technically possible but creates `N·Hout·Wout` slice operations
//! — astronomical node count that would actively harm any optimization
//! pass that consumed it. So [`decompose`] **panics** with a clear
//! pointer to this gap, and the lowering rule's typical "decompose to
//! primitives for backends without a native kernel" fallback is
//! replaced by the executor's `cpu_fallback` path until either:
//!
//! 1. an `Op::Im2Col` primitive lands (step 10 territory; would let
//!    the lowering rule produce a meaningful primitive subgraph), or
//! 2. the registry's `decompose` field becomes `Option<fn(...)>` so
//!    a fused op can architecturally declare "no primitive
//!    decomposition exists" (a follow-up architecture commit).
//!
//! What this commit DOES contribute toward the planned architecture:
//! - the Conv2D builder routes through `Op::Fused(CONV2D, _)`, so the
//!   primitive `Op` enum loses one more variant in step 5;
//! - CPU Conv2D kernels register as per-decision-point `BackendImpl`s
//!   in `fuel_storage::fused::FusedKernelRegistry` (the same shape
//!   FusedLinear uses), populating the alternative-set substrate that
//!   step 9 reads for pre-resolved KernelRef dispatch;
//! - CSE / `op_key` automatically dedupes via [`super::FusedOpParams`]'s
//!   stride/padding/groups payload.
//!
//! The matcher is also stubbed (returns `None`) — Conv2D nodes
//! originate from the `Tensor::conv2d` builder; user-decomposed forms
//! don't exist as a pattern to recognize.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

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

/// Lowering panics with a clear architectural message — see the module
/// preamble for the full rationale. In practice this is unreachable
/// because no current code path enables `RuleRegistry::default_rules`
/// or `lowering_only` on a Conv2D-bearing graph; if a future caller
/// does, this panic surfaces the architectural gap rather than silently
/// emitting nonsense or returning the same id.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "conv2d::decompose: Conv2D has no primitive decomposition in \
         the current Op set. The textbook im2col + matmul lowering \
         needs an Op::Im2Col primitive that isn't part of the closed \
         primitive set yet; the slice-soup alternative is \
         astronomically expensive. Backends without a native Conv2D \
         kernel route through `GraphExecutor::cpu_fallback` to the \
         always-built CPU kernel. See `fuel-graph/src/registry/conv2d.rs` \
         module docs for the full picture.",
    );
}

/// Matcher stub — Conv2D is always produced by the `Tensor::conv2d`
/// builder; there is no user-decomposed pattern to recognize as
/// `Op::Fused(CONV2D, _)`.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
