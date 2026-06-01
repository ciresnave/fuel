//! CausalConv1d — depthwise 1-D convolution + causal masking + optional
//! fused SiLU activation. Second FusedOpRegistry entry added by the
//! re-framed CPU OpKind coverage plan (after FusedSoftmaxCrossEntropy).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[x, weight, bias]`.
//!   - `x`:      `[batch, channels, seq + (kernel - 1)]` — caller
//!     pre-pads with `kernel - 1` zeros on the left for the causal mask.
//!     This matches Mamba-2's prefill convention (mamba2.rs:552 builds
//!     the pad explicitly via `Tensor::cat`).
//!   - `weight`: `[channels, 1, kernel]` — depthwise (one filter per
//!     channel; `groups == channels` in standard conv terminology).
//!   - `bias`:   `[channels]` — required (matches baracuda's
//!     `causal_conv1d_*_run` signature, which has bias as a required
//!     argument; callers without a bias pass a zero vector).
//!
//! Output: `[batch, channels, seq]`, same dtype as inputs. Output time
//! dim is `x_seq - (kernel - 1) = seq`.
//!
//! ## Why this exists (the win)
//!
//! Mamba-2's prefill convolution
//! ([fuel-transformers/src/models/llm/mamba2.rs:554-558](../../../fuel-transformers/src/models/llm/mamba2.rs#L554-L558))
//! is currently a three-op chain: `conv1d + broadcast_add(bias) +
//! silu`. Three kernel launches per layer × N layers per forward call.
//! A fused kernel collapses this to one launch per layer.
//!
//! Note: Mamba's *autoregressive* paths
//! ([mamba.rs:188-194](../../../fuel-transformers/src/models/llm/mamba.rs#L188-L194)
//! and [mamba2.rs:342-356](../../../fuel-transformers/src/models/llm/mamba2.rs#L342-L356))
//! use hand-rolled state-ring-buffer loops and are NOT in scope for
//! this fusion — they need in-place state mutation, which a pure
//! forward fused op can't express.
//!
//! ## Architectural note — no primitive decomposition
//!
//! Mirrors [`super::conv2d`]'s precedent: `fuel-graph` has no
//! `Op::Conv1D` primitive (only `Op::Conv2D`), so a primitive
//! decomposition would require either (a) Reshape→Conv2D→Reshape
//! gymnastics around a unit spatial dim, or (b) Slice + Mul + Sum
//! chains with `kernel * seq` node count. Both are antipatterns. The
//! fused kernel IS the implementation; backends without one fall
//! through to the executor's `cpu_fallback` path. [`decompose`]
//! panics with a clear pointer to this gap, same as Conv2D.
//!
//! ## Why `BackwardKind::NotDifferentiable` for v1
//!
//! Mamba's lazy migration ([docs/session-prompts/mamba-eager-to-lazy-migration.md])
//! is inference-only. Without a backward consumer, training-time
//! gradient support is premature. The backward formula (dX via
//! "transposed" causal conv; dW via cross-correlation; dB via
//! reduce-sum along batch×time) is mechanical to add when the first
//! Mamba training consumer materializes.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for CausalConv1d.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::CAUSAL_CONV1D,
        name:       "CausalConv1d",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output is `[batch, channels, seq]` where `seq =
/// x.dims[2] - (kernel - 1)`. `kernel` is read from the weight shape
/// (weight is `[channels, 1, kernel]`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 3,
        "CausalConv1d takes 3 inputs (x, weight, bias)",
    );
    let x_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert_eq!(x_dims.len(), 3, "CausalConv1d: x must be rank 3 [batch, channels, seq+pad], got {x_dims:?}");
    debug_assert_eq!(w_dims.len(), 3, "CausalConv1d: weight must be rank 3 [channels, 1, kernel], got {w_dims:?}");
    let batch = x_dims[0];
    let channels = x_dims[1];
    let x_seq = x_dims[2];
    let kernel = w_dims[2];
    debug_assert!(
        x_seq >= kernel - 1,
        "CausalConv1d: x time dim {x_seq} must be ≥ kernel - 1 = {} \
         (caller must pre-pad with {} zeros)", kernel - 1, kernel - 1,
    );
    let out_seq = x_seq - (kernel - 1);
    Shape::from_dims(&[batch, channels, out_seq])
}

/// Dtype rule: output dtype matches input 0 (x). All three inputs
/// must agree at construction time (the builder validates).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 3,
        "CausalConv1d takes 3 inputs (x, weight, bias)",
    );
    input_dtypes[0]
}

/// See module preamble — CausalConv1d deliberately has no primitive
/// decomposition. The cpu_fallback path handles backends without a
/// native kernel.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "causal_conv1d::decompose: CausalConv1d has no registry-layer \
         decomposition. fuel-graph doesn't carry an Op::Conv1D primitive, \
         and synthesizing the depthwise conv from Slice + Mul + Sum chains \
         would create kernel*seq nodes — an antipattern for any optimizer. \
         Backends without a native CausalConv1d kernel use the executor's \
         cpu_fallback path. See conv2d::decompose for the same precedent.",
    );
}

/// Matcher stub — CausalConv1d nodes originate from the explicit
/// `Tensor::causal_conv1d` builder. No primitive subgraph pattern to
/// auto-fuse (would require an `Op::Conv1D + Add + Silu` chain
/// pattern, but Op::Conv1D isn't in fuel-graph's primitive set).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
