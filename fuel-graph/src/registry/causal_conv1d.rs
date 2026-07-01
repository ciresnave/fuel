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
//! ## Architectural note — depthwise, so it DOES decompose
//!
//! Unlike [`super::conv2d`] (which mixes channels and is a genuine
//! `Op::Im2Col` basis gap), CausalConv1d is **depthwise**: each output
//! channel convolves only its own input channel, so it lowers to an
//! `O(kernel)` per-channel shift-multiply-accumulate tap sum (`Slice → Mul →
//! Add`), NOT the `O(kernel·seq)` node explosion an earlier note claimed
//! (that confused element count with node count). [`decompose`] emits this
//! real primitive subgraph per G2; the fused kernel is the *fast* path the
//! cost-guided optimizer prefers when present (and `cpu_fallback` covers
//! backends without one), but the decomposition is always available.
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
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

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

/// Decompose the depthwise causal conv into an `O(kernel)` shift-multiply-
/// accumulate tap sum (NOT `O(kernel·seq)` — the old module note confused
/// element count with node count). Because the conv is **depthwise**
/// (`weight [C,1,K]`, one filter per channel, no channel mixing), there is no
/// `Im2Col`/`MatMul` basis gap like `conv2d` — every tap is a per-channel
/// `Slice → Mul → Add`. Inputs `[x, weight, bias]` with `x` pre-padded to
/// `[B, C, seq+(K-1)]`; output `[B, C, seq]`:
///
/// `out[t] = Σ_{k<K} weight[:,0,k] · x[:, :, t+k] + bias`, then optional SiLU.
///
/// Every primitive exists (`Slice`, `Reshape`, `BroadcastTo`, `Mul`, `Add`,
/// `Silu`), so per G2 this is a real decomposition (~`5K+3` nodes; Mamba's
/// `K=4` → ~23), not a basis-gap self-return.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (x_id, w_id, b_id, out_shape, dtype) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.inputs[2], n.shape.clone(), n.dtype)
    };
    let use_silu = match params {
        FusedOpParams::CausalConv1d { use_silu } => *use_silu,
        // G2: total + never-panic — impossible params; return self.
        _ => return id,
    };
    let out_dims = out_shape.dims().to_vec(); // [B, C, out_seq]
    let channels = out_dims[1];
    let out_seq = out_dims[2];
    let kernel = graph.node(w_id).shape.dims()[2]; // weight is [C, 1, K]
    let per_channel = Shape::from_dims(&[1, channels, 1]);
    let full = out_shape.clone();

    // acc = Σ_k weight[:,0,k] · x[:, :, k : k+out_seq]
    let mut acc: Option<NodeId> = None;
    for tap in 0..kernel {
        let x_k = graph.push(Node {
            op: Op::Slice {
                dim: 2,
                start: tap,
                len: out_seq,
            },
            inputs: vec![x_id],
            shape: full.clone(),
            dtype,
        });
        let w_k = graph.push(Node {
            op: Op::Slice {
                dim: 2,
                start: tap,
                len: 1,
            },
            inputs: vec![w_id],
            shape: Shape::from_dims(&[channels, 1, 1]),
            dtype,
        });
        let w_re = graph.push(Node {
            op: Op::Reshape(per_channel.clone()),
            inputs: vec![w_k],
            shape: per_channel.clone(),
            dtype,
        });
        let w_b = graph.push(Node {
            op: Op::BroadcastTo(full.clone()),
            inputs: vec![w_re],
            shape: full.clone(),
            dtype,
        });
        let term = graph.push(Node {
            op: Op::Mul,
            inputs: vec![x_k, w_b],
            shape: full.clone(),
            dtype,
        });
        acc = Some(match acc {
            None => term,
            Some(a) => graph.push(Node {
                op: Op::Add,
                inputs: vec![a, term],
                shape: full.clone(),
                dtype,
            }),
        });
    }
    let acc = acc.expect("CausalConv1d kernel size is ≥ 1");

    // + bias  (broadcast [C] → [1, C, 1] → full)
    let b_re = graph.push(Node {
        op: Op::Reshape(per_channel.clone()),
        inputs: vec![b_id],
        shape: per_channel,
        dtype,
    });
    let b_b = graph.push(Node {
        op: Op::BroadcastTo(full.clone()),
        inputs: vec![b_re],
        shape: full.clone(),
        dtype,
    });
    let biased = graph.push(Node {
        op: Op::Add,
        inputs: vec![acc, b_b],
        shape: full.clone(),
        dtype,
    });

    if use_silu {
        graph.push(Node {
            op: Op::Silu,
            inputs: vec![biased],
            shape: full,
            dtype,
        })
    } else {
        biased
    }
}

/// Matcher stub — CausalConv1d nodes originate from the explicit
/// `Tensor::causal_conv1d` builder. No primitive subgraph pattern to
/// auto-fuse (would require an `Op::Conv1D + Add + Silu` chain
/// pattern, but Op::Conv1D isn't in fuel-graph's primitive set).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
