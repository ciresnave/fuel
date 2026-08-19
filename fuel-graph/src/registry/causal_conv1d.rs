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
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for CausalConv1d.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::CAUSAL_CONV1D,
        name: "CausalConv1d",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
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
        input_shapes.len(),
        3,
        "CausalConv1d takes 3 inputs (x, weight, bias)",
    );
    let x_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert_eq!(
        x_dims.len(),
        3,
        "CausalConv1d: x must be rank 3 [batch, channels, seq+pad], got {x_dims:?}"
    );
    debug_assert_eq!(
        w_dims.len(),
        3,
        "CausalConv1d: weight must be rank 3 [channels, 1, kernel], got {w_dims:?}"
    );
    let batch = x_dims[0];
    let channels = x_dims[1];
    let x_seq = x_dims[2];
    let kernel = w_dims[2];
    debug_assert!(
        x_seq >= kernel - 1,
        "CausalConv1d: x time dim {x_seq} must be ≥ kernel - 1 = {} \
         (caller must pre-pad with {} zeros)",
        kernel - 1,
        kernel - 1,
    );
    let out_seq = x_seq - (kernel - 1);
    Shape::from_dims(&[batch, channels, out_seq])
}

/// Dtype rule: output dtype matches input 0 (x). All three inputs
/// must agree at construction time (the builder validates).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(),
        3,
        "CausalConv1d takes 3 inputs (x, weight, bias)",
    );
    input_dtypes[0]
}

/// CausalConv1d's depthwise shift-multiply-accumulate primitive recipe as
/// portable [`PatternNode`] DATA (Increment C) — the structure-preserving
/// migration of the pre-Increment-C imperative `decompose` body onto the
/// re-emit machinery. It is a per-call `recipe(out_dims, channels, out_seq,
/// kernel, use_silu) -> PatternNode` builder that uses ORDINARY Rust control
/// flow (the config-branch mechanism) to select structure and bake concrete
/// counts — no new bridge machinery:
///   * the **extent-driven tap unroll** is a Rust `for tap in 0..kernel` loop
///     (`kernel` = the concrete weight extent read at decompose time), emitting
///     one shifted `Slice → Reshape → BroadcastTo → Mul` term per tap and
///     folding them left-associatively with `Add` — exactly the imperative
///     `for tap` body;
///   * the optional SiLU tail is a Rust `if use_silu`.
///
/// Because the conv is **depthwise** (`weight [C,1,K]`, one filter per channel,
/// no channel mixing), every tap is a per-channel `Slice → Mul → Add` — there
/// is no `Im2Col`/`MatMul` basis gap like `conv2d`.
///
/// Bind space: `0 = x [B,C,seq+(K-1)]`, `1 = weight [C,1,K]`, `2 = bias [C]` —
/// the fused node's input order. Every shape is a concrete integer at decompose
/// time, so the shape-changer ops (`Reshape`/`BroadcastTo`) and the taps'
/// `Slice`s carry BAKED absolute `target_shape`/`slice_*` attrs (the flash_attn
/// / FSCE posture) — NO open scalar slots, NO shape-relative (`SameAs`/`WithDim`/
/// `DimExpr`) carriers, NO C-4 `DimExpr::Param` threading. Each `BroadcastTo` is
/// preceded by an equal-rank `Reshape` (`[C,1,K]`-slice → `[1,C,1]`, `[C]`-bias →
/// `[1,C,1]`), so emit's D4 auto-pad never fires and the emitted base map is
/// node-for-node identical to the imperative body.
///
/// The lowered form (`full = [B, C, out_seq]`, `out_seq = x_seq − (K−1)`):
///
/// ```text
///   for tap in 0..K:
///     x_k  = Slice{dim:2, start:tap, len:out_seq}(x)      # [B, C, out_seq]
///     w_k  = Slice{dim:2, start:tap, len:1}(weight)       # [C, 1, 1]
///     w_re = Reshape([1, C, 1])(w_k)
///     w_b  = BroadcastTo(full)(w_re)
///     term = Mul(x_k, w_b)
///     acc  = term            (tap 0)  |  Add(acc, term)   (tap > 0)
///   b_re   = Reshape([1, C, 1])(bias)
///   b_b    = BroadcastTo(full)(b_re)
///   biased = Add(acc, b_b)
///   out    = Silu(biased)  if use_silu  else  biased
/// ```
///
/// The caller ([`decompose`]) guarantees `kernel >= 1`, so the accumulator fold
/// always yields at least one tap (the `.expect` is unreachable — it mirrors the
/// imperative body's `acc.expect("CausalConv1d kernel size is ≥ 1")`).
fn recipe(
    out_dims: &[usize],
    channels: usize,
    out_seq: usize,
    kernel: usize,
    use_silu: bool,
) -> PatternNode {
    use OpTag as T;
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs,
        operands,
    };
    let bind = |i: u8| PatternNode::Bind { index: i };
    // A baked absolute `target_shape` attr for a shape-changer (Reshape/BroadcastTo).
    let shape_attr = |dims: &[usize]| OpAttrs {
        target_shape: dims.iter().map(|&d| d as i64).collect(),
        ..OpAttrs::default()
    };
    // A baked concrete `Slice { dim, start, len }` attr (`tag_to_op` reads these).
    let slice_attr = |dim: i64, start: u64, len: u64| OpAttrs {
        axis: Some(dim),
        slice_start: Some(start),
        slice_len: Some(len),
        ..OpAttrs::default()
    };
    let per_channel = [1usize, channels, 1];

    // acc = Σ_k weight[:,0,k] · x[:, :, k : k+out_seq]  — the extent-driven unroll.
    let mut acc: Option<PatternNode> = None;
    for tap in 0..kernel {
        let x_k = op(
            T::Slice,
            slice_attr(2, tap as u64, out_seq as u64),
            vec![bind(0)],
        );
        let w_k = op(T::Slice, slice_attr(2, tap as u64, 1), vec![bind(1)]);
        let w_re = op(T::Reshape, shape_attr(&per_channel), vec![w_k]);
        let w_b = op(T::BroadcastTo, shape_attr(out_dims), vec![w_re]);
        let term = op(T::Mul, OpAttrs::default(), vec![x_k, w_b]);
        acc = Some(match acc {
            None => term,
            Some(a) => op(T::Add, OpAttrs::default(), vec![a, term]),
        });
    }
    let acc = acc.expect("CausalConv1d kernel size is ≥ 1");

    // + bias  (broadcast [C] → [1, C, 1] → full)
    let b_re = op(T::Reshape, shape_attr(&per_channel), vec![bind(2)]);
    let b_b = op(T::BroadcastTo, shape_attr(out_dims), vec![b_re]);
    let biased = op(T::Add, OpAttrs::default(), vec![acc, b_b]);

    if use_silu {
        op(T::Silu, OpAttrs::default(), vec![biased])
    } else {
        biased
    }
}

/// Decompose the depthwise causal conv into an `O(kernel)` shift-multiply-
/// accumulate tap sum (NOT `O(kernel·seq)` — the old module note confused
/// element count with node count). Since Increment C a re-emit of [`recipe`]'s
/// portable data through the [`decompose_via_recipe`] bridge (structure-
/// preserving: the emitted base map is node-for-node identical to the
/// pre-Increment-C imperative body — see the parity test in `tests`). The
/// per-call recipe bakes the concrete shapes and drives the extent-driven tap
/// unroll + the SiLU tail via ordinary Rust control flow (the config-branch
/// mechanism). Inputs `[x, weight, bias]` with `x` pre-padded to
/// `[B, C, seq+(K-1)]`; output `[B, C, seq]`:
///
/// `out[t] = Σ_{k<K} weight[:,0,k] · x[:, :, t+k] + bias`, then optional SiLU.
///
/// Every primitive exists (`Slice`, `Reshape`, `BroadcastTo`, `Mul`, `Add`,
/// `Silu`), so per G2 this is a real decomposition (~`5K+3` nodes; Mamba's
/// `K=4` → ~23), not a basis-gap self-return.
///
/// Per G2 this is total + never-panic: a wrong-params payload, a malformed node
/// (wrong input arity, non-rank-3 output/weight, zero-tap weight), or any bridge
/// decline (validation, bind-arity, emit) returns `id` (the driver's fixpoint
/// signal) rather than crash.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let use_silu = match params {
        FusedOpParams::CausalConv1d { use_silu } => *use_silu,
        // G2: total + never-panic — impossible params; return self.
        _ => return id,
    };
    let (out_dims, kernel) = {
        let n = graph.node(id);
        // Malformed node → fixpoint self-return (never panic).
        if n.inputs.len() != 3 {
            return id;
        }
        let out_dims = n.shape.dims().to_vec(); // [B, C, out_seq]
        let w_dims = graph.node(n.inputs[1]).shape.dims().to_vec(); // weight [C, 1, K]
        if out_dims.len() != 3 || w_dims.len() != 3 {
            return id;
        }
        (out_dims, w_dims[2])
    };
    // kernel >= 1 (weight carries at least one tap); a zero-tap weight is a
    // malformed node whose accumulator fold would be empty — decline to a
    // fixpoint (never panic) rather than build an empty recipe.
    if kernel == 0 {
        return id;
    }
    let channels = out_dims[1];
    let out_seq = out_dims[2];

    let recipe_node = recipe(&out_dims, channels, out_seq, kernel, use_silu);
    // No open scalar slots (every extent/shape is baked per call).
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Matcher stub — CausalConv1d nodes originate from the explicit
/// `Tensor::causal_conv1d` builder. No primitive subgraph pattern to
/// auto-fuse (would require an `Op::Conv1D + Add + Silu` chain
/// pattern, but Op::Conv1D isn't in fuel-graph's primitive set).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    /// FROZEN copy of the pre-Increment-C imperative `causal_conv1d::decompose`
    /// body, verbatim (the explicit-`Shape` `graph.push` spelling), before the
    /// migration replaced the live body with the data recipe. The structure-
    /// preservation oracle: the migrated recipe re-emit must produce a graph
    /// structurally identical to this.
    fn frozen_legacy_causal_conv1d_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, w_id, b_id, out_shape, dtype) = {
            let n = graph.node(id);
            (
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.shape.clone(),
                n.dtype,
            )
        };
        let use_silu = match params {
            FusedOpParams::CausalConv1d { use_silu } => *use_silu,
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

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, recursively-equal inputs). A shared leaf (same NodeId — a
    /// bound external input) matches by identity. Shape/dtype-sensitive at EVERY
    /// node, so it catches any structural drift the recipe migration introduces
    /// (a wrong slice start, a missing tap, a lost bias broadcast, a dropped
    /// SiLU tail, an accidental D4 auto-pad).
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

    /// Build a fused CAUSAL_CONV1D node over `x [b,c,seq+(k-1)]` (caller
    /// pre-pads by `k-1`), `weight [c,1,k]`, `bias [c]` — all F32 `Op::Const`
    /// leaves. Output `[b,c,seq]`. Returns the fused NodeId.
    fn fused_node(
        g: &mut Graph,
        b: usize,
        c: usize,
        seq: usize,
        kernel: usize,
        use_silu: bool,
    ) -> NodeId {
        let x_seq = seq + (kernel - 1);
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[b, c, x_seq]),
            dtype: DType::F32,
        });
        let w = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[c, 1, kernel]),
            dtype: DType::F32,
        });
        let bias = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[c]),
            dtype: DType::F32,
        });
        g.push(Node {
            op: Op::Fused(
                FusedOps::CAUSAL_CONV1D,
                FusedOpParams::CausalConv1d { use_silu },
            ),
            inputs: vec![x, w, bias],
            shape: Shape::from_dims(&[b, c, seq]),
            dtype: DType::F32,
        })
    }

    /// ONE recipe form decomposes across a representative matrix — a couple of
    /// kernel sizes (`K=1` exercises the single-tap no-`Add` fold, `K>1` the
    /// left-associated `Add` accumulator; both prove the extent-driven unroll),
    /// with/without SiLU, and a couple of channel/length/batch shapes — and its
    /// emitted base map is node-for-node identical to the FROZEN pre-migration
    /// imperative body (the structure-preservation contract). Born-red while the
    /// recipe is absent (`decompose` declines to a fixpoint ⇒ `assert_ne` trips).
    #[test]
    fn causal_conv1d_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
        // (b, c, seq)
        let shapes: [(usize, usize, usize); 4] = [
            (1, 2, 3), // Mamba-like: 1 batch, 2 channels, 3 out steps
            (1, 3, 1), // single out step (out_seq == 1)
            (2, 2, 4), // 2 batches
            (1, 1, 5), // single channel
        ];
        let mut fired = 0usize;
        for kernel in [1usize, 2, 4] {
            for use_silu in [false, true] {
                for (b, c, seq) in shapes {
                    let mut g = Graph::new();
                    let fused = fused_node(&mut g, b, c, seq, kernel, use_silu);
                    let out_shape = g.node(fused).shape.clone();
                    let params = FusedOpParams::CausalConv1d { use_silu };

                    let new_root = decompose(&mut g, fused, &params);
                    assert_ne!(
                        new_root, fused,
                        "recipe decompose fires (kernel={kernel}, use_silu={use_silu}, \
                         b={b}, c={c}, seq={seq})"
                    );
                    assert_eq!(
                        g.node(new_root).shape,
                        out_shape,
                        "output shape matches shape_rule (kernel={kernel}, use_silu={use_silu})"
                    );
                    assert_eq!(g.node(new_root).dtype, DType::F32, "output dtype is F32");

                    let legacy_root = frozen_legacy_causal_conv1d_decompose(&mut g, fused, &params);
                    assert_structural_eq(&g, new_root, legacy_root);
                    fired += 1;
                }
            }
        }
        assert_eq!(fired, 3 * 2 * 4, "every config in the matrix was checked");
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn causal_conv1d_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_node(&mut g, 1, 2, 3, 4, true);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }

    /// Totality (G2): a malformed node the builder would never emit — a zero-tap
    /// weight `[C, 1, 0]` — declines to a fixpoint, never a crash (the empty
    /// accumulator fold would otherwise `.expect`-panic).
    #[test]
    fn causal_conv1d_zero_tap_weight_is_a_fixpoint() {
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[1, 2, 3]),
            dtype: DType::F32,
        });
        let w = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 1, 0]), // zero taps — malformed
            dtype: DType::F32,
        });
        let bias = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        let params = FusedOpParams::CausalConv1d { use_silu: false };
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::CAUSAL_CONV1D, params.clone()),
            inputs: vec![x, w, bias],
            shape: Shape::from_dims(&[1, 2, 3]),
            dtype: DType::F32,
        });
        let before = g.len();
        let out = decompose(&mut g, fused, &params);
        assert_eq!(out, fused, "zero-tap weight => fixpoint self-return");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
