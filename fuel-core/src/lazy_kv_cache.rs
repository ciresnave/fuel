//! `LazyKvCache` — functional KV cache backed by [`LazyTensor`].
//!
//! Phase B of the eager-`Tensor` retirement program
//! ([`docs/session-prompts/eager-tensor-retirement-master-plan.md`](
//! ../../../docs/session-prompts/eager-tensor-retirement-master-plan.md)).
//! User-locked decision: **option (b)** — `append()` consumes `self`
//! and returns a new cache instead of mutating in place. Costs more
//! consumer churn but gives a clean graph: every cache state is a
//! distinct value, no shared mutable reference, no `&mut` threading
//! through the forward pass.
//!
//! # Shape contract
//!
//! All buffers laid out as `[max_seq, n_kv_heads, head_dim]` per layer
//! (no batch dim — the typical decode-time KV cache is per-sequence and
//! the caller broadcasts/concats across batches as needed). The
//! sequence axis is always dim 0.
//!
//! Per-layer K/V buffers are held as separate [`LazyTensor`]s so an
//! `append` to layer `l` only emits one [`Op::WriteSlice`] node per
//! K/V, not one per layer × KV. The cost is `Vec` capacity of
//! `2 * n_layers` `LazyTensor`s — cheap since each is a `(graph, id)`
//! pair behind an `Arc`.
//!
//! # Lifecycle and graph anchoring
//!
//! The cache is **per-forward-pass**: every cache buffer is a node on a
//! single [`fuel_graph::Graph`]. Cross-step decoding therefore requires
//! either:
//!
//! 1. Re-creating the cache on each step's graph (`new` on the new
//!    forward's anchor; the previous step's realized K/V values get
//!    rebound via [`LazyTensor::const_f32_like`] — same pattern the
//!    existing eight migrated lazy ports use), or
//! 2. Holding the cache's realized K/V in host buffers between steps
//!    and re-uploading on the next forward.
//!
//! Persistent storage-backed cross-graph re-anchoring (the pattern
//! [`crate::inference_context::KvCache`] uses for its
//! `forward_with_kv_context` consumer) is a documented follow-up — it
//! requires `Arc<RwLock<Storage>>` plumbing through
//! [`crate::inference_context::InferenceContext`] that's already in
//! place for option (a)-style consumers but adds enough complexity to
//! warrant its own session.
//!
//! # Why "consumes self" instead of `&mut self`
//!
//! With option (a)'s `&mut self`, the cache's identity persists across
//! calls — but graph-side that means we hold a mutable reference into
//! a graph whose nodes the executor is also reading. Option (b)'s
//! `self -> Self` makes every cache state a distinct value, which:
//!
//! - Lets autograd snapshots cache versions trivially (each version is
//!   its own [`LazyTensor`] graph).
//! - Eliminates the "is this cache's slice valid right now?"
//!   ambiguity — every cache is the latest one to the holder.
//! - Composes cleanly with future graph-optimization passes that
//!   rewrite the post-write buffer NodeId in place: the rewrite sees
//!   one logical owner, the most recent cache.

use crate::{DType, Device, lazy::LazyTensor};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Per-forward-pass KV cache. See module docs for lifecycle and shape
/// contract.
#[derive(Clone, Debug)]
pub struct LazyKvCache {
    /// Per-layer (K, V) buffers. `layers[l].0` is K, `layers[l].1` is V.
    layers: Vec<(LazyTensor, LazyTensor)>,
    /// Sequence positions filled so far. Slices returned by [`Self::k`]
    /// / [`Self::v`] narrow to `[..current_seq_len]` along dim 0.
    current_seq_len: usize,
    /// Pre-allocated capacity per layer. Exceeding this on `append`
    /// returns a typed error rather than reallocating (the buffer is a
    /// `Op::WriteSlice` target; resizing would need a new buffer node).
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl LazyKvCache {
    /// Allocate a zero-filled cache on the same graph as `anchor`.
    /// The cache lives on the anchor's graph; appending K/V from a
    /// different graph will panic with the standard cross-graph error.
    ///
    /// `dtype` selects the cache's element type — typically matches the
    /// K/V tensors produced by the forward pass (commonly F32 / BF16 /
    /// F16 for inference workloads).
    pub fn new(
        anchor: &LazyTensor,
        n_layers: usize,
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Self {
        assert!(n_layers >= 1, "LazyKvCache::new: n_layers must be ≥ 1");
        assert!(max_seq_len >= 1, "LazyKvCache::new: max_seq_len must be ≥ 1");
        let buffer_shape = Shape::from_dims(&[max_seq_len, n_kv_heads, head_dim]);
        let buffer_elems = max_seq_len * n_kv_heads * head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let k = zero_const_on(anchor, dtype, buffer_shape.clone(), buffer_elems);
            let v = zero_const_on(anchor, dtype, buffer_shape.clone(), buffer_elems);
            layers.push((k, v));
        }
        Self {
            layers,
            current_seq_len: 0,
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    /// Append `k_new` / `v_new` for layer `layer` at the cache's current
    /// position. **Consumes `self`** and returns the updated cache.
    ///
    /// Input shapes: `k_new` and `v_new` must both be
    /// `[seqlen_new, n_kv_heads, head_dim]` where `seqlen_new` is the
    /// number of tokens being appended (1 for the typical decode step,
    /// `prefill_len` for the prefill step). Their last two dims must
    /// match the cache's `n_kv_heads` and `head_dim`.
    ///
    /// The cache's `current_seq_len` advances by `seqlen_new` after the
    /// append; subsequent layer appends in the same step land at the
    /// same position (the position only moves forward across step
    /// boundaries, not across per-layer appends within a step). Call
    /// [`Self::advance_by`] after the last layer's append in each step
    /// to bump the position for the next step.
    ///
    /// Returns an error if `layer >= n_layers`, shapes don't match, or
    /// the append would exceed `max_seq_len`.
    pub fn append(
        mut self,
        layer: usize,
        k_new: &LazyTensor,
        v_new: &LazyTensor,
    ) -> crate::Result<Self> {
        if layer >= self.layers.len() {
            crate::bail!(
                "LazyKvCache::append: layer {layer} out of bounds (n_layers={})",
                self.layers.len(),
            );
        }
        let k_dims = k_new.shape().dims().to_vec();
        let v_dims = v_new.shape().dims().to_vec();
        if k_dims.len() != 3 || v_dims.len() != 3 {
            crate::bail!(
                "LazyKvCache::append: k_new/v_new must be rank 3, got k={k_dims:?} v={v_dims:?}",
            );
        }
        if k_dims != v_dims {
            crate::bail!(
                "LazyKvCache::append: k_new shape {k_dims:?} != v_new shape {v_dims:?}",
            );
        }
        let seqlen_new = k_dims[0];
        if k_dims[1] != self.n_kv_heads || k_dims[2] != self.head_dim {
            crate::bail!(
                "LazyKvCache::append: k_new shape {k_dims:?} doesn't match cache geometry \
                 [n_kv_heads={}, head_dim={}]",
                self.n_kv_heads, self.head_dim,
            );
        }
        let new_end = self.current_seq_len + seqlen_new;
        if new_end > self.max_seq_len {
            crate::bail!(
                "LazyKvCache::append: appending {seqlen_new} tokens at position {} would \
                 exceed max_seq_len {}",
                self.current_seq_len, self.max_seq_len,
            );
        }
        let ranges = vec![
            (self.current_seq_len, new_end),
            (0, self.n_kv_heads),
            (0, self.head_dim),
        ];
        let (k_buffer, v_buffer) = self.layers[layer].clone();
        let new_k = k_buffer.write_slice(k_new, ranges.clone())?;
        let new_v = v_buffer.write_slice(v_new, ranges)?;
        self.layers[layer] = (new_k, new_v);
        Ok(self)
    }

    /// Advance `current_seq_len` by `n` positions. Call this after the
    /// last layer's [`Self::append`] in each generation step.
    ///
    /// Most callers will set `n` to the same `seqlen_new` they appended
    /// in this step (matches across all layers). Separated from
    /// `append` so per-layer appends in one step don't accidentally
    /// move the position past where the cache actually holds data.
    pub fn advance_by(mut self, n: usize) -> Self {
        self.current_seq_len = (self.current_seq_len + n).min(self.max_seq_len);
        self
    }

    /// Slice K-buffer for `layer` to `[0..current_seq_len]` along dim 0.
    /// Returns the slice on the same graph as the cache.
    pub fn k(&self, layer: usize) -> LazyTensor {
        self.layers[layer].0.slice(0_usize, 0, self.current_seq_len.max(1)).unwrap()
    }

    /// Slice V-buffer for `layer` to `[0..current_seq_len]` along dim 0.
    pub fn v(&self, layer: usize) -> LazyTensor {
        self.layers[layer].1.slice(0_usize, 0, self.current_seq_len.max(1)).unwrap()
    }

    /// Underlying full-capacity K-buffer (rank 3, `[max_seq, n_kv_heads,
    /// head_dim]`). Use [`Self::k`] for the active-only slice;
    /// `k_buffer_full` is escape-hatch territory for callers that need
    /// to reason about the underlying buffer (e.g., to copy into
    /// another cache's same-shape buffer).
    pub fn k_buffer_full(&self, layer: usize) -> LazyTensor {
        self.layers[layer].0.clone()
    }

    /// Underlying full-capacity V-buffer; sibling of [`Self::k_buffer_full`].
    pub fn v_buffer_full(&self, layer: usize) -> LazyTensor {
        self.layers[layer].1.clone()
    }

    /// Number of tokens written so far.
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Pre-allocated capacity (per-layer).
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of layers in this cache.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// `n_kv_heads` the cache was constructed with.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// `head_dim` the cache was constructed with.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// Build a zero-initialized [`LazyTensor`] of the given shape/dtype on
/// the same graph as `anchor`. Mirrors [`LazyTensor::zeros_like`] but
/// takes a separate shape rather than copying `anchor`'s, since the
/// cache buffers have a different shape from the anchor.
fn zero_const_on(anchor: &LazyTensor, dtype: DType, shape: Shape, elems: usize) -> LazyTensor {
    match dtype {
        DType::F32 => anchor.const_f32_like(vec![0.0_f32; elems], shape),
        DType::F64 => anchor.const_f64_like(vec![0.0_f64; elems], shape),
        DType::BF16 => anchor.const_bf16_like(vec![half::bf16::ZERO; elems], shape),
        DType::F16 => anchor.const_f16_like(vec![half::f16::ZERO; elems], shape),
        other => panic!("LazyKvCache: unsupported dtype {other:?}"),
    }
}

// Silence the unused-import lint in case the Arc import becomes
// load-bearing later.
#[allow(dead_code)]
fn _arc_marker(_a: Arc<()>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn new_cache_is_empty() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 2, 8, 4, 16, DType::F32);
        assert_eq!(cache.n_layers(), 2);
        assert_eq!(cache.max_seq_len(), 8);
        assert_eq!(cache.current_seq_len(), 0);
        assert_eq!(cache.n_kv_heads(), 4);
        assert_eq!(cache.head_dim(), 16);
    }

    #[test]
    fn append_advances_position() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 1, 4, 2, 3, DType::F32);
        let k_new = anchor.const_f32_like(vec![1.0; 2 * 3], vec![1, 2, 3]);
        let v_new = anchor.const_f32_like(vec![2.0; 2 * 3], vec![1, 2, 3]);
        let cache = cache.append(0, &k_new, &v_new).unwrap();
        let cache = cache.advance_by(1);
        assert_eq!(cache.current_seq_len(), 1);
        let k = cache.k(0);
        assert_eq!(k.shape().dims(), &[1, 2, 3]);
        assert_eq!(k.realize_f32(), vec![1.0; 6]);
        let v = cache.v(0);
        assert_eq!(v.realize_f32(), vec![2.0; 6]);
    }

    #[test]
    fn multi_step_append_concatenates_logically() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 1, 4, 1, 2, DType::F32);
        // Step 1: append 2 tokens.
        let k1 = anchor.const_f32_like(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
        let v1 = anchor.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], vec![2, 1, 2]);
        let cache = cache.append(0, &k1, &v1).unwrap().advance_by(2);
        assert_eq!(cache.current_seq_len(), 2);
        // Step 2: append 1 token.
        let k2 = anchor.const_f32_like(vec![9.0, 10.0], vec![1, 1, 2]);
        let v2 = anchor.const_f32_like(vec![11.0, 12.0], vec![1, 1, 2]);
        let cache = cache.append(0, &k2, &v2).unwrap().advance_by(1);
        assert_eq!(cache.current_seq_len(), 3);
        // K slice should be the 3 appended tokens, end to end.
        let k = cache.k(0);
        assert_eq!(k.shape().dims(), &[3, 1, 2]);
        assert_eq!(k.realize_f32(), vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0]);
    }

    #[test]
    fn multi_layer_append_isolates_layers() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 2, 4, 1, 2, DType::F32);
        // Both layers appended within one step → same position 0 → advance once.
        let k0 = anchor.const_f32_like(vec![1.0, 1.0], vec![1, 1, 2]);
        let v0 = anchor.const_f32_like(vec![2.0, 2.0], vec![1, 1, 2]);
        let cache = cache.append(0, &k0, &v0).unwrap();
        let k1 = anchor.const_f32_like(vec![3.0, 3.0], vec![1, 1, 2]);
        let v1 = anchor.const_f32_like(vec![4.0, 4.0], vec![1, 1, 2]);
        let cache = cache.append(1, &k1, &v1).unwrap().advance_by(1);
        assert_eq!(cache.current_seq_len(), 1);
        assert_eq!(cache.k(0).realize_f32(), vec![1.0, 1.0]);
        assert_eq!(cache.v(0).realize_f32(), vec![2.0, 2.0]);
        assert_eq!(cache.k(1).realize_f32(), vec![3.0, 3.0]);
        assert_eq!(cache.v(1).realize_f32(), vec![4.0, 4.0]);
    }

    #[test]
    fn append_rejects_oob_layer() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 1, 4, 1, 2, DType::F32);
        let k = anchor.const_f32_like(vec![0.0; 2], vec![1, 1, 2]);
        let v = anchor.const_f32_like(vec![0.0; 2], vec![1, 1, 2]);
        assert!(cache.append(5, &k, &v).is_err());
    }

    #[test]
    fn append_rejects_shape_mismatch() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 1, 4, 2, 3, DType::F32);
        let k = anchor.const_f32_like(vec![0.0; 5], vec![1, 5, 1]); // wrong heads, wrong head_dim
        let v = anchor.const_f32_like(vec![0.0; 5], vec![1, 5, 1]);
        assert!(cache.append(0, &k, &v).is_err());
    }

    #[test]
    fn append_rejects_capacity_overflow() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 1, 2, 1, 1, DType::F32);
        let k = anchor.const_f32_like(vec![0.0; 3], vec![3, 1, 1]);
        let v = anchor.const_f32_like(vec![0.0; 3], vec![3, 1, 1]);
        assert!(cache.append(0, &k, &v).is_err()); // 3 tokens > max_seq_len 2
    }

    #[test]
    fn k_buffer_full_returns_max_seq_buffer() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyKvCache::new(&anchor, 1, 5, 1, 2, DType::F32);
        let buf = cache.k_buffer_full(0);
        assert_eq!(buf.shape().dims(), &[5, 1, 2]);
    }
}
