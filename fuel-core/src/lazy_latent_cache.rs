//! `LazyLatentCache` — a general **N-slot** decode cache backed by
//! [`LazyTensor`], the structural generalization of [`crate::lazy_kv_cache::
//! LazyKvCache`] that unblocks latent / pruned KV compression.
//!
//! `LazyKvCache` hardwires a symmetric **K/V pair** — two buffers per
//! layer, both shaped `[max_seq, n_kv_heads, head_dim]`. That shape is
//! wrong for the compression architectures the roadmap targets:
//!
//!   - **Multi-head Latent Attention (DeepSeek-V2 MLA)** caches, per layer,
//!     a low-rank latent `compressed_kv [max_seq, kv_lora_rank]` **and** a
//!     single-head rope key `k_pe [max_seq, qk_rope_head_dim]` — two slots
//!     of *different* trailing shapes, neither a `[n_kv_heads, head_dim]`
//!     K/V. (This is the ~93% cache reduction: cache `kv_lora_rank +
//!     qk_rope_head_dim` per token instead of `2 · n_heads · head_dim`.)
//!   - **Two-projection attention / QKV pruning** caches a *single*
//!     retained projection per layer — one slot.
//!
//! So the container's real degree of freedom is: **per layer, an ordered
//! list of latent buffers, each `[max_seq, …arbitrary trailing dims]`,
//! sharing the sequence axis (dim 0).** The standard K/V cache is then just
//! the two-equal-slots special case; MLA is two-unequal-slots; two-projection
//! is one slot. This type owns that general shape; `LazyKvCache`'s K/V
//! surface can later be re-expressed on top of it (a follow-up — it has live
//! consumers, so it stays as-is for now).
//!
//! # Shape contract
//!
//! Slot `s` of every layer is a buffer `[max_seq, …slot_trailing[s]]` with
//! the sequence axis at dim 0 (matching [`LazyKvCache`]'s no-batch,
//! per-sequence convention; the caller broadcasts/concats across batches).
//! An [`Self::append`] writes a `[seqlen_new, …slot_trailing[s]]` slab into
//! every slot at the cache's current position; all slots in one append
//! share the same `seqlen_new`.
//!
//! # Lifecycle
//!
//! Per-forward-pass and graph-anchored, exactly like [`LazyKvCache`]:
//! every buffer is a node on one [`fuel_graph::Graph`]; cross-step decode
//! either re-creates the cache on the new step's graph (rebinding realized
//! latents via `const_*_like`) or holds realized latents host-side between
//! steps. Persistent storage-backed cross-graph re-anchoring (the
//! [`crate::inference_context::KvCache`] pattern) is the same documented
//! follow-up it is for `LazyKvCache`.

use crate::{DType, Device, lazy::LazyTensor};
use fuel_ir::Shape;

/// Per-forward-pass, N-slot latent cache. See module docs for the shape
/// contract and lifecycle.
#[derive(Clone, Debug)]
pub struct LazyLatentCache {
    /// `layers[l][s]` is slot `s`'s buffer for layer `l`, shaped
    /// `[max_seq, …slot_trailing[s]]`.
    layers: Vec<Vec<LazyTensor>>,
    /// Trailing dims (past the leading seq axis) for each slot; its length
    /// is the per-layer slot count.
    slot_trailing: Vec<Vec<usize>>,
    /// Sequence positions filled so far. [`Self::slot`] narrows to
    /// `[..current_seq_len]` on dim 0.
    current_seq_len: usize,
    /// Pre-allocated per-slot capacity along the sequence axis.
    max_seq_len: usize,
}

impl LazyLatentCache {
    /// Allocate a zero-filled cache on the same graph as `anchor`.
    ///
    /// `slot_trailing` gives the trailing shape (past the seq axis) of each
    /// slot; its length is the number of slots per layer and must be ≥ 1.
    /// A slot's buffer is `[max_seq_len, …trailing]` (an empty trailing ⇒
    /// a `[max_seq_len]` per-token-scalar slot). `dtype` selects the element
    /// type (typically the latent's — F32 / BF16 / F16 for inference).
    pub fn new(
        anchor: &LazyTensor,
        n_layers: usize,
        max_seq_len: usize,
        slot_trailing: Vec<Vec<usize>>,
        dtype: DType,
    ) -> std::result::Result<Self, fuel_ir::Error> {
        if n_layers == 0 {
            return Err(fuel_ir::Error::Msg(
                "LazyLatentCache::new: n_layers must be ≥ 1".into(),
            ).bt());
        }
        if max_seq_len == 0 {
            return Err(fuel_ir::Error::Msg(
                "LazyLatentCache::new: max_seq_len must be ≥ 1".into(),
            ).bt());
        }
        if slot_trailing.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "LazyLatentCache::new: need at least one slot (slot_trailing empty)".into(),
            ).bt());
        }
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut slots = Vec::with_capacity(slot_trailing.len());
            for trailing in &slot_trailing {
                let mut dims = Vec::with_capacity(1 + trailing.len());
                dims.push(max_seq_len);
                dims.extend_from_slice(trailing);
                let shape = Shape::from_dims(&dims);
                let elems: usize = dims.iter().product();
                slots.push(zero_const_on(anchor, dtype, shape, elems)?);
            }
            layers.push(slots);
        }
        Ok(Self {
            layers,
            slot_trailing,
            current_seq_len: 0,
            max_seq_len,
        })
    }

    /// Append fresh latents for `layer` at the cache's current position.
    /// **Consumes `self`** and returns the updated cache (option (b), the
    /// same functional shape as [`LazyKvCache::append`]).
    ///
    /// `new_slots` must have exactly `n_slots` entries, in slot order; entry
    /// `s` must be `[seqlen_new, …slot_trailing[s]]`, and all entries must
    /// agree on `seqlen_new` (the number of tokens appended this step: 1 for
    /// a decode step, `prefill_len` for prefill). Advances the cache's
    /// position implicitly is **not** done here — call [`Self::advance_by`]
    /// after the last layer's append in each step, matching `LazyKvCache`.
    ///
    /// Errors (typed, at build time): `layer` out of bounds, wrong slot
    /// count, a slot shape mismatch, mismatched `seqlen_new` across slots,
    /// or an append that would exceed `max_seq_len`.
    pub fn append(
        mut self,
        layer: usize,
        new_slots: &[&LazyTensor],
    ) -> crate::Result<Self> {
        if layer >= self.layers.len() {
            crate::bail!(
                "LazyLatentCache::append: layer {layer} out of bounds (n_layers={})",
                self.layers.len(),
            );
        }
        let n_slots = self.slot_trailing.len();
        if new_slots.len() != n_slots {
            crate::bail!(
                "LazyLatentCache::append: expected {n_slots} slot tensors, got {}",
                new_slots.len(),
            );
        }
        // Derive seqlen_new from slot 0 and validate every slot against its
        // declared trailing shape + that shared seqlen_new.
        let seqlen_new = {
            let d = new_slots[0].shape().dims().to_vec();
            if d.is_empty() {
                crate::bail!(
                    "LazyLatentCache::append: slot 0 must be rank ≥ 1 (leading seq axis), got {d:?}",
                );
            }
            d[0]
        };
        for (s, tensor) in new_slots.iter().enumerate() {
            let got = tensor.shape().dims().to_vec();
            let mut want = Vec::with_capacity(1 + self.slot_trailing[s].len());
            want.push(seqlen_new);
            want.extend_from_slice(&self.slot_trailing[s]);
            if got != want {
                crate::bail!(
                    "LazyLatentCache::append: slot {s} shape {got:?} != expected {want:?} \
                     (seqlen_new={seqlen_new}, trailing={:?})",
                    self.slot_trailing[s],
                );
            }
        }
        let new_end = self.current_seq_len + seqlen_new;
        if new_end > self.max_seq_len {
            crate::bail!(
                "LazyLatentCache::append: appending {seqlen_new} tokens at position {} would \
                 exceed max_seq_len {}",
                self.current_seq_len, self.max_seq_len,
            );
        }
        // Write each slot's slab at [current_seq_len, new_end) on dim 0,
        // full extent on the trailing dims.
        let mut updated = Vec::with_capacity(n_slots);
        for (s, tensor) in new_slots.iter().enumerate() {
            let mut ranges = Vec::with_capacity(1 + self.slot_trailing[s].len());
            ranges.push((self.current_seq_len, new_end));
            for &d in &self.slot_trailing[s] {
                ranges.push((0, d));
            }
            let buffer = self.layers[layer][s].clone();
            updated.push(buffer.write_slice(tensor, ranges)?);
        }
        self.layers[layer] = updated;
        Ok(self)
    }

    /// Advance `current_seq_len` by `n`. Call after the last layer's
    /// [`Self::append`] in each generation step (mirrors
    /// [`LazyKvCache::advance_by`]).
    pub fn advance_by(mut self, n: usize) -> Self {
        self.current_seq_len = (self.current_seq_len + n).min(self.max_seq_len);
        self
    }

    /// Active slice of `layer`'s slot `s`: `[0..current_seq_len]` on dim 0
    /// (clamped to ≥ 1 so a fresh cache still yields a valid rank).
    pub fn slot(&self, layer: usize, slot: usize) -> LazyTensor {
        self.layers[layer][slot]
            .slice(0_usize, 0, self.current_seq_len.max(1))
            .unwrap()
    }

    /// Full-capacity buffer for `layer`'s slot `s` (`[max_seq, …trailing]`)
    /// — escape hatch mirroring [`LazyKvCache::k_buffer_full`].
    pub fn slot_buffer_full(&self, layer: usize, slot: usize) -> LazyTensor {
        self.layers[layer][slot].clone()
    }

    /// Tokens written so far.
    pub fn current_seq_len(&self) -> usize { self.current_seq_len }
    /// Pre-allocated per-slot capacity along the sequence axis.
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }
    /// Number of layers.
    pub fn n_layers(&self) -> usize { self.layers.len() }
    /// Number of slots per layer.
    pub fn n_slots(&self) -> usize { self.slot_trailing.len() }
    /// Trailing shape of slot `s` (past the seq axis).
    pub fn slot_trailing(&self, slot: usize) -> &[usize] { &self.slot_trailing[slot] }
}

/// Zero-initialized [`LazyTensor`] of the given shape/dtype on `anchor`'s
/// graph. Same helper `LazyKvCache` uses (kept private per-module to avoid
/// coupling the two caches while the K/V one still exists independently).
fn zero_const_on(
    anchor: &LazyTensor, dtype: DType, shape: Shape, elems: usize,
) -> std::result::Result<LazyTensor, fuel_ir::Error> {
    match dtype {
        DType::F32 => Ok(anchor.const_f32_like(vec![0.0_f32; elems], shape)),
        DType::F64 => Ok(anchor.const_f64_like(vec![0.0_f64; elems], shape)),
        DType::BF16 => Ok(anchor.const_bf16_like(vec![half::bf16::ZERO; elems], shape)),
        DType::F16 => Ok(anchor.const_f16_like(vec![half::f16::ZERO; elems], shape)),
        other => Err(fuel_ir::Error::Msg(format!(
            "LazyLatentCache: unsupported dtype {other:?}",
        )).bt()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_f32(data: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(data, shape.to_vec(), &Device::cpu())
    }

    #[test]
    fn new_latent_cache_is_empty() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        // MLA-shaped: slot 0 = compressed_kv trailing [kv_lora_rank=5],
        // slot 1 = k_pe trailing [qk_rope_head_dim=2].
        let cache = LazyLatentCache::new(
            &anchor, 3, 8, vec![vec![5], vec![2]], DType::F32,
        ).unwrap();
        assert_eq!(cache.n_layers(), 3);
        assert_eq!(cache.n_slots(), 2);
        assert_eq!(cache.max_seq_len(), 8);
        assert_eq!(cache.current_seq_len(), 0);
        assert_eq!(cache.slot_trailing(0), &[5]);
        assert_eq!(cache.slot_trailing(1), &[2]);
        assert_eq!(cache.slot_buffer_full(0, 0).shape().dims(), &[8, 5]);
        assert_eq!(cache.slot_buffer_full(0, 1).shape().dims(), &[8, 2]);
    }

    /// The MLA payoff the generalization exists for: two slots of
    /// **different** trailing shapes (`[kv_lora_rank]` and
    /// `[qk_rope_head_dim]`), appended across two decode steps, sliced back
    /// as the logically-concatenated latents. `LazyKvCache` cannot express
    /// this — it forces both buffers to `[n_kv_heads, head_dim]`.
    #[test]
    fn mla_two_unequal_slots_append_and_slice() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        // slot 0 latent trailing [3], slot 1 rope-key trailing [2].
        let mut cache = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![3], vec![2]], DType::F32,
        ).unwrap();

        // Step 1: append 2 tokens.
        let c1 = anchor.const_f32_like(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let p1 = anchor.const_f32_like(
            vec![10.0, 11.0, 12.0, 13.0], vec![2, 2]);
        cache = cache.append(0, &[&c1, &p1]).unwrap().advance_by(2);
        assert_eq!(cache.current_seq_len(), 2);

        // Step 2: append 1 token.
        let c2 = anchor.const_f32_like(vec![7.0, 8.0, 9.0], vec![1, 3]);
        let p2 = anchor.const_f32_like(vec![14.0, 15.0], vec![1, 2]);
        cache = cache.append(0, &[&c2, &p2]).unwrap().advance_by(1);
        assert_eq!(cache.current_seq_len(), 3);

        // slot 0 (latent) = the three tokens' [3]-vectors, end to end.
        let latent = cache.slot(0, 0);
        assert_eq!(latent.shape().dims(), &[3, 3]);
        assert_eq!(
            latent.realize_f32(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        // slot 1 (rope key) = the three tokens' [2]-vectors.
        let kpe = cache.slot(0, 1);
        assert_eq!(kpe.shape().dims(), &[3, 2]);
        assert_eq!(
            kpe.realize_f32(),
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        );
    }

    /// Two-projection / QKV-pruning shape: a single retained latent per
    /// layer (one slot), with a rank-3 trailing shape to exercise
    /// higher-rank slots.
    #[test]
    fn single_slot_rank3_trailing() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        // one slot, trailing [2, 2] → buffer [max_seq, 2, 2].
        let cache = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![2, 2]], DType::F32,
        ).unwrap();
        assert_eq!(cache.n_slots(), 1);
        let t = anchor.const_f32_like(
            vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let cache = cache.append(0, &[&t]).unwrap().advance_by(1);
        let got = cache.slot(0, 0);
        assert_eq!(got.shape().dims(), &[1, 2, 2]);
        assert_eq!(got.realize_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn multi_layer_isolates_slots() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyLatentCache::new(
            &anchor, 2, 4, vec![vec![2], vec![1]], DType::F32,
        ).unwrap();
        // Both layers appended within one step → same position → advance once.
        let a0 = anchor.const_f32_like(vec![1.0, 1.0], vec![1, 2]);
        let b0 = anchor.const_f32_like(vec![9.0], vec![1, 1]);
        let cache = cache.append(0, &[&a0, &b0]).unwrap();
        let a1 = anchor.const_f32_like(vec![2.0, 2.0], vec![1, 2]);
        let b1 = anchor.const_f32_like(vec![8.0], vec![1, 1]);
        let cache = cache.append(1, &[&a1, &b1]).unwrap().advance_by(1);
        assert_eq!(cache.current_seq_len(), 1);
        assert_eq!(cache.slot(0, 0).realize_f32(), vec![1.0, 1.0]);
        assert_eq!(cache.slot(0, 1).realize_f32(), vec![9.0]);
        assert_eq!(cache.slot(1, 0).realize_f32(), vec![2.0, 2.0]);
        assert_eq!(cache.slot(1, 1).realize_f32(), vec![8.0]);
    }

    #[test]
    fn append_rejects_oob_layer() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![2]], DType::F32,
        ).unwrap();
        let t = anchor.const_f32_like(vec![0.0, 0.0], vec![1, 2]);
        assert!(cache.append(5, &[&t]).is_err());
    }

    #[test]
    fn append_rejects_wrong_slot_count() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![2], vec![2]], DType::F32,
        ).unwrap();
        let t = anchor.const_f32_like(vec![0.0, 0.0], vec![1, 2]);
        // 2 slots declared, only 1 tensor supplied.
        assert!(cache.append(0, &[&t]).is_err());
    }

    #[test]
    fn append_rejects_slot_shape_mismatch() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![3]], DType::F32,
        ).unwrap();
        // trailing should be [3], supply [5].
        let t = anchor.const_f32_like(vec![0.0; 5], vec![1, 5]);
        assert!(cache.append(0, &[&t]).is_err());
    }

    #[test]
    fn append_rejects_mismatched_seqlen_across_slots() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![2], vec![2]], DType::F32,
        ).unwrap();
        // slot 0 has seqlen 2, slot 1 has seqlen 1 → inconsistent.
        let a = anchor.const_f32_like(vec![0.0; 4], vec![2, 2]);
        let b = anchor.const_f32_like(vec![0.0; 2], vec![1, 2]);
        assert!(cache.append(0, &[&a, &b]).is_err());
    }

    #[test]
    fn append_rejects_capacity_overflow() {
        let anchor = cpu_f32(vec![0.0], &[1]);
        let cache = LazyLatentCache::new(
            &anchor, 1, 2, vec![vec![1]], DType::F32,
        ).unwrap();
        let t = anchor.const_f32_like(vec![0.0; 3], vec![3, 1]);
        assert!(cache.append(0, &[&t]).is_err()); // 3 tokens > max_seq_len 2
    }
}
