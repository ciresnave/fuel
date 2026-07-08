//! Two-projection attention (shared K = V).
//!
//! Implements the shared-KV attention variant from ["Do Transformers Need
//! Three Projections?"](https://arxiv.org/abs/2606.04032) (ICML 2026).
//! Standard multi-head attention learns three projections per layer — `Q`,
//! `K`, `V`. This variant learns **two**: `q = x @ W_q`, `kv = x @ W_kv`,
//! and then sets `K = V = kv` (the same projected tensor stands in for both
//! the key and value roles). The paper reports the shared-KV variant
//! performs on par with standard QKV attention.
//!
//! # Cache math
//!
//! The payoff is decode-cache size. A standard K/V cache stores two
//! `[n_kv_heads * head_dim]` tensors per token per layer (one for K, one
//! for V) — `2 * n_kv_heads * head_dim` elements/token. Two-projection
//! attention stores **one** `[n_kv_heads * head_dim]` tensor per token per
//! layer (the shared `kv`), since `K` and `V` are the same tensor and never
//! need to be cached separately:
//!
//!   - At equal heads (`n_kv_heads == n_heads`, plain MHA): the cache is
//!     exactly **50%** of a standard K/V cache (one slot instead of two).
//!   - With GQA/MQA-style `n_kv_heads < n_heads`, the saving compounds with
//!     the existing K/V-head reduction. E.g. `n_heads = 32`, `head_dim =
//!     128`, `n_kv_heads = 1` (MQA): standard K/V is `2 * 32 * 128 = 8192`
//!     elements/token; two-projection is `1 * 128 = 128` elements/token —
//!     **1.5625%** of the standard cache (a ~98.4% reduction). See
//!     [`Self::cache_elems_per_token`] / [`Self::standard_kv_elems_per_token`]
//!     and the `cache_size_ratio_matches_mqa_math` test below.
//!
//! # Scope
//!
//! This is a **capability block** — there is no shipped checkpoint consumer
//! yet. It is composable: [`Self::forward`] / [`Self::forward_with_latent_cache`]
//! return the raw attention context (`[B, S, n_heads * head_dim]`); callers
//! apply their own output projection (no `W_o` is baked in here), matching
//! [`crate::lazy_nn::moe`]'s house style of shipping the primitive block
//! rather than a full layer.

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_latent_cache::LazyLatentCache;
use fuel_ir::Shape;

/// Two-projection (shared K = V) attention block. Holds the `Q` and shared
/// `KV` projections; no output projection (composable — see module docs).
#[derive(Debug, Clone)]
pub struct LazyTwoProjAttention {
    /// `[hidden_size, n_heads * head_dim]`.
    w_q: WeightStorage,
    /// `[hidden_size, n_kv_heads * head_dim]` — the shared K = V projection.
    w_kv: WeightStorage,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl LazyTwoProjAttention {
    /// Build a two-projection attention block.
    ///
    /// `w_q` must have `hidden_size * n_heads * head_dim` elements; `w_kv`
    /// must have `hidden_size * n_kv_heads * head_dim` elements (both laid
    /// out `[hidden_size, out_features]`, the [`WeightStorage::apply_linear`]
    /// convention). `n_heads` and `head_dim` must be ≥ 1; `n_kv_heads` must
    /// be ≥ 1 and evenly divide `n_heads` (the GQA-style head-repeat this
    /// block supports — see [`Self::forward`]).
    pub fn new(
        w_q: WeightStorage,
        w_kv: WeightStorage,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        if n_heads == 0 {
            crate::bail!("LazyTwoProjAttention::new: n_heads must be >= 1");
        }
        if n_kv_heads == 0 {
            crate::bail!("LazyTwoProjAttention::new: n_kv_heads must be >= 1");
        }
        if head_dim == 0 {
            crate::bail!("LazyTwoProjAttention::new: head_dim must be >= 1");
        }
        if hidden_size == 0 {
            crate::bail!("LazyTwoProjAttention::new: hidden_size must be >= 1");
        }
        if n_heads % n_kv_heads != 0 {
            crate::bail!(
                "LazyTwoProjAttention::new: n_kv_heads ({n_kv_heads}) must evenly \
                 divide n_heads ({n_heads})",
            );
        }
        let want_q = hidden_size * n_heads * head_dim;
        if w_q.elem_count() != want_q {
            crate::bail!(
                "LazyTwoProjAttention::new: w_q has {} elements but \
                 hidden_size * n_heads * head_dim = {hidden_size} * {n_heads} * {head_dim} = {want_q}",
                w_q.elem_count(),
            );
        }
        let want_kv = hidden_size * n_kv_heads * head_dim;
        if w_kv.elem_count() != want_kv {
            crate::bail!(
                "LazyTwoProjAttention::new: w_kv has {} elements but \
                 hidden_size * n_kv_heads * head_dim = {hidden_size} * {n_kv_heads} * {head_dim} = {want_kv}",
                w_kv.elem_count(),
            );
        }
        Ok(Self { w_q, w_kv, n_heads, n_kv_heads, head_dim, hidden_size })
    }

    pub fn n_heads(&self) -> usize { self.n_heads }
    pub fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn hidden_size(&self) -> usize { self.hidden_size }

    /// Per-token, per-layer decode-cache footprint of the shared `kv`
    /// projection (one slot of `n_kv_heads * head_dim` elements).
    pub fn cache_elems_per_token(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Per-token, per-layer footprint a *standard* K/V cache would need at
    /// this block's head geometry (`2 * n_heads * head_dim` — two full-head
    /// slots), for comparison against [`Self::cache_elems_per_token`].
    pub fn standard_kv_elems_per_token(&self) -> usize {
        2 * self.n_heads * self.head_dim
    }

    /// Dense (prefill/training-shape) forward. `xs: (B, S, hidden_size)`,
    /// causal. Returns `(B, S, n_heads * head_dim)` — no output projection
    /// (see module docs).
    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let dims = xs.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != self.hidden_size {
            crate::bail!(
                "LazyTwoProjAttention::forward: expected input rank 3 (B, S, hidden={}), \
                 got shape {dims:?}",
                self.hidden_size,
            );
        }
        let seq = dims[1];

        // q = x @ W_q, split into heads: (B, H, S, d).
        let q = self
            .w_q
            .apply_linear(xs, self.hidden_size, self.n_heads * self.head_dim)
            .split_heads(self.n_heads, self.head_dim)?;

        // kv = x @ W_kv, split into kv-heads, repeated up to n_heads.
        // K = V = kv (the two-projection trick).
        let kv = self
            .w_kv
            .apply_linear(xs, self.hidden_size, self.n_kv_heads * self.head_dim)
            .split_heads(self.n_kv_heads, self.head_dim)?;
        let n_rep = self.n_heads / self.n_kv_heads;
        let kv = kv.repeat_interleave(1_usize, n_rep)?; // (B, H, S, d)

        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let k_t = kv.transpose()?; // (B, H, d, S)
        let scores = q.matmul(&k_t)?; // (B, H, S, S)
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(xs, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let ctx = probs.matmul(&kv)?; // (B, H, S, d) -- kv doubles as V

        ctx.merge_heads() // (B, S, H * d)
    }

    /// Cached decode (per-pass, one-slot). `xs_step: (1, seq_new, hidden_size)`
    /// — new tokens only. `cache` must be a one-slot [`LazyLatentCache`]
    /// whose slot 0 trailing shape is `[n_kv_heads * head_dim]` (this block
    /// caches only the shared `kv` projection — see module docs). The
    /// cache's `current_seq_len()` at call time is the count of already-
    /// cached tokens for `layer`.
    ///
    /// Mirrors `lazy_deepseek2.rs::mla_attention_cached`'s per-pass shape:
    /// append this step's `kv` slab, read back the full attended prefix
    /// (cached + new) from the cache's full-capacity buffer, attend over it
    /// with a decode causal mask of width `cached_len + seq_new`. Returns
    /// `(out, cache)`; **the caller must `cache.advance_by(seq_new)`** after
    /// the last layer's call in a step, exactly like [`LazyLatentCache`]'s
    /// own convention — this fn does not advance the cache itself.
    pub fn forward_with_latent_cache(
        &self,
        xs_step: &LazyTensor,
        cache: LazyLatentCache,
        layer: usize,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        let dims = xs_step.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != self.hidden_size {
            crate::bail!(
                "LazyTwoProjAttention::forward_with_latent_cache: expected xs_step rank 3 \
                 (1, seq_new, hidden={}), got shape {dims:?}",
                self.hidden_size,
            );
        }
        if dims[0] != 1 {
            crate::bail!(
                "LazyTwoProjAttention::forward_with_latent_cache: xs_step batch dim must be 1, \
                 got {}",
                dims[0],
            );
        }
        if cache.n_slots() != 1 {
            crate::bail!(
                "LazyTwoProjAttention::forward_with_latent_cache: cache must have exactly 1 \
                 slot (the shared kv projection), got {}",
                cache.n_slots(),
            );
        }
        let want_trailing = [self.n_kv_heads * self.head_dim];
        if cache.slot_trailing(0) != want_trailing {
            crate::bail!(
                "LazyTwoProjAttention::forward_with_latent_cache: cache slot 0 trailing shape \
                 must be {want_trailing:?} (n_kv_heads * head_dim), got {:?}",
                cache.slot_trailing(0),
            );
        }
        let s = dims[1];
        let cached_len = cache.current_seq_len();
        let total = cached_len + s;
        if total > cache.max_seq_len() {
            crate::bail!(
                "LazyTwoProjAttention::forward_with_latent_cache: appending {s} tokens at \
                 position {cached_len} would exceed cache max_seq_len {}",
                cache.max_seq_len(),
            );
        }

        // q from the new tokens only: (1, H, s, d).
        let q = self
            .w_q
            .apply_linear(xs_step, self.hidden_size, self.n_heads * self.head_dim)
            .split_heads(self.n_heads, self.head_dim)?;

        // kv_step = W_kv(xs_step): (1, s, Hkv*d) -> [s, Hkv*d] -> append.
        let kv_step = self.w_kv.apply_linear(
            xs_step, self.hidden_size, self.n_kv_heads * self.head_dim,
        );
        let kv_step_2d = kv_step.reshape(Shape::from_dims(&[s, self.n_kv_heads * self.head_dim]))?;
        let cache = cache.append(layer, &[&kv_step_2d])?;

        // Read back the full attended prefix (cached + new) and repeat to
        // n_heads. K = V = kv_all.
        let kv_all = cache
            .slot_buffer_full(layer, 0)
            .slice(0_usize, 0, total)?
            .reshape(Shape::from_dims(&[1, total, self.n_kv_heads * self.head_dim]))?
            .split_heads(self.n_kv_heads, self.head_dim)?; // (1, Hkv, total, d)
        let n_rep = self.n_heads / self.n_kv_heads;
        let kv_all = kv_all.repeat_interleave(1_usize, n_rep)?; // (1, H, total, d)

        let mask_data = crate::lazy::build_decode_causal_mask(cached_len, s, total);
        let mask = xs_step.const_f32_like(mask_data, Shape::from_dims(&[1, 1, s, total]));

        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let k_t = kv_all.transpose()?; // (1, H, d, total)
        let scores = q.matmul(&k_t)?; // (1, H, s, total)
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let ctx = probs.matmul(&kv_all)?; // (1, H, s, d)

        let out = ctx.merge_heads()?; // (1, s, H * d)
        Ok((out, cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};
    use std::sync::Arc;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    /// Host-side reference: `y = x @ w` with `w` laid out `[in, out]`
    /// (matches `WeightStorage::apply_linear`'s convention).
    fn ref_linear(x: &[f32], w: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0_f32;
                for k in 0..in_f {
                    acc += x[r * in_f + k] * w[k * out_f + o];
                }
                out[r * out_f + o] = acc;
            }
        }
        out
    }

    /// Host-side hand reference for two-projection causal attention over a
    /// single batch: `q = x @ w_q`, `kv = x @ w_kv`, `K = V = kv` (repeated
    /// GQA-style from `n_kv_heads` up to `n_heads`), causal softmax, merge
    /// heads. `q_start` shifts the causal boundary for the cached-decode
    /// comparison (absolute position of row 0 of `x` is `q_start`); `kv`
    /// spans the *cached* prefix `[0, total)` while `x`/`q` span only the
    /// `rows` new tokens starting at `q_start`.
    #[allow(clippy::too_many_arguments)]
    fn hand_reference_attn(
        q_proj: &[f32],   // [rows, H*d] -- new-token queries only
        kv_proj: &[f32],  // [total, Hkv*d] -- full attended-prefix kv
        rows: usize,
        total: usize,
        q_start: usize,   // absolute position of q row 0
        h: usize,
        hkv: usize,
        d: usize,
    ) -> Vec<f32> {
        let n_rep = h / hkv;
        let scale = 1.0_f32 / (d as f32).sqrt();
        let mut out = vec![0.0_f32; rows * h * d];
        for head in 0..h {
            let kv_head = head / n_rep;
            for i in 0..rows {
                let abs_i = q_start + i;
                // scores against j in [0, abs_i], scale + softmax.
                let mut scores = vec![0.0_f32; abs_i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0_f32;
                    for dd in 0..d {
                        let qv = q_proj[i * (h * d) + head * d + dd];
                        let kv = kv_proj[j * (hkv * d) + kv_head * d + dd];
                        acc += qv * kv;
                    }
                    *score = acc * scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
                for dd in 0..d {
                    let mut acc = 0.0_f32;
                    for j in 0..=abs_i {
                        acc += probs[j] * kv_proj[j * (hkv * d) + kv_head * d + dd];
                    }
                    out[i * (h * d) + head * d + dd] = acc;
                }
            }
        }
        let _ = total; // total is implicit in kv_proj.len() / (hkv*d)
        out
    }

    /// Test 1 (born-red semantic anchor): tiny dims, deterministic ramp
    /// weights, dense forward vs. a plain-loop host reference.
    #[test]
    fn dense_matches_hand_reference() {
        let hidden = 4;
        let (h, hkv, d) = (2, 1, 2);
        let seq = 3;
        let batch = 1;

        let w_q_data = ramp_f32(hidden * h * d, 0.05, -0.2);
        let w_kv_data = ramp_f32(hidden * hkv * d, 0.03, 0.1);
        let x_data = ramp_f32(batch * seq * hidden, 0.04, -0.3);

        let attn = LazyTwoProjAttention::new(
            WeightStorage::F32(Arc::from(w_q_data.clone())),
            WeightStorage::F32(Arc::from(w_kv_data.clone())),
            h, hkv, d, hidden,
        ).unwrap();

        let xs = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );
        let out = attn.forward(&xs).unwrap();
        assert_eq!(out.shape().dims(), &[batch, seq, h * d]);
        let got = out.realize_f32();

        // Host reference: q/kv projections, then causal K=V attention.
        let q_proj = ref_linear(&x_data, &w_q_data, seq, hidden, h * d);
        let kv_proj = ref_linear(&x_data, &w_kv_data, seq, hidden, hkv * d);
        let expected = hand_reference_attn(&q_proj, &kv_proj, seq, seq, 0, h, hkv, d);

        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "out[{i}] expected {e}, got {a}");
        }
    }

    /// Test 2 (bit-exact acceptance gate): S=4 protocol -- dense forward
    /// over all 4 tokens vs. cached (prefill 2, decode 1, decode 1). If
    /// this fails, print max-diff and STOP: a mismatch is a real bug (GQA
    /// repeat or mask), not something to loosen.
    #[test]
    fn cached_decode_bit_exact_to_dense() {
        let hidden = 4;
        let (h, hkv, d) = (2, 1, 2);
        let seq = 4;
        let batch = 1;

        let w_q_data = ramp_f32(hidden * h * d, 0.05, -0.2);
        let w_kv_data = ramp_f32(hidden * hkv * d, 0.03, 0.1);
        let x_data = ramp_f32(batch * seq * hidden, 0.04, -0.3);

        let attn = LazyTwoProjAttention::new(
            WeightStorage::F32(Arc::from(w_q_data)),
            WeightStorage::F32(Arc::from(w_kv_data)),
            h, hkv, d, hidden,
        ).unwrap();

        // ---- Dense reference over all 4 tokens ----
        let xs_dense = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );
        let dense = attn.forward(&xs_dense).unwrap().realize_f32();

        // ---- Cached: prefill 2, decode 1, decode 1 ----
        let xs_full = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );
        let cache = LazyLatentCache::new(
            &xs_full, 1, seq, vec![vec![hkv * d]], DType::F32,
        ).unwrap();

        let step1 = xs_full.slice(1_usize, 0, 2).unwrap();
        let (out1, cache) = attn.forward_with_latent_cache(&step1, cache, 0).unwrap();
        let cache = cache.advance_by(2);

        let step2 = xs_full.slice(1_usize, 2, 1).unwrap();
        let (out2, cache) = attn.forward_with_latent_cache(&step2, cache, 0).unwrap();
        let cache = cache.advance_by(1);

        let step3 = xs_full.slice(1_usize, 3, 1).unwrap();
        let (out3, cache) = attn.forward_with_latent_cache(&step3, cache, 0).unwrap();
        let _cache = cache.advance_by(1);

        let mut cached: Vec<f32> = Vec::with_capacity(seq * h * d);
        cached.extend(out1.realize_f32());
        cached.extend(out2.realize_f32());
        cached.extend(out3.realize_f32());

        assert_eq!(cached.len(), dense.len());
        let mut max_diff_bits = 0i64;
        let mut max_diff_idx = 0usize;
        for (i, (c, dn)) in cached.iter().zip(dense.iter()).enumerate() {
            let diff = (c.to_bits() as i64 - dn.to_bits() as i64).abs();
            if diff > max_diff_bits {
                max_diff_bits = diff;
                max_diff_idx = i;
            }
        }
        if max_diff_bits != 0 {
            panic!(
                "cached_decode_bit_exact_to_dense: NOT bit-exact -- max diff at index \
                 {max_diff_idx}: cached={} dense={} (bit diff {max_diff_bits})",
                cached[max_diff_idx], dense[max_diff_idx],
            );
        }
        for (i, (c, dn)) in cached.iter().zip(dense.iter()).enumerate() {
            assert_eq!(c.to_bits(), dn.to_bits(), "out[{i}]: cached {c} != dense {dn}");
        }
    }

    /// Test 3: GQA repeat path (`n_kv_heads > 1` and `< n_heads`) --
    /// same cached-vs-dense bit-exact protocol, exercising the
    /// `repeat_interleave` head-repeat rather than the MQA `n_kv_heads==1`
    /// shortcut.
    #[test]
    fn gqa_repeat_correct() {
        let hidden = 4;
        let (h, hkv, d) = (4, 2, 2);
        let seq = 4;
        let batch = 1;

        let w_q_data = ramp_f32(hidden * h * d, 0.02, -0.1);
        let w_kv_data = ramp_f32(hidden * hkv * d, 0.025, 0.05);
        let x_data = ramp_f32(batch * seq * hidden, 0.03, -0.2);

        let attn = LazyTwoProjAttention::new(
            WeightStorage::F32(Arc::from(w_q_data)),
            WeightStorage::F32(Arc::from(w_kv_data)),
            h, hkv, d, hidden,
        ).unwrap();

        let xs_dense = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );
        let dense = attn.forward(&xs_dense).unwrap().realize_f32();

        let xs_full = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );
        let cache = LazyLatentCache::new(
            &xs_full, 1, seq, vec![vec![hkv * d]], DType::F32,
        ).unwrap();

        let step1 = xs_full.slice(1_usize, 0, 2).unwrap();
        let (out1, cache) = attn.forward_with_latent_cache(&step1, cache, 0).unwrap();
        let cache = cache.advance_by(2);

        let step2 = xs_full.slice(1_usize, 2, 1).unwrap();
        let (out2, cache) = attn.forward_with_latent_cache(&step2, cache, 0).unwrap();
        let cache = cache.advance_by(1);

        let step3 = xs_full.slice(1_usize, 3, 1).unwrap();
        let (out3, cache) = attn.forward_with_latent_cache(&step3, cache, 0).unwrap();
        let _cache = cache.advance_by(1);

        let mut cached: Vec<f32> = Vec::with_capacity(seq * h * d);
        cached.extend(out1.realize_f32());
        cached.extend(out2.realize_f32());
        cached.extend(out3.realize_f32());

        assert_eq!(cached.len(), dense.len());
        for (i, (c, dn)) in cached.iter().zip(dense.iter()).enumerate() {
            assert_eq!(c.to_bits(), dn.to_bits(), "out[{i}]: cached {c} != dense {dn} (GQA repeat)");
        }
    }

    /// Test 4: build-time / call-time typed-error rejections.
    #[test]
    fn rejects_bad_geometry() {
        let hidden = 4;
        let (h, hkv, d) = (2, 1, 2);

        // ---- constructor geometry ----
        let good_wq = WeightStorage::F32(Arc::from(ramp_f32(hidden * h * d, 0.01, 0.0)));
        let good_wkv = WeightStorage::F32(Arc::from(ramp_f32(hidden * hkv * d, 0.01, 0.0)));

        // Wrong w_q element count.
        let bad_wq = WeightStorage::F32(Arc::from(vec![0.0_f32; hidden * h * d - 1]));
        assert!(LazyTwoProjAttention::new(
            bad_wq, good_wkv.clone(), h, hkv, d, hidden,
        ).is_err());

        // Wrong w_kv element count.
        let bad_wkv = WeightStorage::F32(Arc::from(vec![0.0_f32; hidden * hkv * d + 1]));
        assert!(LazyTwoProjAttention::new(
            good_wq.clone(), bad_wkv, h, hkv, d, hidden,
        ).is_err());

        // n_kv_heads does not divide n_heads (3 heads, 2 kv-heads).
        let wq3 = WeightStorage::F32(Arc::from(ramp_f32(hidden * 3 * d, 0.01, 0.0)));
        let wkv2 = WeightStorage::F32(Arc::from(ramp_f32(hidden * 2 * d, 0.01, 0.0)));
        assert!(LazyTwoProjAttention::new(wq3, wkv2, 3, 2, d, hidden).is_err());

        // Zero n_heads / n_kv_heads / head_dim / hidden_size.
        assert!(LazyTwoProjAttention::new(
            good_wq.clone(), good_wkv.clone(), 0, hkv, d, hidden,
        ).is_err());
        assert!(LazyTwoProjAttention::new(
            good_wq.clone(), good_wkv.clone(), h, 0, d, hidden,
        ).is_err());
        assert!(LazyTwoProjAttention::new(
            good_wq.clone(), good_wkv.clone(), h, hkv, 0, hidden,
        ).is_err());
        assert!(LazyTwoProjAttention::new(
            good_wq.clone(), good_wkv.clone(), h, hkv, d, 0,
        ).is_err());

        let attn = LazyTwoProjAttention::new(good_wq, good_wkv, h, hkv, d, hidden).unwrap();

        // ---- forward_with_latent_cache: cache/geometry mismatches ----
        let anchor = LazyTensor::from_f32(vec![0.0_f32; hidden], Shape::from_dims(&[1, 1, hidden]), &Device::cpu());
        let step = anchor.clone();

        // Wrong slot count (2 slots instead of 1).
        let cache_2slots = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![hkv * d], vec![hkv * d]], DType::F32,
        ).unwrap();
        assert!(attn.forward_with_latent_cache(&step, cache_2slots, 0).is_err());

        // Wrong slot trailing shape.
        let cache_bad_trailing = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![hkv * d + 1]], DType::F32,
        ).unwrap();
        assert!(attn.forward_with_latent_cache(&step, cache_bad_trailing, 0).is_err());

        // Capacity overflow: max_seq_len=1, already full, appending 1 more.
        let cache_full = LazyLatentCache::new(
            &anchor, 1, 1, vec![vec![hkv * d]], DType::F32,
        ).unwrap();
        let prefill = anchor.clone();
        let (_out, cache_full) = attn.forward_with_latent_cache(&prefill, cache_full, 0).unwrap();
        let cache_full = cache_full.advance_by(1);
        let overflow_step = anchor.clone();
        assert!(attn.forward_with_latent_cache(&overflow_step, cache_full, 0).is_err());

        // Bad xs_step batch dim (batch != 1).
        let cache_ok = LazyLatentCache::new(
            &anchor, 1, 4, vec![vec![hkv * d]], DType::F32,
        ).unwrap();
        let bad_batch_step = LazyTensor::from_f32(
            vec![0.0_f32; 2 * hidden], Shape::from_dims(&[2, 1, hidden]), &Device::cpu(),
        );
        assert!(attn.forward_with_latent_cache(&bad_batch_step, cache_ok, 0).is_err());
    }

    /// Test 5: cache-size ratio documented in the module doc, checked
    /// numerically for an MQA-style fixture (n_heads=32, n_kv_heads=1,
    /// head_dim=128 -- a Llama-7B-shaped head geometry).
    #[test]
    fn cache_size_ratio_matches_mqa_math() {
        let hidden = 4096;
        let (h, hkv, d) = (32, 1, 128);
        let w_q = WeightStorage::F32(Arc::from(vec![0.0_f32; hidden * h * d]));
        let w_kv = WeightStorage::F32(Arc::from(vec![0.0_f32; hidden * hkv * d]));
        let attn = LazyTwoProjAttention::new(w_q, w_kv, h, hkv, d, hidden).unwrap();

        // Two-projection: one shared kv slot of Hkv*d elements/token.
        assert_eq!(attn.cache_elems_per_token(), hkv * d);
        assert_eq!(attn.cache_elems_per_token(), 128);
        // Standard K/V: two full-head slots of H*d elements/token each.
        assert_eq!(attn.standard_kv_elems_per_token(), 2 * h * d);
        assert_eq!(attn.standard_kv_elems_per_token(), 8192);

        let ratio = attn.cache_elems_per_token() as f64 / attn.standard_kv_elems_per_token() as f64;
        assert!((ratio - 0.015625).abs() < 1e-12, "ratio = {ratio}");
        let reduction = 1.0 - ratio;
        assert!((reduction - 0.984375).abs() < 1e-12, "reduction = {reduction}");
    }
}
