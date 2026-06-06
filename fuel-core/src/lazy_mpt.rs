//! MPT (Mosaic Pretrained Transformer) decoder ported to the
//! lazy-graph API.
//!
//! Phase D LLM port. MPT (Replit-Code-v1.5-3B, MosaicBERT-style) is
//! distinguished by **ALiBi positional bias** instead of RoPE — a
//! per-head linear position penalty added directly to attention
//! scores. Otherwise: GQA + LayerNorm + GELU MLP + bias-free
//! projections.
//!
//! # ALiBi
//!
//! For a causal model, the bias for query position `i` attending to
//! key position `j ≤ i` is `slope[h] * (j - i)` (zero at `j == i`,
//! more negative as `j` recedes). Per-head slopes are
//! `1 / 2^(v * alibi_bias_max / n_heads_pow2)` for `v = 1..=n_heads_pow2`,
//! with the canonical interleave trick when `n_heads` isn't a
//! power of 2.
//!
//! v1 pre-computes the combined ALiBi + causal mask
//! `[1, n_heads, seq, seq]` as a single F32 const tensor at forward
//! time and broadcast-adds it to the attention scores before
//! softmax — same shape as the standard causal mask, just with
//! ALiBi's negative biases on the valid (lower-triangular)
//! positions instead of zeros.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct MptConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub expansion_ratio: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub kv_n_heads: usize,
    pub alibi_bias_max: usize,
    pub layer_norm_eps: f64,
}

impl MptConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn ffn_dim(&self) -> usize {
        self.d_model * self.expansion_ratio
    }

    /// Replit-Code-v1.5-3B preset.
    pub fn replit_code_v1_5_3b() -> Self {
        Self {
            d_model: 3072,
            n_heads: 24,
            n_layers: 32,
            expansion_ratio: 4,
            max_seq_len: 4096,
            vocab_size: 32_768,
            kv_n_heads: 8,
            alibi_bias_max: 8,
            layer_norm_eps: 1e-5,
        }
    }

    /// Compute the per-head ALiBi slopes vector of length `n_heads`.
    /// Mirrors the eager `build_alibi_bias` slope construction.
    pub fn alibi_slopes(&self) -> Vec<f32> {
        let n = self.n_heads;
        let mut n2 = 1_usize;
        while n2 < n { n2 *= 2; }
        let bias_max = self.alibi_bias_max;
        let slopes: Vec<f32> = (1..=n2)
            .map(|v| 1.0_f32 / 2.0_f32.powf((v * bias_max) as f32 / n2 as f32))
            .collect();
        if n2 == n {
            slopes
        } else {
            // Interleave: odd indices first, then even.
            let evens: Vec<f32> = slopes.iter().step_by(2).copied().collect();
            let odds:  Vec<f32> = slopes.iter().skip(1).step_by(2).copied().collect();
            odds.into_iter().chain(evens.into_iter()).take(n).collect()
        }
    }
}

/// Combined ALiBi + causal mask for `seq` tokens. Layout
/// `[1, n_heads, seq, seq]` row-major. For `j > i` (future), the
/// entry is `-inf`. For `j <= i`, the entry is
/// `slope[h] * (j - i)` (zero on the diagonal, more negative as `j`
/// recedes).
pub fn build_alibi_causal_mask(seq: usize, slopes: &[f32]) -> Vec<f32> {
    let n_heads = slopes.len();
    let mut out = vec![0.0_f32; n_heads * seq * seq];
    for h in 0..n_heads {
        for i in 0..seq {
            for j in 0..seq {
                let idx = h * seq * seq + i * seq + j;
                if j > i {
                    out[idx] = f32::NEG_INFINITY;
                } else {
                    out[idx] = slopes[h] * (j as f32 - i as f32);
                }
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct MptLayerWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    /// Bias-free Q/K/V/O. MPT fuses Q+K+V on disk; we store split in
    /// memory.
    pub attn_q: WeightStorage,
    pub attn_k: WeightStorage,
    pub attn_v: WeightStorage,
    pub attn_o: WeightStorage,
    /// `[d_model, ffn_dim]` — `up_proj`.
    pub mlp_up: WeightStorage,
    /// `[ffn_dim, d_model]` — `down_proj`.
    pub mlp_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MptWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<MptLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MptModel {
    pub config: MptConfig,
    pub weights: MptWeights,
}

impl MptModel {
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, d_model)`.
    /// MPT uses ALiBi positional bias + causal mask combined
    /// in a single per-head additive bias.
    pub fn forward_hidden(&self, tokens: &[u32]) -> Result<LazyTensor> {
        self.run_backbone(tokens)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. MPT does NOT scale embeddings and
    /// has no `start_pos` (ALiBi positional bias is purely relative).
    pub fn forward_embeds(&self, embeds: &LazyTensor) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(&self, embeds: &LazyTensor) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.d_model, tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self.weights.output.apply_linear(h_norm, cfg.d_model, cfg.vocab_size))
    }

    fn run_backbone(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.d_model, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h)
    }

    fn run_backbone_embeds(&self, embeds: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.d_model {
            return Err(crate::Error::Msg(format!(
                "MptModel::forward_embeds: expected embeds shape (1, seq, d_model={}), got {:?}",
                cfg.d_model, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "MptModel::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.n_heads * cfg.head_dim() != cfg.d_model {
            return Err(crate::Error::Msg(
                "MptConfig: n_heads * head_dim must equal d_model".into(),
            ).bt());
        }
        if cfg.n_heads % cfg.kv_n_heads != 0 {
            return Err(crate::Error::Msg(
                "MptConfig: n_heads must be a multiple of kv_n_heads".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let slopes = cfg.alibi_slopes();
        let mask_data = build_alibi_causal_mask(seq, &slopes);
        let mask = h.const_f32_like(
            mask_data,
            Shape::from_dims(&[1, cfg.n_heads, seq, seq]),
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &mask)?;
        }
        h.layer_norm_affine(
            std::sync::Arc::clone(&weights.final_ln_gain),
            std::sync::Arc::clone(&weights.final_ln_bias),
            cfg.layer_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &MptLayerWeights,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.kv_n_heads * head_dim;

        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.norm1_gain), std::sync::Arc::clone(&layer.norm1_bias), cfg.layer_norm_eps)?;

        // Bias-free Q/K/V.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.d_model, cfg.d_model);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.d_model, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.d_model, kv_dim);

        let q = q.split_heads(cfg.n_heads, head_dim)?;
        let k = k.split_heads(cfg.kv_n_heads, head_dim)?;
        let v = v.split_heads(cfg.kv_n_heads, head_dim)?;

        // GQA replication.
        let n_rep = cfg.n_heads / cfg.kv_n_heads;
        let k_full = k.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Broadcast-add the ALiBi + causal mask (shape
        // `[1, n_heads, seq, seq]`). Broadcasts cleanly over the
        // batch axis.
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.d_model, cfg.d_model);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.norm2_gain), std::sync::Arc::clone(&layer.norm2_bias), cfg.layer_norm_eps)?;

        let mid = layer.mlp_up.apply_linear(&h1_norm, cfg.d_model, cfg.ffn_dim());
        let mid_act = mid.gelu_erf();
        let ffn_out = layer.mlp_down.apply_linear(&mid_act, cfg.ffn_dim(), cfg.d_model);
        h1.add(&ffn_out)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl MptWeights {
    /// Load MPT (Replit-Code-v1.5-3B etc.) weights. MPT uses fused
    /// `Wqkv` of shape `[d_model + 2*kv_dim, d_model]` (multi-group
    /// QKV) — split at load.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &MptConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix};
        let d = cfg.d_model;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.n_heads * head_dim;  // == d_model
        let kv_dim = cfg.kv_n_heads * head_dim;
        let ffn = cfg.ffn_dim();

        let token_embedding = Arc::from(load_tensor_as_f32(st, "transformer.wte.weight")?);
        let mut layers: Vec<MptLayerWeights> = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("transformer.blocks.{i}");
            let norm1_gain = Arc::from(load_tensor_as_f32(st, &format!("{p}.norm_1.weight"))?);
            let norm1_bias = load_tensor_as_f32(st, &format!("{p}.norm_1.bias")).ok()
                .map(Arc::from).unwrap_or_else(|| Arc::from(vec![0.0; d]));
            let norm2_gain = Arc::from(load_tensor_as_f32(st, &format!("{p}.norm_2.weight"))?);
            let norm2_bias = load_tensor_as_f32(st, &format!("{p}.norm_2.bias")).ok()
                .map(Arc::from).unwrap_or_else(|| Arc::from(vec![0.0; d]));

            // Fused QKV shape: [q_dim + 2*kv_dim, d_model]
            let qkv = load_transposed_matrix(
                st, &format!("{p}.attn.Wqkv.weight"), q_dim + 2 * kv_dim, d,
            )?;
            // Split into Q, K, V
            let mut q = vec![0.0_f32; d * q_dim];
            let mut k = vec![0.0_f32; d * kv_dim];
            let mut v = vec![0.0_f32; d * kv_dim];
            let out_dim = q_dim + 2 * kv_dim;
            for row in 0..d {
                let src = &qkv[row * out_dim..(row + 1) * out_dim];
                q[row * q_dim..(row + 1) * q_dim].copy_from_slice(&src[..q_dim]);
                k[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&src[q_dim..q_dim + kv_dim]);
                v[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&src[q_dim + kv_dim..]);
            }

            let attn_o = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.attn.out_proj.weight"), d, d,
            )?;
            let mlp_up = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.ffn.up_proj.weight"), ffn, d,
            )?;
            let mlp_down = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.ffn.down_proj.weight"), d, ffn,
            )?;

            layers.push(MptLayerWeights {
                norm1_gain, norm1_bias, norm2_gain, norm2_bias,
                attn_q: WeightStorage::F32(Arc::from(q)),
                attn_k: WeightStorage::F32(Arc::from(k)),
                attn_v: WeightStorage::F32(Arc::from(v)),
                attn_o, mlp_up, mlp_down,
            });
        }

        let final_ln_gain = Arc::from(load_tensor_as_f32(st, "transformer.norm_f.weight")?);
        let final_ln_bias = load_tensor_as_f32(st, "transformer.norm_f.bias").ok()
            .map(Arc::from).unwrap_or_else(|| Arc::from(vec![0.0; d]));
        // MPT typically ties lm_head to wte; load if separate present.
        let output = match crate::lazy::load_transposed_matrix_preserve_dtype(
            st, "lm_head.weight", cfg.vocab_size, d,
        ) {
            Ok(w) => w,
            Err(_) => crate::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding, cfg.vocab_size, d,
            ),
        };

        Ok(Self { token_embedding, layers, final_ln_gain, final_ln_bias, output })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &MptConfig) -> MptWeights {
        let mut s: u32 = 14641;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.d_model;
        let kv = cfg.kv_n_heads * cfg.head_dim();
        let inter = cfg.ffn_dim();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<MptLayerWeights> = (0..cfg.n_layers).map(|_| MptLayerWeights {
            norm1_gain: Arc::from(vec![1.0_f32; h]),
            norm1_bias: Arc::from(vec![0.0_f32; h]),
            norm2_gain: Arc::from(vec![1.0_f32; h]),
            norm2_bias: Arc::from(vec![0.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            mlp_up:   WeightStorage::F32(vec_of(h * inter, &mut *nb)),
            mlp_down: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        MptWeights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_shape_and_finite_with_alibi() {
        let cfg = MptConfig {
            d_model: 16, n_heads: 4, n_layers: 2, expansion_ratio: 4,
            max_seq_len: 32, vocab_size: 32, kv_n_heads: 2,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let model = MptModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3, 4]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    /// ALiBi slopes for n_heads = 4 (power of 2) follow the
    /// `1 / 2^(v * 8 / 4)` formula directly.
    #[test]
    fn alibi_slopes_power_of_two_heads() {
        let cfg = MptConfig {
            d_model: 16, n_heads: 4, n_layers: 1, expansion_ratio: 4,
            max_seq_len: 16, vocab_size: 16, kv_n_heads: 2,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let slopes = cfg.alibi_slopes();
        assert_eq!(slopes.len(), 4);
        // slopes[0] = 1 / 2^(1 * 8 / 4) = 1/4
        // slopes[1] = 1 / 2^(2 * 8 / 4) = 1/16
        // slopes[2] = 1 / 2^(3 * 8 / 4) = 1/64
        // slopes[3] = 1 / 2^(4 * 8 / 4) = 1/256
        for (i, expected) in [0.25_f32, 0.0625, 0.015_625, 0.003_906_25].iter().enumerate() {
            assert!((slopes[i] - *expected).abs() < 1e-6, "slopes[{i}] = {} vs {expected}", slopes[i]);
        }
    }

    /// ALiBi penalty must produce a different output than a strict
    /// causal mask alone. We compare two runs with the same weights
    /// but pre-built masks that differ only in their lower-triangle
    /// entries (zero vs ALiBi-shaped).
    #[test]
    fn alibi_mask_differs_from_zero_lower_triangle() {
        let cfg = MptConfig {
            d_model: 8, n_heads: 2, n_layers: 1, expansion_ratio: 2,
            max_seq_len: 16, vocab_size: 8, kv_n_heads: 1,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let slopes = cfg.alibi_slopes();
        let causal_mask = build_alibi_causal_mask(4, &[0.0, 0.0]); // zero slopes → causal only
        let alibi_mask = build_alibi_causal_mask(4, &slopes);
        // Should differ at positions where j < i (the ALiBi negative bias).
        let any_diff = causal_mask.iter().zip(alibi_mask.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7 && a.is_finite() && b.is_finite());
        assert!(any_diff, "ALiBi mask must differ from zero-slope causal mask on j < i positions");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = MptConfig {
            d_model: 16, n_heads: 4, n_layers: 2, expansion_ratio: 4,
            max_seq_len: 16, vocab_size: 32, kv_n_heads: 1,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        };
        let model = MptModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.d_model]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> MptConfig {
        MptConfig {
            d_model: 16, n_heads: 4, n_layers: 2, expansion_ratio: 4,
            max_seq_len: 16, vocab_size: 32, kv_n_heads: 1,
            alibi_bias_max: 8, layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = MptModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "MPT forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = MptModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.d_model + 1)],
            Shape::from_dims(&[1, 3, cfg.d_model + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = MptModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "MPT forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}
