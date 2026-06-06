//! MetaVoice main TTS LM — lazy port.
//!
//! Decoder-only transformer (RMSNorm + RoPE + SwiGLU FFN) that
//! predicts EnCodec audio tokens conditioned on text tokens and a
//! speaker embedding produced upstream by
//! [`crate::lazy_metavoice_speaker_encoder`]. Mirrors the
//! stage-2 transformer in
//! `fuel-transformers/src/models/audio/metavoice.rs` — bias-free
//! GQA-capable causal LM with a multi-codebook prediction head.
//!
//! Speaker conditioning: the `(1, 1, speaker_emb_dim)` speaker
//! vector is projected through a `speaker_emb_dim → hidden_size`
//! linear and added (broadcast over the sequence axis) to the
//! token embeddings before the first transformer block. This
//! matches the eager `speaker_cond_pos` linear, but without the
//! eager `spk_cond_mask` row gating — the lazy port reuses the
//! projected vector across all sequence positions (i.e., the
//! gating mask is implicitly all-ones).
//!
//! Multi-codebook head: after the final RmsNorm, the last-position
//! hidden state is projected through `num_codebooks` separate
//! bias-free linears (one per EnCodec codebook), then stacked into
//! a `(batch, num_codebooks, vocab_size)` tensor.
//!
//! Scope: F32 activations, batch == 1, forward-only inference,
//! single shared `vocab_size` across all codebook heads. KV cache
//! is not used — every `forward` call recomputes attention over
//! the full input (matches the LLaMA / Mistral lazy v1 contract).

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// MetaVoice main LM configuration. Mirrors HF stage-2 config
/// fields with one addition: `num_codebooks` selects how many
/// parallel EnCodec codebook heads the model predicts.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaVoiceConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_codebooks: usize,
    pub speaker_emb_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl MetaVoiceConfig {
    /// MetaVoice-1B v0.1 stage-2 preset extended to multi-codebook
    /// output. Matches the eager
    /// `metavoice::transformer::Config::cfg1b_v0_1` shape and uses
    /// 4 EnCodec codebooks (the typical small-EnCodec count).
    pub fn metavoice_1b_v0_1() -> Self {
        Self {
            vocab_size: 2562,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 128,
            num_codebooks: 4,
            speaker_emb_dim: 256,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

/// MetaVoice main LM weights. Per-layer parameters reuse
/// [`crate::lazy::LayerWeights`] since the bias-free GQA decoder
/// shape is identical to LLaMA / Mistral.
#[derive(Debug, Clone)]
pub struct MetaVoiceWeights {
    /// `[vocab_size, hidden_size]` token embedding table.
    pub token_embedding: Arc<[f32]>,
    /// `[speaker_emb_dim, hidden_size]` speaker conditioning
    /// projection (no bias).
    pub speaker_proj: WeightStorage,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// `[hidden_size]` RmsNorm gain before the lm heads.
    pub final_norm_gain: Arc<[f32]>,
    /// One `[hidden_size, vocab_size]` projection per codebook.
    pub lm_heads: Vec<WeightStorage>,
}

/// MetaVoice main LM, lazy-graph form.
#[derive(Debug, Clone)]
pub struct MetaVoiceModel {
    pub config: MetaVoiceConfig,
    pub weights: MetaVoiceWeights,
}

impl MetaVoiceModel {
    /// Run a forward pass on `tokens` with the speaker embedding
    /// `speaker_embed` (shape `(1, 1, speaker_emb_dim)` or
    /// `(1, speaker_emb_dim)`) and return per-codebook logits of
    /// shape `(1, num_codebooks, vocab_size)` for the final
    /// sequence position.
    ///
    /// `start_pos` offsets the RoPE frequencies; pass `0` for the
    /// first forward of a sequence.
    pub fn forward(
        &self,
        tokens: &[u32],
        speaker_embed: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "MetaVoiceModel::forward: tokens must be non-empty");

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        self.forward_embeds(&h, speaker_embed, start_pos)
    }

    /// Forward from pre-computed text embeddings of shape
    /// `(1, seq, hidden_size)`. Used by multimodal wrappers that
    /// build embeddings outside the LM.
    pub fn forward_embeds(
        &self,
        embeds: &LazyTensor,
        speaker_embed: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims().to_vec();
        assert_eq!(dims.len(), 3, "embeds must be rank 3 [b, seq, hidden]");
        let batch = dims[0];
        let seq = dims[1];
        assert_eq!(dims[2], cfg.hidden_size);
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            "MetaVoiceConfig: num_attention_heads * head_dim must equal hidden_size",
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "MetaVoiceConfig: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        );

        let spk_anchored = self.anchor_speaker(speaker_embed, embeds)?;
        let spk_proj = self.weights.speaker_proj.apply_linear(
            &spk_anchored,
            cfg.speaker_emb_dim,
            cfg.hidden_size,
        );
        let spk_bc = spk_proj.broadcast_to(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;
        let mut h = embeds.add(&spk_bc)?;

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);
        let mask = self.build_causal_mask(&h, seq);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask)?;
        }

        let h_norm =
            h.rms_norm_affine(Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)?;

        let last = h_norm.narrow(1_usize, seq - 1, 1)?;
        let mut per_codebook: Vec<LazyTensor> = Vec::with_capacity(cfg.num_codebooks);
        for head in &weights.lm_heads {
            let logits = head.apply_linear(&last, cfg.hidden_size, cfg.vocab_size);
            per_codebook.push(logits.squeeze(1_usize)?);
        }
        let refs: Vec<&LazyTensor> = per_codebook.iter().collect();
        LazyTensor::stack(&refs, 1_usize)
            .map_err(|e| crate::Error::Msg(format!("stack lm_heads: {e}")).bt())
    }

    fn anchor_speaker(
        &self,
        speaker_embed: &LazyTensor,
        anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims: Vec<usize> = speaker_embed.shape().dims().to_vec();
        let n: usize = dims.iter().product();
        let expected = cfg.speaker_emb_dim;
        let trailing_ok = match dims.as_slice() {
            [d] => *d == expected,
            [_, d] => *d == expected,
            [_, _, d] => *d == expected,
            _ => false,
        };
        if !trailing_ok || n != expected {
            return Err(crate::Error::Msg(format!(
                "speaker_embed must flatten to speaker_emb_dim={expected} elements, got shape {dims:?}",
            ))
            .bt());
        }
        let data: Arc<[f32]> = Arc::from(speaker_embed.realize_f32());
        Ok(anchor.const_f32_like(data, Shape::from_dims(&[1, 1, cfg.speaker_emb_dim])))
    }

    fn build_causal_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
        let mut data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &LayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm =
            x.rms_norm_affine(Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim);

        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out =
            layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;

        let h1_norm =
            h1.rms_norm_affine(Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let gate =
            layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out =
            layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size);

        h1.add(&ffn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }

    fn tiny_cfg() -> MetaVoiceConfig {
        MetaVoiceConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            num_codebooks: 4,
            speaker_emb_dim: 8,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    fn tiny_weights(cfg: &MetaVoiceConfig, seed: u32) -> MetaVoiceWeights {
        let mut nb = rng_seed(seed);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut nb);
        let speaker_proj = ws(cfg.speaker_emb_dim * h, &mut nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: ws(h * h, &mut nb),
                attn_q_bias: None,
                attn_k: ws(h * kv, &mut nb),
                attn_k_bias: None,
                attn_v: ws(h * kv, &mut nb),
                attn_v_bias: None,
                attn_o: ws(h * h, &mut nb),
                ffn_gate: ws(h * i, &mut nb),
                ffn_up: ws(h * i, &mut nb),
                ffn_down: ws(i * h, &mut nb),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let lm_heads: Vec<WeightStorage> = (0..cfg.num_codebooks)
            .map(|_| ws(h * cfg.vocab_size, &mut nb))
            .collect();
        MetaVoiceWeights {
            token_embedding,
            speaker_proj,
            layers,
            final_norm_gain,
            lm_heads,
        }
    }

    fn tiny_model() -> MetaVoiceModel {
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg, 2026);
        MetaVoiceModel { config: cfg, weights }
    }

    fn speaker_vec(cfg: &MetaVoiceConfig, fill: f32) -> LazyTensor {
        let data: Vec<f32> = (0..cfg.speaker_emb_dim).map(|_| fill).collect();
        LazyTensor::from_f32(
            data,
            Shape::from_dims(&[1, 1, cfg.speaker_emb_dim]),
            &Device::cpu(),
        )
    }

    #[test]
    fn forward_shape_and_finite() {
        let model = tiny_model();
        let cfg = &model.config;
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let spk = speaker_vec(cfg, 0.1);
        let logits = model.forward(&tokens, &spk, 0).unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[1, cfg.num_codebooks, cfg.vocab_size]
        );
        for (i, &v) in logits.realize_f32().iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    #[test]
    fn multi_codebook_head_output_shape() {
        let mut cfg = tiny_cfg();
        cfg.num_codebooks = 6;
        let weights = tiny_weights(&cfg, 7);
        let model = MetaVoiceModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let spk = speaker_vec(&cfg, 0.0);
        let logits = model.forward(&tokens, &spk, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 6, cfg.vocab_size]);
    }

    #[test]
    fn forward_embeds_matches_forward() {
        let model = tiny_model();
        let cfg = &model.config;
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9];
        let spk = speaker_vec(cfg, 0.05);

        let out_tokens = model.forward(&tokens, &spk, 0).unwrap().realize_f32();

        let embeds = LazyTensor::embed_tokens(
            model.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            &tokens,
            &Device::cpu(),
        )
        .unwrap();
        let out_embeds = model
            .forward_embeds(&embeds, &spk, 0)
            .unwrap()
            .realize_f32();

        assert_eq!(out_tokens.len(), out_embeds.len());
        for (i, (a, b)) in out_tokens.iter().zip(out_embeds.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at {i}: forward={a} forward_embeds={b}",
            );
        }
    }

    #[test]
    fn speaker_conditioning_changes_output() {
        let model = tiny_model();
        let cfg = &model.config;
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

        let spk_zero = speaker_vec(cfg, 0.0);
        let spk_nonzero = speaker_vec(cfg, 0.5);

        let out_zero = model.forward(&tokens, &spk_zero, 0).unwrap().realize_f32();
        let out_nonzero = model
            .forward(&tokens, &spk_nonzero, 0)
            .unwrap()
            .realize_f32();

        let any_diff = out_zero
            .iter()
            .zip(out_nonzero.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            any_diff,
            "zeroed speaker vs non-zero speaker must change logits",
        );
    }
}
