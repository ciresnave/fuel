//! Voxtral (Mistral AI) speech-language model ported to the lazy-graph API.
//!
//! Phase D multimodal port. Voxtral combines a **Whisper-compatible
//! audio encoder** with a **Mistral-style Llama decoder** through a
//! 2-layer GELU-activated projector. The eager reference is HF's
//! `mistralai/Voxtral-Mini-3B-2507` / `Voxtral-Small-24B-2507`.
//!
//! Architecture:
//!
//!   1. **Audio encoder** (Whisper-shape):
//!      `Conv1d(num_mel_bins → hidden, k=3, p=1) → GELU →`
//!      `Conv1d(hidden → hidden, k=3, s=2, p=1) → GELU →`
//!      `transpose(time,channel) + position_embedding →`
//!      `N × pre-LN(LayerNorm + self-attention + LayerNorm + GeLU MLP) →`
//!      `final LayerNorm`.
//!      Reuses `lazy_whisper::conv1d_k3_s{1,2}_p1` for the conv stem.
//!   2. **Multi-modal projector**: `Linear(intermediate_size →`
//!      `text_hidden) → GELU → Linear(text_hidden → text_hidden)`.
//!      Both linears are bias-free.
//!   3. **Voxtral text model**: Llama-shape decoder with one quirk
//!      — `head_dim` is explicit in the config (128) rather than
//!      `hidden_size / num_attention_heads`. For Voxtral 3B this
//!      makes `q_dim = num_heads * head_dim = 4096 != hidden_size
//!      = 3072`, so the Q/K/V/O linears are non-square. GQA,
//!      SwiGLU FFN, RmsNorm, RoPE θ = 100M.
//!
//! Audio token replacement: text input contains a special
//! `audio_token_id` (24 in HF Voxtral); the projected audio embeddings
//! are scatter-substituted into the embeddings at those positions.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32.
//! v1 ships three forward entry points:
//!
//!   - [`VoxtralEncoder::forward`] — mel-spec → audio hidden states.
//!     Input shape `[1, num_mel_bins, mel_time]`, output
//!     `[1, mel_time/2, hidden_size]`.
//!   - [`VoxtralMultiModalProjector::forward`] — audio hidden →
//!     text embedding space. Input `[batch, seq, intermediate_size]`,
//!     output `[batch, seq, text_hidden]`.
//!   - [`VoxtralModel::forward_with_audio`] — full end-to-end
//!     conditional generation: mel + tokens (with `audio_token_id`
//!     placeholders) → logits. Replaces audio tokens 1:1 with
//!     projected audio embeddings.
//!
//! # Deferrals
//!
//! - **KV cache** — every step recomputes the full decoder forward.
//!   A KV cache is a constant-factor speedup best layered on top of
//!   a working greedy decoder, like in `lazy_whisper`.
//! - **Audio preprocessing** (PCM → log-mel spectrogram): lives
//!   outside the lazy graph. Caller passes a pre-computed mel.
//! - **Long-audio chunking / overlap averaging** — the eager
//!   `process_long_audio` recursion. Caller is expected to chunk
//!   if the audio exceeds `max_source_positions`.
//! - **Voxtral-24B `softmax_in_fp32`** path — v1 runs softmax in
//!   the working dtype (F32 throughout this port), matching the
//!   F32-everywhere behavior the lazy-port family inherits.
//! - **Dropout / layer-drop / safe_clamp** — inference-only port,
//!   no stochastic depth.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_whisper::{conv1d_k3_s1_p1, conv1d_k3_s2_p1};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Configs ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
}

impl VoxtralEncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `mistralai/Voxtral-Mini-3B-2507` encoder defaults.
    pub fn voxtral_3b() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            num_mel_bins: 128,
            max_source_positions: 1500,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Voxtral carries head_dim **explicitly** — for 3B,
    /// `num_attention_heads * head_dim = 32 * 128 = 4096`, not
    /// `hidden_size = 3072`. So Q/K/V projections are non-square.
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl VoxtralTextConfig {
    pub fn voxtral_3b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 30,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 100_000_000.0,
            max_position_embeddings: 131072,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralConfig {
    pub audio: VoxtralEncoderConfig,
    pub text: VoxtralTextConfig,
    /// Token id that marks an audio-embedding slot in the text input.
    pub audio_token_id: u32,
}

impl VoxtralConfig {
    pub fn voxtral_3b() -> Self {
        Self {
            audio: VoxtralEncoderConfig::voxtral_3b(),
            text: VoxtralTextConfig::voxtral_3b(),
            audio_token_id: 24,
        }
    }
}

// ---- Encoder weights -------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VoxtralEncoderLayerWeights {
    pub self_attn_q: WeightStorage,
    pub self_attn_q_bias: Arc<[f32]>,
    pub self_attn_k: WeightStorage,
    pub self_attn_v: WeightStorage,
    pub self_attn_v_bias: Arc<[f32]>,
    pub self_attn_o: WeightStorage,
    pub self_attn_o_bias: Arc<[f32]>,
    pub self_attn_ln_gain: Arc<[f32]>,
    pub self_attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct VoxtralEncoderWeights {
    /// Conv1d kernel `[hidden, num_mel_bins, 3]`.
    pub conv1_w: Arc<[f32]>,
    pub conv1_b: Arc<[f32]>,
    /// Conv1d kernel `[hidden, hidden, 3]`.
    pub conv2_w: Arc<[f32]>,
    pub conv2_b: Arc<[f32]>,
    /// `[max_source_positions, hidden]` learned position embedding.
    pub embed_positions: Arc<[f32]>,
    pub layers: Vec<VoxtralEncoderLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

// ---- Encoder ---------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VoxtralEncoder {
    pub config: VoxtralEncoderConfig,
    pub weights: VoxtralEncoderWeights,
}

impl VoxtralEncoder {
    /// Forward the Whisper-shape audio encoder.
    ///
    /// `mel` is a `[1, num_mel_bins, mel_time]` log-mel spectrogram
    /// pre-computed externally. `mel_time` must be even (the stride-2
    /// conv2 halves it).
    ///
    /// Returns hidden states of shape
    /// `[1, mel_time/2, hidden_size]`.
    pub fn forward(&self, mel: &[f32], mel_time: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let n_mel = cfg.num_mel_bins;
        assert_eq!(
            mel.len(), n_mel * mel_time,
            "Voxtral encoder: mel has {} elements, expected {}×{}",
            mel.len(), n_mel, mel_time,
        );
        assert!(mel_time.is_multiple_of(2),
            "Voxtral encoder: mel_time must be even (got {mel_time})");
        let t_half = mel_time / 2;

        let mel_t = LazyTensor::from_f32(
            mel.to_vec(),
            Shape::from_dims(&[1, n_mel, mel_time]),
            &Device::cpu(),
        );

        // ---- conv stem (downsample) ---------------------------------------
        let x = conv1d_k3_s1_p1(
            &mel_t, &self.weights.conv1_w, &self.weights.conv1_b, n_mel, d, mel_time,
        ).gelu();
        let x = conv1d_k3_s2_p1(
            &x, &self.weights.conv2_w, &self.weights.conv2_b, d, d, mel_time,
        ).gelu();

        // ---- transpose to [1, T/2, d] + add learned positions -------------
        let x = x.permute([0, 2, 1_usize])?;
        let pos = x
            .const_f32_like(
                self.weights.embed_positions.clone(),
                Shape::from_dims(&[cfg.max_source_positions, d]),
            )
            .slice(0, 0, t_half)?
            .reshape(Shape::from_dims(&[1, t_half, d]))?
            .broadcast_to(Shape::from_dims(&[1, t_half, d]))?;
        let mut x = x.add(&pos)?;

        // ---- encoder layers -----------------------------------------------
        for lw in &self.weights.layers {
            x = encoder_layer(&x, lw, cfg)?;
        }

        // ---- final LayerNorm ----------------------------------------------
        x.layer_norm_affine(
            Arc::clone(&self.weights.final_ln_gain),
            Arc::clone(&self.weights.final_ln_bias),
            1e-5,
        )
    }
}

fn encoder_layer(
    x: &LazyTensor,
    lw: &VoxtralEncoderLayerWeights,
    cfg: &VoxtralEncoderConfig,
) -> Result<LazyTensor> {
    let d = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let scale = (head_dim as f64).powf(-0.5);

    // Pre-LN self-attention block.
    let x_ln = x.layer_norm_affine(
        Arc::clone(&lw.self_attn_ln_gain),
        Arc::clone(&lw.self_attn_ln_bias),
        1e-5,
    )?;

    // Q (biased) / K (bias-free) / V (biased).
    let q = lw.self_attn_q.apply_linear(&x_ln, d, d)
        .add_trailing_bias(Arc::clone(&lw.self_attn_q_bias))?;
    let k = lw.self_attn_k.apply_linear(&x_ln, d, d);
    let v = lw.self_attn_v.apply_linear(&x_ln, d, d)
        .add_trailing_bias(Arc::clone(&lw.self_attn_v_bias))?;

    // Match the eager reference: scaling is folded into Q rather than
    // into the post-matmul scores. Equivalent at F32.
    let q = q.mul_scalar(scale);

    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;

    let k_t = k.transpose()?;
    let scores = q.matmul(&k_t)?;
    let attn = scores.softmax_last_dim()?;
    let ctx = attn.matmul(&v)?;
    let merged = ctx.merge_heads()?;
    let attn_out = lw.self_attn_o.apply_linear(&merged, d, d)
        .add_trailing_bias(Arc::clone(&lw.self_attn_o_bias))?;
    let h1 = x.add(&attn_out)?;

    // Pre-LN FFN block (Linear → GELU → Linear).
    let h1_ln = h1.layer_norm_affine(
        Arc::clone(&lw.final_ln_gain),
        Arc::clone(&lw.final_ln_bias),
        1e-5,
    )?;
    let hidden = lw.fc1.apply_linear(&h1_ln, d, cfg.intermediate_size)
        .add_trailing_bias(Arc::clone(&lw.fc1_bias))?;
    let hidden = hidden.gelu();
    let ffn_out = lw.fc2.apply_linear(&hidden, cfg.intermediate_size, d)
        .add_trailing_bias(Arc::clone(&lw.fc2_bias))?;
    h1.add(&ffn_out)
}

// ---- Multi-modal projector -------------------------------------------------

#[derive(Debug, Clone)]
pub struct VoxtralMultiModalProjector {
    /// `[intermediate_size, text_hidden]`.
    pub linear_1: WeightStorage,
    /// `[text_hidden, text_hidden]`.
    pub linear_2: WeightStorage,
    pub audio_intermediate_size: usize,
    pub text_hidden: usize,
}

impl VoxtralMultiModalProjector {
    /// Project audio encoder hidden states into the text-embedding
    /// space.
    ///
    /// `audio` may be of any shape `[..., audio_intermediate_size]`;
    /// the input dim is `linear_1`'s in_features. Output is
    /// `[..., text_hidden]`.
    pub fn forward(&self, audio: &LazyTensor) -> Result<LazyTensor> {
        let x = self.linear_1.apply_linear(
            audio, self.audio_intermediate_size, self.text_hidden,
        );
        let x = x.gelu();
        Ok(self.linear_2.apply_linear(&x, self.text_hidden, self.text_hidden))
    }
}

// ---- Text-model weights (Llama-shape with explicit head_dim) ---------------

/// Per-layer weights for the Voxtral text model. Distinct from
/// `crate::lazy::LayerWeights` because Voxtral's Q/K/V are
/// non-square (`num_heads * head_dim` may not equal `hidden_size`).
#[derive(Debug, Clone)]
pub struct VoxtralTextLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    /// `[hidden_size, num_heads * head_dim]`.
    pub attn_q: WeightStorage,
    /// `[hidden_size, num_kv_heads * head_dim]`.
    pub attn_k: WeightStorage,
    /// `[hidden_size, num_kv_heads * head_dim]`.
    pub attn_v: WeightStorage,
    /// `[num_heads * head_dim, hidden_size]`.
    pub attn_o: WeightStorage,
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct VoxtralTextWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<VoxtralTextLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub lm_head: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct VoxtralTextModel {
    pub config: VoxtralTextConfig,
    pub weights: VoxtralTextWeights,
}

impl VoxtralTextModel {
    /// Plain text-only forward: tokens → logits `(1, seq, vocab)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let embeds = self.embed(tokens)?;
        self.forward_embeds(&embeds, start_pos)
    }

    /// Embed tokens via the token embedding table → `(1, seq, hidden)`.
    /// Bootstraps a fresh graph. Use [`Self::embed_with_anchor`] to
    /// share a graph with an existing tensor (e.g. audio features).
    pub fn embed(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32],
            Shape::from_dims(&[1]),
            &Device::cpu(),
        );
        let embed = self.embed_with_anchor(tokens, &anchor)?;
        let _ = cfg;
        Ok(embed)
    }

    /// Embed tokens onto `anchor`'s graph so the result can be
    /// composed with other tensors already on that graph (e.g.
    /// audio embeddings from the projector).
    pub fn embed_with_anchor(
        &self, tokens: &[u32], anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "VoxtralTextModel: tokens must be non-empty");
        let embed_table = anchor.const_f32_like(
            self.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
        );
        let token_ids = anchor.const_u32_like(
            tokens.to_vec(), Shape::from_dims(&[seq]),
        );
        embed_table
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[1, seq, cfg.hidden_size]))
    }

    /// Run the decoder on pre-computed embeddings of shape
    /// `(1, seq, hidden)`. Used by the multimodal forward, which
    /// substitutes projected audio embeddings into the text
    /// embedding tensor before running the decoder.
    pub fn forward_embeds(&self, embeds: &LazyTensor, start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "VoxtralTextModel: embeds must be rank-3");
        assert_eq!(dims[2], cfg.hidden_size,
            "VoxtralTextModel: embeds last dim must equal hidden_size");
        let seq = dims[1];

        let (rope_cos, rope_sin) = embeds.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        let mut h = embeds.clone();
        for layer in &weights.layers {
            h = apply_text_layer(&h, layer, cfg, &rope_cos, &rope_sin)?;
        }
        let h_norm = h.rms_norm_affine(
            Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps,
        )?;
        Ok(weights.lm_head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }
}

fn apply_text_layer(
    x: &LazyTensor,
    layer: &VoxtralTextLayerWeights,
    cfg: &VoxtralTextConfig,
    rope_cos: &LazyTensor,
    rope_sin: &LazyTensor,
) -> Result<LazyTensor> {
    let h = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let dims = x.shape();
    let dims = dims.dims();
    let seq = dims[1];

    let x_norm = x.rms_norm_affine(
        Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps,
    )?;
    let q = layer.attn_q.apply_linear(&x_norm, h, q_dim);
    let k = layer.attn_k.apply_linear(&x_norm, h, kv_dim);
    let v = layer.attn_v.apply_linear(&x_norm, h, kv_dim);

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
    let scores = q_r.matmul(&k_t)?.mul_scalar(scale);
    let mask = LazyTensor::additive_causal_mask_like(x, seq)
        .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
    let scores_masked = scores.broadcast_add(&mask)?;
    let attn = scores_masked.softmax_last_dim()?;
    let attn_v = attn.matmul(&v_full)?;
    let merged = attn_v.merge_heads()?;
    let attn_out = layer.attn_o.apply_linear(&merged, q_dim, h);
    let h1 = x.add(&attn_out)?;

    let h1_norm = h1.rms_norm_affine(
        Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps,
    )?;
    let gate = layer.ffn_gate.apply_linear(&h1_norm, h, cfg.intermediate_size);
    let up = layer.ffn_up.apply_linear(&h1_norm, h, cfg.intermediate_size);
    let swiglu = gate.silu().mul(&up)?;
    let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.intermediate_size, h);
    h1.add(&ffn_out)
}

// ---- Top-level conditional-generation model --------------------------------

#[derive(Debug, Clone)]
pub struct VoxtralWeights {
    pub audio: VoxtralEncoderWeights,
    pub projector_1: WeightStorage,
    pub projector_2: WeightStorage,
    pub text: VoxtralTextWeights,
}

#[derive(Debug, Clone)]
pub struct VoxtralModel {
    pub config: VoxtralConfig,
    pub weights: VoxtralWeights,
}

impl VoxtralModel {
    /// End-to-end audio-conditioned forward.
    ///
    /// `mel` is a `[1, num_mel_bins, mel_time]` pre-computed log-mel.
    /// `tokens` is the input text with `audio_token_id` placeholders
    /// where the projected audio embeddings should land. The number
    /// of `audio_token_id` occurrences MUST equal the number of audio
    /// embedding slots produced by the encoder+projector.
    ///
    /// Returns logits of shape `(1, seq, vocab_size)`.
    pub fn forward_with_audio(
        &self, mel: &[f32], mel_time: usize, tokens: &[u32], start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;

        // 1) Audio encoder → [1, mel_time/2, audio_hidden].
        let encoder = VoxtralEncoder {
            config: cfg.audio.clone(),
            weights: self.weights.audio.clone(),
        };
        let audio_hidden = encoder.forward(mel, mel_time)?;
        let a_dims = audio_hidden.shape();
        let a_dims = a_dims.dims();
        let batch = a_dims[0];
        let a_seq = a_dims[1];
        let a_hidden = a_dims[2];

        // 2) Reshape audio for projector: HF Voxtral reshapes to
        //    `[(batch * a_seq * a_hidden) / intermediate_size,
        //      intermediate_size]`. With Voxtral 3B's
        //    audio_hidden=1280, intermediate_size=5120, seq=1500,
        //    this becomes `[375, 5120]` per `[1, 1500, 1280]`.
        let total = batch * a_seq * a_hidden;
        let intermediate = cfg.audio.intermediate_size;
        assert_eq!(total % intermediate, 0,
            "Voxtral projector reshape: total {total} not divisible by intermediate_size {intermediate}");
        let new_batch = total / intermediate;
        let audio_flat = audio_hidden.reshape(Shape::from_dims(&[new_batch, intermediate]))?;

        // 3) Projector → [new_batch, text_hidden].
        let projector = VoxtralMultiModalProjector {
            linear_1: self.weights.projector_1.clone(),
            linear_2: self.weights.projector_2.clone(),
            audio_intermediate_size: intermediate,
            text_hidden: cfg.text.hidden_size,
        };
        let audio_embeds = projector.forward(&audio_flat)?;

        // 4) Text embeddings anchored on the audio graph so the
        //    substitute step can mix them.
        let text_model = VoxtralTextModel {
            config: cfg.text.clone(),
            weights: self.weights.text.clone(),
        };
        let text_embeds = text_model.embed_with_anchor(tokens, &audio_embeds)?;

        // 5) Substitute audio embeddings at audio_token positions.
        let composed = substitute_audio_embeds(
            &text_embeds, &audio_embeds, tokens, cfg.audio_token_id, cfg.text.hidden_size,
        )?;

        // 6) Run the decoder.
        text_model.forward_embeds(&composed, start_pos)
    }
}

/// Replace the rows of `text_embeds` at positions where
/// `tokens[i] == audio_token_id` with the corresponding rows of
/// `audio_embeds`. The number of `audio_token_id` occurrences must
/// equal `audio_embeds.shape()[0]`.
///
/// Implementation: a constant `mask` tensor `[1, seq, 1]` is built
/// at graph-build time (host-side, since tokens are known) marking
/// audio-token rows. A second `[seq, hidden]` tensor is built by
/// scattering audio_embeds rows in audio-token positions and zeros
/// elsewhere. Then `out = text * (1 - mask) + scatter_audio * mask`.
///
/// We use plain `LazyTensor` ops only — no scatter primitive needed.
fn substitute_audio_embeds(
    text_embeds: &LazyTensor,
    audio_embeds: &LazyTensor,
    tokens: &[u32],
    audio_token_id: u32,
    hidden: usize,
) -> Result<LazyTensor> {
    let seq = tokens.len();
    let a_dims = audio_embeds.shape();
    let a_dims = a_dims.dims();
    assert_eq!(a_dims.len(), 2,
        "substitute_audio_embeds: audio_embeds must be rank-2 [n, hidden]");
    assert_eq!(a_dims[1], hidden,
        "substitute_audio_embeds: audio_embeds last dim {} != text hidden {}",
        a_dims[1], hidden);

    let audio_positions: Vec<usize> = tokens.iter().enumerate()
        .filter_map(|(i, &t)| if t == audio_token_id { Some(i) } else { None })
        .collect();

    // If no audio tokens, return text embeds unchanged.
    if audio_positions.is_empty() {
        return Ok(text_embeds.clone());
    }

    assert_eq!(audio_positions.len(), a_dims[0],
        "substitute_audio_embeds: {} audio tokens vs {} audio embeddings",
        audio_positions.len(), a_dims[0]);

    // Build the scatter index `(seq,)`:
    //   for audio-token positions, the index into `audio_embeds`'s rows
    //   for text-token positions, a dummy index (we mask them out anyway)
    let mut scatter_indices = vec![0_u32; seq];
    for (audio_row, &pos) in audio_positions.iter().enumerate() {
        scatter_indices[pos] = audio_row as u32;
    }
    let idx_t = text_embeds.const_u32_like(
        scatter_indices, Shape::from_dims(&[seq]),
    );
    // [seq, hidden] — audio_embeds row per token position; text-token
    // rows are bogus but masked out.
    let scattered = audio_embeds.index_select(0_usize, &idx_t)?
        .reshape(Shape::from_dims(&[1, seq, hidden]))?;

    // [1, seq, 1] additive mask: 1.0 at audio positions, 0.0 else.
    let mut mask_data = vec![0.0_f32; seq];
    for &p in &audio_positions { mask_data[p] = 1.0; }
    let mask = text_embeds
        .const_f32_like(mask_data, Shape::from_dims(&[seq]))
        .reshape(Shape::from_dims(&[1, seq, 1]))?
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]))?;

    let one_minus = mask.affine(-1.0, 1.0);  // 1 - mask
    let text_part = text_embeds.mul(&one_minus)?;
    let audio_part = scattered.mul(&mask)?;
    text_part.add(&audio_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn tiny_encoder_cfg() -> VoxtralEncoderConfig {
        VoxtralEncoderConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_mel_bins: 4,
            max_source_positions: 8,
        }
    }

    fn tiny_encoder_weights(cfg: &VoxtralEncoderConfig) -> VoxtralEncoderWeights {
        let mut next = rng(11111);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let d = cfg.hidden_size;
        let layers: Vec<VoxtralEncoderLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| VoxtralEncoderLayerWeights {
                self_attn_q: WeightStorage::F32(vec_of(d * d)),
                self_attn_q_bias: vec_of(d),
                self_attn_k: WeightStorage::F32(vec_of(d * d)),
                self_attn_v: WeightStorage::F32(vec_of(d * d)),
                self_attn_v_bias: vec_of(d),
                self_attn_o: WeightStorage::F32(vec_of(d * d)),
                self_attn_o_bias: vec_of(d),
                self_attn_ln_gain: Arc::from(vec![1.0_f32; d]),
                self_attn_ln_bias: Arc::from(vec![0.0_f32; d]),
                fc1: WeightStorage::F32(vec_of(d * cfg.intermediate_size)),
                fc1_bias: vec_of(cfg.intermediate_size),
                fc2: WeightStorage::F32(vec_of(cfg.intermediate_size * d)),
                fc2_bias: vec_of(d),
                final_ln_gain: Arc::from(vec![1.0_f32; d]),
                final_ln_bias: Arc::from(vec![0.0_f32; d]),
            }).collect();
        VoxtralEncoderWeights {
            conv1_w: vec_of(d * cfg.num_mel_bins * 3),
            conv1_b: vec_of(d),
            conv2_w: vec_of(d * d * 3),
            conv2_b: vec_of(d),
            embed_positions: vec_of(cfg.max_source_positions * d),
            layers,
            final_ln_gain: Arc::from(vec![1.0_f32; d]),
            final_ln_bias: Arc::from(vec![0.0_f32; d]),
        }
    }

    #[test]
    fn encoder_forward_shape_and_finite() {
        let cfg = tiny_encoder_cfg();
        let weights = tiny_encoder_weights(&cfg);
        let enc = VoxtralEncoder { config: cfg.clone(), weights };
        let mel_time = 8;
        let mel: Vec<f32> = (0..cfg.num_mel_bins * mel_time)
            .map(|i| (i as f32 * 0.001) - 0.05).collect();
        let out = enc.forward(&mel, mel_time).unwrap();
        assert_eq!(out.shape().dims(), &[1, mel_time / 2, cfg.hidden_size]);
        let v = out.realize_f32();
        for &x in &v {
            assert!(x.is_finite(), "non-finite encoder output: {x}");
        }
    }

    #[test]
    fn projector_forward_shape_and_finite() {
        let mut next = rng(22222);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let audio_in = 8;
        let text_h = 6;
        let projector = VoxtralMultiModalProjector {
            linear_1: WeightStorage::F32(vec_of(audio_in * text_h)),
            linear_2: WeightStorage::F32(vec_of(text_h * text_h)),
            audio_intermediate_size: audio_in,
            text_hidden: text_h,
        };
        let audio = LazyTensor::from_f32(
            (0..4 * audio_in).map(|i| (i as f32 * 0.01) - 0.1).collect::<Vec<_>>(),
            Shape::from_dims(&[4, audio_in]),
            &Device::cpu(),
        );
        let out = projector.forward(&audio).unwrap();
        assert_eq!(out.shape().dims(), &[4, text_h]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite projector output: {v}");
        }
    }

    fn tiny_text_cfg() -> VoxtralTextConfig {
        VoxtralTextConfig {
            vocab_size: 32, hidden_size: 12, intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4, num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            max_position_embeddings: 64, tie_word_embeddings: false,
        }
    }

    fn tiny_text_weights(cfg: &VoxtralTextConfig) -> VoxtralTextWeights {
        let mut next = rng(33333);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let layers = (0..cfg.num_hidden_layers).map(|_| VoxtralTextLayerWeights {
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * q_dim)),
            attn_k: WeightStorage::F32(vec_of(h * kv_dim)),
            attn_v: WeightStorage::F32(vec_of(h * kv_dim)),
            attn_o: WeightStorage::F32(vec_of(q_dim * h)),
            ffn_gate: WeightStorage::F32(vec_of(h * cfg.intermediate_size)),
            ffn_up: WeightStorage::F32(vec_of(h * cfg.intermediate_size)),
            ffn_down: WeightStorage::F32(vec_of(cfg.intermediate_size * h)),
        }).collect();
        VoxtralTextWeights {
            token_embedding: vec_of(cfg.vocab_size * h),
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            lm_head: WeightStorage::F32(vec_of(h * cfg.vocab_size)),
        }
    }

    #[test]
    fn text_model_forward_shape_and_finite() {
        let cfg = tiny_text_cfg();
        let weights = tiny_text_weights(&cfg);
        let model = VoxtralTextModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite text-model logits: {v}");
        }
    }

    #[test]
    fn substitute_audio_embeds_replaces_marked_positions() {
        // 4-token input, audio_token_id = 99 at positions 1 and 3.
        let hidden = 3;
        let tokens = vec![10_u32, 99, 20, 99];
        let audio = LazyTensor::from_f32(
            vec![
                7.0_f32, 7.0, 7.0,
                8.0, 8.0, 8.0,
            ],
            Shape::from_dims(&[2, hidden]),
            &Device::cpu(),
        );
        // Anchor text on audio's graph so the substitute step can mix.
        let text = audio.const_f32_like(
            Arc::<[f32]>::from(vec![
                1.0_f32, 1.0, 1.0,
                2.0, 2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0,
            ]),
            Shape::from_dims(&[1, 4, hidden]),
        );
        let out = substitute_audio_embeds(&text, &audio, &tokens, 99, hidden).unwrap();
        let v = out.realize_f32();
        // Expected: row 0 = text 1, row 1 = audio 7, row 2 = text 3,
        //           row 3 = audio 8.
        let want = [
            1.0_f32, 1.0, 1.0,
            7.0, 7.0, 7.0,
            3.0, 3.0, 3.0,
            8.0, 8.0, 8.0,
        ];
        for (i, (&a, &b)) in v.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "row {} elt: {a} vs {b}", i / hidden);
        }
    }

    #[test]
    fn forward_with_audio_shape_and_finite() {
        // End-to-end: encoder + projector + audio-token substitution
        // + text decoder, on tiny synthetic configs sized so the
        // projector reshape divides evenly (mel_time/2 * audio_hidden
        // must be divisible by audio_intermediate_size, and the
        // resulting #audio_embeds must match #audio_tokens).
        //
        // tiny audio: hidden=8, intermediate=16, mel_time=8 → audio
        // hidden states [1, 4, 8], total = 32, intermediate=16 → 2
        // audio embeddings.
        let audio_cfg = tiny_encoder_cfg();
        let mut text_cfg = tiny_text_cfg();
        // Make text hidden match audio output for the projector tests
        // since the projector goes audio_intermediate → text_hidden.
        text_cfg.hidden_size = 12;
        let audio_w = tiny_encoder_weights(&audio_cfg);
        let text_w = tiny_text_weights(&text_cfg);
        let mut next = rng(44444);
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let projector_1 = WeightStorage::F32(
            vec_of(audio_cfg.intermediate_size * text_cfg.hidden_size));
        let projector_2 = WeightStorage::F32(
            vec_of(text_cfg.hidden_size * text_cfg.hidden_size));

        // audio_token_id must be < vocab_size since it flows through
        // the embedding lookup before being replaced. Pick a
        // distinguished in-vocab id.
        let cfg = VoxtralConfig {
            audio: audio_cfg.clone(), text: text_cfg.clone(), audio_token_id: 7,
        };
        let model = VoxtralModel {
            config: cfg.clone(),
            weights: VoxtralWeights {
                audio: audio_w, projector_1, projector_2, text: text_w,
            },
        };

        let mel_time = 8;
        let mel: Vec<f32> = (0..audio_cfg.num_mel_bins * mel_time)
            .map(|i| (i as f32 * 0.001) - 0.05).collect();

        // 2 audio tokens at positions 1, 3; 3 text tokens elsewhere.
        let tokens: Vec<u32> = vec![1, 7, 2, 7, 3];

        let logits = model.forward_with_audio(&mel, mel_time, &tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), text_cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite forward_with_audio logits: {v}");
        }
    }

    #[test]
    fn substitute_audio_embeds_no_audio_tokens_returns_text_unchanged() {
        let hidden = 2;
        let tokens = vec![1_u32, 2, 3];
        let audio = LazyTensor::from_f32(
            vec![0.0_f32, 0.0],
            Shape::from_dims(&[1, hidden]),
            &Device::cpu(),
        );
        let text = audio.const_f32_like(
            Arc::<[f32]>::from(vec![1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0]),
            Shape::from_dims(&[1, 3, hidden]),
        );
        let out = substitute_audio_embeds(&text, &audio, &tokens, 99, hidden).unwrap();
        let v = out.realize_f32();
        let want = [1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        for (a, b) in v.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
