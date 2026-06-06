//! MusicGen (Facebook / Meta AI) — lazy port.
//!
//! Decoder-only LM that predicts EnCodec audio tokens
//! conditioned on a text encoder output via cross-attention.
//! Architecture (HF `MusicgenForConditionalGeneration`):
//!
//!   - **Text encoder** (typically T5-base in the published
//!     checkpoints) produces `(1, text_len, encoder_hidden)`
//!     conditioning states. This lazy v1 ships a *light-weight*
//!     text adapter (embedding + projection) so the module is
//!     standalone-testable; the real T5 conditioning path is
//!     [`crate::lazy_t5`] and a multimodal wrapper can call
//!     [`MusicGenModel::forward_with_encoder_states`] to feed
//!     pre-computed encoder hidden states directly.
//!
//!   - **MusicGen decoder** — a Transformer LM with:
//!       1. **Multi-codebook input embeddings** — `num_codebooks`
//!          separate `[vocab, hidden]` tables; the per-codebook
//!          embeddings are summed along the codebook axis to
//!          produce the token embedding stream.
//!       2. **Sinusoidal positional embeddings** (Bart/MusicGen
//!          convention — half-sin half-cos concatenated; same
//!          frequency schedule used in `MusicgenSinusoidal
//!          PositionalEmbedding`).
//!       3. **Self-attention + encoder-attention (cross-attention)
//!          + FFN** per layer with LayerNorm pre-norms.
//!       4. **Multi-codebook lm_heads** — `num_codebooks` separate
//!          `[hidden, vocab]` projections; logits stack to
//!          `(batch * num_codebooks, seq, vocab)`.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32,
//! teacher-forced layout (full audio token sequence per call). The
//! text encoder is the lightweight built-in adapter; callers with a
//! real T5 stack should use
//! [`MusicGenModel::forward_with_encoder_states`].

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix, load_transposed_matrix_preserve_dtype,
    LazyTensor, WeightStorage,
};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicGenActivation {
    Relu,
    Gelu,
    GeluPytorchTanh,
    Silu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MusicGenConfig {
    /// Audio token vocab (per codebook). The actual embedding table
    /// width is `vocab_size + 1` so that the trailing pad/bos id
    /// (`vocab_size`) is representable, matching the eager
    /// `embed_dim = vocab_size + 1` convention.
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub num_codebooks: usize,
    /// Hidden size of the (external) text encoder feeding cross-
    /// attention. Equals `hidden_size` for the small/medium
    /// checkpoints (T5-base shares dim with the decoder).
    pub encoder_hidden_size: usize,
    /// Vocab size of the simple built-in text adapter. Ignored when
    /// callers feed encoder states directly via
    /// [`MusicGenModel::forward_with_encoder_states`].
    pub text_vocab_size: usize,
    pub scale_embedding: bool,
    pub activation_function: MusicGenActivation,
}

impl MusicGenConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `facebook/musicgen-small` preset (HF
    /// `MusicgenDecoderConfig`).
    pub fn musicgen_small() -> Self {
        Self {
            vocab_size: 2048,
            max_position_embeddings: 2048,
            num_hidden_layers: 24,
            ffn_dim: 4096,
            num_attention_heads: 16,
            hidden_size: 1024,
            num_codebooks: 4,
            encoder_hidden_size: 1024,
            text_vocab_size: 32128,
            scale_embedding: false,
            activation_function: MusicGenActivation::Gelu,
        }
    }
}

// ---- Weight structures -----------------------------------------------------

/// MusicGen attention weights. All four projections are bias-free
/// (matches the eager `linear_no_bias` calls in
/// `MusicgenAttention::load`).
#[derive(Debug, Clone)]
pub struct MusicGenAttentionWeights {
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub out_proj: WeightStorage,
}

/// Per-decoder-layer weights (self-attn + cross-attn + FFN).
#[derive(Debug, Clone)]
pub struct MusicGenDecoderLayerWeights {
    pub self_attn: MusicGenAttentionWeights,
    pub self_attn_ln_gain: Arc<[f32]>,
    pub self_attn_ln_bias: Arc<[f32]>,
    pub encoder_attn: MusicGenAttentionWeights,
    pub encoder_attn_ln_gain: Arc<[f32]>,
    pub encoder_attn_ln_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc2: WeightStorage,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MusicGenWeights {
    /// `num_codebooks` per-codebook embedding tables, each
    /// `[vocab_size + 1, hidden_size]`.
    pub embed_tokens: Vec<Arc<[f32]>>,
    pub layers: Vec<MusicGenDecoderLayerWeights>,
    /// `[hidden_size]` post-stack LayerNorm gain.
    pub final_ln_gain: Arc<[f32]>,
    /// `[hidden_size]` post-stack LayerNorm bias.
    pub final_ln_bias: Arc<[f32]>,
    /// `num_codebooks` per-codebook `[hidden_size, vocab_size]`
    /// output projections.
    pub lm_heads: Vec<WeightStorage>,
    /// Built-in text adapter: `[text_vocab_size, encoder_hidden_size]`
    /// embedding table the standalone forward looks up to produce
    /// cross-attention K/V. Optional — when the caller passes
    /// encoder states directly (`forward_with_encoder_states`) the
    /// adapter is unused and this can be a 1-element placeholder.
    pub text_encoder_embedding: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MusicGenModel {
    pub config: MusicGenConfig,
    pub weights: MusicGenWeights,
}

impl MusicGenModel {
    /// Standalone forward: looks up text token embeddings via the
    /// built-in text adapter, runs the decoder over `audio_tokens`
    /// cross-attending to those embeddings, returns logits of shape
    /// `(num_codebooks, seq_len, vocab_size)`.
    ///
    /// `audio_tokens` is laid out as `[num_codebooks * seq_len]`
    /// (matches HF's `(batch * num_codebooks, seq)` decoder input).
    /// `start_pos` offsets the sinusoidal position table.
    pub fn forward(
        &self,
        text_tokens: &[u32],
        audio_tokens: &[u32],
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!text_tokens.is_empty(), "text_tokens must be non-empty");
        assert!(!audio_tokens.is_empty(), "audio_tokens must be non-empty");
        let n_cb = cfg.num_codebooks;
        let total = audio_tokens.len();
        assert_eq!(
            total % n_cb,
            0,
            "audio_tokens len ({total}) must be divisible by num_codebooks ({n_cb})",
        );
        let seq_len = total / n_cb;

        // Build a graph anchor: the first codebook embedding table.
        // All other constants are anchored on the same graph through
        // `const_*_like` to keep cross-attention happy.
        let anchor = LazyTensor::from_f32(
            self.weights.embed_tokens[0].clone(),
            Shape::from_dims(&[cfg.vocab_size + 1, cfg.hidden_size]),
            &Device::cpu(),
        );

        // Built-in text adapter → encoder states `(1, text_len, enc_hidden)`.
        let encoder_states = self.encode_text_adapter(&anchor, text_tokens)?;

        self.decode(&anchor, audio_tokens, seq_len, &encoder_states, start_pos)
    }

    /// Forward with externally-computed encoder hidden states (e.g.
    /// from a real T5 stack). `encoder_states` must have shape
    /// `(1, text_len, encoder_hidden_size)`.
    pub fn forward_with_encoder_states(
        &self,
        audio_tokens: &[u32],
        encoder_states: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(!audio_tokens.is_empty(), "audio_tokens must be non-empty");
        let dims = encoder_states.shape();
        let dims = dims.dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.encoder_hidden_size {
            return Err(crate::Error::Msg(format!(
                "forward_with_encoder_states: expected encoder_states shape \
                 (1, text_len, encoder_hidden={}), got {:?}",
                cfg.encoder_hidden_size, dims,
            ))
            .bt());
        }
        let n_cb = cfg.num_codebooks;
        let total = audio_tokens.len();
        assert_eq!(
            total % n_cb,
            0,
            "audio_tokens len ({total}) must be divisible by num_codebooks ({n_cb})",
        );
        let seq_len = total / n_cb;

        // Anchor on the encoder states' graph so cross-attention
        // composes correctly.
        self.decode(encoder_states, audio_tokens, seq_len, encoder_states, start_pos)
    }

    fn encode_text_adapter(
        &self,
        anchor: &LazyTensor,
        text_tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let text_len = text_tokens.len();
        let embed = anchor.const_f32_like(
            Arc::clone(&self.weights.text_encoder_embedding),
            Shape::from_dims(&[cfg.text_vocab_size, cfg.encoder_hidden_size]),
        );
        let ids = anchor.const_u32_like(
            text_tokens.to_vec(),
            Shape::from_dims(&[text_len]),
        );
        embed
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[1, text_len, cfg.encoder_hidden_size]))
    }

    fn decode(
        &self,
        anchor: &LazyTensor,
        audio_tokens: &[u32],
        seq_len: usize,
        encoder_states: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let n_cb = cfg.num_codebooks;
        let batch = 1usize;
        let embed_dim = cfg.vocab_size + 1;
        assert_eq!(
            cfg.num_attention_heads * cfg.head_dim(),
            cfg.hidden_size,
            "num_attention_heads * head_dim must equal hidden_size",
        );

        // Sum multi-codebook embeddings: for each codebook `i`,
        // look up `audio_tokens[i*seq..i*seq+seq]` in
        // `embed_tokens[i]`, then sum.
        let mut summed: Option<LazyTensor> = None;
        for cb in 0..n_cb {
            let slice = &audio_tokens[cb * seq_len..(cb + 1) * seq_len];
            let table = anchor.const_f32_like(
                Arc::clone(&weights.embed_tokens[cb]),
                Shape::from_dims(&[embed_dim, cfg.hidden_size]),
            );
            let ids = anchor
                .const_u32_like(slice.to_vec(), Shape::from_dims(&[seq_len]));
            let part = table
                .index_select(0_usize, &ids)?
                .reshape(Shape::from_dims(&[batch, seq_len, cfg.hidden_size]))?;
            summed = Some(match summed {
                None => part,
                Some(acc) => acc.add(&part)?,
            });
        }
        let mut h = summed.expect("num_codebooks must be >= 1");
        if cfg.scale_embedding {
            h = h.mul_scalar((cfg.hidden_size as f64).sqrt());
        }

        // Sinusoidal positional embedding.
        let pos_table = build_sinusoidal_table(
            cfg.max_position_embeddings, cfg.hidden_size,
        );
        let pos_full = anchor.const_f32_like(
            Arc::from(pos_table),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.hidden_size]),
        );
        let pos_slice = pos_full
            .slice(0_usize, start_pos, seq_len)?
            .reshape(Shape::from_dims(&[1, seq_len, cfg.hidden_size]))?;
        h = h.add(&pos_slice.broadcast_to(
            Shape::from_dims(&[batch, seq_len, cfg.hidden_size]),
        )?)?;

        // Causal mask `[1, 1, seq, seq]`.
        let mut mask_data = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal_mask = anchor.const_f32_like(
            mask_data, Shape::from_dims(&[1, 1, seq_len, seq_len]),
        );

        for layer in &weights.layers {
            h = self.apply_decoder_layer(&h, layer, encoder_states, &causal_mask)?;
        }

        let h = h.layer_norm_affine(
            Arc::clone(&weights.final_ln_gain),
            Arc::clone(&weights.final_ln_bias),
            1e-5,
        )?;

        // Multi-codebook lm_heads, stacked along a new dim → reshape
        // to `(num_codebooks, seq_len, vocab_size)` (batch == 1).
        let mut per_codebook: Vec<LazyTensor> = Vec::with_capacity(n_cb);
        for head in &weights.lm_heads {
            let logits = head.apply_linear(&h, cfg.hidden_size, cfg.vocab_size);
            // `logits` has shape `(1, seq_len, vocab_size)`.
            per_codebook.push(logits.squeeze(0_usize)?);
        }
        let refs: Vec<&LazyTensor> = per_codebook.iter().collect();
        LazyTensor::stack(&refs, 0_usize)
            .map_err(|e| crate::Error::Msg(format!("stack lm_heads: {e}")).bt())
    }

    fn apply_decoder_layer(
        &self,
        x: &LazyTensor,
        layer: &MusicGenDecoderLayerWeights,
        encoder_states: &LazyTensor,
        causal_mask: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;

        // ---- Self-attention sub-block (pre-LN, residual outside) ---
        let residual = x.clone();
        let x_norm = x.layer_norm_affine(
            Arc::clone(&layer.self_attn_ln_gain),
            Arc::clone(&layer.self_attn_ln_bias),
            1e-5,
        )?;
        let self_attn = self.attention(
            &x_norm,
            &x_norm,
            &layer.self_attn,
            cfg.hidden_size,
            cfg.hidden_size,
            Some(causal_mask),
        )?;
        let h1 = residual.add(&self_attn)?;

        // ---- Cross-attention sub-block ----------------------------
        let residual = h1.clone();
        let h1_norm = h1.layer_norm_affine(
            Arc::clone(&layer.encoder_attn_ln_gain),
            Arc::clone(&layer.encoder_attn_ln_bias),
            1e-5,
        )?;
        let cross_attn = self.attention(
            &h1_norm,
            encoder_states,
            &layer.encoder_attn,
            cfg.hidden_size,
            cfg.encoder_hidden_size,
            None,
        )?;
        let h2 = residual.add(&cross_attn)?;

        // ---- FFN ---------------------------------------------------
        let residual = h2.clone();
        let h2_norm = h2.layer_norm_affine(
            Arc::clone(&layer.final_ln_gain),
            Arc::clone(&layer.final_ln_bias),
            1e-5,
        )?;
        let fc1 = layer.fc1.apply_linear(&h2_norm, cfg.hidden_size, cfg.ffn_dim);
        let activated = match cfg.activation_function {
            MusicGenActivation::Relu => fc1.relu(),
            MusicGenActivation::Gelu => fc1.gelu_erf(),
            MusicGenActivation::GeluPytorchTanh => fc1.gelu(),
            MusicGenActivation::Silu => fc1.silu(),
        };
        let fc2 = layer.fc2.apply_linear(&activated, cfg.ffn_dim, cfg.hidden_size);
        residual.add(&fc2)
    }

    /// Standard scaled-dot-product attention (no Q scaling baked
    /// here — the scale lives in `apply_decoder_layer` via the
    /// MusicGen convention of multiplying Q by `head_dim ** -0.5`
    /// inside the attention block). `q_in_dim` and `kv_in_dim` may
    /// differ for cross-attention (encoder hidden dim != decoder
    /// hidden dim).
    fn attention(
        &self,
        q_src: &LazyTensor,
        kv_src: &LazyTensor,
        w: &MusicGenAttentionWeights,
        q_in_dim: usize,
        kv_in_dim: usize,
        attn_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let inner = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let scaling = 1.0_f64 / (head_dim as f64).sqrt();

        let q = w.q_proj.apply_linear(q_src, q_in_dim, inner).mul_scalar(scaling);
        let k = w.k_proj.apply_linear(kv_src, kv_in_dim, inner);
        let v = w.v_proj.apply_linear(kv_src, kv_in_dim, inner);

        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_attention_heads, head_dim)?;
        let v = v.split_heads(cfg.num_attention_heads, head_dim)?;

        let k_t = k.transpose()?;
        let mut scores = q.matmul(&k_t)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        Ok(w.out_proj.apply_linear(&merged, inner, inner))
    }
}

// ---- Sinusoidal positional embedding ---------------------------------------

/// Build the `[max_positions, hidden]` MusicGen positional table.
///
/// Mirrors the eager `get_embedding` helper: `half = hidden / 2`,
/// `inv_freq[i] = exp(-i * ln(10000) / (half - 1))`, table column
/// layout is `[cos(t * inv_freq); sin(t * inv_freq)]`. Odd `hidden`
/// pads the trailing column with zeros (the eager `cat zeros` path).
fn build_sinusoidal_table(max_positions: usize, hidden: usize) -> Vec<f32> {
    assert!(hidden >= 2, "hidden must be >= 2");
    let half = hidden / 2;
    let mut out = vec![0.0_f32; max_positions * hidden];
    let inv_freq: Vec<f32> = if half > 1 {
        let scale = (10_000.0_f64.ln() / (half - 1) as f64) as f32;
        (0..half).map(|v| (-(v as f32) * scale).exp()).collect()
    } else {
        vec![1.0_f32]
    };
    for t in 0..max_positions {
        for i in 0..half {
            let arg = t as f32 * inv_freq[i];
            out[t * hidden + i] = arg.cos();
            out[t * hidden + half + i] = arg.sin();
        }
    }
    out
}

// ---- Safetensors loader ----------------------------------------------------

impl MusicGenWeights {
    /// Load MusicGen decoder weights from a memory-mapped safetensors
    /// file using the HuggingFace `MusicgenForCausalLM` naming
    /// convention (prefix `decoder.model.decoder.` is the standard
    /// `MusicgenForConditionalGeneration` shape).
    ///
    /// Expected tensor names (per HF
    /// `MusicgenForConditionalGeneration` checkpoint):
    /// - `decoder.model.decoder.embed_tokens.{i}.weight` — `[vocab+1, hidden]`
    /// - `decoder.model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight`
    ///   (transposed)
    /// - `decoder.model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj.weight`
    ///   (transposed)
    /// - `decoder.model.decoder.layers.{i}.fc1.weight` (transposed)
    /// - `decoder.model.decoder.layers.{i}.fc2.weight` (transposed)
    /// - `decoder.model.decoder.layers.{i}.{self_attn,encoder_attn,final}_layer_norm.{weight,bias}`
    /// - `decoder.model.decoder.layer_norm.{weight,bias}` (post-stack)
    /// - `decoder.lm_heads.{i}.weight` (transposed)
    /// - `text_encoder.shared.weight` (optional) — feeds the built-in
    ///   text adapter; if absent a single-element placeholder is
    ///   installed (callers must use
    ///   [`MusicGenModel::forward_with_encoder_states`]).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &MusicGenConfig,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let i_dim = cfg.ffn_dim;
        let embed_dim = cfg.vocab_size + 1;
        let prefix = "decoder.model.decoder";

        // ---- Multi-codebook embed_tokens -------------------------
        let mut embed_tokens: Vec<Arc<[f32]>> =
            Vec::with_capacity(cfg.num_codebooks);
        for cb in 0..cfg.num_codebooks {
            let name = format!("{prefix}.embed_tokens.{cb}.weight");
            let v = load_tensor_as_f32(st, &name)?;
            if v.len() != embed_dim * h {
                crate::bail!(
                    "{name}: {} elts, expected {} ({}×{})",
                    v.len(), embed_dim * h, embed_dim, h,
                );
            }
            embed_tokens.push(Arc::from(v));
        }

        // ---- Decoder layers --------------------------------------
        let mut layers: Vec<MusicGenDecoderLayerWeights> =
            Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}.layers.{li}");

            let self_attn = MusicGenAttentionWeights {
                q_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.self_attn.q_proj.weight"), h, h,
                )?,
                k_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.self_attn.k_proj.weight"), h, h,
                )?,
                v_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.self_attn.v_proj.weight"), h, h,
                )?,
                out_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.self_attn.out_proj.weight"), h, h,
                )?,
            };
            let self_attn_ln_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn_layer_norm.weight"),
            )?);
            let self_attn_ln_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn_layer_norm.bias"),
            )?);

            let encoder_attn = MusicGenAttentionWeights {
                q_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.encoder_attn.q_proj.weight"), h, h,
                )?,
                k_proj: load_transposed_matrix_preserve_dtype(
                    st,
                    &format!("{p}.encoder_attn.k_proj.weight"),
                    h, cfg.encoder_hidden_size,
                )?,
                v_proj: load_transposed_matrix_preserve_dtype(
                    st,
                    &format!("{p}.encoder_attn.v_proj.weight"),
                    h, cfg.encoder_hidden_size,
                )?,
                out_proj: load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.encoder_attn.out_proj.weight"), h, h,
                )?,
            };
            let encoder_attn_ln_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.encoder_attn_layer_norm.weight"),
            )?);
            let encoder_attn_ln_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.encoder_attn_layer_norm.bias"),
            )?);

            let fc1 = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.fc1.weight"), i_dim, h,
            )?;
            let fc2 = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.fc2.weight"), h, i_dim,
            )?;
            let final_ln_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.final_layer_norm.weight"),
            )?);
            let final_ln_bias = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.final_layer_norm.bias"),
            )?);

            layers.push(MusicGenDecoderLayerWeights {
                self_attn,
                self_attn_ln_gain,
                self_attn_ln_bias,
                encoder_attn,
                encoder_attn_ln_gain,
                encoder_attn_ln_bias,
                fc1,
                fc2,
                final_ln_gain,
                final_ln_bias,
            });
        }

        // ---- Post-stack LayerNorm --------------------------------
        let final_ln_gain = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.layer_norm.weight"),
        )?);
        let final_ln_bias = Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.layer_norm.bias"),
        )?);

        // ---- Per-codebook lm_heads -------------------------------
        let mut lm_heads: Vec<WeightStorage> = Vec::with_capacity(cfg.num_codebooks);
        for cb in 0..cfg.num_codebooks {
            let v = load_transposed_matrix(
                st,
                &format!("decoder.lm_heads.{cb}.weight"),
                cfg.vocab_size, h,
            )?;
            lm_heads.push(WeightStorage::F32(Arc::from(v)));
        }

        // ---- Optional text encoder embedding ---------------------
        let text_encoder_embedding: Arc<[f32]> =
            match st.get("text_encoder.shared.weight") {
                Ok(_) => {
                    let v = load_tensor_as_f32(st, "text_encoder.shared.weight")?;
                    let expected = cfg.text_vocab_size * cfg.encoder_hidden_size;
                    if v.len() != expected {
                        crate::bail!(
                            "text_encoder.shared.weight: {} elts, expected {expected}",
                            v.len(),
                        );
                    }
                    Arc::from(v)
                }
                Err(_) => Arc::from(vec![0.0_f32; 1]),
            };

        Ok(Self {
            embed_tokens,
            layers,
            final_ln_gain,
            final_ln_bias,
            lm_heads,
            text_encoder_embedding,
        })
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

    fn tiny_cfg() -> MusicGenConfig {
        MusicGenConfig {
            vocab_size: 16,
            max_position_embeddings: 32,
            num_hidden_layers: 2,
            ffn_dim: 16,
            num_attention_heads: 2,
            hidden_size: 8,
            num_codebooks: 4,
            encoder_hidden_size: 8,
            text_vocab_size: 12,
            scale_embedding: false,
            activation_function: MusicGenActivation::Gelu,
        }
    }

    fn tiny_attention_weights(
        out_dim: usize, q_in: usize, kv_in: usize, nb: &mut dyn FnMut() -> f32,
    ) -> MusicGenAttentionWeights {
        MusicGenAttentionWeights {
            q_proj: ws(q_in * out_dim, nb),
            k_proj: ws(kv_in * out_dim, nb),
            v_proj: ws(kv_in * out_dim, nb),
            out_proj: ws(out_dim * out_dim, nb),
        }
    }

    fn tiny_weights(cfg: &MusicGenConfig, seed: u32) -> MusicGenWeights {
        let mut nb = rng_seed(seed);
        let h = cfg.hidden_size;
        let embed_dim = cfg.vocab_size + 1;
        let i_dim = cfg.ffn_dim;
        let embed_tokens: Vec<Arc<[f32]>> = (0..cfg.num_codebooks)
            .map(|_| vec_of(embed_dim * h, &mut nb))
            .collect();
        let layers: Vec<MusicGenDecoderLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| MusicGenDecoderLayerWeights {
                self_attn: tiny_attention_weights(h, h, h, &mut nb),
                self_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
                self_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
                encoder_attn: tiny_attention_weights(
                    h, h, cfg.encoder_hidden_size, &mut nb,
                ),
                encoder_attn_ln_gain: Arc::from(vec![1.0_f32; h]),
                encoder_attn_ln_bias: Arc::from(vec![0.0_f32; h]),
                fc1: ws(h * i_dim, &mut nb),
                fc2: ws(i_dim * h, &mut nb),
                final_ln_gain: Arc::from(vec![1.0_f32; h]),
                final_ln_bias: Arc::from(vec![0.0_f32; h]),
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let lm_heads: Vec<WeightStorage> = (0..cfg.num_codebooks)
            .map(|_| ws(h * cfg.vocab_size, &mut nb))
            .collect();
        let text_encoder_embedding =
            vec_of(cfg.text_vocab_size * cfg.encoder_hidden_size, &mut nb);
        MusicGenWeights {
            embed_tokens,
            layers,
            final_ln_gain,
            final_ln_bias,
            lm_heads,
            text_encoder_embedding,
        }
    }

    fn tiny_model() -> MusicGenModel {
        let cfg = tiny_cfg();
        let weights = tiny_weights(&cfg, 2026);
        MusicGenModel { config: cfg, weights }
    }

    #[test]
    fn forward_shape_and_finite() {
        let model = tiny_model();
        let cfg = &model.config;
        let seq_len: usize = 5;
        let text_tokens: Vec<u32> = vec![1, 2, 3, 4];
        // 4 codebooks × 5 = 20 audio tokens (every value < vocab_size+1)
        let audio_tokens: Vec<u32> = (0..(cfg.num_codebooks * seq_len) as u32)
            .map(|i| i % cfg.vocab_size as u32)
            .collect();
        let logits = model.forward(&text_tokens, &audio_tokens, 0).unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[cfg.num_codebooks, seq_len, cfg.vocab_size],
        );
        for (i, &v) in logits.realize_f32().iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// Cross-attention is wired: changing text tokens must change
    /// the audio logits.
    #[test]
    fn cross_attention_is_wired() {
        let model = tiny_model();
        let cfg = &model.config;
        let seq_len: usize = 4;
        let audio: Vec<u32> = (0..(cfg.num_codebooks * seq_len) as u32)
            .map(|i| i % cfg.vocab_size as u32)
            .collect();
        let a = model.forward(&[1, 2, 3], &audio, 0).unwrap().realize_f32();
        let b = model.forward(&[7, 8, 9], &audio, 0).unwrap().realize_f32();
        let max_diff = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "cross-attention must condition on text tokens, max_diff = {max_diff}",
        );
    }

    /// `forward_with_encoder_states` must agree with `forward` when
    /// the encoder states are produced by the built-in adapter.
    #[test]
    fn forward_with_encoder_states_matches_forward() {
        let model = tiny_model();
        let cfg = &model.config;
        let seq_len: usize = 3;
        let text_tokens: Vec<u32> = vec![2, 3, 5];
        let audio: Vec<u32> = (0..(cfg.num_codebooks * seq_len) as u32)
            .map(|i| i % cfg.vocab_size as u32)
            .collect();
        let direct = model.forward(&text_tokens, &audio, 0).unwrap().realize_f32();

        // Hand-build encoder states the same way the adapter does.
        let anchor = LazyTensor::from_f32(
            model.weights.embed_tokens[0].clone(),
            Shape::from_dims(&[cfg.vocab_size + 1, cfg.hidden_size]),
            &Device::cpu(),
        );
        let enc = model.encode_text_adapter(&anchor, &text_tokens).unwrap();
        let via_enc = model
            .forward_with_encoder_states(&audio, &enc, 0)
            .unwrap()
            .realize_f32();
        assert_eq!(direct.len(), via_enc.len());
        let max_diff = direct.iter().zip(via_enc.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "forward / forward_with_encoder_states diverge, max_diff = {max_diff}",
        );
    }

    /// Causal mask is honored: changing an audio token at position
    /// `t` must NOT alter logits at positions < `t`.
    #[test]
    fn causal_mask_is_enforced() {
        let model = tiny_model();
        let cfg = &model.config;
        let seq_len: usize = 4;
        let text_tokens: Vec<u32> = vec![1, 2, 3];
        let mut audio_a: Vec<u32> = (0..(cfg.num_codebooks * seq_len) as u32)
            .map(|i| i % cfg.vocab_size as u32)
            .collect();
        let mut audio_b = audio_a.clone();
        // Flip the very last token of codebook 0 only — position
        // (seq_len - 1) inside codebook 0.
        let idx = seq_len - 1;
        audio_b[idx] = (audio_a[idx] + 3) % cfg.vocab_size as u32;

        let a = model.forward(&text_tokens, &audio_a, 0).unwrap().realize_f32();
        let b = model.forward(&text_tokens, &audio_b, 0).unwrap().realize_f32();

        // Per-codebook logits live at [cb, t, col] in the
        // (num_codebooks, seq_len, vocab) output. Compare positions
        // t < seq_len - 1 across all codebooks.
        let v = cfg.vocab_size;
        for cb in 0..cfg.num_codebooks {
            for t in 0..(seq_len - 1) {
                for col in 0..v {
                    let i = cb * seq_len * v + t * v + col;
                    assert!(
                        (a[i] - b[i]).abs() < 1e-4,
                        "causal mask violated at cb={cb} t={t} col={col}: {} vs {}",
                        a[i], b[i],
                    );
                }
            }
        }

        // Avoid the warning about unused mut.
        audio_a.clear();
        audio_b.clear();
    }

    #[test]
    fn sinusoidal_table_position_0() {
        let t = build_sinusoidal_table(4, 8);
        let half = 4;
        // Position 0: cos(0)=1, sin(0)=0.
        for i in 0..half {
            assert!((t[i] - 1.0).abs() < 1e-6, "cos[0, {i}] = {} != 1", t[i]);
            assert!(t[half + i].abs() < 1e-6, "sin[0, {i}] = {} != 0", t[half + i]);
        }
    }

    #[test]
    fn musicgen_small_preset_constructs() {
        let cfg = MusicGenConfig::musicgen_small();
        assert_eq!(cfg.num_codebooks, 4);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim(), 64);
    }
}
