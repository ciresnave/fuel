//! GGUF-quantized Whisper speech-recognition model.
//!
//! Architecture parallels [`crate::lazy_whisper`] one-to-one. The only
//! difference is that every `Linear` weight inside the encoder and
//! decoder attention + FFN blocks is stored as a `WeightStorage::Q4_0`
//! block matrix and routed through `qmatmul` instead of a dense F32
//! matmul. The conv1d stem at the front of the encoder stays F32 —
//! that's the standard GGUF Whisper convention (whisper.cpp keeps
//! `conv1`/`conv2` in F32/F16) and matches the eager Candle quantized
//! port which dequantizes the conv1 / conv2 tensors at load time.
//!
//! LayerNorm gains / biases, the (learned) decoder positional table,
//! and the token-embedding table likewise stay F32 — only the Q/K/V/O
//! and FC1/FC2 projection matrices flip to Q4_0.
//!
//! The model can be constructed two ways:
//! - [`QuantizedWhisperModel::from_f32_bake`] — given a fully-loaded
//!   plain [`crate::lazy_whisper::WhisperModel`], quantize each Linear
//!   matrix into Q4_0 on the fly. The F32 → Q4_0 round-trip is the
//!   same `BlockQ4_0::from_float` ggml/llama.cpp uses, so the result
//!   matches what `whisper.cpp` would write out of a fresh quantize
//!   pass.
//! - [`QuantizedWhisperModel::from_gguf`] — read a GGUF file written
//!   by whisper.cpp directly. *(Spec'd surface; the GGUF Whisper file
//!   layout has not been wired in this crate yet — the constructor
//!   returns an error pointing callers at `from_f32_bake` for now.
//!   Adding the loader is a mechanical re-use of the same GGUF reader
//!   the LLaMA port uses; see `lazy::LlamaWeights::load_from_gguf`.)*

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_whisper::{
    WhisperConfig, WhisperModel, conv1d_k3_s1_p1, conv1d_k3_s2_p1,
};
use fuel_core_types::Shape;
use fuel_quantized::{BlockQ4_0, GgmlType, QK4_0};
use std::sync::Arc;

// ---- Quantized weight structs ---------------------------------------------

/// Encoder-layer weights with all six projection matrices in Q4_0.
#[derive(Debug, Clone)]
pub struct QuantizedWhisperEncoderLayerWeights {
    pub self_attn_ln_g: Arc<[f32]>,
    pub self_attn_ln_b: Arc<[f32]>,
    pub q_w: WeightStorage,
    pub q_b: Arc<[f32]>,
    pub k_w: WeightStorage,
    pub v_w: WeightStorage,
    pub v_b: Arc<[f32]>,
    pub out_w: WeightStorage,
    pub out_b: Arc<[f32]>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
    pub fc1_w: WeightStorage,
    pub fc1_b: Arc<[f32]>,
    pub fc2_w: WeightStorage,
    pub fc2_b: Arc<[f32]>,
}

/// Decoder-layer weights with all ten projection matrices in Q4_0.
#[derive(Debug, Clone)]
pub struct QuantizedWhisperDecoderLayerWeights {
    pub self_ln_g: Arc<[f32]>,
    pub self_ln_b: Arc<[f32]>,
    pub self_q_w: WeightStorage,
    pub self_q_b: Arc<[f32]>,
    pub self_k_w: WeightStorage,
    pub self_v_w: WeightStorage,
    pub self_v_b: Arc<[f32]>,
    pub self_out_w: WeightStorage,
    pub self_out_b: Arc<[f32]>,
    pub cross_ln_g: Arc<[f32]>,
    pub cross_ln_b: Arc<[f32]>,
    pub cross_q_w: WeightStorage,
    pub cross_q_b: Arc<[f32]>,
    pub cross_k_w: WeightStorage,
    pub cross_v_w: WeightStorage,
    pub cross_v_b: Arc<[f32]>,
    pub cross_out_w: WeightStorage,
    pub cross_out_b: Arc<[f32]>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
    pub fc1_w: WeightStorage,
    pub fc1_b: Arc<[f32]>,
    pub fc2_w: WeightStorage,
    pub fc2_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct QuantizedWhisperEncoderWeights {
    pub conv1_w: Arc<[f32]>,
    pub conv1_b: Arc<[f32]>,
    pub conv2_w: Arc<[f32]>,
    pub conv2_b: Arc<[f32]>,
    pub positional: Arc<[f32]>,
    pub layers: Vec<QuantizedWhisperEncoderLayerWeights>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct QuantizedWhisperDecoderWeights {
    pub embed_tokens: Arc<[f32]>,
    pub embed_positions: Arc<[f32]>,
    pub layers: Vec<QuantizedWhisperDecoderLayerWeights>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct QuantizedWhisperWeights {
    pub encoder: QuantizedWhisperEncoderWeights,
    pub decoder: QuantizedWhisperDecoderWeights,
}

#[derive(Debug, Clone)]
pub struct QuantizedWhisperModel {
    pub config: WhisperConfig,
    pub weights: QuantizedWhisperWeights,
}

// ---- Quantization helper --------------------------------------------------

/// Quantize a row-major `[in_features, out_features]` F32 weight matrix
/// (the layout `lazy_whisper`'s safetensors loader produces, ready for
/// `x @ W`) into a `WeightStorage::Q4_0` value. The conversion:
///
/// 1. Transposes back to `[out_features, in_features]` — the GGUF /
///    llama.cpp layout that `qmatmul`'s block stream expects.
/// 2. Quantizes each output row of `in_features` floats into
///    `in_features / 32` `BlockQ4_0` structs via the standard
///    `BlockQ4_0::from_float` (same code path as ggml).
/// 3. Reinterprets the resulting block array as raw little-endian
///    bytes and packs them into the `u32` word stream the
///    `WeightStorage::Q4_0` variant stores.
///
/// `in_features` must be a multiple of 32 (the Q4_0 block size).
fn quantize_in_out_to_q4_0(
    w_in_out: &[f32],
    in_features: usize,
    out_features: usize,
) -> crate::Result<WeightStorage> {
    if w_in_out.len() != in_features * out_features {
        return Err(crate::Error::Msg(format!(
            "quantize_in_out_to_q4_0: weight has {} elements, expected {} × {} = {}",
            w_in_out.len(), in_features, out_features, in_features * out_features,
        )).bt());
    }
    if !in_features.is_multiple_of(QK4_0) {
        return Err(crate::Error::Msg(format!(
            "quantize_in_out_to_q4_0: in_features ({in_features}) must be a multiple of Q4_0 block size ({QK4_0})",
        )).bt());
    }
    let mut w_out_in = vec![0.0_f32; out_features * in_features];
    for o in 0..out_features {
        for i in 0..in_features {
            w_out_in[o * in_features + i] = w_in_out[i * out_features + o];
        }
    }
    let blocks_per_row = in_features / QK4_0;
    let n_blocks = out_features * blocks_per_row;
    let mut blocks: Vec<BlockQ4_0> = vec![BlockQ4_0::zeros(); n_blocks];
    BlockQ4_0::from_float(&w_out_in, &mut blocks);

    let bytes_per_block = std::mem::size_of::<BlockQ4_0>();
    let total_bytes = n_blocks * bytes_per_block;
    // SAFETY: BlockQ4_0 is #[repr(C)] with size_of == 18; reinterpreting
    // a contiguous slice as a packed byte stream gives the exact byte
    // representation a GGUF file would store on disk.
    let bytes_view: &[u8] = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total_bytes)
    };
    // The Q4_0 byte stream has 18-byte blocks; for the u32 reinterpretation
    // to be exact, the total byte length must be a multiple of 4. That
    // requires `n_blocks` to be even (since 18 mod 4 == 2). Pad with one
    // extra zero block when odd — those bytes are never read by `qmatmul`
    // since the kernel walks exactly `n * (k / 32)` blocks per matmul.
    let (padded_bytes, padded_byte_len);
    let bytes_for_words: &[u8] = if total_bytes.is_multiple_of(4) {
        padded_byte_len = total_bytes;
        bytes_view
    } else {
        padded_bytes = {
            let pad = 4 - (total_bytes % 4);
            let mut v = Vec::with_capacity(total_bytes + pad);
            v.extend_from_slice(bytes_view);
            v.extend(std::iter::repeat_n(0u8, pad));
            v
        };
        padded_byte_len = padded_bytes.len();
        padded_bytes.as_slice()
    };
    let mut words: Vec<u32> = Vec::with_capacity(padded_byte_len / 4);
    for chunk in bytes_for_words.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    drop(blocks);
    Ok(WeightStorage::Q4_0 {
        words: Arc::from(words),
        bytes_len: total_bytes,
        in_features,
        out_features,
    })
}

// ---- Quantized model forward ---------------------------------------------

impl QuantizedWhisperModel {
    /// Bake every Linear matrix of a plain `WhisperModel` into Q4_0.
    /// Conv stem + LayerNorms + positional tables + token embedding stay
    /// F32. Returns an error if any matmul's `in_features` is not a
    /// multiple of 32 (the Q4_0 block size); for HF Whisper this holds
    /// — `d_model` is always 384 / 512 / 768 / 1024 / 1280 across the
    /// release line.
    pub fn from_f32_bake(plain: &WhisperModel) -> crate::Result<Self> {
        let cfg = plain.config.clone();
        let d = cfg.d_model;
        let h_ff_enc = cfg.encoder_ffn_dim;
        let h_ff_dec = cfg.decoder_ffn_dim;

        let mut enc_layers = Vec::with_capacity(plain.weights.encoder.layers.len());
        for lw in &plain.weights.encoder.layers {
            enc_layers.push(QuantizedWhisperEncoderLayerWeights {
                self_attn_ln_g: Arc::clone(&lw.self_attn_ln_g),
                self_attn_ln_b: Arc::clone(&lw.self_attn_ln_b),
                q_w: quantize_in_out_to_q4_0(&lw.q_w, d, d)?,
                q_b: Arc::clone(&lw.q_b),
                k_w: quantize_in_out_to_q4_0(&lw.k_w, d, d)?,
                v_w: quantize_in_out_to_q4_0(&lw.v_w, d, d)?,
                v_b: Arc::clone(&lw.v_b),
                out_w: quantize_in_out_to_q4_0(&lw.out_w, d, d)?,
                out_b: Arc::clone(&lw.out_b),
                final_ln_g: Arc::clone(&lw.final_ln_g),
                final_ln_b: Arc::clone(&lw.final_ln_b),
                fc1_w: quantize_in_out_to_q4_0(&lw.fc1_w, d, h_ff_enc)?,
                fc1_b: Arc::clone(&lw.fc1_b),
                fc2_w: quantize_in_out_to_q4_0(&lw.fc2_w, h_ff_enc, d)?,
                fc2_b: Arc::clone(&lw.fc2_b),
            });
        }

        let mut dec_layers = Vec::with_capacity(plain.weights.decoder.layers.len());
        for lw in &plain.weights.decoder.layers {
            dec_layers.push(QuantizedWhisperDecoderLayerWeights {
                self_ln_g: Arc::clone(&lw.self_ln_g),
                self_ln_b: Arc::clone(&lw.self_ln_b),
                self_q_w: quantize_in_out_to_q4_0(&lw.self_q_w, d, d)?,
                self_q_b: Arc::clone(&lw.self_q_b),
                self_k_w: quantize_in_out_to_q4_0(&lw.self_k_w, d, d)?,
                self_v_w: quantize_in_out_to_q4_0(&lw.self_v_w, d, d)?,
                self_v_b: Arc::clone(&lw.self_v_b),
                self_out_w: quantize_in_out_to_q4_0(&lw.self_out_w, d, d)?,
                self_out_b: Arc::clone(&lw.self_out_b),
                cross_ln_g: Arc::clone(&lw.cross_ln_g),
                cross_ln_b: Arc::clone(&lw.cross_ln_b),
                cross_q_w: quantize_in_out_to_q4_0(&lw.cross_q_w, d, d)?,
                cross_q_b: Arc::clone(&lw.cross_q_b),
                cross_k_w: quantize_in_out_to_q4_0(&lw.cross_k_w, d, d)?,
                cross_v_w: quantize_in_out_to_q4_0(&lw.cross_v_w, d, d)?,
                cross_v_b: Arc::clone(&lw.cross_v_b),
                cross_out_w: quantize_in_out_to_q4_0(&lw.cross_out_w, d, d)?,
                cross_out_b: Arc::clone(&lw.cross_out_b),
                final_ln_g: Arc::clone(&lw.final_ln_g),
                final_ln_b: Arc::clone(&lw.final_ln_b),
                fc1_w: quantize_in_out_to_q4_0(&lw.fc1_w, d, h_ff_dec)?,
                fc1_b: Arc::clone(&lw.fc1_b),
                fc2_w: quantize_in_out_to_q4_0(&lw.fc2_w, h_ff_dec, d)?,
                fc2_b: Arc::clone(&lw.fc2_b),
            });
        }

        Ok(Self {
            config: cfg,
            weights: QuantizedWhisperWeights {
                encoder: QuantizedWhisperEncoderWeights {
                    conv1_w: Arc::clone(&plain.weights.encoder.conv1_w),
                    conv1_b: Arc::clone(&plain.weights.encoder.conv1_b),
                    conv2_w: Arc::clone(&plain.weights.encoder.conv2_w),
                    conv2_b: Arc::clone(&plain.weights.encoder.conv2_b),
                    positional: Arc::clone(&plain.weights.encoder.positional),
                    layers: enc_layers,
                    final_ln_g: Arc::clone(&plain.weights.encoder.final_ln_g),
                    final_ln_b: Arc::clone(&plain.weights.encoder.final_ln_b),
                },
                decoder: QuantizedWhisperDecoderWeights {
                    embed_tokens: Arc::clone(&plain.weights.decoder.embed_tokens),
                    embed_positions: Arc::clone(&plain.weights.decoder.embed_positions),
                    layers: dec_layers,
                    final_ln_g: Arc::clone(&plain.weights.decoder.final_ln_g),
                    final_ln_b: Arc::clone(&plain.weights.decoder.final_ln_b),
                },
            },
        })
    }

    /// Placeholder for the GGUF-on-disk constructor. The GGUF Whisper
    /// file layout reuses the same key naming whisper.cpp emits; the
    /// loader is a mechanical re-use of the GGUF reader the LLaMA port
    /// already ships. Wiring it up is a follow-up — callers that have
    /// a `WhisperModel` should use `from_f32_bake` instead.
    pub fn from_gguf<P: AsRef<std::path::Path>>(_path: P) -> crate::Result<Self> {
        Err(crate::Error::Msg(
            "QuantizedWhisperModel::from_gguf: GGUF loader not yet wired in this crate. \
             Use QuantizedWhisperModel::from_f32_bake on a loaded WhisperModel as the bridge."
                .into(),
        ).bt())
    }

    /// Run the encoder. Same forward path as `WhisperModel::forward_encoder`
    /// (conv stem + sinusoidal positional add + transformer blocks +
    /// final LN); the only difference is that the attention + FFN
    /// projections inside each block read from Q4_0 weights via
    /// `WeightStorage::apply_linear` → `qmatmul`.
    pub fn forward_encoder(&self, mel: &[f32], mel_time: usize) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let n_mel = cfg.num_mel_bins;
        if mel.len() != n_mel * mel_time {
            return Err(crate::Error::Msg(format!(
                "forward_encoder: mel has {} elements, expected {}×{}",
                mel.len(), n_mel, mel_time,
            )).bt());
        }
        let mel_t = LazyTensor::from_f32(
            mel.to_vec(), Shape::from_dims(&[1, n_mel, mel_time]), &crate::Device::cpu(),
        );

        let x = conv1d_k3_s1_p1(
            &mel_t,
            &self.weights.encoder.conv1_w,
            &self.weights.encoder.conv1_b,
            n_mel, d, mel_time,
        ).gelu();
        if !mel_time.is_multiple_of(2) {
            return Err(crate::Error::Msg(
                "forward_encoder: mel_time must be even for stride-2 conv".into(),
            ).bt());
        }
        let t_half = mel_time / 2;
        let x = conv1d_k3_s2_p1(
            &x,
            &self.weights.encoder.conv2_w,
            &self.weights.encoder.conv2_b,
            d, d, mel_time,
        ).gelu();

        let x = x.permute([0, 2, 1_usize])?;
        let pos = x
            .const_f32_like(
                Arc::clone(&self.weights.encoder.positional),
                Shape::from_dims(&[cfg.max_source_positions, d]),
            )
            .slice(0, 0, t_half)?
            .reshape(Shape::from_dims(&[1, t_half, d]))?
            .broadcast_to(Shape::from_dims(&[1, t_half, d]))?;
        let mut x = x.add(&pos)?;

        for lw in &self.weights.encoder.layers {
            x = encoder_layer(&x, lw, cfg, t_half)?;
        }

        layer_norm_affine(
            &x,
            &self.weights.encoder.final_ln_g,
            &self.weights.encoder.final_ln_b,
            1e-5, d, t_half,
        )
    }

    /// Run the decoder against a precomputed encoder context, returning
    /// `[1, seq, vocab_size]` logits via tied output projection.
    pub fn forward_decoder(&self, tokens: &[u32], encoder_out: &LazyTensor) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let seq = tokens.len();
        if seq == 0 {
            return Err(crate::Error::Msg("forward_decoder: empty tokens".into()).bt());
        }
        if seq > cfg.max_target_positions {
            return Err(crate::Error::Msg(format!(
                "forward_decoder: seq {seq} > max_target_positions {}",
                cfg.max_target_positions,
            )).bt());
        }

        let input_ids = encoder_out.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let embed = encoder_out.const_f32_like(
            Arc::clone(&self.weights.decoder.embed_tokens),
            Shape::from_dims(&[cfg.vocab_size, d]),
        );
        let position_ids_vec: Vec<u32> = (0..seq as u32).collect();
        let position_ids = encoder_out.const_u32_like(position_ids_vec, Shape::from_dims(&[seq]));
        let pos_emb = encoder_out.const_f32_like(
            Arc::clone(&self.weights.decoder.embed_positions),
            Shape::from_dims(&[cfg.max_target_positions, d]),
        );

        let tok = embed.index_select(0, &input_ids)?;
        let pos = pos_emb.index_select(0, &position_ids)?;
        let mut x = tok.add(&pos)?.reshape(Shape::from_dims(&[1, seq, d]))?;

        for lw in &self.weights.decoder.layers {
            x = decoder_layer(&x, encoder_out, lw, cfg, seq)?;
        }

        let x = layer_norm_affine(
            &x,
            &self.weights.decoder.final_ln_g,
            &self.weights.decoder.final_ln_b,
            1e-5, d, seq,
        )?;
        let embed_t = embed.transpose()?;
        Ok(x.matmul(&embed_t)?)
    }

    /// Greedy generation paralleling `WhisperModel::generate_greedy`:
    /// run the encoder once, then re-run the full decoder forward per
    /// step picking the argmax token until EOS or `max_new_tokens`.
    pub fn generate_greedy(
        &self,
        mel: &[f32],
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> crate::Result<Vec<u32>> {
        let mel_time = mel.len() / self.config.num_mel_bins;
        let encoder_out = self.forward_encoder(mel, mel_time)?.realize_f32();
        let t_half = mel_time / 2;
        let enc_shape = Shape::from_dims(&[1, t_half, self.config.d_model]);
        let eos = self.config.eos_token_id;

        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        for _ in 0..max_new_tokens {
            let encoder_t = LazyTensor::from_f32(
                encoder_out.clone(), enc_shape.clone(), &crate::Device::cpu(),
            );
            let logits = self.forward_decoder(&tokens, &encoder_t)?;
            let flat = logits.realize_f32();
            let vocab = self.config.vocab_size;
            let last_row_start = (tokens.len() - 1) * vocab;
            let row = &flat[last_row_start..last_row_start + vocab];
            let (argmax, _) = row.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            );
            let next_tok = argmax as u32;
            tokens.push(next_tok);
            if next_tok == eos { break; }
        }
        Ok(tokens)
    }
}

// ---- Layer primitives -----------------------------------------------------

fn layer_norm_affine(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    eps: f64,
    hidden: usize,
    seq: usize,
) -> crate::Result<LazyTensor> {
    let normed = x.layer_norm_last_dim(eps)?;
    let g = x
        .const_f32_like(Arc::clone(gamma), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))?
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]))?;
    let b = x
        .const_f32_like(Arc::clone(beta), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))?
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]))?;
    Ok(normed.mul(&g)?.add(&b)?)
}

/// Q4_0 linear with optional broadcast bias add. `in_f`/`out_f` are the
/// logical matrix dims; `WeightStorage::apply_linear` dispatches to
/// `qmatmul` for Q4_0 and to plain matmul for F32 (so this same helper
/// can run a hybrid model where only some matrices are quantized).
fn q_linear(
    x: &LazyTensor,
    w: &WeightStorage,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> crate::Result<LazyTensor> {
    let proj = w.apply_linear(x, in_f, out_f);
    match b {
        Some(bias) => {
            let bias_t = proj
                .const_f32_like(Arc::clone(bias), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f]))?
                .broadcast_to(Shape::from_dims(&[1, seq, out_f]))?;
            Ok(proj.add(&bias_t)?)
        }
        None => Ok(proj),
    }
}

#[allow(clippy::too_many_arguments)]
fn multi_head_attn(
    q_src: &LazyTensor,
    k_src: &LazyTensor,
    v_src: &LazyTensor,
    q_w: &WeightStorage, q_b: &Arc<[f32]>,
    k_w: &WeightStorage,
    v_w: &WeightStorage, v_b: &Arc<[f32]>,
    out_w: &WeightStorage, out_b: &Arc<[f32]>,
    d: usize, n_heads: usize, d_head: usize,
    q_seq: usize, kv_seq: usize, causal: bool,
) -> crate::Result<LazyTensor> {
    let q = q_linear(q_src, q_w, Some(q_b), d, d, q_seq)?;
    let k = q_linear(k_src, k_w, None, d, d, kv_seq)?;
    let v = q_linear(v_src, v_w, Some(v_b), d, d, kv_seq)?;

    let q = q.split_heads(n_heads, d_head)?;
    let k = k.split_heads(n_heads, d_head)?;
    let v = v.split_heads(n_heads, d_head)?;

    let k_t = k.permute([0, 1, 3, 2_usize])?;
    let scale = 1.0_f64 / (d_head as f64).sqrt();
    let mut scores = q.matmul(&k_t)?.mul_scalar(scale);

    if causal {
        let mut mask = vec![0.0_f32; q_seq * kv_seq];
        for i in 0..q_seq {
            for j in 0..kv_seq {
                if j > i { mask[i * kv_seq + j] = f32::NEG_INFINITY; }
            }
        }
        let mask_t = scores
            .const_f32_like(mask, Shape::from_dims(&[q_seq, kv_seq]))
            .reshape(Shape::from_dims(&[1, 1, q_seq, kv_seq]))?
            .broadcast_to(Shape::from_dims(&[1, n_heads, q_seq, kv_seq]))?;
        scores = scores.add(&mask_t)?;
    }

    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?.merge_heads()?;
    q_linear(&ctx, out_w, Some(out_b), d, d, q_seq)
}

fn encoder_layer(
    x: &LazyTensor,
    lw: &QuantizedWhisperEncoderLayerWeights,
    cfg: &WhisperConfig,
    seq: usize,
) -> crate::Result<LazyTensor> {
    let d = cfg.d_model;
    let n_heads = cfg.encoder_attention_heads;
    let d_head = cfg.encoder_head_dim();

    let x_ln = layer_norm_affine(x, &lw.self_attn_ln_g, &lw.self_attn_ln_b, 1e-5, d, seq)?;
    let attn = multi_head_attn(
        &x_ln, &x_ln, &x_ln,
        &lw.q_w, &lw.q_b, &lw.k_w, &lw.v_w, &lw.v_b, &lw.out_w, &lw.out_b,
        d, n_heads, d_head, seq, seq, false,
    )?;
    let x = x.add(&attn)?;

    let x_ln = layer_norm_affine(&x, &lw.final_ln_g, &lw.final_ln_b, 1e-5, d, seq)?;
    let h_ff = cfg.encoder_ffn_dim;
    let mid = q_linear(&x_ln, &lw.fc1_w, Some(&lw.fc1_b), d, h_ff, seq)?.gelu();
    let ffn = q_linear(&mid, &lw.fc2_w, Some(&lw.fc2_b), h_ff, d, seq)?;
    Ok(x.add(&ffn)?)
}

fn decoder_layer(
    x: &LazyTensor,
    encoder_out: &LazyTensor,
    lw: &QuantizedWhisperDecoderLayerWeights,
    cfg: &WhisperConfig,
    q_seq: usize,
) -> crate::Result<LazyTensor> {
    let d = cfg.d_model;
    let n_heads = cfg.decoder_attention_heads;
    let d_head = cfg.decoder_head_dim();
    let kv_seq_self = q_seq;

    let x_ln = layer_norm_affine(x, &lw.self_ln_g, &lw.self_ln_b, 1e-5, d, q_seq)?;
    let self_attn = multi_head_attn(
        &x_ln, &x_ln, &x_ln,
        &lw.self_q_w, &lw.self_q_b, &lw.self_k_w, &lw.self_v_w, &lw.self_v_b,
        &lw.self_out_w, &lw.self_out_b,
        d, n_heads, d_head, q_seq, kv_seq_self, true,
    )?;
    let x = x.add(&self_attn)?;

    let x_ln = layer_norm_affine(&x, &lw.cross_ln_g, &lw.cross_ln_b, 1e-5, d, q_seq)?;
    let enc_shape = encoder_out.shape();
    let enc_dims = enc_shape.dims();
    if enc_dims.len() != 3 {
        return Err(crate::Error::Msg(format!(
            "encoder_out must be [1, T, d]; got {enc_dims:?}",
        )).bt());
    }
    let kv_seq_cross = enc_dims[1];
    let cross = multi_head_attn(
        &x_ln, encoder_out, encoder_out,
        &lw.cross_q_w, &lw.cross_q_b, &lw.cross_k_w, &lw.cross_v_w, &lw.cross_v_b,
        &lw.cross_out_w, &lw.cross_out_b,
        d, n_heads, d_head, q_seq, kv_seq_cross, false,
    )?;
    let x = x.add(&cross)?;

    let x_ln = layer_norm_affine(&x, &lw.final_ln_g, &lw.final_ln_b, 1e-5, d, q_seq)?;
    let h_ff = cfg.decoder_ffn_dim;
    let mid = q_linear(&x_ln, &lw.fc1_w, Some(&lw.fc1_b), d, h_ff, q_seq)?.gelu();
    let ffn = q_linear(&mid, &lw.fc2_w, Some(&lw.fc2_b), h_ff, d, q_seq)?;
    Ok(x.add(&ffn)?)
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_whisper::{
        WhisperConfig, WhisperDecoderLayerWeights, WhisperDecoderWeights,
        WhisperEncoderLayerWeights, WhisperEncoderWeights, WhisperModel, WhisperWeights,
    };

    /// Tiny Whisper variant sized for the Q4_0 block constraint:
    /// `d_model` and both FFN dims are multiples of 32, the smallest
    /// power-of-two that satisfies the Q4_0 block size.
    fn tiny_q4_cfg() -> WhisperConfig {
        WhisperConfig {
            vocab_size:              64,
            num_mel_bins:             8,
            d_model:                 32,
            encoder_layers:           2,
            encoder_attention_heads:  4,
            encoder_ffn_dim:         32,
            decoder_layers:           2,
            decoder_attention_heads:  4,
            decoder_ffn_dim:         32,
            max_source_positions:    16,
            max_target_positions:    16,
            scale_embedding:      false,
            bos_token_id:             1,
            eos_token_id:             2,
            pad_token_id:             0,
            decoder_start_token_id:   1,
        }
    }

    /// Deterministic small-magnitude weights — keeps the F32 → Q4_0
    /// round-trip's per-element error well under saturation. Mirrors
    /// the LCG generator used by other lazy-port test helpers.
    fn deterministic_weights(cfg: &WhisperConfig) -> WhisperWeights {
        let mut s: u32 = 42;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let d = cfg.d_model;
        let n_mel = cfg.num_mel_bins;

        let encoder = WhisperEncoderWeights {
            conv1_w: vec_of(d * n_mel * 3, &mut *nb),
            conv1_b: vec_of(d, &mut *nb),
            conv2_w: vec_of(d * d * 3, &mut *nb),
            conv2_b: vec_of(d, &mut *nb),
            positional: vec_of(cfg.max_source_positions * d, &mut *nb),
            layers: (0..cfg.encoder_layers).map(|_| WhisperEncoderLayerWeights {
                self_attn_ln_g: Arc::from(vec![1.0_f32; d]),
                self_attn_ln_b: vec_of(d, &mut *nb),
                q_w: vec_of(d * d, &mut *nb),
                q_b: vec_of(d, &mut *nb),
                k_w: vec_of(d * d, &mut *nb),
                v_w: vec_of(d * d, &mut *nb),
                v_b: vec_of(d, &mut *nb),
                out_w: vec_of(d * d, &mut *nb),
                out_b: vec_of(d, &mut *nb),
                final_ln_g: Arc::from(vec![1.0_f32; d]),
                final_ln_b: vec_of(d, &mut *nb),
                fc1_w: vec_of(d * cfg.encoder_ffn_dim, &mut *nb),
                fc1_b: vec_of(cfg.encoder_ffn_dim, &mut *nb),
                fc2_w: vec_of(cfg.encoder_ffn_dim * d, &mut *nb),
                fc2_b: vec_of(d, &mut *nb),
            }).collect(),
            final_ln_g: Arc::from(vec![1.0_f32; d]),
            final_ln_b: vec_of(d, &mut *nb),
        };
        let decoder = WhisperDecoderWeights {
            embed_tokens: vec_of(cfg.vocab_size * d, &mut *nb),
            embed_positions: vec_of(cfg.max_target_positions * d, &mut *nb),
            layers: (0..cfg.decoder_layers).map(|_| WhisperDecoderLayerWeights {
                self_ln_g: Arc::from(vec![1.0_f32; d]),
                self_ln_b: vec_of(d, &mut *nb),
                self_q_w: vec_of(d * d, &mut *nb),
                self_q_b: vec_of(d, &mut *nb),
                self_k_w: vec_of(d * d, &mut *nb),
                self_v_w: vec_of(d * d, &mut *nb),
                self_v_b: vec_of(d, &mut *nb),
                self_out_w: vec_of(d * d, &mut *nb),
                self_out_b: vec_of(d, &mut *nb),
                cross_ln_g: Arc::from(vec![1.0_f32; d]),
                cross_ln_b: vec_of(d, &mut *nb),
                cross_q_w: vec_of(d * d, &mut *nb),
                cross_q_b: vec_of(d, &mut *nb),
                cross_k_w: vec_of(d * d, &mut *nb),
                cross_v_w: vec_of(d * d, &mut *nb),
                cross_v_b: vec_of(d, &mut *nb),
                cross_out_w: vec_of(d * d, &mut *nb),
                cross_out_b: vec_of(d, &mut *nb),
                final_ln_g: Arc::from(vec![1.0_f32; d]),
                final_ln_b: vec_of(d, &mut *nb),
                fc1_w: vec_of(d * cfg.decoder_ffn_dim, &mut *nb),
                fc1_b: vec_of(cfg.decoder_ffn_dim, &mut *nb),
                fc2_w: vec_of(cfg.decoder_ffn_dim * d, &mut *nb),
                fc2_b: vec_of(d, &mut *nb),
            }).collect(),
            final_ln_g: Arc::from(vec![1.0_f32; d]),
            final_ln_b: vec_of(d, &mut *nb),
        };
        WhisperWeights { encoder, decoder }
    }

    fn make_quantized_model() -> QuantizedWhisperModel {
        let cfg = tiny_q4_cfg();
        let plain = WhisperModel { config: cfg.clone(), weights: deterministic_weights(&cfg) };
        QuantizedWhisperModel::from_f32_bake(&plain).unwrap()
    }

    #[test]
    fn encoder_forward_shape_finite_with_q4_0() {
        let model = make_quantized_model();
        let cfg = &model.config;
        // mel_time = 32 → T/2 = 16 = max_source_positions.
        let mel: Vec<f32> = (0..cfg.num_mel_bins * 32)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let enc = model.forward_encoder(&mel, 32).unwrap();
        assert_eq!(enc.shape().dims(), &[1, 16, cfg.d_model]);
        let flat = enc.realize_f32();
        assert_eq!(flat.len(), 16 * cfg.d_model);
        for &v in &flat {
            assert!(v.is_finite(), "non-finite encoder output: {v}");
        }
    }

    #[test]
    fn decoder_forward_shape_finite_with_q4_0() {
        let model = make_quantized_model();
        let cfg = &model.config;
        let mel: Vec<f32> = (0..cfg.num_mel_bins * 32)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();
        let enc = model.forward_encoder(&mel, 32).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward_decoder(&tokens, &enc).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let flat = logits.realize_f32();
        assert_eq!(flat.len(), tokens.len() * cfg.vocab_size);
        for &v in &flat {
            assert!(v.is_finite(), "non-finite decoder logit: {v}");
        }
    }

    #[test]
    fn attention_projections_use_q4_0_weights() {
        let model = make_quantized_model();
        // Every attention + FFN projection in every encoder and decoder
        // layer must be a Q4_0 block matrix; conv1/conv2 + LN gains +
        // positional + token-embed remain F32 (verified separately by
        // inspecting `WhisperEncoderWeights::conv1_w` etc as Arc<[f32]>).
        for lw in &model.weights.encoder.layers {
            for w in [&lw.q_w, &lw.k_w, &lw.v_w, &lw.out_w, &lw.fc1_w, &lw.fc2_w] {
                assert!(
                    matches!(w, WeightStorage::Q4_0 { .. }),
                    "encoder projection is not Q4_0: {w:?}",
                );
            }
        }
        for lw in &model.weights.decoder.layers {
            for w in [
                &lw.self_q_w, &lw.self_k_w, &lw.self_v_w, &lw.self_out_w,
                &lw.cross_q_w, &lw.cross_k_w, &lw.cross_v_w, &lw.cross_out_w,
                &lw.fc1_w, &lw.fc2_w,
            ] {
                assert!(
                    matches!(w, WeightStorage::Q4_0 { .. }),
                    "decoder projection is not Q4_0: {w:?}",
                );
            }
        }
    }
}
