//! GGUF-quantized T5 ported to the lazy-graph API.
//!
//! T5 (and Flan-T5 / UL2 / MADLAD-400) with Q4_0 block-quantized
//! Linear weights. The forward path is identical to
//! [`crate::lazy_t5::T5Model`]; only the storage of the per-layer
//! Linear matrices (`q`, `k`, `v`, `o`, FFN `wi`/`wi_0`/`wi_1`/`wo`,
//! and the optional `lm_head`) changes from F32 to
//! `WeightStorage::Q4_0`. T5 has no biases anywhere, so the only
//! parameters that stay in F32 are:
//!
//!   * `shared_embedding` (the shared src/tgt embedding table),
//!   * `encoder_rel_bias` / `decoder_rel_bias` (the
//!     `[num_buckets, n_heads]` learned relative-attention bias
//!     tables — they're embeddings, not Linears, and they're tiny),
//!   * the per-layer RmsNorm `*_norm_gain` vectors,
//!   * `encoder_final_norm_gain` / `decoder_final_norm_gain`.
//!
//! T5 is an encoder-decoder model. Each encoder layer has one
//! self-attention `(q, k, v, o)` and one FFN. Each decoder layer
//! has self-attention + cross-attention (each with its own
//! `q, k, v, o`) + FFN. The FFN may be plain Dense (`wi`/`wo`) or
//! Gated (`wi_0`/`wi_1`/`wo`) — both shapes are quantized
//! identically: each weight matrix becomes Q4_0 on its own.
//!
//! Construction paths:
//! - [`QuantizedT5Model::from_f32_bake`] — take f32 source weights
//!   (same `[in, out]` layout as `T5Weights`) and quantize on the
//!   fly. Used by tests and by callers that have unquantized
//!   weights in memory already.
//! - [`QuantizedT5Model::load_from_mmapped`] — convenience: load
//!   HF safetensors via [`T5Weights::load_from_mmapped`] and then
//!   quantize via `from_f32_bake`.
//! - [`QuantizedT5Model::from_gguf`] — load directly from a T5
//!   GGUF file using the llama.cpp T5 tensor namespacing
//!   (`enc.blk.{i}.*` and `dec.blk.{i}.*`), keeping Q4_0 tensors
//!   quantized and dequantizing other GGML dtypes to F32.

use crate::lazy_t5::{
    T5AttentionWeights, T5Config, T5DecoderLayerWeights, T5EncoderLayerWeights, T5FfnWeights,
    T5Model, T5Weights,
};
use crate::lazy::WeightStorage;
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

/// GGUF-quantized T5 (encoder-decoder) model. Wraps a plain
/// [`T5Model`] whose Linear weights have been replaced with
/// `WeightStorage::Q4_0` blocks. Norm gains, the shared embedding
/// table, and the relative-attention bias tables stay in F32.
///
/// The wrapper exists to label the quantization origin and to
/// give the loader its own entry point; once constructed every
/// method delegates to the inner [`T5Model`].
#[derive(Debug, Clone)]
pub struct QuantizedT5Model {
    inner: T5Model,
}

impl QuantizedT5Model {
    /// Full seq2seq forward: encode `src_tokens`, decode
    /// `tgt_tokens` against the encoder output, return logits
    /// `(1, tgt_len, vocab_size)`.
    pub fn forward(&self, src_tokens: &[u32], tgt_tokens: &[u32]) -> Result<crate::lazy::LazyTensor> {
        self.inner.forward(src_tokens, tgt_tokens)
    }

    /// Encoder-only forward over token IDs. Returns post-final-norm
    /// hidden states `(1, src_len, d_model)`.
    pub fn forward_encoder(&self, src_tokens: &[u32]) -> Result<crate::lazy::LazyTensor> {
        self.inner.forward_encoder(src_tokens)
    }

    /// Encoder-only forward over pre-computed embeddings. Skips the
    /// token-embedding lookup. Returns `(1, src_len, d_model)`.
    pub fn forward_encoder_embeds(
        &self, src_embeds: &crate::lazy::LazyTensor,
    ) -> Result<crate::lazy::LazyTensor> {
        self.inner.forward_encoder_embeds(src_embeds)
    }

    /// Decoder-only forward against a cached encoder output.
    /// Returns logits `(1, tgt_len, vocab_size)`.
    pub fn forward_decoder(
        &self, tgt_tokens: &[u32], encoder_out: &crate::lazy::LazyTensor,
    ) -> Result<crate::lazy::LazyTensor> {
        self.inner.forward_decoder(tgt_tokens, encoder_out)
    }

    /// Decoder-only forward over pre-computed target embeddings
    /// and a cached encoder output. Returns logits
    /// `(1, tgt_len, vocab_size)`.
    pub fn forward_decoder_embeds(
        &self,
        tgt_embeds: &crate::lazy::LazyTensor,
        encoder_out: &crate::lazy::LazyTensor,
    ) -> Result<crate::lazy::LazyTensor> {
        self.inner.forward_decoder_embeds(tgt_embeds, encoder_out)
    }

    /// Hidden-state variant of [`Self::forward_decoder_embeds`].
    /// Returns `(1, tgt_len, d_model)` post-RmsNorm states without
    /// the LM head or tied-embedding scaling.
    pub fn forward_decoder_hidden_embeds(
        &self,
        tgt_embeds: &crate::lazy::LazyTensor,
        encoder_out: &crate::lazy::LazyTensor,
    ) -> Result<crate::lazy::LazyTensor> {
        self.inner.forward_decoder_hidden_embeds(tgt_embeds, encoder_out)
    }

    /// Build per-token embeddings without running encoder or
    /// decoder. T5 has tied src/tgt embeddings; one table serves
    /// both sides. Returns `(1, seq, d_model)`.
    pub fn embed_tokens_anchored(
        &self, anchor: &crate::lazy::LazyTensor, tokens: &[u32],
    ) -> Result<crate::lazy::LazyTensor> {
        self.inner.embed_tokens_anchored(anchor, tokens)
    }

    /// Model configuration.
    pub fn config(&self) -> &T5Config { &self.inner.config }

    /// Underlying [`T5Model`] for direct access to the lazy graph
    /// API. The wrapper exists solely to label the quantization
    /// origin.
    pub fn inner(&self) -> &T5Model { &self.inner }

    /// Convenience: load f32 T5 weights from HF safetensors and
    /// quantize each Linear weight to Q4_0. Equivalent to
    /// `Self::from_f32_bake(cfg, T5Weights::load_from_mmapped(st, &cfg)?)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: T5Config,
    ) -> Result<Self> {
        let src = T5Weights::load_from_mmapped(st, &cfg)?;
        Self::from_f32_bake(cfg, src)
    }

    /// Construct from in-memory f32 source weights, quantizing
    /// each Linear weight to Q4_0. Norm gains, the shared
    /// embedding table, and the relative-attention bias tables
    /// stay in F32 — matching the GGUF convention used by
    /// llama.cpp for T5 releases.
    ///
    /// `cfg.d_model`, `cfg.inner_dim()` (= `num_heads * d_kv`),
    /// and `cfg.d_ff` must each be a multiple of the Q4_0 block
    /// size (32). Source weights follow the same
    /// `[in_features, out_features]` row-major layout as
    /// [`T5Weights`].
    pub fn from_f32_bake(cfg: T5Config, src: T5Weights) -> Result<Self> {
        let d = cfg.d_model;
        let inner = cfg.inner_dim();
        let d_ff = cfg.d_ff;
        check_q4_0_divisible("d_model", d)?;
        check_q4_0_divisible("inner_dim (= num_heads * d_kv)", inner)?;
        check_q4_0_divisible("d_ff", d_ff)?;

        let quantize_linear = |w: &WeightStorage, in_features: usize, out_features: usize| -> Result<WeightStorage> {
            let f32_in_out = match w {
                WeightStorage::F32(a) => a.to_vec(),
                _ => return Err(crate::Error::Msg(
                    "QuantizedT5Model::from_f32_bake: source weights must be WeightStorage::F32".into(),
                ).bt()),
            };
            if f32_in_out.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "QuantizedT5Model::from_f32_bake: weight has {} elems, expected {}×{}",
                    f32_in_out.len(), in_features, out_features,
                )).bt());
            }
            quantize_in_out_to_q4_0(&f32_in_out, in_features, out_features)
        };

        let quantize_attn = |a: &T5AttentionWeights, name: &str| -> Result<T5AttentionWeights> {
            let q = quantize_linear(&a.q, d, inner)
                .map_err(|e| crate::Error::Msg(format!("{name}.q: {e}")).bt())?;
            let k = quantize_linear(&a.k, d, inner)
                .map_err(|e| crate::Error::Msg(format!("{name}.k: {e}")).bt())?;
            let v = quantize_linear(&a.v, d, inner)
                .map_err(|e| crate::Error::Msg(format!("{name}.v: {e}")).bt())?;
            let o = quantize_linear(&a.o, inner, d)
                .map_err(|e| crate::Error::Msg(format!("{name}.o: {e}")).bt())?;
            Ok(T5AttentionWeights { q, k, v, o })
        };

        let quantize_ffn = |f: &T5FfnWeights, name: &str| -> Result<T5FfnWeights> {
            match f {
                T5FfnWeights::Dense { wi, wo } => {
                    let wi = quantize_linear(wi, d, d_ff)
                        .map_err(|e| crate::Error::Msg(format!("{name}.wi: {e}")).bt())?;
                    let wo = quantize_linear(wo, d_ff, d)
                        .map_err(|e| crate::Error::Msg(format!("{name}.wo: {e}")).bt())?;
                    Ok(T5FfnWeights::Dense { wi, wo })
                }
                T5FfnWeights::Gated { wi_0, wi_1, wo } => {
                    let wi_0 = quantize_linear(wi_0, d, d_ff)
                        .map_err(|e| crate::Error::Msg(format!("{name}.wi_0: {e}")).bt())?;
                    let wi_1 = quantize_linear(wi_1, d, d_ff)
                        .map_err(|e| crate::Error::Msg(format!("{name}.wi_1: {e}")).bt())?;
                    let wo = quantize_linear(wo, d_ff, d)
                        .map_err(|e| crate::Error::Msg(format!("{name}.wo: {e}")).bt())?;
                    Ok(T5FfnWeights::Gated { wi_0, wi_1, wo })
                }
            }
        };

        let mut encoder_layers: Vec<T5EncoderLayerWeights> =
            Vec::with_capacity(src.encoder_layers.len());
        for (idx, layer) in src.encoder_layers.into_iter().enumerate() {
            let self_attn = quantize_attn(&layer.self_attn, "self_attn")
                .map_err(|e| layer_err(idx, "encoder.self_attn", e))?;
            let ffn = quantize_ffn(&layer.ffn, "ffn")
                .map_err(|e| layer_err(idx, "encoder.ffn", e))?;
            encoder_layers.push(T5EncoderLayerWeights {
                self_attn_norm_gain: layer.self_attn_norm_gain,
                self_attn,
                ffn_norm_gain: layer.ffn_norm_gain,
                ffn,
            });
        }

        let mut decoder_layers: Vec<T5DecoderLayerWeights> =
            Vec::with_capacity(src.decoder_layers.len());
        for (idx, layer) in src.decoder_layers.into_iter().enumerate() {
            let self_attn = quantize_attn(&layer.self_attn, "self_attn")
                .map_err(|e| layer_err(idx, "decoder.self_attn", e))?;
            let cross_attn = quantize_attn(&layer.cross_attn, "cross_attn")
                .map_err(|e| layer_err(idx, "decoder.cross_attn", e))?;
            let ffn = quantize_ffn(&layer.ffn, "ffn")
                .map_err(|e| layer_err(idx, "decoder.ffn", e))?;
            decoder_layers.push(T5DecoderLayerWeights {
                self_attn_norm_gain: layer.self_attn_norm_gain,
                self_attn,
                cross_attn_norm_gain: layer.cross_attn_norm_gain,
                cross_attn,
                ffn_norm_gain: layer.ffn_norm_gain,
                ffn,
            });
        }

        let lm_head = match src.lm_head {
            Some(w) => Some(
                quantize_linear(&w, d, cfg.vocab_size)
                    .map_err(|e| crate::Error::Msg(format!("lm_head: {e}")).bt())?,
            ),
            None => None,
        };

        let inner = T5Model {
            config: cfg,
            weights: T5Weights {
                shared_embedding: src.shared_embedding,
                encoder_rel_bias: src.encoder_rel_bias,
                decoder_rel_bias: src.decoder_rel_bias,
                encoder_layers,
                decoder_layers,
                encoder_final_norm_gain: src.encoder_final_norm_gain,
                decoder_final_norm_gain: src.decoder_final_norm_gain,
                lm_head,
            },
        };
        Ok(Self { inner })
    }

    /// Load a GGUF-quantized T5 checkpoint. Q4_0 weight tensors
    /// stay quantized; norms / embeddings / relative-attention bias
    /// tables dequantize to F32. Other GGML dtypes for Linear
    /// weights are dequantized to F32 — matching the SmolLM3
    /// loader policy.
    ///
    /// Tensor namespacing follows the llama.cpp T5 convention:
    /// encoder weights live under `enc.blk.{i}.*` and decoder
    /// weights under `dec.blk.{i}.*`. Per-layer tensors:
    ///
    ///   * `enc.blk.{i}.attn_q.weight`, `attn_k.weight`,
    ///     `attn_v.weight`, `attn_o.weight`
    ///   * `enc.blk.{i}.attn_norm.weight`, `ffn_norm.weight`
    ///   * Dense FFN: `enc.blk.{i}.ffn_up.weight`, `ffn_down.weight`
    ///   * Gated FFN: `enc.blk.{i}.ffn_gate.weight`,
    ///     `ffn_up.weight`, `ffn_down.weight`
    ///   * Decoder additionally has `cross_attn_q/k/v/o.weight`
    ///     and `cross_attn_norm.weight`.
    ///
    /// Stack-level tensors:
    ///
    ///   * `token_embd.weight` — shared embedding table
    ///     (`[vocab_size, d_model]`).
    ///   * `enc.blk.0.attn_rel_b.weight` /
    ///     `dec.blk.0.attn_rel_b.weight` —
    ///     `[num_buckets, n_heads]` relative-attention bias.
    ///   * `enc.output_norm.weight` / `dec.output_norm.weight` —
    ///     final RmsNorm gains.
    ///   * `output.weight` — LM head; tied to `token_embd.weight`
    ///     when absent.
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P, cfg: &T5Config,
    ) -> Result<Self> {
        use crate::quantized::gguf_mmap::MmapedContent;
        let mc = MmapedContent::from_path(path)?;
        let content = mc.content();
        let mmap_arc = mc.mmap();
        let mmap_bytes: &[u8] = &mmap_arc[..];
        let data_off = content.tensor_data_offset as usize;

        let get_tensor_bytes = |name: &str| -> Result<(&[u8], crate::quantized::GgmlDType, Vec<usize>)> {
            let info = content.tensor_infos.get(name).ok_or_else(|| {
                crate::Error::Msg(format!("gguf: missing tensor {name:?}"))
            })?;
            let elems = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let bytes_len = elems / block_size * info.ggml_dtype.type_size();
            let start = data_off + info.offset as usize;
            Ok((&mmap_bytes[start..start + bytes_len], info.ggml_dtype, info.shape.dims().to_vec()))
        };

        let load_f32 = |name: &str| -> Result<Vec<f32>> {
            let (bytes, dt, _) = get_tensor_bytes(name)?;
            dequant_bytes_to_f32(bytes, dt, name)
        };

        let load_weight = |name: &str, out_features: usize, in_features: usize| -> Result<WeightStorage> {
            let (bytes, dt, dims) = get_tensor_bytes(name)?;
            let expected = out_features * in_features;
            let actual: usize = dims.iter().product();
            if actual != expected {
                return Err(crate::Error::Msg(format!(
                    "gguf: tensor {name:?} has {actual} elements, expected {expected} for [{out_features}, {in_features}]",
                )).bt());
            }
            match dt {
                crate::quantized::GgmlDType::Q4_0 => Ok(WeightStorage::Q4_0 {
                    words: bytes_to_u32_arc(bytes),
                    bytes_len: bytes.len(),
                    in_features,
                    out_features,
                }),
                _ => {
                    // Dequantize to F32 and transpose [out, in] → [in, out].
                    let f32_out_in = dequant_bytes_to_f32(bytes, dt, name)?;
                    let mut f32_in_out = vec![0.0_f32; expected];
                    for o in 0..out_features {
                        for j in 0..in_features {
                            f32_in_out[j * out_features + o] = f32_out_in[o * in_features + j];
                        }
                    }
                    Ok(WeightStorage::F32(Arc::from(f32_in_out)))
                }
            }
        };

        let d = cfg.d_model;
        let inner = cfg.inner_dim();
        let d_ff = cfg.d_ff;
        let n_enc = cfg.num_layers;
        let n_dec = cfg.num_decoder_layers_resolved();
        let gated = cfg.gated_ffn;

        // Shared embedding table: [vocab_size, d_model].
        let shared_embedding = load_f32("token_embd.weight")?;
        if shared_embedding.len() != cfg.vocab_size * d {
            return Err(crate::Error::Msg(format!(
                "gguf token_embd.weight: {} elems, expected {}×{}",
                shared_embedding.len(), cfg.vocab_size, d,
            )).bt());
        }

        // Relative-attention bias tables: [num_buckets, n_heads].
        let encoder_rel_bias = load_f32("enc.blk.0.attn_rel_b.weight")?;
        let decoder_rel_bias = load_f32("dec.blk.0.attn_rel_b.weight")?;
        let expected_rel = cfg.relative_attention_num_buckets * cfg.num_heads;
        if encoder_rel_bias.len() != expected_rel {
            return Err(crate::Error::Msg(format!(
                "gguf enc.blk.0.attn_rel_b.weight: {} elems, expected {}×{}",
                encoder_rel_bias.len(),
                cfg.relative_attention_num_buckets, cfg.num_heads,
            )).bt());
        }
        if decoder_rel_bias.len() != expected_rel {
            return Err(crate::Error::Msg(format!(
                "gguf dec.blk.0.attn_rel_b.weight: {} elems, expected {}×{}",
                decoder_rel_bias.len(),
                cfg.relative_attention_num_buckets, cfg.num_heads,
            )).bt());
        }

        let load_attn = |prefix: &str, q_name: &str, k_name: &str, v_name: &str, o_name: &str| -> Result<T5AttentionWeights> {
            let q = load_weight(&format!("{prefix}.{q_name}.weight"), inner, d)?;
            let k = load_weight(&format!("{prefix}.{k_name}.weight"), inner, d)?;
            let v = load_weight(&format!("{prefix}.{v_name}.weight"), inner, d)?;
            let o = load_weight(&format!("{prefix}.{o_name}.weight"), d, inner)?;
            Ok(T5AttentionWeights { q, k, v, o })
        };

        let load_ffn = |prefix: &str| -> Result<T5FfnWeights> {
            if gated {
                let wi_0 = load_weight(&format!("{prefix}.ffn_gate.weight"), d_ff, d)?;
                let wi_1 = load_weight(&format!("{prefix}.ffn_up.weight"),   d_ff, d)?;
                let wo   = load_weight(&format!("{prefix}.ffn_down.weight"), d,    d_ff)?;
                Ok(T5FfnWeights::Gated { wi_0, wi_1, wo })
            } else {
                let wi = load_weight(&format!("{prefix}.ffn_up.weight"),   d_ff, d)?;
                let wo = load_weight(&format!("{prefix}.ffn_down.weight"), d,    d_ff)?;
                Ok(T5FfnWeights::Dense { wi, wo })
            }
        };

        let mut encoder_layers: Vec<T5EncoderLayerWeights> = Vec::with_capacity(n_enc);
        for i in 0..n_enc {
            let prefix = format!("enc.blk.{i}");
            let self_attn_norm_gain: Arc<[f32]> =
                Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let self_attn = load_attn(&prefix, "attn_q", "attn_k", "attn_v", "attn_o")?;
            let ffn_norm_gain: Arc<[f32]> =
                Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
            let ffn = load_ffn(&prefix)?;
            encoder_layers.push(T5EncoderLayerWeights {
                self_attn_norm_gain, self_attn, ffn_norm_gain, ffn,
            });
        }

        let mut decoder_layers: Vec<T5DecoderLayerWeights> = Vec::with_capacity(n_dec);
        for i in 0..n_dec {
            let prefix = format!("dec.blk.{i}");
            let self_attn_norm_gain: Arc<[f32]> =
                Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let self_attn = load_attn(&prefix, "attn_q", "attn_k", "attn_v", "attn_o")?;
            let cross_attn_norm_gain: Arc<[f32]> =
                Arc::from(load_f32(&format!("{prefix}.cross_attn_norm.weight"))?);
            let cross_attn = load_attn(
                &prefix,
                "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o",
            )?;
            let ffn_norm_gain: Arc<[f32]> =
                Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
            let ffn = load_ffn(&prefix)?;
            decoder_layers.push(T5DecoderLayerWeights {
                self_attn_norm_gain, self_attn,
                cross_attn_norm_gain, cross_attn,
                ffn_norm_gain, ffn,
            });
        }

        let encoder_final_norm_gain: Arc<[f32]> =
            Arc::from(load_f32("enc.output_norm.weight")?);
        let decoder_final_norm_gain: Arc<[f32]> =
            Arc::from(load_f32("dec.output_norm.weight")?);

        // Tied embeddings: T5 by default ties the LM head to the
        // shared embedding table. If `output.weight` is present we
        // honor it; otherwise we leave `lm_head = None` and let
        // `T5Model::forward` use the tied path (which also applies
        // the d_model^-0.5 scaling).
        let lm_head = if content.tensor_infos.contains_key("output.weight") {
            Some(load_weight("output.weight", cfg.vocab_size, d)?)
        } else if cfg.tie_word_embeddings {
            None
        } else {
            return Err(crate::Error::Msg(
                "gguf: output.weight absent and config has tie_word_embeddings=false".into(),
            ).bt());
        };

        let inner_model = T5Model {
            config: cfg.clone(),
            weights: T5Weights {
                shared_embedding: Arc::from(shared_embedding),
                encoder_rel_bias: Arc::from(encoder_rel_bias),
                decoder_rel_bias: Arc::from(decoder_rel_bias),
                encoder_layers, decoder_layers,
                encoder_final_norm_gain, decoder_final_norm_gain,
                lm_head,
            },
        };
        Ok(Self { inner: inner_model })
    }
}

fn layer_err(idx: usize, name: &str, e: crate::Error) -> crate::Error {
    crate::Error::Msg(format!("layer {idx} {name}: {e}")).bt()
}

fn check_q4_0_divisible(name: &str, n: usize) -> Result<()> {
    const QK4_0: usize = 32;
    if !n.is_multiple_of(QK4_0) {
        return Err(crate::Error::Msg(format!(
            "QuantizedT5Model::from_f32_bake: {name} ({n}) must be divisible by Q4_0 block size ({QK4_0})"
        )).bt());
    }
    Ok(())
}

/// Quantize a `[in_features, out_features]` row-major F32 weight
/// matrix into a `WeightStorage::Q4_0` keeping GGUF's native
/// `[out, in]` block layout. The implementation does the
/// `[in, out] → [out, in]` transpose first, then runs the per-row
/// Q4_0 quantization.
fn quantize_in_out_to_q4_0(
    f32_in_out: &[f32], in_features: usize, out_features: usize,
) -> Result<WeightStorage> {
    use fuel_quantized::{BlockQ4_0, GgmlType};
    const QK4_0: usize = 32;
    if !in_features.is_multiple_of(QK4_0) {
        return Err(crate::Error::Msg(format!(
            "Q4_0 quantize: in_features ({in_features}) must be divisible by {QK4_0}"
        )).bt());
    }

    // Transpose [in, out] → [out, in] so each row is contiguous in K.
    let mut f32_out_in = vec![0.0_f32; out_features * in_features];
    for o in 0..out_features {
        for j in 0..in_features {
            f32_out_in[o * in_features + j] = f32_in_out[j * out_features + o];
        }
    }

    let n_blocks = out_features * in_features / QK4_0;
    let mut blocks: Vec<BlockQ4_0> = vec![BlockQ4_0::zeros(); n_blocks];
    BlockQ4_0::from_float(&f32_out_in, &mut blocks);

    // BlockQ4_0 is repr(C) and exactly 18 bytes; reinterpret as bytes.
    let bytes_len = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let byte_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, bytes_len)
    };
    // Q4_0 storage holds u32 words; pad bytes to a multiple of 4 by
    // copying into a Vec<u8> first.
    let padded_len = bytes_len.div_ceil(4) * 4;
    let mut padded = vec![0_u8; padded_len];
    padded[..bytes_len].copy_from_slice(byte_slice);
    let words: Vec<u32> = padded.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(WeightStorage::Q4_0 {
        words: Arc::from(words),
        bytes_len,
        in_features,
        out_features,
    })
}

fn bytes_to_u32_arc(bytes: &[u8]) -> Arc<[u32]> {
    let padded_len = bytes.len().div_ceil(4) * 4;
    let mut padded = vec![0_u8; padded_len];
    padded[..bytes.len()].copy_from_slice(bytes);
    let words: Vec<u32> = padded.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Arc::from(words)
}

/// Dequantize a raw GGUF byte slice of the given GGML dtype to F32.
/// Self-contained so this module stays independent of other lazy
/// quantized modules.
fn dequant_bytes_to_f32(
    bytes: &[u8], dt: crate::quantized::GgmlDType, name: &str,
) -> Result<Vec<f32>> {
    use crate::quantized::GgmlDType;
    use half::{bf16, f16};
    match dt {
        GgmlDType::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::Msg(format!(
                    "gguf {name}: F32 byte count {} not multiple of 4", bytes.len(),
                )).bt());
            }
            Ok(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        GgmlDType::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(crate::Error::Msg(format!(
                    "gguf {name}: F16 byte count {} not multiple of 2", bytes.len(),
                )).bt());
            }
            Ok(bytes.chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect())
        }
        GgmlDType::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(crate::Error::Msg(format!(
                    "gguf {name}: BF16 byte count {} not multiple of 2", bytes.len(),
                )).bt());
            }
            Ok(bytes.chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect())
        }
        GgmlDType::Q4_0 => Ok(cpu_dequant_q4_0_bytes(bytes)),
        other => Err(crate::Error::Msg(format!(
            "gguf {name}: dequant of {other:?} is not supported by lazy_quantized_t5",
        )).bt()),
    }
}

fn cpu_dequant_q4_0_bytes(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    let bpb = 18usize;
    let epb = 32usize;
    let n_blocks = bytes.len() / bpb;
    let mut out = vec![0.0_f32; n_blocks * epb];
    for b in 0..n_blocks {
        let off = b * bpb;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let base = b * epb;
        for kk in 0..16 {
            let packed = bytes[off + 2 + kk];
            let lo = (packed & 0x0F) as i32 - 8;
            let hi = ((packed >> 4) & 0x0F) as i32 - 8;
            out[base + kk]      = lo as f32 * d;
            out[base + 16 + kk] = hi as f32 * d;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LazyTensor;
    use crate::lazy_t5::T5Activation;

    fn test_cfg() -> T5Config {
        // All quantized dims must be multiples of 32:
        //   d_model = 32, inner_dim = num_heads * d_kv = 4 * 8 = 32,
        //   d_ff = 64.
        T5Config {
            vocab_size: 32,
            d_model: 32, d_kv: 8, d_ff: 64,
            num_layers: 2, num_decoder_layers: None,
            num_heads: 4,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 32,
            layer_norm_epsilon: 1e-6,
            activation: T5Activation::Relu,
            gated_ffn: false,
            tie_word_embeddings: true,
        }
    }

    fn tiny_weights(cfg: &T5Config) -> T5Weights {
        let mut s: u32 = 75757;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let d = cfg.d_model;
        let inner = cfg.inner_dim();
        let d_ff = cfg.d_ff;

        let mk_attn = |vec_of: &mut dyn FnMut(usize) -> Arc<[f32]>| T5AttentionWeights {
            q: WeightStorage::F32(vec_of(d * inner)),
            k: WeightStorage::F32(vec_of(d * inner)),
            v: WeightStorage::F32(vec_of(d * inner)),
            o: WeightStorage::F32(vec_of(inner * d)),
        };
        let mk_ffn = |vec_of: &mut dyn FnMut(usize) -> Arc<[f32]>| -> T5FfnWeights {
            if cfg.gated_ffn {
                T5FfnWeights::Gated {
                    wi_0: WeightStorage::F32(vec_of(d * d_ff)),
                    wi_1: WeightStorage::F32(vec_of(d * d_ff)),
                    wo:   WeightStorage::F32(vec_of(d_ff * d)),
                }
            } else {
                T5FfnWeights::Dense {
                    wi: WeightStorage::F32(vec_of(d * d_ff)),
                    wo: WeightStorage::F32(vec_of(d_ff * d)),
                }
            }
        };

        let shared_embedding = vec_of(cfg.vocab_size * d);
        let encoder_rel_bias = vec_of(cfg.relative_attention_num_buckets * cfg.num_heads);
        let decoder_rel_bias = vec_of(cfg.relative_attention_num_buckets * cfg.num_heads);

        let encoder_layers: Vec<T5EncoderLayerWeights> = (0..cfg.num_layers).map(|_| {
            T5EncoderLayerWeights {
                self_attn_norm_gain: Arc::from(vec![1.0_f32; d]),
                self_attn: mk_attn(&mut vec_of),
                ffn_norm_gain: Arc::from(vec![1.0_f32; d]),
                ffn: mk_ffn(&mut vec_of),
            }
        }).collect();
        let decoder_layers: Vec<T5DecoderLayerWeights> = (0..cfg.num_decoder_layers_resolved()).map(|_| {
            T5DecoderLayerWeights {
                self_attn_norm_gain: Arc::from(vec![1.0_f32; d]),
                self_attn: mk_attn(&mut vec_of),
                cross_attn_norm_gain: Arc::from(vec![1.0_f32; d]),
                cross_attn: mk_attn(&mut vec_of),
                ffn_norm_gain: Arc::from(vec![1.0_f32; d]),
                ffn: mk_ffn(&mut vec_of),
            }
        }).collect();
        let encoder_final_norm_gain = Arc::from(vec![1.0_f32; d]);
        let decoder_final_norm_gain = Arc::from(vec![1.0_f32; d]);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(d * cfg.vocab_size)))
        };
        T5Weights {
            shared_embedding,
            encoder_rel_bias, decoder_rel_bias,
            encoder_layers, decoder_layers,
            encoder_final_norm_gain, decoder_final_norm_gain,
            lm_head,
        }
    }

    #[test]
    fn forward_shape_finite_with_q4_0_weights() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedT5Model::from_f32_bake(cfg.clone(), src).unwrap();
        // Sanity: all Linear projections in encoder/decoder layer 0
        // are now Q4_0.
        let e0 = &model.inner().weights.encoder_layers[0];
        assert!(matches!(e0.self_attn.q, WeightStorage::Q4_0 { .. }));
        assert!(matches!(e0.self_attn.k, WeightStorage::Q4_0 { .. }));
        assert!(matches!(e0.self_attn.v, WeightStorage::Q4_0 { .. }));
        assert!(matches!(e0.self_attn.o, WeightStorage::Q4_0 { .. }));
        match &e0.ffn {
            T5FfnWeights::Dense { wi, wo } => {
                assert!(matches!(wi, WeightStorage::Q4_0 { .. }));
                assert!(matches!(wo, WeightStorage::Q4_0 { .. }));
            }
            _ => panic!("expected dense FFN"),
        }
        let d0 = &model.inner().weights.decoder_layers[0];
        assert!(matches!(d0.self_attn.q, WeightStorage::Q4_0 { .. }));
        assert!(matches!(d0.cross_attn.q, WeightStorage::Q4_0 { .. }));

        let src_tokens = [1_u32, 2, 3, 4];
        let tgt_tokens = [5_u32, 6, 7];
        let logits = model.forward(&src_tokens, &tgt_tokens).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt_tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn gated_ffn_quantizes_three_matrices() {
        let cfg = T5Config {
            gated_ffn: true,
            activation: T5Activation::GeluPytorchTanh,
            ..test_cfg()
        };
        let src = tiny_weights(&cfg);
        let model = QuantizedT5Model::from_f32_bake(cfg.clone(), src).unwrap();
        let e0 = &model.inner().weights.encoder_layers[0];
        match &e0.ffn {
            T5FfnWeights::Gated { wi_0, wi_1, wo } => {
                assert!(matches!(wi_0, WeightStorage::Q4_0 { .. }));
                assert!(matches!(wi_1, WeightStorage::Q4_0 { .. }));
                assert!(matches!(wo,   WeightStorage::Q4_0 { .. }));
            }
            _ => panic!("expected gated FFN"),
        }
        let src_tokens = [1_u32, 2, 3];
        let tgt_tokens = [4_u32, 5];
        let logits = model.forward(&src_tokens, &tgt_tokens).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tgt_tokens.len(), cfg.vocab_size]);
    }

    #[test]
    fn separate_lm_head_quantized_when_untied() {
        let cfg = T5Config { tie_word_embeddings: false, ..test_cfg() };
        let src = tiny_weights(&cfg);
        let model = QuantizedT5Model::from_f32_bake(cfg.clone(), src).unwrap();
        let lm_head = model.inner().weights.lm_head.as_ref().expect("lm_head present");
        assert!(matches!(lm_head, WeightStorage::Q4_0 { .. }));
    }

    #[test]
    fn rejects_non_q4_0_divisible_d_model() {
        let cfg = T5Config { d_model: 30, ..test_cfg() };
        let src = tiny_weights(&cfg);
        assert!(QuantizedT5Model::from_f32_bake(cfg, src).is_err());
    }

    #[test]
    fn forward_encoder_embeds_matches_forward_encoder() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedT5Model::from_f32_bake(cfg.clone(), src).unwrap();
        let src_tokens = [1_u32, 2, 3];
        let enc_ref = model.forward_encoder(&src_tokens).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let src_embeds = model.embed_tokens_anchored(&anchor, &src_tokens).unwrap();
        let enc_via_embeds = model.forward_encoder_embeds(&src_embeds).unwrap().realize_f32();
        let max_diff = enc_ref.iter().zip(enc_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-4,
            "Quantized T5 forward_encoder vs forward_encoder_embeds must agree (max diff {max_diff})");
    }
}
