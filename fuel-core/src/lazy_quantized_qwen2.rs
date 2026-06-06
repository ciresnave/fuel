//! GGUF-quantized Qwen2 ported to the lazy-graph API.
//!
//! Qwen2 (non-MoE) with Q4_0 block-quantized Linear weights. The
//! forward path is identical to [`crate::lazy_qwen2::Qwen2Model`];
//! only the weight storage of `attn_{q,k,v,o}` / `ffn_{gate,up,down}`
//! / `output` changes from F32/BF16 to `WeightStorage::Q4_0`. RmsNorm
//! gains, the token embedding table, and the Q/K/V attention biases
//! (which Qwen2 always has) stay in F32 — matching the GGUF
//! convention used by llama.cpp for Qwen2 releases.
//!
//! Construction paths:
//! - [`QuantizedQwen2Model::from_f32_bake`] — take f32 source weights
//!   (same `[in, out]` layout as `Qwen2Weights`) and quantize on the
//!   fly. Used by tests and by callers that have unquantized weights
//!   in memory already.
//! - [`QuantizedQwen2Model::load_from_mmapped`] — convenience: load
//!   f32 Qwen2 weights from HF safetensors then quantize.
//! - [`QuantizedQwen2Model::from_gguf`] — load directly from a Qwen2
//!   GGUF file (`general.architecture = "qwen2"`), keeping Q4_0
//!   tensors quantized and dequantizing other GGML dtypes to F32.
//!   Mirrors the SmolLM3 loader convention.
//!
//! # GGUF tensor name convention (llama.cpp `qwen2` arch)
//!
//! - `token_embd.weight`                — token embedding
//! - `output_norm.weight`               — final RmsNorm gain
//! - `output.weight`                    — lm_head (tied to
//!                                        `token_embd.weight` when
//!                                        absent)
//! - `blk.{i}.attn_norm.weight`         — input layernorm gain
//! - `blk.{i}.ffn_norm.weight`          — post-attention layernorm
//! - `blk.{i}.attn_q.weight`            — Q projection
//! - `blk.{i}.attn_q.bias`              — Q projection bias (Qwen2 has)
//! - `blk.{i}.attn_k.weight`            — K projection
//! - `blk.{i}.attn_k.bias`              — K projection bias
//! - `blk.{i}.attn_v.weight`            — V projection
//! - `blk.{i}.attn_v.bias`              — V projection bias
//! - `blk.{i}.attn_output.weight`       — O projection (no bias)
//! - `blk.{i}.ffn_gate.weight`          — SwiGLU gate
//! - `blk.{i}.ffn_up.weight`            — SwiGLU up
//! - `blk.{i}.ffn_down.weight`          — SwiGLU down

use crate::lazy::{LayerWeights, LazyTensor, WeightStorage};
use crate::lazy_qwen2::{Qwen2Config, Qwen2Model, Qwen2Weights};
use crate::Result;
use std::sync::Arc;

/// GGUF-quantized Qwen2 causal language model. Wraps a plain
/// [`Qwen2Model`] whose Linear weights are `WeightStorage::Q4_0`.
/// The Q/K/V attention biases (an always-present Qwen2 feature) are
/// preserved as F32 — only the weight matrices are quantized.
#[derive(Debug, Clone)]
pub struct QuantizedQwen2Model {
    inner: Qwen2Model,
}

impl QuantizedQwen2Model {
    /// Forward over a token-ID sequence. Returns logits with shape
    /// `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward(tokens, start_pos)
    }

    /// Per-token hidden states up to the final RmsNorm, shape
    /// `(1, seq, hidden_size)`. Skips the `lm_head` projection.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward_hidden(tokens, start_pos)
    }

    /// Forward over pre-computed embeddings, skipping the token-
    /// embedding lookup. Embeds must have shape
    /// `(1, seq, hidden_size)`. Uses the standard per-layer
    /// (sliding-window / strict-causal) mask construction.
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_hidden_embeds(embeds, start_pos)
    }

    /// Forward over pre-computed embeddings with a caller-supplied
    /// `(1, 1, seq, seq)` additive mask used for ALL layers (bypasses
    /// Qwen2's per-layer sliding-window gating). Useful for
    /// bidirectional encoder mode.
    pub fn forward_hidden_embeds_with_mask(
        &self,
        embeds: &LazyTensor,
        attention_mask: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner
            .forward_hidden_embeds_with_mask(embeds, attention_mask, start_pos)
    }

    /// Model configuration.
    pub fn config(&self) -> &Qwen2Config { &self.inner.config }

    /// Underlying [`Qwen2Model`] for direct access to the lazy graph
    /// API. The wrapper exists solely to label the quantization origin.
    pub fn inner(&self) -> &Qwen2Model { &self.inner }

    /// Convenience: load f32 Qwen2 weights from HF safetensors and
    /// quantize each Linear weight to Q4_0. Equivalent to
    /// `Self::from_f32_bake(cfg, Qwen2Weights::load_from_mmapped(st, &cfg)?)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: Qwen2Config,
    ) -> Result<Self> {
        let src = Qwen2Weights::load_from_mmapped(st, &cfg)?;
        Self::from_f32_bake(cfg, src)
    }

    /// Construct from in-memory f32 source weights, quantizing each
    /// Linear weight to Q4_0. Norm gains, Q/K/V biases, and the
    /// token-embedding table stay in F32 — matching the GGUF
    /// convention used by llama.cpp for Qwen2 releases.
    ///
    /// `cfg.hidden_size`, `cfg.intermediate_size`, and
    /// `cfg.num_key_value_heads * cfg.head_dim()` must each be a
    /// multiple of the Q4_0 block size (32). Source weights follow
    /// the same `[in_features, out_features]` row-major layout as
    /// [`Qwen2Weights`].
    pub fn from_f32_bake(cfg: Qwen2Config, src: Qwen2Weights) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv = cfg.num_key_value_heads * head_dim;
        check_q4_0_divisible("hidden_size", h)?;
        check_q4_0_divisible("intermediate_size", i)?;
        check_q4_0_divisible("num_attention_heads * head_dim", q_dim)?;
        check_q4_0_divisible("num_key_value_heads * head_dim", kv)?;

        let quantize_linear = |w: &WeightStorage, in_features: usize, out_features: usize| -> Result<WeightStorage> {
            let f32_in_out = match w {
                WeightStorage::F32(a) => a.to_vec(),
                _ => return Err(crate::Error::Msg(
                    "QuantizedQwen2Model::from_f32_bake: source weights must be WeightStorage::F32".into(),
                ).bt()),
            };
            if f32_in_out.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "QuantizedQwen2Model::from_f32_bake: weight has {} elems, expected {}×{}",
                    f32_in_out.len(), in_features, out_features,
                )).bt());
            }
            quantize_in_out_to_q4_0(&f32_in_out, in_features, out_features)
        };

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (idx, layer) in src.layers.into_iter().enumerate() {
            let attn_q   = quantize_linear(&layer.attn_q,   h, q_dim).map_err(|e| layer_err(idx, "attn_q",   e))?;
            let attn_k   = quantize_linear(&layer.attn_k,   h, kv   ).map_err(|e| layer_err(idx, "attn_k",   e))?;
            let attn_v   = quantize_linear(&layer.attn_v,   h, kv   ).map_err(|e| layer_err(idx, "attn_v",   e))?;
            let attn_o   = quantize_linear(&layer.attn_o,   q_dim, h).map_err(|e| layer_err(idx, "attn_o",   e))?;
            let ffn_gate = quantize_linear(&layer.ffn_gate, h, i    ).map_err(|e| layer_err(idx, "ffn_gate", e))?;
            let ffn_up   = quantize_linear(&layer.ffn_up,   h, i    ).map_err(|e| layer_err(idx, "ffn_up",   e))?;
            let ffn_down = quantize_linear(&layer.ffn_down, i, h    ).map_err(|e| layer_err(idx, "ffn_down", e))?;
            layers.push(LayerWeights {
                attn_q, attn_q_bias: layer.attn_q_bias,
                attn_k, attn_k_bias: layer.attn_k_bias,
                attn_v, attn_v_bias: layer.attn_v_bias,
                attn_o,
                ffn_gate, ffn_up, ffn_down,
                attn_norm_gain: layer.attn_norm_gain,
                ffn_norm_gain:  layer.ffn_norm_gain,
            });
        }

        let output = quantize_linear(&src.output, h, cfg.vocab_size)
            .map_err(|e| crate::Error::Msg(format!("output: {e}")).bt())?;

        let inner = Qwen2Model {
            config: cfg,
            weights: Qwen2Weights {
                token_embedding: src.token_embedding,
                layers,
                final_norm_gain: src.final_norm_gain,
                output,
            },
        };
        Ok(Self { inner })
    }

    /// Load a GGUF-quantized Qwen2 checkpoint. Q4_0 weight tensors
    /// stay quantized; norms / biases / embeddings dequantize to F32.
    /// Other GGML dtypes for Linear weights are dequantized to F32 —
    /// matching the SmolLM3 loader policy. Embedding and `lm_head`
    /// share storage when `output.weight` is absent (tied embeddings,
    /// honored when `cfg.tie_word_embeddings == true`).
    ///
    /// Expects llama.cpp's `qwen2` arch tensor naming. Q/K/V biases
    /// (`blk.{i}.attn_{q,k,v}.bias`) are loaded when present and
    /// validated against `hidden_size` / `kv_dim`.
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P, cfg: &Qwen2Config,
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

        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv = cfg.num_key_value_heads * head_dim;

        let token_embedding = load_f32("token_embd.weight")?;
        if token_embedding.len() != cfg.vocab_size * h {
            return Err(crate::Error::Msg(format!(
                "gguf token_embd.weight: {} elems, expected {}×{}",
                token_embedding.len(), cfg.vocab_size, h,
            )).bt());
        }

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{idx}");
            let attn_q   = load_weight(&format!("{prefix}.attn_q.weight"),      q_dim, h)?;
            let attn_k   = load_weight(&format!("{prefix}.attn_k.weight"),      kv,    h)?;
            let attn_v   = load_weight(&format!("{prefix}.attn_v.weight"),      kv,    h)?;
            let attn_o   = load_weight(&format!("{prefix}.attn_output.weight"), h,     q_dim)?;
            let ffn_gate = load_weight(&format!("{prefix}.ffn_gate.weight"),    i,     h)?;
            let ffn_up   = load_weight(&format!("{prefix}.ffn_up.weight"),      i,     h)?;
            let ffn_down = load_weight(&format!("{prefix}.ffn_down.weight"),    h,     i)?;
            let attn_norm_gain: Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let ffn_norm_gain:  Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
            // Qwen2 always has Q/K/V biases. Be defensive: load when
            // present with the correct length, otherwise leave None.
            let bias = |name: &str, len: usize| -> Option<Arc<[f32]>> {
                load_f32(name).ok().and_then(|v| if v.len() == len { Some(Arc::from(v)) } else { None })
            };
            layers.push(LayerWeights {
                attn_q,
                attn_q_bias: bias(&format!("{prefix}.attn_q.bias"), q_dim),
                attn_k,
                attn_k_bias: bias(&format!("{prefix}.attn_k.bias"), kv),
                attn_v,
                attn_v_bias: bias(&format!("{prefix}.attn_v.bias"), kv),
                attn_o,
                ffn_gate, ffn_up, ffn_down,
                attn_norm_gain, ffn_norm_gain,
            });
        }

        let final_norm_gain: Arc<[f32]> = Arc::from(load_f32("output_norm.weight")?);

        // Tied embeddings: Qwen2 small variants (e.g. 0.5B / 1.5B)
        // tie `lm_head` to the token embedding; larger variants ship
        // a separate `output.weight`. Honor `output.weight` when
        // present in the GGUF; otherwise synthesize an F32 lm_head
        // from the (already-dequantized) token embedding.
        let output = if content.tensor_infos.contains_key("output.weight") {
            load_weight("output.weight", cfg.vocab_size, h)?
        } else {
            let mut f32_in_out = vec![0.0_f32; h * cfg.vocab_size];
            for v in 0..cfg.vocab_size {
                for j in 0..h {
                    f32_in_out[j * cfg.vocab_size + v] = token_embedding[v * h + j];
                }
            }
            WeightStorage::F32(Arc::from(f32_in_out))
        };

        let inner = Qwen2Model {
            config: cfg.clone(),
            weights: Qwen2Weights {
                token_embedding: Arc::from(token_embedding),
                layers, final_norm_gain, output,
            },
        };
        Ok(Self { inner })
    }
}

fn layer_err(idx: usize, name: &str, e: crate::Error) -> crate::Error {
    crate::Error::Msg(format!("layer {idx} {name}: {e}")).bt()
}

fn check_q4_0_divisible(name: &str, n: usize) -> Result<()> {
    const QK4_0: usize = 32;
    if !n.is_multiple_of(QK4_0) {
        return Err(crate::Error::Msg(format!(
            "QuantizedQwen2Model::from_f32_bake: {name} ({n}) must be divisible by Q4_0 block size ({QK4_0})"
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
/// Mirrors the small SmolLM3 helper but lives here so this module
/// stays independent of the SmolLM3 internals.
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
            "gguf {name}: dequant of {other:?} is not supported by lazy_quantized_qwen2",
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

    fn test_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: 4,
            max_window_layers: 1,
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }

    fn tiny_weights(cfg: &Qwen2Config) -> Qwen2Weights {
        let mut s: u32 = 99999;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv = cfg.num_key_value_heads * head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * q_dim)),
            attn_q_bias: Some(vec_of(q_dim)),
            attn_k: WeightStorage::F32(vec_of(h * kv)),
            attn_k_bias: Some(vec_of(kv)),
            attn_v: WeightStorage::F32(vec_of(h * kv)),
            attn_v_bias: Some(vec_of(kv)),
            attn_o: WeightStorage::F32(vec_of(q_dim * h)),
            ffn_gate: WeightStorage::F32(vec_of(h * i)),
            ffn_up:   WeightStorage::F32(vec_of(h * i)),
            ffn_down: WeightStorage::F32(vec_of(i * h)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size));
        Qwen2Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_finite_with_q4_0_weights() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedQwen2Model::from_f32_bake(cfg.clone(), src).unwrap();
        // Sanity: all Linear projections in layer 0 are now Q4_0.
        let l0 = &model.inner().weights.layers[0];
        assert!(matches!(l0.attn_q, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_k, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_v, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_o, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_gate, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_up,   WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_down, WeightStorage::Q4_0 { .. }));
        assert!(matches!(model.inner().weights.output, WeightStorage::Q4_0 { .. }));
        // Q/K/V biases must be preserved (Qwen2 invariant).
        assert!(l0.attn_q_bias.is_some());
        assert!(l0.attn_k_bias.is_some());
        assert!(l0.attn_v_bias.is_some());

        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_hidden_shape_finite() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedQwen2Model::from_f32_bake(cfg.clone(), src).unwrap();
        let hidden = model.forward_hidden(&[1, 2, 3, 4], 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, 4, cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn rejects_non_q4_0_divisible_dims() {
        // hidden_size = 30 isn't divisible by 32 → must fail validation.
        let cfg = Qwen2Config {
            vocab_size: 32,
            hidden_size: 30,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 5,
            num_key_value_heads: 1,
            max_position_embeddings: 64,
            sliding_window: 4,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        // Build a placeholder Qwen2Weights — values don't matter; the
        // divisibility check happens before any quantization.
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv = cfg.num_key_value_heads * head_dim;
        let zeros = |n: usize| -> Arc<[f32]> { Arc::from(vec![0.0_f32; n]) };
        let layer = LayerWeights {
            attn_q: WeightStorage::F32(zeros(h * q_dim)),
            attn_q_bias: Some(zeros(q_dim)),
            attn_k: WeightStorage::F32(zeros(h * kv)),
            attn_k_bias: Some(zeros(kv)),
            attn_v: WeightStorage::F32(zeros(h * kv)),
            attn_v_bias: Some(zeros(kv)),
            attn_o: WeightStorage::F32(zeros(q_dim * h)),
            ffn_gate: WeightStorage::F32(zeros(h * i)),
            ffn_up:   WeightStorage::F32(zeros(h * i)),
            ffn_down: WeightStorage::F32(zeros(i * h)),
            attn_norm_gain: zeros(h),
            ffn_norm_gain:  zeros(h),
        };
        let src = Qwen2Weights {
            token_embedding: zeros(cfg.vocab_size * h),
            layers: vec![layer],
            final_norm_gain: zeros(h),
            output: WeightStorage::F32(zeros(h * cfg.vocab_size)),
        };
        let err = QuantizedQwen2Model::from_f32_bake(cfg, src).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("divisible by Q4_0"), "unexpected error: {msg}");
    }
}
