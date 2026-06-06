//! GGUF-quantized Gemma 3 ported to the lazy-graph API.
//!
//! Gemma 3 with Q4_0 block-quantized Linear weights. The forward path is
//! identical to [`crate::lazy_gemma3::Gemma3Model`]; only the weight
//! storage of `attn_{q,k,v,o}` / `ffn_{gate,up,down}` changes from
//! F32/BF16 to `WeightStorage::Q4_0`. All norm gains (input, post-attn,
//! pre-FFN, post-FFN, per-head Q/K, final), biases (when present), and
//! the token-embedding table stay in F32. Gemma 3 ties `lm_head` to
//! `token_embedding`, so there is no separate output projection to
//! quantize.
//!
//! Gemma-3-specific structural notes the loader honors:
//! - Independent attention/embedding dims: `q_dim = num_attention_heads
//!   * head_dim` is **not** required to equal `hidden_size`. The Q
//!   projection is `[q_dim, hidden_size]`; o_proj inverts that to
//!   `[hidden_size, q_dim]`. K/V project to `kv_dim = num_key_value_heads
//!   * head_dim`.
//! - Per-head Q/K RmsNorm on `head_dim` (offset `(gain + 1)`) sits
//!   between the QK projection split and RoPE — these gains stay F32.
//! - Tied `lm_head`: the GGUF loader synthesizes the output matrix from
//!   the token embedding when `output.weight` is absent (the common case
//!   for Gemma releases).
//!
//! Construction paths:
//! - [`QuantizedGemma3Model::load_from_mmapped`] — convenience wrapper
//!   over [`Gemma3Weights::load_from_mmapped`] + [`Self::from_f32_bake`].
//! - [`QuantizedGemma3Model::from_f32_bake`] — take f32 source weights
//!   (same `[in_features, out_features]` layout as `Gemma3Weights`) and
//!   quantize each Linear weight to Q4_0 on the fly. Used by tests and
//!   by callers that have unquantized weights in memory already.
//! - [`QuantizedGemma3Model::from_gguf`] — load directly from a Gemma 3
//!   GGUF file, keeping Q4_0 tensors quantized and dequantizing other
//!   GGML dtypes for Linear weights to F32. Mirrors the SmolLM3 loader
//!   policy.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_gemma3::{Gemma3Config, Gemma3LayerWeights, Gemma3Model, Gemma3Weights};
use crate::Result;
use std::sync::Arc;

/// GGUF-quantized Gemma 3 causal language model. Wraps a plain
/// [`Gemma3Model`] whose Linear weights are `WeightStorage::Q4_0`. The
/// tied `lm_head` continues to share storage with `token_embedding`
/// (kept in F32), matching the eager Gemma 3 convention.
#[derive(Debug, Clone)]
pub struct QuantizedGemma3Model {
    inner: Gemma3Model,
}

impl QuantizedGemma3Model {
    /// Forward over a token-ID sequence. Returns logits with shape
    /// `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward(tokens, start_pos)
    }

    /// Forward over pre-computed embeddings (skipping the
    /// token-embedding lookup). The caller is responsible for the
    /// `sqrt(hidden_size)` scaling — Gemma 3 expects pre-scaled
    /// embeddings here.
    pub fn forward_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_embeds(scaled_embeds, start_pos)
    }

    /// Per-token hidden states up to the final offset RmsNorm, shape
    /// `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward_hidden(tokens, start_pos)
    }

    /// Pre-embedded variant of [`Self::forward_hidden`].
    pub fn forward_hidden_embeds(
        &self, scaled_embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_hidden_embeds(scaled_embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder. Used by
    /// multimodal compositions to obtain text-side embeddings that will
    /// be spliced with vision features before [`Self::forward_embeds`].
    /// The caller is responsible for the `sqrt(hidden_size)` scaling.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        self.inner.embed_tokens_anchored(anchor, tokens)
    }

    /// Model configuration.
    pub fn config(&self) -> &Gemma3Config { &self.inner.config }

    /// Underlying [`Gemma3Model`] for direct access to the lazy graph
    /// API. The wrapper exists solely to label the quantization origin.
    pub fn inner(&self) -> &Gemma3Model { &self.inner }

    /// Convenience: load f32 Gemma 3 weights from HF safetensors and
    /// quantize each Linear weight to Q4_0. Equivalent to
    /// `Self::from_f32_bake(cfg, Gemma3Weights::load_from_mmapped(st, &cfg)?)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: Gemma3Config,
    ) -> Result<Self> {
        let src = Gemma3Weights::load_from_mmapped(st, &cfg)?;
        Self::from_f32_bake(cfg, src)
    }

    /// Construct from in-memory f32 source weights, quantizing each
    /// Linear weight to Q4_0. Norm gains, biases, and the
    /// token-embedding table stay in F32 — matching the GGUF convention
    /// used by llama.cpp for Gemma 3 releases.
    ///
    /// Q4_0 requires `in_features % 32 == 0` for every quantized
    /// matrix. The relevant Gemma-3 dims are checked up front:
    /// - `attn_q` / `ffn_gate` / `ffn_up`: in = `hidden_size`
    /// - `attn_k` / `attn_v`: in = `hidden_size`
    /// - `attn_o`: in = `q_dim = num_attention_heads * head_dim`
    /// - `ffn_down`: in = `intermediate_size`
    ///
    /// Source weights follow the same `[in_features, out_features]`
    /// row-major layout as [`Gemma3Weights`].
    pub fn from_f32_bake(cfg: Gemma3Config, src: Gemma3Weights) -> Result<Self> {
        let h = cfg.hidden_size;
        let i_dim = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // attn_q in=h; attn_o in=q_dim; ffn_down in=i_dim. K/V also in=h.
        check_q4_0_divisible("hidden_size (attn_q/k/v/ffn_gate/ffn_up in-features)", h)?;
        check_q4_0_divisible("num_attention_heads * head_dim (attn_o in-features)", q_dim)?;
        check_q4_0_divisible("intermediate_size (ffn_down in-features)", i_dim)?;

        let quantize_linear = |w: &WeightStorage, in_features: usize, out_features: usize| -> Result<WeightStorage> {
            let f32_in_out = match w {
                WeightStorage::F32(a) => a.to_vec(),
                _ => return Err(crate::Error::Msg(
                    "QuantizedGemma3Model::from_f32_bake: source weights must be WeightStorage::F32".into(),
                ).bt()),
            };
            if f32_in_out.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "QuantizedGemma3Model::from_f32_bake: weight has {} elems, expected {}×{}",
                    f32_in_out.len(), in_features, out_features,
                )).bt());
            }
            quantize_in_out_to_q4_0(&f32_in_out, in_features, out_features)
        };

        let mut layers: Vec<Gemma3LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (idx, layer) in src.layers.into_iter().enumerate() {
            let attn_q   = quantize_linear(&layer.attn_q,   h,     q_dim ).map_err(|e| layer_err(idx, "attn_q",   e))?;
            let attn_k   = quantize_linear(&layer.attn_k,   h,     kv_dim).map_err(|e| layer_err(idx, "attn_k",   e))?;
            let attn_v   = quantize_linear(&layer.attn_v,   h,     kv_dim).map_err(|e| layer_err(idx, "attn_v",   e))?;
            let attn_o   = quantize_linear(&layer.attn_o,   q_dim, h     ).map_err(|e| layer_err(idx, "attn_o",   e))?;
            let ffn_gate = quantize_linear(&layer.ffn_gate, h,     i_dim ).map_err(|e| layer_err(idx, "ffn_gate", e))?;
            let ffn_up   = quantize_linear(&layer.ffn_up,   h,     i_dim ).map_err(|e| layer_err(idx, "ffn_up",   e))?;
            let ffn_down = quantize_linear(&layer.ffn_down, i_dim, h     ).map_err(|e| layer_err(idx, "ffn_down", e))?;
            layers.push(Gemma3LayerWeights {
                attn_q, attn_q_bias: layer.attn_q_bias,
                attn_k, attn_k_bias: layer.attn_k_bias,
                attn_v, attn_v_bias: layer.attn_v_bias,
                attn_o, attn_o_bias: layer.attn_o_bias,
                q_norm_gain: layer.q_norm_gain,
                k_norm_gain: layer.k_norm_gain,
                input_norm_gain: layer.input_norm_gain,
                post_attn_norm_gain: layer.post_attn_norm_gain,
                pre_ffn_norm_gain: layer.pre_ffn_norm_gain,
                post_ffn_norm_gain: layer.post_ffn_norm_gain,
                ffn_gate, ffn_up, ffn_down,
            });
        }

        let inner = Gemma3Model {
            config: cfg,
            weights: Gemma3Weights {
                token_embedding: src.token_embedding,
                layers,
                final_norm_gain: src.final_norm_gain,
            },
        };
        Ok(Self { inner })
    }

    /// Load a GGUF-quantized Gemma 3 checkpoint. Q4_0 weight tensors
    /// stay quantized; norms / biases / embeddings dequantize to F32.
    /// Other GGML dtypes for Linear weights are dequantized to F32 —
    /// matching the SmolLM3 loader policy. Because Gemma 3 ties
    /// `lm_head` to the token embedding, no `output.weight` lookup is
    /// performed — the embedding table is the lm_head.
    ///
    /// GGUF tensor names follow the llama.cpp convention:
    ///   - `token_embd.weight`       → `token_embedding`
    ///   - `output_norm.weight`      → `final_norm_gain`
    ///   - `blk.{i}.attn_q.weight`   → `attn_q` (also `.bias` if present)
    ///   - `blk.{i}.attn_k.weight`   → `attn_k` (also `.bias` if present)
    ///   - `blk.{i}.attn_v.weight`   → `attn_v` (also `.bias` if present)
    ///   - `blk.{i}.attn_output.weight` → `attn_o` (also `.bias` if present)
    ///   - `blk.{i}.attn_q_norm.weight` → `q_norm_gain` (per-head, head_dim)
    ///   - `blk.{i}.attn_k_norm.weight` → `k_norm_gain` (per-head, head_dim)
    ///   - `blk.{i}.attn_norm.weight`            → `input_norm_gain`
    ///   - `blk.{i}.post_attention_norm.weight`  → `post_attn_norm_gain`
    ///   - `blk.{i}.ffn_norm.weight`             → `pre_ffn_norm_gain`
    ///   - `blk.{i}.post_ffw_norm.weight`        → `post_ffn_norm_gain`
    ///   - `blk.{i}.ffn_gate.weight` → `ffn_gate`
    ///   - `blk.{i}.ffn_up.weight`   → `ffn_up`
    ///   - `blk.{i}.ffn_down.weight` → `ffn_down`
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P, cfg: &Gemma3Config,
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
        let i_dim = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let token_embedding = load_f32("token_embd.weight")?;
        if token_embedding.len() != cfg.vocab_size * h {
            return Err(crate::Error::Msg(format!(
                "gguf token_embd.weight: {} elems, expected {}×{}",
                token_embedding.len(), cfg.vocab_size, h,
            )).bt());
        }

        let mut layers: Vec<Gemma3LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{idx}");
            let attn_q   = load_weight(&format!("{prefix}.attn_q.weight"),      q_dim,  h    )?;
            let attn_k   = load_weight(&format!("{prefix}.attn_k.weight"),      kv_dim, h    )?;
            let attn_v   = load_weight(&format!("{prefix}.attn_v.weight"),      kv_dim, h    )?;
            let attn_o   = load_weight(&format!("{prefix}.attn_output.weight"), h,      q_dim)?;
            let ffn_gate = load_weight(&format!("{prefix}.ffn_gate.weight"),    i_dim,  h    )?;
            let ffn_up   = load_weight(&format!("{prefix}.ffn_up.weight"),      i_dim,  h    )?;
            let ffn_down = load_weight(&format!("{prefix}.ffn_down.weight"),    h,      i_dim)?;

            // Per-head Q/K RmsNorm on head_dim — present on every Gemma 3 layer.
            let q_norm_gain: Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.attn_q_norm.weight"))?);
            let k_norm_gain: Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.attn_k_norm.weight"))?);

            // Four block-level norms (input, post-attn, pre-FFN, post-FFN).
            let input_norm_gain: Arc<[f32]>     = Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let post_attn_norm_gain: Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.post_attention_norm.weight"))?);
            let pre_ffn_norm_gain: Arc<[f32]>   = Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
            let post_ffn_norm_gain: Arc<[f32]>  = Arc::from(load_f32(&format!("{prefix}.post_ffw_norm.weight"))?);

            // Biases are optional and only present when attention_bias is set
            // upstream; honor whatever the file ships.
            let bias = |name: &str, len: usize| -> Option<Arc<[f32]>> {
                load_f32(name).ok().and_then(|v| if v.len() == len { Some(Arc::from(v)) } else { None })
            };

            layers.push(Gemma3LayerWeights {
                attn_q,
                attn_q_bias: bias(&format!("{prefix}.attn_q.bias"),      q_dim),
                attn_k,
                attn_k_bias: bias(&format!("{prefix}.attn_k.bias"),      kv_dim),
                attn_v,
                attn_v_bias: bias(&format!("{prefix}.attn_v.bias"),      kv_dim),
                attn_o,
                attn_o_bias: bias(&format!("{prefix}.attn_output.bias"), h),
                q_norm_gain, k_norm_gain,
                input_norm_gain,
                post_attn_norm_gain,
                pre_ffn_norm_gain,
                post_ffn_norm_gain,
                ffn_gate, ffn_up, ffn_down,
            });
        }

        let final_norm_gain: Arc<[f32]> = Arc::from(load_f32("output_norm.weight")?);

        // Gemma 3 has tied embeddings — there is no `output.weight` to
        // load (and no separate output field on Gemma3Weights). The
        // forward path reuses `token_embedding` as the lm_head.
        let _ = kv_dim; // suppress dead-binding warning if no caller path uses it post-bias

        let inner = Gemma3Model {
            config: cfg.clone(),
            weights: Gemma3Weights {
                token_embedding: Arc::from(token_embedding),
                layers,
                final_norm_gain,
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
            "QuantizedGemma3Model::from_f32_bake: {name} ({n}) must be divisible by Q4_0 block size ({QK4_0})"
        )).bt());
    }
    Ok(())
}

/// Quantize a `[in_features, out_features]` row-major F32 weight matrix
/// into a `WeightStorage::Q4_0` keeping GGUF's native `[out, in]` block
/// layout. The implementation does the `[in, out] → [out, in]` transpose
/// first, then runs the per-row Q4_0 quantization.
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
/// Mirrors the SmolLM3 helper but lives here so this module stays
/// independent of the SmolLM3 internals.
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
            "gguf {name}: dequant of {other:?} is not supported by lazy_quantized_gemma3",
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
    use crate::lazy_gemma3::GemmaActivation;
    use crate::Device;
    use fuel_core_types::Shape;

    fn test_cfg() -> Gemma3Config {
        // Pick num_heads * head_dim != hidden_size to exercise
        // independent attention/embedding dims like real Gemma3.
        // All Q4_0 in-features must be multiples of 32:
        //   hidden_size = 32, q_dim = 32, intermediate_size = 64.
        Gemma3Config {
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,         // q_dim = 32, kv_dim = 16
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            max_position_embeddings: 64,
            sliding_window: 3,
            sliding_window_pattern: 3,
            attention_bias: false,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
        }
    }

    fn tiny_weights(cfg: &Gemma3Config) -> Gemma3Weights {
        let mut s: u32 = 5151;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i_dim = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h);
        let layers: Vec<Gemma3LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Gemma3LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * q_dim)),  attn_q_bias: None,
                attn_k: WeightStorage::F32(vec_of(h * kv_dim)), attn_k_bias: None,
                attn_v: WeightStorage::F32(vec_of(h * kv_dim)), attn_v_bias: None,
                attn_o: WeightStorage::F32(vec_of(q_dim * h)),  attn_o_bias: None,
                q_norm_gain: Arc::from(vec![0.05_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![0.05_f32; cfg.head_dim]),
                input_norm_gain: Arc::from(vec![0.05_f32; h]),
                post_attn_norm_gain: Arc::from(vec![0.05_f32; h]),
                pre_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                post_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                ffn_gate: WeightStorage::F32(vec_of(h * i_dim)),
                ffn_up:   WeightStorage::F32(vec_of(h * i_dim)),
                ffn_down: WeightStorage::F32(vec_of(i_dim * h)),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![0.05_f32; h]);
        Gemma3Weights { token_embedding, layers, final_norm_gain }
    }

    #[test]
    fn forward_shape_finite_with_q4_0_weights() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedGemma3Model::from_f32_bake(cfg.clone(), src).unwrap();
        // Sanity: all Linear projections in layer 0 are now Q4_0.
        let l0 = &model.inner().weights.layers[0];
        assert!(matches!(l0.attn_q, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_k, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_v, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_o, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_gate, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_up,   WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_down, WeightStorage::Q4_0 { .. }));

        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn from_f32_bake_rejects_unaligned_dims() {
        // intermediate_size not divisible by 32 → reject (covers ffn_down in-features).
        let mut cfg = test_cfg();
        cfg.intermediate_size = 48;
        let src = tiny_weights(&cfg);
        assert!(QuantizedGemma3Model::from_f32_bake(cfg, src).is_err());
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedGemma3Model::from_f32_bake(cfg.clone(), src).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let scaled = embeds.mul_scalar((cfg.hidden_size as f64).sqrt());
        let logits_via_embeds = model.forward_embeds(&scaled, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-4,
            "Quantized Gemma3 forward vs forward_embeds (post-scale) must agree (max diff {max_diff})");
    }
}
