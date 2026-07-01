//! GGUF-quantized LLaMA (LLaMA-2 / LLaMA-3 / LLaMA-3.1) ported to the
//! lazy-graph API.
//!
//! Wraps [`crate::lazy_llama_full::Llama3Model`] — the LLaMA family
//! decoder with optional LLaMA-3.1 three-band RoPE scaling — replacing
//! the Linear weight storage of `attn_{q,k,v,o}` / `ffn_{gate,up,down}`
//! / `output` with `WeightStorage::Q4_0`. RmsNorm gains, the token
//! embedding table, and biases (Qwen2-style; absent for canonical
//! LLaMA) stay in F32.
//!
//! The forward path is identical to [`Llama3Model`] — only the storage
//! kind of the projection matrices changes. RoPE scaling (Llama3),
//! EOS token enums (single or multiple), tied embeddings, and the
//! Qwen2-style attention biases are all preserved as-is.
//!
//! Construction paths:
//! - [`QuantizedLlama3Model::from_f32_bake`] — take f32 source weights
//!   (same `[in, out]` layout as
//!   [`crate::lazy::LlamaWeights`](crate::lazy::LlamaWeights)) and
//!   quantize on the fly. Used by tests and by callers that already
//!   have unquantized weights in memory.
//! - [`QuantizedLlama3Model::load_from_mmapped`] — convenience that
//!   pairs `LlamaWeights::load_from_mmapped` with `from_f32_bake`
//!   for callers that have HF-shape safetensors on disk.
//! - [`QuantizedLlama3Model::from_gguf`] — load directly from a
//!   LLaMA GGUF file using llama.cpp's `blk.{i}.*` naming, keeping
//!   Q4_0 tensors quantized and dequantizing other GGML dtypes to
//!   F32 (mirrors the SmolLM3 / Phi-2 loader convention).
//!
//! ## Field set
//!
//! [`crate::lazy::LayerWeights`](crate::lazy::LayerWeights) is shared
//! with SmolLM3 / Mistral / Qwen2 / etc.; the LLaMA wrapper only
//! quantizes the seven Linear weight matrices per layer plus the
//! output projection. Optional Qwen2-style attention biases
//! (`attn_q_bias`, `attn_k_bias`, `attn_v_bias`) — present on Qwen2
//! checkpoints loaded via this LLaMA-shape path — stay F32 and pass
//! through unchanged.
//!
//! ## Tied embeddings
//!
//! LLaMA-2 / LLaMA-3 / LLaMA-3.1 use untied output projections
//! (`tie_word_embeddings = false`); some smaller derivatives tie them.
//! The GGUF loader honors `output.weight` when present and falls back
//! to a transposed token-embedding F32 matrix otherwise — matching the
//! [`crate::lazy_llama_full::tied_lm_head_from_embeddings`] convention.

use crate::lazy::{LayerWeights, LazyTensor, LlamaModel, LlamaWeights, WeightStorage};
use crate::lazy_llama_full::{Llama3Model, LlamaFullConfig};
use crate::Result;
use std::sync::Arc;

/// GGUF-quantized LLaMA-family causal language model with optional
/// LLaMA-3.1 long-context RoPE scaling. Wraps a plain
/// [`Llama3Model`] whose Linear weights are `WeightStorage::Q4_0`.
///
/// All public forward methods delegate to the inner [`Llama3Model`] /
/// [`LlamaModel`]; the wrapper exists solely to label the quantization
/// origin and provide [`from_f32_bake`](Self::from_f32_bake) /
/// [`from_gguf`](Self::from_gguf) constructors.
#[derive(Debug, Clone)]
pub struct QuantizedLlama3Model {
    inner: Llama3Model,
}

impl QuantizedLlama3Model {
    /// Forward over a token-ID sequence. Returns logits with shape
    /// `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward(tokens, start_pos)
    }

    /// Forward over pre-computed embeddings, skipping the token-embedding
    /// lookup. Embeds must have shape `(1, seq, hidden_size)`.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_embeds(embeds, start_pos)
    }

    /// Per-token hidden states up to the final RmsNorm, shape
    /// `(1, seq, hidden_size)`. Builds the embedding lookup using
    /// `anchor` as the graph anchor (mirrors
    /// [`LlamaModel::forward_hidden`]); useful from training loops
    /// where the lm_head parameter supplies the anchor.
    ///
    /// Note: this path does NOT pick up the LLaMA-3.1 RoPE scaling
    /// from the wrapper because `LlamaModel::forward_hidden` uses
    /// the unscaled RoPE base directly. Use
    /// [`forward_hidden_embeds`](Self::forward_hidden_embeds) when
    /// you need scaled RoPE; that path threads through
    /// [`Llama3Model`] and respects `rope_scaling`.
    pub fn forward_hidden(
        &self, tokens: &[u32], start_pos: usize, anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        self.inner.inner.forward_hidden(tokens, start_pos, anchor)
    }

    /// Pre-embedded variant of [`Self::forward_hidden`]. Honors the
    /// LLaMA-3.1 RoPE scaling carried by the inner [`Llama3Model`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_hidden_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder. Uses
    /// `anchor` as the graph anchor for the constant embedding table
    /// — useful when the embeddings will be fed into a separate
    /// graph (multimodal hosts: LLaVA / Pixtral / Qwen-VL).
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.inner.inner.config;
        anchor.embed_tokens_anchored(
            self.inner.inner.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.dim, tokens,
        )
    }

    /// Underlying [`Llama3Model`] for direct access to the lazy graph
    /// API. The wrapper exists solely to label the quantization origin.
    pub fn inner(&self) -> &Llama3Model { &self.inner }

    /// Convenience: load f32 LLaMA weights from HF safetensors and
    /// quantize each Linear weight to Q4_0. Equivalent to
    /// `Self::from_f32_bake(cfg, LlamaWeights::load_from_mmapped(st, &cfg.to_lazy_config())?)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: LlamaFullConfig,
    ) -> Result<Self> {
        let lazy_cfg = cfg.to_lazy_config();
        let src = LlamaWeights::load_from_mmapped(st, &lazy_cfg)?;
        Self::from_f32_bake(cfg, src)
    }

    /// Construct from in-memory f32 source weights, quantizing each
    /// Linear weight to Q4_0. Norm gains, biases (when present, e.g.
    /// Qwen2-style), and the token-embedding table stay in F32 —
    /// matching the GGUF convention used by llama.cpp for LLaMA-family
    /// releases.
    ///
    /// Q4_0 blocks run along each Linear's *in_features* axis, so
    /// every dimension that serves as an in_features must be a
    /// multiple of the Q4_0 block size (32): `cfg.hidden_size`
    /// (attn_q/k/v/o, ffn_gate/up, output) and
    /// `cfg.intermediate_size` (ffn_down). `num_key_value_heads *
    /// head_dim` only ever appears as an out_features (attn_k /
    /// attn_v) and carries no divisibility requirement. Source
    /// weights follow the same `[in_features, out_features]`
    /// row-major layout as [`LlamaWeights`].
    pub fn from_f32_bake(cfg: LlamaFullConfig, src: LlamaWeights) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        // Gate exactly the dims that serve as a Linear in_features —
        // Q4_0 blocks run along K only. kv (= num_key_value_heads *
        // head_dim) is only ever an out_features (attn_k / attn_v)
        // and needs no divisibility (mirrors the gemma3 gate).
        check_q4_0_divisible("hidden_size (attn_q/k/v/o, ffn_gate/up, output in-features)", h)?;
        check_q4_0_divisible("intermediate_size (ffn_down in-features)", i)?;

        let quantize_linear = |w: &WeightStorage, in_features: usize, out_features: usize| -> Result<WeightStorage> {
            let f32_in_out = match w {
                WeightStorage::F32(a) => a.to_vec(),
                _ => return Err(crate::Error::Msg(
                    "QuantizedLlama3Model::from_f32_bake: source weights must be WeightStorage::F32".into(),
                ).bt()),
            };
            if f32_in_out.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "QuantizedLlama3Model::from_f32_bake: weight has {} elems, expected {}×{}",
                    f32_in_out.len(), in_features, out_features,
                )).bt());
            }
            quantize_in_out_to_q4_0(&f32_in_out, in_features, out_features)
        };

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (idx, layer) in src.layers.into_iter().enumerate() {
            let attn_q   = quantize_linear(&layer.attn_q,   h, h ).map_err(|e| layer_err(idx, "attn_q",   e))?;
            let attn_k   = quantize_linear(&layer.attn_k,   h, kv).map_err(|e| layer_err(idx, "attn_k",   e))?;
            let attn_v   = quantize_linear(&layer.attn_v,   h, kv).map_err(|e| layer_err(idx, "attn_v",   e))?;
            let attn_o   = quantize_linear(&layer.attn_o,   h, h ).map_err(|e| layer_err(idx, "attn_o",   e))?;
            let ffn_gate = quantize_linear(&layer.ffn_gate, h, i ).map_err(|e| layer_err(idx, "ffn_gate", e))?;
            let ffn_up   = quantize_linear(&layer.ffn_up,   h, i ).map_err(|e| layer_err(idx, "ffn_up",   e))?;
            let ffn_down = quantize_linear(&layer.ffn_down, i, h ).map_err(|e| layer_err(idx, "ffn_down", e))?;
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

        let lazy_cfg = cfg.to_lazy_config();
        let llama = LlamaModel {
            config: lazy_cfg,
            weights: LlamaWeights {
                token_embedding: src.token_embedding,
                layers,
                final_norm_gain: src.final_norm_gain,
                output,
            },
        };
        let inner = Llama3Model::new(llama, cfg.rope_scaling.clone(), cfg.eos_token_id.clone());
        Ok(Self { inner })
    }

    /// Load a GGUF-quantized LLaMA checkpoint. Q4_0 weight tensors
    /// stay quantized; norms / biases / embeddings dequantize to F32.
    /// Other GGML dtypes (F16 / BF16 / F32 / Q4_0) for Linear weights
    /// are dequantized to F32 — matching the SmolLM3 / Phi-2 loader
    /// policy. Embedding and `lm_head` share storage (transposed) when
    /// `output.weight` is absent (tied embeddings).
    ///
    /// Tensor names follow the llama.cpp LLaMA convention:
    ///   - `token_embd.weight`
    ///   - `blk.{i}.attn_q.weight`
    ///   - `blk.{i}.attn_k.weight`
    ///   - `blk.{i}.attn_v.weight`
    ///   - `blk.{i}.attn_output.weight`
    ///   - `blk.{i}.ffn_gate.weight`
    ///   - `blk.{i}.ffn_up.weight`
    ///   - `blk.{i}.ffn_down.weight`
    ///   - `blk.{i}.attn_norm.weight`
    ///   - `blk.{i}.ffn_norm.weight`
    ///   - `output_norm.weight`
    ///   - `output.weight` (optional; tied to `token_embd.weight` if absent)
    ///
    /// Optional Qwen2-style attention biases
    /// (`blk.{i}.attn_q.bias` / `attn_k.bias` / `attn_v.bias`) are
    /// loaded as F32 when present and ignored otherwise.
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P, cfg: &LlamaFullConfig,
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
        let kv = cfg.num_key_value_heads * cfg.head_dim;

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
            let attn_q   = load_weight(&format!("{prefix}.attn_q.weight"),      h,  h)?;
            let attn_k   = load_weight(&format!("{prefix}.attn_k.weight"),      kv, h)?;
            let attn_v   = load_weight(&format!("{prefix}.attn_v.weight"),      kv, h)?;
            let attn_o   = load_weight(&format!("{prefix}.attn_output.weight"), h,  h)?;
            let ffn_gate = load_weight(&format!("{prefix}.ffn_gate.weight"),    i,  h)?;
            let ffn_up   = load_weight(&format!("{prefix}.ffn_up.weight"),      i,  h)?;
            let ffn_down = load_weight(&format!("{prefix}.ffn_down.weight"),    h,  i)?;
            let attn_norm_gain: Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let ffn_norm_gain:  Arc<[f32]> = Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
            let bias = |name: &str, len: usize| -> Option<Arc<[f32]>> {
                load_f32(name).ok().and_then(|v| if v.len() == len { Some(Arc::from(v)) } else { None })
            };
            layers.push(LayerWeights {
                attn_q,
                attn_q_bias: bias(&format!("{prefix}.attn_q.bias"), h),
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

        // Tied embeddings: if `output.weight` is absent, synthesize an
        // F32 lm_head by transposing [vocab, h] → [h, vocab] from the
        // (already-dequantized) token embedding. Matches
        // `tied_lm_head_from_embeddings` in `lazy_llama_full`.
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

        let lazy_cfg = cfg.to_lazy_config();
        let llama = LlamaModel {
            config: lazy_cfg,
            weights: LlamaWeights {
                token_embedding: Arc::from(token_embedding),
                layers, final_norm_gain, output,
            },
        };
        let inner = Llama3Model::new(llama, cfg.rope_scaling.clone(), cfg.eos_token_id.clone());
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
            "QuantizedLlama3Model::from_f32_bake: {name} ({n}) must be divisible by Q4_0 block size ({QK4_0})"
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
/// Mirrors the SmolLM3 / Phi-2 helpers but lives here so this module
/// stays independent of those modules' internals.
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
            "gguf {name}: dequant of {other:?} is not supported by lazy_quantized_llama",
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
    use crate::lazy::LlamaConfig;
    use crate::lazy_llama_full::{Llama3RopeConfig, Llama3RopeType, LlamaEosToks};
    use crate::Device;
    use fuel_ir::Shape;

    fn test_cfg() -> LlamaFullConfig {
        LlamaFullConfig {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    fn tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 55555;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim;
        let i = cfg.ffn_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h)),  attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h)),
            ffn_gate: WeightStorage::F32(vec_of(h * i)),
            ffn_up:   WeightStorage::F32(vec_of(h * i)),
            ffn_down: WeightStorage::F32(vec_of(i * h)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size));
        LlamaWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_finite_with_q4_0_weights() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg.to_lazy_config());
        let model = QuantizedLlama3Model::from_f32_bake(cfg.clone(), src).unwrap();
        // Sanity: all Linear projections in layer 0 are now Q4_0.
        let l0 = &model.inner().inner.weights.layers[0];
        assert!(matches!(l0.attn_q, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_k, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_v, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_o, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_gate, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_up,   WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_down, WeightStorage::Q4_0 { .. }));
        assert!(matches!(model.inner().inner.weights.output, WeightStorage::Q4_0 { .. }));

        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg.to_lazy_config());
        let model = QuantizedLlama3Model::from_f32_bake(cfg.clone(), src).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-4,
            "Quantized Llama3 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    /// With LLaMA-3.1 RoPE scaling, the quantized wrapper must still
    /// produce finite logits of the expected shape. Doesn't validate
    /// numerical match against unscaled — scaled-RoPE numerics are
    /// exercised by `lazy_llama_full::tests`.
    #[test]
    fn scaled_forward_shape_and_finite() {
        let mut cfg = test_cfg();
        cfg.rope_scaling = Some(Llama3RopeConfig {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            rope_type: Llama3RopeType::Llama3,
        });
        cfg.eos_token_id = Some(LlamaEosToks::Multiple(vec![128001, 128008, 128009]));
        let src = tiny_weights(&cfg.to_lazy_config());
        let model = QuantizedLlama3Model::from_f32_bake(cfg.clone(), src).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
        // RoPE scaling + EOS metadata survive the quantization wrapper.
        assert!(model.inner().rope_scaling.is_some());
        assert!(matches!(model.inner().eos_token_id, Some(LlamaEosToks::Multiple(_))));
    }

    /// A LLaMA `hidden_size` not divisible by Q4_0's block size (32)
    /// must be rejected at construction. We accept the build-time
    /// failure here (matching the SmolLM3 wrapper); callers with
    /// awkward dims should fall back to F32 storage.
    #[test]
    fn rejects_non_q4_0_aligned_hidden_size() {
        let mut cfg = test_cfg();
        cfg.hidden_size = 30; // not a multiple of 32
        cfg.num_attention_heads = 6;
        cfg.num_key_value_heads = 6;
        cfg.head_dim = 5;
        let src = tiny_weights(&cfg.to_lazy_config());
        let err = QuantizedLlama3Model::from_f32_bake(cfg, src).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("hidden_size") && msg.contains("divisible"),
            "expected divisibility error, got: {msg}",
        );
    }
}
