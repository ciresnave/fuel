//! GGUF-quantized LFM2 (Liquid Foundation Model 2) ported to the lazy-graph API.
//!
//! LFM2 with Q4_0 / Q4_K_M block-quantized Linear weights. The forward
//! path is identical to [`crate::lazy_lfm2::LFM2Model`]; only the
//! storage of the per-layer Linear weights changes from F32/BF16 to
//! `WeightStorage::Q4_0`. Norm gains, the token-embedding table, and
//! the depthwise conv kernel stay in F32.
//!
//! Construction paths:
//! - [`QuantizedLFM2Model::from_f32_bake`] — take f32 source weights
//!   (same `[in_features, out_features]` layout as `LFM2Weights`) and
//!   quantize each Linear weight to Q4_0 on the fly. Used by tests and
//!   by callers that already have unquantized weights in memory.
//! - [`QuantizedLFM2Model::from_gguf`] — load directly from an LFM2
//!   GGUF file, keeping Q4_0 tensors quantized and dequantizing other
//!   GGML dtypes for Linear weights to F32. Mirrors the SmolLM3 /
//!   Gemma 3 loader policy. Q4_K_M is dequantized to F32 today — the
//!   lazy graph emits a Q4_0 matmul kernel only, so mixed-block-format
//!   GGUFs run as F32 except for the Q4_0 tensors.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_lfm2::{
    LFM2AttentionWeights, LFM2BlockType, LFM2Config, LFM2ConvWeights,
    LFM2LayerWeights, LFM2MixerWeights, LFM2Model, LFM2Weights,
};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

/// GGUF-quantized LFM2 causal language model. Wraps a plain
/// [`LFM2Model`] whose Linear weights are `WeightStorage::Q4_0`. Norm
/// gains, the token-embedding table, and the depthwise conv kernel
/// stay in F32.
#[derive(Debug, Clone)]
pub struct QuantizedLFM2Model {
    inner: LFM2Model,
}

impl QuantizedLFM2Model {
    /// Forward over a token-ID sequence. Returns logits with shape
    /// `(1, seq, vocab_size)`.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward(tokens, start_pos)
    }

    /// Forward over pre-computed embeddings, skipping the
    /// token-embedding lookup. Embeds must have shape
    /// `(1, seq, hidden_size)`.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_embeds(embeds, start_pos)
    }

    /// Per-token hidden states up to the final RmsNorm, shape
    /// `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.inner.forward_hidden(tokens, start_pos)
    }

    /// Pre-embedded variant of [`Self::forward_hidden`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.inner.forward_hidden_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        self.inner.embed_tokens_anchored(anchor, tokens)
    }

    /// Model configuration.
    pub fn config(&self) -> &LFM2Config { &self.inner.config }

    /// Underlying [`LFM2Model`] for direct access to the lazy graph
    /// API. The wrapper exists solely to label the quantization origin.
    pub fn inner(&self) -> &LFM2Model { &self.inner }

    /// Convenience: load f32 LFM2 weights from HF safetensors and
    /// quantize each Linear weight to Q4_0. Equivalent to
    /// `Self::from_f32_bake(cfg, LFM2Weights::load_from_mmapped(st, &cfg)?)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: LFM2Config,
    ) -> Result<Self> {
        let src = LFM2Weights::load_from_mmapped(st, &cfg)?;
        Self::from_f32_bake(cfg, src)
    }

    /// Construct from in-memory f32 source weights, quantizing each
    /// Linear weight to Q4_0.
    ///
    /// Q4_0 blocks run along each Linear's *in_features* axis, so
    /// every dimension that serves as an in_features must be a
    /// multiple of the Q4_0 block size (32): `cfg.hidden_size`
    /// (attn_q/k/v, conv in_proj/out_proj, ffn_gate/up, output),
    /// `cfg.intermediate_size` (ffn_down), and
    /// `cfg.num_attention_heads * cfg.head_dim` (attn_o).
    /// Out-features carry no divisibility requirement — neither
    /// `num_key_value_heads * head_dim` (attn_k / attn_v outputs)
    /// nor the conv in_proj's `3 * hidden_size`. Source weights
    /// follow the same `[in_features, out_features]` row-major
    /// layout as [`LFM2Weights`].
    pub fn from_f32_bake(cfg: LFM2Config, src: LFM2Weights) -> Result<Self> {
        cfg.validate()?;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        // Gate exactly the dims that serve as a Linear in_features —
        // Q4_0 blocks run along K only. kv_dim (= num_key_value_heads
        // * head_dim, attn_k / attn_v) and the conv in_proj's
        // 3 * hidden are only ever out_features and need no
        // divisibility (mirrors the gemma3 gate).
        check_q4_0_divisible("hidden_size (attn_q/k/v, conv in/out_proj, ffn_gate/up, output in-features)", h)?;
        check_q4_0_divisible("intermediate_size (ffn_down in-features)", inter)?;
        check_q4_0_divisible("num_attention_heads * head_dim (attn_o in-features)", q_dim)?;

        let quantize_linear = |w: &WeightStorage, in_features: usize, out_features: usize| -> Result<WeightStorage> {
            let f32_in_out = match w {
                WeightStorage::F32(a) => a.to_vec(),
                _ => return Err(crate::Error::Msg(
                    "QuantizedLFM2Model::from_f32_bake: source weights must be WeightStorage::F32".into(),
                ).bt()),
            };
            if f32_in_out.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "QuantizedLFM2Model::from_f32_bake: weight has {} elems, expected {in_features}x{out_features}",
                    f32_in_out.len(),
                )).bt());
            }
            quantize_in_out_to_q4_0(&f32_in_out, in_features, out_features)
        };

        let mut layers: Vec<LFM2LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (idx, layer) in src.layers.into_iter().enumerate() {
            let mixer = match layer.mixer {
                LFM2MixerWeights::Attention(a) => {
                    LFM2MixerWeights::Attention(LFM2AttentionWeights {
                        attn_q: quantize_linear(&a.attn_q, h,  q_dim).map_err(|e| layer_err(idx, "attn_q",  e))?,
                        attn_k: quantize_linear(&a.attn_k, h, kv_dim).map_err(|e| layer_err(idx, "attn_k",  e))?,
                        attn_v: quantize_linear(&a.attn_v, h, kv_dim).map_err(|e| layer_err(idx, "attn_v",  e))?,
                        attn_o: quantize_linear(&a.attn_o, q_dim, h).map_err(|e| layer_err(idx, "attn_o",  e))?,
                        q_norm_gain: a.q_norm_gain,
                        k_norm_gain: a.k_norm_gain,
                    })
                }
                LFM2MixerWeights::Conv(c) => {
                    LFM2MixerWeights::Conv(LFM2ConvWeights {
                        in_proj:  quantize_linear(&c.in_proj,  h, 3 * h).map_err(|e| layer_err(idx, "conv.in_proj",  e))?,
                        out_proj: quantize_linear(&c.out_proj, h, h    ).map_err(|e| layer_err(idx, "conv.out_proj", e))?,
                        conv_weight: c.conv_weight, // stays F32
                    })
                }
            };
            layers.push(LFM2LayerWeights {
                operator_norm_gain: layer.operator_norm_gain,
                ffn_norm_gain: layer.ffn_norm_gain,
                mixer,
                ffn_gate: quantize_linear(&layer.ffn_gate, h, inter).map_err(|e| layer_err(idx, "ffn_gate", e))?,
                ffn_up:   quantize_linear(&layer.ffn_up,   h, inter).map_err(|e| layer_err(idx, "ffn_up",   e))?,
                ffn_down: quantize_linear(&layer.ffn_down, inter, h).map_err(|e| layer_err(idx, "ffn_down", e))?,
            });
        }

        let output = quantize_linear(&src.output, h, cfg.vocab_size)
            .map_err(|e| crate::Error::Msg(format!("output: {e}")).bt())?;

        let inner = LFM2Model {
            config: cfg,
            weights: LFM2Weights {
                token_embedding: src.token_embedding,
                layers,
                final_norm_gain: src.final_norm_gain,
                output,
            },
        };
        Ok(Self { inner })
    }

    /// Load a GGUF-quantized LFM2 checkpoint. Q4_0 weight tensors stay
    /// quantized; norms / biases / embeddings / the depthwise conv
    /// kernel dequantize to F32. Other GGML dtypes for Linear weights
    /// are dequantized to F32 — matching the SmolLM3 loader policy.
    /// Embedding and `lm_head` share storage when `output.weight` is
    /// absent (tied embeddings).
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P, cfg: &LFM2Config,
    ) -> Result<Self> {
        use crate::quantized::gguf_mmap::MmapedContent;
        cfg.validate()?;
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
        // Try several candidate names, returning the first hit.
        let load_f32_any = |candidates: &[&str]| -> Result<Vec<f32>> {
            for name in candidates {
                if content.tensor_infos.contains_key(*name) {
                    return load_f32(name);
                }
            }
            Err(crate::Error::Msg(format!(
                "gguf: none of the tensors {candidates:?} are present",
            )).bt())
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
                    // Dequantize and transpose [out, in] -> [in, out].
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
        let load_weight_any = |candidates: &[&str], out_features: usize, in_features: usize| -> Result<WeightStorage> {
            for name in candidates {
                if content.tensor_infos.contains_key(*name) {
                    return load_weight(name, out_features, in_features);
                }
            }
            Err(crate::Error::Msg(format!(
                "gguf: none of the tensors {candidates:?} are present",
            )).bt())
        };

        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let k = cfg.conv_kernel_size;

        let token_embedding = load_f32_any(&[
            "token_embd.weight",
            "tok_embeddings.weight",
            "model.embed_tokens.weight",
        ])?;
        if token_embedding.len() != cfg.vocab_size * h {
            return Err(crate::Error::Msg(format!(
                "gguf token embedding: {} elems, expected {}*{}",
                token_embedding.len(), cfg.vocab_size, h,
            )).bt());
        }

        let mut layers: Vec<LFM2LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, &block) in cfg.block_types.iter().enumerate() {
            let prefix = format!("blk.{i}");

            let operator_norm_gain: Arc<[f32]> = Arc::from(load_f32_any(&[
                &format!("{prefix}.attn_norm.weight"),
                &format!("{prefix}.operator_norm.weight"),
                &format!("{prefix}.attention_norm.weight"),
            ])?);
            let ffn_norm_gain: Arc<[f32]> = Arc::from(load_f32_any(&[
                &format!("{prefix}.ffn_norm.weight"),
                &format!("{prefix}.ffn_norm"),
            ])?);

            let ffn_gate = load_weight_any(&[
                &format!("{prefix}.ffn_gate.weight"),
                &format!("{prefix}.feed_forward.w1.weight"),
            ], inter, h)?;
            let ffn_up = load_weight_any(&[
                &format!("{prefix}.ffn_up.weight"),
                &format!("{prefix}.feed_forward.w3.weight"),
            ], inter, h)?;
            let ffn_down = load_weight_any(&[
                &format!("{prefix}.ffn_down.weight"),
                &format!("{prefix}.feed_forward.w2.weight"),
            ], h, inter)?;

            let mixer = match block {
                LFM2BlockType::Attention => {
                    let attn_q = load_weight_any(&[
                        &format!("{prefix}.attn_q.weight"),
                        &format!("{prefix}.self_attn.q_proj.weight"),
                    ], q_dim, h)?;
                    let attn_k = load_weight_any(&[
                        &format!("{prefix}.attn_k.weight"),
                        &format!("{prefix}.self_attn.k_proj.weight"),
                    ], kv_dim, h)?;
                    let attn_v = load_weight_any(&[
                        &format!("{prefix}.attn_v.weight"),
                        &format!("{prefix}.self_attn.v_proj.weight"),
                    ], kv_dim, h)?;
                    let attn_o = load_weight_any(&[
                        &format!("{prefix}.attn_output.weight"),
                        &format!("{prefix}.self_attn.o_proj.weight"),
                    ], h, q_dim)?;
                    let q_norm_gain: Arc<[f32]> = Arc::from(load_f32_any(&[
                        &format!("{prefix}.attn_q_norm.weight"),
                        &format!("{prefix}.self_attn.q_layernorm.weight"),
                        &format!("{prefix}.attention.q_norm.weight"),
                    ])?);
                    let k_norm_gain: Arc<[f32]> = Arc::from(load_f32_any(&[
                        &format!("{prefix}.attn_k_norm.weight"),
                        &format!("{prefix}.self_attn.k_layernorm.weight"),
                        &format!("{prefix}.attention.k_norm.weight"),
                    ])?);
                    LFM2MixerWeights::Attention(LFM2AttentionWeights {
                        attn_q, attn_k, attn_v, attn_o,
                        q_norm_gain, k_norm_gain,
                    })
                }
                LFM2BlockType::Conv => {
                    let in_proj = load_weight_any(&[
                        &format!("{prefix}.shortconv.in_proj.weight"),
                        &format!("{prefix}.conv.in_proj.weight"),
                    ], 3 * h, h)?;
                    let out_proj = load_weight_any(&[
                        &format!("{prefix}.shortconv.out_proj.weight"),
                        &format!("{prefix}.conv.out_proj.weight"),
                    ], h, h)?;
                    let raw = load_f32_any(&[
                        &format!("{prefix}.shortconv.conv.weight"),
                        &format!("{prefix}.conv.conv.weight"),
                        &format!("{prefix}.shortconv.conv"),
                    ])?;
                    let normalized = normalize_conv_kernel(raw, h, k, i)?;
                    LFM2MixerWeights::Conv(LFM2ConvWeights {
                        in_proj, out_proj,
                        conv_weight: Arc::from(normalized),
                    })
                }
            };

            layers.push(LFM2LayerWeights {
                operator_norm_gain, ffn_norm_gain,
                mixer,
                ffn_gate, ffn_up, ffn_down,
            });
        }

        let final_norm_gain: Arc<[f32]> = Arc::from(load_f32_any(&[
            "output_norm.weight",
            "embedding_norm.weight",
            "model.embedding_norm.weight",
            "model.embedding_norm",
            "token_embd_norm.weight",
        ])?);

        // Tied embeddings: LFM2 frequently reuses the embedding for
        // `lm_head`. If `output.weight` is present we honor it;
        // otherwise we synthesize an F32 lm_head by transposing the
        // (already-dequantized) token embedding [vocab, h] -> [h, vocab].
        let output = if content.tensor_infos.contains_key("output.weight") {
            load_weight("output.weight", cfg.vocab_size, h)?
        } else if content.tensor_infos.contains_key("lm_head.weight") {
            load_weight("lm_head.weight", cfg.vocab_size, h)?
        } else {
            let mut f32_in_out = vec![0.0_f32; h * cfg.vocab_size];
            for v in 0..cfg.vocab_size {
                for j in 0..h {
                    f32_in_out[j * cfg.vocab_size + v] = token_embedding[v * h + j];
                }
            }
            WeightStorage::F32(Arc::from(f32_in_out))
        };

        let inner = LFM2Model {
            config: cfg.clone(),
            weights: LFM2Weights {
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
            "QuantizedLFM2Model::from_f32_bake: {name} ({n}) must be divisible by Q4_0 block size ({QK4_0})"
        )).bt());
    }
    Ok(())
}

/// Normalize a conv kernel pulled from GGUF into the `[hidden, k]`
/// flat layout the `causal_conv1d` op expects. Currently only the
/// length is validated; the per-checkpoint layout heuristic matches
/// the eager loader, which treats the data as already in
/// `[hidden, k]` order (modulo a trailing singleton dim).
fn normalize_conv_kernel(
    raw: Vec<f32>, hidden: usize, k: usize, layer_idx: usize,
) -> Result<Vec<f32>> {
    let want = hidden * k;
    if raw.len() != want {
        return Err(crate::Error::Msg(format!(
            "LFM2 quantized layer {layer_idx} conv kernel: {} elems, expected hidden*k = {hidden}*{k} = {want}",
            raw.len(),
        )).bt());
    }
    Ok(raw)
}

/// Quantize a `[in_features, out_features]` row-major F32 weight
/// matrix into a `WeightStorage::Q4_0` keeping GGUF's native
/// `[out, in]` block layout. Implementation does the
/// `[in, out] -> [out, in]` transpose first, then runs the per-row
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

    let mut f32_out_in = vec![0.0_f32; out_features * in_features];
    for o in 0..out_features {
        for j in 0..in_features {
            f32_out_in[o * in_features + j] = f32_in_out[j * out_features + o];
        }
    }

    let n_blocks = out_features * in_features / QK4_0;
    let mut blocks: Vec<BlockQ4_0> = vec![BlockQ4_0::zeros(); n_blocks];
    BlockQ4_0::from_float(&f32_out_in, &mut blocks);

    let bytes_len = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let byte_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, bytes_len)
    };
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
/// Mirrors the SmolLM3 helper; lives here so this module stays
/// independent of other lazy-quantized loaders.
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
            "gguf {name}: dequant of {other:?} is not supported by lazy_quantized_lfm2 \
             (Q4_K_M and other block formats must be re-quantized upstream or loaded \
              via a future native Q4_K_M matmul path)",
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
    use crate::lazy_lfm2::{LFM2BlockType, LFM2ConvWeights, LFM2AttentionWeights};

    fn test_cfg() -> LFM2Config {
        // hidden/intermediate/q_dim/kv_dim all multiples of 32.
        LFM2Config {
            vocab_size: 32,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 8,
            intermediate_size: 64,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            conv_kernel_size: 4,
            block_types: vec![LFM2BlockType::Attention, LFM2BlockType::Conv],
        }
    }

    fn tiny_weights(cfg: &LFM2Config) -> LFM2Weights {
        let mut s: u32 = 55555;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let k = cfg.conv_kernel_size;
        let token_embedding = vec_of(cfg.vocab_size * h);
        let mut layers: Vec<LFM2LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for &block in cfg.block_types.iter() {
            let mixer = match block {
                LFM2BlockType::Attention => LFM2MixerWeights::Attention(LFM2AttentionWeights {
                    attn_q: WeightStorage::F32(vec_of(h * q_dim)),
                    attn_k: WeightStorage::F32(vec_of(h * kv_dim)),
                    attn_v: WeightStorage::F32(vec_of(h * kv_dim)),
                    attn_o: WeightStorage::F32(vec_of(q_dim * h)),
                    q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                    k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                }),
                LFM2BlockType::Conv => LFM2MixerWeights::Conv(LFM2ConvWeights {
                    in_proj:  WeightStorage::F32(vec_of(h * 3 * h)),
                    out_proj: WeightStorage::F32(vec_of(h * h)),
                    conv_weight: vec_of(h * k),
                }),
            };
            layers.push(LFM2LayerWeights {
                operator_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
                mixer,
                ffn_gate: WeightStorage::F32(vec_of(h * i)),
                ffn_up:   WeightStorage::F32(vec_of(h * i)),
                ffn_down: WeightStorage::F32(vec_of(i * h)),
            });
        }
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size));
        LFM2Weights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_finite_with_q4_0_weights() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedLFM2Model::from_f32_bake(cfg.clone(), src).unwrap();
        // Sanity: in the attention layer 0, all Linear projections are Q4_0.
        let l0 = &model.inner().weights.layers[0];
        match &l0.mixer {
            LFM2MixerWeights::Attention(a) => {
                assert!(matches!(a.attn_q, WeightStorage::Q4_0 { .. }));
                assert!(matches!(a.attn_k, WeightStorage::Q4_0 { .. }));
                assert!(matches!(a.attn_v, WeightStorage::Q4_0 { .. }));
                assert!(matches!(a.attn_o, WeightStorage::Q4_0 { .. }));
            }
            _ => panic!("layer 0 should be attention per the test config"),
        }
        let l1 = &model.inner().weights.layers[1];
        match &l1.mixer {
            LFM2MixerWeights::Conv(c) => {
                assert!(matches!(c.in_proj,  WeightStorage::Q4_0 { .. }));
                assert!(matches!(c.out_proj, WeightStorage::Q4_0 { .. }));
                // Conv kernel itself stays F32.
                assert_eq!(c.conv_weight.len(), cfg.hidden_size * cfg.conv_kernel_size);
            }
            _ => panic!("layer 1 should be conv per the test config"),
        }
        assert!(matches!(l0.ffn_gate, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_up,   WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_down, WeightStorage::Q4_0 { .. }));
        assert!(matches!(model.inner().weights.output, WeightStorage::Q4_0 { .. }));

        let logits = model.forward(&[1, 2, 3, 4], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup_quantized() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedLFM2Model::from_f32_bake(cfg.clone(), src).unwrap();
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
            "Quantized LFM2 forward vs forward_embeds must agree (max diff {max_diff})");
    }
}
