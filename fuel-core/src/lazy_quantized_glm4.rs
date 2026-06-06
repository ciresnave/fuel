//! GGUF-quantized GLM-4 ported to the lazy-graph API.
//!
//! GLM-4 with Q4_0 block-quantized Linear weights. The forward path is
//! identical to [`crate::lazy_glm4::Glm4Model`]; only the weight storage
//! of `attn_{q,k,v,o}` / `ffn_gate_up` / `ffn_down` / `lm_head` changes
//! from F32 to `WeightStorage::Q4_0`. RmsNorm gains, the token embedding
//! table, and the (optional) Q/K/V biases stay in F32.
//!
//! Construction paths:
//! - [`QuantizedGlm4Model::from_f32_bake`] — take f32 source weights
//!   (same `[in, out]` layout as `Glm4Weights`) and quantize on the fly.
//!   Used by tests and by callers that have unquantized weights in memory
//!   already.
//! - [`QuantizedGlm4Model::load_from_mmapped`] — convenience that loads
//!   f32 weights from HF safetensors and then quantizes (round-trips
//!   through [`Glm4Weights::load_from_mmapped`]).
//! - [`QuantizedGlm4Model::from_gguf`] — load directly from a GGUF file
//!   produced by `llama.cpp`'s `convert_hf_to_gguf.py` with `chatglm`
//!   arch tag. Q4_0 tensors stay quantized; other GGML dtypes for Linear
//!   weights are dequantized to F32, mirroring the SmolLM3 loader policy.
//!
//! GGUF tensor naming follows llama.cpp's chatglm/glm-4 mapping:
//!   `token_embd.weight`, `output_norm.weight`, optional `output.weight`
//!   (lm_head), and per-block:
//!     - `blk.{i}.attn_norm.weight`              — input_layernorm
//!     - `blk.{i}.post_attention_norm.weight`    — post_self_attn_layernorm
//!     - `blk.{i}.ffn_norm.weight`               — post_attention_layernorm
//!     - `blk.{i}.post_ffw_norm.weight`          — post_mlp_layernorm
//!     - `blk.{i}.attn_q.weight` / `attn_k.weight` / `attn_v.weight`
//!     - `blk.{i}.attn_q.bias` / `attn_k.bias` / `attn_v.bias` (optional)
//!     - `blk.{i}.attn_output.weight`
//!     - `blk.{i}.ffn_gate.weight` + `blk.{i}.ffn_up.weight` (split form,
//!       which we concatenate along the out-features dim into the fused
//!       `[hidden, 2*intermediate]` matrix the lazy model expects)
//!     - `blk.{i}.ffn_down.weight`
//!
//! A tied lm_head (`tie_word_embeddings`) is honored automatically when
//! `output.weight` is absent in the GGUF file.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_glm4::{Glm4Config, Glm4LayerWeights, Glm4Model, Glm4Weights};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// GGUF-quantized GLM-4 causal language model. Wraps a plain
/// [`Glm4Model`] whose Linear weights are `WeightStorage::Q4_0`. RmsNorm
/// gains, biases (when present), and the token-embedding table stay in
/// F32 — matching the GGUF convention used by llama.cpp for GLM-4 /
/// ChatGLM releases.
#[derive(Debug, Clone)]
pub struct QuantizedGlm4Model {
    inner: Glm4Model,
}

impl QuantizedGlm4Model {
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
    pub fn config(&self) -> &Glm4Config { &self.inner.config }

    /// Underlying [`Glm4Model`] for direct access to the lazy graph API.
    /// The wrapper exists solely to label the quantization origin.
    pub fn inner(&self) -> &Glm4Model { &self.inner }

    /// Convenience: load f32 GLM-4 weights from HF safetensors and
    /// quantize each Linear weight to Q4_0. Equivalent to
    /// `Self::from_f32_bake(cfg, Glm4Weights::load_from_mmapped(st, &cfg)?)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: Glm4Config,
    ) -> Result<Self> {
        let src = Glm4Weights::load_from_mmapped(st, &cfg)?;
        Self::from_f32_bake(cfg, src)
    }

    /// Construct from in-memory f32 source weights, quantizing each
    /// Linear weight to Q4_0. Norm gains, biases (optional), and the
    /// token-embedding table stay in F32.
    ///
    /// `cfg.hidden_size`, `cfg.intermediate_size`,
    /// `cfg.num_attention_heads * cfg.head_dim`, and
    /// `cfg.num_key_value_heads * cfg.head_dim` must each be a multiple
    /// of the Q4_0 block size (32). Source weights follow the same
    /// `[in_features, out_features]` row-major layout as
    /// [`Glm4Weights`]. The fused `ffn_gate_up` matrix has
    /// `out_features = 2 * intermediate_size`, which is automatically a
    /// multiple of 32 once `intermediate_size` is.
    pub fn from_f32_bake(cfg: Glm4Config, src: Glm4Weights) -> Result<Self> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        check_q4_0_divisible("hidden_size", h)?;
        check_q4_0_divisible("intermediate_size", inter)?;
        check_q4_0_divisible("num_attention_heads * head_dim", q_dim)?;
        check_q4_0_divisible("num_key_value_heads * head_dim", kv)?;

        let quantize_linear = |w: &WeightStorage, in_features: usize, out_features: usize| -> Result<WeightStorage> {
            let f32_in_out = match w {
                WeightStorage::F32(a) => a.to_vec(),
                _ => return Err(crate::Error::Msg(
                    "QuantizedGlm4Model::from_f32_bake: source weights must be WeightStorage::F32".into(),
                ).bt()),
            };
            if f32_in_out.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "QuantizedGlm4Model::from_f32_bake: weight has {} elems, expected {}×{}",
                    f32_in_out.len(), in_features, out_features,
                )).bt());
            }
            quantize_in_out_to_q4_0(&f32_in_out, in_features, out_features)
        };

        let mut layers: Vec<Glm4LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for (idx, layer) in src.layers.into_iter().enumerate() {
            let attn_q       = quantize_linear(&layer.attn_q,       h,     q_dim     ).map_err(|e| layer_err(idx, "attn_q",       e))?;
            let attn_k       = quantize_linear(&layer.attn_k,       h,     kv        ).map_err(|e| layer_err(idx, "attn_k",       e))?;
            let attn_v       = quantize_linear(&layer.attn_v,       h,     kv        ).map_err(|e| layer_err(idx, "attn_v",       e))?;
            let attn_o       = quantize_linear(&layer.attn_o,       q_dim, h         ).map_err(|e| layer_err(idx, "attn_o",       e))?;
            let ffn_gate_up  = quantize_linear(&layer.ffn_gate_up,  h,     2 * inter ).map_err(|e| layer_err(idx, "ffn_gate_up",  e))?;
            let ffn_down     = quantize_linear(&layer.ffn_down,     inter, h         ).map_err(|e| layer_err(idx, "ffn_down",     e))?;
            layers.push(Glm4LayerWeights {
                input_norm_gain: layer.input_norm_gain,
                post_self_attn_norm_gain: layer.post_self_attn_norm_gain,
                post_attn_norm_gain: layer.post_attn_norm_gain,
                post_mlp_norm_gain: layer.post_mlp_norm_gain,
                attn_q, attn_q_bias: layer.attn_q_bias,
                attn_k, attn_k_bias: layer.attn_k_bias,
                attn_v, attn_v_bias: layer.attn_v_bias,
                attn_o,
                ffn_gate_up,
                ffn_down,
            });
        }

        let lm_head = match src.lm_head {
            Some(w) => Some(
                quantize_linear(&w, h, cfg.vocab_size)
                    .map_err(|e| crate::Error::Msg(format!("lm_head: {e}")).bt())?
            ),
            None => None, // tied embeddings stay as the f32 token_embedding
        };

        let inner = Glm4Model {
            config: cfg,
            weights: Glm4Weights {
                token_embedding: src.token_embedding,
                layers,
                final_norm_gain: src.final_norm_gain,
                lm_head,
            },
        };
        Ok(Self { inner })
    }

    /// Load a GGUF-quantized GLM-4 / ChatGLM checkpoint. Q4_0 weight
    /// tensors stay quantized; norms / biases / embeddings dequantize to
    /// F32. Other GGML dtypes for Linear weights are dequantized to F32
    /// — matching the SmolLM3 loader policy. Embedding and `lm_head`
    /// share storage when `output.weight` is absent (tied embeddings).
    ///
    /// llama.cpp's `chatglm` converter emits split `ffn_gate.weight` +
    /// `ffn_up.weight`; this loader concatenates them along the
    /// out-features dim into the fused `[hidden, 2*intermediate]` matrix
    /// the lazy `Glm4Model` expects. If a fused `ffn_up.weight` at the
    /// doubled shape is present we use it as-is (some converters emit
    /// the fused form directly).
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P, cfg: &Glm4Config,
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
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;

        let token_embedding = load_f32("token_embd.weight")?;
        if token_embedding.len() != cfg.vocab_size * h {
            return Err(crate::Error::Msg(format!(
                "gguf token_embd.weight: {} elems, expected {}×{}",
                token_embedding.len(), cfg.vocab_size, h,
            )).bt());
        }

        let mut layers: Vec<Glm4LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{idx}");

            // ---- Norms -------------------------------------------------------
            // GLM-4 four-norm convention. Loaders for older ChatGLM
            // converters may name these differently; in v1 we follow
            // the current llama.cpp `chatglm` mapping verbatim.
            let input_norm_gain: Arc<[f32]> = Arc::from(load_f32(
                &format!("{prefix}.attn_norm.weight"),
            )?);
            let post_self_attn_norm_gain: Arc<[f32]> = Arc::from(load_f32(
                &format!("{prefix}.post_attention_norm.weight"),
            )?);
            let post_attn_norm_gain: Arc<[f32]> = Arc::from(load_f32(
                &format!("{prefix}.ffn_norm.weight"),
            )?);
            let post_mlp_norm_gain: Arc<[f32]> = Arc::from(load_f32(
                &format!("{prefix}.post_ffw_norm.weight"),
            )?);

            // ---- Attention projections --------------------------------------
            let attn_q = load_weight(&format!("{prefix}.attn_q.weight"),      q_dim, h)?;
            let attn_k = load_weight(&format!("{prefix}.attn_k.weight"),      kv,    h)?;
            let attn_v = load_weight(&format!("{prefix}.attn_v.weight"),      kv,    h)?;
            let attn_o = load_weight(&format!("{prefix}.attn_output.weight"), h,     q_dim)?;

            let bias = |name: &str, len: usize| -> Option<Arc<[f32]>> {
                load_f32(name).ok().and_then(|v| if v.len() == len { Some(Arc::from(v)) } else { None })
            };
            let attn_q_bias = if cfg.attention_bias { bias(&format!("{prefix}.attn_q.bias"), q_dim) } else { None };
            let attn_k_bias = if cfg.attention_bias { bias(&format!("{prefix}.attn_k.bias"), kv)    } else { None };
            let attn_v_bias = if cfg.attention_bias { bias(&format!("{prefix}.attn_v.bias"), kv)    } else { None };

            // ---- FFN: fused gate_up, or split gate + up --------------------
            let ffn_gate_up = load_fused_or_split_gate_up(
                &content.tensor_infos, &load_weight, &prefix, h, inter,
            )?;
            let ffn_down = load_weight(&format!("{prefix}.ffn_down.weight"), h, inter)?;

            layers.push(Glm4LayerWeights {
                input_norm_gain, post_self_attn_norm_gain,
                post_attn_norm_gain, post_mlp_norm_gain,
                attn_q, attn_q_bias,
                attn_k, attn_k_bias,
                attn_v, attn_v_bias,
                attn_o,
                ffn_gate_up,
                ffn_down,
            });
        }

        let final_norm_gain: Arc<[f32]> = Arc::from(load_f32("output_norm.weight")?);

        // Tied embeddings: ChatGLM commonly reuses the embedding for
        // `lm_head`. If `output.weight` is present we honor it;
        // otherwise we leave `lm_head = None` and the model falls back
        // to the f32 token-embedding matrix at logits time (the eager
        // `Glm4Model` already handles this case).
        let lm_head = if content.tensor_infos.contains_key("output.weight") {
            Some(load_weight("output.weight", cfg.vocab_size, h)?)
        } else {
            None
        };

        let inner = Glm4Model {
            config: cfg.clone(),
            weights: Glm4Weights {
                token_embedding: Arc::from(token_embedding),
                layers, final_norm_gain, lm_head,
            },
        };
        Ok(Self { inner })
    }
}

/// Load the fused `[hidden, 2*intermediate]` gate+up matrix from GGUF.
///
/// Two layouts seen in the wild:
///   1. Fused: a single `blk.{i}.ffn_up.weight` shaped `[2*inter, hidden]`
///      (some converters merge the two halves directly). Detect by the
///      tensor's element count.
///   2. Split: `blk.{i}.ffn_gate.weight` + `blk.{i}.ffn_up.weight`, each
///      shaped `[inter, hidden]`. We concatenate them along the
///      out-features dim — `gate` first, then `up` — matching the lazy
///      `Glm4Model::apply_layer` slice order
///      (`slice(2, 0, inter)` = gate, `slice(2, inter, inter)` = up).
fn load_fused_or_split_gate_up<F>(
    tensor_infos: &std::collections::HashMap<String, crate::quantized::gguf_file::TensorInfo>,
    load_weight: &F,
    prefix: &str,
    h: usize,
    inter: usize,
) -> Result<WeightStorage>
where
    F: Fn(&str, usize, usize) -> Result<WeightStorage>,
{
    let up_name = format!("{prefix}.ffn_up.weight");
    let gate_name = format!("{prefix}.ffn_gate.weight");

    // Case 1: fused — `ffn_up.weight` already has 2*inter rows.
    if let Some(info) = tensor_infos.get(&up_name) {
        let elems = info.shape.elem_count();
        if elems == 2 * inter * h {
            return load_weight(&up_name, 2 * inter, h);
        }
    }

    // Case 2: split. Load gate and up separately as F32 [hidden, inter]
    // (after the load_weight transpose), then concatenate along the
    // out-features (second) dim into [hidden, 2*inter]. We force the
    // concatenation to flow through F32 because Q4_0 is opaque
    // per-block and cannot be re-sliced post-quantization here without
    // a dequant pass — by going through F32 once we keep semantics
    // intact, and the outer `from_gguf` returns this matrix as F32
    // (it will be requantized by `from_f32_bake` if the caller layered
    // these two ctor paths).
    let gate_ws = load_weight(&gate_name, inter, h)?;
    let up_ws   = load_weight(&up_name,   inter, h)?;

    let gate_in_out = weight_storage_to_f32_in_out(&gate_ws, h, inter, &gate_name)?;
    let up_in_out   = weight_storage_to_f32_in_out(&up_ws,   h, inter, &up_name)?;

    // Concat along out-features: row j has gate[j, 0..inter] then
    // up[j, 0..inter]. Storage layout is [in, out] row-major, so for
    // each in-row j we place gate's `inter` values then up's `inter`.
    let mut fused = vec![0.0_f32; h * (2 * inter)];
    for j in 0..h {
        let dst_row_start = j * (2 * inter);
        let g_row_start = j * inter;
        fused[dst_row_start .. dst_row_start + inter]
            .copy_from_slice(&gate_in_out[g_row_start .. g_row_start + inter]);
        fused[dst_row_start + inter .. dst_row_start + 2 * inter]
            .copy_from_slice(&up_in_out[g_row_start .. g_row_start + inter]);
    }
    Ok(WeightStorage::F32(Arc::from(fused)))
}

/// Convert a `WeightStorage` (post `load_weight`) back to a flat f32
/// `[in_features, out_features]` row-major slice. For F32 storage this
/// is a memcpy; for Q4_0 we run the same dequant the dispatch path uses
/// and then transpose the resulting `[out, in]` matrix back to `[in, out]`.
fn weight_storage_to_f32_in_out(
    ws: &WeightStorage, in_features: usize, out_features: usize, name: &str,
) -> Result<Vec<f32>> {
    match ws {
        WeightStorage::F32(a) => {
            if a.len() != in_features * out_features {
                return Err(crate::Error::Msg(format!(
                    "gguf {name}: F32 weight has {} elems, expected {}×{}",
                    a.len(), in_features, out_features,
                )).bt());
            }
            Ok(a.to_vec())
        }
        WeightStorage::Q4_0 { words, bytes_len, .. } => {
            // Reconstruct the byte stream and dequant to [out, in].
            let mut bytes = vec![0_u8; words.len() * 4];
            for (i, w) in words.iter().enumerate() {
                bytes[i*4..i*4+4].copy_from_slice(&w.to_le_bytes());
            }
            let bytes = &bytes[..*bytes_len];
            let dq_out_in = cpu_dequant_q4_0_bytes(bytes);
            if dq_out_in.len() != out_features * in_features {
                return Err(crate::Error::Msg(format!(
                    "gguf {name}: Q4_0 dequant gave {} elems, expected {}×{}",
                    dq_out_in.len(), out_features, in_features,
                )).bt());
            }
            // Transpose [out, in] → [in, out] to match WeightStorage::F32 layout.
            let mut in_out = vec![0.0_f32; in_features * out_features];
            for o in 0..out_features {
                for j in 0..in_features {
                    in_out[j * out_features + o] = dq_out_in[o * in_features + j];
                }
            }
            Ok(in_out)
        }
        _ => Err(crate::Error::Msg(format!(
            "gguf {name}: split-gate-up concatenation only supports F32 / Q4_0 inputs",
        )).bt()),
    }
}

fn layer_err(idx: usize, name: &str, e: crate::Error) -> crate::Error {
    crate::Error::Msg(format!("layer {idx} {name}: {e}")).bt()
}

fn check_q4_0_divisible(name: &str, n: usize) -> Result<()> {
    const QK4_0: usize = 32;
    if !n.is_multiple_of(QK4_0) {
        return Err(crate::Error::Msg(format!(
            "QuantizedGlm4Model::from_f32_bake: {name} ({n}) must be divisible by Q4_0 block size ({QK4_0})"
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
/// Mirrors the small SmolLM3 helper but lives here so this module stays
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
            "gguf {name}: dequant of {other:?} is not supported by lazy_quantized_glm4",
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
    use crate::lazy_glm4::Glm4Activation;

    fn test_cfg() -> Glm4Config {
        // All dims divisible by 32 for Q4_0 quantization.
        Glm4Config {
            vocab_size: 64, hidden_size: 32, intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4, num_key_value_heads: 4, head_dim: 8,
            partial_rotary_factor: 1.0,
            attention_bias: false,
            max_position_embeddings: 64,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            hidden_activation: Glm4Activation::Silu,
            tie_word_embeddings: false,
        }
    }

    fn tiny_weights(cfg: &Glm4Config) -> Glm4Weights {
        let mut s: u32 = 67890;
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
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h);
        let layers: Vec<Glm4LayerWeights> = (0..cfg.num_hidden_layers).map(|_| Glm4LayerWeights {
            input_norm_gain: Arc::from(vec![1.0_f32; h]),
            post_self_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            post_mlp_norm_gain: Arc::from(vec![1.0_f32; h]),
            attn_q: WeightStorage::F32(vec_of(h * q_dim)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv)),    attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv)),    attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(q_dim * h)),
            ffn_gate_up: WeightStorage::F32(vec_of(h * (2 * i))),
            ffn_down:    WeightStorage::F32(vec_of(i * h)),
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let lm_head = Some(WeightStorage::F32(vec_of(h * cfg.vocab_size)));
        Glm4Weights { token_embedding, layers, final_norm_gain, lm_head }
    }

    #[test]
    fn forward_shape_finite_with_q4_0_weights() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedGlm4Model::from_f32_bake(cfg.clone(), src).unwrap();
        // Sanity: all Linear projections in layer 0 are now Q4_0.
        let l0 = &model.inner().weights.layers[0];
        assert!(matches!(l0.attn_q, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_k, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_v, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.attn_o, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_gate_up, WeightStorage::Q4_0 { .. }));
        assert!(matches!(l0.ffn_down,    WeightStorage::Q4_0 { .. }));
        assert!(matches!(model.inner().weights.lm_head.as_ref().unwrap(),
            WeightStorage::Q4_0 { .. }));

        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn tied_embedding_lm_head_stays_none() {
        let cfg = Glm4Config { tie_word_embeddings: true, ..test_cfg() };
        let mut src = tiny_weights(&cfg);
        src.lm_head = None;
        let model = QuantizedGlm4Model::from_f32_bake(cfg.clone(), src).unwrap();
        assert!(model.inner().weights.lm_head.is_none());
        let logits = model.forward(&[1, 2], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 2 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = test_cfg();
        let src = tiny_weights(&cfg);
        let model = QuantizedGlm4Model::from_f32_bake(cfg.clone(), src).unwrap();
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
            "Quantized GLM4 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn from_f32_bake_rejects_non_block_divisible_dims() {
        // hidden_size = 30 is not a multiple of 32.
        let cfg = Glm4Config { hidden_size: 30, ..test_cfg() };
        // We can't actually build the source weights at hidden_size=30
        // because tiny_weights assumes the same cfg; just test the
        // check fires by handing in any source.
        let src = Glm4Weights {
            token_embedding: Arc::from(vec![0.0_f32; cfg.vocab_size * 30]),
            layers: Vec::new(),
            final_norm_gain: Arc::from(vec![1.0_f32; 30]),
            lm_head: None,
        };
        assert!(QuantizedGlm4Model::from_f32_bake(cfg, src).is_err());
    }
}
