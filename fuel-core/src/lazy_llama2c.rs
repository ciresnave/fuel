//! Llama2-C decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Llama2-C is Andrej Karpathy's stripped-down
//! Llama2 implementation (`llama2.c` repo) targeting tiny models
//! trained from scratch. Architecturally **identical to LLaMA**:
//! bias-free GQA + RmsNorm + SwiGLU FFN + RoPE. Only the field
//! names differ (`dim` ↔ `hidden_size`, `n_layers` ↔ `num_hidden_layers`,
//! etc.).
//!
//! Thin wrapper over [`crate::lazy::LlamaModel`] + adapter from
//! [`Llama2cConfig`] to [`crate::lazy::LlamaConfig`].

use crate::inference_context::{InferenceContext, KvCache};
use crate::lazy::{
    LayerWeights, LlamaConfig, LlamaModel, LlamaWeights, LazyTensor,
    SamplingStrategy, WeightStorage,
};
use crate::{DType, Device, Result};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Llama2cConfig {
    /// Transformer dim (== `hidden_size` in HF).
    pub dim: usize,
    /// FFN hidden dim (== `intermediate_size` in HF).
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    /// `dim / n_heads`.
    pub head_dim: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
}

impl Llama2cConfig {
    pub fn from_dim(dim: usize, hidden_dim: usize, n_layers: usize, n_heads: usize, n_kv_heads: usize, vocab_size: usize) -> Self {
        Self {
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
            head_dim: dim / n_heads,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// Convert to the [`LlamaConfig`] shape so the underlying lazy
    /// LLaMA model accepts it.
    pub fn to_llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            vocab_size: self.vocab_size,
            dim:        self.dim,
            n_layers:   self.n_layers,
            n_heads:    self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim:   self.head_dim,
            ffn_dim:    self.hidden_dim,
            norm_eps:   self.norm_eps,
            rope_base:  self.rope_theta,
        }
    }
}

/// Llama2-C language model. Stores its own config naming for
/// safetensors-loader interop with the `llama2.c` checkpoint format;
/// the forward delegates to [`LlamaModel`].
#[derive(Debug, Clone)]
pub struct Llama2cModel {
    pub config: Llama2cConfig,
    pub weights: LlamaWeights,
}

impl Llama2cModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward(tokens, start_pos)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, dim)`. Delegates
    /// to `LlamaModel::forward_hidden` with an internally-built
    /// anchor from the token-embedding constant.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        let anchor = LazyTensor::from_f32(
            llama.weights.token_embedding.clone(),
            fuel_core_types::Shape::from_dims(&[llama.config.vocab_size, llama.config.dim]),
            &crate::Device::cpu(),
        );
        llama.forward_hidden(tokens, start_pos, &anchor)
    }

    /// Variant of [`Self::forward_hidden`] that takes a caller-supplied
    /// graph anchor instead of bootstrapping its own. Used by training
    /// loops that need the model's frozen forward to land on the same
    /// graph as the trainable parameters (e.g. the lm_head being
    /// fine-tuned).
    ///
    /// Delegates to [`LlamaModel::forward_hidden`].
    pub fn forward_hidden_anchored(
        &self,
        tokens: &[u32],
        start_pos: usize,
        anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_hidden(tokens, start_pos, anchor)
    }

    /// Multimodal forward: run the decoder on pre-computed input
    /// embeddings of shape `(batch, seq, dim)`. Returns logits
    /// `(batch, seq, vocab)`. Used by vision-language models
    /// (LLaVA, PaliGemma, Pixtral) that interleave image patch
    /// embeddings with text embeddings before running the LLM.
    ///
    /// Delegates to [`LlamaModel::forward_embeds`].
    pub fn forward_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_embeds(embeds, start_pos)
    }

    /// Multimodal forward returning hidden states (post-final-RmsNorm,
    /// pre-lm_head) instead of logits. Used by adapters / embeddings
    /// pipelines that need the raw representation. Delegates to
    /// [`LlamaModel::forward_hidden_embeds`].
    pub fn forward_hidden_embeds(
        &self,
        embeds: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_hidden_embeds(embeds, start_pos)
    }

    /// [`Self::forward_hidden_embeds`] with a caller-supplied additive
    /// attention mask of shape `(1, 1, seq, seq)` instead of the
    /// internal strict-causal mask. Used by NV-Embed-v2 and other
    /// bidirectional / padded inputs that need a custom mask.
    ///
    /// Delegates to [`LlamaModel::forward_hidden_embeds_with_mask`].
    pub fn forward_hidden_embeds_with_mask(
        &self,
        embeds: &LazyTensor,
        attention_mask: &LazyTensor,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_hidden_embeds_with_mask(embeds, attention_mask, start_pos)
    }

    /// KV-cache-aware forward. Delegates to
    /// [`LlamaModel::forward_with_kv_context`] by building an inline
    /// `LlamaModel` from the current weights + adapted config. The
    /// cache and inference context are owned by the caller so the
    /// same `KvCache` can be reused across decode steps for O(1)
    /// per-token cost instead of O(n).
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_with_kv_context(tokens, cache, ctx)
    }

    /// Greedy / sampled generation with persistent KV cache.
    /// Delegates to [`LlamaModel::generate_with_kv_context`]; the
    /// underlying loop prefills the prompt in one forward then
    /// decodes one token per call until either `max_new_tokens` is
    /// reached or `eos_id` is produced.
    pub fn generate_with_kv_context(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<u32>> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.generate_with_kv_context(
            prompt_tokens, max_new_tokens, strategy, eos_id, device, dtype,
        )
    }

    /// Streaming variant of [`Self::generate_with_kv_context`]. The
    /// `on_token` callback fires once per newly-decoded token (after
    /// the prompt prefill), enabling stdout streaming, token-rate
    /// timing, early stopping, etc.
    pub fn generate_streaming_with_kv_context(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.generate_streaming_with_kv_context(
            prompt_tokens, max_new_tokens, strategy, eos_id, device, dtype, on_token,
        )
    }

    /// Speculative decode against a `draft` Llama2cModel through the
    /// kv-context path. The draft proposes up to `k` next tokens; the
    /// target (this model) verifies them in a single parallel forward
    /// and accepts a prefix. On rejection, both caches are rolled
    /// back via `KvCache::truncate_to`.
    ///
    /// `draft.config.vocab_size` must equal `self.config.vocab_size`.
    ///
    /// Delegates to
    /// [`LlamaModel::generate_streaming_spec_with_kv_context`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_spec_with_kv_context(
        &self,
        draft: &Llama2cModel,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        k: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let target_inline = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        let draft_inline = LlamaModel {
            config: draft.config.to_llama_config(),
            weights: draft.weights.clone(),
        };
        target_inline.generate_streaming_spec_with_kv_context(
            &draft_inline, prompt_tokens, max_new_tokens, k, strategy, eos_id,
            device, dtype, on_token,
        )
    }

    // The device-resident `*_gpu_on` delegate family
    // (`forward_with_cache_gpu_on`, `forward_with_cache_gpu_on_all_positions`,
    // `generate_streaming_gpu_on`, `generate_streaming_spec`,
    // `generate_streaming_cuda`) retired in Unification Session 4
    // (E.3.4) together with the underlying `LlamaModel` methods and
    // `lazy_kv_cache_device::KVCache<B>`. The `*_with_kv_context`
    // family above is the sole forward/generate surface; callers pass
    // a `Device` and the pipelined executor handles backend dispatch.

    /// Download a Llama-2-shape checkpoint (TinyLlama, Llama-2-7B,
    /// etc.) from the HuggingFace Hub and build a ready-to-forward
    /// `Llama2cModel`.
    ///
    /// Parses `config.json` via [`Llama2cConfig::from_hf_json_str`]
    /// and loads weights via the shared
    /// [`crate::lazy::LlamaWeights::load_from_mmapped`] path. Works
    /// with single-file and sharded checkpoints (uses
    /// `model.safetensors.index.json` when present).
    pub fn from_hub(repo_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = Llama2cConfig::from_hf_json_str(&config_str)?;

        let weight_paths: Vec<std::path::PathBuf> = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)
                    .map_err(|e| crate::Error::Msg(format!("parsing index: {e}")))?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|x| x.as_object())
                    .ok_or_else(|| crate::Error::Msg("index: missing weight_map".into()))?;
                let mut unique = std::collections::HashSet::new();
                for v in weight_map.values() {
                    if let Some(s) = v.as_str() {
                        unique.insert(s.to_string());
                    }
                }
                let mut paths = Vec::new();
                for name in &unique {
                    paths.push(
                        repo.get(name)
                            .map_err(|e| crate::Error::Msg(format!("hf-hub {name}: {e}")))?,
                    );
                }
                paths
            }
            Err(_) => vec![repo
                .get("model.safetensors")
                .map_err(|e| {
                    crate::Error::Msg(format!("hf-hub model.safetensors: {e}"))
                })?],
        };

        let st = unsafe { crate::safetensors::MmapedSafetensors::multi(&weight_paths) }?;
        let weights = LlamaWeights::load_from_mmapped(&st, &config.to_llama_config())?;

        Ok(Llama2cModel { config, weights })
    }
}

impl Llama2cConfig {
    /// Parse a Llama-shape `config.json` from HuggingFace. Field
    /// map: HF native names → `Llama2cConfig` native names.
    ///
    ///   `hidden_size` → `dim`,
    ///   `intermediate_size` → `hidden_dim`,
    ///   `num_hidden_layers` → `n_layers`,
    ///   `num_attention_heads` → `n_heads`,
    ///   `num_key_value_heads` → `n_kv_heads` (defaults to `n_heads`
    ///   for non-GQA configs),
    ///   `head_dim` → `head_dim` (defaults to `dim / n_heads`),
    ///   `rms_norm_eps` → `norm_eps` (defaults to 1e-5),
    ///   `rope_theta` → `rope_theta` (defaults to 10000.0).
    ///
    /// Compatible with TinyLlama, Llama-2-7B, Llama-3, Mistral, and
    /// any Llama-shape HF checkpoint.
    pub fn from_hf_json_str(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::Error::Msg(format!("parsing Llama2c config.json: {e}")))?;

        let get_usize = |key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    crate::Error::Msg(format!("Llama2c config.json: missing/invalid {key:?}"))
                })
        };
        let get_f64 = |key: &str| -> Option<f64> { v.get(key).and_then(|x| x.as_f64()) };

        let vocab_size = get_usize("vocab_size")?;
        let dim = get_usize("hidden_size")?;
        let n_layers = get_usize("num_hidden_layers")?;
        let n_heads = get_usize("num_attention_heads")?;
        let n_kv_heads = v
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(n_heads);
        let hidden_dim = get_usize("intermediate_size")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-5);
        let rope_theta = get_f64("rope_theta").unwrap_or(10_000.0);

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            head_dim,
            norm_eps,
            rope_theta,
        })
    }
}

// -----------------------------------------------------------------
// llama2.c binary checkpoint loader (Karpathy format, v0 / legacy).
// -----------------------------------------------------------------

fn read_i32_le<R: std::io::Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| crate::Error::Msg(format!("llama2c bin: short read on i32 header: {e}")))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32_vec<R: std::io::Read>(r: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes).map_err(|e| {
        crate::Error::Msg(format!(
            "llama2c bin: short read of {n} f32s ({} bytes): {e}",
            n * 4,
        ))
    })?;
    let mut out = vec![0.0_f32; n];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

/// In-place row-major transpose. Source has shape `(rows, cols)`;
/// returns a new vec with shape `(cols, rows)`.
fn transpose_rows_cols(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Read a llama2.c binary checkpoint (Karpathy's legacy format) into
/// a ready-to-forward [`Llama2cModel`].
///
/// Header is 7 little-endian `i32`s: `dim`, `hidden_dim`, `n_layers`,
/// `n_heads`, `n_kv_heads`, `vocab_size` (signed — negative signals
/// an untied lm_head present at the end of the file), `seq_len`.
///
/// Then weights are stored in PyTorch row-major `(out_features,
/// in_features)` layout per linear layer, in this order:
///
///   1. token_embedding  `(vocab_size, dim)`
///   2. rms_att          `(n_layers, dim)`
///   3. wq               `(n_layers, dim, dim)`
///   4. wk               `(n_layers, kv_dim, dim)` where `kv_dim = n_kv_heads * head_dim`
///   5. wv               `(n_layers, kv_dim, dim)`
///   6. wo               `(n_layers, dim, dim)`
///   7. rms_ffn          `(n_layers, dim)`
///   8. w1 (gate)        `(n_layers, hidden_dim, dim)`
///   9. w2 (down)        `(n_layers, dim, hidden_dim)`
///  10. w3 (up)          `(n_layers, hidden_dim, dim)`
///  11. rms_final        `(dim,)`
///  12. freq_cis_real    `(seq_len, head_dim/2)`  — skipped (recomputed by graph)
///  13. freq_cis_imag    `(seq_len, head_dim/2)`  — skipped
///  14. lm_head          `(vocab_size, dim)` — only if `vocab_size` was negative
///
/// The lazy port discards the precomputed RoPE tables (the graph rebuilds
/// them host-side from `rope_theta = 10000.0`) and transposes each linear
/// weight to `(in_features, out_features)` — the layout
/// [`WeightStorage::apply_linear`] expects.
///
/// When `shared_classifier == true` (positive vocab_size in the header),
/// the lm_head weight is materialized as the transpose of the token
/// embedding table — exactly what the eager
/// `TransformerWeights::var_builder` path does with
/// `lm_head.weight = tr(token_embedding_table)`.
///
/// # Errors
///
/// Returns an error on truncated input, on `dim % n_heads != 0`, or on
/// `head_dim % 2 != 0` (RoPE prerequisite).
pub fn load_llama2c_bin<R: std::io::Read>(r: &mut R) -> Result<Llama2cModel> {
    let dim = read_i32_le(r)? as usize;
    let hidden_dim = read_i32_le(r)? as usize;
    let n_layers = read_i32_le(r)? as usize;
    let n_heads = read_i32_le(r)? as usize;
    let n_kv_heads = read_i32_le(r)? as usize;
    let vocab_signed = read_i32_le(r)?;
    let shared_classifier = vocab_signed > 0;
    let vocab_size = vocab_signed.unsigned_abs() as usize;
    let seq_len = read_i32_le(r)? as usize;

    if dim == 0 || n_heads == 0 || !dim.is_multiple_of(n_heads) {
        return Err(crate::Error::Msg(format!(
            "llama2c bin: invalid header dim={dim} n_heads={n_heads}",
        )));
    }
    let head_dim = dim / n_heads;
    if !head_dim.is_multiple_of(2) {
        return Err(crate::Error::Msg(format!(
            "llama2c bin: head_dim {head_dim} must be even for RoPE",
        )));
    }
    let kv_dim = n_kv_heads * head_dim;

    let config = Llama2cConfig {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps: 1e-5,
        rope_theta: 10_000.0,
    };

    // 1. Token embedding table — kept as-is (vocab_size, dim) row-major.
    //    Same layout `LlamaWeights::token_embedding` and the lazy
    //    embedding-lookup path expect.
    let token_embedding: Arc<[f32]> = Arc::from(read_f32_vec(r, vocab_size * dim)?);

    // 2. Per-layer RmsNorm gains (attn pre-norm), interleaved across layers.
    let rms_att = read_f32_vec(r, n_layers * dim)?;

    // 3–6. Q/K/V/O projection weights, stored per layer in (out, in) order.
    let wq_all = read_f32_vec(r, n_layers * dim * dim)?;
    let wk_all = read_f32_vec(r, n_layers * kv_dim * dim)?;
    let wv_all = read_f32_vec(r, n_layers * kv_dim * dim)?;
    let wo_all = read_f32_vec(r, n_layers * dim * dim)?;

    // 7. Per-layer RmsNorm gains (ffn pre-norm).
    let rms_ffn = read_f32_vec(r, n_layers * dim)?;

    // 8–10. SwiGLU FFN weights.
    //   w1 = gate  (hidden_dim, dim)
    //   w2 = down  (dim, hidden_dim)
    //   w3 = up    (hidden_dim, dim)
    let w1_all = read_f32_vec(r, n_layers * hidden_dim * dim)?;
    let w2_all = read_f32_vec(r, n_layers * dim * hidden_dim)?;
    let w3_all = read_f32_vec(r, n_layers * hidden_dim * dim)?;

    // 11. Final RmsNorm gain.
    let rms_final: Arc<[f32]> = Arc::from(read_f32_vec(r, dim)?);

    // 12–13. Precomputed RoPE freq_cis tables — skip; the lazy graph
    //        rebuilds them from `rope_theta` each forward.
    if seq_len > 0 {
        let half = head_dim / 2;
        let _ = read_f32_vec(r, seq_len * half)?;
        let _ = read_f32_vec(r, seq_len * half)?;
    }

    // 14. Optional separate lm_head when classifier is not tied.
    let lm_head_raw = if shared_classifier {
        None
    } else {
        Some(read_f32_vec(r, vocab_size * dim)?)
    };

    // Build per-layer LayerWeights by slicing the interleaved per-layer
    // blocks and transposing each linear weight to (in, out).
    let layers: Vec<LayerWeights> = (0..n_layers)
        .map(|i| {
            let wq_layer = &wq_all[i * dim * dim..(i + 1) * dim * dim];
            let wk_layer = &wk_all[i * kv_dim * dim..(i + 1) * kv_dim * dim];
            let wv_layer = &wv_all[i * kv_dim * dim..(i + 1) * kv_dim * dim];
            let wo_layer = &wo_all[i * dim * dim..(i + 1) * dim * dim];
            let w1_layer = &w1_all[i * hidden_dim * dim..(i + 1) * hidden_dim * dim];
            let w2_layer = &w2_all[i * dim * hidden_dim..(i + 1) * dim * hidden_dim];
            let w3_layer = &w3_all[i * hidden_dim * dim..(i + 1) * hidden_dim * dim];

            // Transpose each (out, in) → (in, out) for WeightStorage::apply_linear.
            let attn_q = Arc::from(transpose_rows_cols(wq_layer, dim, dim));
            let attn_k = Arc::from(transpose_rows_cols(wk_layer, kv_dim, dim));
            let attn_v = Arc::from(transpose_rows_cols(wv_layer, kv_dim, dim));
            let attn_o = Arc::from(transpose_rows_cols(wo_layer, dim, dim));
            let ffn_gate = Arc::from(transpose_rows_cols(w1_layer, hidden_dim, dim));
            let ffn_down = Arc::from(transpose_rows_cols(w2_layer, dim, hidden_dim));
            let ffn_up = Arc::from(transpose_rows_cols(w3_layer, hidden_dim, dim));

            let attn_norm_gain: Arc<[f32]> =
                Arc::from(rms_att[i * dim..(i + 1) * dim].to_vec());
            let ffn_norm_gain: Arc<[f32]> =
                Arc::from(rms_ffn[i * dim..(i + 1) * dim].to_vec());

            LayerWeights {
                attn_q: WeightStorage::F32(attn_q),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(attn_k),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(attn_v),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(attn_o),
                ffn_gate: WeightStorage::F32(ffn_gate),
                ffn_up: WeightStorage::F32(ffn_up),
                ffn_down: WeightStorage::F32(ffn_down),
                attn_norm_gain,
                ffn_norm_gain,
            }
        })
        .collect();

    // lm_head: transpose (vocab_size, dim) → (dim, vocab_size).
    let output = match lm_head_raw {
        Some(raw) => WeightStorage::F32(Arc::from(transpose_rows_cols(&raw, vocab_size, dim))),
        None => crate::lazy_llama_full::tied_lm_head_from_embeddings(
            &token_embedding,
            vocab_size,
            dim,
        ),
    };

    let weights = LlamaWeights {
        token_embedding,
        layers,
        final_norm_gain: rms_final,
        output,
    };
    Ok(Llama2cModel { config, weights })
}

/// Convenience: open a path and read a llama2.c bin file.
pub fn load_llama2c_bin_path<P: AsRef<std::path::Path>>(path: P) -> Result<Llama2cModel> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| crate::Error::Msg(format!("llama2c bin open {:?}: {e}", path.as_ref())))?;
    let mut reader = std::io::BufReader::new(file);
    load_llama2c_bin(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Shape;

    #[test]
    fn from_hf_json_str_parses_canonical_tinyllama_fields() {
        // Excerpt from TinyLlama/TinyLlama-1.1B-Chat-v1.0/config.json.
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 2048
        }"#;
        let cfg = Llama2cConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.dim, 2048);
        assert_eq!(cfg.hidden_dim, 5632);
        assert_eq!(cfg.n_layers, 22);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 4);  // GQA model
        assert_eq!(cfg.head_dim, 64);   // 2048 / 32 default
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-9);
    }

    #[test]
    fn from_hf_json_str_applies_optional_defaults() {
        // Minimal Llama-shape config (Llama-2-7B style — no GQA,
        // no head_dim override, no rope_theta override).
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;
        let cfg = Llama2cConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.n_kv_heads, 32);  // defaults to n_heads (no GQA)
        assert_eq!(cfg.head_dim, 128);   // 4096 / 32
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-9);
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        // Verifies that `forward_embeds` (LLaVA-style multimodal entry)
        // produces the same logits as `forward` when the embeddings
        // input is the token embedding table indexed by the same
        // tokens — i.e. the multimodal path is a strict superset of
        // the token-lookup path, with the lookup factored out.
        let cfg = Llama2cConfig {
            dim: 16, hidden_dim: 32, n_layers: 2,
            n_heads: 4, n_kv_heads: 2, vocab_size: 32,
            head_dim: 4, norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let mut s: u32 = 24680;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim; let i = cfg.hidden_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let weights = LlamaWeights {
            token_embedding: Arc::clone(&token_embedding),
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)),
        };
        let model = Llama2cModel { config: cfg.clone(), weights };

        let tokens: Vec<u32> = vec![5, 10, 15];

        // Path A: forward(tokens, 0) — the standard path that does
        // an internal token-embedding lookup.
        let logits_a = model.forward(&tokens, 0).unwrap().realize_f32();

        // Path B: pre-compute the embeddings and call forward_embeds.
        let embeds = LazyTensor::embed_tokens(
            Arc::clone(&token_embedding), cfg.vocab_size, cfg.dim,
            &tokens, &crate::Device::cpu(),
        ).unwrap();
        let logits_b = model.forward_embeds(&embeds, 0).unwrap().realize_f32();

        assert_eq!(logits_a.len(), logits_b.len());
        for (i, (a, b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "logit[{i}]: forward={a} vs forward_embeds={b}",
            );
        }
    }

    #[test]
    fn forward_with_kv_context_returns_matching_logits() {
        // Same model, run forward twice: once via Llama2cModel::forward
        // (no cache), once via forward_with_kv_context (with cache).
        // For the prefill (first call on a fresh cache) the two paths
        // should agree to within float tolerance.
        let cfg = Llama2cConfig {
            dim: 16, hidden_dim: 32, n_layers: 2,
            n_heads: 4, n_kv_heads: 2, vocab_size: 32,
            head_dim: 4, norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let mut s: u32 = 13579;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim; let i = cfg.hidden_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let weights = LlamaWeights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)),
        };
        let model = Llama2cModel { config: cfg.clone(), weights };

        let tokens: Vec<u32> = vec![5, 10, 15];
        let logits_nocache = {
            let l = model.forward(&tokens, 0).unwrap();
            // Pull the last position's logits to match what
            // forward_with_kv_context returns.
            let last = l
                .slice(1_usize, tokens.len() - 1, 1).unwrap()
                .reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap();
            last.realize_f32()
        };

        let device = crate::Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
            tokens.len(), crate::DType::F32, &device,
        ).unwrap();
        let mut ctx = InferenceContext::new(device.clone());
        let logits_cached = model.forward_with_kv_context(
            &tokens, &mut cache, &mut ctx,
        ).unwrap();

        assert_eq!(logits_nocache.len(), logits_cached.len());
        for (i, (a, b)) in logits_nocache.iter().zip(logits_cached.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "logit[{i}] differs: nocache={a} cached={b}",
            );
        }
    }

    /// Greedy spec decode through the Llama2c delegate must equal
    /// plain greedy generation (greedy spec decode is lossless for
    /// any draft). Uses the model as its own draft — the underlying
    /// rollback machinery is covered by the LlamaModel-level
    /// divergent-draft test.
    #[test]
    fn generate_streaming_spec_with_kv_context_matches_greedy() {
        let cfg = Llama2cConfig {
            dim: 16, hidden_dim: 32, n_layers: 2,
            n_heads: 4, n_kv_heads: 2, vocab_size: 32,
            head_dim: 4, norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let mut s: u32 = 86420;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim; let i = cfg.hidden_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let weights = LlamaWeights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)),
        };
        let model = Llama2cModel { config: cfg.clone(), weights };

        let prompt = [2_u32, 9, 4];
        let max_new = 6;
        let device = crate::Device::cpu();

        let baseline = model.generate_with_kv_context(
            &prompt, max_new, SamplingStrategy::Greedy, None,
            &device, crate::DType::F32,
        ).expect("baseline");

        let spec = model.generate_streaming_spec_with_kv_context(
            &model, &prompt, max_new, 2,
            SamplingStrategy::Greedy, None,
            &device, crate::DType::F32, |_| {},
        ).expect("spec");

        assert_eq!(spec, baseline);
    }

    #[test]
    fn from_hf_json_str_round_trips_through_to_llama_config() {
        // After from_hf_json_str() + to_llama_config(), the LlamaConfig
        // should match what the inline `LlamaConfig::from_hf_json_str`
        // would produce. (Documents the field-rename adapter.)
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4
        }"#;
        let llama2c_cfg = Llama2cConfig::from_hf_json_str(json).unwrap();
        let llama_cfg = llama2c_cfg.to_llama_config();
        let direct_llama_cfg = LlamaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(llama_cfg.vocab_size, direct_llama_cfg.vocab_size);
        assert_eq!(llama_cfg.dim, direct_llama_cfg.dim);
        assert_eq!(llama_cfg.n_layers, direct_llama_cfg.n_layers);
        assert_eq!(llama_cfg.n_heads, direct_llama_cfg.n_heads);
        assert_eq!(llama_cfg.n_kv_heads, direct_llama_cfg.n_kv_heads);
        assert_eq!(llama_cfg.head_dim, direct_llama_cfg.head_dim);
        assert_eq!(llama_cfg.ffn_dim, direct_llama_cfg.ffn_dim);
        assert!((llama_cfg.norm_eps - direct_llama_cfg.norm_eps).abs() < 1e-12);
        assert!((llama_cfg.rope_base - direct_llama_cfg.rope_base).abs() < 1e-9);
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Llama2cConfig {
            dim: 16, hidden_dim: 32, n_layers: 2,
            n_heads: 4, n_kv_heads: 2, vocab_size: 32,
            head_dim: 4, norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let mut s: u32 = 99999;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim; let i = cfg.hidden_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let weights = LlamaWeights {
            token_embedding,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)),
        };
        let _ = Shape::from_dims(&[1, 3, cfg.vocab_size]); // unused; included for future debug.
        let model = Llama2cModel { config: cfg.clone(), weights };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn config_field_mapping_matches_llama_config() {
        let cfg = Llama2cConfig::from_dim(64, 128, 4, 8, 2, 256);
        let l = cfg.to_llama_config();
        assert_eq!(l.dim, 64);
        assert_eq!(l.ffn_dim, 128);
        assert_eq!(l.n_layers, 4);
        assert_eq!(l.n_heads, 8);
        assert_eq!(l.n_kv_heads, 2);
        assert_eq!(l.head_dim, 8); // 64 / 8
        assert_eq!(l.vocab_size, 256);
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Llama2cConfig {
            dim: 16, hidden_dim: 32, n_layers: 2,
            n_heads: 4, n_kv_heads: 2, vocab_size: 32,
            head_dim: 4, norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let mut s: u32 = 31415;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.dim; let i = cfg.hidden_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)), attn_q_bias: None,
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_k_bias: None,
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)), attn_v_bias: None,
            attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
            ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_up:   WeightStorage::F32(vec_of(h * i, &mut *nb)),
            ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            attn_norm_gain: Arc::from(vec![1.0_f32; h]),
            ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
        }).collect();
        let weights = LlamaWeights {
            token_embedding, layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            output: WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)),
        };
        let model = Llama2cModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.dim]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    // -----------------------------------------------------------------
    // llama2.c bin format loader tests.
    // -----------------------------------------------------------------

    /// Hand-construct a tiny llama2.c bin blob with deterministic
    /// payloads, then verify `load_llama2c_bin` rebuilds the same
    /// config and produces correctly transposed weights.
    fn build_tiny_bin(
        dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        vocab_size: usize,
        seq_len: usize,
        shared_classifier: bool,
        with_freq_cis: bool,
    ) -> Vec<u8> {
        let head_dim = dim / n_heads;
        let kv_dim = n_kv_heads * head_dim;
        let mut bytes = Vec::new();
        let vocab_signed = if shared_classifier {
            vocab_size as i32
        } else {
            -(vocab_size as i32)
        };
        for v in &[
            dim as i32,
            hidden_dim as i32,
            n_layers as i32,
            n_heads as i32,
            n_kv_heads as i32,
            vocab_signed,
            seq_len as i32,
        ] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Deterministic f32 generator so tests can hand-compute expected values.
        let mut counter: u32 = 0;
        let mut next_f32 = || -> f32 {
            counter = counter.wrapping_add(1);
            (counter as f32) * 0.001
        };
        let push_f32s = |bytes: &mut Vec<u8>, n: usize, src: &mut dyn FnMut() -> f32| {
            for _ in 0..n {
                bytes.extend_from_slice(&src().to_le_bytes());
            }
        };
        push_f32s(&mut bytes, vocab_size * dim, &mut next_f32); // token_embedding
        push_f32s(&mut bytes, n_layers * dim, &mut next_f32); // rms_att
        push_f32s(&mut bytes, n_layers * dim * dim, &mut next_f32); // wq
        push_f32s(&mut bytes, n_layers * kv_dim * dim, &mut next_f32); // wk
        push_f32s(&mut bytes, n_layers * kv_dim * dim, &mut next_f32); // wv
        push_f32s(&mut bytes, n_layers * dim * dim, &mut next_f32); // wo
        push_f32s(&mut bytes, n_layers * dim, &mut next_f32); // rms_ffn
        push_f32s(&mut bytes, n_layers * hidden_dim * dim, &mut next_f32); // w1
        push_f32s(&mut bytes, n_layers * dim * hidden_dim, &mut next_f32); // w2
        push_f32s(&mut bytes, n_layers * hidden_dim * dim, &mut next_f32); // w3
        push_f32s(&mut bytes, dim, &mut next_f32); // rms_final
        if with_freq_cis {
            push_f32s(&mut bytes, seq_len * (head_dim / 2), &mut next_f32);
            push_f32s(&mut bytes, seq_len * (head_dim / 2), &mut next_f32);
        }
        if !shared_classifier {
            push_f32s(&mut bytes, vocab_size * dim, &mut next_f32);
        }
        bytes
    }

    #[test]
    fn load_llama2c_bin_round_trip_no_gqa_tied() {
        let bin = build_tiny_bin(
            /*dim*/ 8, /*hidden_dim*/ 16, /*n_layers*/ 1,
            /*n_heads*/ 2, /*n_kv_heads*/ 2, /*vocab_size*/ 4,
            /*seq_len*/ 4, /*shared_classifier*/ true, /*with_freq_cis*/ true,
        );
        let mut reader = std::io::Cursor::new(&bin);
        let model = load_llama2c_bin(&mut reader).unwrap();
        assert_eq!(model.config.dim, 8);
        assert_eq!(model.config.hidden_dim, 16);
        assert_eq!(model.config.n_layers, 1);
        assert_eq!(model.config.n_heads, 2);
        assert_eq!(model.config.n_kv_heads, 2);
        assert_eq!(model.config.vocab_size, 4);
        assert_eq!(model.config.head_dim, 4);
        assert!((model.config.rope_theta - 10_000.0).abs() < 1e-9);
        // Token embedding stored as-is.
        assert_eq!(model.weights.token_embedding.len(), 4 * 8);
        assert_eq!(model.weights.layers.len(), 1);
        // Q is dim×dim → transposed flat length stays dim*dim = 64.
        let attn_q = match &model.weights.layers[0].attn_q {
            WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32"),
        };
        assert_eq!(attn_q.len(), 8 * 8);
        // FFN gate is (hidden_dim, dim) → transposed (dim, hidden_dim) = 128 floats.
        let ffn_gate = match &model.weights.layers[0].ffn_gate {
            WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32"),
        };
        assert_eq!(ffn_gate.len(), 8 * 16);
        // FFN down is (dim, hidden_dim) → transposed (hidden_dim, dim) = 128 floats.
        let ffn_down = match &model.weights.layers[0].ffn_down {
            WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32"),
        };
        assert_eq!(ffn_down.len(), 16 * 8);
        // Forward should produce finite logits of the expected shape.
        let logits = model.forward(&[0, 1, 2], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, 4]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn load_llama2c_bin_round_trip_gqa_untied() {
        let bin = build_tiny_bin(
            /*dim*/ 8, /*hidden_dim*/ 16, /*n_layers*/ 2,
            /*n_heads*/ 4, /*n_kv_heads*/ 2, /*vocab_size*/ 6,
            /*seq_len*/ 4, /*shared_classifier*/ false, /*with_freq_cis*/ true,
        );
        let mut reader = std::io::Cursor::new(&bin);
        let model = load_llama2c_bin(&mut reader).unwrap();
        assert_eq!(model.config.n_kv_heads, 2);
        assert_eq!(model.config.head_dim, 2); // 8 / 4
        // GQA: wk/wv are (kv_dim=4, dim=8) → transposed 4*8 = 32 floats per layer.
        let attn_k = match &model.weights.layers[0].attn_k {
            WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32"),
        };
        assert_eq!(attn_k.len(), 4 * 8);
        // Untied lm_head: output is read from the file (not tied to embedding).
        // We can't directly check distinctness without a deterministic byte
        // pattern match — instead just verify it has the right size.
        let lm = match &model.weights.output {
            WeightStorage::F32(a) => a.clone(),
            _ => panic!("expected F32"),
        };
        assert_eq!(lm.len(), 8 * 6); // (dim, vocab_size) transposed
        // Forward smoke test.
        let logits = model.forward(&[0, 1], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 2, 6]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn load_llama2c_bin_rejects_invalid_head_dim() {
        // dim=7, n_heads=2 → 7 % 2 != 0.
        let mut bytes = Vec::new();
        for v in &[7_i32, 16, 1, 2, 2, 4, 4] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut reader = std::io::Cursor::new(&bytes);
        let err = load_llama2c_bin(&mut reader).err().expect("should reject");
        let msg = format!("{err}");
        assert!(msg.contains("dim=7"), "unexpected error: {msg}");
    }

    #[test]
    fn load_llama2c_bin_rejects_truncated() {
        // Only 5 of the 7 header i32s present.
        let mut bytes = Vec::new();
        for v in &[8_i32, 16, 1, 2, 2] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut reader = std::io::Cursor::new(&bytes);
        assert!(load_llama2c_bin(&mut reader).is_err());
    }

    /// Spot-check the transpose: for a 2×3 source `[[1,2,3],[4,5,6]]`
    /// stored row-major, the transpose `(3, 2)` = `[[1,4],[2,5],[3,6]]`.
    #[test]
    fn transpose_rows_cols_correctness() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_rows_cols(&src, 2, 3);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
