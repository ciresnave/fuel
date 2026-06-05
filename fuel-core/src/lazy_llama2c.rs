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
use crate::lazy::{KVCache, LlamaConfig, LlamaModel, LlamaWeights, LazyTensor, SamplingStrategy};
use crate::{DType, Device, Result};
use fuel_graph_executor::{GraphBackend, GraphExecutor};

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

    /// Device-resident KV-aware forward, backend-generic. Keys and
    /// values stay on the device that owns `B::Storage` across decode
    /// steps — no D2H/H2D round-trip per token. Used by `llama-lazy-
    /// cuda` and `llama-lazy-vulkan` binaries.
    ///
    /// Delegates to [`LlamaModel::forward_with_cache_gpu_on`].
    pub fn forward_with_cache_gpu_on<B: GraphBackend>(
        &self,
        tokens: &[u32],
        cache: &mut KVCache<B>,
        executor: &mut GraphExecutor<B>,
    ) -> Vec<f32> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_with_cache_gpu_on(tokens, cache, executor)
    }

    /// Same as [`Self::forward_with_cache_gpu_on`] but returns logits
    /// for ALL positions (used by speculative decode's accept/reject
    /// path) instead of just the last position.
    pub fn forward_with_cache_gpu_on_all_positions<B: GraphBackend>(
        &self,
        tokens: &[u32],
        cache: &mut KVCache<B>,
        executor: &mut GraphExecutor<B>,
    ) -> Vec<f32> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.forward_with_cache_gpu_on_all_positions(tokens, cache, executor)
    }

    /// Backend-generic streaming decode with device-resident KV cache.
    /// For `B = CudaBackend` / `VulkanBackend` / future GPU backends
    /// the K/V bytes never leave the device. Collapses to a
    /// host-resident cache for `B = CpuBackend`.
    ///
    /// Delegates to [`LlamaModel::generate_streaming_gpu_on`].
    pub fn generate_streaming_gpu_on<B: GraphBackend>(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        executor: &mut GraphExecutor<B>,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.generate_streaming_gpu_on(
            prompt_tokens, max_new_tokens, strategy, eos_id, executor, on_token,
        )
    }

    /// CUDA-specific convenience wrapper around
    /// [`Self::generate_streaming_gpu_on`] for callers that already
    /// hold a `GraphExecutor<CudaBackend>`. Delegates to
    /// [`LlamaModel::generate_streaming_cuda`].
    #[cfg(feature = "cuda")]
    pub fn generate_streaming_cuda(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        executor: &mut GraphExecutor<fuel_cuda_backend::CudaBackend>,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let llama = LlamaModel {
            config: self.config.to_llama_config(),
            weights: self.weights.clone(),
        };
        llama.generate_streaming_cuda(
            prompt_tokens, max_new_tokens, strategy, eos_id, executor, on_token,
        )
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{LayerWeights, WeightStorage};
    use fuel_core_types::Shape;
    use std::sync::Arc;

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
}
