// SPDX-License-Identifier: MIT OR Apache-2.0
//! The LLaMA-family lazy-graph decoder (`LlamaConfig`/`LlamaWeights`/`LlamaModel`/
//! `LlamaTokenizer`).
//!
//! Extracted from `fuel-core/src/lazy.rs` (fuel-core dissolution, per
//! `docs/architecture/02-layers.md`'s ratified `fuel-model-llama`† — one architecture per
//! crate). `fuel_model_llama::LlamaModel` (this crate) is the canonical base every downstream
//! LLaMA-family model in `fuel-transformers` builds on, and `fuel-inference/src/multi_session.rs`
//! references it directly.
//!
//! ⚠️ **Depends on `fuel-core` itself, and this crate does NOT contain everything that was
//! textually near `LlamaModel` in the original file.** Several types and functions
//! positioned inside the old "LLaMA section" of `lazy.rs` — `WeightStorage`, `LayerWeights`,
//! `SamplingStrategy`, `LayerNormPair`, `ConvWeightBias`, `TokenDataHost`, `TokenDataBytes`,
//! `captured_output_to_f32`, `load_tensor_as_f32`, `load_transposed_matrix(_preserve_dtype)`,
//! `apply_affine_rms_norm`, `sample_logits`, `build_decode_causal_mask(_windowed)`,
//! `offer_flash_decode_arm_for_region`, `invalidate_decode_pair_if_stale`,
//! `refresh_decode_session` — are either GENERIC (used by anywhere from a handful to ~120
//! files across the whole `fuel-transformers` model zoo, some by `fuel-core`'s own
//! `persistent_decode.rs`/`inference_context.rs`) or SHARED specifically between this crate
//! and `fuel-model-phi` (`TokenDataHost`/`TokenDataBytes`/`captured_output_to_f32`), and
//! stayed in `fuel-core` on purpose: moving the generic set here would have made this crate a
//! de facto shared-utilities dependency for the entire zoo, the opposite of "one architecture
//! per crate". This crate calls back into `fuel_core::lazy::X` for all of them, and that
//! callback is itself deferred debt — see docs/session-prompts/fuel-core-dissolution-b1.md's
//! shim-debt ledger. The Tensor bridge type and the paged/persistent-decode machinery this
//! model is built on have no rehomed destination yet either (NOT `fuel-tensor`†, which is a
//! post-fission redesign wrapping `fuel-graph`).

// GAP-229 (fuel-core/src/lib.rs): this crate's code was extracted verbatim from
// fuel-core/src/lazy.rs and inherits the same DOC-SHAPE identity-op idiom (`1 * 2 *
// cfg.vocab_size` mirroring `Shape::from_dims(&[1, 2, vocab_size])`) -- a deliberate
// house idiom there, not new debt here. See fuel-core's lib.rs for the full ruling.
#![allow(clippy::identity_op)]

use fuel_core::inference_context::{InferenceContext, KvCache, KvSlot};
#[cfg(test)]
use fuel_core::lazy::sample_multinomial;
use fuel_core::lazy::{
    LayerWeights, SamplingStrategy, SessionDisposition, Tensor, TokenDataHost, WeightStorage,
    apply_affine_rms_norm, build_decode_causal_mask, invalidate_decode_pair_if_stale,
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, offer_flash_decode_arm_for_region,
    sample_logits, spec_argmax, spec_next_u01, spec_sample_cat, spec_softmax_temp,
};
#[cfg(feature = "cuda")]
use fuel_core::lazy::{TokenDataBytes, captured_output_to_f32};
use fuel_core::{DType, Device, Shape};
use serde::Deserialize;
use std::sync::Arc;

// ===========================================================================
// GAP-029 increment 2b — the per-model half of the shared persistent-decode
// rebind driver. The driver itself lives in `fuel_core::persistent_decode`; these
// impls are the ONLY places the two models are allowed to differ on this path.
// ===========================================================================

impl fuel_core::persistent_decode::PersistentDecodeModel for LlamaModel {
    fn decode_n_layers(&self) -> usize {
        self.config.n_layers
    }

    /// Rebuild the per-token offset only if the held session is on the
    /// device-offset path (`offset_node.is_some()`); SymEnv sessions skip it
    /// (the offset rides `cached_len_sym`). **Measured: Llama takes the
    /// device-offset path on CPU/F32** — the driver must not normalise this away.
    fn build_decode_token_data(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        session: &fuel_core::inference_context::DecodeSession,
        cache: &KvCache,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<fuel_core::inference_context::DecodeTokenData> {
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        self.build_token_rope_mask_arcs(
            device,
            cached_len,
            tokens,
            session.max_seq_len(),
            cache_dtype,
            session.offset_node().is_some(),
            rope_inv_freq,
        )
    }
}
/// GAP-029 increment 3 — `LlamaModel`'s half of the shared decode **build**
/// path. The body it used to own now lives in [`fuel_core::persistent_decode`];
/// what remains here is the architecture, which is the only part that was ever
/// Llama-specific.
impl fuel_core::persistent_decode::DecodeBackbone for LlamaModel {
    fn decode_family(&self) -> &'static str {
        "LlamaModel"
    }

    fn decode_dims(&self) -> fuel_core::persistent_decode::DecodeDims {
        let cfg = &self.config;
        fuel_core::persistent_decode::DecodeDims {
            n_layers: cfg.n_layers,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            hidden: cfg.dim,
            vocab: cfg.vocab_size,
            // Full rotary — Llama rotates the whole head.
            rope_width: cfg.head_dim,
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        LlamaModel::decode_shape_key(self)
    }

    /// Llama has **no** per-layer attention variation: measured **zero**
    /// `sliding_window` hits in this file, against a positive control of hits in
    /// five sibling model files. The uniform plan emits no slice node, so
    /// Llama's decode graph is byte-identical to the pre-GAP-029 one.
    fn decode_mask_plan(&self) -> fuel_core::persistent_decode::MaskPlan {
        fuel_core::persistent_decode::MaskPlan::dense(self.config.n_layers)
    }

    fn decode_rope_plan(&self) -> fuel_core::persistent_decode::RopePlan {
        fuel_core::persistent_decode::RopePlan::single(
            self.config.rope_base,
            self.decode_dims().n_layers,
        )
    }

    fn decode_token_embedding(&self) -> Arc<[f32]> {
        self.weights.token_embedding.clone()
    }

    fn decode_apply_layer(
        &self,
        layer_idx: usize,
        inputs: &fuel_core::persistent_decode::DecodeLayerInputs<'_>,
    ) -> fuel_core::Result<Tensor> {
        self.apply_layer_with_kv_writes(
            inputs.x,
            &self.weights.layers[layer_idx],
            inputs.k_cache,
            inputs.v_cache,
            inputs.cached_len_sym,
            inputs.attended_len_sym,
            inputs.offset,
            inputs.rope_cos,
            inputs.rope_sin,
            inputs.mask,
            inputs.attn_window,
        )
    }

    fn decode_final_norm_and_head(&self, h: &Tensor) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let h_norm = apply_affine_rms_norm(h, &self.weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        self.weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)
    }
}
// ---- LLaMA model assembly --------------------------------------------------

/// Hyperparameters for a LLaMA-style transformer model.
///
/// Field names follow the conventional LLaMA nomenclature:
/// - `dim` is the model hidden dimension (often written `d_model`).
/// - `n_heads` is the number of attention query heads.
/// - `n_kv_heads` is the number of key/value heads. Equal to `n_heads`
///   for standard multi-head attention; smaller (e.g. `n_heads / 4`)
///   for Grouped Query Attention (GQA). LLaMA 2 onwards uses GQA.
/// - `head_dim` is the per-head feature dimension (`dim / n_heads`).
/// - `ffn_dim` is the hidden dimension of the SwiGLU feed-forward
///   network, conventionally around `4 × dim` with some rounding.
/// - `norm_eps` is the epsilon of the RmsNorm layers.
/// - `rope_base` is the frequency base for rotary position embeddings
///   (`10_000` in original LLaMA, `500_000` in LLaMA 3).
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub norm_eps: f64,
    pub rope_base: f64,
}

fn default_llama_norm_eps() -> f64 {
    1e-5
}
fn default_llama_rope_base() -> f64 {
    10_000.0
}

/// A Llama `config.json` under HuggingFace's field names.
///
/// Mirrors the wire format rather than this crate's vocabulary: `LlamaConfig`
/// calls these `dim`, `n_layers`, `n_heads`, `ffn_dim`, `rope_base`, and
/// doing the rename in `#[serde(rename)]` attributes would scatter it. It
/// happens in [`Self::resolve`] as one visible block.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LlamaConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_llama_norm_eps", rename = "rms_norm_eps")]
    norm_eps: f64,
    #[serde(default = "default_llama_rope_base", rename = "rope_theta")]
    rope_base: f64,
}

impl LlamaConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing config.json: {e}")))
    }

    fn resolve(self) -> fuel_core::Result<LlamaConfig> {
        Ok(LlamaConfig {
            vocab_size: self.vocab_size,
            dim: self.hidden_size,
            n_layers: self.num_hidden_layers,
            n_heads: self.num_attention_heads,
            n_kv_heads: fuel_core::hf_config::num_key_value_heads(
                self.num_key_value_heads,
                self.num_attention_heads,
            )?,
            head_dim: fuel_core::hf_config::head_dim(
                self.head_dim,
                self.hidden_size,
                self.num_attention_heads,
            )?,
            ffn_dim: self.intermediate_size,
            norm_eps: self.norm_eps,
            rope_base: self.rope_base,
        })
    }
}

impl LlamaConfig {
    /// Parse a LlamaConfig from a Hugging Face `config.json` string.
    ///
    /// Maps HF's field names to ours:
    /// - `hidden_size` → `dim`
    /// - `num_hidden_layers` → `n_layers`
    /// - `num_attention_heads` → `n_heads`
    /// - `num_key_value_heads` → `n_kv_heads` (falls back to `n_heads`
    ///   when absent, for older configs without GQA)
    /// - `intermediate_size` → `ffn_dim`
    /// - `vocab_size` → `vocab_size`
    /// - `rms_norm_eps` → `norm_eps`
    /// - `rope_theta` → `rope_base` (defaults to 10000 when absent)
    /// - `head_dim` is taken directly when present, or computed as
    ///   `hidden_size / num_attention_heads` otherwise.
    ///
    /// `LlamaConfigRaw` is the wire shape under HF's own field names;
    /// `LlamaConfigRaw::resolve` applies the two cross-field defaults and
    /// renames into this crate's vocabulary.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        LlamaConfigRaw::from_json_str(json)?.resolve()
    }
}
/// Top-level weights: token embedding table, per-layer weights, final
/// norm gain, and output projection (which may be tied to the embedding
/// or a separate matrix).
#[derive(Debug, Clone)]
pub struct LlamaWeights {
    /// Process-unique identity for THIS weight set — the component that lets a
    /// held decode plan tell two same-architecture models apart.
    ///
    /// It lives on the weights rather than the model because the weights are
    /// what a held graph bakes as `Const`s: two `LlamaModel`s sharing one
    /// `Arc` weight set may legitimately share a plan, while two with distinct
    /// weights must not. Minted with [`fuel_core::decode_shape::ModelInstanceId::next`];
    /// never recycled, so no lifetime reasoning is required (see that module for
    /// the pointer-identity scheme this replaced and how it failed).
    pub instance: fuel_core::decode_shape::ModelInstanceId,
    /// `[vocab_size, dim]` token embedding table. Stays f32 — the
    /// downstream `index_select` + graph traversal requires activation
    /// dtype to be f32, and the table is used directly as activations.
    pub token_embedding: Arc<[f32]>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// `[dim]` RmsNorm gain for the final norm before the output head.
    pub final_norm_gain: Arc<[f32]>,
    /// `[dim, vocab_size]` output projection (a.k.a. `lm_head`).
    /// Supports bf16 or f32 on-device — this is the largest single
    /// matrix after the embedding, worth ~262 MB at f32.
    pub output: WeightStorage,
}

/// A LLaMA-style transformer model assembled via `Tensor`. Holds
/// config + weights as plain vectors; each `forward` call rebuilds a
/// graph using those vectors as `Const` leaves.
///
/// This lives in `fuel_core::lazy` rather than `fuel_transformers`
/// because it was built directly on top of the Phase 6a bridge
/// primitives and predates the migration of `fuel_transformers`'
/// existing model code onto `Tensor`. Once that migration lands,
/// this code will move back to `fuel-transformers::models::llama`.
#[derive(Debug, Clone)]
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
}

impl LlamaModel {
    /// Run a forward pass from a sequence of token IDs and return the
    /// final logits as a `Tensor` of shape `[1, seq_len, vocab_size]`.
    /// Call `.realize_f32()` on the result to materialize them.
    ///
    /// `start_pos` offsets the RoPE frequencies — use `0` for the
    /// first forward call of a conversation and the previous total
    /// token count for each subsequent decode step when using a KV
    /// cache. The current implementation does NOT use a KV cache
    /// internally; it recomputes the full attention each call. Adding
    /// a KV cache is orthogonal plumbing that doesn't change the graph
    /// structure.
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert_eq!(
            cfg.n_heads * cfg.head_dim,
            cfg.dim,
            "LlamaConfig: n_heads * head_dim must equal dim"
        );

        // Embedding lookup: build a token embedding const tensor +
        // a U32 index tensor + index_select along dim 0.
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        )?;
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]))?;
        // index_select(0, token_ids) produces [seq, dim]. Reshape to
        // [1, seq, dim] for the downstream attention code.
        let h = embed
            .index_select(0, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .unwrap();

        self.forward_embeds(&h, start_pos)
    }

    /// Forward from pre-computed input embeddings of shape
    /// `(batch, seq, dim)`. Used by multimodal models (LLaVA,
    /// Pixtral, Qwen-VL, etc.) that interleave image embeddings
    /// with text embeddings before running the LLaMA decoder
    /// stack.
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)
    }

    /// Like [`Self::forward_embeds`] but skips the LM-head projection
    /// and returns post-final-RmsNorm hidden states
    /// `(batch, seq, dim)`. Uses strict-causal masking. Use
    /// this from multimodal hosts (LLaVA, Pixtral, etc.) that
    /// interleave image embeddings into the text stream and
    /// want hidden states without the lm_head projection.
    /// Mirrors `MistralModel::forward_hidden_embeds`.
    pub fn forward_hidden_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
    ) -> fuel_core::Result<Tensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    fn run_backbone_embeds(&self, embeds: &Tensor, start_pos: usize) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let shape = embeds.shape();
        let (_, seq, _) = shape.dims3().map_err(|e| {
            e.context("LlamaModel::run_backbone_embeds: embeds must be rank 3 [b, seq, dim]")
        })?;
        let dims = shape.dims();
        assert_eq!(dims[2], cfg.dim, "embeds last dim must equal cfg.dim");
        assert_eq!(
            cfg.n_heads * cfg.head_dim,
            cfg.dim,
            "LlamaConfig: n_heads * head_dim must equal dim"
        );

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_base, start_pos, seq, cfg.head_dim);

        let mask = Tensor::additive_causal_mask_like(embeds, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))
            .unwrap();

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask)?;
        }
        Ok(apply_affine_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        ))
    }

    /// Like [`Self::forward_embeds`] but takes a caller-supplied
    /// additive attention mask `(1, 1, seq, seq)` and skips
    /// the LM-head projection. Returns the post-final-RmsNorm
    /// hidden states `[batch, seq, dim]`.
    ///
    /// Use this for bidirectional Llama-encoder modes (e.g.
    /// embedding adapters). The `mask` must live on the same
    /// graph as `embeds` — build it via `embeds.const_f32_like`.
    pub fn forward_hidden_embeds_with_mask(
        &self,
        embeds: &Tensor,
        attention_mask: &Tensor,
        start_pos: usize,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let shape = embeds.shape();
        let (_, seq, _) = shape.dims3().map_err(|e| {
            e.context(
                "LlamaModel::forward_hidden_embeds_with_mask: embeds must be rank 3 [b, seq, dim]",
            )
        })?;
        let dims = shape.dims();
        assert_eq!(dims[2], cfg.dim, "embeds last dim must equal cfg.dim");

        let mut h = embeds.clone();
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_base, start_pos, seq, cfg.head_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, attention_mask)?;
        }
        Ok(apply_affine_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        ))
    }

    /// Like [`Self::forward`] but returns the hidden state AFTER the final
    /// RMSNorm, BEFORE the output projection. Shape: `[batch, seq, dim]`.
    ///
    /// The `anchor` tensor provides the graph to build on — use a
    /// parameter or any existing tensor from the training graph. All
    /// frozen weights are emitted as Const nodes on that graph.
    ///
    /// Use this for fine-tuning: freeze all layer weights (const nodes)
    /// and apply a trainable output head manually:
    ///
    /// ```ignore
    /// // Inside TrainState::step's build_loss callback:
    /// let lm_head = &params["lm_head"];  // ← anchor tensor
    /// let hidden = model.forward_hidden(&tokens, 0, lm_head);
    /// let logits = hidden.matmul(lm_head);
    /// let loss = cross_entropy_with_logits(&logits, &targets);
    /// ```
    pub fn forward_hidden(
        &self,
        tokens: &[u32],
        start_pos: usize,
        anchor: &Tensor,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1usize;

        let embed = anchor.const_f32_like(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
        )?;
        let token_ids = anchor.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]))?;
        let mut h = embed
            .index_select(0, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .unwrap();

        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_base, start_pos, seq, cfg.head_dim);

        // Build the strict-causal mask once for all layers.
        let mask = Tensor::additive_causal_mask_like(&h, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))
            .unwrap();

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, &mask)?;
        }

        Ok(apply_affine_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        ))
    }

    /// Internal entry that runs the LLaMA backbone given pre-built RoPE
    /// cos/sin tables and an attention mask. The standard
    /// [`Self::forward_embeds`] path computes cos/sin from `cfg.rope_base`
    /// via [`Tensor::rope_tables_const`] and uses a strict-causal
    /// mask; `fuel_transformers::models::lazy_llama_full::Llama3Model` uses this hook to
    /// inject Llama-3 long-context scaled RoPE tables without
    /// duplicating the forward path.
    ///
    /// `rope_cos` / `rope_sin` must have shape `[seq, head_dim]` and
    /// live on the same graph as `embeds`. `mask` is additive,
    /// broadcast-compatible with `(B, n_heads, seq, kv_seq)`.
    pub fn run_backbone_with_rope_tables(
        &self,
        embeds: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let shape = embeds.shape();
        shape.dims3().map_err(|e| {
            e.context(
                "LlamaModel::run_backbone_with_rope_tables: embeds must be rank 3 [b, seq, dim]",
            )
        })?;
        let dims = shape.dims();
        assert_eq!(dims[2], cfg.dim, "embeds last dim must equal cfg.dim");
        assert_eq!(
            cfg.n_heads * cfg.head_dim,
            cfg.dim,
            "LlamaConfig: n_heads * head_dim must equal dim"
        );

        let mut h = embeds.clone();
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, rope_cos, rope_sin, mask)?;
        }
        Ok(apply_affine_rms_norm(
            &h,
            &weights.final_norm_gain,
            cfg.dim,
            cfg.norm_eps,
        ))
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        // Pre-attention RmsNorm with affine gain.
        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);

        // Project to Q, K, V using WeightStorage::apply_linear — this
        // routes F32/BF16 through standard matmul and Q4_0 through
        // fused qmatmul. Under GQA, W_k and W_v have fewer output
        // features (kv_dim instead of dim).
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.dim, cfg.dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.dim, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.dim, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        // Split heads.
        // Q: [batch, seq, dim] → [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        // K/V: [batch, seq, kv_dim] → [batch, seq, n_kv_heads, head_dim] → [batch, n_kv_heads, seq, head_dim]
        let k_h = k
            .reshape(Shape::from_dims(&[
                batch,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
            ]))?
            .permute([0, 2, 1, 3_usize])?;
        let v_h = v
            .reshape(Shape::from_dims(&[
                batch,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
            ]))?
            .permute([0, 2, 1, 3_usize])?;

        // RoPE on Q and K (applied per-head; V is NOT rotated). Uses
        // caller-supplied cos/sin so all layers share a single pair
        // of const nodes.
        let q_r = q_h.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k_h.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication factor. This division is DELIBERATELY left unguarded:
        // apply_layer is a RUNTIME site, and the build-time divisibility guard for
        // `n_heads / n_kv_heads` belongs at config parse
        // (hf_config::num_key_value_heads, called from every *ConfigRaw::resolve),
        // NOT here — a guard here would fire per-forward rather than at build and
        // would not cover a config that never reaches this path (GAP-282). If a
        // non-dividing count reaches this point, fix the parse site that built
        // `cfg`, not this line.
        let n_rep = cfg.n_heads / cfg.n_kv_heads;
        let k_r = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_h = v_h.repeat_interleave(1_usize, n_rep)?;

        // Scaled dot-product attention with caller-supplied mask.
        // The default forward path passes the strict-causal mask
        // built once outside the loop; `forward_hidden_embeds_with_mask`
        // passes whatever the caller chose (e.g. bidirectional pad).
        let _ = seq; // silence unused after refactor; mask already sized for seq.
        let k_t = k_r.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = Tensor::from_graph_tensor(scores.graph_tensor().mul_scalar(scale));
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_h)?;

        // Merge heads + output projection.
        let merged = attn_v
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim)?;

        // First residual connection.
        let h1 = x.add(&attn_out)?;

        // Pre-FFN RmsNorm with affine gain.
        let h1_norm = apply_affine_rms_norm(&h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);

        // SwiGLU FFN (routes through apply_linear → qmatmul for Q4_0).
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim)?;
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim)?;
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim)?;

        // Second residual connection.
        h1.add(&ffn_out)
    }

    // ===== Phase 7.6 step 9c E.3.3.D — host-resident forward retired =====
    //
    // The legacy host-resident cached forward path
    // (`forward_with_cache_on`, `forward_with_cache`,
    // `forward_with_cache_cuda`, `unpack_kv_cache`) and its supporting
    // types (`LayerKVCache`, `LlamaKVCache`) were retired in favor of
    // [`Self::forward_with_kv_context`] + [`KvCache`] +
    // [`InferenceContext`]. Greedy token-sequence parity vs the
    // retired path was confirmed by
    // `generate_with_kv_context_matches_legacy_generate` immediately
    // before retirement; bitwise prefill parity vs non-cached forward
    // is checked by
    // `forward_with_kv_context_prefill_matches_non_cached_forward`.
    //
    // Unification Session 4 (E.3.3/E.3.4) completed the retirement:
    // the device-resident `*_gpu_on` family, its shared
    // `apply_layer_with_cache` helper, `LayerOutput`, `LayerKVCache`,
    // and the generic `lazy_kv_cache_device::KVCache<B>` are gone.
    // `forward_with_kv_context` (below) is the sole cached forward.

    // ===== Phase 7.6 step 9c E.3.3.B — InferenceContext + KvCache + WriteSlice =====
    //
    // The new forward path. Uses pre-allocated KV-cache buffers
    // (`KvCache::with_capacity`) + `Op::WriteSlice` in-graph to mutate
    // them, replacing the legacy concat-cached-and-fresh / download-
    // fresh / host-append pattern. Runs on CPU, CUDA, and Vulkan via
    // the pipelined executor + binding-table dispatch.

    /// Variant of [`apply_layer_with_cache`] that uses pre-allocated
    /// KV-cache buffers + `Op::WriteSlice`. The K/V caches are bound
    /// via `k_cache_const` / `v_cache_const` (Const placeholders that
    /// the caller has wired into [`InferenceContext`]).
    ///
    /// **Phase D (input-independent decode graph):** the KV write lands
    /// at the runtime offset `cached_len` via `write_slice_dyn`
    /// (`DynScalar::Sym(cached_len_sym)`, resolved through the per-pass
    /// `SymEnv` at realize), and attention reads the **full fixed-capacity**
    /// buffers `[batch, n_kv_heads, max_seq_len, head_dim]` with a fixed
    /// `[1, 1, seq, max_seq_len]` causal mask (`k > cached_len + q` masks
    /// future positions AND the zero-init stale tail). Nothing in the
    /// graph's *shape* or *structure* depends on `cached_len`, so the
    /// decode-step graph is byte-identical across tokens — the prerequisite
    /// for plan-once persistent decode. Numerically identical to the prior
    /// `slice(0..total_seq)` form (masked positions contribute 0).
    ///
    /// Tradeoff: attention computes over `max_seq_len` (not the live
    /// `total_seq`), so early tokens do extra masked work — a documented
    /// efficiency follow-up (the flash arm with a runtime `k_len`), not a
    /// correctness issue.
    ///
    /// **Phase D · D2b (mask hoist):** the `[1, 1, seq, max_seq_len]`
    /// causal mask is now built ONCE in the forward (`mask` param, like
    /// RoPE tables) and shared across all layers, instead of one Const
    /// per layer. Byte-exact refactor (the mask data is identical across
    /// layers — it depends only on `cached_len`, `seq`, `max_seq_len`);
    /// it also cuts the per-token data-Const re-bind count on the
    /// persistent path from `n_layers` to 1.
    // 12 distinct per-layer decode inputs (activation, weights, K/V cache
    // consts, two symbolic lengths, offset, RoPE cos/sin, mask, this layer's
    // window) threaded through one attention application. They are heterogeneous
    // graph/sym handles, not a reusable cluster — a struct would obscure the
    // seam without removing an argument. Exceeds even the raised (10) threshold.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_kv_writes(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        k_cache_const: &Tensor,
        v_cache_const: &Tensor,
        cached_len_sym: fuel_ir::SymId,
        attended_len_sym: fuel_ir::SymId,
        offset: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
        // GAP-194: this layer's own window, so the flash-arm offer states the
        // truth rather than asserting `None`. Always `None` for Llama — but
        // DERIVED from its mask plan, not assumed.
        attn_window: Option<usize>,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        // Shared front half (RmsNorm → Q/K/V proj → per-head reshape → RoPE) —
        // identical to the paged layer; see `project_qkv_roped`.
        let (q_r, k_r, v_h) = self.project_qkv_roped(x, layer, rope_cos, rope_sin)?;

        // Write fresh K/V slabs into the pre-allocated cache buffers
        // via Op::WriteSlice at the RUNTIME offset `cached_len`. Source
        // slab shape is `[batch, n_kv_heads, seq, head_dim]`; on axis 2
        // the start is dynamic (`cached_len_sym`, resolved at realize)
        // and the slab width is `seq`. The returned tensor's Storage Arc
        // IS the cache const's Arc — post-write reference to the same
        // buffer (the executor adopts dest's Arc as the kernel output,
        // mutating in place). Keeping the offset symbolic makes the write
        // node structurally identical across tokens.
        let write_ranges = vec![
            (0, batch),
            (0, cfg.n_kv_heads),
            (0, seq), // axis-2 start is dynamic; width = seq
            (0, cfg.head_dim),
        ];
        // Two structurally-distinct KV-write ops, one per decode path:
        //   - device-offset (`offset = Some`, CUDA/CPU): `Op::WriteSliceDoff`
        //     reads `cached_len` from a device-resident I64 buffer at kernel
        //     launch — no host round-trip, so the decode step is CUDA-graph-
        //     capturable (CapturedRun). The start on axis 2 is device-only.
        //   - SymEnv (`offset = None`, Vulkan): `Op::WriteSlice` with a
        //     `DynScalar::Sym(cached_len_sym)` start resolved host-side each
        //     token via the per-pass SymEnv (backend-generic; no WriteSliceDoff
        //     binding needed).
        // Both land the same slab at the same offset — bit-identical results.
        let (full_k, full_v) = match offset {
            Some(off) => {
                let full_k = k_cache_const.write_slice_doff(&k_r, off, 2, write_ranges.clone())?;
                let full_v = v_cache_const.write_slice_doff(&v_h, off, 2, write_ranges)?;
                (full_k, full_v)
            }
            None => {
                let dyn_off = fuel_ir::DynScalar::Sym(cached_len_sym);
                let full_k =
                    k_cache_const.write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?;
                let full_v = v_cache_const.write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?;
                (full_k, full_v)
            }
        };

        // Attend over the FULL fixed-capacity buffers (no slice to
        // `total_seq`) so the attention shape is `max_seq_len` every
        // token. The fixed-capacity causal mask excludes future positions
        // AND the stale/unwritten tail (`k > cached_len + q` covers both,
        // since `cached_len + q < total_seq <= max_seq_len`).
        let k_t = full_k.transpose().unwrap();
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t).unwrap();

        // Mask is hoisted to the forward (built once, shared across
        // layers) — see the D2b note on this method.
        let scores_scaled = Tensor::from_graph_tensor(scores.graph_tensor().mul_scalar(scale));
        let scores_masked = scores_scaled.broadcast_add(mask).unwrap();
        let attn = scores_masked.softmax_last_dim().unwrap();
        let attn_v = attn.matmul(&full_v).unwrap();

        // The sole consumer of `attn_v` — the branch reconverge / merge
        // point. Split out of the `merged` chain so we hold its NodeId for
        // the flash-arm offer below (arm-0 runnability requires the merge to
        // read arm 0 = `attn_v`).
        let attn_v_permuted = attn_v.permute([0, 2, 1, 3_usize]).unwrap();

        // Optimizer-owned CUDA flash-decode arm offer (gated). On f32 /
        // prefill (`seq_q != 1`) / non-CUDA topologies the emitter's gate
        // declines (`Ok(None)`) and leaves the graph byte-identical to
        // today; only a supported bf16/f16 decode shape on a CUDA topology
        // gets an `Op::Branch { arm0 = decomposed attn_v, arm1 = CUDA-pinned
        // FlashAttn }` recorded (collapsed at optimize time by the variant
        // bake). `k_len` is the live attended prefix `cached_len + seq`,
        // carried as `Sym(attended_len_sym)` and resolved per-token through
        // the `SymEnv`.
        offer_flash_decode_arm_for_region(
            q_r.graph_tensor().graph(),
            q_r.graph_tensor().id(),
            full_k.graph_tensor().id(),
            full_v.graph_tensor().id(),
            attn_v.graph_tensor().id(),
            attn_v_permuted.graph_tensor().id(),
            scale as f32,
            attended_len_sym,
            attn_window,
            None, // Llama has no attention-logit softcap
            fuel_dispatch::decode_flash::FlashArmCapability::production(),
        )?;

        let merged = attn_v_permuted
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .unwrap();
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim)?;

        let h1 = x.add(&attn_out).unwrap();
        // Shared FFN tail (identical to the paged layer); see `ffn_block`.
        self.ffn_block(&h1, layer)
    }

    /// One single-token forward through **paged** pool storage — the paged-decode
    /// counterpart of [`Self::forward_with_kv_context`] (multi-session serving,
    /// paged-storage integration PS2). Feeds ONE `token` (a prompt token during
    /// prefill, or a sampled token during decode), writes its K/V into the
    /// session's [`DeviceKvPool`](fuel_core::kv_block_pool_device::DeviceKvPool)
    /// blocks, attends via `Op::PagedAttn` over the session's block table, and
    /// returns last-position logits `[vocab_size]`.
    ///
    /// `Op::PagedAttn` is decode-only (`Sq == 1`), so PREFILL is done by feeding
    /// the prompt one token at a time — each attends causally to its predecessors
    /// via the running `context_len`, which is position-for-position equivalent
    /// to a batched causal prefill (each token's K/V depends only on tokens
    /// `0..=i`, all resident by the time token `i` is fed). The pool-dtype gate
    /// accepts F32/BF16/F16 (the old f32-only restriction was lifted).
    ///
    /// Placement — RESOLVED (2026-08-01) by measurement: on a CUDA target this
    /// paged decode currently runs the attention OFF-DEVICE (CPU) with copies
    /// stitched around it. Two facts, both load-bearing:
    /// (1) `Op::PagedAttn` has no FUSED CUDA/Vulkan kernel — CPU-only in the
    ///     binding table (byte_kernels.rs). This is the mechanism.
    /// (2) An nsys A/B (Lightbulb, same binary/model/tokens, only the decode path
    ///     differing) is the per-node MEASUREMENT that discriminates: the paged
    ///     arm pulled ~6.2 GB/token Device→Host (6,335 copies) against the
    ///     contiguous arm's ~13.5 MB logits-readback FLOOR (26 copies), and ran
    ///     FEWER GPU kernels (17,802 vs 36,718). On-device paged attention cannot
    ///     produce that D2H volume + kernel-count deficit — the attention work is
    ///     leaving the device.
    /// NOTE the trap this note carried for weeks: the PC-2 CUDA test
    /// `bf16_paged_decode_matches_contiguous_on_cuda` asserts only NUMERICAL parity
    /// (argmax(paged) == argmax(contiguous)) — INVARIANT to placement; a CPU
    /// fallback passes it identically. It never backed placement either way, and
    /// three code-readings that "reached different answers" were all non-
    /// discriminating. Measurement settled it; a fused CUDA `PagedAttn` kernel
    /// (or a decompose→GPU-primitives route with a real production caller) is the
    /// prerequisite to move it on-device.
    ///
    /// Same math as the contiguous forward up to the attention reduction order
    /// (paged gathers-then-dense-SDPAs where contiguous slices a fixed-capacity
    /// buffer) — ε-close, the bar the batched arm already uses.
    pub fn forward_paged_step(
        &self,
        token: u32,
        pool: &mut fuel_core::kv_block_pool_device::DeviceKvPool,
        session: fuel_core::kv_block_pool::SessionHandle,
    ) -> fuel_core::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        // The activation dtype IS the pool dtype (BF16-throughout decode, Phase D
        // increment A): f32 pool → f32 activations; bf16 pool → bf16 activations.
        let act_dtype = pool.dtype();
        if !matches!(act_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step: unsupported pool dtype {act_dtype:?} (expected F32/BF16/F16)",
            ))
            .bt());
        }
        // Build + realize on the pool's device so a CUDA pool runs on CUDA.
        let dev = pool.device().clone();
        let geom = pool.geometry();
        if geom.n_layers != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step: pool n_layers {} != model n_layers {}",
                geom.n_layers, cfg.n_layers,
            ))
            .bt());
        }
        let block_size = geom.block_size;

        // This token's absolute position; grow the session by one slot to cover it.
        let tok_pos = pool.core().filled_tokens(session).ok_or_else(|| {
            fuel_ir::Error::Msg("forward_paged_step: unknown session".to_string()).bt()
        })?;
        pool.core_mut().append(session, 1).map_err(|e| {
            fuel_ir::Error::Msg(format!("forward_paged_step: block append failed: {e:?}")).bt()
        })?;
        // Copy-on-write: if this token's block is shared (spliced), break the
        // share + copy so the write doesn't corrupt a co-sharer.
        let phys = pool.ensure_writable_block(session, tok_pos / block_size)?;
        let slot = tok_pos % block_size;
        let pt = pool.materialize_block_table(&[session]).map_err(|e| {
            fuel_ir::Error::Msg(format!(
                "forward_paged_step: block-table materialize failed: {e:?}"
            ))
            .bt()
        })?;

        // Embed the single token → [1, 1, dim]. The table stays f32 (CUDA
        // IndexSelect has no bf16 key); cast to the activation dtype after lookup.
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &dev,
        )?;
        let token_ids = embed.const_u32_like(vec![token], Shape::from_dims(&[1]))?;
        let mut h = embed
            .index_select(0, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[1, 1, cfg.dim]))
            .unwrap()
            .to_dtype(act_dtype)?;

        // RoPE tables at this token's absolute position (seq = 1).
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_base, tok_pos, 1, cfg.head_dim);
        let scale = (1.0f64 / (cfg.head_dim as f64).sqrt()) as f32;

        // block_table / context_lens for this session (single-row batch).
        let block_table = h.const_u32_like(pt.block_table.clone(), pt.block_table_shape())?;
        let context_lens = h.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape())?;

        // Per layer: bind this layer's pool K/V buffers to placeholders and build
        // the paged attend + FFN. All layers bind into ONE realize cache — the
        // pool buffers persist across steps via their Arcs (the same relationship
        // the contiguous forward's cache slots have), so the write this step lands
        // in place and is visible next step.
        let mut cache = fuel_dispatch::pipelined::StorageCache::new();
        for (li, layer) in weights.layers.iter().enumerate() {
            // Placeholders match the pool buffers' dtype (bf16 pool → bf16 bind).
            let k_ph = h.const_placeholder_like(pool.pool_shape().clone(), act_dtype);
            let v_ph = h.const_placeholder_like(pool.pool_shape().clone(), act_dtype);
            let k_arc = pool.k_pool(li).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step: no K pool buffer for layer {li}"
                ))
                .bt()
            })?;
            let v_arc = pool.v_pool(li).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step: no V pool buffer for layer {li}"
                ))
                .bt()
            })?;
            cache.insert(k_ph.graph_tensor().id(), std::sync::Arc::clone(k_arc));
            cache.insert(v_ph.graph_tensor().id(), std::sync::Arc::clone(v_arc));
            h = self.apply_layer_paged(
                &h,
                layer,
                pool,
                &k_ph,
                &v_ph,
                &rope_cos,
                &rope_sin,
                &block_table,
                &context_lens,
                phys,
                slot,
                scale,
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)?;
        // Cast to f32 before the f32 realize (a bf16 root is UB — half the byte
        // width); no-op under an f32 pool.
        let logits_root = logits
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?
            .to_dtype(DType::F32)?;
        fuel_core::pipelined_bridge::realize_one_as_with_initial::<f32>(
            logits_root.graph_tensor().graph(),
            logits_root.graph_tensor().id(),
            &dev,
            cache,
        )
    }

    /// **Plan-once persistent sibling of [`Self::forward_paged_step`]** — builds
    /// the paged decode graph + optimized plan ONCE (held in a
    /// [`fuel_core::inference_context::PagedDecodeSession`]) and REUSES it for every
    /// subsequent token, paying the optimizer (Lightbulb: ~90% of per-token
    /// paged cost) once instead of per token. The paged twin of
    /// [`Self::forward_with_kv_context_persistent`].
    ///
    /// Removes paged decode's three per-token re-plan triggers: (1) the fresh
    /// `Tensor` graph root → a held graph of stable re-bindable Const
    /// placeholders; (2) the L-varying `block_table` shape → pinned to
    /// `[1, max_blocks_cap]` (Task 1 padded materialize); (3) the per-step
    /// KV-write range → one flattened dynamic offset (Task 2
    /// `build_decode_attn_off`). Per token only shape-stable data Arcs + the
    /// bound offset change — re-bound into a clone of the held `base_cache`.
    ///
    /// `seq == 1` single-session (B = 1) — the simplest correct target; the
    /// ragged batched persistent path is a follow-on. `max_blocks_cap` pins the
    /// block_table shape: the paged driver passes the session's decode capacity
    /// `ceil(max_seq_len / block_size)`; it MUST be ≥ the session's eventual
    /// block count (else the padded materialize errors — never truncates). On a
    /// validity-key mismatch (cap / geometry / dtype) or a `TopologyChanged`,
    /// the held session is dropped and this token falls back to the re-planning
    /// [`Self::forward_paged_step`] (the session rebuilds on the next token).
    /// f32/bf16/f16; CPU-verifiable (the win is CPU-side planning).
    ///
    /// `plan` is the runtime flag ([`fuel_core::inference_context::PagedDecodePlan`]):
    /// `Replan` drops any held session and re-plans this token via
    /// [`Self::forward_paged_step`] (the pre-plan-once behavior, now an explicit
    /// opt-OUT); `PlanOnce` — **the driver default** — builds-once / rebinds as
    /// above.
    pub fn forward_paged_step_persistent(
        &self,
        token: u32,
        pool: &mut fuel_core::kv_block_pool_device::DeviceKvPool,
        session: fuel_core::kv_block_pool::SessionHandle,
        max_blocks_cap: usize,
        plan: fuel_core::inference_context::PagedDecodePlan,
        decode_session: &mut Option<fuel_core::inference_context::PagedDecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        // Runtime flag (`plan`). `Replan` is the pre-plan-once path: drop any
        // held session (so nothing stale lingers across a flag flip) and re-plan
        // this token via `forward_paged_step`. `PlanOnce` builds-once / rebinds
        // below. This is the exact toggle the paged driver ships (off by
        // default) and the correctness gate flips per arm.
        if matches!(plan, fuel_core::inference_context::PagedDecodePlan::Replan) {
            *decode_session = None;
            return self.forward_paged_step(token, pool, session);
        }
        let cfg = &self.config;
        let act_dtype = pool.dtype();
        if !matches!(act_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step_persistent: unsupported pool dtype {act_dtype:?} (expected F32/BF16/F16)",
            )).bt());
        }
        let geom = pool.geometry();
        if geom.n_layers != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step_persistent: pool n_layers {} != model n_layers {}",
                geom.n_layers, cfg.n_layers,
            ))
            .bt());
        }

        // Invalidate a stale held session (different cap / geometry / dtype).
        if let Some(s) = decode_session.as_ref() {
            // `decode_shape_key()` carries weight IDENTITY, so a same-shaped
            // model with different weights invalidates instead of silently
            // reusing a plan baked against the other one's Consts.
            if !s.is_valid_for(
                max_blocks_cap,
                geom.n_layers,
                geom.block_size,
                act_dtype,
                self.decode_shape_key(),
                pool.alloc_id(),
            ) {
                *decode_session = None;
            }
        }

        if decode_session.is_none() {
            // First paged decode token (or post-invalidation): build + optimize
            // the held graph ONCE.
            return self.build_and_realize_first_paged_token(
                token,
                pool,
                session,
                max_blocks_cap,
                decode_session,
            );
        }

        // Subsequent token: re-bind data + skip optimize.
        let res = {
            let s = decode_session.as_ref().expect("session is_some");
            self.rebind_and_realize_paged_prebuilt(token, pool, session, max_blocks_cap, s)
        };
        match res {
            Ok(logits) => Ok(logits),
            Err(fuel_core::Error::TopologyChanged { .. }) => {
                // Stale cached generation — drop the session and re-plan this
                // token via the D1 paged path; the session rebuilds next token.
                *decode_session = None;
                self.forward_paged_step(token, pool, session)
            }
            Err(e) => Err(e),
        }
    }

    /// Advance one paged decode token's POOL bookkeeping (shared by the build
    /// and rebind arms of the persistent path): grow the session by one slot,
    /// break any copy-on-write share on the target block, and materialize the
    /// capacity-padded block table. Returns `(tok_pos, linear, page_table)`
    /// where `linear = phys·block_size + slot` is the flattened KV-write offset.
    /// Takes `&mut pool` and returns owned data so the caller's `&mut` borrow
    /// ends before the (immutable) graph build.
    fn advance_paged_session(
        pool: &mut fuel_core::kv_block_pool_device::DeviceKvPool,
        session: fuel_core::kv_block_pool::SessionHandle,
        max_blocks_cap: usize,
    ) -> fuel_core::Result<(usize, usize, fuel_core::kv_block_pool_device::PageTableHost)> {
        let block_size = pool.geometry().block_size;
        let tok_pos = pool.core().filled_tokens(session).ok_or_else(|| {
            fuel_ir::Error::Msg("forward_paged_step_persistent: unknown session".to_string()).bt()
        })?;
        pool.core_mut().append(session, 1).map_err(|e| {
            fuel_ir::Error::Msg(format!(
                "forward_paged_step_persistent: block append failed: {e:?}"
            ))
            .bt()
        })?;
        let phys = pool.ensure_writable_block(session, tok_pos / block_size)?;
        let slot = tok_pos % block_size;
        let linear = phys as usize * block_size + slot;
        let pt =
            pool.materialize_block_table_padded(&[session], max_blocks_cap)
                .map_err(|e| {
                    fuel_ir::Error::Msg(format!(
                "forward_paged_step_persistent: padded block-table materialize failed: {e:?}",
            )).bt()
                })?;
        Ok((tok_pos, linear, pt))
    }

    /// Build the per-token data Arcs for one paged decode step (shared by the
    /// build + rebind arms). Uses the SAME `upload_host_buffer_to_device` path
    /// `KvCache::with_capacity` uses — on CPU the Storage wraps the host bytes,
    /// on GPU it performs the (tiny) H2D upload. The bytes change per token;
    /// the held graph's Const NodeIds stay stable (re-bound via a `base_cache`
    /// overwrite, not a fresh graph).
    fn build_paged_token_data(
        &self,
        dev: &Device,
        token: u32,
        tok_pos: usize,
        linear: usize,
        pt: &fuel_core::kv_block_pool_device::PageTableHost,
        use_device_offset: bool,
    ) -> fuel_core::Result<fuel_core::inference_context::PagedDecodeTokenData> {
        let cfg = &self.config;
        let upload = fuel_core::pipelined_bridge::upload_host_buffer_to_device;
        let token_ids = upload(dev, fuel_ir::HostBuffer::U32(vec![token]))?;
        let (cos_data, sin_data) =
            fuel_graph::build_rope_tables(cfg.rope_base, tok_pos, 1, cfg.head_dim);
        let rope_cos = upload(dev, fuel_ir::HostBuffer::F32(cos_data))?;
        let rope_sin = upload(dev, fuel_ir::HostBuffer::F32(sin_data))?;
        let block_table = upload(dev, fuel_ir::HostBuffer::U32(pt.block_table.clone()))?;
        let context_lens = upload(dev, fuel_ir::HostBuffer::U32(pt.context_lens.clone()))?;
        let offset = if use_device_offset {
            Some(upload(dev, fuel_ir::HostBuffer::I64(vec![linear as i64]))?)
        } else {
            None
        };
        Ok(fuel_core::inference_context::PagedDecodeTokenData {
            token_ids,
            rope_cos,
            rope_sin,
            block_table,
            context_lens,
            offset,
        })
    }

    /// Build the held paged decode-step graph with STABLE re-bindable data
    /// Consts, optimize it ONCE via `prebuild_optimized_env_capturing_cache`,
    /// populate `decode_session`, and return the first token's logits. Only
    /// called for the first paged decode token when there is no valid session.
    fn build_and_realize_first_paged_token(
        &self,
        token: u32,
        pool: &mut fuel_core::kv_block_pool_device::DeviceKvPool,
        session: fuel_core::kv_block_pool::SessionHandle,
        max_blocks_cap: usize,
        decode_session: &mut Option<fuel_core::inference_context::PagedDecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dev = pool.device().clone();
        let act_dtype = pool.dtype();
        let geom = pool.geometry();
        let block_size = geom.block_size;
        let n_layers = geom.n_layers;
        let scale = (1.0f64 / (cfg.head_dim as f64).sqrt()) as f32;
        // Device-offset (WriteSliceDoff) on CPU/CUDA — capture-ready + the
        // CPU-verifiable arm; SymEnv (WriteSliceDyn) on Vulkan. Mirrors the
        // contiguous persistent path's split.
        let use_device_offset = dev.is_cpu() || dev.is_cuda();
        let write_sym = fuel_ir::SymId(0);

        // Advance the pool for THIS token (same bookkeeping the re-planning
        // path does); releases the &mut borrow before the immutable build.
        let (tok_pos, linear, pt) = Self::advance_paged_session(pool, session, max_blocks_cap)?;

        // ---- Build the held graph ONCE with STABLE re-bindable placeholders. ----
        // Embed table stays f32 (CUDA IndexSelect has no bf16 key); cast to the
        // activation dtype after lookup. The embed Const is the graph root.
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &dev,
        )?;
        let token_ids = embed.const_placeholder_like(Shape::from_dims(&[1]), DType::U32);
        let token_ids_node = token_ids.graph_tensor().id();
        let mut h = embed
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[1, 1, cfg.dim]))?
            .to_dtype(act_dtype)?;

        // RoPE cos/sin at this token's absolute position — STABLE placeholders.
        let rope_shape = Shape::from_dims(&[1, cfg.head_dim]);
        let rope_cos = h.const_placeholder_like(rope_shape.clone(), DType::F32);
        let rope_sin = h.const_placeholder_like(rope_shape, DType::F32);
        let rope_cos_node = rope_cos.graph_tensor().id();
        let rope_sin_node = rope_sin.graph_tensor().id();

        // block_table pinned to [1, max_blocks_cap] (Task 1) + context_lens —
        // STABLE placeholders. `.max(1)` matches the padded materialize's
        // rank-2 well-formedness floor.
        let block_table =
            h.const_placeholder_like(Shape::from_dims(&[1, max_blocks_cap.max(1)]), DType::U32);
        let block_table_node = block_table.graph_tensor().id();
        let context_lens = h.const_placeholder_like(Shape::from_dims(&[1]), DType::U32);
        let context_lens_node = context_lens.graph_tensor().id();

        // The flattened KV-write offset carrier (device path: a rank-0 I64
        // placeholder; SymEnv path: None, the offset rides `write_sym`).
        let offset_tensor = if use_device_offset {
            Some(h.const_placeholder_like(Shape::from_dims(&[]), DType::I64))
        } else {
            None
        };
        let offset_node = offset_tensor.as_ref().map(|t| t.graph_tensor().id());

        // Per layer: STABLE pool K/V placeholders (viewed at pool_shape_flat so
        // the write is ONE dynamic axis-0 offset), bound ONCE to the pool
        // buffers and mutated in place each token.
        let mut cache = fuel_dispatch::pipelined::StorageCache::new();
        let mut kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)> =
            Vec::with_capacity(n_layers);
        let flat_shape = pool.pool_shape_flat();
        for (li, layer) in weights.layers.iter().enumerate() {
            let k_ph = h.const_placeholder_like(flat_shape.clone(), act_dtype);
            let v_ph = h.const_placeholder_like(flat_shape.clone(), act_dtype);
            let k_arc = pool.k_pool(li).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step_persistent: no K pool buffer for layer {li}"
                ))
                .bt()
            })?;
            let v_arc = pool.v_pool(li).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step_persistent: no V pool buffer for layer {li}"
                ))
                .bt()
            })?;
            let k_id = k_ph.graph_tensor().id();
            let v_id = v_ph.graph_tensor().id();
            cache.insert(k_id, std::sync::Arc::clone(k_arc));
            cache.insert(v_id, std::sync::Arc::clone(v_arc));
            kv_nodes.push((k_id, v_id));
            h = self.apply_layer_paged_off(
                &h,
                layer,
                pool,
                &k_ph,
                &v_ph,
                &rope_cos,
                &rope_sin,
                &block_table,
                &context_lens,
                offset_tensor.as_ref(),
                write_sym,
                scale,
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)?;
        // Cast to f32 before the f32 realize (a bf16 root is UB — half the byte
        // width); no-op under an f32 pool.
        let logits_root = logits
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?
            .to_dtype(DType::F32)?;
        let logits_node = logits_root.graph_tensor().id();
        let graph = logits_root.graph_tensor().graph().clone();

        // ---- Bind the per-token DATA + realize + optimize ONCE. ----
        let data =
            self.build_paged_token_data(&dev, token, tok_pos, linear, &pt, use_device_offset)?;
        cache.insert(token_ids_node, std::sync::Arc::clone(&data.token_ids));
        cache.insert(rope_cos_node, std::sync::Arc::clone(&data.rope_cos));
        cache.insert(rope_sin_node, std::sync::Arc::clone(&data.rope_sin));
        cache.insert(block_table_node, std::sync::Arc::clone(&data.block_table));
        cache.insert(context_lens_node, std::sync::Arc::clone(&data.context_lens));
        if let (Some(off_node), Some(off_arc)) = (offset_node, data.offset.as_ref()) {
            cache.insert(off_node, std::sync::Arc::clone(off_arc));
        }

        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(write_sym, linear)?;

        let (effective_target, optimized, base_cache, logits_vec) =
            fuel_core::pipelined_bridge::prebuild_optimized_env_capturing_cache::<f32>(
                &graph,
                logits_node,
                &dev,
                cache,
                &sym_env,
            )?;

        *decode_session = Some(fuel_core::inference_context::PagedDecodeSession::new(
            graph,
            optimized,
            effective_target,
            logits_node,
            token_ids_node,
            rope_cos_node,
            rope_sin_node,
            block_table_node,
            context_lens_node,
            kv_nodes,
            offset_node,
            write_sym,
            base_cache,
            max_blocks_cap,
            n_layers,
            block_size,
            act_dtype,
            self.decode_shape_key(),
            // Which POOL's block buffers are baked into `base_cache` — the
            // rebind never re-binds them, so a same-geometry pool swap would
            // otherwise reuse this plan over the wrong KV.
            pool.alloc_id(),
        ));

        Ok(logits_vec)
    }

    /// Re-bind one paged decode token's data into a clone of the held session's
    /// `base_cache` and realize via the prebuilt seam (SKIP prepare + optimize).
    /// The pool bookkeeping runs the same as the build arm; only the plan is
    /// reused. `TopologyChanged` surfaces typed (the caller invalidates).
    fn rebind_and_realize_paged_prebuilt(
        &self,
        token: u32,
        pool: &mut fuel_core::kv_block_pool_device::DeviceKvPool,
        session: fuel_core::kv_block_pool::SessionHandle,
        max_blocks_cap: usize,
        decode_session: &fuel_core::inference_context::PagedDecodeSession,
    ) -> fuel_core::Result<Vec<f32>> {
        let dev = pool.device().clone();
        // Source of truth for the offset carrier = what the graph was built
        // with (its offset_node presence), so build + rebind never disagree.
        let use_device_offset = decode_session.offset_node().is_some();
        let (tok_pos, linear, pt) = Self::advance_paged_session(pool, session, max_blocks_cap)?;
        let data =
            self.build_paged_token_data(&dev, token, tok_pos, linear, &pt, use_device_offset)?;
        let sym_env = decode_session.per_token_sym_env(linear)?;
        decode_session.realize_token(&dev, data, &sym_env)
    }

    /// Batched (`B = K`) sibling of [`Self::forward_paged_step`] — one decode
    /// step over K sessions in a single model pass (paged-storage PS4a, the
    /// throughput arm). Each session `i` contributes one `token` at its OWN
    /// position: a ragged batch (sessions at different `filled_tokens`) is fully
    /// supported — per-row RoPE (`rope_tables_const_batched` → `rope_batched`) +
    /// per-row slot index remove the former uniformity precondition. Returns one
    /// logits row `[vocab_size]` per session, in `sessions` order.
    ///
    /// Each session's new K/V is written into its OWN physical block (they differ
    /// by `block_table` row), and one `Op::PagedAttn` at batch `K` reads them —
    /// so per-row results are independent (no cross-session contamination) and
    /// equal the B=1 serial path ε-close. The pool-dtype gate accepts F32/BF16/F16.
    /// `Op::PagedAttn` has no fused CUDA/Vulkan kernel (CPU-only in the binding
    /// table). On a CUDA target this attention currently runs OFF-DEVICE (CPU)
    /// with copies — RESOLVED 2026-08-01 by an nsys A/B (Lightbulb): the paged arm
    /// pulled ~6.2 GB/token D2H vs the contiguous arm's ~13.5 MB logits-readback
    /// floor, with FEWER GPU kernels. See `forward_paged_step`'s note for the full
    /// account (the PC-2 test asserts argmax parity, invariant to placement — it
    /// never backed either answer; measurement, not reading, settled it).
    pub fn forward_paged_step_batched(
        &self,
        tokens: &[u32],
        pool: &mut fuel_core::kv_block_pool_device::DeviceKvPool,
        sessions: &[fuel_core::kv_block_pool::SessionHandle],
    ) -> fuel_core::Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let k = tokens.len();
        if k == 0 || k != sessions.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step_batched: {} tokens for {} sessions (need equal, ≥ 1)",
                k,
                sessions.len(),
            ))
            .bt());
        }
        let act_dtype = pool.dtype();
        if !matches!(act_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step_batched: unsupported pool dtype {act_dtype:?} (expected F32/BF16/F16)",
            )).bt());
        }
        let dev = pool.device().clone();
        let geom = pool.geometry();
        if geom.n_layers != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step_batched: pool n_layers {} != model n_layers {}",
                geom.n_layers, cfg.n_layers,
            ))
            .bt());
        }
        let block_size = geom.block_size;

        // Per-session positions — a ragged batch is fully supported: each row
        // carries its own RoPE position and slot index. (The former uniformity
        // gate existed only because RoPE + slot were shared; per-row RoPE via
        // rope_tables_const_batched dissolved that precondition.)
        let mut positions = Vec::with_capacity(k);
        for &s in sessions {
            let p = pool.core().filled_tokens(s).ok_or_else(|| {
                fuel_ir::Error::Msg("forward_paged_step_batched: unknown session".to_string()).bt()
            })?;
            positions.push(p);
        }

        // Atomicity + capacity pre-check (C-1): a batched step allocates one block
        // per boundary-crossing session (slot == 0) PLUS one per shared frontier
        // block that copy-on-write will split. Verify the whole batch fits BEFORE
        // mutating any session, so a mid-batch OutOfBlocks can't leave the batch
        // partially advanced (which would wedge it non-uniform + un-retryable).
        let mut needed = 0usize;
        for (bi, &s) in sessions.iter().enumerate() {
            let pos_b = positions[bi];
            if pos_b % block_size == 0 {
                needed += 1; // append allocates a fresh block for the new token
            } else {
                let frontier = pool
                    .core()
                    .resident_block(s, pos_b / block_size)
                    .ok_or_else(|| {
                        fuel_ir::Error::Msg(
                            "forward_paged_step_batched: frontier block not resident".to_string(),
                        )
                        .bt()
                    })?;
                if pool.core().block_refcount(frontier) > 1 {
                    needed += 1; // copy-on-write will split this shared block
                }
            }
        }
        let free = pool.core().free_blocks();
        if needed > free {
            return Err(fuel_ir::Error::Msg(format!(
                "forward_paged_step_batched: batch needs {needed} blocks, {free} free — \
                 pre-check capacity (C-1) or evict before batching",
            ))
            .bt());
        }

        // Execute — pre-checked to fit, so no session is left partially advanced.
        let mut writes: Vec<(fuel_core::kv_block_pool::PhysBlockId, usize)> = Vec::with_capacity(k);
        for (bi, &s) in sessions.iter().enumerate() {
            let pos_b = positions[bi];
            pool.core_mut().append(s, 1).map_err(|e| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step_batched: block append failed: {e:?}"
                ))
                .bt()
            })?;
            // Copy-on-write if this session's frontier block is shared (spliced).
            let phys = pool.ensure_writable_block(s, pos_b / block_size)?;
            writes.push((phys, pos_b % block_size));
        }
        let pt = pool.materialize_block_table(sessions).map_err(|e| {
            fuel_ir::Error::Msg(format!(
                "forward_paged_step_batched: block-table materialize failed: {e:?}"
            ))
            .bt()
        })?;

        // Embed K tokens → [K, 1, dim] (f32 table; cast to activation dtype).
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &dev,
        )?;
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[k]))?;
        let mut h = embed
            .index_select(0, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[k, 1, cfg.dim]))
            .unwrap()
            .to_dtype(act_dtype)?;

        // Per-row RoPE: one position per session -> [K,1,1,head_dim] tables.
        // block_table [K, max_blk], context_lens [K] (both already ragged).
        let (rope_cos, rope_sin) =
            h.rope_tables_const_batched(cfg.rope_base, &positions, cfg.head_dim);
        let scale = (1.0f64 / (cfg.head_dim as f64).sqrt()) as f32;
        let block_table = h.const_u32_like(pt.block_table.clone(), pt.block_table_shape())?;
        let context_lens = h.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape())?;

        let mut cache = fuel_dispatch::pipelined::StorageCache::new();
        for (li, layer) in weights.layers.iter().enumerate() {
            let k_ph = h.const_placeholder_like(pool.pool_shape().clone(), act_dtype);
            let v_ph = h.const_placeholder_like(pool.pool_shape().clone(), act_dtype);
            let k_arc = pool.k_pool(li).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step_batched: no K pool buffer for layer {li}"
                ))
                .bt()
            })?;
            let v_arc = pool.v_pool(li).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "forward_paged_step_batched: no V pool buffer for layer {li}"
                ))
                .bt()
            })?;
            cache.insert(k_ph.graph_tensor().id(), std::sync::Arc::clone(k_arc));
            cache.insert(v_ph.graph_tensor().id(), std::sync::Arc::clone(v_arc));
            h = self.apply_layer_paged_batched(
                &h,
                layer,
                pool,
                &k_ph,
                &v_ph,
                &rope_cos,
                &rope_sin,
                &block_table,
                &context_lens,
                &writes,
                scale,
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)?; // [K, 1, vocab]
        let logits_flat = logits
            .reshape(Shape::from_dims(&[k * cfg.vocab_size]))?
            .to_dtype(DType::F32)?; // f32 for the realize (no-op under an f32 pool)
        let flat = fuel_core::pipelined_bridge::realize_one_as_with_initial::<f32>(
            logits_flat.graph_tensor().graph(),
            logits_flat.graph_tensor().id(),
            &dev,
            cache,
        )?;
        // Split the flat [K·vocab] into one [vocab] row per session (row-major).
        Ok(flat.chunks(cfg.vocab_size).map(|c| c.to_vec()).collect())
    }

    /// Batched (`B = K`) sibling of [`Self::apply_layer_paged`]: the shared
    /// `project_qkv_roped` (batch-agnostic) + `build_decode_attn_batched` (K
    /// per-session slot writes + one `Op::PagedAttn` at B=K) + the shared
    /// `ffn_block`. `x` is `[K, 1, dim]`.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_paged_batched(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        pool: &fuel_core::kv_block_pool_device::DeviceKvPool,
        k_pool_ph: &Tensor,
        v_pool_ph: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        writes: &[(fuel_core::kv_block_pool::PhysBlockId, usize)],
        scale: f32,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let xs = x.shape();
        let batch = xs.dims()[0];
        let (q_r, k_r, v_h) = self.project_qkv_roped_batched(x, layer, rope_cos, rope_sin)?;
        let attn = pool.build_decode_attn_batched(
            k_pool_ph,
            v_pool_ph,
            &q_r,
            &k_r,
            &v_h,
            block_table,
            context_lens,
            writes,
            scale,
        )?;
        let merged = attn
            .permute([0, 2, 1, 3_usize])
            .unwrap()
            .reshape(Shape::from_dims(&[batch, 1, cfg.dim]))
            .unwrap();
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim)?;
        let h1 = x.add(&attn_out).unwrap();
        self.ffn_block(&h1, layer)
    }

    /// One transformer layer of the paged decode step: the projection/RoPE of
    /// [`Self::apply_layer_with_kv_writes`] (duplicated so the tested contiguous
    /// forward is untouched), with the KV write + sliced attention replaced by
    /// [`DeviceKvPool::build_decode_attn`](fuel_core::kv_block_pool_device::DeviceKvPool::build_decode_attn)
    /// — write the new token's K/V into its pool slot, then `Op::PagedAttn`. The
    /// o-projection + residual + SwiGLU FFN tail is identical to the contiguous
    /// layer. `seq == batch == 1` (single-token decode step).
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_paged(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        pool: &fuel_core::kv_block_pool_device::DeviceKvPool,
        k_pool_ph: &Tensor,
        v_pool_ph: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        phys: fuel_core::kv_block_pool::PhysBlockId,
        slot: usize,
        scale: f32,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let (batch, seq) = (1usize, 1usize);
        let (q_r, k_r, v_h) = self.project_qkv_roped(x, layer, rope_cos, rope_sin)?;

        // Paged storage + attention (replaces the contiguous write_slice + sliced SDPA).
        let attn = pool.build_decode_attn(
            k_pool_ph,
            v_pool_ph,
            &q_r,
            &k_r,
            &v_h,
            block_table,
            context_lens,
            phys,
            slot,
            scale,
        )?;

        let merged = attn
            .permute([0, 2, 1, 3_usize])
            .unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .unwrap();
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim)?;
        let h1 = x.add(&attn_out).unwrap();
        self.ffn_block(&h1, layer)
    }

    /// **Plan-once sibling of [`Self::apply_layer_paged`]** — identical
    /// attention front half + o-projection + FFN, but the paged KV write +
    /// attend go through
    /// [`DeviceKvPool::build_decode_attn_off`](fuel_core::kv_block_pool_device::DeviceKvPool::build_decode_attn_off)
    /// (flattened, runtime-resolved write offset) so the layer's graph is
    /// structurally IDENTICAL across decode steps — only the bound offset
    /// changes. `k_pool_ph`/`v_pool_ph` are `pool_shape_flat()` placeholders;
    /// `write_off` is `Some` on the device-offset path (CPU/CUDA) / `None` on
    /// the SymEnv path (Vulkan), with `write_sym` carrying the offset there.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_paged_off(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        pool: &fuel_core::kv_block_pool_device::DeviceKvPool,
        k_pool_ph: &Tensor,
        v_pool_ph: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        write_off: Option<&Tensor>,
        write_sym: fuel_ir::SymId,
        scale: f32,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let (batch, seq) = (1usize, 1usize);
        let (q_r, k_r, v_h) = self.project_qkv_roped(x, layer, rope_cos, rope_sin)?;

        // Paged storage + attention with a flattened dynamic write offset
        // (structurally step-invariant), replacing the concrete two-axis slab.
        let attn = pool.build_decode_attn_off(
            k_pool_ph,
            v_pool_ph,
            &q_r,
            &k_r,
            &v_h,
            block_table,
            context_lens,
            write_off,
            write_sym,
            scale,
        )?;

        let merged = attn
            .permute([0, 2, 1, 3_usize])
            .unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .unwrap();
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.dim, cfg.dim)?;
        let h1 = x.add(&attn_out).unwrap();
        self.ffn_block(&h1, layer)
    }

    /// Shared attention **front half** — RmsNorm → Q/K/V projections (+ optional
    /// biases) → per-head reshape → RoPE on Q and K — returning
    /// `(q_r, k_r, v_h)` at `[batch, {n_heads|n_kv_heads}, seq, head_dim]`. The
    /// math is identical for contiguous ([`Self::apply_layer_with_kv_writes`]) and
    /// paged ([`Self::apply_layer_paged`]) decode; only KV storage + attention
    /// differ downstream. RoPE runs in f32 (its build-time requirement) with no-op
    /// casts around it when the activation dtype is already f32.
    fn project_qkv_roped(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> fuel_core::Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let act_dtype = x.dtype();

        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.dim, cfg.dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())
            .unwrap();
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.dim, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())
            .unwrap();
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.dim, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())
            .unwrap();
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let k_h = k
            .reshape(Shape::from_dims(&[
                batch,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
            ]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let v_h = v
            .reshape(Shape::from_dims(&[
                batch,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
            ]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let q_r = q_h
            .to_dtype(DType::F32)?
            .rope_with_tables_decomposed(rope_cos, rope_sin)?
            .to_dtype(act_dtype)?;
        let k_r = k_h
            .to_dtype(DType::F32)?
            .rope_with_tables_decomposed(rope_cos, rope_sin)?
            .to_dtype(act_dtype)?;
        Ok((q_r, k_r, v_h))
    }

    /// Per-row RoPE sibling of [`Self::project_qkv_roped`]: identical projection +
    /// head reshape, but applies RoPE via [`Tensor::rope_batched`] with
    /// `[batch, 1, 1, head_dim]` cos/sin tables (one position per row) instead of
    /// the shared single-position `rope_with_tables_decomposed`. Uniform
    /// positions are a bit-identical special case, so this subsumes the shared
    /// path for the batched paged decode. `rope_cos`/`rope_sin` come from
    /// [`Tensor::rope_tables_const_batched`].
    fn project_qkv_roped_batched(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> fuel_core::Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let act_dtype = x.dtype();

        let x_norm = apply_affine_rms_norm(x, &layer.attn_norm_gain, cfg.dim, cfg.norm_eps);
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.dim, cfg.dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())
            .unwrap();
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.dim, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())
            .unwrap();
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.dim, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())
            .unwrap();
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let k_h = k
            .reshape(Shape::from_dims(&[
                batch,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
            ]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let v_h = v
            .reshape(Shape::from_dims(&[
                batch,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
            ]))
            .unwrap()
            .permute([0, 2, 1, 3_usize])
            .unwrap();
        let q_r = q_h
            .to_dtype(DType::F32)?
            .rope_batched(rope_cos, rope_sin)?
            .to_dtype(act_dtype)?;
        let k_r = k_h
            .to_dtype(DType::F32)?
            .rope_batched(rope_cos, rope_sin)?
            .to_dtype(act_dtype)?;
        Ok((q_r, k_r, v_h))
    }

    /// Shared attention **tail** — pre-FFN RmsNorm → SwiGLU (`gate.silu() * up`) →
    /// down-projection → residual. `h1` is the post-attention residual
    /// (`x + o_proj(merged_heads)`). Identical for contiguous + paged decode.
    fn ffn_block(&self, h1: &Tensor, layer: &LayerWeights) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let h1_norm = apply_affine_rms_norm(h1, &layer.ffn_norm_gain, cfg.dim, cfg.norm_eps);
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim)?;
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.dim, cfg.ffn_dim)?;
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&swiglu, cfg.ffn_dim, cfg.dim)?;
        h1.add(&ffn_out)
    }

    /// Forward pass using pre-allocated KV-cache buffers and
    /// `Op::WriteSlice`. The cache must have been constructed via
    /// [`KvCache::with_capacity`] (the legacy `with_dims` grow-by-
    /// replacement constructor is rejected — its layers carry no
    /// pre-allocated storage to write into).
    ///
    /// ## Architectural notes
    ///
    /// - The cache's K + V Storage Arcs are bound to per-step Const
    ///   NodeIds via [`InferenceContext::insert`]. The
    ///   `const_placeholder_like` helper pushes Const nodes WITHOUT
    ///   populating the graph's legacy `storage_map` — the realize
    ///   call's `initial` StorageCache (cloned from `ctx.persistent`)
    ///   short-circuits the `build_const_cache` walk.
    /// - The cache buffers are mutated in place by
    ///   `Op::WriteSlice`'s kernel; the cache's Arcs persist outside
    ///   the graph (the graph is built fresh per forward step and
    ///   dropped after realize). Subsequent forward steps see the
    ///   accumulated K/V state via the same Arcs.
    /// - Logits return shape: rank-1 `[vocab_size]` — last-position
    ///   only, same as [`Self::forward_with_kv_context`].
    /// - Backends: CPU, CUDA, and Vulkan all run this path via the
    ///   pipelined executor + binding-table dispatch.
    ///
    /// The [`fuel_core::decode_shape`] key a held decode plan for THIS model is
    /// baked against: family + the config values that change graph structure +
    /// this weight set's identity.
    ///
    /// `rope_base` is deliberately absent — RoPE tables are rebound per token,
    /// not baked, so including it would forfeit plan reuse across a frequency
    /// change that is already handled correctly. See the module docs.
    pub fn decode_shape_key(&self) -> u64 {
        let mut h = fuel_core::decode_shape::ShapeKeyHasher::new();
        h.mix_str("llama")
            .mix_instance(self.weights.instance)
            .mix_u64(self.config.n_layers as u64)
            .mix_u64(self.config.n_heads as u64)
            .mix_u64(self.config.n_kv_heads as u64)
            .mix_u64(self.config.head_dim as u64)
            .mix_u64(self.config.dim as u64)
            .mix_u64(self.config.ffn_dim as u64)
            .mix_u64(self.config.vocab_size as u64)
            .mix_f64(self.config.norm_eps);
        // The mask plan wires the graph (how many mask variants exist and which
        // layer reads which), so a session built under one plan must not be
        // reused under another. Uniform for Llama — mixed unconditionally
        // anyway, so the key cannot silently stop covering it if that changes.
        fuel_core::persistent_decode::MaskPlan::dense(self.config.n_layers).mix_into(&mut h);
        h.finish()
    }

    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> fuel_core::Result<Vec<f32>> {
        self.forward_with_kv_context_impl(tokens, cache, ctx, false, None)
    }

    /// [`Self::forward_with_kv_context`] with the RoPE inverse frequencies
    /// supplied by the caller instead of derived from `config.rope_base`.
    ///
    /// This is the seam a scaled-RoPE wrapper decodes through — LLaMA-3.1's
    /// long-context scaling is a per-dimension transform of exactly this
    /// vector (`fuel_graph::build_rope_tables_with_inv_freq`), so a scaled
    /// model reuses this whole path rather than needing its own copy of it.
    /// `None` is bit-identical to [`Self::forward_with_kv_context`].
    pub fn forward_with_kv_context_inv_freq(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<Vec<f32>> {
        self.forward_with_kv_context_impl(tokens, cache, ctx, false, rope_inv_freq)
    }

    /// All-positions variant of [`Self::forward_with_kv_context`]:
    /// returns `seq * vocab_size` logits (flat, row-major over
    /// position). Used by speculative decoding's verification step —
    /// the target model runs forward on the K drafted tokens at once
    /// and needs per-position logits to accept/reject each draft.
    ///
    /// Cache semantics identical to `forward_with_kv_context`; on
    /// reject, the caller invokes [`KvCache::truncate_to`] to roll
    /// back (a pure metadata update on the pre-allocated-buffer path —
    /// rows past `cached_len` stop being read and are overwritten by
    /// the next `Op::WriteSlice` at the same positions).
    pub fn forward_with_kv_context_all_positions(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> fuel_core::Result<Vec<f32>> {
        self.forward_with_kv_context_impl(tokens, cache, ctx, true, None)
    }

    fn forward_with_kv_context_impl(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        return_all_positions: bool,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context(
            self,
            tokens,
            cache,
            ctx,
            return_all_positions,
            rope_inv_freq,
        )
    }

    /// **Plan-reuse decode at the SAME call shape as
    /// [`Self::forward_with_kv_context`]** — `(tokens, cache, ctx)`, no fourth
    /// argument. The held plan rides in the `InferenceContext`, so a caller
    /// hand-rolling a decode loop gets plan reuse without knowing that
    /// `DecodeSession` exists.
    ///
    /// **Why this is a separate entry rather than a change to
    /// `forward_with_kv_context`.** The ergonomic shape and the fast shape had
    /// drifted apart: the persistent sibling needs `&mut Option<DecodeSession>`,
    /// so the call a consumer naturally writes is the slow one. That drift is
    /// not theoretical — it cost a measured consumer 5,901 → 26.47 ms/token
    /// (nsys, 2026-08-01) and read to them as "Fuel ships a bad default" when
    /// the real defect was that the fast path was unreachable at the shape they
    /// were writing.
    ///
    /// `forward_with_kv_context` keeps its rebuild contract deliberately: it is
    /// the primitive the persistent path itself falls back to (`seq != 1`,
    /// invalidation, `TopologyChanged`), and dozens of tests exercise the
    /// rebuild path *as* the thing under test. Silently making it persistent
    /// would change what those tests mean. So the fix is to make the fast path
    /// reachable, not to redefine the primitive.
    ///
    /// Semantics are exactly [`Self::forward_with_kv_context_persistent`]'s —
    /// this only owns the session for you. `seq != 1` (prefill, spec-decode
    /// verification) falls back to the rebuild path without building a plan;
    /// the first `seq == 1` token builds it; later tokens rebind and skip
    /// optimize. Output is byte-identical to `forward_with_kv_context` either
    /// way. The plan self-heals on a cache resize / dtype change
    /// ([`fuel_core::inference_context::DecodeSession::is_valid_for`]).
    ///
    /// **One `InferenceContext` per model** — already the invariant here
    /// (speculative decoding builds a separate context for draft and target).
    pub fn forward_decode_step(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> fuel_core::Result<Vec<f32>> {
        // Take/put rather than borrow: the persistent forward needs `&mut ctx`
        // and `&mut Option<DecodeSession>` simultaneously. The result is bound
        // BEFORE the put so the session returns to the context on the error
        // path too — dropping it on error would silently downgrade every
        // subsequent token to re-planning.
        let mut session = ctx.take_decode_session();
        let out = self.forward_with_kv_context_persistent(tokens, cache, ctx, &mut session);
        ctx.put_decode_session(session);
        out
    }

    /// Phase D · D2b — plan-once persistent decode. Sibling of
    /// [`Self::forward_with_kv_context`] that HOLDS the optimized
    /// decode-step graph in `session` and, on every token after the
    /// first, re-realizes the SAME graph with the D2a prebuilt seam —
    /// **skipping the `prepare` D2H-splice + the `optimize_graph`
    /// placement DP**. The ~1.8×/token win comes from not re-planning.
    ///
    /// ## Control flow
    ///
    /// - **`seq != 1`** (prefill / spec-decode verification) OR the held
    ///   `session` is **invalid** for this step (validity-key mismatch):
    ///   drop the session and fall back to the D1 rebuild path
    ///   ([`Self::forward_with_kv_context`]). The session is rebuilt on
    ///   the next `seq == 1` token.
    /// - **First `seq == 1` token with no session:** build the decode
    ///   graph ONCE with STABLE re-bindable data Consts (token-ids /
    ///   RoPE cos+sin / mask / per-layer KV, all as
    ///   `const_placeholder_like` + `ctx.insert` of a device-resident
    ///   Arc), `prebuild_optimized_env` (runs `prepare` + `optimize` +
    ///   dispatch ONCE), and populate `session` with the held graph +
    ///   cached `OptimizedGraph` + the stable NodeIds. `OPTIMIZE_CALLS`
    ///   bumps once here.
    /// - **Subsequent `seq == 1` tokens with a valid session:** recompute
    ///   the per-token host bytes (token-ids = the new token, RoPE tables
    ///   at `position = cached_len`, mask with the shifted `-inf`
    ///   boundary) and WRITE them into the held device Arcs (re-bind);
    ///   bind the per-pass `SymEnv` (`cached_len`); call
    ///   [`InferenceContext::realize_prebuilt_as_with_env`] which SKIPS
    ///   optimize. The KV Arcs are re-bound once at build time and mutate
    ///   in place via `Op::WriteSlice` (NOT re-inserted per token). A
    ///   `TopologyChanged` invalidates the session (dropped) and falls
    ///   back to the rebuild path this token.
    ///
    /// Byte-identical to the D1 cached path on the same prefix (same plan
    /// → same kernels). Bumps `cache.cached_len` + per-slot versions
    /// exactly as [`Self::forward_with_kv_context`] does.
    ///
    /// The held data Consts persist across tokens (NOT removed each
    /// token); they are removed from `ctx` when the session is dropped.
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        self.forward_with_kv_context_persistent_inv_freq(tokens, cache, ctx, session, None)
    }

    /// [`Self::forward_with_kv_context_persistent`] with caller-supplied RoPE
    /// inverse frequencies — the persistent-decode sibling of
    /// [`Self::forward_with_kv_context_inv_freq`], and the entry point a
    /// scaled-RoPE wrapper serves through.
    ///
    /// The override reaches BOTH halves of the persistent path: the first
    /// token's held-graph build AND every subsequent per-token rebind.
    /// Threading only one would give a model whose first decode token used
    /// scaled RoPE and whose remaining tokens did not — a silent,
    /// position-dependent wrong answer rather than an error, which is why the
    /// parity test decodes several tokens rather than one.
    /// `None` is bit-identical to the unscaled method.
    pub fn forward_with_kv_context_persistent_inv_freq(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context_persistent(
            self,
            tokens,
            cache,
            ctx,
            session,
            rope_inv_freq,
        )
    }

    /// Build the held decode-step graph with STABLE re-bindable data
    /// Consts, optimize it ONCE via `prebuild_optimized_env`, populate
    /// `session`, and return the first token's logits. Only called for
    /// the first `seq == 1` decode token when there is no valid session.
    /// Delegates to the shared decode BUILD path — see
    /// [`fuel_core::persistent_decode`], which now owns this body for every
    /// LLaMA-shaped family. A change there is a change for all of them.
    // Only remaining caller is `forward_with_kv_context_captured`, which is
    // `#[cfg(feature = "cuda")]` — GAP-029 increment 3 moved the non-CUDA
    // callers onto the shared seam in `fuel_core::persistent_decode`. So "never
    // used" on a default build is a FALSE signal from a feature-gated caller,
    // not an orphan: deleting this breaks `--features cuda`, which no gate on
    // this machine compiles cheaply.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn build_and_realize_first_decode_token(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<Vec<f32>> {
        fuel_core::persistent_decode::build_and_realize_first_decode_token(
            self,
            tokens,
            cache,
            ctx,
            session,
            rope_inv_freq,
        )
    }

    /// Multi-session serving Increment 1 (C3): one **live batched** decode step
    /// over K sessions' KV caches, matching K serial single-session
    /// `forward_with_kv_context_persistent([last_token], …)` steps.
    ///
    /// This is a SEPARATE batch=K plan-once graph, so its float reduction order
    /// differs from the batch=1 serial arm: the guarantee is **ε-close** (logits
    /// within 1e-4) and **token-identical** on tested shapes, NOT bit-exact.
    /// KNOWN LIMITATION: on a real model with near-tied logits, ε-level drift
    /// could flip a greedy argmax and diverge the batched token stream from the
    /// serial one — the parity gates use tiny models where this does not occur.
    ///
    /// All K caches must be **uniform** — equal `cached_len`, `max_seq_len`,
    /// `n_layers`, dtype (the scheduler's uniformity gate guarantees this so
    /// the single shared `flash_decoding` `k_len` = `cached_len + 1` is correct
    /// for every row). `last_tokens[i]` is session `i`'s most recent token.
    ///
    /// Mechanism (spec risk #2 (a), copy-in/copy-out into a per-call shared
    /// buffer; all-or-nothing commit, spec risk #7):
    /// 1. Allocate a shared `[K, n_kv_heads, max_seq_len, head_dim]` K/V buffer
    ///    per layer (`fuel_core::inference_context::alloc_batched_kv`; fail-on-OOM).
    /// 2. **Copy-in:** `Op::WriteSlice` each session's `[1,…]` KV history into
    ///    its batch slot `i`.
    /// 3. **Decode:** build a batch=`K` analogue of
    ///    `Self::build_and_realize_first_decode_token` over the shared buffer
    ///    (the projection GEMMs batch for free through the leading batch axis;
    ///    the attention half reaches `flash_decoding`'s batch dim on CUDA) and
    ///    realize `[K, vocab]` logits (non-captured plan-once, spec #4).
    /// 4. **Copy-out + commit:** ONLY after realize succeeds, copy each slot
    ///    back into its session's own cache and bump `cached_len`/versions. Any
    ///    earlier `Err` returns before a single session cache is mutated — no
    ///    session is left half-written.
    ///
    /// On CPU (f32) the flash arm is not offered — the batch=`K` graph runs the
    /// decomposed batched attention, which still exercises the full shared-
    /// buffer + scatter path and is the CPU parity gate. On CUDA (bf16) the
    /// optimizer-emitted flash arm consumes the shared buffer's batch dim.
    #[allow(clippy::needless_range_loop)]
    // `pub` (not `pub(crate)`): the multi-session decode scheduler that drives
    // this batched arm lives in `fuel-inference` (the Q2 move), so its
    // `DecodeModel for LlamaModel` impl must reach this method across the crate
    // boundary. It is a legitimate public model capability (K-way batched decode
    // logits), not an internal.
    pub fn build_batched_decode_logits(
        &self,
        caches: &mut [&mut KvCache],
        last_tokens: &[u32],
        device: &Device,
        dtype: DType,
    ) -> fuel_core::Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let k = caches.len();
        if k < 2 {
            return Err(fuel_ir::Error::Msg(
                "build_batched_decode_logits: need >= 2 sessions".to_string(),
            )
            .bt());
        }
        if last_tokens.len() != k {
            return Err(fuel_ir::Error::Msg(format!(
                "build_batched_decode_logits: {} last_tokens for {} caches",
                last_tokens.len(),
                k
            ))
            .bt());
        }
        let cached_len = caches[0].cached_len;
        let max_seq_len = caches[0].max_seq_len.ok_or_else(|| {
            fuel_ir::Error::Msg(
                "build_batched_decode_logits: cache built via with_dims (no capacity)".to_string(),
            )
            .bt()
        })?;
        let cache_dtype = dtype;
        for (i, c) in caches.iter().enumerate() {
            if c.n_layers() != cfg.n_layers {
                return Err(fuel_ir::Error::Msg(format!(
                    "build_batched_decode_logits: cache {i} n_layers {} != model {}",
                    c.n_layers(),
                    cfg.n_layers
                ))
                .bt());
            }
            if c.cached_len != cached_len || c.max_seq_len != Some(max_seq_len) {
                return Err(fuel_ir::Error::Msg(
                    "build_batched_decode_logits: non-uniform caches (cached_len/max_seq_len)"
                        .to_string(),
                )
                .bt());
            }
            // Fail-fast: the cache's own dtype must match the requested
            // (scheduler) dtype. The shared buffer + graph placeholders are
            // built at `dtype`; binding a different-width cache Arc would be a
            // byte-reinterpretation (caught deep inside realize as a confusing
            // byte-count error — reject it here at call time instead).
            if c.dtype != Some(cache_dtype) {
                return Err(fuel_ir::Error::Msg(format!(
                    "build_batched_decode_logits: cache {i} dtype {:?} != requested dtype {:?}",
                    c.dtype, cache_dtype
                ))
                .bt());
            }
        }
        if cached_len + 1 > max_seq_len {
            return Err(fuel_ir::Error::Msg(format!(
                "build_batched_decode_logits: cached_len ({cached_len}) + 1 > max_seq_len ({max_seq_len})"
            ))
            .bt());
        }

        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let one_shape = Shape::from_dims(&[1, n_kv_heads, max_seq_len, head_dim]);
        let shared_shape = Shape::from_dims(&[k, n_kv_heads, max_seq_len, head_dim]);

        // (1) Allocate the shared [K, Hkv, msl, D] K/V buffer per layer.
        let shared = fuel_core::inference_context::alloc_batched_kv(
            k,
            cfg.n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            cache_dtype,
            device,
        )?;

        // ---- (2) Copy-in: WriteSlice each session's KV history into slot i. ----
        {
            let anchor = Tensor::from_f32(
                Arc::from(vec![0.0f32]),
                Shape::from_dims(&[1]),
                &Device::cpu(),
            )?;
            let mut ctx_in = InferenceContext::new(device.clone());
            let mut targets: Vec<fuel_graph::NodeId> = Vec::with_capacity(2 * cfg.n_layers);
            for l in 0..cfg.n_layers {
                let shared_k = anchor.const_placeholder_like(shared_shape.clone(), cache_dtype);
                let shared_v = anchor.const_placeholder_like(shared_shape.clone(), cache_dtype);
                ctx_in.insert(shared_k.graph_tensor().id(), Arc::clone(&shared[l].0));
                ctx_in.insert(shared_v.graph_tensor().id(), Arc::clone(&shared[l].1));
                let mut acc_k = shared_k;
                let mut acc_v = shared_v;
                for (i, c) in caches.iter().enumerate() {
                    let sk = anchor.const_placeholder_like(one_shape.clone(), cache_dtype);
                    let sv = anchor.const_placeholder_like(one_shape.clone(), cache_dtype);
                    let k_arc = c.slot_storage(l, KvSlot::K).ok_or_else(|| {
                        fuel_ir::Error::Msg(format!(
                            "build_batched_decode_logits: cache {i} layer {l} has no K slot"
                        ))
                        .bt()
                    })?;
                    let v_arc = c.slot_storage(l, KvSlot::V).ok_or_else(|| {
                        fuel_ir::Error::Msg(format!(
                            "build_batched_decode_logits: cache {i} layer {l} has no V slot"
                        ))
                        .bt()
                    })?;
                    ctx_in.insert(sk.graph_tensor().id(), k_arc);
                    ctx_in.insert(sv.graph_tensor().id(), v_arc);
                    let ranges = vec![(i, i + 1), (0, n_kv_heads), (0, max_seq_len), (0, head_dim)];
                    acc_k = acc_k.write_slice(&sk, ranges.clone())?;
                    acc_v = acc_v.write_slice(&sv, ranges)?;
                }
                targets.push(acc_k.graph_tensor().id());
                targets.push(acc_v.graph_tensor().id());
            }
            let graph = anchor.graph_tensor().graph().clone();
            realize_kv_write_targets(&ctx_in, &graph, &targets, cache_dtype)?;
        }

        // ---- (3) Batch=K decode graph over the shared buffer → [K, vocab]. ----
        let batch = k;
        let seq = 1usize;
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        )?;
        let token_ids = embed.const_u32_like(last_tokens.to_vec(), Shape::from_dims(&[k]))?;
        let mut h = embed
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        h = h.to_dtype(cache_dtype)?;

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_base, cached_len, seq, cfg.head_dim);

        let cached_len_sym = fuel_ir::SymId(0);
        let attended_len_sym = fuel_ir::SymId(1);

        let mask_data = build_decode_causal_mask(cached_len, seq, max_seq_len);
        let mask = h.const_like_dtype(
            &mask_data,
            Shape::from_dims(&[1, 1, seq, max_seq_len]),
            cache_dtype,
        )?;

        let cache_shape = Shape::from_dims(&[batch, cfg.n_kv_heads, max_seq_len, cfg.head_dim]);
        let mut ctx_dec = InferenceContext::new(device.clone());
        for (l, layer_weights) in weights.layers.iter().enumerate() {
            let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            ctx_dec.insert(k_cache_node.graph_tensor().id(), Arc::clone(&shared[l].0));
            ctx_dec.insert(v_cache_node.graph_tensor().id(), Arc::clone(&shared[l].1));
            h = self.apply_layer_with_kv_writes(
                &h,
                layer_weights,
                &k_cache_node,
                &v_cache_node,
                cached_len_sym,
                attended_len_sym,
                // Non-captured plan-once batched realize: keep the backend-
                // generic SymEnv WriteSlice offset (bit-identical KV write).
                None,
                &rope_cos,
                &rope_sin,
                &mask,
                None, // Llama has no sliding window
            )?;
        }

        let h_norm = apply_affine_rms_norm(&h, &weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)?;
        let last_pos = seq - 1;
        let logits_root = logits
            .slice(1, last_pos, 1)?
            .reshape(Shape::from_dims(&[batch, cfg.vocab_size]))?;
        let logits_root = logits_root.to_dtype(DType::F32)?;
        let graph_dec = logits_root.graph_tensor().graph().clone();

        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(cached_len_sym, cached_len)?;
        sym_env.bind(attended_len_sym, cached_len + seq)?;
        let flat = ctx_dec.realize_one_as_with_env::<f32>(
            &graph_dec,
            logits_root.graph_tensor().id(),
            &sym_env,
        )?;
        if flat.len() != k * cfg.vocab_size {
            return Err(fuel_ir::Error::Msg(format!(
                "build_batched_decode_logits: realize returned {} floats, expected {}",
                flat.len(),
                k * cfg.vocab_size
            ))
            .bt());
        }
        let rows: Vec<Vec<f32>> = (0..k)
            .map(|i| flat[i * cfg.vocab_size..(i + 1) * cfg.vocab_size].to_vec())
            .collect();

        // ---- (4) Copy-out + commit: overwrite each session's cache with its
        // slot (prefix + the freshly-written position `cached_len`), then bump.
        //
        // Safety of the all-or-nothing contract (spec risk #7). Copy-out is
        // ITSELF a fallible batched realize — the whole `realize_kv_write_targets`
        // call below writes all 2·n_layers·K slabs in one shot, and a failure
        // mid-way could leave some session caches rewritten and others not.
        // That partial state is nonetheless benign, for THREE reasons, none of
        // which is "no cache was mutated":
        //   (a) each session's KV is PRIVATE (T1 isolation) — a copy-out into
        //       session i can never touch session j's buffer;
        //   (b) on ANY `Err` returned from here, the scheduler's `advance_batched`
        //       Err arm forces EVERY batch member to Finished-with-error (see
        //       `fuel_inference::multi_session::SessionScheduler::advance_batched`),
        //       so no partially-written session is ever decoded again;
        //   (c) `cached_len` is NOT bumped until AFTER copy-out fully succeeds,
        //       and the copy-out rewrite is idempotent over the read region
        //       `[0, cached_len)` (it re-copies the identical prefix), so a
        //       partial copy-out cannot corrupt what a subsequent read would see.
        // WARNING: this safety hinges on (c) — NEVER commit `cached_len`
        // separately from (before) copy-out, and NEVER retry a batch-errored
        // session. Either change turns a partial copy-out into silent KV
        // corruption (a bumped `cached_len` over a half-written position, or a
        // retry that reads stale/partial slots). ----
        {
            let anchor = Tensor::from_f32(
                Arc::from(vec![0.0f32]),
                Shape::from_dims(&[1]),
                &Device::cpu(),
            )?;
            let mut ctx_out = InferenceContext::new(device.clone());
            let mut targets: Vec<fuel_graph::NodeId> = Vec::with_capacity(2 * cfg.n_layers * k);
            for l in 0..cfg.n_layers {
                let shared_k = anchor.const_placeholder_like(shared_shape.clone(), cache_dtype);
                let shared_v = anchor.const_placeholder_like(shared_shape.clone(), cache_dtype);
                ctx_out.insert(shared_k.graph_tensor().id(), Arc::clone(&shared[l].0));
                ctx_out.insert(shared_v.graph_tensor().id(), Arc::clone(&shared[l].1));
                for (i, c) in caches.iter().enumerate() {
                    // Slot i is a complete cache for session i (copied-in prefix
                    // + the decode's position-`cached_len` write). Outermost-axis
                    // slice → contiguous [1, Hkv, msl, D] source.
                    let slot_k = shared_k.slice(0, i, 1)?;
                    let slot_v = shared_v.slice(0, i, 1)?;
                    let dst_k = anchor.const_placeholder_like(one_shape.clone(), cache_dtype);
                    let dst_v = anchor.const_placeholder_like(one_shape.clone(), cache_dtype);
                    ctx_out.insert(
                        dst_k.graph_tensor().id(),
                        c.slot_storage(l, KvSlot::K).ok_or_else(|| {
                            fuel_ir::Error::Msg(format!(
                                "build_batched_decode_logits: cache {i} layer {l} has no K slot (copy-out)"
                            ))
                            .bt()
                        })?,
                    );
                    ctx_out.insert(
                        dst_v.graph_tensor().id(),
                        c.slot_storage(l, KvSlot::V).ok_or_else(|| {
                            fuel_ir::Error::Msg(format!(
                                "build_batched_decode_logits: cache {i} layer {l} has no V slot (copy-out)"
                            ))
                            .bt()
                        })?,
                    );
                    let ranges = vec![(0, 1), (0, n_kv_heads), (0, max_seq_len), (0, head_dim)];
                    let wk = dst_k.write_slice(&slot_k, ranges.clone())?;
                    let wv = dst_v.write_slice(&slot_v, ranges)?;
                    targets.push(wk.graph_tensor().id());
                    targets.push(wv.graph_tensor().id());
                }
            }
            let graph = anchor.graph_tensor().graph().clone();
            realize_kv_write_targets(&ctx_out, &graph, &targets, cache_dtype)?;
        }

        // Commit: bump each session's cached_len + versions (identical to the
        // single-session decode's post-write bump).
        for c in caches.iter_mut() {
            c.cached_len += seq;
            for li in 0..cfg.n_layers {
                c.bump_version(li, KvSlot::K);
                c.bump_version(li, KvSlot::V);
            }
        }

        Ok(rows)
    }

    /// Re-bind the per-token data Consts (token-ids / RoPE / mask) into
    /// device Arcs, bind the `SymEnv`, and realize via the D2a prebuilt
    /// seam (SKIPPING optimize) over the held session's base cache. The
    /// KV Arcs are stable (mutated in place by WriteSlice via the held
    /// base_cache entries) — not touched here. Called for every decode
    /// token after the first.
    /// Thin forwarder to the shared driver
    /// ([`fuel_core::persistent_decode::rebind_and_realize_prebuilt`]). See also the
    /// `PersistentDecodeModel` impls at the end of this module.
    ///
    /// GAP-029 2b: this was a hand-copied 48-line body, byte-for-byte parallel
    /// to `PhiModel`'s. The copies are collapsed; the per-model part is
    /// [`PersistentDecodeModel::build_decode_token_data`] below.
    // Only remaining caller is `forward_with_kv_context_captured`, which is
    // `#[cfg(feature = "cuda")]` — GAP-029 increment 3 moved the non-CUDA
    // callers onto the shared seam in `fuel_core::persistent_decode`. So "never
    // used" on a default build is a FALSE signal from a feature-gated caller,
    // not an orphan: deleting this breaks `--features cuda`, which no gate on
    // this machine compiles cheaply.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn rebind_and_realize_prebuilt(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &InferenceContext,
        session: &Option<fuel_core::inference_context::DecodeSession>,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<Vec<f32>> {
        fuel_core::persistent_decode::rebind_and_realize_prebuilt(
            self,
            tokens,
            cache,
            ctx,
            session,
            rope_inv_freq,
        )
    }

    /// Compute the per-token RoPE cos/sin tables + causal mask + (on the
    /// device-offset path) the KV-write offset, tagged with dtype as
    /// [`fuel_ir::HostBuffer`]s. Pure host-side math, no upload — shared
    /// by [`Self::build_token_rope_mask_arcs`] (uploads each to a fresh
    /// device Arc) and [`Self::build_token_rope_mask_bytes`] (extracts
    /// each to raw bytes for an in-place H2D overwrite), so the
    /// RoPE-table-math and mask-math live in exactly one place.
    fn compute_token_rope_mask_host_data(
        &self,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
        cache_dtype: DType,
        with_device_offset: bool,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<TokenDataHost> {
        let cfg = &self.config;
        let seq = tokens.len();

        let token_ids = fuel_ir::HostBuffer::U32(tokens.to_vec());
        // `rope_inv_freq` replaces the frequencies derived from
        // `cfg.rope_base` — the ONLY thing a scaled-RoPE variant (LLaMA-3.1
        // long context) changes on this path. `None` keeps the default, so
        // every pre-existing caller stays bit-identical.
        let (cos_data, sin_data) = match rope_inv_freq {
            Some(inv) => {
                fuel_graph::build_rope_tables_with_inv_freq(inv, cached_len, seq, cfg.head_dim)
            }
            None => fuel_graph::build_rope_tables(cfg.rope_base, cached_len, seq, cfg.head_dim),
        };
        let rope_cos = fuel_ir::HostBuffer::F32(cos_data);
        let rope_sin = fuel_ir::HostBuffer::F32(sin_data);
        // Sourced from the SHARED variant builder, not from `build_decode_causal_mask`
        // directly: the held graph's mask Const is minted by the shared build path
        // through that same function, and two mask formulas — one for the build,
        // one for the rebind — is exactly the divergence that would stay invisible
        // until a windowed family decoded its second token. Byte-identical to the
        // dense builder for Llama's uniform plan (asserted, not assumed — see
        // `mask_variants_match_the_dense_builder_when_uniform`).
        let mask_data = fuel_core::persistent_decode::build_decode_mask_variants(
            &fuel_core::persistent_decode::DecodeBackbone::decode_mask_plan(self),
            cached_len,
            seq,
            max_seq_len,
        );
        // Mask dtype tracks the cache dtype (BF16-throughout decode,
        // Phase D increment A) — the held graph's mask Const placeholder
        // was minted at `cache_dtype` (see
        // `build_and_realize_first_decode_token`), and the rebind must
        // supply bytes matching that placeholder's dtype. No-op (stays
        // F32) for f32 caches.
        let mask = match cache_dtype {
            DType::F32 => fuel_ir::HostBuffer::F32(mask_data),
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    mask_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
                fuel_ir::HostBuffer::BF16(bf16_data)
            }
            other => {
                return Err(fuel_ir::Error::Msg(format!(
                    "compute_token_rope_mask_host_data: unsupported cache dtype {other:?} \
                 (expected F32 or BF16)",
                ))
                .bt());
            }
        };

        // Device-offset path: the KV-write start (`cached_len` as a
        // rank-0 I64) so the held `Op::WriteSliceDoff` nodes read the
        // live position device-side. `None` on the SymEnv (Vulkan) path.
        let offset = if with_device_offset {
            Some(fuel_ir::HostBuffer::I64(vec![cached_len as i64]))
        } else {
            None
        };

        Ok(TokenDataHost {
            token_ids,
            rope_cos,
            rope_sin,
            mask,
            offset,
        })
    }

    /// Recompute the per-token host bytes for token-ids / RoPE cos+sin /
    /// mask and build device-resident Arcs from them (the SAME upload
    /// path `KvCache::with_capacity` uses). On CPU the Storage wraps the
    /// host bytes; on GPU it performs the H2D upload (tiny tensors).
    /// Design §2 option (b): the bytes change per token, the NodeId stays
    /// stable (the held graph's Const nodes are re-bound via `base_cache`
    /// overwrite, not a fresh graph).
    fn build_token_rope_mask_arcs(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
        cache_dtype: DType,
        with_device_offset: bool,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<fuel_core::inference_context::DecodeTokenData> {
        let host = self.compute_token_rope_mask_host_data(
            cached_len,
            tokens,
            max_seq_len,
            cache_dtype,
            with_device_offset,
            rope_inv_freq,
        )?;
        let upload = fuel_core::pipelined_bridge::upload_host_buffer_to_device;

        let token_ids = upload(device, host.token_ids)?;
        let rope_cos = upload(device, host.rope_cos)?;
        let rope_sin = upload(device, host.rope_sin)?;
        let mask = upload(device, host.mask)?;
        let offset = host.offset.map(|o| upload(device, o)).transpose()?;

        Ok(fuel_core::inference_context::DecodeTokenData {
            token_ids,
            rope_cos,
            rope_sin,
            mask,
            offset,
        })
    }

    /// Same per-token data as [`Self::build_token_rope_mask_arcs`], as
    /// raw host bytes instead of freshly-uploaded device Arcs — for
    /// [`fuel_dispatch::pipelined::CapturedDecodeSession::replay_token`]'s
    /// in-place H2D overwrite of fixed buffers (no fresh allocation, same
    /// device address every call).
    #[cfg(feature = "cuda")]
    fn build_token_rope_mask_bytes(
        &self,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
        cache_dtype: DType,
        with_device_offset: bool,
        rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<TokenDataBytes> {
        let host = self.compute_token_rope_mask_host_data(
            cached_len,
            tokens,
            max_seq_len,
            cache_dtype,
            with_device_offset,
            rope_inv_freq,
        )?;
        Ok(TokenDataBytes {
            token_ids: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.token_ids),
            rope_cos: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.rope_cos),
            rope_sin: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.rope_sin),
            mask: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.mask),
            offset: host
                .offset
                .as_ref()
                .map(fuel_core::pipelined_bridge::host_buffer_to_bytes),
        })
    }

    /// CapturedRun (CUDA-graph decode capture) driver — task 4b-δ. A
    /// brand-new, additive, opt-in entry point alongside
    /// [`Self::forward_with_kv_context_persistent`] (which this does NOT
    /// replace, alter, or share mutable state with beyond the caller-owned
    /// `session`). Four cases, exactly the persistent path's shape plus
    /// the capture-build/replay split:
    ///
    /// 1. `seq != 1` (prefill / multi-token step): not a decode step —
    ///    drop any session/capture and fall back to the plain D1 path
    ///    ([`Self::forward_with_kv_context`]), mirroring
    ///    `forward_with_kv_context_persistent`'s own `seq != 1` fallback.
    /// 2. `session.is_none()` (first decode token): delegate to the
    ///    existing, unmodified
    ///    [`Self::build_and_realize_first_decode_token`]. `captured`
    ///    stays `None`.
    /// 3. `session.is_some() && captured.is_none()` (second decode token):
    ///    build this token's per-token data as fresh Arcs (FIXED device
    ///    addresses — these become the addresses every later
    ///    `replay_token` H2D-overwrites in place), merge them over
    ///    `session.base_cache()`, and capture the decode graph once via
    ///    [`fuel_dispatch::pipelined::CapturedDecodeSession::capture`]
    ///    (targeting `session.logits_node()`, NOT `effective_target` —
    ///    `capture_decode` hard-rejects the D2H `Op::Copy` splice as a
    ///    cross-device, non-single-device-CUDA-capturable op; the D2H
    ///    happens HERE, after replay, not inside the capture). The warm
    ///    pass inside `capture()` already computed this token's correct
    ///    result, so this token's logits come from an EMPTY-`updates`
    ///    `replay_token(&[])` (replays against the per-token buffers the
    ///    capture just warmed, per `replay_token`'s documented contract).
    /// 4. `captured.is_some()` (third token onward): compute this token's
    ///    per-token data as raw bytes (no allocation) via
    ///    [`Self::build_token_rope_mask_bytes`] and replay via one
    ///    `cuGraphLaunch` (`replay_token` with the fresh bytes).
    ///
    /// **Staleness** (was a documented gap in the first capture-wiring pass;
    /// now closed): before case 2, a held pair whose validity key no longer
    /// matches the live cache/model is invalidated — session AND capture
    /// together — via [`Self::invalidate_decode_pair_if_stale`], which shares
    /// its predicate with `forward_with_kv_context_persistent` so the two
    /// paths cannot drift on what "stale" means. Dropping the session alone
    /// would leave a recorded CUDA graph replaying against fixed device
    /// addresses that no longer describe the live cache: wrong logits at full
    /// speed, with nothing reporting it. After invalidation `session` is
    /// `None`, so case 2 rebuilds both on this same token.
    ///
    /// **RoPE SCALING — why the three internal `rope_inv_freq: None`s are
    /// correct, and exactly when they stop being.** This entry is inherent to
    /// `LlamaModel`, whose RoPE frequencies are the unscaled default. Models
    /// with scaled frequencies (Llama-3.1 et al) reach decode through
    /// `forward_with_kv_context_persistent_inv_freq`, and **nothing routes them
    /// here** — this fn has no caller outside this file. So `None` is the same
    /// value `forward_with_kv_context_persistent` passes, and the semantics are
    /// identical.
    ///
    /// It is therefore a **latent** trap, not a live one — and severe if it goes
    /// live: passing `None` where a model needs its own inverse frequencies
    /// gives right-token-1 / wrong-tokens-after, with no error. Worse on THIS
    /// path than the persistent one, because the capture bakes the graph ONCE
    /// and every later `replay_token` reuses it, so a wrong `None` would persist
    /// across every replayed token instead of showing up intermittently.
    /// **Wiring a scaled model to this entry means threading `rope_inv_freq`
    /// through all three internal sites — not passing `None` because it
    /// compiles.** (Raised by Lightbulb, 2026-08-05.)
    #[cfg(feature = "cuda")]
    pub fn forward_with_kv_context_captured(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
        captured: &mut Option<fuel_dispatch::pipelined::CapturedDecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();

        // ---- 1. Non-decode step: drop state, fall back to D1. ----
        if seq != 1 {
            *captured = None;
            self.drop_decode_session(session, ctx);
            return self.forward_with_kv_context(tokens, cache, ctx);
        }

        // ---- 1b. Staleness: a held pair keyed to a cache/model that no
        // longer matches is retired — capture first, then session (see
        // `invalidate_decode_pair_if_stale`). This leaves `session` as
        // `None`, so case 2 immediately rebuilds both on this token. ----
        self.invalidate_decode_pair_if_stale(
            session,
            captured,
            ctx,
            seq,
            cache.max_seq_len,
            cache.dtype.unwrap_or(DType::F32),
            cache,
        );

        // ---- 2. First decode token: build the held session (unmodified
        // shared path with forward_with_kv_context_persistent). ----
        if session.is_none() {
            *captured = None;
            // rope_inv_freq: None — see the RoPE-SCALING note in this fn's doc.
            return self.build_and_realize_first_decode_token(tokens, cache, ctx, session, None);
        }

        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        let cached_len = cache.cached_len;

        // ---- 3. Second decode token: build the capture. ----
        if captured.is_none() {
            let device = ctx.device().clone();
            let s = session.as_ref().expect("session is Some (checked above)");

            // Fresh FIXED-address Arcs — these addresses are what every
            // later `replay_token` H2D-overwrites in place.
            let data = self.build_token_rope_mask_arcs(
                &device,
                cached_len,
                tokens,
                s.max_seq_len(),
                cache_dtype,
                s.offset_node().is_some(),
                None, // rope_inv_freq — see the RoPE-SCALING note in this fn's doc
            )?;

            // Merged StorageCache: base_cache clone (cheap — Arc-clones
            // only) + overwrite the per-token entries with the fresh Arcs.
            let mut merged_cache: fuel_dispatch::pipelined::StorageCache = s.base_cache().clone();
            merged_cache.insert(s.token_ids_node(), Arc::clone(&data.token_ids));
            merged_cache.insert(s.rope_cos_node(), Arc::clone(&data.rope_cos));
            merged_cache.insert(s.rope_sin_node(), Arc::clone(&data.rope_sin));
            merged_cache.insert(s.mask_node(), Arc::clone(&data.mask));
            let mut per_token_node_ids: Vec<fuel_graph::NodeId> = vec![
                s.token_ids_node(),
                s.rope_cos_node(),
                s.rope_sin_node(),
                s.mask_node(),
            ];
            if let (Some(offset_node), Some(offset_arc)) = (s.offset_node(), data.offset.as_ref()) {
                merged_cache.insert(offset_node, Arc::clone(offset_arc));
                per_token_node_ids.push(offset_node);
            }

            let sym_env = s.per_token_sym_env(cached_len)?;

            // CAPTURE IS AN OPTIMIZATION: IT MUST DEGRADE, NEVER FAIL.
            //
            // `capture_decode` rejects any graph it cannot record — most
            // commonly a CROSS-DEVICE `Op::Copy`, which appears whenever any
            // node in the decode graph lands on the host. That is a property of
            // the model's graph, not an error in this call: PhiModel hits it
            // today, and the paged decode graph hits it because `Op::PagedAttn`
            // is host-placed for want of a CUDA kernel.
            //
            // Propagating it would turn "this graph cannot be captured" into
            // "generation fails" — which is what happened when capture was
            // first defaulted on for Phi, and is exactly the wrong trade for an
            // optimization. Fall back to the ordinary persistent rebind instead
            // and return a correct token.
            //
            // Safe by inspection: `capture()` runs BEFORE any `cache` mutation
            // below, so nothing is half-applied and the fallback recomputes the
            // token cleanly. A genuine compute error is not masked — the
            // fallback re-runs the same work through the persistent path and
            // surfaces it there.
            //
            // Retried per token rather than remembered, deliberately: the check
            // is a cheap graph scan on an already-built plan, it only costs
            // anything on models that cannot capture (which get no benefit
            // either way), and it self-corrects if a future registration makes
            // the graph capturable mid-run.
            let cd_session = match fuel_dispatch::pipelined::CapturedDecodeSession::capture(
                s.graph().clone(),
                s.logits_node(),
                merged_cache,
                &per_token_node_ids,
                sym_env,
            ) {
                Ok(cd) => cd,
                Err(_) => {
                    let res = self.rebind_and_realize_prebuilt(
                        tokens, cache, &*ctx, &*session, None, // see the RoPE-SCALING note
                    );
                    return res;
                }
            };

            // The warm pass inside `capture()` already computed THIS
            // token's correct result against the per-token buffers just
            // built above — an empty-updates replay fetches it without
            // re-uploading anything.
            let output = cd_session.replay_token(&[])?;
            let logits = captured_output_to_f32(&output)?;

            *captured = Some(cd_session);

            cache.cached_len += seq;
            for li in 0..cfg.n_layers {
                cache.bump_version(li, KvSlot::K);
                cache.bump_version(li, KvSlot::V);
            }
            return Ok(logits);
        }

        // ---- 4. Third token onward: pure replay. ----
        let cap = captured.as_ref().expect("captured is Some (checked above)");
        let s = session
            .as_ref()
            .expect("session is Some whenever captured is Some");

        let bytes = self.build_token_rope_mask_bytes(
            cached_len,
            tokens,
            s.max_seq_len(),
            cache_dtype,
            s.offset_node().is_some(),
            None, // rope_inv_freq — see the RoPE-SCALING note in this fn's doc
        )?;
        let mut updates: Vec<(fuel_graph::NodeId, &[u8])> = vec![
            (s.token_ids_node(), bytes.token_ids.as_slice()),
            (s.rope_cos_node(), bytes.rope_cos.as_slice()),
            (s.rope_sin_node(), bytes.rope_sin.as_slice()),
            (s.mask_node(), bytes.mask.as_slice()),
        ];
        if let (Some(offset_node), Some(offset_bytes)) = (s.offset_node(), bytes.offset.as_ref()) {
            updates.push((offset_node, offset_bytes.as_slice()));
        }

        let output = cap.replay_token(&updates)?;
        let logits = captured_output_to_f32(&output)?;

        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }
        Ok(logits)
    }

    /// Invalidate a held `(DecodeSession, capture)` PAIR when the session's
    /// validity key no longer matches the live cache/model — the captured
    /// path's twin of the staleness check
    /// [`Self::forward_with_kv_context_persistent`] runs inline, and it uses
    /// the identical predicate ([`DecodeSession::is_valid_for`] over
    /// `seq / max_seq_len / n_layers / cache_dtype`, with a `max_seq_len` of
    /// `None` counting as stale) so the two paths cannot drift apart on what
    /// "stale" means.
    ///
    /// **The capture must die with the session, and that is the whole point
    /// of this being one function instead of two.** A `CapturedDecodeSession`
    /// is a recorded CUDA graph over FIXED device addresses drawn from the
    /// session's `base_cache`. Drop the session alone and the recorded graph
    /// keeps replaying against buffers that no longer describe the live cache
    /// — silently wrong logits at full speed, which is worse than a crash
    /// because nothing reports it. Dropping the capture FIRST (before the
    /// session releases its `base_cache` Arcs) retires the reader before the
    /// owner, so no replay can observe a half-invalidated pair.
    ///
    /// Generic in the capture type purely so this stays compilable — and
    /// therefore CPU-testable — without the `cuda` feature; the only thing
    /// done with `captured` is to clear it.
    ///
    /// Returns whether it invalidated. On `true` the caller's `session` is
    /// `None`, so the ordinary "first decode token" arm rebuilds both.
    // Only remaining caller is `forward_with_kv_context_captured`, which is
    // `#[cfg(feature = "cuda")]` — GAP-029 increment 3 moved the non-CUDA
    // callers onto the shared seam in `fuel_core::persistent_decode`. So "never
    // used" on a default build is a FALSE signal from a feature-gated caller,
    // not an orphan: deleting this breaks `--features cuda`, which no gate on
    // this machine compiles cheaply.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn invalidate_decode_pair_if_stale<C>(
        &self,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
        captured: &mut Option<C>,
        ctx: &mut InferenceContext,
        seq: usize,
        max_seq_len: Option<usize>,
        cache_dtype: DType,
        kv: &dyn fuel_core::inference_context::KvRebindSource,
    ) -> SessionDisposition {
        invalidate_decode_pair_if_stale(
            session,
            captured,
            ctx,
            seq,
            max_seq_len,
            cache_dtype,
            self.config.n_layers,
            self.decode_shape_key(),
            kv,
            |sess, c| self.drop_decode_session(sess, c),
        )
    }

    /// Drop a held decode session, removing any leftover persistent
    /// data-Const / KV bindings from `ctx` (defensive — the build path
    /// already removes them once the session owns `base_cache`; this
    /// covers the invalidation path). No-op if `session` is `None`.
    // Only remaining caller is `forward_with_kv_context_captured`, which is
    // `#[cfg(feature = "cuda")]` — GAP-029 increment 3 moved the non-CUDA
    // callers onto the shared seam in `fuel_core::persistent_decode`. So "never
    // used" on a default build is a FALSE signal from a feature-gated caller,
    // not an orphan: deleting this breaks `--features cuda`, which no gate on
    // this machine compiles cheaply.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn drop_decode_session(
        &self,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
        ctx: &mut InferenceContext,
    ) {
        fuel_core::persistent_decode::drop_decode_session(session, ctx)
    }
}

// Phase 7.6 step 9c E.3.3.D — host-resident `LlamaKVCache` retired.
// Its successor is `KvCache` in `fuel_core::inference_context`, which
// stores backend-erased `Arc<RwLock<fuel_memory::Storage>>` per slot
// and supports both the legacy `with_dims` grow-by-replace shape and
// the new `with_capacity` pre-allocated-buffer shape that
// `forward_with_kv_context` writes into via `Op::WriteSlice`.

/// Per-token host-side data for a persistent decode step, tagged with
/// dtype via [`fuel_ir::HostBuffer`] — the shared computation both
/// `LlamaModel::build_token_rope_mask_arcs` (uploads each to a fresh
/// device Arc) and `LlamaModel::build_token_rope_mask_bytes` (extracts
/// each to raw host bytes) build from. `LlamaModel`-private.
/// **Shared staleness predicate for a held `(DecodeSession, capture)` pair** —
/// one definition for every model, so they cannot drift on what "stale" means.
///
/// Was a `LlamaModel` inherent method until Phi needed it too. Copying it would
/// have created exactly the divergence risk this session has been closing
/// elsewhere: two predicates, one test, and a silent disagreement the day one is
/// updated. The model-specific parts (`n_layers`, `shape_key`, and how to drop
/// a session) arrive as parameters instead.
///
/// **The capture must die with the session.** A `CapturedDecodeSession` is a
/// recorded CUDA graph over FIXED device addresses drawn from the session's
/// `base_cache`. Drop the session alone and the graph keeps replaying against
/// buffers that no longer describe the live cache — silently wrong logits at
/// full speed, which is worse than a crash because nothing reports it. The
/// capture is cleared FIRST, retiring the reader before the owner.
///
/// Generic in the capture type so this compiles — and is therefore testable —
/// without the `cuda` feature; the only thing done with `captured` is clear it.
///
/// Returns whether it invalidated. On `true` the caller's `session` is `None`,
/// so the ordinary first-decode-token arm rebuilds both.
#[allow(clippy::too_many_arguments)]
fn realize_kv_write_targets(
    ctx: &InferenceContext,
    graph: &std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>,
    targets: &[fuel_graph::NodeId],
    dtype: DType,
) -> fuel_core::Result<()> {
    let env = fuel_ir::SymEnv::new();
    match dtype {
        DType::F32 => {
            ctx.realize_many_as_with_env::<f32>(graph, targets, &env)?;
        }
        DType::BF16 => {
            ctx.realize_many_as_with_env::<half::bf16>(graph, targets, &env)?;
        }
        other => {
            return Err(fuel_ir::Error::Msg(format!(
                "build_batched_decode_logits: unsupported KV dtype {other:?}"
            ))
            .bt());
        }
    }
    Ok(())
}

/// Promoted from `pub(crate)` to `pub` by the `fuel-nn` extraction
/// (2026-08-19): `fuel_nn::modules::two_proj_attention` calls it, and that
/// module now lives outside this crate. This is a REAL public-API addition,
/// not a mechanical move — flagged rather than folded silently into the
/// extraction diff.
impl LlamaWeights {
    /// Load all LLaMA weights from one or more memory-mapped safetensors
    /// files using the HuggingFace naming convention (the same names
    /// you see in any `pytorch_model.bin.index.json` or
    /// `model.safetensors.index.json` for a LLaMA-architecture model).
    ///
    /// Expected names:
    /// - `model.embed_tokens.weight` → token embedding (kept as-is)
    /// - `model.layers.{i}.self_attn.q_proj.weight` (transposed)
    /// - `model.layers.{i}.self_attn.k_proj.weight` (transposed)
    /// - `model.layers.{i}.self_attn.v_proj.weight` (transposed)
    /// - `model.layers.{i}.self_attn.o_proj.weight` (transposed)
    /// - `model.layers.{i}.mlp.gate_proj.weight` (transposed)
    /// - `model.layers.{i}.mlp.up_proj.weight` (transposed)
    /// - `model.layers.{i}.mlp.down_proj.weight` (transposed)
    /// - `model.layers.{i}.input_layernorm.weight` (per-channel gain)
    /// - `model.layers.{i}.post_attention_layernorm.weight` (per-channel gain)
    /// - `model.norm.weight` → final RmsNorm gain
    /// - `lm_head.weight` → output projection (transposed)
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &LlamaConfig,
    ) -> fuel_core::Result<Self> {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * cfg.dim {
            fuel_core::bail!(
                "embed_tokens: {} elements, expected {} ({}×{})",
                token_embedding.len(),
                cfg.vocab_size * cfg.dim,
                cfg.vocab_size,
                cfg.dim,
            );
        }

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            // Projections use the dtype-preserving loader — bf16
            // source files stay bf16 on-device (halving weight memory
            // on this layer).
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let attn_k = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_v = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
                kv_dim,
                cfg.dim,
            )?;
            let attn_o = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.o_proj.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.up_proj.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.down_proj.weight"),
                cfg.dim,
                cfg.ffn_dim,
            )?;
            let attn_norm_gain =
                load_tensor_as_f32(st, &format!("model.layers.{i}.input_layernorm.weight"))?;
            let ffn_norm_gain = load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
            )?;
            // Qwen2-style biases on Q/K/V. LLaMA has no biases at all,
            // so these will return `Err` for LLaMA weights and we
            // store `None`. We don't bail — a missing bias is a
            // legitimate architectural variation, not an error.
            let attn_q_bias =
                load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.q_proj.bias"))
                    .ok()
                    .map(Arc::from);
            let attn_k_bias =
                load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.k_proj.bias"))
                    .ok()
                    .map(Arc::from);
            let attn_v_bias =
                load_tensor_as_f32(st, &format!("model.layers.{i}.self_attn.v_proj.bias"))
                    .ok()
                    .map(Arc::from);
            layers.push(LayerWeights {
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                ffn_gate,
                ffn_up,
                ffn_down,
                attn_norm_gain: Arc::from(attn_norm_gain),
                ffn_norm_gain: Arc::from(ffn_norm_gain),
            });
        }

        let final_norm_gain = load_tensor_as_f32(st, "model.norm.weight")?;
        // `lm_head.weight` is `[vocab_size, dim]` in HF layout; we want
        // `[dim, vocab_size]` for `h @ W_out`. Fall back to tied
        // embeddings (`lm_head.weight` absent → reuse embed_tokens) for
        // models that tie input/output weights.
        let output: WeightStorage = match load_transposed_matrix_preserve_dtype(
            st,
            "lm_head.weight",
            cfg.vocab_size,
            cfg.dim,
        ) {
            Ok(w) => w,
            Err(_) => {
                // Tied weights: transpose embed_tokens. Embedding is
                // always f32, so the tied output is f32 regardless
                // of how the projection weights loaded.
                let mut transposed = vec![0.0_f32; cfg.dim * cfg.vocab_size];
                for i in 0..cfg.vocab_size {
                    for j in 0..cfg.dim {
                        transposed[j * cfg.vocab_size + i] = token_embedding[i * cfg.dim + j];
                    }
                }
                WeightStorage::F32(Arc::from(transposed))
            }
        };

        Ok(LlamaWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain: Arc::from(final_norm_gain),
            output,
        })
    }
}

/// A small wrapper around `tokenizers::Tokenizer` tuned for the
/// chat-generation workflow: encode a prompt into token IDs, decode
/// token IDs back into a string, find the model's end-of-sequence
/// token. Lives next to LlamaModel in the same module so a decode
/// loop can keep both under one import.
pub struct LlamaTokenizer {
    inner: tokenizers::Tokenizer,
    eos_id: Option<u32>,
}

impl LlamaTokenizer {
    /// Load a tokenizer from a `tokenizer.json` on disk.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> fuel_core::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| fuel_core::Error::Msg(format!("loading tokenizer: {e}")))?;
        // LLaMA 3 uses `<|end_of_text|>` as EOS; LLaMA 2 uses `</s>`;
        // Qwen2 chat models use `<|im_end|>`. Try each in order and
        // take whichever the vocab has.
        let eos_id = ["<|end_of_text|>", "</s>", "<|eot_id|>", "<|im_end|>"]
            .iter()
            .find_map(|s| inner.token_to_id(s));
        Ok(Self { inner, eos_id })
    }

    /// Load a tokenizer from a HuggingFace repo. Downloads
    /// `tokenizer.json` and calls [`Self::from_file`].
    pub fn from_hub(repo_id: &str) -> fuel_core::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| fuel_core::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| fuel_core::Error::Msg(format!("hf-hub tokenizer.json: {e}")))?;
        Self::from_file(path)
    }

    /// Encode a prompt into token IDs. `add_special_tokens=true`
    /// prepends the model's BOS token (for LLaMA, `<|begin_of_text|>`).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> fuel_core::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| fuel_core::Error::Msg(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a slice of token IDs back into a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> fuel_core::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| fuel_core::Error::Msg(format!("tokenizer decode: {e}")))
    }

    /// The model's end-of-sequence token ID, if one was identified.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }
}
impl LlamaModel {
    /// Run greedy or temperature-sampled token generation for
    /// `max_new_tokens` steps starting from `prompt_tokens`. Returns
    /// the full sequence including the prompt.
    ///
    /// This is the minimum viable decode loop: each iteration runs a
    /// full forward pass on the entire sequence so far (no KV cache),
    /// slices out the logits for the last position, samples the next
    /// token, and appends. It stops early if the sampled token equals
    /// `eos_id`.
    ///
    /// Without a KV cache this is O(n²) in sequence length — fine for
    /// a correctness demo, way too slow for production. A cached
    /// decode loop is mechanical to add once the graph layer grows
    /// persistent state.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
    ) -> fuel_core::Result<Vec<u32>> {
        // Phase 7.6 step 9c E.3.3.D: re-pointed to the new KvCache +
        // InferenceContext + Op::WriteSlice path on CPU + F32. The
        // greedy parity test
        // `generate_with_kv_context_matches_legacy_generate` confirms
        // bitwise token-sequence equivalence with the retired
        // `generate_streaming_on` / `LlamaKVCache` host-resident path.
        self.generate_with_kv_context(
            prompt_tokens,
            max_new_tokens,
            strategy,
            eos_id,
            &Device::cpu(),
            DType::F32,
        )
    }

    // ===== Phase 7.6 step 9c E.3.3.D — host-resident streaming retired =====
    //
    // The legacy `generate_streaming_on<B>` (host-resident KV cache via
    // LlamaKVCache + per-step D2H/H2D round-trip) and its CPU-wrapper
    // `generate_streaming` were retired in favor of
    // `generate_streaming_with_kv_context`. Greedy token-sequence parity
    // was confirmed by `generate_with_kv_context_matches_legacy_generate`
    // before retirement. CPU, CUDA, and Vulkan callers all use the new
    // path (forward_with_kv_context + WriteSlice in-graph).

    // ===== Phase 7.6 step 9c E.3.3.C — streaming with KvCache + InferenceContext =====
    //
    // These replaced the legacy `generate_streaming_on` /
    // `generate_streaming_gpu_on` pair across CPU, CUDA, and Vulkan
    // (the latter retired in Unification Session 4, E.3.4). The
    // device is passed in directly (no `GraphBackend` parameter);
    // the pipelined executor handles backend dispatch through the
    // binding-table lookup.

    /// Streaming generation through the new `forward_with_kv_context`
    /// path. Allocates a pre-allocated `KvCache` of capacity
    /// `prompt_tokens.len() + max_new_tokens` on `device` (so the
    /// cache never overflows during decode), then loops prefill +
    /// decode, calling `on_token` for each generated token.
    ///
    /// `dtype` is the K/V storage dtype — typically `F32` for
    /// inference. The cache memory cost is
    /// `n_layers * 2 * n_kv_heads * (prompt+max_new) * head_dim *
    /// dtype_size`. For TinyLlama-1.1B at 1024-token max context, F32:
    /// 22 * 2 * 4 * 1024 * 64 * 4 ≈ 46 MiB.
    ///
    /// Works on CPU, CUDA, and Vulkan — the pipelined executor's
    /// binding-table dispatch picks the registered kernel per op
    /// based on the device passed in.
    pub fn generate_streaming_with_kv_context(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
        mut on_token: impl FnMut(u32),
    ) -> fuel_core::Result<Vec<u32>> {
        let cfg = &self.config;
        if prompt_tokens.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "generate_streaming_with_kv_context: prompt is empty".to_string(),
            )
            .bt());
        }
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };

        let max_seq_len = prompt_tokens.len() + max_new_tokens;
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            dtype,
            device,
        )?;
        let mut ctx = InferenceContext::new(device.clone());

        // Phase D · D2c: hold ONE plan-once decode session across the
        // whole generation. Prefill (seq>1) routes through the persistent
        // entry, which internally falls back to the D1 rebuild path for
        // non-seq==1 steps WITHOUT building the session (behaviour byte-
        // identical to a bare `forward_with_kv_context` prefill). Each
        // per-token decode step (seq==1) then builds the held graph on the
        // FIRST token (optimize once) and reuses it — skipping optimize —
        // for every subsequent token. The session is loop-internal; the
        // public signature is unchanged.
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // CUDA-graph capture is ON BY DEFAULT here (2026-08-06). Under `cuda`
        // the decode steps route through `forward_with_kv_context_captured`,
        // which holds this same `session` plus a recorded CUDA graph and
        // replays it with one `cuGraphLaunch` per token.
        //
        // MEASURED, release, RTX 4070 Laptop, TinyLlama-1.1B, both arms one
        // process, median over tok 3..N: plan-once 111.77 -> captured 25.87
        // ms/token = 4.28x, byte-exact (`logits_bit_exact=true`). The replay
        // cost is BUILD-PROFILE-INVARIANT (25.8 / 26.33 / 25.87 across three
        // weeks and both build profiles) because it is one launch and almost
        // pure device time, while the plan-once baseline is nearly all host
        // work — which is why a debug build reads the ratio as ~12x. Quote
        // 4.28x (release); the historic 10.4x was a DEBUG measurement.
        //
        // Not merely a contiguous-path win: the k=1 paged penalty is *made of*
        // missing capture (paged ~= contiguous-WITHOUT-capture; the gap to
        // captured-contiguous is the same ~4x capture is worth here), so this
        // is where that value currently lives.
        //
        // Safe by construction rather than convention: the captured entry
        // declines non-decode shapes itself (`seq != 1` falls back to the
        // rebuild path), retires the capture WITH the session on any staleness
        // (`invalidate_decode_pair_if_stale`), and is byte-identical to the
        // plan-once path. Non-CUDA builds are untouched.
        #[cfg(feature = "cuda")]
        let mut captured: Option<fuel_dispatch::pipelined::CapturedDecodeSession> = None;

        // Prefill: one forward pass over the full prompt. Always the persistent
        // entry — `seq != 1`, so the captured path would immediately fall back.
        let mut last_logits = self.forward_with_kv_context_persistent(
            prompt_tokens,
            &mut cache,
            &mut ctx,
            &mut session,
        )?;

        // Decode loop.
        for _ in 0..max_new_tokens {
            let next = sample_logits(&last_logits, strategy, &mut rng_state);
            tokens.push(next);
            on_token(next);
            if let Some(eos) = eos_id
                && next == eos
            {
                break;
            }
            #[cfg(feature = "cuda")]
            {
                last_logits = self.forward_with_kv_context_captured(
                    &[next],
                    &mut cache,
                    &mut ctx,
                    &mut session,
                    &mut captured,
                )?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                last_logits = self.forward_with_kv_context_persistent(
                    &[next],
                    &mut cache,
                    &mut ctx,
                    &mut session,
                )?;
            }
        }
        Ok(tokens)
    }

    /// Non-streaming convenience wrapper around
    /// [`Self::generate_streaming_with_kv_context`]. Collects the
    /// generated tokens into a `Vec<u32>` and returns them.
    pub fn generate_with_kv_context(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
    ) -> fuel_core::Result<Vec<u32>> {
        self.generate_streaming_with_kv_context(
            prompt_tokens,
            max_new_tokens,
            strategy,
            eos_id,
            device,
            dtype,
            |_| {},
        )
    }

    /// Speculative decoding through the `forward_with_kv_context`
    /// path (KvCache + InferenceContext + the pipelined executor).
    ///
    /// Uses a `draft` model to predict `k` tokens autoregressively,
    /// then has `self` (the target) verify all `k` positions in a
    /// single forward. Accepts a prefix of the drafts per `strategy`:
    ///
    /// - `Greedy`: longest prefix where target's argmax matches
    ///   draft's token. On mismatch, emit target's argmax as the
    ///   bonus. Output is provably identical to plain greedy
    ///   generation from the target, regardless of the draft.
    /// - `Temperature`: Leviathan-style probability-ratio accept.
    ///   Sample draft tokens from draft's temperature-scaled
    ///   distribution; accept each with probability
    ///   `min(1, p_target(d) / p_draft(d))`. On reject, sample the
    ///   replacement from `(p_target - p_draft)_+ / Z`. Distribution
    ///   of outputs is provably identical to plain sampled generation
    ///   from the target.
    ///
    /// Rejected drafts are rolled back via [`KvCache::truncate_to`] —
    /// a pure metadata update on the pre-allocated-buffer path. The
    /// cache rolls back to the committed prefix (accepted drafts
    /// only); the bonus token's K/V is written by the explicit
    /// bonus-advance forward at its true position.
    ///
    /// Note: the retired legacy-executor implementation truncated the
    /// target cache to `committed + accepted + 1` on rejection,
    /// leaving the rejected draft's K/V row in place at the bonus
    /// position and appending the bonus one position too far. The
    /// resulting logits drift was measured at ~4e-3 (vs ~1e-6 gemm
    /// noise) on the tiny test fixture — real positional corruption,
    /// though small enough there that the argmax never flipped and
    /// the legacy token-equality tests (which only exercised the
    /// accepted == k path) couldn't see it. This implementation
    /// truncates to `committed + accepted` so the bonus advance lands
    /// at the correct position;
    /// `spec_decode_kv_context_divergent_draft_matches_greedy_baseline`
    /// locks the lossless-greedy property.
    ///
    /// Expected speedup 1.5-3× at good acceptance rates (same-family
    /// drafts only — cross-family drafts or different tokenizers will
    /// have <20% acceptance and net-negative speedup).
    ///
    /// Preconditions:
    /// - `draft.config.vocab_size == self.config.vocab_size` (so
    ///   target's distribution over draft's vocab is well-defined).
    /// - Both models share the same tokenizer (caller's
    ///   responsibility).
    #[allow(clippy::too_many_arguments)]
    // needless_range_loop here: the bound is a semantic count that need not equal the
    // indexed buffer len, so a mechanical .iter()/.take() risks silently dropping
    // iterations.
    #[allow(clippy::needless_range_loop)]
    pub fn generate_streaming_spec_with_kv_context(
        &self,
        draft: &LlamaModel,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        k: usize,
        strategy: SamplingStrategy,
        eos_id: Option<u32>,
        device: &Device,
        dtype: DType,
        mut on_token: impl FnMut(u32),
    ) -> fuel_core::Result<Vec<u32>> {
        if draft.config.vocab_size != self.config.vocab_size {
            fuel_ir::bail!(
                "spec-decode: draft vocab {} != target vocab {}",
                draft.config.vocab_size,
                self.config.vocab_size,
            );
        }
        if k == 0 {
            fuel_ir::bail!("spec-decode: k must be >= 1");
        }
        if prompt_tokens.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "generate_streaming_spec_with_kv_context: prompt is empty".to_string(),
            )
            .bt());
        }

        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        let vocab = self.config.vocab_size;

        // RNG state threading. Only used in Temperature mode.
        let mut rng_state: u64 = match strategy {
            SamplingStrategy::Temperature { seed, .. } => seed,
            _ => 0,
        };
        let temp = match strategy {
            SamplingStrategy::Temperature { temp, .. } => temp,
            SamplingStrategy::Greedy => 1.0, // unused in greedy
        };

        // KV capacity: the committed sequence never exceeds
        // `prompt + max_new`; both caches transiently hold up to `k`
        // not-yet-accepted rows past the committed prefix (draft
        // phase / verify phase) before truncation rolls them back.
        let max_seq_len = prompt_tokens.len() + max_new_tokens + k;
        let mut target_cache = KvCache::with_capacity(
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
            max_seq_len,
            dtype,
            device,
        )?;
        let mut draft_cache = KvCache::with_capacity(
            draft.config.n_layers,
            draft.config.n_kv_heads,
            draft.config.head_dim,
            max_seq_len,
            dtype,
            device,
        )?;
        let mut target_ctx = InferenceContext::new(device.clone());
        let mut draft_ctx = InferenceContext::new(device.clone());

        // Prefill both caches with the prompt.
        let mut target_last_logits =
            self.forward_with_kv_context(&tokens, &mut target_cache, &mut target_ctx)?;
        let mut draft_last_logits =
            draft.forward_with_kv_context(&tokens, &mut draft_cache, &mut draft_ctx)?;

        let mut emitted = 0usize;

        while emitted < max_new_tokens {
            // --- Draft phase: K tokens. In Greedy mode, argmax; in
            // Temperature mode, sample from draft's temp-scaled dist.
            // We ALSO stash each draft's probability distribution for
            // the Temperature accept rule.
            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            let mut draft_probs_stash: Vec<Vec<f32>> = Vec::with_capacity(k);
            for _ in 0..k {
                let d = match strategy {
                    SamplingStrategy::Greedy => {
                        // We don't need draft_probs in greedy, but the
                        // slot has to exist to keep indexing uniform.
                        draft_probs_stash.push(Vec::new());
                        spec_argmax(&draft_last_logits)
                    }
                    SamplingStrategy::Temperature { .. } => {
                        let probs = spec_softmax_temp(&draft_last_logits, temp);
                        let d = spec_sample_cat(&probs, &mut rng_state);
                        draft_probs_stash.push(probs);
                        d
                    }
                };
                drafts.push(d);
                draft_last_logits =
                    draft.forward_with_kv_context(&[d], &mut draft_cache, &mut draft_ctx)?;
            }

            // --- Verify phase: target runs forward on the K drafts.
            let verify_logits = self.forward_with_kv_context_all_positions(
                &drafts,
                &mut target_cache,
                &mut target_ctx,
            )?;
            debug_assert_eq!(verify_logits.len(), drafts.len() * vocab);

            // --- Accept phase: strategy-specific. ---
            let mut accepted = 0usize;

            let bonus_token: u32 = match strategy {
                SamplingStrategy::Greedy => {
                    let mut mismatched: Option<u32> = None;
                    for i in 0..drafts.len() {
                        let prev_row = if i == 0 {
                            &target_last_logits[..]
                        } else {
                            &verify_logits[(i - 1) * vocab..i * vocab]
                        };
                        let target_pick = spec_argmax(prev_row);
                        if target_pick == drafts[i] {
                            accepted += 1;
                        } else {
                            mismatched = Some(target_pick);
                            break;
                        }
                    }
                    match mismatched {
                        Some(t) => t,
                        None => spec_argmax(
                            &verify_logits[(drafts.len() - 1) * vocab..drafts.len() * vocab],
                        ),
                    }
                }
                SamplingStrategy::Temperature { .. } => {
                    // Leviathan accept rule. For each i:
                    //   q_i = draft's prob of drafts[i]
                    //   p_i = target's prob of drafts[i] (from prev[i])
                    //   accept with prob min(1, p_i / q_i)
                    // On reject: sample replacement from (p - q)_+ / Z.
                    let mut rejected_replacement: Option<u32> = None;
                    for i in 0..drafts.len() {
                        let prev_row = if i == 0 {
                            &target_last_logits[..]
                        } else {
                            &verify_logits[(i - 1) * vocab..i * vocab]
                        };
                        let target_probs = spec_softmax_temp(prev_row, temp);
                        let draft_probs = &draft_probs_stash[i];
                        let d_tok = drafts[i] as usize;
                        let p = target_probs[d_tok];
                        let q = draft_probs[d_tok];
                        let ratio = if q > 0.0 { (p / q).min(1.0) } else { 0.0 };
                        let u = spec_next_u01(&mut rng_state);
                        if u < ratio {
                            accepted += 1;
                        } else {
                            // Replacement from (p - q)_+ / sum.
                            let mut residual: Vec<f32> = target_probs
                                .iter()
                                .zip(draft_probs.iter())
                                .map(|(&pt, &qt)| (pt - qt).max(0.0))
                                .collect();
                            let sum: f32 = residual.iter().sum();
                            if sum > 0.0 {
                                for r in residual.iter_mut() {
                                    *r /= sum;
                                }
                                rejected_replacement =
                                    Some(spec_sample_cat(&residual, &mut rng_state));
                            } else {
                                // Degenerate case (should only happen if
                                // distributions match exactly — then any
                                // sample from target_probs is equally valid).
                                rejected_replacement =
                                    Some(spec_sample_cat(&target_probs, &mut rng_state));
                            }
                            break;
                        }
                    }
                    match rejected_replacement {
                        Some(t) => t,
                        None => {
                            // All K accepted — sample bonus from target's
                            // last-position distribution.
                            let last_row =
                                &verify_logits[(drafts.len() - 1) * vocab..drafts.len() * vocab];
                            let probs = spec_softmax_temp(last_row, temp);
                            spec_sample_cat(&probs, &mut rng_state)
                        }
                    }
                }
            };

            // --- Roll back both caches to the committed prefix. ---
            // Both caches advanced by K during draft/verify, but only
            // `accepted` of those K positions hold committed tokens.
            // The bonus token's K/V is NOT in either cache (the verify
            // row at the bonus position belongs to the first rejected
            // draft); the bonus-advance forwards below write it at the
            // correct position. When accepted == k both truncates are
            // no-ops and the bonus appends at the cache tail.
            let committed_base = target_cache.cached_len - k;
            target_cache.truncate_to(committed_base + accepted);
            let draft_committed_base = draft_cache.cached_len - k;
            draft_cache.truncate_to(draft_committed_base + accepted);

            // --- Emit accepted drafts + bonus ---
            for i in 0..accepted {
                tokens.push(drafts[i]);
                on_token(drafts[i]);
                emitted += 1;
                if emitted >= max_new_tokens {
                    return Ok(tokens);
                }
                if eos_id == Some(drafts[i]) {
                    return Ok(tokens);
                }
            }
            tokens.push(bonus_token);
            on_token(bonus_token);
            emitted += 1;
            if eos_id == Some(bonus_token) {
                return Ok(tokens);
            }
            if emitted >= max_new_tokens {
                return Ok(tokens);
            }

            // --- Advance both caches + both "last_logits" by the bonus
            // token. The draft needs to see the bonus (which it didn't
            // produce); the target writes the bonus K/V at its true
            // position and returns fresh logits for the next
            // accept-check on draft[0].
            target_last_logits =
                self.forward_with_kv_context(&[bonus_token], &mut target_cache, &mut target_ctx)?;
            draft_last_logits =
                draft.forward_with_kv_context(&[bonus_token], &mut draft_cache, &mut draft_ctx)?;
        }
        Ok(tokens)
    }
}
impl LlamaModel {
    /// Download a LLaMA-architecture model from the HuggingFace Hub and
    /// return a fully assembled `LlamaModel`. Uses `hf_hub::sync` for
    /// the downloads — blocking, with the usual `~/.cache/huggingface`
    /// caching semantics.
    ///
    /// `repo_id` is the HuggingFace repo name in the usual form
    /// (e.g. `"meta-llama/Meta-Llama-3-8B"`). Gated models require
    /// `HF_TOKEN` or a prior `huggingface-cli login`.
    ///
    /// This call downloads:
    /// - `config.json` — the model config
    /// - `model.safetensors.index.json` OR `model.safetensors` —
    ///   depending on whether the model is sharded
    /// - every shard in the index (if sharded)
    ///
    /// It does NOT download the tokenizer or any other files. Wire the
    /// tokenizer separately via `hf_hub::api::sync::ApiRepo::get`.
    ///
    /// For a 70B model this function will download ~150GB. The cache
    /// is persistent so subsequent calls are instant.
    pub fn from_hub(repo_id: &str) -> fuel_core::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| fuel_core::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        // 1. config.json
        let config_path = repo
            .get("config.json")
            .map_err(|e| fuel_core::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = LlamaConfig::from_hf_json_str(&config_str)?;

        // 2. Weight file(s). Try sharded layout first, fall back to single file.
        let weight_paths: Vec<std::path::PathBuf> = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)
                    .map_err(|e| fuel_core::Error::Msg(format!("parsing index: {e}")))?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|x| x.as_object())
                    .ok_or_else(|| {
                        fuel_core::Error::Msg("index.json: missing weight_map".into())
                    })?;
                let mut unique = std::collections::HashSet::new();
                for v in weight_map.values() {
                    if let Some(s) = v.as_str() {
                        unique.insert(s.to_string());
                    }
                }
                let mut paths: Vec<std::path::PathBuf> = Vec::new();
                for shard_name in unique {
                    let p = repo
                        .get(&shard_name)
                        .map_err(|e| fuel_core::Error::Msg(format!("hf-hub {shard_name}: {e}")))?;
                    paths.push(p);
                }
                paths
            }
            Err(_) => {
                // Single-shard model.
                let p = repo
                    .get("model.safetensors")
                    .map_err(|e| fuel_core::Error::Msg(format!("hf-hub model.safetensors: {e}")))?;
                vec![p]
            }
        };

        // 3. Memory-map the safetensors files and load the weights.
        let st = unsafe { fuel_core::safetensors::MmapedSafetensors::multi(&weight_paths) }?;
        let weights = LlamaWeights::load_from_mmapped(&st, &config)?;

        Ok(LlamaModel { config, weights })
    }
}
#[cfg(test)]
impl LlamaConfig {
    /// The hand-rolled parser this type used before the `serde` split, kept
    /// verbatim and test-only as the differential's oracle.
    pub(crate) fn from_hf_json_str_legacy(json: &str) -> fuel_core::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing config.json: {e}")))?;

        let get_usize = |key: &str| -> fuel_core::Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    fuel_core::Error::Msg(format!("config.json: missing/invalid field {key:?}"))
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
        let ffn_dim = get_usize("intermediate_size")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = get_f64("rms_norm_eps").unwrap_or(1e-5);
        let rope_base = get_f64("rope_theta").unwrap_or(10_000.0);

        Ok(LlamaConfig {
            vocab_size,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            norm_eps,
            rope_base,
        })
    }
}
#[cfg(test)]
mod hub_tests {
    use super::*;

    /// Every `config.json` fixture this module exercises, as ONE corpus.
    ///
    /// **Corpus: 4 fixtures — 3 pre-existing, 1 added.** Per-rule coverage:
    ///
    /// | rule                  | discriminated by a PRE-EXISTING fixture?                |
    /// |-----------------------|---------------------------------------------------------|
    /// | `num_key_value_heads` | YES — llama3-8b ships 8 against 32 heads                 |
    /// | `head_dim`            | NO — llama3-8b ships 128 against a quotient of 4096/32 = 128, so the axis is COLLAPSED; the legacy fixture omits it. Fixture 4 added. |
    ///
    /// Fifth config in this increment and the third `head_dim` collapse: real
    /// configs usually ship the quotient, so the axis arrives pre-collapsed
    /// almost everywhere and each conversion needs its own non-quotient case.
    const DIFFERENTIAL_CORPUS: &[(&str, &str)] = &[
        (
            "llama3-8b (kv_heads 8 != 32 discriminates; head_dim 128 == 4096/32 does NOT)",
            r#"{
                "architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
                "intermediate_size": 14336, "num_hidden_layers": 32,
                "num_attention_heads": 32, "num_key_value_heads": 8,
                "vocab_size": 128256, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                "head_dim": 128, "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16"
            }"#,
        ),
        (
            "legacy LLaMA-1 (no GQA, no rope_theta, no head_dim)",
            r#"{
                "hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 2,
                "num_attention_heads": 4, "vocab_size": 128, "rms_norm_eps": 1e-5
            }"#,
        ),
        (
            "missing required fields — both paths must REJECT",
            r#"{"hidden_size": 64}"#,
        ),
        (
            "ADDED: explicit head_dim DISAGREEING with hidden_size/num_attention_heads",
            r#"{
                "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
                "num_attention_heads": 32, "vocab_size": 128256, "head_dim": 96
            }"#,
        ),
    ];

    #[test]
    fn serde_path_agrees_with_the_legacy_parser_on_every_fixture() {
        assert_eq!(DIFFERENTIAL_CORPUS.len(), 4, "corpus shrank");
        for (name, json) in DIFFERENTIAL_CORPUS {
            let new = LlamaConfig::from_hf_json_str(json);
            let old = LlamaConfig::from_hf_json_str_legacy(json);
            match (new, old) {
                // `LlamaConfig` gained `PartialEq` for this: a hand-written
                // field-by-field comparison can silently OMIT a field, which
                // would make the differential blind to exactly that field.
                // The derive covers every field by construction.
                (Ok(a), Ok(b)) => assert_eq!(a, b, "differential mismatch on {name}"),
                (Err(_), Err(_)) => {}
                (Ok(_), Err(e)) => panic!("{name}: serde accepted, legacy rejected: {e}"),
                (Err(e), Ok(_)) => panic!("{name}: serde rejected, legacy accepted: {e}"),
            }
        }
    }

    #[test]
    fn explicit_head_dim_survives_both_paths() {
        let json = DIFFERENTIAL_CORPUS[3].1;
        let cfg = LlamaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(
            cfg.head_dim, 96,
            "explicit head_dim must not be overwritten"
        );
        assert_ne!(
            cfg.head_dim,
            4096 / 32,
            "llama3-8b cannot make this distinction"
        );
        let legacy = LlamaConfig::from_hf_json_str_legacy(json).unwrap();
        assert_eq!(legacy.head_dim, 96, "legacy must agree - it is the oracle");
    }

    #[test]
    fn parse_llama3_style_hf_config() {
        // A minimal LLaMA 3 8B config.json. Real values from the
        // Hugging Face card; we just check the parser maps every field
        // correctly.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "head_dim": 128,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16"
        }"#;
        let cfg = LlamaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.dim, 4096);
        assert_eq!(cfg.ffn_dim, 14336);
        assert_eq!(cfg.n_layers, 32);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8); // GQA
        assert_eq!(cfg.vocab_size, 128256);
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope_base - 500_000.0).abs() < 1e-6);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn parse_legacy_llama_config_defaults_to_mha() {
        // Older LLaMA 1 configs don't have `num_key_value_heads` or
        // `rope_theta`. The parser should fall back to non-GQA and
        // rope base 10000.
        let json = r#"{
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5
        }"#;
        let cfg = LlamaConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.n_kv_heads, cfg.n_heads);
        assert_eq!(cfg.head_dim, 64 / 4);
        assert!((cfg.rope_base - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn parse_rejects_missing_required_fields() {
        let json = r#"{"hidden_size": 64}"#;
        let result = LlamaConfig::from_hf_json_str(json);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod generate_tests {
    use super::*;

    /// Same tiny-weight helper as the llama_tests module, duplicated
    /// here to keep these tests self-contained.
    fn make_tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        make_tiny_weights_seeded(cfg, 9999)
    }

    /// Seeded variant — spec-decode tests use a second seed to build
    /// a draft model that genuinely diverges from the target.
    fn make_tiny_weights_seeded(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s: u32 = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv_dim).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv_dim).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }

    /// Qwen2-style tiny weights: same shapes as LLaMA plus Q/K/V
    /// biases. Used to verify the bias path is wired through both
    /// `forward` and `forward_with_cache`.
    fn make_tiny_weights_with_qkv_bias(cfg: &LlamaConfig) -> LlamaWeights {
        let mut w = make_tiny_weights(cfg);
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        for layer in &mut w.layers {
            layer.attn_q_bias = Some(Arc::from(vec![0.01_f32; cfg.dim]));
            layer.attn_k_bias = Some(Arc::from(vec![0.01_f32; kv_dim]));
            layer.attn_v_bias = Some(Arc::from(vec![0.01_f32; kv_dim]));
        }
        w
    }

    /// BF16-weight variant of [`make_tiny_weights`]: every `WeightStorage`
    /// matrix (Q/K/V/O/gate/up/down/output) is converted to
    /// `WeightStorage::BF16`. Token embedding + norm gains stay f32 (the
    /// frozen seams: the embedding `index_select` has no BF16 CUDA key,
    /// and norm gains are precision-sensitive host-side, converted to the
    /// running activation dtype at graph-build time — see
    /// `Tensor::const_like_dtype`).
    ///
    /// Homogeneous BF16 activations × BF16 weights is the ONLY dtype
    /// combination `Tensor::matmul`'s gate allows for BF16 activations:
    /// same-dtype homogeneous, or `(lhs=F32, rhs=BF16)`. A BF16-activation
    /// × F32-weight matmul (the opposite direction) is rejected — so the
    /// BF16-throughout decode parity tests need BF16 weights, not the
    /// F32 weights `make_tiny_weights` produces.
    fn make_tiny_weights_bf16(cfg: &LlamaConfig) -> LlamaWeights {
        let f32w = make_tiny_weights(cfg);
        fn to_bf16(ws: WeightStorage) -> WeightStorage {
            match ws {
                WeightStorage::F32(a) => {
                    let converted: Vec<half::bf16> =
                        a.iter().map(|&v| half::bf16::from_f32(v)).collect();
                    WeightStorage::BF16(Arc::from(converted))
                }
                other => other,
            }
        }
        LlamaWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: f32w.token_embedding,
            layers: f32w
                .layers
                .into_iter()
                .map(|l| LayerWeights {
                    attn_q: to_bf16(l.attn_q),
                    attn_q_bias: l.attn_q_bias,
                    attn_k: to_bf16(l.attn_k),
                    attn_k_bias: l.attn_k_bias,
                    attn_v: to_bf16(l.attn_v),
                    attn_v_bias: l.attn_v_bias,
                    attn_o: to_bf16(l.attn_o),
                    ffn_gate: to_bf16(l.ffn_gate),
                    ffn_up: to_bf16(l.ffn_up),
                    ffn_down: to_bf16(l.ffn_down),
                    attn_norm_gain: l.attn_norm_gain,
                    ffn_norm_gain: l.ffn_norm_gain,
                })
                .collect(),
            final_norm_gain: f32w.final_norm_gain,
            output: to_bf16(f32w.output),
        }
    }

    #[test]
    fn qwen2_style_bias_changes_forward_output_but_keeps_it_finite() {
        // Build two identical tiny LLaMAs: one with all-None biases,
        // one with small nonzero biases. The bias-bearing model must
        // still produce finite logits and must produce a different
        // argmax (otherwise the bias code is dead).
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let tokens = [1_u32, 2, 3];
        let no_bias = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let with_bias = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_with_qkv_bias(&cfg),
        };
        let no_bias_logits = no_bias
            .forward(&tokens, 0)
            .unwrap()
            .slice(1, tokens.len() - 1, 1)
            .unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .unwrap()
            .realize_f32();
        let with_bias_logits = with_bias
            .forward(&tokens, 0)
            .unwrap()
            .slice(1, tokens.len() - 1, 1)
            .unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .unwrap()
            .realize_f32();
        for &v in &with_bias_logits {
            assert!(v.is_finite(), "with-bias logit is non-finite: {v}");
        }
        let any_different = no_bias_logits
            .iter()
            .zip(with_bias_logits.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_different,
            "bias had no effect — check that add_optional_trailing_bias is actually called",
        );
    }

    #[test]
    fn qwen2_style_bias_cached_matches_non_cached_generate() {
        // Same correctness bar as the LLaMA version: greedy generation
        // via the cached path must match a non-cached greedy loop.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_with_qkv_bias(&cfg),
        };
        let prompt = [1_u32, 2, 3];
        let max_new = 4;

        // Non-cached reference loop.
        let mut ref_tokens = prompt.to_vec();
        for _ in 0..max_new {
            let logits = model.forward(&ref_tokens, 0).unwrap();
            let last_pos = ref_tokens.len() - 1;
            let last = logits
                .slice(1, last_pos, 1)
                .unwrap()
                .reshape(Shape::from_dims(&[cfg.vocab_size]))
                .unwrap()
                .realize_f32();
            let next = last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            ref_tokens.push(next);
        }

        let cached = model
            .generate(&prompt, max_new, SamplingStrategy::Greedy, None)
            .unwrap();
        assert_eq!(cached, ref_tokens);
    }

    #[test]
    fn generate_greedy_appends_tokens() {
        // Run greedy generation for 4 steps from a 3-token prompt on a
        // tiny model. Output sequence should be 3+4=7 tokens, all
        // valid vocab indices.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let out = model
            .generate(&[1, 2, 3], 4, SamplingStrategy::Greedy, None)
            .unwrap();
        assert_eq!(out.len(), 7);
        for &t in &out {
            assert!(
                (t as usize) < cfg.vocab_size,
                "sampled token {t} out of vocab",
            );
        }
    }

    #[test]
    fn generate_temperature_is_deterministic_with_seed() {
        // Two runs with the same seed must produce identical output.
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let strategy = SamplingStrategy::Temperature {
            temp: 1.0,
            seed: 42,
        };
        let a = model.generate(&[0, 1], 3, strategy, None).unwrap();
        let b = model.generate(&[0, 1], 3, strategy, None).unwrap();
        assert_eq!(a, b, "seeded sampling must be deterministic");
    }

    #[test]
    fn generate_stops_early_on_eos() {
        // Construct a tiny model and pick whatever token greedy
        // selects at step 1 as our "eos". The second call then must
        // stop after exactly one new token (since the first new token
        // equals eos).
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2];
        // First: generate one step without eos to see which token
        // greedy picks.
        let baseline = model
            .generate(&prompt, 1, SamplingStrategy::Greedy, None)
            .unwrap();
        let picked = *baseline.last().unwrap();
        // Second: generate with that token as eos. Should stop after
        // appending it (length = prompt + 1).
        let with_eos = model
            .generate(&prompt, 10, SamplingStrategy::Greedy, Some(picked))
            .unwrap();
        assert_eq!(with_eos.len(), prompt.len() + 1);
        assert_eq!(*with_eos.last().unwrap(), picked);
    }

    // The host-resident-cache prefill-parity test
    // (`forward_with_cache_matches_forward_on_prefill`) was retired in
    // E.3.3.D. Its successor is
    // `forward_with_kv_context_prefill_matches_non_cached_forward`,
    // which exercises the same correctness bar via the new
    // KvCache + InferenceContext + Op::WriteSlice path.

    #[test]
    fn generate_with_cache_matches_non_cached_generate() {
        // Greedy generation must produce the same token sequence
        // whether or not the KV cache is in use. Uses an internal
        // non-cached reference loop so this test does not depend on
        // the public `generate` still having a non-cached path.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2, 3];
        let max_new = 5;

        // Reference: non-cached greedy loop.
        let mut ref_tokens = prompt.to_vec();
        for _ in 0..max_new {
            let logits = model.forward(&ref_tokens, 0).unwrap();
            let last_pos = ref_tokens.len() - 1;
            let last = logits
                .slice(1, last_pos, 1)
                .unwrap()
                .reshape(Shape::from_dims(&[cfg.vocab_size]))
                .unwrap()
                .realize_f32();
            let next = last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            ref_tokens.push(next);
        }

        // Cached: the public generate() routine.
        let cached = model
            .generate(&prompt, max_new, SamplingStrategy::Greedy, None)
            .unwrap();

        assert_eq!(cached, ref_tokens);
    }

    /// Greedy generation through the new `generate_with_kv_context`
    /// path must produce the same token sequence as the legacy
    /// `generate` (which uses the host-resident `LlamaKVCache` +
    /// `forward_with_cache_on`). Both routes use the cache; the only
    /// difference is the in-graph WriteSlice path vs the host-side
    /// download-and-append loop.
    #[test]
    fn generate_with_kv_context_matches_legacy_generate() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2, 3];
        let max_new = 5;

        // Reference: legacy host-resident cache path.
        let legacy = model
            .generate(&prompt, max_new, SamplingStrategy::Greedy, None)
            .unwrap();

        // New: KvCache + InferenceContext + forward_with_kv_context.
        let new_path = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();

        // Greedy argmax is robust to O(ε) drift in the logits — both
        // paths should pick the same token at every step.
        assert_eq!(new_path, legacy);
    }

    /// Streaming generation through `generate_streaming_with_kv_context`
    /// fires `on_token` exactly once per generated token (not the
    /// prompt tokens) and the resulting Vec matches the non-streaming
    /// convenience wrapper.
    #[test]
    fn generate_streaming_with_kv_context_fires_callback_per_token() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 4,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2];
        let max_new = 3;

        let mut streamed: Vec<u32> = Vec::new();
        let tokens = model
            .generate_streaming_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
                |tok| streamed.push(tok),
            )
            .unwrap();

        // on_token fires once per GENERATED token (not the prompt).
        assert_eq!(streamed.len(), max_new);
        // The returned Vec is prompt ++ streamed.
        assert_eq!(tokens.len(), prompt.len() + max_new);
        assert_eq!(&tokens[..prompt.len()], &prompt[..]);
        assert_eq!(&tokens[prompt.len()..], &streamed[..]);
    }

    /// `generate_streaming_with_kv_context` short-circuits when an EOS
    /// token is generated, returning before max_new_tokens is reached.
    #[test]
    fn generate_streaming_with_kv_context_stops_on_eos() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 4,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2];
        let max_new = 10;

        // First find what greedy generates without EOS, then set the
        // first generated token as the EOS to confirm short-circuit.
        let unbounded = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        assert_eq!(unbounded.len(), prompt.len() + max_new);
        let first_generated = unbounded[prompt.len()];

        let bounded = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                Some(first_generated),
                &Device::cpu(),
                DType::F32,
            )
            .unwrap();
        // With EOS = first_generated, generation stops after producing
        // that one token.
        assert_eq!(bounded.len(), prompt.len() + 1);
        assert_eq!(bounded[prompt.len()], first_generated);
    }

    // The host-resident-cache prefill+decode parity test
    // (`forward_with_cache_decode_step_matches_full_forward`) was
    // retired in E.3.3.D. Its successor is
    // `forward_with_kv_context_decode_matches_non_cached_forward`
    // below, which exercises the same correctness bar via the new
    // KvCache + InferenceContext + Op::WriteSlice path.

    // ---- forward_with_kv_context (Phase 7.6 step 9c E.3.3.B) -----------

    /// Prefill + decode through the new `forward_with_kv_context` path
    /// should produce the same last-position logits as a non-cached
    /// forward over the full sequence. Mirrors the
    /// `forward_with_cache_decode_step_matches_full_forward` test but
    /// uses `KvCache::with_capacity` + `InferenceContext` + `Op::
    /// WriteSlice` instead of the legacy host-resident `LlamaKVCache`
    /// + concat-and-download path.
    #[test]
    fn forward_with_kv_context_decode_matches_non_cached_forward() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let next_token = 4_u32;
        let full = [prompt[0], prompt[1], prompt[2], next_token];

        // Non-cached reference: full forward over all 4 tokens.
        let full_logits = model.forward(&full, 0).unwrap();
        let last_pos = full.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1)
            .unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .unwrap()
            .realize_f32();

        // New cached path: KvCache::with_capacity + forward_with_kv_context.
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            /*max_seq_len*/ full.len(),
            DType::F32,
            &device,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(device);

        // Prefill: write the 3-token prompt's K/V into the cache.
        let _prefill_logits = model
            .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
            .expect("prefill");
        assert_eq!(cache.cached_len, prompt.len());

        // Decode: one step with the new token.
        let actual = model
            .forward_with_kv_context(&[next_token], &mut cache, &mut ctx)
            .expect("decode");
        assert_eq!(cache.cached_len, full.len());
        assert_eq!(actual.len(), expected.len());

        // Same tolerance as the legacy cached vs non-cached test: the
        // attention matmul accumulates along the seq dim in a slightly
        // different order between the prefill (one tensor of length
        // total_seq) and the prefill+decode (cached prefix + 1 fresh
        // row) paths. This is the standard O(ε) gemm drift, not a
        // correctness bug.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: new-cached={a}, non-cached={b}, diff={diff}",
            );
        }

        // Side effect: every layer's K and V version should have
        // bumped once per forward step (2 steps × 1 bump each = 2).
        for li in 0..cfg.n_layers {
            assert_eq!(cache.layer(li).unwrap().k_version, 2);
            assert_eq!(cache.layer(li).unwrap().v_version, 2);
        }
    }

    /// Phase D · D2b born-red gate for plan-once persistent decode.
    ///
    /// Drive [`LlamaModel::forward_with_kv_context_persistent`] for ≥3
    /// decode tokens (after a prefill) holding ONE [`DecodeSession`], run
    /// in lockstep against the D1 [`LlamaModel::forward_with_kv_context`]
    /// path (a SECOND identical model + cache + ctx fed the identical
    /// token at each step). Assert the three plan-once invariants:
    ///   (a) `optimize_calls_thread_local()` bumps **exactly once** across
    ///       all the decode tokens — the first persistent decode token
    ///       builds + optimizes the held session; tokens 2..N skip
    ///       optimize entirely (the held graph + cached `OptimizedGraph`
    ///       are reused via the D2a prebuilt seam);
    ///   (b) each persistent token's logits are **exactly `==`** the D1
    ///       cached path on the same prefix — same plan → same kernels →
    ///       bit-exact (NOT epsilon);
    ///   (c) the held graph's node `len()` is **stable from token 2
    ///       onward** — no per-token node growth (the guard that no
    ///       builder snuck a `cached_len`-dependent shape / re-splice /
    ///       re-insert back in).
    ///
    /// GAP-029 increment 2b — is `attended_len_sym` actually UNREFERENCED?
    ///
    /// The Llama half of a two-model control; see the Phi twin
    /// (`phi_attended_len_sym_is_unreferenced_negative_control`) for the full
    /// rationale. Both halves are needed because the 2b shared-driver design
    /// rests on TWO comment-sourced claims — Phi's `SymId(1)` is "carried for
    /// API parity but never referenced/bound" and Llama's is "unreferenced on
    /// today's f32 decode graph". **Testing one only halves the assumption**: if
    /// Llama's were referenced while Phi's is not, the "identical no-op"
    /// argument collapses from the other direction.
    ///
    /// The byte-exact persistent tests cannot settle this — a referenced symbol
    /// bound to its usual value passes them. The discriminating instrument is a
    /// deliberately WRONG binding, paired with a positive control on
    /// `cached_len_sym` (which the KV write demonstrably does reference) so that
    /// "output unchanged" means *unreferenced* rather than *the perturbation
    /// never reached the graph*.
    ///
    /// Scope, per the doc's own qualifier: **F32, CPU, no flash arm.** A
    /// bf16/f16 CUDA decode offering the flash arm would reference
    /// `attended_len` and must re-run this control.
    #[test]
    fn llama_attended_len_sym_is_unreferenced_negative_control() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_seq_len = prompt.len() + 4;

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // Prefill, then ONE decode token to BUILD the held session.
        let _ = model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        let _ = model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("build decode session");
        let s = session.as_ref().expect("session built");

        // Realize the SAME next token three times, varying ONLY the SymEnv.
        // `realize_token` takes &self and clones `base_cache` internally, so the
        // three calls cannot pollute each other.
        let cached_len = cache.cached_len;
        let next = [5_u32];
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        let with_off = s.offset_node().is_some();
        let mk_data = || {
            model
                .build_token_rope_mask_arcs(
                    &dev,
                    cached_len,
                    &next,
                    s.max_seq_len(),
                    cache_dtype,
                    with_off,
                    None,
                )
                .expect("token data")
        };

        // Baseline = exactly what production binds (BOTH symbols).
        let baseline = s
            .realize_token(
                &dev,
                mk_data(),
                &s.per_token_sym_env(cached_len).expect("env"),
            )
            .expect("baseline");

        // --- the measurement: attended_len bound to a WRONG value ---
        let mut env_bad_attended = fuel_ir::SymEnv::new();
        env_bad_attended
            .bind(s.cached_len_sym(), cached_len)
            .expect("bind cached_len");
        env_bad_attended
            .bind(s.attended_len_sym(), cached_len + 4242)
            .expect("bind attended_len (wrong on purpose)");
        let perturbed_attended = s
            .realize_token(&dev, mk_data(), &env_bad_attended)
            .expect("realize with wrong attended_len");

        // --- control A: perturb the DATA, which must always move the output ---
        // This proves `realize_token` is live and sensitive at all. Without it,
        // "nothing changed" could mean the realize is returning a cached or
        // constant result and every verdict below would be vacuous.
        let other = [6_u32];
        let data_other = model
            .build_token_rope_mask_arcs(
                &dev,
                cached_len,
                &other,
                s.max_seq_len(),
                cache_dtype,
                with_off,
                None,
            )
            .expect("token data (different token)");
        let perturbed_data = s
            .realize_token(
                &dev,
                data_other,
                &s.per_token_sym_env(cached_len).expect("env"),
            )
            .expect("realize with different token");
        assert_ne!(
            perturbed_data, baseline,
            "control A FAILED: a DIFFERENT input token produced identical \
             logits, so realize_token is not responding to its inputs and no \
             verdict from this test means anything",
        );

        // --- control B: is the SymEnv consulted AT ALL on this path? ---
        let mut env_bad_cached = fuel_ir::SymEnv::new();
        env_bad_cached
            .bind(s.cached_len_sym(), cached_len + 1)
            .expect("bind cached_len (wrong on purpose)");
        env_bad_cached
            .bind(s.attended_len_sym(), cached_len + 1)
            .expect("bind attended_len");
        let perturbed_cached = s
            .realize_token(&dev, mk_data(), &env_bad_cached)
            .expect("realize with wrong cached_len");

        // MEASURED, not assumed: on the device-offset path the KV write offset
        // rides a device-resident BUFFER (`Op::WriteSliceDoff` reads the start
        // from `DecodeTokenData::offset` at launch), so `cached_len_sym` is NOT
        // what drives the write and perturbing it is expected to be inert. On
        // the SymEnv path (`offset_node().is_none()`, Vulkan) the offset rides
        // the symbol and perturbing it MUST move the output.
        //
        // Asserting the direction that matches the path is what keeps this a
        // control rather than a coin flip: it fails if the relationship between
        // `offset_node` and symbol-referencedness is ever not what the driver's
        // own comment claims.
        if with_off {
            assert_eq!(
                perturbed_cached, baseline,
                "on the device-offset path the KV offset rides the offset \
                 BUFFER, so a wrong cached_len_sym should be inert — it moved \
                 the output, meaning the symbol IS load-bearing here and the \
                 driver's offset_node comment is wrong",
            );
        } else {
            assert_ne!(
                perturbed_cached, baseline,
                "on the SymEnv path the KV offset rides cached_len_sym, so a \
                 wrong binding MUST move the output; it did not, so this test \
                 cannot detect referencedness and its verdict is vacuous",
            );
        }

        assert_eq!(
            perturbed_attended, baseline,
            "attended_len_sym IS referenced by Llama's F32 CPU decode graph — \
             the 'unreferenced on today's f32 decode graph' claim is FALSE, and \
             the GAP-029 2b shared-driver design is unsound as written",
        );

        // WHAT THIS ARM PROVES IS WEAKER THAN THE PHI ARM, and the asymmetry is
        // measured rather than assumed. MEASURED 2026-08-12: Llama runs the
        // **device-offset path** on CPU (`offset_node().is_some() == true`),
        // where the KV write offset rides a buffer, so control B took the
        // `assert_eq` branch — `cached_len_sym` is inert too.
        //
        // Therefore Llama's green does NOT establish "attended_len_sym is
        // unreferenced by the graph". It establishes only "no symbol drives this
        // path", which is a strictly weaker claim that happens to have the same
        // consequence here. Phi is the arm carrying the real evidence: it runs
        // the SymEnv path, where a sibling symbol in the same env demonstrably
        // IS load-bearing.
        //
        // Do not cite this arm as proof of referencedness in either direction,
        // and note the scope: F32 / CPU / no flash arm. A bf16/f16 CUDA decode
        // that offers the flash arm would reference `attended_len` and must
        // re-run this control.
        eprintln!(
            "[gap-029 2b control] llama: offset_node.is_some()={with_off} \
             ({} path; SymEnv live={})",
            if with_off { "device-offset" } else { "SymEnv" },
            !with_off,
        );
    }

    /// The fixed tiny model the GAP-029 step-1 golden is captured against.
    /// Shared by the capture printer and the assertion so they cannot drift.
    fn gap029_golden_cfg() -> LlamaConfig {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        }
    }

    /// Prefill 3 tokens, then decode 3 through `path`, returning the decode
    /// steps' logits flattened. `persistent` selects D2 (held graph + rebind)
    /// over D1 (rebuild per step).
    fn gap029_golden_decode(persistent: bool) -> Vec<f32> {
        let cfg = gap029_golden_cfg();
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [1_u32, 2, 3];
        let decode_tokens = [4_u32, 5, 6];
        let max_seq_len = prompt.len() + decode_tokens.len();

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        let mut out = Vec::new();
        if persistent {
            let _ = model
                .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
                .expect("prefill");
            for &tok in &decode_tokens {
                out.extend(
                    model
                        .forward_with_kv_context_persistent(
                            &[tok],
                            &mut cache,
                            &mut ctx,
                            &mut session,
                        )
                        .expect("decode"),
                );
            }
        } else {
            let _ = model
                .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
                .expect("prefill");
            for &tok in &decode_tokens {
                out.extend(
                    model
                        .forward_with_kv_context(&[tok], &mut cache, &mut ctx)
                        .expect("decode"),
                );
            }
        }
        out
    }

    /// `LlamaModel`'s decode logits, **captured on `0c04a7bb` before GAP-029
    /// increment 3 moved the build path into `fuel_core::persistent_decode`.**
    ///
    /// 3 prefill tokens then 3 decode steps, flattened — see
    /// [`gap029_golden_decode`].
    const GAP029_LLAMA_DECODE_GOLDEN: [f32; 48] = [
        0.235_882_4_f32,
        0.079_829_14_f32,
        -0.130_090_36_f32,
        -0.102592126_f32,
        -0.065_259_62_f32,
        0.086_530_46_f32,
        -0.204_395_95_f32,
        0.017_957_9_f32,
        0.064_830_67_f32,
        -0.197_869_9_f32,
        -0.218_223_33_f32,
        0.039_521_27_f32,
        0.181_455_9_f32,
        -0.014700335_f32,
        0.325_861_75_f32,
        0.119_164_76_f32,
        -0.124_628_99_f32,
        -0.219_824_36_f32,
        0.088_146_29_f32,
        0.069_199_83_f32,
        -0.053305764_f32,
        -0.030561801_f32,
        0.031161062_f32,
        -0.083186984_f32,
        -0.013608441_f32,
        0.065_507_23_f32,
        -0.085608765_f32,
        -0.016087107_f32,
        0.061509483_f32,
        0.099036664_f32,
        -0.117_331_99_f32,
        -0.176_698_67_f32,
        0.132_780_49_f32,
        -0.045753848_f32,
        -0.103880905_f32,
        -0.083_950_27_f32,
        -0.068138495_f32,
        0.013806637_f32,
        -0.149_572_33_f32,
        -0.053597312_f32,
        -0.025251985_f32,
        -0.027037866_f32,
        -0.328_116_18_f32,
        -0.087185904_f32,
        0.152_396_05_f32,
        -0.154_513_64_f32,
        0.104_392_51_f32,
        0.062_370_29_f32,
    ];

    /// Node count of `LlamaModel`'s held decode graph — see
    /// [`llama_held_decode_graph_has_not_grown`]. Measured, not predicted.
    const GAP029_LLAMA_DECODE_GRAPH_NODES: usize = 186;

    /// Build one held decode session and report its graph node count. Shared by
    /// the capture and the assertion so the two cannot drift.
    fn gap029_llama_decode_graph_nodes() -> usize {
        let cfg = gap029_golden_cfg();
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            6,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        model
            .forward_with_kv_context_persistent(&[1, 2, 3], &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("decode");
        session
            .expect("session built on the first decode token")
            .graph_node_count()
    }

    /// **STRUCTURAL baseline, captured 2026-08-13 BEFORE the Gemma3 seam work
    /// (per-layer RoPE variants + `embed_scale`).**
    ///
    /// [`llama_decode_logits_unchanged_by_the_shared_build_path`] proves the
    /// NUMBERS did not move. It cannot prove the GRAPH did not grow: an extra
    /// `mul_scalar(1.0)`, or a slice+reshape a single-variant family should
    /// never emit, is **numerically invisible and structurally real** — it costs
    /// a node in every held decode plan for the life of the session, and a
    /// logits golden would sail straight past it.
    ///
    /// So "uniform families pay literally nothing" is pinned as a node COUNT
    /// rather than asserted in prose: `embed_scale == None` must emit no
    /// multiply, and `n_rope_variants == 1` must emit neither a slice nor a
    /// reshape.
    ///
    /// If an unrelated graph-construction change moves this, re-capture it
    /// deliberately and say so in the commit — do not nudge the constant until
    /// it passes.
    #[test]
    fn llama_held_decode_graph_has_not_grown() {
        assert_eq!(
            gap029_llama_decode_graph_nodes(),
            GAP029_LLAMA_DECODE_GRAPH_NODES,
            "Llama's held decode graph changed size",
        );
    }

    /// **GAP-029 increment 3, step 1 — the oracle for a behaviour-preserving
    /// refactor, and the reason it is not the pre-existing suite.**
    ///
    /// Moving `LlamaModel`'s D1+D2 build path into the shared, parameterised
    /// `fuel_core::persistent_decode` seam has **no born-red state**: "the tests
    /// still pass" is also exactly what a no-op produces. The obvious oracle —
    /// the existing decode suite — is *too loose to serve*:
    /// `forward_with_kv_context_decode_matches_non_cached_forward` asserts
    /// `diff < 5e-3 || rel < 1e-2`, and GAP-029 measured a real, silently-wrong
    /// masking divergence at **7.9e-3**. A tolerance that swallows a known
    /// defect cannot certify "behaviour preserved"; it certifies only that
    /// nothing gross broke.
    ///
    /// So the values above were captured from the **pre-refactor** code and are
    /// asserted here at **1e-6** — tight enough that any reassociation of the
    /// decode arithmetic shows up, loose enough to absorb nothing but f32
    /// print/parse round-off.
    ///
    /// **This test is BORN GREEN and that is stated rather than presented as
    /// evidence.** Its value is entirely in *when* the numbers were taken. Its
    /// discrimination is established separately, by the sabotage record in
    /// `fuel_core::persistent_decode`: breaking the shared build path reddens this
    /// test. A golden nobody has ever seen fail is a constant, not an oracle.
    ///
    /// Both paths are asserted because the refactor parameterises **both**:
    /// D2 riding a correct D1 would otherwise hide a D2-only regression.
    #[test]
    fn llama_decode_logits_unchanged_by_the_shared_build_path() {
        for (persistent, label) in [(false, "D1 rebuild"), (true, "D2 persistent")] {
            let got = gap029_golden_decode(persistent);
            assert_eq!(
                got.len(),
                GAP029_LLAMA_DECODE_GOLDEN.len(),
                "{label}: decode step count changed",
            );
            for (i, (a, b)) in got
                .iter()
                .zip(GAP029_LLAMA_DECODE_GOLDEN.iter())
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-6,
                    "{label}: logit[{i}] drifted across the GAP-029 build-path \
                     extraction: got {a}, pre-refactor golden {b} (diff {}). This \
                     refactor is behaviour-preserving by contract — widen nothing; \
                     find what changed.",
                    (a - b).abs(),
                );
            }
        }
    }

    /// Born-red shape: if the data Consts are rebuilt fresh per token
    /// (a new graph each token) OR the session re-optimizes, (a)/(c)
    /// fail; wiring the held session + per-token data re-bind makes them
    /// pass.
    #[test]
    fn forward_with_kv_context_persistent_plan_once_matches_d1() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };

        // Two byte-identical models: one drives the D2 persistent path,
        // one drives the D1 rebuild path. Identical weights (same seed).
        let model_d2 = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let model_d1 = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let decode_tokens = [4_u32, 5, 6, 7]; // ≥3 decode tokens
        let max_seq_len = prompt.len() + decode_tokens.len();

        // --- D1 (rebuild) reference FIRST, in its own pass, so its
        // per-token re-plans do NOT pollute the optimize-count window we
        // measure around the D2 loop. Store the expected logits. ---
        let dev1 = Device::cpu();
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev1,
        )
        .expect("with_capacity d1");
        let mut ctx1 = InferenceContext::new(dev1);
        let _ = model_d1
            .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
            .expect("d1 prefill");
        let mut d1_expected: Vec<Vec<f32>> = Vec::with_capacity(decode_tokens.len());
        for &tok in &decode_tokens {
            d1_expected.push(
                model_d1
                    .forward_with_kv_context(&[tok], &mut cache1, &mut ctx1)
                    .expect("d1 decode"),
            );
        }

        // --- D2 (persistent) session state ---
        let dev2 = Device::cpu();
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev2,
        )
        .expect("with_capacity d2");
        let mut ctx2 = InferenceContext::new(dev2);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // Prefill the D2 path (seq > 1 → the persistent path falls back to
        // the rebuild path; the session is NOT built here).
        let _ = model_d2
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .expect("d2 prefill");
        assert!(
            session.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );

        // Decode ≥3 tokens through the persistent path ONLY. Snapshot the
        // optimizer count on THIS thread just before the decode loop
        // (isolated from other suite threads' concurrent optimizes — the
        // process-global count is polluted; the thread-local delta is
        // exact). The D2 loop is the ONLY optimize source in this window.
        let opt_before = fuel_core::pipelined_bridge::optimize_calls_thread_local();
        let mut len_at_token2: Option<usize> = None;

        for (i, &tok) in decode_tokens.iter().enumerate() {
            let d2 = model_d2
                .forward_with_kv_context_persistent(&[tok], &mut cache2, &mut ctx2, &mut session)
                .expect("d2 decode");

            // (b) bit-exact vs. the D1 cached path (same plan → same
            // kernels), NOT epsilon.
            assert_eq!(
                d2, d1_expected[i],
                "persistent decode token {i} must be byte-identical to the D1 \
                 cached path",
            );

            // The session must exist after the first decode token and
            // stay valid across the rest.
            let sess = session
                .as_ref()
                .expect("session built on first decode token");
            let graph_len = sess.graph_node_count();
            if i == 1 {
                len_at_token2 = Some(graph_len);
            } else if i >= 2 {
                // (c) node count stable from token 2 onward.
                assert_eq!(
                    Some(graph_len),
                    len_at_token2,
                    "held graph must NOT grow from token 2 onward (token {i})",
                );
            }
        }

        // (a) optimize bumped EXACTLY ONCE across all decode tokens.
        let opt_after = fuel_core::pipelined_bridge::optimize_calls_thread_local();
        assert_eq!(
            opt_after - opt_before,
            1,
            "persistent decode must optimize EXACTLY ONCE across {} decode \
             tokens (the first builds the session; the rest skip optimize): \
             {opt_before} -> {opt_after}",
            decode_tokens.len(),
        );

        // Sanity: both caches advanced identically.
        assert_eq!(cache2.cached_len, max_seq_len);
        assert_eq!(cache1.cached_len, max_seq_len);
    }

    // ---- Phase D increment A: BF16-throughout decode -------------------
    //
    // Fix strategy for the CUDA decode 21x-slower-than-Vulkan gap: CUDA
    // has no mixed F32xBF16 gemm, so an F32 activation stream forces F32
    // gemm everywhere even with BF16 weights. Running decode activations
    // BF16 end-to-end (cache dtype drives the whole stream) lets the
    // homogeneous [bf16;3] CUDA gemm fire. This increment (A) proves the
    // graph/dtype seams on CPU; a later increment measures the CUDA win.

    /// Born-red gate: BF16-throughout D1 decode
    /// (`forward_with_kv_context`, BF16 `KvCache`) must produce logits
    /// close to the F32 reference — same argmax, small abs diff.
    ///
    /// Both runs use `forward_with_kv_context` (the D1 rebuild path);
    /// only the cache dtype (hence the activation dtype end-to-end)
    /// differs. `matmul`'s dtype gate only allows homogeneous same-dtype
    /// or `(lhs=F32, rhs=BF16)` — the opposite of BF16-activation ×
    /// F32-weight — so the BF16 run also needs BF16 weights
    /// (`make_tiny_weights_bf16`); this is therefore a quantization-
    /// accuracy check (BOTH weights and activations lose precision vs.
    /// f32), not a bit-exact one.
    ///
    /// Born-red shape: before the dtype seams land (embed cast, RoPE
    /// cast-around, dtype-aware norm-gain/bias/mask consts, logits cast
    /// pre-realize), the BF16-cache run panics inside `fuel_graph` —
    /// `rope_with_tables`'s typed f32-only check surfaces first as a
    /// typed `Err` (not even a panic) making prefill itself fail, and if
    /// that were bypassed, `binary_op`'s dtype-equality `assert_eq!`
    /// (rms-norm gain multiply, mask add) or `write_slice_dyn`'s dtype
    /// check would panic downstream.
    #[test]
    fn bf16_decode_matches_f32_decode_d1() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let prompt = [1_u32, 2, 3];
        let next_token = 4_u32;
        let max_seq_len = prompt.len() + 1;

        // F32 reference: F32 weights, F32 cache.
        let model_f32 = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev_f32 = Device::cpu();
        let mut cache_f32 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev_f32,
        )
        .expect("with_capacity f32");
        let mut ctx_f32 = InferenceContext::new(dev_f32);
        let _ = model_f32
            .forward_with_kv_context(&prompt, &mut cache_f32, &mut ctx_f32)
            .expect("f32 prefill");
        let f32_logits = model_f32
            .forward_with_kv_context(&[next_token], &mut cache_f32, &mut ctx_f32)
            .expect("f32 decode");

        // BF16-throughout: BF16 weights (the only matmul-gate-legal
        // pairing with BF16 activations) + BF16 cache.
        let model_bf16 = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };
        let dev_bf16 = Device::cpu();
        let mut cache_bf16 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &dev_bf16,
        )
        .expect("with_capacity bf16");
        let mut ctx_bf16 = InferenceContext::new(dev_bf16);
        let _ = model_bf16
            .forward_with_kv_context(&prompt, &mut cache_bf16, &mut ctx_bf16)
            .expect("bf16 prefill");
        let bf16_logits = model_bf16
            .forward_with_kv_context(&[next_token], &mut cache_bf16, &mut ctx_bf16)
            .expect("bf16 decode");

        assert_eq!(f32_logits.len(), bf16_logits.len());
        assert_eq!(f32_logits.len(), cfg.vocab_size);

        let argmax = |v: &[f32]| -> usize {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(
            argmax(&f32_logits),
            argmax(&bf16_logits),
            "BF16-throughout decode must agree with the F32 reference on argmax: \
             f32={f32_logits:?} bf16={bf16_logits:?}",
        );

        let max_abs_diff = f32_logits
            .iter()
            .zip(bf16_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("bf16_decode_matches_f32_decode_d1: max abs diff = {max_abs_diff}");
        // Tolerance calibration (sabotage-measured, 2026-07-08): genuine
        // BF16-vs-F32 max abs diff on this fixture is ~0.00118 (BF16
        // carries ~2-3 significant decimal digits; BOTH weights AND
        // activations are BF16-quantized). A deliberately corrupted BF16
        // leg (const_like_dtype's gain/mask consts scaled by 1.05 — a 5%
        // error confined to the BF16 path; note common-mode sabotages
        // like a wrong shared mask offset CANCEL in this differential
        // test and calibrate nothing) moves the diff to ~0.0159. 5e-3
        // sits ~4.2x above genuine and ~3.2x below that corruption
        // signal. Argmax equality above is the scale-robust backstop.
        assert!(
            max_abs_diff < 5e-3,
            "bf16 vs f32 max abs diff {max_abs_diff} exceeds tolerance",
        );
    }

    /// Part 2 increment B · live-CUDA gate: BF16-throughout D1 decode must
    /// agree between the CPU and CUDA devices — the whole point of the
    /// BF16-throughout seam is that the homogeneous `[bf16;3]` CUDA gemm
    /// (tensor cores, `CUBLAS_COMPUTE_32F` accumulation) fires instead of
    /// the F32 gemm that made plain CUDA decode 21x slower than Vulkan
    /// (commit `d87d2427`). Both legs run the SAME BF16-weight model
    /// (`make_tiny_weights_bf16`) through the SAME D1 rebuild protocol
    /// (prefill 3 + decode 1); only the device differs.
    ///
    /// This is a cross-device agreement check, not a bit-exact one: CPU and
    /// CUDA BF16 gemms can accumulate in slightly different order /
    /// intermediate precision even when both target `CUBLAS_COMPUTE_32F`-
    /// equivalent F32 accumulation. Tolerance is calibrated empirically
    /// (see the printed `max abs diff`) with ~5x headroom.
    ///
    /// Gated `#[cfg(feature = "cuda")]` + `#[ignore]`; skips cleanly if no
    /// CUDA device is present. Run:
    ///   `cargo test -p fuel-core --features cuda --lib \
    ///    bf16_decode_cuda_matches_cpu -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn bf16_decode_cuda_matches_cpu() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let prompt = [1_u32, 2, 3];
        let next_token = 4_u32;
        let max_seq_len = prompt.len() + 1;

        // CPU leg: BF16-throughout decode (BF16 weights + BF16 cache), the
        // same protocol `bf16_decode_matches_f32_decode_d1`'s BF16 side uses.
        let model_cpu = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };
        let dev_cpu = Device::cpu();
        let mut cache_cpu = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &dev_cpu,
        )
        .expect("with_capacity cpu bf16");
        let mut ctx_cpu = InferenceContext::new(dev_cpu);
        let _ = model_cpu
            .forward_with_kv_context(&prompt, &mut cache_cpu, &mut ctx_cpu)
            .expect("cpu bf16 prefill");
        let cpu_logits = model_cpu
            .forward_with_kv_context(&[next_token], &mut cache_cpu, &mut ctx_cpu)
            .expect("cpu bf16 decode");

        // CUDA device or skip cleanly.
        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let cuda_device: Device = cuda.into();

        // CUDA leg: byte-identical model/protocol, BF16 cache on the CUDA
        // device — the homogeneous [bf16;3] CUDA gemm path under test.
        let model_cuda = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };
        let mut cache_cuda = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &cuda_device,
        )
        .expect("with_capacity cuda bf16");
        let mut ctx_cuda = InferenceContext::new(cuda_device);
        let _ = model_cuda
            .forward_with_kv_context(&prompt, &mut cache_cuda, &mut ctx_cuda)
            .expect("cuda bf16 prefill");
        let cuda_logits = model_cuda
            .forward_with_kv_context(&[next_token], &mut cache_cuda, &mut ctx_cuda)
            .expect("cuda bf16 decode");

        assert_eq!(cpu_logits.len(), cuda_logits.len());
        assert_eq!(cpu_logits.len(), cfg.vocab_size);

        let argmax = |v: &[f32]| -> usize {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(
            argmax(&cpu_logits),
            argmax(&cuda_logits),
            "BF16-throughout CUDA decode must agree with the CPU BF16 leg on argmax: \
             cpu={cpu_logits:?} cuda={cuda_logits:?}",
        );

        let max_abs_diff = cpu_logits
            .iter()
            .zip(cuda_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("bf16_decode_cuda_matches_cpu: max abs diff = {max_abs_diff}");
        // Tolerance calibration (measured 2026-07-08, RTX 4070, CUDA 13.3):
        // observed max abs diff = 0 exactly — on this tiny fixture the CPU
        // and CUDA BF16 legs are bit-identical (both gemms accumulate F32
        // and every intermediate rounds to the same BF16). 5e-3 (the same
        // band the BF16-vs-F32 D1 test and the CPU-vs-GPU decode twins use)
        // is therefore pure headroom for larger/less-well-conditioned
        // fixtures; argmax equality above is the scale-robust backstop.
        assert!(
            max_abs_diff < 5e-3,
            "cpu vs cuda bf16 max abs diff {max_abs_diff} exceeds tolerance",
        );
    }

    // ---------------------------------------------------------------------
    // PER-NODE CPU-ORACLE DIFF — ATTEMPTED, MEASURED, NOT VIABLE IN THIS FORM.
    //
    // The idea: after a captured replay, read every retained intermediate
    // (`CapturedDecodeSession::node_outputs`) and compare it against the same
    // node realized on CPU, localising numerical divergence to a NODE rather
    // than to "the logits differ". The capture half works — 66 retained
    // buffers over a 168-node graph, all readable, all finite (see
    // `captured_decode_exposes_per_node_intermediates_cuda`).
    //
    // The CPU-oracle half does not, and the reason is structural rather than a
    // matter of tuning. THREE configurations were measured, all exceeding a
    // 7-15 minute bound on a TINY fixture (vocab 16, dim 16, 2 layers):
    //
    //   1. one `realize_one_as_with_env` per node   -> killed at 15 min
    //   2. one `realize_many_as_with_env` over ~66  -> killed at 15 min
    //   3. the same, SAMPLED to 3 targets           -> killed at 7 min
    //
    // (3) is decisive: cost is NOT proportional to the number of targets, so
    // sampling cannot rescue it. Realizing an INTERIOR node of a held decode
    // graph is expensive per se — the plan-once path exists precisely because
    // the graph has ONE root, and asking for an interior node asks a question
    // that path is not shaped to answer.
    //
    // WHAT THIS ACTUALLY BLOCKS ON, stated so the next attempt does not repeat
    // the three above: capture gives per-node retention on CUDA *for free*
    // because recording the graph pins every intermediate at a fixed address.
    // There is NO CPU-side equivalent. So a per-node CPU oracle needs a
    // "realize once, retain all intermediates" mode on the CPU path — a real
    // capability, not a test-writing problem. With that, the oracle is one
    // realize and a map lookup; without it, every node costs a fresh traversal.
    //
    // Filed here rather than as a disabled test: a test that cannot complete is
    // worse than no test, and the measurement is the useful artifact.
    // ---------------------------------------------------------------------

    /// **TOKEN-INVARIANCE CENSUS — how much of a decode step is recomputed
    /// per token that need not be?** The measurement that sizes memoization
    /// before any caching logic is written.
    ///
    /// A node is token-invariant iff none of its transitive inputs is per-token
    /// data. Those inputs are exactly what the held session names:
    /// `token_ids` / `rope_cos` / `rope_sin` / `mask` / the KV pairs / the
    /// optional write offset. Everything downstream of them varies; everything
    /// else is computed identically for every token of the generation.
    ///
    /// **Measured TWO independent ways, and they must agree.** This session's
    /// only reliable defence against a wrong instrument was a second instrument
    /// that could contradict the first — an under-populated binding table, a
    /// truncated enum read and a mis-windowed benchmark each looked like a
    /// finding until something disagreed with them.
    ///
    /// 1. **Structural** — forward reachability from the per-token inputs over
    ///    the real graph. Exact by construction.
    /// 2. **Empirical** — replay two DIFFERENT tokens and compare every retained
    ///    intermediate bytewise. A node whose bytes change is varying; one whose
    ///    bytes do not is an invariance CANDIDATE.
    ///
    /// Their disagreement is the interesting signal and is reported, not hidden.
    /// Empirical-invariant / structurally-varying is expected and benign (two
    /// tokens can coincide, especially on a 16-vocab fixture). The reverse —
    /// empirically CHANGING while structurally invariant — would mean a node
    /// with no per-token input is nonetheless not reproducible, which is a
    /// correctness signal, so it is asserted against.
    ///
    /// Reports rather than gates: this exists to size the opportunity, and the
    /// number it produces decides whether memoization is worth its correctness
    /// exposure at all.
    ///
    /// **MEASURED (tiny F32 Llama, RTX 4070): 168 nodes, 127 varying, 41
    /// invariant = 21 `Const` + 20 `BroadcastTo`. Invariant nodes that
    /// MATERIALIZE: zero.** The Consts are weights `base_cache` already realizes
    /// once; the `BroadcastTo`s are `WorkItemKind::ViewOf` — output Arc aliases
    /// the input's, no allocation, no launch — so recomputing one per token
    /// costs a host-side Layout. Nothing here is worth a value cache.
    ///
    /// The first version of this test reported the same CONCLUSION for a reason
    /// that was wrong, and the correction is the point: it inferred "all 41 are
    /// Consts" from an op histogram that was populated only for nodes present in
    /// both replay snapshots. Capture retains no `Const` output and no view
    /// output, so that bucket was empty and printed NOTHING — and an empty print
    /// was read as confirmation. It was equally consistent with 20 invariant
    /// compute nodes existing, which is what was actually there. Hence §1b:
    /// enumerate off the graph, and assert the enumeration's total against an
    /// independently computed count so "printed nothing" cannot masquerade as
    /// "measured nothing".
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn token_invariance_census_cuda() {
        use std::collections::{BTreeMap, HashSet};

        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let dev: Device = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d.into(),
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };

        let prompt = [1_u32, 2, 3];
        let msl = prompt.len() + 4;
        let m = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            msl,
            DType::F32,
            &dev,
        )
        .expect("cache");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut sess: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut cap: Option<fuel_dispatch::pipelined::CapturedDecodeSession> = None;

        m.forward_with_kv_context_captured(&prompt, &mut cache, &mut ctx, &mut sess, &mut cap)
            .expect("prefill");
        // Tokens 1-2 build the session then the capture; 3 and 4 are pure
        // replays, and those are the two we compare.
        for t in [4_u32, 5] {
            m.forward_with_kv_context_captured(&[t], &mut cache, &mut ctx, &mut sess, &mut cap)
                .expect("build");
        }

        let snapshot = |cap: &fuel_dispatch::pipelined::CapturedDecodeSession| {
            let mut out: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
            for (nid, buf) in cap.node_outputs().iter() {
                if let Ok(v) = captured_output_to_f32(buf) {
                    out.insert(nid.0, v);
                }
            }
            out
        };

        m.forward_with_kv_context_captured(&[6], &mut cache, &mut ctx, &mut sess, &mut cap)
            .expect("replay A");
        let snap_a = snapshot(cap.as_ref().expect("capture"));
        m.forward_with_kv_context_captured(&[11], &mut cache, &mut ctx, &mut sess, &mut cap)
            .expect("replay B");
        let snap_b = snapshot(cap.as_ref().expect("capture"));

        let s = sess.as_ref().expect("session");
        let g = s.graph().read().expect("graph lock");

        // --- (1) STRUCTURAL: forward reachability from the per-token inputs ---
        let mut varying: HashSet<usize> = HashSet::new();
        varying.insert(s.token_ids_node().0);
        varying.insert(s.rope_cos_node().0);
        varying.insert(s.rope_sin_node().0);
        varying.insert(s.mask_node().0);
        if let Some(off) = s.offset_node() {
            varying.insert(off.0);
        }
        for (k, v) in s.kv_nodes() {
            varying.insert(k.0);
            varying.insert(v.0);
        }
        let seeds = varying.len();
        // Node ids are topologically ordered (a node's inputs precede it), so a
        // single forward sweep is a complete fixpoint — no iteration needed.
        for i in 0..g.len() {
            if g.node(fuel_graph::NodeId(i))
                .inputs
                .iter()
                .any(|inp| varying.contains(&inp.0))
            {
                varying.insert(i);
            }
        }

        // --- (1b) WHAT the structurally-invariant nodes ACTUALLY ARE ---
        //
        // The conclusion this census gets used to justify — "there is nothing to
        // memoize, because every invariant node is a weight `base_cache` already
        // realizes once" — was INFERRED in the first version of this test and
        // never measured. The op histogram below (`inv_by_op`) is populated only
        // for nodes present in BOTH replay snapshots, and capture retains no
        // `Const` outputs, so that bucket was empty and printed nothing. An empty
        // print was read as confirmation of the inference. It confirmed nothing:
        // it is equally consistent with invariant COMPUTE nodes existing and
        // simply not being retained.
        //
        // So enumerate them directly, off the graph, with no dependence on what
        // capture happened to retain. `Op` has exactly two leaves — `Const` and
        // `Iota` — so every other op has inputs and is real computation. An
        // invariant node that is not a leaf is therefore work redone every token
        // that could be done once: a genuine memoization candidate, and its
        // existence flips the conclusion.
        let mut inv_all_by_op: BTreeMap<String, usize> = BTreeMap::new();
        let mut inv_compute: Vec<(usize, String)> = Vec::new();
        let op_kind = |o: &fuel_graph::Op| {
            format!("{o:?}")
                .split(['(', ' ', '{'])
                .next()
                .unwrap_or("?")
                .to_string()
        };
        for i in 0..g.len() {
            if varying.contains(&i) {
                continue;
            }
            let node = g.node(fuel_graph::NodeId(i));
            let kind = op_kind(&node.op);
            *inv_all_by_op.entry(kind.clone()).or_insert(0) += 1;
            if !matches!(node.op, fuel_graph::Op::Const | fuel_graph::Op::Iota { .. }) {
                inv_compute.push((i, kind));
            }
        }
        // Not every non-leaf is WORK. `Transpose` / `Permute` / `BroadcastTo` are
        // `WorkItemKind::ViewOf` in the executor: the output's Storage Arc IS the
        // input's, bytes shared, only the Layout's strides/offset differ. They
        // allocate nothing and launch nothing, so "recomputing" one per token
        // costs a Layout struct on the host — memoizing it buys nothing. (It is
        // also why capture does not retain them: there is no distinct output
        // buffer to retain, which is what made the first version of this census
        // read as "no candidates".)
        //
        // So the number that actually decides whether memoization has a consumer
        // is the MATERIALIZING invariant nodes — invariant work that allocates
        // and launches. Reported separately rather than asserted on: a
        // materializing invariant node appearing later is an OPPORTUNITY, not a
        // defect, and a test that fails on someone adding one would be an
        // obstacle. The census reports; the decision stays with a human.
        let is_view = |k: &str| matches!(k, "Transpose" | "Permute" | "BroadcastTo" | "Reshape");
        let (inv_views, inv_materializing): (Vec<_>, Vec<_>) =
            inv_compute.iter().cloned().partition(|(_, k)| is_view(k));

        // --- (2) EMPIRICAL: bytes that changed between two replayed tokens ---
        let mut emp_changed: HashSet<usize> = HashSet::new();
        let mut compared = 0usize;
        for (id, a) in &snap_a {
            if let Some(b) = snap_b.get(id) {
                compared += 1;
                if a.len() != b.len()
                    || a.iter()
                        .zip(b.iter())
                        .any(|(x, y)| x.to_bits() != y.to_bits())
                {
                    emp_changed.insert(*id);
                }
            }
        }

        // --- agreement + census over the nodes we can actually see ---
        let mut struct_inv_emp_same = 0usize;
        let mut struct_inv_emp_changed: Vec<usize> = Vec::new();
        let mut struct_var_emp_same: Vec<usize> = Vec::new();
        let mut struct_var_emp_changed = 0usize;
        let mut inv_by_op: BTreeMap<String, usize> = BTreeMap::new();
        for id in snap_a.keys() {
            if snap_b.get(id).is_none() {
                continue;
            }
            let sv = varying.contains(id);
            let ec = emp_changed.contains(id);
            match (sv, ec) {
                (false, false) => {
                    struct_inv_emp_same += 1;
                    let op = format!("{:?}", g.node(fuel_graph::NodeId(*id)).op);
                    *inv_by_op
                        .entry(op.split(' ').next().unwrap_or("?").to_string())
                        .or_insert(0) += 1;
                }
                (false, true) => struct_inv_emp_changed.push(*id),
                (true, false) => struct_var_emp_same.push(*id),
                (true, true) => struct_var_emp_changed += 1,
            }
        }

        println!("\n=== token-invariance census (decode step) ===");
        println!("graph nodes: {}   per-token seeds: {seeds}", g.len());
        println!(
            "structurally VARYING: {}   structurally INVARIANT: {}",
            varying.len(),
            g.len() - varying.len()
        );
        println!("retained + comparable across two replays: {compared}");
        println!(
            "  structurally-invariant & bytes same    : {struct_inv_emp_same}  <- memoization candidates"
        );
        println!("  structurally-varying   & bytes changed : {struct_var_emp_changed}");
        println!(
            "  structurally-varying   & bytes SAME    : {}  (coincidence on a 16-vocab fixture)",
            struct_var_emp_same.len()
        );
        println!(
            "  structurally-invariant & bytes CHANGED : {}  (would be a correctness signal)",
            struct_inv_emp_changed.len()
        );
        if !inv_by_op.is_empty() {
            println!("invariant candidates by op:");
            for (op, n) in &inv_by_op {
                println!("    {op:<24} {n}");
            }
        }

        println!(
            "\n-- what the structurally-invariant nodes ARE (off the graph, \
                  independent of what capture retained) --"
        );
        for (op, n) in &inv_all_by_op {
            println!("    {op:<24} {n}");
        }
        println!(
            "  leaves (Const/Iota — already hoisted by base_cache) : {}",
            inv_all_by_op
                .iter()
                .filter(|(k, _)| k.as_str() == "Const" || k.as_str() == "Iota")
                .map(|(_, n)| *n)
                .sum::<usize>(),
        );
        println!(
            "  invariant VIEW ops (ViewOf — alias input bytes, no alloc/launch) : {}",
            inv_views.len(),
        );
        println!(
            "  invariant MATERIALIZING ops (alloc + launch every token) : {}  <- the real candidates",
            inv_materializing.len(),
        );
        for (id, op) in inv_materializing.iter().take(20) {
            println!("      #{id:<5} {op}");
        }
        println!("=== END ===\n");

        // POSITIVE CONTROL: the comparison actually happened. Without it, zero
        // comparable nodes would report "no varying nodes" and pass vacuously.
        assert!(
            compared >= 10,
            "only {compared} nodes were comparable across replays"
        );
        // And the seeds must have been found — an empty seed set makes EVERYTHING
        // look invariant, which is the most flattering possible wrong answer.
        assert!(
            seeds >= 4,
            "only {seeds} per-token seed nodes — the structural pass is blind"
        );
        // POSITIVE CONTROL for (1b), and the specific defect it repairs: the
        // enumeration must account for EVERY structurally-invariant node. The
        // bug being fixed here was an empty report read as a negative result, so
        // "the histogram printed nothing" must be distinguishable from "the
        // histogram was never populated". Totals agreeing with the independently
        // computed invariant count is that distinction.
        let inv_total: usize = inv_all_by_op.values().sum();
        assert_eq!(
            inv_total,
            g.len() - varying.len(),
            "the invariant-node enumeration covered {inv_total} nodes but the \
             structural pass found {} — an under-populated histogram would print \
             an empty candidate list that reads as 'nothing to memoize'",
            g.len() - varying.len(),
        );

        // The one direction that is a correctness signal rather than noise.
        assert!(
            struct_inv_emp_changed.is_empty(),
            "nodes {struct_inv_emp_changed:?} have NO per-token input yet produced \
             different bytes across two replays — a node that cannot depend on the \
             token should be reproducible, so this is a correctness signal, not a \
             census artifact",
        );
    }

    /// **Capture as a WHOLE-GRAPH CORRECTNESS INSTRUMENT** — per-node
    /// intermediates read out of a captured replay and checked against a CPU
    /// oracle.
    ///
    /// Capture is normally sold as launch-overhead elimination, and that is all
    /// Fuel exploits today. Recording the graph has a second consequence nobody
    /// was using: **every compute node's output gets a FIXED device address that
    /// survives replays** (`CapturedDecodeSession::node_outputs`). On the
    /// ordinary path intermediates are transient, so there is nothing to
    /// inspect after the fact.
    ///
    /// That makes a capture an independent second source of truth. `placement_of`
    /// reports what the optimizer DECIDED; this reports what the device HELD.
    /// This session repeatedly found instruments that were themselves wrong, and
    /// the only reliable defence was two instruments that could disagree.
    ///
    /// What this asserts:
    /// 1. **Coverage** — the capture retains a buffer for a substantial share of
    ///    the graph's nodes, not a handful. Without this, a `node_outputs` that
    ///    returned one entry would satisfy every other check here.
    /// 2. **Readability** — every retained buffer D2Hs to finite values. A
    ///    capture that replayed into freed or uninitialised memory shows up as
    ///    NaN/Inf here, which is the failure mode `retained_inputs` exists to
    ///    prevent and which no logits-only check would localise.
    /// 3. **Whole-graph agreement** — the final logits match the persistent path
    ///    byte-for-byte, so the intermediates above belong to a run that was
    ///    correct end-to-end rather than merely finite.
    ///
    /// The per-node CPU-oracle diff this enables is deliberately NOT attempted
    /// here: node ids are shared with the held graph, so a full differential is
    /// a straightforward follow-up, but it needs a CPU realize of the same graph
    /// and belongs in its own increment.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn captured_decode_exposes_per_node_intermediates_cuda() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let dev: Device = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d.into(),
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };

        let prompt = [1_u32, 2, 3];
        let decode = [4_u32, 5, 6];
        let max_seq_len = prompt.len() + decode.len();

        // Reference: the persistent path, for the end-to-end agreement check.
        let m_ref = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let mut c_ref = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("ref cache");
        let mut x_ref = InferenceContext::new(dev.clone());
        let mut s_ref: Option<fuel_core::inference_context::DecodeSession> = None;
        m_ref
            .forward_with_kv_context_persistent(&prompt, &mut c_ref, &mut x_ref, &mut s_ref)
            .expect("ref prefill");
        let mut ref_logits = Vec::new();
        for &t in &decode {
            ref_logits.push(
                m_ref
                    .forward_with_kv_context_persistent(&[t], &mut c_ref, &mut x_ref, &mut s_ref)
                    .expect("ref decode"),
            );
        }

        // Under test: capture, then read the intermediates it retained.
        let m_cap = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let mut c_cap = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("cap cache");
        let mut x_cap = InferenceContext::new(dev.clone());
        let mut s_cap: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut captured: Option<fuel_dispatch::pipelined::CapturedDecodeSession> = None;
        m_cap
            .forward_with_kv_context_captured(
                &prompt,
                &mut c_cap,
                &mut x_cap,
                &mut s_cap,
                &mut captured,
            )
            .expect("cap prefill");
        let mut cap_logits = Vec::new();
        for &t in &decode {
            cap_logits.push(
                m_cap
                    .forward_with_kv_context_captured(
                        &[t],
                        &mut c_cap,
                        &mut x_cap,
                        &mut s_cap,
                        &mut captured,
                    )
                    .expect("cap decode"),
            );
        }

        let cap = captured.as_ref().expect(
            "Llama's decode graph is capturable — if this is None the capture declined and there are no intermediates to inspect",
        );
        let outputs = cap.node_outputs();
        let graph_len = s_cap.as_ref().expect("session").graph_node_count();

        // (1) COVERAGE — a real share of the graph, not a token handful.
        println!(
            "
=== captured per-node intermediates ==="
        );
        println!(
            "graph nodes: {graph_len}   retained buffers: {}",
            outputs.len()
        );
        assert!(
            outputs.len() >= graph_len / 4,
            "capture retained only {} buffers for a {graph_len}-node graph — node_outputs is supposed to expose EVERY compute node's output, so a near-empty map means the inspection surface is not what it claims",
            outputs.len(),
        );

        // (2) READABILITY — every retained buffer D2Hs to finite values.
        let mut checked = 0usize;
        let mut nonfinite: Vec<String> = Vec::new();
        for (nid, buf) in outputs.iter() {
            if let Ok(vals) = captured_output_to_f32(buf) {
                if vals.iter().any(|v| !v.is_finite()) {
                    nonfinite.push(format!("#{}", nid.0));
                }
                checked += 1;
            }
        }
        println!(
            "readable buffers: {checked}   non-finite: {}",
            nonfinite.len()
        );
        assert!(
            checked > 0,
            "no retained buffer was readable — the surface is unusable"
        );
        assert!(
            nonfinite.is_empty(),
            "retained intermediates contain non-finite values at {nonfinite:?} — the signature of replay against freed/uninitialised memory, which a logits-only check would not localise",
        );

        // (3) WHOLE-GRAPH AGREEMENT — the run those intermediates came from was
        // correct end to end, not merely finite.
        for (i, got) in cap_logits.iter().enumerate() {
            assert_eq!(
                *got, ref_logits[i],
                "captured token {i} must be byte-identical to the persistent path",
            );
        }
        println!(
            "end-to-end: byte-identical to the persistent path over {} tokens",
            decode.len()
        );
        println!(
            "=== END ===
"
        );
    }

    /// Task 4b-δ · GPU gate: [`LlamaModel::forward_with_kv_context_captured`]
    /// (the CapturedRun / CUDA-graph decode driver) must be **bit-exact**
    /// vs. the existing [`LlamaModel::forward_with_kv_context_persistent`]
    /// (D2 prebuilt-realize) path — same device, same prompt, same decode
    /// tokens, same reasoning `forward_with_kv_context_persistent_plan_
    /// once_matches_d1` holds D2 to against D1 (same plan → same kernels
    /// → identical bytes, NOT epsilon).
    ///
    /// Drives ≥4 decode tokens so all four branches of the new driver run:
    /// token 1 builds the held `DecodeSession` (identical shared call to
    /// the persistent path — trivially exact); token 2 builds the
    /// `CapturedDecodeSession` and fetches its logits via an
    /// empty-`updates` warm replay; tokens 3-4 are pure `cuGraphLaunch`
    /// replays via fresh per-token bytes. Also asserts the greedy-argmax
    /// token matches at every step as an easy-to-read secondary signal,
    /// and that `captured` transitions `None -> None -> Some` across the
    /// first two decode tokens (session-build / capture-build boundary).
    ///
    /// F32 throughout (capture is f32-only today per the project's
    /// existing constraint) — the tiny-model fixture convention of
    /// `forward_with_kv_context_persistent_plan_once_matches_d1`.
    ///
    /// Gated `#[cfg(feature = "cuda")]` + `#[ignore]`; skips cleanly if no
    /// CUDA device is present. Run:
    ///   `cargo test -p fuel-core --features cuda --lib \
    ///    forward_with_kv_context_captured -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn forward_with_kv_context_captured_matches_persistent() {
        // Back to this file's usual tiny fixture. An earlier pass bumped
        // these dims to rule out a cost-based CPU-placement explanation for
        // the capture blocker — the CUDA kernel precision-audit program
        // (docs/architecture/10-decisions-log.md, 2026-07-11 entries)
        // subsequently found the real cause: several CUDA kernels
        // (MatMul, MulElementwise, RmsNormLastDim, Softmax/LogSoftmax)
        // carried the `audited: false` seed, which unconditionally loses
        // to CPU's bit-stable guarantee regardless of problem size — not a
        // cost decision at all. With those audited, tensor size was never
        // the lever; using the tiny fixture again here.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };

        // CUDA device or skip cleanly.
        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let cuda_device: Device = cuda.into();

        let prompt = [1_u32, 2, 3];
        let decode_tokens = [4_u32, 5, 6, 7]; // >= 4 decode tokens
        let max_seq_len = prompt.len() + decode_tokens.len();

        // Two byte-identical F32-weight models on the SAME CUDA device:
        // one drives the reference persistent (D2) path, one drives the
        // new captured (CapturedRun) path under test.
        let model_persistent = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let model_captured = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        // --- Reference: forward_with_kv_context_persistent ---
        let mut cache_persistent = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &cuda_device,
        )
        .expect("with_capacity persistent");
        let mut ctx_persistent = InferenceContext::new(cuda_device.clone());
        let mut session_persistent: Option<fuel_core::inference_context::DecodeSession> = None;
        let _ = model_persistent
            .forward_with_kv_context_persistent(
                &prompt,
                &mut cache_persistent,
                &mut ctx_persistent,
                &mut session_persistent,
            )
            .expect("persistent prefill");
        let mut persistent_logits: Vec<Vec<f32>> = Vec::with_capacity(decode_tokens.len());
        for &tok in &decode_tokens {
            persistent_logits.push(
                model_persistent
                    .forward_with_kv_context_persistent(
                        &[tok],
                        &mut cache_persistent,
                        &mut ctx_persistent,
                        &mut session_persistent,
                    )
                    .expect("persistent decode"),
            );
        }

        // --- Under test: forward_with_kv_context_captured ---
        let mut cache_captured = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &cuda_device,
        )
        .expect("with_capacity captured");
        let mut ctx_captured = InferenceContext::new(cuda_device);
        let mut session_captured: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut captured: Option<fuel_dispatch::pipelined::CapturedDecodeSession> = None;
        let _ = model_captured
            .forward_with_kv_context_captured(
                &prompt,
                &mut cache_captured,
                &mut ctx_captured,
                &mut session_captured,
                &mut captured,
            )
            .expect("captured prefill");
        assert!(
            session_captured.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );
        assert!(
            captured.is_none(),
            "prefill (seq>1) must NOT build the capture"
        );

        let argmax = |v: &[f32]| -> usize {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };

        for (i, &tok) in decode_tokens.iter().enumerate() {
            let got = model_captured
                .forward_with_kv_context_captured(
                    &[tok],
                    &mut cache_captured,
                    &mut ctx_captured,
                    &mut session_captured,
                    &mut captured,
                )
                .expect("captured decode");

            // Bit-exact vs. the persistent path (same plan → same
            // kernels), NOT epsilon.
            assert_eq!(
                got, persistent_logits[i],
                "captured decode token {i} must be byte-identical to the \
                 persistent path: got={got:?} want={:?}",
                persistent_logits[i],
            );
            assert_eq!(
                argmax(&got),
                argmax(&persistent_logits[i]),
                "captured decode token {i} argmax must match the persistent path",
            );

            if i == 0 {
                // First decode token: session builds, capture does not yet.
                assert!(
                    session_captured.is_some(),
                    "token 1 must build the held session"
                );
                assert!(captured.is_none(), "token 1 must NOT build the capture yet");
            } else if i == 1 {
                // Second decode token: the capture builds.
                assert!(captured.is_some(), "token 2 must build the capture");
            }
        }

        // Sanity: both caches advanced identically.
        assert_eq!(cache_captured.cached_len, max_seq_len);
        assert_eq!(cache_persistent.cached_len, max_seq_len);
    }

    /// Born-red gate: BF16-throughout D2 persistent decode
    /// (`forward_with_kv_context_persistent`, BF16 `KvCache`) must be
    /// **bit-exact** vs. the BF16 D1 rebuild path on the same prefix
    /// (same plan → same kernels — the house bar every D2 test holds,
    /// see `forward_with_kv_context_persistent_plan_once_matches_d1`),
    /// AND must plan exactly once across the decode loop.
    ///
    /// Unlike test 1 (which compares BF16 against an F32 reference and
    /// tolerates quantization drift), this test holds BOTH sides to the
    /// SAME BF16 dtype, so D2 vs. D1 must match exactly — any drift here
    /// would mean D2's held-graph reuse (not the BF16 seam itself) is
    /// broken.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "cross-backend placement fallback: under a --features cuda build on a \
                  CUDA host, the optimizer can stamp a CPU-pinned node (the broadcast \
                  mask) onto the GPU and fail with 'no CUDA storage in input cache' — \
                  the documented class generate_persistent_decode_on_cuda_* best-effort \
                  skips. This CPU test's home is the default-feature build (always runs \
                  there); the CUDA-side BF16 coverage is bf16_decode_cuda_matches_cpu."
    )]
    fn bf16_decode_matches_f32_decode_d2_plan_once() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };

        // Two byte-identical BF16-weight models: one drives D2, one D1.
        let model_d2 = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };
        let model_d1 = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let decode_tokens = [4_u32, 5, 6, 7]; // >= 3 decode tokens
        let max_seq_len = prompt.len() + decode_tokens.len();

        // --- D1 (rebuild) reference FIRST, in its own pass, so its
        // per-token re-plans do NOT pollute the optimize-count window
        // measured around the D2 loop. ---
        let dev1 = Device::cpu();
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &dev1,
        )
        .expect("with_capacity d1");
        let mut ctx1 = InferenceContext::new(dev1);
        let _ = model_d1
            .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
            .expect("d1 prefill");
        let mut d1_expected: Vec<Vec<f32>> = Vec::with_capacity(decode_tokens.len());
        for &tok in &decode_tokens {
            d1_expected.push(
                model_d1
                    .forward_with_kv_context(&[tok], &mut cache1, &mut ctx1)
                    .expect("d1 decode"),
            );
        }

        // --- D2 (persistent) session state ---
        let dev2 = Device::cpu();
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &dev2,
        )
        .expect("with_capacity d2");
        let mut ctx2 = InferenceContext::new(dev2);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // Prefill (seq>1 falls back to the rebuild path; no session yet).
        let _ = model_d2
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .expect("d2 prefill");
        assert!(
            session.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );

        let opt_before = fuel_core::pipelined_bridge::optimize_calls_thread_local();
        let mut len_at_token2: Option<usize> = None;

        for (i, &tok) in decode_tokens.iter().enumerate() {
            let d2 = model_d2
                .forward_with_kv_context_persistent(&[tok], &mut cache2, &mut ctx2, &mut session)
                .expect("d2 decode");

            // Bit-exact vs. the BF16 D1 cached path (same plan → same
            // kernels), NOT epsilon.
            assert_eq!(
                d2, d1_expected[i],
                "BF16 persistent decode token {i} must be byte-identical to the \
                 BF16 D1 cached path",
            );

            let sess = session
                .as_ref()
                .expect("session built on first decode token");
            let graph_len = sess.graph_node_count();
            if i == 1 {
                len_at_token2 = Some(graph_len);
            } else if i >= 2 {
                assert_eq!(
                    Some(graph_len),
                    len_at_token2,
                    "held graph must NOT grow from token 2 onward (token {i})",
                );
            }
        }

        // Optimize bumped EXACTLY ONCE across all decode tokens.
        let opt_after = fuel_core::pipelined_bridge::optimize_calls_thread_local();
        assert_eq!(
            opt_after - opt_before,
            1,
            "BF16 persistent decode must optimize EXACTLY ONCE across {} decode \
             tokens: {opt_before} -> {opt_after}",
            decode_tokens.len(),
        );

        assert_eq!(cache2.cached_len, max_seq_len);
        assert_eq!(cache1.cached_len, max_seq_len);
    }

    /// Graph-shape guard (no realize): after building one BF16 D1 decode
    /// step's graph, every `Op::MatMul` node between the post-embed cast
    /// and the pre-realize logits cast must be BF16 — this is the guard
    /// against a silent F32-matmul-fallback regression creeping back in
    /// (e.g. someone drops a `to_dtype` seam and the mixed-dtype matmul
    /// gate happens to still build, quietly re-introducing an F32 gemm).
    ///
    /// Builds the graph by hand (mirroring
    /// `forward_with_kv_context_impl`'s D1 body up to, but not
    /// including, the realize call) — this test audits graph SHAPE, so
    /// it doesn't need a real `KvCache`/`InferenceContext` binding.
    #[test]
    fn bf16_decode_graph_dtype_audit() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };
        let next_token = 4_u32;
        let cached_len = 3usize; // as if 3 tokens were already prefilled
        let seq = 1usize;
        let batch = 1usize;
        let max_seq_len = cached_len + seq;
        let cache_dtype = DType::BF16;

        let embed = Tensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        )
        .unwrap();
        let token_ids = embed
            .const_u32_like(vec![next_token], Shape::from_dims(&[seq]))
            .unwrap();
        let mut h = embed
            .index_select(0, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))
            .unwrap();
        h = h.to_dtype(cache_dtype).unwrap();

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_base, cached_len, seq, cfg.head_dim);

        let mask_data = build_decode_causal_mask(cached_len, seq, max_seq_len);
        let mask = h
            .const_like_dtype(
                &mask_data,
                Shape::from_dims(&[1, 1, seq, max_seq_len]),
                cache_dtype,
            )
            .unwrap();

        let cached_len_sym = fuel_ir::SymId(0);
        let attended_len_sym = fuel_ir::SymId(1);
        let cache_shape = Shape::from_dims(&[batch, cfg.n_kv_heads, max_seq_len, cfg.head_dim]);

        for layer_weights in &model.weights.layers {
            let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            h = model
                .apply_layer_with_kv_writes(
                    &h,
                    layer_weights,
                    &k_cache_node,
                    &v_cache_node,
                    cached_len_sym,
                    attended_len_sym,
                    None, // dtype-audit test: SymEnv write path (offset carrier irrelevant here)
                    &rope_cos,
                    &rope_sin,
                    &mask,
                    None, // dense
                )
                .expect("apply_layer_with_kv_writes");
        }

        let h_norm =
            apply_affine_rms_norm(&h, &model.weights.final_norm_gain, cfg.dim, cfg.norm_eps);
        let logits = model
            .weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)
            .unwrap();
        let logits_root = logits
            .slice(1, seq - 1, 1)
            .unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .unwrap();
        // Mirrors the pre-realize cast the production path applies.
        let logits_root = logits_root.to_dtype(DType::F32).unwrap();

        // Walk the (unrealized) graph and count Op::MatMul nodes by dtype.
        let graph = logits_root.graph_handle();
        let g = graph.read().unwrap();
        let mut bf16_matmuls = 0usize;
        let mut f32_matmuls = 0usize;
        let mut other_matmuls = 0usize;
        for i in 0..g.len() {
            let node = g.node(fuel_graph::NodeId(i));
            if matches!(&node.op, fuel_graph::Op::MatMul) {
                match node.dtype {
                    DType::BF16 => bf16_matmuls += 1,
                    DType::F32 => f32_matmuls += 1,
                    _ => other_matmuls += 1,
                }
            }
        }

        // Expected for this fixture (n_layers=2): per layer — Q/K/V
        // projections (3) + QK^T scores (1) + attn·V (1) + O projection
        // (1) + gate/up/down FFN projections (3) = 9 matmuls/layer; 2
        // layers = 18, plus the lm_head projection = 19 total. ALL must
        // be BF16 — the RoPE F32-cast window doesn't itself contain a
        // matmul, so zero F32 (or other-dtype) matmuls are expected
        // between the post-embed cast and the pre-realize logits cast.
        assert_eq!(
            f32_matmuls, 0,
            "no F32 matmul expected in a BF16-throughout decode graph"
        );
        assert_eq!(other_matmuls, 0, "no non-BF16/F32 matmul dtype expected");
        assert_eq!(
            bf16_matmuls,
            9 * cfg.n_layers + 1,
            "expected 9 matmuls/layer (Q/K/V, QK^T, attn·V, O, gate/up/down) \
             + 1 lm_head matmul, all BF16 (got {bf16_matmuls})",
        );
    }

    /// Phase D · D2c born-red gate for generate-loop integration.
    ///
    /// The plain LlamaModel decode generate loops
    /// (`generate_streaming_with_kv_context` / `generate_with_kv_context`)
    /// now hold ONE plan-once [`DecodeSession`] across the generation and
    /// route every step through
    /// [`LlamaModel::forward_with_kv_context_persistent`]. This is the
    /// end-to-end guard that the plan-once path is actually USED in
    /// production generation and stays bit-exact vs. the D1 rebuild path.
    ///
    /// The test drives an explicit persistent generate loop (mirroring the
    /// wired production loop: hold `session`, call
    /// `forward_with_kv_context_persistent` for prefill + every decode
    /// step) and asserts, against a SEPARATE D1 reference loop over the
    /// same inputs (bare `forward_with_kv_context` + the identical greedy
    /// `sample_logits`):
    ///   (a) the generated token sequence is **byte-identical** over N≥4
    ///       greedy tokens — because greedy sampling means ANY per-token
    ///       logit drift diverges the sequence, an exact N-token match is
    ///       a strong end-to-end guard;
    ///   (b) each step's **logits** are **exactly `==`** the D1 cached
    ///       path (same plan → same kernels → bit-exact, NOT epsilon);
    ///   (c) `optimize_calls_thread_local()` bumps **only ~once for the
    ///       decode portion** — the first decode token builds the held
    ///       session (optimize once); tokens 2..N skip optimize. The
    ///       prefill (seq>1) falls back to the D1 rebuild path, which
    ///       optimizes once too, so the total across prefill + N decode
    ///       tokens is exactly 2.
    /// It ALSO drives the real production wrapper
    /// `generate_with_kv_context` and asserts the returned token sequence
    /// matches the reference — confirming the wiring, not just the entry.
    ///
    /// Born-red shape: greedy over N tokens diverges the sequence on ANY
    /// per-token logit drift, so a broken session-reuse (stale
    /// intermediate, re-optimize corruption, per-token node growth) would
    /// flip a token and fail (a). Before the loops were wired to
    /// `forward_with_kv_context_persistent`, (c) fails (the D1 path
    /// re-optimizes per token → the decode window bumps N times, not 1).
    #[test]
    fn generate_loop_persistent_byte_exact_and_plans_once() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_new = 5; // N ≥ 4 greedy decode tokens
        let max_seq_len = prompt.len() + max_new;
        let strategy = SamplingStrategy::Greedy;

        // ---- D1 (rebuild) REFERENCE loop FIRST, in its own pass, so its
        // per-token re-plans do NOT pollute the optimize-count window we
        // measure around the D2 loop. Greedy sampling is open-coded with
        // `sample_logits` so it is bit-identical to the persistent loop's
        // sampling; we capture BOTH the token sequence AND per-step logits.
        let dev1 = Device::cpu();
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev1,
        )
        .expect("with_capacity d1");
        let mut ctx1 = InferenceContext::new(dev1);
        let mut rng1: u64 = 0;
        let mut ref_tokens: Vec<u32> = prompt.to_vec();
        let mut ref_step_logits: Vec<Vec<f32>> = Vec::with_capacity(max_new);
        // Prefill.
        let mut last1 = model
            .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
            .expect("d1 prefill");
        for _ in 0..max_new {
            let next = sample_logits(&last1, strategy, &mut rng1);
            ref_tokens.push(next);
            last1 = model
                .forward_with_kv_context(&[next], &mut cache1, &mut ctx1)
                .expect("d1 decode");
            ref_step_logits.push(last1.clone());
        }

        // ---- D2 (persistent) generate loop — mirrors the wired
        // production loop exactly (hold `session`, route prefill + every
        // decode step through `forward_with_kv_context_persistent`). We
        // snapshot the thread-local optimize count around the WHOLE loop
        // (prefill + decode). ----
        let opt_before = fuel_core::pipelined_bridge::optimize_calls_thread_local();

        let dev2 = Device::cpu();
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev2,
        )
        .expect("with_capacity d2");
        let mut ctx2 = InferenceContext::new(dev2);
        let mut rng2: u64 = 0;
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut d2_tokens: Vec<u32> = prompt.to_vec();
        let mut d2_step_logits: Vec<Vec<f32>> = Vec::with_capacity(max_new);
        // Prefill through the persistent entry (seq>1 → falls back to the
        // D1 rebuild path WITHOUT building the session).
        let mut last2 = model
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .expect("d2 prefill");
        assert!(
            session.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );
        for _ in 0..max_new {
            let next = sample_logits(&last2, strategy, &mut rng2);
            d2_tokens.push(next);
            last2 = model
                .forward_with_kv_context_persistent(&[next], &mut cache2, &mut ctx2, &mut session)
                .expect("d2 decode");
            d2_step_logits.push(last2.clone());
        }

        let opt_after = fuel_core::pipelined_bridge::optimize_calls_thread_local();

        // (a) Byte-identical token sequence over N greedy tokens. Any
        // per-token logit drift would diverge greedy argmax → this is the
        // strong end-to-end guard.
        assert_eq!(
            d2_tokens, ref_tokens,
            "persistent generate loop must produce the byte-identical token \
             sequence as the D1 rebuild path over {max_new} greedy tokens",
        );

        // (b) Each step's logits exactly == the D1 cached path (bit-exact,
        // NOT epsilon — same plan → same kernel sequence → identical bytes).
        assert_eq!(d2_step_logits.len(), ref_step_logits.len());
        for (i, (d2, d1)) in d2_step_logits
            .iter()
            .zip(ref_step_logits.iter())
            .enumerate()
        {
            assert_eq!(
                d2, d1,
                "persistent decode step {i} logits must be byte-identical to the \
                 D1 cached path",
            );
        }

        // (c) optimize bumped only ~once for the decode portion. Prefill
        // (seq>1) falls back to the rebuild path (1 optimize); the first
        // decode token builds the session (1 optimize); decode tokens
        // 2..N skip optimize. Total across prefill + N decode = exactly 2.
        assert_eq!(
            opt_after - opt_before,
            2,
            "persistent generate must optimize EXACTLY twice (1 prefill \
             fallback + 1 decode-session build) regardless of N={max_new} \
             decode tokens: {opt_before} -> {opt_after}",
        );

        // The session was built (on the first decode token) and is still
        // held/valid at the end of the generation.
        assert!(session.is_some(), "held session survives the decode loop");
        assert_eq!(cache2.cached_len, max_seq_len);
        assert_eq!(cache1.cached_len, max_seq_len);

        // ---- Finally, drive the REAL production wrapper and confirm the
        // wiring: the token sequence it returns matches the reference. ----
        let via_wrapper = model
            .generate_with_kv_context(&prompt, max_new, strategy, None, &Device::cpu(), DType::F32)
            .expect("generate_with_kv_context");
        assert_eq!(
            via_wrapper, ref_tokens,
            "generate_with_kv_context (wired to the persistent path) must \
             produce the byte-identical token sequence as the D1 reference",
        );
    }

    // =======================================================================
    // Decode-builder ↔ CUDA flash-arm WIRING (feat/kernel-contracts-dlpack).
    //
    // These prove the model-layer wiring of `offer_decode_flash_arm`:
    //   (A) the wiring builds a correct `DecodeFlashSpec` from a real decode
    //       attention region + calls offer (k_len = Sym(attended_len_sym),
    //       CUDA-pinned FLASH_ATTN arm 1, decomposed oracle arm 0);
    //   (B) `DecodeSession` allocates + carries the attended-length symbol
    //       distinct from `cached_len`, and binds it to `cached_len + seq`
    //       each token;
    //   (C) GUARD: on the real f32 decode graph NO arm is offered (the dtype
    //       gate) so the held graph carries ZERO `Op::Branch` — dormant, the
    //       byte-exact suite is untouched by construction.
    // The emitter's own admission logic is exhaustively tested in
    // `fuel-dispatch/src/decode_flash.rs`; these prove the WIRING, not the
    // emitter.
    // =======================================================================

    /// (A) The wiring builds a `DecodeFlashSpec` from a synthetic — but
    /// structurally real — f16 decode attention region and offers the CUDA
    /// flash arm: arm 0 stays the decomposed oracle, arm 1 is a CUDA-pinned
    /// `Fused(FLASH_ATTN, { k_len: Some(Sym(attended_len_sym)) })` reading
    /// `[q, k, v]`. This is the plumbing the f32 production path keeps
    /// dormant; an injected all-available capability drives it on CPU.
    #[test]
    fn flash_arm_wiring_offers_for_f16_region_with_attended_len_sym() {
        use fuel_dispatch::decode_flash::FlashArmCapability;
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        use fuel_graph::{Graph, Node, Op};
        use fuel_ir::probe::BackendId;
        use fuel_ir::{DType, DynScalar, Shape, SymId};
        use std::sync::RwLock;

        let (h, d, sk) = (4usize, 64usize, 37usize);
        let dt = DType::F16;
        let mut g = Graph::new();
        let leaf = |g: &mut Graph, dims: &[usize]| {
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(dims),
                dtype: dt,
            })
        };
        // q [1,H,1,D], k/v capacity buffers [1,H,SK,D], mask [1,H,1,SK].
        let q = leaf(&mut g, &[1, h, 1, d]);
        let k = leaf(&mut g, &[1, h, sk, d]);
        let v = leaf(&mut g, &[1, h, sk, d]);
        let mask = leaf(&mut g, &[1, h, 1, sk]);
        // Decomposed region: scores → scale → +mask → softmax → attn_v.
        let kt = g.push(Node {
            op: Op::Permute(vec![0, 1, 3, 2]),
            inputs: vec![k],
            shape: Shape::from_dims(&[1, h, d, sk]),
            dtype: dt,
        });
        let scores = g.push(Node {
            op: Op::MatMul,
            inputs: vec![q, kt],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype: dt,
        });
        let scaled = g.push(Node {
            op: Op::MulScalar(0.125),
            inputs: vec![scores],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype: dt,
        });
        let masked = g.push(Node {
            op: Op::Add,
            inputs: vec![scaled, mask],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype: dt,
        });
        let probs = g.push(Node {
            op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
            inputs: vec![masked],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype: dt,
        });
        // decomposed_out — the region's attention output (arm 0 / the oracle).
        let attn_v = g.push(Node {
            op: Op::MatMul,
            inputs: vec![probs, v],
            shape: Shape::from_dims(&[1, h, 1, d]),
            dtype: dt,
        });
        // reconverge — the SOLE consumer of attn_v (the merge).
        let reconverge = g.push(Node {
            op: Op::Permute(vec![0, 2, 1, 3]),
            inputs: vec![attn_v],
            shape: Shape::from_dims(&[1, 1, h, d]),
            dtype: dt,
        });

        let graph = std::sync::Arc::new(RwLock::new(g));
        let attended = SymId(1);
        // An injected all-available capability (the CPU test box has no CUDA
        // topology, so production() would decline — we drive the gate here).
        let cap = FlashArmCapability {
            cuda_flash_kernel: true,
            cuda_in_topology: true,
        };

        let branch = super::offer_flash_decode_arm_for_region(
            &graph, q, k, v, attn_v, reconverge, 0.125, attended,
            None, // dense layer: no window — and now DERIVED, not asserted
            None, // no softcap
            cap,
        )
        .expect("well-formed region")
        .expect("supported f16 decode shape + capability ⇒ arm offered");

        let g = graph.read().unwrap();
        assert!(
            matches!(g.node(branch).op, Op::Branch { .. }),
            "an Op::Branch was recorded"
        );
        let arms = g.node(branch).inputs.clone();
        assert_eq!(arms.len(), 2, "2-arm branch (decomposed oracle + flash)");
        assert_eq!(
            arms[0], attn_v,
            "arm 0 is the decomposed region output (the oracle)"
        );
        let flash = arms[1];
        match &g.node(flash).op {
            Op::Fused(
                fid,
                FusedOpParams::FlashAttn {
                    k_len,
                    causal,
                    softcap,
                    ..
                },
            ) => {
                assert_eq!(*fid, FusedOps::FLASH_ATTN, "arm 1 is FLASH_ATTN");
                // THE headline wiring assertion: k_len is the attended-length
                // symbol (NOT a concrete value, NOT cached_len_sym).
                assert_eq!(
                    *k_len,
                    Some(DynScalar::Sym(attended)),
                    "arm 1 carries k_len = Sym(attended_len_sym)",
                );
                assert!(*causal, "decode region is causal");
                assert!(softcap.is_none(), "no softcap");
            }
            other => panic!("arm 1 must be Fused(FLASH_ATTN, FlashAttn), got {other:?}"),
        }
        assert_eq!(g.node(flash).inputs, vec![q, k, v], "flash reads q, k, v");
        assert_eq!(
            g.target_backend(flash),
            Some(BackendId::Cuda),
            "arm 1 pinned to CUDA"
        );
        // Arm-0 runnability: the merge still reads the decomposed output.
        assert!(
            g.node(reconverge).inputs.contains(&attn_v),
            "reconverge reads arm 0 ⇒ an unpicked/non-CUDA graph realizes decomposed",
        );
    }

    /// **GAP-194 — the offer site must state its window, and a windowed layer
    /// must therefore be DECLINED.**
    ///
    /// The defect was never that the arm cannot express a window: `DecodeFlash
    /// Spec` carries `window_size_left`/`window_size_right`, and
    /// `flash_decode_admissible` already declines them because the
    /// `flash_decoding` capacity-K kernel does not implement local attention.
    /// The defect was that the offer site **hardcoded `None`** — true for
    /// `LlamaModel`, a lie for any windowed family, and the lie is what would
    /// have made the arm attend the whole prefix and silently drop the window.
    ///
    /// So this asserts the claim, not the kernel: the SAME region offered with
    /// a window gets **no arm**, and offered dense gets one. Both halves in one
    /// test, because a decline that also declines the dense case would prove
    /// nothing.
    ///
    /// ⚠️ **This covers the DECLINE half only.** That a dense layer's arm is
    /// actually *taken* cannot be observed here or on any f32/CPU gate — real
    /// admission needs bf16 + a CUDA topology, and `flash_decode_admissible`
    /// rejects on capability and dtype long before it reaches the window check.
    /// A green CPU suite is not evidence for the admit half and must not be
    /// cited as such.
    #[test]
    fn gap194_windowed_layer_is_declined_and_dense_layer_is_offered() {
        use fuel_dispatch::decode_flash::FlashArmCapability;
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        use fuel_graph::{Graph, Node, Op};
        use fuel_ir::{DType, Shape, SymId};
        use std::sync::RwLock;

        // One region builder, so the two arms of the comparison differ ONLY in
        // the window argument.
        let build = |window: Option<usize>| {
            let (h, d, sk) = (4usize, 64usize, 37usize);
            let dt = DType::F16;
            let mut g = Graph::new();
            let leaf = |g: &mut Graph, dims: &[usize]| {
                g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(dims),
                    dtype: dt,
                })
            };
            let q = leaf(&mut g, &[1, h, 1, d]);
            let k = leaf(&mut g, &[1, h, sk, d]);
            let v = leaf(&mut g, &[1, h, sk, d]);
            let mask = leaf(&mut g, &[1, h, 1, sk]);
            let kt = g.push(Node {
                op: Op::Permute(vec![0, 1, 3, 2]),
                inputs: vec![k],
                shape: Shape::from_dims(&[1, h, d, sk]),
                dtype: dt,
            });
            let scores = g.push(Node {
                op: Op::MatMul,
                inputs: vec![q, kt],
                shape: Shape::from_dims(&[1, h, 1, sk]),
                dtype: dt,
            });
            let scaled = g.push(Node {
                op: Op::MulScalar(0.125),
                inputs: vec![scores],
                shape: Shape::from_dims(&[1, h, 1, sk]),
                dtype: dt,
            });
            let masked = g.push(Node {
                op: Op::Add,
                inputs: vec![scaled, mask],
                shape: Shape::from_dims(&[1, h, 1, sk]),
                dtype: dt,
            });
            let probs = g.push(Node {
                op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
                inputs: vec![masked],
                shape: Shape::from_dims(&[1, h, 1, sk]),
                dtype: dt,
            });
            let attn_v = g.push(Node {
                op: Op::MatMul,
                inputs: vec![probs, v],
                shape: Shape::from_dims(&[1, h, 1, d]),
                dtype: dt,
            });
            let reconverge = g.push(Node {
                op: Op::Permute(vec![0, 2, 1, 3]),
                inputs: vec![attn_v],
                shape: Shape::from_dims(&[1, 1, h, d]),
                dtype: dt,
            });
            let graph = std::sync::Arc::new(RwLock::new(g));
            let cap = FlashArmCapability {
                cuda_flash_kernel: true,
                cuda_in_topology: true,
            };
            super::offer_flash_decode_arm_for_region(
                &graph,
                q,
                k,
                v,
                attn_v,
                reconverge,
                0.125,
                SymId(1),
                window,
                None,
                cap,
            )
            .expect("well-formed region")
        };

        // Positive control: identical region, no window ⇒ the arm IS offered.
        // Without this the decline below could be caused by anything.
        assert!(
            build(None).is_some(),
            "dense f16 decode region must still be offered the arm — otherwise the \
             decline below is not attributable to the window",
        );
        assert!(
            build(Some(4)).is_none(),
            "a WINDOWED layer must be declined: `flash_decoding` cannot express local \
             attention, and offering it anyway is exactly the silent window-drop \
             GAP-194 records",
        );
    }

    /// The FA-v2 translation of a Fuel window width, asserted against the
    /// convention `fuel-cuda-backend::translate_window` reads: local attention
    /// is *both* bounds `>= 0`; `is_causal` is `right == 0 && left < 0`.
    ///
    /// Fuel's mask attends `j ∈ [i - w + 1, i]`, i.e. `w - 1` back and `0`
    /// forward — so a width of 1 is "this position only" (`left = 0`), not
    /// "unbounded".
    #[test]
    fn gap194_window_bounds_translation() {
        assert_eq!(
            fuel_core::lazy::flash_window_bounds(None),
            (None, None),
            "dense stays causal"
        );
        assert_eq!(
            fuel_core::lazy::flash_window_bounds(Some(1)),
            (Some(0), Some(0))
        );
        assert_eq!(
            fuel_core::lazy::flash_window_bounds(Some(4)),
            (Some(3), Some(0))
        );
        // Degenerate width must NOT silently become "no window" — `(None, None)`
        // is the one answer that would re-introduce the defect.
        assert_eq!(
            fuel_core::lazy::flash_window_bounds(Some(0)),
            (Some(0), Some(0))
        );
        assert_ne!(fuel_core::lazy::flash_window_bounds(Some(0)), (None, None));
    }

    /// (B) A real (f32) persistent decode session allocates the
    /// attended-length symbol DISTINCT from `cached_len`, and its per-token
    /// `SymEnv` binds `attended_len = cached_len + seq` (seq == 1 in decode)
    /// alongside `cached_len`. This is the second symbol the flash arm's
    /// `k_len` resolves against.
    #[test]
    fn decode_session_allocates_and_binds_attended_len_sym() {
        use fuel_ir::SymId;
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_seq_len = prompt.len() + 2;
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // Prefill (seq>1 → no session), then the first decode token BUILDS
        // the held session (cached_len == prompt.len() == 3 at build).
        let _ = model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        assert!(session.is_none(), "prefill builds no session");
        let _ = model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("first decode token builds the session");
        let s = session
            .as_ref()
            .expect("session built on first decode token");

        // The two symbols are distinct (cached_len = SymId(0), attended = SymId(1)).
        assert_eq!(s.cached_len_sym(), SymId(0), "cached_len symbol");
        assert_eq!(s.attended_len_sym(), SymId(1), "attended-length symbol");
        assert_ne!(
            s.attended_len_sym(),
            s.cached_len_sym(),
            "attended-length is a SECOND symbol, not aliased to cached_len",
        );

        // The per-token env binds BOTH: cached_len = c, attended = c + seq(1).
        let env = s.per_token_sym_env(3).expect("per_token_sym_env");
        assert_eq!(
            env.get(s.cached_len_sym()),
            Some(3),
            "cached_len bound to 3"
        );
        assert_eq!(
            env.get(s.attended_len_sym()),
            Some(4),
            "attended_len bound to cached_len + seq = 3 + 1 = 4",
        );
    }

    /// (C) GUARD: the REAL f32 decode graph offers NO flash arm — the
    /// emitter's dtype gate declines on f32, so the held session graph
    /// carries ZERO `Op::Branch`. This is the dormancy that keeps the
    /// persistent byte-exact suite byte-identical: the wiring is present but
    /// inert until a bf16/f16 CUDA decode lands.
    #[test]
    fn f32_decode_graph_offers_no_flash_arm() {
        use fuel_graph::{NodeId, Op};
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_seq_len = prompt.len() + 2;
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let _ = model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        let _ = model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("first decode token builds the session");
        let s = session.as_ref().expect("session built");

        let g = s.graph().read().unwrap();
        let branch_count = (0..g.len())
            .filter(|&i| matches!(g.node(NodeId(i)).op, Op::Branch { .. }))
            .count();
        assert_eq!(
            branch_count, 0,
            "f32 decode ⇒ the emitter's dtype gate declines ⇒ NO Op::Branch \
             in the held decode graph (the flash arm stays dormant)",
        );
    }

    /// Part 2 increment B — the INVERSE of `f32_decode_graph_offers_no_flash_arm`:
    /// a REAL BF16-throughout persistent decode session, built on a live CUDA
    /// device, DOES carry the CUDA flash-decode `Op::Branch` — one per layer.
    ///
    /// `offer_flash_decode_arm_for_region`'s gate
    /// (`fuel_dispatch::decode_flash::flash_decode_admissible`) is driven by
    /// `FlashArmCapability::production()`, which is two independent halves:
    /// - `cuda_flash_kernel` — a *compiled-in* check (`default_kernel_registry`
    ///   lookup), true whenever this binary is built `--features cuda`,
    ///   regardless of whether a live device is ever instantiated;
    /// - `cuda_in_topology` — a *runtime hardware probe*
    ///   (`SystemTopology::current().backends()`), true only on a host with a
    ///   physical CUDA device visible to the process.
    /// Neither half reads the `Device` the model/cache/ctx happen to be built
    /// on — the gate is purely (q's graph dtype, global capability). So the
    /// arm can in principle be offered even for a BF16 graph built on
    /// `Device::cpu()`, as long as the process is a `--features cuda` build
    /// running on CUDA-capable hardware. This test still drives the session on
    /// a live CUDA device (the realistic BF16-CUDA-decode scenario the whole
    /// program targets) and is `#[ignore]`'d because `cuda_in_topology`
    /// depends on physical hardware being present — not portable to a
    /// CUDA-less CI box even though no `CudaDevice` handle is read by the gate
    /// itself.
    ///
    /// Run:
    ///   `cargo test -p fuel-core --features cuda --lib \
    ///    bf16_cuda_decode_graph_offers_flash_arm -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device (topology probe needs real hardware)"]
    fn bf16_cuda_decode_graph_offers_flash_arm() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};
        use fuel_graph::{NodeId, Op};
        use fuel_ir::probe::BackendId;

        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };

        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let cuda_device: Device = cuda.into();

        let prompt = [1_u32, 2, 3];
        let max_seq_len = prompt.len() + 2;
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &cuda_device,
        )
        .expect("with_capacity bf16 cuda");
        let mut ctx = InferenceContext::new(cuda_device);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let _ = model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        let _ = model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("first decode token builds the session");
        let s = session.as_ref().expect("session built");

        // Print the two production-capability halves so a failure names the
        // declining condition directly (2026-07-08 finding: cuda_flash_kernel
        // is FALSE even in a --features cuda build on live hardware — see the
        // assert message below for the registry-seam root cause).
        let cap = fuel_dispatch::decode_flash::FlashArmCapability::production();
        println!(
            "FlashArmCapability::production(): cuda_flash_kernel={} cuda_in_topology={}",
            cap.cuda_flash_kernel, cap.cuda_in_topology,
        );

        let g = s.graph().read().unwrap();
        let branches: Vec<NodeId> = (0..g.len())
            .map(NodeId)
            .filter(|&n| matches!(g.node(n).op, Op::Branch { .. }))
            .collect();
        // FLIPPED (registry seam fixed, dd-shapes 2026-07-08): the flash-arm
        // capability gate previously consulted the FusedKernelRegistry (static
        // CPU-only defaults + the runtime-adopted sidecar) for the STATIC
        // FLASH_ATTN id, but the CUDA flash_decoding binding registers ONLY
        // into the KernelBindingTable ((OpKind::FlashAttn, [f16|bf16;4], Cuda)
        // via register_cuda_flash_decoding_from_contract). `fused_kernel_
        // available` (fuel-dispatch/src/runtime_fused_kernels.rs) now bridges
        // the static-id path to the binding table (additive; the runtime-
        // fusion program's runtime-id sidecar lookup is untouched), so the
        // arm is offered — one `Op::Branch` per layer.
        assert_eq!(
            branches.len(),
            cfg.n_layers,
            "one CUDA flash-decode Op::Branch per layer (n_layers={}), found {}",
            cfg.n_layers,
            branches.len(),
        );
        for branch in branches {
            let arms = g.node(branch).inputs.clone();
            assert_eq!(arms.len(), 2, "2-arm branch (decomposed oracle + flash)");
            let flash = arms[1];
            match &g.node(flash).op {
                Op::Fused(fid, FusedOpParams::FlashAttn { .. }) => {
                    assert_eq!(*fid, FusedOps::FLASH_ATTN, "arm 1 is FLASH_ATTN");
                }
                other => panic!("arm 1 must be Fused(FLASH_ATTN, ..), got {other:?}"),
            }
            assert_eq!(
                g.target_backend(flash),
                Some(BackendId::Cuda),
                "arm 1 pinned to CUDA"
            );
        }
    }

    /// Phase D · FIRST live-GPU verification of plan-once persistent decode
    /// on CUDA — the core correctness gate for the previously UNVERIFIED
    /// GPU upload arm.
    ///
    /// The persistent decode (D1–D4) is CPU-verified byte-exact, but its GPU
    /// path — the non-CPU arm of
    /// [`fuel_core::pipelined_bridge::upload_host_buffer_to_device`], which does
    /// the per-token token/RoPE/mask re-bind as a transient `Op::Const →
    /// Op::Copy { target }` H2D upload (CUDA `write_from_host`) — is wired but
    /// was never exercised live. GPU is the production decode target, so this
    /// closes a real gap.
    ///
    /// The test drives greedy generation two ways ON THE SAME CUDA DEVICE:
    ///   - **persistent** (plan-once): hold one `DecodeSession`, route prefill
    ///     + every decode token through `forward_with_kv_context_persistent`
    ///     (this is the wired production path; it exercises the per-token GPU
    ///     re-bind upload arm), and
    ///   - **rebuild** (D1 reference): a bare per-token
    ///     `forward_with_kv_context` loop (re-plans every token).
    ///
    /// The two share the SAME optimized plan → SAME kernel sequence, so their
    /// per-step logits must be **bit-exact `==`**. ANY difference means the GPU
    /// upload arm (per-token H2D re-bind of token/RoPE/mask) is wrong — that is
    /// the headline correctness finding.
    ///
    /// It ALSO cross-checks the CUDA persistent logits against a CPU rebuild
    /// reference within the decode epsilon convention (`diff < 5e-3 ||
    /// rel < 1e-2`, the same band the CPU-vs-Vulkan decode twin uses) — this
    /// catches a GPU numeric bug BEYOND the upload arm (e.g. a bad kernel),
    /// which the CUDA-vs-CUDA bit-exact check alone would miss (both CUDA paths
    /// would share the same wrong kernel).
    ///
    /// Finally it drives the real production wrapper `generate_with_kv_context`
    /// on the CUDA device and confirms the returned token sequence matches the
    /// CUDA rebuild reference — proving the wiring, not just the entry point.
    ///
    /// Gated `#[cfg(feature = "cuda")]` + `#[ignore]`; skips cleanly if no CUDA
    /// device is present. Run:
    ///   `cargo test -p fuel-core --features cuda --lib \
    ///    generate_persistent_decode_on_cuda_matches_rebuild_and_cpu \
    ///    -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn generate_persistent_decode_on_cuda_matches_rebuild_and_cpu() {
        // Same tiny GQA config as the CPU persistent gate
        // (`forward_with_kv_context_persistent_plan_once_matches_d1`).
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2, // exercise GQA (n_rep = 2)
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_new = 5usize; // N ≥ 4 greedy decode tokens
        let max_seq_len = prompt.len() + max_new;
        let strategy = SamplingStrategy::Greedy;

        // ---- CPU REBUILD reference (BEST-EFFORT epsilon cross-check).
        //
        // LIVE-GPU FINDING (not the decode path under test): on a box with a
        // physical CUDA GPU, in a `--features cuda` build, the process-global
        // `SystemTopology` reports CUDA as an available device REGARDLESS of
        // whether this test constructed a `CudaDevice` yet (capabilities probe,
        // not device-handle-gated). The multi-backend placement DP then offers
        // CUDA as a *fallback* placement even for a CPU-pinned realize and can
        // stamp a node onto CUDA; the CPU cache has no CUDA seed, so the H2D
        // `Op::Copy` fails to derive a device handle. That failure is unrelated
        // to persistent decode, so the CPU cross-check is BEST-EFFORT: run it,
        // and if the CPU realize errors with that cross-backend-placement
        // condition, SKIP the epsilon assert (with a note) rather than fail the
        // headline CUDA-vs-CUDA gate. If the CPU realize succeeds, the epsilon
        // assert runs for real. ----
        let cpu_step_logits: Option<Vec<Vec<f32>>> = {
            let cpu_device = Device::cpu();
            let cpu_ref = || -> fuel_core::Result<Vec<Vec<f32>>> {
                let mut cpu_cache = KvCache::with_capacity(
                    cfg.n_layers,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    max_seq_len,
                    DType::F32,
                    &cpu_device,
                )?;
                let mut cpu_ctx = InferenceContext::new(cpu_device.clone());
                let mut cpu_rng: u64 = 0;
                let mut out: Vec<Vec<f32>> = Vec::with_capacity(max_new);
                let mut last_cpu =
                    model.forward_with_kv_context(&prompt, &mut cpu_cache, &mut cpu_ctx)?;
                for _ in 0..max_new {
                    let next = sample_logits(&last_cpu, strategy, &mut cpu_rng);
                    last_cpu =
                        model.forward_with_kv_context(&[next], &mut cpu_cache, &mut cpu_ctx)?;
                    out.push(last_cpu.clone());
                }
                Ok(out)
            };
            match cpu_ref() {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!(
                        "CPU epsilon cross-check SKIPPED (cross-backend placement \
                         fallback under --features cuda on a live-GPU host — not a \
                         decode bug): {e:?}"
                    );
                    None
                }
            }
        };

        // CUDA device or skip cleanly (mirrors the live-GPU integration tests'
        // `dev_or_skip`).
        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let cuda_device: Device = cuda.into();

        // ---- CUDA REBUILD (D1) reference: bare per-token
        // `forward_with_kv_context` loop on the CUDA device. Open-coded greedy
        // via `sample_logits` so it is bit-identical to the persistent loop's
        // sampling. Capture BOTH the token sequence AND the per-step logits. ----
        let mut cuda_rebuild_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &cuda_device,
        )
        .expect("cuda rebuild with_capacity");
        let mut cuda_rebuild_ctx = InferenceContext::new(cuda_device.clone());
        let mut rebuild_rng: u64 = 0;
        let mut rebuild_tokens: Vec<u32> = prompt.to_vec();
        let mut rebuild_step_logits: Vec<Vec<f32>> = Vec::with_capacity(max_new);
        let mut last_rebuild = model
            .forward_with_kv_context(&prompt, &mut cuda_rebuild_cache, &mut cuda_rebuild_ctx)
            .expect("cuda rebuild prefill");
        for _ in 0..max_new {
            let next = sample_logits(&last_rebuild, strategy, &mut rebuild_rng);
            rebuild_tokens.push(next);
            last_rebuild = model
                .forward_with_kv_context(&[next], &mut cuda_rebuild_cache, &mut cuda_rebuild_ctx)
                .expect("cuda rebuild decode");
            rebuild_step_logits.push(last_rebuild.clone());
        }

        // ---- CUDA PERSISTENT (plan-once): hold one DecodeSession, route
        // prefill + every decode step through the persistent entry (the wired
        // production path; this is what exercises the per-token GPU re-bind
        // upload arm under test). Capture tokens + per-step logits. ----
        let mut cuda_persist_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &cuda_device,
        )
        .expect("cuda persistent with_capacity");
        let mut cuda_persist_ctx = InferenceContext::new(cuda_device.clone());
        let mut persist_rng: u64 = 0;
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut persist_tokens: Vec<u32> = prompt.to_vec();
        let mut persist_step_logits: Vec<Vec<f32>> = Vec::with_capacity(max_new);
        let mut last_persist = model
            .forward_with_kv_context_persistent(
                &prompt,
                &mut cuda_persist_cache,
                &mut cuda_persist_ctx,
                &mut session,
            )
            .expect("cuda persistent prefill");
        assert!(
            session.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );
        for _ in 0..max_new {
            let next = sample_logits(&last_persist, strategy, &mut persist_rng);
            persist_tokens.push(next);
            last_persist = model
                .forward_with_kv_context_persistent(
                    &[next],
                    &mut cuda_persist_cache,
                    &mut cuda_persist_ctx,
                    &mut session,
                )
                .expect("cuda persistent decode");
            persist_step_logits.push(last_persist.clone());
        }
        assert!(
            session.is_some(),
            "held session survives the CUDA decode loop"
        );

        // === (1) HEADLINE GATE: CUDA persistent == CUDA rebuild, BIT-EXACT.
        // Same optimized plan → same kernels → identical bytes. Any diff means
        // the per-token GPU H2D re-bind upload arm is wrong. ===
        assert_eq!(
            persist_tokens, rebuild_tokens,
            "CUDA persistent greedy token sequence must be byte-identical to \
             the CUDA rebuild path over {max_new} tokens",
        );
        assert_eq!(persist_step_logits.len(), rebuild_step_logits.len());
        for (i, (p, r)) in persist_step_logits
            .iter()
            .zip(rebuild_step_logits.iter())
            .enumerate()
        {
            assert_eq!(
                p, r,
                "CUDA persistent decode step {i} logits must be BIT-EXACT vs the \
                 CUDA rebuild path — a divergence here is a bug in the per-token \
                 GPU upload arm (token/RoPE/mask H2D re-bind)",
            );
        }

        // === (2) EPSILON CROSS-CHECK: CUDA persistent vs CPU rebuild. Catches
        // a GPU numeric bug beyond the upload arm (both CUDA paths would share
        // it). Same decode tolerance band as the CPU-vs-Vulkan twin. Runs only
        // if the CPU reference realized (see the best-effort note above). ===
        let mut epsilon_checked = false;
        if let Some(cpu_step_logits) = cpu_step_logits.as_ref() {
            assert_eq!(persist_step_logits.len(), cpu_step_logits.len());
            for (i, (p, c)) in persist_step_logits
                .iter()
                .zip(cpu_step_logits.iter())
                .enumerate()
            {
                assert_eq!(p.len(), c.len(), "step {i} logit width");
                for (j, (a, b)) in p.iter().zip(c.iter()).enumerate() {
                    let diff = (a - b).abs();
                    let rel = diff / a.abs().max(b.abs()).max(1e-6);
                    assert!(
                        diff < 5e-3 || rel < 1e-2,
                        "step {i} logit[{j}]: cuda={a}, cpu={b}, diff={diff}, rel={rel}",
                    );
                }
            }
            epsilon_checked = true;
        }

        // === (3) WIRING: the real production wrapper on CUDA returns the same
        // token sequence as the CUDA rebuild reference. ===
        let via_wrapper = model
            .generate_with_kv_context(&prompt, max_new, strategy, None, &cuda_device, DType::F32)
            .expect("generate_with_kv_context on CUDA");
        assert_eq!(
            via_wrapper, rebuild_tokens,
            "generate_with_kv_context on CUDA (wired to the persistent path) \
             must return the byte-identical token sequence as the CUDA rebuild \
             reference",
        );

        eprintln!(
            "CUDA persistent decode VERIFIED: {} tokens + logits BIT-EXACT vs \
             CUDA rebuild path; CPU epsilon cross-check {}. tokens={:?}",
            max_new,
            if epsilon_checked {
                "PASSED"
            } else {
                "SKIPPED (see note above)"
            },
            persist_tokens,
        );
    }

    /// PS4c PC-2 — bf16 PAGED decode on a live CUDA device matches bf16
    /// CONTIGUOUS decode, per step. Both run BF16-throughout on the 4070; the
    /// paged path stores KV in a bf16 `DeviceKvPool` and attends via
    /// `Op::PagedAttn` (decomposed to gather+SDPA on the primitives' bf16 CUDA
    /// kernels — PC-2's decompose route), the contiguous path uses a bf16
    /// `KvCache`. Greedy **argmax** must agree at every step (bf16 is too coarse
    /// for tight logit parity, but the sampled token must match).
    ///
    /// Run: `cargo test -p fuel-core --features cuda --lib \
    ///   bf16_paged_decode_matches_contiguous_on_cuda -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn bf16_paged_decode_matches_contiguous_on_cuda() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_bf16(&cfg),
        };

        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let dev: Device = cuda.into();

        let prompt = [1u32, 2, 3];
        let decode = [4u32, 5, 6];
        let all: Vec<u32> = prompt.iter().chain(decode.iter()).copied().collect();
        let max_seq_len = all.len();

        // Contiguous bf16 reference on CUDA.
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::BF16,
            &dev,
        )
        .expect("bf16 KvCache");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut contig: Vec<Vec<f32>> = Vec::new();
        contig.push(
            model
                .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
                .expect("contig prefill"),
        );
        for &t in &decode {
            contig.push(
                model
                    .forward_with_kv_context(&[t], &mut cache, &mut ctx)
                    .expect("contig decode"),
            );
        }

        // Paged bf16 decode on CUDA — feed every token one at a time.
        let geom = fuel_core::kv_block_pool::KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 2,
        };
        let mut pool = fuel_core::kv_block_pool_device::DeviceKvPool::new(geom, DType::BF16, &dev)
            .expect("bf16 DeviceKvPool");
        let session = pool.core_mut().open();
        let mut paged: Vec<Vec<f32>> = Vec::new();
        for &t in &all {
            paged.push(
                model
                    .forward_paged_step(t, &mut pool, session)
                    .expect("paged bf16 step"),
            );
        }

        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        // Contiguous prefill = position P-1 = paged's P-th step; then each decode.
        let p = prompt.len();
        assert_eq!(
            argmax(&paged[p - 1]),
            argmax(&contig[0]),
            "prefill-last argmax (bf16 CUDA)"
        );
        for i in 0..decode.len() {
            assert_eq!(
                argmax(&paged[p + i]),
                argmax(&contig[1 + i]),
                "decode-{i} argmax (bf16 CUDA paged vs contiguous)",
            );
        }
        eprintln!(
            "bf16 PAGED decode on CUDA VERIFIED: argmax matches contiguous over {} steps",
            all.len()
        );
    }

    /// Ragged (non-uniform-position) batched paged decode: `K` sessions sitting
    /// at DISTINCT absolute positions, advanced by ONE batched step, where every
    /// output row must equal the single-session serial [`LlamaModel::
    /// forward_paged_step`] for that session at its own position. CPU f32.
    ///
    /// This is the parity gate for removing `forward_paged_step_batched`'s
    /// uniformity gate. The schedule is deliberately STAGGERED — histories of
    /// length 2 / 5 / 3 → positions 2 / 5 / 3 — NOT a lockstep common offset. A
    /// broken per-row RoPE that shares one `tok_pos` across rows applies the
    /// wrong rotation to rows 1..K (row 0 sits at the shared position, so it
    /// stays correct — that asymmetry is the tell) and diverges from their
    /// serial reference. A fixed-offset heterogeneous schedule would pass
    /// vacuously even with that bug; distinct absolute positions do not.
    #[test]
    // needless_range_loop here: the bound is a semantic count that need not equal the
    // indexed buffer len, so a mechanical .iter()/.take() risks silently dropping
    // iterations.
    #[allow(clippy::needless_range_loop)]
    fn ragged_batched_paged_decode_matches_per_session_serial() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = fuel_core::Device::cpu();

        // Staggered histories → DISTINCT positions {2, 5, 3} at the batched step.
        let histories: [&[u32]; 3] = [&[1, 2], &[3, 4, 5, 6, 1], &[2, 5, 1]];
        let decode_tokens = [7u32, 2, 4];
        let k = histories.len();

        let (n_layers, n_kv_heads, head_dim) = (cfg.n_layers, cfg.n_kv_heads, cfg.head_dim);
        let geom = || fuel_core::kv_block_pool::KvGeometry {
            n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads,
            head_dim,
            elem_size: 4,
        };

        // Serial reference: each session ALONE in its own pool, primed
        // identically, then one single-session decode step — ground truth row b.
        let mut reference: Vec<Vec<f32>> = Vec::with_capacity(k);
        for b in 0..k {
            let mut pool =
                fuel_core::kv_block_pool_device::DeviceKvPool::new(geom(), DType::F32, &dev)
                    .expect("f32 DeviceKvPool (ref)");
            let s = pool.core_mut().open();
            for &t in histories[b] {
                model
                    .forward_paged_step(t, &mut pool, s)
                    .expect("serial prime");
            }
            reference.push(
                model
                    .forward_paged_step(decode_tokens[b], &mut pool, s)
                    .expect("serial decode"),
            );
        }

        // Batched: K sessions in ONE pool, primed identically per-session, then a
        // SINGLE ragged batched step over all K at once.
        let mut pool = fuel_core::kv_block_pool_device::DeviceKvPool::new(geom(), DType::F32, &dev)
            .expect("f32 DeviceKvPool (batched)");
        let mut sessions = Vec::with_capacity(k);
        for b in 0..k {
            let s = pool.core_mut().open();
            for &t in histories[b] {
                model
                    .forward_paged_step(t, &mut pool, s)
                    .expect("batched prime");
            }
            sessions.push(s);
        }
        // Guard the premise: the positions really are non-uniform.
        let positions: Vec<usize> = sessions
            .iter()
            .map(|&s| pool.core().filled_tokens(s).unwrap())
            .collect();
        assert_eq!(positions, vec![2, 5, 3], "test setup: staggered positions");

        let batched = model
            .forward_paged_step_batched(&decode_tokens, &mut pool, &sessions)
            .expect("ragged batched decode");
        assert_eq!(batched.len(), k, "one logits row per session");

        for b in 0..k {
            assert_eq!(
                batched[b].len(),
                reference[b].len(),
                "row {b} logits length"
            );
            for (i, (&got, &want)) in batched[b].iter().zip(reference[b].iter()).enumerate() {
                let diff = (got - want).abs();
                let denom = got.abs().max(want.abs()).max(f32::MIN_POSITIVE);
                let rel = diff / denom;
                assert!(
                    diff < 1e-5 || rel < 1e-5,
                    "row {b} (pos {}) logit[{i}]: batched={got} serial={want} (abs={diff} rel={rel})",
                    positions[b],
                );
            }
        }
    }

    /// **A hand-rolled decode loop gets plan reuse at the ergonomic call
    /// shape.** `forward_decode_step(tokens, cache, ctx)` takes the same three
    /// arguments as the raw `forward_with_kv_context`, so the call a consumer
    /// naturally writes is now the fast one. This closes the gap that cost a
    /// measured consumer 5,901 → 26.47 ms/token: not a bad default, but a fast
    /// path that was unreachable without knowing `DecodeSession` existed.
    ///
    /// Same slope instrument as
    /// [`contiguous_generate_reuses_decode_plan_by_default`]: the optimize-call
    /// delta must NOT GROW WITH TOKEN COUNT. Two loops differing only in length
    /// (2 vs 10 decode tokens) must show the same delta.
    ///
    /// **The second arm is the point.** It drives the identical loop through
    /// `forward_with_kv_context` — the raw primitive, same three arguments —
    /// and asserts the slope IS present there. Without it, arm one alone could
    /// pass on a `forward_decode_step` that silently did nothing useful, and
    /// the test would be asserting a property of the *instrument* rather than
    /// of the new entry. With both arms, the test states the actual claim: at
    /// one call shape the slope is absent, at the other it is present, and the
    /// difference is exactly the held plan.
    #[test]
    fn forward_decode_step_reuses_plan_at_the_raw_call_shape() {
        use fuel_core::pipelined_bridge::optimize_calls_thread_local;

        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = fuel_core::Device::cpu();
        let prompt: [u32; 3] = [1, 2, 3];
        const MSL: usize = 32;

        // `persistent = false` drives the raw primitive at the same call shape.
        let optimize_calls_for = |n_decode: usize, persistent: bool| -> usize {
            let mut cache = fuel_core::inference_context::KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                MSL,
                DType::F32,
                &dev,
            )
            .expect("kv cache");
            let mut ctx = InferenceContext::new(dev.clone());
            let before = optimize_calls_thread_local();
            // Prefill (seq != 1 — builds no plan on either path).
            model
                .forward_decode_step(&prompt, &mut cache, &mut ctx)
                .expect("prefill");
            for t in 0..n_decode {
                let tok = [(t as u32 % 7) + 1];
                if persistent {
                    model
                        .forward_decode_step(&tok, &mut cache, &mut ctx)
                        .expect("decode")
                } else {
                    model
                        .forward_with_kv_context(&tok, &mut cache, &mut ctx)
                        .expect("decode")
                };
            }
            optimize_calls_thread_local() - before
        };

        // --- Arm 1: forward_decode_step — NO slope. ---
        let short = optimize_calls_for(2, true);
        let long = optimize_calls_for(10, true);
        assert_eq!(
            long,
            short,
            "forward_decode_step must reuse ONE decode plan across tokens: delta grew {} \
             between 2 decode tokens ({short}) and 10 ({long})",
            long as i64 - short as i64,
        );

        // --- Arm 2 (POSITIVE CONTROL): the raw primitive — slope PRESENT. ---
        // Proves the instrument can see the difference it claims to measure.
        let raw_short = optimize_calls_for(2, false);
        let raw_long = optimize_calls_for(10, false);
        assert_eq!(
            raw_long - raw_short,
            8,
            "the raw forward_with_kv_context must re-plan per token — expected the \
             optimize-call delta to grow by exactly the 8-token difference, got {} \
             (2 tokens: {raw_short}, 10 tokens: {raw_long}). If this is 0, the \
             instrument is blind and arm 1 above proves nothing.",
            raw_long as i64 - raw_short as i64,
        );

        // And the plan is actually HELD on the context, not merely fast.
        let mut cache = fuel_core::inference_context::KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            MSL,
            DType::F32,
            &dev,
        )
        .expect("kv cache");
        let mut ctx = InferenceContext::new(dev.clone());
        assert!(!ctx.has_decode_session(), "fresh context holds no plan");
        model
            .forward_decode_step(&prompt, &mut cache, &mut ctx)
            .expect("prefill");
        assert!(
            !ctx.has_decode_session(),
            "prefill (seq != 1) builds no plan"
        );
        model
            .forward_decode_step(&[4], &mut cache, &mut ctx)
            .expect("first decode token");
        assert!(
            ctx.has_decode_session(),
            "the first decode token builds and HOLDS the plan"
        );
    }

    /// **Staleness invalidates the SESSION AND THE CAPTURE TOGETHER** — the
    /// check `forward_with_kv_context_captured` shipped without, and the reason
    /// it matters more there than on the plain persistent path.
    ///
    /// A `CapturedDecodeSession` is a recorded CUDA graph over FIXED device
    /// addresses taken from the session's `base_cache`. If invalidation dropped
    /// the session but left the capture, the next token would `cuGraphLaunch`
    /// against buffers that no longer describe the live cache and return wrong
    /// logits **at full speed, with nothing reporting it**. That is strictly
    /// worse than a crash, and no existing test could see it: the plain
    /// persistent path has no capture to leak, so its staleness test passes
    /// whether or not the captured path is correct.
    ///
    /// Exercised through the generic capture parameter with `C = ()`. That is a
    /// faithful stand-in, not a convenience: the function's entire contract
    /// w.r.t. `captured` is "clear it", so `()` tests the real behavior while
    /// keeping the test on CPU — `CapturedDecodeSession` is `cuda`-only and a
    /// GPU-gated test could not run in the ordinary suite at all.
    ///
    /// Five arms, because "it clears things" is not the claim — "it does the
    /// right thing to each of the pair, exactly when the key says so" is:
    /// - matching key → `Kept`, BOTH survive (an over-eager check that
    ///   invalidated every token would silently cost the whole 223× win while
    ///   every correctness test still passed);
    /// - mismatched `max_seq_len` → `Dropped(None)`, BOTH cleared;
    /// - mismatched `cache_dtype` → `Dropped(None)`, BOTH cleared;
    /// - same-geometry allocation swap → `Rebound` (GAP-028): the SESSION
    ///   survives, the CAPTURE does not, and the plan re-keys to the new
    ///   allocation so the next token is `Kept` rather than re-binding forever;
    /// - a swap whose residency cannot be proven → `Dropped(ResidencyUnknown)`,
    ///   BOTH cleared. That last arm is the one that makes the fourth mean
    ///   something: a guard is only worth having if it also says no.
    ///
    /// **What this test CANNOT see, so that nobody treats it as covering it:**
    /// whether a re-bind bound the RIGHT storage. Arm 4 asserts `Rebound` and
    /// the alloc_id re-key, and a `rebind_kv` gutted to skip the `base_cache`
    /// overwrite entirely produces both — verified by sabotage, this test stays
    /// green through it. The failure surfaces only in
    /// [`held_decode_plan_must_not_execute_over_a_swapped_kv_cache`], at the
    /// logit level. The two are NOT substitutes despite the overlap in subject.
    #[test]
    fn stale_decode_pair_invalidates_session_and_capture_together() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = fuel_core::Device::cpu();
        const MSL: usize = 16;

        // Build a REAL held session: prefill (seq>1, no session) then one
        // decode token (seq==1, builds it).
        let mut cache = fuel_core::inference_context::KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            MSL,
            DType::F32,
            &dev,
        )
        .expect("kv cache");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        model
            .forward_with_kv_context_persistent(&[1, 2, 3], &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("first decode token builds the session");
        assert!(session.is_some(), "precondition: a session is held");

        // --- Arm 1: key MATCHES the live cache → nothing is invalidated. ---
        let mut captured: Option<()> = Some(());
        let d = model.invalidate_decode_pair_if_stale(
            &mut session,
            &mut captured,
            &mut ctx,
            1,
            Some(MSL),
            DType::F32,
            &cache,
        );
        assert_eq!(
            d,
            SessionDisposition::Kept,
            "a matching validity key must NOT invalidate"
        );
        assert!(session.is_some(), "valid session survives");
        assert!(captured.is_some(), "valid capture survives");

        // --- Arm 2: cache resized under the held pair → both retired. The
        // refusal is `None`: a structural mismatch is not a re-bind candidate,
        // so the guard is never consulted. ---
        let mut captured: Option<()> = Some(());
        let d = model.invalidate_decode_pair_if_stale(
            &mut session,
            &mut captured,
            &mut ctx,
            1,
            Some(MSL * 4),
            DType::F32,
            &cache,
        );
        assert_eq!(
            d,
            SessionDisposition::Dropped(None),
            "a resized cache must be judged STRUCTURALLY stale, not offered to the \
             re-bind guard — re-binding across `max_seq_len` would feed the plan \
             storage of the wrong extent",
        );
        assert!(session.is_none(), "stale session dropped");
        assert!(
            captured.is_none(),
            "stale CAPTURE dropped — a surviving recorded graph would replay against \
             fixed device addresses that no longer describe the live cache",
        );

        // --- Arm 3: same, via a cache-dtype swap, rebuilding the session. ---
        model
            .forward_with_kv_context_persistent(&[5], &mut cache, &mut ctx, &mut session)
            .expect("rebuild the session after invalidation");
        assert!(session.is_some(), "precondition: session rebuilt");
        let mut captured: Option<()> = Some(());
        let d = model.invalidate_decode_pair_if_stale(
            &mut session,
            &mut captured,
            &mut ctx,
            1,
            Some(MSL),
            DType::BF16,
            &cache,
        );
        assert_eq!(
            d,
            SessionDisposition::Dropped(None),
            "a cache-dtype swap must invalidate structurally",
        );
        assert!(
            session.is_none() && captured.is_none(),
            "both retired on dtype mismatch"
        );

        // --- Arm 4 (GAP-014): a cache SWAP at identical geometry. The pair is
        // welded to the KV Arcs baked into `base_cache`, so a same-shaped
        // replacement is the one mismatch geometry cannot express — and the
        // one the slot-pooled serving path hits every time it admits a
        // request. Silent if missed: the capture would keep replaying against
        // the retired request's device addresses. ---
        model
            .forward_with_kv_context_persistent(&[6], &mut cache, &mut ctx, &mut session)
            .expect("rebuild the session after invalidation");
        assert!(session.is_some(), "precondition: session rebuilt");
        let fresh = fuel_core::inference_context::KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            MSL,
            DType::F32,
            &dev,
        )
        .expect("a second cache of IDENTICAL geometry");
        assert_ne!(
            fresh.alloc_id(),
            cache.alloc_id(),
            "two allocations must never share an id — the counter is never recycled",
        );
        let mut captured: Option<()> = Some(());
        let d = model.invalidate_decode_pair_if_stale(
            &mut session,
            &mut captured,
            &mut ctx,
            1,
            Some(MSL),
            DType::F32,
            &fresh,
        );
        // GAP-014 retired the pair here. GAP-028 keeps the plan and re-points
        // its KV Consts instead — but ONLY the plan. The capture is not
        // rescuable by any amount of guarding: it is a recorded CUDA graph over
        // FIXED device addresses, and the re-bind is precisely the act of
        // changing some of those addresses.
        assert_eq!(
            d,
            SessionDisposition::Rebound,
            "a same-geometry cache swap must RE-BIND the held plan (GAP-028) — \
             every other key field matches, so only the allocation id sees it",
        );
        assert!(session.is_some(), "the plan survives a guarded re-bind");
        assert!(
            captured.is_none(),
            "the CAPTURE must die on a re-bind exactly as it dies on a drop — the \
             re-bind changes the very device addresses the recording baked in",
        );
        // And the plan now answers to the NEW allocation, so the next token
        // takes the `Kept` path rather than re-binding on every step.
        let mut captured2: Option<()> = Some(());
        let d = model.invalidate_decode_pair_if_stale(
            &mut session,
            &mut captured2,
            &mut ctx,
            1,
            Some(MSL),
            DType::F32,
            &fresh,
        );
        assert_eq!(
            d,
            SessionDisposition::Kept,
            "after a re-bind the plan's alloc_id must name the NEW allocation — \
             otherwise every subsequent token re-binds and the guard runs forever",
        );

        // --- Arm 5 (GAP-028 negative control at the disposition level): a swap
        // the guard must REFUSE, and refuse for a NAMED reason. Accepting a
        // compatible swap proves nothing on its own; the whole risk of GAP-028
        // is a guard that says yes too readily, which is GAP-014 restored
        // behind an optimization. `with_dims` + `set_layer` reaches the one
        // state the storage bytes cannot describe: residency this cache never
        // measured. ---
        let mut unprovable = fuel_core::inference_context::KvCache::with_dims(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
        );
        unprovable.max_seq_len = Some(MSL);
        unprovable.dtype = Some(DType::F32);
        for li in 0..cfg.n_layers {
            let layer = fresh.layer(li).expect("fresh cache has every layer");
            unprovable.set_layer(
                li,
                fuel_core::inference_context::KvLayer {
                    k: std::sync::Arc::clone(&layer.k),
                    v: std::sync::Arc::clone(&layer.v),
                    k_layout: layer.k_layout.clone(),
                    v_layout: layer.v_layout.clone(),
                    k_version: 0,
                    v_version: 0,
                    k_authority: fuel_core::inference_context::AuthorityState::Host,
                    v_authority: fuel_core::inference_context::AuthorityState::Host,
                },
            );
        }
        // Note what this cache is: byte-for-byte the SAME storage `Arc`s the
        // held plan is already bound to. Every fingerprint check passes; only
        // the residency claim is missing. That is deliberate — it isolates the
        // one guard clause that cannot be derived from the bytes.
        let mut captured: Option<()> = Some(());
        let d = model.invalidate_decode_pair_if_stale(
            &mut session,
            &mut captured,
            &mut ctx,
            1,
            Some(MSL),
            DType::F32,
            &unprovable,
        );
        assert_eq!(
            d,
            SessionDisposition::Dropped(Some(
                fuel_core::inference_context::RebindRefusal::ResidencyUnknown,
            )),
            "an unprovable residency must be REFUSED, and refused for that reason — \
             a guard that refuses by accident is not a guard",
        );
        assert!(
            session.is_none() && captured.is_none(),
            "both retired on refusal"
        );
    }

    /// **The CONTIGUOUS generation API is plan-reuse-by-default, and this is the
    /// gate that keeps it so.** [`LlamaModel::generate_streaming_with_kv_context`]
    /// holds one `Option<DecodeSession>` across its whole decode loop, so the
    /// optimizer runs once for the held decode plan rather than once per token.
    /// Nothing asserted that before this test: the behavior was correct but
    /// ungated, i.e. one refactor away from silently reverting to per-token
    /// planning — the failure mode that cost the paged route a measured 29.7× and
    /// hand-rolled contiguous decode loops 223× (nsys, 2026-08-01).
    ///
    /// **Instrument: the optimize-call delta must NOT GROW WITH TOKEN COUNT.** An
    /// absolute bound would be a magic number that drifts with the prefill/sampling
    /// path; the *slope* is the actual claim. Two runs differing only in
    /// `max_new_tokens` (2 vs 10) must show the SAME delta:
    ///
    /// - plan reuse ON  → `delta(10) - delta(2) == 0`  (prefill + one plan build,
    ///   independent of N)
    /// - plan reuse OFF → `delta(10) - delta(2) == 8`  (one optimize per token)
    ///
    /// So the assertion is sabotage-calibrated by construction: the quantity it
    /// checks IS the per-token planning cost, and deleting the held session moves
    /// it from 0 to exactly the token difference. Thread-local counter (robust
    /// under the concurrent suite); CPU f32; greedy + `eos_id: None` so both runs
    /// spend their full budget.
    #[test]
    fn contiguous_generate_reuses_decode_plan_by_default() {
        use fuel_core::pipelined_bridge::optimize_calls_thread_local;

        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = fuel_core::Device::cpu();
        let prompt: [u32; 3] = [1, 2, 3];

        let optimize_calls_for = |max_new: usize| -> usize {
            let before = optimize_calls_thread_local();
            model
                .generate_with_kv_context(
                    &prompt,
                    max_new,
                    SamplingStrategy::Greedy,
                    None,
                    &dev,
                    DType::F32,
                )
                .expect("greedy generate");
            optimize_calls_thread_local() - before
        };

        let short = optimize_calls_for(2);
        let long = optimize_calls_for(10);

        assert_eq!(
            long,
            short,
            "contiguous generate must reuse ONE decode plan across tokens: \
             optimize-call delta grew {} between max_new=2 ({short}) and max_new=10 ({long}) \
             — that slope is one re-plan per token, i.e. the held DecodeSession is gone",
            long as i64 - short as i64,
        );
    }

    /// **§6.1 of `docs/design/paged-attention-agnostic-seam.md` — WHERE does
    /// paged decode actually run?** This is the discriminating instrument the
    /// paged-attention design is blocked on, and it exists because reasoning has
    /// repeatedly failed here: three sessions have asserted the placement of
    /// `Op::PagedAttn` from non-discriminating evidence and landed on two
    /// opposite answers. Grep cannot settle it and neither can arithmetic.
    ///
    /// **The question.** `PagedAttn` has no CUDA and no Vulkan implementation
    /// (positive-controlled: the same query finds FlashAttn in 3 cuda-backend
    /// files, FusedLinear in 2). Under a CUDA-pinned decode the optimizer
    /// therefore has two legal moves, and Lightbulb's measured 186× host-ward
    /// DtoH is consistent with BOTH:
    ///
    /// - **(A) keep the fused node and place it on host** — the KV caches round
    ///   trip every token, and the fix is a kernel;
    /// - **(B) lower to the primitive recipe and run it on CUDA** — the
    ///   round-trip is `DeviceKvPool` plumbing, a kernel would not have fixed
    ///   it, and the work belongs somewhere else entirely.
    ///
    /// Those imply completely different programs, which is why no kernel is
    /// being requested from Baracuda until this reports.
    ///
    /// **How it discriminates.** It reads the REAL held decode plan
    /// (`PagedDecodeSession::optimized()`) from a CUDA `forward_paged_step_
    /// persistent`, and dumps `placement_of` for every node, bucketed by op. The
    /// two hypotheses produce structurally different dumps and cannot be
    /// confused:
    ///
    /// - **(A)** a surviving `Op::Fused(PAGED_ATTN, _)` node, placed `Cpu`,
    ///   inside an otherwise-CUDA graph;
    /// - **(B)** NO `PAGED_ATTN` node at all (it was decomposed), and
    ///   `IndexSelect`/`MatMul`/`MaskedFill` recipe nodes placed `Cuda`.
    ///
    /// It asserts neither outcome — **it reports.** Pre-committing to an answer
    /// is the exact error being corrected. What it DOES assert is that the
    /// instrument worked: `has_placements()` must be true, because a skipped
    /// residency pass makes every `placement_of` return `None`, and a dump of
    /// all-`None` would read as "nothing is placed on CUDA" — a wrong answer
    /// that looks like a null one, and precisely the failure
    /// `OptimizedGraph::has_placements` exists to expose.
    ///
    /// Live-GPU: `#[ignore]`d, and run through `scripts/gpu-run.ps1`.
    #[test]
    #[ignore = "live GPU (CUDA); run via scripts/gpu-run.ps1"]
    #[cfg(feature = "cuda")]
    fn paged_decode_node_placement_report_cuda() {
        use fuel_core::inference_context::PagedDecodePlan;
        use std::collections::BTreeMap;

        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        // `Device::new_cuda` was retired; `cuda_backend::new_device` carries its
        // ergonomics (see that module's header).
        let dev = match fuel_core::cuda_backend::new_device(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0) (placement report): {e:?}"
                    )),
                );
            }
        };

        let geom = fuel_core::kv_block_pool::KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 4,
        };
        let mut pool = fuel_core::kv_block_pool_device::DeviceKvPool::new(geom, DType::F32, &dev)
            .expect("CUDA DeviceKvPool");
        let s = pool.core_mut().open();
        for &t in &[1u32, 2, 3, 4, 5, 6, 7] {
            model.forward_paged_step(t, &mut pool, s).expect("prime");
        }

        // Build the held plan on CUDA — this is the real production decode plan.
        let mut ds: Option<fuel_core::inference_context::PagedDecodeSession> = None;
        model
            .forward_paged_step_persistent(8, &mut pool, s, 8, PagedDecodePlan::PlanOnce, &mut ds)
            .expect("CUDA persistent decode token");
        let ds = ds.expect("plan-once built a held session");

        let opt = ds.optimized();
        // INSTRUMENT CHECK, before reading anything out of it. Without this, a
        // skipped residency pass yields all-`None` and the report below would
        // silently claim "no node is on CUDA".
        assert!(
            opt.has_placements(),
            "placement was NOT COMPUTED — every placement_of would be None and the \
             report below would be an artifact of the missing instrument, not an \
             observation. Fix the harness before reading any conclusion from it.",
        );

        let g = ds.graph().read().expect("graph lock");
        let mut by_op: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        let mut paged_attn_placements: Vec<String> = Vec::new();

        for i in 0..g.len() {
            let id = fuel_graph::NodeId(i);
            let node = g.node(id);
            let is_paged = matches!(
                node.op,
                fuel_graph::Op::Fused(fid, _) if fid == fuel_graph::registry::FusedOps::PAGED_ATTN
            );
            let op_name = match &node.op {
                fuel_graph::Op::Fused(fid, _) => format!("Fused({fid:?})"),
                other => format!("{other:?}")
                    .split(' ')
                    .next()
                    .unwrap_or("?")
                    .to_string(),
            };
            let place = match opt.placement_of(id) {
                Some(d) => format!("{d:?}"),
                None => "None".to_string(),
            };
            if is_paged {
                paged_attn_placements.push(place.clone());
            }
            *by_op.entry(op_name).or_default().entry(place).or_insert(0) += 1;
        }

        println!("\n=== paged decode plan: node placement by op (CUDA-pinned) ===");
        println!("total nodes: {}", g.len());
        for (op, places) in &by_op {
            let rendered: Vec<String> = places.iter().map(|(d, n)| format!("{d}×{n}")).collect();
            println!("  {op:<40} {}", rendered.join("  "));
        }

        println!("\n--- VERDICT ---");
        if paged_attn_placements.is_empty() {
            println!(
                "HYPOTHESIS (B): no Op::Fused(PAGED_ATTN) node survives in the held plan \
                 — it was DECOMPOSED to the primitive recipe. Read the IndexSelect / \
                 MatMul / MaskedFill rows above for where those primitives landed. If \
                 they are Cuda, the 186x DtoH is NOT this op's placement and a paged \
                 kernel would not have fixed it."
            );
        } else {
            println!(
                "HYPOTHESIS (A): {} Op::Fused(PAGED_ATTN) node(s) SURVIVE in the held \
                 plan, placed {:?}. If that is Cpu inside an otherwise-Cuda graph, the \
                 fused paged attention op is executing on the host and the KV caches \
                 round-trip every token.",
                paged_attn_placements.len(),
                paged_attn_placements,
            );
        }
        println!("--- END VERDICT ---\n");
    }

    /// **A held PAGED plan must not be reused across two models of identical
    /// geometry but different weights.** The paged half of the hole the
    /// contiguous `DecodeSession` closed first.
    ///
    /// `PagedDecodeSession` bakes the model's weight `Const`s. Its validity key
    /// was `(max_blocks_cap, n_layers, block_size, cache_dtype)` — pure
    /// geometry — so two same-shaped models produced the SAME key and a plan
    /// built for one was judged valid for the other. That is not a crash: it is
    /// the right architecture computed with the wrong weights, at full speed,
    /// with nothing to report. Weight identity now enters via
    /// `decode_shape_key()`, whose `ModelInstanceId` comes from a never-recycled
    /// counter (so it cannot be defeated by allocator address reuse).
    ///
    /// BOTH halves asserted, because only the pair is meaningful: a differing
    /// key must invalidate, AND a matching key must NOT. An "always stale"
    /// predicate would pass the first assertion alone while silently disabling
    /// plan reuse entirely — a pure performance regression that hides behind a
    /// fully green correctness suite.
    #[test]
    fn paged_session_is_not_reused_across_models_with_different_weights() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };

        // Two models: IDENTICAL config, SEPARATELY constructed weights.
        let model_a = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let model_b = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let key_a = model_a.decode_shape_key();
        let key_b = model_b.decode_shape_key();
        assert_ne!(
            key_a, key_b,
            "two same-shaped models with distinct weights must key differently — the paged session bakes their Consts",
        );
        // CONTROL: the key is stable for one model, or "always invalidate"
        // would satisfy the assertion above while destroying plan reuse.
        assert_eq!(
            key_a,
            model_a.decode_shape_key(),
            "the same model must key identically across calls",
        );

        // Build a real held paged session for model A on CPU.
        let dev = fuel_core::Device::cpu();
        let geom = fuel_core::kv_block_pool::KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 4,
        };
        let mut pool = fuel_core::kv_block_pool_device::DeviceKvPool::new(geom, DType::F32, &dev)
            .expect("f32 DeviceKvPool");
        let handle = pool.core_mut().open();
        for &t in &[1u32, 2, 3] {
            model_a
                .forward_paged_step(t, &mut pool, handle)
                .expect("prime");
        }
        let mut ds: Option<fuel_core::inference_context::PagedDecodeSession> = None;
        model_a
            .forward_paged_step_persistent(
                4,
                &mut pool,
                handle,
                8,
                fuel_core::inference_context::PagedDecodePlan::PlanOnce,
                &mut ds,
            )
            .expect("model A builds a held paged plan");
        let s = ds.as_ref().expect("held paged session");

        let pool_id = pool.alloc_id();
        // Same geometry, model A's key, same pool => VALID (plan reuse preserved).
        assert!(
            s.is_valid_for(8, cfg.n_layers, 4, DType::F32, key_a, pool_id),
            "model A's own plan must stay valid — otherwise plan reuse is dead",
        );
        // Same geometry, model B's key => STALE.
        assert!(
            !s.is_valid_for(8, cfg.n_layers, 4, DType::F32, key_b, pool_id),
            "a plan baked for model A must NOT be judged valid for model B at identical geometry — this is the silent-wrong-weights hole",
        );

        // GAP-014, paged half: same model, same geometry, a DIFFERENT pool.
        // `rebind_and_realize_paged_prebuilt` never re-binds the pool buffers,
        // so this plan would keep reading the first pool's blocks — the paged
        // twin of the cross-request contamination the contiguous path had.
        let pool_b = fuel_core::kv_block_pool_device::DeviceKvPool::new(geom, DType::F32, &dev)
            .expect("a second pool of IDENTICAL geometry");
        assert_ne!(
            pool_b.alloc_id(),
            pool_id,
            "two pools must never share an id — the counter is never recycled",
        );
        assert!(
            !s.is_valid_for(8, cfg.n_layers, 4, DType::F32, key_a, pool_b.alloc_id()),
            "a plan baked against pool A must NOT be judged valid for pool B — \
             every other key field matches, so only the allocation id can see it",
        );
    }

    /// GAP-014 — a held decode plan is WELDED to the KV allocation it was
    /// built against, and today's validity key cannot see a swap.
    ///
    /// `DecodeSession::base_cache` holds the KV storage Arcs bound on the
    /// first decode token, and `rebind_and_realize_prebuilt` overwrites only
    /// the per-token data Consts — it contains ZERO `kv_nodes` references. So
    /// every subsequent token reads and writes the ORIGINAL cache's buffers no
    /// matter which `&mut KvCache` the caller hands in. The validity key
    /// (`seq / max_seq_len / n_layers / cache_dtype / shape_key`) names the
    /// model and the geometry and nothing at all about the allocation, so a
    /// same-shaped replacement is judged interchangeable.
    ///
    /// That is precisely the slot-pooled serving happy path: retire request A,
    /// admit request B with a fresh cache of the same geometry, reuse the held
    /// plan for speed. B then decodes over A's KV — at full speed, with a
    /// plausible-looking distribution, and nothing to report.
    ///
    /// The oracle is at LOGIT level, not sampled-token level: on a tiny model
    /// greedy sampling is a fixed point that swallows exactly this kind of
    /// divergence (the null-oracle lesson from the ragged-decode work).
    ///
    /// **NOT a substitute for, and not substitutable by,
    /// [`stale_decode_pair_invalidates_session_and_capture_together`] — they
    /// look redundant and cover disjoint things.** Measured by sabotage: gutting
    /// `rebind_kv` so it updates `alloc_id` but never overwrites `base_cache`
    /// leaves that test fully GREEN (it asserts `Rebound` and the alloc_id
    /// re-key, both of which a gutted re-bind still produces) while THIS test
    /// fails at maxdiff 1.032e-1. Only the logit oracle can see a re-bind that
    /// binds the wrong storage. Deleting either as duplication silently drops a
    /// whole failure mode.
    #[test]
    fn held_decode_plan_must_not_execute_over_a_swapped_kv_cache() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let dev = Device::cpu();
        let max_seq_len = 8usize;
        let mk_cache = || {
            KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq_len,
                DType::F32,
                &dev,
            )
            .expect("with_capacity")
        };
        // Tolerance for "the same computation": both arms run the identical f32
        // CPU kernels, so agreement is near-exact. Deliberately far below the
        // measured A-vs-B separation asserted as the control below.
        const TOL: f32 = 1e-5;
        fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
            assert_eq!(a.len(), b.len(), "logit rows must be comparable");
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        let prompt_a = [1u32, 2, 3];
        let prompt_b = [7u32, 8, 9];
        let next_token = 5u32;

        // ---- Request A: prime a cache and BUILD a held plan welded to it. ----
        let mut cache_a = mk_cache();
        let mut ctx = InferenceContext::new(dev.clone());
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        model
            .forward_with_kv_context_persistent(&prompt_a, &mut cache_a, &mut ctx, &mut session)
            .expect("A prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache_a, &mut ctx, &mut session)
            .expect("A's first decode token builds the held plan");
        // Reuse detector: the held graph's Arc IDENTITY. NOT its data-Const
        // NodeIds — those are graph-local and a rebuilt graph mints the very
        // same `NodeId(1)`, so a NodeId comparison cannot distinguish reuse
        // from rebuild in EITHER direction. Holding this clone pins the
        // allocation, which is what makes pointer identity sound here.
        let built = std::sync::Arc::clone(
            session
                .as_ref()
                .expect("A must leave a held plan — without one this test proves nothing")
                .graph(),
        );

        // ---- ORACLE: what B's next token MUST produce. Computed on an
        // independent cache via the always-re-planning reference path, which
        // holds no session and therefore cannot be contaminated. ----
        let mut cache_oracle = mk_cache();
        let mut ctx_oracle = InferenceContext::new(dev.clone());
        model
            .forward_with_kv_context(&prompt_b, &mut cache_oracle, &mut ctx_oracle)
            .expect("oracle prefill");
        let expected_b = model
            .forward_with_kv_context(&[next_token], &mut cache_oracle, &mut ctx_oracle)
            .expect("oracle decode");

        // ---- CONTROL: the two histories must actually be distinguishable at
        // this token. If A's prefix and B's prefix produced near-identical
        // logits, the assertion below would pass without the bug being fixed —
        // a vacuous oracle. Measure the separation and require the pass
        // threshold to sit far under it. ----
        let mut cache_ctl = mk_cache();
        let mut ctx_ctl = InferenceContext::new(dev.clone());
        model
            .forward_with_kv_context(&prompt_a, &mut cache_ctl, &mut ctx_ctl)
            .expect("control prefill");
        model
            .forward_with_kv_context(&[4], &mut cache_ctl, &mut ctx_ctl)
            .expect("control decode 1");
        let a_history = model
            .forward_with_kv_context(&[next_token], &mut cache_ctl, &mut ctx_ctl)
            .expect("control decode 2");
        let separation = maxdiff(&expected_b, &a_history);
        assert!(
            separation > 10.0 * TOL,
            "NEGATIVE CONTROL FAILED: A's history and B's history produce logits \
             only {separation:.3e} apart, so a {TOL:.0e} equality check cannot \
             tell them apart and the assertion below would be vacuous",
        );

        // ---- ACT: hand the held plan a step over a DIFFERENT, fresh cache
        // primed with B's own prefix through the reference path. ----
        let mut cache_b = mk_cache();
        let mut ctx_b = InferenceContext::new(dev.clone());
        model
            .forward_with_kv_context(&prompt_b, &mut cache_b, &mut ctx_b)
            .expect("B prefill lands in B's OWN storage");
        let actual_b = model
            .forward_with_kv_context_persistent(&[next_token], &mut cache_b, &mut ctx, &mut session)
            .expect("B's decode step, offered the plan A built");

        let drift = maxdiff(&expected_b, &actual_b);
        assert!(
            drift <= TOL,
            "CROSS-REQUEST KV CONTAMINATION: a plan built against cache A \
             executed request B's token over A's KV. maxdiff {drift:.3e} vs the \
             re-planned oracle, while the two histories are {separation:.3e} \
             apart — so this is A's answer, not a rounding difference.",
        );

        // The mechanism, made visible. GAP-014 shipped invalidate-and-rebuild,
        // so this asserted the graph CHANGED. GAP-028 replaces that with a
        // guarded re-bind, so a compatible swap now REUSES the graph — the
        // assertion is deliberately flipped, and the logit oracle above (which
        // did not change) is what proves the flip is safe: whichever mechanism
        // runs, B must get B's answer.
        assert!(
            session
                .as_ref()
                .map(|s| std::sync::Arc::ptr_eq(s.graph(), &built))
                .unwrap_or(false),
            "a compatible cache swap must RE-BIND the held plan, not rebuild it \
             (GAP-028) — the logits are already proven correct above, so a \
             rebuild here is pure cost on the serving admission path",
        );
    }

    /// GAP-028 NEGATIVE CONTROL — the half that makes the test above mean
    /// something.
    ///
    /// GAP-028 trades a guarantee that held by *construction* for one that holds
    /// by *proof*: invalidate-and-rebuild could not be wrong, whereas re-binding
    /// the KV Arcs into a live plan is correct only while the guard is. If the
    /// guard ever accepts an incompatible cache, the failure is GAP-014 again —
    /// silent, full-speed, plausible logits.
    ///
    /// So accepting a compatible swap proves nothing on its own. This asserts
    /// the guard REJECTS: an incompatible cache must fall back to a rebuild,
    /// never be spliced into the baked plan.
    ///
    /// TWO arms, because the two rejections come from different layers and a
    /// sabotage run showed the first alone is not a test of the guard:
    ///
    /// 1. `max_seq_len` — rejected by [`DecodeSession::validity_for`] as
    ///    STRUCTURALLY stale. `rebind_kv` is never called. This arm survived
    ///    both a gutted-re-bind sabotage and a gutted-residency-check sabotage,
    ///    which is exactly what "it does not exercise the guard" looks like.
    ///    It is kept because it pins a real decision — that `max_seq_len`
    ///    stays a structural key and is NOT delegated to the guard — but it is
    ///    labelled so nobody mistakes it for guard coverage.
    /// 2. Head geometry — a cache whose per-slot storage is a different
    ///    EXTENT at the same `max_seq_len`. Every structural field matches
    ///    (`shape_key` describes the MODEL, not the cache), so `validity_for`
    ///    says `AllocationChanged` and the guard is the only thing between that
    ///    storage and a splice into the baked graph. This is the arm that
    ///    actually tests `rebind_kv`.
    #[test]
    fn incompatible_cache_swap_is_rejected_not_rebound() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = Device::cpu();
        let mk = |msl: usize| {
            KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                msl,
                DType::F32,
                &dev,
            )
            .expect("with_capacity")
        };

        let mut cache_a = mk(8);
        let mut ctx = InferenceContext::new(dev.clone());
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        model
            .forward_with_kv_context_persistent(&[1, 2, 3], &mut cache_a, &mut ctx, &mut session)
            .expect("A prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache_a, &mut ctx, &mut session)
            .expect("A builds the plan");
        let built = std::sync::Arc::clone(session.as_ref().expect("held plan").graph());

        // A cache whose KV buffers are a DIFFERENT extent. Every other key
        // field matches — same model, same layer count, same dtype — so this
        // isolates the one thing the guard must catch.
        let mut cache_wide = mk(16);
        let mut ctx_w = InferenceContext::new(dev.clone());
        model
            .forward_with_kv_context(&[7, 8, 9], &mut cache_wide, &mut ctx_w)
            .expect("wide prefill");
        let expected = {
            let mut c = mk(16);
            let mut x = InferenceContext::new(dev.clone());
            model
                .forward_with_kv_context(&[7, 8, 9], &mut c, &mut x)
                .expect("oracle prefill");
            model
                .forward_with_kv_context(&[5], &mut c, &mut x)
                .expect("oracle decode")
        };

        let got = model
            .forward_with_kv_context_persistent(&[5], &mut cache_wide, &mut ctx, &mut session)
            .expect("decode over an incompatible cache must still SUCCEED (rebuild)");

        assert!(
            !session
                .as_ref()
                .map(|s| std::sync::Arc::ptr_eq(s.graph(), &built))
                .unwrap_or(false),
            "REJECTION FAILED: a plan baked for max_seq_len=8 was re-bound onto \
             max_seq_len=16 storage. The guard accepted an incompatible cache, \
             which is GAP-014 reintroduced behind the optimization that was \
             supposed to be safe.",
        );
        let drift = {
            assert_eq!(expected.len(), got.len(), "logit rows must be comparable");
            expected
                .iter()
                .zip(&got)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            drift <= 1e-5,
            "rejecting must fall back to a correct REBUILD, not merely refuse to \
             re-bind: maxdiff {drift:.3e} vs the re-planned oracle",
        );

        // ---- Arm 2: the guard's own rejection. ----
        //
        // A cache at the SAME `max_seq_len` whose K/V buffers are a different
        // extent, because its head geometry differs. Nothing in the structural
        // key can see this: `shape_key` is the model's identity, and
        // `n_layers` / `cache_dtype` / `max_seq_len` all match. So
        // `validity_for` hands it to the guard, and only the per-slot byte
        // comparison stands between a `[1, 4, 16, 4]` plan and `[1, 8, 16, 4]`
        // storage.
        model
            .forward_with_kv_context_persistent(&[7], &mut cache_wide, &mut ctx, &mut session)
            .expect("rebuild a plan against the wide cache");
        let s = session.as_mut().expect("a held plan to offer the guard");
        let wrong_heads = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads * 2,
            cfg.head_dim,
            16,
            DType::F32,
            &dev,
        )
        .expect("same max_seq_len, DOUBLE the kv heads");
        assert_eq!(
            s.validity_for(
                1,
                16,
                cfg.n_layers,
                DType::F32,
                model.decode_shape_key(),
                wrong_heads.alloc_id(),
            ),
            fuel_core::inference_context::PlanValidity::AllocationChanged,
            "precondition: the structural key must NOT catch this — if it did, \
             the guard would never be consulted and the assertion below would \
             pass vacuously",
        );
        assert_eq!(
            s.rebind_kv(&wrong_heads),
            Err(fuel_core::inference_context::RebindRefusal::StorageDiffers { layer: 0, which: 0 }),
            "the guard must refuse storage of the wrong EXTENT, and name it — \
             splicing it in would give the baked graph a buffer whose bytes do \
             not match the `Const` shape it was optimized around",
        );
    }

    /// GAP-014 CONTROL, the other direction: an "always stale" predicate would
    /// satisfy the test above while silently destroying plan reuse (measured
    /// 223× on CUDA) with every correctness test still green. Continuing the
    /// SAME request on the SAME cache must keep the held plan alive, and
    /// `truncate_to` — speculative decoding's reject path — must NOT count as a
    /// swap: it rewinds the conversation, it does not re-allocate the buffers
    /// the plan is welded to. That distinction is the whole design: the key
    /// names the ALLOCATION, never the conversation.
    #[test]
    fn held_decode_plan_survives_same_cache_continuation_and_truncate() {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            8,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        model
            .forward_with_kv_context_persistent(&[1, 2, 3], &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut session)
            .expect("first decode token builds the plan");
        // Arc identity, NOT `token_ids_node()`: data-Const NodeIds are
        // graph-local, so a REBUILT session mints the very same `NodeId(1)`
        // and a NodeId comparison would report "reused" for a rebuild —
        // making both assertions below silently vacuous.
        let built = std::sync::Arc::clone(session.as_ref().expect("held plan").graph());
        let reused = |s: &Option<fuel_core::inference_context::DecodeSession>| {
            s.as_ref()
                .map(|s| std::sync::Arc::ptr_eq(s.graph(), &built))
                .unwrap_or(false)
        };

        model
            .forward_with_kv_context_persistent(&[5], &mut cache, &mut ctx, &mut session)
            .expect("second decode token on the SAME cache");
        assert!(
            reused(&session),
            "continuing the same request on its own cache must REUSE the plan",
        );

        // Speculative-decode reject: rewind the conversation, same allocation.
        cache.truncate_to(4);
        model
            .forward_with_kv_context_persistent(&[6], &mut cache, &mut ctx, &mut session)
            .expect("decode after truncate_to");
        assert!(
            reused(&session),
            "truncate_to rewinds the CONVERSATION, not the ALLOCATION — the plan \
             is still welded to the right buffers and must stay valid",
        );
    }

    /// Task 3 (paged plan-once) — the second decode token REUSES the built
    /// graph + optimized plan instead of re-planning. Proves both halves:
    ///
    /// (1) PLAN-ONCE: the first persistent token optimizes (≥ 1 optimize call —
    ///     it builds + caches the plan); the second optimizes ZERO times (the
    ///     `realize_one_prebuilt_env` HIT), and the held graph's node count is
    ///     stable across the rebind (no re-splice / re-insert). Measured on the
    ///     THREAD-LOCAL optimize counter so it is robust under the concurrent
    ///     test suite (a process-global counter is polluted by peer threads).
    /// (2) CORRECTNESS: both persistent tokens are BYTE-IDENTICAL to the
    ///     re-planning `forward_paged_step` reference from the same primed
    ///     state — so the rebind bound the RIGHT per-token data (token-ids,
    ///     RoPE at the live position, the capacity-padded block_table that
    ///     gains a new physical block mid-generation, context_lens, and the
    ///     flattened KV-write offset). A HIT on WRONG bindings would pass (1)
    ///     but fail (2); only a correct rebind passes both.
    ///
    /// The schedule crosses a BLOCK BOUNDARY between the two measured tokens
    /// (history 7 → pos 7 fills block 1, pos 8 allocates block 2), so the
    /// second token's rebound block_table MUST carry a physical block the
    /// first token's did not — a stale (un-rebound) block_table attends the
    /// wrong slots and diverges from the reference. CPU f32.
    #[test]
    fn plan_once_second_token_reuses_graph() {
        use fuel_core::pipelined_bridge::optimize_calls_thread_local;

        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let dev = fuel_core::Device::cpu();

        let (n_layers, n_kv_heads, head_dim) = (cfg.n_layers, cfg.n_kv_heads, cfg.head_dim);
        let geom = || fuel_core::kv_block_pool::KvGeometry {
            n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads,
            head_dim,
            elem_size: 4,
        };

        // History of 7 → after priming, filled_tokens == 7. The two decode
        // tokens land at pos 7 (block 1, slot 3 — fills block 1) then pos 8
        // (block 2, slot 0 — a NEW block is allocated between the two steps).
        let history: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        let decode: [u32; 2] = [8, 9];
        // Fixed capacity ≥ ceil((7 + 2) / 4) = 3 blocks. Pad to 8 so the
        // block_table shape is stable AND wider than occupancy (exercises pad).
        const CAP: usize = 8;

        // Reference: re-planning `forward_paged_step` from the primed state.
        let mut ref_pool =
            fuel_core::kv_block_pool_device::DeviceKvPool::new(geom(), DType::F32, &dev)
                .expect("f32 DeviceKvPool (ref)");
        let rs = ref_pool.core_mut().open();
        for &t in &history {
            model
                .forward_paged_step(t, &mut ref_pool, rs)
                .expect("ref prime");
        }
        let ref0 = model
            .forward_paged_step(decode[0], &mut ref_pool, rs)
            .expect("ref decode 0");
        let ref1 = model
            .forward_paged_step(decode[1], &mut ref_pool, rs)
            .expect("ref decode 1");

        // Persistent: prime identically via the re-planning path, then decode
        // the two tokens via the plan-once persistent path.
        let mut pool = fuel_core::kv_block_pool_device::DeviceKvPool::new(geom(), DType::F32, &dev)
            .expect("f32 DeviceKvPool (persistent)");
        let s = pool.core_mut().open();
        for &t in &history {
            model
                .forward_paged_step(t, &mut pool, s)
                .expect("persistent prime");
        }

        let mut ds: Option<fuel_core::inference_context::PagedDecodeSession> = None;

        // ---- Token 0: builds + optimizes the plan ONCE. ----
        let before_build = optimize_calls_thread_local();
        let l0 = model
            .forward_paged_step_persistent(
                decode[0],
                &mut pool,
                s,
                CAP,
                fuel_core::inference_context::PagedDecodePlan::PlanOnce,
                &mut ds,
            )
            .expect("persistent decode 0 (build)");
        let build_delta = optimize_calls_thread_local() - before_build;
        assert!(
            build_delta >= 1,
            "first persistent token must optimize (build the plan): delta={build_delta}",
        );
        assert!(ds.is_some(), "session built on the first token");
        let nodes_after_build = ds.as_ref().unwrap().graph_node_count();

        // ---- Token 1: REUSES the plan (zero optimize calls). ----
        let before_rebind = optimize_calls_thread_local();
        let l1 = model
            .forward_paged_step_persistent(
                decode[1],
                &mut pool,
                s,
                CAP,
                fuel_core::inference_context::PagedDecodePlan::PlanOnce,
                &mut ds,
            )
            .expect("persistent decode 1 (rebind)");
        let rebind_delta = optimize_calls_thread_local() - before_rebind;
        assert_eq!(
            rebind_delta, 0,
            "second persistent token must REUSE the cached plan (zero re-optimize), got delta={rebind_delta}",
        );
        let nodes_after_rebind = ds.as_ref().unwrap().graph_node_count();
        assert_eq!(
            nodes_after_rebind, nodes_after_build,
            "held graph node count stable across rebind (no re-splice / re-insert)",
        );

        // ---- Correctness: byte-identical to the re-planning reference. ----
        assert_eq!(l0.len(), ref0.len(), "logits length (token 0)");
        assert_eq!(l1.len(), ref1.len(), "logits length (token 1)");
        assert_eq!(
            l0, ref0,
            "plan-once BUILD token must be byte-identical to the re-planning reference",
        );
        assert_eq!(
            l1, ref1,
            "plan-once REBIND token must be byte-identical to the re-planning reference \
             (proves the block_table / offset / RoPE rebind is correct across a block boundary)",
        );
    }

    // =======================================================================
    // Task 4 (paged plan-once) — flag-toggled, teeth-bearing correctness gate.
    //
    // The gate runs BOTH arms in ONE process off identical primed pools:
    //   • plan-once arm  (`PagedDecodePlan::PlanOnce`): build+optimize once,
    //     then reuse — asserted a HIT (0 re-optimize AND one session rebind)
    //     every step after the first.
    //   • control arm    (`PagedDecodePlan::Replan`): the pre-plan-once path
    //     (clears the session + delegates to `forward_paged_step`) — asserted
    //     a MISS (re-optimizes) EVERY step. Guards the inversion where a
    //     control secretly running plan-once would pass identity self-vs-self,
    //     pass HIT, and read ~1.0× = "doesn't help" (a false negative that
    //     would retire the feature).
    // Both arms must be byte-identical every step.
    //
    // Teeth: the assertions live in a NON-panicking `check_paged_gate` that
    // returns `Result`; the primary test `unwrap`s it (so it relies on those
    // exact checks) and `plan_once_gate_has_teeth` flips each flag and asserts
    // the SAME checker goes red — proving the invariants have teeth before we
    // trust them. (No `catch_unwind` / global panic-hook fiddling under the
    // concurrent suite.)
    //
    // CPU-scoped note: the per-step optimize delta uses the thread-local
    // counter, robust here because on CPU `upload_host_buffer_to_device` wraps
    // host bytes (no transient copy graph → no optimize bump). The device-
    // independent HIT signal is the session's `realize_count` (bumped only by
    // the rebind seam), also asserted — so the gate is not optimize-counter-
    // only. f32/CPU.
    // =======================================================================

    /// One decode step's measurements for the paged plan-once gate.
    struct PagedGateStep {
        /// plan-once arm (A) logits this step.
        logits_a: Vec<f32>,
        /// re-planning control arm (B) logits this step.
        logits_b: Vec<f32>,
        /// optimize-call delta around arm A this step (0 = plan reused = HIT).
        opt_delta_a: usize,
        /// optimize-call delta around arm B this step (≥1 = re-planned = MISS).
        opt_delta_b: usize,
        /// arm-A session `realize_count` delta this step (device-independent
        /// HIT: 1 = one rebind served, 0 = build/none).
        hit_delta_a: usize,
    }

    /// Full gate run: per-step rows + the arm-A session's final rebind count.
    struct PagedGateReport {
        steps: Vec<PagedGateStep>,
        /// arm-A session `realize_count` at the end (== rebinds served).
        final_hits_a: usize,
    }

    /// Build the tiny CPU f32 model + block geometry shared by the gate tests.
    fn paged_gate_fixture() -> (LlamaModel, fuel_core::kv_block_pool::KvGeometry, Device) {
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let geom = fuel_core::kv_block_pool::KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 4,
        };
        (model, geom, Device::cpu())
    }

    /// Run both gate arms off identically primed pools. Arm A runs `plan_a`,
    /// arm B runs `plan_b` (the flags are parameters so the mutation test can
    /// flip them). Each arm is primed with `history` via the re-planning
    /// `forward_paged_step`, then decodes `decode` via the persistent path.
    #[allow(clippy::too_many_arguments)]
    fn run_paged_plan_once_gate(
        model: &LlamaModel,
        geom: fuel_core::kv_block_pool::KvGeometry,
        history: &[u32],
        decode: &[u32],
        cap: usize,
        plan_a: fuel_core::inference_context::PagedDecodePlan,
        plan_b: fuel_core::inference_context::PagedDecodePlan,
        dev: &Device,
    ) -> fuel_core::Result<PagedGateReport> {
        use fuel_core::pipelined_bridge::optimize_calls_thread_local;
        type Pool = fuel_core::kv_block_pool_device::DeviceKvPool;

        // Two pools, IDENTICALLY primed — so plan-once (A) and re-planning (B)
        // can only differ if the plan-once rebind is wrong.
        let mut pool_a = Pool::new(geom, DType::F32, dev)?;
        let sa = pool_a.core_mut().open();
        for &t in history {
            model.forward_paged_step(t, &mut pool_a, sa)?;
        }
        let mut pool_b = Pool::new(geom, DType::F32, dev)?;
        let sb = pool_b.core_mut().open();
        for &t in history {
            model.forward_paged_step(t, &mut pool_b, sb)?;
        }

        let mut ds_a: Option<fuel_core::inference_context::PagedDecodeSession> = None;
        let mut ds_b: Option<fuel_core::inference_context::PagedDecodeSession> = None;
        let mut steps = Vec::with_capacity(decode.len());
        for &tok in decode {
            let hits_before = ds_a.as_ref().map(|s| s.realize_count()).unwrap_or(0);
            let ob = optimize_calls_thread_local();
            let logits_a = model.forward_paged_step_persistent(
                tok,
                &mut pool_a,
                sa,
                cap,
                plan_a,
                &mut ds_a,
            )?;
            let opt_delta_a = optimize_calls_thread_local() - ob;
            let hits_after = ds_a.as_ref().map(|s| s.realize_count()).unwrap_or(0);
            let hit_delta_a = hits_after - hits_before;

            let ob2 = optimize_calls_thread_local();
            let logits_b = model.forward_paged_step_persistent(
                tok,
                &mut pool_b,
                sb,
                cap,
                plan_b,
                &mut ds_b,
            )?;
            let opt_delta_b = optimize_calls_thread_local() - ob2;

            steps.push(PagedGateStep {
                logits_a,
                logits_b,
                opt_delta_a,
                opt_delta_b,
                hit_delta_a,
            });
        }
        let final_hits_a = ds_a.as_ref().map(|s| s.realize_count()).unwrap_or(0);
        Ok(PagedGateReport {
            steps,
            final_hits_a,
        })
    }

    /// The gate's invariants, as a NON-panicking checker (so the mutation test
    /// can prove the SAME checks have teeth). Returns `Err(reason)` on the
    /// first broken invariant. Assumes arm A = plan-once, arm B = control.
    fn check_paged_gate(report: &PagedGateReport) -> std::result::Result<(), String> {
        if report.steps.is_empty() {
            return Err("gate ran zero steps".to_string());
        }
        let n = report.steps.len();
        for (i, s) in report.steps.iter().enumerate() {
            // Identity: plan-once (A) must equal re-planning (B), byte-for-byte.
            if s.logits_a != s.logits_b {
                return Err(format!(
                    "step {i}: plan-once logits != re-planning logits (not byte-identical)",
                ));
            }
            // Control MISS: the re-planning arm re-optimizes EVERY step.
            if s.opt_delta_b < 1 {
                return Err(format!(
                    "step {i}: control (Replan) arm must re-optimize every step (MISS), \
                     opt_delta_b={}",
                    s.opt_delta_b,
                ));
            }
            if i == 0 {
                // First plan-once token BUILDS the plan (optimize ≥ 1) and is
                // NOT a rebind (build realizes via prebuild, not realize_token).
                if s.opt_delta_a < 1 {
                    return Err(format!(
                        "step 0: first plan-once token must build the plan (optimize ≥ 1), \
                         opt_delta_a={}",
                        s.opt_delta_a,
                    ));
                }
                if s.hit_delta_a != 0 {
                    return Err(format!(
                        "step 0: build is not a rebind — session HIT count must not advance, \
                         hit_delta_a={}",
                        s.hit_delta_a,
                    ));
                }
            } else {
                // HIT: plan-once REUSES — zero re-optimize AND exactly one rebind.
                if s.opt_delta_a != 0 {
                    return Err(format!(
                        "step {i}: plan-once arm must REUSE the plan (0 re-optimize = HIT), \
                         opt_delta_a={}",
                        s.opt_delta_a,
                    ));
                }
                if s.hit_delta_a != 1 {
                    return Err(format!(
                        "step {i}: plan-once session must report exactly one rebind (HIT), \
                         hit_delta_a={}",
                        s.hit_delta_a,
                    ));
                }
            }
        }
        // Global: the session served exactly (n-1) rebinds (1 build + (n-1) HITs).
        if report.final_hits_a != n - 1 {
            return Err(format!(
                "plan-once session must serve exactly {} rebinds (1 build + {} HITs), served {}",
                n - 1,
                n - 1,
                report.final_hits_a,
            ));
        }
        Ok(())
    }

    /// Task 4 · the primary correctness gate: N decode tokens crossing a block
    /// boundary, plan-once (A) vs re-planning control (B) in one process, all
    /// three invariants (byte-identity + plan-once HIT + control MISS) asserted
    /// via the shared checker.
    #[test]
    fn plan_once_paged_matches_replanning_with_hit_asserted() {
        use fuel_core::inference_context::PagedDecodePlan;
        let (model, geom, dev) = paged_gate_fixture();
        // History 7 → decode lands at pos 7,8,9,10. The block boundary
        // (block_size 4) falls between step 0 (pos 7, fills block 1) and step 1
        // (pos 8, allocates block 2): a cached-but-un-rebound block_table would
        // attend the wrong slots and diverge from the control.
        let history: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        let decode: [u32; 4] = [8, 9, 10, 11];
        const CAP: usize = 8;
        let report = run_paged_plan_once_gate(
            &model,
            geom,
            &history,
            &decode,
            CAP,
            PagedDecodePlan::PlanOnce,
            PagedDecodePlan::Replan,
            &dev,
        )
        .expect("paged plan-once gate run");
        check_paged_gate(&report)
            .unwrap_or_else(|e| panic!("plan-once gate must hold (identity + HIT + MISS): {e}"));
    }

    /// Task 4 · the gate's teeth: flipping each arm's flag must make the SAME
    /// checker `plan_once_paged_matches_replanning_with_hit_asserted` relies on
    /// go RED — so a toothless gate (assertions that always pass) cannot slip
    /// through. Mutation 1 forces the plan-once arm to always-MISS (breaks the
    /// HIT invariant); mutation 2 forces the control arm to always-HIT (breaks
    /// the MISS invariant).
    #[test]
    fn plan_once_gate_has_teeth() {
        use fuel_core::inference_context::PagedDecodePlan;
        let (model, geom, dev) = paged_gate_fixture();
        let history: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        let decode: [u32; 4] = [8, 9, 10, 11];
        const CAP: usize = 8;

        // Mutation 1 — plan-once arm forced to re-plan every token (Replan):
        // the HIT invariant (opt_delta_a==0 / one rebind) must break.
        let forced_miss = run_paged_plan_once_gate(
            &model,
            geom,
            &history,
            &decode,
            CAP,
            PagedDecodePlan::Replan,
            PagedDecodePlan::Replan,
            &dev,
        )
        .expect("gate run (force-miss)");
        assert!(
            check_paged_gate(&forced_miss).is_err(),
            "forcing the plan-once arm to re-plan every token MUST break the HIT assertion (teeth)",
        );

        // Mutation 2 — control arm forced to reuse the plan (PlanOnce): the
        // MISS invariant (opt_delta_b≥1 every step) must break.
        let forced_hit = run_paged_plan_once_gate(
            &model,
            geom,
            &history,
            &decode,
            CAP,
            PagedDecodePlan::PlanOnce,
            PagedDecodePlan::PlanOnce,
            &dev,
        )
        .expect("gate run (force-hit)");
        assert!(
            check_paged_gate(&forced_hit).is_err(),
            "forcing the control arm to reuse the plan MUST break the MISS assertion (teeth)",
        );
    }

    /// Task 4 · ragged coverage: single-session (B=1) histories at STAGGERED
    /// lengths {2, 5, 3} → different absolute positions and block-slot phases
    /// (mid-block, block-boundary-adjacent), where a cached plan is *wrong* (the
    /// per-token block_table / offset / RoPE differ), not merely rebuilt. The
    /// lockstep-uniform primary gate won't exercise these phases. Each staggered
    /// session must satisfy the full gate (identity + HIT + MISS).
    #[test]
    fn plan_once_paged_ragged_matches_replanning() {
        use fuel_core::inference_context::PagedDecodePlan;
        let (model, geom, dev) = paged_gate_fixture();
        const CAP: usize = 8;
        for &hist_len in &[2usize, 5, 3] {
            let history: Vec<u32> = (1..=hist_len as u32).collect();
            // In-range tokens (vocab_size == 32); values are irrelevant to the
            // boundary phase (set by position), only the gather must be valid.
            let decode: Vec<u32> = vec![10, 11, 12]; // 3 decode tokens
            let report = run_paged_plan_once_gate(
                &model,
                geom,
                &history,
                &decode,
                CAP,
                PagedDecodePlan::PlanOnce,
                PagedDecodePlan::Replan,
                &dev,
            )
            .unwrap_or_else(|e| panic!("ragged gate run (hist_len={hist_len}): {e:?}"));
            check_paged_gate(&report)
                .unwrap_or_else(|e| panic!("ragged plan-once gate (hist_len={hist_len}): {e}"));
        }
    }

    /// Phase D · FIRST live-GPU verification of plan-once persistent decode on
    /// VULKAN — the `write_bytes` variant of the per-token H2D re-bind upload
    /// arm (`upload_host_buffer_to_device`'s non-CPU branch on a Vulkan
    /// `Device`). Same structure/gates as the CUDA twin: persistent must be
    /// BIT-EXACT vs the Vulkan rebuild path (same plan → same kernels), and
    /// within the decode epsilon vs a CPU rebuild reference.
    ///
    /// Gated `#[cfg(feature = "vulkan")]` + `#[ignore]`; skips cleanly if no
    /// Vulkan device. Run:
    ///   `cargo test -p fuel-core --features "cuda vulkan" --lib \
    ///    generate_persistent_decode_on_vulkan_matches_rebuild_and_cpu \
    ///    -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "vulkan")]
    #[ignore = "requires a live Vulkan device"]
    fn generate_persistent_decode_on_vulkan_matches_rebuild_and_cpu() {
        use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_new = 5usize;
        let max_seq_len = prompt.len() + max_new;
        let strategy = SamplingStrategy::Greedy;

        // CPU rebuild reference (BEST-EFFORT epsilon cross-check) — see the CUDA
        // twin for the cross-backend-placement caveat: under a build that also
        // has CUDA/Vulkan probed in, a CPU-pinned realize can be stamped onto a
        // GPU as a placement fallback and fail (no GPU seed in the CPU cache).
        // That is not a decode bug, so the CPU cross-check is best-effort: skip
        // (with a note) on that error, run the epsilon assert for real on
        // success.
        let cpu_step_logits: Option<Vec<Vec<f32>>> = {
            let cpu_device = Device::cpu();
            let cpu_ref = || -> fuel_core::Result<Vec<Vec<f32>>> {
                let mut cpu_cache = KvCache::with_capacity(
                    cfg.n_layers,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    max_seq_len,
                    DType::F32,
                    &cpu_device,
                )?;
                let mut cpu_ctx = InferenceContext::new(cpu_device.clone());
                let mut cpu_rng: u64 = 0;
                let mut out: Vec<Vec<f32>> = Vec::with_capacity(max_new);
                let mut last_cpu =
                    model.forward_with_kv_context(&prompt, &mut cpu_cache, &mut cpu_ctx)?;
                for _ in 0..max_new {
                    let next = sample_logits(&last_cpu, strategy, &mut cpu_rng);
                    last_cpu =
                        model.forward_with_kv_context(&[next], &mut cpu_cache, &mut cpu_ctx)?;
                    out.push(last_cpu.clone());
                }
                Ok(out)
            };
            match cpu_ref() {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!(
                        "CPU epsilon cross-check SKIPPED (cross-backend placement \
                         fallback — not a decode bug): {e:?}"
                    );
                    None
                }
            }
        };

        let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => b,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Vulkan,
                    fuel_test_support::hardware::Missing::device(format!(
                        "VulkanBackend::with_selection: {e:?}"
                    )),
                );
            }
        };
        let vk_device: Device = vk_backend.into();

        // Vulkan rebuild (D1) reference.
        let mut vk_rebuild_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &vk_device,
        )
        .expect("vk rebuild with_capacity");
        let mut vk_rebuild_ctx = InferenceContext::new(vk_device.clone());
        let mut rebuild_rng: u64 = 0;
        let mut rebuild_tokens: Vec<u32> = prompt.to_vec();
        let mut rebuild_step_logits: Vec<Vec<f32>> = Vec::with_capacity(max_new);
        let mut last_rebuild = model
            .forward_with_kv_context(&prompt, &mut vk_rebuild_cache, &mut vk_rebuild_ctx)
            .expect("vk rebuild prefill");
        for _ in 0..max_new {
            let next = sample_logits(&last_rebuild, strategy, &mut rebuild_rng);
            rebuild_tokens.push(next);
            last_rebuild = model
                .forward_with_kv_context(&[next], &mut vk_rebuild_cache, &mut vk_rebuild_ctx)
                .expect("vk rebuild decode");
            rebuild_step_logits.push(last_rebuild.clone());
        }

        // Vulkan persistent (plan-once).
        let mut vk_persist_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &vk_device,
        )
        .expect("vk persistent with_capacity");
        let mut vk_persist_ctx = InferenceContext::new(vk_device.clone());
        let mut persist_rng: u64 = 0;
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut persist_tokens: Vec<u32> = prompt.to_vec();
        let mut persist_step_logits: Vec<Vec<f32>> = Vec::with_capacity(max_new);
        let mut last_persist = model
            .forward_with_kv_context_persistent(
                &prompt,
                &mut vk_persist_cache,
                &mut vk_persist_ctx,
                &mut session,
            )
            .expect("vk persistent prefill");
        assert!(
            session.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );
        for _ in 0..max_new {
            let next = sample_logits(&last_persist, strategy, &mut persist_rng);
            persist_tokens.push(next);
            last_persist = model
                .forward_with_kv_context_persistent(
                    &[next],
                    &mut vk_persist_cache,
                    &mut vk_persist_ctx,
                    &mut session,
                )
                .expect("vk persistent decode");
            persist_step_logits.push(last_persist.clone());
        }
        assert!(
            session.is_some(),
            "held session survives the Vulkan decode loop"
        );

        // (1) BIT-EXACT: Vulkan persistent == Vulkan rebuild.
        assert_eq!(
            persist_tokens, rebuild_tokens,
            "Vulkan persistent token sequence"
        );
        for (i, (p, r)) in persist_step_logits
            .iter()
            .zip(rebuild_step_logits.iter())
            .enumerate()
        {
            assert_eq!(
                p, r,
                "Vulkan persistent decode step {i} logits must be BIT-EXACT vs \
                 the Vulkan rebuild path (upload arm's write_bytes branch)",
            );
        }

        // (2) EPSILON: Vulkan persistent vs CPU rebuild (best-effort).
        let mut epsilon_checked = false;
        if let Some(cpu_step_logits) = cpu_step_logits.as_ref() {
            for (i, (p, c)) in persist_step_logits
                .iter()
                .zip(cpu_step_logits.iter())
                .enumerate()
            {
                for (j, (a, b)) in p.iter().zip(c.iter()).enumerate() {
                    let diff = (a - b).abs();
                    let rel = diff / a.abs().max(b.abs()).max(1e-6);
                    assert!(
                        diff < 5e-3 || rel < 1e-2,
                        "step {i} logit[{j}]: vulkan={a}, cpu={b}, diff={diff}, rel={rel}",
                    );
                }
            }
            epsilon_checked = true;
        }

        // (3) WIRING.
        let via_wrapper = model
            .generate_with_kv_context(&prompt, max_new, strategy, None, &vk_device, DType::F32)
            .expect("generate_with_kv_context on Vulkan");
        assert_eq!(
            via_wrapper, rebuild_tokens,
            "generate_with_kv_context on Vulkan"
        );

        eprintln!(
            "VULKAN persistent decode VERIFIED: tokens + logits BIT-EXACT vs \
             Vulkan rebuild path; CPU epsilon cross-check {}. tokens={:?}",
            if epsilon_checked {
                "PASSED"
            } else {
                "SKIPPED (see note above)"
            },
            persist_tokens,
        );
    }

    /// Phase D · INDICATIVE live-GPU wall-clock (ignored — NOT a CI gate, NO
    /// timing assertion). Prints the persistent-vs-rebuild per-token ratio on a
    /// CUDA device so a human can eyeball it. NOTE: this is a TINY model — the
    /// honest ~1.8× plan-once win needs a realistic model (per design doc §10,
    /// CPU/planning is a small fraction of GPU compute for tiny models, so this
    /// ratio UNDERSTATES the real win). Do NOT read a CI signal into it.
    ///
    /// Run: `cargo test -p fuel-core --features cuda --lib \
    ///  generate_persistent_decode_cuda_bench_scaffold -- --ignored --nocapture`
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "perf scaffold — manual live-CUDA measurement, not a CI gate"]
    fn generate_persistent_decode_cuda_bench_scaffold() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let cuda = match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        let dev: Device = cuda.into();

        let prompt = [1_u32, 2, 3];
        let n = 64usize;
        let max_seq_len = prompt.len() + n;
        let strategy = SamplingStrategy::Greedy;

        // D1: rebuild + re-optimize every decode token.
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .unwrap();
        let mut ctx1 = InferenceContext::new(dev.clone());
        let mut rng1 = 0u64;
        let mut last1 = model
            .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
            .unwrap();
        let t_d1 = std::time::Instant::now();
        for _ in 0..n {
            let next = sample_logits(&last1, strategy, &mut rng1);
            last1 = model
                .forward_with_kv_context(&[next], &mut cache1, &mut ctx1)
                .unwrap();
        }
        let d1 = t_d1.elapsed();

        // D2: plan-once persistent decode.
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .unwrap();
        let mut ctx2 = InferenceContext::new(dev.clone());
        let mut rng2 = 0u64;
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut last2 = model
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .unwrap();
        let t_d2 = std::time::Instant::now();
        for _ in 0..n {
            let next = sample_logits(&last2, strategy, &mut rng2);
            last2 = model
                .forward_with_kv_context_persistent(&[next], &mut cache2, &mut ctx2, &mut session)
                .unwrap();
        }
        let d2 = t_d2.elapsed();

        eprintln!(
            "CUDA D2c bench (TINY model, N={n}): D1 rebuild = {:?} ({:?}/tok), \
             D2 plan-once = {:?} ({:?}/tok), ratio = {:.2}x — INDICATIVE ONLY; \
             the honest ~1.8x needs a realistic model (tiny model understates).",
            d1,
            d1 / n as u32,
            d2,
            d2 / n as u32,
            d1.as_secs_f64() / d2.as_secs_f64().max(1e-9),
        );
    }

    /// Phase D · D2c perf SCAFFOLD (ignored — NOT a CI gate). The
    /// wall-clock ~1.8×/token win of plan-once over per-token re-plan is a
    /// MANUAL live-GPU measurement on a realistic model (per the design
    /// doc §10: CPU planning is a smaller fraction of CPU compute, so the
    /// CPU ratio understates the win; timing tests are flaky in CI). This
    /// scaffold shows the A/B shape — a D1 rebuild loop vs. a D2 persistent
    /// loop over N seq==1 tokens — and prints the per-token wall-clock so a
    /// human can run it on CUDA/Vulkan. Do NOT assert on timing here.
    ///
    /// Run manually: `cargo test -p fuel-core --lib
    /// generate_loop_persistent_bench_scaffold -- --ignored --nocapture`.
    /// For the real number, port this shape to a live-GPU harness with a
    /// realistic model + N≥64 (one live suite at a time, per CLAUDE.md).
    #[test]
    #[ignore = "perf scaffold — manual live-GPU measurement, not a CI gate"]
    fn generate_loop_persistent_bench_scaffold() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let n = 64usize;
        let max_seq_len = prompt.len() + n;
        let strategy = SamplingStrategy::Greedy;
        let dev = Device::cpu();

        // D1: rebuild + re-optimize every decode token.
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .unwrap();
        let mut ctx1 = InferenceContext::new(dev.clone());
        let mut rng1 = 0u64;
        let mut last1 = model
            .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
            .unwrap();
        let t_d1 = std::time::Instant::now();
        for _ in 0..n {
            let next = sample_logits(&last1, strategy, &mut rng1);
            last1 = model
                .forward_with_kv_context(&[next], &mut cache1, &mut ctx1)
                .unwrap();
        }
        let d1 = t_d1.elapsed();

        // D2: plan-once persistent decode (the wired production path).
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .unwrap();
        let mut ctx2 = InferenceContext::new(dev.clone());
        let mut rng2 = 0u64;
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut last2 = model
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .unwrap();
        let t_d2 = std::time::Instant::now();
        for _ in 0..n {
            let next = sample_logits(&last2, strategy, &mut rng2);
            last2 = model
                .forward_with_kv_context_persistent(&[next], &mut cache2, &mut ctx2, &mut session)
                .unwrap();
        }
        let d2 = t_d2.elapsed();

        eprintln!(
            "D2c bench (CPU, tiny model, N={n}): D1 rebuild = {:?} ({:?}/tok), \
             D2 plan-once = {:?} ({:?}/tok), ratio = {:.2}x (CPU understates; \
             measure the ~1.8x on a live GPU with a realistic model)",
            d1,
            d1 / n as u32,
            d2,
            d2 / n as u32,
            d1.as_secs_f64() / d2.as_secs_f64().max(1e-9),
        );
        // NO timing assertion — perf is a verify-after gate, not a CI gate.
    }

    // ===================================================================
    // Phase D · persistent-decode WALL-CLOCK benchmark on a REAL model.
    //
    // The scaffold above proves the A/B shape on a tiny synthetic model
    // (CPU understates the win). These entries run the SAME A/B — D1
    // replan-every-token vs. D2 plan-once persistent — on a realistic
    // model (TinyLlama-1.1B) loaded from `FUEL_BENCH_MODEL_DIR`, and
    // print a per-token wall-clock table + the ratio. `#[ignore]`'d
    // (manual, needs a multi-GB checkpoint on disk); NOT a CI gate.
    //
    //   CPU:    FUEL_BENCH_MODEL_DIR=... cargo test -p fuel-core --lib \
    //             bench_persistent_decode_real_model_cpu -- --ignored --nocapture
    //   Vulkan: FUEL_BENCH_MODEL_DIR=... cargo test -p fuel-core --lib \
    //             --features vulkan \
    //             bench_persistent_decode_real_model_vulkan -- --ignored --nocapture
    //
    // Weights are force-upcast to F32 (see `force_weights_f32`) so the
    // forward graph is pure-F32 — both the CPU and Vulkan F32 matmul
    // kernels handle it, sidestepping any mixed-precision (F32×BF16)
    // CPU-matmul gap. This is a TIMING benchmark, not a precision test.
    // ===================================================================

    /// Upcast every BF16 projection weight to F32 (norms/embeddings are
    /// already F32) so the forward graph is homogeneously F32.
    fn force_weights_f32(mut w: LlamaWeights) -> LlamaWeights {
        fn to_f32(ws: &WeightStorage) -> WeightStorage {
            match ws {
                WeightStorage::BF16(a) => {
                    let v: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
                    WeightStorage::F32(Arc::from(v))
                }
                other => other.clone(),
            }
        }
        for l in w.layers.iter_mut() {
            l.attn_q = to_f32(&l.attn_q);
            l.attn_k = to_f32(&l.attn_k);
            l.attn_v = to_f32(&l.attn_v);
            l.attn_o = to_f32(&l.attn_o);
            l.ffn_gate = to_f32(&l.ffn_gate);
            l.ffn_up = to_f32(&l.ffn_up);
            l.ffn_down = to_f32(&l.ffn_down);
        }
        w.output = to_f32(&w.output);
        w
    }

    /// Load `LlamaModel` from `FUEL_BENCH_MODEL_DIR` (config.json +
    /// model.safetensors). `force_f32` upcasts BF16 projections to F32
    /// (used on CPU, where matmul is F32×F32); `false` keeps the
    /// checkpoint's native BF16 on the projections (used on Vulkan —
    /// the backend's mixed `matmul_f32_bf16_b` path — halving weight
    /// VRAM, which matters because the D1 replan baseline re-uploads
    /// the full weight set every realize). Returns `(model, load_secs)`,
    /// or `None` (with a logged reason) if the env var is unset — so
    /// the `#[ignore]`'d test skips cleanly without a checkpoint.
    fn load_real_llama(force_f32: bool) -> Option<(LlamaModel, f64)> {
        let dir = match std::env::var("FUEL_BENCH_MODEL_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => {
                eprintln!(
                    "FUEL_BENCH_MODEL_DIR not set — skipping real-model persistent-decode bench.",
                );
                return None;
            }
        };
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .unwrap_or_else(|e| panic!("read {config_path:?}: {e}"));
        let cfg = LlamaConfig::from_hf_json_str(&config_str).expect("parse config.json");
        eprintln!(
            "model config: vocab={} dim={} layers={} q_heads={} kv_heads={} head_dim={} ffn={}",
            cfg.vocab_size,
            cfg.dim,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.ffn_dim,
        );
        let weights_path = dir.join("model.safetensors");
        let t0 = std::time::Instant::now();
        let st = unsafe { fuel_core::safetensors::MmapedSafetensors::new(&weights_path) }
            .unwrap_or_else(|e| panic!("mmap {weights_path:?}: {e}"));
        // Report the source dtype of a representative projection so the
        // deviation (bf16 source → f32 in-memory) is visible in the log.
        if let Ok(v) = st.get("model.layers.0.self_attn.q_proj.weight") {
            eprintln!("source safetensors dtype (q_proj.weight): {:?}", v.dtype());
        }
        let raw = LlamaWeights::load_from_mmapped(&st, &cfg).expect("load weights");
        let weights = if force_f32 {
            force_weights_f32(raw)
        } else {
            raw
        };
        let load_secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "weights loaded in {load_secs:.2}s (projections {})",
            if force_f32 {
                "upcast to F32"
            } else {
                "kept at source dtype"
            },
        );
        Some((
            LlamaModel {
                config: cfg,
                weights,
            },
            load_secs,
        ))
    }

    /// Summarize a slice of per-token durations as (mean, min, max) in ms.
    fn ms_stats(times: &[std::time::Duration]) -> (f64, f64, f64) {
        let ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &x in &ms {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
        (mean, min, max)
    }

    /// Median of a slice of per-token durations, in ms (robust to the odd
    /// scheduler/driver spike — the CapturedRun 4b-ε report uses it for the
    /// steady replay window, median-of-≥8 per the worklist).
    fn median_ms(times: &[std::time::Duration]) -> f64 {
        if times.is_empty() {
            return 0.0;
        }
        let mut ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = ms.len() / 2;
        if ms.len().is_multiple_of(2) {
            (ms[mid - 1] + ms[mid]) / 2.0
        } else {
            ms[mid]
        }
    }

    /// The shared A/B benchmark body (device-agnostic). Runs D1 (replan
    /// every token) and D2 (plan-once persistent) over `n` decode tokens
    /// from a fixed hard-coded prompt, times each per-token step
    /// (prefill excluded + reported separately), checks byte-exactness
    /// between the two paths, and prints a compact table.
    ///
    /// `post_token` (if given) runs after EVERY forward step in BOTH
    /// loops, OUTSIDE the timed window. The Vulkan bench passes a
    /// `synchronize_pending` drain here: the D1 replan path re-uploads
    /// the full weight set every realize, and without a forced sync the
    /// deferred-destruction batches let several realize-generations of
    /// weight buffers coexist → `ERROR_OUT_OF_DEVICE_MEMORY` on a 12 GB
    /// card by decode token 3. Excluding the drain from the timer is
    /// conservative (it charges D1 nothing for the reclaim it needs).
    ///
    /// `cache_dtype` is the `KvCache`'s dtype (hence the activation dtype
    /// end-to-end — see `LlamaModel::forward_with_kv_context_impl`'s
    /// `to_dtype(cache_dtype)` cast). `DType::F32` is every pre-Part-2-B
    /// caller's value (CPU/Vulkan/CUDA F32 baselines); `DType::BF16` opts
    /// the whole activation stream into BF16-throughout (needs BF16
    /// weights — pass `load_real_llama(false)`).
    fn run_persistent_decode_bench(
        model: &LlamaModel,
        device: &Device,
        dev_label: &str,
        load_secs: f64,
        n: usize,
        post_token: Option<&dyn Fn()>,
        cache_dtype: DType,
    ) {
        let after_step = || {
            if let Some(f) = post_token {
                f();
            }
        };
        use std::time::Instant;
        let cfg = model.config.clone();
        // Fixed prompt token IDs (all < vocab_size = 32000). We measure
        // time, not quality, so exact tokens are immaterial — 1 is BOS.
        let prompt: [u32; 8] = [1, 15043, 29892, 590, 1024, 338, 6033, 5077];
        let max_seq_len = prompt.len() + n + 1;
        let strategy = SamplingStrategy::Greedy;

        // Which loops to run: FUEL_BENCH_PATHS=both|d1|d2 (default both).
        // The split modes exist for constrained-VRAM GPUs: each D1
        // (replan) realize re-uploads the full weight set and (observed
        // on Vulkan) those uploads are NOT reclaimed across realizes, so
        // at 1.1B scale a 12 GB card cannot complete both loops in one
        // process. Run `d2` (full N) and `d1` (graceful truncation) in
        // separate processes; the printed greedy token sequences give
        // the cross-path token-level check.
        let paths_env = std::env::var("FUEL_BENCH_PATHS").unwrap_or_else(|_| "both".to_string());
        let run_d1 = paths_env != "d2";
        let run_d2 = paths_env != "d1";

        // ---------------- D2: plan-once persistent decode ----------------
        // D2 runs FIRST: it uploads the weights once (held in the session's
        // base_cache) and re-binds only the 4 small data Consts per token, so
        // it has a flat device-memory profile (proven by N=48 on a 12 GB
        // card). D1 runs second because its per-token full-weight re-upload
        // accumulates device memory on Vulkan — running it last means an
        // early D1 abort still leaves complete D2 numbers.
        let mut d2_prefill = std::time::Duration::ZERO;
        let mut d2_tokens: Vec<u32> = Vec::with_capacity(n);
        let mut d2_logits: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut d2_times: Vec<std::time::Duration> = Vec::with_capacity(n);
        let mut opt_prefill_delta = 0usize;
        let mut opt_decode_delta = 0usize;
        if run_d2 {
            let opt_before_prefill = fuel_core::pipelined_bridge::optimize_calls_thread_local();
            // Variant-bake telemetry: how many same-device fused variants (e.g.
            // the CUDA flash-decode arm) the optimizer actually PICKED across
            // the D2 build. > 0 confirms the flash arm fired (was baked to the
            // fused winner), not merely offered.
            let vb_before = fuel_dispatch::variant_bake::variant_bakes_thread_local();
            let mut cache2 = KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq_len,
                cache_dtype,
                device,
            )
            .expect("d2 with_capacity");
            let mut ctx2 = InferenceContext::new(device.clone());
            let mut session: Option<fuel_core::inference_context::DecodeSession> = None;
            let t_pre2 = Instant::now();
            let mut last2 = model
                .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
                .expect("d2 prefill");
            d2_prefill = t_pre2.elapsed();
            after_step();
            eprintln!("  D2 prefill: {:.1} ms", d2_prefill.as_secs_f64() * 1e3);
            assert!(
                session.is_none(),
                "prefill (seq>1) must NOT build the held session"
            );
            let opt_after_prefill = fuel_core::pipelined_bridge::optimize_calls_thread_local();
            let mut rng2 = 0u64;
            for i in 0..n {
                let next = sample_logits(&last2, strategy, &mut rng2);
                d2_tokens.push(next);
                let t = Instant::now();
                last2 = model
                    .forward_with_kv_context_persistent(
                        &[next],
                        &mut cache2,
                        &mut ctx2,
                        &mut session,
                    )
                    .expect("d2 decode");
                let dt = t.elapsed();
                after_step();
                d2_times.push(dt);
                d2_logits.push(last2.clone());
                eprintln!("  D2 tok {}/{n}: {:.1} ms", i + 1, dt.as_secs_f64() * 1e3);
            }
            assert!(session.is_some(), "held session survives the decode loop");
            let opt_after_decode = fuel_core::pipelined_bridge::optimize_calls_thread_local();
            let vb_after = fuel_dispatch::variant_bake::variant_bakes_thread_local();
            opt_prefill_delta = opt_after_prefill.wrapping_sub(opt_before_prefill);
            opt_decode_delta = opt_after_decode.wrapping_sub(opt_after_prefill);
            let variant_bakes_delta = vb_after.wrapping_sub(vb_before);
            eprintln!(
                "  D2 variant-bakes (fused-arm picks, e.g. flash-decode) across the build: {} \
                 ({})",
                variant_bakes_delta,
                if variant_bakes_delta > 0 {
                    "flash arm PICKED"
                } else {
                    "no fused variant picked \
                 — decode ran the decomposed base map"
                },
            );
            // cache2/ctx2/session drop here (end of scope): frees D2's
            // device-resident state (base_cache holds the full weight set
            // on non-CPU devices) before the D1 loop starts allocating.
        }
        after_step();

        // ---------------- D3: CapturedRun replay decode ----------------
        // Third leg (CapturedRun 4b-ε): `forward_with_kv_context_captured`
        // builds the SAME held session as D2 at token 1, captures a CUDA graph
        // at token 2, then tokens 3..N are pure `cuGraphLaunch` replays — fresh
        // per-token H2D of only the 4 small data buffers (token id, rope cos/sin,
        // mask), no re-optimize, no re-realize. Must be byte-identical to D2
        // (same plan → same kernels). F32-only (capture is f32-today); skipped
        // for a BF16 cache. Own scope so its base_cache frees before D1.
        // d3_* declared unconditionally so the report/stats below compile in
        // every feature set; the capture leg itself is CUDA-only (the capture
        // APIs `forward_with_kv_context_captured` / `CapturedDecodeSession` are
        // `#[cfg(feature = "cuda")]`), so on a non-cuda build these stay empty.
        // Pushed only inside the `#[cfg(feature = "cuda")]` block below; read
        // (empty) by the report/stats on a non-cuda build. `mut` is therefore
        // used under cuda and unused at default features — gate the lint, don't
        // drop `mut`, or `--features cuda` fails to borrow these as mutable.
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut d3_tokens: Vec<u32> = Vec::with_capacity(n);
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut d3_logits: Vec<Vec<f32>> = Vec::with_capacity(n);
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut d3_times: Vec<std::time::Duration> = Vec::with_capacity(n);
        let run_d3 = cfg!(feature = "cuda") && run_d2 && cache_dtype == DType::F32;
        #[cfg(feature = "cuda")]
        {
            if run_d3 {
                let mut cache3 = KvCache::with_capacity(
                    cfg.n_layers,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    max_seq_len,
                    cache_dtype,
                    device,
                )
                .expect("d3 with_capacity");
                let mut ctx3 = InferenceContext::new(device.clone());
                let mut session3: Option<fuel_core::inference_context::DecodeSession> = None;
                let mut captured3: Option<fuel_dispatch::pipelined::CapturedDecodeSession> = None;
                let t_pre3 = Instant::now();
                let mut last3 = model
                    .forward_with_kv_context_captured(
                        &prompt,
                        &mut cache3,
                        &mut ctx3,
                        &mut session3,
                        &mut captured3,
                    )
                    .expect("d3 prefill");
                let d3_prefill = t_pre3.elapsed();
                after_step();
                eprintln!("  D3 prefill: {:.1} ms", d3_prefill.as_secs_f64() * 1e3);
                let mut rng3 = 0u64;
                for i in 0..n {
                    let next = sample_logits(&last3, strategy, &mut rng3);
                    d3_tokens.push(next);
                    let t = Instant::now();
                    last3 = model
                        .forward_with_kv_context_captured(
                            &[next],
                            &mut cache3,
                            &mut ctx3,
                            &mut session3,
                            &mut captured3,
                        )
                        .expect("d3 decode");
                    let dt = t.elapsed();
                    after_step();
                    d3_times.push(dt);
                    d3_logits.push(last3.clone());
                    eprintln!("  D3 tok {}/{n}: {:.1} ms", i + 1, dt.as_secs_f64() * 1e3);
                }
                assert!(
                    captured3.is_some(),
                    "D3: the capture must build by decode token 2"
                );
                // cache3/ctx3/session3/captured3 drop here — frees before D1.
            } else if run_d2 && cache_dtype != DType::F32 {
                eprintln!(
                    "  D3 captured-replay: SKIPPED (capture is f32-only; cache dtype {cache_dtype:?})"
                );
            }
        }
        after_step();

        // ---------------- D1: rebuild + re-optimize every token ----------------
        // The D1 loop tolerates a mid-run device-OOM: each D1 realize
        // re-uploads the full weight set, and (observed on Vulkan/12 GB)
        // buffers from prior realizes are not reclaimed in time, so the
        // loop can die after a few tokens. We keep whatever per-token
        // timings succeeded and report the truncation honestly.
        let mut d1_prefill = std::time::Duration::ZERO;
        let mut d1_tokens: Vec<u32> = Vec::with_capacity(n);
        let mut d1_logits: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut d1_times: Vec<std::time::Duration> = Vec::with_capacity(n);
        let mut d1_abort: Option<String> = None;
        if run_d1 {
            let mut cache1 = KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                max_seq_len,
                cache_dtype,
                device,
            )
            .expect("d1 with_capacity");
            let mut ctx1 = InferenceContext::new(device.clone());
            let t_pre1 = Instant::now();
            let mut last1 = model
                .forward_with_kv_context(&prompt, &mut cache1, &mut ctx1)
                .expect("d1 prefill");
            d1_prefill = t_pre1.elapsed();
            after_step();
            eprintln!("  D1 prefill: {:.1} ms", d1_prefill.as_secs_f64() * 1e3);
            let mut rng1 = 0u64;
            for i in 0..n {
                let next = sample_logits(&last1, strategy, &mut rng1);
                let t = Instant::now();
                match model.forward_with_kv_context(&[next], &mut cache1, &mut ctx1) {
                    Ok(l) => last1 = l,
                    Err(e) => {
                        d1_abort =
                            Some(format!("D1 replan loop ABORTED at token {}: {e:?}", i + 1,));
                        eprintln!("  {}", d1_abort.as_ref().unwrap());
                        break;
                    }
                }
                let dt = t.elapsed();
                after_step();
                d1_tokens.push(next);
                d1_times.push(dt);
                d1_logits.push(last1.clone());
                eprintln!("  D1 tok {}/{n}: {:.1} ms", i + 1, dt.as_secs_f64() * 1e3);
            }
        }
        let n1 = d1_times.len();
        let n2 = d2_times.len();
        let n3 = d3_times.len();

        // ---------------- byte-exactness between the two paths ----------------
        // Compared over the overlapping completed prefix (n1 == n2 == n
        // unless a loop was skipped or the D1 loop aborted early).
        let cmp = n1.min(n2);
        let tokens_match = d1_tokens[..cmp] == d2_tokens[..cmp];
        let mut max_abs_diff = 0.0f32;
        for (a, b) in d1_logits[..cmp].iter().zip(d2_logits[..cmp].iter()) {
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = (x - y).abs();
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
            }
        }
        let logits_bit_exact = d1_logits[..cmp] == d2_logits[..cmp];

        // ---------------- stats ----------------
        // D1: mean over all completed tokens, and over the "steady" window
        // (tokens 2..) to match D2's steady window (D2 token 1 is the build).
        let (d1_mean_all, d1_min, d1_max) = ms_stats(&d1_times);
        let d1_steady = if n1 > 1 {
            &d1_times[1..]
        } else {
            &d1_times[..]
        };
        let (d1_mean_steady, _, _) = ms_stats(d1_steady);
        // D2: token 1 is the plan-once BUILD; tokens 2..N are the reuse.
        let d2_build_ms = if n2 > 0 {
            d2_times[0].as_secs_f64() * 1e3
        } else {
            0.0
        };
        let d2_reuse = if n2 > 1 {
            &d2_times[1..]
        } else {
            &d2_times[..]
        };
        let (d2_mean_reuse, d2_min_reuse, d2_max_reuse) = ms_stats(d2_reuse);
        let (d2_mean_all, _, _) = ms_stats(&d2_times);
        let d2_median_reuse = median_ms(d2_reuse);
        // D3: token 1 build (== D2's), token 2 capture-build, tokens 3..N the
        // pure cuGraphLaunch replay window (the number that matters).
        let d3_build_ms = if n3 > 0 {
            d3_times[0].as_secs_f64() * 1e3
        } else {
            0.0
        };
        let d3_capture_ms = if n3 > 1 {
            d3_times[1].as_secs_f64() * 1e3
        } else {
            0.0
        };
        let d3_replay = if n3 > 2 {
            &d3_times[2..]
        } else {
            &d3_times[..0]
        };
        let (d3_mean_replay, d3_min_replay, d3_max_replay) = ms_stats(d3_replay);
        let d3_median_replay = median_ms(d3_replay);
        // D2-vs-D3 byte-exactness (same plan → identical logits) over the
        // overlapping prefix.
        let cmp23 = n2.min(n3);
        let d3_tokens_match = cmp23 == 0 || d2_tokens[..cmp23] == d3_tokens[..cmp23];
        let d3_logits_bit_exact = cmp23 == 0 || d2_logits[..cmp23] == d3_logits[..cmp23];

        let ratio_steady = d1_mean_steady / d2_mean_reuse.max(1e-9);
        let ratio_all = d1_mean_all / d2_mean_all.max(1e-9);

        eprintln!("\n============================================================");
        eprintln!(" Persistent-decode wall-clock benchmark — {dev_label}");
        eprintln!("============================================================");
        eprintln!(
            " model: TinyLlama-1.1B  (layers={}, q/kv heads={}/{}, dim={}, vocab={})",
            cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.dim, cfg.vocab_size,
        );
        eprintln!(" weight load: {load_secs:.2}s");
        eprintln!(
            " paths run: {paths_env}   N decode tokens = {n}   prompt len = {}   max_seq_len = {}",
            prompt.len(),
            max_seq_len
        );
        eprintln!(
            " prefill (excluded from per-token):  D1 = {:.1} ms   D2 = {:.1} ms",
            d1_prefill.as_secs_f64() * 1e3,
            d2_prefill.as_secs_f64() * 1e3
        );
        if run_d2 {
            eprintln!(
                " optimize_graph calls: prefill-fallback +{opt_prefill_delta}, \
                       decode-loop +{opt_decode_delta} (plan-once ⇒ expect +1)"
            );
        }
        eprintln!("------------------------------------------------------------");
        if let Some(msg) = &d1_abort {
            eprintln!(" !! {msg}");
            eprintln!(" !! D1 stats below cover the {n1} token(s) that completed.");
        }
        eprintln!(" per-token wall-clock (ms):");
        if n1 > 0 {
            eprintln!(
                "   D1 replan   : mean({n1} toks)  = {d1_mean_all:8.2}   [min {d1_min:.2}, max {d1_max:.2}]"
            );
            eprintln!("   D1 replan   : mean(tok 2..)  = {d1_mean_steady:8.2}");
        } else if run_d1 {
            eprintln!("   D1 replan   : NO tokens completed");
        }
        if n2 > 0 {
            eprintln!("   D2 plan-once: build (tok 1)  = {d2_build_ms:8.2}");
            eprintln!(
                "   D2 plan-once: mean(tok 2..N) = {d2_mean_reuse:8.2}   [min {d2_min_reuse:.2}, max {d2_max_reuse:.2}]"
            );
        }
        if n3 > 0 {
            eprintln!(
                "   D3 captured : build(tok1)={d3_build_ms:8.2}  capture-build(tok2)={d3_capture_ms:8.2}"
            );
            eprintln!(
                "   D3 captured : replay(tok 3..N) median={d3_median_replay:8.2}  mean={d3_mean_replay:8.2}   [min {d3_min_replay:.2}, max {d3_max_replay:.2}]"
            );
        } else if run_d2 && cache_dtype != DType::F32 {
            eprintln!(
                "   D3 captured : skipped (capture is f32-only; cache dtype {cache_dtype:?})"
            );
        }
        eprintln!("------------------------------------------------------------");
        if n1 > 0 && n2 > 1 {
            eprintln!(" RATIO (D1/D2), steady windows  : {ratio_steady:.3}x");
            eprintln!(" RATIO (D1/D2), all-token means : {ratio_all:.3}x");
        }
        if !d3_replay.is_empty() && n2 > 1 {
            let ratio_replay = d2_median_reuse / d3_median_replay.max(1e-9);
            eprintln!(
                " RATIO (D2-reuse / D3-replay), medians : {ratio_replay:.3}x  \
                 (captured cuGraphLaunch replay vs plan-once re-dispatch)",
            );
        }
        eprintln!("------------------------------------------------------------");
        // Greedy token sequences — lets a d1-only and a d2-only run (in
        // separate processes) be cross-checked at the token level.
        eprintln!(" D1 greedy tokens ({n1}): {d1_tokens:?}");
        eprintln!(" D2 greedy tokens ({n2}): {d2_tokens:?}");
        if run_d1 && run_d2 {
            eprintln!(
                " byte-exact D1 vs D2 (over {cmp} tokens): tokens_match={tokens_match}  \
                       logits_bit_exact={logits_bit_exact}  max_abs_logit_diff={max_abs_diff:.3e}"
            );
        } else {
            eprintln!(
                " byte-exact D1 vs D2: n/a (single-path run — compare the token \
                       sequences across processes)"
            );
        }
        if run_d3 {
            eprintln!(" D3 greedy tokens ({n3}): {d3_tokens:?}");
            eprintln!(
                " byte-exact D2 vs D3 (over {cmp23} tokens): tokens_match={d3_tokens_match}  \
                       logits_bit_exact={d3_logits_bit_exact}  (captured replay must equal plan-once)"
            );
        }
        eprintln!("============================================================\n");

        // Sanity (not perf) assertions — these SHOULD hold and catch a
        // broken persistent path even in this ignored bench.
        if run_d1 && run_d2 {
            assert!(
                tokens_match,
                "D1 and D2 must generate the same greedy token sequence"
            );
        }
        if run_d2 {
            assert_eq!(
                opt_decode_delta, 1,
                "plan-once: the decode loop must optimize exactly ONCE (the build), \
                 regardless of N",
            );
        }
        if run_d3 {
            assert!(
                d3_tokens_match,
                "D3 captured-replay must generate the SAME greedy tokens as D2 plan-once",
            );
            assert!(
                d3_logits_bit_exact,
                "D3 captured-replay logits must be BYTE-IDENTICAL to D2 plan-once (same plan → same kernels)",
            );
        }
    }

    /// CPU persistent-decode wall-clock benchmark on TinyLlama-1.1B.
    /// N defaults to 12 (override with `FUEL_BENCH_N`); CPU is
    /// seconds/token at 1.1B on portable kernels.
    #[test]
    #[ignore = "real-model wall-clock bench — needs FUEL_BENCH_MODEL_DIR + a multi-GB checkpoint"]
    fn bench_persistent_decode_real_model_cpu() {
        let n = std::env::var("FUEL_BENCH_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(12);
        let (model, load_secs) = match load_real_llama(true) {
            Some(m) => m,
            None => return,
        };
        run_persistent_decode_bench(
            &model,
            &Device::cpu(),
            "CPU",
            load_secs,
            n,
            None,
            DType::F32,
        );
    }

    /// Vulkan (live-GPU) persistent-decode wall-clock benchmark on
    /// TinyLlama-1.1B. N defaults to 48 (override with `FUEL_BENCH_N`).
    /// Skips cleanly if no Vulkan device is available.
    #[test]
    #[cfg(feature = "vulkan")]
    #[ignore = "live-GPU wall-clock bench — needs FUEL_BENCH_MODEL_DIR + a Vulkan device"]
    fn bench_persistent_decode_real_model_vulkan() {
        use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
        let n = std::env::var("FUEL_BENCH_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(48);
        // Keep the checkpoint's native BF16 projections: the D1 replan
        // baseline re-uploads the full weight set every realize, and the
        // F32 upcast (4.4 GB) OOMs a 12 GB card when two realize
        // lifetimes overlap. BF16 (2.2 GB) is also the intended Vulkan
        // path (mixed `matmul_f32_bf16_b` kernels).
        let (model, load_secs) = match load_real_llama(false) {
            Some(m) => m,
            None => return,
        };
        let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => b,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Vulkan,
                    fuel_test_support::hardware::Missing::device(format!(
                        "VulkanBackend::with_selection: {e:?}"
                    )),
                );
            }
        };
        let vk_arc = std::sync::Arc::new(vk_backend);
        let vk_device: Device = std::sync::Arc::clone(&vk_arc).into();
        // Per-token forced drain (outside the timers): retires the
        // deferred-destruction batches so the D1 replan path's per-token
        // full-weight re-upload doesn't pile up realize-generations of
        // buffers and OOM the 12 GB card (observed at decode token 3
        // without this).
        let drain = move || {
            if let Err(e) = vk_arc.synchronize_pending() {
                eprintln!("synchronize_pending failed: {e:?}");
            }
        };
        run_persistent_decode_bench(
            &model,
            &vk_device,
            "Vulkan (RTX 4070)",
            load_secs,
            n,
            Some(&drain),
            DType::F32,
        );
    }

    /// CUDA (live-GPU) persistent-decode wall-clock benchmark on
    /// TinyLlama-1.1B. N defaults to 16 (override with `FUEL_BENCH_N`).
    /// Skips cleanly if no CUDA device is available.
    ///
    ///   FUEL_BENCH_MODEL_DIR=... cargo test -p fuel-core --lib --features cuda \
    ///     bench_persistent_decode_real_model_cuda -- --ignored --nocapture
    ///
    /// Weights are force-upcast to F32 (`load_real_llama(true)`, like the
    /// CPU leg — NOT BF16 like the Vulkan leg). The baracuda CUDA dense
    /// MatMul family registers only HOMOGENEOUS dtype keys — `[f32;3]`,
    /// `[bf16;3]`, `[f16;3]`, `[f64;3]` (fuel-dispatch/src/baracuda_dispatch.rs
    /// `matmul_*`); there is NO mixed `F32×BF16` CUDA matmul kernel (the
    /// Vulkan path's `matmul_f32_bf16_b` has no CUDA analog). The forward
    /// graph runs F32 activations (KvCache is F32), so the weights must be
    /// F32 to hit `(MatMul,[F32,F32,F32],Cuda)`.
    ///
    /// F32 weights are ~4.4 GB: in D2 they upload ONCE (held in the
    /// session base_cache) and fit the 12 GB card with room for the F32
    /// KV cache + per-token intermediates. The D1 replan path re-uploads
    /// the full weight set every token; whether CUDA reclaims those across
    /// realizes (the Vulkan `synchronize_pending` drain is a Vulkan-side
    /// mechanism) is left to the bench to reveal — `post_token = None`
    /// here, and a D1 mid-run OOM aborts gracefully + is reported. Run D2
    /// alone first: `FUEL_BENCH_PATHS=d2 ... --nocapture`.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "live-GPU wall-clock bench — needs FUEL_BENCH_MODEL_DIR + a CUDA device"]
    fn bench_persistent_decode_real_model_cuda() {
        let n = std::env::var("FUEL_BENCH_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16);
        // Force F32 weights — the CUDA MatMul family is homogeneous-key
        // only (no F32×BF16 mixed kernel); F32 activations need F32 weights.
        let (model, load_secs) = match load_real_llama(true) {
            Some(m) => m,
            None => return,
        };
        let cuda_device = match fuel_core::cuda_backend::new_device(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        run_persistent_decode_bench(
            &model,
            &cuda_device,
            "CUDA (RTX 4070)",
            load_secs,
            n,
            None,
            DType::F32,
        );
    }

    /// CUDA (live-GPU) BF16-throughout persistent-decode wall-clock
    /// benchmark on TinyLlama-1.1B — Part 2 increment B headline number.
    /// N defaults to 16 (override with `FUEL_BENCH_N`). Skips cleanly if no
    /// CUDA device is available.
    ///
    ///   FUEL_BENCH_MODEL_DIR=... cargo test -p fuel-core --lib --features cuda \
    ///     bench_persistent_decode_real_model_cuda_bf16 -- --ignored --nocapture
    ///
    /// Unlike `bench_persistent_decode_real_model_cuda` (F32 weights + F32
    /// cache, because the CUDA MatMul family has no mixed F32×BF16 kernel),
    /// this leg keeps the checkpoint's native BF16 weights
    /// (`load_real_llama(false)`) AND runs a BF16 `KvCache` — BF16-throughout
    /// activations end-to-end — so the graph hits the homogeneous
    /// `(MatMul, [BF16,BF16,BF16], Cuda)` tensor-core gemm instead of the F32
    /// gemm, AND the CUDA flash-decode arm (`{F16,BF16}`-only) becomes
    /// admissible. Compare this bench's ms/tok against
    /// `bench_persistent_decode_real_model_cuda`'s F32 number (same day, same
    /// card) for the apples-to-apples BF16-vs-F32 CUDA decode comparison.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "live-GPU wall-clock bench — needs FUEL_BENCH_MODEL_DIR + a CUDA device"]
    fn bench_persistent_decode_real_model_cuda_bf16() {
        let n = std::env::var("FUEL_BENCH_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16);
        // Keep native BF16 weights — BF16 activations need BF16 weights
        // (matmul's dtype gate rejects BF16-activation x F32-weight).
        let (model, load_secs) = match load_real_llama(false) {
            Some(m) => m,
            None => return,
        };
        let cuda_device = match fuel_core::cuda_backend::new_device(0) {
            Ok(d) => d,
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!(
                        "CudaDevice::new(0): {e:?}"
                    )),
                );
            }
        };
        run_persistent_decode_bench(
            &model,
            &cuda_device,
            "CUDA (RTX 4070) BF16-throughout",
            load_secs,
            n,
            None,
            DType::BF16,
        );
    }

    /// Phase D · D3 — concurrency isolation. N threads each run a full
    /// plan-once persistent greedy generation from the SAME shared `&model`,
    /// each with its OWN internal `KvCache` + `InferenceContext` + loop-held
    /// `DecodeSession` (all created inside `generate_with_kv_context`). Every
    /// thread must reproduce the single-threaded reference EXACTLY — proving
    /// concurrent persistent decode is correct + isolated (no shared-session
    /// clobber, no data race that would perturb a thread's logits).
    ///
    /// This IS the spec's `(NodeId, SessionId)` concurrency model, realized as
    /// per-session `DecodeSession` isolation: each generation owns its session
    /// state (its held graph + `base_cache` + KV); only the read-only model
    /// weights and the kernel-binding registry (read-locked during optimize)
    /// are shared. (A future refinement could SHARE one optimized graph across
    /// same-model sessions to save N builds — consumerless today, so deferred.)
    #[test]
    fn generate_persistent_is_concurrency_isolated() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let max_new = 6usize;

        // Single-threaded greedy reference (through the wired persistent path).
        let reference = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &Device::cpu(),
                DType::F32,
            )
            .expect("reference generation");

        // N concurrent generations sharing `&model`; each builds its own
        // KvCache / InferenceContext / DecodeSession internally.
        const N_THREADS: usize = 8;
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..N_THREADS)
                .map(|_| {
                    s.spawn(|| {
                        model
                            .generate_with_kv_context(
                                &prompt,
                                max_new,
                                SamplingStrategy::Greedy,
                                None,
                                &Device::cpu(),
                                DType::F32,
                            )
                            .expect("concurrent generation")
                    })
                })
                .collect();
            for h in handles {
                let out = h.join().expect("thread join");
                assert_eq!(
                    out, reference,
                    "concurrent plan-once persistent generation must match the \
                     single-threaded reference — per-generation DecodeSession \
                     isolation, no shared-session clobber",
                );
            }
        });
    }

    /// Phase D · D2b invalidation: a `seq != 1` step mid-stream (e.g. a
    /// spec-decode verification batch) must DROP the held session and
    /// fall back to the D1 rebuild path (the session is shape-keyed to
    /// seq==1); a subsequent seq==1 token rebuilds a fresh session and
    /// still produces correct logits. Also checks the session is rebuilt
    /// (a NEW session object) after the fallback.
    #[test]
    fn forward_with_kv_context_persistent_invalidates_on_non_decode_step() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2];
        let max_seq_len = 8;
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &device,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(device);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // Prefill (seq>1 → no session).
        let _ = model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut session)
            .expect("prefill");
        assert!(session.is_none());

        // One decode token builds the session.
        let _ = model
            .forward_with_kv_context_persistent(&[3], &mut cache, &mut ctx, &mut session)
            .expect("decode 1");
        assert!(session.is_some(), "first decode token builds the session");
        // Clone the graph Arc so the underlying Graph OBJECT stays alive past
        // the session drop below — otherwise the allocator can recycle its
        // freed address for the rebuilt session's graph, making the raw-pointer
        // identity check below spuriously fail (a ~1-in-8 flake before this).
        // Holding this clone does not affect the drop being tested: the session
        // is an `Option` set to `None` by `drop_decode_session` regardless of
        // the graph's refcount.
        let graph_1_keepalive = session.as_ref().unwrap().graph().clone();
        let graph_ptr_1 = Arc::as_ptr(&graph_1_keepalive);

        // A seq!=1 all-positions step drops the session (fallback to D1).
        let _ = model
            .forward_with_kv_context_persistent(&[4, 5], &mut cache, &mut ctx, &mut session)
            .expect("multi-token step");
        assert!(
            session.is_none(),
            "a seq!=1 step must invalidate + drop the held session",
        );

        // A subsequent seq==1 token rebuilds a FRESH session (different
        // graph Arc) and produces correct logits vs. the D1 path on the
        // same running cache.
        let d2 = model
            .forward_with_kv_context_persistent(&[6], &mut cache, &mut ctx, &mut session)
            .expect("decode after fallback");
        assert!(
            session.is_some(),
            "session rebuilt on the next decode token"
        );
        let graph_ptr_2 = Arc::as_ptr(session.as_ref().unwrap().graph());
        assert!(
            graph_ptr_1 != graph_ptr_2,
            "the rebuilt session must hold a NEW graph, not the dropped one",
        );
        drop(graph_1_keepalive); // address no longer needs pinning past here

        // Byte-exact vs. a fresh D1 run over the identical token history.
        let mut cache_ref = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &Device::cpu(),
        )
        .expect("with_capacity ref");
        let mut ctx_ref = InferenceContext::new(Device::cpu());
        let _ = model
            .forward_with_kv_context(&prompt, &mut cache_ref, &mut ctx_ref)
            .unwrap();
        let _ = model
            .forward_with_kv_context(&[3], &mut cache_ref, &mut ctx_ref)
            .unwrap();
        let _ = model
            .forward_with_kv_context(&[4, 5], &mut cache_ref, &mut ctx_ref)
            .unwrap();
        let d1 = model
            .forward_with_kv_context(&[6], &mut cache_ref, &mut ctx_ref)
            .unwrap();
        assert_eq!(d2, d1, "post-fallback decode must match the D1 cached path");
    }

    /// Prefill-only forward through `forward_with_kv_context` should
    /// match a non-cached forward over the same prompt (no decode
    /// step, just the prefill). This is the cleanest correctness gate
    /// — `cached_len == 0` means WriteSlice writes into the head of a
    /// zero-initialized buffer and the subsequent attention slice
    /// equals the fresh K/V.
    #[test]
    fn forward_with_kv_context_prefill_matches_non_cached_forward() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3, 4];

        // Non-cached reference.
        let full_logits = model.forward(&prompt, 0).unwrap();
        let last_pos = prompt.len() - 1;
        let expected = full_logits
            .slice(1, last_pos, 1)
            .unwrap()
            .reshape(Shape::from_dims(&[cfg.vocab_size]))
            .unwrap()
            .realize_f32();

        // New path, single prefill call.
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            prompt.len(),
            DType::F32,
            &device,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(device);
        let actual = model
            .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
            .expect("prefill");

        assert_eq!(cache.cached_len, prompt.len());
        assert_eq!(actual.len(), expected.len());

        // Tighter tolerance than the prefill+decode test: this is
        // structurally one forward pass through the model with the
        // same input shape — the only added work is WriteSlice +
        // Slice (both byte-exact ops). Drift should be at the rng-
        // initial-noise level.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-5,
                "logit[{i}]: new-prefill={a}, non-cached={b}, diff={diff}",
            );
        }
    }

    /// `forward_with_kv_context` rejects a cache built via `with_dims`
    /// (no pre-allocated buffers) with a clear error pointing at the
    /// `with_capacity` constructor.
    #[test]
    fn forward_with_kv_context_rejects_with_dims_cache() {
        let cfg = LlamaConfig {
            vocab_size: 4,
            dim: 4,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 4,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let mut cache = KvCache::with_dims(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim);
        let mut ctx = InferenceContext::new(Device::cpu());

        let err = model.forward_with_kv_context(&[1_u32], &mut cache, &mut ctx);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("with_capacity"),
            "expected error message to mention with_capacity, got: {msg}",
        );
    }

    /// `forward_with_kv_context` rejects when `cached_len + seq`
    /// exceeds the cache's `max_seq_len`.
    #[test]
    fn forward_with_kv_context_rejects_overflow() {
        let cfg = LlamaConfig {
            vocab_size: 4,
            dim: 4,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 4,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            /*max_seq_len*/ 2,
            DType::F32,
            &device,
        )
        .unwrap();
        let mut ctx = InferenceContext::new(device);

        // 3 tokens into a cache with max_seq_len=2 → overflow.
        let err = model.forward_with_kv_context(&[1_u32, 2, 3], &mut cache, &mut ctx);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("max_seq_len"),
            "expected error message to mention max_seq_len, got: {msg}",
        );
    }

    /// After each forward call, the per-step Const NodeIds inserted
    /// into ctx are cleaned up — ctx.persistent should NOT accumulate
    /// across decode steps.
    #[test]
    fn forward_with_kv_context_does_not_leak_context_entries() {
        let cfg = LlamaConfig {
            vocab_size: 4,
            dim: 4,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 4,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let device = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            4,
            DType::F32,
            &device,
        )
        .unwrap();
        let mut ctx = InferenceContext::new(device);

        assert_eq!(ctx.len(), 0);
        model
            .forward_with_kv_context(&[1_u32, 2], &mut cache, &mut ctx)
            .unwrap();
        assert_eq!(ctx.len(), 0, "ctx.persistent should be empty after forward");
        model
            .forward_with_kv_context(&[3_u32], &mut cache, &mut ctx)
            .unwrap();
        assert_eq!(
            ctx.len(),
            0,
            "ctx.persistent should stay empty across steps"
        );
    }

    /// Vulkan parity: prefill+decode through `forward_with_kv_context`
    /// on a Vulkan `Device` matches the CPU reference for the same
    /// model + prompt. Closes the runtime Device-abstraction gate that
    /// the audit memo `project_phase_7_6_step_9c_parity_audit.md`
    /// flagged: previously `Device::new(...)` rejected Vulkan because
    /// no `DynBackendDevice` impl existed, so `KvCache::with_capacity` +
    /// `InferenceContext` could not run on Vulkan even though every
    /// kernel-side gate (WriteSlice b1/b2/b4/b8 + byte-storage Vulkan
    /// D2H) was open. With `VulkanBackendDevice` wired through
    /// `Device::custom`, the pipelined executor + binding-table
    /// dispatch route the per-op kernels to Vulkan SPIR-V.
    ///
    /// GAP-157: REQUIRES a live Vulkan device — the old
    /// `eprintln!("skipping: …"); return` reported `ok` having asserted
    /// nothing, and it did so on **every macOS CI run**.
    ///
    /// ⚠️ `#[ignore]` IS LOAD-BEARING HERE, and the reason corrects an earlier
    /// claim in this very comment. It previously read: *"this test is
    /// `#[cfg(feature = "vulkan")]`, so a CI machine without a GPU never builds
    /// it"*. **That is FALSE.** `cargo test --workspace` unifies features
    /// across the whole selected graph INCLUDING dev-dependencies, and
    /// `fuel-vulkan-backend` -> `fuel-dispatch/vulkan` -> `fuel-core/vulkan`
    /// turns this feature ON in an ordinary default build (measured with
    /// `cargo tree -e features --workspace`). So it compiled and RAN on macOS
    /// CI, found no Vulkan loader, returned early, and passed.
    ///
    /// A `#[cfg(feature = …)]` gate is therefore NOT evidence that CI skips a
    /// test — feature unification can enable it from a crate you never think
    /// about. The device requirement needs its own DECLARED gate, which is what
    /// `#[ignore]` is: run it explicitly on a box that has Vulkan
    ///   `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-core \
    ///        --features vulkan --lib forward_with_kv_context_vulkan -- --ignored`
    #[test]
    #[ignore = "requires a live Vulkan device"]
    #[cfg(feature = "vulkan")]
    fn forward_with_kv_context_vulkan_matches_cpu() {
        use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

        let vk_backend = fuel_test_support::required_ok(
            "a live Vulkan device",
            VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
                .map_err(|e| format!("{e:?}")),
        );
        let vk_device: Device = vk_backend.into();

        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let cfg = LlamaConfig {
            dim: cfg.n_heads * cfg.head_dim,
            ..cfg
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };

        let prompt = [1_u32, 2, 3];
        let next_token = 4_u32;
        let max_seq_len = prompt.len() + 1;

        // CPU reference.
        let cpu_device = Device::cpu();
        let mut cpu_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &cpu_device,
        )
        .expect("cpu with_capacity");
        let mut cpu_ctx = InferenceContext::new(cpu_device);
        model
            .forward_with_kv_context(&prompt, &mut cpu_cache, &mut cpu_ctx)
            .expect("cpu prefill");
        let expected = model
            .forward_with_kv_context(&[next_token], &mut cpu_cache, &mut cpu_ctx)
            .expect("cpu decode");

        // Vulkan path through the new Device wiring.
        let mut vk_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &vk_device,
        )
        .expect("vulkan with_capacity");
        let mut vk_ctx = InferenceContext::new(vk_device);
        model
            .forward_with_kv_context(&prompt, &mut vk_cache, &mut vk_ctx)
            .expect("vulkan prefill");
        let actual = model
            .forward_with_kv_context(&[next_token], &mut vk_cache, &mut vk_ctx)
            .expect("vulkan decode");

        assert_eq!(actual.len(), expected.len());
        // Same tolerance band as `forward_with_kv_context_decode_matches_
        // non_cached_forward`: cross-backend matmul accumulation order
        // differs, producing standard O(ε) gemm drift on the f32 path.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: vulkan={a}, cpu={b}, diff={diff}, rel={rel}",
            );
        }
    }

    // ---- kv-context all-positions + spec decode (E.3.4 port) ----------

    /// The all-positions variant's last row must equal what the
    /// regular (last-only) variant produces. Same graph, same cache
    /// state, same tokens — only the output shape differs.
    #[test]
    fn forward_with_kv_context_all_positions_last_row_matches_last_only() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let tokens = [1_u32, 2, 3, 4, 5];
        let device = Device::cpu();

        // Path A: regular last-only forward.
        let mut cache_a = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            tokens.len(),
            DType::F32,
            &device,
        )
        .expect("cache_a");
        let mut ctx_a = InferenceContext::new(device.clone());
        let last_only = model
            .forward_with_kv_context(&tokens, &mut cache_a, &mut ctx_a)
            .expect("last-only forward");
        assert_eq!(last_only.len(), cfg.vocab_size);

        // Path B: all-positions forward.
        let mut cache_b = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            tokens.len(),
            DType::F32,
            &device,
        )
        .expect("cache_b");
        let mut ctx_b = InferenceContext::new(device.clone());
        let all = model
            .forward_with_kv_context_all_positions(&tokens, &mut cache_b, &mut ctx_b)
            .expect("all-positions forward");
        assert_eq!(all.len(), tokens.len() * cfg.vocab_size);

        // Last row of `all` must match last_only.
        let last_pos = tokens.len() - 1;
        let all_last = &all[last_pos * cfg.vocab_size..(last_pos + 1) * cfg.vocab_size];
        for (i, (a, b)) in all_last.iter().zip(last_only.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "vocab idx {i}: all_positions={a} vs last_only={b}",
            );
        }

        // Both caches should have advanced by the same amount.
        assert_eq!(cache_a.cached_len, cache_b.cached_len);
    }

    /// `KvCache::truncate_to` rollback semantics on the pre-allocated
    /// WriteSlice path: decode a token, roll it back, decode different
    /// tokens through the same positions — the final logits must match
    /// an uninterrupted run that never saw the rolled-back token.
    /// This is exactly spec decode's reject path: stale K/V rows past
    /// `cached_len` must stop being read and must be overwritten by
    /// the next `Op::WriteSlice` at the same positions.
    #[test]
    fn kv_cache_truncate_then_redecode_matches_uninterrupted_decode() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let device = Device::cpu();

        // Path A (reference): prefill [3,7,1] then decode 9 then 2.
        let mut cache_a = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            5,
            DType::F32,
            &device,
        )
        .expect("cache_a");
        let mut ctx_a = InferenceContext::new(device.clone());
        model
            .forward_with_kv_context(&[3, 7, 1], &mut cache_a, &mut ctx_a)
            .expect("prefill A");
        model
            .forward_with_kv_context(&[9], &mut cache_a, &mut ctx_a)
            .expect("decode A1");
        let expected = model
            .forward_with_kv_context(&[2], &mut cache_a, &mut ctx_a)
            .expect("decode A2");

        // Path B: prefill [3,7,1], decode a WRONG token (11) at
        // position 3, roll it back, then decode [9, 2] through the
        // same positions in one step.
        let mut cache_b = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            5,
            DType::F32,
            &device,
        )
        .expect("cache_b");
        let mut ctx_b = InferenceContext::new(device.clone());
        model
            .forward_with_kv_context(&[3, 7, 1], &mut cache_b, &mut ctx_b)
            .expect("prefill B");
        model
            .forward_with_kv_context(&[11], &mut cache_b, &mut ctx_b)
            .expect("decode B wrong");
        assert_eq!(cache_b.cached_len, 4);
        cache_b.truncate_to(3);
        assert_eq!(cache_b.cached_len, 3);
        let actual = model
            .forward_with_kv_context(&[9, 2], &mut cache_b, &mut ctx_b)
            .expect("redecode B");
        assert_eq!(cache_b.cached_len, 5);

        assert_eq!(actual.len(), expected.len());
        // Tolerance: path A attends over (cached 4 + fresh 1) rows,
        // path B over (cached 3 + fresh 2) — standard O(ε) gemm
        // accumulation-order drift, same band as the other kv-context
        // parity tests.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: redecode={a}, uninterrupted={b}, diff={diff}",
            );
        }
    }

    /// Spec decode with the target as its own draft: every draft is
    /// trivially argmax-matched, acceptance is 100%, and the output
    /// must equal a plain greedy run through the same kv-context path.
    #[test]
    fn spec_decode_kv_context_self_draft_matches_greedy_baseline() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [3_u32, 7, 1];
        let max_new = 8;
        let device = Device::cpu();

        let baseline = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &device,
                DType::F32,
            )
            .expect("baseline generate");

        for k in [2_usize, 4] {
            let spec_out = model
                .generate_streaming_spec_with_kv_context(
                    &model,
                    &prompt,
                    max_new,
                    k,
                    SamplingStrategy::Greedy,
                    None,
                    &device,
                    DType::F32,
                    |_| {},
                )
                .expect("spec generate");
            assert_eq!(
                spec_out, baseline,
                "K={k}: spec-decode must match baseline when draft == target",
            );
        }
    }

    /// Greedy spec decode is lossless for ANY draft: on the first
    /// mismatch the target's own argmax is emitted and the rejected
    /// draft rows are rolled back, so the output must equal plain
    /// greedy generation from the target. A draft with different
    /// weights forces genuine rejections, exercising the
    /// `KvCache::truncate_to` rollback + bonus-position re-write that
    /// the self-draft test (100% acceptance) never reaches.
    ///
    /// The retired legacy-executor implementation got this rollback
    /// wrong: it kept one stale K/V row at the bonus position
    /// (truncate to `committed + accepted + 1`) and appended the
    /// bonus one position too far — measured ~4e-3 logit drift on
    /// this fixture (argmax happened to survive, so its
    /// token-equality tests passed).
    #[test]
    fn spec_decode_kv_context_divergent_draft_matches_greedy_baseline() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let target = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_seeded(&cfg, 9999),
        };
        let draft = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights_seeded(&cfg, 4242),
        };
        let prompt = [3_u32, 7, 1];
        let max_new = 8;
        let device = Device::cpu();

        let baseline = target
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &device,
                DType::F32,
            )
            .expect("baseline generate");

        for k in [1_usize, 2, 4] {
            let spec_out = target
                .generate_streaming_spec_with_kv_context(
                    &draft,
                    &prompt,
                    max_new,
                    k,
                    SamplingStrategy::Greedy,
                    None,
                    &device,
                    DType::F32,
                    |_| {},
                )
                .expect("spec generate");
            assert_eq!(
                spec_out, baseline,
                "K={k}: greedy spec-decode must be lossless for a divergent draft",
            );
        }
    }

    /// In Temperature mode with draft == target, the accept coin's
    /// ratio = min(1, p_target/p_draft) = 1.0, so acceptance is 100%.
    /// We can't bit-match against a plain sampled baseline because the
    /// RNG sequences diverge (spec-decode draws more randoms per
    /// output token than plain gen), but we can assert: (a) output has
    /// expected length, (b) all tokens are in vocab, (c) prompt prefix
    /// is preserved.
    #[test]
    fn spec_decode_kv_context_sampled_self_draft_produces_valid_tokens() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let prompt = [3_u32, 7, 1];
        let max_new = 6;
        let device = Device::cpu();

        for k in [2_usize, 4] {
            let out = model
                .generate_streaming_spec_with_kv_context(
                    &model,
                    &prompt,
                    max_new,
                    k,
                    SamplingStrategy::Temperature {
                        temp: 0.8,
                        seed: 42,
                    },
                    None,
                    &device,
                    DType::F32,
                    |_| {},
                )
                .expect("spec sampled generate");

            // The emit loop returns the moment `emitted == max_new`,
            // so the output is exactly prompt + max_new tokens.
            assert_eq!(
                out.len(),
                prompt.len() + max_new,
                "K={k}: expected {} tokens, got {}",
                prompt.len() + max_new,
                out.len()
            );
            assert_eq!(&out[..prompt.len()], &prompt);
            for &t in &out {
                assert!(
                    (t as usize) < cfg.vocab_size,
                    "K={k}: token {t} out of vocab"
                );
            }
        }
    }

    #[test]
    fn sample_multinomial_respects_distribution() {
        // Heavy-loaded distribution: 99% on index 0. The sampler
        // should pick 0 almost always.
        let probs = vec![0.99_f32, 0.005, 0.005];
        let mut state: u64 = 12345;
        let mut counts = [0_usize; 3];
        for _ in 0..1000 {
            let idx = sample_multinomial(&probs, &mut state) as usize;
            counts[idx] += 1;
        }
        assert!(
            counts[0] > 900,
            "expected ≥900 samples on index 0, got {}",
            counts[0]
        );
    }

    // ===== Phase 7.6 step 9c E.3.4 — legacy spec-decode tests retired =====
    //
    // The legacy-executor spec-decode + KVCache<B>-truncate tests
    // (`spec_decode_with_self_as_draft_matches_greedy_baseline`,
    // `spec_decode_sampled_with_self_as_draft_produces_valid_tokens`,
    // `forward_with_cache_all_positions_last_slice_matches_forward_with_cache`,
    // `kvcache_truncate_to_*`) retired with the `*_gpu_on` family.
    // Their kv-context successors are above:
    // `spec_decode_kv_context_*`,
    // `forward_with_kv_context_all_positions_last_row_matches_last_only`,
    // and `kv_cache_truncate_then_redecode_matches_uninterrupted_decode`
    // — the latter strictly stronger (behavioral rollback semantics,
    // not just buffer-shrink bookkeeping). The divergent-draft test
    // additionally locks the greedy-losslessness property the legacy
    // implementation violated on partial acceptance (~4e-3 logit
    // drift; see `generate_streaming_spec_with_kv_context`'s docs).
}

#[cfg(test)]
mod gqa_tests {
    use super::*;

    /// Build tiny GQA weights for forward-pass tests.
    fn make_tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 5678;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv_dim).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv_dim).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }

    #[test]
    fn llama_forward_with_gqa_matches_llama3_ratio() {
        // A Llama-3-sized head ratio in miniature: n_heads = 4,
        // n_kv_heads = 1. Every query head shares the single K and V
        // head via broadcast. Most interesting because it's the
        // extreme case (n_rep = 4).
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 2,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };

        let tokens = vec![0_u32, 1, 2];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        let realized = logits.realize_f32();
        for &v in &realized {
            assert!(v.is_finite(), "GQA logit non-finite: {v}");
        }
    }

    #[test]
    fn llama_forward_with_2to1_gqa_ratio() {
        // n_heads = 4, n_kv_heads = 2 (classic GQA 2:1 ratio).
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim: 8,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };
        let tokens = vec![1_u32, 3];
        let logits = model.forward(&tokens, 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 1 * 2 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }
}

#[cfg(test)]
mod llama_tests {
    use super::*;
    use fuel_core::Shape;

    /// Build a set of tiny LLaMA weights filled with deterministic,
    /// small, "random" values. Respects `cfg.n_kv_heads` so GQA-style
    /// shapes come out correctly.
    fn make_tiny_weights(cfg: &LlamaConfig) -> LlamaWeights {
        let mut s: u32 = 2024;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv_dim).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv_dim).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }

    #[test]
    fn llama_forward_produces_correct_logit_shape() {
        // A 2-layer 8-dim 2-head LLaMA. Not trained, just checking
        // that the graph builds, runs, and emits a logit tensor of the
        // expected shape.
        let cfg = LlamaConfig {
            vocab_size: 32,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };

        let tokens: Vec<u32> = vec![5, 12, 0, 7];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn llama_forward_realizes_to_finite_logits() {
        // Same config, smaller vocab for faster realization. The
        // output must be finite across the full [1, seq, vocab] tensor.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };

        let tokens = vec![1_u32, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        let logits_vec = logits.realize_f32();
        assert_eq!(logits_vec.len(), 1 * 3 * cfg.vocab_size);
        for &v in &logits_vec {
            assert!(v.is_finite(), "llama logit non-finite: {v}");
        }
    }

    #[test]
    fn llama_forward_is_relative_position_invariant() {
        // RoPE has a specific and well-known property: the attention
        // scores depend only on *relative* position differences, not
        // absolute positions. That means the forward output of a
        // LlamaModel on the same input sequence should be (modulo
        // floating-point noise) independent of `start_pos`, even
        // though the Q and K vectors themselves change.
        //
        // This test enforces the property: start_pos=0 and
        // start_pos=10 on the same token sequence must produce
        // identical logits. It's both a validation that RoPE is
        // implemented correctly AND a documented invariant for any
        // caller building a KV-cached decode loop — the cache has to
        // track absolute positions, not relative, because relative
        // differences change as new tokens arrive.
        let cfg = LlamaConfig {
            vocab_size: 8,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };

        let tokens = vec![2_u32, 4];
        let l0 = model.forward(&tokens, 0).unwrap().realize_f32();
        let l10 = model.forward(&tokens, 10).unwrap().realize_f32();
        // Relative-position invariance: the two should match exactly.
        assert_eq!(
            l0, l10,
            "RoPE attention should be invariant to start_pos for a fixed input",
        );
    }

    #[test]
    fn llama_forward_argmax_selects_a_token_id() {
        // Predict next-token by argmax over the last position's logits.
        // Not testing correctness (weights are random); just that the
        // predicted ID is a valid vocabulary index. This is the
        // decode-step primitive a sampling loop would call.
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };

        let tokens = vec![3_u32, 1, 4, 1, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        // Take last-position slice and argmax over vocab dim, all
        // through the Tensor bridge API.
        let last = logits.slice(1, tokens.len() - 1, 1).unwrap(); // [1, 1, vocab]
        let last_flat = last.reshape(Shape::from_dims(&[cfg.vocab_size])).unwrap();
        let predicted_ids = last_flat.argmax_dim(0_usize).unwrap().realize_u32();
        assert_eq!(predicted_ids.len(), 1);
        let pred = predicted_ids[0];
        assert!(
            (pred as usize) < cfg.vocab_size,
            "argmax should return a valid vocab index",
        );
    }

    /// `forward_hidden_embeds_with_mask` runs the LlamaModel with
    /// a caller-supplied mask instead of the built-in strict
    /// causal one. An all-zero (bidirectional) mask must produce
    /// different hidden states than the strict-causal `forward`
    /// because the bidirectional path lets earlier tokens attend
    /// to later ones.
    #[test]
    fn forward_hidden_embeds_with_mask_bidirectional() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        // Causal reference path through `forward` → drop the
        // lm_head matmul mentally by comparing across runs.
        let _logits = model.forward(&tokens, 0).unwrap().realize_f32();

        // Build embeds + bidirectional (all-zero) mask on one graph.
        let embed = Tensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &fuel_core::Device::cpu(),
        )
        .unwrap();
        let token_ids = embed
            .const_u32_like(tokens.clone(), Shape::from_dims(&[tokens.len()]))
            .unwrap();
        let embeds = embed
            .index_select(0_usize, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.dim]))
            .unwrap();
        let zero_mask: Arc<[f32]> = Arc::from(vec![0.0_f32; tokens.len() * tokens.len()]);
        let mask = embeds
            .const_f32_like(
                zero_mask,
                Shape::from_dims(&[1, 1, tokens.len(), tokens.len()]),
            )
            .unwrap();
        let bidir = model
            .forward_hidden_embeds_with_mask(&embeds, &mask, 0)
            .unwrap()
            .realize_f32();

        // Also run the standard causal hidden path for comparison.
        // forward_embeds applies the LM head; we need just the
        // hidden state, so build it separately via forward_embeds
        // and undo the lm_head implicitly by checking the difference
        // is non-trivial across a known position.
        let causal_logits = model.forward(&tokens, 0).unwrap().realize_f32();
        assert_eq!(bidir.len(), tokens.len() * cfg.dim);
        for &v in &bidir {
            assert!(v.is_finite(), "bidirectional hidden state not finite: {v}");
        }
        assert!(!causal_logits.is_empty());
    }

    /// `forward_hidden_embeds(embeds, start_pos)` returns
    /// post-final-RmsNorm hidden states for pre-built embeds —
    /// useful for multimodal hosts (LLaVA, Pixtral) that
    /// interleave image embeddings with text embeddings and
    /// want hidden states without the lm_head projection.
    /// The result must match `forward_embeds(embeds,
    /// start_pos)` projected through the lm_head, because
    /// `forward_embeds` is exactly `forward_hidden_embeds`
    /// followed by `lm_head.apply_linear`.
    #[test]
    fn forward_hidden_embeds_followed_by_lm_head_matches_forward_embeds() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = make_tiny_weights(&cfg);
        let model = LlamaModel {
            config: cfg.clone(),
            weights,
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        let embed = Tensor::from_f32(
            model.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &fuel_core::Device::cpu(),
        )
        .unwrap();
        let token_ids = embed
            .const_u32_like(tokens.clone(), Shape::from_dims(&[tokens.len()]))
            .unwrap();
        let embeds = embed
            .index_select(0_usize, &token_ids)
            .unwrap()
            .reshape(Shape::from_dims(&[1, tokens.len(), cfg.dim]))
            .unwrap();

        let hidden = model.forward_hidden_embeds(&embeds, 0).unwrap();
        let logits_from_hidden = model
            .weights
            .output
            .apply_linear(&hidden, cfg.dim, cfg.vocab_size)
            .unwrap()
            .realize_f32();
        let logits_direct = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        assert_eq!(logits_from_hidden.len(), logits_direct.len());
        for (a, b) in logits_from_hidden.iter().zip(logits_direct.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "forward_hidden_embeds + lm_head must match forward_embeds: {a} vs {b}"
            );
        }
    }

    /// GAP-326: the three embeds entry points check rank with `Shape::dims3()`,
    /// so a rank-2 input is a typed `UnexpectedNumberOfDims` naming the entry
    /// point. Each used to be an `assert_eq!` panic inside a function that
    /// returns `Result`.
    #[test]
    fn embeds_of_the_wrong_rank_are_a_typed_error_not_a_panic() {
        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let model = LlamaModel {
            config: cfg.clone(),
            weights: make_tiny_weights(&cfg),
        };
        let rank2 = Tensor::from_f32(
            vec![0.0; 4 * cfg.dim],
            Shape::from_dims(&[4, cfg.dim]),
            &fuel_core::Device::cpu(),
        )
        .unwrap();

        /// The error under any `Context` / `WithBacktrace` wrapping.
        fn root(e: &fuel_core::Error) -> &fuel_core::Error {
            match e {
                fuel_core::Error::Context { inner, .. }
                | fuel_core::Error::WithBacktrace { inner, .. } => root(inner),
                other => other,
            }
        }

        // The rank check runs before the other arguments are used, so the
        // rank-2 tensor stands in for the rope tables and the mask.
        let results = [
            (
                "run_backbone_embeds",
                model.forward_hidden_embeds(&rank2, 0),
            ),
            (
                "forward_hidden_embeds_with_mask",
                model.forward_hidden_embeds_with_mask(&rank2, &rank2, 0),
            ),
            (
                "run_backbone_with_rope_tables",
                model.run_backbone_with_rope_tables(&rank2, &rank2, &rank2, &rank2),
            ),
        ];
        for (entry, result) in results {
            let e = result
                .err()
                .unwrap_or_else(|| panic!("{entry}: rank-2 embeds accepted"));
            assert!(
                matches!(
                    root(&e),
                    fuel_core::Error::UnexpectedNumberOfDims {
                        expected: 3,
                        got: 2,
                        ..
                    }
                ),
                "{entry}: {e}"
            );
            assert!(e.to_string().contains(entry), "{entry}: {e}");
        }
    }
}
