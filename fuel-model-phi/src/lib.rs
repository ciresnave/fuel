// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Phi-family lazy-graph decoder (`PhiConfig`/`PhiWeights`/`PhiModel`).
//!
//! Extracted from `fuel-core/src/lazy.rs` (fuel-core dissolution, per
//! `docs/architecture/02-layers.md`'s ratified `fuel-model-phi`† — one architecture per
//! crate). `fuel-core` keeps no shim for this type (see `fuel-model-llama`'s doc comment for
//! why: `fuel-model-phi` depends on `fuel-core`, so a re-export back would be a build cycle).
//! Every real consumer was repointed to `fuel_model_phi::` directly.
//!
//! ⚠️ Same stepping-stone status as `fuel-model-llama`, including the same generic-utility
//! exclusion: this crate does not own `WeightStorage`, `load_tensor_as_f32`,
//! `load_transposed_matrix*`, or `apply_affine_rms_norm`, all textually near `PhiModel` in
//! the original file but used by dozens of unrelated models and left in `fuel-core`.

use fuel_core::inference_context::{InferenceContext, KvCache, KvSlot};
use fuel_core::lazy::{
    SamplingStrategy, Tensor, TokenDataHost, WeightStorage, build_decode_causal_mask,
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype, refresh_decode_session,
    sample_logits,
};
#[cfg(feature = "cuda")]
use fuel_core::lazy::{TokenDataBytes, captured_output_to_f32};
use fuel_core::{DType, Device, Shape};
use serde::Deserialize;
use std::sync::Arc;

impl fuel_core::persistent_decode::PersistentDecodeModel for PhiModel {
    fn decode_n_layers(&self) -> usize {
        self.config.n_layers
    }

    /// **Measured: Phi takes the SymEnv path on CPU/F32** (`offset: None`), so
    /// there is no device-offset operand and its builder takes no dtype/offset
    /// arguments. `rope_inv_freq` is ignored: Phi has no LLaMA-3-style RoPE
    /// scaling override, and accepting-then-dropping it here is what lets one
    /// driver serve both without the driver knowing which model it has.
    fn build_decode_token_data(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        session: &fuel_core::inference_context::DecodeSession,
        _cache: &KvCache,
        _rope_inv_freq: Option<&[f64]>,
    ) -> fuel_core::Result<fuel_core::inference_context::DecodeTokenData> {
        self.build_token_rope_mask_arcs(device, cached_len, tokens, session.max_seq_len())
    }
}
// ---- Phi-2 model assembly ---------------------------------------------------
//
// Phi-2 (microsoft/phi-2, 2.7B params) differs from LLaMA in four
// meaningful ways, each of which exercises a different code path:
//
//   1. Norm: LayerNorm with gain + bias (not RMSNorm with gain only)
//   2. MLP: standard fc1 → GELU → fc2 (not SwiGLU's gate ⊗ up → down)
//   3. Residual structure: parallel attention + MLP — both branches
//      consume the same pre-block-norm input and are summed with x:
//        h' = x + attn(LN(x)) + mlp(LN(x))
//      compared to LLaMA's sequential:
//        h1 = x + attn(LN1(x))
//        h2 = h1 + mlp(LN2(h1))
//   4. Partial RoPE: only the first `rotary_dim` entries of each head
//      get rotated (rotary_dim=32 for head_dim=80 in Phi-2). The rest
//      pass through unchanged. We slice → rope → concat.
//
// Phi-2 also has biases on Q/K/V/dense and on fc1/fc2, plus a bias on
// the LayerNorm. Every one of those is a real `broadcast_add` in the
// graph, which exercises the lazy broadcast path we built for the
// stride-aware binary work.

/// Phi-2 model hyperparameters. Field semantics match the LLaMA config
/// where they overlap; the `layer_norm_eps`, `partial_rotary_factor`,
/// and `rotary_dim` fields are Phi-specific.
#[derive(Debug, Clone, PartialEq)]
pub struct PhiConfig {
    pub vocab_size: usize,
    pub dim: usize, // hidden_size
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize, // intermediate_size
    pub layer_norm_eps: f64,
    pub rope_base: f64,
    pub partial_rotary_factor: f64,
    /// Number of dims at the start of head_dim that get rotated.
    /// `rotary_dim = (partial_rotary_factor * head_dim).round() as usize`.
    /// Must be even for the half-split RoPE layout.
    pub rotary_dim: usize,
    pub tie_word_embeddings: bool,
}

fn default_phi_layer_norm_eps() -> f64 {
    1e-5
}
fn default_phi_rope_base() -> f64 {
    10_000.0
}
fn default_partial_rotary_factor() -> f64 {
    0.4
}

/// A Phi `config.json` under HuggingFace's field names.
///
/// ⚠️ THIS IS THE ONE CONFIG WHOSE RESOLUTION IS A CHAIN RATHER THAN A MAP,
/// which is why it was converted last: the shape had to be settled by the
/// other six before the constraint could be honoured rather than discovered.
///
/// ```text
///   head_dim   <- "head_dim" if present, else dim / n_heads
///   rotary_dim <- round(partial_rotary_factor * head_dim)     depends on ^
///   then rotary_dim must be even
/// ```
///
/// `rotary_dim` is derived from the RESOLVED `head_dim`, not from the raw
/// field, so `resolve` computes `head_dim` into a binding first and uses that
/// binding. Writing the two as sibling entries in a struct literal would
/// happen to work — Rust evaluates fields in source order — but it would
/// encode the ordering as an accident of layout rather than as a dependency,
/// and a later field reshuffle would silently change the result.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PhiConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_phi_layer_norm_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_phi_rope_base", rename = "rope_theta")]
    rope_base: f64,
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

impl PhiConfigRaw {
    fn from_json_str(json: &str) -> fuel_core::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| fuel_core::Error::Msg(format!("parsing config.json: {e}")))
    }

    fn resolve(self) -> fuel_core::Result<PhiConfig> {
        // ORDERED, not a flat map: rotary_dim reads the resolved head_dim.
        let head_dim = fuel_core::hf_config::head_dim(
            self.head_dim,
            self.hidden_size,
            self.num_attention_heads,
        )?;
        let rotary_dim = (self.partial_rotary_factor * head_dim as f64).round() as usize;
        if !rotary_dim.is_multiple_of(2) {
            fuel_core::bail!(
                "PhiConfig: rotary_dim {rotary_dim} must be even (partial_rotary_factor={}, head_dim={head_dim})",
                self.partial_rotary_factor
            );
        }
        Ok(PhiConfig {
            vocab_size: self.vocab_size,
            dim: self.hidden_size,
            n_layers: self.num_hidden_layers,
            n_heads: self.num_attention_heads,
            head_dim,
            ffn_dim: self.intermediate_size,
            layer_norm_eps: self.layer_norm_eps,
            rope_base: self.rope_base,
            partial_rotary_factor: self.partial_rotary_factor,
            rotary_dim,
            tie_word_embeddings: self.tie_word_embeddings,
        })
    }
}

impl PhiConfig {
    ///
    /// `PhiConfigRaw` is the wire shape; `PhiConfigRaw::resolve` applies
    /// the CHAINED derivation and the evenness check.
    pub fn from_hf_json_str(json: &str) -> fuel_core::Result<Self> {
        PhiConfigRaw::from_json_str(json)?.resolve()
    }
}

/// How Q/K/V projections are stored for a Phi layer.
///
/// - `Split`: separate Q, K, V weights + biases (matches HF safetensors
///   layout — `q_proj.weight`, `k_proj.weight`, `v_proj.weight`).
/// - `Packed`: single `[3*dim, dim]` weight + `[3*dim]` bias (matches
///   llama.cpp GGUF layout — `attn_qkv.weight`). The forward pass does
///   one big matmul producing `[*, 3*dim]`, then slices that output
///   into Q, K, V. Critically, the slice happens on the OUTPUT side
///   rather than up-front on the weights — this matches Candle's
///   `qkv.reshape(3, n_head, head_dim).i((.., .., 0..3))` exactly and
///   avoids any potential byte-split-order hazards on the weight side.
#[derive(Debug, Clone)]
pub enum PhiQkv {
    Split {
        q: WeightStorage,
        q_bias: Arc<[f32]>,
        k: WeightStorage,
        k_bias: Arc<[f32]>,
        v: WeightStorage,
        v_bias: Arc<[f32]>,
    },
    Packed {
        /// `[3*dim, dim]` weight (GGUF layout).
        qkv: WeightStorage,
        /// `[3*dim]` bias, Q first then K then V (standard Candle convention).
        qkv_bias: Arc<[f32]>,
    },
}

/// Per-layer Phi-2 weights. Every projection has a bias (unlike LLaMA).
#[derive(Debug, Clone)]
pub struct PhiLayerWeights {
    pub attn_qkv: PhiQkv,
    /// Output projection (called "dense" in Phi-2, not "o_proj").
    pub attn_dense: WeightStorage,
    pub attn_dense_bias: Arc<[f32]>,
    pub mlp_fc1: WeightStorage, // [dim, ffn_dim]
    pub mlp_fc1_bias: Arc<[f32]>,
    pub mlp_fc2: WeightStorage, // [ffn_dim, dim]
    pub mlp_fc2_bias: Arc<[f32]>,
    /// Pre-block LayerNorm (single norm for Phi-2's parallel attn+MLP).
    pub norm_gain: Arc<[f32]>,
    pub norm_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PhiWeights {
    /// See `LlamaWeights::instance`.
    pub instance: fuel_core::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>, // [vocab_size, dim]
    pub layers: Vec<PhiLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub final_norm_bias: Arc<[f32]>,
    pub output: WeightStorage, // [dim, vocab_size]
    pub output_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct PhiModel {
    pub config: PhiConfig,
    pub weights: PhiWeights,
}

impl PhiModel {
    /// See `LlamaModel::decode_shape_key`.
    pub fn decode_shape_key(&self) -> u64 {
        let mut h = fuel_core::decode_shape::ShapeKeyHasher::new();
        h.mix_str("phi")
            .mix_instance(self.weights.instance)
            .mix_u64(self.config.n_layers as u64)
            .mix_u64(self.config.n_heads as u64)
            .mix_u64(self.config.head_dim as u64)
            .mix_u64(self.config.dim as u64)
            .mix_u64(self.config.vocab_size as u64);
        h.finish()
    }

    // ===== Phase 7.6 step 9c E.3.3/E.3.4 — KvCache + InferenceContext =====
    //
    // The pipelined-executor forward/generate family, mirroring
    // `LlamaModel::forward_with_kv_context`. Pre-allocated KV buffers
    // (`KvCache::with_capacity`) + `Op::WriteSlice` in-graph mutation;
    // runs on CPU, CUDA, and Vulkan via binding-table dispatch. Phi-2
    // has no GQA, so the cache's `n_kv_heads` slot carries `n_heads`.

    /// Variant of [`Self::apply_layer_with_cache`] that uses
    /// pre-allocated KV-cache buffers + `Op::WriteSlice`. The K/V
    /// caches are bound via `k_cache_const` / `v_cache_const` (Const
    /// placeholders the caller has wired into [`InferenceContext`]).
    ///
    /// **Phase D · D4 (input-independent decode graph — the Phi mirror
    /// of the LlamaModel D1/D2b transform):** the KV write lands at the
    /// runtime offset `cached_len` via `write_slice_dyn`
    /// (`DynScalar::Sym(cached_len_sym)`, resolved through the per-pass
    /// `SymEnv` at realize), and attention reads the **full fixed-capacity**
    /// buffers `[batch, n_heads, max_seq_len, head_dim]` with a fixed
    /// `[1, 1, seq, max_seq_len]` causal `mask` (`k > cached_len + q` masks
    /// future positions AND the zero-init stale tail). Nothing in the
    /// graph's *shape* or *structure* depends on `cached_len`, so the
    /// decode-step graph is byte-identical across tokens — the prerequisite
    /// for plan-once persistent decode. Numerically identical to the prior
    /// `slice(0..total_seq)` form (masked positions contribute 0).
    ///
    /// Phi specifics preserved from the sliced form: parallel attention +
    /// MLP over a SHARED pre-block LayerNorm, bias on every projection,
    /// partial RoPE (only the first `rotary_dim` head entries rotate),
    /// no GQA (`kv_dim == n_heads * head_dim`), and the parallel residual
    /// `x + attn_out + mlp_out`. The `mask` is hoisted to ONE shared Const
    /// built in the forward (was one Const per layer) — byte-exact (it
    /// depends only on `cached_len` / `seq` / `max_seq_len`), and it cuts
    /// the per-token data-Const re-bind on the persistent path to 1.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_kv_writes(
        &self,
        x: &Tensor,
        layer: &PhiLayerWeights,
        k_cache_const: &Tensor,
        v_cache_const: &Tensor,
        cached_len_sym: fuel_ir::SymId,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
    ) -> fuel_core::Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_heads * cfg.head_dim; // no GQA in Phi-2

        // Shared pre-block LayerNorm.
        let x_norm = x.layer_norm_affine(
            Arc::clone(&layer.norm_gain),
            Arc::clone(&layer.norm_bias),
            cfg.layer_norm_eps,
        )?;

        // Q/K/V projections with bias — identical to apply_layer_with_cache.
        let (q, k, v) = match &layer.attn_qkv {
            PhiQkv::Split {
                q,
                q_bias,
                k,
                k_bias,
                v,
                v_bias,
            } => {
                let q_out =
                    q.apply_linear_with_bias(&x_norm, cfg.dim, cfg.dim, Arc::clone(q_bias))?;
                let k_out =
                    k.apply_linear_with_bias(&x_norm, cfg.dim, kv_dim, Arc::clone(k_bias))?;
                let v_out =
                    v.apply_linear_with_bias(&x_norm, cfg.dim, kv_dim, Arc::clone(v_bias))?;
                (q_out, k_out, v_out)
            }
            PhiQkv::Packed { qkv, qkv_bias } => {
                let combined = qkv.apply_linear_with_bias(
                    &x_norm,
                    cfg.dim,
                    3 * cfg.dim,
                    Arc::clone(qkv_bias),
                )?;
                let last = combined.rank() - 1;
                let q_out = combined.slice(last, 0, cfg.dim)?;
                let k_out = combined.slice(last, cfg.dim, cfg.dim)?;
                let v_out = combined.slice(last, 2 * cfg.dim, cfg.dim)?;
                (q_out, k_out, v_out)
            }
        };

        // Split heads → [batch, n_heads, seq, head_dim].
        let q_h = q
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_heads, cfg.head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        // Partial RoPE on Q and K (first `rotary_dim` entries rotate).
        let q_r = partial_rope(&q_h, rope_cos, rope_sin, cfg.rotary_dim, cfg.head_dim);
        let k_r = partial_rope(&k_h, rope_cos, rope_sin, cfg.rotary_dim, cfg.head_dim);

        // Write fresh K/V into the pre-allocated cache buffers via
        // Op::WriteSlice at the RUNTIME offset `cached_len`. On axis 2
        // the start is dynamic (`cached_len_sym`, resolved at realize)
        // and the slab width is `seq`. The returned tensor's Storage Arc
        // IS the cache const's Arc — post-write reference to the same
        // buffer (the executor adopts dest's Arc as the kernel output,
        // mutating in place). Keeping the offset symbolic makes the write
        // node structurally identical across tokens.
        let write_ranges = vec![
            (0, batch),
            (0, cfg.n_heads),
            (0, seq), // axis-2 start is dynamic; width = seq
            (0, cfg.head_dim),
        ];
        let dyn_off = fuel_ir::DynScalar::Sym(cached_len_sym);
        let full_k = k_cache_const.write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?;
        let full_v = v_cache_const.write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?;

        // Attend over the FULL fixed-capacity buffers (no slice to
        // `total_seq`) so the attention shape is `max_seq_len` every
        // token. The fixed-capacity causal mask (built once in the
        // forward, shared across layers) excludes future positions AND
        // the stale/unwritten tail.
        let k_t = full_k.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = Tensor::from_graph_tensor(scores.graph_tensor().mul_scalar(scale));
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&full_v)?;

        // Merge heads: [batch, n_heads, seq, head_dim] → [batch, seq, dim].
        let merged = attn_v
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;
        let attn_out = layer.attn_dense.apply_linear_with_bias(
            &merged,
            cfg.dim,
            cfg.dim,
            Arc::clone(&layer.attn_dense_bias),
        )?;

        // MLP branch (shares x_norm with the attention branch).
        let fc1_out = layer.mlp_fc1.apply_linear_with_bias(
            &x_norm,
            cfg.dim,
            cfg.ffn_dim,
            Arc::clone(&layer.mlp_fc1_bias),
        )?;
        let gelu_out = fc1_out.gelu();
        let mlp_out = layer.mlp_fc2.apply_linear_with_bias(
            &gelu_out,
            cfg.ffn_dim,
            cfg.dim,
            Arc::clone(&layer.mlp_fc2_bias),
        )?;

        // Parallel residual: x + attn_out + mlp_out.
        x.add(&attn_out)?.add(&mlp_out)
    }

    /// Forward pass using pre-allocated KV-cache buffers and
    /// `Op::WriteSlice`; returns last-position logits. Mirrors
    /// `LlamaModel::forward_with_kv_context` — see its docs for the
    /// architectural notes. The cache must have been constructed via
    /// [`KvCache::with_capacity`] with `n_kv_heads == n_heads` (Phi-2
    /// has no GQA).
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> fuel_core::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;

        if seq == 0 {
            return Err(fuel_ir::Error::Msg(
                "PhiModel::forward_with_kv_context: zero tokens".to_string(),
            )
            .bt());
        }
        if cache.n_layers() != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context: cache n_layers {} != model n_layers {}",
                cache.n_layers(),
                cfg.n_layers,
            ))
            .bt());
        }
        let max_seq_len = cache.max_seq_len.ok_or_else(|| {
            fuel_ir::Error::Msg(
                "PhiModel::forward_with_kv_context: cache was constructed via with_dims \
                 (no pre-allocated buffers); call KvCache::with_capacity(...) for the \
                 WriteSlice path"
                    .to_string(),
            )
            .bt()
        })?;
        if cached_len + seq > max_seq_len {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context: cached_len ({cached_len}) + seq \
                 ({seq}) > max_seq_len ({max_seq_len})",
            ))
            .bt());
        }
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);
        if cache.n_kv_heads != cfg.n_heads || cache.head_dim != cfg.head_dim {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context: cache shape (n_kv_heads={}, \
                 head_dim={}) disagrees with model config (n_heads={}, head_dim={})",
                cache.n_kv_heads, cache.head_dim, cfg.n_heads, cfg.head_dim,
            ))
            .bt());
        }

        // Embed lookup + reshape to [batch, seq, dim].
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        )?;
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]))?;
        let mut h = embed
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;

        // RoPE tables are sized for `rotary_dim`, not the full
        // head_dim — partial RoPE rotates only the first `rotary_dim`
        // entries.
        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_base, cached_len, seq, cfg.rotary_dim);

        // Phase D · D4: the per-token KV-write offset (`cached_len`) is a
        // runtime symbol bound through the per-pass `SymEnv` at realize,
        // not baked into the graph. One symbol shared across all layers
        // (they all append at the same offset); a fixed id keeps the
        // decode-step graph structurally identical across tokens.
        let cached_len_sym = fuel_ir::SymId(0);

        // Phase D · D4: the causal mask is hoisted to ONE shared Const
        // (was one Const per layer) — byte-identical across layers (it
        // depends only on `cached_len` / `seq` / `max_seq_len`).
        let mask_data = build_decode_causal_mask(cached_len, seq, max_seq_len);
        let mask = h.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, max_seq_len]))?;

        // Per-layer: bind the cache K + V Arcs to fresh Const NodeIds,
        // dispatch through the WriteSlice variant, clean up the
        // per-step bindings after realize.
        let cache_shape = Shape::from_dims(&[batch, cfg.n_heads, max_seq_len, cfg.head_dim]);
        let mut bound_node_ids: Vec<fuel_graph::NodeId> = Vec::with_capacity(2 * cfg.n_layers);
        for (li, layer_weights) in weights.layers.iter().enumerate() {
            let k_arc = cache.slot_storage(li, KvSlot::K).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "PhiModel::forward_with_kv_context: cache layer {li} has no K slot \
                     (with_capacity should have populated all layers)",
                ))
                .bt()
            })?;
            let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "PhiModel::forward_with_kv_context: cache layer {li} has no V slot",
                ))
                .bt()
            })?;
            let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let k_id = k_cache_node.graph_tensor().id();
            let v_id = v_cache_node.graph_tensor().id();
            ctx.insert(k_id, k_arc);
            ctx.insert(v_id, v_arc);
            bound_node_ids.push(k_id);
            bound_node_ids.push(v_id);

            h = self.apply_layer_with_kv_writes(
                &h,
                layer_weights,
                &k_cache_node,
                &v_cache_node,
                cached_len_sym,
                &rope_cos,
                &rope_sin,
                &mask,
            )?;
        }

        // Final LayerNorm, output projection (+ optional bias).
        let h_norm = h.layer_norm_affine(
            Arc::clone(&weights.final_norm_gain),
            Arc::clone(&weights.final_norm_bias),
            cfg.layer_norm_eps,
        )?;
        let logits_no_bias = weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)?;
        let logits = match &weights.output_bias {
            Some(b) => {
                let b_t =
                    h_norm.const_f32_like(Arc::clone(b), Shape::from_dims(&[cfg.vocab_size]))?;
                logits_no_bias.broadcast_add(&b_t)?
            }
            None => logits_no_bias,
        };

        let last_pos = seq - 1;
        let last_logits = logits
            .slice(1, last_pos, 1)?
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?;

        // Realize through InferenceContext. The WriteSlice nodes mutate
        // the cache buffers in place at the runtime offset `cached_len`,
        // supplied for this pass via the `SymEnv`; downstream attention
        // reads the post-write full-capacity buffers.
        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(cached_len_sym, cached_len)?;
        let logits_vec = ctx.realize_one_as_with_env::<f32>(
            last_logits.graph_tensor().graph(),
            last_logits.graph_tensor().id(),
            &sym_env,
        )?;

        // Clean up per-step bindings so they don't accumulate across
        // decode steps (each step gets a fresh graph; the previous
        // step's NodeIds are dead).
        for id in bound_node_ids {
            ctx.remove(id);
        }

        // Bump cache state.
        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }

        Ok(logits_vec)
    }

    /// Phase D · D4 — plan-once persistent decode (the Phi mirror of
    /// `LlamaModel::forward_with_kv_context_persistent`). Sibling of
    /// [`Self::forward_with_kv_context`] that HOLDS the optimized
    /// decode-step graph in `session` and, on every token after the
    /// first, re-realizes the SAME graph with the D2a prebuilt seam —
    /// **skipping the `prepare` D2H-splice + the `optimize_graph`
    /// placement DP**. The per-token re-plan win comes from not
    /// re-planning. See the LlamaModel sibling for the full control-flow
    /// contract; the Phi version differs only in the model body it
    /// builds (parallel attn+MLP, LayerNorm, partial RoPE, projection
    /// biases, optional output bias).
    ///
    /// Byte-identical to the D1 cached path ([`Self::forward_with_kv_context`])
    /// on the same prefix (same plan → same kernels). Bumps
    /// `cache.cached_len` + per-slot versions exactly as the D1 path does.
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();
        let max_seq_len = cache.max_seq_len;
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);

        // A non-`seq==1` step (prefill / spec-decode verification) is
        // shape-distinct from the held decode graph — drop any session and
        // fall back to the D1 rebuild path (the session rebuilds on the
        // next decode token).
        if seq != 1 {
            self.drop_decode_session(session, ctx);
            return self.forward_with_kv_context(tokens, cache, ctx);
        }

        // seq == 1. If a session exists but its validity keys no longer
        // match the live cache/model (max_seq_len / n_layers / dtype), it
        // is stale — drop it so we rebuild fresh below. A key differing ONLY
        // in the KV allocation gets a guarded re-bind first (GAP-028). No
        // capture exists on this path, hence the no-op reader retirement.
        refresh_decode_session(
            session,
            ctx,
            seq,
            max_seq_len,
            cache_dtype,
            cfg.n_layers,
            self.decode_shape_key(),
            cache,
            || {},
            |s, c| self.drop_decode_session(s, c),
        );

        match session.as_ref() {
            None => {
                // First decode token (or post-invalidation): build +
                // optimize the held graph ONCE.
                self.build_and_realize_first_decode_token(tokens, cache, ctx, session)
            }
            Some(_) => {
                // Subsequent decode token: re-bind data + skip optimize.
                let res = self.rebind_and_realize_prebuilt(tokens, cache, &*ctx, &*session);
                match res {
                    Ok(logits) => Ok(logits),
                    Err(fuel_core::Error::TopologyChanged { .. }) => {
                        self.drop_decode_session(session, ctx);
                        self.forward_with_kv_context(tokens, cache, ctx)
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Build the held Phi decode-step graph with STABLE re-bindable data
    /// Consts, optimize it ONCE via the capturing prebuild, populate
    /// `session`, and return the first token's logits. Only called for
    /// the first `seq == 1` decode token when there is no valid session.
    fn build_and_realize_first_decode_token(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        let cached_len = cache.cached_len;
        let max_seq_len = cache.max_seq_len.ok_or_else(|| {
            fuel_ir::Error::Msg(
                "PhiModel::forward_with_kv_context_persistent: cache built via with_dims \
                 (no pre-allocated buffers); use KvCache::with_capacity"
                    .to_string(),
            )
            .bt()
        })?;
        if cache.n_layers() != cfg.n_layers {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context_persistent: cache n_layers {} != model {}",
                cache.n_layers(),
                cfg.n_layers,
            ))
            .bt());
        }
        if cached_len + seq > max_seq_len {
            return Err(fuel_ir::Error::Msg(format!(
                "PhiModel::forward_with_kv_context_persistent: cached_len ({cached_len}) + \
                 seq ({seq}) > max_seq_len ({max_seq_len})",
            ))
            .bt());
        }
        let cache_dtype = cache.dtype.unwrap_or(DType::F32);

        // Embed lookup + reshape to [batch, seq, dim]. Token-ids is a
        // STABLE re-bindable placeholder Const (bytes bound via ctx).
        let embed = Tensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.dim]),
            &Device::cpu(),
        )?;
        let token_ids = embed.const_placeholder_like(Shape::from_dims(&[seq]), DType::U32);
        let token_ids_node = token_ids.graph_tensor().id();
        let mut h = embed
            .index_select(0, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.dim]))?;

        // RoPE cos/sin: STABLE re-bindable placeholder Consts. Phi's
        // tables are sized for `rotary_dim` (partial RoPE), NOT head_dim.
        let rope_shape = Shape::from_dims(&[seq, cfg.rotary_dim]);
        let rope_cos = h.const_placeholder_like(rope_shape.clone(), DType::F32);
        let rope_sin = h.const_placeholder_like(rope_shape, DType::F32);
        let rope_cos_node = rope_cos.graph_tensor().id();
        let rope_sin_node = rope_sin.graph_tensor().id();

        // Mask: STABLE re-bindable placeholder Const (hoisted; shared).
        let mask =
            h.const_placeholder_like(Shape::from_dims(&[1, 1, seq, max_seq_len]), DType::F32);
        let mask_node = mask.graph_tensor().id();

        let cached_len_sym = fuel_ir::SymId(0);
        // No GQA in Phi-2: the KV cache carries `n_heads`.
        let cache_shape = Shape::from_dims(&[batch, cfg.n_heads, max_seq_len, cfg.head_dim]);

        // Per-layer KV placeholder Consts (STABLE). The Arcs are bound
        // ONCE here and mutate in place via Op::WriteSlice each token.
        let mut kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)> =
            Vec::with_capacity(cfg.n_layers);
        for (li, layer_weights) in weights.layers.iter().enumerate() {
            let k_arc = cache.slot_storage(li, KvSlot::K).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "PhiModel::forward_with_kv_context_persistent: cache layer {li} has no K slot",
                ))
                .bt()
            })?;
            let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "PhiModel::forward_with_kv_context_persistent: cache layer {li} has no V slot",
                ))
                .bt()
            })?;
            let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
            let k_id = k_cache_node.graph_tensor().id();
            let v_id = v_cache_node.graph_tensor().id();
            ctx.insert(k_id, k_arc);
            ctx.insert(v_id, v_arc);
            kv_nodes.push((k_id, v_id));

            h = self.apply_layer_with_kv_writes(
                &h,
                layer_weights,
                &k_cache_node,
                &v_cache_node,
                cached_len_sym,
                &rope_cos,
                &rope_sin,
                &mask,
            )?;
        }

        // Final LayerNorm, output projection (+ optional output bias).
        let h_norm = h.layer_norm_affine(
            Arc::clone(&weights.final_norm_gain),
            Arc::clone(&weights.final_norm_bias),
            cfg.layer_norm_eps,
        )?;
        let logits_no_bias = weights
            .output
            .apply_linear(&h_norm, cfg.dim, cfg.vocab_size)?;
        let logits = match &weights.output_bias {
            Some(b) => {
                let b_t =
                    h_norm.const_f32_like(Arc::clone(b), Shape::from_dims(&[cfg.vocab_size]))?;
                logits_no_bias.broadcast_add(&b_t)?
            }
            None => logits_no_bias,
        };
        let last_pos = seq - 1;
        let logits_root = logits
            .slice(1, last_pos, 1)?
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?;
        let logits_node = logits_root.graph_tensor().id();
        let graph = logits_root.graph_tensor().graph().clone();

        // Bind the per-token DATA into ctx (token-ids / RoPE / mask) as
        // device-resident Arcs so the FIRST realize's const-cache walk
        // resolves them (they are placeholders, not in graph.storage_map).
        // KV Arcs were already inserted above. The optimize + realize then
        // runs ONCE, capturing the reusable artifacts + the full realized
        // cache (weights + KV + data) for the held session.
        let data =
            self.build_token_rope_mask_arcs(ctx.device(), cached_len, tokens, max_seq_len)?;
        ctx.insert(token_ids_node, Arc::clone(&data.token_ids));
        ctx.insert(rope_cos_node, Arc::clone(&data.rope_cos));
        ctx.insert(rope_sin_node, Arc::clone(&data.rope_sin));
        ctx.insert(mask_node, Arc::clone(&data.mask));

        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(cached_len_sym, cached_len)?;

        let (effective_target, optimized, base_cache, logits_vec) =
            ctx.prebuild_optimized_capturing_as_with_env::<f32>(&graph, logits_node, &sym_env)?;

        // The held session now owns the graph + base_cache; drop the
        // transient ctx bindings.
        ctx.remove(token_ids_node);
        ctx.remove(rope_cos_node);
        ctx.remove(rope_sin_node);
        ctx.remove(mask_node);
        for (k, v) in &kv_nodes {
            ctx.remove(*k);
            ctx.remove(*v);
        }

        *session = Some(fuel_core::inference_context::DecodeSession::new(
            graph,
            optimized,
            effective_target,
            logits_node,
            token_ids_node,
            rope_cos_node,
            rope_sin_node,
            mask_node,
            kv_nodes,
            // Phi decode stays on the SymEnv `Op::WriteSlice` path (no
            // device-offset / CapturedRun yet — sequenced behind Llama).
            None,
            cached_len_sym,
            // PhiModel decode does not offer the CUDA flash-decode arm yet
            // (only LlamaModel is wired), so this attended-length symbol is
            // carried for API parity but never referenced/bound in Phi's
            // per-token env — a placeholder distinct from `cached_len_sym`.
            fuel_ir::SymId(1),
            base_cache,
            seq,
            max_seq_len,
            cfg.n_layers,
            cache_dtype,
            self.decode_shape_key(),
            // Which ALLOCATION's KV Arcs are baked into `base_cache`, and
            // where they live — both read from the one source (GAP-028).
            cache,
        ));

        // Bump cache state (identical to the D1 path).
        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }
        Ok(logits_vec)
    }

    /// Re-bind the per-token data Consts (token-ids / RoPE / mask) into
    /// device Arcs, bind the `SymEnv`, and realize via the D2a prebuilt
    /// seam (SKIPPING optimize) over the held session's base cache. The
    /// KV Arcs are stable (mutated in place by WriteSlice) — not touched
    /// here. Called for every decode token after the first.
    fn rebind_and_realize_prebuilt(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &InferenceContext,
        session: &Option<fuel_core::inference_context::DecodeSession>,
    ) -> fuel_core::Result<Vec<f32>> {
        // GAP-029 2b: forwards to the shared driver. Phi previously built its
        // own `SymEnv` binding ONLY `cached_len_sym`; the shared
        // `per_token_sym_env` also binds `attended_len_sym`. That extra binding
        // is a measured no-op for Phi — negative-controlled in
        // `phi_attended_len_sym_is_unreferenced_negative_control`, on the SymEnv
        // path where a wrong `cached_len_sym` DOES move the logits, so the env
        // is provably live while this symbol is inert.
        fuel_core::persistent_decode::rebind_and_realize_prebuilt(
            self, tokens, cache, ctx, session, None,
        )
    }

    /// Recompute the per-token host bytes (token-ids / RoPE cos+sin sized
    /// for `rotary_dim` / mask) and build device-resident Arcs from them
    /// (the SAME upload path `KvCache::with_capacity` uses). The bytes
    /// change per token; the NodeId stays stable (re-bound via a
    /// `base_cache` overwrite, not a fresh graph).
    /// Phi's per-token decode data as HOST buffers — the single source of
    /// truth shared by [`Self::build_token_rope_mask_arcs`] (which uploads it)
    /// and [`Self::build_token_rope_mask_bytes`] (which serialises it for
    /// capture replay).
    ///
    /// Extracted rather than duplicated deliberately. If the upload path and
    /// the replay path computed this data separately they could drift, and the
    /// failure would be silent: replay would H2D the *wrong bytes* into the
    /// right buffers and produce confident garbage. One computation, two
    /// consumers, drift unrepresentable.
    ///
    /// **Phi-specific:** RoPE tables are sized for `rotary_dim`, not `head_dim`
    /// — Phi uses PARTIAL rotary embedding. Getting that wrong here is exactly
    /// the silent-wrong-bytes failure above, since the shapes would still match.
    fn compute_token_rope_mask_host_data(
        &self,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
    ) -> fuel_core::Result<TokenDataHost> {
        let cfg = &self.config;
        let seq = tokens.len();
        // Phi's RoPE tables are sized for `rotary_dim` (partial RoPE).
        let (cos_data, sin_data) =
            fuel_graph::build_rope_tables(cfg.rope_base, cached_len, seq, cfg.rotary_dim);
        Ok(TokenDataHost {
            token_ids: fuel_ir::HostBuffer::U32(tokens.to_vec()),
            rope_cos: fuel_ir::HostBuffer::F32(cos_data),
            rope_sin: fuel_ir::HostBuffer::F32(sin_data),
            mask: fuel_ir::HostBuffer::F32(build_decode_causal_mask(cached_len, seq, max_seq_len)),
            // Phi decode stays on the SymEnv `Op::WriteSlice` path — the KV
            // write offset rides `cached_len_sym`, so there is no device-offset
            // operand to rebind. Capture handles this: the offset entry is
            // conditional on `offset_node().is_some()`.
            offset: None,
        })
    }

    fn build_token_rope_mask_arcs(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
    ) -> fuel_core::Result<fuel_core::inference_context::DecodeTokenData> {
        let host = self.compute_token_rope_mask_host_data(cached_len, tokens, max_seq_len)?;
        let upload = fuel_core::pipelined_bridge::upload_host_buffer_to_device;
        Ok(fuel_core::inference_context::DecodeTokenData {
            token_ids: upload(device, host.token_ids)?,
            rope_cos: upload(device, host.rope_cos)?,
            rope_sin: upload(device, host.rope_sin)?,
            mask: upload(device, host.mask)?,
            offset: None,
        })
    }

    /// The same per-token data as raw host bytes, for
    /// [`fuel_dispatch::pipelined::CapturedDecodeSession::replay_token`]'s
    /// in-place H2D overwrite of the fixed capture buffers. Shares
    /// [`Self::compute_token_rope_mask_host_data`] with the Arc path, so the
    /// captured and uncaptured routes cannot disagree about what a token's
    /// data is.
    #[cfg(feature = "cuda")]
    fn build_token_rope_mask_bytes(
        &self,
        cached_len: usize,
        tokens: &[u32],
        max_seq_len: usize,
    ) -> fuel_core::Result<TokenDataBytes> {
        let host = self.compute_token_rope_mask_host_data(cached_len, tokens, max_seq_len)?;
        Ok(TokenDataBytes {
            token_ids: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.token_ids),
            rope_cos: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.rope_cos),
            rope_sin: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.rope_sin),
            mask: fuel_core::pipelined_bridge::host_buffer_to_bytes(&host.mask),
            offset: None,
        })
    }

    /// CapturedRun decode for Phi — the sibling of
    /// [`LlamaModel::forward_with_kv_context_captured`], same four cases:
    ///
    /// 1. `seq != 1` — not a decode step; drop session + capture, fall back to
    ///    the rebuild path.
    /// 1b. stale held pair — retire BOTH (see below), leaving case 2 to rebuild.
    /// 2. first decode token — build the held session; `captured` stays `None`.
    /// 3. second decode token — build this token's data as fresh FIXED-address
    ///    Arcs, merge over `session.base_cache()`, and capture once targeting
    ///    `logits_node()` (NOT `effective_target`: `capture_decode` rejects the
    ///    D2H `Op::Copy` splice as non-single-device-CUDA-capturable, so the
    ///    D2H happens here, after replay). The warm pass inside `capture()`
    ///    already computed this token, so its logits come from an empty-updates
    ///    `replay_token(&[])`.
    /// 4. third token onward — recompute the per-token data as raw bytes and
    ///    replay with one `cuGraphLaunch`.
    ///
    /// **Phi is on the SymEnv path** (`offset: None`), so there is no
    /// device-offset operand to rebind — the KV write offset rides
    /// `cached_len_sym`. The offset entries below are conditional for exactly
    /// that reason and are simply absent for Phi today.
    ///
    /// Staleness retires the capture WITH the session via the shared
    /// [`invalidate_decode_pair_if_stale`]: a recorded CUDA graph outliving its
    /// session would replay against fixed device addresses that no longer
    /// describe the live cache — wrong logits at full speed. Sharing the helper
    /// with `LlamaModel` rather than copying it means the two models cannot
    /// drift on what "stale" means.
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

        // ---- 1. Non-decode step: drop state, fall back. ----
        if seq != 1 {
            *captured = None;
            self.drop_decode_session(session, ctx);
            return self.forward_with_kv_context(tokens, cache, ctx);
        }

        // ---- 1b. Staleness: retire capture-then-session together. ----
        invalidate_decode_pair_if_stale(
            session,
            captured,
            ctx,
            seq,
            cache.max_seq_len,
            cache.dtype.unwrap_or(DType::F32),
            cfg.n_layers,
            self.decode_shape_key(),
            cache,
            |s, ctx| self.drop_decode_session(s, ctx),
        );

        // ---- 2. First decode token: build the held session. ----
        if session.is_none() {
            *captured = None;
            return self.build_and_realize_first_decode_token(tokens, cache, ctx, session);
        }

        let cached_len = cache.cached_len;

        // ---- 3. Second decode token: build the capture. ----
        if captured.is_none() {
            let device = ctx.device().clone();
            let s = session.as_ref().expect("session is Some (checked above)");

            let data =
                self.build_token_rope_mask_arcs(&device, cached_len, tokens, s.max_seq_len())?;

            let mut merged_cache: fuel_dispatch::pipelined::StorageCache = s.base_cache().clone();
            merged_cache.insert(s.token_ids_node(), Arc::clone(&data.token_ids));
            merged_cache.insert(s.rope_cos_node(), Arc::clone(&data.rope_cos));
            merged_cache.insert(s.rope_sin_node(), Arc::clone(&data.rope_sin));
            merged_cache.insert(s.mask_node(), Arc::clone(&data.mask));
            let per_token_node_ids: Vec<fuel_graph::NodeId> = vec![
                s.token_ids_node(),
                s.rope_cos_node(),
                s.rope_sin_node(),
                s.mask_node(),
            ];

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
                    let res = self.rebind_and_realize_prebuilt(tokens, cache, &*ctx, &*session);
                    return res;
                }
            };

            // The warm pass inside `capture()` already computed THIS token.
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

        let bytes = self.build_token_rope_mask_bytes(cached_len, tokens, s.max_seq_len())?;
        let updates: Vec<(fuel_graph::NodeId, &[u8])> = vec![
            (s.token_ids_node(), bytes.token_ids.as_slice()),
            (s.rope_cos_node(), bytes.rope_cos.as_slice()),
            (s.rope_sin_node(), bytes.rope_sin.as_slice()),
            (s.mask_node(), bytes.mask.as_slice()),
        ];

        let output = cap.replay_token(&updates)?;
        let logits = captured_output_to_f32(&output)?;

        cache.cached_len += seq;
        for li in 0..cfg.n_layers {
            cache.bump_version(li, KvSlot::K);
            cache.bump_version(li, KvSlot::V);
        }
        Ok(logits)
    }

    /// Drop a held decode session, removing any leftover persistent
    /// data-Const / KV bindings from `ctx` (defensive). No-op if `None`.
    fn drop_decode_session(
        &self,
        session: &mut Option<fuel_core::inference_context::DecodeSession>,
        ctx: &mut InferenceContext,
    ) {
        if let Some(s) = session.take() {
            ctx.remove(s.token_ids_node());
            ctx.remove(s.rope_cos_node());
            ctx.remove(s.rope_sin_node());
            ctx.remove(s.mask_node());
            for (k, v) in s.kv_nodes() {
                ctx.remove(*k);
                ctx.remove(*v);
            }
        }
    }

    /// Streaming generation through [`Self::forward_with_kv_context`].
    /// Allocates a pre-allocated [`KvCache`] of capacity
    /// `prompt_tokens.len() + max_new_tokens` on `device`, then loops
    /// prefill + decode, calling `on_token` for each generated token.
    /// Mirrors `LlamaModel::generate_streaming_with_kv_context`.
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
                "PhiModel::generate_streaming_with_kv_context: prompt is empty".to_string(),
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
            cfg.n_heads,
            cfg.head_dim,
            max_seq_len,
            dtype,
            device,
        )?;
        let mut ctx = InferenceContext::new(device.clone());

        // Phase D · D4: hold ONE plan-once decode session across the whole
        // generation (the Phi mirror of the LlamaModel D2c wiring). Prefill
        // (seq>1) routes through the persistent entry, which falls back to
        // the D1 rebuild path WITHOUT building the session; each per-token
        // decode step (seq==1) builds the held graph on the FIRST token
        // (optimize once) and reuses it — skipping optimize — thereafter.
        // The session is loop-internal; the public signature is unchanged.
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // CUDA-graph capture is ON BY DEFAULT here, matching
        // `LlamaModel::generate_streaming_with_kv_context`. Measured on Llama:
        // 4.28x at k=1 in release, byte-exact, replay cost build-profile-
        // invariant. Phi shares the decode substrate (`DecodeSession`, the
        // capture/replay machinery, the staleness predicate), so the same win
        // applies; the ratio itself has not been separately measured for Phi.
        //
        // Phi stays on the SymEnv path (`offset: None`) — the KV write offset
        // rides `cached_len_sym` rather than a device-offset operand — which
        // capture handles, since the offset entry is conditional.
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
    /// [`Self::generate_streaming_with_kv_context`].
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

    /// Load weights from a HuggingFace Hub repo (e.g. "microsoft/phi-2").
    pub fn from_hub(repo_id: &str) -> fuel_core::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| fuel_core::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| fuel_core::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = PhiConfig::from_hf_json_str(&config_str)?;

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
                let p = repo
                    .get("model.safetensors")
                    .map_err(|e| fuel_core::Error::Msg(format!("hf-hub model.safetensors: {e}")))?;
                vec![p]
            }
        };

        let st = unsafe { fuel_core::safetensors::MmapedSafetensors::multi(&weight_paths) }?;
        let weights = PhiWeights::load_from_mmapped(&st, &config)?;
        Ok(PhiModel { config, weights })
    }

    /// Load a Phi-2 model from a GGUF file (e.g. one of TheBloke's
    /// quantized Phi-2 releases). Q4_0 tensors stay quantized on-device;
    /// other dtypes dequantize to F32 at load time. Config is derived
    /// from the GGUF metadata.
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P) -> fuel_core::Result<Self> {
        use fuel_core::quantized::gguf_mmap::MmapedContent;
        let mc = MmapedContent::from_path(&path)?;
        let meta = mc.metadata();
        let get_u32 = |k: &str| -> fuel_core::Result<u32> {
            meta.get(k)
                .ok_or_else(|| fuel_core::Error::Msg(format!("gguf metadata: missing {k:?}")))?
                .to_u32()
                .map_err(|e| fuel_core::Error::Msg(format!("gguf metadata {k:?}: {e:?}")))
        };
        let get_f32 = |k: &str| -> fuel_core::Result<f32> {
            meta.get(k)
                .ok_or_else(|| fuel_core::Error::Msg(format!("gguf metadata: missing {k:?}")))?
                .to_f32()
                .map_err(|e| fuel_core::Error::Msg(format!("gguf metadata {k:?}: {e:?}")))
        };
        // Phi-2 metadata keys (llama.cpp convention).
        let dim = get_u32("phi2.embedding_length")? as usize;
        let n_layers = get_u32("phi2.block_count")? as usize;
        let n_heads = get_u32("phi2.attention.head_count")? as usize;
        let ffn_dim = get_u32("phi2.feed_forward_length")? as usize;
        let head_dim = dim / n_heads;
        let layer_norm_eps = get_f32("phi2.attention.layer_norm_epsilon").unwrap_or(1e-5) as f64;
        let rope_base = get_f32("phi2.rope.freq_base").unwrap_or(10_000.0) as f64;
        let rotary_dim = get_u32("phi2.rope.dimension_count").unwrap_or(32) as usize;
        let partial_rotary_factor = rotary_dim as f64 / head_dim as f64;

        // Derive vocab_size from the token_embd shape (no explicit
        // metadata key for it in GGUF; llama.cpp infers from the
        // tokenizer array which needs a dedicated API). token_embd has
        // shape [vocab, dim].
        let vocab_size = mc
            .content()
            .tensor_infos
            .get("token_embd.weight")
            .ok_or_else(|| fuel_core::Error::Msg("gguf: missing token_embd.weight".into()))?
            .shape
            .dims()[0];

        let config = PhiConfig {
            vocab_size,
            dim,
            n_layers,
            n_heads,
            head_dim,
            ffn_dim,
            layer_norm_eps,
            rope_base,
            partial_rotary_factor,
            rotary_dim,
            tie_word_embeddings: false,
        };

        // MmapedContent drops here; the load_from_gguf path re-opens.
        // In practice this is two mmaps in flight, both pointing at the
        // same file — cheap on modern OSes. If this becomes a hotspot,
        // refactor to hand the Arc<Mmap> through.
        drop(mc);
        let weights = PhiWeights::load_from_gguf(&path, &config)?;
        Ok(PhiModel { config, weights })
    }
}

impl PhiWeights {
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &PhiConfig,
    ) -> fuel_core::Result<Self> {
        let kv_dim = cfg.n_heads * cfg.head_dim;
        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * cfg.dim {
            fuel_core::bail!(
                "embed_tokens: {} elements, expected {}",
                token_embedding.len(),
                cfg.vocab_size * cfg.dim,
            );
        }

        let mut layers: Vec<PhiLayerWeights> = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            // Phi-2 uses `dense` for the output projection (not `o_proj`)
            // and `fc1`/`fc2` for the MLP (not `gate_proj`/`up_proj`/`down_proj`).
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
            let attn_dense = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.self_attn.dense.weight"),
                cfg.dim,
                cfg.dim,
            )?;
            let mlp_fc1 = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.fc1.weight"),
                cfg.ffn_dim,
                cfg.dim,
            )?;
            let mlp_fc2 = load_transposed_matrix_preserve_dtype(
                st,
                &format!("model.layers.{i}.mlp.fc2.weight"),
                cfg.dim,
                cfg.ffn_dim,
            )?;

            let attn_q_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.q_proj.bias"),
            )?);
            let attn_k_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.k_proj.bias"),
            )?);
            let attn_v_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.v_proj.bias"),
            )?);
            let attn_dense_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.self_attn.dense.bias"),
            )?);
            let mlp_fc1_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.mlp.fc1.bias"),
            )?);
            let mlp_fc2_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.mlp.fc2.bias"),
            )?);

            // Phi-2's pre-block LayerNorm is `input_layernorm.{weight,bias}`.
            let norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.input_layernorm.weight"),
            )?);
            let norm_bias = Arc::from(load_tensor_as_f32(
                st,
                &format!("model.layers.{i}.input_layernorm.bias"),
            )?);

            layers.push(PhiLayerWeights {
                attn_qkv: PhiQkv::Split {
                    q: attn_q,
                    q_bias: attn_q_bias,
                    k: attn_k,
                    k_bias: attn_k_bias,
                    v: attn_v,
                    v_bias: attn_v_bias,
                },
                attn_dense,
                attn_dense_bias,
                mlp_fc1,
                mlp_fc1_bias,
                mlp_fc2,
                mlp_fc2_bias,
                norm_gain,
                norm_bias,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.final_layernorm.weight")?);
        let final_norm_bias = Arc::from(load_tensor_as_f32(st, "model.final_layernorm.bias")?);

        let output: WeightStorage = if cfg.tie_word_embeddings {
            // Tied: transpose embed_tokens.
            let mut transposed = vec![0.0_f32; cfg.dim * cfg.vocab_size];
            for i in 0..cfg.vocab_size {
                for j in 0..cfg.dim {
                    transposed[j * cfg.vocab_size + i] = token_embedding[i * cfg.dim + j];
                }
            }
            WeightStorage::F32(Arc::from(transposed))
        } else {
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, cfg.dim)?
        };
        let output_bias = load_tensor_as_f32(st, "lm_head.bias").ok().map(Arc::from);

        Ok(PhiWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain,
            final_norm_bias,
            output,
            output_bias,
        })
    }

    /// Load Phi-2 weights from a GGUF file. Q4_0 tensors stay quantized
    /// (go into `WeightStorage::Q4_0`); other GGML dtypes are dequantized
    /// to F32 at load time and stored as `WeightStorage::F32` (or
    /// `Arc<[f32]>` for biases, norms, embedding).
    ///
    /// GGUF key layout for Phi-2:
    ///   token_embd.weight / output.weight / output_norm.{weight,bias}
    ///   blk.{i}.attn_qkv.{weight,bias}           (packed 3*dim × dim)
    ///   blk.{i}.attn_output.{weight,bias}
    ///   blk.{i}.ffn_up.{weight,bias}
    ///   blk.{i}.ffn_down.{weight,bias}
    ///   blk.{i}.attn_norm.{weight,bias}
    pub fn load_from_gguf<P: AsRef<std::path::Path>>(
        path: P,
        cfg: &PhiConfig,
    ) -> fuel_core::Result<Self> {
        use fuel_core::quantized::gguf_mmap::MmapedContent;
        let mc = MmapedContent::from_path(path)?;
        let content = mc.content();
        let (mmap_arc, _) = (mc.mmap(), ());
        let mmap_bytes: &[u8] = &mmap_arc[..];
        let data_off = content.tensor_data_offset as usize;

        // Extract a raw byte slice for a tensor.
        let get_tensor_bytes = |name: &str| -> fuel_core::Result<(
            &[u8],
            fuel_core::quantized::GgmlDType,
            Vec<usize>,
        )> {
            let info = content
                .tensor_infos
                .get(name)
                .ok_or_else(|| fuel_core::Error::Msg(format!("gguf: missing tensor {name:?}")))?;
            let elems = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let bytes_len = elems / block_size * info.ggml_dtype.type_size();
            let start = data_off + info.offset as usize;
            Ok((
                &mmap_bytes[start..start + bytes_len],
                info.ggml_dtype,
                info.shape.dims().to_vec(),
            ))
        };

        // Load an F32 vector (for biases, norms, embedding). Dequantizes
        // if necessary.
        let load_f32 = |name: &str| -> fuel_core::Result<Vec<f32>> {
            let (bytes, dt, _dims) = get_tensor_bytes(name)?;
            dequant_gguf_bytes_to_f32(bytes, dt, name)
        };

        // Load a weight matrix as WeightStorage. For Q4_0 bytes, keep
        // them quantized; for other dtypes, dequantize to F32.
        // `out_features × in_features` is the GGUF/llama.cpp convention.
        let load_weight = |name: &str,
                           out_features: usize,
                           in_features: usize|
         -> fuel_core::Result<WeightStorage> {
            let (bytes, dt, dims) = get_tensor_bytes(name)?;
            // GGUF stores weights as [out, in] — matches our Q4_0 block layout.
            let expected_elems = out_features * in_features;
            let actual_elems: usize = dims.iter().product();
            if actual_elems != expected_elems {
                fuel_core::bail!(
                    "gguf: tensor {name:?} has {actual_elems} elements, expected {expected_elems} for [{out_features}, {in_features}]",
                );
            }
            // Debug fallback: FUEL_FORCE_F32=1 dequantizes every weight
            // at load time to isolate Q4_0-path bugs from model-structure
            // bugs. Useful for validating the PhiModel/loader against a
            // known-good computation path.
            let force_f32 = std::env::var("FUEL_FORCE_F32").is_ok();
            match dt {
                fuel_core::quantized::GgmlDType::Q4_0 if !force_f32 => Ok(WeightStorage::Q4_0 {
                    words: bytes_to_u32_arc(bytes),
                    bytes_len: bytes.len(),
                    in_features,
                    out_features,
                }),
                _ => {
                    // Dequantized data is in GGUF's native [out, in]
                    // row-major layout. Our standard F32/BF16 matmul
                    // expects [in, out], so transpose before storing.
                    // (Q4_0 keeps its native layout because qmatmul
                    // reads blocks as [N, K/32] directly.)
                    let f32_out_in = dequant_gguf_bytes_to_f32(bytes, dt, name)?;
                    let mut f32_in_out = vec![0.0_f32; out_features * in_features];
                    for o in 0..out_features {
                        for i in 0..in_features {
                            f32_in_out[i * out_features + o] = f32_out_in[o * in_features + i];
                        }
                    }
                    Ok(WeightStorage::F32(Arc::from(f32_in_out)))
                }
            }
        };

        let token_embedding = load_f32("token_embd.weight")?;
        if token_embedding.len() != cfg.vocab_size * cfg.dim {
            fuel_core::bail!(
                "gguf token_embd: {} elems, expected {}×{}",
                token_embedding.len(),
                cfg.vocab_size,
                cfg.dim,
            );
        }

        let mut layers: Vec<PhiLayerWeights> = Vec::with_capacity(cfg.n_layers);
        let kv_dim = cfg.n_heads * cfg.head_dim;

        for i in 0..cfg.n_layers {
            let prefix = format!("blk.{i}");

            // Phi-2 GGUF packs Q/K/V into a single attn_qkv tensor of
            // shape [3*dim, dim]. We keep it PACKED as a single
            // WeightStorage and let the forward pass do one big matmul
            // + slice after (matching Candle's eager approach). This
            // avoids any hazards around byte-level Q/K/V splits on the
            // weight side.
            let attn_qkv_weight =
                load_weight(&format!("{prefix}.attn_qkv.weight"), 3 * cfg.dim, cfg.dim)?;
            let qkv_bias_vec = load_f32(&format!("{prefix}.attn_qkv.bias"))?;
            if qkv_bias_vec.len() != 3 * cfg.dim {
                fuel_core::bail!(
                    "gguf attn_qkv.bias: {} elems, expected {}",
                    qkv_bias_vec.len(),
                    3 * cfg.dim
                );
            }
            let qkv_bias: Arc<[f32]> = Arc::from(qkv_bias_vec);
            let _ = kv_dim; // Phi-2 has no GQA; kv_dim == dim

            let attn_dense =
                load_weight(&format!("{prefix}.attn_output.weight"), cfg.dim, cfg.dim)?;
            let attn_dense_bias = Arc::from(load_f32(&format!("{prefix}.attn_output.bias"))?);

            let mlp_fc1 = load_weight(&format!("{prefix}.ffn_up.weight"), cfg.ffn_dim, cfg.dim)?;
            let mlp_fc1_bias = Arc::from(load_f32(&format!("{prefix}.ffn_up.bias"))?);
            let mlp_fc2 = load_weight(&format!("{prefix}.ffn_down.weight"), cfg.dim, cfg.ffn_dim)?;
            let mlp_fc2_bias = Arc::from(load_f32(&format!("{prefix}.ffn_down.bias"))?);

            let norm_gain = Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
            let norm_bias = Arc::from(load_f32(&format!("{prefix}.attn_norm.bias"))?);

            layers.push(PhiLayerWeights {
                attn_qkv: PhiQkv::Packed {
                    qkv: attn_qkv_weight,
                    qkv_bias,
                },
                attn_dense,
                attn_dense_bias,
                mlp_fc1,
                mlp_fc1_bias,
                mlp_fc2,
                mlp_fc2_bias,
                norm_gain,
                norm_bias,
            });
        }

        let final_norm_gain = Arc::from(load_f32("output_norm.weight")?);
        let final_norm_bias = Arc::from(load_f32("output_norm.bias")?);

        // Output projection. In GGUF: `output.weight` has shape [vocab, dim].
        let output = load_weight("output.weight", cfg.vocab_size, cfg.dim)?;
        let output_bias = load_f32("output.bias").ok().map(Arc::from);

        Ok(PhiWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain,
            final_norm_bias,
            output,
            output_bias,
        })
    }
}

/// Dequantize a raw byte slice from GGUF (of the given GGML dtype) into
/// a flat `Vec<f32>`. Used by the lazy GGUF loader for non-Q4_0 tensors
/// (biases, norms, embeddings, and weight matrices of any dtype that
/// lacks a fused on-device dequant path).
fn dequant_gguf_bytes_to_f32(
    bytes: &[u8],
    dt: fuel_core::quantized::GgmlDType,
    name: &str,
) -> fuel_core::Result<Vec<f32>> {
    use fuel_core::quantized::GgmlDType;
    use half::{bf16, f16};
    match dt {
        GgmlDType::F32 => {
            if !bytes.len().is_multiple_of(4) {
                fuel_core::bail!(
                    "gguf {name}: F32 byte count {} not multiple of 4",
                    bytes.len()
                );
            }
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        GgmlDType::F16 => {
            if !bytes.len().is_multiple_of(2) {
                fuel_core::bail!(
                    "gguf {name}: F16 byte count {} not multiple of 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        GgmlDType::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                fuel_core::bail!(
                    "gguf {name}: BF16 byte count {} not multiple of 2",
                    bytes.len()
                );
            }
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        GgmlDType::Q4_0 => {
            // Should rarely be requested this way (prefer keeping Q4_0
            // quantized), but support it for biases or other oddities.
            Ok(cpu_dequant_q4_0_bytes(bytes))
        }
        GgmlDType::Q8_0 => Ok(cpu_dequant_q8_0_bytes(bytes)),
        // k-quants: dequant via the reference-CPU GgmlType trait impls.
        GgmlDType::Q6K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ6K>(bytes)),
        GgmlDType::Q5K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ5K>(bytes)),
        GgmlDType::Q4K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ4K>(bytes)),
        GgmlDType::Q3K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ3K>(bytes)),
        GgmlDType::Q2K => Ok(cpu_dequant_via_trait::<fuel_quantized::BlockQ2K>(bytes)),
        other => fuel_core::bail!(
            "gguf {name}: dequant-to-f32 for dtype {other:?} not implemented in lazy loader"
        ),
    }
}

/// Dequantize an arbitrary k-quant block stream to F32 via the
/// reference `GgmlType::to_float` trait. Callers give the concrete
/// block type `T` (e.g. `BlockQ6K`); the function reinterprets the
/// byte slice as `&[T]` and calls the impl. Used for dtypes that
/// don't have a fused on-device dequant kernel (yet).
fn cpu_dequant_via_trait<T: fuel_quantized::GgmlType>(bytes: &[u8]) -> Vec<f32> {
    let block_bytes = std::mem::size_of::<T>();
    assert!(
        bytes.len().is_multiple_of(block_bytes),
        "cpu_dequant_via_trait: bytes {} not multiple of block_bytes {}",
        bytes.len(),
        block_bytes
    );
    let n_blocks = bytes.len() / block_bytes;
    // SAFETY: T is #[repr(C)]; GGUF bytes are laid out as a dense array
    // of T structs. The source mmap is 8-byte aligned per memmap2, which
    // satisfies every block struct's alignment (≤ 4 in practice).
    let blocks: &[T] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n_blocks) };
    let mut out = vec![0.0_f32; n_blocks * T::BLCK_SIZE];
    T::to_float(blocks, &mut out);
    out
}

/// Reinterpret a byte slice as a u32 `Arc` by reading little-endian
/// u32 words. Input length must be a multiple of 4. This performs one
/// copy at load time — all subsequent uses are cheap Arc clones.
fn bytes_to_u32_arc(bytes: &[u8]) -> Arc<[u32]> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "bytes_to_u32_arc: len must be multiple of 4"
    );
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Arc::from(words)
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
            out[base + kk] = lo as f32 * d;
            out[base + 16 + kk] = hi as f32 * d;
        }
    }
    out
}

fn cpu_dequant_q8_0_bytes(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    let bpb = 34usize;
    let epb = 32usize;
    let n_blocks = bytes.len() / bpb;
    let mut out = vec![0.0_f32; n_blocks * epb];
    for b in 0..n_blocks {
        let off = b * bpb;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let base = b * epb;
        for kk in 0..32 {
            let q = bytes[off + 2 + kk] as i8 as i32;
            out[base + kk] = q as f32 * d;
        }
    }
    out
}

/// Apply rotary embeddings to only the first `rotary_dim` entries of
/// the last dimension; pass the remaining `head_dim - rotary_dim` entries
/// through unchanged. Used by Phi-2 and Phi-3 which rotate only a
/// fraction of each head's feature dim.
///
/// Input shape: `[..., head_dim]`. Output shape: same.
fn partial_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
) -> Tensor {
    if rotary_dim == head_dim {
        return x.rope_with_tables(cos, sin).unwrap();
    }
    let rank = x.shape().dims().len();
    let last = rank - 1;
    let x_rot = x.slice(last, 0, rotary_dim).unwrap();
    let x_pass = x.slice(last, rotary_dim, head_dim - rotary_dim).unwrap();
    let x_rot_rotated = x_rot.rope_with_tables(cos, sin).unwrap();
    x_rot_rotated.concat(&x_pass, last).unwrap()
}

#[cfg(test)]
impl PhiConfig {
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
        let ffn_dim = get_usize("intermediate_size")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(dim / n_heads);
        let layer_norm_eps = get_f64("layer_norm_eps").unwrap_or(1e-5);
        let rope_base = get_f64("rope_theta").unwrap_or(10_000.0);
        let partial_rotary_factor = get_f64("partial_rotary_factor").unwrap_or(0.4);
        let rotary_dim = (partial_rotary_factor * head_dim as f64).round() as usize;
        if !rotary_dim.is_multiple_of(2) {
            fuel_core::bail!(
                "PhiConfig: rotary_dim {rotary_dim} must be even (partial_rotary_factor={partial_rotary_factor}, head_dim={head_dim})"
            );
        }
        let tie_word_embeddings = v
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);

        Ok(PhiConfig {
            vocab_size,
            dim,
            n_layers,
            n_heads,
            head_dim,
            ffn_dim,
            layer_norm_eps,
            rope_base,
            partial_rotary_factor,
            rotary_dim,
            tie_word_embeddings,
        })
    }
}
#[cfg(test)]
mod phi_config_tests {
    use super::PhiConfig;

    /// ⚠️ THIS CONFIG HAD NO TEST FIXTURES AT ALL BEFORE THIS CORPUS.
    ///
    /// Its only caller was `PhiModel::from_hub`, which reads `config.json`
    /// over the network — so the parser had never been exercised by anything
    /// runnable offline. Seventh config in the increment and the most extreme
    /// coverage case: not a collapsed axis, not a partially-covered rule, but
    /// an EMPTY corpus. **All 4 fixtures are ADDED; none is pre-existing.**
    ///
    /// | behaviour                          | pre-existing fixture? |
    /// |------------------------------------|-----------------------|
    /// | anything at all                    | NO — there were none  |
    const DIFFERENTIAL_CORPUS: &[(&str, &str)] = &[
        (
            "ADDED: phi-2 shaped, explicit head_dim equal to the quotient",
            r#"{
                "vocab_size": 51200, "hidden_size": 2560, "num_hidden_layers": 32,
                "num_attention_heads": 32, "intermediate_size": 10240,
                "head_dim": 80, "layer_norm_eps": 1e-5, "rope_theta": 10000.0,
                "partial_rotary_factor": 0.4
            }"#,
        ),
        (
            "ADDED: minimal — every optional absent, head_dim derived",
            r#"{
                "vocab_size": 51200, "hidden_size": 1024, "num_hidden_layers": 16,
                "num_attention_heads": 16, "intermediate_size": 4096
            }"#,
        ),
        (
            "ADDED: explicit head_dim 96 != 2560/32 = 80, and rotary_dim CHAINS off it",
            r#"{
                "vocab_size": 51200, "hidden_size": 2560, "num_hidden_layers": 32,
                "num_attention_heads": 32, "intermediate_size": 10240,
                "head_dim": 96, "partial_rotary_factor": 0.4
            }"#,
        ),
        (
            "ADDED: odd rotary_dim — both paths must REJECT (0.5 * 10 = 5)",
            r#"{
                "vocab_size": 100, "hidden_size": 40, "num_hidden_layers": 2,
                "num_attention_heads": 4, "intermediate_size": 80,
                "partial_rotary_factor": 0.5
            }"#,
        ),
    ];

    #[test]
    fn serde_path_agrees_with_the_legacy_parser_on_every_fixture() {
        assert_eq!(DIFFERENTIAL_CORPUS.len(), 4, "corpus shrank");
        for (name, json) in DIFFERENTIAL_CORPUS {
            let new = PhiConfig::from_hf_json_str(json);
            let old = PhiConfig::from_hf_json_str_legacy(json);
            match (new, old) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "differential mismatch on {name}"),
                (Err(_), Err(_)) => {}
                (Ok(_), Err(e)) => panic!("{name}: serde accepted, legacy rejected: {e}"),
                (Err(e), Ok(_)) => panic!("{name}: serde rejected, legacy accepted: {e}"),
            }
        }
    }

    #[test]
    fn rotary_dim_chains_off_the_resolved_head_dim() {
        // The ordering constraint, made falsifiable. With head_dim explicit at
        // 96, rotary_dim must be round(0.4 * 96) = 38. If the chain read the
        // QUOTIENT instead (2560/32 = 80) it would be round(0.4 * 80) = 32.
        // Both are even, so the evenness check cannot tell them apart — only
        // this assertion can.
        let cfg = PhiConfig::from_hf_json_str(DIFFERENTIAL_CORPUS[2].1).unwrap();
        assert_eq!(cfg.head_dim, 96, "explicit head_dim must survive");
        assert_eq!(
            cfg.rotary_dim, 38,
            "rotary_dim must chain off the RESOLVED head_dim"
        );
        assert_ne!(
            cfg.rotary_dim, 32,
            "32 is what reading the raw quotient would give"
        );
    }

    #[test]
    fn odd_rotary_dim_is_rejected_by_both_paths() {
        let json = DIFFERENTIAL_CORPUS[3].1;
        assert!(
            PhiConfig::from_hf_json_str(json).is_err(),
            "0.5 * 10 = 5 is odd"
        );
        assert!(
            PhiConfig::from_hf_json_str_legacy(json).is_err(),
            "legacy must agree"
        );
    }
}

#[cfg(test)]
mod phi_kv_context_tests {
    use super::*;
    use fuel_core::inference_context::{InferenceContext, KvCache};

    // Parity with the retired Phi `_gpu_on` family
    // (`forward_with_cache_gpu_on` prefill + decode logits;
    // `generate_streaming_gpu_on` greedy token sequences) was
    // confirmed by `phi_forward_with_kv_context_matches_legacy_gpu_on`
    // and `phi_generate_with_kv_context_matches_legacy_generate`
    // immediately before retirement (commit 03df5c49); those tests
    // retired together with the legacy methods they referenced.

    /// Build tiny Phi-2-shaped weights (Split QKV + biases everywhere,
    /// partial RoPE) for kv-context forward tests.
    fn make_tiny_phi(cfg: &PhiConfig, seed: u32) -> PhiWeights {
        let mut s: u32 = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let d = cfg.dim;
        let kv_dim = cfg.n_heads * cfg.head_dim;
        PhiWeights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * d),
            layers: (0..cfg.n_layers)
                .map(|_| PhiLayerWeights {
                    attn_qkv: PhiQkv::Split {
                        q: vec_of(d * d).into(),
                        q_bias: vec_of(d),
                        k: vec_of(d * kv_dim).into(),
                        k_bias: vec_of(kv_dim),
                        v: vec_of(d * kv_dim).into(),
                        v_bias: vec_of(kv_dim),
                    },
                    attn_dense: vec_of(d * d).into(),
                    attn_dense_bias: vec_of(d),
                    mlp_fc1: vec_of(d * cfg.ffn_dim).into(),
                    mlp_fc1_bias: vec_of(cfg.ffn_dim),
                    mlp_fc2: vec_of(cfg.ffn_dim * d).into(),
                    mlp_fc2_bias: vec_of(d),
                    norm_gain: Arc::from(vec![1.0_f32; d]),
                    norm_bias: vec_of(d),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0_f32; d]),
            final_norm_bias: vec_of(d),
            output: vec_of(d * cfg.vocab_size).into(),
            output_bias: Some(vec_of(cfg.vocab_size)),
        }
    }

    fn tiny_cfg() -> PhiConfig {
        PhiConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            layer_norm_eps: 1e-5,
            rope_base: 10000.0,
            partial_rotary_factor: 0.5,
            rotary_dim: 2,
            tie_word_embeddings: false,
        }
    }

    /// KV-cache self-consistency on the new path: a monolithic prefill
    /// over N tokens must produce the same last-position logits as a
    /// shorter prefill followed by single-token decode steps through
    /// the same positions. Catches cache-position bugs without
    /// referencing the legacy path (survives its retirement).
    #[test]
    fn phi_kv_context_decode_consistent_with_monolithic_prefill() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let tokens = [1_u32, 5, 9, 12];
        let device = Device::cpu();

        // Path A: monolithic prefill over all 4 tokens.
        let mut cache_a = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            tokens.len(),
            DType::F32,
            &device,
        )
        .expect("cache_a");
        let mut ctx_a = InferenceContext::new(device.clone());
        let expected = model
            .forward_with_kv_context(&tokens, &mut cache_a, &mut ctx_a)
            .expect("monolithic prefill");

        // Path B: prefill 3, decode 1.
        let mut cache_b = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            tokens.len(),
            DType::F32,
            &device,
        )
        .expect("cache_b");
        let mut ctx_b = InferenceContext::new(device.clone());
        model
            .forward_with_kv_context(&tokens[..3], &mut cache_b, &mut ctx_b)
            .expect("prefill B");
        let actual = model
            .forward_with_kv_context(&tokens[3..], &mut cache_b, &mut ctx_b)
            .expect("decode B");

        assert_eq!(actual.len(), expected.len());
        // Same O(ε) gemm accumulation-order band as the LLaMA
        // kv-context parity tests.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: chunked={a}, monolithic={b}, diff={diff}",
            );
        }
        assert_eq!(cache_a.cached_len, cache_b.cached_len);
    }

    /// Greedy generation through the Phi kv-context path: correct
    /// shape (prompt preserved, max_new appended, tokens in vocab)
    /// and fully deterministic across runs.
    #[test]
    fn phi_generate_with_kv_context_greedy_is_deterministic() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let prompt = [1_u32, 5, 9];
        let max_new = 8;
        let device = Device::cpu();

        let mut streamed = Vec::new();
        let run_a = model
            .generate_streaming_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &device,
                DType::F32,
                |t| streamed.push(t),
            )
            .expect("run a");
        let run_b = model
            .generate_with_kv_context(
                &prompt,
                max_new,
                SamplingStrategy::Greedy,
                None,
                &device,
                DType::F32,
            )
            .expect("run b");

        assert_eq!(run_a, run_b, "greedy generation must be deterministic");
        assert_eq!(run_a.len(), prompt.len() + max_new);
        assert_eq!(&run_a[..prompt.len()], &prompt);
        assert_eq!(
            streamed,
            &run_a[prompt.len()..],
            "callback sees exactly the new tokens"
        );
        for &t in &run_a {
            assert!((t as usize) < cfg.vocab_size, "token {t} out of vocab");
        }
    }

    /// `forward_with_kv_context` build-time validation: with_dims
    /// caches (no pre-allocated buffers) and capacity overflows are
    /// rejected with typed errors, not panics.
    #[test]
    fn phi_forward_with_kv_context_rejects_invalid_cache() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let device = Device::cpu();
        let mut ctx = InferenceContext::new(device.clone());

        // with_dims cache → typed error.
        let mut dims_cache = KvCache::with_dims(cfg.n_layers, cfg.n_heads, cfg.head_dim);
        let err = model
            .forward_with_kv_context(&[1, 2], &mut dims_cache, &mut ctx)
            .expect_err("with_dims cache must be rejected");
        assert!(format!("{err}").contains("with_capacity"));

        // Capacity overflow → typed error.
        let mut small_cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            2,
            DType::F32,
            &device,
        )
        .expect("small cache");
        model
            .forward_with_kv_context(&[1, 2], &mut small_cache, &mut ctx)
            .expect("fits exactly");
        let err = model
            .forward_with_kv_context(&[3], &mut small_cache, &mut ctx)
            .expect_err("overflow must be rejected");
        assert!(format!("{err}").contains("max_seq_len"));
    }

    /// Phase D · D4 correctness gate (the Phi mirror of the LlamaModel D1
    /// `forward_with_kv_context_decode_matches_non_cached_forward`).
    ///
    /// PhiModel has no non-cached `forward` reference, so — like the
    /// existing `phi_kv_context_decode_consistent_with_monolithic_prefill`
    /// — this compares the input-independent decode graph (write_slice_dyn
    /// at a symbolic offset + full-capacity attention + fixed-capacity
    /// mask) against a monolithic prefill over the same token history. A
    /// prefill (seq>1) + a seq==1 decode step exercise BOTH the multi-row
    /// and single-row shapes of the transformed `apply_layer_with_kv_writes`.
    /// Within the existing O(ε) gemm accumulation-order band the two paths
    /// must agree — masked positions contribute exactly 0, so the extra
    /// masked compute over `max_seq_len` (vs the live `total_seq`) is a
    /// no-op numerically.
    ///
    /// Born-red shape: if the write offset were baked concretely (breaking
    /// the symbolic path) or the fixed-capacity mask failed to null the
    /// stale tail, the decode logits would diverge from the monolithic
    /// prefill and this fails.
    #[test]
    fn phi_decode_matches_non_cached_forward() {
        let cfg = tiny_cfg(); // partial RoPE (rotary_dim=2, head_dim=4)
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };

        let prompt = [1_u32, 5, 9];
        let next_token = 12_u32;
        let full = [prompt[0], prompt[1], prompt[2], next_token];
        let device = Device::cpu();

        // Reference: monolithic prefill over all 4 tokens.
        let mut cache_ref = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            full.len(),
            DType::F32,
            &device,
        )
        .expect("with_capacity ref");
        let mut ctx_ref = InferenceContext::new(device.clone());
        let expected = model
            .forward_with_kv_context(&full, &mut cache_ref, &mut ctx_ref)
            .expect("monolithic prefill");

        // Input-independent path: prefill(3) then decode(1) through the
        // transformed apply_layer_with_kv_writes.
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            full.len(),
            DType::F32,
            &device,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(device);
        let _prefill = model
            .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
            .expect("prefill");
        assert_eq!(cache.cached_len, prompt.len());
        let actual = model
            .forward_with_kv_context(&[next_token], &mut cache, &mut ctx)
            .expect("decode");
        assert_eq!(cache.cached_len, full.len());
        assert_eq!(actual.len(), expected.len());

        // Same O(ε) gemm accumulation-order band as the other Phi kv-context
        // parity tests.
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / a.abs().max(b.abs()).max(1e-6);
            assert!(
                diff < 5e-3 || rel < 1e-2,
                "logit[{i}]: input-independent={a}, monolithic={b}, diff={diff}",
            );
        }
    }

    /// Phase D · D4 born-red gate for plan-once persistent decode (the Phi
    /// mirror of the LlamaModel
    /// `forward_with_kv_context_persistent_plan_once_matches_d1`).
    ///
    /// Drive [`PhiModel::forward_with_kv_context_persistent`] for ≥3 decode
    /// tokens (after a prefill) holding ONE `DecodeSession`, in lockstep
    /// against the D1 [`PhiModel::forward_with_kv_context`] rebuild path
    /// (a SECOND identical model + cache + ctx fed the identical token each
    /// step). Assert the three plan-once invariants:
    ///   (a) `optimize_calls_thread_local()` bumps **exactly once** across
    ///       all the decode tokens — the first persistent decode token
    ///       builds + optimizes the held session; tokens 2..N skip optimize
    ///       (reuse via the D2a prebuilt seam);
    ///   (b) each persistent token's logits are **exactly `==`** the D1
    ///       cached path on the same prefix — same plan → same kernels →
    ///       bit-exact (NOT epsilon);
    ///   (c) the held graph's node `len()` is **stable from token 2 onward**
    ///       (no per-token node growth).
    #[test]
    fn phi_persistent_plan_once_matches_d1() {
        let cfg = tiny_cfg(); // partial RoPE + parallel block + biases
        // Two byte-identical models (same seed): one drives the D2
        // persistent path, one the D1 rebuild path.
        let model_d2 = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let model_d1 = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };

        let prompt = [1_u32, 5, 9];
        let decode_tokens = [12_u32, 3, 7, 2]; // ≥3 decode tokens
        let max_seq_len = prompt.len() + decode_tokens.len();

        // --- D1 (rebuild) reference FIRST, in its own pass, so its
        // per-token re-plans don't pollute the optimize-count window we
        // measure around the D2 loop. ---
        let dev1 = Device::cpu();
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
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
            cfg.n_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev2,
        )
        .expect("with_capacity d2");
        let mut ctx2 = InferenceContext::new(dev2);
        let mut session: Option<fuel_core::inference_context::DecodeSession> = None;

        // Prefill (seq>1 → falls back to the rebuild path; NO session).
        let _ = model_d2
            .forward_with_kv_context_persistent(&prompt, &mut cache2, &mut ctx2, &mut session)
            .expect("d2 prefill");
        assert!(
            session.is_none(),
            "prefill (seq>1) must NOT build the held session"
        );

        // Decode ≥3 tokens through the persistent path ONLY. Snapshot the
        // thread-local optimize count just before the loop (isolated from
        // other suite threads' concurrent optimizes).
        let opt_before = fuel_core::pipelined_bridge::optimize_calls_thread_local();
        let mut len_at_token2: Option<usize> = None;

        for (i, &tok) in decode_tokens.iter().enumerate() {
            let d2 = model_d2
                .forward_with_kv_context_persistent(&[tok], &mut cache2, &mut ctx2, &mut session)
                .expect("d2 decode");

            // (b) bit-exact vs. the D1 cached path (same plan → same kernels).
            assert_eq!(
                d2, d1_expected[i],
                "persistent decode token {i} must be byte-identical to the D1 cached path",
            );

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
            "persistent decode must optimize EXACTLY ONCE across {} decode tokens \
             (the first builds the session; the rest skip optimize): {opt_before} -> {opt_after}",
            decode_tokens.len(),
        );

        assert_eq!(cache2.cached_len, max_seq_len);
        assert_eq!(cache1.cached_len, max_seq_len);
    }

    /// Phase D · D4 generate-loop integration (the Phi mirror of the
    /// LlamaModel `generate_loop_persistent_byte_exact_and_plans_once`).
    ///
    /// The plain PhiModel decode generate loops
    /// (`generate_streaming_with_kv_context` / `generate_with_kv_context`)
    /// now hold ONE plan-once `DecodeSession` and route every step through
    /// [`PhiModel::forward_with_kv_context_persistent`]. This is the
    /// end-to-end guard that the plan-once path is USED in production Phi
    /// generation and stays bit-exact vs the D1 rebuild path.
    ///
    /// Drives an explicit persistent generate loop (mirroring the wired
    /// production loop) against a SEPARATE D1 reference loop over the same
    /// inputs, asserting:
    ///   (a) the generated token sequence is **byte-identical** over N≥4
    ///       greedy tokens (greedy diverges on ANY logit drift — a strong
    ///       end-to-end guard);
    ///   (b) each step's **logits** are **exactly `==`** the D1 cached path;
    ///   (c) `optimize_calls_thread_local()` bumps **exactly 2** across
    ///       prefill + N decode (1 prefill fallback + 1 decode-session
    ///       build) regardless of N — plan-once at the loop level.
    /// GAP-029 increment 2b — is `attended_len_sym` actually UNREFERENCED?
    ///
    /// The shared-driver de-duplication rests on a claim made only in comments:
    /// Phi's session carries `attended_len_sym = SymId(1)` "for API parity but
    /// never referenced/bound", and Llama's own doc says its attended-length
    /// binding "is unreferenced on today's f32 decode graph (no flash arm)". If
    /// both are true, Phi adopting Llama's shared `per_token_sym_env` (which
    /// binds both symbols) is a no-op and one driver can serve both.
    ///
    /// **The byte-exact persistent tests CANNOT settle this.** If the symbol
    /// were referenced but bound to its usual value, byte-exactness passes and
    /// says nothing about referencedness — the comment and the passing test are
    /// the same evidence, and neither discriminates.
    ///
    /// The discriminating instrument is a **negative control on the binding**:
    /// bind the symbol to a deliberately WRONG value and look at the output.
    ///   - output changes  ⇒ referenced ⇒ the no-op claim is FALSE
    ///   - output identical ⇒ genuinely unreferenced (positive evidence, not a
    ///     comment's word)
    ///
    /// **A wrong-binding test that sees no change is vacuous on its own**, since
    /// "the perturbation never reached the graph" produces the identical result.
    /// So each arm below is paired with a positive control that perturbs
    /// `cached_len_sym` — a symbol the KV write demonstrably DOES reference —
    /// and requires the output to change. Only then does "unchanged" mean
    /// unreferenced rather than "the instrument is blind".
    ///
    /// Scope of the claim, per Llama's own qualifier: **F32, CPU, today's decode
    /// graph, no flash arm.** A bf16/f16 CUDA decode that offers the flash arm
    /// would reference `attended_len` and must re-run this control.
    #[test]
    fn phi_attended_len_sym_is_unreferenced_negative_control() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };
        let prompt = [1_u32, 5, 9];
        let max_seq_len = prompt.len() + 4;

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
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
            .forward_with_kv_context_persistent(&[12], &mut cache, &mut ctx, &mut session)
            .expect("build decode session");
        let s = session.as_ref().expect("session built");

        // Realize the SAME next token three times, varying ONLY the SymEnv.
        // `realize_token` takes &self and clones `base_cache` internally, so the
        // three calls cannot pollute each other.
        let cached_len = cache.cached_len;
        let next = [3_u32];
        let mk_data = || {
            model
                .build_token_rope_mask_arcs(&dev, cached_len, &next, s.max_seq_len())
                .expect("token data")
        };

        let mut env_ok = fuel_ir::SymEnv::new();
        env_ok
            .bind(s.cached_len_sym(), cached_len)
            .expect("bind cached_len");
        let baseline = s.realize_token(&dev, mk_data(), &env_ok).expect("baseline");

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
        // Proves `realize_token` is live and sensitive at all; without it,
        // "nothing changed" could mean the realize returns a cached or constant
        // result and every verdict below would be vacuous.
        let other = [6_u32];
        let data_other = model
            .build_token_rope_mask_arcs(&dev, cached_len, &other, s.max_seq_len())
            .expect("token data (different token)");
        let mut env_ok2 = fuel_ir::SymEnv::new();
        env_ok2
            .bind(s.cached_len_sym(), cached_len)
            .expect("bind cached_len");
        let perturbed_data = s
            .realize_token(&dev, data_other, &env_ok2)
            .expect("realize with different token");
        assert_ne!(
            perturbed_data, baseline,
            "control A FAILED: a DIFFERENT input token produced identical \
             logits, so realize_token is not responding to its inputs and no \
             verdict from this test means anything",
        );

        // --- control B: is the SymEnv consulted AT ALL on this path? ---
        let with_off = s.offset_node().is_some();
        let mut env_bad_cached = fuel_ir::SymEnv::new();
        env_bad_cached
            .bind(s.cached_len_sym(), cached_len + 1)
            .expect("bind cached_len (wrong on purpose)");
        let perturbed_cached = s
            .realize_token(&dev, mk_data(), &env_bad_cached)
            .expect("realize with wrong cached_len");

        // See the Llama twin for the full reasoning: on the device-offset path
        // the KV write offset rides a device-resident BUFFER, so cached_len_sym
        // is expected inert; on the SymEnv path it drives the write and MUST
        // move the output. Asserting the direction that matches the path keeps
        // this a control rather than a coin flip.
        if with_off {
            assert_eq!(
                perturbed_cached, baseline,
                "on the device-offset path the KV offset rides the offset \
                 BUFFER, so a wrong cached_len_sym should be inert — it moved \
                 the output, meaning the symbol IS load-bearing here",
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
            "attended_len_sym IS referenced by Phi's decode graph — the \
             'carried for API parity but never referenced' comment at \
             lazy.rs:12088 is FALSE, and sharing per_token_sym_env between \
             Llama and Phi is NOT the no-op the GAP-029 2b design assumes",
        );

        // MEASURED 2026-08-12: Phi runs the **SymEnv path** on CPU
        // (`offset_node().is_none()`), so control B above took the `assert_ne`
        // branch and PASSED — a wrong `cached_len_sym` does move Phi's output.
        // The SymEnv is therefore demonstrably LIVE here, which is what makes
        // the `attended_len` verdict non-vacuous: it is inert while a sibling
        // symbol in the same env is load-bearing.
        //
        // So for Phi this is positive evidence, not a comment's word:
        // `attended_len_sym` is genuinely unreferenced, and adopting the shared
        // `per_token_sym_env` (which binds it) is a real no-op.
        eprintln!(
            "[gap-029 2b control] phi: offset_node.is_some()={with_off} \
             ({} path; SymEnv live={})",
            if with_off { "device-offset" } else { "SymEnv" },
            !with_off,
        );
    }

    /// It ALSO drives the real production wrapper `generate_with_kv_context`
    /// and asserts the returned token sequence matches the reference.
    #[test]
    fn phi_generate_loop_persistent_byte_exact_and_plans_once() {
        let cfg = tiny_cfg();
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7777),
        };

        let prompt = [1_u32, 5, 9];
        let max_new = 5; // N ≥ 4 greedy decode tokens
        let max_seq_len = prompt.len() + max_new;
        let strategy = SamplingStrategy::Greedy;

        // ---- D1 (rebuild) REFERENCE loop FIRST, in its own pass. Greedy
        // sampling open-coded with `sample_logits` so it is bit-identical to
        // the persistent loop's sampling. ----
        let dev1 = Device::cpu();
        let mut cache1 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
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

        // ---- D2 (persistent) generate loop — mirrors the wired production
        // loop. Snapshot the thread-local optimize count around the WHOLE
        // loop (prefill + decode). ----
        let opt_before = fuel_core::pipelined_bridge::optimize_calls_thread_local();

        let dev2 = Device::cpu();
        let mut cache2 = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
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

        // (a) Byte-identical token sequence over N greedy tokens.
        assert_eq!(
            d2_tokens, ref_tokens,
            "persistent generate loop must produce the byte-identical token sequence \
             as the D1 rebuild path over {max_new} greedy tokens",
        );

        // (b) Each step's logits exactly == the D1 cached path (bit-exact).
        assert_eq!(d2_step_logits.len(), ref_step_logits.len());
        for (i, (d2, d1)) in d2_step_logits
            .iter()
            .zip(ref_step_logits.iter())
            .enumerate()
        {
            assert_eq!(
                d2, d1,
                "persistent decode step {i} logits must be byte-identical to the D1 cached path",
            );
        }

        // (c) optimize bumped exactly twice (1 prefill fallback + 1
        // decode-session build) regardless of N.
        assert_eq!(
            opt_after - opt_before,
            2,
            "persistent generate must optimize EXACTLY twice (1 prefill fallback + 1 \
             decode-session build) regardless of N={max_new} decode tokens: \
             {opt_before} -> {opt_after}",
        );

        assert!(session.is_some(), "held session survives the decode loop");
        assert_eq!(cache2.cached_len, max_seq_len);
        assert_eq!(cache1.cached_len, max_seq_len);

        // ---- Drive the REAL production wrapper and confirm the wiring. ----
        let via_wrapper = model
            .generate_with_kv_context(&prompt, max_new, strategy, None, &Device::cpu(), DType::F32)
            .expect("generate_with_kv_context");
        assert_eq!(
            via_wrapper, ref_tokens,
            "generate_with_kv_context (wired to the persistent path) must produce the \
             byte-identical token sequence as the D1 reference",
        );
    }

    /// **WHICH Phi decode nodes land on the host?** — diagnostic for the
    /// capture rejection (`cross-device Op::Copy ... target Cpu`).
    ///
    /// Capture records CUDA operations issued to a stream. Host code is not a
    /// CUDA operation, so it cannot be recorded — and a cross-device `Copy` is
    /// the SYMPTOM of host compute sitting mid-graph, not the cause. This says
    /// which node(s) actually caused it, which decides the fix: a load-bearing
    /// host op means capture genuinely cannot apply; an accidentally-placed one
    /// means the fix is placement, and capture then works for free.
    ///
    /// Reports rather than asserts. The one thing it DOES assert is that
    /// placement was computed (`has_placements`) — an all-`None` dump would
    /// read as "nothing is on CUDA", a wrong answer wearing a null answer's
    /// clothes.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn phi_decode_node_placement_report_cuda() {
        use std::collections::BTreeMap;

        let cfg = tiny_cfg();
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
        let model = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7),
        };
        let prompt = [1_u32, 2, 3];
        let max_seq_len = prompt.len() + 4;

        let mut cache = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut sess: Option<fuel_core::inference_context::DecodeSession> = None;
        model
            .forward_with_kv_context_persistent(&prompt, &mut cache, &mut ctx, &mut sess)
            .expect("prefill");
        model
            .forward_with_kv_context_persistent(&[4], &mut cache, &mut ctx, &mut sess)
            .expect("first decode token builds the held plan");
        let s = sess.as_ref().expect("held session");

        let opt = s.optimized();
        assert!(
            opt.has_placements(),
            "placement NOT COMPUTED — every lookup would be None and the report below would describe a missing instrument, not the graph",
        );

        let g = s.graph().read().expect("graph lock");
        let mut by_op: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        let mut host_nodes: Vec<String> = Vec::new();
        for i in 0..g.len() {
            let id = fuel_graph::NodeId(i);
            let op_name = match &g.node(id).op {
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
            if place.starts_with("Cpu") {
                host_nodes.push(format!("#{i} {op_name}"));
            }
            *by_op.entry(op_name).or_default().entry(place).or_insert(0) += 1;
        }

        println!(
            "
=== Phi decode plan: node placement (CUDA) ==="
        );
        println!("total nodes: {}", g.len());
        for (op, places) in &by_op {
            let r: Vec<String> = places.iter().map(|(d, n)| format!("{d}x{n}")).collect();
            println!("  {op:<34} {}", r.join("  "));
        }
        println!(
            "
--- HOST-PLACED NODES ({}) ---",
            host_nodes.len()
        );
        for h in &host_nodes {
            println!("    {h}");
        }
        println!(
            "--- END ---
"
        );
    }

    /// **GPU gate for Phi's CapturedRun decode** — the mirror of
    /// `forward_with_kv_context_captured_matches_persistent` (Llama).
    ///
    /// Phi's capture shipped default-on verified only by `cargo check`, and its
    /// captured entry is `#[cfg(feature = "cuda")]` — so every CPU test in this
    /// crate passes with it arbitrarily broken. This closes that.
    ///
    /// **The specific failure it guards is silent.** Phi uses PARTIAL rotary:
    /// its RoPE tables are sized for `rotary_dim`, not `head_dim`. The replay
    /// path serialises those tables to raw bytes and H2Ds them into fixed
    /// capture buffers, so a wrong-but-well-formed table would write the WRONG
    /// VALUES into the RIGHT BUFFERS — correct shapes, no error, fluent wrong
    /// tokens. Byte-exactness against the persistent path is the only thing
    /// that catches it.
    ///
    /// Drives >= 4 decode tokens so all four driver branches run: token 1
    /// builds the held `DecodeSession`; token 2 builds the
    /// `CapturedDecodeSession` and returns its logits from an empty-`updates`
    /// warm replay; tokens 3-4 are pure `cuGraphLaunch` replays fed by freshly
    /// serialised per-token bytes — which is where a partial-rotary mistake
    /// would surface and nowhere earlier.
    ///
    /// Bit-exact, not epsilon: same plan, same kernels, same bytes.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a live CUDA device"]
    fn phi_forward_with_kv_context_captured_matches_persistent() {
        let cfg = tiny_cfg();

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
        let decode_tokens = [4_u32, 5, 6, 7]; // >= 4 so every branch runs
        let max_seq_len = prompt.len() + decode_tokens.len();

        // Two models with byte-identical weights (same seed): one drives the
        // reference persistent path, one the captured path under test.
        let model_ref = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7),
        };
        let model_cap = PhiModel {
            config: cfg.clone(),
            weights: make_tiny_phi(&cfg, 7),
        };

        // --- Reference: forward_with_kv_context_persistent ---
        let mut cache_ref = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity ref");
        let mut ctx_ref = InferenceContext::new(dev.clone());
        let mut sess_ref: Option<fuel_core::inference_context::DecodeSession> = None;
        model_ref
            .forward_with_kv_context_persistent(
                &prompt,
                &mut cache_ref,
                &mut ctx_ref,
                &mut sess_ref,
            )
            .expect("ref prefill");
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(decode_tokens.len());
        for &tok in &decode_tokens {
            ref_logits.push(
                model_ref
                    .forward_with_kv_context_persistent(
                        &[tok],
                        &mut cache_ref,
                        &mut ctx_ref,
                        &mut sess_ref,
                    )
                    .expect("ref decode"),
            );
        }

        // --- Under test: forward_with_kv_context_captured ---
        let mut cache_cap = KvCache::with_capacity(
            cfg.n_layers,
            cfg.n_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("with_capacity captured");
        let mut ctx_cap = InferenceContext::new(dev.clone());
        let mut sess_cap: Option<fuel_core::inference_context::DecodeSession> = None;
        let mut captured: Option<fuel_dispatch::pipelined::CapturedDecodeSession> = None;
        model_cap
            .forward_with_kv_context_captured(
                &prompt,
                &mut cache_cap,
                &mut ctx_cap,
                &mut sess_cap,
                &mut captured,
            )
            .expect("captured prefill");
        assert!(
            captured.is_none(),
            "prefill (seq != 1) must NOT build a capture"
        );

        for (i, &tok) in decode_tokens.iter().enumerate() {
            let got = model_cap
                .forward_with_kv_context_captured(
                    &[tok],
                    &mut cache_cap,
                    &mut ctx_cap,
                    &mut sess_cap,
                    &mut captured,
                )
                .expect("captured decode");

            assert_eq!(
                got, ref_logits[i],
                "Phi captured decode token {i} must be BYTE-IDENTICAL to the persistent path (same plan => same kernels). A mismatch here on tokens 3+ is the partial-rotary replay-bytes failure this test exists for.",
            );

            if i == 0 {
                assert!(sess_cap.is_some(), "token 1 builds the held session");
                assert!(captured.is_none(), "token 1 must NOT build the capture yet");
            }
        }

        // Both caches advanced identically — the fallback path must bump the
        // cache exactly as the captured path would.
        assert_eq!(cache_cap.cached_len, max_seq_len);
        assert_eq!(cache_ref.cached_len, max_seq_len);

        // --- Report whether capture actually formed (state, not verdict) ---
        //
        // Byte-exactness above is the CLAIM and is asserted unconditionally.
        // Whether the capture forms is a property of the model's GRAPH, and
        // Phi's is currently NOT capturable: its decode plan carries 3 fused
        // `LayerNormLastDim` and 4 fused `ROPE` nodes with no CUDA binding, so
        // they land on the host and force the cross-device `Op::Copy` that
        // `capture_decode` rejects (measured — see
        // `phi_decode_node_placement_report_cuda`). Llama captures because its
        // rope runs DECOMPOSED on CUDA instead of as a fused node.
        //
        // So this asserts the INVARIANT that holds either way — capture is an
        // optimization, and declining it must never change the answer — and
        // reports which branch ran. If a future registration puts LayerNorm and
        // ROPE on CUDA, capture will start forming and this test keeps passing
        // while the line below changes: a state report, not a frozen
        // expectation.
        if captured.is_some() {
            println!("Phi capture FORMED — the decode graph is fully CUDA-resident");
        } else {
            println!(
                "Phi capture DECLINED (expected today) — host-placed fused LayerNormLastDim/ROPE force a cross-device Copy. Results were byte-identical via the persistent fallback, which is the property that matters: capture degrades, never fails."
            );
        }
    }
}
