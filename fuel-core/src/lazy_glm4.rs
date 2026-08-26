// SPDX-License-Identifier: MIT OR Apache-2.0
//! GLM-4 (new architecture) decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. GLM-4 keeps the Llama-family overall
//! shape (RmsNorm, GQA, RoPE, SwiGLU FFN) and adds three
//! architectural twists worth honoring distinctly:
//!
//!   1. **Interleaved RoPE (`rope_i`)** — pairs are adjacent
//!      `(x_0, x_1), (x_2, x_3), …` rather than the standard
//!      split-half layout `(x_i, x_{i+d/2})` that
//!      [`fuel_graph::build_rope_tables`] /
//!      [`Tensor::rope_with_tables`] assume. We emulate
//!      the interleaved variant by **reshape-permuting** the
//!      input from `(..., d)` to `(..., d/2, 2)`, swapping the
//!      last two dims to `(..., 2, d/2)`, applying standard
//!      split-half RoPE on the resulting `(..., d)`, then
//!      reversing the permute. This is exactly equivalent to
//!      pair-adjacent rotation and avoids a new graph op.
//!   2. **Optional partial rotary** — `partial_rotary_factor`
//!      controls the rotated prefix per head, same as Phi /
//!      StableLM. The pass-through tail is untouched.
//!   3. **Four norms per block** — `input_layernorm` and
//!      `post_self_attn_layernorm` wrap the attention path;
//!      `post_attention_layernorm` and `post_mlp_layernorm`
//!      wrap the FFN path. Two residual sums, four norms.
//!   4. **Fused `gate_up_proj`** — a single linear
//!      `hidden → 2 * intermediate` is split into the gate and
//!      up halves (same pattern as Phi-3). FFN uses
//!      `down(act(gate) * up)`.
//!
//! Optional Q/K/V biases (`attention_bias`, default false) and
//! a tied lm_head (`tie_word_embeddings`) are supported via
//! flag fields. v1 supports the GLM-4 default configuration:
//! SwiGLU activation, full or partial rotary, no sliding window.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache
//! (recomputes each call), F32. The mask is causal-only (no
//! sliding window — the config's `sliding_window` is read but
//! ignored in v1, mirroring the eager GLM-4 default).

use crate::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use crate::lazy::{Tensor, WeightStorage};
use crate::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel,
};
use crate::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm4Activation {
    Silu,
    Gelu,
    GeluPytorchTanh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Glm4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Fraction of `head_dim` to rotate. `1.0` = full rotary
    /// (default for GLM-4-9B).
    pub partial_rotary_factor: f64,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_activation: Glm4Activation,
    pub tie_word_embeddings: bool,
}

impl Glm4Config {
    pub fn rope_dim(&self) -> usize {
        let r = (self.partial_rotary_factor * self.head_dim as f64) as usize;
        r & !1 // RoPE expects even
    }
}

#[derive(Debug, Clone)]
pub struct Glm4LayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub post_self_attn_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub post_mlp_norm_gain: Arc<[f32]>,

    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage, // never has bias in GLM-4

    /// Fused gate+up: `[hidden, 2*intermediate]`. First half is
    /// the gated path, second half is the up path.
    pub ffn_gate_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Glm4Weights {
    /// Identity of this weight set, folded into [`Glm4Model::decode_shape_key`]
    /// so a held decode plan (which bakes these weights as graph Consts) is
    /// never reused for a different Glm4 model that happens to share a config.
    /// Under-keying here is a silent wrong answer, not a slowdown.
    pub instance: crate::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Glm4LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    /// If `tie_word_embeddings`, the caller passes `None` and
    /// `token_embedding` is reused as the lm_head matrix.
    pub lm_head: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct Glm4Model {
    pub config: Glm4Config,
    pub weights: Glm4Weights,
}

impl Glm4Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. GLM4 does NOT scale embeddings.
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(&self, anchor: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let lm_head_w = match &self.weights.lm_head {
            Some(w) => w.clone(),
            None => WeightStorage::F32(self.weights.token_embedding.clone()),
        };
        lm_head_w.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Glm4Model::forward: tokens must be non-empty");

        let h = Tensor::embed_tokens(
            weights.token_embedding.clone(),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "Glm4Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(
                crate::Error::Msg("Glm4Model::forward_embeds: seq must be > 0".into()).bt(),
            );
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(crate::Error::Msg(
                "Glm4Config: num_attention_heads must be a multiple of num_key_value_heads".into(),
            )
            .bt());
        }
        let rope_dim = cfg.rope_dim();
        if rope_dim == 0 || rope_dim > cfg.head_dim || !rope_dim.is_multiple_of(2) {
            return Err(crate::Error::Msg(format!(
                "Glm4Config: rope_dim ({rope_dim}) must be even and in (0, head_dim ({})]",
                cfg.head_dim,
            ))
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, start_pos, seq, rope_dim);

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &Glm4LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        // ---- Attention sublayer ---------------------------------------------
        let residual = x.clone();
        let x_norm = x.rms_norm_affine(
            std::sync::Arc::clone(&layer.input_norm_gain),
            cfg.rms_norm_eps,
        )?;

        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, q_dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Interleaved RoPE on the first `rope_dim` features.
        let q_r = apply_interleaved_partial_rope(&q, rope_cos, rope_sin, cfg.head_dim, rope_dim)?;
        let k_r = apply_interleaved_partial_rope(&k, rope_cos, rope_sin, cfg.head_dim, rope_dim)?;

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Strict causal mask.
        let mask = Tensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, q_dim, cfg.hidden_size)?;
        let attn_normed = attn_out.rms_norm_affine(
            std::sync::Arc::clone(&layer.post_self_attn_norm_gain),
            cfg.rms_norm_eps,
        )?;
        let h1 = residual.add(&attn_normed)?;

        // ---- FFN sublayer ---------------------------------------------------
        let residual2 = h1.clone();
        let h1_norm = h1.rms_norm_affine(
            std::sync::Arc::clone(&layer.post_attn_norm_gain),
            cfg.rms_norm_eps,
        )?;

        // Fused gate_up: [hidden, 2 * intermediate]. Split last dim.
        let gate_up =
            layer
                .ffn_gate_up
                .apply_linear(&h1_norm, cfg.hidden_size, 2 * cfg.intermediate_size)?;
        let gate = gate_up.slice(2_usize, 0, cfg.intermediate_size)?;
        let up = gate_up.slice(2_usize, cfg.intermediate_size, cfg.intermediate_size)?;
        let activated = match cfg.hidden_activation {
            Glm4Activation::Silu => gate.silu(),
            Glm4Activation::Gelu => gate.gelu_erf(),
            Glm4Activation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size)?;
        let ffn_normed = ffn_out.rms_norm_affine(
            std::sync::Arc::clone(&layer.post_mlp_norm_gain),
            cfg.rms_norm_eps,
        )?;
        residual2.add(&ffn_normed)
    }

    // ---- Persistent KV-context decode (GAP-029 family 5) --------------------

    /// Decode/prefill through a pre-allocated [`KvCache`], rebuilding the graph
    /// each step. The primitive the persistent path falls back to on `seq != 1`.
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        crate::persistent_decode::forward_with_kv_context(self, tokens, cache, ctx, false, None)
    }

    /// Plan-once persistent decode: the first `seq == 1` token builds and
    /// optimizes the graph, every later token rebinds data and skips optimize.
    /// `seq != 1` falls back to [`Self::forward_with_kv_context`].
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> Result<Vec<f32>> {
        crate::persistent_decode::forward_with_kv_context_persistent(
            self, tokens, cache, ctx, session, None,
        )
    }

    /// [`Self::forward_with_kv_context_persistent`] at the ergonomic call shape
    /// — the session rides in the `InferenceContext`, so a hand-rolled decode
    /// loop gets plan reuse without knowing `DecodeSession` exists.
    pub fn forward_decode_step(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        let mut session = ctx.take_decode_session();
        let out = self.forward_with_kv_context_persistent(tokens, cache, ctx, &mut session);
        ctx.put_decode_session(session);
        out
    }

    /// Identity of a held decode plan for THIS model: family + the config values
    /// that change graph *structure* + this weight set.
    ///
    /// `rope_dim` is folded in because it sets the graph's rope-table shape;
    /// `rope_theta` / `partial_rotary_factor` are NOT — the theta is per-token
    /// data (the tables are rebound every step) and the factor only reaches the
    /// graph through `rope_dim`, already mixed. `instance` distinguishes two
    /// Glm4 models that share a config: the held plan bakes these weights, so
    /// under-keying it is a silent wrong answer, not a slowdown.
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = crate::decode_shape::ShapeKeyHasher::new();
        h.mix_str("glm4")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim as u64)
            .mix_u64(cfg.rope_dim() as u64)
            .mix_u64(cfg.hidden_size as u64)
            .mix_u64(cfg.intermediate_size as u64)
            .mix_u64(cfg.vocab_size as u64)
            .mix_f64(cfg.rms_norm_eps);
        self.decode_mask_plan().mix_into(&mut h);
        h.finish()
    }

    /// Glm4 has **no sliding window** (its config's `sliding_window` is unused),
    /// so every layer runs the strict-causal mask — a single dense variant
    /// (N=1). No `MaskPlan::split_window`, and the N=1 stacked mask collapses
    /// byte-identically to the plain dense builder.
    pub fn decode_mask_plan(&self) -> MaskPlan {
        MaskPlan::dense(self.config.num_hidden_layers)
    }

    /// One Glm4 layer against the pre-allocated KV buffers — the decode twin of
    /// [`Self::apply_layer`]. This step's K/V slab is written at the runtime
    /// offset, then attention reads the **full fixed-capacity** buffers under
    /// the seam's `[1, 1, seq, max_seq_len]` mask. Nothing in the graph's shape
    /// depends on `cached_len`, which is what makes the step reusable per token.
    ///
    /// Same math as the prefill [`Self::apply_layer`] — interleaved partial
    /// rotary, biased Q/K/V, GLM-4's four sandwich norms, fused-gate SwiGLU —
    /// with the seam's two deliberate decode differences (shared by every ported
    /// family):
    ///
    /// - **GQA is left to `matmul`'s head broadcast**, not `repeat_interleave`:
    ///   replicating K/V here would expand the whole `max_seq_len` cache every
    ///   token, not just this step's row.
    /// - **No flash-decode arm.** Glm4 has no windowed layer so the arm would be
    ///   numerically fine, but the seam declines it uniformly and this port does
    ///   not special-case that.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_kv_writes(
        &self,
        x: &Tensor,
        layer: &Glm4LayerWeights,
        k_cache_const: &Tensor,
        v_cache_const: &Tensor,
        cached_len_sym: fuel_ir::SymId,
        offset: Option<&Tensor>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let act_dtype = x.dtype();

        // ---- Attention sublayer ---------------------------------------------
        let residual = x.clone();
        let x_norm = x.rms_norm_affine(Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;

        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, q_dim)?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let q_h = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k_h = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v_h = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        // Interleaved partial RoPE on the first `rope_dim` features, run in f32
        // (its build-time requirement); the casts are no-ops at f32 activation.
        let q_r = apply_interleaved_partial_rope(
            &q_h.to_dtype(DType::F32)?,
            rope_cos,
            rope_sin,
            head_dim,
            rope_dim,
        )?
        .to_dtype(act_dtype)?;
        let k_r = apply_interleaved_partial_rope(
            &k_h.to_dtype(DType::F32)?,
            rope_cos,
            rope_sin,
            head_dim,
            rope_dim,
        )?
        .to_dtype(act_dtype)?;

        // Write this step's K/V into the pre-allocated buffers at the runtime
        // offset. K is post-RoPE, V is raw; GQA is NOT replicated here (the cache
        // holds `num_key_value_heads`, matmul broadcasts to Q's heads).
        let write_ranges = vec![
            (0, batch),
            (0, cfg.num_key_value_heads),
            (0, seq),
            (0, head_dim),
        ];
        let (full_k, full_v) = match offset {
            // Device-resident offset (`Op::WriteSliceDoff`, CPU/CUDA): read at
            // kernel launch, so the step is CUDA-graph-capturable.
            Some(off) => (
                k_cache_const.write_slice_doff(&k_r, off, 2, write_ranges.clone())?,
                v_cache_const.write_slice_doff(&v_h, off, 2, write_ranges)?,
            ),
            // Backend-generic `SymEnv` offset (Vulkan). Bit-identical write.
            None => {
                let dyn_off = fuel_ir::DynScalar::Sym(cached_len_sym);
                (
                    k_cache_const.write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?,
                    v_cache_const.write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?,
                )
            }
        };

        let k_t = full_k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&full_v)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, q_dim, cfg.hidden_size)?;
        let attn_normed = attn_out.rms_norm_affine(
            Arc::clone(&layer.post_self_attn_norm_gain),
            cfg.rms_norm_eps,
        )?;
        let h1 = residual.add(&attn_normed)?;

        // ---- FFN sublayer ---------------------------------------------------
        let residual2 = h1.clone();
        let h1_norm =
            h1.rms_norm_affine(Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;
        let gate_up =
            layer
                .ffn_gate_up
                .apply_linear(&h1_norm, cfg.hidden_size, 2 * cfg.intermediate_size)?;
        let gate = gate_up.slice(2_usize, 0, cfg.intermediate_size)?;
        let up = gate_up.slice(2_usize, cfg.intermediate_size, cfg.intermediate_size)?;
        let activated = match cfg.hidden_activation {
            Glm4Activation::Silu => gate.silu(),
            Glm4Activation::Gelu => gate.gelu_erf(),
            Glm4Activation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_in = activated.mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&ffn_in, cfg.intermediate_size, cfg.hidden_size)?;
        let ffn_normed =
            ffn_out.rms_norm_affine(Arc::clone(&layer.post_mlp_norm_gain), cfg.rms_norm_eps)?;
        residual2.add(&ffn_normed)
    }
}

impl PersistentDecodeModel for Glm4Model {
    fn decode_n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn build_decode_token_data(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        session: &DecodeSession,
        cache: &KvCache,
        rope_inv_freq: Option<&[f64]>,
    ) -> Result<DecodeTokenData> {
        let host = crate::persistent_decode::compute_decode_token_host(
            self,
            cached_len,
            tokens,
            session.max_seq_len(),
            rope_inv_freq,
        );
        crate::persistent_decode::upload_decode_token_data(
            device,
            &host,
            cache.dtype.unwrap_or(DType::F32),
            // Only the device-offset path carries an offset buffer; a SymEnv
            // session gets its offset through `cached_len_sym` instead.
            session.offset_node().is_some().then_some(cached_len),
        )
    }
}

impl DecodeBackbone for Glm4Model {
    fn decode_family(&self) -> &'static str {
        "Glm4Model"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            // Partial rotary: the rope tables (and thus the graph's rope-table
            // shape) are `rope_dim = partial_rotary_factor * head_dim` wide, not
            // the full head. This single value is Glm4's entire seam-level
            // divergence from the LLaMA shape; the interleaved *application*
            // lives in `apply_layer_with_kv_writes`.
            rope_width: cfg.rope_dim(),
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        Glm4Model::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        Glm4Model::decode_mask_plan(self)
    }

    fn decode_rope_plan(&self) -> crate::persistent_decode::RopePlan {
        crate::persistent_decode::RopePlan::single(
            self.config.rope_theta,
            self.decode_dims().n_layers,
        )
    }

    fn decode_token_embedding(&self) -> Arc<[f32]> {
        self.weights.token_embedding.clone()
    }

    fn decode_apply_layer(
        &self,
        layer_idx: usize,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor> {
        self.apply_layer_with_kv_writes(
            inputs.x,
            &self.weights.layers[layer_idx],
            inputs.k_cache,
            inputs.v_cache,
            inputs.cached_len_sym,
            inputs.offset,
            inputs.rope_cos,
            inputs.rope_sin,
            inputs.mask,
        )
    }

    fn decode_final_norm_and_head(&self, h: &Tensor) -> Result<Tensor> {
        let h_norm = h.rms_norm_affine(
            Arc::clone(&self.weights.final_norm_gain),
            self.config.rms_norm_eps,
        )?;
        self.apply_lm_head(&h_norm)
    }
}

/// Apply interleaved RoPE (`rope_i`) to the first `rope_dim`
/// features of each head. `qk` is shaped `[batch, n_heads, seq, head_dim]`
/// and `rope_cos` / `rope_sin` are the standard split-half tables
/// shaped `[seq, rope_dim]` produced by `fuel_graph::build_rope_tables`.
///
/// The trick: reshape `(..., d)` to `(..., d/2, 2)`, permute the
/// last two dims, then `reshape(..., d)` — adjacent pairs become
/// "first half + second half" which exactly matches the standard
/// split-half RoPE convention. Reverse the permute afterward.
pub fn apply_interleaved_partial_rope(
    qk: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    head_dim: usize,
    rope_dim: usize,
) -> Result<Tensor> {
    if rope_dim == 0 {
        return Ok(qk.clone());
    }
    let shape = qk.shape();
    let dims = shape.dims();
    assert_eq!(dims.len(), 4);
    let batch = dims[0];
    let n_heads = dims[1];
    let seq = dims[2];
    let pass_dim = head_dim - rope_dim;

    // Split rotated prefix vs pass-through tail.
    let rot = qk.slice(3_usize, 0, rope_dim)?;
    let pass = if pass_dim > 0 {
        Some(qk.slice(3_usize, rope_dim, pass_dim)?)
    } else {
        None
    };

    // Permute (..., rope_dim) → (..., 2, rope_dim/2) by reshape + permute.
    let half = rope_dim / 2;
    let rot_pairs = rot.reshape(Shape::from_dims(&[batch, n_heads, seq, half, 2]))?;
    // Swap last two dims: (..., half, 2) → (..., 2, half).
    let rot_split = rot_pairs.permute([0, 1, 2, 4, 3_usize])?;
    // Flatten back to (..., rope_dim).
    let rot_flat = rot_split.reshape(Shape::from_dims(&[batch, n_heads, seq, rope_dim]))?;

    // Now standard split-half RoPE.
    let rotated = rot_flat.rope_with_tables(rope_cos, rope_sin)?;

    // Reverse: (..., rope_dim) → (..., 2, half) → (..., half, 2) → flatten.
    let rotated_split = rotated.reshape(Shape::from_dims(&[batch, n_heads, seq, 2, half]))?;
    let rotated_pairs = rotated_split.permute([0, 1, 2, 4, 3_usize])?;
    let rotated_flat = rotated_pairs.reshape(Shape::from_dims(&[batch, n_heads, seq, rope_dim]))?;

    match pass {
        None => Ok(rotated_flat),
        Some(pass_tensor) => rotated_flat.concat(&pass_tensor, 3_usize),
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl Glm4Weights {
    /// Load GLM-4 (THUDM/glm-4-9b-chat etc.) weights from HF safetensors.
    /// GLM-4 has split sandwich-norm structure: input_norm + post-self-attn
    /// + post-attn + post-mlp; fused gate_up_proj in MLP.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Glm4Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let opt_bias = |name: String| -> Option<Arc<[f32]>> {
            load_tensor_as_f32(st, &name).ok().map(Arc::from)
        };

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let input_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let post_self_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_self_attn_layernorm.weight"),
            )?);
            let post_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
            let post_mlp_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_mlp_layernorm.weight"),
            )?);
            let attn_q = ltm(st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            let attn_q_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.q_proj.bias"))
            } else {
                None
            };
            let attn_k = ltm(st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            let attn_k_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.k_proj.bias"))
            } else {
                None
            };
            let attn_v = ltm(st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            let attn_v_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.v_proj.bias"))
            } else {
                None
            };
            let attn_o = ltm(st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            let ffn_gate_up = ltm(st, &format!("{p}.mlp.gate_up_proj.weight"), 2 * inter, h)?;
            let ffn_down = ltm(st, &format!("{p}.mlp.down_proj.weight"), h, inter)?;
            layers.push(Glm4LayerWeights {
                input_norm_gain,
                post_self_attn_norm_gain,
                post_attn_norm_gain,
                post_mlp_norm_gain,
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                ffn_gate_up,
                ffn_down,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(ltm(st, "lm_head.weight", cfg.vocab_size, h)?)
        };

        Ok(Self {
            instance: crate::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            lm_head,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Glm4Config) -> Glm4Weights {
        let mut s: u32 = 67890;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<Glm4LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Glm4LayerWeights {
                input_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_self_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                post_mlp_norm_gain: Arc::from(vec![1.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                attn_q_bias: if cfg.attention_bias {
                    Some(vec_of(q_dim, &mut *nb))
                } else {
                    None
                },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: if cfg.attention_bias {
                    Some(vec_of(kv, &mut *nb))
                } else {
                    None
                },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: if cfg.attention_bias {
                    Some(vec_of(kv, &mut *nb))
                } else {
                    None
                },
                attn_o: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                ffn_gate_up: WeightStorage::F32(vec_of(h * (2 * i), &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)))
        };
        Glm4Weights {
            instance: crate::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            lm_head,
        }
    }

    fn tiny_config() -> Glm4Config {
        Glm4Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            partial_rotary_factor: 0.5, // rope_dim = 2
            attention_bias: false,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            hidden_activation: Glm4Activation::Silu,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn tied_embedding_lm_head() {
        let cfg = Glm4Config {
            tie_word_embeddings: true,
            ..tiny_config()
        };
        let weights = tiny_weights(&cfg);
        assert!(weights.lm_head.is_none());
        let model = Glm4Model {
            config: cfg.clone(),
            weights,
        };
        let logits = model.forward(&[2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 2 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn full_rotary() {
        let mut cfg = tiny_config();
        cfg.partial_rotary_factor = 1.0;
        assert_eq!(cfg.rope_dim(), cfg.head_dim);
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }

    /// Interleaved RoPE on rope_dim == head_dim — identical input
    /// shape but different rotation convention than split-half.
    /// Verify rotation is applied: zero RoPE tables (cos = 1, sin = 0)
    /// should be the identity; with non-trivial tables, output changes.
    #[test]
    fn interleaved_rope_is_applied() {
        let cfg = Glm4Config {
            num_hidden_layers: 1,
            partial_rotary_factor: 1.0,
            ..tiny_config()
        };
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        let dev = Device::cpu();
        // Build a (1, 1, 1, head_dim) tensor.
        let qk = Tensor::from_f32(
            Arc::from(
                (0..head_dim)
                    .map(|i| (i as f32 + 1.0) * 0.1)
                    .collect::<Vec<_>>(),
            ),
            Shape::from_dims(&[1, 1, 1, head_dim]),
            &dev,
        );
        // Identity-ish RoPE tables: cos=1, sin=0 ⇒ rotation is identity.
        let cos_id = qk.const_f32_like(
            Arc::from(vec![1.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let sin_id = qk.const_f32_like(
            Arc::from(vec![0.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let id_out = apply_interleaved_partial_rope(&qk, &cos_id, &sin_id, head_dim, rope_dim)
            .unwrap()
            .realize_f32();
        let in_data = qk.realize_f32();
        for (a, b) in in_data.iter().zip(id_out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "identity RoPE must round-trip: {a} vs {b}"
            );
        }

        // Non-trivial RoPE: cos=0, sin=1 ⇒ pair (x_0, x_1) becomes (-x_1, x_0).
        // For interleaved, this means: [x0, x1, x2, x3] → [-x1, x0, -x3, x2].
        let cos_rot = qk.const_f32_like(
            Arc::from(vec![0.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let sin_rot = qk.const_f32_like(
            Arc::from(vec![1.0_f32; rope_dim]),
            Shape::from_dims(&[1, rope_dim]),
        );
        let rot_out = apply_interleaved_partial_rope(&qk, &cos_rot, &sin_rot, head_dim, rope_dim)
            .unwrap()
            .realize_f32();
        // Expected: pair (a, b) → (-b, a) per interleaved RoPE convention.
        // in_data = [0.1, 0.2, 0.3, 0.4]
        // expected = [-0.2, 0.1, -0.4, 0.3]
        let expected: Vec<f32> = in_data
            .chunks(2)
            .flat_map(|pair| vec![-pair[1], pair[0]])
            .collect();
        for (a, e) in rot_out.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-5,
                "interleaved rotation: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = tiny_config();
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref
            .iter()
            .zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "GLM4 forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config();
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let bad = Tensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]),
            &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config();
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model
            .forward_hidden_embeds(&embeds, 0)
            .unwrap()
            .realize_f32();
        let max_diff = h_ref
            .iter()
            .zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "GLM4 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }

    // ==== GAP-029 family 5: persistent KV-context decode =====================

    /// Prefill `tokens[..prefill]`, then decode the rest one token at a time
    /// through the persistent path; return each decode step's logits.
    ///
    /// **`>= 3` decode steps are load-bearing, not padding.** One decode token
    /// exercises only the held-graph BUILD path; the per-token REBIND path —
    /// where this position's RoPE/mask bytes are recomputed into the held Const
    /// — is first reached on step 2. A 1-step test passes on a model that is
    /// wrong from token 2 onward.
    fn decode_steps(model: &Glm4Model, tokens: &[u32], prefill: usize) -> Vec<Vec<f32>> {
        let cfg = &model.config;
        let n_decode = tokens.len() - prefill;
        assert!(
            n_decode >= 3,
            "decode_steps: need >= 3 decode tokens to reach the rebind path (got {n_decode})",
        );
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            tokens.len(),
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<DecodeSession> = None;

        model
            .forward_with_kv_context_persistent(
                &tokens[..prefill],
                &mut cache,
                &mut ctx,
                &mut session,
            )
            .expect("prefill");
        assert!(
            session.is_none(),
            "prefill (seq > 1) must NOT build the held session"
        );

        let mut out = Vec::with_capacity(n_decode);
        for i in prefill..tokens.len() {
            out.push(
                model
                    .forward_with_kv_context_persistent(
                        &tokens[i..=i],
                        &mut cache,
                        &mut ctx,
                        &mut session,
                    )
                    .expect("decode"),
            );
            assert!(session.is_some(), "decode must hold a session from token 1");
        }
        assert_eq!(
            cache.cached_len,
            tokens.len(),
            "cache must advance every step"
        );
        out
    }

    /// Max |logit diff| between each decode step and the shipped prefill
    /// [`Glm4Model::forward`] at the same absolute position.
    ///
    /// `forward` is an INDEPENDENT correct reference: the born-red sabotage
    /// lives only in the decode layer (`apply_layer_with_kv_writes`), so this is
    /// an absolute oracle against unsabotaged code — NOT a relative A-vs-B over
    /// shared code, which would be blind to a defect both sides run.
    fn decode_vs_forward_max_abs(model: &Glm4Model, tokens: &[u32], prefill: usize) -> Vec<f32> {
        let cfg = &model.config;
        let steps = decode_steps(model, tokens, prefill);
        steps
            .iter()
            .enumerate()
            .map(|(k, got)| {
                let pos = prefill + k;
                let full = model.forward(&tokens[..=pos], 0).unwrap().realize_f32();
                let expected = &full[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
                assert_eq!(got.len(), expected.len());
                got.iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max)
            })
            .collect()
    }

    /// The oracle threshold — **measured, not inherited**, and set BETWEEN the
    /// correct drift and the sabotaged divergence.
    ///
    /// The ambient decode template (`diff < 5e-3 || rel < 1e-2`) is a trap here,
    /// and this port's own measurement proves it: the split-half rope sabotage
    /// diverges by only `1.74e-4`, which is `< 5e-3`, so the ambient tolerance
    /// would PASS a rope-swapped port — a vacuous oracle. (Same failure the
    /// Qwen2 lane measured cross-project; recorded in CLAUDE.md.)
    ///
    /// Measured on the full-rotary config (prefill 3, decode 3), decode vs the
    /// shipped prefill `forward`, via [`measure_glm4_decode_drift`]:
    ///
    /// ```text
    /// (a) correct interleaved decode : [0.0, 0.0, 0.0]                         (bit-exact)
    /// (b) split-half (sabotaged)     : [1.74e-4, 8.37e-5, 1.62e-4]  max 1.74e-4 (divergence)
    /// control (rope_dim == 2)        : [0.0, 0.0, 0.0] under BOTH bodies       (insensitive)
    /// ```
    ///
    /// `1e-5` sits ~17x below the divergence while leaving headroom above the
    /// bit-exact `0.0` for a legitimate future reassociation (a fused decode
    /// kernel, a different reduction order). The separation is total because (a)
    /// is exactly zero; the margin is tighter in absolute terms than Qwen2's
    /// (whose divergence was `7e-3`) only because this rope config diverges less.
    const DECODE_ORACLE_ABS: f32 = 1e-5;

    /// DISCRIMINATOR — full rotary (`rope_dim == head_dim`), where interleaved
    /// RoPE and the split-half convention DIFFER. Swapping the interleaved
    /// application in `apply_layer_with_kv_writes` for split-half reddens
    /// exactly this test (the born-red).
    #[test]
    fn glm4_decode_matches_forward_full_rotary() {
        let cfg = Glm4Config {
            partial_rotary_factor: 1.0,
            ..tiny_config()
        };
        assert_eq!(
            cfg.rope_dim(),
            cfg.head_dim,
            "this config must be full-rotary"
        );
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let drift = decode_vs_forward_max_abs(&model, &tokens, 3);
        for (k, d) in drift.iter().enumerate() {
            assert!(
                *d < DECODE_ORACLE_ABS,
                "full-rotary decode step {k} diverges from forward by {d} (>= {DECODE_ORACLE_ABS})",
            );
        }
    }

    /// NON-DISCRIMINATION CONTROL — read this first.
    ///
    /// At `rope_dim = 2` (head_dim 4, factor 0.5) the interleave permute has
    /// `half = 1` and is provably the IDENTITY, so interleaved RoPE and the
    /// split-half convention are byte-identical BY CONSTRUCTION. The rope
    /// sabotage that reddens the discriminator is a NO-OP here, so this stays
    /// GREEN under both bodies. A green isolates "the seam + port plumbing work"
    /// from "the rope pairing is right": if this ever fails, the discriminator
    /// proves nothing — the instrument would be measuring plumbing, not pairing.
    #[test]
    fn control_decode_matches_forward_at_rope_dim_2() {
        let cfg = tiny_config(); // factor 0.5, head_dim 4 => rope_dim 2
        assert_eq!(
            cfg.rope_dim(),
            2,
            "control requires rope_dim == 2 (permute is identity)"
        );
        let model = Glm4Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let drift = decode_vs_forward_max_abs(&model, &tokens, 3);
        for (k, d) in drift.iter().enumerate() {
            assert!(
                *d < DECODE_ORACLE_ABS,
                "control decode step {k} diverges from forward by {d}"
            );
        }
    }

    /// Prints the measured drift for both configs — run with `--nocapture` to
    /// read (a) and (b). Not an assertion; the discriminator/control carry those.
    #[test]
    fn measure_glm4_decode_drift() {
        let full = Glm4Config {
            partial_rotary_factor: 1.0,
            ..tiny_config()
        };
        let m_full = Glm4Model {
            config: full.clone(),
            weights: tiny_weights(&full),
        };
        let ctrl = tiny_config();
        let m_ctrl = Glm4Model {
            config: ctrl.clone(),
            weights: tiny_weights(&ctrl),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        println!(
            "GLM4-DRIFT full_rotary(rope_dim={})={:?} control(rope_dim={})={:?}",
            full.rope_dim(),
            decode_vs_forward_max_abs(&m_full, &tokens, 3),
            ctrl.rope_dim(),
            decode_vs_forward_max_abs(&m_ctrl, &tokens, 3),
        );
    }
}
