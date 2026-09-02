// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Qwen3 evolves Qwen2 with **per-head QK-norm**:
//! a RmsNorm is applied to Q and K AFTER the per-head reshape, along
//! the `head_dim` axis. Norm gains are `[head_dim]` (not
//! `[hidden_size]` like OLMo2). Otherwise identical to Qwen2:
//! GQA + RmsNorm + SwiGLU + RoPE + per-layer sliding-window gating
//! + optional Q/K/V/O biases.
//!
//! Reuses `fuel_core::lazy::LayerWeights` for the standard fields and
//! stores the per-head QK-norm gains in `Qwen3LayerExtras`.

use fuel_core::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use fuel_core::lazy::{LayerWeights, Tensor, WeightStorage};
use fuel_core::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel,
};
use fuel_core::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen3LayerExtras {
    /// `[head_dim]` — per-head RmsNorm gain for Q.
    pub q_norm_gain: Arc<[f32]>,
    /// `[head_dim]` — per-head RmsNorm gain for K.
    pub k_norm_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen3Weights {
    /// Process-unique identity for THIS weight set — what lets a held decode
    /// plan tell two same-architecture models apart (GAP-029). Mint with
    /// [`fuel_core::decode_shape::ModelInstanceId::next`].
    pub instance: fuel_core::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub layer_extras: Vec<Qwen3LayerExtras>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3Model {
    pub config: Qwen3Config,
    pub weights: Qwen3Weights,
}

impl Qwen3Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Qwen3-specific: per-layer sliding-window gate
    /// (`use_sliding_window && layer_idx < max_window_layers`)
    /// and Q/K-norm gains are honored.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips the token-embedding step and
    /// runs the decoder over pre-embedded inputs (e.g. concatenated
    /// vision + text embeddings for Qwen3-VL composition).
    ///
    /// `embeds` shape: `(1, seq, hidden_size)`. Unlike Gemma, Qwen
    /// does NOT scale embeddings by `sqrt(hidden_size)` — the
    /// caller passes raw embeddings.
    ///
    /// Returns logits `(1, seq, vocab_size)`.
    pub fn forward_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`]. Returns the
    /// post-final-RmsNorm states `(1, seq, hidden_size)` — used by
    /// LLaVA-style multimodal hosts that consume hidden states
    /// without the lm_head projection.
    pub fn forward_hidden_embeds(&self, embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder. Used by
    /// multimodal compositions to obtain text-side embeddings that
    /// will be concatenated with vision features before
    /// [`Self::forward_embeds`].
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
        self.weights
            .output
            .apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

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
            return Err(fuel_core::Error::Msg(format!(
                "Qwen3Model::forward_embeds: expected embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            ))
            .bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(fuel_core::Error::Msg(
                "Qwen3Model::forward_embeds: seq must be > 0".into(),
            )
            .bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(fuel_core::Error::Msg(
                "Qwen3Config: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        if weights.layers.len() != weights.layer_extras.len() {
            return Err(fuel_core::Error::Msg(format!(
                "Qwen3Weights: layers ({}) must have matching layer_extras ({})",
                weights.layers.len(),
                weights.layer_extras.len(),
            ))
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for (layer_idx, (layer, extras)) in weights
            .layers
            .iter()
            .zip(weights.layer_extras.iter())
            .enumerate()
        {
            let uses_window = cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, extras, &rope_cos, &rope_sin, uses_window)?;
        }

        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    fn build_layer_mask(&self, anchor: &Tensor, seq: usize, uses_window: bool) -> Tensor {
        let cfg = &self.config;
        let window = if uses_window {
            cfg.sliding_window.unwrap_or(seq + 1)
        } else {
            seq + 1
        };
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || j + window <= i {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        extras: &Qwen3LayerExtras,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        uses_window: bool,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = x.rms_norm_affine(
            std::sync::Arc::clone(&layer.attn_norm_gain),
            cfg.rms_norm_eps,
        )?;

        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)?
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

        // Per-head QK-norm: RmsNorm along the head_dim (last axis).
        let q = q.rms_norm_affine(std::sync::Arc::clone(&extras.q_norm_gain), cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine(std::sync::Arc::clone(&extras.k_norm_gain), cfg.rms_norm_eps)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_layer_mask(x, seq, uses_window);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?;

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(
            std::sync::Arc::clone(&layer.ffn_norm_gain),
            cfg.rms_norm_eps,
        )?;
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size)?;
        h1.add(&ffn_out)
    }
}

// ---- GAP-029 increment 3 · persistent-KV decode -----------------------------

/// The Qwen3-family attention block's weights + geometry, borrowed.
///
/// **Exists so Qwen3 and Qwen3Moe do not get two copies of one attention
/// block.** Their attention halves are identical — biased Q/K/V, per-head
/// QK-norm, GQA, explicit `head_dim` — and differ only in the *type* holding
/// the weights (`LayerWeights` + [`Qwen3LayerExtras`] versus
/// `Qwen3MoeLayerWeights`, which inlines the norm gains). Two hand-maintained
/// copies of a decode attention block is the reproduction mechanism GAP-029
/// increment 3 exists to avoid, so the differing part is a borrow and the
/// identical part is written once.
///
/// The FFN is deliberately **not** here: that is where the two families really
/// diverge (SwiGLU versus router + experts).
pub(crate) struct Qwen3AttnBlock<'a> {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub attn_norm_gain: &'a Arc<[f32]>,
    pub attn_q: &'a WeightStorage,
    pub attn_q_bias: Option<&'a Arc<[f32]>>,
    pub attn_k: &'a WeightStorage,
    pub attn_k_bias: Option<&'a Arc<[f32]>>,
    pub attn_v: &'a WeightStorage,
    pub attn_v_bias: Option<&'a Arc<[f32]>>,
    pub attn_o: &'a WeightStorage,
    /// `[head_dim]` per-head QK-norm gains — the Qwen3 addition over Qwen2.
    pub q_norm_gain: &'a Arc<[f32]>,
    pub k_norm_gain: &'a Arc<[f32]>,
}

/// Qwen3-family attention against the pre-allocated KV buffers, returning the
/// post-attention residual `x + attn_out` (the caller adds its own FFN).
///
/// This step's K/V slab is written at the runtime offset `cached_len`, then
/// attention reads the **full fixed-capacity** buffers under this layer's mask
/// variant — which excludes future positions, the unwritten tail, and (on a
/// windowed layer) everything older than the window.
///
/// Two deliberate differences from the prefill twin, both shared with Qwen2:
/// GQA rides `matmul`'s head broadcast instead of materialising
/// `repeat_interleave` over the whole `max_seq_len` cache; and **no
/// flash-decode arm is offered**, because the CUDA arm's single
/// `k_len = cached_len + seq` cannot represent a sliding window and would
/// silently drop it on bf16/CUDA.
pub(crate) fn qwen3_attn_with_kv_writes(
    blk: &Qwen3AttnBlock<'_>,
    inputs: &DecodeLayerInputs<'_>,
) -> Result<Tensor> {
    let x = inputs.x;
    let x_shape = x.shape();
    let dims = x_shape.dims();
    let batch = dims[0];
    let seq = dims[1];
    let kv_dim = blk.num_key_value_heads * blk.head_dim;
    let act_dtype = x.dtype();

    let x_norm = x.rms_norm_affine(Arc::clone(blk.attn_norm_gain), blk.rms_norm_eps)?;

    let q = blk
        .attn_q
        .apply_linear(&x_norm, blk.hidden_size, blk.hidden_size)?
        .add_optional_trailing_bias(blk.attn_q_bias)?;
    let k = blk
        .attn_k
        .apply_linear(&x_norm, blk.hidden_size, kv_dim)?
        .add_optional_trailing_bias(blk.attn_k_bias)?;
    let v = blk
        .attn_v
        .apply_linear(&x_norm, blk.hidden_size, kv_dim)?
        .add_optional_trailing_bias(blk.attn_v_bias)?;

    let q = q.split_heads(blk.num_attention_heads, blk.head_dim)?;
    let k = k.split_heads(blk.num_key_value_heads, blk.head_dim)?;
    let v_h = v.split_heads(blk.num_key_value_heads, blk.head_dim)?;

    // Per-head QK-norm: RmsNorm along head_dim, BEFORE RoPE — same order as
    // the prefill path, which is what makes decode agree with it.
    let q = q.rms_norm_affine(Arc::clone(blk.q_norm_gain), blk.rms_norm_eps)?;
    let k = k.rms_norm_affine(Arc::clone(blk.k_norm_gain), blk.rms_norm_eps)?;

    // RoPE runs in f32 (build-time requirement); no-op casts under f32 caches.
    let q_r = q
        .to_dtype(DType::F32)?
        .rope_with_tables(inputs.rope_cos, inputs.rope_sin)?
        .to_dtype(act_dtype)?;
    let k_r = k
        .to_dtype(DType::F32)?
        .rope_with_tables(inputs.rope_cos, inputs.rope_sin)?
        .to_dtype(act_dtype)?;

    let write_ranges = vec![
        (0, batch),
        (0, blk.num_key_value_heads),
        (0, seq), // axis-2 start is dynamic; width = seq
        (0, blk.head_dim),
    ];
    let (full_k, full_v) = match inputs.offset {
        // Device-resident offset (`Op::WriteSliceDoff`, CPU/CUDA) — read at
        // kernel launch, so the step stays CUDA-graph-capturable.
        Some(off) => (
            inputs
                .k_cache
                .write_slice_doff(&k_r, off, 2, write_ranges.clone())?,
            inputs
                .v_cache
                .write_slice_doff(&v_h, off, 2, write_ranges)?,
        ),
        // Backend-generic `SymEnv` offset (Vulkan). Bit-identical write.
        None => {
            let dyn_off = fuel_ir::DynScalar::Sym(inputs.cached_len_sym);
            (
                inputs
                    .k_cache
                    .write_slice_dyn(&k_r, write_ranges.clone(), 2, dyn_off)?,
                inputs
                    .v_cache
                    .write_slice_dyn(&v_h, write_ranges, 2, dyn_off)?,
            )
        }
    };

    let k_t = full_k.transpose()?;
    let scale = 1.0_f64 / (blk.head_dim as f64).sqrt();
    let scores = q_r.matmul(&k_t)?;
    let scores_scaled = scores.mul_scalar(scale);
    let scores_masked = scores_scaled.broadcast_add(inputs.mask)?;
    let attn = scores_masked.softmax_last_dim()?;
    let attn_v = attn.matmul(&full_v)?;

    // `merge_heads()` inlined as permute + reshape so `attn_v`'s SOLE consumer
    // (the permute) can be named as the flash arm's reconverge — arm-0
    // runnability requires the merge to read arm 0. Same split `LlamaModel`
    // makes, for the same reason.
    let attn_v_permuted = attn_v.permute([0, 2, 1, 3_usize])?;
    fuel_core::lazy::offer_flash_decode_arm_for_region(
        q_r.graph(),
        q_r.node_id(),
        full_k.node_id(),
        full_v.node_id(),
        attn_v.node_id(),
        attn_v_permuted.node_id(),
        scale as f32,
        inputs.attended_len_sym,
        // GAP-194: this layer's OWN window — declined by the admissibility gate
        // on a windowed layer, eligible on a dense one. Asserting `None` here
        // would make the arm attend the whole prefix.
        inputs.attn_window,
        None, // no attention-logit softcap in the Qwen3 family
        fuel_dispatch::decode_flash::FlashArmCapability::production(),
    )?;
    let merged = attn_v_permuted.reshape(Shape::from_dims(&[
        batch,
        seq,
        blk.num_attention_heads * blk.head_dim,
    ]))?;
    let attn_out = blk
        .attn_o
        .apply_linear(&merged, blk.hidden_size, blk.hidden_size)?;
    x.add(&attn_out)
}

// ===========================================================================
// SABOTAGE RECORD — QWEN3 / QWEN3MOE (GAP-029 increment 3, families 3 & 4)
//
// Three sabotages, because "does this test test THIS family" and "what is the
// shared block's blast radius" are different questions. All runs carried
// `Compiling fuel-core`.
//
// (a) Qwen3's OWN `decode_apply_layer` — drop the FFN residual:
//       lazy_qwen3::tests  x3 FAILED    -> 82 passed; 3 failed
//     Qwen2, Qwen3Moe, Llama, Llama3 and Phi all GREEN. Exactly one family, as
//     increment 3's constraint (1) requires.
//
// (b) Qwen3Moe's OWN `decode_apply_layer` — same defect:
//       lazy_qwen3_moe::tests x3 FAILED -> 82 passed; 3 failed
//     Exactly one family again, and notably NOT Qwen3 — the two share an
//     attention block, so this is the run that shows the sharing did not make
//     their suites interchangeable.
//
// (c) The SHARED `qwen3_attn_with_kv_writes` — drop the attention residual:
//       lazy_qwen3::tests x3 + lazy_qwen3_moe::tests x3 FAILED
//                                       -> 79 passed; 6 failed
//     BOTH Qwen3 families red, and Qwen2 / Llama / Llama3 / Phi still green.
//     That is the block's blast radius MEASURED rather than intended: it proves
//     both families really execute the shared code (neither kept a private copy
//     the compiler happily retained), and that sharing it did not silently widen
//     its reach into the families that do not use it.
// ===========================================================================

impl Qwen3Model {
    /// **Per-layer attention variation**, gated exactly as the prefill path
    /// gates it (`use_sliding_window && layer_idx < max_window_layers`).
    ///
    /// ⚠️ **`sliding_window: None` is DENSE, not "windowed with some default".**
    /// Prefill computes its width as `sliding_window.unwrap_or(seq + 1)`, and a
    /// window of `seq + 1` cannot exclude anything — so a config with
    /// `use_sliding_window: true` and no width is dense at every layer, and the
    /// plan must say so. Reading the flag alone and inventing a width would
    /// window layers prefill leaves dense.
    pub fn decode_mask_plan(&self) -> MaskPlan {
        let cfg = &self.config;
        match (cfg.use_sliding_window, cfg.sliding_window) {
            (true, Some(w)) => {
                MaskPlan::split_window(cfg.num_hidden_layers, cfg.max_window_layers, w)
            }
            _ => MaskPlan::dense(cfg.num_hidden_layers),
        }
    }

    /// Identity a held decode plan is baked against. `rope_theta` and the window
    /// *width* are absent deliberately — both are per-token data, rebound every
    /// step; the plan contributes only structure (variant count + per-layer
    /// assignment).
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = fuel_core::decode_shape::ShapeKeyHasher::new();
        h.mix_str("qwen3")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim as u64)
            .mix_u64(cfg.hidden_size as u64)
            .mix_u64(cfg.intermediate_size as u64)
            .mix_u64(cfg.vocab_size as u64)
            .mix_f64(cfg.rms_norm_eps);
        self.decode_mask_plan().mix_into(&mut h);
        h.finish()
    }

    /// Decode/prefill through a pre-allocated [`KvCache`], rebuilding the graph
    /// each step. The primitive the persistent path falls back to.
    pub fn forward_with_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
    ) -> Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context(self, tokens, cache, ctx, false, None)
    }

    /// Plan-once persistent decode: the first `seq == 1` token builds and
    /// optimizes the graph, later tokens rebind data and skip optimize.
    pub fn forward_with_kv_context_persistent(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        ctx: &mut InferenceContext,
        session: &mut Option<DecodeSession>,
    ) -> Result<Vec<f32>> {
        fuel_core::persistent_decode::forward_with_kv_context_persistent(
            self, tokens, cache, ctx, session, None,
        )
    }

    /// Persistent decode with the session owned by the `InferenceContext`.
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
}

impl PersistentDecodeModel for Qwen3Model {
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
        let host = fuel_core::persistent_decode::compute_decode_token_host(
            self,
            cached_len,
            tokens,
            session.max_seq_len(),
            rope_inv_freq,
        );
        fuel_core::persistent_decode::upload_decode_token_data(
            device,
            &host,
            cache.dtype.unwrap_or(DType::F32),
            session.offset_node().is_some().then_some(cached_len),
        )
    }
}

impl DecodeBackbone for Qwen3Model {
    fn decode_family(&self) -> &'static str {
        "Qwen3Model"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            // Full rotary over the explicit `head_dim` — no partial-rotary field.
            rope_width: cfg.head_dim,
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        Qwen3Model::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        Qwen3Model::decode_mask_plan(self)
    }

    fn decode_rope_plan(&self) -> fuel_core::persistent_decode::RopePlan {
        fuel_core::persistent_decode::RopePlan::single(
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
        let cfg = &self.config;
        let layer = &self.weights.layers[layer_idx];
        let extras = &self.weights.layer_extras[layer_idx];
        let h1 = qwen3_attn_with_kv_writes(
            &Qwen3AttnBlock {
                hidden_size: cfg.hidden_size,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                attn_norm_gain: &layer.attn_norm_gain,
                attn_q: &layer.attn_q,
                attn_q_bias: layer.attn_q_bias.as_ref(),
                attn_k: &layer.attn_k,
                attn_k_bias: layer.attn_k_bias.as_ref(),
                attn_v: &layer.attn_v,
                attn_v_bias: layer.attn_v_bias.as_ref(),
                attn_o: &layer.attn_o,
                q_norm_gain: &extras.q_norm_gain,
                k_norm_gain: &extras.k_norm_gain,
            },
            inputs,
        )?;

        let h1_norm = h1.rms_norm_affine(Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size)?;
        let swiglu = gate.silu().mul(&up)?;
        let ffn_out =
            layer
                .ffn_down
                .apply_linear(&swiglu, cfg.intermediate_size, cfg.hidden_size)?;
        h1.add(&ffn_out)
    }

    fn decode_final_norm_and_head(&self, h: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let h_norm =
            h.rms_norm_affine(Arc::clone(&self.weights.final_norm_gain), cfg.rms_norm_eps)?;
        self.weights
            .output
            .apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl Qwen3Weights {
    /// Load Qwen3 weights from HF safetensors (e.g. `Qwen/Qwen3-7B`).
    /// HF naming follows LLaMA + per-head QK-norm:
    ///   model.embed_tokens / model.layers.{i}.self_attn.{q,k,v,o}_proj +
    ///   .q_norm / .k_norm + model.layers.{i}.{input_layernorm,
    ///   post_attention_layernorm}.weight + model.layers.{i}.mlp.{gate,up,down}_proj
    ///   + model.norm + lm_head (or tied).
    pub fn load_from_mmapped(
        st: &fuel_core::safetensors::MmapedSafetensors,
        cfg: &Qwen3Config,
    ) -> Result<Self> {
        use fuel_core::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        let mut layer_extras: Vec<Qwen3LayerExtras> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_q = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                h,
            )?;
            let attn_k = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_v = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                h,
            )?;
            let attn_o = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                q_dim,
            )?;
            let (attn_q_bias, attn_k_bias, attn_v_bias) = if cfg.attention_bias {
                (
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.q_proj.bias"),
                    )?)),
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.k_proj.bias"),
                    )?)),
                    Some(Arc::from(load_tensor_as_f32(
                        st,
                        &format!("{p}.self_attn.v_proj.bias"),
                    )?)),
                )
            } else {
                (None, None, None)
            };
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.gate_proj.weight"),
                inter,
                h,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.up_proj.weight"),
                inter,
                h,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.mlp.down_proj.weight"),
                h,
                inter,
            )?;
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
            )?);
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
                attn_norm_gain,
                ffn_norm_gain,
            });
            let q_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.q_norm.weight"),
            )?);
            let k_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.k_norm.weight"),
            )?);
            layer_extras.push(Qwen3LayerExtras {
                q_norm_gain,
                k_norm_gain,
            });
        }
        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = if cfg.tie_word_embeddings {
            crate::models::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding,
                cfg.vocab_size,
                h,
            )
        } else {
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, h)?
        };
        Ok(Self {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            layer_extras,
            final_norm_gain,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &Qwen3Config) -> Qwen3Weights {
        let mut s: u32 = 24680;
        let next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let mut layers = Vec::new();
        let mut layer_extras = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: if cfg.attention_bias {
                    Some(vec_of(h, &mut *nb))
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
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            });
            layer_extras.push(Qwen3LayerExtras {
                q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
            });
        }
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Qwen3Weights {
            instance: fuel_core::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            layer_extras,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_with_per_head_qk_norm() {
        let cfg = Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            max_position_embeddings: 64,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            tie_word_embeddings: false,
        };
        let model = Qwen3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            max_position_embeddings: 64,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            tie_word_embeddings: false,
        };
        let model = Qwen3Model {
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

    fn forward_embeds_test_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            max_position_embeddings: 64,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        // forward_embeds(embed_tokens(tokens)) must produce the same
        // logits as forward(tokens). Unlike Gemma, Qwen3 does NOT apply
        // any sqrt(hidden_size) scaling — embeds are passed raw.
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu());
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        assert_eq!(logits_ref.len(), logits_via_embeds.len());
        let max_diff = logits_ref
            .iter()
            .zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Qwen3 forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let bad_embeds = Tensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]),
            &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad_embeds, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![2, 5];
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
            "Qwen3 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }

    // ---- GAP-029 increment 3, family 3: persistent decode -------------------

    /// Oracle threshold, measured rather than inherited — see the identical
    /// argument in `lazy_qwen2`. The natural template
    /// (`forward_with_kv_context_decode_matches_non_cached_forward`) asserts
    /// `diff < 5e-3 || rel < 1e-2`, which sits ABOVE the ~7e-3 single-mask
    /// divergence and would go green on the very defect this tests for.
    const DECODE_ORACLE_ABS: f32 = 1e-5;

    /// 2 layers, window 4, `max_window_layers: 1` — layer 0 windowed, layer 1
    /// dense. `sliding_window` is `Some`, which matters: `None` is dense.
    fn mixed_window_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            sliding_window: Some(4),
            max_window_layers: 1,
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: true,
            tie_word_embeddings: false,
        }
    }

    /// Max |logit diff| per decode step against the shipped, per-layer-gated
    /// non-cached forward at the same absolute position.
    ///
    /// **The `>= 3` decode steps are load-bearing:** one decode token exercises
    /// only the held-graph BUILD path, and the per-token REBIND path is first
    /// reached on step 2. The assert lives here so a caller cannot weaken it.
    fn decode_vs_forward_max_abs(cfg: &Qwen3Config, tokens: &[u32], prefill: usize) -> Vec<f32> {
        let n_decode = tokens.len() - prefill;
        assert!(
            n_decode >= 3,
            "need >= 3 decode tokens to reach the rebind path"
        );
        let model = Qwen3Model {
            config: cfg.clone(),
            weights: tiny_weights(cfg),
        };

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
        for pos in prefill..tokens.len() {
            let got = model
                .forward_with_kv_context_persistent(
                    &tokens[pos..=pos],
                    &mut cache,
                    &mut ctx,
                    &mut session,
                )
                .expect("decode");
            assert!(session.is_some(), "decode must hold a session from token 1");
            let full = model.forward(&tokens[..=pos], 0).unwrap().realize_f32();
            let expected = &full[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
            out.push(
                got.iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max),
            );
        }
        assert_eq!(
            cache.cached_len,
            tokens.len(),
            "cache must advance every step"
        );
        out
    }

    /// ⚠️ **NON-DISCRIMINATION CONTROL — what makes the sibling's red mean
    /// anything.** With `max_window_layers: 0` the plan collapses to one dense
    /// variant, so this passes under BOTH a correct windowed mask plan and one
    /// that ignores windowing entirely. It certifies the seam, the QK-norm
    /// attention block, the KV writes and the rebind — never the windowing.
    #[test]
    fn qwen3_decode_matches_forward_when_no_layer_is_windowed() {
        let cfg = Qwen3Config {
            max_window_layers: 0,
            ..mixed_window_cfg()
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        for (k, diff) in decode_vs_forward_max_abs(&cfg, &tokens, 3)
            .iter()
            .enumerate()
        {
            assert!(
                *diff < DECODE_ORACLE_ABS,
                "unwindowed Qwen3 decode step {k} (absolute position {}) diverged by \
                 {diff}. This is the CONTROL — the windowed test below proves nothing \
                 until this is green.",
                3 + k,
            );
        }
    }

    /// **GAP-029 family 3 — Qwen3 windowed persistent decode.**
    ///
    /// **Born red, observed.** With `decode_mask_plan` returning
    /// `MaskPlan::dense(..)` — precisely what a single-mask decode port
    /// computes — the measured per-step divergence was
    ///
    /// ```text
    /// absolute position 3, 4, 5 : [0.0, 7.854598e-3, 6.385114e-3]
    /// ```
    ///
    /// while the control above and the absent-width sibling stayed green
    /// (`7 passed; 1 failed`). Restoring the real `split_window` plan took every
    /// step to **0.0**.
    ///
    /// **The leading zero is the discrimination evidence, not a weakness:** a
    /// window of 4 cannot exclude anything until absolute position 4, so a
    /// degenerate oracle would have shown three zeros in the red run and this
    /// showed one. **Both failing steps are REBIND steps** — the session is built
    /// on decode step 0 — so a single-decode-token test could not see this.
    #[test]
    fn qwen3_windowed_decode_matches_per_layer_gated_forward() {
        let cfg = mixed_window_cfg();
        let window = cfg
            .sliding_window
            .expect("mixed config carries a window width");
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        assert!(
            tokens.len() > window,
            "non-vacuity: the window must actually bite"
        );
        let diffs = decode_vs_forward_max_abs(&cfg, &tokens, 3);
        assert!(
            3 + diffs.len() > window,
            "non-vacuity: no decoded position is far enough in for the window to bite",
        );
        // Report the WHOLE per-step vector on any failure: which steps diverge
        // is the discrimination evidence (a window of 4 must leave position 3
        // untouched), and a first-failure-only message throws that away.
        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "windowed Qwen3 decode diverged from the per-layer-gated forward: \
             per-step max|diff| at absolute positions 3..=5 = {diffs:?} (limit \
             {DECODE_ORACLE_ABS:e}). A single mask applied to every layer produces \
             exactly this signature, and leaves position 3 clean.",
        );
    }

    /// ⚠️ **`use_sliding_window: true` with `sliding_window: None` is DENSE.**
    ///
    /// Prefill computes its width as `sliding_window.unwrap_or(seq + 1)`, and a
    /// window of `seq + 1` excludes nothing. Reading the flag alone and
    /// inventing a width would window layers the shipped path leaves dense —
    /// a divergence no test of a `Some(..)` config can see.
    #[test]
    fn qwen3_absent_window_width_is_dense_at_every_layer() {
        let cfg = Qwen3Config {
            sliding_window: None,
            max_window_layers: 2, // would window BOTH layers if a width existed
            ..mixed_window_cfg()
        };
        let plan = Qwen3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        }
        .decode_mask_plan();
        assert_eq!(
            plan.n_variants(),
            1,
            "an absent window width must collapse to a single dense variant",
        );
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        for (k, diff) in decode_vs_forward_max_abs(&cfg, &tokens, 3)
            .iter()
            .enumerate()
        {
            assert!(*diff < DECODE_ORACLE_ABS, "step {k} diverged by {diff}");
        }
    }
}
