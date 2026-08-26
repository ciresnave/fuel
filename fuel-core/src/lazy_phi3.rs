// SPDX-License-Identifier: MIT OR Apache-2.0
//! Phi-3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Phi-3 (Phi-3-mini-4k-instruct etc.) is a
//! standard GQA transformer with HuggingFace's "fused projection"
//! quirk: a single `qkv_proj` packs Q + K + V along the last dim,
//! and a single `gate_up_proj` packs gate + up. On disk the
//! safetensors stores them fused; in lazy we store them split
//! (matching [`crate::lazy::LayerWeights`]), with the safetensors
//! loader doing the narrow at load time.
//!
//! **Deferred to a follow-up** (don't block other ports on it):
//!   - LongRoPE long-context scaling (short_factor / long_factor /
//!     `original_max_position_embeddings`). Phi-3-mini-4k doesn't
//!     use it; Phi-3-mini-128k does.
//!   - `partial_rotary_factor` < 1.0 (apply RoPE to only a prefix of
//!     each head's dim). Default is 1.0 (full rotary) — that's what
//!     this port assumes.
//!
//! Both can be added by augmenting `Phi3Config` + the RoPE table
//! builder when a Phi-3-128k checkpoint needs to run.
//!
//! # Scope (v1, same as the other Phase D ports)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache, F32
//! activations. Strict lower-triangular causal mask.

use crate::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use crate::lazy::{LayerWeights, Tensor, WeightStorage};
use crate::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel,
};
use crate::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `microsoft/Phi-3-mini-4k-instruct`.
    pub fn phi3_mini_4k() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Phi3Weights {
    /// Process-unique identity for THIS weight set — what lets a held decode
    /// plan tell two same-architecture models apart (GAP-029). Mint with
    /// [`crate::decode_shape::ModelInstanceId::next`].
    pub instance: crate::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Phi3Model {
    pub config: Phi3Config,
    pub weights: Phi3Weights,
}

impl Phi3Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Mirrors the
    /// `forward_hidden` pattern shipped across the LLM family.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips the token-embedding step and runs
    /// the decoder over pre-embedded inputs — the precursor for a
    /// future Phi-3-Vision / Phi-3.5-V lazy composition. Phi3 does
    /// NOT scale embeddings — `embeds` is passed raw.
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
        self.weights
            .output
            .apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size)
    }

    /// Shared backbone: embed → RoPE → per-layer attn + MLP →
    /// final RmsNorm. Used by both `forward` (then matmuls
    /// with `lm_head`) and `forward_hidden`.
    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "Phi3Model: tokens must be non-empty");

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
                "Phi3Model::forward_embeds: expected embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            ))
            .bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(
                crate::Error::Msg("Phi3Model::forward_embeds: seq must be > 0".into()).bt(),
            );
        }
        let head_dim = cfg.head_dim();
        if cfg.num_attention_heads * head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Phi3Config: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(crate::Error::Msg(
                "Phi3Config: num_attention_heads must be a multiple of num_key_value_heads".into(),
            )
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, start_pos, seq, head_dim);

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
        layer: &LayerWeights,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let x_norm = x.rms_norm_affine(
            std::sync::Arc::clone(&layer.attn_norm_gain),
            cfg.rms_norm_eps,
        )?;

        // Bias-free Q / K / V (Phi-3 uses linear_no_bias for all).
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        // Strict causal mask.
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
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

        // SwiGLU FFN (Phi-3's MLP is SwiGLU even though it stores
        // gate+up fused on disk).
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

// ---- HuggingFace safetensors loader ----------------------------------------

/// Split a fused QKV transposed matrix (shape [hidden_size, qkv_out]) into Q/K/V.
/// Phi3 uses MQA-like qkv_out = q_dim + 2*kv_dim with Q occupying the first
/// q_dim columns then K then V.
fn split_phi3_qkv(
    transposed: &[f32],
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let qkv_out = q_dim + 2 * kv_dim;
    let mut q = vec![0.0_f32; hidden_size * q_dim];
    let mut k = vec![0.0_f32; hidden_size * kv_dim];
    let mut v = vec![0.0_f32; hidden_size * kv_dim];
    for row in 0..hidden_size {
        let src = &transposed[row * qkv_out..(row + 1) * qkv_out];
        q[row * q_dim..(row + 1) * q_dim].copy_from_slice(&src[0..q_dim]);
        k[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&src[q_dim..q_dim + kv_dim]);
        v[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&src[q_dim + kv_dim..]);
    }
    (q, k, v)
}

// ---- GAP-029 increment 3 · persistent-KV decode -----------------------------
//
// Phi3 is family 6, and it is the FIRST UNIFORM one on this seam: a strict
// lower-triangular causal mask at every layer, no sliding window, no per-layer
// variation of any kind. That is not a gap in the port — it is what the mask
// plan's `n_variants == 1` case exists for, and it means Phi3's decode graph
// carries no mask slice node at all.
//
// ⚠️ NAME COLLISION WORTH READING ONCE: `Phi3Model` (here) is NOT `PhiModel`
// (`crate::lazy`). They are different architectures — Phi has a parallel
// attention block, LayerNorm + bias, and its OWN hand-written decode body that
// GAP-029 deliberately does not touch. Phi3 measured LLaMA-shaped despite the
// lineage in its name, which is why it is on the shared seam and Phi is not.
// The sabotage record below pins that boundary by measurement.

impl Phi3Model {
    /// **No per-layer attention variation.** Measured for GAP-029 increment 3:
    /// zero `sliding_window` hits in this file, against a positive control of
    /// hits in five sibling model files. `apply_layer` builds a strict causal
    /// mask unconditionally.
    ///
    /// The uniform plan emits no slice node and its host bytes are byte-identical
    /// to `build_decode_causal_mask` (asserted in
    /// `crate::persistent_decode`), so a family with no windowing pays literally
    /// nothing for the N-variant machinery.
    pub fn decode_mask_plan(&self) -> MaskPlan {
        MaskPlan::dense(self.config.num_hidden_layers)
    }

    /// Identity a held decode plan is baked against. `rope_theta` is absent
    /// deliberately — RoPE tables are rebound per token, so baking it would
    /// forfeit plan reuse across a change already handled correctly.
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = crate::decode_shape::ShapeKeyHasher::new();
        h.mix_str("phi3")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim() as u64)
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
        crate::persistent_decode::forward_with_kv_context(self, tokens, cache, ctx, false, None)
    }

    /// Plan-once persistent decode.
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

    /// One Phi3 layer against the pre-allocated KV buffers.
    ///
    /// Phi3's attention is Qwen2's **without the Q/K/V biases** (`linear_no_bias`
    /// throughout). The fused `qkv_proj` / `gate_up_proj` are a *load-time*
    /// concern — they are already split into `LayerWeights` fields before any
    /// graph is built — so nothing about the fusion reaches this path.
    ///
    /// Same two deliberate differences from the prefill twin as every family on
    /// this seam: GQA rides `matmul`'s head broadcast rather than materialising
    /// `repeat_interleave` over the whole cache, and **no flash-decode arm is
    /// offered** (see `lazy_qwen2` for the correctness argument; it is not
    /// window-specific — an unwindowed family simply has no benefit yet to
    /// justify shipping an arm this lane cannot test).
    fn apply_layer_with_kv_writes(
        &self,
        layer: &LayerWeights,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x = inputs.x;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let act_dtype = x.dtype();

        let x_norm = x.rms_norm_affine(Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        // Bias-free Q / K / V — Phi-3 uses `linear_no_bias` for all.
        let q = layer
            .attn_q
            .apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size)?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, cfg.hidden_size, kv_dim)?;

        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v_h = v.split_heads(cfg.num_key_value_heads, head_dim)?;

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
            (0, cfg.num_key_value_heads),
            (0, seq), // axis-2 start is dynamic; width = seq
            (0, head_dim),
        ];
        let (full_k, full_v) = match inputs.offset {
            Some(off) => (
                inputs
                    .k_cache
                    .write_slice_doff(&k_r, off, 2, write_ranges.clone())?,
                inputs
                    .v_cache
                    .write_slice_doff(&v_h, off, 2, write_ranges)?,
            ),
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
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(inputs.mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&full_v)?;

        // `merge_heads()` inlined as permute + reshape so `attn_v`'s SOLE
        // consumer (the permute) can be named as the flash arm's reconverge —
        // arm-0 runnability requires the merge to read arm 0.
        let attn_v_permuted = attn_v.permute([0, 2, 1, 3_usize])?;
        crate::lazy::offer_flash_decode_arm_for_region(
            q_r.inner.graph(),
            q_r.inner.id(),
            full_k.inner.id(),
            full_v.inner.id(),
            attn_v.inner.id(),
            attn_v_permuted.inner.id(),
            scale as f32,
            inputs.attended_len_sym,
            // Phi3 is uniformly dense, so this is always `None` — but DERIVED
            // from its mask plan (GAP-194), not asserted.
            inputs.attn_window,
            None, // no attention-logit softcap
            fuel_dispatch::decode_flash::FlashArmCapability::production(),
        )?;
        let merged = attn_v_permuted.reshape(Shape::from_dims(&[
            batch,
            seq,
            cfg.num_attention_heads * head_dim,
        ]))?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, cfg.hidden_size, cfg.hidden_size)?;

        let h1 = x.add(&attn_out)?;
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
}

// ===========================================================================
// SABOTAGE RECORD — PHI3 (GAP-029 increment 3, family 6, 2026-08-13)
//
// `phi3_decode_matches_non_cached_forward` is BORN GREEN — Phi3 is uniform, so
// the single-mask born-red that works for the windowed families does not exist
// here (a single mask is CORRECT for Phi3). Its discrimination therefore has to
// be established separately, and this is it.
//
// Sabotage: drop the FFN residual in Phi3's OWN `apply_layer_with_kv_writes`.
// Run carried `Compiling fuel-core`:
//
//   lazy_phi3::phi3_decode_matches_non_cached_forward   FAILED
//   test result: FAILED. 91 passed; 1 failed
//
// EXACTLY ONE test red, and the 91 green are what make it mean anything.
//
// ⚠️ THE GREEN THAT MATTERS MOST IS `phi_kv_context::*` — `PhiModel`, which is
// a DIFFERENT architecture that keeps its own hand-written decode body and is
// deliberately NOT on this seam. Two families one character apart in name, one
// on the shared path and one off it: this run is what makes that boundary a
// measurement instead of an intention. Qwen2, Qwen3, Qwen3Moe, Llama and
// Llama3 also stayed green.
//
// NOTE `phi3_mask_plan_is_a_single_dense_variant` stayed green too, correctly:
// it asserts plan STRUCTURE, not numerics, so it cannot see this defect and
// must not be cited as covering it.
// ===========================================================================

impl PersistentDecodeModel for Phi3Model {
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
            session.offset_node().is_some().then_some(cached_len),
        )
    }
}

impl DecodeBackbone for Phi3Model {
    fn decode_family(&self) -> &'static str {
        "Phi3Model"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            // Full rotary. The module doc mentions `partial_rotary_factor`, but
            // no such field exists and `apply_layer` rotates the whole head —
            // measured for increment 3's step 0, and the reason Phi3 sits on
            // this seam rather than beside Phi.
            rope_width: cfg.head_dim(),
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        Phi3Model::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        Phi3Model::decode_mask_plan(self)
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
        self.apply_layer_with_kv_writes(&self.weights.layers[layer_idx], inputs)
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

/// Split fused gate_up_proj [hidden_size, 2*intermediate] into gate and up.
fn split_phi3_gate_up(
    transposed: &[f32],
    hidden_size: usize,
    inter: usize,
) -> (Vec<f32>, Vec<f32>) {
    let out_dim = 2 * inter;
    let mut gate = vec![0.0_f32; hidden_size * inter];
    let mut up = vec![0.0_f32; hidden_size * inter];
    for row in 0..hidden_size {
        let src = &transposed[row * out_dim..(row + 1) * out_dim];
        gate[row * inter..(row + 1) * inter].copy_from_slice(&src[0..inter]);
        up[row * inter..(row + 1) * inter].copy_from_slice(&src[inter..]);
    }
    (gate, up)
}

impl Phi3Weights {
    /// Load Phi-3 weights from HF safetensors (e.g. `microsoft/Phi-3-mini-4k-instruct`).
    /// Phi-3 uses fused qkv_proj + fused gate_up_proj — split at load time.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Phi3Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix};
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let qkv = load_transposed_matrix(
                st,
                &format!("{p}.self_attn.qkv_proj.weight"),
                q_dim + 2 * kv_dim,
                h,
            )?;
            let (q, k, v) = split_phi3_qkv(&qkv, h, q_dim, kv_dim);
            let attn_o = crate::lazy::load_transposed_matrix_preserve_dtype(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                q_dim,
            )?;

            let gate_up =
                load_transposed_matrix(st, &format!("{p}.mlp.gate_up_proj.weight"), 2 * inter, h)?;
            let (gate, up) = split_phi3_gate_up(&gate_up, h, inter);
            let ffn_down = crate::lazy::load_transposed_matrix_preserve_dtype(
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
                attn_q: WeightStorage::F32(Arc::from(q)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(Arc::from(k)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(Arc::from(v)),
                attn_v_bias: None,
                attn_o,
                ffn_gate: WeightStorage::F32(Arc::from(gate)),
                ffn_up: WeightStorage::F32(Arc::from(up)),
                ffn_down,
                attn_norm_gain,
                ffn_norm_gain,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = if cfg.tie_word_embeddings {
            crate::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding,
                cfg.vocab_size,
                h,
            )
        } else {
            crate::lazy::load_transposed_matrix_preserve_dtype(
                st,
                "lm_head.weight",
                cfg.vocab_size,
                h,
            )?
        };

        Ok(Self {
            instance: crate::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Phi3Config) -> Phi3Weights {
        let mut s: u32 = 8888;
        let next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim();
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *next_box)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *next_box)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        Phi3Weights {
            instance: crate::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_2_layer() {
        let cfg = Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Phi3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// `forward_hidden` returns post-RmsNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        };
        let model = Phi3Model {
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

    fn forward_embeds_test_cfg() -> Phi3Config {
        Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = Phi3Model {
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
            "Phi3 forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = Phi3Model {
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
        let cfg = forward_embeds_test_cfg();
        let model = Phi3Model {
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
            "Phi3 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }

    // ---- GAP-029 increment 3, family 6: persistent decode -------------------

    /// Measured, not inherited — the natural template's `diff < 5e-3 ||
    /// rel < 1e-2` sits above the ~7e-3 divergences the windowed families
    /// measured, so it cannot certify a decode port on this seam.
    const DECODE_ORACLE_ABS: f32 = 1e-5;

    fn decode_cfg() -> Phi3Config {
        Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2, // exercise GQA (n_rep = 2)
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }

    /// **Phi3 is the first UNIFORM family on this seam, so it has NO born-red
    /// and this test is BORN GREEN. That is stated rather than presented as
    /// evidence.**
    ///
    /// The windowed families (Qwen2/Qwen3/Qwen3Moe) could be born red by handing
    /// the port a single-mask plan, because a single mask is *wrong* for them.
    /// Phi3 has a strict causal mask at every layer — a single mask is exactly
    /// **correct** — so that instrument does not exist here, and inventing one
    /// would be theatre.
    ///
    /// What this test IS: an independent oracle. `forward` is a different code
    /// path (no KV cache, mask rebuilt per layer, `repeat_interleave` GQA) that
    /// already shipped, so decode agreeing with it position-for-position is a
    /// real claim, not a tautology. What certifies that it *discriminates* is
    /// the sabotage record below — not its passing.
    ///
    /// `>= 3` decode steps so the assertions land on the per-token REBIND path.
    #[test]
    fn phi3_decode_matches_non_cached_forward() {
        let cfg = decode_cfg();
        let model = Phi3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let prefill = 3;
        let head_dim = cfg.head_dim();

        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            head_dim,
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

        let mut diffs = Vec::new();
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
            diffs.push(
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
        assert!(
            diffs.len() >= 3,
            "need >= 3 decode steps to reach the rebind path"
        );

        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "Phi3 persistent decode diverged from the non-cached forward: per-step \
             max|diff| at absolute positions 3..=5 = {diffs:?} (limit \
             {DECODE_ORACLE_ABS:e})",
        );
    }

    /// Node count of Phi3's held decode graph — see
    /// [`phi3_held_decode_graph_has_not_grown`]. Measured, not predicted.
    const PHI3_DECODE_GRAPH_NODES: usize = 130;

    fn gap029_phi3_decode_graph_nodes() -> usize {
        let cfg = decode_cfg();
        let model = Phi3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let dev = Device::cpu();
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            6,
            DType::F32,
            &dev,
        )
        .expect("with_capacity");
        let mut ctx = InferenceContext::new(dev);
        let mut session: Option<DecodeSession> = None;
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

    /// **STRUCTURAL baseline for a fully uniform family**, captured 2026-08-13
    /// before the Gemma3 seam work. Phi3 is single-variant on BOTH axes (one
    /// mask, one RoPE base) and has no embedding scale, so it is the strictest
    /// witness that the new machinery costs a non-Gemma family literally
    /// nothing: `embed_scale == None` must emit no multiply and
    /// `n_rope_variants == 1` must emit neither a slice nor a reshape.
    ///
    /// A logits golden cannot see node growth. This can.
    #[test]
    fn phi3_held_decode_graph_has_not_grown() {
        assert_eq!(
            gap029_phi3_decode_graph_nodes(),
            PHI3_DECODE_GRAPH_NODES,
            "Phi3's held decode graph changed size",
        );
    }

    /// Phi3's uniform plan must stay single-variant: that is what makes its
    /// decode graph carry no mask slice node and its mask bytes byte-identical
    /// to the pre-GAP-029 dense builder. A regression here would be silent —
    /// correct output, extra nodes — so it is asserted rather than assumed.
    #[test]
    fn phi3_mask_plan_is_a_single_dense_variant() {
        let cfg = decode_cfg();
        let plan = Phi3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        }
        .decode_mask_plan();
        assert_eq!(
            plan.n_variants(),
            1,
            "Phi3 has no per-layer attention variation"
        );
        assert_eq!(plan.n_layers(), cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            assert_eq!(
                plan.variant_for_layer(li),
                0,
                "layer {li} must take the dense variant"
            );
        }
    }
}
