// SPDX-License-Identifier: MIT OR Apache-2.0
//! SmolLM3 decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. SmolLM3 (HuggingFace small-model line) is a
//! Qwen2-shape transformer with two notable extras:
//! - **Per-layer RoPE gating** — `uses_rope_per_layer` is a per-layer 0/1
//!   flag, NOT a list of layer indices: `uses_rope_per_layer[i] == 1` means
//!   layer `i` **uses** RoPE; `== 0` marks a **NoPE** (position-agnostic)
//!   layer. This matches the code (`layer_uses_rope`) and HuggingFace
//!   Transformers `configuration_smollm3.py`: *"A `1` at an index
//!   position indicates that the corresponding layer will use RoPE,
//!   while a `0` indicates that it's a NoPE layer."* This field is named for
//!   what it holds; HuggingFace's config key is `no_rope_layers`, kept
//!   verbatim as the wire name in the loader even though it reads backwards
//!   (`no_rope_layers[i] == 1` is a *RoPE* layer). Renamed from
//!   `no_rope_layers` to remove that second wrong signal (GAP-196).
//! - **Optional sliding window** (Mistral-style).
//!
//! Otherwise: GQA + RmsNorm + SwiGLU FFN + optional Q/K/V/O biases
//! via `cfg.attention_bias`. Reuses `crate::lazy::LayerWeights`.

use crate::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use crate::lazy::{LayerWeights, Tensor, WeightStorage};
use crate::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel,
};
use crate::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SmolLm3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    /// One entry per layer: `1` = use RoPE on that layer, `0` = skip
    /// RoPE. `None` = use RoPE on every layer (Llama default).
    pub uses_rope_per_layer: Option<Vec<usize>>,
}

impl SmolLm3Config {
    fn layer_uses_rope(&self, layer_idx: usize) -> bool {
        match &self.uses_rope_per_layer {
            Some(v) => v.get(layer_idx).copied().unwrap_or(1) == 1,
            None => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmolLm3Weights {
    /// Identity of this weight set, folded into [`SmolLm3Model::decode_shape_key`]
    /// so a held decode plan (which bakes these weights as graph Consts) is
    /// never reused for a different SmolLm3 model that shares a config.
    pub instance: crate::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct SmolLm3Model {
    pub config: SmolLm3Config,
    pub weights: SmolLm3Weights,
}

impl SmolLm3Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// SmolLM3-specific: every Nth layer skips RoPE
    /// (NoPE pattern). The hook honors the same per-layer
    /// RoPE-on/off schedule as `forward`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. SmolLM3 does NOT scale embeddings.
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
            return Err(crate::Error::Msg(format!(
                "SmolLm3Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(
                crate::Error::Msg("SmolLm3Model::forward_embeds: seq must be > 0".into()).bt(),
            );
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "SmolLm3Config: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_rope = cfg.layer_uses_rope(layer_idx);
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_rope)?;
        }
        h.rms_norm_affine(
            std::sync::Arc::clone(&weights.final_norm_gain),
            cfg.rms_norm_eps,
        )
    }

    fn build_mask(&self, anchor: &Tensor, seq: usize) -> Tensor {
        let cfg = &self.config;
        let window = cfg.sliding_window.unwrap_or(seq + 1);
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
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        uses_rope: bool,
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

        // Conditional RoPE — skipped only on NoPE layers (`uses_rope_per_layer[i] == 0`).
        let (q_r, k_r) = if uses_rope {
            (
                q.rope_with_tables(rope_cos, rope_sin)?,
                k.rope_with_tables(rope_cos, rope_sin)?,
            )
        } else {
            (q, k)
        };

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_mask(x, seq);
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

    // ---- Persistent KV-context decode (GAP-029 family 6) --------------------

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

    /// Plan-once persistent decode: the first `seq == 1` token builds + optimizes
    /// the graph, later tokens rebind data and skip optimize. `seq != 1` falls
    /// back to [`Self::forward_with_kv_context`].
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

    /// [`Self::forward_with_kv_context_persistent`] at the ergonomic call shape —
    /// the session rides in the `InferenceContext`.
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

    /// Identity of a held decode plan for THIS model: family + config values that
    /// change graph *structure* + this weight set.
    ///
    /// The per-layer **RoPE-on/off pattern is STRUCTURAL** — a no-rope layer
    /// omits the rope op, so two models with different patterns build different
    /// graphs, and the pattern is folded in. The sliding-window **width is NOT**
    /// (it is per-token data, rebound every step; only the mask-plan structure
    /// — how many variants and which layer reads which — is folded, via
    /// [`MaskPlan::mix_into`]). `instance` distinguishes weight sets.
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = crate::decode_shape::ShapeKeyHasher::new();
        h.mix_str("smollm3")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim as u64)
            .mix_u64(cfg.hidden_size as u64)
            .mix_u64(cfg.intermediate_size as u64)
            .mix_u64(cfg.vocab_size as u64)
            .mix_f64(cfg.rms_norm_eps);
        for i in 0..cfg.num_hidden_layers {
            h.mix_u64(cfg.layer_uses_rope(i) as u64);
        }
        self.decode_mask_plan().mix_into(&mut h);
        h.finish()
    }

    /// SmolLm3's sliding window is MODEL-UNIFORM (every layer shares one mask),
    /// so the plan is a single variant: dense when `sliding_window` is `None`,
    /// or one windowed variant (the `split >= n_layers` collapse of
    /// [`MaskPlan::split_window`]) when `Some`. No per-layer split — SmolLm3's
    /// per-layer axis is RoPE-on/off, not the mask.
    pub fn decode_mask_plan(&self) -> MaskPlan {
        let n = self.config.num_hidden_layers;
        match self.config.sliding_window {
            None => MaskPlan::dense(n),
            Some(w) => MaskPlan::split_window(n, n, w),
        }
    }

    /// One SmolLm3 layer against the pre-allocated KV buffers — the decode twin
    /// of [`Self::apply_layer`]. Same math as the prefill twin (biased Q/K/V,
    /// GQA, SwiGLU) plus **per-layer conditional RoPE** (`uses_rope`), with the
    /// seam's two decode differences shared by every ported family: GQA left to
    /// `matmul`'s head broadcast (no `repeat_interleave`), and no flash-decode arm.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_kv_writes(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        uses_rope: bool,
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
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let act_dtype = x.dtype();

        let x_norm = x.rms_norm_affine(Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

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

        let q_h = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k_h = k.split_heads(cfg.num_key_value_heads, head_dim)?;
        let v_h = v.split_heads(cfg.num_key_value_heads, head_dim)?;

        // Per-layer conditional RoPE — skipped only on NoPE layers (`uses_rope_per_layer[i] == 0`). RoPE
        // runs in f32 (build-time requirement); the casts are no-ops at f32.
        let (q_r, k_r) = if uses_rope {
            (
                q_h.to_dtype(DType::F32)?
                    .rope_with_tables(rope_cos, rope_sin)?
                    .to_dtype(act_dtype)?,
                k_h.to_dtype(DType::F32)?
                    .rope_with_tables(rope_cos, rope_sin)?
                    .to_dtype(act_dtype)?,
            )
        } else {
            (q_h, k_h)
        };

        // Write this step's K/V at the runtime offset — K is post-RoPE-or-not, V
        // raw. GQA is NOT replicated here (the cache holds `num_key_value_heads`,
        // matmul broadcasts to Q's heads).
        let write_ranges = vec![
            (0, batch),
            (0, cfg.num_key_value_heads),
            (0, seq),
            (0, head_dim),
        ];
        let (full_k, full_v) = match offset {
            Some(off) => (
                k_cache_const.write_slice_doff(&k_r, off, 2, write_ranges.clone())?,
                v_cache_const.write_slice_doff(&v_h, off, 2, write_ranges)?,
            ),
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

impl PersistentDecodeModel for SmolLm3Model {
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

impl DecodeBackbone for SmolLm3Model {
    fn decode_family(&self) -> &'static str {
        "SmolLm3Model"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            // Full rotary — SmolLm3 has no partial-rotary factor. The rope tables
            // are built unconditionally; no-rope layers just don't apply them.
            rope_width: cfg.head_dim,
            // No embedding scale — that is a Gemma-family trait.
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        SmolLm3Model::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        SmolLm3Model::decode_mask_plan(self)
    }

    /// **One RoPE base for every layer** — and this is the family that makes the
    /// distinction concrete. SmolLm3 varies RoPE *per layer*, but by SKIPPING it
    /// (`uses_rope_per_layer`), not by changing its base. Skipping needs no different
    /// table bytes, so it stays inside `decode_apply_layer` and the plan is
    /// single-variant; Gemma3's dual base genuinely differs in bytes and is what
    /// `RopePlan` exists for.
    ///
    /// Variation in DATA needs a variant axis; variation in WHETHER-TO-APPLY
    /// does not.
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
            self.config.layer_uses_rope(layer_idx),
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

// ---- HuggingFace safetensors loader ----------------------------------------

impl SmolLm3Weights {
    /// Load SmolLM3 weights from HF safetensors.
    /// HF naming follows LLaMA conventions: model.embed_tokens.weight,
    /// model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,optional bias},
    /// model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight,
    /// model.layers.{i}.mlp.{gate,up,down}_proj.weight, model.norm.weight,
    /// lm_head.weight (always present in HF SmolLM3 checkpoints).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &SmolLm3Config,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let h = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
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

            let attn_q_bias = if cfg.attention_bias {
                Some(Arc::from(load_tensor_as_f32(
                    st,
                    &format!("{p}.self_attn.q_proj.bias"),
                )?))
            } else {
                None
            };
            let attn_k_bias = if cfg.attention_bias {
                Some(Arc::from(load_tensor_as_f32(
                    st,
                    &format!("{p}.self_attn.k_proj.bias"),
                )?))
            } else {
                None
            };
            let attn_v_bias = if cfg.attention_bias {
                Some(Arc::from(load_tensor_as_f32(
                    st,
                    &format!("{p}.self_attn.v_proj.bias"),
                )?))
            } else {
                None
            };

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
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output =
            load_transposed_matrix_preserve_dtype(st, "lm_head.weight", cfg.vocab_size, h)?;

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
    fn tiny_weights(cfg: &SmolLm3Config) -> SmolLm3Weights {
        let mut s: u32 = 55555;
        let mut next = || -> f32 {
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
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
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
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        SmolLm3Weights {
            instance: crate::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_shape_and_finite_all_rope() {
        let cfg = SmolLm3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            sliding_window: None,
            uses_rope_per_layer: None,
        };
        let model = SmolLm3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// `uses_rope_per_layer = [0, 1]` skips RoPE on layer 0 only.
    /// Output must differ from the all-RoPE configuration.
    #[test]
    fn skipping_rope_on_one_layer_changes_output() {
        let mut cfg = SmolLm3Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            attention_bias: false,
            sliding_window: None,
            uses_rope_per_layer: None,
        };
        let weights = tiny_weights(&cfg);
        let out_all = SmolLm3Model {
            config: cfg.clone(),
            weights: weights.clone(),
        }
        .forward(&[1, 2, 3, 4], 0)
        .unwrap()
        .realize_f32();
        cfg.uses_rope_per_layer = Some(vec![0, 1]); // skip RoPE on layer 0
        let out_partial = SmolLm3Model {
            config: cfg,
            weights,
        }
        .forward(&[1, 2, 3, 4], 0)
        .unwrap()
        .realize_f32();
        let any_diff = out_all
            .iter()
            .zip(out_partial.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "skipping RoPE on layer 0 must change output");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = SmolLm3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            sliding_window: None,
            uses_rope_per_layer: None,
        };
        let model = SmolLm3Model {
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

    fn forward_embeds_test_cfg() -> SmolLm3Config {
        SmolLm3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            sliding_window: None,
            uses_rope_per_layer: None,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = SmolLm3Model {
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
            "SmolLm3 forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = SmolLm3Model {
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
        let model = SmolLm3Model {
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
            "SmolLm3 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }

    // ==== GAP-029 family 6: persistent KV-context decode =====================

    /// Standard tiny config with GQA (4 heads / 2 KV heads) — override
    /// `uses_rope_per_layer` / `sliding_window` per test.
    fn base_cfg() -> SmolLm3Config {
        SmolLm3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            attention_bias: false,
            sliding_window: None,
            uses_rope_per_layer: None,
        }
    }

    /// Prefill `tokens[..prefill]`, then decode the rest one token at a time
    /// through the persistent path; return each decode step's logits.
    ///
    /// **`>= 3` decode steps are load-bearing:** the per-token REBIND path is
    /// first reached on step 2, so a 1-step test passes on a model wrong from
    /// token 2.
    fn decode_steps(model: &SmolLm3Model, tokens: &[u32], prefill: usize) -> Vec<Vec<f32>> {
        let cfg = &model.config;
        let n_decode = tokens.len() - prefill;
        assert!(
            n_decode >= 3,
            "need >= 3 decode tokens to reach the rebind path (got {n_decode})"
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
    /// [`SmolLm3Model::forward`] at the same absolute position.
    ///
    /// `forward` is an INDEPENDENT correct reference: the born-red sabotage lives
    /// only in the decode layer, so this is an absolute oracle against
    /// unsabotaged code — NOT a relative A-vs-B over shared code.
    fn decode_vs_forward_max_abs(model: &SmolLm3Model, tokens: &[u32], prefill: usize) -> Vec<f32> {
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

    /// The oracle threshold — **measured, not inherited**, set BETWEEN the
    /// correct drift and the sabotaged divergence.
    ///
    /// The ambient decode template (`diff < 5e-3 || rel < 1e-2`) would badly
    /// miss this: SmolLm3's rope-gating divergence on a tiny model is **small
    /// and does not amplify with position** (measured identical at 3 and 9
    /// decode steps). Measured on the mixed-rope config (prefill 3, decode 9),
    /// decode vs the shipped prefill `forward`, via [`measure_smollm3_decode_drift`]:
    ///
    /// ```text
    /// (a) correct conditional-rope decode : [0.0; 9]                    bit-exact
    /// (b) rope-everywhere (sabotaged)     : 1.14e-5 .. 3.52e-5 (max)    divergence
    /// control (uses_rope_per_layer = None)     : [0.0; 9] under BOTH bodies  insensitive
    /// ```
    ///
    /// `1e-6` sits **~11.4x below the SMALLEST sabotaged step (1.14e-5)** — the
    /// margin standard set for Glm4 (which was 8.4x). (a) is bit-exact `0.0`
    /// **measured on this box**, so the accept side has full headroom; the ~1e-7
    /// cross-machine gemm-reassociation figure below is an **ESTIMATE for other
    /// hardware, NOT a measurement** — do not cite it as observed. `1e-6` leaves
    /// ~10x over that estimate. Tighter than Glm4's `1e-5` **because this
    /// divergence is ~40x smaller**, not because the accept side drifts, and set
    /// below the SMALLEST divergent step so every decode position discriminates,
    /// not just the largest.
    const DECODE_ORACLE_ABS: f32 = 1e-6;

    /// DISCRIMINATOR — mixed RoPE (`uses_rope_per_layer = Some(vec![0,1])`: layer 0
    /// NoPE, layer 1 RoPE). Dropping the per-layer conditional in
    /// `apply_layer_with_kv_writes` (applying RoPE everywhere, the "copied
    /// always-rope Qwen2 line" mistake) reddens exactly this test — the born-red.
    #[test]
    fn smollm3_decode_matches_forward_mixed_rope() {
        let cfg = SmolLm3Config {
            uses_rope_per_layer: Some(vec![0, 1]),
            ..base_cfg()
        };
        assert!(
            !cfg.layer_uses_rope(0) && cfg.layer_uses_rope(1),
            "layer 0 NoPE, layer 1 RoPE"
        );
        let model = SmolLm3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = (1u32..=12).collect();
        for (k, d) in decode_vs_forward_max_abs(&model, &tokens, 3)
            .iter()
            .enumerate()
        {
            assert!(
                *d < DECODE_ORACLE_ABS,
                "mixed-rope decode step {k} diverges from forward by {d} (>= {DECODE_ORACLE_ABS})",
            );
        }
    }

    /// NON-DISCRIMINATION CONTROL — read first. With `uses_rope_per_layer = None`,
    /// `layer_uses_rope` is always true, so "apply RoPE everywhere" (the
    /// sabotage) is IDENTICAL to the correct conditional BY CONSTRUCTION — the
    /// sabotage CANNOT be sensitive here. Green isolates "seam/plumbing works"
    /// from "the gating is right": if this fails, the discriminator proves
    /// nothing (the instrument would be measuring plumbing, not gating).
    #[test]
    fn control_decode_matches_forward_all_rope() {
        let cfg = base_cfg(); // uses_rope_per_layer = None
        assert!(
            cfg.layer_uses_rope(0) && cfg.layer_uses_rope(1),
            "control: every layer uses rope"
        );
        let model = SmolLm3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let tokens: Vec<u32> = (1u32..=12).collect();
        for (k, d) in decode_vs_forward_max_abs(&model, &tokens, 3)
            .iter()
            .enumerate()
        {
            assert!(
                *d < DECODE_ORACLE_ABS,
                "control decode step {k} diverges from forward by {d}"
            );
        }
    }

    /// Axis-B coverage: model-uniform sliding window (`Some(4)`) with all-rope.
    /// Exercises the `MaskPlan::split_window(n, n, w)` single-windowed-variant
    /// collapse end-to-end (decode vs windowed `forward`).
    #[test]
    fn smollm3_decode_matches_forward_windowed() {
        let cfg = SmolLm3Config {
            sliding_window: Some(4),
            ..base_cfg()
        };
        assert_eq!(
            SmolLm3Model {
                config: cfg.clone(),
                weights: tiny_weights(&cfg)
            }
            .decode_mask_plan()
            .n_variants(),
            1,
            "model-uniform window is a single variant"
        );
        let model = SmolLm3Model {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        // seq 6 > window 4 so the window actually bites.
        let tokens: Vec<u32> = (1u32..=12).collect();
        for (k, d) in decode_vs_forward_max_abs(&model, &tokens, 3)
            .iter()
            .enumerate()
        {
            assert!(
                *d < DECODE_ORACLE_ABS,
                "windowed decode step {k} diverges from forward by {d}"
            );
        }
    }

    /// Prints measured drift for both bodies — run with `--nocapture` for (a)/(b).
    #[test]
    fn measure_smollm3_decode_drift() {
        let mixed = SmolLm3Config {
            uses_rope_per_layer: Some(vec![0, 1]),
            ..base_cfg()
        };
        let m_mixed = SmolLm3Model {
            config: mixed.clone(),
            weights: tiny_weights(&mixed),
        };
        let ctrl = base_cfg();
        let m_ctrl = SmolLm3Model {
            config: ctrl.clone(),
            weights: tiny_weights(&ctrl),
        };
        let tokens: Vec<u32> = (1u32..=12).collect();
        println!(
            "SMOLLM3-DRIFT mixed_rope={:?} control_all_rope={:?}",
            decode_vs_forward_max_abs(&m_mixed, &tokens, 3),
            decode_vs_forward_max_abs(&m_ctrl, &tokens, 3),
        );
    }
}
