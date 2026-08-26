// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen3-MoE decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Qwen3-MoE = Qwen3 attention (per-head QK-norm +
//! per-layer sliding-window gating + optional Q/K/V/O biases) +
//! per-layer FFN alternation between a dense SwiGLU MLP and a
//! Mixtral-style sparse MoE. `decoder_sparse_step` controls the
//! cadence: layer `i` uses MoE when `(i + 1) % decoder_sparse_step
//! == 0`; other layers run a single SwiGLU.
//!
//! v1 uses **dense routing** for the MoE layers (every expert
//! evaluated, weighted by full router softmax) — same trade-off
//! as Mixtral. No shared expert.

use crate::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache};
use crate::lazy::{Tensor, WeightStorage};
use crate::lazy_qwen3::{Qwen3AttnBlock, qwen3_attn_with_kv_writes};
use crate::persistent_decode::{
    DecodeBackbone, DecodeDims, DecodeLayerInputs, MaskPlan, PersistentDecodeModel,
};
use crate::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Layer `i` uses MoE iff `(i + 1) % decoder_sparse_step == 0`.
    /// `1` → every layer is MoE; `2` → every other; etc.
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl Qwen3MoeConfig {
    pub fn layer_uses_moe(&self, layer_idx: usize) -> bool {
        self.num_experts > 0 && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeExpertWeights {
    pub gate_w: WeightStorage,
    pub up_w: WeightStorage,
    pub down_w: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    /// Per-head QK-norm gains (`[head_dim]` each).
    pub q_norm_gain: Arc<[f32]>,
    pub k_norm_gain: Arc<[f32]>,
    /// FFN variant. `Dense` → single SwiGLU; `Moe` → router + experts.
    pub ffn: Qwen3MoeFfn,
}

#[derive(Debug, Clone)]
pub enum Qwen3MoeFfn {
    Dense {
        gate_w: WeightStorage,
        up_w: WeightStorage,
        down_w: WeightStorage,
    },
    Moe {
        /// `[hidden_size, num_experts]` router.
        router_w: Arc<[f32]>,
        experts: Vec<Qwen3MoeExpertWeights>,
    },
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeWeights {
    /// Process-unique identity for THIS weight set — what lets a held decode
    /// plan tell two same-architecture models apart (GAP-029). Mint with
    /// [`crate::decode_shape::ModelInstanceId::next`].
    pub instance: crate::decode_shape::ModelInstanceId,
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Qwen3MoeLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeModel {
    pub config: Qwen3MoeConfig,
    pub weights: Qwen3MoeWeights,
}

impl Qwen3MoeModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Qwen3-MoE-specific: per-layer sliding-window gate
    /// (`use_sliding_window && layer_idx < max_window_layers`)
    /// and per-token MoE FFN routing are honored.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. Qwen3-MoE does NOT scale embeddings.
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
                "Qwen3MoeModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(
                crate::Error::Msg("Qwen3MoeModel::forward_embeds: seq must be > 0".into()).bt(),
            );
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Qwen3MoeConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            )
            .bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) =
            h.rope_tables_const(cfg.rope_theta, start_pos, seq, cfg.head_dim);

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window = cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_window)?;
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
        layer: &Qwen3MoeLayerWeights,
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

        // Per-head QK-norm.
        let q = q.rms_norm_affine(std::sync::Arc::clone(&layer.q_norm_gain), cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine(std::sync::Arc::clone(&layer.k_norm_gain), cfg.rms_norm_eps)?;

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

        let ffn_out = self.apply_ffn(&h1_norm, &layer.ffn, batch, seq)?;
        h1.add(&ffn_out)
    }

    fn apply_ffn(&self, x: &Tensor, ffn: &Qwen3MoeFfn, batch: usize, seq: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        match ffn {
            Qwen3MoeFfn::Dense {
                gate_w,
                up_w,
                down_w,
            } => {
                let inter = cfg.intermediate_size;
                let gate = gate_w.apply_linear(x, h, inter)?;
                let up = up_w.apply_linear(x, h, inter)?;
                let swiglu = gate.silu().mul(&up)?;
                Ok(down_w.apply_linear(&swiglu, inter, h)?)
            }
            Qwen3MoeFfn::Moe { router_w, experts } => {
                let inter = cfg.moe_intermediate_size;
                let router_w_t =
                    x.const_f32_like(router_w.clone(), Shape::from_dims(&[h, cfg.num_experts]));
                let router_logits = x.matmul(&router_w_t)?;
                let router_weights = router_logits.softmax_last_dim()?;

                let mut routed_sum: Option<Tensor> = None;
                for (ei, ew) in experts.iter().enumerate() {
                    let gate = ew.gate_w.apply_linear(x, h, inter)?;
                    let up = ew.up_w.apply_linear(x, h, inter)?;
                    let swiglu = gate.silu().mul(&up)?;
                    let expert_out = ew.down_w.apply_linear(&swiglu, inter, h)?;

                    let w_col = router_weights.slice(2_usize, ei, 1)?;
                    let w_bc = w_col.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
                    let gated = expert_out.mul(&w_bc)?;
                    routed_sum = Some(match routed_sum {
                        Some(s) => s.add(&gated)?,
                        None => gated,
                    });
                }
                Ok(routed_sum.expect("Qwen3-MoE: must have at least one expert"))
            }
        }
    }
}

// ---- GAP-029 increment 3 · persistent-KV decode -----------------------------

impl Qwen3MoeModel {
    /// **Per-layer attention variation** — the same predicate and the same
    /// `sliding_window: None` ⇒ dense subtlety as [`crate::lazy_qwen3`]: prefill
    /// widens an absent window to `seq + 1`, which excludes nothing, so
    /// `use_sliding_window` without a width is dense at every layer.
    ///
    /// Note this is the **attention** axis only. Qwen3Moe's *other* per-layer
    /// variation — dense-vs-MoE FFN on the `decoder_sparse_step` cadence — needs
    /// nothing from the seam: it already lives inside the layer hook, at the
    /// right granularity.
    pub fn decode_mask_plan(&self) -> MaskPlan {
        let cfg = &self.config;
        match (cfg.use_sliding_window, cfg.sliding_window) {
            (true, Some(w)) => {
                MaskPlan::split_window(cfg.num_hidden_layers, cfg.max_window_layers, w)
            }
            _ => MaskPlan::dense(cfg.num_hidden_layers),
        }
    }

    /// Identity a held decode plan is baked against.
    ///
    /// `decoder_sparse_step`, `num_experts` and `moe_intermediate_size` ARE
    /// mixed: unlike the window width they are **structural** — they decide
    /// which layers emit a router plus `num_experts` expert subgraphs, so two
    /// configs differing only there produce genuinely different graphs and must
    /// never share a held plan.
    pub fn decode_shape_key(&self) -> u64 {
        let cfg = &self.config;
        let mut h = crate::decode_shape::ShapeKeyHasher::new();
        h.mix_str("qwen3_moe")
            .mix_instance(self.weights.instance)
            .mix_u64(cfg.num_hidden_layers as u64)
            .mix_u64(cfg.num_attention_heads as u64)
            .mix_u64(cfg.num_key_value_heads as u64)
            .mix_u64(cfg.head_dim as u64)
            .mix_u64(cfg.hidden_size as u64)
            .mix_u64(cfg.intermediate_size as u64)
            .mix_u64(cfg.vocab_size as u64)
            .mix_u64(cfg.decoder_sparse_step as u64)
            .mix_u64(cfg.num_experts as u64)
            .mix_u64(cfg.moe_intermediate_size as u64)
            .mix_f64(cfg.rms_norm_eps);
        self.decode_mask_plan().mix_into(&mut h);
        h.finish()
    }

    /// Decode/prefill through a pre-allocated [`KvCache`], rebuilding the graph
    /// each step.
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
}

impl PersistentDecodeModel for Qwen3MoeModel {
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

impl DecodeBackbone for Qwen3MoeModel {
    fn decode_family(&self) -> &'static str {
        "Qwen3MoeModel"
    }

    fn decode_dims(&self) -> DecodeDims {
        let cfg = &self.config;
        DecodeDims {
            n_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            rope_width: cfg.head_dim,
            embed_scale: None,
        }
    }

    fn decode_shape_key(&self) -> u64 {
        Qwen3MoeModel::decode_shape_key(self)
    }

    fn decode_mask_plan(&self) -> MaskPlan {
        Qwen3MoeModel::decode_mask_plan(self)
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

    /// Attention comes from the shared Qwen3-family block; the FFN is where this
    /// family actually differs, and it reuses [`Self::apply_ffn`] unchanged —
    /// the routing was already at the right granularity for a decode step
    /// (`batch = seq = 1`).
    fn decode_apply_layer(
        &self,
        layer_idx: usize,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let layer = &self.weights.layers[layer_idx];
        let dims = inputs.x.shape();
        let dims = dims.dims();
        let (batch, seq) = (dims[0], dims[1]);

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
                q_norm_gain: &layer.q_norm_gain,
                k_norm_gain: &layer.k_norm_gain,
            },
            inputs,
        )?;

        let h1_norm = h1.rms_norm_affine(Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let ffn_out = self.apply_ffn(&h1_norm, &layer.ffn, batch, seq)?;
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

impl Qwen3MoeWeights {
    /// Load Qwen3-MoE (Qwen/Qwen3-MoE-A*) weights from HF safetensors.
    /// Layer FFN selects Dense vs MoE per `cfg.layer_uses_moe(i)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Qwen3MoeConfig,
    ) -> Result<Self> {
        use crate::lazy::{
            load_tensor_as_f32, load_transposed_matrix,
            load_transposed_matrix_preserve_dtype as ltm,
        };
        let h = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let moe_int = cfg.moe_intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(st, "model.embed_tokens.weight")?);

        let opt_bias = |name: String| -> Option<Arc<[f32]>> {
            load_tensor_as_f32(st, &name).ok().map(Arc::from)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.post_attention_layernorm.weight"),
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
            let q_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.q_norm.weight"),
            )?);
            let k_norm_gain = Arc::from(load_tensor_as_f32(
                st,
                &format!("{p}.self_attn.k_norm.weight"),
            )?);

            let ffn = if cfg.layer_uses_moe(i) {
                // HF gate weight: `[num_experts, hidden]`; transpose to
                // `[hidden, num_experts]` for matmul layout.
                let router_w = Arc::from(load_transposed_matrix(
                    st,
                    &format!("{p}.mlp.gate.weight"),
                    cfg.num_experts,
                    h,
                )?);
                let mut experts = Vec::with_capacity(cfg.num_experts);
                for e in 0..cfg.num_experts {
                    let ep = format!("{p}.mlp.experts.{e}");
                    let gate_w_e = ltm(st, &format!("{ep}.gate_proj.weight"), moe_int, h)?;
                    let up_w = ltm(st, &format!("{ep}.up_proj.weight"), moe_int, h)?;
                    let down_w = ltm(st, &format!("{ep}.down_proj.weight"), h, moe_int)?;
                    experts.push(Qwen3MoeExpertWeights {
                        gate_w: gate_w_e,
                        up_w,
                        down_w,
                    });
                }
                Qwen3MoeFfn::Moe { router_w, experts }
            } else {
                let gate_w = ltm(st, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
                let up_w = ltm(st, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
                let down_w = ltm(st, &format!("{p}.mlp.down_proj.weight"), h, inter)?;
                Qwen3MoeFfn::Dense {
                    gate_w,
                    up_w,
                    down_w,
                }
            };

            layers.push(Qwen3MoeLayerWeights {
                attn_norm_gain,
                ffn_norm_gain,
                attn_q,
                attn_q_bias,
                attn_k,
                attn_k_bias,
                attn_v,
                attn_v_bias,
                attn_o,
                q_norm_gain,
                k_norm_gain,
                ffn,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(st, "model.norm.weight")?);
        let output = match ltm(st, "lm_head.weight", cfg.vocab_size, h) {
            Ok(w) => w,
            Err(_) => crate::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding,
                cfg.vocab_size,
                h,
            ),
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
    fn tiny_weights(cfg: &Qwen3MoeConfig) -> Qwen3MoeWeights {
        let mut s: u32 = 13579;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<Qwen3MoeLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|li| {
                let ffn = if cfg.layer_uses_moe(li) {
                    let router_w = vec_of(h * cfg.num_experts, &mut *nb);
                    let experts: Vec<Qwen3MoeExpertWeights> = (0..cfg.num_experts)
                        .map(|_| Qwen3MoeExpertWeights {
                            gate_w: WeightStorage::F32(vec_of(h * moe_inter, &mut *nb)),
                            up_w: WeightStorage::F32(vec_of(h * moe_inter, &mut *nb)),
                            down_w: WeightStorage::F32(vec_of(moe_inter * h, &mut *nb)),
                        })
                        .collect();
                    Qwen3MoeFfn::Moe { router_w, experts }
                } else {
                    Qwen3MoeFfn::Dense {
                        gate_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                        up_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                        down_w: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                    }
                };
                Qwen3MoeLayerWeights {
                    attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                    ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
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
                    q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                    k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                    ffn,
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Qwen3MoeWeights {
            instance: crate::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        }
    }

    #[test]
    fn forward_with_alternating_dense_and_moe() {
        // decoder_sparse_step = 2 → layers 1 and 3 (0-indexed) use MoE,
        // layer 0 and 2 use dense.
        let cfg = Qwen3MoeConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            head_dim: 4,
            attention_bias: false,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            decoder_sparse_step: 2,
            moe_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
        };
        // Confirm the FFN-mode mapping is what we expect.
        assert!(!cfg.layer_uses_moe(0));
        assert!(cfg.layer_uses_moe(1));
        assert!(!cfg.layer_uses_moe(2));
        assert!(cfg.layer_uses_moe(3));
        let model = Qwen3MoeModel {
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
        let cfg = Qwen3MoeConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 2,
            max_position_embeddings: 32,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            decoder_sparse_step: 2,
            moe_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            attention_bias: false,
        };
        let model = Qwen3MoeModel {
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

    fn forward_embeds_test_cfg() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 2,
            max_position_embeddings: 32,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            decoder_sparse_step: 2,
            moe_intermediate_size: 8,
            num_experts: 2,
            num_experts_per_tok: 1,
            attention_bias: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3MoeModel {
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
            "Qwen3MoE forward vs forward_embeds must agree (max diff {max_diff})"
        );
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3MoeModel {
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
        let model = Qwen3MoeModel {
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
            "Qwen3MoE forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})"
        );
    }

    // ---- GAP-029 increment 3, family 4: persistent decode -------------------

    /// Measured, not inherited — the natural template's `diff < 5e-3 ||
    /// rel < 1e-2` sits ABOVE the ~7e-3 single-mask divergence.
    const DECODE_ORACLE_ABS: f32 = 1e-5;

    /// 2 layers, window 4, `max_window_layers: 1`. `decoder_sparse_step: 1` so
    /// **every** layer is MoE — the decode path must carry router + experts, not
    /// just the dense-FFN arm.
    fn mixed_window_cfg() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            head_dim: 4,
            attention_bias: true,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            sliding_window: Some(4),
            max_window_layers: 1,
            use_sliding_window: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            decoder_sparse_step: 1,
            moe_intermediate_size: 16,
            num_experts: 3,
            num_experts_per_tok: 3,
        }
    }

    /// Max |logit diff| per decode step against the per-layer-gated non-cached
    /// forward. `>= 3` decode steps so the assertions reach the REBIND path.
    fn decode_vs_forward_max_abs(cfg: &Qwen3MoeConfig, tokens: &[u32], prefill: usize) -> Vec<f32> {
        let n_decode = tokens.len() - prefill;
        assert!(
            n_decode >= 3,
            "need >= 3 decode tokens to reach the rebind path"
        );
        let model = Qwen3MoeModel {
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

    /// ⚠️ **NON-DISCRIMINATION CONTROL.** With `max_window_layers: 0` the plan is
    /// a single dense variant, so this passes under BOTH a correct windowed plan
    /// and one that ignores windowing. It certifies the seam, the shared Qwen3
    /// attention block **and the MoE routing under decode** — never the
    /// windowing.
    #[test]
    fn qwen3_moe_decode_matches_forward_when_no_layer_is_windowed() {
        let cfg = Qwen3MoeConfig {
            max_window_layers: 0,
            ..mixed_window_cfg()
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let diffs = decode_vs_forward_max_abs(&cfg, &tokens, 3);
        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "unwindowed Qwen3Moe decode diverged: per-step max|diff| = {diffs:?}. \
             This is the CONTROL — the windowed test proves nothing until it is green.",
        );
    }

    /// **GAP-029 family 4 — Qwen3Moe windowed persistent decode, every layer MoE.**
    ///
    /// **Born red, observed.** With `decode_mask_plan` returning
    /// `MaskPlan::dense(..)` — precisely what a single-mask decode port
    /// computes — the measured per-step divergence was
    ///
    /// ```text
    /// absolute position 3, 4, 5 : [0.0, 4.654333e-3, 1.5804386e-2]
    /// ```
    ///
    /// while the control stayed green (`7 passed; 2 failed`, the other failure
    /// being the dense-FFN sibling). Restoring the real `split_window` plan took
    /// every step to **0.0**.
    ///
    /// **The leading zero is the discrimination evidence:** a window of 4 cannot
    /// exclude anything until absolute position 4, so a degenerate oracle would
    /// have shown three zeros and this showed one. **Both failing steps are
    /// REBIND steps.**
    ///
    /// Note the divergence GROWS with position here (4.7e-3 → 1.6e-2) where
    /// Qwen3's shrank — expected, since more prefix falls outside the window as
    /// position advances, and nothing about decode forces the trend either way.
    #[test]
    fn qwen3_moe_windowed_decode_matches_per_layer_gated_forward() {
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
        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "windowed Qwen3Moe decode diverged from the per-layer-gated forward: \
             per-step max|diff| at absolute positions 3..=5 = {diffs:?} (limit \
             {DECODE_ORACLE_ABS:e}).",
        );
    }

    /// **The same claim on the OTHER FFN cadence**, which is a different code
    /// path through `apply_ffn`, not a different tolerance.
    ///
    /// `decoder_sparse_step: 2` over 2 layers puts a **dense SwiGLU** on layer 0
    /// (the windowed one) and MoE on layer 1 — the mirror of the sibling above,
    /// where every layer is MoE. Both cadences are real Qwen3Moe configurations
    /// and both must agree with prefill.
    ///
    /// Measured born-red under a single-mask plan:
    ///
    /// ```text
    /// absolute position 3, 4, 5 : [0.0, 2.6300699e-3, 9.746924e-3]
    /// ```
    ///
    /// **0.0 at every step** under the real plan.
    ///
    /// ⚠️ **A prediction of mine was refuted here and the record is kept rather
    /// than tidied.** Before measuring, I expected the all-MoE sibling to be
    /// *unable* to discriminate windowing — reasoning that dense routing
    /// softmaxes over every expert and would average the layer-0 masking
    /// difference away before it reached the logits. **That was wrong: the
    /// all-MoE config diverges MORE (1.58e-2) than this one (9.7e-3).** The
    /// argument was plausible and unmeasured, and had it not been run it would
    /// have been written into the file as a reason. Both tests are kept for the
    /// real reason — two FFN cadences, two code paths — not the invented one.
    #[test]
    fn qwen3_moe_dense_ffn_layers_expose_the_windowed_mask() {
        let cfg = Qwen3MoeConfig {
            decoder_sparse_step: 2,
            ..mixed_window_cfg()
        };
        assert!(
            !cfg.layer_uses_moe(0),
            "layer 0 (the windowed one) must be dense here"
        );
        assert!(cfg.layer_uses_moe(1), "layer 1 must still be MoE");
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let diffs = decode_vs_forward_max_abs(&cfg, &tokens, 3);
        let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            worst < DECODE_ORACLE_ABS,
            "windowed Qwen3Moe decode (dense FFN on the windowed layer) diverged: \
             per-step max|diff| at absolute positions 3..=5 = {diffs:?} (limit \
             {DECODE_ORACLE_ABS:e}). A single mask on every layer produces exactly \
             this signature and leaves position 3 clean.",
        );
    }

    /// ⚠️ `use_sliding_window: true` with `sliding_window: None` is **dense** —
    /// prefill widens an absent width to `seq + 1`, which excludes nothing.
    #[test]
    fn qwen3_moe_absent_window_width_is_dense_at_every_layer() {
        let cfg = Qwen3MoeConfig {
            sliding_window: None,
            max_window_layers: 2,
            ..mixed_window_cfg()
        };
        let plan = Qwen3MoeModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        }
        .decode_mask_plan();
        assert_eq!(
            plan.n_variants(),
            1,
            "absent width must collapse to one dense variant"
        );
    }
}
