//! Mixtral (sparse Mixture-of-Experts) decoder ported to the
//! lazy-graph API.
//!
//! Phase D LLM port. Mixtral is "Mistral + per-layer MoE FFN":
//! standard Mistral attention (GQA + sliding-window mask) plus a
//! sparse top-K-routed MoE block instead of the dense SwiGLU FFN.
//! Sibling of `crate::lazy_qwen2_moe::Qwen2MoeModel`; the two
//! share the dense-routing pattern (every expert evaluated, output
//! weighted by full router softmax) for the v1 lazy port.
//!
//! # Dense vs top-K routing (v1 trade-off)
//!
//! Trained Mixtral uses **top-K routing** (top-2 of 8 experts per
//! token, with renormalization). The dense-routing v1 here evaluates
//! all 8 experts on every token and sums their outputs weighted by
//! the full router softmax. Mathematical implications:
//!
//! - **Pros**: pure-functional graph, no dynamic top-K selection
//!   in the IR. The lazy stack stays static.
//! - **Cons**: 4× FFN compute vs trained top-K (Mixtral evaluates
//!   2/8 = 25% of experts), and the output isn't a bit-exact match
//!   for the trained top-K model. The summed activations are still
//!   in the right neighborhood per token but the model's per-token
//!   routing-sparsity behavior is gone.
//!
//! Adding true top-K routing needs either a new IR op
//! (`Op::TopKRoute` returning (indices, weights, gated experts))
//! or a lazy primitive for per-token sparse expert dispatch. Deferred
//! until a Mixtral-class consumer needs trained-routing parity.
//!
//! # Scope (v1)
//!
//! - Forward-only, single sequence (`batch == 1`), no KV cache.
//! - Sliding-window mask (Mistral semantics).
//! - F32 activations; weights via `WeightStorage`.
//! - **No shared expert** (Mixtral has none; Qwen2-MoE does).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct MixtralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub num_experts_per_tok: usize,
    pub num_local_experts: usize,
}

impl MixtralConfig {
    /// `mistralai/Mixtral-8x7B-v0.1`.
    pub fn mixtral_8x7b_v01() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 4096,
            intermediate_size: 14_336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 32_768,
            rms_norm_eps: 1e-5,
            rope_theta: 1e6,
            sliding_window: Some(4096),
            num_experts_per_tok: 2,
            num_local_experts: 8,
        }
    }
}

/// One Mixtral expert's SwiGLU MLP weights.
#[derive(Debug, Clone)]
pub struct MixtralExpertWeights {
    pub gate_w: WeightStorage,
    pub up_w: WeightStorage,
    pub down_w: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MixtralLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    // Bias-free attention (Mistral lineage).
    pub attn_q: WeightStorage,
    pub attn_k: WeightStorage,
    pub attn_v: WeightStorage,
    pub attn_o: WeightStorage,
    // Router: `[hidden_size, num_experts]`.
    pub gate_w: Arc<[f32]>,
    pub experts: Vec<MixtralExpertWeights>,
}

#[derive(Debug, Clone)]
pub struct MixtralWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<MixtralLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MixtralModel {
    pub config: MixtralConfig,
    pub weights: MixtralWeights,
}

impl MixtralModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection — useful for Mixtral-
    /// based embedding adapters (analogous to NV-Embed-v2
    /// running on Mistral). Mirrors the `forward_hidden`
    /// pattern across the LLM family.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips the token-embedding step and runs
    /// the decoder over pre-embedded inputs. Mixtral does NOT scale
    /// embeddings — `embeds` is passed raw.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.hidden_size, tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self.weights.output.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Shared backbone: embed → RoPE → per-layer attn+MoE →
    /// final RmsNorm. Used by both `forward` (then matmuls
    /// with `lm_head`) and `forward_hidden` (returns hidden
    /// states directly).
    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "MixtralModel: tokens must be non-empty");

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "MixtralModel::forward_embeds: expected embeds shape \
                 (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "MixtralModel::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "MixtralConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(crate::Error::Msg(
                "MixtralConfig: num_attention_heads must be a multiple of num_key_value_heads".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn build_sliding_window_mask(&self, anchor: &LazyTensor, seq: usize) -> LazyTensor {
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
        x: &LazyTensor,
        layer: &MixtralLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        // Pre-attention RmsNorm.
        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        // Q / K / V — bias-free Mistral-style.
        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size);
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim);
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim);

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // GQA replication.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_sliding_window_mask(x, seq);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;

        // Pre-FFN RmsNorm + MoE block.
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;
        let ffn_out = self.apply_moe(&h1_norm, layer, batch, seq)?;
        h1.add(&ffn_out)
    }

    /// Dense MoE: evaluate every expert on every token, weight by
    /// the router softmax, sum. v1 trade-off documented at the
    /// module level.
    fn apply_moe(
        &self,
        x: &LazyTensor,
        layer: &MixtralLayerWeights,
        batch: usize,
        seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let e = cfg.num_local_experts;

        // Router: `[batch, seq, hidden]` @ `[hidden, e]` → `[batch, seq, e]`
        // → softmax over expert axis.
        let gate_w = x.const_f32_like(layer.gate_w.clone(), Shape::from_dims(&[h, e]));
        let router_logits = x.matmul(&gate_w)?;
        let router_weights = router_logits.softmax_last_dim()?;  // [batch, seq, e]

        let mut routed_sum: Option<LazyTensor> = None;
        for (ei, ew) in layer.experts.iter().enumerate() {
            // SwiGLU expert: `down(silu(gate(x)) * up(x))`.
            let gate = ew.gate_w.apply_linear(x, h, inter);
            let up = ew.up_w.apply_linear(x, h, inter);
            let swiglu = gate.silu().mul(&up)?;
            let expert_out = ew.down_w.apply_linear(&swiglu, inter, h);

            // Gate weight column for this expert: [batch, seq, 1].
            let w_col = router_weights.slice(2_usize, ei, 1)?;
            let w_bc = w_col.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
            let gated = expert_out.mul(&w_bc)?;
            routed_sum = Some(match routed_sum {
                Some(s) => s.add(&gated)?,
                None => gated,
            });
        }
        Ok(routed_sum.expect("MoE: must have at least one expert"))
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl MixtralWeights {
    /// Load Mixtral (mistralai/Mixtral-8x*) weights from HuggingFace safetensors.
    /// Standard Mistral-style attn (no biases) + per-expert SwiGLU under
    /// `model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &MixtralConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let token_embedding = Arc::from(load_tensor_as_f32(
            st, "model.embed_tokens.weight",
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.post_attention_layernorm.weight"),
            )?);
            let attn_q = ltm(st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            let attn_k = ltm(st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            let attn_v = ltm(st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            let attn_o = ltm(st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            // Router gate.
            let gate_w = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.block_sparse_moe.gate.weight"),
            )?);
            let mut experts = Vec::with_capacity(cfg.num_local_experts);
            for e in 0..cfg.num_local_experts {
                let ep = format!("{p}.block_sparse_moe.experts.{e}");
                let gate_w_e = ltm(st, &format!("{ep}.w1.weight"), inter, h)?;
                let down_w = ltm(st, &format!("{ep}.w2.weight"), h, inter)?;
                let up_w = ltm(st, &format!("{ep}.w3.weight"), inter, h)?;
                experts.push(MixtralExpertWeights {
                    gate_w: gate_w_e, up_w, down_w,
                });
            }
            layers.push(MixtralLayerWeights {
                attn_norm_gain, ffn_norm_gain,
                attn_q, attn_k, attn_v, attn_o,
                gate_w, experts,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(
            st, "model.norm.weight",
        )?);
        let output = ltm(st, "lm_head.weight", cfg.vocab_size, h)?;

        Ok(Self { token_embedding, layers, final_norm_gain, output })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &MixtralConfig) -> MixtralWeights {
        let mut s: u32 = 91011;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<MixtralLayerWeights> = (0..cfg.num_hidden_layers).map(|_| {
            let attn_q = WeightStorage::F32(vec_of(h * h, &mut *next_box));
            let attn_k = WeightStorage::F32(vec_of(h * kv, &mut *next_box));
            let attn_v = WeightStorage::F32(vec_of(h * kv, &mut *next_box));
            let attn_o = WeightStorage::F32(vec_of(h * h, &mut *next_box));
            let gate_w = vec_of(h * cfg.num_local_experts, &mut *next_box);
            let experts: Vec<MixtralExpertWeights> = (0..cfg.num_local_experts).map(|_| {
                MixtralExpertWeights {
                    gate_w: WeightStorage::F32(vec_of(h * inter, &mut *next_box)),
                    up_w:   WeightStorage::F32(vec_of(h * inter, &mut *next_box)),
                    down_w: WeightStorage::F32(vec_of(inter * h, &mut *next_box)),
                }
            }).collect();
            MixtralLayerWeights {
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
                attn_q, attn_k, attn_v, attn_o,
                gate_w, experts,
            }
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        MixtralWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_shape_and_finite_2_layer_4_experts() {
        let cfg = MixtralConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: Some(8),
            num_experts_per_tok: 2,
            num_local_experts: 4,
        };
        let model = MixtralModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// Removing all experts but the first should produce a different
    /// output (the dense-routing sum changes with the expert pool).
    #[test]
    fn fewer_experts_changes_output() {
        let mut cfg = MixtralConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: None,
            num_experts_per_tok: 1,
            num_local_experts: 4,
        };
        let w_full = tiny_weights(&cfg);
        // Truncate to 1 expert. Need to rebuild gate_w to match new size.
        let mut w_one = w_full.clone();
        for l in &mut w_one.layers {
            l.experts.truncate(1);
            // gate_w shape was [h, 4]; need [h, 1] now. Take a slice.
            let h = cfg.hidden_size;
            let new_gate: Vec<f32> = (0..h).map(|i| l.gate_w[i * 4]).collect();
            l.gate_w = Arc::from(new_gate);
        }
        let out_full = MixtralModel { config: cfg.clone(), weights: w_full }
            .forward(&[1, 2, 3], 0).unwrap().realize_f32();
        cfg.num_local_experts = 1;
        let out_one = MixtralModel { config: cfg, weights: w_one }
            .forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let any_diff = out_full.iter().zip(out_one.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "4-expert vs 1-expert dense MoE must differ");
    }

    /// `forward_hidden` returns post-RmsNorm hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    /// Same backbone as `forward`, just skips the final
    /// projection.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = MixtralConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: None,
            num_experts_per_tok: 2,
            num_local_experts: 4,
        };
        let model = MixtralModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> MixtralConfig {
        MixtralConfig {
            vocab_size: 32, hidden_size: 16, intermediate_size: 32,
            num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2,
            head_dim: 4, max_position_embeddings: 64,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
            sliding_window: None,
            num_experts_per_tok: 2, num_local_experts: 4,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = MixtralModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Mixtral forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = MixtralModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = MixtralModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Mixtral forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}
