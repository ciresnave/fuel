//! DeepSeek-V2 (Multi-head Latent Attention + MoE) ported to the
//! lazy-graph API.
//!
//! Phase D specialized port. DeepSeek-V2 introduces **Multi-head
//! Latent Attention (MLA)** — a compression-based attention
//! mechanism designed to slash the KV-cache cost during decode
//! while preserving multi-head expressiveness:
//!
//!   - **Q** is split into a NoPE part (`qk_nope_head_dim` per
//!     head) and a RoPE part (`qk_rope_head_dim` per head).
//!     Optionally produced via LoRA (`q_a_proj → norm →
//!     q_b_proj`) when `q_lora_rank` is set; falls back to a
//!     plain projection otherwise.
//!   - **KV** flows through a low-rank latent path:
//!     ```text
//!     compressed_kv, k_pe = kv_a_proj_with_mqa(x).split(
//!                                kv_lora_rank, qk_rope_head_dim)
//!     k_nope, v = kv_b_proj(layernorm(compressed_kv))
//!                     .split(qk_nope_head_dim, v_head_dim)
//!     ```
//!     `k_pe` is **single-head** (MQA-shared) and gets broadcast
//!     across all heads.
//!   - **Attention**: `Q = cat(q_nope, q_pe)`,
//!     `K = cat(k_nope, k_pe_repeated)`. Softmax-scaled with an
//!     mscale-adjusted scale if YaRN scaling is on (v1: plain
//!     RoPE only, YaRN deferred — `softmax_scale = 1 /
//!     sqrt(q_head_dim)`).
//!
//! The MoE block follows the Qwen2-MoE pattern adopted by Phase
//! D batch B: dense routing (full softmax × every expert),
//! plus an always-on **shared-expert** branch (`n_shared_experts
//! > 0`). The `first_k_dense_replace` config skips MoE for the
//! first K layers (they use a plain SwiGLU MLP instead).
//!
//! v1 deferrals:
//!   - **YaRN / Su / Dynamic / Linear RoPE scaling**. v1 uses
//!     plain RoPE with `rope_theta`.
//!   - **Group-limited top-K routing** (`n_group`, `topk_group`,
//!     `TopkMethod::GroupLimitedGreedy`). v1 uses dense softmax
//!     routing (every expert evaluated, weighted).
//!   - **routed_scaling_factor**. Applied as a no-op (factor=1)
//!     by default.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! F32. Both LoRA-Q (DeepSeek-V2) and plain-Q (DeepSeek-V2-Lite)
//! configurations supported.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_glm4::apply_interleaved_partial_rope;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeek2Activation {
    Silu,
    Gelu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeek2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    /// Layer `i` uses MoE iff `i >= first_k_dense_replace && (i %
    /// moe_layer_freq == 0)` and `n_routed_experts > 0`. Default
    /// is `1` (every layer past the dense replace boundary).
    pub moe_layer_freq: usize,
    pub first_k_dense_replace: usize,
    pub norm_topk_prob: bool,
    pub hidden_activation: DeepSeek2Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub attention_bias: bool,
    /// MLA Q-LoRA rank. `None` → plain Q projection.
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
}

impl DeepSeek2Config {
    pub fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }
    /// True iff this layer uses MoE (else plain dense MLP).
    pub fn layer_uses_moe(&self, layer_idx: usize) -> bool {
        let n_routed = self.n_routed_experts.unwrap_or(0);
        n_routed > 0
            && layer_idx >= self.first_k_dense_replace
            && (layer_idx - self.first_k_dense_replace) % self.moe_layer_freq == 0
    }
}

#[derive(Debug, Clone)]
pub enum DeepSeek2QProj {
    Plain(WeightStorage),
    Lora {
        a: WeightStorage,
        norm_gain: Arc<[f32]>,
        b: WeightStorage,
    },
}

#[derive(Debug, Clone)]
pub struct DeepSeek2MlaWeights {
    pub q_proj: DeepSeek2QProj,
    /// `[hidden, kv_lora_rank + qk_rope_head_dim]`.
    pub kv_a_proj_with_mqa: WeightStorage,
    pub kv_a_layernorm_gain: Arc<[f32]>,
    /// `[kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]`.
    pub kv_b_proj: WeightStorage,
    /// `[num_heads * v_head_dim, hidden]`.
    pub o_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2DenseMlpWeights {
    pub gate: WeightStorage,
    pub up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2ExpertWeights {
    pub gate: WeightStorage,
    pub up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2MoeWeights {
    /// `[hidden, n_routed_experts]` routing matrix.
    pub router: Arc<[f32]>,
    pub experts: Vec<DeepSeek2ExpertWeights>,
    /// Shared expert (always-on). Intermediate size =
    /// `n_shared_experts * moe_intermediate_size`.
    pub shared_gate: WeightStorage,
    pub shared_up: WeightStorage,
    pub shared_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub enum DeepSeek2FfnWeights {
    Dense(DeepSeek2DenseMlpWeights),
    Moe(DeepSeek2MoeWeights),
}

#[derive(Debug, Clone)]
pub struct DeepSeek2LayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub mla: DeepSeek2MlaWeights,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub ffn: DeepSeek2FfnWeights,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<DeepSeek2LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    /// Optional separate lm_head. None ⇒ tied to token_embedding.
    pub lm_head: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2Model {
    pub config: DeepSeek2Config,
    pub weights: DeepSeek2Weights,
}

impl DeepSeek2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let lm_head_w = match &weights.lm_head {
            Some(w) => w.clone(),
            None => WeightStorage::F32(weights.token_embedding.clone()),
        };
        Ok(lm_head_w.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// DeepSeek-V2-specific: MLA attention, per-layer dense /
    /// MoE FFN selection (first `n` dense layers, then MoE).
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert_eq!(
            weights.layers.len(), cfg.num_hidden_layers,
            "weights.layers length must match num_hidden_layers",
        );

        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.qk_rope_head_dim,
        );

        for (idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, idx, &rope_cos, &rope_sin)?;
        }
        Ok(h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)?)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &DeepSeek2LayerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;
        let attn = self.mla_attention(&x_norm, &layer.mla, rope_cos, rope_sin)?;
        let h1 = x.add(&attn)?;

        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;
        let expected_moe = cfg.layer_uses_moe(layer_idx);
        let mlp_out = match (&layer.ffn, expected_moe) {
            (DeepSeek2FfnWeights::Dense(w), false) => self.apply_dense_mlp(&h1_norm, w)?,
            (DeepSeek2FfnWeights::Moe(w), true) => self.apply_moe(&h1_norm, w)?,
            _ => panic!(
                "layer {layer_idx}: FFN weight kind does not match config-derived kind (uses_moe={expected_moe})",
            ),
        };
        h1.add(&mlp_out)
    }

    fn mla_attention(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MlaWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;

        // ---- Q projection (plain or LoRA) -----------------------------------
        let q = match &w.q_proj {
            DeepSeek2QProj::Plain(plain) => {
                plain.apply_linear(x, cfg.hidden_size, n_heads * q_head_dim)
            }
            DeepSeek2QProj::Lora { a, norm_gain, b } => {
                let lo = a.apply_linear(x, cfg.hidden_size, norm_gain.len());
                let lo_norm = lo.rms_norm_affine(Arc::clone(norm_gain), cfg.rms_norm_eps)?;
                b.apply_linear(&lo_norm, norm_gain.len(), n_heads * q_head_dim)
            }
        };
        let _ = (batch, seq);
        let q = q.split_heads(n_heads, q_head_dim)?;
        // Split Q on the last dim into (q_nope, q_pe).
        let q_nope = q.slice(3_usize, 0, nope)?;
        let q_pe = q.slice(3_usize, nope, rope)?;

        // ---- KV compressed projection ---------------------------------------
        let kv_a = w.kv_a_proj_with_mqa.apply_linear(
            x, cfg.hidden_size, cfg.kv_lora_rank + rope,
        );
        let compressed_kv = kv_a.slice(2_usize, 0, cfg.kv_lora_rank)?;
        let k_pe_single = kv_a.slice(2_usize, cfg.kv_lora_rank, rope)?;
        // k_pe shape (b, seq, rope) → (b, 1, seq, rope) for MQA broadcast.
        let k_pe_single_h = k_pe_single.split_heads(1, rope)?;

        let compressed_kv_norm = compressed_kv.rms_norm_affine(std::sync::Arc::clone(&w.kv_a_layernorm_gain), cfg.rms_norm_eps)?;
        let kv = w.kv_b_proj.apply_linear(
            &compressed_kv_norm, cfg.kv_lora_rank, n_heads * (nope + v_dim),
        );
        let kv = kv.split_heads(n_heads, nope + v_dim)?;
        let k_nope = kv.slice(3_usize, 0, nope)?;
        let v = kv.slice(3_usize, nope, v_dim)?;

        // ---- RoPE on q_pe and k_pe (interleaved) ----------------------------
        let q_pe_rot = apply_interleaved_partial_rope(&q_pe, rope_cos, rope_sin, rope, rope)?;
        let k_pe_rot = apply_interleaved_partial_rope(&k_pe_single_h, rope_cos, rope_sin, rope, rope)?;

        // Broadcast k_pe_rot from (b, 1, seq, rope) to (b, n_heads, seq, rope).
        let k_pe_full = k_pe_rot
            .broadcast_to(Shape::from_dims(&[batch, n_heads, seq, rope]))?;

        // Cat Q and K along the head_dim axis.
        let q_full = q_nope.concat(&q_pe_rot, 3_usize)?;
        let k_full = k_nope.concat(&k_pe_full, 3_usize)?;

        // ---- Attention ------------------------------------------------------
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (q_head_dim as f64).sqrt();
        let scores = q_full.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?; // (b, n_heads, seq, v_dim)

        let merged = ctx.merge_heads()?;
        Ok(w.o_proj.apply_linear(&merged, n_heads * v_dim, cfg.hidden_size))
    }

    fn apply_dense_mlp(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2DenseMlpWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let gate = w.gate.apply_linear(x, h, inter);
        let up = w.up.apply_linear(x, h, inter);
        let activated = match cfg.hidden_activation {
            DeepSeek2Activation::Silu => gate.silu(),
            DeepSeek2Activation::Gelu => gate.gelu_erf(),
        };
        let inner = activated.mul(&up)?;
        Ok(w.down.apply_linear(&inner, inter, h))
    }

    fn apply_moe(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MoeWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let n_routed = cfg.n_routed_experts.unwrap_or(0);
        let n_shared = cfg.n_shared_experts.unwrap_or(0);
        assert!(n_routed > 0, "MoE block requires n_routed_experts > 0");
        assert_eq!(w.experts.len(), n_routed,
            "MoE weights expert count {} != n_routed_experts {n_routed}",
            w.experts.len());

        // Routed path (dense routing — full softmax × every expert).
        let router_t = x.const_f32_like(
            w.router.clone(),
            Shape::from_dims(&[h, n_routed]),
        );
        let router_logits = x.matmul(&router_t)?;
        let routing_weights = router_logits.softmax_last_dim()?;

        let mut routed_sum: Option<LazyTensor> = None;
        for (ei, ew) in w.experts.iter().enumerate() {
            let gate = ew.gate.apply_linear(x, h, inter);
            let up = ew.up.apply_linear(x, h, inter);
            let activated = match cfg.hidden_activation {
                DeepSeek2Activation::Silu => gate.silu(),
                DeepSeek2Activation::Gelu => gate.gelu_erf(),
            };
            let inner = activated.mul(&up)?;
            let expert_out = ew.down.apply_linear(&inner, inter, h);
            let w_col = routing_weights.slice(2_usize, ei, 1)?;
            let w_bc = w_col.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
            let gated = expert_out.mul(&w_bc)?;
            routed_sum = Some(match routed_sum {
                Some(s) => s.add(&gated)?,
                None => gated,
            });
        }
        let routed = routed_sum.expect("MoE: at least one expert");

        // Shared-expert path (always on, no gating).
        if n_shared == 0 {
            return Ok(routed);
        }
        let shared_inter = n_shared * inter;
        let s_gate = w.shared_gate.apply_linear(x, h, shared_inter);
        let s_up = w.shared_up.apply_linear(x, h, shared_inter);
        let s_act = match cfg.hidden_activation {
            DeepSeek2Activation::Silu => s_gate.silu(),
            DeepSeek2Activation::Gelu => s_gate.gelu_erf(),
        };
        let s_inner = s_act.mul(&s_up)?;
        let s_out = w.shared_down.apply_linear(&s_inner, shared_inter, h);
        routed.add(&s_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_mla_weights(cfg: &DeepSeek2Config, nb: &mut Box<dyn FnMut() -> f32>) -> DeepSeek2MlaWeights {
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;

        let q_proj = match cfg.q_lora_rank {
            None => DeepSeek2QProj::Plain(WeightStorage::F32(vec_of(h * n_heads * q_head_dim, &mut **nb))),
            Some(lora) => DeepSeek2QProj::Lora {
                a: WeightStorage::F32(vec_of(h * lora, &mut **nb)),
                norm_gain: Arc::from(vec![1.0_f32; lora]),
                b: WeightStorage::F32(vec_of(lora * n_heads * q_head_dim, &mut **nb)),
            },
        };
        DeepSeek2MlaWeights {
            q_proj,
            kv_a_proj_with_mqa: WeightStorage::F32(vec_of(h * (cfg.kv_lora_rank + rope), &mut **nb)),
            kv_a_layernorm_gain: Arc::from(vec![1.0_f32; cfg.kv_lora_rank]),
            kv_b_proj: WeightStorage::F32(vec_of(cfg.kv_lora_rank * n_heads * (nope + v_dim), &mut **nb)),
            o_proj: WeightStorage::F32(vec_of(n_heads * v_dim * h, &mut **nb)),
        }
    }

    fn tiny_dense_mlp(cfg: &DeepSeek2Config, nb: &mut Box<dyn FnMut() -> f32>) -> DeepSeek2DenseMlpWeights {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        DeepSeek2DenseMlpWeights {
            gate: WeightStorage::F32(vec_of(h * i, &mut **nb)),
            up: WeightStorage::F32(vec_of(h * i, &mut **nb)),
            down: WeightStorage::F32(vec_of(i * h, &mut **nb)),
        }
    }

    fn tiny_moe(cfg: &DeepSeek2Config, nb: &mut Box<dyn FnMut() -> f32>) -> DeepSeek2MoeWeights {
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let n_routed = cfg.n_routed_experts.unwrap_or(0);
        let n_shared = cfg.n_shared_experts.unwrap_or(0);
        let shared_inter = n_shared * inter;
        let router = vec_of(h * n_routed, &mut **nb);
        let experts: Vec<DeepSeek2ExpertWeights> = (0..n_routed)
            .map(|_| DeepSeek2ExpertWeights {
                gate: WeightStorage::F32(vec_of(h * inter, &mut **nb)),
                up: WeightStorage::F32(vec_of(h * inter, &mut **nb)),
                down: WeightStorage::F32(vec_of(inter * h, &mut **nb)),
            })
            .collect();
        DeepSeek2MoeWeights {
            router, experts,
            shared_gate: WeightStorage::F32(vec_of(h * shared_inter, &mut **nb)),
            shared_up: WeightStorage::F32(vec_of(h * shared_inter, &mut **nb)),
            shared_down: WeightStorage::F32(vec_of(shared_inter * h, &mut **nb)),
        }
    }

    fn tiny_weights(cfg: &DeepSeek2Config) -> DeepSeek2Weights {
        let mut s: u32 = 99999;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<DeepSeek2LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let ffn = if cfg.layer_uses_moe(i) {
                    DeepSeek2FfnWeights::Moe(tiny_moe(cfg, &mut nb))
                } else {
                    DeepSeek2FfnWeights::Dense(tiny_dense_mlp(cfg, &mut nb))
                };
                DeepSeek2LayerWeights {
                    input_norm_gain: Arc::from(vec![1.0_f32; h]),
                    mla: tiny_mla_weights(cfg, &mut nb),
                    post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                    ffn,
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)))
        };
        DeepSeek2Weights {
            token_embedding, layers,
            final_norm_gain, lm_head,
        }
    }

    fn tiny_config_lora_q() -> DeepSeek2Config {
        DeepSeek2Config {
            vocab_size: 16, hidden_size: 16,
            intermediate_size: 32, moe_intermediate_size: 8,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            n_shared_experts: Some(1),
            n_routed_experts: Some(2),
            num_experts_per_tok: Some(1),
            moe_layer_freq: 1,
            first_k_dense_replace: 1,  // layer 0 is dense; layers 1, 2 are MoE
            norm_topk_prob: false,
            hidden_activation: DeepSeek2Activation::Silu,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            rope_theta: 10_000.0,
            attention_bias: false,
            q_lora_rank: Some(8),
            qk_rope_head_dim: 4,
            kv_lora_rank: 8,
            v_head_dim: 4,
            qk_nope_head_dim: 4,
        }
    }

    fn tiny_config_plain_q() -> DeepSeek2Config {
        DeepSeek2Config { q_lora_rank: None, ..tiny_config_lora_q() }
    }

    #[test]
    fn forward_shape_and_finite_lora_q() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn forward_shape_and_finite_plain_q() {
        let cfg = tiny_config_plain_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// `first_k_dense_replace` actually skips MoE in early layers.
    #[test]
    fn dense_replace_layer_uses_dense_mlp() {
        let cfg = tiny_config_lora_q();
        assert!(!cfg.layer_uses_moe(0));
        assert!(cfg.layer_uses_moe(1));
        assert!(cfg.layer_uses_moe(2));
    }

    /// MLA k_pe is MQA-shared (single head, broadcast). Zero
    /// out the kv_a_proj_with_mqa columns that produce k_pe
    /// (the last `qk_rope_head_dim` columns) and confirm the
    /// output changes.
    #[test]
    fn mla_k_pe_is_wired() {
        let cfg = DeepSeek2Config { num_hidden_layers: 1, ..tiny_config_lora_q() };
        let h = cfg.hidden_size;
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        let rope = cfg.qk_rope_head_dim;
        let kv_a_full_size = cfg.kv_lora_rank + rope;
        // Zero the k_pe slice (the last `rope` columns of kv_a_proj_with_mqa).
        let mut kv_a_v = match &zeroed.layers[0].mla.kv_a_proj_with_mqa {
            WeightStorage::F32(v) => v.to_vec(),
            _ => panic!(),
        };
        for row in 0..h {
            for j in cfg.kv_lora_rank..kv_a_full_size {
                kv_a_v[row * kv_a_full_size + j] = 0.0;
            }
        }
        zeroed.layers[0].mla.kv_a_proj_with_mqa = WeightStorage::F32(Arc::from(kv_a_v));
        let m_base = DeepSeek2Model { config: cfg.clone(), weights: base };
        let m_zero = DeepSeek2Model { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, 0).unwrap().realize_f32();
        let b = m_zero.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-8,
            "k_pe path must be wired (zeroing kv_a's rope cols alters output), max_diff = {max_diff}");
    }

    /// Shared expert must contribute alongside routed experts.
    #[test]
    fn shared_expert_contributes() {
        let cfg = DeepSeek2Config {
            // One MoE-only layer.
            num_hidden_layers: 1,
            first_k_dense_replace: 0,
            ..tiny_config_lora_q()
        };
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        if let DeepSeek2FfnWeights::Moe(m) = &mut zeroed.layers[0].ffn {
            let h = cfg.hidden_size;
            let n_shared = cfg.n_shared_experts.unwrap_or(0);
            let shared_inter = n_shared * cfg.moe_intermediate_size;
            m.shared_gate = WeightStorage::F32(Arc::from(vec![0.0_f32; h * shared_inter]));
            m.shared_up = WeightStorage::F32(Arc::from(vec![0.0_f32; h * shared_inter]));
            m.shared_down = WeightStorage::F32(Arc::from(vec![0.0_f32; shared_inter * h]));
        } else {
            panic!("expected MoE FFN");
        }
        let m_base = DeepSeek2Model { config: cfg.clone(), weights: base };
        let m_zero = DeepSeek2Model { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, 0).unwrap().realize_f32();
        let b = m_zero.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-8,
            "shared expert path must contribute, max_diff = {max_diff}");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}
