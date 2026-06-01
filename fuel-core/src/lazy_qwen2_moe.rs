//! Qwen2-MoE decoder ported to the lazy-graph API.
//!
//! Phase 6a anchor #7 — the seventh and final anchor for the Phase 6a
//! exit criterion. Qwen2-MoE is a Mixture-of-Experts extension of the
//! Qwen2 architecture: each FFN layer is replaced by a router that
//! softmax-selects `num_experts_per_tok` experts (4 of 60 for
//! Qwen1.5-MoE-A2.7B) plus a shared expert that's always active. The
//! attention block is unchanged from Qwen2 — causal multi-head with
//! Q/K/V biases, RMSNorm, RoPE.
//!
//! # Architectural first (vs prior anchors)
//!
//! **`moe_block`** — dense-routing MoE FFN. For Phase 6a's proof-of-
//! architecture scope we compute the full softmax over all experts and
//! run every expert on every token, weighting each expert's output by
//! its per-token routing probability. This is mathematically different
//! from the trained top-k routing (which keeps only the top-k experts'
//! mass and optionally renormalizes), but exercises every piece of the
//! MoE plumbing — gate network, per-expert SwiGLU MLPs, weighted
//! combination, shared expert with a sigmoid-gated residual. A true
//! top-k implementation is a future addition once the lazy graph
//! grows a `top_k` primitive; until then, dense routing trades
//! 15× compute for every-path-exercised correctness.
//!
//! # Scope
//!
//! - Forward-only, single sequence, no KV cache — same scoping as the
//!   initial BERT / Whisper ports. A real decode loop layers on top.
//! - Dense routing (full softmax, all experts always evaluated) rather
//!   than the trained top-k sparse routing. Output shapes + norms are
//!   plausible; numerical equivalence to HF's sparse path is not
//!   expected.
//! - Model loading lives in this module but relies on the caller to
//!   point at an MoE checkpoint. The smallest public Qwen-MoE is
//!   `Qwen/Qwen1.5-MoE-A2.7B-Chat` at ~14 GB — real-weights validation
//!   needs that download.

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use serde::Deserialize;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
}

fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_norm_topk_prob() -> bool { false }

impl Qwen2MoeConfig {
    pub fn from_hf_json_str(s: &str) -> crate::Result<Self> {
        serde_json::from_str::<Self>(s)
            .map_err(|e| crate::Error::Msg(format!("parsing qwen2_moe config: {e}")).bt())
    }

    pub fn head_dim(&self) -> usize {
        assert_eq!(self.hidden_size % self.num_attention_heads, 0);
        self.hidden_size / self.num_attention_heads
    }
}

// ---- Weight storage --------------------------------------------------------

/// One SwiGLU expert: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub gate_w: Arc<[f32]>,
    pub up_w:   Arc<[f32]>,
    pub down_w: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen2MoeLayerWeights {
    // attention (Qwen2 shape: Q/K/V have biases, O has none)
    pub input_ln:     Arc<[f32]>,
    pub q_w: Arc<[f32]>, pub q_b: Arc<[f32]>,
    pub k_w: Arc<[f32]>, pub k_b: Arc<[f32]>,
    pub v_w: Arc<[f32]>, pub v_b: Arc<[f32]>,
    pub o_w: Arc<[f32]>,
    // MoE FFN
    pub post_attn_ln: Arc<[f32]>,
    /// Gate network: `[num_experts, hidden]` (stored pre-transpose).
    pub gate_w: Arc<[f32]>,
    pub experts: Vec<ExpertWeights>,
    /// Shared expert (always active, wider than routed experts).
    pub shared_gate_w: Arc<[f32]>,
    pub shared_up_w:   Arc<[f32]>,
    pub shared_down_w: Arc<[f32]>,
    /// Shared expert gating: `[1, hidden]` scalar gate.
    pub shared_expert_gate_w: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen2MoeWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers:   Vec<Qwen2MoeLayerWeights>,
    pub final_ln: Arc<[f32]>,
    pub lm_head:  Arc<[f32]>,
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Qwen2MoeModel {
    pub config:  Qwen2MoeConfig,
    pub weights: Qwen2MoeWeights,
}

impl Qwen2MoeModel {
    /// Run a forward pass on a single sequence of token IDs. Returns
    /// `[1, seq, vocab_size]` logits. No KV cache — this is the
    /// "prefill from scratch" path.
    pub fn forward(&self, tokens: &[u32]) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let seq = tokens.len();
        assert!(seq > 0);

        // Anchor the graph on the token embedding.
        let embed = LazyTensor::from_f32(
            self.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, h]),
            &crate::Device::cpu(),
        );
        let input_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut x = embed
            .index_select(0, &input_ids).unwrap()
            .reshape(Shape::from_dims(&[1, seq, h])).unwrap();

        for lw in &self.weights.layers {
            x = decoder_layer(&x, lw, cfg, seq);
        }

        let x = rms_norm_affine(&x, &self.weights.final_ln, cfg.rms_norm_eps, h, seq);
        let lm = x.const_f32_like(self.weights.lm_head.clone(), Shape::from_dims(&[h, cfg.vocab_size]));
        Ok(x.matmul(&lm).unwrap())
    }
}

fn decoder_layer(x: &LazyTensor, lw: &Qwen2MoeLayerWeights, cfg: &Qwen2MoeConfig, seq: usize) -> LazyTensor {
    let h = cfg.hidden_size;
    // Attention sublayer
    let x_ln = rms_norm_affine(x, &lw.input_ln, cfg.rms_norm_eps, h, seq);
    let attn = qwen2_attn(&x_ln, lw, cfg, seq);
    let x = x.add(&attn).unwrap();

    // MoE sublayer
    let x_ln = rms_norm_affine(&x, &lw.post_attn_ln, cfg.rms_norm_eps, h, seq);
    let moe = moe_block(&x_ln, lw, cfg, seq);
    x.add(&moe).unwrap()
}

fn qwen2_attn(x: &LazyTensor, lw: &Qwen2MoeLayerWeights, cfg: &Qwen2MoeConfig, seq: usize) -> LazyTensor {
    let h = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let d_head = cfg.head_dim();
    let n_kv = cfg.num_key_value_heads;

    let q = linear(x, &lw.q_w, Some(&lw.q_b), h, h, seq);
    let k = linear(x, &lw.k_w, Some(&lw.k_b), h, n_kv * d_head, seq);
    let v = linear(x, &lw.v_w, Some(&lw.v_b), h, n_kv * d_head, seq);

    let q = q
        .reshape(Shape::from_dims(&[1, seq, n_heads, d_head])).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap();
    // For Qwen1.5-MoE num_kv_heads == num_attention_heads, so no GQA
    // replication needed; but keep the rehape path general.
    let k = k
        .reshape(Shape::from_dims(&[1, seq, n_kv, d_head])).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap();
    let v = v
        .reshape(Shape::from_dims(&[1, seq, n_kv, d_head])).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap();

    // RoPE.
    let (cos, sin) = rope_tables(cfg.rope_theta, seq, d_head);
    let q = apply_rope(&q, &cos, &sin, seq, d_head);
    let k = apply_rope(&k, &cos, &sin, seq, d_head);

    let k_t = k.permute([0, 1, 3, 2_usize]).unwrap();
    let scale = 1.0_f64 / (d_head as f64).sqrt();
    let mut scores = q.matmul(&k_t).unwrap().mul_scalar(scale);
    // causal mask
    let mut mask = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i { mask[i * seq + j] = f32::NEG_INFINITY; }
        }
    }
    let mask_t = scores
        .const_f32_like(mask, Shape::from_dims(&[seq, seq]))
        .reshape(Shape::from_dims(&[1, 1, seq, seq])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, n_heads, seq, seq])).unwrap();
    scores = scores.add(&mask_t).unwrap();
    let probs = scores.softmax_last_dim().unwrap();
    let ctx = probs
        .matmul(&v).unwrap()
        .permute([0, 2, 1, 3_usize]).unwrap()
        .reshape(Shape::from_dims(&[1, seq, h])).unwrap();
    linear(&ctx, &lw.o_w, None, h, h, seq)
}

/// MoE FFN block. Dense routing: every expert runs on every token;
/// outputs are weighted by the full gate softmax. See module docs for
/// the scoping rationale vs the trained top-k path.
fn moe_block(
    x: &LazyTensor,
    lw: &Qwen2MoeLayerWeights,
    cfg: &Qwen2MoeConfig,
    seq: usize,
) -> LazyTensor {
    let h = cfg.hidden_size;
    let e = cfg.num_experts;

    // Router: [1, seq, h] → gate.matmul → [1, seq, E].
    let gate = x.const_f32_like(lw.gate_w.clone(), Shape::from_dims(&[h, e]));
    let router_logits = x.matmul(&gate).unwrap();
    let router_weights = router_logits.softmax_last_dim().unwrap();  // [1, seq, E]

    // Each expert's SwiGLU output, weighted by its per-token gate weight.
    // Accumulate into `routed_sum` : [1, seq, h].
    let moe_int = cfg.moe_intermediate_size;
    let mut routed_sum: Option<LazyTensor> = None;
    for (ei, ew) in lw.experts.iter().enumerate() {
        let expert_out = swiglu_mlp(x, &ew.gate_w, &ew.up_w, &ew.down_w, h, moe_int, seq);
        // Slice router_weights to get the column for this expert: [1, seq, 1].
        let w_col = router_weights
            .slice(2, ei, 1).unwrap();  // [1, seq, 1]
        let w_bc = w_col.broadcast_to(Shape::from_dims(&[1, seq, h])).unwrap();
        let gated = expert_out.mul(&w_bc).unwrap();
        routed_sum = Some(match routed_sum {
            Some(s) => s.add(&gated).unwrap(),
            None => gated,
        });
    }
    let routed = routed_sum.expect("moe: must have at least one expert");

    // Shared expert (always active, sigmoid-gated by a scalar per token).
    let shared_int = cfg.shared_expert_intermediate_size;
    let shared_out = swiglu_mlp(x, &lw.shared_gate_w, &lw.shared_up_w, &lw.shared_down_w, h, shared_int, seq);
    // Shared expert gate: Linear(h → 1), then sigmoid.
    let sg_w = x.const_f32_like(lw.shared_expert_gate_w.clone(), Shape::from_dims(&[h, 1]));
    let sg = x.matmul(&sg_w).unwrap().sigmoid();  // [1, seq, 1]
    let sg_bc = sg.broadcast_to(Shape::from_dims(&[1, seq, h])).unwrap();
    let shared_gated = shared_out.mul(&sg_bc).unwrap();

    routed.add(&shared_gated).unwrap()
}

fn swiglu_mlp(
    x: &LazyTensor,
    gate_w: &Arc<[f32]>,
    up_w: &Arc<[f32]>,
    down_w: &Arc<[f32]>,
    h: usize,
    h_ff: usize,
    seq: usize,
) -> LazyTensor {
    let g = linear(x, gate_w, None, h, h_ff, seq).silu();
    let u = linear(x, up_w, None, h, h_ff, seq);
    let gated = g.mul(&u).unwrap();
    linear(&gated, down_w, None, h_ff, h, seq)
}

fn rms_norm_affine(x: &LazyTensor, gamma: &Arc<[f32]>, eps: f64, hidden: usize, seq: usize) -> LazyTensor {
    // RMS norm: x * rsqrt(mean(x^2) + eps) * gamma
    let sq = x.mul(x).unwrap();
    let ms = sq.mean_dim(2).unwrap();  // [1, seq]
    let rstd = ms.add_scalar(eps).sqrt();
    let rstd_bc = rstd
        .reshape(Shape::from_dims(&[1, seq, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, hidden])).unwrap();
    let normed = x.div(&rstd_bc).unwrap();
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, hidden])).unwrap();
    normed.mul(&g).unwrap()
}

fn rope_tables(theta: f64, seq: usize, d_head: usize) -> (Vec<f32>, Vec<f32>) {
    let half = d_head / 2;
    let mut cos = vec![0.0_f32; seq * d_head];
    let mut sin = vec![0.0_f32; seq * d_head];
    for pos in 0..seq {
        for i in 0..half {
            let freq = (theta as f32).powf(-2.0 * (i as f32) / (d_head as f32));
            let arg = (pos as f32) * freq;
            cos[pos * d_head + i] = arg.cos();
            cos[pos * d_head + i + half] = arg.cos();
            sin[pos * d_head + i] = arg.sin();
            sin[pos * d_head + i + half] = arg.sin();
        }
    }
    (cos, sin)
}

fn apply_rope(
    x: &LazyTensor,  // [1, H, seq, d_head]
    cos: &[f32],
    sin: &[f32],
    seq: usize,
    d_head: usize,
) -> LazyTensor {
    let x_shape = x.shape();
    let x_dims = x_shape.dims();
    let n_heads = x_dims[1];
    let cos_t = x
        .const_f32_like(cos.to_vec(), Shape::from_dims(&[seq, d_head]))
        .reshape(Shape::from_dims(&[1, 1, seq, d_head])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, n_heads, seq, d_head])).unwrap();
    let sin_t = x
        .const_f32_like(sin.to_vec(), Shape::from_dims(&[seq, d_head]))
        .reshape(Shape::from_dims(&[1, 1, seq, d_head])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, n_heads, seq, d_head])).unwrap();
    let half = d_head / 2;
    let x1 = x.slice(3, 0, half).unwrap();
    let x2 = x.slice(3, half, half).unwrap();
    // rotate_half: concat(-x2, x1) along dim 3.
    let neg_x2 = x2.neg();
    let rotated = neg_x2.concat(&x1, 3).unwrap();
    x.mul(&cos_t).unwrap().add(&rotated.mul(&sin_t).unwrap()).unwrap()
}

fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t).unwrap();
    match b {
        Some(b) => {
            let bias = x
                .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f])).unwrap()
                .broadcast_to(Shape::from_dims(&[1, seq, out_f])).unwrap();
            proj.add(&bias).unwrap()
        }
        None => proj,
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen2MoeConfig {
        // Minimum viable MoE: 3 experts, hidden=8, 1 layer.
        Qwen2MoeConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            moe_intermediate_size: 12,
            shared_expert_intermediate_size: 16,
            num_experts: 3,
            num_experts_per_tok: 2,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            norm_topk_prob: false,
        }
    }

    #[test]
    fn moe_forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let h = cfg.hidden_size;
        let moe_int = cfg.moe_intermediate_size;
        let shared_int = cfg.shared_expert_intermediate_size;
        let z = |n| Arc::from(vec![0.0_f32; n]) as Arc<[f32]>;
        let o = |n| Arc::from(vec![1.0_f32; n]) as Arc<[f32]>;
        let experts: Vec<ExpertWeights> = (0..cfg.num_experts)
            .map(|_| ExpertWeights {
                gate_w: z(h * moe_int),
                up_w:   z(h * moe_int),
                down_w: z(moe_int * h),
            })
            .collect();
        let layer = Qwen2MoeLayerWeights {
            input_ln: o(h),
            q_w: z(h * h), q_b: z(h),
            k_w: z(h * h), k_b: z(h),
            v_w: z(h * h), v_b: z(h),
            o_w: z(h * h),
            post_attn_ln: o(h),
            gate_w: z(h * cfg.num_experts),
            experts,
            shared_gate_w: z(h * shared_int),
            shared_up_w:   z(h * shared_int),
            shared_down_w: z(shared_int * h),
            shared_expert_gate_w: z(h),
        };
        let weights = Qwen2MoeWeights {
            token_embedding: z(cfg.vocab_size * h),
            layers: vec![layer],
            final_ln: o(h),
            lm_head: z(h * cfg.vocab_size),
        };
        let model = Qwen2MoeModel { config: cfg.clone(), weights };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward(&tokens).unwrap();
        let flat = logits.realize_f32();
        assert_eq!(flat.len(), 1 * tokens.len() * cfg.vocab_size);
        assert!(flat.iter().all(|v| v.is_finite()));

        // Phase 6a oracle gate.
        let flat_ref = logits.realize_f32_reference();
        crate::test_utils::assert_allclose_f32(&flat, &flat_ref, 1e-4, 1e-3);
    }

    #[test]
    fn config_head_dim() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.head_dim(), 4);
    }
}
