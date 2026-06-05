//! Recurrent Gemma decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. Recurrent Gemma alternates between
//! attention layers (Gemma-shaped GQA with halved-feature partial
//! rotary) and **recurrent** layers built around a Real-Gated
//! Linear Recurrent Unit (RG-LRU). Block types are configured by
//! a `block_types: Vec<TemporalBlockType>` cycle, e.g.
//! `[R, R, A]` repeats: two recurrent then one attention.
//!
//! # Recurrent block (RecurrentBlock)
//!
//!   ```text
//!   y = act(linear_y(x))                          ; gating branch
//!   x_branch = linear_x(x)                        ; recurrence input
//!   x_branch = causal_conv1d(x_branch, w=4)       ; depthwise, kernel 4
//!   x_branch = rg_lru(x_branch)                   ; recurrent unit
//!   out      = linear_out(x_branch * y)
//!   ```
//!
//! ## RG-LRU
//!
//! Per layer, a block-structured recurrence with `n_heads` blocks
//! each of `block_width = lru_width / n_heads` features:
//!
//!   ```text
//!   i_gate    = sigmoid(per_head_W_i @ x + b_i)       ; input gate
//!   r_gate    = sigmoid(per_head_W_r @ x + b_r)       ; recurrent gate
//!   log_decay = -8 * r_gate * softplus(recurrent_param)
//!   decay     = exp(log_decay)
//!   a_square  = exp(2 * log_decay)
//!   gated_x   = x * i_gate
//!   mult      = reset + (1 - reset) * sqrt(1 - a_square)
//!   x_in      = gated_x * mult
//!   state[t]  = decay[t] * (1 - reset[t]) * state[t-1] + x_in[t]
//!   ```
//!
//! At `pos == 0` reset is `1` so `state[0] = x_in[0]`; thereafter
//! reset is `0`. v1 only supports prefill from zero state — no
//! cross-call state resumption — so the model only sees the
//! "first chunk" reset path.
//!
//! # Attention block
//!
//! Standard Gemma GQA. Partial rotary with `partial_rotary_factor
//! == 0.5` is hard-coded by the eager reference (and asserted by
//! this port): only the first half of each head's features are
//! rotated. The `attention_window_size` is read but applied as a
//! sliding-window mask (a v1 simplification — eager uses local
//! causal masking).
//!
//! # MLP
//!
//! `down(act(gate(x)) * up(x))`, with intermediate size
//! `intermediate_size / 2` per the eager reference (a Gemma-RG
//! quirk — the config's `intermediate_size` is the *fused* width,
//! halved for SwiGLU's two-branch path).
//!
//! # Other carries
//!
//! Offset RmsNorm `(gain + 1)` (Gemma family convention).
//! Tied lm_head to `token_embedding`. Soft-cap on final logits via
//! `logits_soft_cap`.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! no cross-call state, F32. The recurrent scan is unrolled at
//! graph-build time (same shape as the RWKV-5 port — long
//! prompts produce large but well-formed graphs).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

pub use crate::lazy_gemma::GemmaActivation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBlockType {
    Attention,
    Recurrent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentGemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// LRU width; defaults to `hidden_size` if `None`.
    pub lru_width: Option<usize>,
    pub attention_window_size: usize,
    pub conv1d_width: usize,
    pub logits_soft_cap: f64,
    pub hidden_activation: GemmaActivation,
    /// Must equal 0.5 to match the eager reference (asserted).
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub block_types: Vec<TemporalBlockType>,
    pub attention_bias: bool,
    pub max_seq_len: usize,
}

impl RecurrentGemmaConfig {
    pub fn lru_width_or_default(&self) -> usize {
        self.lru_width.unwrap_or(self.hidden_size)
    }
    pub fn block_width(&self) -> usize {
        self.lru_width_or_default() / self.num_attention_heads
    }
    pub fn mlp_intermediate(&self) -> usize {
        self.intermediate_size / 2
    }
    pub fn block_type(&self, layer_idx: usize) -> TemporalBlockType {
        self.block_types[layer_idx % self.block_types.len()]
    }
}

#[derive(Debug, Clone)]
pub struct RgluWeights {
    /// `[lru_width]` — softplus'd per-feature parameter for the decay.
    pub recurrent_param: Arc<[f32]>,
    /// `[n_heads, block_width, block_width]` — per-head input gate matrix.
    pub input_gate_weight: Arc<[f32]>,
    /// `[n_heads, block_width]` — per-head input gate bias.
    pub input_gate_bias: Arc<[f32]>,
    pub recurrent_gate_weight: Arc<[f32]>,
    pub recurrent_gate_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct RecurrentBlockWeights {
    pub linear_y_w: WeightStorage, // hidden → lru_width
    pub linear_y_b: Arc<[f32]>,
    pub linear_x_w: WeightStorage, // hidden → lru_width
    pub linear_x_b: Arc<[f32]>,
    pub linear_out_w: WeightStorage, // lru_width → hidden
    pub linear_out_b: Arc<[f32]>,
    /// `[lru_width, 1, conv1d_width]` depthwise kernel.
    pub conv1d_w: Arc<[f32]>,
    /// `[lru_width]`.
    pub conv1d_b: Arc<[f32]>,
    pub rg_lru: RgluWeights,
}

#[derive(Debug, Clone)]
pub struct AttentionBlockWeights {
    pub q_w: WeightStorage,
    pub q_b: Option<Arc<[f32]>>,
    pub k_w: WeightStorage,
    pub k_b: Option<Arc<[f32]>>,
    pub v_w: WeightStorage,
    pub v_b: Option<Arc<[f32]>>,
    pub o_w: WeightStorage,
    pub o_b: Arc<[f32]>, // o_proj always has bias in recurrent_gemma
}

#[derive(Debug, Clone)]
pub enum TemporalBlockWeights {
    Attention(AttentionBlockWeights),
    Recurrent(RecurrentBlockWeights),
}

#[derive(Debug, Clone)]
pub struct RecurrentGemmaLayerWeights {
    pub temporal_pre_norm_gain: Arc<[f32]>,
    pub channel_pre_norm_gain: Arc<[f32]>,
    pub temporal: TemporalBlockWeights,
    pub mlp_gate_w: WeightStorage,
    pub mlp_gate_b: Arc<[f32]>,
    pub mlp_up_w: WeightStorage,
    pub mlp_up_b: Arc<[f32]>,
    pub mlp_down_w: WeightStorage,
    pub mlp_down_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct RecurrentGemmaWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<RecurrentGemmaLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    // lm_head is tied to token_embedding.
}

#[derive(Debug, Clone)]
pub struct RecurrentGemmaModel {
    pub config: RecurrentGemmaConfig,
    pub weights: RecurrentGemmaWeights,
}

impl RecurrentGemmaModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let h_norm = self.run_backbone(tokens, start_pos)?;
        let lm_head = WeightStorage::F32(weights.token_embedding.clone());
        let logits = lm_head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size);
        let sc = cfg.logits_soft_cap;
        if sc > 0.0 {
            Ok(logits.mul_scalar(1.0 / sc).tanh().mul_scalar(sc))
        } else {
            Ok(logits)
        }
    }

    /// Run the decoder forward up to the final offset RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// RecurrentGemma-specific: per-layer Attention vs. Recurrent
    /// (LRU) temporal block selection from `block_types`.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "RecurrentGemmaModel: tokens must be non-empty");
        assert!(
            (cfg.partial_rotary_factor - 0.5).abs() < 1e-9,
            "RecurrentGemmaConfig: partial_rotary_factor must be exactly 0.5 (got {})",
            cfg.partial_rotary_factor,
        );
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads, 0,
            "num_attention_heads must be a multiple of num_key_value_heads",
        );
        let lru_width = cfg.lru_width_or_default();
        assert_eq!(
            lru_width % cfg.num_attention_heads, 0,
            "lru_width ({lru_width}) must be a multiple of num_attention_heads ({})",
            cfg.num_attention_heads,
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

        let rope_dim = cfg.head_dim / 2;
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, rope_dim,
        );

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, layer_idx, &rope_cos, &rope_sin)?;
        }
        apply_offset_rms_norm(
            &h, &weights.final_norm_gain, cfg.hidden_size, cfg.rms_norm_eps,
        )
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &RecurrentGemmaLayerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        // Temporal sublayer: pre_norm → temporal_block → residual add.
        let residual = x.clone();
        let x_norm = apply_offset_rms_norm(
            x, &layer.temporal_pre_norm_gain, h, cfg.rms_norm_eps,
        )?;
        let temporal_out = match (&layer.temporal, cfg.block_type(layer_idx)) {
            (TemporalBlockWeights::Attention(a), TemporalBlockType::Attention) => {
                self.apply_attention(&x_norm, a, rope_cos, rope_sin)?
            }
            (TemporalBlockWeights::Recurrent(r), TemporalBlockType::Recurrent) => {
                self.apply_recurrent(&x_norm, r)?
            }
            _ => panic!(
                "Layer {layer_idx}: weight kind does not match block_types[{layer_idx} % {}]",
                cfg.block_types.len(),
            ),
        };
        let h1 = residual.add(&temporal_out)?;

        // Channel sublayer: pre_norm → MLP → residual add.
        let residual2 = h1.clone();
        let h1_norm = apply_offset_rms_norm(
            &h1, &layer.channel_pre_norm_gain, h, cfg.rms_norm_eps,
        )?;
        let mlp_out = self.apply_mlp(&h1_norm, layer)?;
        residual2.add(&mlp_out)
    }

    fn apply_attention(
        &self,
        x: &LazyTensor,
        a: &AttentionBlockWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let rope_dim = cfg.head_dim / 2;
        let window = cfg.attention_window_size;

        let q = opt_bias(
            a.q_w.apply_linear(x, cfg.hidden_size, q_dim),
            a.q_b.as_ref(), q_dim,
        )?;
        let k = opt_bias(
            a.k_w.apply_linear(x, cfg.hidden_size, kv_dim),
            a.k_b.as_ref(), kv_dim,
        )?;
        let v = opt_bias(
            a.v_w.apply_linear(x, cfg.hidden_size, kv_dim),
            a.v_b.as_ref(), kv_dim,
        )?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Partial rotary on first head_dim/2 features.
        let q_r = q.rope_partial(rope_cos, rope_sin, rope_dim)?;
        let k_r = k.rope_partial(rope_cos, rope_sin, rope_dim)?;

        // GQA expand.
        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        // Sliding-window causal mask.
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || (window > 0 && j + window <= i) {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = x.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]));
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = a.o_w.apply_linear(&merged, q_dim, cfg.hidden_size);
        add_bias(attn_out, &a.o_b, cfg.hidden_size)
    }

    fn apply_recurrent(
        &self,
        x: &LazyTensor,
        r: &RecurrentBlockWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let lru_width = cfg.lru_width_or_default();
        let n_heads = cfg.num_attention_heads;
        let block_width = cfg.block_width();
        let kernel = cfg.conv1d_width;

        // Gating branch.
        let y = add_bias(
            r.linear_y_w.apply_linear(x, h, lru_width),
            &r.linear_y_b, lru_width,
        )?;
        let y_act = match cfg.hidden_activation {
            GemmaActivation::Gelu => y.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => y.gelu(),
        };

        // Recurrence input.
        let x_branch = add_bias(
            r.linear_x_w.apply_linear(x, h, lru_width),
            &r.linear_x_b, lru_width,
        )?;

        // Causal conv1d: (batch, seq, lru_width) → (batch, lru_width, seq),
        // pad left with (kernel - 1) zeros, run causal_conv1d, transpose back.
        let x_b_t = x_branch.permute([0, 2, 1_usize])?; // (b, lru, seq)
        let pad_zeros = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * lru_width * (kernel - 1)]),
            Shape::from_dims(&[batch, lru_width, kernel - 1]),
        );
        let x_b_padded = pad_zeros.concat(&x_b_t, 2_usize)?;
        let conv_w = x.const_f32_like(
            Arc::clone(&r.conv1d_w),
            Shape::from_dims(&[lru_width, 1, kernel]),
        );
        let conv_b = x.const_f32_like(
            Arc::clone(&r.conv1d_b),
            Shape::from_dims(&[lru_width]),
        );
        let x_conv = x_b_padded.causal_conv1d(&conv_w, &conv_b, false); // (b, lru, seq)
        let x_back = x_conv.permute([0, 2, 1_usize])?; // (b, seq, lru_width)

        // RG-LRU.
        let x_lru = self.apply_rg_lru(&x_back, &r.rg_lru, batch, seq, n_heads, block_width)?;

        // Gate × output.
        let gated = x_lru.mul(&y_act)?;
        let out_proj = r.linear_out_w.apply_linear(&gated, lru_width, h);
        add_bias(out_proj, &r.linear_out_b, h)
    }

    fn apply_rg_lru(
        &self,
        x: &LazyTensor,
        rg: &RgluWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        block_width: usize,
    ) -> Result<LazyTensor> {
        let lru_width = n_heads * block_width;

        // Reshape x to (b, seq, n_heads, block_width).
        let xh = x.reshape(Shape::from_dims(&[batch, seq, n_heads, block_width]))?;

        // Per-head gate projection: for each head h, compute
        //   gate[..., h, :] = sigmoid(W[h] @ x[..., h, :] + b[h])
        // The per-head W has shape (block_width, block_width). We do
        // batched matmul: reshape x to (..., n_heads, 1, block_width) and
        // W to (1, 1, n_heads, block_width, block_width), then matmul.
        let project = |w: &Arc<[f32]>, b: &Arc<[f32]>| -> Result<LazyTensor> {
            let w_t = x.const_f32_like(
                Arc::clone(w),
                Shape::from_dims(&[1, 1, n_heads, block_width, block_width]),
            );
            let w_bc = w_t.broadcast_to(Shape::from_dims(&[
                batch, seq, n_heads, block_width, block_width,
            ]))?;
            let x_row = xh.reshape(Shape::from_dims(&[batch, seq, n_heads, 1, block_width]))?;
            let res = x_row.matmul(&w_bc)?; // (b, seq, n_heads, 1, block_width)
            let res = res.reshape(Shape::from_dims(&[batch, seq, n_heads, block_width]))?;
            // Add per-head bias (n_heads, block_width).
            let b_t = x.const_f32_like(
                Arc::clone(b),
                Shape::from_dims(&[1, 1, n_heads, block_width]),
            );
            let b_bc = b_t.broadcast_to(Shape::from_dims(&[
                batch, seq, n_heads, block_width,
            ]))?;
            res.add(&b_bc)
        };
        let input_gate = project(&rg.input_gate_weight, &rg.input_gate_bias)?.sigmoid();
        let recurrent_gate = project(&rg.recurrent_gate_weight, &rg.recurrent_gate_bias)?
            .sigmoid();

        // Flatten back to (b, seq, lru_width).
        let input_gate = input_gate.reshape(Shape::from_dims(&[batch, seq, lru_width]))?;
        let recurrent_gate = recurrent_gate
            .reshape(Shape::from_dims(&[batch, seq, lru_width]))?;

        // log_decay = -8 * recurrent_gate * softplus(recurrent_param)
        // softplus(y) = log(exp(y) + 1)
        let rp = x.const_f32_like(
            Arc::clone(&rg.recurrent_param),
            Shape::from_dims(&[lru_width]),
        );
        let softplus_rp = rp.exp().add_scalar(1.0).log();
        // broadcast (lru_width) → (1, 1, lru_width)
        let softplus_rp_bc = softplus_rp
            .reshape(Shape::from_dims(&[1, 1, lru_width]))?
            .broadcast_to(Shape::from_dims(&[batch, seq, lru_width]))?;
        let log_decay = recurrent_gate.mul_scalar(-8.0).mul(&softplus_rp_bc)?;
        let decay = log_decay.exp();
        let a_square = log_decay.mul_scalar(2.0).exp();

        // gated_x = x * input_gate
        let gated_x = x.mul(&input_gate)?;
        // mult = reset + (1 - reset) * sqrt(1 - a_square)
        //   at t=0, reset=1 ⇒ mult=1
        //   at t>0, reset=0 ⇒ mult = sqrt(1 - a_square)
        // Build reset mask shape (1, seq, 1): [1.0, 0.0, 0.0, ...].
        let mut reset_data = vec![0.0_f32; seq];
        reset_data[0] = 1.0;
        let reset = x.const_f32_like(
            Arc::from(reset_data),
            Shape::from_dims(&[1, seq, 1]),
        );
        let one_minus_reset = reset.mul_scalar(-1.0).add_scalar(1.0); // 1 - reset
        let one_minus_reset_bc = one_minus_reset
            .broadcast_to(Shape::from_dims(&[batch, seq, lru_width]))?;
        // 1 - a_square (clamp away from negatives via straight subtraction; in
        // valid range a_square ∈ (0, 1] so 1 - a_square ≥ 0).
        let one_minus_a_square = a_square.mul_scalar(-1.0).add_scalar(1.0);
        let sqrt_term = one_minus_a_square.sqrt();
        let reset_bc = reset.broadcast_to(Shape::from_dims(&[batch, seq, lru_width]))?;
        let mult = reset_bc.add(&one_minus_reset_bc.mul(&sqrt_term)?)?;
        let normalized_x = gated_x.mul(&mult)?;

        // Effective decay = decay * (1 - reset).
        let decay_eff = decay.mul(&one_minus_reset_bc)?;

        // Sequential recurrence:
        //   state[t] = decay_eff[t] * state[t-1] + normalized_x[t]
        // Stack states into (b, seq, lru_width). State starts at zeros.
        let mut state: Option<LazyTensor> = None;
        let mut out_steps: Vec<LazyTensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let x_t = normalized_x.slice(1_usize, t, 1)?; // (b, 1, lru_width)
            let d_t = decay_eff.slice(1_usize, t, 1)?;     // (b, 1, lru_width)
            let new_state = match state {
                None => x_t,
                Some(s) => d_t.mul(&s)?.add(&x_t)?,
            };
            state = Some(new_state.clone());
            out_steps.push(new_state);
        }
        // Concat along seq axis.
        let mut all: Option<LazyTensor> = None;
        for step in out_steps.into_iter() {
            all = Some(match all {
                None => step,
                Some(s) => s.concat(&step, 1_usize)?,
            });
        }
        Ok(all.expect("at least one step"))
    }

    fn apply_mlp(
        &self,
        x: &LazyTensor,
        layer: &RecurrentGemmaLayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.mlp_intermediate();
        let gate = add_bias(
            layer.mlp_gate_w.apply_linear(x, h, inter),
            &layer.mlp_gate_b, inter,
        )?;
        let up = add_bias(
            layer.mlp_up_w.apply_linear(x, h, inter),
            &layer.mlp_up_b, inter,
        )?;
        let activated = match cfg.hidden_activation {
            GemmaActivation::Gelu => gate.gelu_erf(),
            GemmaActivation::GeluPytorchTanh => gate.gelu(),
        };
        let inner = activated.mul(&up)?;
        let down = layer.mlp_down_w.apply_linear(&inner, inter, h);
        add_bias(down, &layer.mlp_down_b, h)
    }
}

fn apply_offset_rms_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    dim: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let _ = dim;
    x.rms_norm_affine_with_offset(gain, 1.0, eps)
}

fn opt_bias(x: LazyTensor, b: Option<&Arc<[f32]>>, n: usize) -> Result<LazyTensor> {
    let _ = n;
    x.add_optional_trailing_bias(b)
}

fn add_bias(x: LazyTensor, bias: &Arc<[f32]>, n: usize) -> Result<LazyTensor> {
    let _ = n;
    x.add_trailing_bias(Arc::clone(bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &RecurrentGemmaConfig) -> RecurrentGemmaWeights {
        let mut s: u32 = 24681;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let lru = cfg.lru_width_or_default();
        let n_heads = cfg.num_attention_heads;
        let block_w = cfg.block_width();
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.mlp_intermediate();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<RecurrentGemmaLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|li| {
                let temporal = match cfg.block_type(li) {
                    TemporalBlockType::Attention => {
                        TemporalBlockWeights::Attention(AttentionBlockWeights {
                            q_w: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                            q_b: if cfg.attention_bias { Some(vec_of(q_dim, &mut *nb)) } else { None },
                            k_w: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                            k_b: if cfg.attention_bias { Some(vec_of(kv_dim, &mut *nb)) } else { None },
                            v_w: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                            v_b: if cfg.attention_bias { Some(vec_of(kv_dim, &mut *nb)) } else { None },
                            o_w: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                            o_b: vec_of(h, &mut *nb),
                        })
                    }
                    TemporalBlockType::Recurrent => {
                        TemporalBlockWeights::Recurrent(RecurrentBlockWeights {
                            linear_y_w: WeightStorage::F32(vec_of(h * lru, &mut *nb)),
                            linear_y_b: vec_of(lru, &mut *nb),
                            linear_x_w: WeightStorage::F32(vec_of(h * lru, &mut *nb)),
                            linear_x_b: vec_of(lru, &mut *nb),
                            linear_out_w: WeightStorage::F32(vec_of(lru * h, &mut *nb)),
                            linear_out_b: vec_of(h, &mut *nb),
                            conv1d_w: vec_of(lru * cfg.conv1d_width, &mut *nb),
                            conv1d_b: vec_of(lru, &mut *nb),
                            rg_lru: RgluWeights {
                                recurrent_param: vec_of(lru, &mut *nb),
                                input_gate_weight: vec_of(n_heads * block_w * block_w, &mut *nb),
                                input_gate_bias: vec_of(n_heads * block_w, &mut *nb),
                                recurrent_gate_weight: vec_of(n_heads * block_w * block_w, &mut *nb),
                                recurrent_gate_bias: vec_of(n_heads * block_w, &mut *nb),
                            },
                        })
                    }
                };
                RecurrentGemmaLayerWeights {
                    temporal_pre_norm_gain: Arc::from(vec![0.05_f32; h]),
                    channel_pre_norm_gain: Arc::from(vec![0.05_f32; h]),
                    temporal,
                    mlp_gate_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    mlp_gate_b: vec_of(inter, &mut *nb),
                    mlp_up_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    mlp_up_b: vec_of(inter, &mut *nb),
                    mlp_down_w: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                    mlp_down_b: vec_of(h, &mut *nb),
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![0.05_f32; h]);
        RecurrentGemmaWeights { token_embedding, layers, final_norm_gain }
    }

    fn tiny_config() -> RecurrentGemmaConfig {
        RecurrentGemmaConfig {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 3, num_attention_heads: 2, num_key_value_heads: 2,
            head_dim: 4, lru_width: Some(8),
            attention_window_size: 8, conv1d_width: 4,
            logits_soft_cap: 30.0,
            hidden_activation: GemmaActivation::GeluPytorchTanh,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-6, rope_theta: 10_000.0,
            block_types: vec![
                TemporalBlockType::Recurrent,
                TemporalBlockType::Recurrent,
                TemporalBlockType::Attention,
            ],
            attention_bias: false,
            max_seq_len: 32,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn single_token() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[3], 0).unwrap().realize_f32();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    /// Recurrent state propagates: first-token swap changes
    /// last-token output via the LRU recurrence.
    #[test]
    fn recurrent_state_propagates() {
        let cfg = RecurrentGemmaConfig {
            // Force layer 0 to be Recurrent so state actually flows.
            block_types: vec![TemporalBlockType::Recurrent],
            num_hidden_layers: 1,
            ..tiny_config()
        };
        let weights = tiny_weights(&cfg);
        let model = RecurrentGemmaModel { config: cfg.clone(), weights };
        let a = model.forward(&[0, 5, 5, 5], 0).unwrap().realize_f32();
        let b = model.forward(&[7, 5, 5, 5], 0).unwrap().realize_f32();
        let last_a = &a[a.len() - cfg.vocab_size..];
        let last_b = &b[b.len() - cfg.vocab_size..];
        let mut max_diff = 0.0_f32;
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny-weight test (weights ∈ [-0.025, 0.025]) — the
        // recurrent contribution is real but small; we just
        // require it to be measurably non-zero.
        assert!(max_diff > 1e-8,
            "recurrent state must propagate first→last, max_diff = {max_diff}");
    }

    /// Soft-cap on logits is wired: removing it changes output.
    #[test]
    fn logits_soft_cap_changes_output() {
        let cfg_a = RecurrentGemmaConfig { logits_soft_cap: 0.0, ..tiny_config() };
        let cfg_b = RecurrentGemmaConfig { logits_soft_cap: 5.0, ..tiny_config() };
        let weights = tiny_weights(&cfg_a);
        let m_a = RecurrentGemmaModel { config: cfg_a, weights: weights.clone() };
        let m_b = RecurrentGemmaModel { config: cfg_b, weights };
        let a = m_a.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let b = m_b.forward(&[1, 2, 3], 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "logits soft-cap must alter output, max_diff = {max_diff}");
    }

    #[test]
    fn block_type_alternation() {
        let cfg = tiny_config();
        // block_types = [R, R, A], num_hidden_layers = 3
        assert_eq!(cfg.block_type(0), TemporalBlockType::Recurrent);
        assert_eq!(cfg.block_type(1), TemporalBlockType::Recurrent);
        assert_eq!(cfg.block_type(2), TemporalBlockType::Attention);
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config();
        let model = RecurrentGemmaModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }
}
