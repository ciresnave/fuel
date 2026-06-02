//! RWKV v5 ("Eagle") decoder ported to the lazy-graph API.
//!
//! Phase D specialized port. RWKV v5 is fundamentally different
//! from a transformer — a *recurrent* model with linear-time
//! attention via an explicit per-head state matrix and a
//! "token-shift" channel mixer in place of the FFN. Each layer
//! carries three state buffers across time steps:
//!
//!   - `extract_key_value`: the previous token's *hidden* input
//!     (used to mix into K/V/R/Gate inputs).
//!   - `linear_attention`: a `[n_heads, head_size, head_size]`
//!     recurrent state matrix that takes the place of softmax
//!     attention.
//!   - `feed_forward`: the previous token's input to the FFN
//!     (used to mix into the FFN's K/R inputs).
//!
//! # The time-mix recurrence (per layer, per token)
//!
//!   ```text
//!   shifted     = prev_token_x        (or zeros at t=0)
//!   k_in        = mix * x + (1-mix) * shifted     ; per K/V/R/G mix
//!   K, V, R, G  = projections of the mixed inputs
//!   G           = silu(G)                          ; gate
//!
//!   per head (reshape h * s):
//!     at        = K[:, h, :, t] @ V[:, h, t, :]    ; outer product [S, S]
//!     out_t     = R[:, h, t, :] @ (faaaa * at + state[h])
//!     state[h]  = at + decay[h] * state[h]
//!     where decay[h] = exp(-exp(time_decay[h]))
//!
//!   out = GroupNorm(concat over t of out_t)        ; per-head LN
//!   y   = (out * G) projected through `output`
//!   ```
//!
//! # The channel-mix (FFN) per layer
//!
//!   ```text
//!   k_in     = mix * x + (1-mix) * prev_ffn_x
//!   r_in     = same with receptance mix
//!   k_out    = relu(K_proj(k_in)).square()
//!   v_out    = V_proj(k_out)
//!   r_out    = sigmoid(R_proj(r_in))
//!   y        = r_out * v_out                       ; no residual here; outer block adds it
//!   ```
//!
//! # Scope (v1)
//!
//! Forward-only prefill: zero initial state, batch == 1, F32.
//! The recurrent time loop is unrolled at graph-build time —
//! one chain of nodes per token. This is fine for short prompts
//! used in tests; long-prompt prefill produces a large but
//! still well-formed graph. Streaming generation (resuming
//! state across calls) is a follow-up.
//!
//! Rescale-every (which halves activations every N layers when
//! casting to half-precision) is honored as a no-op in F32, the
//! v1 dtype — `Self::layers_are_rescaled` is always false for
//! F32 weights.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Rwkv5Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub attention_hidden_size: usize,
    pub head_size: usize,
    /// Defaults to `hidden_size / head_size` if zero.
    pub num_attention_heads: usize,
    pub intermediate_size: Option<usize>,
    pub layer_norm_epsilon: f64,
    /// Activation rescale gate (no-op for F32 weights — see module docs).
    pub rescale_every: usize,
}

impl Rwkv5Config {
    pub fn n_heads(&self) -> usize {
        if self.num_attention_heads == 0 {
            self.hidden_size / self.head_size
        } else {
            // Eager treats num_attention_heads as hidden_size /
            // num_attention_heads-as-divisor (a weird convention),
            // but the model boils down to n_heads = hidden / head_size.
            self.hidden_size / self.head_size
        }
    }
    pub fn ffn_intermediate(&self) -> usize {
        self.intermediate_size
            .unwrap_or((((self.hidden_size as f64) * 3.5) as usize) / 32 * 32)
    }
}

#[derive(Debug, Clone)]
pub struct Rwkv5LayerWeights {
    pub ln1_gain: Arc<[f32]>,
    pub ln1_bias: Arc<[f32]>,
    pub ln2_gain: Arc<[f32]>,
    pub ln2_bias: Arc<[f32]>,
    /// Only present on layer 0 (mirrors HF `pre_ln`).
    pub pre_ln: Option<(Arc<[f32]>, Arc<[f32]>)>,

    // ---- Time-mix (attention) ----
    pub attn_time_mix_key: Arc<[f32]>,        // [hidden_size]
    pub attn_time_mix_value: Arc<[f32]>,      // [hidden_size]
    pub attn_time_mix_receptance: Arc<[f32]>, // [hidden_size]
    pub attn_time_mix_gate: Arc<[f32]>,       // [hidden_size]
    /// `[n_heads, head_size]`; decay = `exp(-exp(time_decay))`.
    pub attn_time_decay: Arc<[f32]>,
    /// `[n_heads, head_size]` per-head bonus factor.
    pub attn_time_faaaa: Arc<[f32]>,
    pub attn_key: WeightStorage,        // hidden_size → attention_hidden_size
    pub attn_value: WeightStorage,      // hidden_size → attention_hidden_size
    pub attn_receptance: WeightStorage, // hidden_size → attention_hidden_size
    pub attn_gate: WeightStorage,       // hidden_size → attention_hidden_size
    pub attn_output: WeightStorage,     // attention_hidden_size → hidden_size
    /// GroupNorm gain `[hidden_size]`.
    pub attn_ln_x_gain: Arc<[f32]>,
    pub attn_ln_x_bias: Arc<[f32]>,

    // ---- Channel-mix (FFN) ----
    pub ffn_time_mix_key: Arc<[f32]>,
    pub ffn_time_mix_receptance: Arc<[f32]>,
    pub ffn_key: WeightStorage,        // hidden_size → ffn_intermediate
    pub ffn_value: WeightStorage,      // ffn_intermediate → hidden_size
    pub ffn_receptance: WeightStorage, // hidden_size → hidden_size
}

#[derive(Debug, Clone)]
pub struct Rwkv5Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Rwkv5LayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub head: WeightStorage, // hidden_size → vocab_size
}

#[derive(Debug, Clone)]
pub struct Rwkv5Model {
    pub config: Rwkv5Config,
    pub weights: Rwkv5Weights,
}

impl Rwkv5Model {
    pub fn forward(&self, tokens: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0, "Rwkv5Model::forward: tokens must be non-empty");
        let n_heads = cfg.n_heads();
        let head_size = cfg.head_size;
        assert_eq!(
            n_heads * head_size, cfg.hidden_size,
            "Rwkv5Config: n_heads({n_heads}) * head_size({head_size}) must equal hidden_size",
        );
        // Embed.
        let embed = LazyTensor::from_f32(
            weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.hidden_size]),
            &Device::cpu(),
        );
        let token_ids = embed.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let mut h = embed
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.hidden_size]))?;

        for layer in &weights.layers {
            h = self.apply_block(&h, layer, batch, seq, n_heads, head_size)?;
        }

        let h_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h, &weights.final_ln_gain, &weights.final_ln_bias,
            cfg.hidden_size, cfg.layer_norm_epsilon,
        );
        Ok(weights.head.apply_linear(&h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn apply_block(
        &self,
        xs: &LazyTensor,
        layer: &Rwkv5LayerWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        head_size: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        // Optional layer-0 pre-LN.
        let xs = if let Some((g, b)) = &layer.pre_ln {
            crate::lazy::apply_affine_layer_norm_pub(xs, g, b, h, cfg.layer_norm_epsilon)
        } else {
            xs.clone()
        };

        // Attention sublayer.
        let xs_ln1 = crate::lazy::apply_affine_layer_norm_pub(
            &xs, &layer.ln1_gain, &layer.ln1_bias, h, cfg.layer_norm_epsilon,
        );
        let attn = self.time_mix(&xs_ln1, layer, batch, seq, n_heads, head_size)?;
        let xs = xs.add(&attn)?;

        // FFN sublayer.
        let xs_ln2 = crate::lazy::apply_affine_layer_norm_pub(
            &xs, &layer.ln2_gain, &layer.ln2_bias, h, cfg.layer_norm_epsilon,
        );
        let ff = self.channel_mix(&xs_ln2, layer, batch, seq)?;
        xs.add(&ff)
    }

    fn time_mix(
        &self,
        x: &LazyTensor,
        layer: &Rwkv5LayerWeights,
        batch: usize,
        seq: usize,
        n_heads: usize,
        head_size: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let ah = cfg.attention_hidden_size;
        assert_eq!(
            ah, h,
            "Rwkv5: v1 assumes attention_hidden_size == hidden_size \
             (decoupled is a follow-up)",
        );

        // Time-mix: shift x by 1 along seq; prepend a zero row.
        let shifted = self.shift_seq(x, batch, seq, h);

        let mix_key = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_key),
            Shape::from_dims(&[1, 1, h]),
        );
        let mix_val = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_value),
            Shape::from_dims(&[1, 1, h]),
        );
        let mix_rec = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_receptance),
            Shape::from_dims(&[1, 1, h]),
        );
        let mix_gate = x.const_f32_like(
            Arc::clone(&layer.attn_time_mix_gate),
            Shape::from_dims(&[1, 1, h]),
        );
        let one_minus = |m: &LazyTensor| -> Result<LazyTensor> {
            // 1.0 - m, returning a tensor shaped (1, 1, h).
            let ones: Vec<f32> = vec![1.0_f32; h];
            let ones_t = x.const_f32_like(Arc::from(ones), Shape::from_dims(&[1, 1, h]));
            ones_t.sub(m)
        };
        let mk_inv = one_minus(&mix_key)?;
        let mv_inv = one_minus(&mix_val)?;
        let mr_inv = one_minus(&mix_rec)?;
        let mg_inv = one_minus(&mix_gate)?;

        let key_in = x.broadcast_mul(&mix_key)?.add(&shifted.broadcast_mul(&mk_inv)?)?;
        let val_in = x.broadcast_mul(&mix_val)?.add(&shifted.broadcast_mul(&mv_inv)?)?;
        let rec_in = x.broadcast_mul(&mix_rec)?.add(&shifted.broadcast_mul(&mr_inv)?)?;
        let gat_in = x.broadcast_mul(&mix_gate)?.add(&shifted.broadcast_mul(&mg_inv)?)?;

        // Project to attention space.
        let k = layer.attn_key.apply_linear(&key_in, h, h);
        let v = layer.attn_value.apply_linear(&val_in, h, h);
        let r = layer.attn_receptance.apply_linear(&rec_in, h, h);
        let g = layer.attn_gate.apply_linear(&gat_in, h, h).silu();

        // Reshape per-head.
        // K → (b, h, head_size, seq)  via reshape + permute(0, 2, 3, 1)
        let k_h = k
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0_usize, 2, 3, 1])?;
        // V → (b, h, seq, head_size)  via reshape + transpose(1, 2) ≡ permute(0,2,1,3)
        let v_h = v
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0_usize, 2, 1, 3])?;
        let r_h = r
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?
            .permute([0_usize, 2, 1, 3])?;

        // decay[h, s, 1] = exp(-exp(time_decay))
        let td = x.const_f32_like(
            Arc::clone(&layer.attn_time_decay),
            Shape::from_dims(&[n_heads, head_size]),
        );
        let decay = td.exp().neg().exp();
        let decay = decay.reshape(Shape::from_dims(&[n_heads, head_size, 1]))?;

        let faaaa = x.const_f32_like(
            Arc::clone(&layer.attn_time_faaaa),
            Shape::from_dims(&[n_heads, head_size]),
        );
        let faaaa = faaaa.reshape(Shape::from_dims(&[n_heads, head_size, 1]))?;

        // Initial state: zeros (b, n_heads, head_size, head_size).
        let state_init = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * n_heads * head_size * head_size]),
            Shape::from_dims(&[batch, n_heads, head_size, head_size]),
        );
        let mut state = state_init;
        let mut outs: Vec<LazyTensor> = Vec::with_capacity(seq);
        for t in 0..seq {
            let kt = k_h.slice(3_usize, t, 1)?; // (b, n_heads, head_size, 1)
            let vt = v_h.slice(2_usize, t, 1)?; // (b, n_heads, 1, head_size)
            let rt = r_h.slice(2_usize, t, 1)?; // (b, n_heads, 1, head_size)
            let at = kt.matmul(&vt)?; // (b, n_heads, head_size, head_size)
            let rhs = faaaa.broadcast_mul(&at)?.add(&state)?; // (b, n_heads, S, S)
            let out_t = rt.matmul(&rhs)?.squeeze(2_usize)?; // (b, n_heads, head_size)
            state = at.add(&decay.broadcast_mul(&state)?)?;
            outs.push(out_t);
        }

        // Concat along a new seq dim. Each out_t is (b, n_heads, head_size).
        // Stack into (b, seq, n_heads, head_size).
        let mut stacked: Option<LazyTensor> = None;
        for out_t in outs.into_iter() {
            let with_seq = out_t.reshape(Shape::from_dims(&[batch, 1, n_heads, head_size]))?;
            stacked = Some(match stacked {
                None => with_seq,
                Some(s) => s.concat(&with_seq, 1_usize)?,
            });
        }
        let stacked = stacked.expect("at least one time step");
        // Reshape to (batch, seq, hidden_size).
        let stacked = stacked.reshape(Shape::from_dims(&[batch, seq, h]))?;

        // Per-head GroupNorm with num_groups = n_heads, num_channels = h.
        // Normalize over each (head_size,) group, then per-channel affine.
        let out = group_norm(
            &stacked, &layer.attn_ln_x_gain, &layer.attn_ln_x_bias,
            batch, seq, n_heads, head_size, 1e-5,
        )?;

        // Output projection: (b, seq, h) * g → out_proj.
        let gated = out.mul(&g)?;
        Ok(layer.attn_output.apply_linear(&gated, h, h))
    }

    fn channel_mix(
        &self,
        x: &LazyTensor,
        layer: &Rwkv5LayerWeights,
        batch: usize,
        seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.ffn_intermediate();
        let shifted = self.shift_seq(x, batch, seq, h);

        let mix_key = x.const_f32_like(
            Arc::clone(&layer.ffn_time_mix_key),
            Shape::from_dims(&[1, 1, h]),
        );
        let mix_rec = x.const_f32_like(
            Arc::clone(&layer.ffn_time_mix_receptance),
            Shape::from_dims(&[1, 1, h]),
        );
        let ones_t = x.const_f32_like(
            Arc::from(vec![1.0_f32; h]),
            Shape::from_dims(&[1, 1, h]),
        );
        let mk_inv = ones_t.sub(&mix_key)?;
        let mr_inv = ones_t.sub(&mix_rec)?;

        let k_in = x.broadcast_mul(&mix_key)?.add(&shifted.broadcast_mul(&mk_inv)?)?;
        let r_in = x.broadcast_mul(&mix_rec)?.add(&shifted.broadcast_mul(&mr_inv)?)?;

        // key = relu(K_proj(k_in)).square()
        let k = layer.ffn_key.apply_linear(&k_in, h, inter).relu().sqr();
        let v = layer.ffn_value.apply_linear(&k, inter, h);
        let r = layer.ffn_receptance.apply_linear(&r_in, h, h).sigmoid();
        r.mul(&v)
    }

    /// Shift `x` (shape `[batch, seq, h]`) one position along the
    /// seq axis: position t in the output equals position t-1 in
    /// the input, with position 0 = zeros.
    fn shift_seq(&self, x: &LazyTensor, batch: usize, seq: usize, h: usize) -> LazyTensor {
        let zero = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * h]),
            Shape::from_dims(&[batch, 1, h]),
        );
        if seq == 1 {
            return zero;
        }
        let earlier = x.slice(1_usize, 0, seq - 1).unwrap(); // (b, seq-1, h)
        zero.concat(&earlier, 1_usize).unwrap()
    }
}

/// Per-head GroupNorm: split `(batch, seq, hidden)` into
/// `(batch, seq, n_heads, head_size)`, normalize each group on
/// the last dim, then apply per-channel gain + bias.
fn group_norm(
    x: &LazyTensor,
    gain: &Arc<[f32]>,
    bias: &Arc<[f32]>,
    batch: usize,
    seq: usize,
    n_heads: usize,
    head_size: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let hidden = n_heads * head_size;
    assert_eq!(gain.len(), hidden);
    assert_eq!(bias.len(), hidden);
    let xh = x.reshape(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let mean = xh.mean_dim(3_usize)?; // (b, seq, n_heads)
    let mean_bc = mean
        .reshape(Shape::from_dims(&[batch, seq, n_heads, 1]))?
        .broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let centered = xh.sub(&mean_bc)?;
    let var = centered.sqr().mean_dim(3_usize)?;
    let inv_std = var.add_scalar(eps).rsqrt();
    let inv_std_bc = inv_std
        .reshape(Shape::from_dims(&[batch, seq, n_heads, 1]))?
        .broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let normed = centered.mul(&inv_std_bc)?;
    // Affine — gain/bias are per-channel (hidden,), reshape to (n_heads, head_size).
    let gain_t = x.const_f32_like(
        Arc::clone(gain),
        Shape::from_dims(&[1, 1, n_heads, head_size]),
    );
    let bias_t = x.const_f32_like(
        Arc::clone(bias),
        Shape::from_dims(&[1, 1, n_heads, head_size]),
    );
    let gain_bc = gain_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let bias_bc = bias_t.broadcast_to(Shape::from_dims(&[batch, seq, n_heads, head_size]))?;
    let scaled = normed.mul(&gain_bc)?.add(&bias_bc)?;
    scaled.reshape(Shape::from_dims(&[batch, seq, hidden]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &Rwkv5Config) -> Rwkv5Weights {
        let mut s: u32 = 91009;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let n_heads = cfg.n_heads();
        let inter = cfg.ffn_intermediate();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<Rwkv5LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|li| {
                let pre_ln = if li == 0 {
                    Some((Arc::from(vec![1.0_f32; h]), Arc::from(vec![0.0_f32; h])))
                } else {
                    None
                };
                Rwkv5LayerWeights {
                    ln1_gain: Arc::from(vec![1.0_f32; h]),
                    ln1_bias: Arc::from(vec![0.0_f32; h]),
                    ln2_gain: Arc::from(vec![1.0_f32; h]),
                    ln2_bias: Arc::from(vec![0.0_f32; h]),
                    pre_ln,
                    attn_time_mix_key: vec_of(h, &mut *nb),
                    attn_time_mix_value: vec_of(h, &mut *nb),
                    attn_time_mix_receptance: vec_of(h, &mut *nb),
                    attn_time_mix_gate: vec_of(h, &mut *nb),
                    // time_decay: small positive → decay close to 1.
                    attn_time_decay: vec_of(n_heads * cfg.head_size, &mut *nb),
                    attn_time_faaaa: vec_of(n_heads * cfg.head_size, &mut *nb),
                    attn_key: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_value: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_receptance: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_gate: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_output: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                    attn_ln_x_gain: Arc::from(vec![1.0_f32; h]),
                    attn_ln_x_bias: Arc::from(vec![0.0_f32; h]),
                    ffn_time_mix_key: vec_of(h, &mut *nb),
                    ffn_time_mix_receptance: vec_of(h, &mut *nb),
                    ffn_key: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    ffn_value: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                    ffn_receptance: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                }
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let head = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Rwkv5Weights { token_embedding, layers, final_ln_gain, final_ln_bias, head }
    }

    fn tiny_config() -> Rwkv5Config {
        Rwkv5Config {
            vocab_size: 16, hidden_size: 8,
            num_hidden_layers: 2, attention_hidden_size: 8,
            head_size: 4, num_attention_heads: 0,
            intermediate_size: Some(16),
            layer_norm_epsilon: 1e-5, rescale_every: 6,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Rwkv5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    /// Time-shift correctness: the state is initialized to zeros, and
    /// position 0 sees `shifted = zeros` regardless of the input.
    /// So a single-token forward must run without panic.
    #[test]
    fn single_token_runs() {
        let cfg = tiny_config();
        let model = Rwkv5Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[7]).unwrap().realize_f32();
        assert_eq!(logits.len(), cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    /// Test that the recurrent state actually propagates: the
    /// output at position t > 0 depends on position 0. Zero the
    /// embedding for token 0 vs leave it default; output at the
    /// last position must differ.
    #[test]
    fn state_carries_across_time() {
        let cfg = Rwkv5Config { num_hidden_layers: 1, ..tiny_config() };
        let base = tiny_weights(&cfg);
        // Run identical sequences but with different first tokens.
        let model = Rwkv5Model { config: cfg.clone(), weights: base };
        let a = model.forward(&[0, 5, 5, 5]).unwrap().realize_f32();
        let b = model.forward(&[1, 5, 5, 5]).unwrap().realize_f32();
        let last_a = &a[a.len() - cfg.vocab_size..];
        let last_b = &b[b.len() - cfg.vocab_size..];
        let mut max_diff = 0.0_f32;
        for (x, y) in last_a.iter().zip(last_b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "state must propagate: first-token change should affect last-token output, max_diff = {max_diff}");
    }

    /// The channel-mix (FFN) path is a pure function of the
    /// post-LN input + the previous token's input — toggle the
    /// FFN value-projection weights to zero, and the FFN contribution
    /// should drop, changing the output.
    #[test]
    fn ffn_value_zero_changes_output() {
        let cfg = Rwkv5Config { num_hidden_layers: 1, ..tiny_config() };
        let base = tiny_weights(&cfg);
        let mut zero_ffn_v = base.clone();
        let inter = cfg.ffn_intermediate();
        zero_ffn_v.layers[0].ffn_value = WeightStorage::F32(Arc::from(vec![0.0_f32; inter * cfg.hidden_size]));
        let m_a = Rwkv5Model { config: cfg.clone(), weights: base };
        let m_b = Rwkv5Model { config: cfg, weights: zero_ffn_v };
        let toks = [1_u32, 2, 3];
        let a = m_a.forward(&toks).unwrap().realize_f32();
        let b = m_b.forward(&toks).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6, "zeroing ffn_value must change output, max_diff = {max_diff}");
    }
}
