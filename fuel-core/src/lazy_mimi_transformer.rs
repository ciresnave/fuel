//! Mimi ProjectedTransformer — lazy port.
//!
//! Pre-LN transformer with rotary positional embeddings, learnable
//! layer-scale on each sublayer residual, optional input/output
//! linear projections. Used by Mimi between SeaNet's convolutional
//! encoder/decoder and the RVQ.
//!
//! v1 scope = the canonical Mimi v0.1 configuration:
//!   - `causal = true`, `norm_first = true` (pre-LN)
//!   - `bias_attn = false`, `bias_ff = false`
//!   - `layer_scale = 0.01` (learnable per-channel scale on each
//!     sublayer residual; one parameter vector per sublayer)
//!   - `positional_embedding = Rope` with `max_period = 10000`
//!   - `kv_repeat = 1` (no grouped-query)
//!   - `gating = None` (plain `fc1 → GELU(erf) → fc2` MLP)
//!   - `norm = LayerNorm` (ε = 1e-5)
//!   - `cross_attention = false`
//!   - `use_conv_block = false`
//!   - `conv_layout = true` (input/output transpose for the
//!     SeaNet's (B, C, T) layout)
//!
//! Forward-only inference: no rotating KV cache, no streaming
//! `step` API. The full sequence is processed in a single call.
//!
//! v1 scope: F32, batch == 1, prefill only.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// `dim_feedforward` from the eager config — hidden dim of the MLP.
    pub dim_feedforward: usize,
    pub max_period: f32,
    /// `conv_layout = true` means input shape is `(B, C, T)`; we
    /// transpose to `(B, T, C)` on entry and back on exit.
    pub conv_layout: bool,
    /// LayerNorm ε.
    pub layer_norm_eps: f64,
}

impl MimiTransformerConfig {
    /// Mimi v0.1 transformer preset.
    pub fn mimi_v0_1() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            num_layers: 8,
            dim_feedforward: 2048,
            max_period: 10_000.0,
            conv_layout: true,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> usize { self.d_model / self.num_heads }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MimiAttentionWeights {
    /// `(d_model, d_model)` each — bias-less.
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub o_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MimiMlpWeights {
    pub fc1: WeightStorage,
    pub fc2: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct MimiTransformerLayerWeights {
    pub norm1: LayerNormWeights,
    pub norm2: LayerNormWeights,
    pub attn: MimiAttentionWeights,
    pub mlp: MimiMlpWeights,
    /// Per-channel scale applied to each sublayer residual before
    /// `+ x`. `(d_model,)` each.
    pub layer_scale_1: Arc<[f32]>,
    pub layer_scale_2: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MimiTransformerWeights {
    pub layers: Vec<MimiTransformerLayerWeights>,
}

#[derive(Debug, Clone)]
pub struct ProjectedTransformerWeights {
    pub transformer: MimiTransformerWeights,
    /// Optional input projection `(input_dim, d_model)` when
    /// `input_dim != d_model`.
    pub input_proj: Option<WeightStorage>,
    /// One output projection per output dim. Each `(d_model,
    /// output_dim)` (or `None` if `output_dim == d_model`).
    pub output_projs: Vec<(Option<WeightStorage>, usize)>,
}

#[derive(Debug, Clone)]
pub struct ProjectedTransformerModel {
    pub config: MimiTransformerConfig,
    pub input_dim: usize,
    pub weights: ProjectedTransformerWeights,
}

// ---- Forward ---------------------------------------------------------------

impl ProjectedTransformerModel {
    /// Run the transformer and return one tensor per `output_dims`
    /// configured at build time. With `conv_layout = true`, both
    /// input and output are `(B, C, T)`; the transformer internally
    /// works on `(B, T, C)`.
    pub fn forward(&self, xs: &LazyTensor) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let dims = xs.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 3, "ProjectedTransformer expects rank 3 input");
        // Optional (B, C, T) → (B, T, C) transpose.
        let xs = if cfg.conv_layout {
            xs.permute([0, 2, 1_usize])?
        } else { xs.clone() };
        // Optional input projection.
        let h = match &self.weights.input_proj {
            None => xs,
            Some(w) => w.apply_linear(&xs, self.input_dim, cfg.d_model),
        };

        // Build RoPE cos/sin tables for the input sequence length.
        let in_dims = h.shape();
        let in_dims = in_dims.dims();
        let b = in_dims[0]; let t = in_dims[1];
        let head_dim = cfg.head_dim();
        let (cos, sin) = build_rope_tables(&h, t, head_dim, cfg.max_period);
        // Build the strict causal mask `(t, t)`.
        let causal_mask = build_causal_mask(&h, t);

        let mut hidden = h;
        for layer in &self.weights.transformer.layers {
            hidden = apply_layer(&hidden, layer, &cos, &sin, &causal_mask, cfg, b, t)?;
        }

        // Apply output projections.
        let mut outs = Vec::with_capacity(self.weights.output_projs.len());
        for (proj, out_dim) in &self.weights.output_projs {
            let y = match proj {
                None => hidden.clone(),
                Some(w) => w.apply_linear(&hidden, cfg.d_model, *out_dim),
            };
            let y = if cfg.conv_layout { y.permute([0, 2, 1_usize])? } else { y };
            outs.push(y);
        }
        Ok(outs)
    }
}

fn apply_layer(
    x: &LazyTensor,
    w: &MimiTransformerLayerWeights,
    cos: &LazyTensor, sin: &LazyTensor, causal_mask: &LazyTensor,
    cfg: &MimiTransformerConfig,
    b: usize, t: usize,
) -> Result<LazyTensor> {
    // Pre-LN self-attention.
    let n1 = apply_layer_norm(x, &w.norm1, cfg.d_model, cfg.layer_norm_eps)?;
    let attn = apply_attention(&n1, &w.attn, cos, sin, causal_mask, cfg, b, t)?;
    let scaled = apply_per_channel_scale(&attn, &w.layer_scale_1, cfg.d_model)?;
    let after_attn = x.add(&scaled)?;

    // Pre-LN MLP.
    let n2 = apply_layer_norm(&after_attn, &w.norm2, cfg.d_model, cfg.layer_norm_eps)?;
    let mlp = apply_mlp(&n2, &w.mlp, cfg)?;
    let scaled_mlp = apply_per_channel_scale(&mlp, &w.layer_scale_2, cfg.d_model)?;
    after_attn.add(&scaled_mlp)
}

fn apply_attention(
    x: &LazyTensor,
    w: &MimiAttentionWeights,
    cos: &LazyTensor, sin: &LazyTensor, causal_mask: &LazyTensor,
    cfg: &MimiTransformerConfig,
    b: usize, t: usize,
) -> Result<LazyTensor> {
    let d = cfg.d_model;
    let heads = cfg.num_heads;
    let head_dim = cfg.head_dim();
    let scale = 1.0_f64 / (head_dim as f64).sqrt();

    // (B, T, D) → (B, T, heads, head_dim) → (B, heads, T, head_dim)
    let q = w.q_proj.apply_linear(x, d, d)
        .reshape(Shape::from_dims(&[b, t, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let k = w.k_proj.apply_linear(x, d, d)
        .reshape(Shape::from_dims(&[b, t, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;
    let v = w.v_proj.apply_linear(x, d, d)
        .reshape(Shape::from_dims(&[b, t, heads, head_dim]))?
        .permute([0, 2, 1, 3_usize])?;

    // Apply RoPE-interleaved to q and k.
    let q = apply_rope_interleaved(&q, cos, sin, b, heads, t, head_dim)?;
    let k = apply_rope_interleaved(&k, cos, sin, b, heads, t, head_dim)?;

    // Collapse leading dims for the matmul.
    let batch = b * heads;
    let q3 = q.reshape(Shape::from_dims(&[batch, t, head_dim]))?;
    let k3 = k.reshape(Shape::from_dims(&[batch, t, head_dim]))?;
    let v3 = v.reshape(Shape::from_dims(&[batch, t, head_dim]))?;
    let kt = k3.permute([0, 2, 1_usize])?;
    let scores = q3.matmul(&kt)?.mul_scalar(scale);
    // Add the causal mask (`(t, t)` → broadcast to `(batch, t, t)`).
    let mask = causal_mask
        .reshape(Shape::from_dims(&[1, t, t]))?
        .broadcast_to(Shape::from_dims(&[batch, t, t]))?;
    let scores = scores.add(&mask)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v3)?
        .reshape(Shape::from_dims(&[b, heads, t, head_dim]))?
        .permute([0, 2, 1, 3_usize])?
        .reshape(Shape::from_dims(&[b, t, d]))?;
    Ok(w.o_proj.apply_linear(&ctx, d, d))
}

fn apply_mlp(
    x: &LazyTensor, w: &MimiMlpWeights, cfg: &MimiTransformerConfig,
) -> Result<LazyTensor> {
    let hidden = cfg.dim_feedforward;
    let d = cfg.d_model;
    let h = w.fc1.apply_linear(x, d, hidden).gelu_erf();
    Ok(w.fc2.apply_linear(&h, hidden, d))
}

fn apply_layer_norm(
    x: &LazyTensor, ln: &LayerNormWeights, hidden: usize, eps: f64,
) -> Result<LazyTensor> {
    let normed = x.layer_norm_last_dim(eps)?;
    let dims_v = x.shape().dims().to_vec();
    let mut shape = vec![1_usize; dims_v.len()];
    shape[dims_v.len() - 1] = hidden;
    let g = normed
        .const_f32_like(Arc::clone(&ln.gain), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&shape))?
        .broadcast_to(Shape::from_dims(&dims_v))?;
    let bias = normed
        .const_f32_like(Arc::clone(&ln.bias), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&shape))?
        .broadcast_to(Shape::from_dims(&dims_v))?;
    Ok(normed.mul(&g)?.add(&bias)?)
}

fn apply_per_channel_scale(
    x: &LazyTensor, scale: &Arc<[f32]>, hidden: usize,
) -> Result<LazyTensor> {
    let dims_v = x.shape().dims().to_vec();
    let mut shape = vec![1_usize; dims_v.len()];
    shape[dims_v.len() - 1] = hidden;
    let s = x
        .const_f32_like(Arc::clone(scale), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&shape))?
        .broadcast_to(Shape::from_dims(&dims_v))?;
    x.mul(&s)
}

// ---- RoPE ------------------------------------------------------------------

/// Build cos/sin tables shaped `(T, head_dim / 2)` for the
/// interleaved-RoPE convention used by Mimi (`rope_i`).
fn build_rope_tables(
    anchor: &LazyTensor, t: usize, head_dim: usize, max_period: f32,
) -> (LazyTensor, LazyTensor) {
    let half = head_dim / 2;
    let mut cos_v = Vec::with_capacity(t * half);
    let mut sin_v = Vec::with_capacity(t * half);
    for pos in 0..t {
        for i in 0..half {
            let freq = 1.0_f32 / max_period.powf((2 * i) as f32 / head_dim as f32);
            let theta = (pos as f32) * freq;
            cos_v.push(theta.cos());
            sin_v.push(theta.sin());
        }
    }
    let cos = anchor.const_f32_like(
        Arc::from(cos_v), Shape::from_dims(&[t, half]),
    );
    let sin = anchor.const_f32_like(
        Arc::from(sin_v), Shape::from_dims(&[t, half]),
    );
    (cos, sin)
}

/// Apply **interleaved** RoPE: pairs `(x[2i], x[2i+1])` rotate as
///   `x'[2i]   = x[2i]*cos - x[2i+1]*sin`
///   `x'[2i+1] = x[2i]*sin + x[2i+1]*cos`
/// Input `(B, heads, T, head_dim)`, cos/sin `(T, head_dim/2)`.
fn apply_rope_interleaved(
    x: &LazyTensor, cos: &LazyTensor, sin: &LazyTensor,
    b: usize, heads: usize, t: usize, head_dim: usize,
) -> Result<LazyTensor> {
    let half = head_dim / 2;
    // Reshape (B, H, T, D) → (B, H, T, half, 2) and split into the
    // (even, odd) pair via `narrow` on the last dim.
    let x_pairs = x.reshape(Shape::from_dims(&[b, heads, t, half, 2]))?;
    let x_even = x_pairs.narrow(4_usize, 0, 1)?
        .reshape(Shape::from_dims(&[b, heads, t, half]))?;
    let x_odd = x_pairs.narrow(4_usize, 1, 1)?
        .reshape(Shape::from_dims(&[b, heads, t, half]))?;
    // Broadcast cos/sin (T, half) → (B, H, T, half).
    let cos_b = cos
        .reshape(Shape::from_dims(&[1, 1, t, half]))?
        .broadcast_to(Shape::from_dims(&[b, heads, t, half]))?;
    let sin_b = sin
        .reshape(Shape::from_dims(&[1, 1, t, half]))?
        .broadcast_to(Shape::from_dims(&[b, heads, t, half]))?;
    let new_even = x_even.mul(&cos_b)?.sub(&x_odd.mul(&sin_b)?)?;
    let new_odd = x_even.mul(&sin_b)?.add(&x_odd.mul(&cos_b)?)?;
    // Re-interleave via stack along a fresh axis + reshape.
    let stacked = LazyTensor::stack(&[&new_even, &new_odd], 4_usize)?;
    stacked.reshape(Shape::from_dims(&[b, heads, t, head_dim]))
}

fn build_causal_mask(anchor: &LazyTensor, t: usize) -> LazyTensor {
    let mut data = vec![0.0_f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    anchor.const_f32_like(Arc::from(data), Shape::from_dims(&[t, t]))
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }
    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn tiny_cfg() -> MimiTransformerConfig {
        MimiTransformerConfig {
            d_model: 8, num_heads: 2, num_layers: 2,
            dim_feedforward: 16,
            max_period: 10_000.0,
            conv_layout: true,
            layer_norm_eps: 1e-5,
        }
    }

    fn build_layer(d: usize, ff: usize, nb: &mut dyn FnMut() -> f32) -> MimiTransformerLayerWeights {
        MimiTransformerLayerWeights {
            norm1: ln_w(d),
            norm2: ln_w(d),
            attn: MimiAttentionWeights {
                q_proj: ws(d * d, nb),
                k_proj: ws(d * d, nb),
                v_proj: ws(d * d, nb),
                o_proj: ws(d * d, nb),
            },
            mlp: MimiMlpWeights {
                fc1: ws(d * ff, nb),
                fc2: ws(ff * d, nb),
            },
            layer_scale_1: vec_of(d, nb),
            layer_scale_2: vec_of(d, nb),
        }
    }

    fn tiny_model(input_dim: usize, output_dims: Vec<usize>) -> ProjectedTransformerModel {
        let cfg = tiny_cfg();
        let mut nb = rng_seed(2026);
        let d = cfg.d_model;
        let ff = cfg.dim_feedforward;
        let transformer = MimiTransformerWeights {
            layers: (0..cfg.num_layers).map(|_| build_layer(d, ff, &mut nb)).collect(),
        };
        let input_proj = if input_dim == d {
            None
        } else {
            Some(ws(input_dim * d, &mut nb))
        };
        let output_projs: Vec<_> = output_dims.iter().map(|&od| {
            let proj = if od == d { None } else { Some(ws(d * od, &mut nb)) };
            (proj, od)
        }).collect();
        ProjectedTransformerModel {
            config: cfg,
            input_dim,
            weights: ProjectedTransformerWeights { transformer, input_proj, output_projs },
        }
    }

    #[test]
    fn forward_returns_one_per_output_dim() {
        let d = 8;
        let model = tiny_model(d, vec![d]);
        let xs = LazyTensor::from_f32(
            (0..(1 * d * 5)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, d, 5]), &Device::cpu(),
        );
        let outs = model.forward(&xs).unwrap();
        assert_eq!(outs.len(), 1);
        // conv_layout = true → (B, C, T) preserved.
        assert_eq!(outs[0].shape().dims(), &[1, d, 5]);
        for &v in &outs[0].realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn forward_with_different_input_dim_uses_input_proj() {
        let input_dim = 4; let d = 8;
        let model = tiny_model(input_dim, vec![d]);
        let xs = LazyTensor::from_f32(
            (0..(1 * input_dim * 5)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, input_dim, 5]), &Device::cpu(),
        );
        let outs = model.forward(&xs).unwrap();
        assert_eq!(outs[0].shape().dims(), &[1, d, 5]);
    }

    #[test]
    fn forward_multiple_output_projections() {
        let d = 8;
        let model = tiny_model(d, vec![d, 4]);
        let xs = LazyTensor::from_f32(
            (0..(1 * d * 5)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, d, 5]), &Device::cpu(),
        );
        let outs = model.forward(&xs).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape().dims(), &[1, d, 5]);
        assert_eq!(outs[1].shape().dims(), &[1, 4, 5]);
    }

    #[test]
    fn rope_zero_position_is_identity() {
        // At pos=0, cos=1, sin=0 → RoPE is identity.
        let b = 1; let h = 2; let t = 1; let d = 4;
        let x_data: Vec<f32> = (0..(b * h * t * d)).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let x = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[b, h, t, d]), &Device::cpu(),
        );
        let (cos, sin) = build_rope_tables(&x, t, d, 10_000.0);
        let out = apply_rope_interleaved(&x, &cos, &sin, b, h, t, d).unwrap();
        let out_data = out.realize_f32();
        for (a, b) in x_data.iter().zip(out_data.iter()) {
            assert!((a - b).abs() < 1e-6, "RoPE at pos=0 should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn causal_mask_enforces_position() {
        // Run forward over a sequence; mutate the last token's input
        // and verify earlier-position outputs are unchanged.
        let d = 8;
        let model = tiny_model(d, vec![d]);
        let t = 4;
        let xs_a = LazyTensor::from_f32(
            (0..(1 * d * t)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, d, t]), &Device::cpu(),
        );
        // Build xs_b with last (T-1) column replaced.
        let mut data_b: Vec<f32> = (0..(1 * d * t)).map(|i| (i as f32) * 0.01).collect();
        for c in 0..d {
            data_b[c * t + (t - 1)] = 9.0; // bump last token in every channel
        }
        let xs_b = LazyTensor::from_f32(
            data_b, Shape::from_dims(&[1, d, t]), &Device::cpu(),
        );
        let oa = model.forward(&xs_a).unwrap();
        let ob = model.forward(&xs_b).unwrap();
        let da = oa[0].realize_f32();
        let db = ob[0].realize_f32();
        // Output is (1, d, t). Positions 0..t-2 must match across runs.
        let mut max_pos_diff = vec![0.0_f32; t];
        for c in 0..d {
            for pos in 0..t {
                let i = c * t + pos;
                let d_pos = (da[i] - db[i]).abs();
                if d_pos > max_pos_diff[pos] { max_pos_diff[pos] = d_pos; }
            }
        }
        for pos in 0..(t - 1) {
            assert!(max_pos_diff[pos] < 1e-5,
                "causal mask violated at pos {pos}: max_diff = {}", max_pos_diff[pos]);
        }
        assert!(max_pos_diff[t - 1] > 1e-6,
            "last-pos output should respond to last-token input change");
    }

    #[test]
    fn preset_mimi_v0_1() {
        let p = MimiTransformerConfig::mimi_v0_1();
        assert_eq!(p.d_model, 512);
        assert_eq!(p.num_heads, 8);
        assert_eq!(p.num_layers, 8);
        assert_eq!(p.head_dim(), 64);
    }
}
