//! Lazy LSTM — PyTorch-shape multi-layer LSTM unrolled at
//! graph-build time.
//!
//! Reference (per-layer, per-time-step):
//!   gates = x_t · Wᵢₕᵀ + bᵢₕ + h_{t-1} · Wₕₕᵀ + bₕₕ
//!   (i, f, g, o) = chunk(gates, 4) along the hidden dim
//!   i = σ(i); f = σ(f); g = tanh(g); o = σ(o)
//!   c_t = f ⊙ c_{t-1} + i ⊙ g
//!   h_t = o ⊙ tanh(c_t)
//!
//! Input shape `(B, T, D_in)`; output `(B, T, D_hidden)`.
//! Initial hidden / cell states default to zero.
//!
//! Multi-layer stacks just chain the output of layer L as the
//! input of layer L+1.
//!
//! `LstmStack::forward_with_residual` additionally adds the
//! original input to the output of the last layer — matches the
//! EnCodec / SNAC / Parler-TTS pattern of a residual around the
//! LSTM block.
//!
//! v1 scope:
//!   - F32, batch == 1.
//!   - Forward-only (autograd through LSTM gates is fuel-graph's
//!     concern, not handled here).
//!   - Bidirectional and packed sequences are out of scope.

use crate::lazy::LazyTensor;
use crate::Result;
use fuel_ir::Shape;
use std::sync::Arc;

/// Per-layer weights matching PyTorch's `nn.LSTM` convention:
/// gates are stored in the order `[i, f, g, o]` concatenated
/// along the hidden dim — total `4·hidden`.
#[derive(Debug, Clone)]
pub struct LstmCellWeights {
    /// `[4·hidden, input_dim]`.
    pub w_ih: Arc<[f32]>,
    /// `[4·hidden, hidden]`.
    pub w_hh: Arc<[f32]>,
    /// `[4·hidden]`.
    pub b_ih: Arc<[f32]>,
    /// `[4·hidden]`.
    pub b_hh: Arc<[f32]>,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

#[derive(Debug, Clone)]
pub struct LstmStack {
    pub layers: Vec<LstmCellWeights>,
}

impl LstmStack {
    /// Forward pass without residual. Input `(1, T, D_in)`; output
    /// `(1, T, D_hidden)` where `D_hidden` is the last layer's
    /// hidden dim.
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = lstm_layer_forward(&h, layer)?;
        }
        Ok(h)
    }

    /// Forward pass with a residual add against the input (skip
    /// connection around the whole stack). Used by EnCodec / SNAC.
    /// Requires `x.shape() == output.shape()` (which holds when
    /// the input dim equals the last layer's hidden dim).
    pub fn forward_with_residual(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let out = self.forward(x)?;
        out.add(x)
    }
}

/// Single-layer LSTM forward, unrolled over time.
fn lstm_layer_forward(
    x: &LazyTensor, w: &LstmCellWeights,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims.len(), 3, "LSTM input must be rank 3 [B, T, D_in]");
    let b = dims[0];
    let t = dims[1];
    let d_in = dims[2];
    assert_eq!(d_in, w.input_dim,
        "input dim mismatch: got {d_in}, expected {}", w.input_dim);
    assert!(t > 0, "LSTM requires non-empty time axis");
    let h_dim = w.hidden_dim;
    let four_h = 4 * h_dim;

    // Weight + bias constants on the input's graph.
    let w_ih = x.const_f32_like(
        Arc::clone(&w.w_ih), Shape::from_dims(&[four_h, d_in]),
    );
    let w_hh = x.const_f32_like(
        Arc::clone(&w.w_hh), Shape::from_dims(&[four_h, h_dim]),
    );
    let b_ih = x.const_f32_like(
        Arc::clone(&w.b_ih), Shape::from_dims(&[four_h]),
    );
    let b_hh = x.const_f32_like(
        Arc::clone(&w.b_hh), Shape::from_dims(&[four_h]),
    );
    let b_combined = b_ih.add(&b_hh)?;
    // Broadcast bias to (B, 4·H) for elementwise add per time step.
    let bias = b_combined
        .reshape(Shape::from_dims(&[1, four_h]))?
        .broadcast_to(Shape::from_dims(&[b, four_h]))?;

    // Initial h and c = zeros, shape (B, H).
    let zeros_bh = x.const_f32_like(
        Arc::<[f32]>::from(vec![0.0_f32; b * h_dim]),
        Shape::from_dims(&[b, h_dim]),
    );
    let mut h_prev = zeros_bh.clone();
    let mut c_prev = zeros_bh;

    let w_ih_t = w_ih.transpose()?;
    let w_hh_t = w_hh.transpose()?;
    let mut outputs: Vec<LazyTensor> = Vec::with_capacity(t);
    for step in 0..t {
        // x_t: (B, D_in)
        let x_t = x
            .slice(1_usize, step, 1)?
            .reshape(Shape::from_dims(&[b, d_in]))?;
        // gates = x_t · w_ihᵀ + h_prev · w_hhᵀ + bias_combined
        let gx = x_t.matmul(&w_ih_t)?;
        let gh = h_prev.matmul(&w_hh_t)?;
        let gates = gx.add(&gh)?.add(&bias)?;

        // Split into i, f, g, o each (B, H).
        let i = gates.slice(1_usize, 0, h_dim)?;
        let f = gates.slice(1_usize, h_dim, h_dim)?;
        let g = gates.slice(1_usize, 2 * h_dim, h_dim)?;
        let o = gates.slice(1_usize, 3 * h_dim, h_dim)?;

        let i = i.sigmoid();
        let f = f.sigmoid();
        let g = g.tanh();
        let o = o.sigmoid();

        // c_t = f ⊙ c_prev + i ⊙ g
        let c_t = f.mul(&c_prev)?.add(&i.mul(&g)?)?;
        // h_t = o ⊙ tanh(c_t)
        let h_t = o.mul(&c_t.tanh())?;

        // Stash (1, 1, H) sliced for the eventual concat.
        outputs.push(h_t.reshape(Shape::from_dims(&[b, 1, h_dim]))?);
        h_prev = h_t;
        c_prev = c_t;
    }
    // Concat along time dim into (B, T, H).
    let mut out = outputs[0].clone();
    for o_t in &outputs[1..] {
        out = out.concat(o_t, 1_usize)?;
    }
    Ok(out)
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    /// Reference Rust implementation of a single LSTM layer with
    /// zero initial state, applied to a single batch and explicit
    /// gate values. Hidden / input dims are 2 in this fixture.
    fn lstm_layer_reference(
        x: &[f32], b: usize, t: usize, d_in: usize, d_h: usize,
        w_ih: &[f32], w_hh: &[f32], b_ih: &[f32], b_hh: &[f32],
    ) -> Vec<f32> {
        let four_h = 4 * d_h;
        assert_eq!(x.len(), b * t * d_in);
        assert_eq!(w_ih.len(), four_h * d_in);
        assert_eq!(w_hh.len(), four_h * d_h);
        assert_eq!(b_ih.len(), four_h);
        assert_eq!(b_hh.len(), four_h);

        let mut out = vec![0.0_f32; b * t * d_h];
        let mut h_prev = vec![0.0_f32; b * d_h];
        let mut c_prev = vec![0.0_f32; b * d_h];
        let mut gates = vec![0.0_f32; b * four_h];
        for step in 0..t {
            // gates = x_t · W_ihᵀ + h_prev · W_hhᵀ + b_ih + b_hh
            for bi in 0..b {
                for k in 0..four_h {
                    let mut g = b_ih[k] + b_hh[k];
                    for j in 0..d_in {
                        g += x[(bi * t + step) * d_in + j] * w_ih[k * d_in + j];
                    }
                    for j in 0..d_h {
                        g += h_prev[bi * d_h + j] * w_hh[k * d_h + j];
                    }
                    gates[bi * four_h + k] = g;
                }
            }
            // Apply gating + state update.
            let mut h_new = vec![0.0_f32; b * d_h];
            let mut c_new = vec![0.0_f32; b * d_h];
            for bi in 0..b {
                for k in 0..d_h {
                    let i = sigmoid(gates[bi * four_h + k]);
                    let f = sigmoid(gates[bi * four_h + d_h + k]);
                    let g = tanh(gates[bi * four_h + 2 * d_h + k]);
                    let o = sigmoid(gates[bi * four_h + 3 * d_h + k]);
                    let c_t = f * c_prev[bi * d_h + k] + i * g;
                    let h_t = o * tanh(c_t);
                    h_new[bi * d_h + k] = h_t;
                    c_new[bi * d_h + k] = c_t;
                }
            }
            for bi in 0..b {
                for k in 0..d_h {
                    out[(bi * t + step) * d_h + k] = h_new[bi * d_h + k];
                }
            }
            h_prev = h_new;
            c_prev = c_new;
        }
        out
    }

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    fn tanh(x: f32) -> f32 { x.tanh() }

    #[test]
    fn single_layer_matches_reference() {
        let b = 1; let t = 4; let d_in = 2; let d_h = 2;
        let four_h = 4 * d_h;
        let x_data: Vec<f32> = (0..(b * t * d_in)).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w_ih: Vec<f32> = (0..(four_h * d_in)).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let w_hh: Vec<f32> = (0..(four_h * d_h)).map(|i| (i as f32) * 0.07 - 0.2).collect();
        let b_ih: Vec<f32> = (0..four_h).map(|i| (i as f32) * 0.02).collect();
        let b_hh: Vec<f32> = (0..four_h).map(|i| (i as f32) * 0.03 - 0.1).collect();

        let expected = lstm_layer_reference(
            &x_data, b, t, d_in, d_h, &w_ih, &w_hh, &b_ih, &b_hh,
        );

        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[b, t, d_in]), &Device::cpu(),
        );
        let stack = LstmStack {
            layers: vec![LstmCellWeights {
                w_ih: Arc::from(w_ih),
                w_hh: Arc::from(w_hh),
                b_ih: Arc::from(b_ih),
                b_hh: Arc::from(b_hh),
                input_dim: d_in,
                hidden_dim: d_h,
            }],
        };
        let out = stack.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[b, t, d_h]);
        let got = out.realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5,
                "lstm[{i}] expected {e}, got {a}");
        }
    }

    #[test]
    fn two_layer_stack_chain_is_correct() {
        // Layer 1: d_in=3 → d_h=4. Layer 2: d_in=4 → d_h=4.
        let b = 1; let t = 3;
        let d_in1 = 3; let d_h1 = 4;
        let d_in2 = d_h1; let d_h2 = 4;
        let x_data: Vec<f32> = (0..(b * t * d_in1)).map(|i| (i as f32) * 0.1).collect();
        let w_ih1: Vec<f32> = (0..(4 * d_h1 * d_in1)).map(|i| (i as f32) * 0.03 - 0.4).collect();
        let w_hh1: Vec<f32> = (0..(4 * d_h1 * d_h1)).map(|i| (i as f32) * 0.04 - 0.3).collect();
        let b_ih1: Vec<f32> = (0..(4 * d_h1)).map(|i| (i as f32) * 0.01).collect();
        let b_hh1: Vec<f32> = (0..(4 * d_h1)).map(|i| (i as f32) * 0.015 - 0.05).collect();
        let w_ih2: Vec<f32> = (0..(4 * d_h2 * d_in2)).map(|i| (i as f32) * 0.025 - 0.2).collect();
        let w_hh2: Vec<f32> = (0..(4 * d_h2 * d_h2)).map(|i| (i as f32) * 0.035 - 0.25).collect();
        let b_ih2: Vec<f32> = (0..(4 * d_h2)).map(|i| (i as f32) * 0.005).collect();
        let b_hh2: Vec<f32> = (0..(4 * d_h2)).map(|i| (i as f32) * 0.008 - 0.03).collect();

        let after_l1 = lstm_layer_reference(
            &x_data, b, t, d_in1, d_h1, &w_ih1, &w_hh1, &b_ih1, &b_hh1,
        );
        let expected = lstm_layer_reference(
            &after_l1, b, t, d_in2, d_h2, &w_ih2, &w_hh2, &b_ih2, &b_hh2,
        );

        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[b, t, d_in1]), &Device::cpu(),
        );
        let stack = LstmStack {
            layers: vec![
                LstmCellWeights {
                    w_ih: Arc::from(w_ih1), w_hh: Arc::from(w_hh1),
                    b_ih: Arc::from(b_ih1), b_hh: Arc::from(b_hh1),
                    input_dim: d_in1, hidden_dim: d_h1,
                },
                LstmCellWeights {
                    w_ih: Arc::from(w_ih2), w_hh: Arc::from(w_hh2),
                    b_ih: Arc::from(b_ih2), b_hh: Arc::from(b_hh2),
                    input_dim: d_in2, hidden_dim: d_h2,
                },
            ],
        };
        let got = stack.forward(&x).unwrap().realize_f32();
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4,
                "two-layer[{i}] expected {e}, got {a}");
        }
    }

    /// Residual variant adds input to output. Use a config where
    /// `input_dim == hidden_dim` so the shapes match.
    #[test]
    fn forward_with_residual_adds_input() {
        let b = 1; let t = 2; let d = 3;
        let four_d = 4 * d;
        let x_data: Vec<f32> = vec![0.5_f32, 0.25, -0.1, 0.7, 0.3, -0.4];
        let w_ih: Vec<f32> = vec![0.0_f32; four_d * d];
        let w_hh: Vec<f32> = vec![0.0_f32; four_d * d];
        let b_ih: Vec<f32> = vec![0.0_f32; four_d];
        let b_hh: Vec<f32> = vec![0.0_f32; four_d];

        // All zero weights → gates are zero → i=f=o=σ(0)=0.5, g=tanh(0)=0.
        // c_t = 0.5*c_{t-1} + 0.5*0 = 0.5*c_{t-1}; since c_0 = 0, c_t = 0
        // for all t. h_t = 0.5 * tanh(0) = 0 for all t. So plain
        // `forward` output is all zeros; `forward_with_residual` output
        // must equal the input.
        let x = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[b, t, d]), &Device::cpu(),
        );
        let stack = LstmStack {
            layers: vec![LstmCellWeights {
                w_ih: Arc::from(w_ih), w_hh: Arc::from(w_hh),
                b_ih: Arc::from(b_ih), b_hh: Arc::from(b_hh),
                input_dim: d, hidden_dim: d,
            }],
        };
        let plain = stack.forward(&x).unwrap().realize_f32();
        for &v in &plain {
            assert!(v.abs() < 1e-6, "expected zero, got {v}");
        }
        let with_res = stack.forward_with_residual(&x).unwrap().realize_f32();
        for (i, (a, e)) in with_res.iter().zip(x_data.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6,
                "residual[{i}] expected {e}, got {a}");
        }
    }

    /// Sanity: input changes propagate to output.
    #[test]
    fn responds_to_input() {
        let b = 1; let t = 4; let d_in = 4; let d_h = 4;
        let four_h = 4 * d_h;
        let w_ih: Vec<f32> = (0..(four_h * d_in)).map(|i| (i as f32) * 0.02).collect();
        let w_hh: Vec<f32> = (0..(four_h * d_h)).map(|i| (i as f32) * 0.02 - 0.05).collect();
        let b_ih: Vec<f32> = vec![0.0_f32; four_h];
        let b_hh: Vec<f32> = vec![0.0_f32; four_h];
        let stack = LstmStack {
            layers: vec![LstmCellWeights {
                w_ih: Arc::from(w_ih), w_hh: Arc::from(w_hh),
                b_ih: Arc::from(b_ih), b_hh: Arc::from(b_hh),
                input_dim: d_in, hidden_dim: d_h,
            }],
        };
        let xa = LazyTensor::from_f32(
            (0..(b * t * d_in)).map(|i| (i as f32) * 0.05).collect::<Vec<_>>(),
            Shape::from_dims(&[b, t, d_in]), &Device::cpu(),
        );
        let xb = LazyTensor::from_f32(
            (0..(b * t * d_in)).map(|i| (i as f32) * 0.05 + 0.3).collect::<Vec<_>>(),
            Shape::from_dims(&[b, t, d_in]), &Device::cpu(),
        );
        let oa = stack.forward(&xa).unwrap().realize_f32();
        let ob = stack.forward(&xb).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (a, b) in oa.iter().zip(ob.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-7,
            "LSTM must respond to input changes, max_diff = {max_diff}");
    }
}
