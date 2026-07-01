//! Lazy GRU — PyTorch-shape multi-layer GRU unrolled at
//! graph-build time.
//!
//! Reference (per-layer, per-time-step):
//!   r_t = σ(W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr)
//!   z_t = σ(W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz)
//!   n_t = tanh(W_in·x_t + b_in + r_t ⊙ (W_hn·h_{t-1} + b_hn))
//!   h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h_{t-1}
//!
//! Input shape `(B, T, D_in)`; output `(B, T, D_hidden)`. Gates
//! are stored in the order `[r, z, n]` concatenated along the
//! hidden dim — total `3·hidden` — matching PyTorch's
//! `nn.GRU` weight layout.
//!
//! Initial hidden state defaults to zero; the
//! [`GruStack::forward_with_initial_state`] variant takes a
//! pre-built `(num_layers, B, H)` tensor (sliced per layer
//! along axis 0).
//!
//! Multi-layer stacks just chain the output of layer L as the
//! input of layer L+1, like `nn.GRU(num_layers=N)`.
//!
//! v1 scope:
//!   - F32, arbitrary batch size.
//!   - Forward-only (autograd through GRU gates is fuel-graph's
//!     concern, not handled here).
//!   - Bidirectional and packed sequences are out of scope.

use crate::lazy::LazyTensor;
use crate::Result;
use fuel_ir::Shape;
use std::sync::Arc;

/// Per-layer weights matching PyTorch's `nn.GRU` convention:
/// gates are stored in the order `[r, z, n]` concatenated
/// along the hidden dim — total `3·hidden`.
#[derive(Debug, Clone)]
pub struct GruCellWeights {
    /// `[3·hidden, input_dim]`.
    pub w_ih: Arc<[f32]>,
    /// `[3·hidden, hidden]`.
    pub w_hh: Arc<[f32]>,
    /// `[3·hidden]`.
    pub b_ih: Arc<[f32]>,
    /// `[3·hidden]`.
    pub b_hh: Arc<[f32]>,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

#[derive(Debug, Clone)]
pub struct GruStack {
    pub layers: Vec<GruCellWeights>,
}

impl GruStack {
    /// Forward pass with zero initial hidden state. Input
    /// `(B, T, D_in)`; output `(B, T, D_hidden)` where
    /// `D_hidden` is the last layer's hidden dim.
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = gru_layer_forward(&h, layer, None)?;
        }
        Ok(h)
    }

    /// Forward pass with an explicit initial hidden state.
    ///
    /// `h0` shape: `(num_layers, B, H)`, matching PyTorch's
    /// `nn.GRU(h_0)` convention. Each layer's slice along axis
    /// 0 is reshaped to `(B, H)` and fed as the layer's
    /// `h_{-1}`. The per-layer hidden dim must match the
    /// stacked layer's `hidden_dim`.
    pub fn forward_with_initial_state(
        &self, x: &LazyTensor, h0: &LazyTensor,
    ) -> Result<LazyTensor> {
        let h0_shape = h0.shape();
        let h0_dims = h0_shape.dims();
        if h0_dims.len() != 3 {
            return Err(crate::Error::Msg(format!(
                "GRU: initial state must be rank 3 [num_layers, B, H], got {h0_dims:?}",
            ))
            .bt());
        }
        if h0_dims[0] != self.layers.len() {
            return Err(crate::Error::Msg(format!(
                "GRU: initial state has {} layers, stack has {}",
                h0_dims[0], self.layers.len(),
            ))
            .bt());
        }
        let x_dims = x.shape();
        let x_dims = x_dims.dims();
        if x_dims.len() != 3 {
            return Err(crate::Error::Msg(format!(
                "GRU: input must be rank 3 [B, T, D_in], got {x_dims:?}",
            ))
            .bt());
        }
        let b = x_dims[0];
        if h0_dims[1] != b {
            return Err(crate::Error::Msg(format!(
                "GRU: initial state batch {} != input batch {}",
                h0_dims[1], b,
            ))
            .bt());
        }
        let mut h = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            if h0_dims[2] != layer.hidden_dim {
                return Err(crate::Error::Msg(format!(
                    "GRU: initial state hidden {} != layer {idx} hidden {}",
                    h0_dims[2], layer.hidden_dim,
                ))
                .bt());
            }
            // Slice out (1, B, H) and squeeze to (B, H).
            let h0_l = h0
                .slice(0_usize, idx, 1)?
                .reshape(Shape::from_dims(&[b, layer.hidden_dim]))?;
            h = gru_layer_forward(&h, layer, Some(h0_l))?;
        }
        Ok(h)
    }
}

/// Single-layer GRU forward, unrolled over time. `h0` is the
/// per-layer initial hidden state of shape `(B, H)`; `None`
/// means zero.
fn gru_layer_forward(
    x: &LazyTensor, w: &GruCellWeights, h0: Option<LazyTensor>,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    assert_eq!(dims.len(), 3, "GRU input must be rank 3 [B, T, D_in]");
    let b = dims[0];
    let t = dims[1];
    let d_in = dims[2];
    assert_eq!(d_in, w.input_dim,
        "input dim mismatch: got {d_in}, expected {}", w.input_dim);
    assert!(t > 0, "GRU requires non-empty time axis");
    let h_dim = w.hidden_dim;
    let three_h = 3 * h_dim;

    // Weight + bias constants on the input's graph.
    let w_ih = x.const_f32_like(
        Arc::clone(&w.w_ih), Shape::from_dims(&[three_h, d_in]),
    );
    let w_hh = x.const_f32_like(
        Arc::clone(&w.w_hh), Shape::from_dims(&[three_h, h_dim]),
    );
    let b_ih = x.const_f32_like(
        Arc::clone(&w.b_ih), Shape::from_dims(&[three_h]),
    );
    let b_hh = x.const_f32_like(
        Arc::clone(&w.b_hh), Shape::from_dims(&[three_h]),
    );
    // Broadcast biases to (B, 3·H) for elementwise add per time step.
    let b_ih_b = b_ih
        .reshape(Shape::from_dims(&[1, three_h]))?
        .broadcast_to(Shape::from_dims(&[b, three_h]))?;
    let b_hh_b = b_hh
        .reshape(Shape::from_dims(&[1, three_h]))?
        .broadcast_to(Shape::from_dims(&[b, three_h]))?;

    // Initial h: zero or supplied (B, H).
    let mut h_prev = match h0 {
        Some(h) => h,
        None => x.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; b * h_dim]),
            Shape::from_dims(&[b, h_dim]),
        ),
    };

    let w_ih_t = w_ih.transpose()?;
    let w_hh_t = w_hh.transpose()?;
    let mut outputs: Vec<LazyTensor> = Vec::with_capacity(t);
    for step in 0..t {
        // x_t: (B, D_in).
        let x_t = x
            .slice(1_usize, step, 1)?
            .reshape(Shape::from_dims(&[b, d_in]))?;
        // gates_ih = x_t · W_ihᵀ + b_ih   (shape (B, 3·H))
        // gates_hh = h_prev · W_hhᵀ + b_hh
        let gates_ih = x_t.matmul(&w_ih_t)?.add(&b_ih_b)?;
        let gates_hh = h_prev.matmul(&w_hh_t)?.add(&b_hh_b)?;

        // Split into r, z, n along the hidden dim.
        let r_ih = gates_ih.slice(1_usize, 0, h_dim)?;
        let z_ih = gates_ih.slice(1_usize, h_dim, h_dim)?;
        let n_ih = gates_ih.slice(1_usize, 2 * h_dim, h_dim)?;
        let r_hh = gates_hh.slice(1_usize, 0, h_dim)?;
        let z_hh = gates_hh.slice(1_usize, h_dim, h_dim)?;
        let n_hh = gates_hh.slice(1_usize, 2 * h_dim, h_dim)?;

        let r = r_ih.add(&r_hh)?.sigmoid();
        let z = z_ih.add(&z_hh)?.sigmoid();
        // n = tanh(n_ih + r ⊙ n_hh)
        let n = n_ih.add(&r.mul(&n_hh)?)?.tanh();

        // h_t = (1 − z) ⊙ n + z ⊙ h_prev
        //     = n − z ⊙ n + z ⊙ h_prev
        let one_minus_z_n = n.sub(&z.mul(&n)?)?;
        let h_t = one_minus_z_n.add(&z.mul(&h_prev)?)?;

        outputs.push(h_t.reshape(Shape::from_dims(&[b, 1, h_dim]))?);
        h_prev = h_t;
    }
    // Concat along time dim into (B, T, H).
    let mut out = outputs[0].clone();
    for o_t in &outputs[1..] {
        out = out.concat(o_t, 1_usize)?;
    }
    Ok(out)
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl GruCellWeights {
    /// Load a single layer's weights from a HuggingFace
    /// `MmapedSafetensors` checkpoint at `{prefix}` following
    /// PyTorch's `nn.GRU` naming convention:
    /// `{prefix}weight_ih_l{layer}`, `{prefix}weight_hh_l{layer}`,
    /// `{prefix}bias_ih_l{layer}`, `{prefix}bias_hh_l{layer}`.
    /// `prefix` is typically empty (`""`) or a module prefix
    /// ending in `.` (e.g. `"encoder.rnn."`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        layer: usize,
        input_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        use crate::lazy::load_tensor_as_f32;
        let three_h = 3 * hidden_dim;
        let w_ih = load_tensor_as_f32(
            st, &format!("{prefix}weight_ih_l{layer}"),
        )?;
        if w_ih.len() != three_h * input_dim {
            crate::bail!(
                "{prefix}weight_ih_l{layer}: {} elements, expected {} ({three_h}×{input_dim})",
                w_ih.len(), three_h * input_dim,
            );
        }
        let w_hh = load_tensor_as_f32(
            st, &format!("{prefix}weight_hh_l{layer}"),
        )?;
        if w_hh.len() != three_h * hidden_dim {
            crate::bail!(
                "{prefix}weight_hh_l{layer}: {} elements, expected {} ({three_h}×{hidden_dim})",
                w_hh.len(), three_h * hidden_dim,
            );
        }
        let b_ih = load_tensor_as_f32(
            st, &format!("{prefix}bias_ih_l{layer}"),
        )?;
        if b_ih.len() != three_h {
            crate::bail!(
                "{prefix}bias_ih_l{layer}: {} elements, expected {three_h}",
                b_ih.len(),
            );
        }
        let b_hh = load_tensor_as_f32(
            st, &format!("{prefix}bias_hh_l{layer}"),
        )?;
        if b_hh.len() != three_h {
            crate::bail!(
                "{prefix}bias_hh_l{layer}: {} elements, expected {three_h}",
                b_hh.len(),
            );
        }
        Ok(GruCellWeights {
            w_ih: Arc::<[f32]>::from(w_ih),
            w_hh: Arc::<[f32]>::from(w_hh),
            b_ih: Arc::<[f32]>::from(b_ih),
            b_hh: Arc::<[f32]>::from(b_hh),
            input_dim,
            hidden_dim,
        })
    }
}

impl GruStack {
    /// Load a stacked GRU from a HuggingFace safetensors
    /// checkpoint. `layer_dims` is `[(input_dim, hidden_dim);
    /// num_layers]` and the on-disk keys are
    /// `{prefix}weight_ih_l{idx}` etc. for each layer's index.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        layer_dims: &[(usize, usize)],
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(layer_dims.len());
        for (idx, &(in_d, h_d)) in layer_dims.iter().enumerate() {
            layers.push(GruCellWeights::load_from_mmapped(
                st, prefix, idx, in_d, h_d,
            )?);
        }
        Ok(GruStack { layers })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    /// Reference Rust implementation of a single GRU layer with
    /// an explicit initial hidden state. Mirrors PyTorch's
    /// `nn.GRU` per-step formulae.
    fn gru_layer_reference(
        x: &[f32], h0: &[f32], b: usize, t: usize, d_in: usize, d_h: usize,
        w_ih: &[f32], w_hh: &[f32], b_ih: &[f32], b_hh: &[f32],
    ) -> Vec<f32> {
        let three_h = 3 * d_h;
        assert_eq!(x.len(), b * t * d_in);
        assert_eq!(h0.len(), b * d_h);
        assert_eq!(w_ih.len(), three_h * d_in);
        assert_eq!(w_hh.len(), three_h * d_h);
        assert_eq!(b_ih.len(), three_h);
        assert_eq!(b_hh.len(), three_h);

        let mut out = vec![0.0_f32; b * t * d_h];
        let mut h_prev = h0.to_vec();
        let mut gates_ih = vec![0.0_f32; b * three_h];
        let mut gates_hh = vec![0.0_f32; b * three_h];
        for step in 0..t {
            for bi in 0..b {
                for k in 0..three_h {
                    let mut gi = b_ih[k];
                    for j in 0..d_in {
                        gi += x[(bi * t + step) * d_in + j] * w_ih[k * d_in + j];
                    }
                    gates_ih[bi * three_h + k] = gi;

                    let mut gh = b_hh[k];
                    for j in 0..d_h {
                        gh += h_prev[bi * d_h + j] * w_hh[k * d_h + j];
                    }
                    gates_hh[bi * three_h + k] = gh;
                }
            }
            let mut h_new = vec![0.0_f32; b * d_h];
            for bi in 0..b {
                for k in 0..d_h {
                    let r = sigmoid(
                        gates_ih[bi * three_h + k]
                            + gates_hh[bi * three_h + k],
                    );
                    let z = sigmoid(
                        gates_ih[bi * three_h + d_h + k]
                            + gates_hh[bi * three_h + d_h + k],
                    );
                    let n = tanh(
                        gates_ih[bi * three_h + 2 * d_h + k]
                            + r * gates_hh[bi * three_h + 2 * d_h + k],
                    );
                    let h_t = (1.0 - z) * n + z * h_prev[bi * d_h + k];
                    h_new[bi * d_h + k] = h_t;
                }
            }
            for bi in 0..b {
                for k in 0..d_h {
                    out[(bi * t + step) * d_h + k] = h_new[bi * d_h + k];
                }
            }
            h_prev = h_new;
        }
        out
    }

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    fn tanh(x: f32) -> f32 { x.tanh() }

    #[test]
    fn single_layer_matches_reference() {
        let b = 1; let t = 4; let d_in = 2; let d_h = 3;
        let three_h = 3 * d_h;
        let x_data: Vec<f32> = (0..(b * t * d_in))
            .map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w_ih: Vec<f32> = (0..(three_h * d_in))
            .map(|i| (i as f32) * 0.05 - 0.3).collect();
        let w_hh: Vec<f32> = (0..(three_h * d_h))
            .map(|i| (i as f32) * 0.07 - 0.2).collect();
        let b_ih: Vec<f32> = (0..three_h).map(|i| (i as f32) * 0.02).collect();
        let b_hh: Vec<f32> = (0..three_h).map(|i| (i as f32) * 0.03 - 0.1).collect();
        let h0 = vec![0.0_f32; b * d_h];

        let expected = gru_layer_reference(
            &x_data, &h0, b, t, d_in, d_h, &w_ih, &w_hh, &b_ih, &b_hh,
        );

        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[b, t, d_in]), &Device::cpu(),
        );
        let stack = GruStack {
            layers: vec![GruCellWeights {
                w_ih: Arc::<[f32]>::from(w_ih),
                w_hh: Arc::<[f32]>::from(w_hh),
                b_ih: Arc::<[f32]>::from(b_ih),
                b_hh: Arc::<[f32]>::from(b_hh),
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
                "gru[{i}] expected {e}, got {a}");
        }
    }

    #[test]
    fn two_layer_stack_chain_is_correct() {
        // Layer 1: d_in=3 → d_h=4. Layer 2: d_in=4 → d_h=4.
        let b = 1; let t = 3;
        let d_in1 = 3; let d_h1 = 4;
        let d_in2 = d_h1; let d_h2 = 4;
        let x_data: Vec<f32> = (0..(b * t * d_in1))
            .map(|i| (i as f32) * 0.1).collect();
        let w_ih1: Vec<f32> = (0..(3 * d_h1 * d_in1))
            .map(|i| (i as f32) * 0.03 - 0.4).collect();
        let w_hh1: Vec<f32> = (0..(3 * d_h1 * d_h1))
            .map(|i| (i as f32) * 0.04 - 0.3).collect();
        let b_ih1: Vec<f32> = (0..(3 * d_h1)).map(|i| (i as f32) * 0.01).collect();
        let b_hh1: Vec<f32> = (0..(3 * d_h1))
            .map(|i| (i as f32) * 0.015 - 0.05).collect();
        let w_ih2: Vec<f32> = (0..(3 * d_h2 * d_in2))
            .map(|i| (i as f32) * 0.025 - 0.2).collect();
        let w_hh2: Vec<f32> = (0..(3 * d_h2 * d_h2))
            .map(|i| (i as f32) * 0.035 - 0.25).collect();
        let b_ih2: Vec<f32> = (0..(3 * d_h2)).map(|i| (i as f32) * 0.005).collect();
        let b_hh2: Vec<f32> = (0..(3 * d_h2))
            .map(|i| (i as f32) * 0.008 - 0.03).collect();

        let h0_l1 = vec![0.0_f32; b * d_h1];
        let h0_l2 = vec![0.0_f32; b * d_h2];
        let after_l1 = gru_layer_reference(
            &x_data, &h0_l1, b, t, d_in1, d_h1, &w_ih1, &w_hh1, &b_ih1, &b_hh1,
        );
        let expected = gru_layer_reference(
            &after_l1, &h0_l2, b, t, d_in2, d_h2, &w_ih2, &w_hh2, &b_ih2, &b_hh2,
        );

        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[b, t, d_in1]), &Device::cpu(),
        );
        let stack = GruStack {
            layers: vec![
                GruCellWeights {
                    w_ih: Arc::<[f32]>::from(w_ih1),
                    w_hh: Arc::<[f32]>::from(w_hh1),
                    b_ih: Arc::<[f32]>::from(b_ih1),
                    b_hh: Arc::<[f32]>::from(b_hh1),
                    input_dim: d_in1, hidden_dim: d_h1,
                },
                GruCellWeights {
                    w_ih: Arc::<[f32]>::from(w_ih2),
                    w_hh: Arc::<[f32]>::from(w_hh2),
                    b_ih: Arc::<[f32]>::from(b_ih2),
                    b_hh: Arc::<[f32]>::from(b_hh2),
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

    /// With z-gate forced to all-ones (so `(1 - z) = 0` and the
    /// new candidate is ignored), the hidden state must
    /// pass through unchanged from `h0`. We arrange this by
    /// setting `b_iz + b_hz` to a large positive constant and
    /// every other weight/bias to zero.
    #[test]
    fn forward_with_initial_state_z_one_passthrough() {
        let b = 1; let t = 3; let d = 2;
        let three_d = 3 * d;
        let x_data: Vec<f32> = vec![0.5_f32, -0.25, 1.0, -1.0, 0.3, 0.7];

        // All zero weights.
        let w_ih: Vec<f32> = vec![0.0_f32; three_d * d];
        let w_hh: Vec<f32> = vec![0.0_f32; three_d * d];
        // Bias: only the z-slice (positions [d .. 2*d]) gets a
        // large positive value so σ(z) ≈ 1 and 1 − z ≈ 0.
        let mut b_ih: Vec<f32> = vec![0.0_f32; three_d];
        for k in 0..d { b_ih[d + k] = 50.0; }
        let b_hh: Vec<f32> = vec![0.0_f32; three_d];

        let h0_data: Vec<f32> = vec![0.7_f32, -0.2];
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[b, t, d]), &Device::cpu(),
        );
        let h0 = x.const_f32_like(
            Arc::<[f32]>::from(h0_data.clone()),
            Shape::from_dims(&[1, b, d]),
        );
        let stack = GruStack {
            layers: vec![GruCellWeights {
                w_ih: Arc::<[f32]>::from(w_ih),
                w_hh: Arc::<[f32]>::from(w_hh),
                b_ih: Arc::<[f32]>::from(b_ih),
                b_hh: Arc::<[f32]>::from(b_hh),
                input_dim: d, hidden_dim: d,
            }],
        };
        let out = stack
            .forward_with_initial_state(&x, &h0)
            .unwrap()
            .realize_f32();
        // Every time-step output should equal h0 (because z ≈ 1).
        assert_eq!(out.len(), b * t * d);
        for step in 0..t {
            for k in 0..d {
                let got = out[step * d + k];
                let want = h0_data[k];
                assert!((got - want).abs() < 1e-5,
                    "passthrough[t={step}, k={k}] expected {want}, got {got}");
            }
        }
    }

    /// Sanity: outputs change when the input changes.
    #[test]
    fn responds_to_input() {
        let b = 1; let t = 4; let d_in = 4; let d_h = 4;
        let three_h = 3 * d_h;
        let w_ih: Vec<f32> = (0..(three_h * d_in))
            .map(|i| (i as f32) * 0.02).collect();
        let w_hh: Vec<f32> = (0..(three_h * d_h))
            .map(|i| (i as f32) * 0.02 - 0.05).collect();
        let b_ih: Vec<f32> = vec![0.0_f32; three_h];
        let b_hh: Vec<f32> = vec![0.0_f32; three_h];
        let stack = GruStack {
            layers: vec![GruCellWeights {
                w_ih: Arc::<[f32]>::from(w_ih),
                w_hh: Arc::<[f32]>::from(w_hh),
                b_ih: Arc::<[f32]>::from(b_ih),
                b_hh: Arc::<[f32]>::from(b_hh),
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
            "GRU must respond to input changes, max_diff = {max_diff}");
    }
}
