//! Lazy LoRA-adapted linear layer.
//!
//! Wraps a frozen base [`WeightStorage`] (F32, BF16, or Q4_0) with a
//! trainable low-rank delta:
//!
//! ```text
//! y = base(x) + (alpha / rank) · x @ A @ B
//! ```
//!
//! where `A` has shape `[in_features, rank]` and `B` has shape
//! `[rank, out_features]` — both in the same `[in, out]` layout
//! convention every shipped lazy port uses. The whole graph is
//! constructed by [`WeightStorage::apply_linear`] on a
//! [`WeightStorage::WithLoRA`] variant; this module is the
//! high-level `LazyModule` wrapper that bundles the adapter
//! parameters with optional bias.

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_nn::LazyModule;
use fuel_core_types::Shape;
use std::sync::Arc;

/// LoRA-adapted linear layer over `LazyTensor`.
#[derive(Debug, Clone)]
pub struct LazyLoraLinear {
    base_weight:  WeightStorage,
    bias:         Option<Arc<[f32]>>,
    lora_a:       Arc<[f32]>,
    lora_b:       Arc<[f32]>,
    rank:         usize,
    alpha:        f32,
    in_features:  usize,
    out_features: usize,
}

impl LazyLoraLinear {
    /// Build a LoRA-adapted linear layer.
    ///
    /// - `base_weight` — frozen base in `[in_features, out_features]` layout.
    ///   Must not already be a [`WeightStorage::WithLoRA`].
    /// - `bias` — optional length-`out_features` bias, added after the
    ///   adapter contribution.
    /// - `lora_a` — `[in_features, rank]` adapter A.
    /// - `lora_b` — `[rank, out_features]` adapter B.
    /// - `alpha` — LoRA scaling factor; effective scale is `alpha / rank`.
    pub fn new(
        base_weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        lora_a: Arc<[f32]>,
        lora_b: Arc<[f32]>,
        rank: usize,
        alpha: f32,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if rank == 0 {
            return Err(crate::Error::Msg(
                "LazyLoraLinear::new: rank must be > 0".into(),
            ).bt());
        }
        if base_weight.elem_count() != in_features * out_features {
            return Err(crate::Error::Msg(format!(
                "LazyLoraLinear::new: base_weight has {} elements but \
                 in_features * out_features = {} * {} = {}",
                base_weight.elem_count(),
                in_features,
                out_features,
                in_features * out_features,
            )).bt());
        }
        if matches!(base_weight, WeightStorage::WithLoRA { .. }) {
            return Err(crate::Error::Msg(
                "LazyLoraLinear::new: base_weight is already WithLoRA \
                 (nested adapters unsupported)".into(),
            ).bt());
        }
        if lora_a.len() != in_features * rank {
            return Err(crate::Error::Msg(format!(
                "LazyLoraLinear::new: lora_a has {} elements but \
                 in_features * rank = {} * {} = {}",
                lora_a.len(), in_features, rank, in_features * rank,
            )).bt());
        }
        if lora_b.len() != rank * out_features {
            return Err(crate::Error::Msg(format!(
                "LazyLoraLinear::new: lora_b has {} elements but \
                 rank * out_features = {} * {} = {}",
                lora_b.len(), rank, out_features, rank * out_features,
            )).bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != out_features {
                return Err(crate::Error::Msg(format!(
                    "LazyLoraLinear::new: bias has length {} but \
                     out_features = {}",
                    b.len(), out_features,
                )).bt());
            }
        }
        Ok(Self {
            base_weight,
            bias,
            lora_a,
            lora_b,
            rank,
            alpha,
            in_features,
            out_features,
        })
    }

    /// Reference to the frozen base weight storage.
    pub fn base_weight(&self) -> &WeightStorage {
        &self.base_weight
    }

    /// Bias buffer, if present.
    pub fn bias(&self) -> Option<&Arc<[f32]>> {
        self.bias.as_ref()
    }

    /// `[in_features, rank]` adapter A.
    pub fn lora_a(&self) -> &Arc<[f32]> {
        &self.lora_a
    }

    /// `[rank, out_features]` adapter B.
    pub fn lora_b(&self) -> &Arc<[f32]> {
        &self.lora_b
    }

    /// LoRA rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// LoRA alpha; effective scale is `alpha / rank`.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// In-features.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Out-features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl LazyModule for LazyLoraLinear {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let merged = self.base_weight.clone().with_lora(
            Arc::clone(&self.lora_a),
            Arc::clone(&self.lora_b),
            self.rank,
            self.alpha,
            self.in_features,
            self.out_features,
        );
        let y = merged.apply_linear(xs, self.in_features, self.out_features);
        match &self.bias {
            Some(b) => {
                let bias_t = y.const_f32_like(
                    Arc::clone(b),
                    Shape::from_dims(&[self.out_features]),
                );
                y.broadcast_add(&bias_t)
            }
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use crate::lazy_nn::LazyLinear;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    #[test]
    fn lora_zero_alpha_matches_base_linear() {
        // With alpha=0 the LoRA delta vanishes and the forward must
        // equal a plain LazyLinear over the same base weight + bias.
        let in_features = 6;
        let out_features = 4;
        let rank = 3;
        let seq = 5;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.05, -0.1);
        let bias: Vec<f32> = ramp_f32(out_features, 0.2, -0.3);
        let lora_a: Vec<f32> = ramp_f32(in_features * rank, 0.07, 0.15);
        let lora_b: Vec<f32> = ramp_f32(rank * out_features, 0.04, -0.2);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.03, -0.25);

        let lora = LazyLoraLinear::new(
            WeightStorage::F32(Arc::from(w.clone())),
            Some(Arc::from(bias.clone())),
            Arc::from(lora_a),
            Arc::from(lora_b),
            rank,
            0.0,
            in_features,
            out_features,
        ).unwrap();
        let plain = LazyLinear::new(
            WeightStorage::F32(Arc::from(w)),
            Some(Arc::from(bias)),
            in_features,
            out_features,
        ).unwrap();

        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[seq, in_features]),
            &Device::cpu(),
        );
        let y_lora = lora.forward(&x).unwrap();
        let x2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let y_plain = plain.forward(&x2).unwrap();

        assert_eq!(y_lora.shape().dims(), &[seq, out_features]);
        let got = y_lora.realize_f32();
        let expected = y_plain.realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "lora alpha=0 [{i}] expected {e}, got {a}",
            );
        }
    }

    #[test]
    fn lora_unit_alpha_adds_lora_correction_to_base() {
        // alpha == rank -> scale == 1; output should equal
        // base @ x  +  x @ A @ B  +  bias.
        let in_features = 4;
        let out_features = 3;
        let rank = 2;
        let seq = 3;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.05, -0.2);
        let bias: Vec<f32> = ramp_f32(out_features, 0.1, 0.0);
        let lora_a: Vec<f32> = ramp_f32(in_features * rank, 0.08, -0.1);
        let lora_b: Vec<f32> = ramp_f32(rank * out_features, 0.06, 0.05);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.03, -0.4);

        // Hand reference: y[b,o] = bias[o] + sum_k x[b,k]*w[k,o]
        //                        + sum_r (sum_k x[b,k]*A[k,r]) * B[r,o]
        let mut expected = vec![0.0_f32; seq * out_features];
        for bi in 0..seq {
            for o in 0..out_features {
                let mut acc = bias[o];
                for k in 0..in_features {
                    acc += x_data[bi * in_features + k] * w[k * out_features + o];
                }
                for r in 0..rank {
                    let mut xa = 0.0_f32;
                    for k in 0..in_features {
                        xa += x_data[bi * in_features + k] * lora_a[k * rank + r];
                    }
                    acc += xa * lora_b[r * out_features + o];
                }
                expected[bi * out_features + o] = acc;
            }
        }

        let lora = LazyLoraLinear::new(
            WeightStorage::F32(Arc::from(w)),
            Some(Arc::from(bias)),
            Arc::from(lora_a),
            Arc::from(lora_b),
            rank,
            rank as f32, // alpha == rank => scale == 1
            in_features,
            out_features,
        ).unwrap();

        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let y = lora.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, out_features]);
        let got = y.realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "lora unit-alpha [{i}] expected {e}, got {a}",
            );
        }
    }
}
