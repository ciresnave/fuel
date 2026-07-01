//! Lazy `Linear` layer — `y = x @ W + b` over `LazyTensor`.
//!
//! Weight is held as a [`WeightStorage`] in `[in_features, out_features]`
//! layout (the layout [`WeightStorage::apply_linear`] expects). This
//! matches every shipped lazy port's convention and is the inverse of
//! eager `fuel-nn::Linear`, which stores `[out_features, in_features]`
//! and transposes inside `forward`.
//!
//! Bias, if present, is a `[out_features]` `Arc<[f32]>` materialized
//! fresh on the activation's graph at forward time and broadcast-added
//! across the leading dims of the projection.

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_nn::LazyModule;
use crate::lazy_nn_varbuilder::LazyVarBuilder;
use fuel_ir::Shape;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;

/// Linear (fully connected) layer over `LazyTensor`.
#[derive(Debug, Clone)]
pub struct LazyLinear {
    weight: WeightStorage,
    bias: Option<Arc<[f32]>>,
    in_features: usize,
    out_features: usize,
}

impl LazyLinear {
    /// Build a linear layer from a weight storage and optional bias.
    ///
    /// `weight` must already be laid out as `[in_features, out_features]`
    /// — the same convention every shipped lazy port uses. `bias`, when
    /// present, must have length `out_features`.
    pub fn new(
        weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if weight.elem_count() != in_features * out_features {
            return Err(crate::Error::Msg(format!(
                "LazyLinear::new: weight has {} elements but \
                 in_features * out_features = {} * {} = {}",
                weight.elem_count(),
                in_features,
                out_features,
                in_features * out_features,
            )).bt());
        }
        if let Some(b) = bias.as_ref() {
            if b.len() != out_features {
                return Err(crate::Error::Msg(format!(
                    "LazyLinear::new: bias has length {} but \
                     out_features = {}",
                    b.len(),
                    out_features,
                )).bt());
            }
        }
        Ok(Self { weight, bias, in_features, out_features })
    }

    /// Convenience constructor for a bias-less linear layer.
    pub fn new_no_bias(
        weight: WeightStorage,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        Self::new(weight, None, in_features, out_features)
    }

    /// Returns a reference to the weight storage.
    pub fn weight(&self) -> &WeightStorage {
        &self.weight
    }

    /// Returns the bias buffer, if present.
    pub fn bias(&self) -> Option<&Arc<[f32]>> {
        self.bias.as_ref()
    }

    /// In-features (last dim of the input expected by `forward`).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Out-features (last dim of the projection produced by `forward`).
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl LazyModule for LazyLinear {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let y = self.weight.apply_linear(xs, self.in_features, self.out_features);
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

// ============================================================================
// Free factories — `lazy_nn::linear(in, out, vs)` style constructors
// ============================================================================

/// Kaiming-like fan-in uniform sample of `n` f32 values in
/// `(-bound, +bound)` with `bound = 1 / sqrt(in_features)`. Matches
/// PyTorch's `nn.Linear` default init (weight and bias both ~
/// `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`) and is close enough to the
/// retired `fuel_nn::linear` recipe to keep small-fixture forward
/// outputs bounded.
///
/// Seeded deterministically from `in_features`, `n`, and `seed_salt`
/// so successive `linear()` calls in the same process produce stable
/// values across runs without forcing the caller to thread an RNG.
fn fan_in_kaiming_uniform(in_features: usize, n: usize, seed_salt: u64) -> Vec<f32> {
    use rand::Rng;
    let bound = 1.0_f32 / (in_features as f32).sqrt();
    let seed = (in_features as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((n as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(seed_salt);
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-bound..bound)).collect()
}

/// Free factory: build a [`LazyLinear`] with weight + bias registered
/// into `vs`'s underlying [`crate::lazy_nn_varmap::LazyVarMap`] under
/// the names `"<prefix>.weight"` and `"<prefix>.bias"`.
///
/// The weight is laid out `[in_features, out_features]` (the layout
/// [`crate::lazy::WeightStorage::apply_linear`] expects). Init follows
/// a Kaiming-fan-in uniform: `U(-1/sqrt(in_features), +1/sqrt(in_features))`,
/// approximating the retired `fuel_nn::linear` semantics.
pub fn linear(
    in_features: usize,
    out_features: usize,
    vs: &LazyVarBuilder,
) -> Result<LazyLinear> {
    let weight_var = vs.get_with(
        Shape::from_dims(&[in_features, out_features]),
        "weight",
        |s| fan_in_kaiming_uniform(in_features, s.elem_count(), 0),
    )?;
    // Bias gets the same Kaiming-fan-in uniform bound (this matches
    // PyTorch's `nn.Linear` default: bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))).
    let bias_var = vs.get_with(
        Shape::from_dims(&[out_features]),
        "bias",
        |s| fan_in_kaiming_uniform(in_features, s.elem_count(), 1),
    )?;
    let weight = WeightStorage::F32(Arc::from(weight_var.to_vec()));
    let bias: Arc<[f32]> = Arc::from(bias_var.to_vec());
    LazyLinear::new(weight, Some(bias), in_features, out_features)
}

/// Free factory: bias-less variant of [`linear`]. Only `"<prefix>.weight"`
/// is registered into the underlying [`crate::lazy_nn_varmap::LazyVarMap`].
pub fn linear_no_bias(
    in_features: usize,
    out_features: usize,
    vs: &LazyVarBuilder,
) -> Result<LazyLinear> {
    let weight_var = vs.get_with(
        Shape::from_dims(&[in_features, out_features]),
        "weight",
        |s| fan_in_kaiming_uniform(in_features, s.elem_count(), 0),
    )?;
    let weight = WeightStorage::F32(Arc::from(weight_var.to_vec()));
    LazyLinear::new_no_bias(weight, in_features, out_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    /// Reference `y = x @ W + bias` with W laid out `[in, out]`.
    fn ref_linear(
        x: &[f32], w: &[f32], bias: Option<&[f32]>,
        b_outer: usize, in_features: usize, out_features: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; b_outer * out_features];
        for bi in 0..b_outer {
            for o in 0..out_features {
                let mut acc = 0.0_f32;
                for k in 0..in_features {
                    acc += x[bi * in_features + k] * w[k * out_features + o];
                }
                if let Some(b) = bias {
                    acc += b[o];
                }
                out[bi * out_features + o] = acc;
            }
        }
        out
    }

    #[test]
    fn linear_forward_shape_and_finite() {
        let in_features = 4;
        let out_features = 3;
        let seq = 5;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.05, -0.2);
        let b: Vec<f32> = ramp_f32(out_features, 0.1, 0.0);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.03, -0.4);

        let layer = LazyLinear::new(
            WeightStorage::F32(Arc::from(w)),
            Some(Arc::from(b)),
            in_features,
            out_features,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, out_features]);
        let got = y.realize_f32();
        assert_eq!(got.len(), seq * out_features);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "linear out[{i}] = {v} not finite");
        }
    }

    #[test]
    fn linear_with_bias_matches_apply_linear_plus_broadcast_add_golden() {
        let in_features = 6;
        let out_features = 4;
        let seq = 3;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.02, 0.1);
        let bias: Vec<f32> = ramp_f32(out_features, 0.25, -0.5);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.07, -0.3);

        let expected = ref_linear(
            &x_data, &w, Some(&bias), seq, in_features, out_features,
        );

        let layer = LazyLinear::new(
            WeightStorage::F32(Arc::from(w)),
            Some(Arc::from(bias)),
            in_features,
            out_features,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, out_features]);
        let got = y.realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "linear[{i}] expected {e}, got {a}",
            );
        }
    }

    #[test]
    fn linear_no_bias_matches_apply_linear() {
        let in_features = 5;
        let out_features = 3;
        let seq = 4;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.03, -0.15);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.04, 0.2);

        let expected = ref_linear(
            &x_data, &w, None, seq, in_features, out_features,
        );

        let weight = WeightStorage::F32(Arc::from(w.clone()));
        let layer = LazyLinear::new_no_bias(
            weight.clone(), in_features, out_features,
        ).unwrap();
        let x = LazyTensor::from_f32(
            x_data.clone(),
            Shape::from_dims(&[seq, in_features]),
            &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, out_features]);
        let got = y.realize_f32();

        let x2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let direct = weight
            .apply_linear(&x2, in_features, out_features)
            .realize_f32();

        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "linear_no_bias[{i}] expected {e}, got {a}",
            );
        }
        for (i, (a, d)) in got.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - d).abs() < 1e-6,
                "linear_no_bias[{i}] forward {a} != apply_linear {d}",
            );
        }
    }

    #[test]
    fn factory_registers_weight_and_bias_and_forward_shape_matches() {
        use crate::DType;
        use crate::lazy_nn_varbuilder::LazyVarBuilder;
        use crate::lazy_nn_varmap::LazyVarMap;

        let in_features = 4;
        let out_features = 3;
        let seq = 5;

        let map = LazyVarMap::new();
        let vs = LazyVarBuilder::from_varmap(map.clone(), DType::F32, Device::cpu());

        let layer = super::linear(in_features, out_features, &vs.pp("proj")).unwrap();
        assert_eq!(layer.in_features(), in_features);
        assert_eq!(layer.out_features(), out_features);

        // Both parameters should be registered under the prefixed paths.
        assert!(map.get("proj.weight").is_some(), "weight not registered");
        assert!(map.get("proj.bias").is_some(), "bias not registered");
        assert_eq!(map.get("proj.weight").unwrap().shape().dims(), &[in_features, out_features]);
        assert_eq!(map.get("proj.bias").unwrap().shape().dims(), &[out_features]);

        // Forward gives the expected output shape on a small fixture.
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.05, -0.1);
        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, out_features]);
        let got = y.realize_f32();
        assert_eq!(got.len(), seq * out_features);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "factory linear out[{i}] = {v} not finite");
        }

        // `linear_no_bias` registers only weight.
        let map2 = LazyVarMap::new();
        let vs2 = LazyVarBuilder::from_varmap(map2.clone(), DType::F32, Device::cpu());
        let layer_nb = super::linear_no_bias(in_features, out_features, &vs2.pp("nb")).unwrap();
        assert!(layer_nb.bias().is_none());
        assert!(map2.get("nb.weight").is_some());
        assert!(map2.get("nb.bias").is_none());
    }
}
