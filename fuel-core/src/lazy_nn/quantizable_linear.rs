//! Lazy `QuantizableLinear` — a `Linear` layer that transparently
//! accepts F32, BF16, or Q4_0 base weight storages.
//!
//! Mirrors the eager `fuel-nn::QuantizableLinear` API surface so
//! lazy ports written against either weight format can dispatch
//! through a single concrete type. The forward pass is the same
//! shape as [`super::LazyLinear`] — the only difference is that
//! this type doesn't refuse `WeightStorage::Q4_0` at construction
//! time. Per the eager-side convention, this is kept as a separate
//! type rather than collapsing into `LazyLinear` to make port
//! intent visible at the model-struct level (i.e. "this layer is
//! intentionally checkpoint-format-polymorphic").

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_nn::LazyModule;
use fuel_core_types::Shape;
use std::sync::Arc;

/// Linear layer over `LazyTensor` whose weight may be F32, BF16,
/// or Q4_0. `LazyLoraLinear` covers the LoRA case; this is the
/// non-LoRA polymorphic surface.
#[derive(Debug, Clone)]
pub struct LazyQuantizableLinear {
    weight:       WeightStorage,
    bias:         Option<Arc<[f32]>>,
    in_features:  usize,
    out_features: usize,
}

impl LazyQuantizableLinear {
    /// Build a quantization-polymorphic linear layer.
    ///
    /// `weight` must be a non-LoRA [`WeightStorage`] variant
    /// (`F32`, `BF16`, or `Q4_0`) in `[in_features, out_features]`
    /// layout, and its `elem_count` must equal `in_features * out_features`.
    /// `bias`, when present, must have length `out_features`.
    pub fn new(
        weight: WeightStorage,
        bias: Option<Arc<[f32]>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if matches!(weight, WeightStorage::WithLoRA { .. }) {
            return Err(crate::Error::Msg(
                "LazyQuantizableLinear::new: WithLoRA must be \
                 wrapped in LazyLoraLinear, not LazyQuantizableLinear"
                .into(),
            ).bt());
        }
        if weight.elem_count() != in_features * out_features {
            return Err(crate::Error::Msg(format!(
                "LazyQuantizableLinear::new: weight has {} elements but \
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
                    "LazyQuantizableLinear::new: bias has length {} but \
                     out_features = {}",
                    b.len(), out_features,
                )).bt());
            }
        }
        Ok(Self { weight, bias, in_features, out_features })
    }

    /// Convenience constructor for a bias-less quantizable linear layer.
    pub fn new_no_bias(
        weight: WeightStorage,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        Self::new(weight, None, in_features, out_features)
    }

    /// Returns `true` if the underlying weight is Q4_0.
    pub fn is_quantized(&self) -> bool {
        matches!(self.weight, WeightStorage::Q4_0 { .. })
    }

    /// Reference to the underlying weight storage.
    pub fn weight(&self) -> &WeightStorage {
        &self.weight
    }

    /// Bias buffer, if present.
    pub fn bias(&self) -> Option<&Arc<[f32]>> {
        self.bias.as_ref()
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

impl LazyModule for LazyQuantizableLinear {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use crate::lazy_nn::LazyLinear;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    /// Quantize `[in, out]` F32 to a `WeightStorage::Q4_0` using the
    /// same path the shipped ports use (transpose to `[out, in]`,
    /// per-row Q4_0 block-encode, reinterpret as u32 words).
    fn quantize_in_out_to_q4_0(
        f32_in_out: &[f32], in_features: usize, out_features: usize,
    ) -> WeightStorage {
        use fuel_quantized::{BlockQ4_0, GgmlType};
        const QK4_0: usize = 32;
        assert_eq!(in_features % QK4_0, 0);
        let mut f32_out_in = vec![0.0_f32; out_features * in_features];
        for o in 0..out_features {
            for j in 0..in_features {
                f32_out_in[o * in_features + j] = f32_in_out[j * out_features + o];
            }
        }
        let n_blocks = out_features * in_features / QK4_0;
        let mut blocks: Vec<BlockQ4_0> = vec![BlockQ4_0::zeros(); n_blocks];
        BlockQ4_0::from_float(&f32_out_in, &mut blocks);
        let bytes_len = n_blocks * std::mem::size_of::<BlockQ4_0>();
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, bytes_len)
        };
        let padded_len = bytes_len.div_ceil(4) * 4;
        let mut padded = vec![0_u8; padded_len];
        padded[..bytes_len].copy_from_slice(byte_slice);
        let words: Vec<u32> = padded.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        WeightStorage::Q4_0 {
            words: Arc::from(words),
            bytes_len,
            in_features,
            out_features,
        }
    }

    #[test]
    fn quantizable_linear_with_f32_weight_matches_lazy_linear() {
        let in_features = 6;
        let out_features = 4;
        let seq = 3;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.04, -0.1);
        let bias: Vec<f32> = ramp_f32(out_features, 0.15, -0.2);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.05, -0.3);

        let qlin = LazyQuantizableLinear::new(
            WeightStorage::F32(Arc::from(w.clone())),
            Some(Arc::from(bias.clone())),
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
        let got = qlin.forward(&x).unwrap().realize_f32();
        let x2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let expected = plain.forward(&x2).unwrap().realize_f32();
        assert_eq!(got.len(), expected.len());
        for (i, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "qlin f32 [{i}] expected {e}, got {a}",
            );
        }
    }

    #[test]
    fn quantizable_linear_with_q4_0_weight_runs_and_finite() {
        // in_features must be a multiple of QK4_0 = 32.
        let in_features = 32;
        let out_features = 4;
        let seq = 2;

        let w: Vec<f32> = ramp_f32(in_features * out_features, 0.01, -0.05);
        let bias: Vec<f32> = ramp_f32(out_features, 0.1, 0.0);
        let x_data: Vec<f32> = ramp_f32(seq * in_features, 0.005, -0.1);

        let q_weight = quantize_in_out_to_q4_0(&w, in_features, out_features);
        let qlin = LazyQuantizableLinear::new(
            q_weight,
            Some(Arc::from(bias)),
            in_features,
            out_features,
        ).unwrap();
        assert!(qlin.is_quantized());

        let x = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[seq, in_features]), &Device::cpu(),
        );
        let y = qlin.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[seq, out_features]);
        let got = y.realize_f32();
        assert_eq!(got.len(), seq * out_features);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "qlin q4_0 out[{i}] = {v} not finite");
        }
    }
}
