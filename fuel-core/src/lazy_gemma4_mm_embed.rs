//! Gemma4 multimodal embedder — lazy port.
//!
//! Projects encoder features (from the vision tower or audio tower)
//! into the language model's embedding space. Two steps:
//!
//!   1. **Pre-projection RMS norm** with NO learnable scale —
//!      `x / sqrt(mean(x²) + eps)`. Just the last-dim RMS divider
//!      without a gain multiplication.
//!   2. **Linear projection** (no bias) from `multimodal_hidden_size`
//!      to `text_hidden_size`.
//!
//! Used by [`Model`] in the eager `gemma4::Model` to project both
//! the vision tower and (optionally) the audio tower outputs into
//! the Gemma4 text decoder's embedding space before concatenating
//! them with text token embeddings.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4MmEmbedConfig {
    pub multimodal_hidden_size: usize,
    pub text_hidden_size: usize,
    pub eps: f64,
}

#[derive(Debug, Clone)]
pub struct Gemma4MmEmbedWeights {
    /// `[multimodal_hidden_size, text_hidden_size]` (no bias).
    pub projection: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Gemma4MmEmbedder {
    pub config: Gemma4MmEmbedConfig,
    pub weights: Gemma4MmEmbedWeights,
}

impl Gemma4MmEmbedder {
    /// Normalise then project soft encoder features into text
    /// embedding space.
    ///
    /// `soft_features` shape: `(..., multimodal_hidden_size)`.
    /// Returns shape `(..., text_hidden_size)`.
    pub fn forward(&self, soft_features: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = soft_features.shape();
        let dims = dims.dims();
        assert!(
            !dims.is_empty() && *dims.last().unwrap() == cfg.multimodal_hidden_size,
            "Gemma4MmEmbed: last dim must equal multimodal_hidden_size={}, got shape {:?}",
            cfg.multimodal_hidden_size, dims,
        );

        // Step 1: RMS normalize over the last dim (no learnable gain).
        let normed = soft_features.rms_norm_last_dim(cfg.eps)?;
        // Step 2: Linear projection (no bias).
        Ok(self.weights.projection.apply_linear(
            &normed,
            cfg.multimodal_hidden_size,
            cfg.text_hidden_size,
        ))
    }
}

// ---- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_core_types::Shape;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.02
        }
    }

    #[test]
    fn mm_embed_forward_shape_and_finite() {
        let cfg = Gemma4MmEmbedConfig {
            multimodal_hidden_size: 8,
            text_hidden_size: 12,
            eps: 1e-6,
        };
        let mut next = rng(42);
        let proj: Vec<f32> = (0..cfg.multimodal_hidden_size * cfg.text_hidden_size)
            .map(|_| next()).collect();
        let model = Gemma4MmEmbedder {
            config: cfg.clone(),
            weights: Gemma4MmEmbedWeights {
                projection: WeightStorage::F32(Arc::from(proj)),
            },
        };
        // (1, seq, multimodal_hidden) input.
        let seq = 5;
        let input_data: Vec<f32> = (0..1 * seq * cfg.multimodal_hidden_size)
            .map(|i| ((i as f32) * 0.05) - 0.1).collect();
        let input = LazyTensor::from_f32(
            input_data,
            Shape::from_dims(&[1, seq, cfg.multimodal_hidden_size]),
            &Device::cpu(),
        );
        let out = model.forward(&input).unwrap();
        assert_eq!(out.shape().dims(), &[1, seq, cfg.text_hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite mm_embed output: {v}");
        }
    }

    #[test]
    fn mm_embed_rms_norm_makes_input_unit_rms() {
        let cfg = Gemma4MmEmbedConfig {
            multimodal_hidden_size: 4,
            text_hidden_size: 4,
            eps: 1e-12,
        };
        // Identity projection so we can read the post-norm value directly.
        let mut identity = vec![0.0_f32; 16];
        for i in 0..4 {
            identity[i * 4 + i] = 1.0;
        }
        let model = Gemma4MmEmbedder {
            config: cfg.clone(),
            weights: Gemma4MmEmbedWeights {
                projection: WeightStorage::F32(Arc::from(identity)),
            },
        };
        // Input: each row is (1, 2, 3, 4) — RMS = sqrt((1+4+9+16)/4) = sqrt(7.5).
        let input = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            Shape::from_dims(&[1, 1, 4]),
            &Device::cpu(),
        );
        let out = model.forward(&input).unwrap().realize_f32();
        let expected = [
            1.0_f32 / 7.5_f32.sqrt(),
            2.0      / 7.5_f32.sqrt(),
            3.0      / 7.5_f32.sqrt(),
            4.0      / 7.5_f32.sqrt(),
        ];
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5,
                "row[{i}]: got {got} expected {want}");
        }
    }
}
