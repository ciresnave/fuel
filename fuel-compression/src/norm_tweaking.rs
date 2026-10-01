// SPDX-License-Identifier: MIT OR Apache-2.0
//! Norm Tweaking: post-quantization LayerNorm/RMSNorm calibration.
//!
//! Ported from `lightbulb/src/quantization/norm_tweaking.rs`
//! (candlelight-based) as part of the Candle-to-Fuel migration
//! (lightbulb #106). Named for Norm Tweaking (Li et al., AAAI 2024), a
//! universal plugin reported to recover 1.5-3% accuracy on quantized models
//! by recalibrating normalization-layer parameters.
//!
//! **This is an analytical approximation, not the paper's algorithm** — and
//! that was already true of the source this was ported from, not something
//! introduced by the port. The paper's actual method is iterative: run
//! calibration data through both the unquantized and quantized models,
//! and gradient-descend the gamma/beta parameters to minimize a divergence
//! loss between the two activation distributions. What's implemented here
//! (and in the lightbulb source) is a **closed-form approximation**: given only
//! the pre- and post-quantization mean/std of a layer's activations, scale
//! gamma by the std ratio and shift beta to match the mean — a single-step
//! moment-matching correction, not an optimization loop. The ported
//! source's own `NormTweakingConfig` still carries `learning_rate` and
//! `num_steps` fields (iterative-optimizer vocabulary) and `calibrate`
//! still takes a `calibration_data: &Tensor` parameter, but NONE of these
//! are read by the actual algorithm — [`NormTweaker::compute_adjustment`]
//! is pure closed-form arithmetic over [`LayerStats`], never a forward
//! pass. This port keeps the real algorithm and drops the dead
//! vocabulary rather than reproducing it (see [`NormTweakingConfig`]'s own
//! doc for what survived and what didn't).
//!
//! This is the SAME pattern already documented elsewhere in this
//! portfolio for a different ported artifact: MLMF's `calibrate()`
//! likewise binds a `calibration_data` parameter it never reads,
//! iterating weights instead of activations (`fuel/ROADMAP.md`, the
//! imatrix-generation backlog entry). Two independently-ported pieces of
//! "calibration" code from two different source projects both turned out
//! to be weight/moment statistics wearing an activation-calibration API —
//! worth knowing if a third one shows up.

use fuel_core::Device;
use fuel_core::lazy::Tensor;
use fuel_ir::Result;

/// Configuration for Norm Tweaking calibration.
///
/// The source this was ported from (`lightbulb/src/quantization/
/// norm_tweaking.rs`) also carried `learning_rate: f64` and
/// `num_steps: usize` fields, named after an iterative gradient-descent
/// calibration the paper describes — but its own `compute_adjustment` is a
/// closed-form calculation that never reads either field. Dropped here
/// rather than ported as dead configuration; see the module doc for the
/// full picture. If a real iterative calibrator is ever built on top of
/// this, those fields belong on ITS config, not on this analytical one's.
#[derive(Debug, Clone)]
pub struct NormTweakingConfig {
    /// Number of calibration samples used to collect [`LayerStats`]
    /// upstream of this crate (not read by [`NormTweaker`] itself — stats
    /// collection is the caller's responsibility via
    /// [`NormTweaker::collect_statistics`]).
    pub calibration_samples: usize,

    /// Target layer names (e.g. `["input_layernorm",
    /// "post_attention_layernorm"]`). If empty, calibrate every layer
    /// [`NormTweaker::calibrate`] is given statistics for.
    pub target_layers: Vec<String>,
}

impl Default for NormTweakingConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 50,
            target_layers: vec![],
        }
    }
}

/// Pre/post-quantization activation statistics for one normalization layer,
/// plus its current gamma/beta — the input [`NormTweaker::compute_adjustment`]
/// needs to compute a correction.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer name.
    pub name: String,

    /// Mean activation value, pre-quantization.
    pub pre_quant_mean: f64,

    /// Standard deviation, pre-quantization.
    pub pre_quant_std: f64,

    /// Mean activation value, post-quantization.
    pub post_quant_mean: f64,

    /// Standard deviation, post-quantization.
    pub post_quant_std: f64,

    /// Current gamma (scale) parameter.
    pub gamma: Vec<f32>,

    /// Current beta (shift) parameter.
    pub beta: Vec<f32>,
}

impl LayerStats {
    /// Distribution-shift magnitude: `|Δmean| + |Δstd|`. Larger means the
    /// layer's activation distribution moved further under quantization,
    /// and is a candidate for prioritizing which layers to calibrate first.
    pub fn shift_magnitude(&self) -> f64 {
        let mean_shift = (self.post_quant_mean - self.pre_quant_mean).abs();
        let std_shift = (self.post_quant_std - self.pre_quant_std).abs();
        mean_shift + std_shift
    }
}

/// Adjusted gamma/beta for one normalization layer, computed by
/// [`NormTweaker::compute_adjustment`].
#[derive(Debug, Clone)]
pub struct LayerAdjustments {
    pub layer_name: String,

    /// Multiplicative scale applied to every element of `gamma`.
    pub gamma_scale: f64,

    /// Additive shift applied to every element of `beta`.
    pub beta_shift: f64,

    /// `gamma` after applying `gamma_scale`.
    pub adjusted_gamma: Vec<f32>,

    /// `beta` after applying `beta_shift`.
    pub adjusted_beta: Vec<f32>,
}

/// Norm Tweaking calibrator. See the module doc for what algorithm this
/// actually runs (a closed-form moment-matching correction, not the
/// paper's iterative gradient descent).
pub struct NormTweaker {
    config: NormTweakingConfig,
    device: Device,
}

impl NormTweaker {
    pub fn new(config: NormTweakingConfig, device: Device) -> Self {
        Self { config, device }
    }

    /// The device statistics are collected on — exposed so a caller
    /// building `Tensor`s to pass to [`Self::collect_statistics`] can
    /// reuse it rather than threading a second copy through.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Compute gamma/beta adjustments for every layer in `layer_stats`
    /// whose name matches [`NormTweakingConfig::target_layers`] (or every
    /// layer, if that list is empty).
    pub fn calibrate(&self, layer_stats: &[LayerStats]) -> Result<Vec<LayerAdjustments>> {
        let mut adjustments = Vec::new();
        for stats in layer_stats {
            if !self.config.target_layers.is_empty()
                && !self
                    .config
                    .target_layers
                    .iter()
                    .any(|t| stats.name.contains(t))
            {
                continue;
            }
            adjustments.push(self.compute_adjustment(stats));
        }
        Ok(adjustments)
    }

    /// Closed-form gamma/beta adjustment for a single layer: scale gamma
    /// by the pre/post standard-deviation ratio (compensating for the
    /// variance change quantization introduced), then shift beta so the
    /// scaled mean matches the pre-quantization mean.
    ///
    /// `post_quant_std` near zero (a degenerate/collapsed distribution)
    /// leaves gamma unscaled (`gamma_scale = 1.0`) rather than dividing by
    /// a near-zero denominator — a typed no-op, never a panic or an `inf`.
    fn compute_adjustment(&self, stats: &LayerStats) -> LayerAdjustments {
        let gamma_scale = if stats.post_quant_std > 1e-6 {
            stats.pre_quant_std / stats.post_quant_std
        } else {
            1.0
        };
        let beta_shift = stats.pre_quant_mean - (stats.post_quant_mean * gamma_scale);

        let adjusted_gamma: Vec<f32> = stats
            .gamma
            .iter()
            .map(|&g| (g as f64 * gamma_scale) as f32)
            .collect();
        let adjusted_beta: Vec<f32> = stats
            .beta
            .iter()
            .map(|&b| (b as f64 + beta_shift) as f32)
            .collect();

        LayerAdjustments {
            layer_name: stats.name.clone(),
            gamma_scale,
            beta_shift,
            adjusted_gamma,
            adjusted_beta,
        }
    }

    /// Collect activation statistics for one normalization layer from a
    /// real forward pass. `activations` is `[batch, seq_len, hidden_size]`
    /// (or any shape where dim 0 is the axis to average the per-position
    /// variance over — matching the source this was ported from).
    ///
    /// Population variance (`correction = 0.0`, KISS-Ops §6.13's default)
    /// — not Bessel's correction — since this is describing the
    /// distribution of the SAMPLES ACTUALLY OBSERVED in this calibration
    /// batch, not estimating a wider population's variance from a sample
    /// of it.
    ///
    /// Returns [`LayerStats`] with `post_quant_mean`/`post_quant_std` at
    /// `0.0` — fill those by calling this again on the quantized model's
    /// activations for the same layer and overwriting them, or construct
    /// [`LayerStats`] directly if collecting both halves in one pass.
    pub fn collect_statistics(
        &self,
        activations: &Tensor,
        layer_name: &str,
        gamma: &[f32],
        beta: &[f32],
    ) -> Result<LayerStats> {
        let mean = activations.mean_all().realize_f32()[0] as f64;
        let variance = activations.var(0, 0.0)?.mean_all().realize_f32()[0] as f64;
        let std = variance.sqrt();

        Ok(LayerStats {
            name: layer_name.to_string(),
            pre_quant_mean: mean,
            pre_quant_std: std,
            post_quant_mean: 0.0,
            post_quant_std: 0.0,
            gamma: gamma.to_vec(),
            beta: beta.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_tweaking_config_default() {
        let config = NormTweakingConfig::default();
        assert_eq!(config.calibration_samples, 50);
        assert!(config.target_layers.is_empty());
    }

    #[test]
    fn layer_stats_shift_magnitude() {
        let stats = LayerStats {
            name: "layer.0".to_string(),
            pre_quant_mean: 0.0,
            pre_quant_std: 1.0,
            post_quant_mean: 0.1,
            post_quant_std: 0.9,
            gamma: vec![1.0; 128],
            beta: vec![0.0; 128],
        };
        let shift = stats.shift_magnitude();
        assert!(
            (shift - 0.2).abs() < 1e-6,
            "|0.1-0| + |0.9-1.0| = 0.2, got {shift}"
        );
    }

    #[test]
    fn compute_adjustment_scales_gamma_and_shifts_beta() {
        let config = NormTweakingConfig::default();
        let tweaker = NormTweaker::new(config, Device::cpu());

        let stats = LayerStats {
            name: "layer.0".to_string(),
            pre_quant_mean: 0.0,
            pre_quant_std: 1.0,
            post_quant_mean: 0.1,
            post_quant_std: 0.8,
            gamma: vec![1.0; 4],
            beta: vec![0.0; 4],
        };
        let adjustment = tweaker.compute_adjustment(&stats);

        assert!(
            (adjustment.gamma_scale - 1.25).abs() < 0.01,
            "1.0 / 0.8 = 1.25, got {}",
            adjustment.gamma_scale
        );
        // target = 0.0, post_quant = 0.1 * 1.25 = 0.125, shift = -0.125
        assert!(
            (adjustment.beta_shift - (-0.125)).abs() < 0.01,
            "expected -0.125, got {}",
            adjustment.beta_shift
        );
    }

    /// Degenerate case: a collapsed (near-zero-variance) post-quant
    /// distribution must not divide by ~zero. `gamma_scale` stays 1.0
    /// (unscaled) rather than exploding.
    #[test]
    fn compute_adjustment_degenerate_post_quant_std_does_not_explode() {
        let config = NormTweakingConfig::default();
        let tweaker = NormTweaker::new(config, Device::cpu());
        let stats = LayerStats {
            name: "layer.0".to_string(),
            pre_quant_mean: 1.0,
            pre_quant_std: 1.0,
            post_quant_mean: 0.5,
            post_quant_std: 0.0,
            gamma: vec![2.0],
            beta: vec![0.0],
        };
        let adjustment = tweaker.compute_adjustment(&stats);
        assert_eq!(
            adjustment.gamma_scale, 1.0,
            "near-zero post_quant_std must not scale gamma"
        );
        assert!(adjustment.gamma_scale.is_finite());
        assert!(adjustment.beta_shift.is_finite());
    }

    #[test]
    fn calibrate_filters_by_target_layers() {
        let config = NormTweakingConfig {
            calibration_samples: 50,
            target_layers: vec!["attn".to_string()],
        };
        let tweaker = NormTweaker::new(config, Device::cpu());
        let make = |name: &str| LayerStats {
            name: name.to_string(),
            pre_quant_mean: 0.0,
            pre_quant_std: 1.0,
            post_quant_mean: 0.0,
            post_quant_std: 1.0,
            gamma: vec![1.0],
            beta: vec![0.0],
        };
        let stats = vec![
            make("layer.0.attn_norm"),
            make("layer.0.mlp_norm"),
            make("layer.1.attn_norm"),
        ];
        let adjustments = tweaker.calibrate(&stats).unwrap();
        assert_eq!(
            adjustments.len(),
            2,
            "only the 2 'attn' layers must pass the filter"
        );
        assert!(adjustments.iter().all(|a| a.layer_name.contains("attn")));
    }

    /// KISS-Ops §6.13: population variance (correction=0.0), matching the
    /// invariant pinned for `Tensor::var` itself (fuel-core's own
    /// `var_population_with_correction_0` test) — collect_statistics must
    /// use the SAME convention, not silently default to Bessel's.
    #[test]
    fn collect_statistics_uses_population_variance() {
        let device = Device::cpu();
        let config = NormTweakingConfig::default();
        let tweaker = NormTweaker::new(config, device.clone());

        // [[1,2,3],[4,5,6]], dim 0 reduced: per-column dev around the
        // column mean (2.5,3.5,4.5) is ±1.5 each -> var = 2.25 per column
        // (population: /2, not Bessel's /1) -> mean of [2.25,2.25,2.25] = 2.25.
        let data =
            Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device).unwrap();
        let stats = tweaker
            .collect_statistics(&data, "test_layer", &[1.0], &[0.0])
            .unwrap();
        let want_var = 2.25_f64;
        let want_std = want_var.sqrt();
        assert!(
            (stats.pre_quant_std - want_std).abs() < 1e-4,
            "population std = sqrt(2.25) = {want_std}, got {}",
            stats.pre_quant_std
        );
    }
}
