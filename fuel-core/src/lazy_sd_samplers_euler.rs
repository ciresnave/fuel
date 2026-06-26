//! Euler-ancestral discrete Stable-Diffusion scheduler — lazy-graph port.
//!
//! Ports the eager `EulerAncestralDiscreteScheduler` from
//! `fuel-transformers/.../euler_ancestral_discrete.rs`. Like the other
//! schedulers in [`crate::lazy_sd_samplers`], all per-step coefficients
//! are precomputed host-side `f64` and folded into the graph via
//! [`LazyTensor::mul_scalar`] / [`LazyTensor::add`] — no new graph ops.
//!
//! Euler-ancestral is a noise-injecting stepper: each `step` adds a
//! noise term with deviation `sigma_up`. The caller supplies the noise
//! tensor so RNG ownership stays out of the scheduler — pass an
//! all-zero tensor for deterministic behaviour, or
//! [`LazyTensor::const_f32_like`] with samples from any RNG for the
//! ancestral path.

use crate::lazy::LazyTensor;
use crate::lazy_sd_samplers::{BetaSchedule, PredictionType, SdScheduler, TimestepSpacing};
use crate::{Error, Result};
use fuel_ir::Shape;

/// Configuration for [`EulerAncestralDiscreteScheduler`].
#[derive(Debug, Clone, Copy)]
pub struct EulerAncestralDiscreteSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub steps_offset: usize,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
    pub timestep_spacing: TimestepSpacing,
}

impl Default for EulerAncestralDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
            timestep_spacing: TimestepSpacing::Leading,
        }
    }
}

/// Euler-ancestral discrete scheduler — Katherine Crowson's
/// `k-diffusion` ancestral Euler step
/// (<https://github.com/crowsonkb/k-diffusion>).
#[derive(Debug, Clone)]
pub struct EulerAncestralDiscreteScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    pub config: EulerAncestralDiscreteSchedulerConfig,
}

impl EulerAncestralDiscreteScheduler {
    /// Builds a scheduler from `config` and immediately precomputes
    /// the sigma schedule for `inference_steps` reverse steps.
    pub fn new(
        inference_steps: usize,
        config: EulerAncestralDiscreteSchedulerConfig,
    ) -> Result<Self> {
        if inference_steps == 0 {
            return Err(Error::Msg(
                "EulerAncestralDiscreteScheduler: inference_steps must be > 0".into(),
            )
            .bt());
        }
        if config.train_timesteps == 0 {
            return Err(Error::Msg(
                "EulerAncestralDiscreteScheduler: train_timesteps must be > 0".into(),
            )
            .bt());
        }
        let mut s = Self {
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            init_noise_sigma: 0.0,
            config,
        };
        s.recompute_schedule(inference_steps)?;
        Ok(s)
    }

    fn recompute_schedule(&mut self, inference_steps: usize) -> Result<()> {
        let cfg = self.config;
        let step_ratio = cfg.train_timesteps / inference_steps;
        let timesteps: Vec<usize> = match cfg.timestep_spacing {
            TimestepSpacing::Leading => (0..inference_steps)
                .map(|s| s * step_ratio + cfg.steps_offset)
                .rev()
                .collect(),
            TimestepSpacing::Trailing => {
                std::iter::successors(Some(cfg.train_timesteps), |n| {
                    if *n > step_ratio { Some(n - step_ratio) } else { None }
                })
                .map(|n| n - 1)
                .collect()
            }
            TimestepSpacing::Linspace => linspace(0.0, (cfg.train_timesteps - 1) as f64, inference_steps)
                .into_iter()
                .map(|f| f as usize)
                .rev()
                .collect(),
        };

        let betas = build_betas(cfg.beta_schedule, cfg.beta_start, cfg.beta_end, cfg.train_timesteps);
        let alphas_cumprod = build_alphas_cumprod(&betas);
        let train_sigmas: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&f| ((1.0 - f) / f).sqrt())
            .collect();

        let xa: Vec<f64> = (0..train_sigmas.len()).map(|i| i as f64).collect();
        let mut sigmas = interp(
            &timesteps.iter().map(|&t| t as f64).collect::<Vec<_>>(),
            &xa,
            &train_sigmas,
        );
        sigmas.push(0.0);

        let max_sigma = sigmas
            .iter()
            .copied()
            .fold(0.0f64, |a, b| if a > b { a } else { b });

        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.init_noise_sigma = match cfg.timestep_spacing {
            TimestepSpacing::Trailing | TimestepSpacing::Linspace => max_sigma,
            TimestepSpacing::Leading => (max_sigma * max_sigma + 1.0).sqrt(),
        };
        Ok(())
    }

    /// Precomputed sigma grid (length `inference_steps + 1`, ending in 0).
    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// Scales the denoiser input by `1 / sqrt(sigma_t^2 + 1)`, matching
    /// the K-LMS convention used by k-diffusion samplers.
    pub fn scale_model_input(&self, sample: &LazyTensor, timestep_idx: usize) -> Result<LazyTensor> {
        if timestep_idx >= self.sigmas.len() {
            return Err(Error::Msg(format!(
                "scale_model_input: timestep_idx {timestep_idx} out of bounds (sigmas.len={})",
                self.sigmas.len()
            ))
            .bt());
        }
        let sigma = self.sigmas[timestep_idx];
        let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
        Ok(sample.mul_scalar(scale))
    }

    /// One ancestral-Euler reverse step. `timestep_idx` indexes into
    /// [`Self::timesteps`] (and [`Self::sigmas`]); the caller is
    /// responsible for supplying a `noise` tensor matching `sample`'s
    /// shape (pass an all-zero tensor for a deterministic step).
    pub fn step(
        &self,
        model_output: &LazyTensor,
        timestep_idx: usize,
        sample: &LazyTensor,
        noise: &LazyTensor,
    ) -> Result<LazyTensor> {
        if timestep_idx + 1 >= self.sigmas.len() {
            return Err(Error::Msg(format!(
                "step: timestep_idx {timestep_idx} out of bounds (sigmas.len={})",
                self.sigmas.len()
            ))
            .bt());
        }
        let sigma_from = self.sigmas[timestep_idx];
        let sigma_to = self.sigmas[timestep_idx + 1];

        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample.sub(&model_output.mul_scalar(sigma_from))?,
            PredictionType::VPrediction => {
                let denom = (sigma_from * sigma_from + 1.0).sqrt();
                let a = model_output.mul_scalar(-sigma_from / denom);
                let b = sample.mul_scalar(1.0 / (sigma_from * sigma_from + 1.0));
                a.add(&b)?
            }
            PredictionType::Sample => {
                return Err(Error::Msg(
                    "EulerAncestralDiscreteScheduler: PredictionType::Sample not implemented".into(),
                )
                .bt());
            }
        };

        let sigma_from_sq = sigma_from * sigma_from;
        let sigma_to_sq = sigma_to * sigma_to;
        let sigma_up = if sigma_from_sq > 0.0 {
            (sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq).max(0.0).sqrt()
        } else {
            0.0
        };
        let sigma_down = (sigma_to_sq - sigma_up * sigma_up).max(0.0).sqrt();

        let dt = sigma_down - sigma_from;
        let prev_sample = if sigma_from > 0.0 {
            let inv = 1.0 / sigma_from;
            let derivative = sample.sub(&pred_original_sample)?.mul_scalar(inv);
            sample.add(&derivative.mul_scalar(dt))?
        } else {
            sample.clone()
        };

        prev_sample.add(&noise.mul_scalar(sigma_up))
    }

    /// Forward-process noising at a single timestep index:
    /// `original + noise * sigma[timestep_idx]`.
    pub fn add_noise(
        &self,
        original: &LazyTensor,
        noise: &LazyTensor,
        timestep_idx: usize,
    ) -> Result<LazyTensor> {
        if timestep_idx >= self.sigmas.len() {
            return Err(Error::Msg(format!(
                "add_noise: timestep_idx {timestep_idx} out of bounds (sigmas.len={})",
                self.sigmas.len()
            ))
            .bt());
        }
        let sigma = self.sigmas[timestep_idx];
        original.add(&noise.mul_scalar(sigma))
    }
}

impl SdScheduler for EulerAncestralDiscreteScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Msg(
                "EulerAncestralDiscreteScheduler::set_timesteps: must be > 0".into(),
            )
            .bt());
        }
        self.recompute_schedule(num_inference_steps)
    }

    fn step(
        &mut self,
        model_output: &LazyTensor,
        timestep: usize,
        sample: &LazyTensor,
    ) -> Result<LazyTensor> {
        let idx = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "EulerAncestralDiscreteScheduler::step: timestep {timestep} not in schedule"
                ))
                .bt()
            })?;
        let noise = randn_like_on_graph(sample, 0.0, 1.0)?;
        EulerAncestralDiscreteScheduler::step(self, model_output, idx, sample, &noise)
    }

    fn add_noise(
        &self,
        original: &LazyTensor,
        noise: &LazyTensor,
        timesteps: &[usize],
    ) -> Result<LazyTensor> {
        if timesteps.is_empty() {
            return Err(Error::Msg(
                "EulerAncestralDiscreteScheduler::add_noise: timesteps must be non-empty".into(),
            )
            .bt());
        }
        let idx = self
            .timesteps
            .iter()
            .position(|&t| t == timesteps[0])
            .ok_or_else(|| {
                Error::Msg(format!(
                    "EulerAncestralDiscreteScheduler::add_noise: timestep {} not in schedule",
                    timesteps[0]
                ))
                .bt()
            })?;
        EulerAncestralDiscreteScheduler::add_noise(self, original, noise, idx)
    }
}

// ---- host-side helpers (duplicated from lazy_sd_samplers since they're
// crate-private over there; trivially small and keeps that module's
// surface untouched). -----------------------------------------------------

fn linspace(start: f64, stop: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        Vec::new()
    } else if steps == 1 {
        vec![start]
    } else {
        let delta = (stop - start) / (steps - 1) as f64;
        (0..steps).map(|i| start + i as f64 * delta).collect()
    }
}

fn betas_for_alpha_bar(num_diffusion_timesteps: usize, max_beta: f64) -> Vec<f64> {
    let alpha_bar = |t: usize| {
        f64::cos((t as f64 + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2)
    };
    let mut betas = Vec::with_capacity(num_diffusion_timesteps);
    for i in 0..num_diffusion_timesteps {
        let t1 = i / num_diffusion_timesteps;
        let t2 = (i + 1) / num_diffusion_timesteps;
        betas.push((1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta));
    }
    betas
}

fn build_betas(schedule: BetaSchedule, beta_start: f64, beta_end: f64, n: usize) -> Vec<f64> {
    match schedule {
        BetaSchedule::Linear => linspace(beta_start, beta_end, n),
        BetaSchedule::ScaledLinear => linspace(beta_start.sqrt(), beta_end.sqrt(), n)
            .into_iter()
            .map(|x| x * x)
            .collect(),
        BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(n, 0.999),
    }
}

fn build_alphas_cumprod(betas: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(betas.len());
    let mut acc = 1.0f64;
    for &b in betas {
        acc *= 1.0 - b;
        out.push(acc);
    }
    out
}

/// Piecewise-linear interpolation matching numpy.interp semantics
/// for in-range xs (caller guarantees `x[i] <= xp.last()`).
fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    debug_assert_eq!(xp.len(), fp.len());
    debug_assert!(xp.len() >= 2);
    x.iter()
        .map(|&xi| {
            if xi <= xp[0] {
                return fp[0];
            }
            if xi >= *xp.last().unwrap() {
                return *fp.last().unwrap();
            }
            let idx = xp.partition_point(|&v| v <= xi).saturating_sub(1);
            let idx = idx.min(xp.len() - 2);
            let x_l = xp[idx];
            let x_h = xp[idx + 1];
            let y_l = fp[idx];
            let y_h = fp[idx + 1];
            let dx = x_h - x_l;
            if dx > 0.0 { y_l + (xi - x_l) / dx * (y_h - y_l) } else { y_l }
        })
        .collect()
}

fn randn_like_on_graph(anchor: &LazyTensor, mean: f64, stdev: f64) -> Result<LazyTensor> {
    use rand_distr::{Distribution, Normal};
    let shape = anchor.shape();
    let n = shape.elem_count();
    let normal = Normal::new(mean, stdev).map_err(|e| {
        Error::Msg(format!("randn_like_on_graph: invalid stdev={stdev}: {e}")).bt()
    })?;
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng) as f32).collect();
    Ok(anchor.const_f32_like(data, Shape::from_dims(shape.dims())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn paired(a: &[f32], b: &[f32], shape: &[usize]) -> (LazyTensor, LazyTensor) {
        let anchor = LazyTensor::from_f32(a.to_vec(), Shape::from_dims(shape), &Device::cpu());
        let other = anchor.const_f32_like(b.to_vec(), Shape::from_dims(shape));
        (anchor, other)
    }

    #[test]
    fn sigmas_descend() {
        let sched = EulerAncestralDiscreteScheduler::new(
            10,
            EulerAncestralDiscreteSchedulerConfig::default(),
        )
        .unwrap();
        let sigmas = sched.sigmas();
        assert!(sigmas.len() == 11);
        for w in sigmas[..sigmas.len() - 1].windows(2) {
            assert!(w[0] > w[1], "non-descending sigmas: {sigmas:?}");
        }
        assert_eq!(sigmas[sigmas.len() - 1], 0.0);
    }

    #[test]
    fn step_finite_on_tiny_sample() {
        let sched = EulerAncestralDiscreteScheduler::new(
            10,
            EulerAncestralDiscreteSchedulerConfig::default(),
        )
        .unwrap();
        let (sample, model_out) = paired(
            &[0.1, -0.2, 0.3, 0.0],
            &[0.05, 0.01, -0.04, 0.02],
            &[1, 4],
        );
        let noise = sample.const_f32_like(vec![0.0; 4], Shape::from_dims(&[1, 4]));
        let next = sched.step(&model_out, 0, &sample, &noise).unwrap();
        let out = next.realize_f32();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    }

    #[test]
    fn set_timesteps_zero_noise_matches_deterministic() {
        let mut sched = EulerAncestralDiscreteScheduler::new(
            4,
            EulerAncestralDiscreteSchedulerConfig::default(),
        )
        .unwrap();
        sched.set_timesteps(8).unwrap();

        let sample_vals: [f32; 4] = [0.5, -0.25, 0.75, 0.1];
        let model_vals: [f32; 4] = [0.2, 0.4, -0.1, 0.05];
        let (sample, model_out) = paired(&sample_vals, &model_vals, &[1, 4]);
        let noise = sample.const_f32_like(vec![0.0; 4], Shape::from_dims(&[1, 4]));

        let idx = 2usize;
        let sigma_from = sched.sigmas()[idx];
        let sigma_to = sched.sigmas()[idx + 1];
        let sigma_from_sq = sigma_from * sigma_from;
        let sigma_to_sq = sigma_to * sigma_to;
        let sigma_up = (sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq)
            .max(0.0)
            .sqrt();
        let sigma_down = (sigma_to_sq - sigma_up * sigma_up).max(0.0).sqrt();
        let dt = sigma_down - sigma_from;

        let next = sched.step(&model_out, idx, &sample, &noise).unwrap();
        let got = next.realize_f32();
        for i in 0..4 {
            let x = sample_vals[i] as f64;
            let eps = model_vals[i] as f64;
            let pred_original = x - eps * sigma_from;
            let derivative = (x - pred_original) / sigma_from;
            let expected = (x + derivative * dt) as f32;
            assert!(
                (got[i] - expected).abs() < 1e-4,
                "idx {i}: got {} expected {}",
                got[i],
                expected
            );
        }
    }

    #[test]
    fn init_noise_sigma_correct() {
        let sched = EulerAncestralDiscreteScheduler::new(
            10,
            EulerAncestralDiscreteSchedulerConfig::default(),
        )
        .unwrap();
        let max_sigma = sched
            .sigmas()
            .iter()
            .copied()
            .fold(0.0f64, |a, b| if a > b { a } else { b });
        let expected = (max_sigma * max_sigma + 1.0).sqrt();
        assert!(
            (sched.init_noise_sigma() - expected).abs() < 1e-9,
            "leading init_noise_sigma mismatch: got {}, expected {}",
            sched.init_noise_sigma(),
            expected
        );

        let cfg = EulerAncestralDiscreteSchedulerConfig {
            timestep_spacing: TimestepSpacing::Linspace,
            ..Default::default()
        };
        let sched = EulerAncestralDiscreteScheduler::new(10, cfg).unwrap();
        let max_sigma = sched
            .sigmas()
            .iter()
            .copied()
            .fold(0.0f64, |a, b| if a > b { a } else { b });
        assert!(
            (sched.init_noise_sigma() - max_sigma).abs() < 1e-9,
            "linspace init_noise_sigma mismatch: got {}, expected {}",
            sched.init_noise_sigma(),
            max_sigma
        );
    }
}
