//! Stable-Diffusion noise schedulers ported to the lazy-graph API.
//!
//! Schedulers are pure host-side scalar control: given a noisy sample
//! and a model-predicted noise / velocity / clean sample, they produce
//! the next sample for the reverse-diffusion loop. The per-step
//! coefficients are precomputed `f64` and folded into the graph as
//! [`LazyTensor::mul_scalar`] / [`LazyTensor::add_scalar`] — no new
//! graph ops are introduced.
//!
//! This sub-port covers the [`SdScheduler`] trait + [`DdimScheduler`]
//! + [`DdpmScheduler`]. UniPC, Euler-ancestral and the SD attention
//! blocks ship in separate sub-ports.

use crate::lazy::LazyTensor;
use crate::{Error, Result};
use fuel_core_types::Shape;

/// Variance schedule shape used during training.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
    SquaredcosCapV2,
}

/// What the UNet predicts at each step.
#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}

/// How the inference timestep grid is laid out over the training range.
#[derive(Debug, Default, Clone, Copy)]
pub enum TimestepSpacing {
    #[default]
    Leading,
    Linspace,
    Trailing,
}

/// Variance variant for the DDPM reverse step.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DdpmVarianceType {
    #[default]
    FixedSmall,
    FixedSmallLog,
    FixedLarge,
    FixedLargeLog,
    Learned,
}

/// Common interface for Stable-Diffusion noise schedulers.
pub trait SdScheduler {
    /// Returns the precomputed inference timesteps, ordered from
    /// noisiest to cleanest (the order the denoiser is called in).
    fn timesteps(&self) -> &[usize];

    /// Initial noise scale used when seeding `x_T`.
    fn init_noise_sigma(&self) -> f64;

    /// Recompute the inference timestep grid for `num_inference_steps`.
    fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()>;

    /// One reverse-diffusion update. Returns the next-step sample.
    fn step(
        &mut self,
        model_output: &LazyTensor,
        timestep: usize,
        sample: &LazyTensor,
    ) -> Result<LazyTensor>;

    /// Forward-process noising: `sqrt(alpha_bar_t) * original
    /// + sqrt(1 - alpha_bar_t) * noise`. `timesteps[0]` is the
    /// (single, batched) timestep used to look up the schedule.
    fn add_noise(
        &self,
        original: &LazyTensor,
        noise: &LazyTensor,
        timesteps: &[usize],
    ) -> Result<LazyTensor>;
}

// ---- host-side beta / alpha schedule helpers --------------------------------

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

fn clamp_timestep(t: usize, len: usize) -> usize {
    if t >= len { len - 1 } else { t }
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

// ---- DDIM -------------------------------------------------------------------

/// DDIM scheduler config (matches eager `DDIMSchedulerConfig`).
#[derive(Debug, Clone, Copy)]
pub struct DdimSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub eta: f64,
    pub steps_offset: usize,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
    pub timestep_spacing: TimestepSpacing,
}

impl Default for DdimSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            eta: 0.0,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
            timestep_spacing: TimestepSpacing::Leading,
        }
    }
}

/// DDIM scheduler (Song et al., 2020 — <https://arxiv.org/abs/2010.02502>).
#[derive(Debug, Clone)]
pub struct DdimScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
    init_noise_sigma: f64,
    pub config: DdimSchedulerConfig,
}

impl DdimScheduler {
    /// Builds a DDIM scheduler ready for `inference_steps` reverse steps.
    pub fn new(inference_steps: usize, config: DdimSchedulerConfig) -> Result<Self> {
        if inference_steps == 0 {
            return Err(Error::Msg("DdimScheduler: inference_steps must be > 0".into()).bt());
        }
        if config.train_timesteps == 0 {
            return Err(Error::Msg("DdimScheduler: train_timesteps must be > 0".into()).bt());
        }
        let betas = build_betas(
            config.beta_schedule,
            config.beta_start,
            config.beta_end,
            config.train_timesteps,
        );
        let alphas_cumprod = build_alphas_cumprod(&betas);
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps = build_ddim_timesteps(inference_steps, step_ratio, &config);
        Ok(Self {
            timesteps,
            alphas_cumprod,
            step_ratio,
            init_noise_sigma: 1.0,
            config,
        })
    }
}

fn build_ddim_timesteps(
    inference_steps: usize,
    step_ratio: usize,
    config: &DdimSchedulerConfig,
) -> Vec<usize> {
    match config.timestep_spacing {
        TimestepSpacing::Leading => (0..inference_steps)
            .map(|s| s * step_ratio + config.steps_offset)
            .rev()
            .collect(),
        TimestepSpacing::Trailing => std::iter::successors(Some(config.train_timesteps), |n| {
            if *n > step_ratio { Some(n - step_ratio) } else { None }
        })
        .map(|n| n - 1)
        .collect(),
        TimestepSpacing::Linspace => {
            let xs = linspace(0.0, (config.train_timesteps - 1) as f64, inference_steps);
            xs.into_iter().map(|v| v as usize).rev().collect()
        }
    }
}

impl SdScheduler for DdimScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Msg(
                "DdimScheduler::set_timesteps: num_inference_steps must be > 0".into(),
            )
            .bt());
        }
        self.step_ratio = self.config.train_timesteps / num_inference_steps;
        self.timesteps = build_ddim_timesteps(num_inference_steps, self.step_ratio, &self.config);
        Ok(())
    }

    fn step(
        &mut self,
        model_output: &LazyTensor,
        timestep: usize,
        sample: &LazyTensor,
    ) -> Result<LazyTensor> {
        let t = clamp_timestep(timestep, self.alphas_cumprod.len());
        let prev_t = t.saturating_sub(self.step_ratio);
        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_t_prev = self.alphas_cumprod[prev_t];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        let (pred_original_sample, pred_epsilon) = match self.config.prediction_type {
            PredictionType::Epsilon => {
                let scaled_eps = model_output.mul_scalar(beta_prod_t.sqrt());
                let diff = sample.sub(&scaled_eps)?;
                let pred_original = diff.mul_scalar(1.0 / alpha_prod_t.sqrt());
                (pred_original, model_output.clone())
            }
            PredictionType::VPrediction => {
                let pred_original = sample
                    .mul_scalar(alpha_prod_t.sqrt())
                    .sub(&model_output.mul_scalar(beta_prod_t.sqrt()))?;
                let pred_eps = model_output
                    .mul_scalar(alpha_prod_t.sqrt())
                    .add(&sample.mul_scalar(beta_prod_t.sqrt()))?;
                (pred_original, pred_eps)
            }
            PredictionType::Sample => {
                let pred_original = model_output.clone();
                let diff = sample.sub(&pred_original.mul_scalar(alpha_prod_t.sqrt()))?;
                let pred_eps = diff.mul_scalar(1.0 / beta_prod_t.sqrt());
                (pred_original, pred_eps)
            }
        };

        let variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.sqrt();
        let dir_coef = (1.0 - alpha_prod_t_prev - std_dev_t * std_dev_t).max(0.0).sqrt();
        let pred_sample_direction = pred_epsilon.mul_scalar(dir_coef);
        let prev_sample = pred_original_sample
            .mul_scalar(alpha_prod_t_prev.sqrt())
            .add(&pred_sample_direction)?;

        if self.config.eta > 0.0 {
            let noise = randn_like_on_graph(&prev_sample, 0.0, std_dev_t)?;
            prev_sample.add(&noise)
        } else {
            Ok(prev_sample)
        }
    }

    fn add_noise(
        &self,
        original: &LazyTensor,
        noise: &LazyTensor,
        timesteps: &[usize],
    ) -> Result<LazyTensor> {
        if timesteps.is_empty() {
            return Err(Error::Msg(
                "DdimScheduler::add_noise: timesteps must be non-empty".into(),
            )
            .bt());
        }
        let t = clamp_timestep(timesteps[0], self.alphas_cumprod.len());
        let sqrt_alpha = self.alphas_cumprod[t].sqrt();
        let sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt();
        original
            .mul_scalar(sqrt_alpha)
            .add(&noise.mul_scalar(sqrt_one_minus))
    }
}

// ---- DDPM -------------------------------------------------------------------

/// DDPM scheduler config (matches eager `DDPMSchedulerConfig`).
#[derive(Debug, Clone, Copy)]
pub struct DdpmSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub clip_sample: bool,
    pub variance_type: DdpmVarianceType,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
}

impl Default for DdpmSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            clip_sample: false,
            variance_type: DdpmVarianceType::FixedSmall,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

/// DDPM scheduler (Ho et al., 2020 — <https://arxiv.org/abs/2006.11239>).
#[derive(Debug, Clone)]
pub struct DdpmScheduler {
    alphas_cumprod: Vec<f64>,
    init_noise_sigma: f64,
    timesteps: Vec<usize>,
    step_ratio: usize,
    pub config: DdpmSchedulerConfig,
}

impl DdpmScheduler {
    /// Builds a DDPM scheduler ready for `inference_steps` reverse steps.
    pub fn new(inference_steps: usize, config: DdpmSchedulerConfig) -> Result<Self> {
        if inference_steps == 0 {
            return Err(Error::Msg("DdpmScheduler: inference_steps must be > 0".into()).bt());
        }
        if config.train_timesteps == 0 {
            return Err(Error::Msg("DdpmScheduler: train_timesteps must be > 0".into()).bt());
        }
        let betas = build_betas(
            config.beta_schedule,
            config.beta_start,
            config.beta_end,
            config.train_timesteps,
        );
        let alphas_cumprod = build_alphas_cumprod(&betas);
        let inference_steps = inference_steps.min(config.train_timesteps);
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> = (0..inference_steps).map(|s| s * step_ratio).rev().collect();
        Ok(Self {
            alphas_cumprod,
            init_noise_sigma: 1.0,
            timesteps,
            step_ratio,
            config,
        })
    }

    fn get_variance(&self, timestep: usize) -> f64 {
        let prev_t = timestep as isize - self.step_ratio as isize;
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev;
        let variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t;
        match self.config.variance_type {
            DdpmVarianceType::FixedSmall => variance.max(1e-20),
            DdpmVarianceType::FixedSmallLog => {
                let v = variance.max(1e-20).ln();
                (v * 0.5).exp()
            }
            DdpmVarianceType::FixedLarge => current_beta_t,
            DdpmVarianceType::FixedLargeLog => current_beta_t.ln(),
            DdpmVarianceType::Learned => variance,
        }
    }
}

impl SdScheduler for DdpmScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        if num_inference_steps == 0 {
            return Err(Error::Msg(
                "DdpmScheduler::set_timesteps: num_inference_steps must be > 0".into(),
            )
            .bt());
        }
        let n = num_inference_steps.min(self.config.train_timesteps);
        self.step_ratio = self.config.train_timesteps / n;
        self.timesteps = (0..n).map(|s| s * self.step_ratio).rev().collect();
        Ok(())
    }

    fn step(
        &mut self,
        model_output: &LazyTensor,
        timestep: usize,
        sample: &LazyTensor,
    ) -> Result<LazyTensor> {
        let prev_t = timestep as isize - self.step_ratio as isize;
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        let current_alpha_t = alpha_prod_t / alpha_prod_t_prev;
        let current_beta_t = 1.0 - current_alpha_t;

        let mut pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample
                .sub(&model_output.mul_scalar(beta_prod_t.sqrt()))?
                .mul_scalar(1.0 / alpha_prod_t.sqrt()),
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => sample
                .mul_scalar(alpha_prod_t.sqrt())
                .sub(&model_output.mul_scalar(beta_prod_t.sqrt()))?,
        };

        if self.config.clip_sample {
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0);
        }

        let pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t;
        let current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t;

        let pred_prev_sample = pred_original_sample
            .mul_scalar(pred_original_sample_coeff)
            .add(&sample.mul_scalar(current_sample_coeff))?;

        if timestep > 0 {
            let std = if self.config.variance_type == DdpmVarianceType::FixedSmallLog {
                self.get_variance(timestep)
            } else {
                self.get_variance(timestep).sqrt()
            };
            let noise = randn_like_on_graph(&pred_prev_sample, 0.0, std)?;
            pred_prev_sample.add(&noise)
        } else {
            Ok(pred_prev_sample)
        }
    }

    fn add_noise(
        &self,
        original: &LazyTensor,
        noise: &LazyTensor,
        timesteps: &[usize],
    ) -> Result<LazyTensor> {
        if timesteps.is_empty() {
            return Err(Error::Msg(
                "DdpmScheduler::add_noise: timesteps must be non-empty".into(),
            )
            .bt());
        }
        let t = clamp_timestep(timesteps[0], self.alphas_cumprod.len());
        let sqrt_alpha = self.alphas_cumprod[t].sqrt();
        let sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt();
        original
            .mul_scalar(sqrt_alpha)
            .add(&noise.mul_scalar(sqrt_one_minus))
    }
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
    fn ddim_step_finite_on_tiny_sample() {
        let mut sched = DdimScheduler::new(10, DdimSchedulerConfig::default()).unwrap();
        let t = sched.timesteps()[0];
        let (sample, model_out) =
            paired(&[0.1, -0.2, 0.3, 0.0], &[0.05, 0.01, -0.04, 0.02], &[1, 4]);
        let next = sched.step(&model_out, t, &sample).unwrap();
        let out = next.realize_f32();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    }

    #[test]
    fn ddpm_step_finite_on_tiny_sample() {
        let mut sched = DdpmScheduler::new(10, DdpmSchedulerConfig::default()).unwrap();
        let t = sched.timesteps()[0];
        let (sample, model_out) =
            paired(&[0.1, -0.2, 0.3, 0.0], &[0.05, 0.01, -0.04, 0.02], &[1, 4]);
        let next = sched.step(&model_out, t, &sample).unwrap();
        let out = next.realize_f32();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    }

    #[test]
    fn set_timesteps_produces_descending_schedule() {
        let mut sched = DdimScheduler::new(10, DdimSchedulerConfig::default()).unwrap();
        sched.set_timesteps(25).unwrap();
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 25);
        for w in ts.windows(2) {
            assert!(w[0] > w[1], "DDIM schedule not strictly descending: {ts:?}");
        }

        let mut sched = DdpmScheduler::new(10, DdpmSchedulerConfig::default()).unwrap();
        sched.set_timesteps(50).unwrap();
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 50);
        for w in ts.windows(2) {
            assert!(w[0] > w[1], "DDPM schedule not strictly descending: {ts:?}");
        }
    }

    #[test]
    fn add_noise_matches_alpha_blend() {
        let original_vals: [f32; 4] = [0.5, -0.25, 0.75, 0.1];
        let noise_vals: [f32; 4] = [0.2, 0.4, -0.1, 0.05];

        let sched = DdimScheduler::new(10, DdimSchedulerConfig::default()).unwrap();
        let (original, noise) = paired(&original_vals, &noise_vals, &[1, 4]);
        let blended = sched.add_noise(&original, &noise, &[0]).unwrap().realize_f32();

        let alpha = sched.alphas_cumprod[0];
        let sqrt_a = alpha.sqrt() as f32;
        let sqrt_one_minus = (1.0 - alpha).sqrt() as f32;
        for (i, (&o, &n)) in original_vals.iter().zip(noise_vals.iter()).enumerate() {
            let expected = sqrt_a * o + sqrt_one_minus * n;
            assert!(
                (blended[i] - expected).abs() < 1e-5,
                "DDIM idx {i}: got {} expected {}",
                blended[i],
                expected
            );
        }

        let sched = DdpmScheduler::new(10, DdpmSchedulerConfig::default()).unwrap();
        let (original, noise) = paired(&original_vals, &noise_vals, &[1, 4]);
        let blended = sched.add_noise(&original, &noise, &[0]).unwrap().realize_f32();
        for (i, (&o, &n)) in original_vals.iter().zip(noise_vals.iter()).enumerate() {
            let expected = sqrt_a * o + sqrt_one_minus * n;
            assert!(
                (blended[i] - expected).abs() < 1e-5,
                "DDPM idx {i}: got {} expected {}",
                blended[i],
                expected
            );
        }
    }
}
