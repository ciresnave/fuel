//! UniPC scheduler ported to the lazy-graph API.
//!
//! UniPC is a training-free predictor-corrector framework for fast sampling
//! of diffusion ODE. See <https://arxiv.org/abs/2302.04867>. This is sub-port
//! 4 of `port-sd-samplers.md`.
//!
//! The scheduler's coefficient arithmetic is pure host-side `f64`: sigmas,
//! lambdas, the `rks` / `r` / `b` arrays and their matrix inverse are all
//! computed off-graph. Tensor work is limited to a single linear combination
//! of the stored model-output history (the `d1s` tensors), folded in via
//! `mul_scalar` / `add` — no new graph ops are introduced.
//!
//! # Scope of this port
//!
//! - `solver_order` in `{1, 2, 3}` — covering both guided (order=2) and
//!   unconditional (order=3) recommended modes.
//! - `SolverType::Bh1` (default) and `SolverType::Bh2`.
//! - `AlgorithmType::DpmSolverPlusPlus` (deterministic).
//! - `PredictionType::Epsilon` / `Sample` / `VPrediction`.
//! - Both `Karras` and `Exponential` sigma schedules with `Linspace` or
//!   `FromSigmas` timestep grids.
//! - `lower_order_final` ramp-down.
//! - UniC corrector with configurable skip set.
//!
//! # Deferred (TODOs for follow-ups)
//!
//! - `thresholding` / `dynamic_thresholding_ratio` / `sample_max_value`:
//!   used in pixel-space DPMs but not standard SD latent-space pipelines.
//! - `AlgorithmType::SdeDpmSolverPlusPlus` (stochastic variant).

use std::collections::HashSet;

use crate::lazy::LazyTensor;
use crate::lazy_sd_samplers::{PredictionType, SdScheduler};
use crate::{Error, Result};

/// Sigma-to-timestep mapping strategy.
#[derive(Debug, Clone, Copy)]
pub enum SigmaSchedule {
    Karras(KarrasSigmaSchedule),
    Exponential(ExponentialSigmaSchedule),
}

impl Default for SigmaSchedule {
    fn default() -> Self {
        Self::Karras(KarrasSigmaSchedule::default())
    }
}

impl SigmaSchedule {
    fn sigma_t(&self, t: f64) -> f64 {
        match self {
            Self::Karras(x) => x.sigma_t(t),
            Self::Exponential(x) => x.sigma_t(t),
        }
    }
}

/// Karras sigma schedule parameterised by min/max sigma and power `rho`.
#[derive(Debug, Clone, Copy)]
pub struct KarrasSigmaSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rho: f64,
}

impl Default for KarrasSigmaSchedule {
    fn default() -> Self {
        Self { sigma_max: 10.0, sigma_min: 0.1, rho: 4.0 }
    }
}

impl KarrasSigmaSchedule {
    fn sigma_t(&self, t: f64) -> f64 {
        let min_inv_rho = self.sigma_min.powf(1.0 / self.rho);
        let max_inv_rho = self.sigma_max.powf(1.0 / self.rho);
        (max_inv_rho + (1.0 - t) * (min_inv_rho - max_inv_rho)).powf(self.rho)
    }
}

/// Exponential sigma schedule on a log-linear scale.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialSigmaSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
}

impl Default for ExponentialSigmaSchedule {
    fn default() -> Self {
        Self { sigma_max: 80.0, sigma_min: 0.1 }
    }
}

impl ExponentialSigmaSchedule {
    fn sigma_t(&self, t: f64) -> f64 {
        (t * (self.sigma_max.ln() - self.sigma_min.ln()) + self.sigma_min.ln()).exp()
    }
}

/// UniPC predictor variant. `Bh1` and `Bh2` differ in how the prediction `B(h)` term is formed.
#[derive(Debug, Default, Clone, Copy)]
pub enum SolverType {
    #[default]
    Bh1,
    Bh2,
}

/// Diffusion algorithm variant. Only the deterministic `DpmSolverPlusPlus` is
/// implemented in this port; the stochastic `SdeDpmSolverPlusPlus` is a TODO.
#[derive(Debug, Default, Clone, Copy)]
pub enum AlgorithmType {
    #[default]
    DpmSolverPlusPlus,
}

/// How to space the inference timesteps across the full training range.
#[derive(Debug, Clone)]
pub enum TimestepSchedule {
    /// Derive timesteps from the sigma schedule via log-sigma interpolation.
    FromSigmas,
    /// Evenly distribute timesteps over `[0, num_training_timesteps - 1]`.
    Linspace,
}

/// Configures whether the UniC corrector runs at a given step.
#[derive(Debug, Clone)]
pub enum CorrectorConfiguration {
    Disabled,
    Enabled { skip_steps: HashSet<usize> },
}

impl Default for CorrectorConfiguration {
    fn default() -> Self {
        Self::Enabled { skip_steps: [0, 1, 2].into_iter().collect() }
    }
}

impl CorrectorConfiguration {
    /// Enables the corrector and skips the listed step indices.
    pub fn new(skip_steps: impl IntoIterator<Item = usize>) -> Self {
        Self::Enabled { skip_steps: skip_steps.into_iter().collect() }
    }
}

/// Full UniPC scheduler configuration.
#[derive(Debug, Clone)]
pub struct UniPcSchedulerConfig {
    pub corrector: CorrectorConfiguration,
    pub sigma_schedule: SigmaSchedule,
    pub timestep_schedule: TimestepSchedule,
    /// Solver order. 1, 2 or 3. Recommended: 2 for guided sampling, 3 for unconditional.
    pub solver_order: usize,
    pub prediction_type: PredictionType,
    pub num_training_timesteps: usize,
    pub solver_type: SolverType,
    pub algorithm_type: AlgorithmType,
    /// Whether to fall back to lower-order solvers near the end of the schedule.
    pub lower_order_final: bool,
}

impl Default for UniPcSchedulerConfig {
    fn default() -> Self {
        Self {
            corrector: CorrectorConfiguration::default(),
            timestep_schedule: TimestepSchedule::FromSigmas,
            sigma_schedule: SigmaSchedule::default(),
            prediction_type: PredictionType::Epsilon,
            num_training_timesteps: 1000,
            solver_order: 2,
            solver_type: SolverType::Bh1,
            algorithm_type: AlgorithmType::DpmSolverPlusPlus,
            lower_order_final: true,
        }
    }
}

#[derive(Debug, Clone)]
struct Schedule {
    timesteps: Vec<usize>,
    num_training_timesteps: usize,
    sigma_schedule: SigmaSchedule,
}

impl Schedule {
    fn new(
        timestep_schedule: &TimestepSchedule,
        sigma_schedule: SigmaSchedule,
        num_inference_steps: usize,
        num_training_timesteps: usize,
    ) -> Result<Self> {
        let timesteps = build_timesteps(
            timestep_schedule,
            &sigma_schedule,
            num_inference_steps,
            num_training_timesteps,
        )?;
        Ok(Self { timesteps, num_training_timesteps, sigma_schedule })
    }

    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn t(&self, step: usize) -> f64 {
        (step as f64 + 1.0) / self.num_training_timesteps as f64
    }

    fn alpha_t(&self, t: usize) -> f64 {
        (1.0 / (self.sigma_schedule.sigma_t(self.t(t)).powi(2) + 1.0)).sqrt()
    }

    fn sigma_t(&self, t: usize) -> f64 {
        self.sigma_schedule.sigma_t(self.t(t)) * self.alpha_t(t)
    }

    fn lambda_t(&self, t: usize) -> f64 {
        self.alpha_t(t).ln() - self.sigma_t(t).ln()
    }
}

fn linspace_f64(start: f64, stop: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        Vec::new()
    } else if steps == 1 {
        vec![start]
    } else {
        let delta = (stop - start) / (steps - 1) as f64;
        (0..steps).map(|i| start + i as f64 * delta).collect()
    }
}

/// Piecewise-linear interpolation: for each `x[i]`, look it up against the
/// sorted `xp` and linearly interpolate `fp`. Matches `numpy.interp` semantics.
fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xv| {
            if xp.is_empty() {
                return f64::NAN;
            }
            if xv <= xp[0] {
                return fp[0];
            }
            if xv >= xp[xp.len() - 1] {
                return fp[xp.len() - 1];
            }
            let idx = xp.partition_point(|o| *o <= xv).saturating_sub(1);
            let (xl, xh) = (xp[idx], xp[idx + 1]);
            let (yl, yh) = (fp[idx], fp[idx + 1]);
            let dx = xh - xl;
            if dx > 0.0 { yl + (xv - xl) / dx * (yh - yl) } else { f64::NAN }
        })
        .collect()
}

fn build_timesteps(
    schedule: &TimestepSchedule,
    sigma_schedule: &SigmaSchedule,
    num_inference_steps: usize,
    num_training_timesteps: usize,
) -> Result<Vec<usize>> {
    if num_inference_steps == 0 {
        return Err(Error::Msg(
            "UniPcScheduler: num_inference_steps must be > 0".into(),
        )
        .bt());
    }
    if num_training_timesteps == 0 {
        return Err(Error::Msg(
            "UniPcScheduler: num_training_timesteps must be > 0".into(),
        )
        .bt());
    }
    match schedule {
        TimestepSchedule::FromSigmas => {
            let sigmas: Vec<f64> = linspace_f64(1.0, 0.0, num_inference_steps)
                .into_iter()
                .map(|t| sigma_schedule.sigma_t(t))
                .collect();
            let log_sigmas: Vec<f64> = sigmas.iter().map(|s| s.ln()).collect();
            let rev_log_sigmas: Vec<f64> =
                log_sigmas.iter().copied().rev().collect();
            let query = linspace_f64(
                log_sigmas[log_sigmas.len() - 1] - 0.001,
                log_sigmas[0] + 0.001,
                num_inference_steps,
            );
            let grid = linspace_f64(
                0.0,
                num_training_timesteps as f64,
                num_inference_steps,
            );
            let interped = interp(&rev_log_sigmas, &query, &grid);
            Ok(interped
                .into_iter()
                .map(|f| (num_training_timesteps - 1) - (f as usize))
                .collect())
        }
        TimestepSchedule::Linspace => Ok(linspace_f64(
            (num_training_timesteps - 1) as f64,
            0.0,
            num_inference_steps,
        )
        .into_iter()
        .map(|f| f as usize)
        .collect()),
    }
}

/// Solves `A x = b` for a small square matrix via Gauss-Jordan. `A` is `n x n` row-major.
/// Returns the solution vector or an error if the matrix is singular.
fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
        return Err(Error::Msg(format!(
            "solve_linear: dim mismatch a.len={} b.len={} n={n}",
            a.len(),
            b.len()
        ))
        .bt());
    }
    let mut m = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            m[i * (n + 1) + j] = a[i * n + j];
        }
        m[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col * (n + 1) + col].abs();
        for r in (col + 1)..n {
            let v = m[r * (n + 1) + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-30 {
            return Err(Error::Msg("solve_linear: singular matrix".into()).bt());
        }
        if pivot != col {
            for j in 0..=n {
                let (ai, bi) = (col * (n + 1) + j, pivot * (n + 1) + j);
                m.swap(ai, bi);
            }
        }
        let piv = m[col * (n + 1) + col];
        for j in col..=n {
            m[col * (n + 1) + j] /= piv;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r * (n + 1) + col];
            if factor == 0.0 {
                continue;
            }
            for j in col..=n {
                m[r * (n + 1) + j] -= factor * m[col * (n + 1) + j];
            }
        }
    }
    Ok((0..n).map(|i| m[i * (n + 1) + n]).collect())
}

#[derive(Debug, Clone)]
struct UniPcState {
    model_outputs: Vec<Option<LazyTensor>>,
    timestep_history: Vec<Option<usize>>,
    lower_order_nums: usize,
    order: usize,
    last_sample: Option<LazyTensor>,
}

impl UniPcState {
    fn new(solver_order: usize) -> Self {
        Self {
            model_outputs: vec![None; solver_order],
            timestep_history: vec![None; solver_order],
            lower_order_nums: 0,
            order: 0,
            last_sample: None,
        }
    }
}

/// UniPC scheduler implementing predictor (UniP) + corrector (UniC) updates.
#[derive(Debug, Clone)]
pub struct UniPcScheduler {
    schedule: Schedule,
    state: UniPcState,
    pub config: UniPcSchedulerConfig,
}

impl UniPcScheduler {
    /// Builds a UniPC scheduler ready for `inference_steps` reverse steps.
    pub fn new(inference_steps: usize, config: UniPcSchedulerConfig) -> Result<Self> {
        if config.solver_order == 0 || config.solver_order > 3 {
            return Err(Error::Msg(format!(
                "UniPcScheduler: solver_order must be 1, 2 or 3, got {}",
                config.solver_order
            ))
            .bt());
        }
        let schedule = Schedule::new(
            &config.timestep_schedule,
            config.sigma_schedule,
            inference_steps,
            config.num_training_timesteps,
        )?;
        let state = UniPcState::new(config.solver_order);
        Ok(Self { schedule, state, config })
    }

    fn step_index(&self, timestep: usize) -> usize {
        let candidates: Vec<usize> = self
            .schedule
            .timesteps()
            .iter()
            .enumerate()
            .filter_map(|(i, t)| if *t == timestep { Some(i) } else { None })
            .collect();
        match candidates.len() {
            0 => 0,
            1 => candidates[0],
            _ => candidates[1],
        }
    }

    fn timestep_at(&self, step_idx: usize) -> usize {
        self.schedule.timesteps().get(step_idx).copied().unwrap_or(0)
    }

    fn convert_model_output(
        &self,
        model_output: &LazyTensor,
        sample: &LazyTensor,
        timestep: usize,
    ) -> Result<LazyTensor> {
        let alpha_t = self.schedule.alpha_t(timestep);
        let sigma_t = self.schedule.sigma_t(timestep);
        match self.config.prediction_type {
            PredictionType::Epsilon => sample
                .sub(&model_output.mul_scalar(sigma_t))
                .map(|t| t.mul_scalar(1.0 / alpha_t)),
            PredictionType::Sample => Ok(model_output.clone()),
            PredictionType::VPrediction => sample
                .mul_scalar(alpha_t)
                .sub(&model_output.mul_scalar(sigma_t)),
        }
    }

    /// Predictor (UniP) update. Combines the latest model output `m0` and
    /// up to `order - 1` older outputs (the `d1s` differences) into a
    /// single linear combination weighted by host-computed `rhos_p`.
    ///
    /// Note on indexing: the upstream diffusers Python reference uses
    /// `pi = step_index - (i + 1)` with `i` ranging `0..order-1`. The
    /// eager Rust translation ranges `i` `1..order` while keeping the
    /// `(i + 1)` term, which is an off-by-one. This port uses
    /// `step_index - i` (Rust `i` starts at 1) to match the algorithmic
    /// intent: at order=k we look up the `k-1` most recent past steps.
    fn unip_bh_update(&self, sample: &LazyTensor, timestep: usize) -> Result<LazyTensor> {
        let step_index = self.step_index(timestep);
        let ns = &self.schedule;
        let model_outputs = self.state.model_outputs.as_slice();
        let m0 = model_outputs
            .last()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| {
                Error::Msg("UniP: missing latest model output".into()).bt()
            })?;
        let order = self.state.order;
        if order == 0 {
            return Err(Error::Msg("UniP: order must be >= 1".into()).bt());
        }

        let (t0, tt) = (timestep, self.timestep_at(step_index + 1));
        let sigma_t = ns.sigma_t(tt);
        let sigma_s0 = ns.sigma_t(t0);
        let alpha_t = ns.alpha_t(tt);
        let lambda_t = ns.lambda_t(tt);
        let lambda_s0 = ns.lambda_t(t0);
        let h = lambda_t - lambda_s0;

        // Cap order so all past timesteps are distinct and present.
        let effective_order = order.min(step_index + 1);

        let mut rks = Vec::with_capacity(effective_order);
        let mut d1s: Vec<LazyTensor> = Vec::with_capacity(effective_order.saturating_sub(1));
        for i in 1..effective_order {
            let ti = self.timestep_at(step_index - i);
            let history_idx = model_outputs.len().saturating_sub(i + 1);
            let mi = model_outputs
                .get(history_idx)
                .and_then(|o| o.as_ref())
                .ok_or_else(|| {
                    Error::Msg(format!("UniP: missing history idx {history_idx}")).bt()
                })?;
            let alpha_si = ns.alpha_t(ti);
            let sigma_si = ns.sigma_t(ti);
            let lambda_si = alpha_si.ln() - sigma_si.ln();
            let rk = (lambda_si - lambda_s0) / h;
            if rk == 0.0 {
                return Err(Error::Msg("UniP: zero rk".into()).bt());
            }
            rks.push(rk);
            d1s.push(mi.sub(m0)?.mul_scalar(1.0 / rk));
        }
        rks.push(1.0);

        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = match self.config.solver_type {
            SolverType::Bh1 => hh,
            SolverType::Bh2 => hh.exp_m1(),
        };
        if b_h == 0.0 {
            return Err(Error::Msg("UniP: zero b_h".into()).bt());
        }

        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0f64;
        let mut r_mat: Vec<Vec<f64>> = Vec::with_capacity(effective_order);
        let mut b_vec: Vec<f64> = Vec::with_capacity(effective_order);
        for i in 1..=effective_order {
            let row: Vec<f64> = rks.iter().map(|rk| rk.powf(i as f64 - 1.0)).collect();
            r_mat.push(row);
            b_vec.push(h_phi_k * factorial_i / b_h);
            factorial_i = i as f64 + 1.0;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }

        let x_t_ = sample
            .mul_scalar(sigma_t / sigma_s0)
            .sub(&m0.mul_scalar(alpha_t * h_phi_1))?;

        if d1s.is_empty() {
            return Ok(x_t_);
        }

        let rhos_p: Vec<f64> = match effective_order {
            2 => vec![0.5],
            _ => {
                let n = effective_order - 1;
                let mut a_flat = vec![0.0f64; n * n];
                let mut b_flat = vec![0.0f64; n];
                for i in 0..n {
                    for j in 0..n {
                        a_flat[i * n + j] = r_mat[i][j];
                    }
                    b_flat[i] = b_vec[i];
                }
                solve_linear(&a_flat, &b_flat, n)?
            }
        };

        let pred_res = weighted_sum(&rhos_p, &d1s)?;
        x_t_.sub(&pred_res.mul_scalar(alpha_t * b_h))
    }

    /// Corrector (UniC) update. Mirrors the predictor but uses the full
    /// `order` weights and a final correction term from the current model
    /// output `model_t`.
    fn unic_bh_update(
        &self,
        model_output: &LazyTensor,
        model_outputs: &[Option<LazyTensor>],
        last_sample: &LazyTensor,
        _sample: &LazyTensor,
        timestep: usize,
    ) -> Result<LazyTensor> {
        let step_index = self.step_index(timestep);
        let m0 = model_outputs
            .last()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| {
                Error::Msg("UniC: missing latest model output".into()).bt()
            })?;
        let ns = &self.schedule;
        let order = self.state.order;
        if order == 0 {
            return Err(Error::Msg("UniC: order must be >= 1".into()).bt());
        }

        let t0 = self.timestep_at(step_index.saturating_sub(1));
        let tt = timestep;
        let sigma_t = ns.sigma_t(tt);
        let sigma_s0 = ns.sigma_t(t0);
        let alpha_t = ns.alpha_t(tt);
        let lambda_t = ns.lambda_t(tt);
        let lambda_s0 = ns.lambda_t(t0);
        let h = lambda_t - lambda_s0;

        let mut rks = Vec::with_capacity(order);
        let mut d1s: Vec<LazyTensor> = Vec::with_capacity(order.saturating_sub(1));
        for i in 1..order {
            let ti = self.timestep_at(step_index.saturating_sub(i + 1));
            let history_idx = model_outputs.len().saturating_sub(i + 1);
            let mi = model_outputs
                .get(history_idx)
                .and_then(|o| o.as_ref())
                .ok_or_else(|| {
                    Error::Msg(format!("UniC: missing history idx {history_idx}")).bt()
                })?;
            let alpha_si = ns.alpha_t(ti);
            let sigma_si = ns.sigma_t(ti);
            let lambda_si = alpha_si.ln() - sigma_si.ln();
            let rk = (lambda_si - lambda_s0) / h;
            if rk == 0.0 {
                return Err(Error::Msg("UniC: zero rk".into()).bt());
            }
            rks.push(rk);
            d1s.push(mi.sub(m0)?.mul_scalar(1.0 / rk));
        }
        rks.push(1.0);

        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = match self.config.solver_type {
            SolverType::Bh1 => hh,
            SolverType::Bh2 => hh.exp_m1(),
        };
        if b_h == 0.0 {
            return Err(Error::Msg("UniC: zero b_h".into()).bt());
        }

        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0f64;
        let mut r_mat: Vec<Vec<f64>> = Vec::with_capacity(order);
        let mut b_vec: Vec<f64> = Vec::with_capacity(order);
        for i in 1..=order {
            let row: Vec<f64> = rks.iter().map(|rk| rk.powf(i as f64 - 1.0)).collect();
            r_mat.push(row);
            b_vec.push(h_phi_k * factorial_i / b_h);
            factorial_i = i as f64 + 1.0;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }

        let rhos_c: Vec<f64> = match order {
            1 => vec![0.5],
            _ => {
                let n = order;
                let mut a_flat = vec![0.0f64; n * n];
                let mut b_flat = vec![0.0f64; n];
                for i in 0..n {
                    for j in 0..n {
                        a_flat[i * n + j] = r_mat[i][j];
                    }
                    b_flat[i] = b_vec[i];
                }
                solve_linear(&a_flat, &b_flat, n)?
            }
        };

        let x_t_ = last_sample
            .mul_scalar(sigma_t / sigma_s0)
            .sub(&m0.mul_scalar(alpha_t * h_phi_1))?;

        let corr_res = if d1s.is_empty() {
            None
        } else {
            Some(weighted_sum(&rhos_c[..rhos_c.len() - 1], &d1s)?)
        };
        let d1_t = model_output.sub(m0)?;
        let final_corr_coef = rhos_c[rhos_c.len() - 1];
        let combined = match corr_res {
            Some(cr) => cr.add(&d1_t.mul_scalar(final_corr_coef))?,
            None => d1_t.mul_scalar(final_corr_coef),
        };
        x_t_.sub(&combined.mul_scalar(alpha_t * b_h))
    }
}

/// Returns `sum_i weights[i] * tensors[i]`. Requires non-empty inputs of equal length.
fn weighted_sum(weights: &[f64], tensors: &[LazyTensor]) -> Result<LazyTensor> {
    if weights.is_empty() || weights.len() != tensors.len() {
        return Err(Error::Msg(format!(
            "weighted_sum: weights.len={} tensors.len={}",
            weights.len(),
            tensors.len()
        ))
        .bt());
    }
    let mut acc = tensors[0].mul_scalar(weights[0]);
    for i in 1..weights.len() {
        acc = acc.add(&tensors[i].mul_scalar(weights[i]))?;
    }
    Ok(acc)
}

impl SdScheduler for UniPcScheduler {
    fn timesteps(&self) -> &[usize] {
        self.schedule.timesteps()
    }

    fn init_noise_sigma(&self) -> f64 {
        self.schedule
            .sigma_schedule
            .sigma_t(self.schedule.t(self.schedule.num_training_timesteps))
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        self.schedule = Schedule::new(
            &self.config.timestep_schedule,
            self.config.sigma_schedule,
            num_inference_steps,
            self.config.num_training_timesteps,
        )?;
        self.state = UniPcState::new(self.config.solver_order);
        Ok(())
    }

    fn step(
        &mut self,
        model_output: &LazyTensor,
        timestep: usize,
        sample: &LazyTensor,
    ) -> Result<LazyTensor> {
        let step_index = self.step_index(timestep);
        let model_output_converted =
            self.convert_model_output(model_output, sample, timestep)?;

        let corrected_sample: LazyTensor = match (
            &self.config.corrector,
            self.state.last_sample.clone(),
        ) {
            (CorrectorConfiguration::Enabled { skip_steps }, Some(last_sample))
                if !skip_steps.contains(&step_index) && step_index > 0 =>
            {
                self.unic_bh_update(
                    &model_output_converted,
                    &self.state.model_outputs.clone(),
                    &last_sample,
                    sample,
                    timestep,
                )?
            }
            _ => sample.clone(),
        };

        let solver_order = self.config.solver_order;
        for i in 0..solver_order.saturating_sub(1) {
            self.state.model_outputs[i] = self.state.model_outputs[i + 1].take();
            self.state.timestep_history[i] = self.state.timestep_history[i + 1].take();
        }
        let last_idx = self.state.model_outputs.len() - 1;
        self.state.model_outputs[last_idx] = Some(model_output_converted);
        self.state.timestep_history[last_idx] = Some(timestep);

        let mut this_order = self.config.solver_order;
        if self.config.lower_order_final {
            let remaining = self.schedule.timesteps.len().saturating_sub(step_index);
            this_order = this_order.min(remaining);
        }
        let new_order = this_order.min(self.state.lower_order_nums + 1).max(1);
        self.state.order = new_order;

        self.state.last_sample = Some(corrected_sample.clone());
        let prev_sample = self.unip_bh_update(&corrected_sample, timestep)?;

        if self.state.lower_order_nums < self.config.solver_order {
            self.state.lower_order_nums += 1;
        }

        Ok(prev_sample)
    }

    fn add_noise(
        &self,
        original: &LazyTensor,
        noise: &LazyTensor,
        timesteps: &[usize],
    ) -> Result<LazyTensor> {
        if timesteps.is_empty() {
            return Err(Error::Msg(
                "UniPcScheduler::add_noise: timesteps must be non-empty".into(),
            )
            .bt());
        }
        let t = timesteps[0];
        let alpha_t = self.schedule.alpha_t(t);
        let sigma_t = self.schedule.sigma_t(t);
        original
            .mul_scalar(alpha_t)
            .add(&noise.mul_scalar(sigma_t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_ir::Shape;

    fn lazy_from(values: &[f32], shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(values.to_vec(), Shape::from_dims(shape), &Device::cpu())
    }

    fn lazy_like(anchor: &LazyTensor, values: &[f32], shape: &[usize]) -> LazyTensor {
        anchor.const_f32_like(values.to_vec(), Shape::from_dims(shape))
    }

    fn finite(out: &[f32]) -> bool {
        out.iter().all(|v| v.is_finite())
    }

    #[test]
    fn set_timesteps_produces_descending_schedule() {
        let mut sched = UniPcScheduler::new(
            10,
            UniPcSchedulerConfig {
                timestep_schedule: TimestepSchedule::Linspace,
                ..UniPcSchedulerConfig::default()
            },
        )
        .unwrap();
        sched.set_timesteps(20).unwrap();
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 20);
        for w in ts.windows(2) {
            assert!(
                w[0] > w[1],
                "UniPC schedule must be strictly descending: {ts:?}"
            );
        }
        let mut sched = UniPcScheduler::new(10, UniPcSchedulerConfig::default()).unwrap();
        sched.set_timesteps(25).unwrap();
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 25);
        for w in ts.windows(2) {
            assert!(
                w[0] >= w[1],
                "UniPC schedule must be non-increasing: {ts:?}"
            );
        }
    }

    #[test]
    fn order_2_predictor_step_finite() {
        let config = UniPcSchedulerConfig {
            solver_order: 2,
            timestep_schedule: TimestepSchedule::Linspace,
            corrector: CorrectorConfiguration::Disabled,
            lower_order_final: false,
            ..UniPcSchedulerConfig::default()
        };
        let mut sched = UniPcScheduler::new(10, config).unwrap();
        let sample_vals = [0.1f32, -0.2, 0.3, 0.0, 0.5, -0.4];
        let model_vals = [0.05f32, 0.01, -0.04, 0.02, -0.03, 0.07];
        let anchor = lazy_from(&sample_vals, &[1, 6]);
        let model_out = lazy_like(&anchor, &model_vals, &[1, 6]);

        let ts = sched.timesteps().to_vec();
        assert!(ts.len() >= 3);
        let mut sample = anchor;
        for &t in &ts[..3] {
            sample = sched.step(&model_out, t, &sample).unwrap();
            let out = sample.realize_f32();
            assert_eq!(out.len(), 6);
            assert!(
                finite(&out),
                "non-finite predictor output at t={t}: {out:?}"
            );
        }
        assert!(sched.state.order >= 1);
    }

    #[test]
    fn order_3_predictor_corrector_step_finite() {
        let config = UniPcSchedulerConfig {
            solver_order: 3,
            timestep_schedule: TimestepSchedule::Linspace,
            corrector: CorrectorConfiguration::new([0]),
            lower_order_final: true,
            ..UniPcSchedulerConfig::default()
        };
        let mut sched = UniPcScheduler::new(8, config).unwrap();
        let sample_vals = [0.2f32, -0.1, 0.05, 0.4, -0.3, 0.0, 0.15, -0.25];
        let model_vals = [0.02f32, 0.04, -0.01, 0.03, 0.05, -0.02, 0.01, 0.0];
        let anchor = lazy_from(&sample_vals, &[1, 8]);
        let model_out = lazy_like(&anchor, &model_vals, &[1, 8]);

        let ts = sched.timesteps().to_vec();
        let mut sample = anchor;
        for &t in &ts[..5] {
            sample = sched.step(&model_out, t, &sample).unwrap();
            let out = sample.realize_f32();
            assert_eq!(out.len(), 8);
            assert!(
                finite(&out),
                "non-finite corrector output at t={t}: {out:?}"
            );
        }
    }

    #[test]
    fn history_window_advances_per_step() {
        let config = UniPcSchedulerConfig {
            solver_order: 3,
            timestep_schedule: TimestepSchedule::Linspace,
            corrector: CorrectorConfiguration::Disabled,
            lower_order_final: false,
            ..UniPcSchedulerConfig::default()
        };
        let mut sched = UniPcScheduler::new(10, config).unwrap();
        let sample_vals = [0.0f32, 0.1, 0.2, 0.3];
        let model_vals = [0.01f32, 0.02, 0.03, 0.04];
        let anchor = lazy_from(&sample_vals, &[1, 4]);
        let model_out = lazy_like(&anchor, &model_vals, &[1, 4]);

        let ts = sched.timesteps().to_vec();
        let mut sample = anchor;
        let mut prior_count = 0usize;
        for (i, &t) in ts.iter().enumerate().take(5) {
            sample = sched.step(&model_out, t, &sample).unwrap();
            let filled = sched
                .state
                .model_outputs
                .iter()
                .filter(|o| o.is_some())
                .count();
            assert!(
                filled >= prior_count,
                "history window shrank at step {i}: {filled} < {prior_count}"
            );
            prior_count = filled;
            // After 3+ steps, the order-3 ring buffer is fully populated.
            if i >= 2 {
                assert_eq!(
                    filled,
                    sched.config.solver_order,
                    "ring buffer should be full after {} steps",
                    i + 1
                );
            }
            assert_eq!(
                sched.state.timestep_history.last().unwrap().as_ref(),
                Some(&t),
                "latest history slot should match timestep at step {i}"
            );
        }
    }

    #[test]
    fn solve_linear_3x3_recovers_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let b = vec![1.0, 4.0, 9.0];
        let x = solve_linear(&a, &b, 3).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn add_noise_alpha_sigma_blend() {
        let sched = UniPcScheduler::new(
            10,
            UniPcSchedulerConfig {
                timestep_schedule: TimestepSchedule::Linspace,
                ..UniPcSchedulerConfig::default()
            },
        )
        .unwrap();
        let orig_vals = [0.5f32, -0.25, 0.75, 0.1];
        let noise_vals = [0.2f32, 0.4, -0.1, 0.05];
        let anchor = lazy_from(&orig_vals, &[1, 4]);
        let noise = lazy_like(&anchor, &noise_vals, &[1, 4]);
        let t = sched.timesteps()[0];
        let alpha = sched.schedule.alpha_t(t) as f32;
        let sigma = sched.schedule.sigma_t(t) as f32;
        let blended = sched.add_noise(&anchor, &noise, &[t]).unwrap().realize_f32();
        for (i, (&o, &n)) in orig_vals.iter().zip(noise_vals.iter()).enumerate() {
            let expected = alpha * o + sigma * n;
            assert!(
                (blended[i] - expected).abs() < 1e-5,
                "blend idx {i}: got {} expected {}",
                blended[i],
                expected
            );
        }
    }
}
