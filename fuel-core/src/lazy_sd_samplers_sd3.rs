//! Stable Diffusion 3 / 3.5 flow-match Euler sampler with Skip Layer
//! Guidance — lazy-graph port.
//!
//! Ports the eager `euler_sample` from
//! `fuel-examples/examples/_stable-diffusion-3_retired/sampling.rs`
//! (eager source recovered from git history at
//! `cfcb35cf~1:fuel-examples/examples/stable-diffusion-3/sampling.rs`,
//! 84 LOC).
//!
//! Three SD3-specific knobs distinguish this from
//! [`crate::lazy_sd_samplers_euler`]:
//!
//! - **time_snr_shift** — non-linear remap of the linear `t` schedule
//!   `t → α·t / (1 + (α − 1)·t)`. Default `α = 3.0` for SD3.5-medium.
//!   Paper: "Resolution-dependent shifting of timestep schedules"
//!   (SD3 tech report — <https://arxiv.org/pdf/2403.03206>).
//! - **Classifier-Free Guidance (CFG)** — standard
//!   `cfg_scale · cond − (cfg_scale − 1) · uncond`, implemented by
//!   stacking `(cond, uncond)` along the batch dim and `narrow`-ing
//!   apart the model's two outputs.
//! - **Skip Layer Guidance (SLG)** — for SD 3.5-medium. Within a
//!   `[start, end]` fractional-step window, runs the denoiser a
//!   third time with a subset of DoubleStream blocks skipped, and
//!   pushes guidance toward the non-skipped output via
//!   `+slg_scale · (cond − cond_skipped)`. Reference:
//!   <https://github.com/Stability-AI/sd3.5/blob/4e484e05/sd3_infer.py#L388-L394>.
//!
//! # Model trait
//!
//! The sampler is generic over any denoiser that exposes the SD3
//! shape — `forward(latent, timestep, y, context, skip_layers)`. The
//! eventual `MmDitFullModel` (forthcoming with the §1.3 mmdit
//! extension in `docs/session-prompts/lazy-sd3-port.md`) will
//! implement [`Sd3Denoiser`]; pre-MmDitFullModel callers can wire any
//! shape-compatible model.
//!
//! # RNG ownership
//!
//! Following the existing scheduler convention, the caller supplies
//! the initial `latent` tensor (the seeded `x_T` noise). The sampler
//! holds no RNG state. Pass a noise tensor constructed with
//! [`LazyTensor::const_f32_like`] (or any of the typed equivalents)
//! seeded from whatever RNG the binary owns.

use crate::lazy::LazyTensor;
use crate::{Error, Result};
use fuel_ir::Shape;

/// Skip-Layer-Guidance + CFG + time-shift configuration for the SD3
/// flow-match Euler sampler.
///
/// Defaults match the SD 3.5-medium reference implementation:
/// `num_steps = 28`, `time_snr_shift = 3.0`, `guidance_scale = 4.5`,
/// `slg_layers = [7, 8, 9]`, `slg_scale = 2.5`, `slg_start = 0.01`,
/// `slg_end = 0.20`.
#[derive(Debug, Clone)]
pub struct Sd3SamplerConfig {
    /// Number of Euler integration steps.
    pub num_steps: usize,
    /// SD3 timestep-shift α. `1.0` reduces to the linear flow-match
    /// schedule; SD3.5-medium uses `3.0`.
    pub time_snr_shift: f64,
    /// CFG weight. Standard `cfg_scale · cond − (cfg_scale − 1) ·
    /// uncond`. `1.0` disables CFG; SD 3.5-medium default is `4.5`.
    pub guidance_scale: f64,
    /// DoubleStream block indices to skip during the SLG pass. Empty
    /// (or [`Sd3SamplerConfig::slg_disabled`]) disables SLG.
    pub slg_layers: Vec<usize>,
    /// SLG strength. The SLG contribution is
    /// `+slg_scale · (cond − cond_skipped)`.
    pub slg_scale: f64,
    /// SLG window start as a fraction of `num_steps`. The SLG pass is
    /// only run when `num_steps · slg_start < step < num_steps ·
    /// slg_end`.
    pub slg_start: f64,
    /// SLG window end as a fraction of `num_steps`.
    pub slg_end: f64,
}

impl Default for Sd3SamplerConfig {
    fn default() -> Self {
        // SD 3.5-medium defaults; see
        // <https://github.com/Stability-AI/sd3.5/blob/4e484e05/sd3_infer.py>.
        Self {
            num_steps: 28,
            time_snr_shift: 3.0,
            guidance_scale: 4.5,
            slg_layers: vec![7, 8, 9],
            slg_scale: 2.5,
            slg_start: 0.01,
            slg_end: 0.20,
        }
    }
}

impl Sd3SamplerConfig {
    /// Build a config with SLG disabled (empty `slg_layers`). Used for
    /// SD3-medium and SD 3.5-large which do not require SLG.
    pub fn slg_disabled(num_steps: usize, time_snr_shift: f64, guidance_scale: f64) -> Self {
        Self {
            num_steps,
            time_snr_shift,
            guidance_scale,
            slg_layers: Vec::new(),
            slg_scale: 0.0,
            slg_start: 0.0,
            slg_end: 0.0,
        }
    }

    /// True iff this config enables SLG (non-empty layer list and the
    /// `[start, end]` window is non-empty).
    pub fn slg_active(&self) -> bool {
        !self.slg_layers.is_empty() && self.slg_end > self.slg_start
    }
}

/// Denoiser shape consumed by [`flow_match_euler_sample`]. The eventual
/// `lazy_mmdit::MmDitFullModel` (per §1.3 of the SD3 port plan) will
/// implement this trait once the patchify / unpatchify wrappers land;
/// the trait keeps the sampler decoupled from the not-yet-shipped
/// struct so this sub-port can land independently.
///
/// Implementations must respect these shape contracts:
///
/// - `latent`: `(B, C, H, W)` — already in image space; the model
///   handles patchification internally.
/// - `timestep`: `(B,)` — diffusion-step scalar per batch element.
/// - `y`: `(B, adm_in_channels)` — pooled text conditioning.
/// - `context`: `(B, S_text, context_embed_size)` — per-token text
///   conditioning.
/// - `skip_layers`: optional set of DoubleStream block indices to
///   skip; passing `None` runs every block.
///
/// Output shape must equal `latent`'s shape.
pub trait Sd3Denoiser {
    fn forward(
        &self,
        latent: &LazyTensor,
        timestep: &LazyTensor,
        y: &LazyTensor,
        context: &LazyTensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<LazyTensor>;
}

/// SD3 flow-match Euler sampler with classifier-free guidance and
/// optional Skip-Layer Guidance.
///
/// Mirrors the eager `euler_sample` line-for-line (recovered from git
/// history at `cfcb35cf~1:fuel-examples/examples/stable-diffusion-3/
/// sampling.rs`).
///
/// # Arguments
///
/// - `model`: any [`Sd3Denoiser`]. Will be `&MmDitFullModel` once the
///   §1.3 mmdit extension lands; named here as a trait to allow
///   independent landing.
/// - `latent`: initial noise `(1, 16, H/8, W/8)` for SD3 latent space.
///   The caller seeds this from whatever RNG it owns (typically
///   `lazy_flux::sampling::get_noise` cast to F16).
/// - `context`: positive per-token conditioning `(1, 154, 4096)`.
/// - `y`: positive pooled conditioning `(1, 2048)`.
/// - `neg_context`: negative per-token conditioning, same shape as
///   `context`.
/// - `neg_y`: negative pooled conditioning, same shape as `y`.
/// - `config`: timestep-shift, CFG, and SLG knobs.
///
/// # Returns
///
/// The final denoised latent, shape == `latent.shape()`, ready for
/// VAE decode.
#[allow(clippy::too_many_arguments)]
pub fn flow_match_euler_sample<M: Sd3Denoiser>(
    model: &M,
    latent: LazyTensor,
    context: LazyTensor,
    y: LazyTensor,
    config: &Sd3SamplerConfig,
    neg_context: LazyTensor,
    neg_y: LazyTensor,
) -> Result<LazyTensor> {
    if config.num_steps == 0 {
        return Err(Error::Msg(
            "flow_match_euler_sample: num_steps must be > 0".into(),
        )
        .bt());
    }
    let latent_dims = latent.shape().dims().to_vec();
    if latent_dims.is_empty() || latent_dims[0] != 1 {
        return Err(Error::Msg(format!(
            "flow_match_euler_sample: latent batch dim must be 1, got shape {:?}",
            latent_dims,
        ))
        .bt());
    }
    // CFG conditioning must already be batched at B == 1 (we stack
    // along batch ourselves below).
    let y_dims = y.shape().dims().to_vec();
    if y_dims.first() != Some(&1) {
        return Err(Error::Msg(format!(
            "flow_match_euler_sample: y batch dim must be 1, got shape {:?}",
            y_dims,
        ))
        .bt());
    }
    let ctx_dims = context.shape().dims().to_vec();
    if ctx_dims.first() != Some(&1) {
        return Err(Error::Msg(format!(
            "flow_match_euler_sample: context batch dim must be 1, got shape {:?}",
            ctx_dims,
        ))
        .bt());
    }
    if neg_y.shape().dims() != y.shape().dims() {
        return Err(Error::Msg(format!(
            "flow_match_euler_sample: neg_y shape {:?} != y shape {:?}",
            neg_y.shape().dims(),
            y.shape().dims(),
        ))
        .bt());
    }
    if neg_context.shape().dims() != context.shape().dims() {
        return Err(Error::Msg(format!(
            "flow_match_euler_sample: neg_context shape {:?} != context shape {:?}",
            neg_context.shape().dims(),
            context.shape().dims(),
        ))
        .bt());
    }

    let sigmas = sigma_schedule(config.num_steps, config.time_snr_shift);

    // Pre-stack the CFG conditioning along the batch dim: row 0 is
    // the positive branch, row 1 is the negative branch. Mirrors the
    // eager binary's `Tensor::cat([context, context_uncond], 0)`.
    let context_cfg = context.concat(&neg_context, 0_usize)?;
    let y_cfg = y.concat(&neg_y, 0_usize)?;

    let slg_active = config.slg_active();
    let slg_lo = (config.num_steps as f64) * config.slg_start;
    let slg_hi = (config.num_steps as f64) * config.slg_end;

    let mut x = latent;
    for (step, window) in sigmas.windows(2).enumerate() {
        let s_curr = window[0];
        let s_prev = window[1];

        // Eager scales σ ∈ [0,1] up to the model's expected
        // [0, 1000] timestep range.
        let timestep_val = s_curr * 1000.0;

        // ----- CFG pass: stack `x` twice along batch, run, split. ----
        let x_pair = x.concat(&x, 0_usize)?;
        let t_pair = x.const_f32_like(
            vec![timestep_val as f32; 2],
            Shape::from_dims(&[2]),
        );
        let noise_pred = model.forward(&x_pair, &t_pair, &y_cfg, &context_cfg, None)?;
        let pred_cond = noise_pred.narrow(0_usize, 0, 1)?;
        let pred_uncond = noise_pred.narrow(0_usize, 1, 1)?;

        // guidance = cfg · cond − (cfg − 1) · uncond
        let mut guidance = pred_cond
            .mul_scalar(config.guidance_scale)
            .sub(&pred_uncond.mul_scalar(config.guidance_scale - 1.0))?;

        // ----- SLG pass: run a third time on the positive branch, ---
        // skipping the configured layers, only inside the window.
        if slg_active && (step as f64) > slg_lo && (step as f64) < slg_hi {
            let t_single = x.const_f32_like(
                vec![timestep_val as f32; 1],
                Shape::from_dims(&[1]),
            );
            let slg_pred = model.forward(
                &x,
                &t_single,
                &y,
                &context,
                Some(&config.slg_layers),
            )?;
            // +slg_scale · (cond − cond_skipped)
            let delta = pred_cond.sub(&slg_pred)?.mul_scalar(config.slg_scale);
            guidance = guidance.add(&delta)?;
        }

        // x ← x + guidance · (σ_prev − σ_curr).  σ is descending so
        // (σ_prev − σ_curr) is negative — this is the standard
        // flow-match Euler step.
        let dt = s_prev - s_curr;
        x = x.add(&guidance.mul_scalar(dt))?;
    }

    Ok(x)
}

/// Build the descending σ schedule for `num_steps` flow-match Euler
/// integration steps with the SD3 SNR shift applied.
///
/// Returns `num_steps + 1` values, descending from
/// `time_snr_shift(α, 1.0)` (==1.0 for any α) to
/// `time_snr_shift(α, 0.0) == 0.0`. Adjacent windows
/// `[σ_curr, σ_prev]` feed each Euler step.
pub fn sigma_schedule(num_steps: usize, time_snr_shift_alpha: f64) -> Vec<f64> {
    (0..=num_steps)
        .map(|i| (num_steps - i) as f64 / num_steps as f64)
        .map(|t| time_snr_shift(time_snr_shift_alpha, t))
        .collect()
}

/// SD3 timestep shift `α · t / (1 + (α − 1) · t)`. With `α = 1.0`
/// this is the identity; with `α > 1` it pushes samples toward
/// higher-noise (larger σ) regions of the schedule.
///
/// Reference: SD3 paper "Resolution-dependent shifting of timestep
/// schedules" (<https://arxiv.org/pdf/2403.03206>) and ComfyUI
/// `comfy/model_sampling.py#L181`.
pub fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    /// Dummy denoiser used for shape / panic tests. Returns the input
    /// `latent` scaled by a tiny constant so the sampler exercises the
    /// `narrow` + `concat` + `mul_scalar` + `add` paths but never
    /// blows up.
    struct IdentityDenoiser;

    impl Sd3Denoiser for IdentityDenoiser {
        fn forward(
            &self,
            latent: &LazyTensor,
            _timestep: &LazyTensor,
            _y: &LazyTensor,
            _context: &LazyTensor,
            _skip_layers: Option<&[usize]>,
        ) -> Result<LazyTensor> {
            // Return a tiny multiple of the input so the sampler has
            // a non-zero `guidance` to integrate but the result stays
            // bounded.
            Ok(latent.mul_scalar(0.01))
        }
    }

    /// Denoiser that tracks how many times the SLG branch (with
    /// `Some(skip_layers)`) was taken.
    struct CountingDenoiser {
        slg_calls: std::cell::Cell<usize>,
    }

    impl Sd3Denoiser for CountingDenoiser {
        fn forward(
            &self,
            latent: &LazyTensor,
            _timestep: &LazyTensor,
            _y: &LazyTensor,
            _context: &LazyTensor,
            skip_layers: Option<&[usize]>,
        ) -> Result<LazyTensor> {
            if skip_layers.is_some() {
                self.slg_calls.set(self.slg_calls.get() + 1);
            }
            Ok(latent.mul_scalar(0.01))
        }
    }

    /// σ schedule is monotonically descending, starts at exactly 1,
    /// ends at exactly 0, and reduces to the linear schedule when
    /// `α = 1`.
    #[test]
    fn sigma_schedule_correctness() {
        // α = 1 → identity → linear descending [1, 1-1/N, ..., 0].
        let linear = sigma_schedule(8, 1.0);
        assert_eq!(linear.len(), 9);
        assert!((linear[0] - 1.0).abs() < 1e-12, "linear[0] = {}", linear[0]);
        assert!(linear.last().unwrap().abs() < 1e-12);
        for (i, &v) in linear.iter().enumerate() {
            let expected = (8 - i) as f64 / 8.0;
            assert!(
                (v - expected).abs() < 1e-12,
                "linear σ[{i}] = {v} expected {expected}",
            );
        }

        // α = 3.0 (SD3.5-medium) → endpoints fixed at 1.0 / 0.0,
        // monotone descending, midpoint pushed above the linear
        // value (the shift biases toward higher σ).
        let shifted = sigma_schedule(8, 3.0);
        assert_eq!(shifted.len(), 9);
        assert!((shifted[0] - 1.0).abs() < 1e-12, "shifted[0] = {}", shifted[0]);
        assert!(shifted.last().unwrap().abs() < 1e-12);
        for w in shifted.windows(2) {
            assert!(
                w[0] > w[1],
                "shifted σ not strictly descending: {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
        // Midpoint t = 0.5 → 3·0.5 / (1 + 2·0.5) = 1.5 / 2.0 = 0.75
        // vs linear 0.5 — confirm the shift pushes upward.
        let mid_linear = 0.5;
        let mid_shifted = time_snr_shift(3.0, 0.5);
        assert!(
            mid_shifted > mid_linear,
            "shifted midpoint {} should exceed linear {}",
            mid_shifted,
            mid_linear
        );
        assert!(
            (mid_shifted - 0.75).abs() < 1e-12,
            "α=3, t=0.5 → {} expected 0.75",
            mid_shifted
        );

        // α = 0 is degenerate (numerator = 0) but the call still
        // produces zeros for non-endpoint t (and 1.0 at the t=1
        // endpoint since 0/(1 + -1·1) = 0 / 0 — guard via not
        // testing α=0; the SD3 spec disallows it).

        // Endpoint identities at any α:
        // time_snr_shift(α, 1) = α / α = 1.
        // time_snr_shift(α, 0) = 0 / 1 = 0.
        for &alpha in &[1.5_f64, 2.0, 3.0, 5.0, 10.0] {
            assert!((time_snr_shift(alpha, 1.0) - 1.0).abs() < 1e-12);
            assert!(time_snr_shift(alpha, 0.0).abs() < 1e-12);
        }
    }

    /// End-to-end smoke test: the sampler runs without panicking on a
    /// tiny fixture, returns a tensor with the input latent's shape,
    /// and produces finite values. Also verifies the SLG branch is
    /// taken inside the configured window and skipped outside.
    #[test]
    fn flow_match_euler_runs_on_tiny_fixture() {
        let device = Device::cpu();
        // Tiny SD3-shaped latent: B=1, C=4, H=2, W=2 (the channel
        // count doesn't matter for the IdentityDenoiser).
        let latent = LazyTensor::from_f32(
            vec![0.1_f32; 16],
            Shape::from_dims(&[1, 4, 2, 2]),
            &device,
        );
        let context = latent.const_f32_like(
            vec![0.2_f32; 48],
            Shape::from_dims(&[1, 6, 8]),
        );
        let y = latent.const_f32_like(vec![0.3_f32; 8], Shape::from_dims(&[1, 8]));
        let neg_context = latent.const_f32_like(
            vec![-0.2_f32; 48],
            Shape::from_dims(&[1, 6, 8]),
        );
        let neg_y =
            latent.const_f32_like(vec![-0.3_f32; 8], Shape::from_dims(&[1, 8]));

        // SLG disabled: just run a few steps end-to-end.
        let cfg_no_slg = Sd3SamplerConfig::slg_disabled(4, 3.0, 4.5);
        let denoiser = IdentityDenoiser;
        let out = flow_match_euler_sample(
            &denoiser,
            latent.clone(),
            context.clone(),
            y.clone(),
            &cfg_no_slg,
            neg_context.clone(),
            neg_y.clone(),
        )
        .unwrap();
        assert_eq!(
            out.shape().dims(),
            &[1, 4, 2, 2],
            "output shape mismatch",
        );
        let realized = out.realize_f32();
        assert_eq!(realized.len(), 16);
        assert!(
            realized.iter().all(|v| v.is_finite()),
            "non-finite output with SLG off: {:?}",
            realized
        );

        // SLG enabled with a wide window so several steps fire.
        let cfg_slg = Sd3SamplerConfig {
            num_steps: 8,
            time_snr_shift: 3.0,
            guidance_scale: 4.5,
            slg_layers: vec![0],
            slg_scale: 2.5,
            slg_start: 0.0,
            slg_end: 1.0,
        };
        let counting = CountingDenoiser { slg_calls: std::cell::Cell::new(0) };
        let out = flow_match_euler_sample(
            &counting,
            latent.clone(),
            context.clone(),
            y.clone(),
            &cfg_slg,
            neg_context.clone(),
            neg_y.clone(),
        )
        .unwrap();
        assert_eq!(out.shape().dims(), &[1, 4, 2, 2]);
        let realized = out.realize_f32();
        assert!(
            realized.iter().all(|v| v.is_finite()),
            "non-finite output with SLG on: {:?}",
            realized
        );
        // With slg_start=0.0 and slg_end=1.0 the condition is
        // `0 < step < 8` for `num_steps = 8`, i.e. steps 1..=7
        // (7 SLG calls).
        assert_eq!(
            counting.slg_calls.get(),
            7,
            "SLG should fire on 7 of 8 steps (window covers steps 1..=7)",
        );

        // SLG disabled (empty layer list): no SLG calls.
        let counting = CountingDenoiser { slg_calls: std::cell::Cell::new(0) };
        let cfg_disabled = Sd3SamplerConfig::slg_disabled(4, 3.0, 4.5);
        let _ = flow_match_euler_sample(
            &counting,
            latent.clone(),
            context.clone(),
            y.clone(),
            &cfg_disabled,
            neg_context.clone(),
            neg_y.clone(),
        )
        .unwrap();
        assert_eq!(
            counting.slg_calls.get(),
            0,
            "SLG must not fire when slg_layers is empty",
        );
    }

    /// Empty `num_steps` is rejected at build time, not silently
    /// returning an unmodified latent (validate-at-graph-build).
    #[test]
    fn zero_num_steps_errors() {
        let device = Device::cpu();
        let latent = LazyTensor::from_f32(
            vec![0.0_f32; 4],
            Shape::from_dims(&[1, 1, 2, 2]),
            &device,
        );
        let context =
            latent.const_f32_like(vec![0.0_f32; 4], Shape::from_dims(&[1, 1, 4]));
        let y = latent.const_f32_like(vec![0.0_f32; 4], Shape::from_dims(&[1, 4]));
        let neg_context = context.clone();
        let neg_y = y.clone();
        let cfg = Sd3SamplerConfig::slg_disabled(0, 3.0, 4.5);
        let denoiser = IdentityDenoiser;
        let err = flow_match_euler_sample(
            &denoiser,
            latent,
            context,
            y,
            &cfg,
            neg_context,
            neg_y,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("num_steps must be > 0"),
            "unexpected error message: {msg}",
        );
    }

    /// Shape-mismatched negatives are rejected at build time.
    #[test]
    fn neg_shape_mismatch_errors() {
        let device = Device::cpu();
        let latent = LazyTensor::from_f32(
            vec![0.0_f32; 4],
            Shape::from_dims(&[1, 1, 2, 2]),
            &device,
        );
        let context =
            latent.const_f32_like(vec![0.0_f32; 6], Shape::from_dims(&[1, 2, 3]));
        let y = latent.const_f32_like(vec![0.0_f32; 4], Shape::from_dims(&[1, 4]));
        // Wrong context shape.
        let bad_neg_context =
            latent.const_f32_like(vec![0.0_f32; 4], Shape::from_dims(&[1, 1, 4]));
        let neg_y = y.clone();
        let cfg = Sd3SamplerConfig::slg_disabled(2, 3.0, 4.5);
        let denoiser = IdentityDenoiser;
        let err = flow_match_euler_sample(
            &denoiser,
            latent,
            context,
            y,
            &cfg,
            bad_neg_context,
            neg_y,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("neg_context shape"),
            "unexpected error message: {msg}",
        );
    }
}
