//! Training augmentations: gradient accumulation, mixed-precision casts,
//! and in-place parameter updates.
//!
//! Sub-ports 3 + 4 + 5 of `docs/session-prompts/port-training-augmentations.md`.
//! Sub-ports 1 + 2 (LR schedulers + gradient clipping) ship in
//! [`crate::lazy_training_augmentations`].
//!
//! Scope of this file:
//!
//! - **Gradient accumulation** — [`GradAccumulator`] sums gradients across
//!   `microbatches` calls of [`GradAccumulator::accumulate`] and produces
//!   the mean on [`GradAccumulator::take_and_scale`].
//!
//! - **Mixed precision** — [`MixedPrecisionConfig`] + [`cast_for_forward`]
//!   + [`cast_grads_back`]. Forward dtype is typically `BF16` (compute
//!   precision); master dtype is typically `F32` (optimizer-state
//!   precision).
//!
//! - **In-place parameter update** — [`apply_inplace_sgd_step`]. Lazy
//!   `LazyTensor` does not currently expose an in-place add primitive
//!   (only the unary activations `relu_inplace` / `silu_inplace` / etc.
//!   from the Phase 4-5 in-place infrastructure). The implementation
//!   therefore uses the functional `param.sub(grad.mul_scalar(lr))` form
//!   and rebinds `*param` to the new tensor — semantically equivalent to
//!   an in-place SGD step from the caller's point of view, but graph-
//!   allocates a new node. The gap is documented inline.

use crate::lazy::LazyTensor;
use crate::Result;
use fuel_ir::DType;
use std::collections::HashMap;

// ---------- Gradient accumulation ----------

/// Sum gradient bundles across `microbatches` micro-batch passes and
/// return the mean on [`Self::take_and_scale`].
///
/// `microbatches` is required to be `>= 1`. The struct does not enforce
/// the exact accumulation count — partial fills are allowed and the
/// caller decides when to take. Scaling is always by `1.0 / microbatches`
/// (the configured target), matching the standard "average over the
/// effective batch" semantics.
#[derive(Debug)]
pub struct GradAccumulator {
    accum: HashMap<String, LazyTensor>,
    microbatches: usize,
    count: usize,
}

impl GradAccumulator {
    /// Build a fresh accumulator with target micro-batch count
    /// `microbatches`. Errors if `microbatches == 0`.
    pub fn new(microbatches: usize) -> Result<Self> {
        if microbatches == 0 {
            return Err(crate::Error::Msg(
                "GradAccumulator::new: microbatches must be >= 1".into(),
            )
            .bt());
        }
        Ok(Self {
            accum: HashMap::new(),
            microbatches,
            count: 0,
        })
    }

    /// How many micro-batches have been accumulated so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Configured target micro-batch count.
    pub fn microbatches(&self) -> usize {
        self.microbatches
    }

    /// Add a micro-batch's gradient bundle. The first call seeds the
    /// internal accumulator; subsequent calls element-wise add. Names
    /// must match across calls — a new name on a later call is an
    /// error.
    pub fn accumulate(&mut self, grads: HashMap<String, LazyTensor>) -> Result<()> {
        if self.accum.is_empty() {
            self.accum = grads;
            self.count = 1;
            return Ok(());
        }
        if grads.len() != self.accum.len() {
            return Err(crate::Error::Msg(format!(
                "GradAccumulator::accumulate: bundle size mismatch (had {} params, got {})",
                self.accum.len(),
                grads.len(),
            ))
            .bt());
        }
        for (name, g) in grads {
            let Some(prev) = self.accum.get(&name) else {
                return Err(crate::Error::Msg(format!(
                    "GradAccumulator::accumulate: unknown parameter name '{name}'"
                ))
                .bt());
            };
            let sum = prev.add(&g).map_err(|e| {
                crate::Error::Msg(format!(
                    "GradAccumulator::accumulate: add failed for '{name}': {e}"
                ))
                .bt()
            })?;
            self.accum.insert(name, sum);
        }
        self.count += 1;
        Ok(())
    }

    /// Return the accumulated gradients scaled by `1.0 / microbatches`
    /// and clear the internal state. After this call, [`Self::count`]
    /// returns `0` and a subsequent `take_and_scale` returns an empty
    /// `HashMap`.
    pub fn take_and_scale(&mut self) -> Result<HashMap<String, LazyTensor>> {
        let scale = 1.0 / self.microbatches as f64;
        let taken = std::mem::take(&mut self.accum);
        self.count = 0;
        let mut out = HashMap::with_capacity(taken.len());
        for (name, g) in taken {
            out.insert(name, g.mul_scalar(scale));
        }
        Ok(out)
    }
}

// ---------- Mixed precision ----------

/// Mixed-precision dtype pair. `forward_dtype` is what the forward pass
/// runs in (typically `BF16` or `F16`); `master_dtype` is what the
/// optimizer state and the long-lived parameter copies live in
/// (typically `F32`).
#[derive(Debug, Clone, Copy)]
pub struct MixedPrecisionConfig {
    pub forward_dtype: DType,
    pub master_dtype: DType,
}

/// Cast a master-precision parameter to forward precision for the forward
/// pass. A no-op when `param` already has `cfg.forward_dtype`.
pub fn cast_for_forward(
    param: &LazyTensor,
    cfg: &MixedPrecisionConfig,
) -> Result<LazyTensor> {
    if param.dtype() == cfg.forward_dtype {
        return Ok(param.clone());
    }
    param.to_dtype(cfg.forward_dtype).map_err(|e| {
        crate::Error::Msg(format!(
            "cast_for_forward: cast to {:?} failed: {e}",
            cfg.forward_dtype,
        ))
        .bt()
    })
}

/// Cast a gradient bundle from forward precision back to master precision
/// for the optimizer step. Values already in `cfg.master_dtype` are
/// passed through unchanged.
pub fn cast_grads_back(
    grads: HashMap<String, LazyTensor>,
    cfg: &MixedPrecisionConfig,
) -> Result<HashMap<String, LazyTensor>> {
    let mut out = HashMap::with_capacity(grads.len());
    for (name, g) in grads {
        if g.dtype() == cfg.master_dtype {
            out.insert(name, g);
            continue;
        }
        let cast = g.to_dtype(cfg.master_dtype).map_err(|e| {
            crate::Error::Msg(format!(
                "cast_grads_back: cast of '{name}' to {:?} failed: {e}",
                cfg.master_dtype,
            ))
            .bt()
        })?;
        out.insert(name, cast);
    }
    Ok(out)
}

// ---------- In-place SGD step ----------

/// Apply an in-place SGD step: `param <- param - lr * grad`.
///
/// `LazyTensor` does not currently expose a `Op::Add` / `Op::Sub`
/// in-place primitive (only the unary activations `relu_inplace` etc.
/// from the Phase 4-5 in-place infrastructure). This implementation
/// therefore uses the functional form `param = param.sub(grad.mul_scalar(lr))`
/// and rebinds `*param` to the new tensor. Semantically this matches a
/// destructive update from the caller's POV: after the call, `*param`
/// is the post-step tensor. Graph-wise, a fresh node is allocated.
///
/// When an in-place add primitive lands, swap the body for a single
/// in-place add of `grad.mul_scalar(-lr)` and remove this doc-noted gap.
pub fn apply_inplace_sgd_step(
    param: &mut LazyTensor,
    grad: &LazyTensor,
    lr: f64,
) -> Result<()> {
    let scaled = grad.mul_scalar(lr);
    let updated = param.sub(&scaled).map_err(|e| {
        crate::Error::Msg(format!("apply_inplace_sgd_step: sub failed: {e}")).bt()
    })?;
    *param = updated;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_ir::Shape;

    fn cpu_f32(values: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(values, Shape::from_dims(shape), &Device::cpu())
    }

    // ---------- Gradient accumulation ----------

    #[test]
    fn grad_accumulator_two_microbatches_average() {
        let mut acc = GradAccumulator::new(2).unwrap();
        let seed = cpu_f32(vec![1.0], &[1]);
        let mut g1 = HashMap::new();
        g1.insert("w".to_string(), seed.const_f32_like(vec![1.0_f32], Shape::from_dims(&[1])));
        acc.accumulate(g1).unwrap();
        assert_eq!(acc.count(), 1);

        let mut g2 = HashMap::new();
        g2.insert("w".to_string(), seed.const_f32_like(vec![3.0_f32], Shape::from_dims(&[1])));
        acc.accumulate(g2).unwrap();
        assert_eq!(acc.count(), 2);

        let avg = acc.take_and_scale().unwrap();
        let host = avg["w"].realize_f32();
        assert!((host[0] - 2.0).abs() < 1e-6, "averaged grad = {}", host[0]);
    }

    #[test]
    fn grad_accumulator_take_clears_state() {
        let mut acc = GradAccumulator::new(1).unwrap();
        let mut g = HashMap::new();
        g.insert("w".to_string(), cpu_f32(vec![5.0], &[1]));
        acc.accumulate(g).unwrap();
        let _ = acc.take_and_scale().unwrap();
        assert_eq!(acc.count(), 0);

        let again = acc.take_and_scale().unwrap();
        assert!(again.is_empty(), "second take returned {} entries", again.len());
    }

    #[test]
    fn grad_accumulator_zero_microbatches_errors() {
        assert!(GradAccumulator::new(0).is_err());
    }

    #[test]
    fn grad_accumulator_unknown_name_errors() {
        let mut acc = GradAccumulator::new(2).unwrap();
        let mut g1 = HashMap::new();
        g1.insert("w".to_string(), cpu_f32(vec![1.0], &[1]));
        acc.accumulate(g1).unwrap();
        let mut g2 = HashMap::new();
        g2.insert("z".to_string(), cpu_f32(vec![1.0], &[1]));
        assert!(acc.accumulate(g2).is_err());
    }

    // ---------- Mixed precision ----------

    #[test]
    fn mixed_precision_bf16_forward_roundtrips_within_bf16_tol() {
        let cfg = MixedPrecisionConfig {
            forward_dtype: DType::BF16,
            master_dtype: DType::F32,
        };
        let param = cpu_f32(vec![1.0, 0.5, -2.0, 3.14159], &[4]);
        let fwd = cast_for_forward(&param, &cfg).unwrap();
        assert_eq!(fwd.dtype(), DType::BF16);
        let host = fwd.realize_bf16();
        assert_eq!(host.len(), 4);
        let expected = [1.0_f32, 0.5, -2.0, 3.14159];
        for (got, exp) in host.iter().zip(expected.iter()) {
            let got_f32 = got.to_f32();
            let tol = 0.5_f32.max((*exp).abs() / 64.0);
            assert!(
                (got_f32 - exp).abs() <= tol,
                "bf16 roundtrip: got {got_f32}, expected {exp}, tol {tol}",
            );
        }
    }

    #[test]
    fn cast_for_forward_same_dtype_is_identity() {
        let cfg = MixedPrecisionConfig {
            forward_dtype: DType::F32,
            master_dtype: DType::F32,
        };
        let param = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let fwd = cast_for_forward(&param, &cfg).unwrap();
        assert_eq!(fwd.dtype(), DType::F32);
        assert_eq!(fwd.realize_f32(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cast_grads_back_to_master_dtype() {
        let cfg = MixedPrecisionConfig {
            forward_dtype: DType::BF16,
            master_dtype: DType::F32,
        };
        let grad_bf16 = LazyTensor::from_bf16(
            vec![
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(-2.0),
            ],
            Shape::from_dims(&[2]),
            &Device::cpu(),
        );
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), grad_bf16);
        let casted = cast_grads_back(grads, &cfg).unwrap();
        let g = &casted["w"];
        assert_eq!(g.dtype(), DType::F32);
        let host = g.realize_f32();
        assert!((host[0] - 1.0).abs() < 1e-3);
        assert!((host[1] - (-2.0)).abs() < 1e-3);
    }

    #[test]
    fn cast_grads_back_already_master_is_identity() {
        let cfg = MixedPrecisionConfig {
            forward_dtype: DType::BF16,
            master_dtype: DType::F32,
        };
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), cpu_f32(vec![1.0, 2.0], &[2]));
        let out = cast_grads_back(grads, &cfg).unwrap();
        assert_eq!(out["w"].dtype(), DType::F32);
        assert_eq!(out["w"].realize_f32(), vec![1.0, 2.0]);
    }

    // ---------- In-place SGD step ----------

    #[test]
    fn inplace_sgd_step_subtracts_lr_times_grad() {
        // param = [1.0, 2.0, 3.0]
        // grad  = [0.5, 0.5, 0.5]
        // lr    = 0.1
        // After step: param - 0.1 * grad = [0.95, 1.95, 2.95]
        let mut param = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let grad = param.const_f32_like(vec![0.5_f32, 0.5, 0.5], Shape::from_dims(&[3]));
        apply_inplace_sgd_step(&mut param, &grad, 0.1).unwrap();
        let host = param.realize_f32();
        assert_eq!(host.len(), 3);
        assert!((host[0] - 0.95).abs() < 1e-6, "host[0]={}", host[0]);
        assert!((host[1] - 1.95).abs() < 1e-6, "host[1]={}", host[1]);
        assert!((host[2] - 2.95).abs() < 1e-6, "host[2]={}", host[2]);
    }

    #[test]
    fn inplace_sgd_step_zero_lr_is_noop() {
        let mut param = cpu_f32(vec![1.0, 2.0, 3.0], &[3]);
        let grad = param.const_f32_like(vec![1.0_f32, 1.0, 1.0], Shape::from_dims(&[3]));
        apply_inplace_sgd_step(&mut param, &grad, 0.0).unwrap();
        let host = param.realize_f32();
        assert!((host[0] - 1.0).abs() < 1e-6);
        assert!((host[1] - 2.0).abs() < 1e-6);
        assert!((host[2] - 3.0).abs() < 1e-6);
    }
}
