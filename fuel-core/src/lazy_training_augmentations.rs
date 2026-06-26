//! Training augmentations: LR schedulers and gradient clipping.
//!
//! Sub-ports 1 + 2 of `docs/session-prompts/port-training-augmentations.md`.
//!
//! Scope of this file:
//!
//! - **LR schedulers** — pure host-side `f64` functions implementing
//!   the [`LrSchedule`] trait. Four canonical schedules ship today:
//!   [`CosineSchedule`], [`LinearWarmupSchedule`], [`PolynomialSchedule`],
//!   [`StepSchedule`]. Adding a new schedule is a single trait impl.
//!
//! - **Gradient clipping** — pure tensor algebra over a
//!   `HashMap<String, LazyTensor>` gradient bundle. Two variants:
//!   [`clip_grad_norm`] (global-norm L_p clipping) and
//!   [`clip_grad_value`] (elementwise clamp).
//!
//! Gradient accumulation, mixed-precision, and in-place parameter
//! update are out of scope here — they ship in follow-up commits per
//! the master plan.

use crate::lazy::LazyTensor;
use crate::Result;
use std::collections::HashMap;

/// Host-side learning-rate schedule.
///
/// `step` is the 0-indexed optimizer step about to be taken. Schedules
/// return the LR to use for that step in `f64` — the optimizer's
/// downstream conversion to `f32` happens at the call site.
pub trait LrSchedule {
    fn lr_at(&self, step: usize) -> f64;
}

/// Linear warmup followed by half-cosine decay to zero.
///
/// - `step < warmup_steps`: linear ramp `base_lr * step / warmup_steps`.
/// - `warmup_steps <= step <= total_steps`: cosine decay
///   `0.5 * base_lr * (1 + cos(pi * progress))` where
///   `progress = (step - warmup_steps) / (total_steps - warmup_steps)`.
/// - `step > total_steps`: clamped to `0`.
///
/// Edge cases: `warmup_steps == 0` skips the ramp; `total_steps ==
/// warmup_steps` makes the decay degenerate (returns `base_lr` at the
/// boundary, `0` after).
#[derive(Debug, Clone, Copy)]
pub struct CosineSchedule {
    pub base_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LrSchedule for CosineSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            return self.base_lr * (step as f64) / (self.warmup_steps as f64);
        }
        if step >= self.total_steps {
            return 0.0;
        }
        let decay_steps = self.total_steps - self.warmup_steps;
        if decay_steps == 0 {
            return self.base_lr;
        }
        let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
        0.5 * self.base_lr * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Linear warmup followed by a constant `base_lr` for all subsequent
/// steps.
///
/// `warmup_steps == 0` returns `base_lr` for every step.
#[derive(Debug, Clone, Copy)]
pub struct LinearWarmupSchedule {
    pub base_lr: f64,
    pub warmup_steps: usize,
}

impl LrSchedule for LinearWarmupSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if self.warmup_steps == 0 || step >= self.warmup_steps {
            return self.base_lr;
        }
        self.base_lr * (step as f64) / (self.warmup_steps as f64)
    }
}

/// Polynomial decay: `base_lr * (1 - step/total_steps)^power`, clamped
/// to `>= 0`. Steps past `total_steps` return `0`.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialSchedule {
    pub base_lr: f64,
    pub total_steps: usize,
    pub power: f64,
}

impl LrSchedule for PolynomialSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if self.total_steps == 0 || step >= self.total_steps {
            return 0.0;
        }
        let remaining = 1.0 - (step as f64) / (self.total_steps as f64);
        let v = self.base_lr * remaining.powf(self.power);
        if v < 0.0 { 0.0 } else { v }
    }
}

/// Step decay: multiply `base_lr` by `gamma` after each milestone.
///
/// At step `s`, the returned LR is
/// `base_lr * gamma^count(milestones[i] <= s)`. Milestones do not need
/// to be sorted — the count is taken over the raw slice.
#[derive(Debug, Clone)]
pub struct StepSchedule {
    pub base_lr: f64,
    pub milestones: Vec<usize>,
    pub gamma: f64,
}

impl LrSchedule for StepSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        let crossed = self.milestones.iter().filter(|&&m| step >= m).count();
        self.base_lr * self.gamma.powi(crossed as i32)
    }
}

/// Clip the global L_p norm of a gradient bundle to `max_norm`.
///
/// `norm_type` selects p: finite values use `(sum |x|^p)^(1/p)`;
/// [`f64::INFINITY`] uses `max |x|`. The standard LLM-training choice
/// is `norm_type = 2.0`.
///
/// Returns a new `HashMap` whose values are either the input gradients
/// untouched (when `total_norm <= max_norm`) or each input scaled by
/// `max_norm / total_norm`. The check is host-side: the total norm is
/// realized as an `f32` scalar via `realize_f32` and compared eagerly,
/// so the lazy graph downstream stays branch-free.
///
/// Errors when `max_norm <= 0.0`, `norm_type <= 0.0`, or `grads` is
/// empty.
pub fn clip_grad_norm(
    grads: &HashMap<String, LazyTensor>,
    max_norm: f64,
    norm_type: f64,
) -> Result<HashMap<String, LazyTensor>> {
    if grads.is_empty() {
        return Err(crate::Error::Msg(
            "clip_grad_norm: gradient bundle is empty".into(),
        )
        .bt());
    }
    if !(max_norm > 0.0) {
        return Err(crate::Error::Msg(format!(
            "clip_grad_norm: max_norm must be > 0, got {max_norm}",
        ))
        .bt());
    }
    if !(norm_type > 0.0) {
        return Err(crate::Error::Msg(format!(
            "clip_grad_norm: norm_type must be > 0, got {norm_type}",
        ))
        .bt());
    }

    // Host-side total-norm computation. We realize per-tensor reductions
    // to host scalars and aggregate there:
    //   * L2 (the LLM-training default): one `sqr().sum_all()` per tensor
    //     stays fully in-graph and produces an f32 scalar.
    //   * L1: one `abs().sum_all()` per tensor.
    //   * Linf: one `abs().reduce_max_to([])` per tensor; host takes the
    //     running maximum.
    //   * General Lp (any other p): realize `|g|` and compute the
    //     `|x|^p` aggregation on the host. Avoids requiring a graph-side
    //     `powf` with a scalar exponent.
    //
    // Keeping the cross-tensor reduction on the host gives us a clean
    // host-side branch for the no-clip-needed case.
    let is_inf = norm_type.is_infinite();
    let total_norm: f64 = if is_inf {
        let mut best: f64 = 0.0;
        for g in grads.values() {
            let abs_host = g.abs().realize_f32();
            for x in abs_host {
                let v = x as f64;
                if v > best {
                    best = v;
                }
            }
        }
        best
    } else if (norm_type - 2.0).abs() < f64::EPSILON {
        let mut acc: f64 = 0.0;
        for g in grads.values() {
            let host = g.sqr().sum_all().realize_f32();
            acc += host[0] as f64;
        }
        acc.sqrt()
    } else if (norm_type - 1.0).abs() < f64::EPSILON {
        let mut acc: f64 = 0.0;
        for g in grads.values() {
            let host = g.abs().sum_all().realize_f32();
            acc += host[0] as f64;
        }
        acc
    } else {
        let mut acc: f64 = 0.0;
        for g in grads.values() {
            let abs_host = g.abs().realize_f32();
            for x in abs_host {
                acc += (x as f64).powf(norm_type);
            }
        }
        acc.powf(1.0 / norm_type)
    };

    // No-op fast path when already within budget.
    if total_norm <= max_norm {
        return Ok(grads
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect());
    }

    let scale = max_norm / total_norm;
    let mut out = HashMap::with_capacity(grads.len());
    for (name, g) in grads {
        out.insert(name.clone(), g.mul_scalar(scale));
    }
    Ok(out)
}

/// Elementwise clamp every gradient to `[-clip_value, clip_value]`.
///
/// Errors when `clip_value < 0.0`.
pub fn clip_grad_value(
    grads: &HashMap<String, LazyTensor>,
    clip_value: f64,
) -> Result<HashMap<String, LazyTensor>> {
    if clip_value < 0.0 {
        return Err(crate::Error::Msg(format!(
            "clip_grad_value: clip_value must be >= 0, got {clip_value}",
        ))
        .bt());
    }
    let mut out = HashMap::with_capacity(grads.len());
    for (name, g) in grads {
        out.insert(name.clone(), g.clamp(-clip_value, clip_value));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_ir::Shape;

    // ---------- LR schedule tests ----------

    #[test]
    fn cosine_schedule_warmup_then_decay() {
        let sched = CosineSchedule {
            base_lr: 1.0,
            warmup_steps: 10,
            total_steps: 100,
        };
        assert!((sched.lr_at(0) - 0.0).abs() < 1e-12);
        assert!((sched.lr_at(10) - 1.0).abs() < 1e-12);
        assert!((sched.lr_at(100) - 0.0).abs() < 1e-12);
        // Linear ramp midpoint:
        assert!((sched.lr_at(5) - 0.5).abs() < 1e-12);
        // Cosine midpoint (progress = 0.5): 0.5 * 1 * (1 + cos(pi/2)) = 0.5
        let mid = 10 + (100 - 10) / 2;
        assert!((sched.lr_at(mid) - 0.5).abs() < 1e-12);
        // Steps past total clamp to 0.
        assert!((sched.lr_at(200) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn linear_warmup_schedule_linear_ramp() {
        let sched = LinearWarmupSchedule {
            base_lr: 2.0,
            warmup_steps: 4,
        };
        assert!((sched.lr_at(0) - 0.0).abs() < 1e-12);
        assert!((sched.lr_at(1) - 0.5).abs() < 1e-12);
        assert!((sched.lr_at(2) - 1.0).abs() < 1e-12);
        assert!((sched.lr_at(3) - 1.5).abs() < 1e-12);
        assert!((sched.lr_at(4) - 2.0).abs() < 1e-12);
        assert!((sched.lr_at(50) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn polynomial_schedule_power_2_decays_quadratically() {
        let sched = PolynomialSchedule {
            base_lr: 1.0,
            total_steps: 10,
            power: 2.0,
        };
        // step 0 → 1.0 * (1 - 0/10)^2 = 1.0
        assert!((sched.lr_at(0) - 1.0).abs() < 1e-12);
        // step 5 → (0.5)^2 = 0.25
        assert!((sched.lr_at(5) - 0.25).abs() < 1e-12);
        // step 9 → (0.1)^2 = 0.01
        assert!((sched.lr_at(9) - 0.01).abs() < 1e-12);
        // step 10 → 0
        assert!((sched.lr_at(10) - 0.0).abs() < 1e-12);
        assert!((sched.lr_at(99) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn step_schedule_drops_at_milestones() {
        let sched = StepSchedule {
            base_lr: 1.0,
            milestones: vec![10, 20, 30],
            gamma: 0.1,
        };
        assert!((sched.lr_at(0) - 1.0).abs() < 1e-12);
        assert!((sched.lr_at(9) - 1.0).abs() < 1e-12);
        // At step 10, exactly one milestone passed → gamma^1 = 0.1
        assert!((sched.lr_at(10) - 0.1).abs() < 1e-12);
        assert!((sched.lr_at(19) - 0.1).abs() < 1e-12);
        // At step 20, two milestones passed → gamma^2 = 0.01
        assert!((sched.lr_at(20) - 0.01).abs() < 1e-12);
        // At step 30, three → gamma^3 = 0.001
        assert!((sched.lr_at(30) - 0.001).abs() < 1e-12);
        assert!((sched.lr_at(1000) - 0.001).abs() < 1e-12);
    }

    // ---------- Gradient clipping tests ----------

    fn make_grad(values: Vec<f32>, shape: &[usize]) -> LazyTensor {
        LazyTensor::from_f32(values, Shape::from_dims(shape), &Device::cpu())
    }

    #[test]
    fn clip_grad_norm_total_under_max_is_identity() {
        // grads with L2 norm = sqrt(1 + 4 + 4) = 3.0, max_norm = 5.0.
        let mut grads = HashMap::new();
        grads.insert("a".to_string(), make_grad(vec![1.0_f32], &[1]));
        grads.insert("b".to_string(), make_grad(vec![2.0_f32, 2.0], &[2]));
        let clipped = clip_grad_norm(&grads, 5.0, 2.0).unwrap();
        for (name, g) in &clipped {
            let original = grads[name].realize_f32();
            let new = g.realize_f32();
            assert_eq!(original.len(), new.len(), "shape preserved for {name}");
            for (a, b) in original.iter().zip(new.iter()) {
                assert!((a - b).abs() < 1e-6, "no scaling for {name}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn clip_grad_norm_total_over_max_scales_to_max() {
        // tensor a = [3, 4]  → ||a||_2^2 = 25
        // tensor b = [12]    → ||b||_2^2 = 144
        // total ||.||_2 = sqrt(25 + 144) = sqrt(169) = 13.
        // max_norm = 6.5 → scale = 6.5 / 13 = 0.5 → halve every grad.
        let mut grads = HashMap::new();
        grads.insert("a".to_string(), make_grad(vec![3.0_f32, 4.0], &[2]));
        grads.insert("b".to_string(), make_grad(vec![12.0_f32], &[1]));
        let clipped = clip_grad_norm(&grads, 6.5, 2.0).unwrap();
        let a_out = clipped["a"].realize_f32();
        let b_out = clipped["b"].realize_f32();
        assert!((a_out[0] - 1.5).abs() < 1e-5, "a[0] = {}", a_out[0]);
        assert!((a_out[1] - 2.0).abs() < 1e-5, "a[1] = {}", a_out[1]);
        assert!((b_out[0] - 6.0).abs() < 1e-5, "b[0] = {}", b_out[0]);

        // And the new total L2 norm is now == max_norm.
        let new_total_sq = a_out.iter().chain(b_out.iter()).map(|x| (x * x) as f64).sum::<f64>();
        let new_total = new_total_sq.sqrt();
        assert!((new_total - 6.5).abs() < 1e-5, "clipped norm = {new_total}");
    }

    #[test]
    fn clip_grad_value_clamps_elementwise() {
        let mut grads = HashMap::new();
        grads.insert(
            "w".to_string(),
            make_grad(vec![-5.0_f32, -1.0, 0.0, 1.0, 5.0], &[5]),
        );
        let clipped = clip_grad_value(&grads, 1.0).unwrap();
        let w = clipped["w"].realize_f32();
        assert_eq!(w.len(), 5);
        assert!((w[0] - (-1.0)).abs() < 1e-6, "w[0]={}", w[0]);
        assert!((w[1] - (-1.0)).abs() < 1e-6, "w[1]={}", w[1]);
        assert!((w[2] - 0.0).abs() < 1e-6, "w[2]={}", w[2]);
        assert!((w[3] - 1.0).abs() < 1e-6, "w[3]={}", w[3]);
        assert!((w[4] - 1.0).abs() < 1e-6, "w[4]={}", w[4]);
    }
}
