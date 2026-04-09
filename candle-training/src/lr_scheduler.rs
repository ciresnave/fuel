//! Learning rate schedulers for training.
//!
//! Schedulers adjust the learning rate over the course of training, typically
//! reducing it according to a predefined policy. All schedulers implement the
//! [`LrScheduler`] trait.
//!
//! # Example
//!
//! ```rust
//! use candle_training::lr_scheduler::{LrScheduler, CosineAnnealingLr};
//!
//! let mut sched = CosineAnnealingLr::new(0.001, 1000);
//! assert!((sched.lr() - 0.001).abs() < 1e-9);
//!
//! // Advance 500 steps (halfway through cosine cycle)
//! for _ in 0..500 {
//!     sched.step();
//! }
//! // At halfway, cosine annealing yields ~half the initial LR
//! assert!(sched.lr() < 0.001);
//! assert!(sched.lr() > 0.0);
//! ```

use std::f64::consts::PI;

/// The interface all learning rate schedulers implement.
///
/// # Usage
///
/// Call [`step()`](LrScheduler::step) once per optimizer step, then apply the
/// new learning rate to your optimizer via
/// [`Optimizer::set_learning_rate`](candle_nn::Optimizer::set_learning_rate).
pub trait LrScheduler {
    /// Return the current learning rate.
    fn lr(&self) -> f64;

    /// Advance the scheduler by one step and update the internal learning rate.
    fn step(&mut self);

    /// Return the current step count (number of times `step()` has been called).
    fn current_step(&self) -> usize;
}

// ─── Constant LR ─────────────────────────────────────────────────────────────

/// A no-op scheduler that returns a fixed learning rate.
///
/// Useful as a baseline or when you want to opt out of scheduling without
/// changing the training loop structure.
///
/// # Example
///
/// ```rust
/// use candle_training::lr_scheduler::{LrScheduler, ConstantLr};
///
/// let mut sched = ConstantLr::new(0.01);
/// sched.step();
/// assert!((sched.lr() - 0.01).abs() < 1e-12);
/// ```
pub struct ConstantLr {
    lr: f64,
    current_step: usize,
}

impl ConstantLr {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            current_step: 0,
        }
    }
}

impl LrScheduler for ConstantLr {
    fn lr(&self) -> f64 {
        self.lr
    }
    fn step(&mut self) {
        self.current_step += 1;
    }
    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ─── Step Decay ──────────────────────────────────────────────────────────────

/// Multiply the learning rate by `gamma` every `step_size` steps.
///
/// This is equivalent to PyTorch's `StepLR`.
///
/// # Example
///
/// ```rust
/// use candle_training::lr_scheduler::{LrScheduler, StepLr};
///
/// let mut sched = StepLr::new(0.1, 30, 0.1);
/// // After 30 steps the LR drops to 0.01
/// for _ in 0..30 {
///     sched.step();
/// }
/// assert!((sched.lr() - 0.01).abs() < 1e-9);
/// ```
pub struct StepLr {
    initial_lr: f64,
    lr: f64,
    step_size: usize,
    gamma: f64,
    current_step: usize,
}

impl StepLr {
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            lr: initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LrScheduler for StepLr {
    fn lr(&self) -> f64 {
        self.lr
    }

    fn step(&mut self) {
        self.current_step += 1;
        let num_decays = self.current_step / self.step_size;
        self.lr = self.initial_lr * self.gamma.powi(num_decays as i32);
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ─── Cosine Annealing ────────────────────────────────────────────────────────

/// Cosine annealing schedule that decays the learning rate from `initial_lr`
/// to `eta_min` over `t_max` steps following a cosine curve.
///
/// Equivalent to PyTorch's `CosineAnnealingLR`.
///
/// After `t_max` steps the learning rate stays at `eta_min`.
///
/// # Example
///
/// ```rust
/// use candle_training::lr_scheduler::{LrScheduler, CosineAnnealingLr};
///
/// let mut sched = CosineAnnealingLr::new(0.001, 1000);
/// for _ in 0..1000 {
///     sched.step();
/// }
/// assert!(sched.lr() < 1e-9); // eta_min defaults to 0
/// ```
pub struct CosineAnnealingLr {
    initial_lr: f64,
    eta_min: f64,
    t_max: usize,
    current_step: usize,
}

impl CosineAnnealingLr {
    /// Create a cosine annealing scheduler with `eta_min = 0.0`.
    pub fn new(initial_lr: f64, t_max: usize) -> Self {
        Self {
            initial_lr,
            eta_min: 0.0,
            t_max,
            current_step: 0,
        }
    }

    /// Create a cosine annealing scheduler with a custom minimum learning rate.
    pub fn with_eta_min(initial_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            initial_lr,
            eta_min,
            t_max,
            current_step: 0,
        }
    }
}

impl LrScheduler for CosineAnnealingLr {
    fn lr(&self) -> f64 {
        if self.current_step >= self.t_max {
            return self.eta_min;
        }
        let progress = self.current_step as f64 / self.t_max as f64;
        self.eta_min + (self.initial_lr - self.eta_min) * (1.0 + (PI * progress).cos()) / 2.0
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ─── Linear Warmup ───────────────────────────────────────────────────────────

/// Linear warmup from `start_lr` to `target_lr` over `warmup_steps`, then
/// holds at `target_lr`.
///
/// Commonly used as the first phase of a two-phase schedule. Compose with
/// another scheduler using [`SequentialLr`] for warmup + decay.
///
/// # Example
///
/// ```rust
/// use candle_training::lr_scheduler::{LrScheduler, LinearWarmupLr};
///
/// let mut sched = LinearWarmupLr::new(0.0, 0.001, 100);
/// for _ in 0..100 {
///     sched.step();
/// }
/// assert!((sched.lr() - 0.001).abs() < 1e-9);
/// ```
pub struct LinearWarmupLr {
    start_lr: f64,
    target_lr: f64,
    warmup_steps: usize,
    current_step: usize,
}

impl LinearWarmupLr {
    pub fn new(start_lr: f64, target_lr: f64, warmup_steps: usize) -> Self {
        Self {
            start_lr,
            target_lr,
            warmup_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for LinearWarmupLr {
    fn lr(&self) -> f64 {
        if self.current_step >= self.warmup_steps {
            return self.target_lr;
        }
        let t = self.current_step as f64 / self.warmup_steps as f64;
        self.start_lr + (self.target_lr - self.start_lr) * t
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ─── Cosine With Warmup ──────────────────────────────────────────────────────

/// Linear warmup followed by cosine annealing — the most common schedule used
/// for transformer pretraining and fine-tuning.
///
/// For the first `warmup_steps`, the LR ramps linearly from 0 to `peak_lr`.
/// Then it decays via cosine annealing to `eta_min` over the remaining
/// `total_steps - warmup_steps`.
///
/// # Example
///
/// ```rust
/// use candle_training::lr_scheduler::{LrScheduler, CosineWithWarmupLr};
///
/// let mut sched = CosineWithWarmupLr::new(0.001, 100, 1000);
/// // Warmup phase
/// for _ in 0..100 {
///     sched.step();
/// }
/// assert!((sched.lr() - 0.001).abs() < 1e-9);
/// // Decay phase
/// for _ in 100..1000 {
///     sched.step();
/// }
/// assert!(sched.lr() < 1e-9);
/// ```
pub struct CosineWithWarmupLr {
    peak_lr: f64,
    eta_min: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl CosineWithWarmupLr {
    /// Create with `eta_min = 0.0`.
    pub fn new(peak_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            eta_min: 0.0,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Create with a custom minimum learning rate.
    pub fn with_eta_min(
        peak_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
        eta_min: f64,
    ) -> Self {
        Self {
            peak_lr,
            eta_min,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for CosineWithWarmupLr {
    fn lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup from 0 to peak_lr
            let t = self.current_step as f64 / self.warmup_steps as f64;
            return self.peak_lr * t;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return self.peak_lr;
        }
        let decay_step = self.current_step.saturating_sub(self.warmup_steps);
        if decay_step >= decay_steps {
            return self.eta_min;
        }
        let progress = decay_step as f64 / decay_steps as f64;
        self.eta_min + (self.peak_lr - self.eta_min) * (1.0 + (PI * progress).cos()) / 2.0
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ─── Sequential Scheduler ────────────────────────────────────────────────────

/// Compose two schedulers end-to-end: run `first` for `switch_step` steps,
/// then switch to `second`.
///
/// # Example
///
/// ```rust
/// use candle_training::lr_scheduler::{LrScheduler, SequentialLr, LinearWarmupLr, StepLr};
///
/// let warmup = LinearWarmupLr::new(0.0, 0.01, 100);
/// let decay = StepLr::new(0.01, 50, 0.5);
/// let mut sched = SequentialLr::new(warmup, decay, 100);
///
/// // First 100 steps: warmup
/// for _ in 0..100 {
///     sched.step();
/// }
/// assert!((sched.lr() - 0.01).abs() < 1e-9);
/// ```
pub struct SequentialLr<A: LrScheduler, B: LrScheduler> {
    first: A,
    second: B,
    switch_step: usize,
    current_step: usize,
}

impl<A: LrScheduler, B: LrScheduler> SequentialLr<A, B> {
    pub fn new(first: A, second: B, switch_step: usize) -> Self {
        Self {
            first,
            second,
            switch_step,
            current_step: 0,
        }
    }
}

impl<A: LrScheduler, B: LrScheduler> LrScheduler for SequentialLr<A, B> {
    fn lr(&self) -> f64 {
        if self.current_step < self.switch_step {
            self.first.lr()
        } else {
            self.second.lr()
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step <= self.switch_step {
            self.first.step();
        } else {
            self.second.step();
        }
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_lr() {
        let mut s = ConstantLr::new(0.05);
        for _ in 0..100 {
            s.step();
        }
        assert!((s.lr() - 0.05).abs() < 1e-12);
        assert_eq!(s.current_step(), 100);
    }

    #[test]
    fn step_lr_decays() {
        let mut s = StepLr::new(0.1, 10, 0.5);
        for _ in 0..10 {
            s.step();
        }
        assert!((s.lr() - 0.05).abs() < 1e-9);
        for _ in 0..10 {
            s.step();
        }
        assert!((s.lr() - 0.025).abs() < 1e-9);
    }

    #[test]
    fn cosine_annealing_endpoints() {
        let mut s = CosineAnnealingLr::new(1.0, 100);
        // Step 0: should be full LR
        assert!((s.lr() - 1.0).abs() < 1e-9);
        for _ in 0..100 {
            s.step();
        }
        // After t_max steps: should be at eta_min (0)
        assert!(s.lr().abs() < 1e-9);
    }

    #[test]
    fn cosine_annealing_midpoint() {
        let mut s = CosineAnnealingLr::new(1.0, 100);
        for _ in 0..50 {
            s.step();
        }
        // At midpoint of cosine: should be ~0.5
        assert!((s.lr() - 0.5).abs() < 0.01);
    }

    #[test]
    fn linear_warmup() {
        let mut s = LinearWarmupLr::new(0.0, 1.0, 100);
        assert!(s.lr().abs() < 1e-12);
        for _ in 0..50 {
            s.step();
        }
        assert!((s.lr() - 0.5).abs() < 0.01);
        for _ in 0..50 {
            s.step();
        }
        assert!((s.lr() - 1.0).abs() < 1e-9);
        // Past warmup: stays at target
        s.step();
        assert!((s.lr() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_with_warmup() {
        let mut s = CosineWithWarmupLr::new(1.0, 100, 1100);
        // Before any step: lr = 0 (warmup starts at 0)
        assert!(s.lr().abs() < 1e-12);
        // After warmup
        for _ in 0..100 {
            s.step();
        }
        assert!((s.lr() - 1.0).abs() < 1e-9);
        // After full schedule
        for _ in 100..1100 {
            s.step();
        }
        assert!(s.lr().abs() < 1e-9);
    }

    #[test]
    fn sequential_switches() {
        let warmup = LinearWarmupLr::new(0.0, 1.0, 10);
        let decay = StepLr::new(1.0, 5, 0.5);
        let mut s = SequentialLr::new(warmup, decay, 10);

        for _ in 0..10 {
            s.step();
        }
        assert!((s.lr() - 1.0).abs() < 1e-9);

        // Now in decay phase: after 5 steps, lr should halve
        for _ in 0..5 {
            s.step();
        }
        assert!((s.lr() - 0.5).abs() < 1e-9);
    }
}
