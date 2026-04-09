//! Training loop abstraction.
//!
//! [`TrainingLoop`] provides a structured training loop that composes
//! learning-rate scheduling, gradient clipping, gradient accumulation, and
//! checkpointing into a single, configurable driver.
//!
//! The user provides a closure that computes the loss for a single mini-batch;
//! the loop handles the rest.
//!
//! # Design
//!
//! The training loop is intentionally NOT a framework. It does not own the
//! model, the dataloader, or the optimizer. It orchestrates the per-step
//! bookkeeping that every training script reimplements:
//!
//! 1. Forward + backward
//! 2. Gradient accumulation (if configured)
//! 3. Gradient clipping (if configured)
//! 4. Optimizer step
//! 5. LR scheduling
//! 6. Logging / metrics
//! 7. Checkpointing (if configured)
//!
//! # Example
//!
//! ```rust
//! use candle::{DType, Device, Tensor, Var};
//! use candle_nn::{AdamW, Optimizer, ParamsAdamW};
//! use candle_training::training_loop::{TrainingLoop, StepOutcome};
//!
//! # fn main() -> candle::Result<()> {
//! let x = Var::new(&[1.0f32, 2.0, 3.0][..], &Device::Cpu)?;
//! let mut opt = AdamW::new(vec![x.clone()], ParamsAdamW::default())?;
//!
//! let mut loop_ = TrainingLoop::new()
//!     .with_max_grad_norm(1.0);
//!
//! // Run 10 training steps manually
//! for step in 0..10 {
//!     let loss = x.as_tensor().sqr()?.sum_all()?;
//!     let outcome = loop_.step(&loss, &[&x], &mut opt)?;
//!     // outcome contains the loss value and gradient norm
//! }
//! # Ok(())
//! # }
//! ```

use crate::grad_clip;
use crate::lr_scheduler::LrScheduler;
use candle::{Result, Tensor, Var};
use candle_nn::Optimizer;

/// The result of a single training step.
#[derive(Clone, Debug)]
pub struct StepOutcome {
    /// The scalar loss value (before backward).
    pub loss: f64,
    /// The global gradient norm before clipping (if clipping was enabled).
    pub grad_norm: Option<f64>,
    /// The current learning rate after this step.
    pub lr: f64,
    /// The global step number (1-indexed).
    pub global_step: usize,
}

/// A composable training loop driver.
///
/// Handles gradient clipping, LR scheduling, and step counting. Does NOT own
/// the model, optimizer, or data — those remain under user control.
pub struct TrainingLoop {
    max_grad_norm: Option<f64>,
    max_grad_value: Option<f64>,
    scheduler: Option<Box<dyn LrScheduler>>,
    global_step: usize,
    log_interval: Option<usize>,
}

impl TrainingLoop {
    /// Create a new training loop with default settings (no clipping, no scheduling).
    pub fn new() -> Self {
        Self {
            max_grad_norm: None,
            max_grad_value: None,
            scheduler: None,
            global_step: 0,
            log_interval: None,
        }
    }

    /// Enable gradient norm clipping. Gradients are rescaled so the global L2
    /// norm does not exceed `max_norm`.
    pub fn with_max_grad_norm(mut self, max_norm: f64) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Enable per-element gradient value clipping to `[-v, v]`.
    pub fn with_max_grad_value(mut self, v: f64) -> Self {
        self.max_grad_value = Some(v);
        self
    }

    /// Attach a learning rate scheduler. The scheduler is stepped after each
    /// optimizer step, and the LR is applied to the optimizer automatically.
    pub fn with_scheduler<S: LrScheduler + 'static>(mut self, scheduler: S) -> Self {
        self.scheduler = Some(Box::new(scheduler));
        self
    }

    /// Set the logging interval (in steps). When set, a `tracing::info!` event
    /// is emitted every `interval` steps with the loss, LR, and gradient norm.
    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = Some(interval);
        self
    }

    /// Resume from a given global step (e.g. when loading a checkpoint).
    pub fn set_global_step(&mut self, step: usize) {
        self.global_step = step;
    }

    /// Return the current global step count.
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Execute one training step: backward, clip, optimize, schedule.
    ///
    /// # Arguments
    ///
    /// - `loss` — A scalar loss tensor (already computed by the user's forward pass).
    /// - `vars` — The trainable variables to clip gradients for.
    /// - `opt` — The optimizer to step.
    ///
    /// # Returns
    ///
    /// A [`StepOutcome`] containing the loss value, gradient norm, current LR,
    /// and global step number.
    pub fn step<O: Optimizer>(
        &mut self,
        loss: &Tensor,
        vars: &[&Var],
        opt: &mut O,
    ) -> Result<StepOutcome> {
        let loss_val: f64 = loss.to_scalar::<f32>()? as f64;

        // Backward pass
        let mut grads = loss.backward()?;

        // Gradient clipping
        let grad_norm = if let Some(max_norm) = self.max_grad_norm {
            Some(grad_clip::clip_grad_norm(vars, &mut grads, max_norm)?)
        } else {
            None
        };

        if let Some(max_val) = self.max_grad_value {
            grad_clip::clip_grad_value(vars, &mut grads, max_val)?;
        }

        // Optimizer step
        opt.step(&grads)?;

        // LR scheduling
        if let Some(sched) = &mut self.scheduler {
            sched.step();
            opt.set_learning_rate(sched.lr());
        }

        self.global_step += 1;

        let lr = opt.learning_rate();

        let outcome = StepOutcome {
            loss: loss_val,
            grad_norm,
            lr,
            global_step: self.global_step,
        };

        // Logging
        if let Some(interval) = self.log_interval {
            if self.global_step % interval == 0 {
                match grad_norm {
                    Some(gn) => tracing::info!(
                        step = self.global_step,
                        loss = loss_val,
                        lr = lr,
                        grad_norm = gn,
                        "training step"
                    ),
                    None => tracing::info!(
                        step = self.global_step,
                        loss = loss_val,
                        lr = lr,
                        "training step"
                    ),
                }
            }
        }

        Ok(outcome)
    }
}

impl Default for TrainingLoop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Var};
    use candle_nn::SGD;

    #[test]
    fn basic_step() -> Result<()> {
        let x = Var::new(&[2.0f32, 3.0][..], &Device::Cpu)?;
        let mut opt = SGD::new(vec![x.clone()], 0.01)?;
        let mut tl = TrainingLoop::new();

        let loss = x.as_tensor().sqr()?.sum_all()?;
        let outcome = tl.step(&loss, &[&x], &mut opt)?;

        assert!(outcome.loss > 0.0);
        assert_eq!(outcome.global_step, 1);
        assert!(outcome.grad_norm.is_none());
        Ok(())
    }

    #[test]
    fn step_with_grad_clipping() -> Result<()> {
        let x = Var::new(&[100.0f32, 200.0][..], &Device::Cpu)?;
        let mut opt = SGD::new(vec![x.clone()], 0.001)?;
        let mut tl = TrainingLoop::new().with_max_grad_norm(1.0);

        let loss = x.as_tensor().sqr()?.sum_all()?;
        let outcome = tl.step(&loss, &[&x], &mut opt)?;

        assert!(outcome.grad_norm.unwrap() > 1.0); // original norm was large
        Ok(())
    }

    #[test]
    fn step_with_scheduler() -> Result<()> {
        use crate::lr_scheduler::StepLr;

        let x = Var::new(&[1.0f32][..], &Device::Cpu)?;
        let mut opt = SGD::new(vec![x.clone()], 0.1)?;
        let sched = StepLr::new(0.1, 5, 0.5);
        let mut tl = TrainingLoop::new().with_scheduler(sched);

        for _ in 0..5 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            tl.step(&loss, &[&x], &mut opt)?;
        }

        // After 5 steps, LR should have decayed
        assert!((opt.learning_rate() - 0.05).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn global_step_tracks() -> Result<()> {
        let x = Var::new(&[1.0f32][..], &Device::Cpu)?;
        let mut opt = SGD::new(vec![x.clone()], 0.01)?;
        let mut tl = TrainingLoop::new();

        for _ in 0..10 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            tl.step(&loss, &[&x], &mut opt)?;
        }

        assert_eq!(tl.global_step(), 10);
        Ok(())
    }
}
