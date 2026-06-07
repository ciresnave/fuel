//! Training loop abstraction (lazy-graph).
//!
//! [`TrainingLoop`] provides a structured training loop that composes
//! gradient clipping, optional gradient accumulation, learning-rate
//! scheduling, and step counting into a single configurable driver.
//!
//! # Design
//!
//! The training loop is intentionally NOT a framework. It does not own
//! the model, the dataloader, or the optimizer. It orchestrates the
//! per-step bookkeeping that every training script reimplements:
//!
//! 1. Backward (`loss.backward()`)
//! 2. Gradient harvest (per [`LazyVar`] in `opt.params()`)
//! 3. Gradient clipping (norm and/or value, if configured)
//! 4. Optimizer step
//! 5. LR scheduling
//! 6. Logging / metrics
//!
//! All gradient operations run on the lazy graph: clipping uses
//! [`fuel::lazy_training_augmentations::clip_grad_norm`] /
//! [`clip_grad_value`], which return new `HashMap<String, LazyTensor>`
//! values.
//!
//! # Example
//!
//! ```rust,no_run
//! use fuel::lazy_nn_optim::{LazyAdamW, LazyOptimizer, LazyVar, AdamWConfig};
//! use fuel::lazy_training_augmentations::CosineSchedule;
//! use fuel_training::training_loop::TrainingLoop;
//!
//! # fn main() -> fuel::Result<()> {
//! let var = LazyVar::new("w", vec![1.0_f32, 2.0, 3.0]);
//! let mut opt = LazyAdamW::new(vec![var.clone()], AdamWConfig::default())?;
//! let sched = CosineSchedule { base_lr: 1e-3, warmup_steps: 10, total_steps: 100 };
//! let mut loop_ = TrainingLoop::new()
//!     .with_max_grad_norm(1.0)
//!     .with_scheduler(sched);
//! // Then per step:
//! //   let loss = ... build forward graph using var.tensor() ...;
//! //   let outcome = loop_.step(&loss, &mut opt)?;
//! # Ok(())
//! # }
//! ```

use fuel::Result;
use fuel::lazy::LazyTensor;
use fuel::lazy_nn_optim::{LazyOptimizer, LazyVar};
use fuel::lazy_training_augmentations::{LrSchedule, clip_grad_norm, clip_grad_value};
use std::collections::HashMap;

/// The result of a single training step.
#[derive(Clone, Debug)]
pub struct StepOutcome {
    /// The scalar loss value (realized to f32 host).
    pub loss: f64,
    /// The global gradient L2 norm before clipping, when norm clipping
    /// is enabled.
    pub grad_norm: Option<f64>,
    /// The learning rate that was applied to this step (after scheduler).
    pub lr: f64,
    /// The 1-indexed global step number.
    pub global_step: usize,
}

/// A composable training loop driver (lazy-graph variant).
pub struct TrainingLoop {
    max_grad_norm: Option<f64>,
    max_grad_value: Option<f64>,
    scheduler: Option<Box<dyn LrSchedule>>,
    global_step: usize,
    log_interval: Option<usize>,
}

impl TrainingLoop {
    pub fn new() -> Self {
        Self {
            max_grad_norm: None,
            max_grad_value: None,
            scheduler: None,
            global_step: 0,
            log_interval: None,
        }
    }

    /// Enable global L2-norm gradient clipping at `max_norm`.
    pub fn with_max_grad_norm(mut self, max_norm: f64) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Enable per-element gradient value clipping to `[-v, v]`.
    pub fn with_max_grad_value(mut self, v: f64) -> Self {
        self.max_grad_value = Some(v);
        self
    }

    /// Attach a learning rate scheduler. The scheduler is consulted at
    /// step time via [`LrSchedule::lr_at`] and applied to the optimizer
    /// via [`LazyOptimizer::set_learning_rate`].
    pub fn with_scheduler<S: LrSchedule + 'static>(mut self, scheduler: S) -> Self {
        self.scheduler = Some(Box::new(scheduler));
        self
    }

    /// Set the logging interval (in steps). When set, a `tracing::info!`
    /// event is emitted every `interval` steps.
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

    /// Execute one training step against a [`LazyOptimizer`]: backward,
    /// harvest, clip, optimize, schedule.
    pub fn step<O: LazyOptimizer>(
        &mut self,
        loss: &LazyTensor,
        opt: &mut O,
    ) -> Result<StepOutcome> {
        let loss_val = loss.realize_f32()[0] as f64;

        let grads = harvest_grads(loss, opt.params());

        let grads = if let Some(max_norm) = self.max_grad_norm {
            clip_grad_norm(&grads, max_norm, 2.0)?
        } else {
            grads
        };
        let grad_norm = if self.max_grad_norm.is_some() {
            Some(global_l2_norm(&grads))
        } else {
            None
        };

        let grads = if let Some(max_val) = self.max_grad_value {
            clip_grad_value(&grads, max_val)?
        } else {
            grads
        };

        opt.step(&grads)?;

        if let Some(sched) = &self.scheduler {
            opt.set_learning_rate(sched.lr_at(self.global_step));
        }

        self.global_step += 1;
        let lr = opt.learning_rate();

        let outcome = StepOutcome {
            loss: loss_val,
            grad_norm,
            lr,
            global_step: self.global_step,
        };

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

/// Walk `vars`, look up each one's gradient in the backward map produced
/// by `loss.backward()`, and collect into a `name -> LazyTensor` map.
/// Parameters that didn't contribute to `loss` are silently skipped
/// (same semantics as [`LazyOptimizer::backward_step`]).
fn harvest_grads(loss: &LazyTensor, vars: &[LazyVar]) -> HashMap<String, LazyTensor> {
    let grad_map = loss.backward();
    let mut grads = HashMap::with_capacity(vars.len());
    for var in vars {
        let Some(node_id) = var.last_node_id() else {
            continue;
        };
        let handle = fuel_graph::Tensor::from_existing(
            loss.graph_tensor().graph().clone(),
            node_id,
        );
        if let Some(grad) = grad_map.get(&handle) {
            grads.insert(
                var.name().to_string(),
                LazyTensor::from_graph_tensor(grad),
            );
        }
    }
    grads
}

fn global_l2_norm(grads: &HashMap<String, LazyTensor>) -> f64 {
    let mut acc: f64 = 0.0;
    for g in grads.values() {
        let host = g.sqr().sum_all().realize_f32();
        acc += host[0] as f64;
    }
    acc.sqrt()
}
