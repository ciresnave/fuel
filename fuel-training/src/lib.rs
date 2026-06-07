//! # fuel-training
//!
//! **Layer**: Training  |  **Stability**: experimental
//!
//! Training orchestration for the Fuel ML framework. This crate is the
//! canonical home for training-loop infrastructure on top of the lazy
//! substrate in `fuel-core` (LazyOptimizer, LazyVar, LazyVarMap, the
//! `lazy_training_augmentations` schedulers and grad clippers).
//!
//! ## Modules
//!
//! - [`lr_scheduler`] — Learning rate schedulers (cosine annealing, warmup,
//!   step decay, sequential composition).
//! - [`grad_clip`] — Gradient clipping (global L2 norm, per-element value).
//! - [`grad_accum`] — Gradient accumulation for simulating larger batch sizes.
//! - [`checkpoint`] — Checkpoint save/load with training metadata (epoch, step,
//!   metrics) for resumable training.
//! - [`training_loop`] — Composable training loop driver that wires together
//!   clipping, scheduling, and logging.
//!
//! ## What is NOT here
//!
//! - Model definitions (stay in `fuel-transformers`)
//! - Inference-specific code (use `fuel-inference`)
//! - Dataset loading or preprocessing (use `fuel-datasets`)
//! - Optimiser kernels (stay in `fuel-nn`)
//!
//! ## Layer placement
//!
//! ```text
//! fuel-training    ← you are here (training orchestration)
//! fuel-nn          (layers, optimisers, VarBuilder)
//! fuel-core        (tensors, devices, autograd)
//! ```
//!
//! Nothing in `fuel-core` depends on this crate. It is a leaf that
//! aggregates; it does not define.
//!
//! ## Example
//!
//! ```rust,no_run
//! use fuel::lazy_nn_optim::{LazyAdamW, LazyOptimizer, LazyVar, AdamWConfig};
//! use fuel::lazy_training_augmentations::CosineSchedule;
//! use fuel_training::training_loop::TrainingLoop;
//!
//! # fn main() -> fuel::Result<()> {
//! let x = LazyVar::new("x", fuel::Shape::from_dims(&[3]), vec![1.0_f32, 2.0, 3.0])?;
//! let mut opt = LazyAdamW::new(vec![x.clone()], AdamWConfig::default())?;
//! let sched = CosineSchedule { base_lr: 1e-3, warmup_steps: 10, total_steps: 100 };
//! let mut loop_ = TrainingLoop::new()
//!     .with_max_grad_norm(1.0)
//!     .with_scheduler(sched);
//! // Per step: build a `LazyTensor` loss using `x.tensor(&anchor)`, then
//! //   `loop_.step(&loss, &mut opt)?;`
//! # Ok(())
//! # }
//! ```

pub mod checkpoint;
pub mod training_loop;

// Re-exports from the lazy training substrate in fuel-core. These used to
// live in fuel-training as eager wrappers around fuel-nn's Optimizer; the
// lazy equivalents have strictly broader coverage so the eager modules are
// retired and only re-exported here for source-compatibility.

pub mod grad_accum {
    //! Gradient accumulation (lazy). Re-export of
    //! [`fuel::lazy_training_augmentations_extras::GradAccumulator`].
    pub use fuel::lazy_training_augmentations_extras::GradAccumulator;
}

pub mod grad_clip {
    //! Gradient clipping (lazy). Re-exports of
    //! [`fuel::lazy_training_augmentations::{clip_grad_norm, clip_grad_value}`].
    pub use fuel::lazy_training_augmentations::{clip_grad_norm, clip_grad_value};
}

pub mod lr_scheduler {
    //! Learning-rate schedulers (lazy). Re-exports of
    //! [`fuel::lazy_training_augmentations::{LrSchedule, CosineSchedule,
    //! LinearWarmupSchedule, PolynomialSchedule, StepSchedule}`].
    pub use fuel::lazy_training_augmentations::{
        CosineSchedule, LinearWarmupSchedule, LrSchedule, PolynomialSchedule, StepSchedule,
    };
}
