//! # fuel-training
//!
//! **Layer**: Training  |  **Stability**: experimental
//!
//! Training orchestration for the Fuel ML framework. This crate is the
//! canonical home for training-loop infrastructure on top of `fuel-core`
//! and `fuel-nn`.
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
//! Nothing in `fuel-core` or `fuel-nn` depends on this crate. It is a
//! leaf that aggregates; it does not define.
//!
//! ## Example
//!
//! ```rust
//! use fuel::{DType, Device, Tensor, Var};
//! use fuel_nn::{AdamW, Optimizer, ParamsAdamW};
//! use fuel_training::training_loop::TrainingLoop;
//! use fuel_training::lr_scheduler::CosineWithWarmupLr;
//!
//! # fn main() -> fuel::Result<()> {
//! let x = Var::new(&[1.0f32, 2.0, 3.0][..], &Device::Cpu)?;
//! let mut opt = AdamW::new(vec![x.clone()], ParamsAdamW::default())?;
//!
//! let sched = CosineWithWarmupLr::new(0.001, 10, 100);
//! let mut loop_ = TrainingLoop::new()
//!     .with_max_grad_norm(1.0)
//!     .with_scheduler(sched);
//!
//! for step in 0..100 {
//!     let loss = x.as_tensor().sqr()?.sum_all()?;
//!     let outcome = loop_.step(&loss, &[&x], &mut opt)?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod checkpoint;
pub mod grad_accum;
pub mod grad_clip;
pub mod lr_scheduler;
pub mod training_loop;
