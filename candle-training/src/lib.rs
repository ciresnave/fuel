//! # candle-training
//!
//! **Layer**: Training  |  **Stability**: experimental
//!
//! Training orchestration for the Candle ML framework. This crate is the
//! canonical home for training-loop infrastructure on top of `candle-core`
//! and `candle-nn`.
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
//! - Model definitions (stay in `candle-transformers`)
//! - Inference-specific code (use `candle-inference`)
//! - Dataset loading or preprocessing (use `candle-datasets`)
//! - Optimiser kernels (stay in `candle-nn`)
//!
//! ## Layer placement
//!
//! ```text
//! candle-training    ← you are here (training orchestration)
//! candle-nn          (layers, optimisers, VarBuilder)
//! candle-core        (tensors, devices, autograd)
//! ```
//!
//! Nothing in `candle-core` or `candle-nn` depends on this crate. It is a
//! leaf that aggregates; it does not define.
//!
//! ## Example
//!
//! ```rust
//! use candle::{DType, Device, Tensor, Var};
//! use candle_nn::{AdamW, Optimizer, ParamsAdamW};
//! use candle_training::training_loop::TrainingLoop;
//! use candle_training::lr_scheduler::CosineWithWarmupLr;
//!
//! # fn main() -> candle::Result<()> {
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
