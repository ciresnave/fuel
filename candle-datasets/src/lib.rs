//! # candle-datasets
//!
//! **Layer**: IO — parallel to `candle-core`'s serialization layer. Provides Rust
//! access to standard machine-learning datasets and a generic batching utility.
//!
//! **Stability**: `evolving`
//!
//! ## What this crate is for
//!
//! `candle-datasets` simplifies dataset loading for training and evaluation:
//!
//! - **[`vision`]**: MNIST, CIFAR-10, CIFAR-100, and other image datasets returned
//!   as `(images, labels)` tensor pairs.
//! - **[`nlp`]**: Text dataset utilities (tokenized batches, sequence packing).
//! - **[`hub`]**: HuggingFace Hub dataset helpers.
//! - **[`Batcher`]**: Generic mini-batch iterator that shuffles and chunks any
//!   dataset tensor into fixed-size batches.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use candle_datasets::vision::mnist;
//! # fn main() -> candle::Result<()> {
//! let dataset = mnist::load()?; // downloads if needed
//! println!("train images: {:?}", dataset.train_images.dims());
//! // → train images: [60000, 1, 28, 28]
//! # Ok(()) }
//! ```
//!
//! ## What is explicitly NOT here
//!
//! - **No model code.** Architecture definitions belong in `candle-transformers`.
//! - **No training loops.** Use `candle-training` (Phase 2) or write your own.
//! - **No inference.** This crate produces input tensors; what you do with them
//!   is not its concern.
pub mod batcher;
pub mod hub;
pub mod nlp;
pub mod vision;

pub use batcher::Batcher;
