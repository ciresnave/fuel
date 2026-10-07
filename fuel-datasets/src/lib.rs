// SPDX-License-Identifier: MIT OR Apache-2.0
//! # fuel-datasets
//!
//! **Layer**: IO — parallel to `fuel-core`'s serialization layer. Provides Rust
//! access to standard machine-learning datasets and a generic batching utility.
//!
//! **Stability**: `evolving`
//!
//! ## What this crate is for
//!
//! `fuel-datasets` simplifies dataset loading for training and evaluation:
//!
//! - **[`vision`]**: MNIST, CIFAR-10, CIFAR-100, and other image datasets returned
//!   as `(images, labels)` host buffers plus explicit dimensions.
//! - **[`nlp`]**: Text dataset utilities (tokenized batches, sequence packing).
//! - **[`hub`]**: HuggingFace Hub dataset helpers.
//!
//! ## Host buffers, not tensors
//!
//! Loaders return `Vec<f32>` / `Vec<u32>`, never a tensor. Decoding a dataset is
//! file I/O plus a normalization pass, and the consumer must build its tensors on
//! *its own* graph anyway — `Tensor::from_*` mints a NEW graph per call, so a
//! tensor handed out by a loader could never be combined with the caller's
//! activations. Host buffers are the shape that composes.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use fuel_datasets::vision::mnist;
//! # fn main() -> fuel_core::Result<()> {
//! let dataset = mnist::load()?; // downloads if needed
//! println!(
//!     "train: {} samples of {:?} ({} floats)",
//!     dataset.train_samples, dataset.image_dims, dataset.train_images.len(),
//! );
//! // → train: 60000 samples of [28, 28] (47040000 floats)
//! # Ok(()) }
//! ```
//!
//! ## What is explicitly NOT here
//!
//! - **No model code.** Architecture definitions belong in `fuel-transformers`.
//! - **No training loops.** Use `fuel-training` (Phase 2) or write your own.
//! - **No inference.** This crate produces input tensors; what you do with them
//!   is not its concern.
pub mod hub;
pub mod nlp;
pub mod vision;

// GAP-347 PR 4 (bail! consolidation): before/after test for this consumer
// crate family. `fuel_core::bail!` is now `fuel_ir::bail!` re-exported
// (the fuel-core-local `macro_rules!` duplicate was deleted), which can
// change `$crate::Error` resolution silently across a macro-path swap.
// This proves the error text/variant this crate actually relies on
// (nlp/tinystories.rs's `fuel_core::bail!`) is unchanged.
#[cfg(test)]
mod bail_consolidation_tests {
    fn always_bails() -> fuel_core::Result<()> {
        fuel_core::bail!(
            "deliberate failure for the fuel-core dissolution bail! consolidation test"
        );
        #[allow(unreachable_code)]
        Ok(())
    }

    #[test]
    fn fuel_core_bail_still_produces_a_typed_err_with_the_message() {
        let err = always_bails().expect_err("bail! must produce an Err, never panic");
        assert!(err.to_string().contains("deliberate failure"));
    }
}
