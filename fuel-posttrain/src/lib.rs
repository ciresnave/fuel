// SPDX-License-Identifier: MIT OR Apache-2.0
//! Post-training model compression for the Fuel ML framework.
//!
//! **Layer**: Use-Case Orchestration | **Stability**: experimental
//!
//! Techniques applied to a model's weights *after* it has been loaded and
//! (optionally) quantized — as opposed to `fuel-training`, which owns the
//! training-LOOP infrastructure (optimizers, schedulers, checkpoints) for a
//! model still being trained. This crate's consumers already have a frozen
//! set of weights and want to recover accuracy lost to quantization, or
//! shrink the weight set further via pruning.
//!
//! ## Modules
//!
//! - [`norm_tweaking`] — Norm Tweaking (Li et al., AAAI 2024): recalibrates
//!   normalization-layer gamma/beta parameters to compensate for the
//!   activation-distribution shift quantization introduces. Ported from
//!   `lightbulb/src/quantization/norm_tweaking.rs` (candlelight-based) as
//!   part of the Candle-to-Fuel migration (lightbulb #106). **Read the
//!   module doc before using this**: the ported implementation is an
//!   analytical approximation of the paper's algorithm, not the paper's own
//!   iterative gradient-based calibration — the source this was ported from
//!   never implemented that half either.
//!
//! ## What is NOT here
//!
//! - Training-loop infrastructure (optimizers, LR schedulers, gradient
//!   clipping) — stays in `fuel-training`.
//! - Model definitions — stay in `fuel-transformers`.
//! - Quantization FORMAT numerics (GGML block types, dequant kernels) —
//!   stay in `fuel-quantized`.

pub mod norm_tweaking;
