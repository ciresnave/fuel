// SPDX-License-Identifier: MIT OR Apache-2.0
//! Load-progress reporting.
//!
//! Moved to [`fuel_model_loader::model_progress`] (fuel-core dissolution Slice 2,
//! `docs/session-prompts/fuel-core-dissolution-b1.md`). This module re-exports it so
//! existing `fuel_core::model_progress` call sites keep compiling.
pub use fuel_model_loader::model_progress::*;
