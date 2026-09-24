// SPDX-License-Identifier: MIT OR Apache-2.0
//! HF `config.json` resolution rules.
//!
//! Moved to [`fuel_model_loader::hf_config`] (fuel-core dissolution Slice 2,
//! `docs/session-prompts/fuel-core-dissolution-b1.md`). This module re-exports it so
//! existing `fuel_core::hf_config` / `fuel::hf_config` call sites keep compiling.
pub use fuel_model_loader::hf_config::*;
