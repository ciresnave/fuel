// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors mmap/buffer loading (`MmapedSafetensors` / `BufferedSafetensors`).
//!
//! Moved to [`fuel_loaders::safetensors`] (fuel-core dissolution Slice 3,
//! `docs/session-prompts/fuel-core-dissolution-b1.md`). This module re-exports it so
//! existing `fuel_core::safetensors` / `fuel::safetensors` call sites keep compiling.
//! Deferred work, not permanent state — see that doc's shim-debt ledger.
pub use fuel_loaders::safetensors::*;
