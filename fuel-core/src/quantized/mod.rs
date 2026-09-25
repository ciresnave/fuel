// SPDX-License-Identifier: MIT OR Apache-2.0
//! Quantized (ggml/gguf block-format) support for fuel-core.
//!
//! Moved to [`fuel_loaders::quantized`] (fuel-core dissolution Slice 2,
//! `docs/session-prompts/fuel-core-dissolution-b1.md`). This module re-exports it —
//! including its `arch`/`gguf_file`/`gguf_mmap`/`imatrix_file`/`tokenizer` submodules,
//! which a glob `pub use` brings along as public items — so existing
//! `fuel_core::quantized::*` / `fuel::quantized::*` call sites keep compiling.
pub use fuel_loaders::quantized::*;
