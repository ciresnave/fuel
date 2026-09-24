// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model-loading support for the fuel ML framework.
//!
//! Slice 2 of the fuel-core dissolution (`docs/session-prompts/fuel-core-dissolution-b1.md`):
//! HF `config.json` resolution rules, load-progress reporting, and GGUF/imatrix/tokenizer
//! glue that sits on top of `fuel-formats`' transport-independent wire parsers. Like
//! `fuel-formats`, **no item in this crate references `Tensor`, `Device`, `Storage`, or any
//! other backend-frontend type** — verified by grep against `fuel-core`'s copies before this
//! crate was cut, not assumed from the module names.
//!
//! `fuel-core`'s own `hf_config`, `model_progress`, and `quantized::{arch, gguf_file,
//! gguf_mmap, imatrix_file, mod, tokenizer}` modules now re-export from here, matching the
//! same shim pattern already used for `dtype`/`layout`/`shape`/`backend`/`storage` from the
//! earlier B0 extraction — existing `fuel_core::` / `fuel::` import paths keep compiling.

pub mod hf_config;
pub mod model_progress;
pub mod quantized;
