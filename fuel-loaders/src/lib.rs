// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model-loading support for the fuel ML framework.
//!
//! Named `fuel-loaders` to match the crate `docs/architecture/02-layers.md` already
//! ratifies for this scope (diagrammed with the `†` unbuilt-but-ratified marker; also
//! named in `fuel-formats`' own doc comment as its downstream consumer). Originally cut
//! as `fuel-model-loader`, an unnoticed drift from the constitution rather than a
//! deliberate rename — corrected within hours, before any consumer outside `fuel-core`'s
//! own shims existed.
//!
//! Slices 2-3 of the fuel-core dissolution (`docs/session-prompts/fuel-core-dissolution-b1.md`):
//! HF `config.json` resolution rules, load-progress reporting, GGUF/imatrix/tokenizer glue, and
//! safetensors file reading, all sitting on top of `fuel-formats`' transport-independent wire
//! parsers (safetensors' own byte-level parsing lives in the upstream `safetensors` crate).
//! Like `fuel-formats`, **no item in this crate references `Tensor`, `Device`, `Storage`, or
//! any other backend-frontend type** — verified by grep against `fuel-core`'s copies before
//! each move, not assumed from the module names.
//!
//! `fuel-core`'s own `hf_config`, `model_progress`, `quantized::{arch, gguf_file, gguf_mmap,
//! imatrix_file, mod, tokenizer}`, and `safetensors` modules now re-export from here, matching
//! the same shim pattern already used for `dtype`/`layout`/`shape`/`backend`/`storage` from the
//! earlier B0 extraction — existing `fuel_core::` / `fuel::` import paths keep compiling.

pub mod hf_config;
pub mod model_progress;
pub mod quantized;
pub mod safetensors;
