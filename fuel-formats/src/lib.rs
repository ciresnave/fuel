//! Transport-independent parsers for tensor wire formats.
//!
//! `fuel-formats` provides pure-Rust parsers for the serialization formats
//! Fuel reads and writes:
//!
//! - [`safetensors`] — HuggingFace's tensor container format
//! - [`pickle`] — Python pickle (`.pth` / `.bin` PyTorch checkpoints)
//! - [`gguf`] — llama.cpp's quantized tensor file format (file + mmap)
//! - [`ggml`] — legacy GGML tensor format
//! - [`imatrix`] — llama.cpp activation-importance matrix format
//!
//! # Design contract
//!
//! Every public API in this crate operates on transport primitives —
//! `impl Read` / `impl Seek` / `&[u8]` / `Cow<'_, [u8]>` — and returns
//! format-typed structs. **No item in this crate references `Tensor`,
//! `Device`, `Storage`, or any other backend-frontend type.** The only
//! external types it borrows are foundational primitives from
//! [`fuel-core-types`]: [`DType`](fuel_core_types::DType),
//! [`Shape`](fuel_core_types::Shape), and
//! [`GgmlDType`](fuel_core_types::GgmlDType).
//!
//! This split is what lets the same parser code serve file loading,
//! HTTP/S3 streaming, mmap, Unix-socket IPC, and shared-memory
//! tensor exchange between cooperating processes — including the
//! eventual `RemoteHostStorage` consumers in Phase 7c. The thin
//! Tensor-construction wrappers that build a [`fuel_core::Tensor`]
//! from parsed metadata live in `fuel-loaders` (post-Phase 7.5
//! work item E) or in `fuel-core` directly (until then).
//!
//! # Status
//!
//! This crate is in active extraction from `fuel-core` as part of
//! Phase 7.5 work item A. Module bodies will be migrated module-by-
//! module; the public surface here is the single source of truth for
//! the post-extraction API.

pub mod ggml;
pub mod gguf;
pub mod imatrix;
pub mod pickle;
pub mod safetensors;
