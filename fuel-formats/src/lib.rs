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
//! [`fuel-core-types`]: [`DType`](fuel_ir::DType),
//! [`Shape`](fuel_ir::Shape), and
//! [`GgmlDType`](fuel_ir::GgmlDType).
//!
//! This split is what lets the same parser code serve file loading,
//! HTTP/S3 streaming, mmap, Unix-socket IPC, and shared-memory
//! tensor exchange between cooperating processes — including the
//! eventual `RemoteHostStorage` consumers in Phase 7c. There is no
//! longer a Tensor-construction layer stacked on top: B6 deleted the
//! eager `Tensor` those wrappers built, and `fuel-core`'s lazy loaders
//! read parsed bytes straight out of the mmap.
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
