//! Safetensors parser.
//!
//! Migration target for `fuel-core/src/safetensors.rs` (Phase 7.5
//! work item A). This module owns the format-parsing surface:
//! header decoding, metadata structs, byte-range computation,
//! `MmapedFile` plumbing. The thin Tensor-construction wrappers
//! (`fn load(path, device)`, `MmapedSafetensors::load`, etc.) stay
//! in `fuel-core` for now and migrate to `fuel-loaders` once
//! work item E lands.
//!
//! Pre-extraction reference: [`fuel-core/src/safetensors.rs`].
