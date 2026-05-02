//! GGUF (GGML Universal File) parser — llama.cpp quantized model format.
//!
//! Migration target for:
//! - `fuel-core/src/quantized/gguf_file.rs` (header, metadata, value
//!   decoding, tensor-info table, write path)
//! - `fuel-core/src/quantized/gguf_mmap.rs` (mmap-backed reader)
//!
//! Owns: `VersionedMagic`, `TensorInfo`, `Content`, `Value`,
//! `ValueType`, `MmapedGguf` (or equivalent), and the byte-level
//! reader / writer. The Tensor-construction wrappers (calls into
//! `Device::qzeros` + `load_quantized`) stay in `fuel-core` until
//! work item E lands.
//!
//! Pre-extraction reference: [`fuel-core/src/quantized/gguf_file.rs`],
//! [`fuel-core/src/quantized/gguf_mmap.rs`].
