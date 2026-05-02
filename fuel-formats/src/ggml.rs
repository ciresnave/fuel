//! GGML legacy tensor format parser.
//!
//! Migration target for `fuel-core/src/quantized/ggml_file.rs`
//! (Phase 7.5 work item A). Owns the magic/version detection,
//! hyperparameter blocks, vocabulary table, and tensor descriptor
//! decoding. Tensor-construction wrappers stay in `fuel-core` until
//! work item E lands.
//!
//! Pre-extraction reference: [`fuel-core/src/quantized/ggml_file.rs`].
