//! Python pickle parser (PyTorch `.pth` / `.bin` checkpoints).
//!
//! Migration target for `fuel-core/src/pickle.rs` (Phase 7.5
//! work item A). Owns the opcode interpreter, `Stack`,
//! `Object`, `TensorInfo`, and the bytes-to-stream readers. The
//! Tensor-builder wrappers (`fn read_all`, `fn read_pth_tensor_info`,
//! `PthTensors`) stay in `fuel-core` until work item E lands.
//!
//! Pre-extraction reference: [`fuel-core/src/pickle.rs`].
