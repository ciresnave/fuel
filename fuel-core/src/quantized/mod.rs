//! Quantized (ggml/gguf block-format) support for fuel-core.
//!
//! Per-backend kernels live in their own crates:
//! - CPU: `fuel-cpu-backend::quantized::CpuQStorage`
//! - CUDA: `fuel-cuda-backend::quantized::QCudaStorage`
//! - Metal: `fuel-metal-backend::quantized::QMetalStorage`
//!
//! fuel-core holds the file-format readers (gguf/imatrix). All per-backend
//! dispatch goes through
//! [`fuel_backend_contract::quantized::DynQuantizedStorage`] and
//! [`fuel_backend_contract::quantized::QuantizedDeviceKernels`] — adding a new
//! backend requires zero edits in this module.
//!
//! B6 removed the eager front-end: `QTensor`, `QMatMul`, `load_quantized`,
//! `check_shape`, `impl Device`, `impl CustomOp1 for QTensor` and
//! `impl Module for QMatMul` were all built on the eager `Tensor` — as was
//! `ggml_file` in its entirety, which existed only to promote a
//! [`fuel_formats::ggml::RawTensor`] into a `QTensor`. Raw GGML wire-format
//! parsing is unaffected and still lives in [`fuel_formats::ggml`].
//!
//! What survives is what the lazy stack actually uses: [`GgmlDType`],
//! [`gguf_mmap::MmapedContent`], and `gguf_file::{Content, Value, TensorInfo}`.
//! Lazy loaders take the mmap plus a `TensorInfo` and slice the tensor bytes
//! out themselves.

pub use fuel_backend_contract::quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
pub use fuel_ir::quantized::GgmlDType;
pub use fuel_quantized::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8_1, BlockQ8K, GgmlType, QuantizedType,
};

pub mod arch;
pub mod gguf_file;
#[cfg(not(target_arch = "wasm32"))]
pub mod gguf_mmap;
pub mod imatrix_file;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;

/// A backend-agnostic quantized storage buffer, owned by exactly one device.
///
/// Produced by [`QuantizedDeviceKernels`] implementations; consumed by the
/// per-backend quantized matmul kernels.
pub type QStorage = Box<dyn DynQuantizedStorage>;
