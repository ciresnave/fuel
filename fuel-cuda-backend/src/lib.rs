//! CUDA GPU executor for `fuel-graph` computation graphs + low-level
//! CUDA primitives (storage, device, kernel dispatch, cuBLAS / cuDNN
//! wrappers). The parallel Vulkan stack (`vulkane` → `fuel-vulkan-backend`)
//! uses the same shape: the external FFI crate family (`baracuda-*` here,
//! `vulkane` there) provides raw bindings; this crate layers the
//! ML-specific dtype-tagged storage, kernel dispatch, and graph
//! integration on top.
//!
//! ## Modules
//!
//! - [`device`] — `CudaDevice` wrapping `baracuda-driver`'s Context + Stream
//!   + module cache + cuBLAS handle + curand generator.
//! - [`storage`] — `CudaStorage` (dtype-tagged tensor) + all tensor-op
//!   dispatch (matmul, conv, softmax, rope, rms_norm, quantized matmul,
//!   gather/scatter, pooling, upsample).
//! - [`utils`] — `Map1` / `Map2` / `Map3` / `Map*Any` dtype-dispatch traits.
//! - [`dyn_impl`] — object-safe `BackendDevice` / `BackendStorage` impls.
//! - [`error`] — `CudaError` + `WrapErr` trait for baracuda error conversion.
//! - [`cudnn`] — optional convolution wrapper (feature: `cudnn`).
//!
//! ## Execution model
//!
//! All intermediates stay in GPU memory; host↔device transfer happens
//! only at `Const` upload (H2D) and `realize_*` readback (D2H).
//!
//! Model weights upload **once** (first forward pass) and persist in
//! the executor's `const_pool` for the executor's lifetime. KV-cache
//! consts and computed intermediates are owned per-realize and freed
//! at the end of each call.

// --- fuel-cuda primitives (formerly a separate crate) -----------------------
pub use fuel_ir::{DType, Error, Layout, Result, Shape};

// `crate::cudnn` retired in Phase 5b of the fuel-cuda-kernels retirement
// (2026-05-25). Fuel's internal cuDNN wrapper for conv2d/conv1d (252 LOC)
// is no longer needed — conv dispatch goes through
// `baracuda-kernels-sys::baracuda_kernels_conv_*_run` instead. The
// `cudnn` Cargo feature is now a near-no-op (only used to gate the
// transitive `baracuda-cudnn{,-sys}` deps that don't have other users
// in Fuel).
// The baracuda FA2 launcher (`flash_attn::launch`). Staged but not yet
// wired into `Op::FlashAttn` dispatch — see the module docs. Formerly
// gated behind the `flash-attn` Cargo feature, which existed only to
// reach the now-deleted eager `fuel-flash-attn-cuda{,-sys}` crates; the
// launcher itself depends only on `baracuda-kernels-sys` (always present)
// so it compiles unconditionally now.
pub mod flash_attn;
pub mod baracuda;
/// Re-export of `baracuda_kernels_sys` so downstream crates (like
/// `fuel-core`) can call baracuda FFI symbols without pulling
/// `baracuda-kernels-sys` in as a direct dep.
pub use baracuda_kernels_sys;
pub mod byte_storage;
pub mod capture;
pub mod cutlass;
pub mod device;
pub mod dyn_impl;
pub mod error;
pub mod pinned;
pub mod probe;
pub mod quantized;
pub mod storage;
#[cfg(feature = "ug")]
pub mod ug;
pub mod utils;

pub use byte_storage::CudaStorageBytes;
pub use capture::CapturedRun;
/// Step E A4b-1: the async-completion primitive the executor defers waits on.
/// Re-exported from `baracuda_driver` so `fuel-dispatch`'s `CudaCompletion`
/// can name the type without depending on baracuda directly.
pub use baracuda_driver::Event;
pub use device::{CublasHandle, CudaDevice, CudaFunc, DeviceId, LaunchArgs, LaunchConfig};
pub use dyn_impl::{CudaBackendDevice, CudaBackendStorage};
pub use error::{CudaError, WrapErr};
pub use pinned::PinnedHostStorage;
pub use storage::{CudaStorage, CudaStorageSlice, SlicePtrOrNull, kernel_name};
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

// --- graph executor integration (RETIRED) -----------------------------------
// The legacy `CudaGraphExecutor` (a per-node graph walker with a
// `fuel_reference_backend::exec::eval_node_with_op` CPU fallback) was deleted
// 2026-07-04 alongside the retirement of `fuel-reference-backend`. It had no
// live constructor — production CUDA realize has run entirely through
// `fuel-dispatch`'s PipelinedExecutor since executor-unification. The
// per-op correctness oracle is now CPU-backend-vs-CUDA pairwise consensus
// (`LazyTensor::realize_f32_reference` + `test_utils::assert_cuda_matches_reference`).
// The FA2 launcher survives in `crate::flash_attn::launch`.
