//! Baracuda kernel integration — alpha.27 ML op surface as alternative
//! implementations in Fuel's binding table.
//!
//! Architecture v1.0 + Phase 7.6 step 9a: every baracuda kernel
//! registers as a sibling alternative alongside Fuel's existing PTX
//! kernels (in [`fuel_cuda_kernels`]) and any NVIDIA-library-backed
//! paths. The route picker (step 9b) ranks competing impls by
//! [`crate::fused::PrecisionGuarantee`] + telemetry.
//!
//! ## Why -sys not the safe `baracuda-kernels` Plan wrapper
//!
//! Baracuda's safe layer caches kernel selection in a `Plan` instance
//! (`Plan::select(stream, descriptor, preference) → Plan`). That
//! doesn't fit Fuel's KernelRef function-pointer dispatch — Plans own
//! state, KernelRefs are fungible function pointers. Wrapping each
//! `extern "C"` symbol directly here keeps the binding-table entries
//! pointer-equal and lets Phase 7.6 step 9a's append-on-register
//! semantics treat baracuda kernels as ordinary alternatives.
//!
//! ## Layout
//!
//! - [`status`] — `i32` status-code → [`crate::CudaError`] mapping per
//!   baracuda's documented contract (`0` = ok, `1` = misalign,
//!   `2` = invalid problem, `3` = unsupported, `4` = workspace too
//!   small, `5` = internal launch failure).
//! - [`scratch`] — per-call workspace allocation helper. Each kernel's
//!   `_workspace_size(...)` is called before `_run(...)`; on non-zero
//!   need, a fresh device buffer is allocated for the call. (A
//!   per-stream scratch pool is a future optimization — for now the
//!   simple alloc-per-call model is correct and small enough to be
//!   bounded.)
//! - [`shape_strides`] — converts Fuel's [`fuel_core_types::Layout`] to
//!   the `(rank: i32, shape: *const i32, stride: *const i64)` triple
//!   baracuda expects. Validates that dims fit in `i32` (baracuda's
//!   shape dtype).
//! - Per-family submodules (`elementwise`, `reduce`, `softmax`, `norm`,
//!   `attention`, `gguf`, `moe`, `gemm_int`, `gemm_fp8`, `gemm_int4`,
//!   `indexing`, `segment`, `embedding`, `sort`, `image`, `quantize`,
//!   `loss`, `random`, `fft`, `linalg`, `scan`) each wrap their slice
//!   of the symbol space as `KernelRef`s and expose a `register_*`
//!   helper for the binding-table side.

#![allow(dead_code)] // submodules land incrementally; intermediate dead-code is expected.

pub mod binary;
pub mod elementwise;
pub mod norm;
pub mod reduce;
pub mod scratch;
pub mod shape_strides;
pub mod status;
