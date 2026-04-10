//! CUDA backend implementation for the candle ML framework.
//!
//! This crate provides [`CudaStorage`] and [`CudaDevice`] types that
//! implement all tensor operations via NVIDIA CUDA. It depends only on
//! `candle-core-types` (not `candle-core`) so that the higher-level crate
//! can provide the thin `BackendStorage` / `BackendDevice` trait delegation.

pub use candle_core_types::{DType, Error, Layout, Result, Shape};

#[cfg(feature = "cudnn")]
pub mod cudnn;
pub mod device;
pub mod dyn_impl;
pub mod error;
pub mod storage;
pub mod utils;

pub use device::{CudaDevice, DeviceId};
pub use dyn_impl::{CudaBackendDevice, CudaBackendStorage};
pub use error::{CudaError, WrapErr};
pub use storage::{CudaStorage, CudaStorageSlice, SlicePtrOrNull, kernel_name};
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

// Re-export underlying CUDA bindings for downstream use.
pub use candle_kernels as kernels;
pub use cudarc;
