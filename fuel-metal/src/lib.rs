//! Metal backend implementation for the fuel ML framework.
//!
//! This crate provides [`MetalStorage`] and [`MetalDevice`] types that
//! implement all tensor operations via Apple Metal. It depends only on
//! `fuel-core-types` (not `fuel-core`) so that the higher-level crate
//! can provide the thin `BackendStorage` / `BackendDevice` trait delegation.
//!
//! On non-Apple platforms the crate compiles but is empty.

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod device;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod dyn_impl;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod quantized;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod storage;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "ug"))]
pub mod ug;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use fuel_core_types::{DType, Error, Layout, Result, Shape};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use device::{DeviceId, MetalDevice};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use dyn_impl::{MetalBackendDevice, MetalBackendStorage};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use storage::{LockError, MetalError, MetalStorage, buffer_o};

// Re-export underlying Metal bindings for downstream use.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use fuel_metal_kernels;
