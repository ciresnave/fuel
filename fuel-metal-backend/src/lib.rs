// SPDX-License-Identifier: MIT OR Apache-2.0
//! Metal backend implementation for the fuel ML framework.
//!
//! This crate provides [`MetalStorage`] and [`MetalDevice`] types that
//! implement all tensor operations via Apple Metal. It depends only on
//! `fuel-core-types` (not `fuel-core`) so that the higher-level crate
//! can provide the thin `BackendStorage` / `BackendDevice` trait delegation.
//!
//! On non-Apple platforms the crate compiles but is otherwise empty —
//! [`metal_is_available`] is the one function with no platform gate (see
//! its own doc comment for why).

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod byte_storage;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod device;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod dyn_impl;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod quantized;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod storage;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use fuel_ir::{D, DType, Error, Layout, Result, Shape};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use byte_storage::MetalStorageBytes;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use device::{DeviceId, MetalDevice};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use dyn_impl::{MetalBackendDevice, MetalBackendStorage};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use storage::{LockError, MetalError, MetalStorage, buffer_o};

// Re-export underlying Metal bindings for downstream use.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use fuel_metal_kernels;

/// Returns `true` unconditionally.
///
/// Moved from `fuel_core::utils::metal_is_available` (fuel-core
/// dissolution, GAP-347 PR 7). Unlike `fuel-cpu-backend`'s
/// `accelerate`/`mkl` features, this crate has no `metal` feature of its
/// own to check: it is an `optional = true` dependency gated entirely by
/// the CALLER's `metal` feature (`dep:fuel-metal-backend` in
/// `fuel-core/Cargo.toml`, forwarded by `fuel`'s own `metal` feature). So
/// this function being reachable at all — the crate being present in the
/// dependency graph — IS the availability signal; there is no separate
/// condition left to evaluate. Deliberately NOT platform-gated: the
/// function it replaces was also platform-blind, a bare `cfg!(feature =
/// "metal")` check with no `target_os` condition, so a build that forces
/// `--features metal` on a non-Apple platform returns `true` here exactly
/// as it did before this move (the REST of this crate still compiles to
/// empty on those platforms — see the module doc comment).
pub fn metal_is_available() -> bool {
    true
}
