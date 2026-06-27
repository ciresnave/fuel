//! Backend-contract traits for the fuel ML framework.
//!
//! This crate holds the object-safe trait surface that **every backend
//! implements** — the contract Fuel dispatches through without naming
//! concrete backend types. It sits **above [`fuel_ir`]** (the vocabulary
//! crate it borrows `DType`/`Shape`/`Layout`/`HostBuffer`/the capability
//! data-types from) and **below the backend crates** (`fuel-cpu-backend`,
//! `fuel-cuda-backend`, `fuel-vulkan-backend`, `fuel-metal-backend`, …),
//! which depend on it to implement the traits.
//!
//! ## What lives here
//!
//! - [`dyn_backend`] — [`DynBackendStorage`] / [`DynBackendDevice`], the
//!   object-safe storage + device traits at the heart of dispatch.
//! - [`backend`] — the capability/runtime traits [`HostStorage`],
//!   [`BackendStorage`], [`BackendCapabilityProvider`], [`BackendRuntime`].
//!   (Their *data* types — `SubstrateClass`, `TransferPath`,
//!   `BackendCapabilities`, `FitStatus` — stay in [`fuel_ir::backend`].)
//! - [`quantized`] — [`DynQuantizedStorage`] / [`QuantizedDeviceKernels`].
//!   (The `GgmlDType` *data* tag stays in [`fuel_ir::quantized`].)
//! - [`inplace_op`] — [`InplaceOp1`] / [`InplaceOp2`] / [`InplaceOp3`].
//! - [`storage`] — [`Storage`], the type-erased `Box<dyn DynBackendStorage>`
//!   handle that carries a backend's storage, plus `allocate_bundled_storage`.
//!   (The bundle *data* — `OutputView`, `OutputViewSpec`, `compose_bundle` —
//!   stays in [`fuel_ir::storage`].)

pub mod backend;
pub mod dyn_backend;
pub mod inplace_op;
pub mod quantized;
pub mod storage;

pub use backend::{BackendCapabilityProvider, BackendRuntime, BackendStorage, HostStorage};
pub use dyn_backend::{DynBackendDevice, DynBackendStorage};
pub use inplace_op::{InplaceOp1, InplaceOp2, InplaceOp3};
pub use quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
pub use storage::{Storage, allocate_bundled_storage};
