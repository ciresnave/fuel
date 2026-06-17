//! Core types and traits for the fuel ML framework.
//!
//! This crate contains the foundational types (`DType`, `Shape`, `Layout`, `Error`),
//! the dyn backend traits (`DynBackendStorage`, `DynBackendDevice`), the
//! orthogonal `HostStorage` capability marker, and CPU storage types
//! (`HostBuffer`, `HostBufferRef`, `CpuDevice`) that are shared across all
//! fuel backend crates.

/// A small-vector type for dimension storage (sizes are non-negative).
/// Avoids heap allocation for tensors with up to 6 dimensions.
pub type DimVec = smallvec::SmallVec<[usize; 6]>;

/// A small-vector type for **signed** strides. Strides are signed
/// because metadata-only view ops can produce negative steps —
/// `Op::Flip` reverses iteration along a dim by negating that dim's
/// stride and adjusting `start_offset`, with zero kernel work. Most
/// strides are positive in practice; the signed type widens the
/// representation to cover the negative-stride case without forcing
/// every kernel to allocate or convert.
pub type StrideVec = smallvec::SmallVec<[isize; 6]>;

pub mod backend;
pub mod capability;
pub mod conv;
pub mod cpu;
mod cpu_storage;
mod device;
pub mod dispatch;
pub mod dtype;
pub mod dummy_dtype;
pub mod dyn_backend;
pub mod error;
pub mod inplace_op;
pub mod layout;
pub mod op;
pub mod probe;
pub mod quant_scale;
pub mod quantized;
pub mod scalar;
pub mod shape;
pub mod storage;
pub mod symbol;
pub mod strided_index;

pub use capability::Capability;
pub use probe::{BackendId, BackendProbe, DeviceDescriptor, EquivalenceKey};
pub use cpu_storage::{CpuDevice, CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef};
pub use device::DeviceLocation;
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use dyn_backend::{DynBackendDevice, DynBackendStorage};
pub use error::{Context, Error, Result};
pub use inplace_op::{InplaceOp1, InplaceOp2, InplaceOp3};
pub use layout::Layout;
pub use quant_scale::{ScaleGranularity, ScalePair};
pub use quantized::{DynQuantizedStorage, GgmlDType, QuantizedDeviceKernels};
pub use scalar::Scalar;
pub use shape::{D, Dim, Dims, Shape, ShapeWithOneHole};
pub use storage::Storage;
pub use strided_index::{StridedBlocks, StridedIndex};
pub use symbol::{DynScalar, SymEnv, SymGen, SymId};
