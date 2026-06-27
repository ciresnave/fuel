//! Core vocabulary types for the fuel ML framework.
//!
//! This crate contains the foundational types (`DType`, `Shape`, `Layout`, `Error`),
//! the backend capability **data** (`BackendCapabilities`, `SubstrateClass`,
//! `TransferPath`, `FitStatus`, `GgmlDType`, bundle `OutputView`s), and CPU
//! storage types (`HostBuffer`, `HostBufferRef`, `CpuDevice`) shared across all
//! fuel backend crates.
//!
//! The object-safe backend-contract **traits** (`DynBackendStorage`,
//! `DynBackendDevice`, `HostStorage`, `BackendStorage`, `BackendRuntime`,
//! `BackendCapabilityProvider`, `DynQuantizedStorage`, `QuantizedDeviceKernels`,
//! `InplaceOp1/2/3`) and the type-erased `Storage` handle live in the
//! `fuel-backend-contract` crate (above this crate, below the backends).

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
mod cpu_storage;
mod device;
pub mod dispatch;
pub mod dtype;
pub mod dummy_dtype;
pub mod error;
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
pub use cpu_storage::{CpuDevice, CpuStorage, CpuStorageRef, HostBuffer, HostBufferRef, HostDType};
pub use device::DeviceLocation;
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use layout::Layout;
pub use quant_scale::{ScaleGranularity, ScalePair};
pub use quantized::GgmlDType;
pub use scalar::Scalar;
pub use shape::{D, Dim, Dims, Shape, ShapeWithOneHole};
pub use strided_index::{StridedBlocks, StridedIndex};
pub use symbol::{DynScalar, SymEnv, SymGen, SymId};
