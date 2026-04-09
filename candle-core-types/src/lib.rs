//! Core types and traits for the candle ML framework.
//!
//! This crate contains the foundational types (`DType`, `Shape`, `Layout`, `Error`),
//! backend traits (`BackendStorage`, `BackendDevice`), and CPU storage types
//! (`CpuStorage`, `CpuStorageRef`, `CpuDevice`) that are shared across all candle
//! backend crates.

/// A small-vector type for dimension/stride storage.
/// Avoids heap allocation for tensors with up to 6 dimensions.
pub type DimVec = smallvec::SmallVec<[usize; 6]>;

pub mod backend;
pub mod conv;
pub mod cpu;
mod cpu_storage;
mod device;
pub mod dtype;
pub mod dummy_dtype;
pub mod error;
pub mod layout;
mod metal_error;
pub mod op;
pub mod scalar;
pub mod shape;
pub mod strided_index;

pub use cpu_storage::{CpuDevice, CpuStorage, CpuStorageRef};
pub use device::DeviceLocation;
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use layout::Layout;
pub use metal_error::MetalError;
pub use scalar::Scalar;
pub use shape::{D, Dim, Dims, Shape, ShapeWithOneHole};
pub use strided_index::{StridedBlocks, StridedIndex};
