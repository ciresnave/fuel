//! Error types for CUDA Virtual Memory Management operations.

use thiserror::Error;

/// Result type for VMM operations.
pub type Result<T> = std::result::Result<T, VmmError>;

/// Errors that can occur during CUDA Virtual Memory Management operations.
#[derive(Debug, Error)]
pub enum VmmError {
    /// A CUDA driver API call failed.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Out of virtual address space.
    #[error(
        "Out of virtual address space: requested {requested} bytes, available {available} bytes"
    )]
    OutOfVirtualMemory { requested: usize, available: usize },

    /// Out of physical GPU memory.
    #[error("Out of physical memory: requested {requested} bytes, available {available} bytes")]
    OutOfPhysicalMemory { requested: usize, available: usize },

    /// Invalid offset into virtual address range.
    #[error("Invalid offset: {offset} (size: {size}, capacity: {capacity})")]
    InvalidOffset {
        offset: usize,
        size: usize,
        capacity: usize,
    },

    /// Memory mapping operation failed.
    #[error("Mapping failed: {0}")]
    MappingFailed(String),

    /// Memory unmapping operation failed.
    #[error("Unmapping failed: {0}")]
    UnmappingFailed(String),

    /// Invalid alignment for memory operation.
    #[error("Invalid alignment: {actual}, required: {required}")]
    InvalidAlignment { actual: usize, required: usize },

    /// Range is already mapped.
    #[error("Range already mapped: offset {offset}, size {size}")]
    AlreadyMapped { offset: usize, size: usize },

    /// Range is not mapped.
    #[error("Range not mapped: offset {offset}, size {size}")]
    NotMapped { offset: usize, size: usize },

    /// Invalid page size.
    #[error("Invalid page size: {0} (must be power of 2 and >= 64KB)")]
    InvalidPageSize(usize),

    /// Device not compatible with VMM.
    #[error("Device does not support CUDA Virtual Memory Management")]
    UnsupportedDevice,

    /// Candle error.
    #[error("Candle error: {0}")]
    CandleError(#[from] candle::Error),

    /// Model not found in shared pool.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Model already registered in shared pool.
    #[error("Model already registered: {0}")]
    ModelAlreadyExists(String),

    /// Generic error with custom message.
    #[error("{0}")]
    Other(String),
}

impl VmmError {
    /// Create a CUDA error from a cudarc result.
    pub fn from_cuda_result(result: cudarc::driver::result::DriverError) -> Self {
        VmmError::CudaError(format!("{:?}", result))
    }

    /// Create a CUDA error with custom message.
    pub fn cuda<S: Into<String>>(msg: S) -> Self {
        VmmError::CudaError(msg.into())
    }

    /// Create a generic error with custom message.
    pub fn other<S: Into<String>>(msg: S) -> Self {
        VmmError::Other(msg.into())
    }
}
