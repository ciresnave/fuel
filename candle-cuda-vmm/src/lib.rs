//! # candle-cuda-vmm
//!
//! CUDA Virtual Memory Management bindings for elastic KV cache allocation in Candle.
//!
//! This crate provides safe Rust bindings to CUDA's Virtual Memory Management (VMM) APIs,
//! enabling elastic memory allocation for LLM inference workloads. It integrates with the
//! Candle deep learning framework and supports:
//!
//! - **Elastic KV Cache Allocation**: Allocate memory on-demand rather than pre-allocating
//!   large static buffers
//! - **Multi-Model Serving**: Share GPU memory pools across multiple models with dynamic
//!   allocation
//! - **Reduced TTFT**: Faster time-to-first-token (1.2-28×) in multi-model scenarios vs
//!   static allocation
//! - **Memory Efficiency**: Optimal memory usage for bursty multi-tenant workloads
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//!
//! - [`error`]: Error types for VMM operations
//! - [`cuda_ffi`]: Low-level CUDA VMM FFI bindings
//! - [`physical_memory`]: Physical GPU memory allocation with RAII
//! - [`mapping`]: Virtual address space reservation and mapping operations
//! - [`virtual_memory`]: High-level elastic memory pool abstractions
//!
//! ## Quick Start
//!
//! ```no_run
//! use candle_cuda_vmm::{VirtualMemoryPool, Result};
//! use candle::Device;
//!
//! fn main() -> Result<()> {
//!     let device = Device::new_cuda(0)?;
//!     
//!     // Create a pool with 128GB virtual capacity, 2MB pages
//!     let mut pool = VirtualMemoryPool::new(
//!         128 * 1024 * 1024 * 1024, // 128GB virtual
//!         2 * 1024 * 1024,          // 2MB pages
//!         device,
//!     )?;
//!     
//!     // Allocate 1GB of physical memory on-demand
//!     let addr = pool.allocate(0, 1024 * 1024 * 1024)?;
//!     println!("Allocated at virtual address: 0x{:x}", addr);
//!     
//!     // Physical memory usage: ~1GB
//!     println!("Physical usage: {} bytes", pool.physical_memory_usage());
//!     
//!     // Deallocate when done
//!     pool.deallocate(0, 1024 * 1024 * 1024)?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-Model Serving
//!
//! ```no_run
//! use candle_cuda_vmm::{SharedMemoryPool, Result};
//! use candle::Device;
//!
//! fn main() -> Result<()> {
//!     let device = Device::new_cuda(0)?;
//!     let mut shared_pool = SharedMemoryPool::new(
//!         32 * 1024 * 1024 * 1024, // 32GB global physical limit
//!         device,
//!     )?;
//!     
//!     // Register models
//!     shared_pool.register_model("llama-7b", 64 * 1024 * 1024 * 1024)?;
//!     shared_pool.register_model("gpt2", 32 * 1024 * 1024 * 1024)?;
//!     
//!     // Allocate for specific model
//!     let addr = shared_pool.allocate_for_model("llama-7b", 1024 * 1024 * 1024)?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Requirements
//!
//! - CUDA 11.2 or later (CUDA VMM APIs introduced in 11.2)
//! - NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
//! - Rust 1.70+
//!
//! ## Performance
//!
//! Based on KVCached benchmarks:
//!
//! - **Allocation Latency**: <100μs per 2MB page
//! - **TTFT Improvement**: 1.2-28× faster vs static allocation (multi-model scenarios)
//! - **Memory Overhead**: <5% metadata overhead
//! - **Throughput**: No degradation vs static allocation for single-model workloads

pub mod error;
pub mod cuda_ffi;
pub mod physical_memory;
pub mod mapping;
pub mod virtual_memory;

// Re-export main types
pub use error::{Result, VmmError};
pub use physical_memory::PhysicalMemoryHandle;
pub use mapping::{VirtualAddressRange, map_memory, unmap_memory, set_memory_access};
pub use virtual_memory::{
    VirtualMemoryPool, SharedMemoryPool, MemoryStats, GlobalMemoryStats
};
pub use cuda_ffi::AccessFlags;

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if CUDA VMM is supported on the current system.
///
/// # Returns
/// True if CUDA VMM is available, false otherwise.
pub fn is_vmm_supported() -> bool {
    // Try to get granularity for device 0 - if this fails, VMM is not supported
    cuda_ffi::get_recommended_granularity(0).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
