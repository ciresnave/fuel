//! Virtual memory pool for elastic memory allocation.

use crate::cuda_ffi::{self, AccessFlags};
use crate::error::{Result, VmmError};
use crate::mapping::{map_memory, set_memory_access, unmap_memory, VirtualAddressRange};
use crate::physical_memory::PhysicalMemoryHandle;
use candle::{Device, DeviceLocation};
use std::collections::HashMap;

/// Helper function to extract device ordinal from Candle Device
fn get_device_ordinal(device: &Device) -> Result<i32> {
    match device.location() {
        DeviceLocation::Cuda { gpu_id } => Ok(gpu_id as i32),
        _ => Err(VmmError::other("Device must be a CUDA device")),
    }
}

/// Page state in the virtual memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageState {
    /// Page is not allocated (no physical memory).
    Free,
    /// Page is allocated and mapped to physical memory.
    Allocated,
}

/// Elastic memory pool with virtual memory backing.
///
/// This pool reserves a large virtual address space but only allocates physical
/// memory on-demand when `allocate()` is called. This enables:
/// - Large virtual capacity (e.g., 128GB) with minimal initial physical usage
/// - Dynamic allocation/deallocation based on workload
/// - Reduced memory waste for bursty workloads
///
/// # Example
/// ```no_run
/// use candle_cuda_vmm::VirtualMemoryPool;
/// use candle::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::new_cuda(0)?;
/// let mut pool = VirtualMemoryPool::new(
///     128 * 1024 * 1024 * 1024, // 128GB virtual capacity
///     2 * 1024 * 1024,          // 2MB page size
///     device,
/// )?;
///
/// // Allocate 1GB of physical memory on-demand
/// let addr = pool.allocate(0, 1024 * 1024 * 1024)?;
/// println!("Physical usage: {} bytes", pool.physical_memory_usage());
///
/// // Deallocate when done
/// pool.deallocate(0, 1024 * 1024 * 1024)?;
/// # Ok(())
/// # }
/// ```
pub struct VirtualMemoryPool {
    /// Virtual address range reservation.
    virtual_range: VirtualAddressRange,
    /// Physical memory handles for each page (indexed by page number).
    physical_pages: HashMap<usize, PhysicalMemoryHandle>,
    /// Page state tracking.
    page_states: Vec<PageState>,
    /// Page size in bytes.
    page_size: usize,
    /// Total virtual capacity in bytes.
    total_capacity: usize,
    /// Currently mapped size in bytes.
    mapped_size: usize,
    /// Device ordinal.
    device_ordinal: i32,
}

impl VirtualMemoryPool {
    /// Create a new virtual memory pool.
    ///
    /// # Arguments
    /// * `capacity` - Maximum virtual address space (e.g., 128GB).
    /// * `page_size` - Page granularity (e.g., 2MB for large pages).
    /// * `device` - CUDA device.
    ///
    /// # Returns
    /// Pool with reserved virtual address space, no physical memory allocated.
    ///
    /// # Errors
    /// Returns error if:
    /// - Device is not a CUDA device
    /// - Page size is invalid (not power of 2 or < 64KB)
    /// - Virtual address reservation fails
    pub fn new(capacity: usize, page_size: usize, device: Device) -> Result<Self> {
        // Validate device
        let device_ordinal = get_device_ordinal(&device)?;

        // Validate page size
        if !page_size.is_power_of_two() || page_size < 64 * 1024 {
            return Err(VmmError::InvalidPageSize(page_size));
        }

        // Ensure capacity is multiple of page size
        let capacity = (capacity + page_size - 1) / page_size * page_size;

        // Reserve virtual address space
        let virtual_range = VirtualAddressRange::new(capacity, page_size)?;

        // Calculate number of pages
        let num_pages = capacity / page_size;

        Ok(Self {
            virtual_range,
            physical_pages: HashMap::new(),
            page_states: vec![PageState::Free; num_pages],
            page_size,
            total_capacity: capacity,
            mapped_size: 0,
            device_ordinal,
        })
    }

    /// Allocate and map physical pages on-demand.
    ///
    /// # Arguments
    /// * `offset` - Offset in virtual address space (bytes).
    /// * `size` - Number of bytes to allocate.
    ///
    /// # Returns
    /// Base virtual address of allocated region.
    ///
    /// # Errors
    /// Returns error if:
    /// - Offset/size out of bounds
    /// - Region already allocated
    /// - Physical memory allocation fails
    pub fn allocate(&mut self, offset: usize, size: usize) -> Result<usize> {
        // Validate parameters
        if offset + size > self.total_capacity {
            return Err(VmmError::InvalidOffset {
                offset,
                size,
                capacity: self.total_capacity,
            });
        }

        // Align offset and size to page boundaries
        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        // Check if any pages are already allocated
        for page_idx in start_page..end_page {
            if self.page_states[page_idx] == PageState::Allocated {
                return Err(VmmError::AlreadyMapped {
                    offset: page_idx * self.page_size,
                    size: self.page_size,
                });
            }
        }

        // Allocate and map each page
        for page_idx in start_page..end_page {
            // Allocate physical memory for this page
            let device = Device::new_cuda(self.device_ordinal as usize)?;
            let physical_handle = PhysicalMemoryHandle::new(self.page_size, &device)?;

            // Map physical memory to virtual address
            let page_offset = page_idx * self.page_size;
            map_memory(
                &self.virtual_range,
                page_offset,
                &physical_handle,
                0,
                self.page_size,
            )?;

            // Set memory access permissions
            set_memory_access(
                &self.virtual_range,
                page_offset,
                self.page_size,
                self.device_ordinal,
                AccessFlags::ReadWrite,
            )?;

            // Store physical handle and update state
            self.physical_pages.insert(page_idx, physical_handle);
            self.page_states[page_idx] = PageState::Allocated;
            self.mapped_size += self.page_size;
        }

        Ok(self.virtual_range.base_address() + offset)
    }

    /// Unmap and free physical pages.
    ///
    /// # Arguments
    /// * `offset` - Offset in virtual address space (bytes).
    /// * `size` - Number of bytes to free.
    ///
    /// # Errors
    /// Returns error if:
    /// - Offset/size out of bounds
    /// - Region not allocated
    pub fn deallocate(&mut self, offset: usize, size: usize) -> Result<()> {
        // Validate parameters
        if offset + size > self.total_capacity {
            return Err(VmmError::InvalidOffset {
                offset,
                size,
                capacity: self.total_capacity,
            });
        }

        // Align offset and size to page boundaries
        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        // Check if all pages are allocated
        for page_idx in start_page..end_page {
            if self.page_states[page_idx] == PageState::Free {
                return Err(VmmError::NotMapped {
                    offset: page_idx * self.page_size,
                    size: self.page_size,
                });
            }
        }

        // Unmap and free each page
        for page_idx in start_page..end_page {
            let page_offset = page_idx * self.page_size;

            // Unmap virtual memory
            unmap_memory(&self.virtual_range, page_offset, self.page_size)?;

            // Remove physical handle (automatically freed via Drop)
            self.physical_pages.remove(&page_idx);
            self.page_states[page_idx] = PageState::Free;
            self.mapped_size -= self.page_size;
        }

        Ok(())
    }

    /// Get current physical memory usage in bytes.
    pub fn physical_memory_usage(&self) -> usize {
        self.mapped_size
    }

    /// Get virtual address space capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.total_capacity
    }

    /// Get base virtual address.
    pub fn base_address(&self) -> usize {
        self.virtual_range.base_address()
    }

    /// Get page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Check if a range is currently mapped.
    ///
    /// # Arguments
    /// * `offset` - Offset in virtual address space (bytes).
    /// * `size` - Size to check (bytes).
    ///
    /// # Returns
    /// True if entire range is mapped, false otherwise.
    pub fn is_mapped(&self, offset: usize, size: usize) -> bool {
        if offset + size > self.total_capacity {
            return false;
        }

        let start_page = offset / self.page_size;
        let end_page = (offset + size + self.page_size - 1) / self.page_size;

        for page_idx in start_page..end_page {
            if self.page_states[page_idx] != PageState::Allocated {
                return false;
            }
        }

        true
    }

    /// Compact pool by coalescing free pages (no-op for now, future optimization).
    pub fn compact(&mut self) -> Result<()> {
        // Future: Implement compaction to reduce fragmentation
        Ok(())
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let allocated_pages = self
            .page_states
            .iter()
            .filter(|&&state| state == PageState::Allocated)
            .count();

        let total_pages = self.page_states.len();
        let fragmentation_ratio = if total_pages > 0 {
            1.0 - (allocated_pages as f32 / total_pages as f32)
        } else {
            0.0
        };

        MemoryStats {
            virtual_capacity: self.total_capacity,
            physical_usage: self.mapped_size,
            mapped_pages: allocated_pages,
            fragmentation_ratio,
        }
    }
}

/// Memory statistics for a pool.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Virtual address space capacity in bytes.
    pub virtual_capacity: usize,
    /// Physical memory usage in bytes.
    pub physical_usage: usize,
    /// Number of mapped pages.
    pub mapped_pages: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = completely fragmented).
    pub fragmentation_ratio: f32,
}

/// Shared memory pool for multiple models.
///
/// Manages multiple virtual memory pools with a global physical memory limit.
/// Enables memory sharing across models with per-model statistics.
///
/// # Example
/// ```no_run
/// use candle_cuda_vmm::SharedMemoryPool;
/// use candle::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::new_cuda(0)?;
/// let mut shared_pool = SharedMemoryPool::new(
///     32 * 1024 * 1024 * 1024, // 32GB global physical limit
///     device,
/// )?;
///
/// // Register models
/// shared_pool.register_model("llama-7b", 64 * 1024 * 1024 * 1024)?; // 64GB virtual
/// shared_pool.register_model("gpt2", 32 * 1024 * 1024 * 1024)?;     // 32GB virtual
///
/// // Allocate for specific model
/// let addr = shared_pool.allocate_for_model("llama-7b", 1024 * 1024 * 1024)?;
/// # Ok(())
/// # }
/// ```
pub struct SharedMemoryPool {
    /// Per-model virtual memory pools.
    pools: HashMap<String, VirtualMemoryPool>,
    /// Global physical memory limit in bytes.
    global_physical_limit: usize,
    /// Current global physical usage in bytes.
    current_physical_usage: usize,
    /// Device ordinal.
    device_ordinal: i32,
    /// Default page size for new pools.
    default_page_size: usize,
}

impl SharedMemoryPool {
    /// Create shared pool with global physical memory limit.
    ///
    /// # Arguments
    /// * `physical_limit` - Global physical memory limit (bytes).
    /// * `device` - CUDA device.
    ///
    /// # Returns
    /// Shared memory pool.
    pub fn new(physical_limit: usize, device: Device) -> Result<Self> {
        let device_ordinal = get_device_ordinal(&device)?;

        // Get recommended page size
        let default_page_size = cuda_ffi::get_recommended_granularity(device_ordinal)?;

        Ok(Self {
            pools: HashMap::new(),
            global_physical_limit: physical_limit,
            current_physical_usage: 0,
            device_ordinal,
            default_page_size,
        })
    }

    /// Register a model with virtual address space reservation.
    ///
    /// # Arguments
    /// * `model_id` - Unique model identifier.
    /// * `virtual_capacity` - Virtual address space for this model (bytes).
    ///
    /// # Errors
    /// Returns error if model already registered.
    pub fn register_model(&mut self, model_id: &str, virtual_capacity: usize) -> Result<()> {
        if self.pools.contains_key(model_id) {
            return Err(VmmError::ModelAlreadyExists(model_id.to_string()));
        }

        let device = Device::new_cuda(self.device_ordinal as usize)?;
        let pool = VirtualMemoryPool::new(virtual_capacity, self.default_page_size, device)?;

        self.pools.insert(model_id.to_string(), pool);
        Ok(())
    }

    /// Allocate from specific model's pool.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier.
    /// * `size` - Size to allocate (bytes).
    ///
    /// # Returns
    /// Virtual address of allocated region.
    ///
    /// # Errors
    /// Returns error if:
    /// - Model not found
    /// - Global physical limit exceeded
    /// - Allocation fails
    pub fn allocate_for_model(&mut self, model_id: &str, size: usize) -> Result<usize> {
        let pool = self
            .pools
            .get_mut(model_id)
            .ok_or_else(|| VmmError::ModelNotFound(model_id.to_string()))?;

        // Check global physical limit
        let rounded_size =
            (size + self.default_page_size - 1) / self.default_page_size * self.default_page_size;
        if self.current_physical_usage + rounded_size > self.global_physical_limit {
            return Err(VmmError::OutOfPhysicalMemory {
                requested: rounded_size,
                available: self.global_physical_limit - self.current_physical_usage,
            });
        }

        // Allocate from model's pool
        let addr = pool.allocate(0, size)?;
        self.current_physical_usage += rounded_size;

        Ok(addr)
    }

    /// Free from specific model's pool.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier.
    /// * `offset` - Offset in model's virtual address space (bytes).
    /// * `size` - Size to free (bytes).
    pub fn deallocate_for_model(
        &mut self,
        model_id: &str,
        offset: usize,
        size: usize,
    ) -> Result<()> {
        let pool = self
            .pools
            .get_mut(model_id)
            .ok_or_else(|| VmmError::ModelNotFound(model_id.to_string()))?;

        let rounded_size =
            (size + self.default_page_size - 1) / self.default_page_size * self.default_page_size;

        pool.deallocate(offset, size)?;
        self.current_physical_usage = self.current_physical_usage.saturating_sub(rounded_size);

        Ok(())
    }

    /// Get per-model memory statistics.
    pub fn get_model_stats(&self, model_id: &str) -> Option<MemoryStats> {
        self.pools.get(model_id).map(|pool| pool.stats())
    }

    /// Global memory statistics.
    pub fn global_stats(&self) -> GlobalMemoryStats {
        GlobalMemoryStats {
            physical_limit: self.global_physical_limit,
            physical_usage: self.current_physical_usage,
            num_models: self.pools.len(),
        }
    }

    /// Unregister a model and free its resources.
    pub fn unregister_model(&mut self, model_id: &str) -> Result<()> {
        if let Some(pool) = self.pools.remove(model_id) {
            let usage = pool.physical_memory_usage();
            self.current_physical_usage = self.current_physical_usage.saturating_sub(usage);
            Ok(())
        } else {
            Err(VmmError::ModelNotFound(model_id.to_string()))
        }
    }
}

/// Global memory statistics for shared pool.
#[derive(Debug, Clone)]
pub struct GlobalMemoryStats {
    /// Global physical memory limit in bytes.
    pub physical_limit: usize,
    /// Current global physical usage in bytes.
    pub physical_usage: usize,
    /// Number of registered models.
    pub num_models: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            virtual_capacity: 1024 * 1024,
            physical_usage: 512 * 1024,
            mapped_pages: 256,
            fragmentation_ratio: 0.5,
        };

        assert_eq!(stats.virtual_capacity, 1024 * 1024);
        assert_eq!(stats.physical_usage, 512 * 1024);
    }

    #[test]
    fn test_global_memory_stats() {
        let stats = GlobalMemoryStats {
            physical_limit: 32 * 1024 * 1024 * 1024,
            physical_usage: 16 * 1024 * 1024 * 1024,
            num_models: 3,
        };

        assert_eq!(stats.num_models, 3);
    }
}
