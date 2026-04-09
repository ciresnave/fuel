//! Virtual address space reservation and memory mapping operations.

use crate::cuda_ffi::{self, AccessFlags, DevicePtr};
use crate::error::{Result, VmmError};
use crate::physical_memory::PhysicalMemoryHandle;

/// Virtual address space reservation.
///
/// Reserves a contiguous range of virtual addresses without allocating physical memory.
/// Memory is automatically freed when the range is dropped (RAII pattern).
///
/// # Example
/// ```no_run
/// use candle_cuda_vmm::VirtualAddressRange;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Reserve 1GB of virtual address space with 2MB alignment
/// let range = VirtualAddressRange::new(1024 * 1024 * 1024, 2 * 1024 * 1024)?;
/// println!("Reserved virtual addresses: 0x{:x}-0x{:x}",
///          range.base_address(), range.base_address() + range.size());
/// // Address space automatically freed when range goes out of scope
/// # Ok(())
/// # }
/// ```
pub struct VirtualAddressRange {
    /// Base virtual address.
    ptr: DevicePtr,
    /// Total size in bytes.
    size: usize,
    /// Alignment in bytes.
    alignment: usize,
}

impl VirtualAddressRange {
    /// Reserve contiguous virtual address space.
    ///
    /// # Arguments
    /// * `size` - Size in bytes.
    /// * `alignment` - Alignment in bytes (must be power of 2).
    ///
    /// # Returns
    /// Virtual address range reservation.
    ///
    /// # Errors
    /// Returns error if:
    /// - Alignment is not a power of 2
    /// - Virtual address space cannot be reserved
    pub fn new(size: usize, alignment: usize) -> Result<Self> {
        // Validate alignment (must be power of 2)
        if !alignment.is_power_of_two() {
            return Err(VmmError::InvalidAlignment {
                actual: alignment,
                required: 0, // any power of 2
            });
        }

        // Reserve virtual address space
        let ptr = unsafe { cuda_ffi::mem_address_reserve(size, alignment, 0)? };

        Ok(Self {
            ptr,
            size,
            alignment,
        })
    }

    /// Create from raw pointer (unsafe - caller must ensure pointer is valid).
    ///
    /// # Safety
    /// - Pointer must be a valid reserved virtual address
    /// - Size and alignment must match the actual reservation
    /// - Caller must ensure pointer is not used elsewhere (ownership transfer)
    pub unsafe fn from_raw(ptr: DevicePtr, size: usize, alignment: usize) -> Self {
        Self {
            ptr,
            size,
            alignment,
        }
    }

    /// Get base virtual address.
    pub fn base_address(&self) -> usize {
        self.ptr as usize
    }

    /// Get total size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get alignment in bytes.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get raw device pointer.
    pub fn as_ptr(&self) -> DevicePtr {
        self.ptr
    }

    /// Get pointer at offset.
    ///
    /// # Arguments
    /// * `offset` - Offset in bytes from base address.
    ///
    /// # Returns
    /// Device pointer at offset.
    ///
    /// # Errors
    /// Returns error if offset is out of bounds.
    pub fn ptr_at_offset(&self, offset: usize) -> Result<DevicePtr> {
        if offset >= self.size {
            return Err(VmmError::InvalidOffset {
                offset,
                size: 0,
                capacity: self.size,
            });
        }
        Ok(self.ptr + offset as u64)
    }

    /// Consume self and return raw pointer (caller takes ownership).
    ///
    /// # Safety
    /// Caller must ensure pointer is properly freed via cuda_ffi::mem_address_free.
    pub fn into_raw(self) -> (DevicePtr, usize) {
        let ptr = self.ptr;
        let size = self.size;
        std::mem::forget(self); // Prevent Drop from running
        (ptr, size)
    }
}

impl Drop for VirtualAddressRange {
    fn drop(&mut self) {
        // Free virtual address space
        // Note: Errors during drop are logged but cannot be propagated
        unsafe {
            if let Err(e) = cuda_ffi::mem_address_free(self.ptr, self.size) {
                eprintln!("Warning: Failed to free virtual address space: {}", e);
            }
        }
    }
}

// VirtualAddressRange can be sent between threads
unsafe impl Send for VirtualAddressRange {}

/// Map physical memory to virtual address range.
///
/// # Arguments
/// * `virtual_range` - Virtual address range to map into.
/// * `offset` - Offset in virtual address range (bytes).
/// * `physical_handle` - Physical memory to map.
/// * `physical_offset` - Offset in physical memory (bytes).
/// * `size` - Size to map (bytes).
///
/// # Errors
/// Returns error if mapping fails or parameters are invalid.
pub fn map_memory(
    virtual_range: &VirtualAddressRange,
    offset: usize,
    physical_handle: &PhysicalMemoryHandle,
    physical_offset: usize,
    size: usize,
) -> Result<()> {
    // Validate parameters
    if offset + size > virtual_range.size() {
        return Err(VmmError::InvalidOffset {
            offset,
            size,
            capacity: virtual_range.size(),
        });
    }

    if physical_offset + size > physical_handle.size() {
        return Err(VmmError::InvalidOffset {
            offset: physical_offset,
            size,
            capacity: physical_handle.size(),
        });
    }

    // Get virtual address pointer
    let ptr = virtual_range.ptr_at_offset(offset)?;

    // Map physical memory to virtual address
    unsafe {
        cuda_ffi::mem_map(ptr, size, physical_offset, physical_handle.as_raw())?;
    }

    Ok(())
}

/// Unmap memory from virtual address range.
///
/// # Arguments
/// * `virtual_range` - Virtual address range to unmap from.
/// * `offset` - Offset in virtual address range (bytes).
/// * `size` - Size to unmap (bytes).
///
/// # Errors
/// Returns error if unmapping fails or parameters are invalid.
pub fn unmap_memory(virtual_range: &VirtualAddressRange, offset: usize, size: usize) -> Result<()> {
    // Validate parameters
    if offset + size > virtual_range.size() {
        return Err(VmmError::InvalidOffset {
            offset,
            size,
            capacity: virtual_range.size(),
        });
    }

    // Get virtual address pointer
    let ptr = virtual_range.ptr_at_offset(offset)?;

    // Unmap memory
    unsafe {
        cuda_ffi::mem_unmap(ptr, size)?;
    }

    Ok(())
}

/// Set memory access permissions.
///
/// # Arguments
/// * `virtual_range` - Virtual address range to set access for.
/// * `offset` - Offset in virtual address range (bytes).
/// * `size` - Size to set access for (bytes).
/// * `device_ordinal` - Device to set access for.
/// * `flags` - Access permissions.
///
/// # Errors
/// Returns error if setting access fails or parameters are invalid.
pub fn set_memory_access(
    virtual_range: &VirtualAddressRange,
    offset: usize,
    size: usize,
    device_ordinal: i32,
    flags: AccessFlags,
) -> Result<()> {
    // Validate parameters
    if offset + size > virtual_range.size() {
        return Err(VmmError::InvalidOffset {
            offset,
            size,
            capacity: virtual_range.size(),
        });
    }

    // Get virtual address pointer
    let ptr = virtual_range.ptr_at_offset(offset)?;

    // Set access permissions
    unsafe {
        cuda_ffi::mem_set_access(ptr, size, device_ordinal, flags)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_address_range_properties() {
        // Test basic properties without actual CUDA operations
        let range = unsafe { VirtualAddressRange::from_raw(0x1000000, 1024 * 1024, 4096) };
        assert_eq!(range.base_address(), 0x1000000);
        assert_eq!(range.size(), 1024 * 1024);
        assert_eq!(range.alignment(), 4096);

        // Prevent actual CUDA cleanup
        std::mem::forget(range);
    }

    #[test]
    fn test_ptr_at_offset() {
        let range = unsafe { VirtualAddressRange::from_raw(0x1000000, 1024 * 1024, 4096) };

        let ptr = range.ptr_at_offset(4096).unwrap();
        assert_eq!(ptr, 0x1000000 + 4096);

        // Out of bounds should error
        assert!(range.ptr_at_offset(2 * 1024 * 1024).is_err());

        // Prevent actual CUDA cleanup
        std::mem::forget(range);
    }

    #[test]
    fn test_invalid_alignment() {
        // Non-power-of-2 alignment should fail (but we can't test actual CUDA call)
        // This would fail: VirtualAddressRange::new(1024, 3)
    }
}
