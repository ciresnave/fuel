//! Physical GPU memory allocation with RAII.

use crate::cuda_ffi::{self, AllocationProp, MemGenericAllocationHandle};
use crate::error::{Result, VmmError};
use candle::{Device, DeviceLocation};

/// Helper function to extract device ordinal from Candle Device
fn get_device_ordinal(device: &Device) -> Result<i32> {
    match device.location() {
        DeviceLocation::Cuda { gpu_id } => Ok(gpu_id as i32),
        _ => Err(VmmError::other("Device must be a CUDA device")),
    }
}

/// Handle to physical GPU memory allocation.
///
/// This type manages physical GPU memory using CUDA's Virtual Memory Management.
/// Memory is automatically released when the handle is dropped (RAII pattern).
///
/// # Example
/// ```no_run
/// use candle_cuda_vmm::PhysicalMemoryHandle;
/// use candle::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::new_cuda(0)?;
/// let handle = PhysicalMemoryHandle::new(2 * 1024 * 1024, &device)?; // 2MB
/// println!("Allocated {} bytes", handle.size());
/// // Memory automatically released when handle goes out of scope
/// # Ok(())
/// # }
/// ```
pub struct PhysicalMemoryHandle {
    /// Raw CUDA memory handle.
    handle: MemGenericAllocationHandle,
    /// Size of allocation in bytes.
    size: usize,
    /// Device this memory belongs to.
    device_ordinal: i32,
}

impl PhysicalMemoryHandle {
    /// Allocate physical GPU memory.
    ///
    /// # Arguments
    /// * `size` - Size in bytes. Must be a multiple of the device's allocation granularity.
    /// * `device` - Candle device to allocate on.
    ///
    /// # Returns
    /// Handle to physical memory allocation.
    ///
    /// # Errors
    /// Returns error if:
    /// - Device is not a CUDA device
    /// - Size is not aligned to granularity
    /// - Allocation fails (out of memory, etc.)
    pub fn new(size: usize, device: &Device) -> Result<Self> {
        let device_ordinal = get_device_ordinal(device)?;

        // Validate size alignment
        let granularity = cuda_ffi::get_recommended_granularity(device_ordinal)?;
        if size % granularity != 0 {
            return Err(VmmError::InvalidAlignment {
                actual: size,
                required: granularity,
            });
        }

        // Allocate physical memory
        let prop = AllocationProp::device(device_ordinal);
        let handle = unsafe { cuda_ffi::mem_create(size, &prop)? };

        Ok(Self {
            handle,
            size,
            device_ordinal,
        })
    }

    /// Create from raw handle (unsafe - caller must ensure handle is valid).
    ///
    /// # Safety
    /// - Handle must be a valid CUDA memory allocation handle
    /// - Size and device_ordinal must match the actual allocation
    /// - Caller must ensure handle is not used elsewhere (ownership transfer)
    pub unsafe fn from_raw(
        handle: MemGenericAllocationHandle,
        size: usize,
        device_ordinal: i32,
    ) -> Self {
        Self {
            handle,
            size,
            device_ordinal,
        }
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get device ordinal.
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Get raw CUDA handle.
    ///
    /// # Safety
    /// Handle remains owned by this PhysicalMemoryHandle. Do not release it manually.
    pub fn as_raw(&self) -> MemGenericAllocationHandle {
        self.handle
    }

    /// Consume self and return raw handle (caller takes ownership).
    ///
    /// # Safety
    /// Caller must ensure handle is properly released via cuda_ffi::mem_release.
    pub fn into_raw(self) -> MemGenericAllocationHandle {
        let handle = self.handle;
        std::mem::forget(self); // Prevent Drop from running
        handle
    }
}

impl Drop for PhysicalMemoryHandle {
    fn drop(&mut self) {
        // Release physical memory
        // Note: Errors during drop are logged but cannot be propagated
        unsafe {
            if let Err(e) = cuda_ffi::mem_release(self.handle) {
                eprintln!("Warning: Failed to release physical memory: {}", e);
            }
        }
    }
}

// PhysicalMemoryHandle can be sent between threads (no shared mutable state)
unsafe impl Send for PhysicalMemoryHandle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_memory_handle_properties() {
        // Test basic properties without actual CUDA allocation
        let handle = unsafe { PhysicalMemoryHandle::from_raw(12345, 2 * 1024 * 1024, 0) };
        assert_eq!(handle.size(), 2 * 1024 * 1024);
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(handle.as_raw(), 12345);

        // Prevent actual CUDA cleanup
        std::mem::forget(handle);
    }
}
