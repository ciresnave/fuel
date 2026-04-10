//! Low-level CUDA Virtual Memory Management FFI bindings.
//!
//! This module provides safe wrappers around CUDA's VMM APIs using cudarc.
//! All functions perform proper error checking and return Result types.

use crate::error::{Result, VmmError};
use cudarc::driver::sys;

/// Memory location type for CUDA allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocationType {
    /// Memory allocated on a specific device.
    Device,
}

/// Memory allocation properties.
#[derive(Debug, Clone)]
pub struct AllocationProp {
    /// Type of memory location.
    pub location_type: MemoryLocationType,
    /// Device ordinal for device memory.
    pub device_ordinal: i32,
}

impl AllocationProp {
    /// Create allocation properties for a specific device.
    pub fn device(device_ordinal: i32) -> Self {
        Self {
            location_type: MemoryLocationType::Device,
            device_ordinal,
        }
    }

    /// Convert to CUDA allocation properties.
    fn to_cuda_prop(&self) -> sys::CUmemAllocationProp {
        sys::CUmemAllocationProp {
            type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device_ordinal,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: unsafe { std::mem::zeroed() },
        }
    }
}

/// Memory access flags for setting access permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessFlags {
    /// No access.
    None = 0,
    /// Read-only access.
    Read = 1,
    /// Read and write access.
    ReadWrite = 3,
}

impl AccessFlags {
    /// Convert to CUDA access flags.
    fn to_cuda_flags(&self) -> sys::CUmemAccess_flags {
        match self {
            AccessFlags::None => sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE,
            AccessFlags::Read => sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
            AccessFlags::ReadWrite => sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        }
    }
}

/// Handle to physical GPU memory allocation.
pub type MemGenericAllocationHandle = sys::CUmemGenericAllocationHandle;

/// Device pointer (virtual address).
pub type DevicePtr = sys::CUdeviceptr;

/// Allocate physical GPU memory.
///
/// # Arguments
/// * `size` - Size in bytes (must be multiple of granularity).
/// * `prop` - Allocation properties (device, type, etc.).
///
/// # Returns
/// Handle to physical memory allocation.
pub unsafe fn mem_create(size: usize, prop: &AllocationProp) -> Result<MemGenericAllocationHandle> {
    let mut handle: MemGenericAllocationHandle = 0;
    let cuda_prop = prop.to_cuda_prop();

    let result = unsafe {
        sys::cuMemCreate(
            &mut handle,
            size,
            &cuda_prop,
            0, // flags
        )
    };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::cuda(format!(
            "cuMemCreate failed with code {:?}",
            result
        )));
    }

    Ok(handle)
}

/// Release physical GPU memory.
///
/// # Arguments
/// * `handle` - Handle to physical memory allocation.
pub unsafe fn mem_release(handle: MemGenericAllocationHandle) -> Result<()> {
    let result = unsafe { sys::cuMemRelease(handle) };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::cuda(format!(
            "cuMemRelease failed with code {:?}",
            result
        )));
    }

    Ok(())
}

/// Reserve virtual address space.
///
/// # Arguments
/// * `size` - Size in bytes.
/// * `alignment` - Alignment in bytes (must be power of 2).
/// * `addr` - Requested starting address (0 for any address).
///
/// # Returns
/// Base virtual address of reserved range.
pub unsafe fn mem_address_reserve(
    size: usize,
    alignment: usize,
    addr: DevicePtr,
) -> Result<DevicePtr> {
    let mut ptr: DevicePtr = 0;

    let result = unsafe {
        sys::cuMemAddressReserve(
            &mut ptr, size, alignment, addr, 0, // flags
        )
    };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::cuda(format!(
            "cuMemAddressReserve failed with code {:?}",
            result
        )));
    }

    Ok(ptr)
}

/// Free virtual address space.
///
/// # Arguments
/// * `ptr` - Base virtual address to free.
/// * `size` - Size in bytes.
pub unsafe fn mem_address_free(ptr: DevicePtr, size: usize) -> Result<()> {
    let result = unsafe { sys::cuMemAddressFree(ptr, size) };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::cuda(format!(
            "cuMemAddressFree failed with code {:?}",
            result
        )));
    }

    Ok(())
}

/// Map physical memory to virtual address range.
///
/// # Arguments
/// * `ptr` - Virtual address to map to.
/// * `size` - Size in bytes.
/// * `offset` - Offset into physical memory handle.
/// * `handle` - Physical memory handle.
pub unsafe fn mem_map(
    ptr: DevicePtr,
    size: usize,
    offset: usize,
    handle: MemGenericAllocationHandle,
) -> Result<()> {
    let result = unsafe {
        sys::cuMemMap(
            ptr, size, offset, handle, 0, // flags
        )
    };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::MappingFailed(format!(
            "cuMemMap failed with code {:?}",
            result
        )));
    }

    Ok(())
}

/// Unmap memory from virtual address range.
///
/// # Arguments
/// * `ptr` - Virtual address to unmap.
/// * `size` - Size in bytes.
pub unsafe fn mem_unmap(ptr: DevicePtr, size: usize) -> Result<()> {
    let result = unsafe { sys::cuMemUnmap(ptr, size) };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::UnmappingFailed(format!(
            "cuMemUnmap failed with code {:?}",
            result
        )));
    }

    Ok(())
}

/// Set memory access permissions.
///
/// # Arguments
/// * `ptr` - Virtual address.
/// * `size` - Size in bytes.
/// * `device_ordinal` - Device to set access for.
/// * `flags` - Access permissions.
pub unsafe fn mem_set_access(
    ptr: DevicePtr,
    size: usize,
    device_ordinal: i32,
    flags: AccessFlags,
) -> Result<()> {
    let access_desc = sys::CUmemAccessDesc {
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device_ordinal,
        },
        flags: flags.to_cuda_flags(),
    };

    let result = unsafe { sys::cuMemSetAccess(ptr, size, &access_desc, 1) };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::cuda(format!(
            "cuMemSetAccess failed with code {:?}",
            result
        )));
    }

    Ok(())
}

/// Get minimum allocation granularity for a device.
///
/// # Arguments
/// * `prop` - Allocation properties.
/// * `option` - Granularity option.
///
/// # Returns
/// Minimum granularity in bytes.
pub unsafe fn mem_get_allocation_granularity(
    prop: &AllocationProp,
    option: sys::CUmemAllocationGranularity_flags,
) -> Result<usize> {
    let mut granularity: usize = 0;
    let cuda_prop = prop.to_cuda_prop();

    let result =
        unsafe { sys::cuMemGetAllocationGranularity(&mut granularity, &cuda_prop, option) };

    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(VmmError::cuda(format!(
            "cuMemGetAllocationGranularity failed with code {:?}",
            result
        )));
    }

    Ok(granularity)
}

/// Get recommended granularity for a device.
pub fn get_recommended_granularity(device_ordinal: i32) -> Result<usize> {
    let prop = AllocationProp::device(device_ordinal);
    unsafe {
        mem_get_allocation_granularity(
            &prop,
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    }
}

/// Get minimum granularity for a device.
pub fn get_minimum_granularity(device_ordinal: i32) -> Result<usize> {
    let prop = AllocationProp::device(device_ordinal);
    unsafe {
        mem_get_allocation_granularity(
            &prop,
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_prop_creation() {
        let prop = AllocationProp::device(0);
        assert_eq!(prop.location_type, MemoryLocationType::Device);
        assert_eq!(prop.device_ordinal, 0);
    }

    #[test]
    fn test_access_flags() {
        assert_eq!(AccessFlags::None as i32, 0);
        assert_eq!(AccessFlags::Read as i32, 1);
        assert_eq!(AccessFlags::ReadWrite as i32, 3);
    }
}
