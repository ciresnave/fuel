//! Low-level CUDA Virtual Memory Management FFI bindings.
//!
//! Safe wrappers over CUDA's VMM APIs using baracuda-cuda-sys. The raw
//! driver symbols are resolved lazily at runtime by baracuda's dynamic
//! loader (`driver()`), so merely linking this crate doesn't pull in
//! `libcuda`. Each public fn performs `CUresult::SUCCESS` checking and
//! maps failures to [`VmmError`].

use crate::error::{Result, VmmError};
use baracuda_cuda_sys::{
    driver, types as sys, CUdeviceptr, CUmemAccessDesc, CUmemAllocationProp, CUmemLocation,
    CUresult,
};

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
    fn to_cuda_prop(&self) -> CUmemAllocationProp {
        CUmemAllocationProp {
            type_: sys::CUmemAllocationType::PINNED,
            requested_handle_types: sys::CUmemAllocationHandleType::NONE,
            location: CUmemLocation {
                type_: sys::CUmemLocationType::DEVICE,
                id: self.device_ordinal,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: Default::default(),
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
    fn to_cuda_flags(&self) -> i32 {
        match self {
            AccessFlags::None => sys::CUmemAccess_flags::NONE,
            AccessFlags::Read => sys::CUmemAccess_flags::READ,
            AccessFlags::ReadWrite => sys::CUmemAccess_flags::READWRITE,
        }
    }
}

/// Handle to physical GPU memory allocation.
pub type MemGenericAllocationHandle = sys::CUmemGenericAllocationHandle;

/// Device pointer (virtual address).
pub type DevicePtr = CUdeviceptr;

/// Granularity option (passed through to `cuMemGetAllocationGranularity`).
///
/// Mirrors `CUmemAllocationGranularity_flags` constants: MINIMUM=0,
/// RECOMMENDED=1.
pub type GranularityFlags = i32;

fn check(result: CUresult, what: &str) -> Result<()> {
    if result == CUresult::SUCCESS {
        Ok(())
    } else {
        Err(VmmError::cuda(format!("{what} failed: {result:?}")))
    }
}

fn load_err<E: core::fmt::Debug>(e: E, what: &str) -> VmmError {
    VmmError::cuda(format!("{what} loader error: {e:?}"))
}

/// Allocate physical GPU memory.
pub unsafe fn mem_create(size: usize, prop: &AllocationProp) -> Result<MemGenericAllocationHandle> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_create()
        .map_err(|e| load_err(e, "cuMemCreate"))?;
    let mut handle: MemGenericAllocationHandle = 0;
    let cuda_prop = prop.to_cuda_prop();
    check(unsafe { cu(&mut handle, size, &cuda_prop, 0) }, "cuMemCreate")?;
    Ok(handle)
}

/// Release physical GPU memory.
pub unsafe fn mem_release(handle: MemGenericAllocationHandle) -> Result<()> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_release()
        .map_err(|e| load_err(e, "cuMemRelease"))?;
    check(unsafe { cu(handle) }, "cuMemRelease")
}

/// Reserve virtual address space.
pub unsafe fn mem_address_reserve(
    size: usize,
    alignment: usize,
    addr: DevicePtr,
) -> Result<DevicePtr> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_address_reserve()
        .map_err(|e| load_err(e, "cuMemAddressReserve"))?;
    let mut ptr: CUdeviceptr = CUdeviceptr(0);
    check(
        unsafe { cu(&mut ptr, size, alignment, addr, 0) },
        "cuMemAddressReserve",
    )?;
    Ok(ptr)
}

/// Free virtual address space.
pub unsafe fn mem_address_free(ptr: DevicePtr, size: usize) -> Result<()> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_address_free()
        .map_err(|e| load_err(e, "cuMemAddressFree"))?;
    check(unsafe { cu(ptr, size) }, "cuMemAddressFree")
}

/// Map physical memory to virtual address range.
pub unsafe fn mem_map(
    ptr: DevicePtr,
    size: usize,
    offset: usize,
    handle: MemGenericAllocationHandle,
) -> Result<()> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d.cu_mem_map().map_err(|e| load_err(e, "cuMemMap"))?;
    let r = unsafe { cu(ptr, size, offset, handle, 0) };
    if r == CUresult::SUCCESS {
        Ok(())
    } else {
        Err(VmmError::MappingFailed(format!(
            "cuMemMap failed: {r:?}"
        )))
    }
}

/// Unmap memory from virtual address range.
pub unsafe fn mem_unmap(ptr: DevicePtr, size: usize) -> Result<()> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_unmap()
        .map_err(|e| load_err(e, "cuMemUnmap"))?;
    let r = unsafe { cu(ptr, size) };
    if r == CUresult::SUCCESS {
        Ok(())
    } else {
        Err(VmmError::UnmappingFailed(format!(
            "cuMemUnmap failed: {r:?}"
        )))
    }
}

/// Set memory access permissions.
pub unsafe fn mem_set_access(
    ptr: DevicePtr,
    size: usize,
    device_ordinal: i32,
    flags: AccessFlags,
) -> Result<()> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_set_access()
        .map_err(|e| load_err(e, "cuMemSetAccess"))?;
    let access_desc = CUmemAccessDesc {
        location: CUmemLocation {
            type_: sys::CUmemLocationType::DEVICE,
            id: device_ordinal,
        },
        flags: flags.to_cuda_flags(),
    };
    check(
        unsafe { cu(ptr, size, &access_desc, 1) },
        "cuMemSetAccess",
    )
}

/// Get minimum / recommended allocation granularity for a device.
pub unsafe fn mem_get_allocation_granularity(
    prop: &AllocationProp,
    option: GranularityFlags,
) -> Result<usize> {
    let d = driver().map_err(|e| load_err(e, "cuda driver"))?;
    let cu = d
        .cu_mem_get_allocation_granularity()
        .map_err(|e| load_err(e, "cuMemGetAllocationGranularity"))?;
    let mut granularity: usize = 0;
    let cuda_prop = prop.to_cuda_prop();
    check(
        unsafe { cu(&mut granularity, &cuda_prop, option) },
        "cuMemGetAllocationGranularity",
    )?;
    Ok(granularity)
}

/// Get recommended granularity for a device.
pub fn get_recommended_granularity(device_ordinal: i32) -> Result<usize> {
    let prop = AllocationProp::device(device_ordinal);
    unsafe {
        mem_get_allocation_granularity(
            &prop,
            sys::CUmemAllocationGranularity_flags::RECOMMENDED,
        )
    }
}

/// Get minimum granularity for a device.
pub fn get_minimum_granularity(device_ordinal: i32) -> Result<usize> {
    let prop = AllocationProp::device(device_ordinal);
    unsafe {
        mem_get_allocation_granularity(&prop, sys::CUmemAllocationGranularity_flags::MINIMUM)
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
