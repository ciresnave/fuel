//! Thin re-export layer for CUDA backend types, plus the bridge that turns
//! a [`fuel_cuda_backend::CudaDevice`] into a [`crate::Device`].
//!
//! After step B1 of the backend extraction, all CUDA logic lives in
//! `fuel-cuda-backend`; this module owns only the `From<CudaDevice> for Device`
//! impl (orphan rule keeps it on this side) and a handful of free functions
//! for the cases that need fuel-core types in their signature (Device-or-CPU
//! fallback, `Device` downcast).
//!
//! The free constructors (`new_device`, `new_device_with_stream`) are
//! always defined: when `--features cuda` is off they return
//! [`Error::NotCompiledWithCudaSupport`], matching the prior
//! `Device::new_cuda` ergonomics.

use crate::{Device, Error, Result};

#[cfg(feature = "cuda")]
pub use fuel_cuda_backend::*;

#[cfg(feature = "cuda")]
impl From<fuel_cuda_backend::CudaDevice> for Device {
    fn from(dev: fuel_cuda_backend::CudaDevice) -> Self {
        Device::custom(std::sync::Arc::new(dev))
    }
}

/// Creates a new CUDA device with the given GPU ordinal.
#[cfg(feature = "cuda")]
pub fn new_device(ordinal: usize) -> Result<Device> {
    Ok(fuel_cuda_backend::CudaDevice::new(ordinal)?.into())
}

#[cfg(not(feature = "cuda"))]
pub fn new_device(_ordinal: usize) -> Result<Device> {
    Err(Error::NotCompiledWithCudaSupport.bt())
}

/// Creates a new CUDA device with a dedicated stream.
#[cfg(feature = "cuda")]
pub fn new_device_with_stream(ordinal: usize) -> Result<Device> {
    Ok(fuel_cuda_backend::CudaDevice::new_with_stream(ordinal)?.into())
}

#[cfg(not(feature = "cuda"))]
pub fn new_device_with_stream(_ordinal: usize) -> Result<Device> {
    Err(Error::NotCompiledWithCudaSupport.bt())
}

/// Returns a CUDA device if available, otherwise falls back to CPU.
pub fn device_if_available(ordinal: usize) -> Result<Device> {
    if crate::utils::cuda_is_available() {
        new_device(ordinal)
    } else {
        Ok(Device::cpu())
    }
}

/// Returns the underlying [`CudaDevice`] handle, or an error if `device` is
/// not a CUDA device.
#[cfg(feature = "cuda")]
pub fn as_device(device: &Device) -> Result<&fuel_cuda_backend::CudaDevice> {
    device
        .inner
        .as_any()
        .downcast_ref::<fuel_cuda_backend::CudaDevice>()
        .ok_or_else(|| Error::Msg("expected a cuda device".into()).bt())
}
