//! Thin re-export layer for Metal backend types, plus the bridge that turns
//! a [`fuel_metal_backend::MetalDevice`] into a [`crate::Device`].
//!
//! After step B1 of the backend extraction, all Metal logic lives in
//! `fuel-metal-backend`; this module owns only the `From<MetalDevice> for Device`
//! impl (orphan rule keeps it on this side) and a handful of free functions
//! for the cases that need fuel-core types in their signature (Device-or-CPU
//! fallback, `Device` downcast).
//!
//! The free constructor `new_device` is always defined: when
//! `--features metal` is off it returns
//! [`Error::NotCompiledWithMetalSupport`], matching the prior
//! `Device::new_metal` ergonomics.

use crate::{Device, Error, Result};

#[cfg(feature = "metal")]
pub use fuel_metal_backend::*;

#[cfg(feature = "metal")]
impl From<fuel_metal_backend::MetalDevice> for Device {
    fn from(dev: fuel_metal_backend::MetalDevice) -> Self {
        Device::custom(std::sync::Arc::new(dev))
    }
}

/// Creates a new Metal device with the given ordinal.
#[cfg(feature = "metal")]
pub fn new_device(ordinal: usize) -> Result<Device> {
    Ok(fuel_metal_backend::MetalDevice::new(ordinal)?.into())
}

#[cfg(not(feature = "metal"))]
pub fn new_device(_ordinal: usize) -> Result<Device> {
    Err(Error::NotCompiledWithMetalSupport.bt())
}

/// Returns a Metal device if available, otherwise falls back to CPU.
pub fn device_if_available(ordinal: usize) -> Result<Device> {
    if crate::utils::metal_is_available() {
        new_device(ordinal)
    } else {
        Ok(Device::cpu())
    }
}

/// Returns the underlying [`MetalDevice`] handle, or an error if `device` is
/// not a Metal device.
#[cfg(feature = "metal")]
pub fn as_device(device: &Device) -> Result<&fuel_metal_backend::MetalDevice> {
    device
        .inner
        .as_any()
        .downcast_ref::<fuel_metal_backend::MetalDevice>()
        .ok_or_else(|| Error::Msg("expected a metal device".into()).bt())
}
