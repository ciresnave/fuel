//! Device enumeration for the CUDA backend.
//!
//! Walks `baracuda_driver::Device::count()`, queries each device's name
//! + compute capability + total memory, and produces one
//! [`DeviceDescriptor`] per ordinal. Vendor ID is hardcoded to NVIDIA's
//! 0x10DE since the CUDA runtime only loads on NVIDIA hardware; CUDA
//! doesn't expose PCI vendor IDs as a device attribute anyway.
//!
//! Returns `Ok(vec![])` — not an error — when the CUDA runtime loaded
//! but no devices are visible (headless-server case, VM without
//! passthrough). Returns `Err` only when the driver dynamic-load
//! itself fails.

use fuel_ir::probe::{BackendId, BackendProbe, DeviceDescriptor};
use fuel_ir::{DeviceLocation, Error, Result};

/// NVIDIA's PCI-SIG vendor ID. Hardcoded because (a) CUDA only ever
/// runs on NVIDIA silicon and (b) the CUDA device-attribute API does
/// not expose `PCI_VENDOR_ID` — only bus/device/domain.
pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;

pub struct CudaBackendProbe;

impl BackendProbe for CudaBackendProbe {
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
        enumerate_devices()
    }
}

/// Enumerate every CUDA device currently visible. Cheap — creates no
/// contexts or streams, allocates no device memory.
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    let count = baracuda_driver::Device::count()
        .map_err(|e| Error::Msg(format!("cuda probe: device count failed: {e}")).bt())?;
    let driver_ver = baracuda_driver::version()
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let mut out = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        let dev = baracuda_driver::Device::get(ordinal)
            .map_err(|e| Error::Msg(format!("cuda probe: Device::get({ordinal}) failed: {e}")).bt())?;
        let name = dev.name()
            .unwrap_or_else(|_| format!("cuda:{ordinal} (name query failed)"));
        let cc = dev.compute_capability().ok();
        let total_mem = dev.total_memory().unwrap_or(0);
        let pci_device_id = dev
            .attribute(baracuda_cuda_sys::types::CUdevice_attribute::PCI_DEVICE_ID as i32)
            .map(|v| v as u32)
            .unwrap_or(0);

        out.push(DeviceDescriptor {
            backend:            BackendId::Cuda,
            device_index:       ordinal,
            hardware_sku:       name,
            vendor_id:          NVIDIA_VENDOR_ID,
            device_id:          pci_device_id,
            compute_capability: cc,
            driver_version:     driver_ver.clone(),
            total_memory_bytes: total_mem,
            location:           DeviceLocation::Cuda { gpu_id: ordinal as usize },
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On machines without any CUDA device (headless CI, for example)
    /// the probe should return `Ok(vec![])`, not error. Where a GPU
    /// does exist, every descriptor should carry NVIDIA's vendor id
    /// and a CUDA `DeviceLocation` matching its ordinal.
    #[test]
    fn cuda_probe_is_total() {
        let devices = match enumerate_devices() {
            Ok(d) => d,
            Err(e) => {
                // Driver dynamic-load failed — only acceptable on
                // hosts without the CUDA runtime at all. Log and
                // bail cleanly.
                eprintln!("cuda probe skipped: {e}");
                return;
            }
        };
        for d in &devices {
            assert_eq!(d.backend, BackendId::Cuda);
            assert_eq!(d.vendor_id, NVIDIA_VENDOR_ID);
            match d.location {
                DeviceLocation::Cuda { gpu_id } => {
                    assert_eq!(gpu_id, d.device_index as usize);
                }
                other => panic!("expected Cuda location, got {other:?}"),
            }
        }
    }
}
