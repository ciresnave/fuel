//! Device enumeration — the hardware-discovery half of what used to be
//! `fuel-core`'s `BackendFactory`. Each compiled-in backend contributes a
//! [`HardwareEnumerator`] that delegates to the backend crate's free
//! `probe::enumerate_devices`. [`crate::probe::ProbeReport::probe_all`] walks
//! [`registry`] to assemble a report.
//!
//! This is the discovery concern ONLY. The realize-seam half
//! (`try_make_realizer` / `LazyRealizer`) stays in `fuel-core::factories`,
//! because it constructs a `LazyTensor` realizer and must live above the IR.

use fuel_ir::probe::{BackendId, DeviceDescriptor};
use fuel_ir::Result;

/// One backend's device-enumeration capability. Implementors are zero-sized
/// tags; enumeration delegates to the backend crate's `probe::enumerate_devices`.
pub trait HardwareEnumerator: Send + Sync {
    /// Stable identifier — the same one the probe + dispatch use.
    fn id(&self) -> BackendId;
    /// Devices this backend currently sees on the host.
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>>;
}

/// Every enumerator compiled into this build — CPU always present, the rest
/// cfg-gated — in the same order the probe lists them.
pub fn registry() -> Vec<&'static dyn HardwareEnumerator> {
    #[allow(unused_mut)]
    let mut v: Vec<&'static dyn HardwareEnumerator> = vec![&CpuEnumerator];
    #[cfg(feature = "cuda")]
    v.push(&CudaEnumerator);
    #[cfg(feature = "vulkan")]
    v.push(&VulkanEnumerator);
    v
}

pub struct CpuEnumerator;
impl HardwareEnumerator for CpuEnumerator {
    fn id(&self) -> BackendId {
        BackendId::Cpu
    }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_cpu_backend::probe::enumerate_devices()
    }
}

#[cfg(feature = "cuda")]
pub struct CudaEnumerator;
#[cfg(feature = "cuda")]
impl HardwareEnumerator for CudaEnumerator {
    fn id(&self) -> BackendId {
        BackendId::Cuda
    }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_cuda_backend::probe::enumerate_devices()
    }
}

#[cfg(feature = "vulkan")]
pub struct VulkanEnumerator;
#[cfg(feature = "vulkan")]
impl HardwareEnumerator for VulkanEnumerator {
    fn id(&self) -> BackendId {
        BackendId::Vulkan
    }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        fuel_vulkan_backend::probe::enumerate_devices()
    }
}
