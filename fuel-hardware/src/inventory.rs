// SPDX-License-Identifier: MIT OR Apache-2.0
//! One device inventory with free memory (ROADMAP row 131), for a placement layer that must not
//! probe the OS itself (Lightbulb, lightbulb#276).
//!
//! [`ProbeReport::probe_all`](crate::probe::ProbeReport::probe_all) already lists every device of
//! every compiled-in backend with its total memory. What it cannot say is how much is FREE: that
//! needs a live handle (CUDA `cuMemGetInfo`, the Vulkan `VK_EXT_memory_budget` budget minus usage,
//! host `MemAvailable` or `ullAvailPhys`). [`inventory`] pairs the two in one call, keyed by
//! `(backend, device_index)`.
//!
//! A separate [`DeviceMemory`] record, not a field on `DeviceDescriptor`: that descriptor is
//! constructed in many crates and is a cache key's source, and free memory changes by the second.
//!
//! Cost: the live query opens a handle per device (a CUDA context, a Vulkan instance and logical
//! device), so call this when placing, not per token. `free_bytes` is `None` when the backend
//! cannot report it (Vulkan without `VK_EXT_memory_budget`), never a guess.

use fuel_ir::probe::{BackendId, DeviceDescriptor, DeviceIdentity};

use crate::probe::ProbeReport;

/// Memory state of one `(backend, device_index)` pair at the moment of the call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemory {
    pub backend: BackendId,
    /// The ordinal the backend's own API uses (the same one `DeviceDescriptor::device_index` holds).
    pub device_index: u32,
    /// Driver-reported device name.
    pub name: String,
    /// What names this physical GPU across processes and APIs (UUID, PCI address); `None` when the
    /// backend could not report one (the CPU, or a device below Vulkan 1.1). `device_index` is only
    /// a within-process handle: identify a device by this, then find its index.
    pub identity: Option<DeviceIdentity>,
    /// Total device-visible bytes from the probe; 0 when the backend did not report one.
    pub total_bytes: u64,
    /// Bytes free right now; `None` when it could not be measured. Never above `total_bytes` when
    /// the total is known.
    pub free_bytes: Option<u64>,
}

/// Build the inventory from already-enumerated descriptors, a free-memory lookup and an identity
/// lookup. Each lookup is called once per descriptor, in order, and gets that descriptor (so two
/// devices of one backend are asked about separately). Pure: the synthetic-enumeration tests drive
/// this.
pub fn inventory_from(
    devices: &[DeviceDescriptor],
    free_of: impl Fn(&DeviceDescriptor) -> Option<u64>,
    identity_of: impl Fn(&DeviceDescriptor) -> Option<DeviceIdentity>,
) -> Vec<DeviceMemory> {
    devices
        .iter()
        .map(|d| {
            let total = d.total_memory_bytes;
            let free = free_of(d).map(|f| if total > 0 { f.min(total) } else { f });
            DeviceMemory {
                backend: d.backend,
                device_index: d.device_index,
                name: d.hardware_sku.clone(),
                identity: identity_of(d),
                total_bytes: total,
                free_bytes: free,
            }
        })
        .collect()
}

/// Every device of every compiled-in backend with its live free memory. Enumeration errors skip
/// that backend, exactly as [`ProbeReport::probe_all`] does.
pub fn inventory() -> Vec<DeviceMemory> {
    let mut devices = ProbeReport::probe_all().devices;
    // The CPU descriptor reports no total (0); the host query knows it.
    for d in devices
        .iter_mut()
        .filter(|d| d.backend == BackendId::Cpu && d.total_memory_bytes == 0)
    {
        d.total_memory_bytes = fuel_cpu_backend::system_memory::total_bytes().unwrap_or(0);
    }
    #[cfg(feature = "cuda")]
    let cuda_ids = fuel_cuda_backend::probe::device_identities().unwrap_or_default();
    #[cfg(feature = "vulkan")]
    let vulkan_ids = fuel_vulkan_backend::probe::device_identities().unwrap_or_default();
    inventory_from(&devices, live_free_bytes, |d| match d.backend {
        #[cfg(feature = "cuda")]
        BackendId::Cuda => cuda_ids.get(d.device_index as usize).copied().flatten(),
        #[cfg(feature = "vulkan")]
        BackendId::Vulkan => vulkan_ids.get(d.device_index as usize).copied().flatten(),
        _ => None,
    })
}

/// Free bytes of one device from its backend's live query; `None` when unavailable.
fn live_free_bytes(d: &DeviceDescriptor) -> Option<u64> {
    match d.backend {
        BackendId::Cpu => fuel_cpu_backend::system_memory::available_bytes(),
        #[cfg(feature = "cuda")]
        BackendId::Cuda => fuel_cuda_backend::CudaDevice::new(d.device_index as usize)
            .ok()?
            .vram_info()
            .ok()
            .map(|(free, _total)| free),
        #[cfg(feature = "vulkan")]
        BackendId::Vulkan => {
            use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
            let b = VulkanBackend::with_selection(DeviceSelection::Index(d.device_index as usize))
                .ok()?;
            b.has_memory_budget_support()
                .then(|| b.vram_budget().saturating_sub(b.vram_used()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DeviceLocation;

    fn desc(backend: BackendId, index: u32, name: &str, total: u64) -> DeviceDescriptor {
        DeviceDescriptor {
            backend,
            device_index: index,
            hardware_sku: name.to_string(),
            vendor_id: 0,
            device_id: 0,
            compute_capability: None,
            subgroup_width: None,
            driver_version: String::new(),
            total_memory_bytes: total,
            location: match backend {
                BackendId::Vulkan => DeviceLocation::Vulkan {
                    gpu_id: index as usize,
                },
                _ => DeviceLocation::Cpu,
            },
        }
    }

    /// The shape: one record per descriptor, in order, carrying backend, index, name and total, with
    /// each device asked about separately (two Vulkan GPUs get their own free value).
    #[test]
    fn one_record_per_device_and_each_is_asked_separately() {
        let devices = [
            desc(BackendId::Cpu, 0, "cpu", 64),
            desc(BackendId::Vulkan, 0, "gpu a", 8_000),
            desc(BackendId::Vulkan, 1, "gpu b", 4_000),
        ];
        let inv = inventory_from(
            &devices,
            |d| Some(100 + u64::from(d.device_index) * 10),
            |_| None,
        );
        assert_eq!(inv.len(), 3);
        assert_eq!(
            (inv[0].backend, inv[0].device_index, inv[0].name.as_str()),
            (BackendId::Cpu, 0, "cpu")
        );
        assert_eq!((inv[1].total_bytes, inv[1].free_bytes), (8_000, Some(100)));
        assert_eq!((inv[2].total_bytes, inv[2].free_bytes), (4_000, Some(110)));
        assert_eq!(inv[2].device_index, 1, "the second GPU keeps its own index");
    }

    /// Each device gets ITS OWN identity (looked up per descriptor, not shared or swapped), and a
    /// device the backend could not identify stays `None`.
    #[test]
    fn identity_is_looked_up_per_device() {
        let devices = [
            desc(BackendId::Vulkan, 0, "gpu a", 8_000),
            desc(BackendId::Vulkan, 1, "gpu b", 4_000),
            desc(BackendId::Cpu, 0, "cpu", 64),
        ];
        let id = |b: u8| DeviceIdentity {
            uuid: Some([b; 16]),
            pci: None,
        };
        let inv = inventory_from(
            &devices,
            |_| None,
            |d| (d.backend == BackendId::Vulkan).then(|| id(d.device_index as u8 + 1)),
        );
        assert_eq!(inv[0].identity, Some(id(1)));
        assert_eq!(inv[1].identity, Some(id(2)));
        assert!(
            !inv[0]
                .identity
                .unwrap()
                .same_physical_device(&inv[1].identity.unwrap())
        );
        assert_eq!(inv[2].identity, None);
    }

    /// An unmeasurable device reports `None`, never a guess, and does not disturb its neighbours.
    #[test]
    fn unmeasurable_free_memory_is_none_not_zero() {
        let devices = [
            desc(BackendId::Vulkan, 0, "no budget ext", 8_000),
            desc(BackendId::Vulkan, 1, "has budget", 4_000),
        ];
        let inv = inventory_from(
            &devices,
            |d| (d.device_index == 1).then_some(1_234),
            |_| None,
        );
        assert_eq!(inv[0].free_bytes, None);
        assert_eq!(inv[1].free_bytes, Some(1_234));
    }

    /// Free is clamped to a known total (a budget can exceed the reported heap), and left alone when
    /// the total is unknown (0).
    #[test]
    fn free_is_clamped_to_a_known_total_only() {
        let devices = [
            desc(BackendId::Vulkan, 0, "known", 1_000),
            desc(BackendId::Cpu, 0, "unknown total", 0),
        ];
        let inv = inventory_from(&devices, |_| Some(5_000), |_| None);
        assert_eq!(inv[0].free_bytes, Some(1_000));
        assert_eq!(inv[1].free_bytes, Some(5_000));
    }

    /// The live call returns at least the CPU. Where the host query is implemented (Linux, Windows)
    /// free memory is measured and no larger than the total; elsewhere (macOS today) both are
    /// honestly absent, so this goes red the day the query lands there and the cfg below should widen.
    #[test]
    fn live_inventory_includes_the_host_with_measured_free_memory() {
        let inv = inventory();
        for m in &inv {
            eprintln!("inventory: {m:?}");
        }
        // A usable UUID names one GPU: no two devices of one backend may share one (the all-zero
        // trap would make every device "the same card"). Vacuous without a GPU backend.
        for backend in [BackendId::Cuda, BackendId::Vulkan] {
            let uuids: Vec<[u8; 16]> = inv
                .iter()
                .filter(|m| m.backend == backend)
                .filter_map(|m| m.identity.and_then(|i| i.usable_uuid()))
                .collect();
            let mut unique = uuids.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                unique.len(),
                uuids.len(),
                "{backend:?} devices share a UUID: {uuids:?}"
            );
        }
        // Cross-API identity (built with both GPU backends only): every CUDA device is the same
        // physical GPU as exactly one Vulkan device, found by identity, not by index.
        #[cfg(all(feature = "cuda", feature = "vulkan"))]
        for c in inv.iter().filter(|m| m.backend == BackendId::Cuda) {
            let cid = c.identity.expect("a CUDA device reports its UUID");
            let matches: Vec<_> = inv
                .iter()
                .filter(|m| m.backend == BackendId::Vulkan)
                .filter(|m| m.identity.is_some_and(|v| v.same_physical_device(&cid)))
                .collect();
            assert_eq!(matches.len(), 1, "CUDA {} matches {matches:?}", c.name);
        }
        let cpu = inv
            .iter()
            .find(|m| m.backend == BackendId::Cpu)
            .expect("the CPU is always enumerated");
        let implemented = cfg!(any(target_os = "linux", target_os = "windows"));
        assert_eq!(cpu.free_bytes.is_some(), implemented, "{cpu:?}");
        if implemented {
            assert!(cpu.free_bytes.unwrap() > 0);
            assert!(
                cpu.total_bytes > 0,
                "the host total is filled from the host query"
            );
            assert!(cpu.free_bytes.unwrap() <= cpu.total_bytes);
        }
    }
}
