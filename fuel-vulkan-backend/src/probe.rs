//! Device enumeration for the Vulkan backend.
//!
//! Creates a throwaway `Instance`, walks `enumerate_physical_devices`,
//! and produces one [`DeviceDescriptor`] per physical device. Each
//! descriptor carries vendor_id, device_id, device_name, driver_version,
//! and the sum of `DEVICE_LOCAL` heap sizes as `total_memory_bytes`.
//!
//! Returns `Ok(vec![])` (not an error) when the Vulkan loader is
//! present but no physical devices are visible. Returns `Err` only
//! when the loader itself cannot be created (missing runtime, no
//! compatible driver).

use fuel_ir::probe::{BackendId, BackendProbe, DeviceDescriptor};
use fuel_ir::{DeviceLocation, Error, Result};
use vulkane::safe::*;

pub struct VulkanBackendProbe;

impl BackendProbe for VulkanBackendProbe {
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
        enumerate_devices()
    }
}

fn vk_err(e: impl std::fmt::Debug) -> Error {
    Error::Msg(format!("vulkan probe: {e:?}"))
}

/// Enumerate every Vulkan physical device currently visible to the
/// loader. Cheap — creates an `Instance` but never a logical
/// `Device`, queue, or any allocations on GPU memory.
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    let instance = Instance::new(InstanceCreateInfo {
        engine_name: Some("fuel-vulkan-backend probe"),
        api_version: ApiVersion::V1_2,
        ..Default::default()
    }).map_err(vk_err)?;
    let physicals = instance.enumerate_physical_devices().map_err(vk_err)?;

    Ok(physicals.iter().enumerate().map(|(idx, p)| {
        let props = p.properties();
        let name = props.device_name();
        let vendor_id = props.vendor_id();
        let device_id = props.device_id();
        let driver_version = props.driver_version();
        // Vulkan's raw `driver_version` is a bare u32 whose bit-packing is
        // VENDOR-DEFINED — NVIDIA packs (22,14,6,10), AMD (22,10,10,10),
        // Intel-on-Windows (18,14) — so it cannot be decoded portably and is
        // good only for equality. Worse, it does not distinguish ICDs: RADV
        // and AMDVLK drive the same silicon, make different codegen choices,
        // and therefore profile differently, but can present the same u32.
        //
        // vulkane 0.9.0's `driver_properties()` (VK_KHR_driver_properties,
        // Vulkan 1.2 core) gives a portable, legible identity instead, with
        // `driver_id` naming the ICD exactly. That is the right axis for both
        // shader-cache keys and driver-quirk gating.
        //
        // CACHE NOTE: `driver_version` is a field of `EquivalenceKey`, so
        // changing its FORMAT invalidates every cached Judge profile exactly
        // once. That is intended and is the same "cheap insurance" the key's
        // own docs describe for a driver upgrade — the new key is strictly
        // more discriminating. Falls back to the raw hex when the device
        // declines (pre-1.2 effective API and no VK_KHR_driver_properties),
        // so the descriptor is still total.
        let driver_version_str = match p.driver_properties() {
            Some(d) => format!("{:?} {} {}", d.driver_id, d.driver_name, d.driver_info),
            None => format!("0x{driver_version:08x}"),
        };
        // Subgroup ("wave"/"warp") width — the single most important Vulkan
        // kernel-specialization axis. `None` on a pre-1.1 effective API,
        // where the property struct would read back zeroed; vulkane declines
        // honestly rather than reporting 0 as an answer.
        //
        // NOTE: `SubgroupProperties::size_control` (the pinnable min/max
        // range) is Vulkan 1.3 core / VK_EXT_subgroup_size_control, and our
        // instances are created at V1_2 — so it is always `None` here today.
        // Raising the instance to V1_3 would unlock it, but that changes
        // instance-creation compatibility and belongs in its own change.
        let subgroup_width = p.subgroup_properties().map(|s| s.subgroup_size);
        let total_mem = total_device_local_memory(p);

        DeviceDescriptor {
            backend:            BackendId::Vulkan,
            device_index:       idx as u32,
            hardware_sku:       name,
            vendor_id,
            device_id,
            compute_capability: None,
            subgroup_width,
            driver_version:     driver_version_str,
            total_memory_bytes: total_mem,
            location:           DeviceLocation::Vulkan { gpu_id: idx },
        }
    }).collect())
}

fn total_device_local_memory(p: &PhysicalDevice) -> u64 {
    let mp = p.memory_properties();
    (0..mp.heap_count())
        .map(|i| mp.memory_heap(i))
        .filter(|h| h.flags().contains(MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a box with no Vulkan runtime at all, `Instance::new` fails
    /// and `enumerate_devices` returns `Err`. On a normal dev box it
    /// returns `Ok(vec)` with at least one entry (software or
    /// hardware rasterizer). In either case, every entry we do
    /// return must key-match its own ordinal and carry non-zero
    /// vendor/device ids (vulkane-side guarantees).
    #[test]
    fn vulkan_probe_is_total() {
        let devices = match enumerate_devices() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("vulkan probe skipped: {e}");
                return;
            }
        };
        for d in &devices {
            assert_eq!(d.backend, BackendId::Vulkan);
            match d.location {
                DeviceLocation::Vulkan { gpu_id } => {
                    assert_eq!(gpu_id, d.device_index as usize);
                }
                other => panic!("expected Vulkan location, got {other:?}"),
            }
        }
    }

    /// vulkane 0.9.0 adoption. `subgroup_properties()` declines honestly
    /// (`None`) rather than reporting a zeroed struct, so the ONLY invalid
    /// state is `Some(0)` — which is exactly what a naive pNext read would
    /// produce on an implementation that ignored the struct. Vulkan
    /// guarantees a subgroup size that is a power of two in `1..=128`, so
    /// anything else means we misread the property.
    #[test]
    fn subgroup_width_is_never_a_zeroed_read() {
        let devices = match enumerate_devices() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("vulkan probe skipped: {e}");
                return;
            }
        };
        // Report what was actually seen. A device list that is EMPTY passes
        // every loop below vacuously, which is indistinguishable from a real
        // pass in the summary line — so print the count and let `--nocapture`
        // show whether this test had anything to assert on.
        eprintln!("vulkan probe: {} device(s) enumerated", devices.len());
        for d in &devices {
            eprintln!(
                "  [{}] {} | subgroup_width={:?} | driver={:?}",
                d.device_index, d.hardware_sku, d.subgroup_width, d.driver_version,
            );
        }
        for d in &devices {
            if let Some(w) = d.subgroup_width {
                assert!(
                    w > 0 && w <= 128 && w.is_power_of_two(),
                    "{}: subgroup_width {w} is not a power of two in 1..=128 — \
                     a zeroed pNext read, not a real answer",
                    d.hardware_sku,
                );
            }
        }
    }

    /// The driver identity is an `EquivalenceKey` field, so an empty or
    /// whitespace-only string would silently collapse distinct drivers into
    /// one Judge profile class. Whether it came from `driver_properties()`
    /// or the raw-hex fallback, it must be non-empty.
    #[test]
    fn driver_version_key_is_never_empty() {
        let devices = match enumerate_devices() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("vulkan probe skipped: {e}");
                return;
            }
        };
        for d in &devices {
            assert!(
                !d.driver_version.trim().is_empty(),
                "{}: empty driver_version would collapse EquivalenceKey classes",
                d.hardware_sku,
            );
        }
    }
}
