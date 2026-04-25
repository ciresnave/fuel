//! Device enumeration for the default CPU backend.
//!
//! Reports one descriptor per physical CPU in scope. Today that's
//! always exactly one — NUMA-aware splits (two Xeon sockets → two CPU
//! descriptors) are Phase 7b territory. Hardware SKU detection uses
//! `std::env::consts::ARCH` and `std::thread::available_parallelism`
//! as placeholders; if the dispatch signal ever needs genuine CPU
//! identification (AVX-512 vs AVX2 dispatch, e.g.) that's the cue to
//! add a real cpuid crate.

use fuel_core_types::probe::{BackendId, BackendProbe, DeviceDescriptor};
use fuel_core_types::{DeviceLocation, error::Result};

pub struct CpuBackendProbe;

impl BackendProbe for CpuBackendProbe {
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
        enumerate_devices()
    }
}

/// Enumerate CPU "devices" — one per physical CPU. NUMA splits are a
/// future refinement; today this always returns exactly one entry.
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let sku = format!("{} ({} logical cores)", std::env::consts::ARCH, cores);
    Ok(vec![DeviceDescriptor {
        backend:            BackendId::Cpu,
        device_index:       0,
        hardware_sku:       sku,
        vendor_id:          0,
        device_id:          0,
        compute_capability: None,
        // No direct driver concept for the CPU backend; use the crate
        // version so the Judge invalidates on fuel-cpu-backend updates.
        driver_version:     env!("CARGO_PKG_VERSION").to_string(),
        total_memory_bytes: 0,
        location:           DeviceLocation::Cpu,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_enumerates_exactly_one_device() {
        let d = enumerate_devices().expect("cpu probe never fails");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].backend, BackendId::Cpu);
        assert_eq!(d[0].device_index, 0);
        assert_eq!(d[0].location, DeviceLocation::Cpu);
        // SKU should at least contain the arch.
        assert!(d[0].hardware_sku.contains(std::env::consts::ARCH));
    }
}
