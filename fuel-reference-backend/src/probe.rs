//! Device enumeration for the reference backend.
//!
//! The reference backend is always available (it's pure Rust) and
//! always reports exactly one "device" — the CPU it's currently
//! executing on. Its role in the dispatch table is as a last-resort
//! correctness fallback, so the descriptor it produces is intentionally
//! bare: no vendor/device IDs, no driver version, no memory count.
//! Anything that needs those facts should be dispatching to
//! `fuel-cpu-backend` instead.

use fuel_core_types::probe::{BackendId, BackendProbe, DeviceDescriptor};
use fuel_core_types::{DeviceLocation, error::Result};

pub struct ReferenceBackendProbe;

impl BackendProbe for ReferenceBackendProbe {
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
        enumerate_devices()
    }
}

/// Enumerate the reference backend's devices — always exactly one
/// entry representing "the CPU as a textbook-math executor."
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    Ok(vec![DeviceDescriptor {
        backend:            BackendId::Reference,
        device_index:       0,
        hardware_sku:       "reference (pure-Rust oracle)".to_string(),
        vendor_id:          0,
        device_id:          0,
        compute_capability: None,
        driver_version:     env!("CARGO_PKG_VERSION").to_string(),
        total_memory_bytes: 0,
        location:           DeviceLocation::Cpu,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_enumerates_exactly_one_device() {
        let d = enumerate_devices().expect("reference probe never fails");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].backend, BackendId::Reference);
        assert_eq!(d[0].device_index, 0);
        assert_eq!(d[0].location, DeviceLocation::Cpu);
    }
}
