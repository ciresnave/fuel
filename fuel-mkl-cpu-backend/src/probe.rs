//! Device enumeration for the oneMKL CPU backend.
//!
//! Returns one descriptor per physical CPU when oneMKL's runtime
//! library loads, otherwise an empty vec (not an error — the absent
//! library is the "this backend is unavailable on this host" signal).
//! The Phase 6b probe collector swallows empty enumerations silently.

use fuel_core_types::probe::{BackendId, BackendProbe, DeviceDescriptor};
use fuel_core_types::{DeviceLocation, error::Result};

pub struct MklBackendProbe;

impl BackendProbe for MklBackendProbe {
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
        enumerate_devices()
    }
}

/// Enumerate MKL "devices" — one entry if `mkl_rt` loads, empty
/// otherwise.
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    if crate::probe_mkl_loadable().is_err() {
        return Ok(Vec::new());
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let sku = format!("{} ({} logical cores) [oneMKL]",
        std::env::consts::ARCH, cores);
    Ok(vec![DeviceDescriptor {
        backend:            BackendId::Mkl,
        device_index:       0,
        hardware_sku:       sku,
        // Intel PCI vendor ID — proxy for "Intel-tuned BLAS path."
        // Like AOCL's vendor_id=0x1022, this is a label, not a runtime
        // gate. MKL works on AMD CPUs too.
        vendor_id:          0x8086,
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
    fn enumerate_returns_zero_or_one_descriptor() {
        let d = enumerate_devices().expect("mkl probe never errors; absence → empty vec");
        match d.len() {
            0 => {
                eprintln!("oneMKL not present on this host; that's fine");
            }
            1 => {
                assert_eq!(d[0].backend, BackendId::Mkl);
                assert_eq!(d[0].location, DeviceLocation::Cpu);
            }
            n => panic!("MKL probe returned {n} descriptors; expected 0 or 1"),
        }
    }
}
