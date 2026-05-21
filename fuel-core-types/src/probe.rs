//! Cross-backend device enumeration — the foundation for Phase 6b's
//! `probe → judge → dispatch` lifecycle.
//!
//! # Why
//!
//! Fuel dispatches ops across multiple backends (CPU, reference, CUDA,
//! Vulkan, later Metal / MKL / AOCL). Each backend may see zero, one,
//! or many physical devices. A single physical GPU may be reachable
//! through more than one backend (an NVIDIA card is visible to both
//! CUDA and Vulkan) and will produce meaningfully different
//! performance profiles depending on which backend routes it. The
//! profiling unit — and therefore the dispatch-table key — is not
//! "backend" but `(backend, device)`.
//!
//! This module defines the data types that represent that pair in a
//! persistence-friendly shape. The actual enumeration logic lives in
//! each backend crate (a free function `enumerate_devices() ->
//! Result<Vec<DeviceDescriptor>>`, not a trait method — see "Why a
//! convention, not a trait" below). A top-level collector in
//! `fuel-graph-router` (or a future `fuel-judge`) calls every
//! backend's enumerator and assembles the combined device table.
//!
//! # Equivalence classes
//!
//! Four identical RTX 4090s through CUDA produce four essentially-
//! identical performance profiles — the Judge doesn't need to run
//! four times. Two [`DeviceDescriptor`]s that share an
//! [`EquivalenceKey`] can share a profile: profile once per key,
//! apply to every device in the class. Cross-backend equivalence
//! never holds (CUDA and Vulkan on the same silicon still key
//! differently because they have different driver paths and
//! submission overhead).
//!
//! # Why a convention, not a trait
//!
//! `BackendProbe` *could* be a trait implemented by each backend's
//! device or storage type. But enumeration is a **static** operation —
//! it doesn't need a live backend instance, and requiring one creates
//! a circular bootstrap problem (you can't know which backends exist
//! until you've enumerated them, but trait impls need types, and the
//! types may not be loadable if their runtime isn't present). The
//! cleaner shape is a plain free function per backend crate that can
//! be called via the crate's public API without constructing any
//! state. The [`BackendProbe`] trait below is a *documentation anchor*
//! — a marker trait each backend-crate-level probe function satisfies
//! in spirit, so the contract is in one place and the collector has
//! a single import to follow.

use crate::DeviceLocation;
use crate::error::Result;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Identifier for a Fuel backend implementation. Stable across runs
/// so the Judge's persisted profile tables remain valid when the
/// process restarts. Adding a new variant is a non-breaking change;
/// the enum is `#[non_exhaustive]` so downstream pattern matches stay
/// forward-compatible.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BackendId {
    /// `fuel-reference-backend` — textbook-correct CPU executor used as
    /// the oracle. Always available; included in the dispatch table
    /// mostly as a correctness-check last-resort fallback.
    Reference,
    /// `fuel-cpu-backend` — optimized CPU executor (gemm + rayon).
    /// One descriptor per physical CPU in the current scope; NUMA
    /// splits are Phase 7b territory.
    Cpu,
    /// `fuel-cuda-backend` via baracuda.
    Cuda,
    /// `fuel-vulkan-backend` via vulkane.
    Vulkan,
    /// `fuel-metal-backend` (future — not yet probe-wired).
    Metal,
    /// `fuel-mkl-cpu-backend` (Phase 7b, Intel CPU variant).
    Mkl,
    /// `fuel-aocl-cpu-backend` (Phase 7b, AMD CPU variant).
    Aocl,
}

impl BackendId {
    /// A short lowercase identifier suitable for log lines, directory
    /// paths, and JSON keys. Stable across Fuel versions.
    pub fn as_str(self) -> &'static str {
        match self {
            BackendId::Reference => "reference",
            BackendId::Cpu       => "cpu",
            BackendId::Cuda      => "cuda",
            BackendId::Vulkan    => "vulkan",
            BackendId::Metal     => "metal",
            BackendId::Mkl       => "mkl",
            BackendId::Aocl      => "aocl",
        }
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything a Judge / dispatch table needs to know about a single
/// `(backend, device)` pair. Constructed by each backend's
/// `enumerate_devices()` and consumed by the collector + the Judge.
///
/// Field layout is stable: adding fields is non-breaking so long as
/// they're additive (`#[serde(default)]` on new optional fields keeps
/// older profile files loading).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeviceDescriptor {
    /// Which backend this descriptor belongs to.
    pub backend: BackendId,
    /// Zero-based index within the backend. For CPU this is always 0
    /// until NUMA splits arrive; for CUDA/Vulkan it's the device
    /// ordinal the backend's own API uses.
    pub device_index: u32,
    /// Human-readable device name as reported by the driver.
    /// Examples: `"NVIDIA GeForce RTX 4070"`, `"AMD Radeon RX 7900 XT"`,
    /// `"Intel(R) Core(TM) i9-14900K"`. Used primarily for logs.
    pub hardware_sku: String,
    /// PCI-SIG vendor ID (NVIDIA=0x10DE, AMD=0x1002, Intel=0x8086).
    /// `0` for CPU entries that don't map to a PCI vendor.
    pub vendor_id: u32,
    /// PCI-SIG device ID for discrete hardware; zero for CPU.
    pub device_id: u32,
    /// CUDA compute capability, when applicable. `None` for non-CUDA
    /// entries. Stored as `(major, minor)` — e.g. `(8, 9)` for sm_89.
    pub compute_capability: Option<(u32, u32)>,
    /// Driver version string as reported by the backend. Cache
    /// invalidation key: if the driver rev changes, Judge re-runs.
    pub driver_version: String,
    /// Total device-visible memory in bytes. Zero if unknown.
    pub total_memory_bytes: u64,
    /// Fuel's placement enum for this descriptor. Populated by the
    /// backend so the router can build `DeviceLocation` handles
    /// without re-querying.
    pub location: DeviceLocation,
}

impl DeviceDescriptor {
    /// The key the Judge uses to share a profile across identical
    /// devices. Two descriptors that compare equal here can share a
    /// single profile entry; the table's runtime dispatch just looks
    /// up the equivalence key rather than the full descriptor.
    pub fn equivalence_key(&self) -> EquivalenceKey {
        EquivalenceKey {
            backend:            self.backend,
            vendor_id:          self.vendor_id,
            device_id:          self.device_id,
            compute_capability: self.compute_capability,
            driver_version:     self.driver_version.clone(),
        }
    }
}

/// Hash key that groups [`DeviceDescriptor`]s likely to profile
/// identically. Explicitly **scoped by backend** — a CUDA RTX 4090
/// and a Vulkan RTX 4090 are the same silicon but different
/// submission paths, so they never share an equivalence class.
///
/// The driver version is part of the key because the same
/// `(vendor, device, capability)` can perform measurably
/// differently across driver releases; invalidating on driver
/// upgrade is cheap insurance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EquivalenceKey {
    pub backend:            BackendId,
    pub vendor_id:          u32,
    pub device_id:          u32,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version:     String,
}

/// Marker trait documenting the enumeration contract each backend
/// crate satisfies. **Not dispatch-through** — each backend provides
/// a plain `pub fn enumerate_devices()` at crate root, and the
/// collector calls them by name. This trait is a single place to
/// reference the expected signature.
///
/// See the module docs for why this is a convention rather than a
/// dispatch trait.
pub trait BackendProbe {
    /// Enumerate every `(backend, device)` pair this backend can
    /// currently reach. Empty vec = backend loaded but no usable
    /// hardware visible (e.g. CUDA runtime present but zero GPUs).
    /// `Err` = backend could not be probed at all (e.g. driver
    /// dynamic-load failed).
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id_as_str_is_stable() {
        assert_eq!(BackendId::Reference.as_str(), "reference");
        assert_eq!(BackendId::Cpu.as_str(), "cpu");
        assert_eq!(BackendId::Cuda.as_str(), "cuda");
        assert_eq!(BackendId::Vulkan.as_str(), "vulkan");
        assert_eq!(BackendId::Metal.as_str(), "metal");
    }

    #[test]
    fn equivalence_key_collapses_identical_cuda_devices() {
        let make = |idx: u32| DeviceDescriptor {
            backend:            BackendId::Cuda,
            device_index:       idx,
            hardware_sku:       "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id:          0x10DE,
            device_id:          0x2684,
            compute_capability: Some((8, 9)),
            driver_version:     "550.54.14".to_string(),
            total_memory_bytes: 25_769_803_776,
            location:           DeviceLocation::Cuda { gpu_id: idx as usize },
        };
        let a = make(0);
        let b = make(1);
        // Same silicon, different ordinals → same equivalence class.
        assert_eq!(a.equivalence_key(), b.equivalence_key());
        // Different device_index → descriptors still distinct overall.
        assert_ne!(a, b);
    }

    #[test]
    fn equivalence_key_splits_same_silicon_across_backends() {
        let cuda = DeviceDescriptor {
            backend:            BackendId::Cuda,
            device_index:       0,
            hardware_sku:       "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id:          0x10DE,
            device_id:          0x2684,
            compute_capability: Some((8, 9)),
            driver_version:     "550.54.14".to_string(),
            total_memory_bytes: 25_769_803_776,
            location:           DeviceLocation::Cuda { gpu_id: 0 },
        };
        let vulkan = DeviceDescriptor {
            backend:            BackendId::Vulkan,
            device_index:       0,
            hardware_sku:       "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id:          0x10DE,
            device_id:          0x2684,
            compute_capability: None,  // Vulkan doesn't expose CUDA CC.
            driver_version:     "550.54.14".to_string(),
            total_memory_bytes: 25_769_803_776,
            location:           DeviceLocation::Cpu,  // placeholder; vulkane uses a different variant
        };
        // Same chip, different submission paths → different classes.
        assert_ne!(cuda.equivalence_key(), vulkan.equivalence_key());
    }

    #[test]
    fn equivalence_key_splits_on_driver_version() {
        let base = DeviceDescriptor {
            backend:            BackendId::Cuda,
            device_index:       0,
            hardware_sku:       "X".to_string(),
            vendor_id:          1,
            device_id:          2,
            compute_capability: Some((8, 9)),
            driver_version:     "550".to_string(),
            total_memory_bytes: 0,
            location:           DeviceLocation::Cuda { gpu_id: 0 },
        };
        let newer = DeviceDescriptor { driver_version: "560".to_string(), ..base.clone() };
        assert_ne!(base.equivalence_key(), newer.equivalence_key());
    }
}
