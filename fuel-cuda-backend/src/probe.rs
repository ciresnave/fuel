// SPDX-License-Identifier: MIT OR Apache-2.0
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
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Memoized enumeration result — the whole point of this module's caching.
///
/// The `Err` arm is cached deliberately, exactly as in the Vulkan twin: caching
/// the FAILURE is what stops a box with no CUDA runtime from re-entering the
/// driver load on every call.
///
/// **This is the CUDA half of the fan-out fixed for Vulkan in `9bb68e6b`.**
/// That commit's own caveat recorded that `gpu-run` serializes runs BETWEEN
/// processes and does nothing for WITHIN-process fan-out, which needs separate
/// memoization — this is that unfixed half. Symptom before the fix:
/// `cargo test -p fuel-dispatch --features cuda --lib` stalls indefinitely at
/// default parallelism while `-- --test-threads=1` passes 743/0 in 19s, with the
/// stalling test set VARYING between runs (a logic bug picks the same victim;
/// contention picks whoever races).
///
/// **Hot-plug caveat:** the cache lives for the process lifetime, so a device
/// attached after the first enumeration will not appear. Correct for a test or
/// CLI run and the deliberate tradeoff for killing the fan-out.
static DEVICE_CACHE: OnceLock<std::result::Result<Vec<DeviceDescriptor>, String>> = OnceLock::new();

/// How many times the driver was ACTUALLY probed, process-global TOTAL across
/// the memoized path and any direct [`enumerate_devices_uncached`] caller.
///
/// Incremented inside the real probe. Because it is a total, it is NOT safe for
/// the concurrency test to assert `== 1` against — a sibling test calling the
/// uncached path directly would pollute it. That is what
/// [`MEMOIZED_PROBE_CALLS`] exists for.
static PROBE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// How many times the MEMOIZED path ran the real probe — incremented inside the
/// `OnceLock::get_or_init` closure, so it is exactly one per process however
/// many threads race, and immune to direct uncached callers.
///
/// Keeping the counter here rather than in the wrapper is the difference between
/// an assertion of `== 1` (which proves memoization) and `>= 1` (which proves
/// nothing).
static MEMOIZED_PROBE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Real driver probes so far this process — TOTAL across both paths. Test seam.
#[doc(hidden)]
pub fn probe_call_count() -> usize {
    PROBE_CALLS.load(Ordering::SeqCst)
}

/// Real driver probes performed via the MEMOIZED path — exactly one per
/// process. Test seam; this is the counter a concurrency assertion belongs on.
#[doc(hidden)]
pub fn memoized_probe_call_count() -> usize {
    MEMOIZED_PROBE_CALLS.load(Ordering::SeqCst)
}

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

/// Enumerate every CUDA device currently visible — **memoized**, one real
/// probe per process.
///
/// The old doc-comment called the probe "cheap — creates no contexts or
/// streams". That is true of the work it *asks for* and false of what it
/// *costs* under concurrency: `Device::count()` forces driver initialization,
/// and K threads reaching a cold probe simultaneously is what stalled the test
/// suite. Cheap-per-call and safe-under-fan-out are different properties.
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    DEVICE_CACHE
        .get_or_init(|| {
            MEMOIZED_PROBE_CALLS.fetch_add(1, Ordering::SeqCst);
            enumerate_devices_uncached().map_err(|e| e.to_string())
        })
        .clone()
        .map_err(Error::Msg)
}

/// The real probe: re-enters the CUDA driver on **every** call.
///
/// Production code must call [`enumerate_devices`]. This is `pub` only so the
/// fan-out characterization test can exercise the pre-memoization behaviour
/// honestly — racing this directly *is* what the old code did — rather than by
/// temporarily breaking the cache, which would leave a window where a
/// half-sabotaged probe could be committed by accident.
#[doc(hidden)]
pub fn enumerate_devices_uncached() -> Result<Vec<DeviceDescriptor>> {
    PROBE_CALLS.fetch_add(1, Ordering::SeqCst);
    let count = baracuda_driver::Device::count()
        .map_err(|e| Error::Msg(format!("cuda probe: device count failed: {e}")).bt())?;
    let driver_ver = baracuda_driver::version()
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let mut out = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        let dev = baracuda_driver::Device::get(ordinal).map_err(|e| {
            Error::Msg(format!("cuda probe: Device::get({ordinal}) failed: {e}")).bt()
        })?;
        let name = dev
            .name()
            .unwrap_or_else(|_| format!("cuda:{ordinal} (name query failed)"));
        let cc = dev.compute_capability().ok();
        let total_mem = dev.total_memory().unwrap_or(0);
        let pci_device_id = dev
            .attribute(baracuda_cuda_sys::types::CUdevice_attribute::PCI_DEVICE_ID as i32)
            .map(|v| v as u32)
            .unwrap_or(0);

        out.push(DeviceDescriptor {
            backend: BackendId::Cuda,
            device_index: ordinal,
            hardware_sku: name,
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: pci_device_id,
            compute_capability: cc,
            // TODO(cuda): CUDA *can* report this — it is
            // `CU_DEVICE_ATTRIBUTE_WARP_SIZE`, readable via the same
            // `dev.attribute(..)` path used for PCI_DEVICE_ID above (32 on
            // every shipping NVIDIA architecture). Left `None` here because
            // this crate needs the `cuda` feature + nvcc to build, which the
            // vulkane-0.9.0 change that introduced this field could not
            // exercise — shipping either an unverified enum-variant name or a
            // hardcoded constant into a probe would be worse than a documented
            // `None`. Populate it in a CUDA-buildable change.
            subgroup_width: None,
            driver_version: driver_ver.clone(),
            total_memory_bytes: total_mem,
            location: DeviceLocation::Cuda {
                gpu_id: ordinal as usize,
            },
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    // GAP-157/GAP-243: these tests REQUIRE a live CUDA device. They are
    // `#[ignore]`d, so they run only when explicitly asked for — which makes a
    // missing device a FAILURE, not a silent early return that reports `ok`
    // having asserted nothing. Run them with the documented runner:
    //   pwsh scripts/gpu-run.ps1 -Project fuel -- \
    //       cargo test -p fuel-cuda-backend --features cuda -- --ignored
    //
    // The policy now lives in `fuel_test_support::hardware`, whose CUDA default
    // is `Fatal` for the same reason this note gives — every call site is an
    // `#[ignore]`d test, so running one is itself the declaration. Behaviour is
    // therefore unchanged where a device exists; what is new is that
    // `FUEL_REQUIRE_CUDA=0` can deliberately disarm it, so a wrong default is a
    // nuisance rather than a wall.

    /// On machines without any CUDA device (headless CI, for example)
    /// the probe should return `Ok(vec![])`, not error. Where a GPU
    /// does exist, every descriptor should carry NVIDIA's vendor id
    /// and a CUDA `DeviceLocation` matching its ordinal.
    #[ignore = "requires a live CUDA device"]
    #[test]
    fn cuda_probe_is_total() {
        // ⚠️ GAP-243. This took `required_ok`, which panics only on `Err` —
        // but `enumerate_devices_uncached` returns **`Ok(vec![])`** when
        // `Device::count()` is 0, which is what a CUDA RUNTIME WITH NO USABLE
        // DEVICE reports (the doc comment above says so in as many words). So
        // on such a box the empty vec sailed through the guard, the loop below
        // iterated ZERO times, and this test reported `ok` having asserted
        // nothing — GAP-243's exact defect, inside the file carrying the
        // GAP-157 fix, because the guard was aimed at *enumeration failing*
        // rather than at *the device being absent*.
        //
        // Not an exotic configuration: a box with the CUDA SDK and no usable
        // GPU is ordinary (it is compile-only CI), and the same state arises
        // from a GPU claimed by another process or a driver/runtime mismatch.
        let devices = match enumerate_devices() {
            Ok(d) if !d.is_empty() => d,
            Ok(_) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(
                        "the CUDA runtime loaded but reported 0 devices",
                    ),
                );
            }
            Err(e) => {
                return fuel_test_support::hardware::skip(
                    fuel_test_support::hardware::Hardware::Cuda,
                    fuel_test_support::hardware::Missing::device(format!("enumerate_devices: {e}")),
                );
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

#[cfg(test)]
mod probe_memoization_tests {
    use super::*;
    // GAP-157: `#[ignore]`d — see the note in `tests` above. A missing CUDA
    // runtime is a FAILURE here, not a silent `return` that reports `ok`.
    // GAP-243: these two require the RUNTIME, not a device, and that is
    // correct rather than an oversight — their property is the number of real
    // driver probes, which holds perfectly well when `Device::count()` is 0. Do
    // NOT "fix" them to demand a non-empty device list: that would make them
    // fail on a machine where the thing they actually measure works. Only
    // `cuda_probe_is_total` needed strengthening, because its assertions live
    // inside a `for` over the device list and an empty list skips them all.

    /// K threads racing a cold probe must produce **exactly one** real driver
    /// enumeration.
    ///
    /// This is the CUDA half of the fan-out fixed for Vulkan in `9bb68e6b`.
    /// Before memoization, `cargo test -p fuel-dispatch --features cuda --lib`
    /// stalled indefinitely at default parallelism while `--test-threads=1`
    /// passed 743/0 in 19s — and the STALLING SET VARIED between runs, which is
    /// the signature of contention rather than a logic bug.
    ///
    /// The assertion is on [`MEMOIZED_PROBE_CALLS`], not [`PROBE_CALLS`].
    /// `PROBE_CALLS` is a process-global total shared with any direct
    /// `enumerate_devices_uncached` caller, so asserting `== 1` on it would be
    /// polluted by a parallel sibling; the memoized counter is incremented
    /// inside the `get_or_init` closure and no caller outside this module can
    /// touch it. Getting that wrong turns an `== 1` proof of memoization into a
    /// `>= 1` proof of nothing.
    ///
    /// Requires a CUDA runtime: a cached `Err` is still exactly one probe, so
    /// the property holds vacuously where the driver never loads — which is why
    /// this is `#[ignore]`d and REQUIRES the device rather than returning early
    /// (GAP-157: an early return reports `ok` having asserted nothing).
    #[ignore = "requires a live CUDA device"]
    #[test]
    fn concurrent_enumeration_probes_the_driver_exactly_once() {
        if let Err(e) = enumerate_devices() {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Cuda,
                fuel_test_support::hardware::Missing::device(format!("no CUDA runtime: {e}")),
            );
        }
        let before = memoized_probe_call_count();

        let threads: Vec<_> = (0..16)
            .map(|_| std::thread::spawn(|| enumerate_devices().map(|d| d.len())))
            .collect();
        let lens: Vec<usize> = threads
            .into_iter()
            .map(|t| t.join().expect("probe thread panicked").expect("probe"))
            .collect();

        // Every racer must see the SAME device list — a torn or per-thread
        // result would mean the cache is not actually shared.
        assert!(
            lens.windows(2).all(|w| w[0] == w[1]),
            "racing threads saw different device counts: {lens:?}",
        );

        assert_eq!(
            memoized_probe_call_count(),
            before,
            "16 racing threads triggered {} additional real driver probes; the memoized path must probe exactly once per process",
            memoized_probe_call_count() - before,
        );
    }

    /// Positive control for the test above. If `enumerate_devices_uncached` did
    /// NOT actually re-probe per call, the memoization assertion would pass
    /// trivially and prove nothing — the same vacuous-oracle shape that has
    /// bitten this codebase repeatedly.
    #[ignore = "requires a live CUDA device"]
    #[test]
    fn uncached_enumeration_really_does_re_probe() {
        if let Err(e) = enumerate_devices() {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Cuda,
                fuel_test_support::hardware::Missing::device(format!("no CUDA runtime: {e}")),
            );
        }
        let before = probe_call_count();
        let _ = enumerate_devices_uncached();
        let _ = enumerate_devices_uncached();
        assert_eq!(
            probe_call_count() - before,
            2,
            "the uncached path must re-probe on every call, or the memoization test above is vacuous",
        );
    }
}
