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
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use vulkane::safe::*;

/// Memoized result of the one real loader probe this process performs.
///
/// The error is stored as a `String` because [`Error`] is not `Clone` and the
/// cache must hand out an owned value per call. Failures are cached too: a
/// loader that is missing now will not appear later in the same run, and
/// caching the failure is what stops a box without Vulkan from re-entering
/// `Instance::new` on every single call.
static DEVICE_CACHE: OnceLock<std::result::Result<Vec<DeviceDescriptor>, String>> =
    OnceLock::new();

/// How many times the loader was *actually* probed — i.e. how many
/// `vkCreateInstance` calls this process has made from here — as opposed to
/// how many times [`enumerate_devices`] was called.
///
/// Incremented inside [`enumerate_devices_uncached`], the real probe, and
/// deliberately NOT inside the memoizing wrapper. That placement is what lets
/// the concurrency test assert `== 1` in absolute terms: memoization
/// guarantees the uncached path runs at most once per process, so the
/// assertion cannot be diluted by test ordering or by other callers.
static PROBE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Number of real loader probes (`vkCreateInstance` calls) performed so far in
/// this process. Test seam.
#[doc(hidden)]
pub fn probe_call_count() -> usize {
    PROBE_CALLS.load(Ordering::SeqCst)
}

pub struct VulkanBackendProbe;

impl BackendProbe for VulkanBackendProbe {
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
        enumerate_devices()
    }
}

fn vk_err(e: impl std::fmt::Debug) -> Error {
    Error::Msg(format!("vulkan probe: {e:?}"))
}

/// Enumerate every Vulkan physical device currently visible to the loader.
///
/// The loader is probed **once per process**; every later call returns a clone
/// of the memoized result. This matters well beyond saving a few milliseconds:
/// `SystemTopology::current()` (fuel-dispatch) deliberately builds its snapshot
/// *outside* its cache lock — "we may race with another rebuild; the last
/// writer wins" — so on a cold cache every racing thread reaches this function.
/// Un-memoized, K threads meant **K concurrent `vkCreateInstance` calls in one
/// process**, which is a fan-out no machine-wide GPU lock can see: such a lock
/// admits the process once and the process then fans out internally. Under
/// `cargo test` K is cargo's thread count.
///
/// Creates an `Instance` but never a logical `Device`, queue, or any GPU
/// allocation.
///
/// **Hot-plug caveat:** the cache lives for the process lifetime, so a device
/// attached after the first enumeration will not appear. That is correct for a
/// test or CLI run and is the deliberate tradeoff for killing the fan-out; a
/// long-lived process that must see hot-plug needs an explicit invalidation
/// hook, which does not exist today.
pub fn enumerate_devices() -> Result<Vec<DeviceDescriptor>> {
    DEVICE_CACHE
        .get_or_init(|| enumerate_devices_uncached().map_err(|e| e.to_string()))
        .clone()
        .map_err(Error::Msg)
}

/// The real probe: creates a fresh `VkInstance` on **every** call.
///
/// Production code must call [`enumerate_devices`] instead. This is `pub` only
/// so the fan-out characterization test can exercise the pre-memoization
/// behaviour honestly — racing this directly *is* what the old code did —
/// rather than by temporarily breaking the cache and re-running, which would
/// leave a window where a half-sabotaged probe could be committed by accident.
#[doc(hidden)]
pub fn enumerate_devices_uncached() -> Result<Vec<DeviceDescriptor>> {
    PROBE_CALLS.fetch_add(1, Ordering::SeqCst);
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

    /// The fan-out this memoization exists to kill.
    ///
    /// `SystemTopology::current()` builds its snapshot outside its own cache
    /// lock by design, so on a cold cache every racing thread lands here.
    /// Before memoization that meant one `vkCreateInstance` per thread — a
    /// concurrent-Vulkan-setup storm inside a single process, which a
    /// machine-wide GPU lock cannot prevent because it admits the process once
    /// and the process then fans out.
    ///
    /// Asserting on the probe counter rather than on timing or on device
    /// contents is what gives this teeth on any box: it holds identically
    /// whether the loader is present (probe succeeds) or absent (probe fails
    /// and the failure is cached), and it fails loudly if someone reverts the
    /// `OnceLock` or adds a second un-memoized call path.
    ///
    /// Note this test creates at most ONE `VkInstance` for the whole process —
    /// it is the *pre-fix* behaviour that was expensive, not this. The
    /// assertion is absolute rather than a delta because memoization makes the
    /// uncached path run at most once per process, so no amount of other
    /// activity can dilute it.
    #[test]
    fn concurrent_enumeration_probes_the_loader_exactly_once() {
        const THREADS: usize = 16;
        let start = std::sync::Barrier::new(THREADS);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    // Release all threads together so they genuinely contend
                    // on the OnceLock rather than trickling through it.
                    start.wait();
                    let _ = enumerate_devices();
                });
            }
        });

        assert_eq!(
            probe_call_count(),
            1,
            "the loader must be probed exactly once across {THREADS} racing \
             threads; got {} probes, which means the per-thread \
             vkCreateInstance fan-out is back",
            probe_call_count(),
        );
    }

    /// Characterizes the fan-out the memoization removes, by racing the
    /// **uncached** entry point directly — which is faithfully what the old
    /// code did on every `SystemTopology::current()` cache miss.
    ///
    /// `#[ignore]` and deliberately bounded to K=4. This test intentionally
    /// fires concurrent `vkCreateInstance` calls, which is the mechanism
    /// implicated in the 2026-07-31 `VIDEO_MEMORY_MANAGEMENT_INTERNAL` /
    /// `DdiMapCpuHostAperture` bugcheck on this hardware. K=4 proves the
    /// fan-out unambiguously (4 != 1) at a quarter of the blast radius of the
    /// K=16 the green test uses. Run it deliberately, serialized against other
    /// GPU work, and alone:
    ///
    /// ```text
    /// cargo test -p fuel-vulkan-backend -- --ignored --exact \
    ///     probe::tests::uncached_enumeration_fans_out_per_thread \
    ///     --test-threads=1
    /// ```
    #[test]
    #[ignore = "deliberately fires concurrent vkCreateInstance; run serialized under the GPU lock"]
    fn uncached_enumeration_fans_out_per_thread() {
        const THREADS: usize = 4;
        let before = probe_call_count();
        let start = std::sync::Barrier::new(THREADS);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    start.wait();
                    let _ = enumerate_devices_uncached();
                });
            }
        });

        assert_eq!(
            probe_call_count() - before,
            THREADS,
            "the uncached path must probe once per thread — that is the \
             fan-out `enumerate_devices` exists to collapse",
        );
    }

    /// Memoization must not change what callers see: repeated calls agree, and
    /// they agree with what the single underlying probe returned.
    #[test]
    fn repeated_enumeration_is_stable() {
        let a = enumerate_devices();
        let b = enumerate_devices();
        match (a, b) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "cached enumeration changed between calls"),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("cached enumeration flipped success/failure: {a:?} vs {b:?}"),
        }
    }

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
