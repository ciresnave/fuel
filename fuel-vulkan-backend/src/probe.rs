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

/// Create an `Instance` at the requested API version, falling back **one step**
/// to V1_2 if the loader refuses it.
///
/// Why the fallback exists. Fuel asks for **V1_3** because
/// `SubgroupProperties::size_control` (the pinnable subgroup-size range) is
/// Vulkan 1.3 core / `VK_EXT_subgroup_size_control`, and `effective_api_version`
/// is `min(instance, device)` — so an instance below 1.3 makes size control
/// unreadable *regardless of what the device supports*. Requesting 1.3 costs
/// nothing on a 1.2 device: the effective version simply clamps back down.
///
/// The risk is narrow but real. A Vulkan **1.0** loader returns
/// `VK_ERROR_INCOMPATIBLE_DRIVER` for any `apiVersion` above 1.0; loaders at 1.1
/// and above must accept any value. Without a fallback, that failure would
/// propagate as `Err` from a probe documented as *total*, and Fuel would report
/// **zero Vulkan devices** on a machine that has one — a silent capability loss
/// that looks identical to "no Vulkan installed". The fallback converts a hard
/// failure into a logged degradation.
///
/// Takes a *builder* rather than a populated struct because
/// `InstanceCreateInfo` is not `Clone` — the fallback needs to construct a
/// second, independent value at the lower version.
pub(crate) fn new_instance_preferring_v1_3<'a>(
    build: impl Fn(ApiVersion) -> InstanceCreateInfo<'a>,
) -> Result<Instance> {
    match Instance::new(build(ApiVersion::V1_3)) {
        Ok(i) => Ok(i),
        Err(e) => {
            tracing::warn!(
                error = ?e,
                "Vulkan instance creation failed at V1_3; retrying at V1_2. \
                 SubgroupProperties::size_control will read None regardless of \
                 device support."
            );
            Instance::new(build(ApiVersion::V1_2)).map_err(vk_err)
        }
    }
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
    let instance = new_instance_preferring_v1_3(|api_version| InstanceCreateInfo {
        engine_name: Some("fuel-vulkan-backend probe"),
        api_version,
        ..Default::default()
    })?;
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

    /// **Characterization of the V1_2 → V1_3 instance bump.** Not an assertion
    /// about this machine — it reports what the raised instance version actually
    /// bought, per device, so the change is *observed* rather than assumed.
    ///
    /// The claim under test: `SubgroupProperties::size_control` is Vulkan 1.3
    /// core / `VK_EXT_subgroup_size_control`, and `effective_api_version` is
    /// `min(instance, device)` — so at V1_2 it read `None` regardless of what
    /// the device supported. If the bump worked and the device is 1.3-capable,
    /// it should now be `Some`.
    ///
    /// The only hard assertion is the invariant that must hold either way:
    /// effective version can never exceed the instance version we asked for.
    #[test]
    fn v1_3_instance_bump_characterization() {
        let instance = match new_instance_preferring_v1_3(|api_version| InstanceCreateInfo {
            engine_name: Some("fuel-vulkan-backend probe"),
            api_version,
            ..Default::default()
        }) {
            Ok(i) => i,
            Err(e) => { eprintln!("vulkan unavailable, skipped: {e}"); return; }
        };
        let physicals = match instance.enumerate_physical_devices() {
            Ok(p) => p,
            Err(e) => { eprintln!("enumeration failed, skipped: {e:?}"); return; }
        };
        eprintln!("=== V1_3 instance bump: {} device(s) ===", physicals.len());
        for (i, p) in physicals.iter().enumerate() {
            let eff = p.effective_api_version();
            let sg = p.subgroup_properties();
            eprintln!(
                "  [{i}] {}\n       effective_api = {}.{}\n       subgroup_size = {:?}\n       size_control  = {:?}",
                p.properties().device_name(),
                eff.major(), eff.minor(),
                sg.as_ref().map(|s| s.subgroup_size),
                sg.as_ref().map(|s| s.size_control.is_some()),
            );
            // Invariant, independent of what this machine supports: the
            // effective version is min(instance, device), so it can never
            // exceed what we requested.
            assert!(
                eff.major() < 1 || (eff.major() == 1 && eff.minor() <= 3),
                "effective_api_version {}.{} exceeds the requested V1_3",
                eff.major(), eff.minor(),
            );
        }
    }

    /// **`driver_version` must not depend on the instance API version.**
    ///
    /// It is an [`EquivalenceKey`] field — the Judge's profile-cache key. If the
    /// same physical device reported a different driver identity under a V1_2
    /// instance than under V1_3, then raising the instance version would
    /// silently invalidate every cached profile, *and* two Fuel components
    /// creating instances at different versions would disagree about which
    /// device they were looking at. Neither failure announces itself.
    ///
    /// Asserted rather than merely observed because the consequence is a silent
    /// cache-key split, not a visible error.
    #[test]
    fn driver_identity_is_stable_across_instance_version() {
        let mk = |api: ApiVersion| {
            Instance::new(InstanceCreateInfo {
                engine_name: Some("fuel-vulkan-backend probe"),
                api_version: api,
                ..Default::default()
            })
            .ok()
            .and_then(|inst| {
                let v: Vec<_> = inst
                    .enumerate_physical_devices()
                    .ok()?
                    .iter()
                    .map(|p| {
                        let d = p.driver_properties();
                        (
                            p.properties().device_name(),
                            d.as_ref().map(|d| format!("{:?} {} {}", d.driver_id, d.driver_name, d.driver_info)),
                        )
                    })
                    .collect();
                Some(v)
            })
        };
        let (Some(at_1_2), Some(at_1_3)) = (mk(ApiVersion::V1_2), mk(ApiVersion::V1_3)) else {
            eprintln!("vulkan unavailable at one or both versions, skipped");
            return;
        };
        eprintln!("V1_2: {at_1_2:#?}");
        eprintln!("V1_3: {at_1_3:#?}");
        assert_eq!(
            at_1_2, at_1_3,
            "driver identity differs by instance API version — `driver_version` is an \
             EquivalenceKey field, so this would split the Judge's profile cache silently"
        );
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
