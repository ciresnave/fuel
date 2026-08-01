//! Live-device validation that the host-visible mapped-byte meter is actually
//! wired into the H2D staging path (increment 2 of the aperture instrument).
//! Gated `#[ignore]` — run under the GPU lock:
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- \
//!   cargo test -p fuel-vulkan-backend --test mapped_meter_live -- --ignored --nocapture
//! ```
//!
//! Deliberately its OWN test binary (a single test) so the process-global meter
//! is pristine while it runs — no contention with the parallel upload tests in
//! `byte_storage_live`, which share a process and would race the global counter.

use fuel_vulkan_backend::{
    mapped_host_visible_bytes, mapped_host_visible_peak_bytes, reset_host_mapped_peak,
    DeviceSelection, VulkanBackend,
};

fn backend_or_skip() -> Option<VulkanBackend> {
    match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            None
        }
    }
}

/// A real H2D upload must (a) lift the process-wide mapped-byte PEAK by at least
/// the staged size while the staging buffer is mapped, and (b) return the CURRENT
/// mapped total to its baseline once `upload_bytes` returns and the RAII guard
/// drops. Together these prove `MappedGuard` is wired into `upload_bytes` AND that
/// it releases — i.e. the instrument tracks the real staging mapping and does not
/// ratchet.
#[test]
#[ignore = "requires a live Vulkan device"]
fn upload_bytes_accounts_host_visible_mapping() {
    let Some(b) = backend_or_skip() else { return };

    // Open a clean measurement window; nothing else touches the global in this
    // single-test binary.
    reset_host_mapped_peak();
    let baseline_current = mapped_host_visible_bytes();
    let baseline_peak = mapped_host_visible_peak_bytes();

    let n: usize = 4 * 1024 * 1024; // 4 MiB — unmistakable against any noise
    let src = vec![0xABu8; n];
    let storage = b.upload_bytes(&src).expect("h2d upload");
    assert_eq!(storage.len_bytes(), n);

    // (a) The peak captured the mapping while it was live.
    let after_peak = mapped_host_visible_peak_bytes();
    assert!(
        after_peak >= baseline_peak + n as u64,
        "upload of {n} bytes should have lifted the mapped-byte peak by >= {n}; \
         baseline_peak={baseline_peak}, after_peak={after_peak}",
    );

    // (b) The guard released on return — current is back to baseline (the returned
    // storage is device-local, hence not host-visible and not counted here).
    let after_current = mapped_host_visible_bytes();
    assert_eq!(
        after_current, baseline_current,
        "the staging mapping must be released once upload_bytes returns \
         (guard Drop); baseline_current={baseline_current}, after_current={after_current}",
    );

    eprintln!(
        "mapped-meter live: baseline_peak={baseline_peak} after_peak={after_peak} \
         (Δpeak={}), current back to {after_current}",
        after_peak - baseline_peak,
    );
}
