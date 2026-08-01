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

/// Both transfer directions must (a) lift the process-wide mapped-byte PEAK by at
/// least the staged size while the staging buffer is mapped, and (b) return the
/// CURRENT mapped total to its baseline once the call returns and the RAII guard
/// drops. Together these prove `MappedGuard` is wired into the H2D path
/// (`upload_bytes`) AND the D2H path (`download_bytes`), and that both release —
/// i.e. the instrument tracks the real staging mappings and does not ratchet.
///
/// One test (not two), because the two directions share the process-global meter
/// and separate `#[ignore]` tests in the same binary would race it; sequencing
/// them with a `reset_host_mapped_peak()` between gives each a clean window.
#[test]
#[ignore = "requires a live Vulkan device"]
fn upload_and_download_account_host_visible_mappings() {
    let Some(b) = backend_or_skip() else { return };

    let n: usize = 4 * 1024 * 1024; // 4 MiB — unmistakable against any noise
    let src: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();

    // ---- H2D: upload_bytes ------------------------------------------------
    reset_host_mapped_peak();
    let h2d_base_current = mapped_host_visible_bytes();
    let h2d_base_peak = mapped_host_visible_peak_bytes();

    let storage = b.upload_bytes(&src).expect("h2d upload");
    assert_eq!(storage.len_bytes(), n);

    let h2d_peak = mapped_host_visible_peak_bytes();
    assert!(
        h2d_peak >= h2d_base_peak + n as u64,
        "H2D upload of {n} bytes should lift the mapped-byte peak by >= {n}; \
         base_peak={h2d_base_peak}, peak={h2d_peak}",
    );
    assert_eq!(
        mapped_host_visible_bytes(),
        h2d_base_current,
        "H2D staging mapping must be released once upload_bytes returns (guard Drop)",
    );

    // ---- D2H: download_bytes ---------------------------------------------
    reset_host_mapped_peak();
    let d2h_base_current = mapped_host_visible_bytes();
    let d2h_base_peak = mapped_host_visible_peak_bytes();

    let got = b.download_bytes(&storage).expect("d2h download");
    assert_eq!(got, src, "round-trip bytes must match (correctness sanity)");

    let d2h_peak = mapped_host_visible_peak_bytes();
    assert!(
        d2h_peak >= d2h_base_peak + n as u64,
        "D2H download of {n} bytes should lift the mapped-byte peak by >= {n}; \
         base_peak={d2h_base_peak}, peak={d2h_peak}",
    );
    assert_eq!(
        mapped_host_visible_bytes(),
        d2h_base_current,
        "D2H staging mapping must be released once download_bytes returns (guard Drop)",
    );

    eprintln!(
        "mapped-meter live: H2D Δpeak={} D2H Δpeak={} (both released to baseline)",
        h2d_peak - h2d_base_peak,
        d2h_peak - d2h_base_peak,
    );
}
