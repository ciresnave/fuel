//! Live-device tests for the Phase 7.5 A4 substrate methods on
//! [`VulkanBackend`] / [`VulkanStorageBytes`]. Gated `#[ignore]` —
//! run with:
//!
//! ```sh
//! cargo test -p fuel-vulkan-backend --test byte_storage_live -- --ignored --nocapture
//! ```

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

fn backend_or_skip() -> Option<VulkanBackend> {
    match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            None
        }
    }
}

/// Smoke: alloc_bytes(byte_count) reports the right len_bytes and is
/// readable back via download_bytes.
#[test]
#[ignore]
fn alloc_then_download() {
    let Some(b) = backend_or_skip() else { return };
    let storage = b.alloc_bytes(32).expect("alloc");
    assert_eq!(storage.len_bytes(), 32);
    let got = b.download_bytes(&storage).expect("d2h");
    // alloc_bytes does NOT zero — the GPU buffer is uninitialized.
    // We only assert the length round-trips, not the content.
    assert_eq!(got.len(), 32);
}

/// H2D + D2H roundtrip: upload bytes, read them back, expect exact
/// byte equality.
#[test]
#[ignore]
fn upload_download_roundtrip_preserves_bytes() {
    let Some(b) = backend_or_skip() else { return };
    let src: Vec<u8> = (0..=255).collect();
    let storage = b.upload_bytes(&src).expect("h2d");
    assert_eq!(storage.len_bytes(), src.len());
    let got = b.download_bytes(&storage).expect("d2h");
    assert_eq!(got, src);
}

/// Zero-length transfer is sound: upload(empty) and alloc(0) both
/// succeed; download produces an empty Vec.
#[test]
#[ignore]
fn zero_length_transfers_round_trip() {
    let Some(b) = backend_or_skip() else { return };
    let from_empty = b.upload_bytes(&[]).expect("h2d 0");
    assert_eq!(from_empty.len_bytes(), 0);
    let got = b.download_bytes(&from_empty).expect("d2h 0");
    assert!(got.is_empty());

    let storage = b.alloc_bytes(0).expect("alloc 0");
    assert_eq!(storage.len_bytes(), 0);
    let got = b.download_bytes(&storage).expect("d2h 0 (alloc)");
    assert!(got.is_empty());
}

/// Larger transfer: 1 MiB pattern, exercises the non-trivial copy path.
#[test]
#[ignore]
fn one_mib_roundtrip_preserves_bytes() {
    let Some(b) = backend_or_skip() else { return };
    let src: Vec<u8> = (0..1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let storage = b.upload_bytes(&src).expect("h2d");
    let got = b.download_bytes(&storage).expect("d2h");
    assert_eq!(got, src);
}

/// BDA: a device-resident storage buffer yields a valid non-zero device
/// address — proving the `bufferDeviceAddress` device feature + the BDA
/// allocator option + the `SHADER_DEVICE_ADDRESS` buffer usage are all wired.
/// This is exactly the value the FDX Vulkan path (spec §3.3.1) sources as a
/// `kDLVulkan` tensor's `data` (the base; `byte_offset` is folded at dispatch).
#[test]
#[ignore]
fn device_storage_has_valid_buffer_device_address() {
    let Some(b) = backend_or_skip() else { return };
    let storage = b.alloc_bytes(256).expect("alloc");
    let buf = storage.buffer_opt().expect("device-resident storage must carry a buffer");
    let addr = buf
        .device_address()
        .expect("device_address must succeed (BDA feature + usage must be enabled)");
    assert_ne!(addr, 0, "a SHADER_DEVICE_ADDRESS buffer must have a non-zero device address");
    eprintln!("BDA device_address = {addr:#018x}");
}

/// Regression: enabling BDA (the device feature + the address usage bit on
/// every storage buffer) must NOT disturb the existing descriptor/transfer
/// path. The same storage that exposes a device address still round-trips
/// H2D/D2H byte-for-byte.
#[test]
#[ignore]
fn bda_does_not_disturb_transfer_path() {
    let Some(b) = backend_or_skip() else { return };
    let src: Vec<u8> = (0..=255).collect();
    let storage = b.upload_bytes(&src).expect("h2d");
    let addr = storage
        .buffer_opt()
        .expect("buffer")
        .device_address()
        .expect("bda");
    assert_ne!(addr, 0);
    let got = b.download_bytes(&storage).expect("d2h");
    assert_eq!(got, src, "BDA enablement must not disturb the transfer path");
}
