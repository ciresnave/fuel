//! Live-device tests for the Phase 7.5 A4 substrate methods on
//! [`CudaStorageBytes`]. Gated `#[ignore]` — run with
//! `cargo test --features cuda -- --ignored` on a machine with an
//! NVIDIA GPU + CUDA Runtime SDK installed.

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

/// Smoke: alloc(byte_count) produces a zero-initialized buffer
/// readable back via to_cpu_bytes.
#[test]
#[ignore]
fn alloc_then_read_is_zero() {
    let Some(dev) = dev_or_skip() else { return };
    let storage = CudaStorageBytes::alloc(&dev, 32).expect("alloc");
    assert_eq!(storage.len_bytes(), 32);
    let bytes = storage.to_cpu_bytes().expect("d2h");
    assert_eq!(bytes.len(), 32);
    assert!(bytes.iter().all(|&b| b == 0));
}

/// H2D + D2H roundtrip: build from a host slice, read back, expect
/// exact byte equality.
#[test]
#[ignore]
fn h2d_d2h_roundtrip_preserves_bytes() {
    let Some(dev) = dev_or_skip() else { return };
    let src: Vec<u8> = (0..=255).collect();
    let storage = CudaStorageBytes::from_cpu_bytes(&dev, &src).expect("h2d");
    assert_eq!(storage.len_bytes(), src.len());
    let got = storage.to_cpu_bytes().expect("d2h");
    assert_eq!(got, src);
}

/// Zero-length transfer is sound: alloc(0), from_cpu_bytes(empty),
/// and to_cpu_bytes on an empty storage all succeed and return
/// empty results.
#[test]
#[ignore]
fn zero_length_transfers_round_trip() {
    let Some(dev) = dev_or_skip() else { return };
    let storage = CudaStorageBytes::alloc(&dev, 0).expect("alloc 0");
    assert_eq!(storage.len_bytes(), 0);
    let bytes = storage.to_cpu_bytes().expect("d2h 0");
    assert!(bytes.is_empty());

    let from_empty = CudaStorageBytes::from_cpu_bytes(&dev, &[]).expect("h2d 0");
    assert_eq!(from_empty.len_bytes(), 0);
}

/// Larger transfer: 1 MiB pattern of [0..=255] tiled, exercise the
/// non-trivial copy path.
#[test]
#[ignore]
fn one_mib_roundtrip_preserves_bytes() {
    let Some(dev) = dev_or_skip() else { return };
    let src: Vec<u8> = (0..1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let storage = CudaStorageBytes::from_cpu_bytes(&dev, &src).expect("h2d");
    let got = storage.to_cpu_bytes().expect("d2h");
    assert_eq!(got, src);
}
