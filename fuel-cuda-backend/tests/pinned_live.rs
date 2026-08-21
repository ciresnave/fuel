// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-device smoke tests for [`PinnedHostStorage`].
//!
//! Gated with `#[ignore]` — run with `cargo test -- --ignored` on a
//! machine with an NVIDIA GPU + CUDA Runtime SDK installed.

use fuel_backend_contract::backend::HostStorage;
use fuel_cuda_backend::{CudaDevice, PinnedHostStorage};
use fuel_ir::{DType, HostBufferRef};

/// Acquire CUDA device 0 for a live test. GAP-224: asserts the machine-wide GPU
/// mutex is held (`require_gpu_run_lock`); GAP-157: a missing device is a FAILURE
/// (`required_ok`), not a silent skip. On a device-less box, an `--ignored` run of
/// this file now FAILS loudly (intended) rather than passing green having asserted
/// nothing.
fn dev() -> CudaDevice {
    fuel_test_support::require_gpu_run_lock();
    fuel_test_support::required_ok("CUDA device 0", CudaDevice::new(0))
}

#[test]
#[ignore]
fn pinned_zeros_f32_is_zero() {
    let dev = dev();
    let buf = PinnedHostStorage::zeros_f32(&dev, 64).expect("alloc");
    let view = buf.as_host_buffer_ref().expect("view");
    assert_eq!(view.dtype(), DType::F32);
    assert_eq!(view.len(), 64);
    match view {
        HostBufferRef::F32(s) => assert!(s.iter().all(|v| *v == 0.0)),
        _ => panic!("unexpected dtype"),
    }
}

#[test]
#[ignore]
fn pinned_write_then_read() {
    let dev = dev();
    let mut buf = PinnedHostStorage::zeros_f32(&dev, 8).expect("alloc");
    {
        let slice = buf.as_mut_slice_f32().expect("mut");
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }
    }
    match buf.as_host_buffer_ref().expect("view") {
        HostBufferRef::F32(s) => {
            assert_eq!(s, &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
#[ignore]
fn pinned_zero_length_round_trip() {
    // Baracuda 235c37e made `cuMemHostAlloc(0)` sound by returning a
    // `NonNull::dangling` sentinel (same trick stdlib uses for
    // empty-`Vec`). Derefing to `&[T]` and re-materializing through
    // `as_host_buffer_ref` on a zero-length buffer both stay sound.
    let dev = dev();
    let buf = PinnedHostStorage::zeros_f32(&dev, 0).expect("alloc");
    assert!(buf.is_empty());
    let view = buf.as_host_buffer_ref().expect("view");
    assert_eq!(view.len(), 0);
    match view {
        HostBufferRef::F32(s) => assert!(s.is_empty()),
        _ => panic!("unexpected dtype"),
    }
}

#[test]
#[ignore]
fn pinned_zeros_by_dtype() {
    let dev = dev();
    for dt in [DType::U8, DType::I32, DType::F16, DType::BF16, DType::F64] {
        let buf = PinnedHostStorage::zeros(&dev, dt, 4).expect("alloc");
        assert_eq!(buf.dtype(), dt);
        assert_eq!(buf.len(), 4);
    }
}
