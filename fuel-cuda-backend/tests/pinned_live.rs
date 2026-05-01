//! Live-device smoke tests for [`PinnedHostStorage`].
//!
//! Gated with `#[ignore]` — run with `cargo test -- --ignored` on a
//! machine with an NVIDIA GPU + CUDA Runtime SDK installed.

use fuel_core_types::backend::HostStorage;
use fuel_core_types::{DType, HostBufferRef};
use fuel_cuda_backend::{CudaDevice, PinnedHostStorage};

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

#[test]
#[ignore]
fn pinned_zeros_f32_is_zero() {
    let Some(dev) = dev_or_skip() else { return };
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
    let Some(dev) = dev_or_skip() else { return };
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
    let Some(dev) = dev_or_skip() else { return };
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
    let Some(dev) = dev_or_skip() else { return };
    for dt in [DType::U8, DType::I32, DType::F16, DType::BF16, DType::F64] {
        let buf = PinnedHostStorage::zeros(&dev, dt, 4).expect("alloc");
        assert_eq!(buf.dtype(), dt);
        assert_eq!(buf.len(), 4);
    }
}
