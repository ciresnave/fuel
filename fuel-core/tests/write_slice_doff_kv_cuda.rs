//! Phase 1 (CapturedRun) — Op::WriteSliceDoff live-CUDA end-to-end.
//!
//! Mirrors the CPU integration tests in `write_slice_doff_kv.rs` but
//! realizes through `realize_f32_cuda`. Validates that:
//!   - the offset stays DEVICE-resident (the CUDA wrapper threads its
//!     device pointer straight to baracuda's `write_slice_*_doff`
//!     launcher — NO D2H, unlike the rotating wrapper),
//!   - the kernel reads the start device-side and writes the slab,
//!   - and the final cache state matches the CPU oracle.
//!
//! This is the production path CapturedRun builds on (Phase 2/3): the
//! captured decode graph replays at the host-updated `*offset`.

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_ir::{probe::BackendId, Shape};

fn cuda_present() -> bool {
    let probe = fuel_core::probe::ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Cuda)
}

fn cuda_device() -> fuel_cuda_backend::CudaDevice {
    fuel_cuda_backend::CudaDevice::new(0)
        .expect("cuda device 0 should be available")
}

/// Basic append at device offset 1 on CUDA — matches the CPU
/// `doff_writes_at_device_offset` oracle.
#[test]
fn doff_writes_at_device_offset_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_i64_like(vec![1_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)])
        .expect("write_slice_doff builds");

    let cuda_dev = cuda_device();
    let out = post_write.realize_f32_cuda(&cuda_dev);
    assert_eq!(out, vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Offset 0 → leading row (no wrap) on CUDA.
#[test]
fn doff_offset_zero_writes_leading_row_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let offset = dest.const_i64_like(vec![0_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, 0, vec![(0, 1), (0, 2)])
        .expect("write_slice_doff builds");

    let cuda_dev = cuda_device();
    let out = post_write.realize_f32_cuda(&cuda_dev);
    assert_eq!(out, vec![7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Capacity-4 decode loop on CUDA appending at the live `cached_len`
/// offset — the DecodeSession/CapturedRun access pattern, every append
/// running the baracuda `_doff` kernel with a device-resident start.
#[test]
fn doff_decode_loop_appends_at_cached_len_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let max_seq = 4_usize;
    let head_dim = 2_usize;
    let mut cache = LazyTensor::from_f32(
        vec![0.0_f32; max_seq * head_dim],
        Shape::from_dims(&[max_seq, head_dim]),
        &device,
    );
    let tokens = [
        vec![1.0_f32, 1.1],
        vec![2.0_f32, 2.1],
        vec![3.0_f32, 3.1],
        vec![4.0_f32, 4.1],
    ];
    for (step, token) in tokens.iter().enumerate() {
        let token_t = cache.const_f32_like(token.clone(), Shape::from_dims(&[1, head_dim]));
        let offset = cache.const_i64_like(vec![step as i64], Shape::from_dims(&[]));
        cache = cache
            .write_slice_doff(&token_t, &offset, 0, vec![(0, 1), (0, head_dim)])
            .expect("doff append");
    }
    let cuda_dev = cuda_device();
    let out = cache.realize_f32_cuda(&cuda_dev);
    assert_eq!(out, vec![1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1]);
}

/// Interior write on a non-leading axis on CUDA: offset 2 on axis 1.
#[test]
fn doff_writes_on_non_leading_axis_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 10], Shape::from_dims(&[2, 5]), &device);
    let src = dest.const_f32_like(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
    );
    let offset = dest.const_i64_like(vec![2_i64], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_doff(&src, &offset, /* axis */ 1, vec![(0, 2), (0, 2)])
        .expect("write_slice_doff builds");

    let cuda_dev = cuda_device();
    let out = post_write.realize_f32_cuda(&cuda_dev);
    assert_eq!(
        out,
        vec![
            0.0, 0.0, 1.0, 2.0, 0.0,
            0.0, 0.0, 3.0, 4.0, 0.0,
        ]
    );
}
