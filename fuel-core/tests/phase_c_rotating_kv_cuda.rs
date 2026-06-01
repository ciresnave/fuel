//! Phase C — Op::WriteSliceRotating live-CUDA end-to-end.
//!
//! Mirrors the CPU integration tests in `phase_c_rotating_kv.rs` but
//! realizes through `realize_f32_cuda`. Validates that:
//!   - the CUDA wrapper D2H-reads the position scalar correctly,
//!   - baracuda's existing `write_slice_b*` kernels are dispatched
//!     for the first/second ring-boundary chunks,
//!   - and the final cache state matches the CPU oracle.

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_core_types::{probe::BackendId, Shape};

fn cuda_present() -> bool {
    let probe = fuel_core::probe::ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Cuda)
}

fn cuda_device() -> fuel_cuda_backend::CudaDevice {
    fuel_cuda_backend::CudaDevice::new(0)
        .expect("cuda device 0 should be available")
}

/// Within-window write on CUDA: matches the CPU `rotating_within_window`
/// oracle.
#[test]
fn rotating_within_window_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(vec![7.0_f32, 8.0], Shape::from_dims(&[1, 2]));
    let position = dest.const_u32_like(vec![1_u32], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_rotating(&src, &position, 0, 4, vec![(0, 1), (0, 2)])
        .expect("write_slice_rotating builds");

    let cuda_dev = cuda_device();
    let out = post_write.realize_f32_cuda(&cuda_dev);
    assert_eq!(out, vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Boundary split on CUDA: position 3, slab 2, modulus 4 — exercises
/// the two-chunk dispatch path (first row at index 3, wrapped row at 0).
#[test]
fn rotating_splits_across_boundary_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let dest = LazyTensor::from_f32(vec![0.0_f32; 8], Shape::from_dims(&[4, 2]), &device);
    let src = dest.const_f32_like(
        vec![10.0_f32, 11.0, 20.0, 21.0],
        Shape::from_dims(&[2, 2]),
    );
    let position = dest.const_u32_like(vec![3_u32], Shape::from_dims(&[]));
    let post_write = dest
        .write_slice_rotating(&src, &position, 0, 4, vec![(0, 2), (0, 2)])
        .expect("write_slice_rotating builds");

    let cuda_dev = cuda_device();
    let out = post_write.realize_f32_cuda(&cuda_dev);
    assert_eq!(out, vec![20.0, 21.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0]);
}

/// Mistral-style 4-step decode loop on CUDA: same shape as the CPU
/// oracle, but every rotating-write executes baracuda kernels.
#[test]
fn rotating_mistral_style_decode_loop_cuda() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }
    let device = fuel_core::Device::cpu();
    let window = 3_usize;
    let head_dim = 2_usize;
    let mut cache = LazyTensor::from_f32(
        vec![0.0_f32; window * head_dim],
        Shape::from_dims(&[window, head_dim]),
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
        let position = cache.const_u32_like(vec![step as u32], Shape::from_dims(&[]));
        cache = cache
            .write_slice_rotating(&token_t, &position, 0, window, vec![(0, 1), (0, head_dim)])
            .expect("rotating append");
    }
    let cuda_dev = cuda_device();
    let out = cache.realize_f32_cuda(&cuda_dev);
    assert_eq!(out, vec![4.0, 4.1, 2.0, 2.1, 3.0, 3.1]);
}
