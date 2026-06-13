//! Live-Vulkan tests for Op::WriteSliceRotating (Phase C).
//!
//! Mirrors the WriteSlice live-Vulkan tests in
//! `vulkan_dispatch_live.rs`: directly invokes the dispatch table's
//! kernel wrapper rather than going through PipelinedExecutor (which
//! Vulkan-resident `LazyTensor::from_f32` doesn't support yet — see
//! `VulkanBackendDevice::storage_from_host_buffer_owned_dyn`).
//!
//! Requires a working Vulkan device. Tests are `#[ignore]` so the
//! default `cargo test` stays green on CI without a GPU; run with
//! `--include-ignored` for the GPU sweep.

#![cfg(feature = "vulkan")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Layout, Shape};
use fuel_dispatch::{kernel::{KernelBindingTable, OpParams}, vulkan_dispatch::register_vulkan_kernels};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::VulkanBackend;

fn backend_or_skip() -> Option<Arc<VulkanBackend>> {
    VulkanBackend::new().ok().map(Arc::new)
}

fn upload_f32(backend: &Arc<VulkanBackend>, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let vk_bytes = backend
        .upload_bytes_handle(bytes)
        .expect("vulkan upload_bytes_handle");
    Storage::new(BackendStorage::Vulkan(vk_bytes), DType::F32)
}

fn upload_u32(backend: &Arc<VulkanBackend>, host: &[u32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let vk_bytes = backend
        .upload_bytes_handle(bytes)
        .expect("vulkan upload_bytes_handle");
    Storage::new(BackendStorage::Vulkan(vk_bytes), DType::U32)
}

fn download_f32(backend: &Arc<VulkanBackend>, s: &Storage) -> Vec<f32> {
    let bytes = match &s.inner {
        BackendStorage::Vulkan(v) => backend.download_bytes(v).expect("d2h"),
        _ => panic!("not on Vulkan"),
    };
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn lookup_rotating_kernel(table: &KernelBindingTable) -> fuel_dispatch::kernel::KernelRef {
    table
        .lookup_alternatives(
            OpKind::WriteSliceRotating,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel
}

/// Within-window write on Vulkan via the binding-table dispatch.
#[test]
#[ignore]
fn vulkan_dispatch_write_slice_rotating_within_window() {
    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    // dest [4, 2] zero-initialized; write [7, 8] at position 1.
    let dst_init = vec![0.0_f32; 8];
    let dst_storage = upload_f32(&backend, &dst_init);
    let src = vec![7.0_f32, 8.0];
    let src_storage = upload_f32(&backend, &src);
    let position = vec![1_u32];
    let pos_storage = upload_u32(&backend, &position);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let pos_arc = Arc::new(RwLock::new(pos_storage));
    let dst_arc = Arc::new(RwLock::new(dst_storage));

    let kernel = lookup_rotating_kernel(&table);
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[1, 2])),
        Layout::contiguous(Shape::from_dims(&[])),
        Layout::contiguous(Shape::from_dims(&[4, 2])),
    ];
    kernel(
        &[Arc::clone(&src_arc), Arc::clone(&pos_arc)],
        &mut [Arc::clone(&dst_arc)],
        &layouts,
        &OpParams::WriteSliceRotating {
            dest_shape: vec![4, 2],
            axis: 0,
            modulus: 4,
            ranges: vec![(0, 1), (0, 2)],
        },
    ).expect("write_slice_rotating dispatch");

    let got = download_f32(&backend, &dst_arc.read().unwrap());
    assert_eq!(got, vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Boundary split: position 3, slab 2, modulus 4 — exercises the
/// two-chunk dispatch through slot_copy_to_new_handle.
#[test]
#[ignore]
fn vulkan_dispatch_write_slice_rotating_splits_across_boundary() {
    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let dst_init = vec![0.0_f32; 8];
    let dst_storage = upload_f32(&backend, &dst_init);
    let src = vec![10.0_f32, 11.0, 20.0, 21.0];
    let src_storage = upload_f32(&backend, &src);
    let position = vec![3_u32];
    let pos_storage = upload_u32(&backend, &position);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let pos_arc = Arc::new(RwLock::new(pos_storage));
    let dst_arc = Arc::new(RwLock::new(dst_storage));

    let kernel = lookup_rotating_kernel(&table);
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[2, 2])),
        Layout::contiguous(Shape::from_dims(&[])),
        Layout::contiguous(Shape::from_dims(&[4, 2])),
    ];
    kernel(
        &[Arc::clone(&src_arc), Arc::clone(&pos_arc)],
        &mut [Arc::clone(&dst_arc)],
        &layouts,
        &OpParams::WriteSliceRotating {
            dest_shape: vec![4, 2],
            axis: 0,
            modulus: 4,
            ranges: vec![(0, 2), (0, 2)],
        },
    ).expect("write_slice_rotating dispatch");

    let got = download_f32(&backend, &dst_arc.read().unwrap());
    // row 3 = (10, 11), wrapped to row 0 = (20, 21).
    assert_eq!(got, vec![20.0, 21.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0]);
}

/// Position wraps modulo modulus: position 4 with modulus 4 → start 0.
#[test]
#[ignore]
fn vulkan_dispatch_write_slice_rotating_wraps_position() {
    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let dst_init = vec![0.0_f32; 8];
    let dst_storage = upload_f32(&backend, &dst_init);
    let src = vec![7.0_f32, 8.0];
    let src_storage = upload_f32(&backend, &src);
    let position = vec![4_u32]; // == modulus
    let pos_storage = upload_u32(&backend, &position);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let pos_arc = Arc::new(RwLock::new(pos_storage));
    let dst_arc = Arc::new(RwLock::new(dst_storage));

    let kernel = lookup_rotating_kernel(&table);
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[1, 2])),
        Layout::contiguous(Shape::from_dims(&[])),
        Layout::contiguous(Shape::from_dims(&[4, 2])),
    ];
    kernel(
        &[Arc::clone(&src_arc), Arc::clone(&pos_arc)],
        &mut [Arc::clone(&dst_arc)],
        &layouts,
        &OpParams::WriteSliceRotating {
            dest_shape: vec![4, 2],
            axis: 0,
            modulus: 4,
            ranges: vec![(0, 1), (0, 2)],
        },
    ).expect("write_slice_rotating dispatch");

    let got = download_f32(&backend, &dst_arc.read().unwrap());
    // position 4 % 4 = 0 → writes to row 0.
    assert_eq!(got, vec![7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}
