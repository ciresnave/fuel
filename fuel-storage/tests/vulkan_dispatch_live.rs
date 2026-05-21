//! Live-Vulkan tests for the pipelined-executor binding-table
//! dispatch on Vulkan. Phase 7.6 step 9c Vulkan catch-up V.1.C —
//! the proof-of-life that one op (Add f32) flows end-to-end through
//! the new path on the Vulkan device.
//!
//! Requires a working Vulkan device (RTX 4070 on the dev machine
//! per the dev-environment memory). Tests are `#[ignore]` so the
//! default `cargo test` doesn't require a GPU; run with
//! `--include-ignored` for the GPU sweep.

#![cfg(feature = "vulkan")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Layout, Shape};
use fuel_storage::{
    kernel::{KernelBindingTable, OpParams},
    vulkan_dispatch::register_vulkan_kernels,
    BackendStorage, Storage,
};
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

fn download_f32(backend: &Arc<VulkanBackend>, s: &Storage) -> Vec<f32> {
    let bytes = match &s.inner {
        BackendStorage::Vulkan(v) => backend.download_bytes(v).expect("d2h"),
        _ => panic!("not on Vulkan"),
    };
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

/// Direct kernel-wrapper invocation. Skips the pipelined executor's
/// output-allocation arm and proves the dispatch wrapper + Slang
/// kernel produce correct bytes in isolation.
#[test]
#[ignore]
fn vulkan_dispatch_binary_add_f32_direct_wrapper() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let n = a_data.len();

    let a_storage = upload_f32(&backend, &a_data);
    let b_storage = upload_f32(&backend, &b_data);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc_bytes_handle");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let alts = table.lookup_alternatives(
        OpKind::AddElementwise,
        &[DType::F32, DType::F32, DType::F32],
        BackendId::Vulkan,
    );
    assert!(!alts.is_empty(), "no Vulkan AddElementwise registration");
    let kernel = alts[0].kernel;

    // Per-input layouts: contiguous rank-1 [n].
    let layout = Layout::contiguous(Shape::from_dims(&[n]));
    let layouts = vec![layout.clone(), layout.clone(), layout];

    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("kernel dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
    assert_eq!(got, expected, "Vulkan Add f32 result mismatch");
}

/// Dispatch-table presence check (no GPU required). Confirms the
/// binding-table key for `(AddElementwise, [F32, F32, F32], Vulkan)`
/// is populated after `register_vulkan_kernels` runs.
#[test]
fn vulkan_dispatch_add_f32_registered() {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let alts = table.lookup_alternatives(
        OpKind::AddElementwise,
        &[DType::F32, DType::F32, DType::F32],
        BackendId::Vulkan,
    );
    assert_eq!(
        alts.len(),
        1,
        "expected exactly 1 Vulkan AddElementwise alternative after register_vulkan_kernels, \
         got {}",
        alts.len(),
    );
}

/// Rank-2 contiguous Add: proves the kernel handles multi-dim
/// shapes through the dispatch wrapper. Strided-input correctness
/// is exercised by V.2's downstream tests once more ops + view ops
/// land; this V.1.C smoke-test sticks to contiguous shapes.
#[test]
#[ignore]
fn vulkan_dispatch_binary_add_f32_rank2_contig() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let a_storage = upload_f32(&backend, &a_data);
    let b_storage = upload_f32(&backend, &b_data);
    let out_bytes = backend.alloc_bytes_handle(6 * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;

    let layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let layouts = vec![layout.clone(), layout.clone(), layout];

    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("kernel dispatch (rank-2 contig)");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    assert_eq!(got, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}
