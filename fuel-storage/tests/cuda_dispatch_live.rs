//! Live-CUDA integration test for the Phase 7.5 first CUDA op
//! through the unified binding table.
//!
//! Runs the full end-to-end path: register_cuda_kernels →
//! KernelBindingTable::lookup → wrapper invocation → kernel launch →
//! D2H read-back. Gated `#[ignore]` because it requires an NVIDIA
//! GPU + CUDA Runtime SDK; invoke explicitly:
//!
//! ```sh
//! cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_storage::{
    dispatch::{register_cuda_kernels, register_cpu_kernels},
    kernel::{KernelBindingTable, OpParams},
    BackendStorage, Storage,
};

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

fn build_storage_cuda(dev: &CudaDevice, src_f32: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(src_f32);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

/// End-to-end: register the CUDA wrapper, look it up via the
/// binding table, invoke it on two F32 CUDA inputs, read back via
/// D2H, assert elementwise sum.
#[test]
#[ignore]
fn add_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    // Build the binding table — register both CPU + CUDA so the
    // table is populated for both, then look up the CUDA entry.
    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let lhs = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0]);
    let rhs = build_storage_cuda(&dev, &[10.0_f32, 20.0, 30.0, 40.0]);
    // Pre-allocated output: 16 bytes of zeros on the same device.
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::AddElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (AddElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[11.0_f32, 22.0, 33.0, 44.0]);
}

/// End-to-end: same as the AddElementwise test but for SubElementwise.
/// Verifies the second CUDA op of Tier 1 binary fanout reaches its
/// kernel through the binding table and produces the expected
/// elementwise difference (lhs - rhs).
#[test]
#[ignore]
fn sub_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let lhs = build_storage_cuda(&dev, &[10.0_f32, 20.0, 30.0, 40.0]);
    let rhs = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SubElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (SubElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[9.0_f32, 18.0, 27.0, 36.0]);
}

/// End-to-end: MulElementwise F32 through the binding table.
#[test]
#[ignore]
fn mul_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let lhs = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0]);
    let rhs = build_storage_cuda(&dev, &[10.0_f32, 20.0, 30.0, 40.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MulElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (MulElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[10.0_f32, 40.0, 90.0, 160.0]);
}

/// End-to-end: DivElementwise F32 through the binding table.
#[test]
#[ignore]
fn div_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let lhs = build_storage_cuda(&dev, &[10.0_f32, 40.0, 90.0, 160.0]);
    let rhs = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::DivElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (DivElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[10.0_f32, 20.0, 30.0, 40.0]);
}

/// End-to-end: MaximumElementwise F32 through the binding table.
#[test]
#[ignore]
fn maximum_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let lhs = build_storage_cuda(&dev, &[1.0_f32, 5.0, -2.0, 4.0]);
    let rhs = build_storage_cuda(&dev, &[3.0_f32, 2.0, 0.0, 4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MaximumElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (MaximumElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[3.0_f32, 5.0, 0.0, 4.0]);
}

/// End-to-end: MinimumElementwise F32 through the binding table.
#[test]
#[ignore]
fn minimum_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let lhs = build_storage_cuda(&dev, &[1.0_f32, 5.0, -2.0, 4.0]);
    let rhs = build_storage_cuda(&dev, &[3.0_f32, 2.0, 0.0, 4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MinimumElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (MinimumElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[1.0_f32, 2.0, -2.0, 4.0]);
}

/// End-to-end: ReluElementwise F32 through the binding table.
/// First unary op of Tier 1 unary fanout — exercises the shared
/// `unary_elementwise_f32` helper in fuel-cuda-backend::byte_kernels.
#[test]
#[ignore]
fn relu_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, &[-2.0_f32, -0.5, 0.0, 1.5, 3.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 20).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ReluElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (ReluElementwise, F32, Cuda)");

    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[0.0_f32, 0.0, 0.0, 1.5, 3.0]);
}

/// End-to-end: NegElementwise F32 through the binding table.
#[test]
#[ignore]
fn neg_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, &[1.0_f32, -2.0, 3.0, -4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::NegElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (NegElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[-1.0_f32, 2.0, -3.0, 4.0]);
}

/// End-to-end: SqrElementwise F32 through the binding table.
#[test]
#[ignore]
fn sqr_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SqrElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (SqrElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[1.0_f32, 4.0, 9.0, 16.0]);
}

/// End-to-end: SqrtElementwise F32 through the binding table.
#[test]
#[ignore]
fn sqrt_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, &[1.0_f32, 4.0, 9.0, 16.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SqrtElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (SqrtElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[1.0_f32, 2.0, 3.0, 4.0]);
}

/// Smoke: looking up a binding before registration returns a clear
/// `NoBackendForOp` error rather than panicking. Doesn't need a
/// live GPU since we never actually invoke a kernel.
#[test]
fn lookup_without_registration_errors_clean() {
    let table = KernelBindingTable::new();
    let r = table.lookup(OpKind::AddElementwise, DType::F32, BackendId::Cuda);
    assert!(r.is_err(), "expected NoBackendForOp error");
}
