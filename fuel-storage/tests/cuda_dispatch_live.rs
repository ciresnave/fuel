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

/// Element-wise approximate equality for transcendental unary tests.
/// CUDA's PTX intrinsics for tanh/exp/log/sin/cos/sigmoid/silu/gelu
/// are not bit-exact with the host `f32::tanh` etc., so the tests
/// compare against a small epsilon (1e-5 is comfortable for these
/// kernels at the magnitudes tested).
fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len(), "len mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= eps,
            "idx {i}: |{a} - {e}| > {eps}",
        );
    }
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

/// End-to-end: RecipElementwise F32 through the binding table.
#[test]
#[ignore]
fn recip_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, &[1.0_f32, 2.0, 4.0, 8.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::RecipElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (RecipElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[1.0_f32, 0.5, 0.25, 0.125]);
}

/// End-to-end: AbsElementwise F32 through the binding table.
#[test]
#[ignore]
fn abs_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, &[-1.0_f32, -2.0, 3.0, -4.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 16).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::AbsElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (AbsElementwise, F32, Cuda)");

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

/// End-to-end: TanhElementwise F32 through the binding table.
/// Compared with an epsilon since CUDA's `tanhg` PTX intrinsic
/// isn't bit-exact with the host `f32::tanh`.
#[test]
#[ignore]
fn tanh_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::TanhElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (TanhElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| x.tanh()).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: ExpElementwise F32 through the binding table.
#[test]
#[ignore]
fn exp_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [-1.0_f32, 0.0, 0.5, 1.0, 2.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ExpElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (ExpElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| x.exp()).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: LogElementwise F32 through the binding table.
#[test]
#[ignore]
fn log_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [0.5_f32, 1.0, 2.0, std::f32::consts::E, 10.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::LogElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (LogElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| x.ln()).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: SinElementwise F32 through the binding table.
#[test]
#[ignore]
fn sin_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [
        0.0_f32,
        std::f32::consts::FRAC_PI_6,
        std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
    ];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SinElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (SinElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| x.sin()).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: CosElementwise F32 through the binding table.
#[test]
#[ignore]
fn cos_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [
        0.0_f32,
        std::f32::consts::FRAC_PI_6,
        std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
    ];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::CosElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (CosElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| x.cos()).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: SigmoidElementwise F32 through the binding table.
#[test]
#[ignore]
fn sigmoid_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SigmoidElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (SigmoidElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: SiluElementwise F32 through the binding table.
#[test]
#[ignore]
fn silu_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SiluElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (SiluElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = xs.iter().map(|x| x / (1.0 + (-x).exp())).collect();
    assert_close(host_f32, &expected, 1e-5);
}

/// End-to-end: GeluElementwise F32 through the binding table
/// (tanh approximation, matching `OpKind::GeluElementwise` semantics
/// and CPU `gelu_f32`).
#[test]
#[ignore]
fn gelu_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::GeluElementwise, DType::F32, BackendId::Cuda)
        .expect("lookup (GeluElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const COEFF: f32 = 0.797_884_56;
    let expected: Vec<f32> = xs
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (COEFF * (x + 0.044_715 * x * x * x)).tanh()))
        .collect();
    assert_close(host_f32, &expected, 1e-5);
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
