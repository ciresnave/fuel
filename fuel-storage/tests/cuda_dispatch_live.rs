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

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Layout, Shape};
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

/// Build a 3-element layouts slice `[lhs, rhs, output]` for binary-op
/// direct-invocation tests where all three tensors share the same
/// contiguous shape. The wrapper reads `layouts[0]` and `layouts[1]`
/// to decide fast vs strided path; the output layout is unused by the
/// wrapper today (it allocates from `lhs_layout.shape()`) but must be
/// present to satisfy the binary-input layouts contract.
fn binary_layouts(shape: &[usize]) -> Vec<Layout> {
    let l = Layout::contiguous(Shape::from(shape.to_vec()));
    vec![l.clone(), l.clone(), l]
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
        .lookup(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (AddElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[4]),
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
        .lookup(OpKind::SubElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SubElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[4]),
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
        .lookup(OpKind::MulElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MulElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[4]),
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
        .lookup(OpKind::DivElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (DivElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[4]),
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
        .lookup(OpKind::MaximumElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MaximumElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[4]),
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
        .lookup(OpKind::MinimumElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MinimumElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[4]),
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

/// PR 2 broadcast validation. The binary F32 wrapper now reads input
/// layouts and routes broadcast (non-contiguous) inputs through the
/// PTX kernel's strided path instead of demanding equal byte lengths.
/// Direct-invocation test: `lhs` is `[B, N, 1]` actual storage with a
/// broadcast layout to `[B, N, M]` (stride [N, 1, 0], same start
/// offset 0); `rhs` is `[B, N, M]` contiguous. Result is element-wise
/// `lhs[b, n, 0] - rhs[b, n, m]` walked over the broadcasted shape.
#[test]
#[ignore]
fn sub_elementwise_f32_broadcast_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    // Shape: B=2, N=3, M=4. lhs is [B, N, 1] with 6 storage elements
    // that broadcast across M; rhs is [B, N, M] contiguous with 24
    // elements; output is [B, N, M] (24 elements).
    const B: usize = 2;
    const N: usize = 3;
    const M: usize = 4;
    let lhs_storage: Vec<f32> = (0..(B * N)).map(|i| (i + 1) as f32).collect();
    let rhs_storage: Vec<f32> = (0..(B * N * M)).map(|i| (i + 1) as f32 * 0.5).collect();

    let lhs = build_storage_cuda(&dev, &lhs_storage);
    let rhs = build_storage_cuda(&dev, &rhs_storage);
    let out_bytes = CudaStorageBytes::alloc(&dev, B * N * M * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    // lhs layout: shape [B, N, M] but strides [N, 1, 0] — the trailing
    // M axis broadcasts (stride 0 reads the same element repeatedly).
    // Built by chaining contiguous([B,N,1]) → broadcast_as([B,N,M]).
    let lhs_layout = Layout::contiguous(Shape::from(vec![B, N, 1]))
        .broadcast_as(Shape::from(vec![B, N, M]))
        .expect("broadcast_as");
    let rhs_layout = Layout::contiguous(Shape::from(vec![B, N, M]));
    let out_layout = Layout::contiguous(Shape::from(vec![B, N, M]));
    let layouts = vec![lhs_layout, rhs_layout, out_layout];

    let kernel = table
        .lookup(OpKind::SubElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SubElementwise, F32, Cuda)");

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &layouts,
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);

    // CPU reference — `lhs[b, n, m] = lhs_storage[b*N + n]` (broadcast),
    // `rhs[b, n, m] = rhs_storage[b*N*M + n*M + m]` (contiguous).
    let mut expected = Vec::with_capacity(B * N * M);
    for b in 0..B {
        for n in 0..N {
            for m in 0..M {
                let l = lhs_storage[b * N + n];
                let r = rhs_storage[b * N * M + n * M + m];
                expected.push(l - r);
            }
        }
    }
    assert_eq!(host_f32, expected.as_slice());
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
        .lookup(OpKind::ReluElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (ReluElementwise, F32, Cuda)");

    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
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
        .lookup(OpKind::NegElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (NegElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::SqrElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SqrElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::SqrtElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SqrtElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::RecipElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (RecipElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::AbsElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (AbsElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::TanhElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (TanhElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::ExpElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (ExpElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::LogElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (LogElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::SinElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SinElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::CosElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (CosElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::SigmoidElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SigmoidElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::SiluElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SiluElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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
        .lookup(OpKind::GeluElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (GeluElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
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

/// End-to-end: StepElementwise F32 through the binding table.
/// Heaviside step (1.0 where x > 0, else 0.0). Exercises the new
/// `ustep_f32` PTX kernel introduced in `fuel-cuda-kernels`.
#[test]
#[ignore]
fn step_elementwise_f32_through_binding_table() {
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
        .lookup(OpKind::StepElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (StepElementwise, F32, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    // x > 0 ? 1 : 0 — note 0.0 maps to 0.0 (strict inequality).
    assert_eq!(host_f32, &[0.0_f32, 0.0, 0.0, 1.0, 1.0]);
}

/// End-to-end: SumReduce F32 through the binding table. First
/// reduction op of Tier 1 — exercises the shared `reduce_f32` helper
/// in fuel-cuda-backend::byte_kernels with `fast_sum_f32`. Input
/// `[2, 3]` of `[1..6]`, reduce axis `[1]`, expected `[6, 15]`.
#[test]
#[ignore]
fn sum_reduce_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = build_storage_cuda(&dev, &xs);
    // Output: 2 elements (the kept dim 0 is size 2).
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::SumReduce, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (SumReduce, F32, Cuda)");

    let params = OpParams::Reduce {
        dims: vec![1],
        keepdim: false,
    };
    let in_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[6.0_f32, 15.0]);
}

/// End-to-end: MaxReduce F32 through the binding table. Reuses the
/// shared `reduce_f32` helper with `fast_max_f32`. Input `[2, 3]`
/// of `[1, 5, 2, 4, 3, 6]`, reduce axis `[1]`, expected `[5, 6]`.
#[test]
#[ignore]
fn max_reduce_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 5.0, 2.0, 4.0, 3.0, 6.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MaxReduce, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MaxReduce, F32, Cuda)");

    let params = OpParams::Reduce {
        dims: vec![1],
        keepdim: false,
    };
    let in_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[5.0_f32, 6.0]);
}

/// End-to-end: MinReduce F32 through the binding table. Reuses the
/// shared `reduce_f32` helper with `fast_min_f32`. Input `[2, 3]`
/// of `[1, 5, 2, 4, 3, 6]`, reduce axis `[1]`, expected `[1, 3]`.
#[test]
#[ignore]
fn min_reduce_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 5.0, 2.0, 4.0, 3.0, 6.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MinReduce, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MinReduce, F32, Cuda)");

    let params = OpParams::Reduce {
        dims: vec![1],
        keepdim: false,
    };
    let in_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[1.0_f32, 3.0]);
}

/// End-to-end: MeanReduce F32 through the binding table. Composed
/// op: `fast_sum_f32` followed by `affine_f32` scaling by
/// `1/divisor`. Input `[2, 3]` of `[1..6]`, reduce axis `[1]`,
/// expected sums `[6, 15]` divided by 3 → `[2, 5]`. Uses
/// `assert_close` because the two-launch composition introduces
/// the affine kernel's `x * mul + add` rounding (single ULP at
/// these magnitudes).
#[test]
#[ignore]
fn mean_reduce_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MeanReduce, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MeanReduce, F32, Cuda)");

    let params = OpParams::Reduce {
        dims: vec![1],
        keepdim: false,
    };
    let in_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_close(host_f32, &[2.0_f32, 5.0], 1e-5);
}

/// End-to-end: rank-2 MatMul F32 through the binding table. First
/// non-PTX, non-element-wise op — the underlying call is cuBLAS
/// `gemm_strided_batched_ex` with `batch_count = 1`. lhs `[2, 3]` of
/// `[1..6]` @ rhs `[3, 2]` of `[1..6]` → `[2, 2]` = `[[22, 28], [49, 64]]`.
/// `assert_close` with `eps = 1e-4` because cuBLAS isn't bit-exact
/// with naive CPU matmul.
#[test]
#[ignore]
fn matmul_f32_rank2_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    // lhs [2,3] = [[1,2,3],[4,5,6]], rhs [3,2] = [[1,2],[3,4],[5,6]].
    let lhs = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = build_storage_cuda(&dev, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MatMul, F32, Cuda)");

    let params = OpParams::Matmul {
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        m: 2,
        n: 2,
        k: 3,
    };

    kernel(&[lhs_arc.clone(), rhs_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    // Expected:
    //   [1*1+2*3+3*5, 1*2+2*4+3*6,
    //    4*1+5*3+6*5, 4*2+5*4+6*6]
    // = [22, 28, 49, 64].
    assert_close(host_f32, &[22.0_f32, 28.0, 49.0, 64.0], 1e-4);
}

/// End-to-end: equal-batch MatMul F32 through the binding table.
/// Exercises the cuBLAS `batch_count > 1` code path. lhs `[2, 2, 3]`
/// (two distinct 2×3 matrices) @ rhs `[2, 3, 2]` (two distinct 3×2
/// matrices) → `[2, 2, 2]`. Sets up batch 0 to use the same
/// matrices as the rank-2 test (expected `[22, 28, 49, 64]`) and
/// batch 1 to use all-ones (expected `[3, 3, 3, 3]`).
#[test]
#[ignore]
fn matmul_f32_batched_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    // batch 0: rank-2 inputs from above; batch 1: all ones.
    let lhs_data = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let rhs_data = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let lhs = build_storage_cuda(&dev, &lhs_data);
    let rhs = build_storage_cuda(&dev, &rhs_data);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 2 * 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MatMul, F32, Cuda)");

    let params = OpParams::Matmul {
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
        m: 2,
        n: 2,
        k: 3,
    };

    kernel(&[lhs_arc.clone(), rhs_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_close(
        host_f32,
        &[22.0_f32, 28.0, 49.0, 64.0, 3.0, 3.0, 3.0, 3.0],
        1e-4,
    );
}

/// End-to-end: GQA-divisible MatMul F32 through the binding table.
/// Exercises the per-batch-loop GQA path. lhs `[4, 2, 3]` (4 lhs
/// batches) @ rhs `[2, 3, 2]` (2 rhs batches) → `[4, 2, 2]`, with
/// `n_rep = [2]` (each rhs batch shared by 2 consecutive lhs
/// batches). lhs batches 0,1 share rhs batch 0; lhs batches 2,3
/// share rhs batch 1. Verifies against a CPU reference for all 4
/// lhs batches.
#[test]
#[ignore]
fn matmul_f32_gqa_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    // 4 distinct 2×3 lhs matrices, 2 distinct 3×2 rhs matrices.
    let lhs_data: Vec<f32> = (0..(4 * 2 * 3)).map(|i| i as f32).collect();
    let rhs_data: Vec<f32> = (0..(2 * 3 * 2)).map(|i| (i + 1) as f32).collect();
    let lhs = build_storage_cuda(&dev, &lhs_data);
    let rhs = build_storage_cuda(&dev, &rhs_data);
    let out_bytes = CudaStorageBytes::alloc(&dev, 4 * 2 * 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (MatMul, F32, Cuda)");

    let params = OpParams::Matmul {
        lhs_batch_dims: vec![4],
        rhs_batch_dims: vec![2],
        m: 2,
        n: 2,
        k: 3,
    };

    kernel(&[lhs_arc.clone(), rhs_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    // CPU reference: per-lhs-batch, with rhs_batch = lhs_batch / 2.
    let mut expected = vec![0.0_f32; 4 * 2 * 2];
    for b in 0..4_usize {
        let r = b / 2;
        let lhs_off = b * (2 * 3);
        let rhs_off = r * (3 * 2);
        let out_off = b * (2 * 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0_f32;
                for kk in 0..3 {
                    acc += lhs_data[lhs_off + i * 3 + kk]
                        * rhs_data[rhs_off + kk * 2 + j];
                }
                expected[out_off + i * 2 + j] = acc;
            }
        }
    }

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_close(host_f32, &expected, 1e-4);
}

/// End-to-end: Affine F32 through the binding table. Element-wise
/// `y = mul * x + add` via the `affine_f32` PTX kernel — also the
/// kernel that backs `Op::AddScalar` (mul=1) and `Op::MulScalar`
/// (add=0) at the graph layer. Inputs `[1, 2, 3, 4]`, mul=2, add=10
/// → `[12, 14, 16, 18]`.
#[test]
#[ignore]
fn affine_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 2.0, 3.0, 4.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, (xs.len() * 4) as usize)
        .expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::Affine, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (Affine, F32, Cuda)");

    let params = OpParams::Affine { mul: 2.0, add: 10.0 };

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_close(host_f32, &[12.0_f32, 14.0, 16.0, 18.0], 1e-5);
}

/// Smoke: looking up a binding before registration returns a clear
/// `NoBackendForOp` error rather than panicking. Doesn't need a
/// live GPU since we never actually invoke a kernel.
#[test]
fn lookup_without_registration_errors_clean() {
    let table = KernelBindingTable::new();
    let r = table.lookup(
        OpKind::AddElementwise,
        &[DType::F32, DType::F32, DType::F32],
        BackendId::Cuda,
    );
    assert!(r.is_err(), "expected NoBackendForOp error");
}

// =============================================================================
// Cast — first op through the unified path with input dtype != output dtype.
// =============================================================================

/// Helper: build a Storage on CUDA from raw host bytes + dtype.
/// Lets cast tests upload BF16/F16/U8/etc. without a per-dtype f32
/// adapter.
fn build_storage_cuda_from_bytes(dev: &CudaDevice, bytes: &[u8], dtype: DType) -> Storage {
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), dtype)
}

/// Helper: build a CUDA-resident output Storage of the requested
/// dtype, sized for `elem_count` elements. Used to receive cast
/// results.
fn build_output_cuda(dev: &CudaDevice, dtype: DType, elem_count: usize) -> Storage {
    let bytes = elem_count * dtype.size_in_bytes();
    let out_bytes = CudaStorageBytes::alloc(dev, bytes).expect("out alloc");
    Storage::new(BackendStorage::Cuda(out_bytes), dtype)
}

/// Run a Cast through the binding table from `src` (dtype `src_dt`)
/// to `dst_dt`, returning the device output's host bytes.
fn run_cast(
    table: &KernelBindingTable,
    dev: &CudaDevice,
    src_bytes: &[u8],
    src_dt: DType,
    dst_dt: DType,
    elem_count: usize,
) -> Vec<u8> {
    let src = build_storage_cuda_from_bytes(dev, src_bytes, src_dt);
    let out = build_output_cuda(dev, dst_dt, elem_count);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = table
        .lookup(OpKind::Cast, &[src_dt, dst_dt], BackendId::Cuda)
        .unwrap_or_else(|e| panic!("lookup (Cast, [{src_dt:?}, {dst_dt:?}], Cuda): {e:?}"));
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Cast,
    )
    .expect("cast kernel call");
    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    c.to_cpu_bytes().expect("d2h")
}

/// F32 → BF16 round-trip on small integer values. Integers ≤ 256
/// are bit-exact in BF16 (they fit the 7-bit mantissa with one bit
/// of headroom for the implicit leading 1), so the cast is lossless
/// here and the BF16 → F32 path back to the host should reproduce
/// the originals exactly.
#[test]
#[ignore]
fn cast_f32_to_bf16_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src_f32: &[f32] = &[1.0, 2.0, 3.0, 4.0, -1.5, 0.0, 100.0, 256.0];
    let src_bytes: &[u8] = bytemuck::cast_slice(src_f32);
    let host = run_cast(&table, &dev, src_bytes, DType::F32, DType::BF16, src_f32.len());

    let bf16_bits: &[u16] = bytemuck::cast_slice(&host);
    let decoded: Vec<f32> = bf16_bits.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
    assert_close(&decoded, src_f32, 0.0);
}

/// BF16 → F32 round-trip — inverse direction of the prior test.
/// Constructs BF16 source bytes on the host and verifies the
/// up-converted F32 matches.
#[test]
#[ignore]
fn cast_bf16_to_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let values: &[f32] = &[1.0, 2.0, 3.0, -1.5, 0.0];
    let bf16_bits: Vec<u16> = values.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
    let src_bytes: &[u8] = bytemuck::cast_slice(&bf16_bits);
    let host = run_cast(&table, &dev, src_bytes, DType::BF16, DType::F32, values.len());

    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_close(host_f32, values, 0.0);
}

/// F32 → F16 round-trip. Like BF16, small ints are bit-exact in F16
/// (10-bit mantissa). 1024 is the largest integer with bit-exact
/// representation; stay well below that.
#[test]
#[ignore]
fn cast_f32_to_f16_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src_f32: &[f32] = &[1.0, 2.0, 3.0, 4.0, -1.5, 0.0, 100.0, 256.0];
    let src_bytes: &[u8] = bytemuck::cast_slice(src_f32);
    let host = run_cast(&table, &dev, src_bytes, DType::F32, DType::F16, src_f32.len());

    let f16_bits: &[u16] = bytemuck::cast_slice(&host);
    let decoded: Vec<f32> = f16_bits.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    assert_close(&decoded, src_f32, 0.0);
}

/// F32 → F64 widening cast. Lossless; expect bit-exact outputs.
#[test]
#[ignore]
fn cast_f32_to_f64_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src_f32: &[f32] = &[1.0, 2.0, 3.5, -0.25, 1e6, 1e-6];
    let src_bytes: &[u8] = bytemuck::cast_slice(src_f32);
    let host = run_cast(&table, &dev, src_bytes, DType::F32, DType::F64, src_f32.len());

    let host_f64: &[f64] = bytemuck::cast_slice(&host);
    let expected: Vec<f64> = src_f32.iter().map(|&v| v as f64).collect();
    assert_eq!(host_f64, expected.as_slice());
}

/// U8 → F32 widening cast. Bit-exact; verifies the integer-source
/// path through the unified binding table.
#[test]
#[ignore]
fn cast_u8_to_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src_u8: &[u8] = &[0, 1, 2, 127, 128, 255];
    let host = run_cast(&table, &dev, src_u8, DType::U8, DType::F32, src_u8.len());

    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: Vec<f32> = src_u8.iter().map(|&v| v as f32).collect();
    assert_eq!(host_f32, expected.as_slice());
}

/// F32 → I64 truncating cast. Verifies negative + positive values
/// truncate toward zero (or the platform's standard direction; the
/// kernel uses C `static_cast` semantics which is truncation).
#[test]
#[ignore]
fn cast_f32_to_i64_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let src_f32: &[f32] = &[0.0, 1.5, -1.5, 100.9, -100.9];
    let src_bytes: &[u8] = bytemuck::cast_slice(src_f32);
    let host = run_cast(&table, &dev, src_bytes, DType::F32, DType::I64, src_f32.len());

    let host_i64: &[i64] = bytemuck::cast_slice(&host);
    // C static_cast<int64_t>(float) truncates toward zero.
    assert_eq!(host_i64, &[0_i64, 1, -1, 100, -100]);
}

/// PR 3.5 follow-up live test: `(ReduceSumTo, F32, Cuda)` through the
/// binding table. Input `[2, 3]` of `[1..6]`, target `[2, 1]` —
/// keepdim sum along the last axis, exactly the shape the lowered
/// SoftmaxLastDim subgraph emits. Expected `[6, 15]`.
#[test]
#[ignore]
fn reduce_sum_to_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = build_storage_cuda(&dev, &xs);
    // Output: 2 elements, shape [2, 1].
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ReduceSumTo, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (ReduceSumTo, F32, Cuda)");

    let params = OpParams::ReduceSumTo {
        input_shape: vec![2, 3],
        output_shape: vec![2, 1],
    };
    let in_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2, 1]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[6.0_f32, 15.0]);
}

/// PR 3.5 follow-up live test: `(ReduceMaxTo, F32, Cuda)` through the
/// binding table. Symmetric of `reduce_sum_to_f32_through_binding_table`.
/// Input `[2, 3]` of `[1, 5, 2, 4, 3, 6]`, target `[2, 1]` — keepdim
/// max along the last axis. Expected `[5, 6]`.
#[test]
#[ignore]
fn reduce_max_to_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);

    let xs = [1.0_f32, 5.0, 2.0, 4.0, 3.0, 6.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ReduceMaxTo, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (ReduceMaxTo, F32, Cuda)");

    let params = OpParams::ReduceMaxTo {
        input_shape: vec![2, 3],
        output_shape: vec![2, 1],
    };
    let in_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2, 1]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(host_f32, &[5.0_f32, 6.0]);
}
