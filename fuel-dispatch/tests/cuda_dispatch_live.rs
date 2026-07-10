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

use fuel_ir::{dispatch::OpKind, probe::BackendId, DType, Layout, Shape};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::{register_cuda_kernels, register_cpu_kernels}, kernel::{KernelBindingTable, MatmulM, OpParams}};
use fuel_memory::{BackendStorage, Storage};

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
        m_compute: MatmulM::All,
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
    register_baracuda_cuda_kernels(&mut table);

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
        m_compute: MatmulM::All,
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
    register_baracuda_cuda_kernels(&mut table);

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
        m_compute: MatmulM::All,
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

/// CPU bf16 matmul reference (row-major Rrr). Accumulates in f32 to
/// match how CUTLASS / cuBLAS bf16 GEMMs do the math — input precision
/// is bf16 but the multiply-accumulate runs at f32. Only used by the
/// bf16 matmul tests below; small enough not to belong in a shared
/// helper.
fn cpu_bf16_matmul_rrr(
    a: &[half::bf16],
    b: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<half::bf16> {
    let mut out = vec![half::bf16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk].to_f32() * b[kk * n + j].to_f32();
            }
            out[i * n + j] = half::bf16::from_f32(acc);
        }
    }
    out
}

/// End-to-end: rank-2 MatMul BF16 through the binding table.
/// First real exercise of the CUTLASS `LayoutSku::Rrr` path:
/// `Op::MatMul`-shaped row-major × row-major GEMM, no transpose
/// trick.
///
/// Shape is `(M=16, N=16, K=32)` — CUTLASS sm80 requires 128-bit
/// alignment on each operand, which for bf16 means M, N, K all
/// multiples of 8. The architecture's eventual route picker will
/// fall back to cuBLAS for misaligned shapes; today the binding
/// table holds the CUTLASS impl unconditionally. Reference is a
/// CPU matmul accumulating in f32 (matches CUTLASS's accumulator);
/// tolerance is `K * 5e-3` per the alpha.13 smoke-test convention.
#[test]
#[ignore]
fn matmul_bf16_rank2_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let (m, n, k) = (16_usize, 16_usize, 32_usize);
    let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.013).cos()).collect();
    let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let expected_bf16 = cpu_bf16_matmul_rrr(&a_bf16, &b_bf16, m, n, k);
    let expected: Vec<f32> = expected_bf16.iter().map(|x| x.to_f32()).collect();

    let lhs = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&a_bf16), DType::BF16);
    let rhs = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&b_bf16), DType::BF16);
    let out_bytes = CudaStorageBytes::alloc(&dev, m * n * 2).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::BF16);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MatMul, &[DType::BF16, DType::BF16, DType::BF16], BackendId::Cuda)
        .expect("lookup (MatMul, BF16, Cuda)");

    let params = OpParams::Matmul {
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        m,
        n,
        k,
        m_compute: MatmulM::All,
    };

    kernel(&[lhs_arc.clone(), rhs_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let got_bf16: &[half::bf16] = bytemuck::cast_slice(&host);
    let got: Vec<f32> = got_bf16.iter().map(|x| x.to_f32()).collect();
    let tol = (k as f32) * 5e-3;
    assert_close(&got, &expected, tol);
}

/// End-to-end: equal-batch MatMul BF16 through the binding table.
/// `[2, M, K]` @ `[2, K, N]` → `[2, M, N]` with `(M=16, N=16, K=32)`.
/// Routes through cutlass_matmul_bf16's per-batch loop —
/// BatchedGemmPlan native dispatch lands in Phase B6. Each batch
/// uses an independent seed so the matmuls don't collapse to
/// identical work.
#[test]
#[ignore]
fn matmul_bf16_batched_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let (m, n, k, batches) = (16_usize, 16_usize, 32_usize, 2_usize);
    let mut a_bf16 = vec![half::bf16::ZERO; batches * m * k];
    let mut b_bf16 = vec![half::bf16::ZERO; batches * k * n];
    for b in 0..batches {
        for i in 0..m * k {
            a_bf16[b * m * k + i] =
                half::bf16::from_f32(((b * 1000 + i) as f32 * 0.01).sin());
        }
        for i in 0..k * n {
            b_bf16[b * k * n + i] =
                half::bf16::from_f32(((b * 1000 + i) as f32 * 0.013).cos());
        }
    }
    let mut expected: Vec<f32> = Vec::with_capacity(batches * m * n);
    for b in 0..batches {
        let a_off = b * m * k;
        let b_off = b * k * n;
        let part = cpu_bf16_matmul_rrr(
            &a_bf16[a_off..a_off + m * k],
            &b_bf16[b_off..b_off + k * n],
            m, n, k,
        );
        expected.extend(part.iter().map(|x| x.to_f32()));
    }

    let lhs = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&a_bf16), DType::BF16);
    let rhs = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&b_bf16), DType::BF16);
    let out_bytes = CudaStorageBytes::alloc(&dev, batches * m * n * 2).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::BF16);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MatMul, &[DType::BF16, DType::BF16, DType::BF16], BackendId::Cuda)
        .expect("lookup (MatMul, BF16, Cuda)");

    let params = OpParams::Matmul {
        lhs_batch_dims: vec![batches],
        rhs_batch_dims: vec![batches],
        m,
        n,
        k,
        m_compute: MatmulM::All,
    };

    kernel(&[lhs_arc.clone(), rhs_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let got_bf16: &[half::bf16] = bytemuck::cast_slice(&host);
    let got: Vec<f32> = got_bf16.iter().map(|x| x.to_f32()).collect();
    let tol = (k as f32) * 5e-3;
    assert_close(&got, &expected, tol);
}

/// CPU f16 matmul reference (row-major Rrr). Mirrors
/// [`cpu_bf16_matmul_rrr`] at `f16` dtype; accumulates in f32 to
/// match the GEMM kernel's accumulator.
fn cpu_f16_matmul_rrr(
    a: &[half::f16],
    b: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<half::f16> {
    let mut out = vec![half::f16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk].to_f32() * b[kk * n + j].to_f32();
            }
            out[i * n + j] = half::f16::from_f32(acc);
        }
    }
    out
}

/// End-to-end: rank-2 MatMul F16 through the binding table. Mirror
/// of [`matmul_bf16_rank2_through_binding_table`] at `f16` dtype;
/// same CUTLASS `LayoutSku::Rrr` path. F16 has more mantissa than
/// bf16 (11-bit vs 8-bit) so tolerances are tighter — `K * 1e-3`.
#[test]
#[ignore]
fn matmul_f16_rank2_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let (m, n, k) = (16_usize, 16_usize, 32_usize);
    let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.013).cos()).collect();
    let a_f16: Vec<half::f16> = a_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let b_f16: Vec<half::f16> = b_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let expected_f16 = cpu_f16_matmul_rrr(&a_f16, &b_f16, m, n, k);
    let expected: Vec<f32> = expected_f16.iter().map(|x| x.to_f32()).collect();

    let lhs = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&a_f16), DType::F16);
    let rhs = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&b_f16), DType::F16);
    let out_bytes = CudaStorageBytes::alloc(&dev, m * n * 2).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F16);

    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::MatMul, &[DType::F16, DType::F16, DType::F16], BackendId::Cuda)
        .expect("lookup (MatMul, F16, Cuda)");

    let params = OpParams::Matmul {
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        m,
        n,
        k,
        m_compute: MatmulM::All,
    };

    kernel(&[lhs_arc.clone(), rhs_arc.clone()], &mut [out_arc.clone()], &[], &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let got_f16: &[half::f16] = bytemuck::cast_slice(&host);
    let got: Vec<f32> = got_f16.iter().map(|x| x.to_f32()).collect();
    let tol = (k as f32) * 1e-3;
    assert_close(&got, &expected, tol);
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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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
    register_baracuda_cuda_kernels(&mut table);

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

/// End-to-end: gather rows from an F32 source via U32 indices through
/// the unified binding-table dispatch on CUDA. Source is a 4×3 row
/// matrix; indices `[2, 0, 3, 1]` permute the rows; output is the
/// 4×3 permuted matrix. Exercises the `(IndexSelect, [F32, U32, F32],
/// Cuda)` entry, which dispatches `is_u32_f32` from the INDEXING PTX
/// module on byte-shaped CUDA storage.
#[test]
#[ignore]
fn index_select_f32_u32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // Source: 4 rows × 3 columns of F32, distinct values per row so a
    // permutation of rows is easy to verify.
    let src_f32: [f32; 12] = [
        100.0, 101.0, 102.0,  // row 0
        200.0, 201.0, 202.0,  // row 1
        300.0, 301.0, 302.0,  // row 2
        400.0, 401.0, 402.0,  // row 3
    ];
    let src = build_storage_cuda(&dev, &src_f32);

    // Indices: select rows in the order [2, 0, 3, 1].
    let ids_u32: [u32; 4] = [2, 0, 3, 1];
    let ids_bytes: &[u8] = bytemuck::cast_slice(&ids_u32);
    let ids = build_storage_cuda_from_bytes(&dev, ids_bytes, DType::U32);

    // Output: 4 selected rows × 3 columns of F32 (16 elements * 4 bytes).
    let out_bytes = CudaStorageBytes::alloc(&dev, 4 * 3 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let ids_arc = Arc::new(RwLock::new(ids));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(
            OpKind::IndexSelect,
            &[DType::F32, DType::U32, DType::F32],
            BackendId::Cuda,
        )
        .expect("lookup (IndexSelect, [F32, U32, F32], Cuda)");

    // OpParams::IndexSelect — selecting along dim 0 of a [4, 3] source
    // with 4 indices: outer_count=1, source_dim_size=4, n_indices=4,
    // inner_count=3.
    let params = OpParams::IndexSelect {
        outer_count: 1,
        source_dim_size: 4,
        n_indices: 4,
        inner_count: 3,
    };
    let src_layout = Layout::contiguous(Shape::from_dims(&[4, 3]));
    let ids_layout = Layout::contiguous(Shape::from_dims(&[4]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[4, 3]));
    let layouts = vec![src_layout, ids_layout, out_layout];

    kernel(
        &[src_arc.clone(), ids_arc.clone()],
        &mut [out_arc.clone()],
        &layouts,
        &params,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: [f32; 12] = [
        300.0, 301.0, 302.0,  // row 2
        100.0, 101.0, 102.0,  // row 0
        400.0, 401.0, 402.0,  // row 3
        200.0, 201.0, 202.0,  // row 1
    ];
    assert_eq!(host_f32, &expected);
}

/// End-to-end: N-dimensional gather along dim 1 through the unified
/// binding-table dispatch on CUDA. Source [2, 4]; indices [2, 3]
/// (same rank as source, output_shape == indices_shape, only the
/// gathered dim differs); output [2, 3] picks per-row columns via
/// the per-element index. Exercises the `(Gather, [F32, U32, F32],
/// Cuda)` entry, which dispatches `gather_u32_f32` from the INDEXING
/// PTX module.
#[test]
#[ignore]
fn gather_f32_u32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // Source: 2 rows × 4 columns of F32.
    let src_f32: [f32; 8] = [
        10.0, 11.0, 12.0, 13.0,  // row 0
        20.0, 21.0, 22.0, 23.0,  // row 1
    ];
    let src = build_storage_cuda(&dev, &src_f32);

    // Indices: 2 rows × 3 columns of U32. Per row, picks 3 source
    // columns by index. Note that indices and source share rank
    // (matching the Gather contract); only `dim` (=1) varies.
    let ids_u32: [u32; 6] = [
        3, 0, 2,  // row 0 picks src[0,3], src[0,0], src[0,2]
        1, 2, 0,  // row 1 picks src[1,1], src[1,2], src[1,0]
    ];
    let ids_bytes: &[u8] = bytemuck::cast_slice(&ids_u32);
    let ids = build_storage_cuda_from_bytes(&dev, ids_bytes, DType::U32);

    // Output: 2 rows × 3 columns of F32 (6 elements * 4 bytes).
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 3 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let ids_arc = Arc::new(RwLock::new(ids));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(
            OpKind::Gather,
            &[DType::F32, DType::U32, DType::F32],
            BackendId::Cuda,
        )
        .expect("lookup (Gather, [F32, U32, F32], Cuda)");

    let params = OpParams::Gather {
        source_shape: vec![2, 4],
        output_shape: vec![2, 3],
        dim: 1,
    };
    let src_layout = Layout::contiguous(Shape::from_dims(&[2, 4]));
    let ids_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let layouts = vec![src_layout, ids_layout, out_layout];

    kernel(
        &[src_arc.clone(), ids_arc.clone()],
        &mut [out_arc.clone()],
        &layouts,
        &params,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: [f32; 6] = [
        13.0, 10.0, 12.0,  // row 0: src[0,3], src[0,0], src[0,2]
        21.0, 22.0, 20.0,  // row 1: src[1,1], src[1,2], src[1,0]
    ];
    assert_eq!(host_f32, &expected);
}

/// End-to-end: element-wise clamp through the unified binding-table
/// dispatch on CUDA. Source contains values both inside and outside
/// the clamp range; clamp(-1.0, 2.0) folds them to the boundary.
/// Exercises the `(ClampElementwise, F32, Cuda)` entry, which
/// dispatches `uclamp_f32` from the UNARY PTX module via the new
/// UNARY_OP2 macro.
#[test]
#[ignore]
fn clamp_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let xs: [f32; 7] = [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 1.5];
    let src = build_storage_cuda(&dev, &xs);

    let out_bytes = CudaStorageBytes::alloc(&dev, 7 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ClampElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (ClampElementwise, F32, Cuda)");

    let params = OpParams::Clamp { min: -1.0, max: 2.0 };
    let in_layout = Layout::contiguous(Shape::from_dims(&[7]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[7]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    let expected: [f32; 7] = [-1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 1.5];
    assert_eq!(host_f32, &expected);
}

/// End-to-end: integer-power through the unified binding-table
/// dispatch on CUDA. Tests both positive (exp=3) and the rust-std
/// f32::powi parity for negatives via square-and-multiply.
/// Exercises the `(PowIElementwise, F32, Cuda)` entry, which
/// dispatches `upowi_f32` from the UNARY PTX module via the new
/// UPOWI_OP macro.
#[test]
#[ignore]
fn powi_elementwise_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let xs: [f32; 5] = [-2.0, -1.0, 0.0, 1.5, 3.0];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 5 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::PowIElementwise, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (PowIElementwise, F32, Cuda)");

    let params = OpParams::PowI { exp: 3 };
    let in_layout = Layout::contiguous(Shape::from_dims(&[5]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[5]));
    let layouts = vec![in_layout, out_layout];

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &layouts, &params)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    // Expected: (-2)^3 = -8, (-1)^3 = -1, 0^3 = 0, 1.5^3 = 3.375, 3^3 = 27
    let expected: [f32; 5] = [-8.0, -1.0, 0.0, 3.375, 27.0];
    assert_eq!(host_f32, &expected);
}

/// End-to-end: 3-input concat along the inner dim through the
/// unified binding-table dispatch on CUDA. Each input has shape
/// [2, dim_i, 2] with dim_i ∈ {1, 2, 1}; the output shape is
/// [2, 4, 2]. Exercises `(Concat, [F32, F32], Cuda)` which dispatches
/// `concat_f32` from the INDEXING PTX module (one launch per input).
#[test]
#[ignore]
fn concat_f32_three_inputs_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // Three inputs with shapes [2, 1, 2], [2, 2, 2], [2, 1, 2].
    // outer_count=2, input_dim_sizes=[1, 2, 1], inner_count=2.
    let a_f32: [f32; 4] = [
        1.0, 2.0,    // outer 0, dim 0
        9.0, 9.0,    // outer 1, dim 0
    ];
    let b_f32: [f32; 8] = [
        3.0, 4.0,    // outer 0, dim 0
        5.0, 6.0,    // outer 0, dim 1
        7.0, 8.0,    // outer 1, dim 0
        7.5, 8.5,    // outer 1, dim 1
    ];
    let c_f32: [f32; 4] = [
        100.0, 200.0,  // outer 0, dim 0
        300.0, 400.0,  // outer 1, dim 0
    ];
    let a = build_storage_cuda(&dev, &a_f32);
    let b = build_storage_cuda(&dev, &b_f32);
    let c = build_storage_cuda(&dev, &c_f32);

    // Output: [2, 4, 2] → 16 elements * 4 bytes.
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4 * 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let a_arc = Arc::new(RwLock::new(a));
    let b_arc = Arc::new(RwLock::new(b));
    let c_arc = Arc::new(RwLock::new(c));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::Concat, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (Concat, [F32, F32], Cuda)");

    let params = OpParams::Concat {
        outer_count: 2,
        input_dim_sizes: vec![1, 2, 1],
        inner_count: 2,
        axis: 1,
    };
    // Layouts: per-input + output. Per-input shapes are [2, dim_i, 2].
    let a_layout = Layout::contiguous(Shape::from_dims(&[2, 1, 2]));
    let b_layout = Layout::contiguous(Shape::from_dims(&[2, 2, 2]));
    let c_layout = Layout::contiguous(Shape::from_dims(&[2, 1, 2]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2, 4, 2]));
    let layouts = vec![a_layout, b_layout, c_layout, out_layout];

    kernel(
        &[a_arc.clone(), b_arc.clone(), c_arc.clone()],
        &mut [out_arc.clone()],
        &layouts,
        &params,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    // Expected output [2, 4, 2]:
    // outer 0: a[0,0,*], b[0,0,*], b[0,1,*], c[0,0,*]
    //         = 1,2 | 3,4 | 5,6 | 100,200
    // outer 1: a[1,0,*], b[1,0,*], b[1,1,*], c[1,0,*]
    //         = 9,9 | 7,8 | 7.5,8.5 | 300,400
    let expected: [f32; 16] = [
        1.0, 2.0,  3.0, 4.0,  5.0, 6.0,  100.0, 200.0,
        9.0, 9.0,  7.0, 8.0,  7.5, 8.5,  300.0, 400.0,
    ];
    assert_eq!(host_f32, &expected);
}

/// End-to-end: ArgMaxDim along the inner dim through the unified
/// binding-table dispatch on CUDA. Source [2, 3] of F32; output [2]
/// of U32 (indices into the reduced axis). Exercises
/// `(ArgMaxDim, [F32, U32], Cuda)` which dispatches `fast_argmax_f32`
/// from the REDUCE PTX module after dim-reordering.
#[test]
#[ignore]
fn argmax_dim_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // [2, 3]: row 0 max at idx 1 (5.0), row 1 max at idx 2 (6.0).
    let xs: [f32; 6] = [
        1.0, 5.0, 2.0,
        4.0, 3.0, 6.0,
    ];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::U32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ArgMaxDim, &[DType::F32, DType::U32], BackendId::Cuda)
        .expect("lookup (ArgMaxDim, [F32, U32], Cuda)");

    let params = OpParams::Reduce { dims: vec![1], keepdim: false };
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
    let host_u32: &[u32] = bytemuck::cast_slice(&host);
    assert_eq!(host_u32, &[1_u32, 2]);
}

/// End-to-end: ArgMinDim along the inner dim — sister of the
/// argmax test. `(ArgMinDim, [F32, U32], Cuda)` dispatches
/// `fast_argmin_f32`.
#[test]
#[ignore]
fn argmin_dim_f32_through_binding_table() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // [2, 3]: row 0 min at idx 0 (1.0), row 1 min at idx 1 (3.0).
    let xs: [f32; 6] = [
        1.0, 5.0, 2.0,
        4.0, 3.0, 6.0,
    ];
    let src = build_storage_cuda(&dev, &xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, 2 * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::U32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ArgMinDim, &[DType::F32, DType::U32], BackendId::Cuda)
        .expect("lookup (ArgMinDim, [F32, U32], Cuda)");

    let params = OpParams::Reduce { dims: vec![1], keepdim: false };
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
    let host_u32: &[u32] = bytemuck::cast_slice(&host);
    assert_eq!(host_u32, &[0_u32, 1]);
}

// =============================================================================
// Judge cuda:0 coverage gap closure (2026-06-11) — Ceil / Floor /
// Round / Erf / Rsqrt / Pow / Rem. One test per newly registered op.
// =============================================================================

/// Run a unary elementwise op through the binding table on F32 input
/// and return the host F32 output. Shared by the coverage-gap tests
/// below.
fn run_unary_f32(op: OpKind, xs: &[f32]) -> Vec<f32> {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let src = build_storage_cuda(&dev, xs);
    let out_bytes = CudaStorageBytes::alloc(&dev, xs.len() * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(op, &[DType::F32, DType::F32], BackendId::Cuda)
        .unwrap_or_else(|e| panic!("lookup ({op:?}, F32, Cuda): {e:?}"));

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    bytemuck::cast_slice::<u8, f32>(&host).to_vec()
}

/// Run a binary elementwise op through the binding table on F32
/// inputs and return the host F32 output.
fn run_binary_f32(op: OpKind, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    assert_eq!(lhs.len(), rhs.len());
    let dev = CudaDevice::new(0).expect("CUDA device");
    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let lhs_s = build_storage_cuda(&dev, lhs);
    let rhs_s = build_storage_cuda(&dev, rhs);
    let out_bytes = CudaStorageBytes::alloc(&dev, lhs.len() * 4).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let lhs_arc = Arc::new(RwLock::new(lhs_s));
    let rhs_arc = Arc::new(RwLock::new(rhs_s));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(op, &[DType::F32, DType::F32, DType::F32], BackendId::Cuda)
        .unwrap_or_else(|e| panic!("lookup ({op:?}, F32, Cuda): {e:?}"));

    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &binary_layouts(&[lhs.len()]),
        &OpParams::None,
    )
    .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    bytemuck::cast_slice::<u8, f32>(&host).to_vec()
}

/// End-to-end: FloorElementwise F32 through the binding table.
#[test]
#[ignore]
fn floor_elementwise_f32_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let xs = [-1.5_f32, -0.5, 0.0, 0.5, 1.5, 2.999, -3.001];
    let got = run_unary_f32(OpKind::FloorElementwise, &xs);
    let expected: Vec<f32> = xs.iter().map(|v| v.floor()).collect();
    assert_eq!(got, expected);
}

/// End-to-end: CeilElementwise F32 through the binding table.
#[test]
#[ignore]
fn ceil_elementwise_f32_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let xs = [-1.5_f32, -0.5, 0.0, 0.5, 1.5, 2.999, -3.001];
    let got = run_unary_f32(OpKind::CeilElementwise, &xs);
    let expected: Vec<f32> = xs.iter().map(|v| v.ceil()).collect();
    assert_eq!(got, expected);
}

/// End-to-end: RoundElementwise F32 through the binding table.
/// Fuel's contract is **banker's rounding** (round-half-to-even,
/// CPU = `f32::round_ties_even`); baracuda's kernel is `rintf` which
/// matches in the default rounding mode. The halfway points are the
/// load-bearing cases — `f32::round` (half-away-from-zero) would give
/// 1, 3, -1, -3 instead.
#[test]
#[ignore]
fn round_elementwise_f32_halfway_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let xs = [0.5_f32, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 1.4999, 2.5001];
    let got = run_unary_f32(OpKind::RoundElementwise, &xs);
    // Same formula the CPU binding-table kernel uses.
    let expected: Vec<f32> = xs.iter().map(|v| v.round_ties_even()).collect();
    assert_eq!(got, expected, "round must be ties-to-even, not half-away-from-zero");
    // Spot-check the halves explicitly so a convention regression
    // reads clearly: 0.5 → 0, 1.5 → 2, 2.5 → 2, -0.5 → -0.
    assert_eq!(&got[..3], &[0.0_f32, 2.0, 2.0]);
    assert_eq!(got[4], 0.0);
}

/// End-to-end: ErfElementwise F32 through the binding table. Value
/// check against reference Gauss-error-function values — these
/// distinguish plain `erf(x)` from both gelu flavors (gelu_erf(1) ≈
/// 0.84134 vs erf(1) ≈ 0.84270; the 1e-6 epsilon catches a flavor
/// mix-up like the one fixed in 9b53da38).
#[test]
#[ignore]
fn erf_elementwise_f32_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let xs = [0.0_f32, 0.5, 1.0, -1.0, 2.0, -2.0];
    let got = run_unary_f32(OpKind::ErfElementwise, &xs);
    // Reference values (Abramowitz & Stegun / mpmath, f32-rounded).
    let expected = [
        0.0_f32,
        0.520_499_88,
        0.842_700_79,
        -0.842_700_79,
        0.995_322_27,
        -0.995_322_27,
    ];
    assert_close(&got, &expected, 1e-6);
}

/// End-to-end: RsqrtElementwise F32 through the binding table. The
/// wrapper existed since Tier 1 but was never registered — this is
/// the first test that can reach it.
#[test]
#[ignore]
fn rsqrt_elementwise_f32_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let xs = [1.0_f32, 4.0, 0.25, 100.0, 2.0];
    let got = run_unary_f32(OpKind::RsqrtElementwise, &xs);
    let expected: Vec<f32> = xs.iter().map(|v| 1.0 / v.sqrt()).collect();
    assert_close(&got, &expected, 1e-6);
}

/// End-to-end: PowElementwise F32 (tensor^tensor) through the binding
/// table. IEEE-754 semantics: negative base with non-integer exponent
/// is NaN.
#[test]
#[ignore]
fn pow_elementwise_f32_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let lhs = [2.0_f32, 4.0, -2.0, 9.0, 5.0];
    let rhs = [3.0_f32, 0.5, 2.0, 0.5, 0.0];
    let got = run_binary_f32(OpKind::PowElementwise, &lhs, &rhs);
    let expected = [8.0_f32, 2.0, 4.0, 3.0, 1.0];
    assert_close(&got, &expected, 1e-5);

    // pow(-2, 0.5) = NaN per IEEE-754 (matches CPU `powf`).
    let nan_out = run_binary_f32(OpKind::PowElementwise, &[-2.0], &[0.5]);
    assert!(nan_out[0].is_nan(), "pow(-2, 0.5) must be NaN, got {}", nan_out[0]);
}

/// End-to-end: RemElementwise F32 through the binding table. Fuel's
/// contract is the **PyTorch convention** `a - floor(a/b) * b` (sign
/// of the divisor) — bound to baracuda's `binary_mod_*`. The
/// mixed-sign cases are the load-bearing ones: C99 `fmod` (baracuda's
/// `binary_remainder_*`) would give -2 and 2 for the middle pair.
#[test]
#[ignore]
fn rem_elementwise_f32_pytorch_convention_through_binding_table() {
    if dev_or_skip().is_none() { return; }
    let lhs = [5.0_f32, -5.0, 5.0, -5.0, 7.5];
    let rhs = [3.0_f32, 3.0, -3.0, -3.0, 2.0];
    let got = run_binary_f32(OpKind::RemElementwise, &lhs, &rhs);
    // PyTorch / Python `%`: 5%3=2, -5%3=1, 5%-3=-1, -5%-3=-2, 7.5%2=1.5.
    let expected = [2.0_f32, 1.0, -1.0, -2.0, 1.5];
    assert_close(&got, &expected, 1e-6);
}

// =============================================================================
// Elementwise NaN-convention pins (2026-07-08, torch parity —
// docs/architecture/10-decisions-log.md). These are the REAL CUDA-kernel
// pins for the NaN-propagating convention: direct binding-table
// invocation guarantees the CUDA wrapper (and thus baracuda's kernel)
// actually executed — unlike a lazy `realize_f32_cuda` graph, where
// cost-based placement may route a tiny op to CPU on both legs (the
// hole that made the fuel-core end-to-end NaN tests hollow; see
// `fuel-core/src/lazy.rs::relu_nan_convention_lazy_realize_smoke`'s doc
// comment).
// =============================================================================

/// CUDA `relu` is NaN-PROPAGATING (torch parity): `relu(NaN) == NaN`,
/// payload aside. Pins the baracuda alpha.76 rebind of
/// `OpKind::ReluElementwise` to `unary_relu_propagating_f32` — under the
/// old scrubbing `unary_relu_f32` (`fmaxf`) binding this test fails at
/// index 0 (NaN scrubbed to 0.0), which was verified born-red against a
/// deliberately sabotaged stem before the rebind shipped.
#[test]
#[ignore]
fn cuda_relu_propagates_nan_f32() {
    if dev_or_skip().is_none() { return; }
    let xs = [f32::NAN, -2.0, 3.0];
    let got = run_unary_f32(OpKind::ReluElementwise, &xs);
    assert!(
        got[0].is_nan(),
        "CUDA relu(NaN) must propagate NaN (torch parity, alpha.76 \
         unary_relu_propagating_* rebind); got {} — a non-NaN here means \
         the binding regressed to the scrubbing unary_relu_* family",
        got[0],
    );
    assert_eq!(got[1], 0.0, "relu(-2.0)");
    assert_eq!(got[2], 3.0, "relu(3.0)");
}

/// CUDA `relu` NaN propagation, BF16 sibling — same pin as the f32 test
/// above for the `unary_relu_propagating_bf16` binding.
#[test]
#[ignore]
fn cuda_relu_propagates_nan_bf16() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    let xs_bf16: Vec<half::bf16> = [f32::NAN, -2.0, 3.0, 0.5]
        .iter()
        .map(|&x| half::bf16::from_f32(x))
        .collect();
    let src = build_storage_cuda_from_bytes(&dev, bytemuck::cast_slice(&xs_bf16), DType::BF16);
    let out_bytes = CudaStorageBytes::alloc(&dev, xs_bf16.len() * 2).expect("out alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::BF16);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = table
        .lookup(OpKind::ReluElementwise, &[DType::BF16, DType::BF16], BackendId::Cuda)
        .expect("lookup (ReluElementwise, BF16, Cuda)");

    kernel(&[src_arc.clone()], &mut [out_arc.clone()], &[], &OpParams::None)
        .expect("kernel call");

    let result_storage = out_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("output not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_bf16: &[half::bf16] = bytemuck::cast_slice(&host);
    assert!(
        host_bf16[0].to_f32().is_nan(),
        "CUDA bf16 relu(NaN) must propagate NaN; got {}",
        host_bf16[0],
    );
    assert_eq!(host_bf16[1].to_f32(), 0.0, "relu(-2.0)");
    assert_eq!(host_bf16[2].to_f32(), 3.0, "relu(3.0)");
    assert_eq!(host_bf16[3].to_f32(), 0.5, "relu(0.5)");
}

/// CUDA `maximum` / `minimum` are NaN-PROPAGATING (torch parity):
/// either-operand NaN → NaN out. Baracuda's `binary_maximum_fp.cu` /
/// `binary_minimum_fp.cu` were already propagating before alpha.76 —
/// this pins that against regression at the actual CUDA binding (the
/// fuel-core end-to-end test can't guarantee the CUDA leg ran; see the
/// block comment above).
#[test]
#[ignore]
fn cuda_maximum_minimum_propagate_nan_f32() {
    if dev_or_skip().is_none() { return; }
    // Index 0: NaN in lhs only. Index 1: NaN in rhs only. Index 2: NaN
    // in both. Indices 3-4: non-NaN sanity.
    let lhs = [f32::NAN, -2.0, f32::NAN, 1.0, -3.0];
    let rhs = [1.0_f32, f32::NAN, f32::NAN, 4.0, -5.0];

    let max_got = run_binary_f32(OpKind::MaximumElementwise, &lhs, &rhs);
    for i in 0..3 {
        assert!(
            max_got[i].is_nan(),
            "maximum[{i}] must be NaN (either-operand propagation); got {}",
            max_got[i],
        );
    }
    assert_eq!(max_got[3], 4.0, "maximum(1, 4)");
    assert_eq!(max_got[4], -3.0, "maximum(-3, -5)");

    let min_got = run_binary_f32(OpKind::MinimumElementwise, &lhs, &rhs);
    for i in 0..3 {
        assert!(
            min_got[i].is_nan(),
            "minimum[{i}] must be NaN (either-operand propagation); got {}",
            min_got[i],
        );
    }
    assert_eq!(min_got[3], 1.0, "minimum(1, 4)");
    assert_eq!(min_got[4], -5.0, "minimum(-3, -5)");
}

/// CUDA in-place `relu` (`OpKind::ReluInplace`) is NaN-PROPAGATING too —
/// the in-place sibling of `cuda_relu_propagates_nan_f32`. Pins the
/// 2026-07-08 rebind of the `unary_inplace_relu_*` stems to
/// `unary_relu_propagating_*` (closing the residual gap left by the
/// forward-only `ReluElementwise` rebind in alpha.76). The in-place
/// wrapper takes 0 inputs + 1 output — the target the executor adopts —
/// which the kernel mutates same-pointer; this also exercises that the
/// propagating kernel is in-place-safe (elementwise, so it is). Under
/// the old scrubbing `unary_relu_f32` stem this fails at index 0.
#[test]
#[ignore]
fn cuda_relu_inplace_propagates_nan_f32() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // The target buffer is both input and output for an in-place op.
    let xs = [f32::NAN, -2.0_f32, 3.0];
    let target = build_storage_cuda(&dev, &xs);
    let target_arc = Arc::new(RwLock::new(target));

    let kernel = table
        .lookup(OpKind::ReluInplace, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (ReluInplace, F32, Cuda)");

    // In-place contract: 0 inputs, 1 output (the target adopted by the
    // executor's WorkItemKind::InplaceKernel arm).
    kernel(&[], &mut [target_arc.clone()], &[], &OpParams::None)
        .expect("in-place kernel call");

    let result_storage = target_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("target not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let got: &[f32] = bytemuck::cast_slice(&host);
    assert!(
        got[0].is_nan(),
        "CUDA relu_inplace(NaN) must propagate NaN (torch parity); got {} — \
         a non-NaN means the in-place stem regressed to scrubbing unary_relu_*",
        got[0],
    );
    assert_eq!(got[1], 0.0, "relu_inplace(-2.0)");
    assert_eq!(got[2], 3.0, "relu_inplace(3.0)");
}

/// Build a rank-0 I64 device scalar (the device-resident `_doff`
/// offset). `from_cpu_bytes` H2D-copies the 8 native-endian bytes.
fn build_i64_scalar_cuda(dev: &CudaDevice, v: i64) -> Storage {
    let bytes = v.to_ne_bytes();
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, &bytes).expect("h2d i64");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::I64)
}

/// End-to-end: `OpKind::WriteSliceDoff` F32 through the binding table
/// with a DEVICE-RESIDENT offset of 1. This is the CapturedRun form-B
/// KV-cache append — the offset is read device-side (NO D2H), which is
/// what lets a captured graph replay at the host-updated position.
///
/// Sabotage-calibration: the offset is NON-ZERO (1), so the write MUST
/// land at row 1. If the kernel ignored `dyn_start_dev` and used the
/// `range_start[axis]` placeholder (0), the result would be
/// `[7, 8, 0, 0, 0, 0, 0, 0]` — this test distinguishes "device offset
/// read" from "placeholder baked". Direct dispatch guarantees the CUDA
/// kernel runs (no cost-based placement routing it to CPU).
#[test]
#[ignore]
fn write_slice_doff_f32_at_device_offset_cuda() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // dest [4, 2] zeros; source [1, 2] = [7, 8]; device offset = 1.
    let dest = build_storage_cuda(&dev, &[0.0_f32; 8]);
    let source = build_storage_cuda(&dev, &[7.0_f32, 8.0]);
    let offset = build_i64_scalar_cuda(&dev, 1);

    let source_arc = Arc::new(RwLock::new(source));
    let offset_arc = Arc::new(RwLock::new(offset));
    let dest_arc = Arc::new(RwLock::new(dest));

    let kernel = table
        .lookup(OpKind::WriteSliceDoff, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (WriteSliceDoff, F32, Cuda)");

    let params = OpParams::WriteSliceDoff {
        dest_shape: vec![4, 2],
        axis: 0,
        ranges: vec![(0, 1), (0, 2)],
    };
    // Executor passes [source, offset] as inputs, [dest] as output.
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[1, 2])),
        Layout::contiguous(Shape::from_dims(&[])),
        Layout::contiguous(Shape::from_dims(&[4, 2])),
    ];
    kernel(
        &[source_arc.clone(), offset_arc.clone()],
        &mut [dest_arc.clone()],
        &layouts,
        &params,
    )
    .expect("write_slice_doff kernel call");

    let result_storage = dest_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("dest not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(
        host_f32,
        &[0.0_f32, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        "write must land at DEVICE offset 1 (row 1); [7,8,0,..] means the \
         placeholder range_start was used instead of dyn_start_dev",
    );
}

/// End-to-end CapturedRun access pattern: a capacity-4 decode loop
/// appending one token per step at the live `cached_len` offset, all
/// into the SAME dest buffer (in-place), each append running the
/// baracuda `_doff` kernel with a device-resident start. Verifies the
/// full KV history — no wrap, each token at its own row.
#[test]
#[ignore]
fn write_slice_doff_f32_decode_loop_cuda() {
    let Some(dev) = dev_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_cpu_kernels(&mut table);
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // A single persistent [4, 2] cache buffer, appended across 4 steps.
    let dest = build_storage_cuda(&dev, &[0.0_f32; 8]);
    let dest_arc = Arc::new(RwLock::new(dest));

    let kernel = table
        .lookup(OpKind::WriteSliceDoff, &[DType::F32, DType::F32], BackendId::Cuda)
        .expect("lookup (WriteSliceDoff, F32, Cuda)");

    let tokens = [
        [1.0_f32, 1.1],
        [2.0_f32, 2.1],
        [3.0_f32, 3.1],
        [4.0_f32, 4.1],
    ];
    for (step, token) in tokens.iter().enumerate() {
        let source = build_storage_cuda(&dev, token);
        let offset = build_i64_scalar_cuda(&dev, step as i64);
        let source_arc = Arc::new(RwLock::new(source));
        let offset_arc = Arc::new(RwLock::new(offset));
        let params = OpParams::WriteSliceDoff {
            dest_shape: vec![4, 2],
            axis: 0,
            ranges: vec![(0, 1), (0, 2)],
        };
        let layouts = vec![
            Layout::contiguous(Shape::from_dims(&[1, 2])),
            Layout::contiguous(Shape::from_dims(&[])),
            Layout::contiguous(Shape::from_dims(&[4, 2])),
        ];
        kernel(
            &[source_arc.clone(), offset_arc.clone()],
            &mut [dest_arc.clone()],
            &layouts,
            &params,
        )
        .expect("write_slice_doff decode-step kernel call");
    }

    let result_storage = dest_arc.read().unwrap();
    let BackendStorage::Cuda(c) = &result_storage.inner else {
        panic!("dest not on CUDA");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    let host_f32: &[f32] = bytemuck::cast_slice(&host);
    assert_eq!(
        host_f32,
        &[1.0_f32, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1],
        "each token must land at its own device-offset row (full KV history)",
    );
}
