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

/// Dispatch-table presence check (no GPU required). Confirms all
/// 6 binary-f32 ops register after V.2.A.
#[test]
fn vulkan_dispatch_binary_f32_registered() {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let key = [DType::F32, DType::F32, DType::F32];
    for op in [
        OpKind::AddElementwise,
        OpKind::SubElementwise,
        OpKind::MulElementwise,
        OpKind::DivElementwise,
        OpKind::MaximumElementwise,
        OpKind::MinimumElementwise,
    ] {
        let alts = table.lookup_alternatives(op, &key, BackendId::Vulkan);
        assert_eq!(
            alts.len(), 1,
            "expected 1 Vulkan alternative for {op:?} f32 after register_vulkan_kernels, got {}",
            alts.len(),
        );
    }
}

/// Helper for V.2.A binary correctness tests — uploads `a` and `b`,
/// dispatches `op` against `(F32, F32, F32)`, downloads + returns.
fn run_binary_f32(
    backend: &Arc<VulkanBackend>,
    op: OpKind,
    a_data: &[f32],
    b_data: &[f32],
) -> Vec<f32> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let n = a_data.len();
    let a_storage = upload_f32(backend, a_data);
    let b_storage = upload_f32(backend, b_data);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));
    let alts = table.lookup_alternatives(
        op,
        &[DType::F32, DType::F32, DType::F32],
        BackendId::Vulkan,
    );
    let kernel = alts[0].kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[n]));
    let layouts = vec![layout.clone(), layout.clone(), layout];
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("kernel dispatch");
    download_f32(backend, &out_arc.read().unwrap())
}

#[test]
#[ignore]
fn vulkan_dispatch_binary_sub_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_binary_f32(
        &backend,
        OpKind::SubElementwise,
        &[10.0, 20.0, 30.0, 40.0],
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(got, vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_binary_mul_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_binary_f32(
        &backend,
        OpKind::MulElementwise,
        &[2.0, 3.0, 4.0, 5.0],
        &[10.0, 10.0, 10.0, 10.0],
    );
    assert_eq!(got, vec![20.0, 30.0, 40.0, 50.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_binary_div_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_binary_f32(
        &backend,
        OpKind::DivElementwise,
        &[100.0, 80.0, 60.0, 40.0],
        &[2.0, 4.0, 5.0, 8.0],
    );
    assert_eq!(got, vec![50.0, 20.0, 12.0, 5.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_binary_maximum_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_binary_f32(
        &backend,
        OpKind::MaximumElementwise,
        &[1.0, 5.0, 3.0, 7.0],
        &[2.0, 4.0, 6.0, 1.0],
    );
    assert_eq!(got, vec![2.0, 5.0, 6.0, 7.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_binary_minimum_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_binary_f32(
        &backend,
        OpKind::MinimumElementwise,
        &[1.0, 5.0, 3.0, 7.0],
        &[2.0, 4.0, 6.0, 1.0],
    );
    assert_eq!(got, vec![1.0, 4.0, 3.0, 1.0]);
}

// ---- V.2.B — unary f32 ---------------------------------------------------

/// Dispatch-table presence check for all 13 unary-f32 ops.
#[test]
fn vulkan_dispatch_unary_f32_registered() {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let key = [DType::F32, DType::F32];
    for op in [
        OpKind::NegElementwise,
        OpKind::SqrElementwise,
        OpKind::SqrtElementwise,
        OpKind::ExpElementwise,
        OpKind::LogElementwise,
        OpKind::SinElementwise,
        OpKind::CosElementwise,
        OpKind::TanhElementwise,
        OpKind::SigmoidElementwise,
        OpKind::SiluElementwise,
        OpKind::GeluElementwise,
        OpKind::ReluElementwise,
        OpKind::StepElementwise,
    ] {
        let alts = table.lookup_alternatives(op, &key, BackendId::Vulkan);
        assert_eq!(
            alts.len(), 1,
            "expected 1 Vulkan alternative for {op:?} f32, got {}",
            alts.len(),
        );
    }
}

fn run_unary_f32(
    backend: &Arc<VulkanBackend>,
    op: OpKind,
    a_data: &[f32],
) -> Vec<f32> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let n = a_data.len();
    let a_storage = upload_f32(backend, a_data);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let a_arc = Arc::new(RwLock::new(a_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));
    let alts = table.lookup_alternatives(
        op,
        &[DType::F32, DType::F32],
        BackendId::Vulkan,
    );
    let kernel = alts[0].kernel;
    kernel(
        &[Arc::clone(&a_arc)],
        &mut [Arc::clone(&out_arc)],
        &[],
        &OpParams::None,
    ).expect("kernel dispatch");
    download_f32(backend, &out_arc.read().unwrap())
}

/// Tolerance helper for transcendental ops where bit-exact won't hold.
fn assert_close(got: &[f32], expected: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        let rel = diff / a.abs().max(b.abs()).max(1e-12);
        assert!(
            diff < abs_tol || rel < rel_tol,
            "[{i}] got={a} want={b} diff={diff} rel={rel}",
        );
    }
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_neg_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::NegElementwise, &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(got, vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_sqr_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::SqrElementwise, &[1.0, 2.0, -3.0, 4.0]);
    assert_eq!(got, vec![1.0, 4.0, 9.0, 16.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_sqrt_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::SqrtElementwise, &[1.0, 4.0, 9.0, 16.0]);
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_exp_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::ExpElementwise, &[0.0, 1.0, 2.0]);
    let want = [1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E];
    assert_close(&got, &want, 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_log_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(
        &backend,
        OpKind::LogElementwise,
        &[1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E],
    );
    assert_close(&got, &[0.0, 1.0, 2.0], 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_sin_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(
        &backend,
        OpKind::SinElementwise,
        &[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
    );
    assert_close(&got, &[0.0, 1.0, 0.0], 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_cos_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(
        &backend,
        OpKind::CosElementwise,
        &[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
    );
    assert_close(&got, &[1.0, 0.0, -1.0], 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_tanh_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::TanhElementwise, &[0.0, 1.0, -1.0]);
    assert_close(&got, &[0.0, 0.7615942, -0.7615942], 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_sigmoid_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::SigmoidElementwise, &[0.0, 1.0, -1.0]);
    assert_close(&got, &[0.5, 0.7310586, 0.26894143], 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_silu_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // silu(x) = x * sigmoid(x): silu(0) = 0, silu(1) = 0.7310586
    let got = run_unary_f32(&backend, OpKind::SiluElementwise, &[0.0, 1.0, -1.0]);
    assert_close(&got, &[0.0, 0.7310586, -0.26894143], 1e-5, 1e-5);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_gelu_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // gelu_tanh approximation: gelu(0)=0, gelu(1)≈0.8413, gelu(-1)≈-0.1587
    let got = run_unary_f32(&backend, OpKind::GeluElementwise, &[0.0, 1.0, -1.0]);
    assert_close(&got, &[0.0, 0.8413, -0.1587], 1e-3, 1e-3);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_relu_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_unary_f32(&backend, OpKind::ReluElementwise, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    assert_eq!(got, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_unary_step_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // step(x) = 1 if x > 0 else 0
    let got = run_unary_f32(&backend, OpKind::StepElementwise, &[-2.0, -0.5, 0.0, 0.5, 2.0]);
    // step at 0 — unary.slang's exact convention varies; check the
    // unambiguous values + relax 0.
    assert_eq!(got[0], 0.0);
    assert_eq!(got[1], 0.0);
    assert_eq!(got[3], 1.0);
    assert_eq!(got[4], 1.0);
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
