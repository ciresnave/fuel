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

// ===========================================================================
// V.2.C — Softmax + RmsNorm last-dim (Fused ops, f32)
// ===========================================================================

/// Presence check: both ops register on `[F32, F32]` against Vulkan.
#[test]
fn vulkan_dispatch_softmax_norm_registered() {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let key = [DType::F32, DType::F32];
    for op in [OpKind::SoftmaxLastDim, OpKind::RmsNormLastDim, OpKind::Rope] {
        let alts = table.lookup_alternatives(op, &key, BackendId::Vulkan);
        assert_eq!(
            alts.len(), 1,
            "expected 1 Vulkan alternative for {op:?} f32 after register_vulkan_kernels, got {}",
            alts.len(),
        );
    }
}

/// Softmax row-wise: 2 rows × 4 cols. Each row should sum to ~1.0.
#[test]
#[ignore]
fn vulkan_dispatch_softmax_last_dim_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let outer = 2usize;
    let last = 4usize;
    let n = outer * last;
    let host: Vec<f32> = vec![
        // row 0
        1.0, 2.0, 3.0, 4.0,
        // row 1 — shift-invariant
        -1.0, 0.0, 1.0, 2.0,
    ];

    let in_storage = upload_f32(&backend, &host);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::SoftmaxLastDim,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[outer, last]));
    let layouts = vec![layout.clone(), layout];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::SoftmaxLastDim { outer_count: outer, last_dim: last },
    ).expect("softmax dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    // Row sums to ~1
    for row in 0..outer {
        let s: f32 = got[row * last .. (row + 1) * last].iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax row {row} sum {s} != 1.0");
    }
    // Both rows have the same shape (shift-invariant), so the two
    // rows should agree element-wise.
    for c in 0..last {
        let a = got[c];
        let b = got[last + c];
        assert!((a - b).abs() < 1e-5,
            "softmax shift-invariance broken at col {c}: row0={a} row1={b}");
    }
}

// ===========================================================================
// V.3.B — Cast (f32 ↔ f16, f32 ↔ bf16)
// ===========================================================================

#[test]
#[ignore]
fn vulkan_dispatch_cast_f32_to_bf16() {
    use half::bf16;

    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host_f32: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0, 1e-3, 1e3];
    let n = host_f32.len();

    let in_storage = upload_f32(&backend, &host_f32);
    let out_bytes_h = backend.alloc_bytes_handle(n * 2).expect("alloc bf16 out");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::BF16);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Cast,
            &[DType::F32, DType::BF16],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[n])),
        Layout::contiguous(Shape::from_dims(&[n])),
    ];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("cast f32→bf16 dispatch");

    // Download bf16 bytes and reinterpret.
    let raw = match &out_arc.read().unwrap().inner {
        BackendStorage::Vulkan(v) => backend.download_bytes(v).expect("d2h"),
        _ => panic!("not on Vulkan"),
    };
    let got: Vec<bf16> = bytemuck::cast_slice::<u8, bf16>(&raw).to_vec();
    for (i, (g, src)) in got.iter().zip(host_f32.iter()).enumerate() {
        let expected = bf16::from_f32(*src);
        // The kernel uses truncation (bits >> 16), not round-to-nearest;
        // bf16::from_f32 also truncates. Should be bit-identical.
        assert_eq!(g.to_bits(), expected.to_bits(),
            "cast[{i}]: f32={src}, got bf16 bits={:#x}, expected {:#x}",
            g.to_bits(), expected.to_bits());
    }
}

#[test]
#[ignore]
fn vulkan_dispatch_cast_bf16_to_f32() {
    use half::bf16;

    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host_bf16: Vec<bf16> = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(-2.5),
        bf16::from_f32(3.25),
        bf16::from_f32(0.0),
        bf16::from_f32(1e-3),
        bf16::from_f32(1e3),
    ];
    let n = host_bf16.len();

    let bf16_bytes: &[u8] = bytemuck::cast_slice(&host_bf16);
    let in_vk = backend.upload_bytes_handle(bf16_bytes).expect("upload bf16");
    let in_storage = Storage::new(BackendStorage::Vulkan(in_vk), DType::BF16);
    let out_bytes_h = backend.alloc_bytes_handle(n * 4).expect("alloc f32 out");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Cast,
            &[DType::BF16, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[n])),
        Layout::contiguous(Shape::from_dims(&[n])),
    ];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("cast bf16→f32 dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    for (i, (g, src)) in got.iter().zip(host_bf16.iter()).enumerate() {
        let expected = src.to_f32();
        assert!((g - expected).abs() < 1e-3,
            "cast[{i}]: bf16={}, got f32={g}, expected {expected}", src.to_f32());
    }
}

#[test]
#[ignore]
fn vulkan_dispatch_cast_f32_to_f16() {
    use half::f16;

    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host_f32: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0, 1e-3, 1e3];
    let n = host_f32.len();

    let in_storage = upload_f32(&backend, &host_f32);
    let out_bytes_h = backend.alloc_bytes_handle(n * 2).expect("alloc f16 out");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::F16);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Cast,
            &[DType::F32, DType::F16],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[n])),
        Layout::contiguous(Shape::from_dims(&[n])),
    ];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("cast f32→f16 dispatch");

    let raw = match &out_arc.read().unwrap().inner {
        BackendStorage::Vulkan(v) => backend.download_bytes(v).expect("d2h"),
        _ => panic!("not on Vulkan"),
    };
    let got: Vec<f16> = bytemuck::cast_slice::<u8, f16>(&raw).to_vec();
    for (i, (g, src)) in got.iter().zip(host_f32.iter()).enumerate() {
        let expected = f16::from_f32(*src);
        // f16 round-to-nearest-even; allow 1 ULP slack.
        let g_bits = g.to_bits() as i32;
        let e_bits = expected.to_bits() as i32;
        assert!((g_bits - e_bits).abs() <= 1,
            "cast[{i}]: f32={src}, got f16 bits={g_bits:#x}, expected {e_bits:#x}");
    }
}

#[test]
#[ignore]
fn vulkan_dispatch_cast_f16_to_f32() {
    use half::f16;

    let Some(backend) = backend_or_skip() else { return };
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host_f16: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(-2.5),
        f16::from_f32(3.25),
        f16::from_f32(0.0),
        f16::from_f32(1e-3),
        f16::from_f32(1e3),
    ];
    let n = host_f16.len();

    let f16_bytes: &[u8] = bytemuck::cast_slice(&host_f16);
    let in_vk = backend.upload_bytes_handle(f16_bytes).expect("upload f16");
    let in_storage = Storage::new(BackendStorage::Vulkan(in_vk), DType::F16);
    let out_bytes_h = backend.alloc_bytes_handle(n * 4).expect("alloc f32 out");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Cast,
            &[DType::F16, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[n])),
        Layout::contiguous(Shape::from_dims(&[n])),
    ];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::None,
    ).expect("cast f16→f32 dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    for (i, (g, src)) in got.iter().zip(host_f16.iter()).enumerate() {
        let expected = src.to_f32();
        // f16 → f32 is exact (no rounding); allow tight tolerance.
        assert!((g - expected).abs() < 1e-3,
            "cast[{i}]: f16={}, got f32={g}, expected {expected}", src.to_f32());
    }
}

/// PowI y = x^3 across mixed-sign inputs.
#[test]
#[ignore]
fn vulkan_dispatch_powi_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let n = host.len();

    let in_storage = upload_f32(&backend, &host);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::PowIElementwise,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[n]));
    let layouts = vec![layout.clone(), layout];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::PowI { exp: 3 },
    ).expect("powi dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    let expected: Vec<f32> = host.iter().map(|x| x.powi(3)).collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "powi at {i}: got {g}, expected {e}");
    }
}

/// PowI y = x^2 across the same inputs (hits the x*x fast path).
#[test]
#[ignore]
fn vulkan_dispatch_powi_squared_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host: Vec<f32> = vec![-2.5, -1.0, 0.0, 0.5, 1.5, 3.0];
    let n = host.len();

    let in_storage = upload_f32(&backend, &host);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::PowIElementwise,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[n]));
    let layouts = vec![layout.clone(), layout];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::PowI { exp: 2 },
    ).expect("powi dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    let expected: Vec<f32> = host.iter().map(|x| x * x).collect();
    assert_eq!(got, expected);
}

/// Clamp y = clamp(x, -1, 1) across 6 elements spanning the bounds.
#[test]
#[ignore]
fn vulkan_dispatch_clamp_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host: Vec<f32> = vec![-3.0, -1.5, -0.5, 0.5, 1.5, 3.0];
    let n = host.len();

    let in_storage = upload_f32(&backend, &host);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::ClampElementwise,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[n]));
    let layouts = vec![layout.clone(), layout];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Clamp { min: -1.0, max: 1.0 },
    ).expect("clamp dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    assert_eq!(got, vec![-1.0, -1.0, -0.5, 0.5, 1.0, 1.0]);
}

/// Affine y = 2*x + 3 across 6 elements.
#[test]
#[ignore]
fn vulkan_dispatch_affine_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n = host.len();

    let in_storage = upload_f32(&backend, &host);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Affine,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[n]));
    let layouts = vec![layout.clone(), layout];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Affine { mul: 2.0, add: 3.0 },
    ).expect("affine dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    let expected: Vec<f32> = host.iter().map(|x| 2.0 * x + 3.0).collect();
    assert_eq!(got, expected);
}

/// Helper for matmul: builds storages, dispatches, returns output.
fn run_matmul_f32(
    backend: &Arc<VulkanBackend>,
    lhs: &[f32], lhs_batch: &[usize],
    rhs: &[f32], rhs_batch: &[usize],
    m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let lhs_storage = upload_f32(backend, lhs);
    let rhs_storage = upload_f32(backend, rhs);
    let lhs_total: usize = lhs_batch.iter().product::<usize>().max(1);
    let out_n = lhs_total * m * n;
    let out_bytes = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let lhs_arc = Arc::new(RwLock::new(lhs_storage));
    let rhs_arc = Arc::new(RwLock::new(rhs_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));
    let kernel = table
        .lookup_alternatives(
            OpKind::MatMul,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let mut lhs_dims = lhs_batch.to_vec();
    lhs_dims.extend_from_slice(&[m, k]);
    let mut rhs_dims = rhs_batch.to_vec();
    rhs_dims.extend_from_slice(&[k, n]);
    let mut out_dims = lhs_batch.to_vec();
    out_dims.extend_from_slice(&[m, n]);
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&lhs_dims)),
        Layout::contiguous(Shape::from_dims(&rhs_dims)),
        Layout::contiguous(Shape::from_dims(&out_dims)),
    ];
    kernel(
        &[Arc::clone(&lhs_arc), Arc::clone(&rhs_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Matmul {
            lhs_batch_dims: lhs_batch.to_vec(),
            rhs_batch_dims: rhs_batch.to_vec(),
            m, n, k,
        },
    ).expect("matmul dispatch");
    download_f32(backend, &out_arc.read().unwrap())
}

/// Mixed-precision matmul f32 × bf16 → f32 (V.3.D). Small-M path.
#[test]
#[ignore]
fn vulkan_dispatch_matmul_f32_bf16_b_small_m() {
    use half::bf16;

    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    // A = [[1,2,3], [4,5,6]] (2×3 f32); B = [[1,2], [3,4], [5,6]] (3×2 bf16).
    // A @ B = [[22, 28], [49, 64]] — but bf16 has limited precision so
    //         results are approximate.
    let a_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_bf16: Vec<bf16> = vec![
        bf16::from_f32(1.0), bf16::from_f32(2.0),
        bf16::from_f32(3.0), bf16::from_f32(4.0),
        bf16::from_f32(5.0), bf16::from_f32(6.0),
    ];

    let a_bytes: &[u8] = bytemuck::cast_slice(&a_f32);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b_bf16);
    let a_vk = backend.upload_bytes_handle(a_bytes).expect("a upload");
    let b_vk = backend.upload_bytes_handle(b_bytes).expect("b upload");
    let a_storage = Storage::new(BackendStorage::Vulkan(a_vk), DType::F32);
    let b_storage = Storage::new(BackendStorage::Vulkan(b_vk), DType::BF16);
    let out_n = 2 * 2;
    let out_bytes_h = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::MatMul,
            &[DType::F32, DType::BF16, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[2, 3])),
        Layout::contiguous(Shape::from_dims(&[3, 2])),
        Layout::contiguous(Shape::from_dims(&[2, 2])),
    ];
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m: 2, n: 2, k: 3,
        },
    ).expect("mixed-bf16 matmul dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    let expected = vec![22.0_f32, 28.0, 49.0, 64.0];
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        // bf16 has ~7-bit mantissa; small integer values are exact but
        // the multiply-accumulate can accumulate small drift. Tolerance
        // of 0.5 covers worst-case bf16 rounding for these magnitudes.
        assert!((g - e).abs() < 0.5, "mixed-bf16 matmul [{i}]: got {g}, expected {e}");
    }
}

/// Mixed-precision matmul f32 × bf16 → f32 with m=16, n=16, k=16 to
/// hit the cooperative-matrix (tensor-core) path when available.
/// Falls through to matmul_tiled_bf16_b otherwise — either way the
/// math must match.
#[test]
#[ignore]
fn vulkan_dispatch_matmul_f32_bf16_b_coop_size() {
    use half::bf16;

    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let m = 16usize;
    let n = 16usize;
    let k = 16usize;

    // A[i, j] = 1 for all (deterministic; simplifies expected).
    // B[i, j] = j (column index). Then (A @ B)[i, j] = sum_k(1 * j) = k * j.
    let a_f32: Vec<f32> = vec![1.0; m * k];
    let mut b_bf16: Vec<bf16> = Vec::with_capacity(k * n);
    for _i in 0..k {
        for j in 0..n {
            b_bf16.push(bf16::from_f32(j as f32));
        }
    }

    let a_bytes: &[u8] = bytemuck::cast_slice(&a_f32);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b_bf16);
    let a_vk = backend.upload_bytes_handle(a_bytes).expect("a upload");
    let b_vk = backend.upload_bytes_handle(b_bytes).expect("b upload");
    let a_storage = Storage::new(BackendStorage::Vulkan(a_vk), DType::F32);
    let b_storage = Storage::new(BackendStorage::Vulkan(b_vk), DType::BF16);
    let out_n = m * n;
    let out_bytes_h = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::MatMul,
            &[DType::F32, DType::BF16, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[m, k])),
        Layout::contiguous(Shape::from_dims(&[k, n])),
        Layout::contiguous(Shape::from_dims(&[m, n])),
    ];
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m, n, k,
        },
    ).expect("mixed-bf16 coop-size matmul dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    for i in 0..m {
        for j in 0..n {
            let expected = (k * j) as f32;
            let g = got[i * n + j];
            assert!((g - expected).abs() < 0.5,
                "coop-size mixed-bf16 [{i}, {j}]: got {g}, expected {expected}");
        }
    }
}

/// Mixed-precision matmul f32 × bf16 → f32, m == 1 (matvec_bf16_b path).
#[test]
#[ignore]
fn vulkan_dispatch_matmul_f32_bf16_b_matvec() {
    use half::bf16;

    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let a_f32: Vec<f32> = vec![1.0, 2.0, 3.0]; // 1×3
    let b_bf16: Vec<bf16> = vec![
        bf16::from_f32(1.0), bf16::from_f32(2.0),
        bf16::from_f32(3.0), bf16::from_f32(4.0),
        bf16::from_f32(5.0), bf16::from_f32(6.0),
    ]; // 3×2

    let a_bytes: &[u8] = bytemuck::cast_slice(&a_f32);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b_bf16);
    let a_vk = backend.upload_bytes_handle(a_bytes).expect("a upload");
    let b_vk = backend.upload_bytes_handle(b_bytes).expect("b upload");
    let a_storage = Storage::new(BackendStorage::Vulkan(a_vk), DType::F32);
    let b_storage = Storage::new(BackendStorage::Vulkan(b_vk), DType::BF16);
    let out_n = 1 * 2;
    let out_bytes_h = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes_h), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::MatMul,
            &[DType::F32, DType::BF16, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[1, 3])),
        Layout::contiguous(Shape::from_dims(&[3, 2])),
        Layout::contiguous(Shape::from_dims(&[1, 2])),
    ];
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m: 1, n: 2, k: 3,
        },
    ).expect("mixed-bf16 matvec dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    let expected = vec![22.0_f32, 28.0];
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 0.5, "mixed-bf16 matvec [{i}]: got {g}, expected {e}");
    }
}

/// Matvec path: m == 1. 1×3 @ 3×2 → 1×2.
#[test]
#[ignore]
fn vulkan_dispatch_matmul_matvec_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // A = [[1, 2, 3]] (1×3); B = [[1, 2], [3, 4], [5, 6]] (3×2).
    // A @ B = [[1+6+15, 2+8+18]] = [[22, 28]].
    let got = run_matmul_f32(
        &backend,
        &[1.0, 2.0, 3.0], &[],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[],
        1, 2, 3,
    );
    assert_eq!(got, vec![22.0, 28.0]);
}

/// Small-M path: 2 ≤ m < 32. 2×3 @ 3×2 → 2×2.
#[test]
#[ignore]
fn vulkan_dispatch_matmul_small_m_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // A = [[1, 2, 3], [4, 5, 6]]; B = [[1, 2], [3, 4], [5, 6]].
    // Row 0: [22, 28]; Row 1: [4+15+30, 8+20+36] = [49, 64].
    let got = run_matmul_f32(
        &backend,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[],
        2, 2, 3,
    );
    assert_eq!(got, vec![22.0, 28.0, 49.0, 64.0]);
}

/// Tiled-M path: m >= 32. Build a 32×4 @ 4×3 with deterministic values.
#[test]
#[ignore]
fn vulkan_dispatch_matmul_tiled_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let m = 32usize;
    let n = 3usize;
    let k = 4usize;
    // A[i, j] = i; B[i, j] = j+1. Row sum across k is i * sum(j+1 for j..k=4)
    //                                          = i * (1+1+1+1) for each col? No.
    // Actually A[i,j] = i. So row i is [i,i,i,i]. B[i,j] = j+1.
    // (A @ B)[i, j] = sum_k(A[i,k] * B[k,j]) = sum_k(i * (j+1)) = i * k * (j+1).
    // Wait B[k,j] = j+1 (depends only on j). So sum_k(i * (j+1)) = i * k * (j+1) = i * 4 * (j+1).
    let mut a = Vec::with_capacity(m * k);
    for i in 0..m {
        for _j in 0..k {
            a.push(i as f32);
        }
    }
    let mut b = Vec::with_capacity(k * n);
    for _i in 0..k {
        for j in 0..n {
            b.push((j + 1) as f32);
        }
    }
    let got = run_matmul_f32(&backend, &a, &[], &b, &[], m, n, k);
    for i in 0..m {
        for j in 0..n {
            let expected = (i * k * (j + 1)) as f32;
            let g = got[i * n + j];
            assert!((g - expected).abs() < 1e-3,
                "tiled matmul [{i}, {j}]: got {g}, expected {expected}");
        }
    }
}

/// Batched matmul: B=2 batch heads. lhs [2, 2, 3] @ rhs [2, 3, 2] → [2, 2, 2].
#[test]
#[ignore]
fn vulkan_dispatch_matmul_batched_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // Batch 0: same as small_m_f32 → [22, 28, 49, 64]
    // Batch 1: A=[[1,1,1],[2,2,2]], B=[[1,1],[1,1],[1,1]] → [[3,3],[6,6]]
    let lhs: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  // batch 0
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0,  // batch 1
    ];
    let rhs: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  // batch 0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  // batch 1
    ];
    let got = run_matmul_f32(&backend, &lhs, &[2], &rhs, &[2], 2, 2, 3);
    assert_eq!(got, vec![
        22.0, 28.0, 49.0, 64.0,        // batch 0
        3.0, 3.0, 6.0, 6.0,            // batch 1
    ]);
}

/// GQA: lhs has 2× the batch heads of rhs. lhs[4,1,3] @ rhs[2,3,2] → [4,1,2].
#[test]
#[ignore]
fn vulkan_dispatch_matmul_gqa_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // 4 query heads share 2 key heads (GQA factor = 2).
    // Q[head_q] @ K[head_q / 2].
    let lhs: Vec<f32> = vec![
        1.0, 2.0, 3.0,    // q0
        4.0, 5.0, 6.0,    // q1
        1.0, 0.0, 0.0,    // q2
        0.0, 1.0, 0.0,    // q3
    ];
    let rhs: Vec<f32> = vec![
        // k0 (used by q0, q1): same B as small_m
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        // k1 (used by q2, q3): identity-like 3x2
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];
    let got = run_matmul_f32(&backend, &lhs, &[4], &rhs, &[2], 1, 2, 3);
    // q0 @ k0 = [22, 28] (same as 1x3 @ 3x2)
    // q1 @ k0 = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    // q2 @ k1 = [1*1+0*0+0*1, 1*0+0*1+0*1] = [1, 0]
    // q3 @ k1 = [0*1+1*0+0*1, 0*0+1*1+0*1] = [0, 1]
    assert_eq!(got, vec![22.0, 28.0, 49.0, 64.0, 1.0, 0.0, 0.0, 1.0]);
}

/// Concat N=3 along last dim: chains via one intermediate allocation.
#[test]
#[ignore]
fn vulkan_dispatch_concat_n3_along_last_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    // 3 inputs, each [2 rows]:
    //   a (2×2): [[1, 2], [3, 4]]
    //   b (2×3): [[10, 20, 30], [40, 50, 60]]
    //   c (2×1): [[100], [200]]
    // Concat along dim=1 → 2×6: [[1, 2, 10, 20, 30, 100],
    //                            [3, 4, 40, 50, 60, 200]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let c_data: Vec<f32> = vec![100.0, 200.0];

    let a_storage = upload_f32(&backend, &a_data);
    let b_storage = upload_f32(&backend, &b_data);
    let c_storage = upload_f32(&backend, &c_data);
    let out_bytes = backend.alloc_bytes_handle(12 * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let c_arc = Arc::new(RwLock::new(c_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Concat,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![
        Layout::contiguous(Shape::from_dims(&[2, 2])),
        Layout::contiguous(Shape::from_dims(&[2, 3])),
        Layout::contiguous(Shape::from_dims(&[2, 1])),
        Layout::contiguous(Shape::from_dims(&[2, 6])),
    ];
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc), Arc::clone(&c_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Concat {
            outer_count: 2,
            input_dim_sizes: vec![2, 3, 1],
            inner_count: 1,
        },
    ).expect("N=3 concat dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    assert_eq!(got, vec![
        1.0, 2.0, 10.0, 20.0, 30.0, 100.0,
        3.0, 4.0, 40.0, 50.0, 60.0, 200.0,
    ]);
}

/// Concat N=4 along the leading dim: tests the chain ping-pong.
#[test]
#[ignore]
fn vulkan_dispatch_concat_n4_along_leading_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    // 4 inputs, each shape [?, 2]:
    //   a (1×2): [[1, 2]]
    //   b (2×2): [[3, 4], [5, 6]]
    //   c (1×2): [[7, 8]]
    //   d (1×2): [[9, 10]]
    // Concat dim=0 → 5×2.
    let a = vec![1.0_f32, 2.0];
    let b = vec![3.0_f32, 4.0, 5.0, 6.0];
    let c = vec![7.0_f32, 8.0];
    let d = vec![9.0_f32, 10.0];

    let arc_for = |v: &[f32], dim0: usize| {
        let st = upload_f32(&backend, v);
        (Arc::new(RwLock::new(st)), Layout::contiguous(Shape::from_dims(&[dim0, 2])))
    };
    let (a_arc, a_l) = arc_for(&a, 1);
    let (b_arc, b_l) = arc_for(&b, 2);
    let (c_arc, c_l) = arc_for(&c, 1);
    let (d_arc, d_l) = arc_for(&d, 1);

    let out_n = 5 * 2;
    let out_bytes = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Concat,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layouts = vec![a_l, b_l, c_l, d_l, Layout::contiguous(Shape::from_dims(&[5, 2]))];
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc), Arc::clone(&c_arc), Arc::clone(&d_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Concat {
            outer_count: 1,
            input_dim_sizes: vec![1, 2, 1, 1],
            inner_count: 2,
        },
    ).expect("N=4 concat dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    assert_eq!(got, vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
        9.0, 10.0,
    ]);
}

/// Concat binary along last dim: [[1,2,3], [4,5,6]] + [[7,8], [9,10]]
/// → [[1,2,3,7,8], [4,5,6,9,10]].
#[test]
#[ignore]
fn vulkan_dispatch_concat_along_last_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];          // 2×2

    let a_storage = upload_f32(&backend, &a_data);
    let b_storage = upload_f32(&backend, &b_data);
    let out_n = 2 * 5;
    let out_bytes = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let a_arc = Arc::new(RwLock::new(a_storage));
    let b_arc = Arc::new(RwLock::new(b_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Concat,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let a_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
    let b_layout = Layout::contiguous(Shape::from_dims(&[2, 2]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[2, 5]));
    let layouts = vec![a_layout, b_layout, out_layout];
    // concat dim = 1 (last). outer_count = prod(dims[..1]) = 2; inner_count = 1.
    kernel(
        &[Arc::clone(&a_arc), Arc::clone(&b_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Concat {
            outer_count: 2,
            input_dim_sizes: vec![3, 2],
            inner_count: 1,
        },
    ).expect("concat dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    assert_eq!(got, vec![1.0, 2.0, 3.0, 7.0, 8.0,   4.0, 5.0, 6.0, 9.0, 10.0]);
}

/// Helper for V.2.D reduce tests — uploads `data` of shape `dims`,
/// dispatches `op` with the supplied reduce dims + keepdim flag, and
/// returns the downloaded result. Output buffer is sized to fit the
/// largest legal output (full-reduce = 1 elem; last-dim = n_rows).
fn run_reduce_f32(
    backend: &Arc<VulkanBackend>,
    op: OpKind,
    data: &[f32],
    dims: &[usize],
    reduce_dims: Vec<usize>,
) -> Vec<f32> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let in_storage = upload_f32(backend, data);
    // Output element count: 1 for full reduce; n_rows for last-dim
    // reduce (n_rows = product of dims[..rank-1]).
    let rank = dims.len();
    let out_elems = if reduce_dims.is_empty() || reduce_dims.len() == rank {
        1
    } else if reduce_dims.len() == 1 && reduce_dims[0] == rank - 1 {
        dims[..rank - 1].iter().product::<usize>().max(1)
    } else {
        panic!("test helper only supports full + last-dim reduce")
    };
    let out_bytes = backend.alloc_bytes_handle(out_elems * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));
    let kernel = table
        .lookup_alternatives(op, &[DType::F32, DType::F32], BackendId::Vulkan)[0]
        .kernel;
    let layout = Layout::contiguous(Shape::from_dims(dims));
    let out_layout = if reduce_dims.is_empty() || reduce_dims.len() == rank {
        Layout::contiguous(Shape::from_dims(&[1]))
    } else {
        Layout::contiguous(Shape::from_dims(&dims[..rank - 1]))
    };
    let layouts = vec![layout, out_layout];
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Reduce { dims: reduce_dims, keepdim: false },
    ).expect("reduce dispatch");
    download_f32(backend, &out_arc.read().unwrap())
}

#[test]
#[ignore]
fn vulkan_dispatch_reduce_sum_full_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_reduce_f32(
        &backend, OpKind::SumReduce,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        vec![0, 1],
    );
    assert_eq!(got.len(), 1);
    assert!((got[0] - 21.0).abs() < 1e-5, "sum got {got:?}");
}

#[test]
#[ignore]
fn vulkan_dispatch_reduce_max_full_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_reduce_f32(
        &backend, OpKind::MaxReduce,
        &[1.0, 7.0, 3.0, 4.0, 2.0, 6.0],
        &[2, 3],
        vec![0, 1],
    );
    assert_eq!(got, vec![7.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_reduce_min_full_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_reduce_f32(
        &backend, OpKind::MinReduce,
        &[5.0, 2.0, 3.0, -1.0, 4.0, 0.0],
        &[2, 3],
        vec![0, 1],
    );
    assert_eq!(got, vec![-1.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_reduce_mean_full_f32() {
    let Some(backend) = backend_or_skip() else { return };
    let got = run_reduce_f32(
        &backend, OpKind::MeanReduce,
        &[2.0, 4.0, 6.0, 8.0],
        &[2, 2],
        vec![0, 1],
    );
    assert_eq!(got.len(), 1);
    assert!((got[0] - 5.0).abs() < 1e-5, "mean got {got:?}");
}

#[test]
#[ignore]
fn vulkan_dispatch_reduce_mean_last_dim_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // 2 rows × 3 cols → per-row mean: [2, 5]
    let got = run_reduce_f32(
        &backend, OpKind::MeanReduce,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        vec![1],
    );
    assert_eq!(got, vec![2.0, 5.0]);
}

#[test]
#[ignore]
fn vulkan_dispatch_reduce_sum_last_dim_f32() {
    let Some(backend) = backend_or_skip() else { return };
    // 2 rows × 3 cols → per-row sum: [6, 15]
    let got = run_reduce_f32(
        &backend, OpKind::SumReduce,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        vec![1],
    );
    assert_eq!(got, vec![6.0, 15.0]);
}

/// IndexSelect along dim 0: pick rows 0, 2, 1 from a 4×3 matrix.
#[test]
#[ignore]
fn vulkan_dispatch_index_select_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    // src shape [4, 3] — rows: [1..3], [4..6], [7..9], [10..12].
    let src_data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let ids_data: Vec<u32> = vec![0, 2, 1];

    let src_storage = upload_f32(&backend, &src_data);
    let ids_bytes: &[u8] = bytemuck::cast_slice(&ids_data);
    let ids_vk = backend.upload_bytes_handle(ids_bytes).expect("ids upload");
    let ids_storage = Storage::new(BackendStorage::Vulkan(ids_vk), DType::U32);

    let outer_count = 1usize;        // dims before axis 0
    let source_dim_size = 4usize;    // src.dims[0]
    let n_indices = 3usize;          // ids.len()
    let inner_count = 3usize;        // dims after axis 0
    let out_n = outer_count * n_indices * inner_count;
    let out_bytes = backend.alloc_bytes_handle(out_n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let ids_arc = Arc::new(RwLock::new(ids_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::IndexSelect,
            &[DType::F32, DType::U32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let src_layout = Layout::contiguous(Shape::from_dims(&[4, 3]));
    let ids_layout = Layout::contiguous(Shape::from_dims(&[3]));
    let out_layout = Layout::contiguous(Shape::from_dims(&[3, 3]));
    let layouts = vec![src_layout, ids_layout, out_layout];
    kernel(
        &[Arc::clone(&src_arc), Arc::clone(&ids_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::IndexSelect { outer_count, source_dim_size, n_indices, inner_count },
    ).expect("index_select dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    // Row 0 → [1,2,3]; Row 2 → [7,8,9]; Row 1 → [4,5,6].
    assert_eq!(got, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
}

/// RoPE round-trip with `cos = [1,1,...]`, `sin = [0,0,...]` — should
/// be the identity (kernel emits `x` unchanged). Proves the 3-input
/// dispatch wiring + backend handle propagation work.
#[test]
#[ignore]
fn vulkan_dispatch_rope_identity_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    // x has shape [outer=1, seq=2, head_dim=4]. head_dim must be even.
    let outer = 1usize;
    let seq = 2usize;
    let head_dim = 4usize;

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0,   5.0, 6.0, 7.0, 8.0];
    // cos = 1, sin = 0 → identity rotation. Shape is [seq, head_dim]
    // (the kernel reads `cos[s, i]` for i in [0, head_dim), not just
    // the first half).
    let cos_data: Vec<f32> = vec![1.0; seq * head_dim];
    let sin_data: Vec<f32> = vec![0.0; seq * head_dim];

    let x_storage = upload_f32(&backend, &x_data);
    let cos_storage = upload_f32(&backend, &cos_data);
    let sin_storage = upload_f32(&backend, &sin_data);
    let n = outer * seq * head_dim;
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let x_arc = Arc::new(RwLock::new(x_storage));
    let cos_arc = Arc::new(RwLock::new(cos_storage));
    let sin_arc = Arc::new(RwLock::new(sin_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Rope,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let x_layout = Layout::contiguous(Shape::from_dims(&[outer, seq, head_dim]));
    let cos_layout = Layout::contiguous(Shape::from_dims(&[seq, head_dim]));
    let sin_layout = Layout::contiguous(Shape::from_dims(&[seq, head_dim]));
    let out_layout = x_layout.clone();
    let layouts = vec![x_layout, cos_layout, sin_layout, out_layout];
    kernel(
        &[Arc::clone(&x_arc), Arc::clone(&cos_arc), Arc::clone(&sin_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Rope { outer_count: outer, seq, head_dim },
    ).expect("rope dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    for (i, (g, x)) in got.iter().zip(x_data.iter()).enumerate() {
        assert!((g - x).abs() < 1e-5, "rope identity mismatch at {i}: got {g}, expected {x}");
    }
}

/// RoPE π/2 rotation: cos=0, sin=1. The kernel's rotate-half formula
/// then gives `out[i] = -x[i+h]` and `out[i+h] = x[i]`.
#[test]
#[ignore]
fn vulkan_dispatch_rope_quarter_rotation_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let outer = 1usize;
    let seq = 1usize;
    let head_dim = 4usize;
    let h = head_dim / 2;

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // (x0, x1, x2, x3); h=2 → pairs (1,3) and (2,4).
    let cos_data: Vec<f32> = vec![0.0; seq * head_dim];
    let sin_data: Vec<f32> = vec![1.0; seq * head_dim];

    let x_storage = upload_f32(&backend, &x_data);
    let cos_storage = upload_f32(&backend, &cos_data);
    let sin_storage = upload_f32(&backend, &sin_data);
    let n = outer * seq * head_dim;
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);

    let x_arc = Arc::new(RwLock::new(x_storage));
    let cos_arc = Arc::new(RwLock::new(cos_storage));
    let sin_arc = Arc::new(RwLock::new(sin_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::Rope,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let x_layout = Layout::contiguous(Shape::from_dims(&[outer, seq, head_dim]));
    let cs_layout = Layout::contiguous(Shape::from_dims(&[seq, head_dim]));
    let layouts = vec![x_layout.clone(), cs_layout.clone(), cs_layout, x_layout];
    kernel(
        &[Arc::clone(&x_arc), Arc::clone(&cos_arc), Arc::clone(&sin_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::Rope { outer_count: outer, seq, head_dim },
    ).expect("rope dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    // Per shader formula with cos=0, sin=1:
    //   out[i]   = x[i] * 0 - x[i+h] * 1 = -x[i+h]
    //   out[i+h] = x[i+h] * 0 + x[i] * 1 = x[i]
    let _ = h;
    let expected: Vec<f32> = vec![-3.0, -4.0, 1.0, 2.0];
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "rope π/2 mismatch at {i}: got {g}, expected {e}");
    }
}

/// RmsNorm row-wise: each row's RMS should be ~1.0 after normalization
/// (eps small relative to data). 2 rows × 4 cols.
#[test]
#[ignore]
fn vulkan_dispatch_rms_norm_last_dim_f32() {
    let Some(backend) = backend_or_skip() else { return };

    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);

    let outer = 2usize;
    let last = 4usize;
    let n = outer * last;
    let host: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
    ];

    let in_storage = upload_f32(&backend, &host);
    let out_bytes = backend.alloc_bytes_handle(n * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Vulkan(out_bytes), DType::F32);
    let in_arc = Arc::new(RwLock::new(in_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = table
        .lookup_alternatives(
            OpKind::RmsNormLastDim,
            &[DType::F32, DType::F32],
            BackendId::Vulkan,
        )[0]
    .kernel;
    let layout = Layout::contiguous(Shape::from_dims(&[outer, last]));
    let layouts = vec![layout.clone(), layout];
    let eps = 1e-6f64;
    kernel(
        &[Arc::clone(&in_arc)],
        &mut [Arc::clone(&out_arc)],
        &layouts,
        &OpParams::NormLastDim { outer_count: outer, last_dim: last, eps },
    ).expect("rmsnorm dispatch");

    let got = download_f32(&backend, &out_arc.read().unwrap());
    // Reference RmsNorm: y = x / sqrt(mean(x^2) + eps).
    for row in 0..outer {
        let xs = &host[row * last .. (row + 1) * last];
        let ys = &got[row * last .. (row + 1) * last];
        let mean_sq: f32 = xs.iter().map(|x| x * x).sum::<f32>() / last as f32;
        let scale = (mean_sq + eps as f32).sqrt();
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let expected = x / scale;
            assert!((y - expected).abs() < 1e-4,
                "rmsnorm row {row} col {i}: got {y}, expected {expected}");
        }
    }
}
