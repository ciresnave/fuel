//! Live-CUDA tests for baracuda-kernels-sys-backed binary
//! elementwise operations.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_ir::{dispatch::OpKind, probe::BackendId, DType, Result};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_memory::{BackendStorage, Storage};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

fn upload_f32(dev: &CudaDevice, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

fn download_f32(s: &Storage) -> Vec<f32> {
    let bytes = match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    };
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    expected: fuel_dispatch::KernelRef,
) -> fuel_dispatch::KernelRef {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // sole CUDA source for these binary ops; the legacy PTX path no
    // longer registers a duplicate alternative. Test still verifies
    // the baracuda KernelRef is registered.
    let alternatives = table.lookup_alternatives(op, dtypes, BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "expected ≥ 1 alternative at ({op:?}, {dtypes:?}, Cuda); got 0",
    );
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!(
        "expected baracuda KernelRef not found among {} alternatives",
        alternatives.len(),
    )
}

fn run_binary_f32(
    op: OpKind,
    expected: fuel_dispatch::KernelRef,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let lhs = upload_f32(&dev, a);
    let rhs = upload_f32(&dev, b);
    let out_bytes = CudaStorageBytes::alloc(&dev, a.len() * 4)?;
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);
    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(
        &table,
        op,
        &[DType::F32, DType::F32, DType::F32],
        expected,
    );
    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::None,
    )?;
    let guard = out_arc.read().unwrap();
    Ok(download_f32(&guard))
}

#[test]
#[ignore]
fn baracuda_binary_add_f32_runs_through_binding_table() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_binary_f32(
        OpKind::AddElementwise,
        fuel_dispatch::baracuda_dispatch::binary::add_f32,
        &[1.0_f32, 2.0, 3.0, 4.0],
        &[10.0_f32, 20.0, 30.0, 40.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![11.0_f32, 22.0, 33.0, 44.0]);
}

#[test]
#[ignore]
fn baracuda_binary_mul_f32_runs_through_binding_table() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_binary_f32(
        OpKind::MulElementwise,
        fuel_dispatch::baracuda_dispatch::binary::mul_f32,
        &[1.0_f32, 2.0, 3.0, 4.0],
        &[10.0_f32, 20.0, 30.0, 40.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![10.0_f32, 40.0, 90.0, 160.0]);
}

#[test]
#[ignore]
fn baracuda_binary_div_f32_runs_through_binding_table() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_binary_f32(
        OpKind::DivElementwise,
        fuel_dispatch::baracuda_dispatch::binary::div_f32,
        &[10.0_f32, 20.0, 30.0, 40.0],
        &[1.0_f32, 2.0, 3.0, 4.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![10.0_f32, 10.0, 10.0, 10.0]);
}

#[test]
#[ignore]
fn baracuda_binary_maximum_f32_runs_through_binding_table() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_binary_f32(
        OpKind::MaximumElementwise,
        fuel_dispatch::baracuda_dispatch::binary::maximum_f32,
        &[1.0_f32, 20.0, 3.0, 40.0],
        &[10.0_f32, 2.0, 30.0, 4.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![10.0_f32, 20.0, 30.0, 40.0]);
}

// ---------------------------------------------------------------------------
// Comparison family (T, T -> U8)
//
// These are the FIRST tests to actually EXECUTE a CUDA comparison kernel.
// Everything else in the registration chain — the 28-minute compile, the FKC
// corpus lint, the placement report — establishes that the kernels EXIST,
// that the contract is well-formed, and that `ge[F32,F32,U8]` RESOLVES. None
// of them runs one.
//
// The specific thing under test is the output-width split in
// `binary_run{,_into}`. Those drivers used a single `dtype_size_bytes` for two
// jobs: deriving element counts from INPUT byte lengths, and sizing the
// OUTPUT. That is correct only while every caller is arithmetic (in == out).
// A comparison is `T, T -> U8`, so the output buffer here is `n` bytes while
// the inputs are `4n` — if the split were wrong, the size check would reject
// this correctly-sized buffer, or the kernel would write past it.
// ---------------------------------------------------------------------------

fn upload_u8_out(dev: &CudaDevice, n: usize) -> Result<Storage> {
    // n BYTES, not n*4 — the whole point of the width split.
    let out_bytes = CudaStorageBytes::alloc(dev, n)?;
    Ok(Storage::new(BackendStorage::Cuda(out_bytes), DType::U8))
}

fn download_u8(s: &Storage) -> Vec<u8> {
    match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    }
}

fn run_compare_f32(
    kernel: fuel_dispatch::KernelRef,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<u8>> {
    let dev = CudaDevice::new(0).expect("cuda");
    let lhs = upload_f32(&dev, a);
    let rhs = upload_f32(&dev, b);
    let out = upload_u8_out(&dev, a.len())?;
    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));
    kernel(
        &[lhs_arc, rhs_arc],
        &mut [out_arc.clone()],
        &[],
        &OpParams::None,
    )?;
    let guard = out_arc.read().unwrap();
    Ok(download_u8(&guard))
}

/// `ge` is the node that kept `Op::PagedAttn`'s recipe off the device, so it
/// gets the full check: ordering both ways, equality (>= is inclusive), and
/// NaN.
#[test]
#[ignore]
fn baracuda_compare_ge_f32_matches_cpu_semantics() {
    if dev_or_skip().is_none() {
        return;
    }
    let a = [1.0_f32, 2.0, 3.0, 2.0, f32::NAN, 0.0];
    let b = [2.0_f32, 2.0, 1.0, 3.0, 1.0, -0.0];
    let got = run_compare_f32(
        fuel_dispatch::baracuda_dispatch::binary::ge_f32_u8,
        &a,
        &b,
    )
    .expect("ge kernel call");

    // Computed from IEEE-754 semantics, NOT from the kernel: 1>=2 false;
    // 2>=2 TRUE (inclusive); 3>=1 true; 2>=3 false; NaN>=1 FALSE (ordered
    // comparison with NaN is false, per both baracuda's documented behaviour
    // and fuel-cpu-backend's `compare.rs`); 0.0 >= -0.0 TRUE (IEEE-754 says
    // +0 == -0).
    assert_eq!(got, vec![0u8, 1, 1, 0, 0, 1], "ge disagrees with IEEE-754/CPU");

    // The kernel documents that it writes ONLY 0 and 1. A wrongly-sized output
    // buffer is the failure this whole change is about, and it would most
    // likely show up as garbage bytes rather than as a size error.
    assert!(got.iter().all(|&v| v == 0 || v == 1), "non-boolean byte in mask: {got:?}");
    assert_eq!(got.len(), a.len(), "output length must be n BYTES, not n*4");
}

/// `ne` is the one comparison that returns 1 on NaN. Pinning it here is what
/// makes the CUDA family's NaN behaviour a tested claim rather than a quote
/// from an FFI doc-comment.
#[test]
#[ignore]
fn baracuda_compare_ne_f32_returns_one_on_nan() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_compare_f32(
        fuel_dispatch::baracuda_dispatch::binary::ne_f32_u8,
        &[f32::NAN, 1.0, 1.0],
        &[f32::NAN, 1.0, 2.0],
    )
    .expect("ne kernel call");
    // NaN != NaN is TRUE (the ONE op that returns 1 on NaN) -> 1;
    // 1 != 1 -> 0; 1 != 2 -> 1. Matches fuel-cpu-backend and PyTorch.
    assert_eq!(got, vec![1u8, 0, 1], "ne NaN semantics diverge from CPU");
}

/// REGRESSION GUARD for the arithmetic family. The width split rewrote every
/// arithmetic call site to pass its element width twice; if that were wrong,
/// `add`/`sub`/`mul`/`div` on CUDA would be broken for every existing consumer
/// and no compile, lint, or placement check would notice. This executes one.
#[test]
#[ignore]
fn baracuda_binary_add_f32_still_correct_after_width_split() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_binary_f32(
        OpKind::AddElementwise,
        fuel_dispatch::baracuda_dispatch::binary::add_f32,
        &[1.0_f32, -2.5, 3.0, 1e30],
        &[10.0_f32, 20.0, -30.0, 1e30],
    )
    .expect("add kernel call");
    assert_eq!(got, vec![11.0_f32, 17.5, -27.0, 2e30]);
}
