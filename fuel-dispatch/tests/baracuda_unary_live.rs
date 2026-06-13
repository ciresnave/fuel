//! Live-CUDA tests for baracuda-kernels-sys-backed unary elementwise
//! operations registered as sibling alternatives at
//! `(OpKind::*Elementwise, [dt, dt], BackendId::Cuda)` decision points.
//!
//! Each test:
//! 1. Builds a binding table with both the PTX path
//!    (`register_cuda_kernels`) and the baracuda path
//!    (`register_baracuda_cuda_kernels`).
//! 2. Pulls *all* alternatives at the relevant key via
//!    `KernelBindingTable::lookup_alternatives`.
//! 3. Asserts there are at least two alternatives (proving step 9a's
//!    append-on-register semantics work for primitive ops too).
//! 4. Picks the baracuda alternative by function-pointer identity
//!    and invokes it directly. Validates the result against an
//!    analytic reference.
//!
//! All tests are `#[ignore]`d so `cargo test --workspace` stays
//! GPU-free by default; run on this host with `cargo test -p
//! fuel-storage --features cuda --test baracuda_unary_live --
//! --ignored`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Result};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_memory::{BackendStorage, Storage};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

/// Upload f32 host data to CUDA and wrap it as a `Storage`.
fn upload_f32(dev: &CudaDevice, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

/// Read back an output `Storage` containing f32 bytes.
fn download_f32(s: &Storage) -> Vec<f32> {
    let bytes = match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    };
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

/// Pick a binding-table alternative by function-pointer identity.
/// Returns the alternative whose `KernelRef` is `expected`; panics
/// when no such alternative is registered (a test bug).
fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    expected: fuel_dispatch::KernelRef,
) -> fuel_dispatch::KernelRef {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // sole CUDA source for these unary ops; PTX path no longer
    // registers a duplicate. Test verifies baracuda KernelRef is
    // registered.
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

fn run_unary_f32(
    table: &KernelBindingTable,
    op: OpKind,
    expected: fuel_dispatch::KernelRef,
    input: &[f32],
) -> Result<Vec<f32>> {
    let dev = CudaDevice::new(0).expect("cuda");
    let src = upload_f32(&dev, input);
    let out_bytes = CudaStorageBytes::alloc(&dev, input.len() * 4)?;
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(table, op, &[DType::F32, DType::F32], expected);
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::None,
    )?;
    let guard = out_arc.read().unwrap();
    Ok(download_f32(&guard))
}

/// Construct a binding table that has BOTH the PTX path and the
/// baracuda path registered. The order matters: PTX first (legacy
/// `lookup` returns first-registered), baracuda second. Step 9a's
/// `lookup_alternatives` surfaces both for the route picker.
fn dual_table() -> KernelBindingTable {
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    table
}

#[test]
#[ignore]
fn baracuda_unary_neg_f32_runs_through_binding_table() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let got = run_unary_f32(
        &table,
        OpKind::NegElementwise,
        fuel_dispatch::baracuda_dispatch::unary::neg_f32,
        &[1.0_f32, -2.0, 3.0, -4.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![-1.0_f32, 2.0, -3.0, 4.0]);
}

#[test]
#[ignore]
fn baracuda_unary_abs_f32_runs_through_binding_table() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let got = run_unary_f32(
        &table,
        OpKind::AbsElementwise,
        fuel_dispatch::baracuda_dispatch::unary::abs_f32,
        &[1.0_f32, -2.0, 3.0, -4.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![1.0_f32, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore]
fn baracuda_unary_sqrt_f32_runs_through_binding_table() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let got = run_unary_f32(
        &table,
        OpKind::SqrtElementwise,
        fuel_dispatch::baracuda_dispatch::unary::sqrt_f32,
        &[1.0_f32, 4.0, 9.0, 16.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![1.0_f32, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore]
fn baracuda_unary_relu_f32_runs_through_binding_table() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let got = run_unary_f32(
        &table,
        OpKind::ReluElementwise,
        fuel_dispatch::baracuda_dispatch::unary::relu_f32,
        &[1.0_f32, -2.0, 0.0, 3.0],
    )
    .expect("kernel call");
    assert_eq!(got, vec![1.0_f32, 0.0, 0.0, 3.0]);
}

/// Verify the binding-table really holds multiple alternatives at one
/// key — step 9a's contract. Doesn't need GPU; runs without
/// `#[ignore]`.
#[test]
fn baracuda_is_sole_unary_source() {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // single source of truth for CUDA unary ops; the legacy PTX path
    // no longer registers duplicate alternatives.
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(
            OpKind::NegElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(
            OpKind::NegElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32 neg");
    assert_eq!(after, 1, "baracuda is the sole F32 neg source");
}
