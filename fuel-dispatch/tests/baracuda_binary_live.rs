//! Live-CUDA tests for baracuda-kernels-sys-backed binary
//! elementwise operations.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Result};
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
