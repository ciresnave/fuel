//! Live-CUDA tests for baracuda-kernels-sys-backed Affine
//! (`y = mul * x + add`) registered as sibling alternatives at
//! `(OpKind::Affine, [dt, dt], BackendId::Cuda)` decision points.
//!
//! Coverage: F32 (sibling to PTX path) + F64/F16/BF16/I32/I64/U8
//! (net-new CUDA dtypes — baracuda fills the gap; PTX path was f32
//! only). See `baracuda_unary_live.rs` for the dual-table + pick-by-
//! fn-pointer pattern.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_storage::{
    baracuda_dispatch::register_baracuda_cuda_kernels,
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
    BackendStorage, Storage,
};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

fn dual_table() -> KernelBindingTable {
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    table
}

fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    expected: fuel_storage::KernelRef,
) -> fuel_storage::KernelRef {
    let alternatives = table.lookup_alternatives(op, dtypes, BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "no alternatives at ({op:?}, {dtypes:?}, Cuda)",
    );
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

fn upload<T: bytemuck::Pod>(dev: &CudaDevice, dt: DType, host: &[T]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda), dt)
}

fn alloc_out(dev: &CudaDevice, dt: DType, n_elems: usize, elem_size: usize) -> Storage {
    let buf = CudaStorageBytes::alloc(dev, n_elems * elem_size).expect("alloc");
    Storage::new(BackendStorage::Cuda(buf), dt)
}

fn download_bytes(s: &Storage) -> Vec<u8> {
    match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    }
}

fn run_affine<T: bytemuck::Pod>(
    table: &KernelBindingTable,
    dt: DType,
    elem_size: usize,
    expected: fuel_storage::KernelRef,
    input: &[T],
    mul: f64,
    add: f64,
) -> Vec<u8> {
    let dev = CudaDevice::new(0).expect("cuda");
    let src = upload(&dev, dt, input);
    let out = alloc_out(&dev, dt, input.len(), elem_size);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(table, OpKind::Affine, &[dt, dt], expected);
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Affine { mul, add },
    )
    .expect("kernel call");
    download_bytes(&out_arc.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_affine_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let out = run_affine(
        &table,
        DType::F32,
        4,
        fuel_storage::baracuda_dispatch::affine::affine_f32,
        &input,
        2.0,
        1.0,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[3.0, 5.0, 7.0, 9.0]);
}

#[test]
#[ignore]
fn baracuda_affine_f64() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let out = run_affine(
        &table,
        DType::F64,
        8,
        fuel_storage::baracuda_dispatch::affine::affine_f64,
        &input,
        2.0,
        1.0,
    );
    let got: &[f64] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[3.0, 5.0, 7.0, 9.0]);
}

#[test]
#[ignore]
fn baracuda_affine_i32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<i32> = vec![1, 2, 3, 4];
    let out = run_affine(
        &table,
        DType::I32,
        4,
        fuel_storage::baracuda_dispatch::affine::affine_i32,
        &input,
        3.0,
        10.0,
    );
    let got: &[i32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[13, 16, 19, 22]);
}

#[test]
#[ignore]
fn baracuda_affine_i64() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<i64> = vec![1, 2, 3, 4];
    let out = run_affine(
        &table,
        DType::I64,
        8,
        fuel_storage::baracuda_dispatch::affine::affine_i64,
        &input,
        -2.0,
        100.0,
    );
    let got: &[i64] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[98, 96, 94, 92]);
}

#[test]
#[ignore]
fn baracuda_affine_u8() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<u8> = vec![1, 2, 3, 4];
    let out = run_affine(
        &table,
        DType::U8,
        1,
        fuel_storage::baracuda_dispatch::affine::affine_u8,
        &input,
        2.0,
        10.0,
    );
    assert_eq!(out, vec![12, 14, 16, 18]);
}

#[test]
fn baracuda_is_sole_affine_source() {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // single source of truth for CUDA Affine; PTX path stripped.
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(OpKind::Affine, &[DType::F32, DType::F32], BackendId::Cuda)
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(OpKind::Affine, &[DType::F32, DType::F32], BackendId::Cuda)
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32 affine");
    assert_eq!(after, 1, "baracuda is the sole F32 affine source");
}
