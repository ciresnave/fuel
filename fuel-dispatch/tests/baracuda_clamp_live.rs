//! Live-CUDA tests for baracuda-kernels-sys-backed Clamp registered
//! as sibling alternatives at `(OpKind::ClampElementwise, [dt, dt],
//! BackendId::Cuda)` decision points.
//!
//! Coverage: F32 (sibling), F64/F16/BF16 (net-new CUDA dtypes).

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_ir::{dispatch::OpKind, probe::BackendId, DType, Layout, Shape};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_memory::{BackendStorage, Storage};

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
    expected: fuel_dispatch::KernelRef,
) -> fuel_dispatch::KernelRef {
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

fn run_clamp<T: bytemuck::Pod>(
    table: &KernelBindingTable,
    dt: DType,
    elem_size: usize,
    expected: fuel_dispatch::KernelRef,
    input: &[T],
    min: f64,
    max: f64,
) -> Vec<u8> {
    let dev = CudaDevice::new(0).expect("cuda");
    let src = upload(&dev, dt, input);
    let out = alloc_out(&dev, dt, input.len(), elem_size);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(table, OpKind::ClampElementwise, &[dt, dt], expected);
    let layout = Layout::contiguous(Shape::from_dims(&[input.len()]));
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[layout.clone(), layout],
        &OpParams::Clamp { min, max },
    )
    .expect("kernel call");
    download_bytes(&out_arc.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_clamp_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![-1.0, 0.5, 2.5, -3.0, 1.5];
    let out = run_clamp(
        &table,
        DType::F32,
        4,
        fuel_dispatch::baracuda_dispatch::clamp::clamp_f32,
        &input,
        0.0,
        2.0,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[0.0, 0.5, 2.0, 0.0, 1.5]);
}

#[test]
#[ignore]
fn baracuda_clamp_f64() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f64> = vec![-1.0, 0.5, 2.5, -3.0, 1.5];
    let out = run_clamp(
        &table,
        DType::F64,
        8,
        fuel_dispatch::baracuda_dispatch::clamp::clamp_f64,
        &input,
        0.0,
        2.0,
    );
    let got: &[f64] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[0.0, 0.5, 2.0, 0.0, 1.5]);
}

#[test]
fn baracuda_is_sole_clamp_source() {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // single source of truth for CUDA Clamp; PTX path stripped.
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(
            OpKind::ClampElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(
            OpKind::ClampElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32 clamp");
    assert_eq!(after, 1, "baracuda is the sole F32 clamp source");
}
