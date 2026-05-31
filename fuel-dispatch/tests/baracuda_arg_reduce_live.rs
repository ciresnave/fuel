//! Live-CUDA tests for baracuda-kernels-sys-backed ArgMaxDim /
//! ArgMinDim (U32 output, alpha.28) registered as sibling alternatives
//! at `(OpKind::Arg{Max,Min}Dim, [input_dt, U32], BackendId::Cuda)`.
//!
//! Coverage: F32 sibling to PTX `fast_arg{max,min}` + F64/F16/BF16
//! net-new (PTX path is F32 only).

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Layout, Shape};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_storage::{BackendStorage, Storage};

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

fn alloc_out(dev: &CudaDevice, n_elems: usize) -> Storage {
    let buf = CudaStorageBytes::alloc(dev, n_elems * 4).expect("alloc");
    Storage::new(BackendStorage::Cuda(buf), DType::U32)
}

fn download_u32(s: &Storage) -> Vec<u32> {
    match &s.inner {
        BackendStorage::Cuda(c) => {
            let bytes = c.to_cpu_bytes().expect("d2h");
            bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
        }
        _ => panic!("not on CUDA"),
    }
}

fn run_arg<T: bytemuck::Pod>(
    table: &KernelBindingTable,
    op: OpKind,
    dt: DType,
    expected: fuel_dispatch::KernelRef,
    input: &[T],
    shape: &[usize],
    dim: usize,
) -> Vec<u32> {
    let dev = CudaDevice::new(0).expect("cuda");
    let src = upload(&dev, dt, input);
    let out_numel: usize = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if i == dim { None } else { Some(d) })
        .product();
    let out = alloc_out(&dev, out_numel);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let layout = Layout::contiguous(Shape::from_dims(shape));
    let kernel = pick_alt(table, op, &[dt, DType::U32], expected);
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[layout],
        &OpParams::Reduce {
            dims: vec![dim],
            keepdim: false,
        },
    )
    .expect("kernel call");
    download_u32(&out_arc.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_argmax_dim_f32_last_axis() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // shape [2, 3] — argmax along axis 1.
    // row 0: [1, 5, 3] -> max at index 1
    // row 1: [4, 2, 6] -> max at index 2
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let got = run_arg(
        &table,
        OpKind::ArgMaxDim,
        DType::F32,
        fuel_dispatch::baracuda_dispatch::arg_reduce::argmax_dim_u32_f32,
        &input,
        &[2, 3],
        1,
    );
    assert_eq!(got, vec![1, 2]);
}

#[test]
#[ignore]
fn baracuda_argmin_dim_f32_last_axis() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let got = run_arg(
        &table,
        OpKind::ArgMinDim,
        DType::F32,
        fuel_dispatch::baracuda_dispatch::arg_reduce::argmin_dim_u32_f32,
        &input,
        &[2, 3],
        1,
    );
    assert_eq!(got, vec![0, 1]);
}

#[test]
#[ignore]
fn baracuda_argmax_dim_f32_first_axis() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // shape [3, 2]; argmax along axis 0.
    // column 0: rows = [1, 4, 2] -> max at row 1
    // column 1: rows = [5, 2, 6] -> max at row 2
    let input: Vec<f32> = vec![1.0, 5.0, 4.0, 2.0, 2.0, 6.0];
    let got = run_arg(
        &table,
        OpKind::ArgMaxDim,
        DType::F32,
        fuel_dispatch::baracuda_dispatch::arg_reduce::argmax_dim_u32_f32,
        &input,
        &[3, 2],
        0,
    );
    assert_eq!(got, vec![1, 2]);
}

#[test]
#[ignore]
fn baracuda_argmax_dim_f64_ties_break_to_smallest_index() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // shape [2, 3]; argmax along axis 1, ties everywhere.
    let input: Vec<f64> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let got = run_arg(
        &table,
        OpKind::ArgMaxDim,
        DType::F64,
        fuel_dispatch::baracuda_dispatch::arg_reduce::argmax_dim_u32_f64,
        &input,
        &[2, 3],
        1,
    );
    // baracuda's tie-break: first-occurrence (smallest index) wins.
    assert_eq!(got, vec![0, 0]);
}

#[test]
fn baracuda_is_sole_argmax_source() {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // single source of truth for CUDA ArgMaxDim; PTX path stripped.
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(
            OpKind::ArgMaxDim,
            &[DType::F32, DType::U32],
            BackendId::Cuda,
        )
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(
            OpKind::ArgMaxDim,
            &[DType::F32, DType::U32],
            BackendId::Cuda,
        )
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32 argmax");
    assert_eq!(after, 1, "baracuda is the sole F32 argmax source");
}
