//! Live-CUDA tests for baracuda-kernels-sys-backed single-axis +
//! multi-axis reductions.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Layout, Result, Shape};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_storage::{BackendStorage, Storage};

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
    let alternatives = table.lookup_alternatives(op, dtypes, BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "expected ≥ 1 alternative at ({op:?}, {dtypes:?}, Cuda)",
    );
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

fn run_reduce_f32(
    op: OpKind,
    expected: fuel_dispatch::KernelRef,
    input: &[f32],
    in_shape: &[usize],
    dims: Vec<usize>,
    out_elem_count: usize,
) -> Result<Vec<f32>> {
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let src = upload_f32(&dev, input);
    let out_bytes = CudaStorageBytes::alloc(&dev, out_elem_count * 4)?;
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(&table, op, &[DType::F32, DType::F32], expected);
    let layout = Layout::contiguous(Shape::from_dims(in_shape));
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[layout],
        &OpParams::Reduce {
            dims,
            keepdim: false,
        },
    )?;
    let guard = out_arc.read().unwrap();
    Ok(download_f32(&guard))
}

#[test]
#[ignore]
fn baracuda_reduce_sum_f32_axis1_runs() {
    if dev_or_skip().is_none() {
        return;
    }
    // [[1, 2, 3], [4, 5, 6]] reduce axis 1 → [6, 15]
    let got = run_reduce_f32(
        OpKind::SumReduce,
        fuel_dispatch::baracuda_dispatch::reduce::sum_f32,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        vec![1],
        2,
    )
    .expect("kernel call");
    assert_eq!(got, vec![6.0_f32, 15.0]);
}

#[test]
#[ignore]
fn baracuda_reduce_max_f32_axis0_runs() {
    if dev_or_skip().is_none() {
        return;
    }
    // [[1, 5, 2], [4, 3, 6]] reduce axis 0 → [4, 5, 6]
    let got = run_reduce_f32(
        OpKind::MaxReduce,
        fuel_dispatch::baracuda_dispatch::reduce::max_f32,
        &[1.0_f32, 5.0, 2.0, 4.0, 3.0, 6.0],
        &[2, 3],
        vec![0],
        3,
    )
    .expect("kernel call");
    assert_eq!(got, vec![4.0_f32, 5.0, 6.0]);
}

#[test]
#[ignore]
fn baracuda_reduce_mean_f32_axis1_runs() {
    if dev_or_skip().is_none() {
        return;
    }
    let got = run_reduce_f32(
        OpKind::MeanReduce,
        fuel_dispatch::baracuda_dispatch::reduce::mean_f32,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        vec![1],
        2,
    )
    .expect("kernel call");
    assert_eq!(got, vec![2.0_f32, 5.0]);
}

#[test]
#[ignore]
fn baracuda_reduce_sum_f32_multi_axis_runs() {
    if dev_or_skip().is_none() {
        return;
    }
    // Multi-axis reduce: [[[1,2],[3,4]],[[5,6],[7,8]]] reduce axes
    // 1 and 2 → scalar per outer batch [10, 26]
    let got = run_reduce_f32(
        OpKind::SumReduce,
        fuel_dispatch::baracuda_dispatch::reduce::sum_f32,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 2],
        vec![1, 2],
        2,
    )
    .expect("kernel call");
    assert_eq!(got, vec![10.0_f32, 26.0]);
}
