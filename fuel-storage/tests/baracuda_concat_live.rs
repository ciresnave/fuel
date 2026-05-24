//! Live-CUDA tests for baracuda-kernels-sys-backed Concat
//! (N-ary via N-1 chained concat2 calls) registered as sibling
//! alternatives at `(OpKind::Concat, [dt, dt], BackendId::Cuda)`.
//!
//! Coverage: F32 (sibling to PTX), F64/F16/BF16 (net-new CUDA dtypes).

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

fn download<T: bytemuck::Pod + Copy>(s: &Storage) -> Vec<T> {
    match &s.inner {
        BackendStorage::Cuda(c) => {
            let bytes = c.to_cpu_bytes().expect("d2h");
            bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
        }
        _ => panic!("not on CUDA"),
    }
}

/// Run baracuda Concat through the binding table.
/// `inputs` carries the per-input host data (rank-3 reshape:
/// outer × dim × inner). `input_dim_sizes` holds each input's middle-
/// dim size.
fn run_concat_f32(
    table: &KernelBindingTable,
    inputs_host: &[&[f32]],
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
) -> Vec<f32> {
    let dev = CudaDevice::new(0).expect("cuda");
    let total_out_dim: usize = input_dim_sizes.iter().sum();
    let out_numel = outer_count * total_out_dim * inner_count;

    let in_storages: Vec<Arc<RwLock<Storage>>> = inputs_host
        .iter()
        .map(|h| Arc::new(RwLock::new(upload(&dev, DType::F32, h))))
        .collect();
    let out = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, out_numel, 4)));

    let kernel = pick_alt(
        table,
        OpKind::Concat,
        &[DType::F32, DType::F32],
        fuel_storage::baracuda_dispatch::concat::concat_f32,
    );
    let params = OpParams::Concat {
        outer_count,
        input_dim_sizes: input_dim_sizes.to_vec(),
        inner_count,
        // Helper always reshapes inputs to rank-3 [outer, dim, inner];
        // axis=1 is the concat dim in that reshape.
        axis: 1,
    };
    let inputs_clone: Vec<_> = in_storages.iter().cloned().collect();
    kernel(&inputs_clone, &mut [out.clone()], &[], &params).expect("kernel call");

    download::<f32>(&out.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_concat_f32_pair() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // Two rank-3 inputs [outer=2, dim, inner=2]; concat along dim.
    // a: [[[1,2],[3,4]], [[5,6],[7,8]]] (outer=2, dim=2, inner=2)
    // b: [[[9,10]], [[11,12]]]          (outer=2, dim=1, inner=2)
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: Vec<f32> = vec![9.0, 10.0, 11.0, 12.0];
    let got = run_concat_f32(&table, &[&a, &b], 2, &[2, 1], 2);
    // Expected layout after concat (outer × (a_dim + b_dim) × inner):
    //   outer 0: [1,2, 3,4, 9,10]
    //   outer 1: [5,6, 7,8, 11,12]
    assert_eq!(
        got,
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0],
    );
}

#[test]
#[ignore]
fn baracuda_concat_f32_n_3_chained() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // Three rank-1 inputs concatenated along the only dim.
    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0, 5.0];
    let c: Vec<f32> = vec![6.0];
    let got = run_concat_f32(&table, &[&a, &b, &c], 1, &[2, 3, 1], 1);
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
#[ignore]
fn baracuda_concat_f32_single_input_is_copy() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let got = run_concat_f32(&table, &[&a], 1, &[4], 1);
    assert_eq!(got, a);
}

#[test]
fn baracuda_is_sole_concat_source() {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // single source of truth for CUDA Concat; PTX path stripped.
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(OpKind::Concat, &[DType::F32, DType::F32], BackendId::Cuda)
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(OpKind::Concat, &[DType::F32, DType::F32], BackendId::Cuda)
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32 concat");
    assert_eq!(after, 1, "baracuda is the sole F32 concat source");
}
