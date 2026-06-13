//! Live-CUDA tests for baracuda-kernels-sys-backed CumSum
//! (inclusive prefix sum along one axis).

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType};
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

/// 1-D cumsum: [1,2,3,4,5] → [1,3,6,10,15].
#[test]
#[ignore]
fn baracuda_cumsum_f32_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 5, 4)));

    let alts = table.lookup_alternatives(
        OpKind::CumSum,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::CumSum {
        outer_count: 1, dim_size: 5, inner_count: 1, axis: 0,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("cumsum");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

/// Middle-axis cumsum on a [2, 4, 2] tensor: each outer × inner block
/// gets its own running sum along the middle axis.
#[test]
#[ignore]
fn baracuda_cumsum_f32_3d_middle_axis() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![
        // outer 0
        1.0_f32, 1.0,    // dim 0
        2.0,     2.0,    // dim 1
        3.0,     3.0,    // dim 2
        4.0,     4.0,    // dim 3
        // outer 1
        10.0, 20.0,
        10.0, 20.0,
        10.0, 20.0,
        10.0, 20.0,
    ];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 16, 4)));

    let alts = table.lookup_alternatives(
        OpKind::CumSum,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::CumSum {
        outer_count: 2, dim_size: 4, inner_count: 2, axis: 1,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("cumsum");

    let got = download::<f32>(&out_arc.read().unwrap());
    let expected = vec![
        // outer 0 — inner-block (col 0, col 1) accumulating independently
        1.0, 1.0,
        3.0, 3.0,
        6.0, 6.0,
        10.0, 10.0,
        // outer 1
        10.0, 20.0,
        20.0, 40.0,
        30.0, 60.0,
        40.0, 80.0,
    ];
    assert_eq!(got, expected);
}

#[test]
fn cumsum_registered_for_4_float_dtypes() {
    let table = dual_table();
    for dt in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let alts = table.lookup_alternatives(
            OpKind::CumSum,
            &[dt, dt],
            BackendId::Cuda,
        );
        assert!(!alts.is_empty(), "no CumSum CUDA registration for dtype {dt:?}");
    }
}
