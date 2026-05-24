//! Live-CUDA tests for baracuda-kernels-sys-backed Roll
//! (single-axis cyclic shift).

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

/// 1-D roll by +1: [1,2,3,4,5] → [5,1,2,3,4] (each element shifts
/// right by 1, last wraps around to the front).
#[test]
#[ignore]
fn baracuda_roll_f32_1d_shift_plus1() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 5, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Roll,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Roll {
        outer_count: 1, dim_size: 5, inner_count: 1, shift: 1, axis: 0,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("roll");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![5.0, 1.0, 2.0, 3.0, 4.0]);
}

/// 1-D roll by -2: [1,2,3,4,5] → [3,4,5,1,2] (each element shifts
/// left by 2; first two wrap around to the back).
#[test]
#[ignore]
fn baracuda_roll_f32_1d_shift_minus2() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 5, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Roll,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Roll {
        outer_count: 1, dim_size: 5, inner_count: 1, shift: -2, axis: 0,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("roll");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![3.0, 4.0, 5.0, 1.0, 2.0]);
}

/// Roll middle axis of a [2, 3, 2] tensor by +1.
#[test]
#[ignore]
fn baracuda_roll_f32_3d_middle_axis() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![
        // outer 0
        1.0_f32, 2.0,
        3.0,     4.0,
        5.0,     6.0,
        // outer 1
        7.0,     8.0,
        9.0,     10.0,
        11.0,    12.0,
    ];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 12, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Roll,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Roll {
        outer_count: 2, dim_size: 3, inner_count: 2, shift: 1, axis: 1,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("roll");

    let got = download::<f32>(&out_arc.read().unwrap());
    let expected = vec![
        // outer 0 — rows shifted by 1 (row 2 wraps to row 0)
        5.0, 6.0,
        1.0, 2.0,
        3.0, 4.0,
        // outer 1
        11.0, 12.0,
        7.0,  8.0,
        9.0,  10.0,
    ];
    assert_eq!(got, expected);
}

#[test]
fn roll_registered_for_4_float_dtypes() {
    let table = dual_table();
    for dt in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let alts = table.lookup_alternatives(
            OpKind::Roll,
            &[dt, dt],
            BackendId::Cuda,
        );
        assert!(!alts.is_empty(), "no Roll CUDA registration for dtype {dt:?}");
    }
}
