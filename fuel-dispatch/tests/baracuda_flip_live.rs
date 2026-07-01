//! Live-CUDA tests for baracuda-kernels-sys-backed Flip.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_ir::{dispatch::OpKind, probe::BackendId, DType};
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

/// Single-axis 1-D flip: [1,2,3,4,5] → [5,4,3,2,1].
/// outer=1, dim=5, inner=1.
#[test]
#[ignore]
fn baracuda_flip_f32_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 5, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Flip,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Flip { outer_count: 1, dim_size: 5, inner_count: 1, axis: 0 };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("flip");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

/// Flip middle axis of a [2, 3, 2] tensor — outer=2, dim=3, inner=2.
/// Each "row" of length 3 gets reversed; the inner block of size 2
/// rides along.
#[test]
#[ignore]
fn baracuda_flip_f32_3d_middle_axis() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![
        // outer 0
        1.0_f32, 2.0,    // row 0
        3.0,     4.0,    // row 1
        5.0,     6.0,    // row 2
        // outer 1
        7.0,     8.0,
        9.0,     10.0,
        11.0,    12.0,
    ];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 12, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Flip,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Flip { outer_count: 2, dim_size: 3, inner_count: 2, axis: 1 };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("flip");

    let got = download::<f32>(&out_arc.read().unwrap());
    let expected = vec![
        // outer 0 — rows reversed
        5.0, 6.0,
        3.0, 4.0,
        1.0, 2.0,
        // outer 1 — rows reversed
        11.0, 12.0,
        9.0,  10.0,
        7.0,  8.0,
    ];
    assert_eq!(got, expected);
}

/// Flip on bf16 — dtype-dispatch sanity.
#[test]
#[ignore]
fn baracuda_flip_bf16_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::BF16, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::BF16, 4, 2)));

    let alts = table.lookup_alternatives(
        OpKind::Flip,
        &[DType::BF16, DType::BF16],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Flip { outer_count: 1, dim_size: 4, inner_count: 1, axis: 0 };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("flip");

    let got = download::<half::bf16>(&out_arc.read().unwrap());
    let expected: Vec<half::bf16> = [4.0_f32, 3.0, 2.0, 1.0]
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn flip_registered_for_4_float_dtypes() {
    let table = dual_table();
    for dt in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let alts = table.lookup_alternatives(
            OpKind::Flip,
            &[dt, dt],
            BackendId::Cuda,
        );
        assert!(!alts.is_empty(), "no Flip CUDA registration for dtype {dt:?}");
    }
}
