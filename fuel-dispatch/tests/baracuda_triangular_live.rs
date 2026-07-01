//! Live-CUDA tests for baracuda-kernels-sys-backed Triu / Tril
//! (Phase 7.6 step 9c E.3.2.4). Verifies the per-dtype triangular
//! mask kernels produce the right output for a few canonical shapes.

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

/// Triu on a 3×3 matrix with diagonal=0 — keeps upper triangle
/// including the main diagonal. Expected:
///   [[1, 2, 3],
///    [0, 5, 6],
///    [0, 0, 9]]
#[test]
#[ignore]
fn baracuda_triu_f32_3x3_diag0() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![
        1.0_f32, 2.0, 3.0,
        4.0,     5.0, 6.0,
        7.0,     8.0, 9.0,
    ];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 9, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Triu,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    assert!(!alts.is_empty(), "no Triu CUDA registration");
    let kernel = alts[0].kernel;

    let params = OpParams::Triangular {
        batch_count: 1,
        rows: 3,
        cols: 3,
        diagonal: 0,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("triu");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![
        1.0, 2.0, 3.0,
        0.0, 5.0, 6.0,
        0.0, 0.0, 9.0,
    ]);
}

/// Tril on a 3×3 matrix with diagonal=0 — keeps lower triangle
/// including the main diagonal. Expected:
///   [[1, 0, 0],
///    [4, 5, 0],
///    [7, 8, 9]]
#[test]
#[ignore]
fn baracuda_tril_f32_3x3_diag0() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![
        1.0_f32, 2.0, 3.0,
        4.0,     5.0, 6.0,
        7.0,     8.0, 9.0,
    ];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 9, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Tril,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Triangular {
        batch_count: 1,
        rows: 3,
        cols: 3,
        diagonal: 0,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("tril");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![
        1.0, 0.0, 0.0,
        4.0, 5.0, 0.0,
        7.0, 8.0, 9.0,
    ]);
}

/// Triu with diagonal=1 — shifts the kept region one column to the
/// right (the main diagonal is dropped).
#[test]
#[ignore]
fn baracuda_triu_f32_3x3_diag1() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![
        1.0_f32, 2.0, 3.0,
        4.0,     5.0, 6.0,
        7.0,     8.0, 9.0,
    ];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 9, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Triu,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Triangular {
        batch_count: 1, rows: 3, cols: 3, diagonal: 1,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("triu");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![
        0.0, 2.0, 3.0,
        0.0, 0.0, 6.0,
        0.0, 0.0, 0.0,
    ]);
}

/// Triu on bf16 — verifies dtype dispatch picks the right kernel.
#[test]
#[ignore]
fn baracuda_triu_bf16_2x3_diag0() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::BF16, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::BF16, 6, 2)));

    let alts = table.lookup_alternatives(
        OpKind::Triu,
        &[DType::BF16, DType::BF16],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Triangular {
        batch_count: 1, rows: 2, cols: 3, diagonal: 0,
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("triu");

    let got = download::<half::bf16>(&out_arc.read().unwrap());
    let expected: Vec<half::bf16> = [1.0_f32, 2.0, 3.0, 0.0, 5.0, 6.0]
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    assert_eq!(got, expected);
}

/// Dispatch-table sanity: both Triu and Tril are registered for the
/// 6 dtype keys fuel covers. CPU-only — no device required.
#[test]
fn triangular_registered_for_all_6_dtypes() {
    let table = dual_table();
    let dtypes = [
        DType::F32, DType::F64, DType::F16, DType::BF16,
        DType::I32, DType::I64,
    ];
    for dt in dtypes {
        for op in [OpKind::Triu, OpKind::Tril] {
            let alts = table.lookup_alternatives(
                op,
                &[dt, dt],
                BackendId::Cuda,
            );
            assert!(
                !alts.is_empty(),
                "no {op:?} CUDA registration for dtype {dt:?}",
            );
        }
    }
}
