//! Live-CUDA tests for baracuda-kernels-sys-backed WriteSlice
//! (Phase 7.6 step 9c E.3.2.4). Verifies the byte-width-dispatched
//! kernel writes the right slab into the destination's in-place
//! buffer and leaves bytes outside the slab untouched.
//!
//! Coverage: f32 (b4) KV-cache append shape, f16 (b2) interior 2-D
//! slab, dispatch-table presence of all 9 dtype entries.

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

fn download<T: bytemuck::Pod + Copy>(s: &Storage) -> Vec<T> {
    match &s.inner {
        BackendStorage::Cuda(c) => {
            let bytes = c.to_cpu_bytes().expect("d2h");
            bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
        }
        _ => panic!("not on CUDA"),
    }
}

/// Canonical KV-cache append shape: source [1, 3, 2] written into
/// dest [4, 3, 2] at axis-0 row 2.
#[test]
#[ignore]
fn baracuda_write_slice_f32_kv_append() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    // dest: 4×3×2 zeros (KV cache buffer with all slots empty).
    let dest_host = vec![0.0_f32; 24];
    // source: 1×3×2 with values 1..6.
    let src_host = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let dest_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &dest_host)));
    let src_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src_host)));

    let alts = table.lookup_alternatives(
        OpKind::WriteSlice,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    assert!(!alts.is_empty(), "no WriteSlice CUDA alternatives");
    let kernel = alts[0].kernel;

    let params = OpParams::WriteSlice {
        dest_shape: vec![4, 3, 2],
        ranges: vec![(2, 3), (0, 3), (0, 2)],
        deferred_dyn_offset: None,
    };
    let inputs = vec![src_arc];
    let mut outputs = vec![dest_arc.clone()];
    kernel(&inputs, &mut outputs, &[], &params).expect("write_slice");

    let got = download::<f32>(&dest_arc.read().unwrap());
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    // row 0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    // row 1
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    // row 2 ← source
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    // row 3
    ];
    assert_eq!(got, expected);
}

/// Interior 2-D slab: source [2, 2] written into dest [3, 4] at
/// rows 0..2, columns 1..3. Verifies non-full slab along both axes
/// (the generic kernel path, not the contiguous-prefix fast path).
#[test]
#[ignore]
fn baracuda_write_slice_f32_interior_2d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let dest_host = vec![
        10.0_f32, 11.0, 12.0, 13.0,
        14.0,     15.0, 16.0, 17.0,
        18.0,     19.0, 20.0, 21.0,
    ];
    let src_host = vec![
        100.0_f32, 101.0,
        102.0,     103.0,
    ];

    let dest_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &dest_host)));
    let src_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src_host)));

    let alts = table.lookup_alternatives(
        OpKind::WriteSlice,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::WriteSlice {
        dest_shape: vec![3, 4],
        ranges: vec![(0, 2), (1, 3)],
        deferred_dyn_offset: None,
    };
    let inputs = vec![src_arc];
    let mut outputs = vec![dest_arc.clone()];
    kernel(&inputs, &mut outputs, &[], &params).expect("write_slice");

    let got = download::<f32>(&dest_arc.read().unwrap());
    let expected = vec![
        10.0, 100.0, 101.0, 13.0,
        14.0, 102.0, 103.0, 17.0,
        18.0, 19.0,  20.0,  21.0,
    ];
    assert_eq!(got, expected);
}

/// f16 (b2 byte width) — the BF16/F16 dtypes both route through
/// the b2 wrapper. Tests the byte-width dispatch correctness.
#[test]
#[ignore]
fn baracuda_write_slice_f16_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    // 5-element dest, write 2 elements at offset 2.
    let dest_host: Vec<half::f16> = (0..5).map(|i| half::f16::from_f32(i as f32)).collect();
    let src_host: Vec<half::f16> = vec![
        half::f16::from_f32(100.0),
        half::f16::from_f32(200.0),
    ];

    let dest_arc = Arc::new(RwLock::new(upload(&dev, DType::F16, &dest_host)));
    let src_arc = Arc::new(RwLock::new(upload(&dev, DType::F16, &src_host)));

    let alts = table.lookup_alternatives(
        OpKind::WriteSlice,
        &[DType::F16, DType::F16],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::WriteSlice {
        dest_shape: vec![5],
        ranges: vec![(2, 4)],
        deferred_dyn_offset: None,
    };
    let inputs = vec![src_arc];
    let mut outputs = vec![dest_arc.clone()];
    kernel(&inputs, &mut outputs, &[], &params).expect("write_slice");

    let got = download::<half::f16>(&dest_arc.read().unwrap());
    let expected: Vec<half::f16> = [0.0_f32, 1.0, 100.0, 200.0, 4.0]
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    assert_eq!(got, expected);
}

/// Dispatch-table sanity: WriteSlice is registered for all 9 dtypes
/// that fuel covers (F32/F64/F16/BF16/I32/I64/U32/U8/I8). No CUDA
/// device required.
#[test]
fn write_slice_registered_for_all_9_dtypes() {
    let table = dual_table();
    let dtypes = [
        DType::F32, DType::F64, DType::F16, DType::BF16,
        DType::I32, DType::I64, DType::U32, DType::U8, DType::I8,
    ];
    for dt in dtypes {
        let alts = table.lookup_alternatives(
            OpKind::WriteSlice,
            &[dt, dt],
            BackendId::Cuda,
        );
        assert!(
            !alts.is_empty(),
            "no WriteSlice CUDA registration for dtype {dt:?}",
        );
    }
}
