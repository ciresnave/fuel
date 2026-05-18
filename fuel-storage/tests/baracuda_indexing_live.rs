//! Live-CUDA tests for baracuda-kernels-sys-backed indexing ops.

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

fn upload_f32(dev: &CudaDevice, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

fn upload_u32(dev: &CudaDevice, host: &[u32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::U32)
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
    expected: fuel_storage::KernelRef,
) -> fuel_storage::KernelRef {
    let alternatives = table.lookup_alternatives(op, dtypes, BackendId::Cuda);
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

#[test]
#[ignore]
fn baracuda_index_select_f32_picks_rows() {
    if dev_or_skip().is_none() {
        return;
    }
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // Source: 4 rows × 3 inner-cols.
    //   row 0: [10, 11, 12]
    //   row 1: [20, 21, 22]
    //   row 2: [30, 31, 32]
    //   row 3: [40, 41, 42]
    let source: Vec<f32> = vec![
        10.0, 11.0, 12.0,
        20.0, 21.0, 22.0,
        30.0, 31.0, 32.0,
        40.0, 41.0, 42.0,
    ];
    // Indices: pick rows [2, 0, 3] in that order.
    let indices: Vec<u32> = vec![2, 0, 3];
    // Expected: [30, 31, 32, 10, 11, 12, 40, 41, 42] (3 selected rows × 3 cols).
    let expected: Vec<f32> = vec![
        30.0, 31.0, 32.0,
        10.0, 11.0, 12.0,
        40.0, 41.0, 42.0,
    ];

    let src_storage = upload_f32(&dev, &source);
    let idx_storage = upload_u32(&dev, &indices);
    let out_bytes = CudaStorageBytes::alloc(&dev, expected.len() * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let idx_arc = Arc::new(RwLock::new(idx_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = pick_alt(
        &table,
        OpKind::IndexSelect,
        &[DType::F32, DType::U32, DType::F32],
        fuel_storage::baracuda_dispatch::indexing::index_select_f32,
    );

    kernel(
        &[src_arc.clone(), idx_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::IndexSelect {
            outer_count: 1,
            source_dim_size: 4,
            n_indices: 3,
            inner_count: 3,
        },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());
    assert_eq!(got, expected);
}

fn upload_u8(dev: &CudaDevice, host: &[u8]) -> Storage {
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, host).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::U8)
}

#[test]
#[ignore]
fn baracuda_masked_fill_f32_replaces_masked_positions() {
    if dev_or_skip().is_none() {
        return;
    }
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // src: [1, 2, 3, 4, 5, 6]
    // mask: [0, 1, 0, 1, 0, 1]  (nonzero = "replace")
    // fill = -99.0_f32
    // expected: [1, -99, 3, -99, 5, -99]
    let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask: Vec<u8> = vec![0, 1, 0, 1, 0, 1];
    let fill_value: f32 = -99.0;
    let fill_bytes = fill_value.to_le_bytes().to_vec();
    let expected: Vec<f32> = vec![1.0, -99.0, 3.0, -99.0, 5.0, -99.0];

    let src_storage = upload_f32(&dev, &src);
    let mask_storage = upload_u8(&dev, &mask);
    let out_bytes = CudaStorageBytes::alloc(&dev, src.len() * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let mask_arc = Arc::new(RwLock::new(mask_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = pick_alt(
        &table,
        OpKind::MaskedFill,
        &[DType::F32, DType::U8, DType::F32],
        fuel_storage::baracuda_dispatch::indexing::masked_fill_f32,
    );

    kernel(
        &[src_arc.clone(), mask_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::MaskedFill { fill_bytes },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());
    assert_eq!(got, expected);
}

#[test]
#[ignore]
fn baracuda_index_select_f32_with_outer_batch() {
    if dev_or_skip().is_none() {
        return;
    }
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // Source: 2 outer batches × 3 select rows × 2 inner cols.
    //   batch 0: row 0 = [10, 11], row 1 = [20, 21], row 2 = [30, 31]
    //   batch 1: row 0 = [40, 41], row 1 = [50, 51], row 2 = [60, 61]
    let source: Vec<f32> = vec![
        10.0, 11.0, 20.0, 21.0, 30.0, 31.0,
        40.0, 41.0, 50.0, 51.0, 60.0, 61.0,
    ];
    let indices: Vec<u32> = vec![2, 0];
    // Each batch picks rows [2, 0]:
    //   batch 0: [30, 31, 10, 11]
    //   batch 1: [60, 61, 40, 41]
    let expected: Vec<f32> = vec![
        30.0, 31.0, 10.0, 11.0,
        60.0, 61.0, 40.0, 41.0,
    ];

    let src_storage = upload_f32(&dev, &source);
    let idx_storage = upload_u32(&dev, &indices);
    let out_bytes = CudaStorageBytes::alloc(&dev, expected.len() * 4).expect("alloc");
    let out_storage = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src_storage));
    let idx_arc = Arc::new(RwLock::new(idx_storage));
    let out_arc = Arc::new(RwLock::new(out_storage));

    let kernel = pick_alt(
        &table,
        OpKind::IndexSelect,
        &[DType::F32, DType::U32, DType::F32],
        fuel_storage::baracuda_dispatch::indexing::index_select_f32,
    );

    kernel(
        &[src_arc.clone(), idx_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::IndexSelect {
            outer_count: 2,
            source_dim_size: 3,
            n_indices: 2,
            inner_count: 2,
        },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());
    assert_eq!(got, expected);
}
