//! Live-CUDA tests for baracuda-kernels-sys-backed Pad + PadBackward.
//! Coverage: Constant/Reflect/Replicate forward + Constant backward.

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

/// 1-D Constant Pad: [1,2,3] padded (1, 2) with value 0 → [0,1,2,3,0,0].
#[test]
#[ignore]
fn baracuda_pad_const_f32_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 6, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Pad,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Pad {
        in_shape: vec![3],
        out_shape: vec![6],
        padding: vec![(1, 2)],
        mode_tag: 0,
        fill_bytes: 0.0_f32.to_le_bytes().to_vec(),
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("pad const");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
}

/// 1-D Reflect Pad: [1,2,3,4] padded (1, 2) → [2,1,2,3,4,3,2]
/// (PyTorch reflect, no edge duplication).
#[test]
#[ignore]
fn baracuda_pad_reflect_f32_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 7, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Pad,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Pad {
        in_shape: vec![4],
        out_shape: vec![7],
        padding: vec![(1, 2)],
        mode_tag: 1,
        fill_bytes: vec![],
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("pad reflect");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

/// 1-D Replicate Pad: [1,2,3] padded (2, 1) → [1,1,1,2,3,3].
#[test]
#[ignore]
fn baracuda_pad_replicate_f32_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 6, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Pad,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Pad {
        in_shape: vec![3],
        out_shape: vec![6],
        padding: vec![(2, 1)],
        mode_tag: 2,
        fill_bytes: vec![],
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("pad replicate");

    let got = download::<f32>(&out_arc.read().unwrap());
    assert_eq!(got, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
}

/// 2-D Constant Pad: [[1,2],[3,4]] padded ((1,1),(0,2)) with value 0
/// → [[0,0,0,0],[1,2,0,0],[3,4,0,0],[0,0,0,0]].
#[test]
#[ignore]
fn baracuda_pad_const_f32_2d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let src = vec![1.0_f32, 2.0, 3.0, 4.0];
    let in_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &src)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 16, 4)));

    let alts = table.lookup_alternatives(
        OpKind::Pad,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::Pad {
        in_shape: vec![2, 2],
        out_shape: vec![4, 4],
        padding: vec![(1, 1), (0, 2)],
        mode_tag: 0,
        fill_bytes: 0.0_f32.to_le_bytes().to_vec(),
    };
    kernel(&[in_arc], &mut [out_arc.clone()], &[], &params).expect("pad const 2d");

    let got = download::<f32>(&out_arc.read().unwrap());
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 2.0, 0.0, 0.0,
        3.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(got, expected);
}

/// PadBackward (Constant mode): gradient at pad-region is discarded.
/// Forward pad: x=[1,2,3] → y=[0,1,2,3,0,0] with padding (1,2).
/// dy=[10,20,30,40,50,60] → dx=[dy[1], dy[2], dy[3]] = [20,30,40].
#[test]
#[ignore]
fn baracuda_pad_backward_constant_f32_1d() {
    let Some(dev) = dev_or_skip() else { return };
    let table = dual_table();
    let dy = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let dy_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &dy)));
    let dx_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, 3, 4)));

    let alts = table.lookup_alternatives(
        OpKind::PadBackward,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    let kernel = alts[0].kernel;

    let params = OpParams::PadBackward {
        in_shape: vec![3],
        out_shape: vec![6],
        padding: vec![(1, 2)],
        mode_tag: 0,
    };
    kernel(&[dy_arc], &mut [dx_arc.clone()], &[], &params).expect("pad backward");

    let got = download::<f32>(&dx_arc.read().unwrap());
    assert_eq!(got, vec![20.0, 30.0, 40.0]);
}

#[test]
fn pad_registered_for_4_float_dtypes() {
    let table = dual_table();
    for dt in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        for op in [OpKind::Pad, OpKind::PadBackward] {
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
