//! Live-CUDA tests for baracuda-kernels-sys-backed PowI registered as
//! sibling alternatives at `(OpKind::PowIElementwise, [dt, dt],
//! BackendId::Cuda)` decision points.
//!
//! Coverage: F32 (sibling to PTX), F64/F16/BF16 (net-new CUDA dtypes
//! — PTX path is F32 only).

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

fn alloc_out(dev: &CudaDevice, dt: DType, n_elems: usize, elem_size: usize) -> Storage {
    let buf = CudaStorageBytes::alloc(dev, n_elems * elem_size).expect("alloc");
    Storage::new(BackendStorage::Cuda(buf), dt)
}

fn download_bytes(s: &Storage) -> Vec<u8> {
    match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    }
}

fn run_powi<T: bytemuck::Pod>(
    table: &KernelBindingTable,
    dt: DType,
    elem_size: usize,
    expected: fuel_dispatch::KernelRef,
    input: &[T],
    exp: i32,
) -> Vec<u8> {
    let dev = CudaDevice::new(0).expect("cuda");
    let src = upload(&dev, dt, input);
    let out = alloc_out(&dev, dt, input.len(), elem_size);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(table, OpKind::PowIElementwise, &[dt, dt], expected);
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::PowI { exp },
    )
    .expect("kernel call");
    download_bytes(&out_arc.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_powi_f32_exp_3() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![1.0, 2.0, -3.0, 4.0, -5.0];
    let out = run_powi(
        &table,
        DType::F32,
        4,
        fuel_dispatch::baracuda_dispatch::powi::powi_f32,
        &input,
        3,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    // Power-by-squaring preserves sign on odd exponents.
    assert_eq!(got, &[1.0, 8.0, -27.0, 64.0, -125.0]);
}

#[test]
#[ignore]
fn baracuda_powi_f32_exp_neg_2() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![1.0, 2.0, 4.0];
    let out = run_powi(
        &table,
        DType::F32,
        4,
        fuel_dispatch::baracuda_dispatch::powi::powi_f32,
        &input,
        -2,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    // x^(-2) = 1 / x^2.
    for (g, want) in got.iter().zip(&[1.0_f32, 0.25, 0.0625]) {
        assert!((g - want).abs() < 1e-6, "got {g}, want {want}");
    }
}

#[test]
#[ignore]
fn baracuda_powi_f64_exp_4() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f64> = vec![1.0, 2.0, -3.0, 4.0];
    let out = run_powi(
        &table,
        DType::F64,
        8,
        fuel_dispatch::baracuda_dispatch::powi::powi_f64,
        &input,
        4,
    );
    let got: &[f64] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[1.0, 16.0, 81.0, 256.0]);
}

#[test]
#[ignore]
fn baracuda_powi_f32_exp_0_returns_ones() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![0.0, 1.5, -7.0, 100.0];
    let out = run_powi(
        &table,
        DType::F32,
        4,
        fuel_dispatch::baracuda_dispatch::powi::powi_f32,
        &input,
        0,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    // x^0 = 1 for every x. baracuda's power-by-squaring returns 1 for n=0.
    assert_eq!(got, &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
#[ignore]
fn baracuda_powi_backward_f32_exp_3() {
    // grad_x = 3 · x^2 · upstream
    let Some(_dev) = dev_or_skip() else { return };
    let dev = CudaDevice::new(0).expect("cuda");
    let table = dual_table();
    let x: Vec<f32> = vec![1.0, 2.0, -3.0, 4.0, -5.0];
    let upstream: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let x_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &x)));
    let up_arc = Arc::new(RwLock::new(upload(&dev, DType::F32, &upstream)));
    let out_arc = Arc::new(RwLock::new(alloc_out(&dev, DType::F32, x.len(), 4)));

    let kernel = pick_alt(
        &table,
        OpKind::PowIElementwiseBackward,
        &[DType::F32, DType::F32, DType::F32],
        fuel_dispatch::baracuda_dispatch::powi_backward::powi_backward_f32,
    );
    let layout = fuel_ir::Layout::contiguous(
        fuel_ir::Shape::from_dims(&[x.len()]),
    );
    kernel(
        &[x_arc, up_arc],
        &mut [out_arc.clone()],
        &[layout.clone(), layout],
        &OpParams::PowI { exp: 3 },
    ).expect("powi_backward dispatch");

    let bytes = download_bytes(&out_arc.read().unwrap());
    let got: &[f32] = bytemuck::cast_slice(&bytes);
    // 3 · x^2 for each input, upstream = 1.
    let want: Vec<f32> = x.iter().map(|&v| 3.0 * v * v).collect();
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
    }
}

#[test]
fn baracuda_is_sole_powi_source() {
    // Post-fuel-cuda-kernels-cleanup (2026-05-25): baracuda is the
    // single source of truth for CUDA PowI; the legacy PTX path no
    // longer registers a duplicate alternative.
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(
            OpKind::PowIElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(
            OpKind::PowIElementwise,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32 powi");
    assert_eq!(after, 1, "baracuda is the sole F32 powi source");
}
