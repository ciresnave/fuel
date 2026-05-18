//! Live-CUDA tests for baracuda-kernels-sys-backed RMSNorm +
//! LayerNorm at Fuel's `*LastDim` shape.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Result};
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
    expected: fuel_storage::KernelRef,
) -> fuel_storage::KernelRef {
    let alternatives =
        table.lookup_alternatives(op, &[DType::F32, DType::F32], BackendId::Cuda);
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

fn run_norm_f32(
    op: OpKind,
    expected: fuel_storage::KernelRef,
    input: &[f32],
    outer_count: usize,
    last_dim: usize,
    eps: f64,
) -> Result<Vec<f32>> {
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let src = upload_f32(&dev, input);
    let out_bytes = CudaStorageBytes::alloc(&dev, input.len() * 4)?;
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(&table, op, expected);
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::NormLastDim {
            outer_count,
            last_dim,
            eps,
        },
    )?;
    let guard = out_arc.read().unwrap();
    Ok(download_f32(&guard))
}

/// Analytic RMSNorm reference for the [outer_count, last_dim] shape.
fn cpu_rms_norm(input: &[f32], outer: usize, last: usize, eps: f64) -> Vec<f32> {
    let mut out = vec![0.0_f32; input.len()];
    for o in 0..outer {
        let off = o * last;
        let mean_sq = (0..last).map(|i| input[off + i].powi(2)).sum::<f32>() / last as f32;
        let inv_rms = 1.0 / (mean_sq + eps as f32).sqrt();
        for i in 0..last {
            out[off + i] = input[off + i] * inv_rms;
        }
    }
    out
}

/// Analytic LayerNorm reference (no affine).
fn cpu_layer_norm(input: &[f32], outer: usize, last: usize, eps: f64) -> Vec<f32> {
    let mut out = vec![0.0_f32; input.len()];
    for o in 0..outer {
        let off = o * last;
        let mean: f32 = (0..last).map(|i| input[off + i]).sum::<f32>() / last as f32;
        let var: f32 =
            (0..last).map(|i| (input[off + i] - mean).powi(2)).sum::<f32>() / last as f32;
        let inv_std = 1.0 / (var + eps as f32).sqrt();
        for i in 0..last {
            out[off + i] = (input[off + i] - mean) * inv_std;
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= eps,
            "idx {i}: |{a} - {e}| = {diff} > {eps}",
        );
    }
}

#[test]
#[ignore]
fn baracuda_rms_norm_last_dim_f32_matches_reference() {
    if dev_or_skip().is_none() {
        return;
    }
    let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let expected = cpu_rms_norm(&input, 2, 4, 1e-5);
    let got = run_norm_f32(
        OpKind::RmsNormLastDim,
        fuel_storage::baracuda_dispatch::norm::rms_f32,
        &input,
        2,
        4,
        1e-5,
    )
    .expect("kernel call");
    assert_close(&got, &expected, 1e-4);
}

#[test]
#[ignore]
fn baracuda_layer_norm_last_dim_f32_matches_reference() {
    if dev_or_skip().is_none() {
        return;
    }
    let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let expected = cpu_layer_norm(&input, 2, 4, 1e-5);
    let got = run_norm_f32(
        OpKind::LayerNormLastDim,
        fuel_storage::baracuda_dispatch::norm::layer_f32,
        &input,
        2,
        4,
        1e-5,
    )
    .expect("kernel call");
    assert_close(&got, &expected, 1e-4);
}
