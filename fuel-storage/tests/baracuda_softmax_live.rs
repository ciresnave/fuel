//! Live-CUDA tests for baracuda-kernels-sys-backed softmax /
//! log-softmax at Fuel's `*LastDim` shape.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Layout, Result, Shape};
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

/// Numerically-stable softmax reference.
fn cpu_softmax(input: &[f32], outer: usize, last: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; input.len()];
    for o in 0..outer {
        let off = o * last;
        let mx = input[off..off + last]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for i in 0..last {
            let e = (input[off + i] - mx).exp();
            out[off + i] = e;
            sum += e;
        }
        for i in 0..last {
            out[off + i] /= sum;
        }
    }
    out
}

fn cpu_log_softmax(input: &[f32], outer: usize, last: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; input.len()];
    for o in 0..outer {
        let off = o * last;
        let mx = input[off..off + last]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = (0..last).map(|i| (input[off + i] - mx).exp()).sum();
        let log_sum = sum_exp.ln();
        for i in 0..last {
            out[off + i] = input[off + i] - mx - log_sum;
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(diff <= eps, "idx {i}: |{a} - {e}| = {diff} > {eps}");
    }
}

fn run_softmax_f32(
    op: OpKind,
    expected_fn: fuel_storage::KernelRef,
    op_params: OpParams,
    input: &[f32],
    outer_count: usize,
    last_dim: usize,
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
    let kernel = pick_alt(&table, op, expected_fn);
    let layout = Layout::contiguous(Shape::from_dims(&[outer_count, last_dim]));
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[layout.clone(), layout],
        &op_params,
    )?;
    Ok(download_f32(&out_arc.read().unwrap()))
}

#[test]
#[ignore]
fn baracuda_softmax_last_dim_f32_matches_reference() {
    if dev_or_skip().is_none() {
        return;
    }
    let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let expected = cpu_softmax(&input, 2, 4);
    let got = run_softmax_f32(
        OpKind::SoftmaxLastDim,
        fuel_storage::baracuda_dispatch::softmax::softmax_f32,
        OpParams::SoftmaxLastDim {
            outer_count: 2,
            last_dim: 4,
        },
        &input,
        2,
        4,
    )
    .expect("kernel call");
    assert_close(&got, &expected, 1e-5);
}

#[test]
#[ignore]
fn baracuda_log_softmax_last_dim_f32_matches_reference() {
    if dev_or_skip().is_none() {
        return;
    }
    let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let expected = cpu_log_softmax(&input, 2, 4);
    let got = run_softmax_f32(
        OpKind::LogSoftmaxLastDim,
        fuel_storage::baracuda_dispatch::softmax::log_softmax_f32,
        OpParams::LogSoftmaxLastDim {
            outer_count: 2,
            last_dim: 4,
        },
        &input,
        2,
        4,
    )
    .expect("kernel call");
    assert_close(&got, &expected, 1e-5);
}
