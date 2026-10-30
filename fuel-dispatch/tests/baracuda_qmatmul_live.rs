// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-CUDA test for the `OpKind::QMatMul` binding (ROADMAP row 105): the contract-registered
//! wrapper against an f64 reference built from the dequantized weights.
//!
//! Cases cover the three routes of `baracuda_dispatch::qmatmul::qmatmul_f32`: one row (baracuda
//! `mmvq_<fmt>`), many rows (`mmvq_rows_<fmt>`), and the host fallback (a `k` off the format's read
//! chunk). Formats: Q4_0 and Q8_0 (the row's target) and Q4_K (a 256-column format).
//!
//! The reference dequantizes with `fuel_quantized`'s `to_float` and accumulates in f64, so it shares
//! no code with the kernels, so the comparison is a tolerance (the reduction order is scheduler-
//! dependent), not a bit-match, and a sibling check shows the tolerance rejects a wrong answer.
//! Measured on the RTX 4070 the device routes agree with the f64 reference to about 1e-7 of the
//! largest output (the baracuda kernels take f32 activations); the host fallback, whose CPU kernel
//! quantizes the activations to 8 bits, is off by 6e-4 to 8e-4. The limits below sit between.
//!
//! Not `#[ignore]`d: the file is `cfg(feature = "cuda")`, so no CI leg builds it, and
//! `require_gpu_run_lock` plus `CudaDevice::new` fail loudly without the wrapper or a device.
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-dispatch --features cuda
//! --test baracuda_qmatmul_live`.

#![cfg(feature = "cuda")]

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{
    baracuda_dispatch::{qmatmul, register_baracuda_cuda_kernels},
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_graph::QuantType;
use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_quantized::{BlockQ4_0, BlockQ4K, BlockQ8_0, k_quants::GgmlType};

const KEY: [DType; 3] = [DType::F32, DType::U32, DType::F32];

fn upload(dev: &CudaDevice, bytes: &[u8], dt: DType) -> Arc<RwLock<Storage>> {
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(cuda), dt)))
}

/// Deterministic, non-degenerate f32 data in roughly [-1, 1].
fn data(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 31 + salt * 17) % 97) as f32 - 48.0) / 48.0)
        .collect()
}

/// Quantize `n` rows of `k` weights; returns the block bytes and the dequantized f32 weights.
fn quantize<T: GgmlType>(n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let w = data(n * k, 3);
    let mut blocks = vec![T::zeros(); n * k / T::BLCK_SIZE];
    T::from_float(&w, &mut blocks);
    let mut deq = vec![0.0f32; n * k];
    T::to_float(&blocks, &mut deq);
    (bytemuck::cast_slice(&blocks).to_vec(), deq)
}

struct Run {
    got: Vec<f32>,
    want: Vec<f64>,
    /// (device calls, host calls) this case added.
    routes: (u64, u64),
}

/// Device 0 and the registered CUDA QMatMul wrapper, looked up the way the executor does.
fn cuda_kernel() -> (CudaDevice, fuel_dispatch::kernel::KernelRef) {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = table
        .lookup(OpKind::QMatMul, &KEY, BackendId::Cuda)
        .expect("QMatMul F32 binding must be registered on CUDA");
    (dev, kernel)
}

/// The f64 reference: `x[rows, k]` times the dequantized `w[n, k]` transposed.
fn reference(x: &[f32], w_deq: &[f32], (rows, n, k): (usize, usize, usize)) -> Vec<f64> {
    let mut want = vec![0.0f64; rows * n];
    for r in 0..rows {
        for o in 0..n {
            want[r * n + o] = (0..k)
                .map(|c| f64::from(x[r * k + c]) * f64::from(w_deq[o * k + c]))
                .sum();
        }
    }
    want
}

/// The CPU kernel alone on the same Q4_0 problem, timed after a warm-up call.
fn cpu_kernel_time(
    x: &[f32],
    w_bytes: &[u8],
    (rows, n, k): (usize, usize, usize),
) -> std::time::Duration {
    let once = || {
        let act = fuel_cpu_backend::CpuStorageBytes::from_slice(x);
        let w = fuel_cpu_backend::CpuStorageBytes::from_bytes(w_bytes);
        let mut out = fuel_cpu_backend::CpuStorageBytes::from_zero_bytes(rows * n * 4);
        let t = std::time::Instant::now();
        fuel_cpu_backend::byte_kernels::qmatmul_q4_0_f32(&act, &w, &mut out, 1, rows, n, k)
            .expect("cpu qmatmul");
        t.elapsed()
    };
    let _ = once();
    once()
}

/// The route counters are process-wide, so the cases take turns.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run(q: QuantType, weights: (Vec<u8>, Vec<f32>), rows: usize, n: usize, k: usize) -> Run {
    fuel_test_support::require_gpu_run_lock();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (dev, kernel) = cuda_kernel();

    let (w_bytes, w_deq) = weights;
    let x = data(rows * k, 5);
    let out = {
        let buf = CudaStorageBytes::alloc(&dev, rows * n * 4).expect("alloc");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(buf),
            DType::F32,
        )))
    };
    let before = (
        qmatmul::DEVICE_CALLS.load(Relaxed),
        qmatmul::HOST_CALLS.load(Relaxed),
    );
    kernel(
        &[
            upload(&dev, bytemuck::cast_slice(&x), DType::F32),
            upload(&dev, &w_bytes, DType::U32),
        ],
        &mut [out.clone()],
        &[],
        &OpParams::QMatMul {
            quant_type: q,
            batch_count: 1,
            m: rows,
            n,
            k,
        },
    )
    .expect("qmatmul call");
    let routes = (
        qmatmul::DEVICE_CALLS.load(Relaxed) - before.0,
        qmatmul::HOST_CALLS.load(Relaxed) - before.1,
    );
    let got_bytes = match &out.read().unwrap().inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("output not on CUDA"),
    };
    let got: Vec<f32> = bytemuck::cast_slice(&got_bytes).to_vec();
    let want = reference(&x, &w_deq, (rows, n, k));
    Run { got, want, routes }
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0.0, f64::max)
}

/// `device`: the case must have run on the baracuda kernels (else on the host fallback).
fn check(label: &str, r: &Run, device: bool) {
    assert_eq!(
        r.routes,
        if device { (1, 0) } else { (0, 1) },
        "{label}: wrong route (device calls, host calls)"
    );
    let scale = r.want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-3);
    let err = max_err(&r.got, &r.want);
    eprintln!("{label}: max_err/scale = {:.3e}", err / scale);
    // Device routes measured <= 1.1e-7 of scale (limit 1e-5, ~90x margin); the host fallback's
    // 8-bit activation quantization measured <= 8.4e-4 (limit 5e-3).
    let limit = if device { 1e-5 } else { 5e-3 } * scale;
    assert!(
        err <= limit,
        "{label}: max_err={err} scale={scale} (limit {limit})"
    );
    // Sabotage sibling: the same output compared against a reference rotated by one output
    // column must be FAR, so the tolerance cannot be met by a wrong result.
    let mut rotated = r.want.clone();
    rotated.rotate_left(1);
    let wrong = max_err(&r.got, &rotated);
    assert!(
        wrong > limit,
        "{label}: tolerance too loose, a rotated reference is within it (wrong={wrong})"
    );
}

#[test]
fn q4_0_one_row_uses_the_decode_gemv() {
    check(
        "q4_0 rows=1",
        &run(QuantType::Q4_0, quantize::<BlockQ4_0>(48, 128), 1, 48, 128),
        true,
    );
}

#[test]
fn q4_0_many_rows_use_the_batched_kernel() {
    check(
        "q4_0 rows=7",
        &run(QuantType::Q4_0, quantize::<BlockQ4_0>(48, 128), 7, 48, 128),
        true,
    );
}

#[test]
fn q8_0_one_and_many_rows() {
    for rows in [1, 9] {
        check(
            &format!("q8_0 rows={rows}"),
            &run(
                QuantType::Q8_0,
                quantize::<BlockQ8_0>(48, 128),
                rows,
                48,
                128,
            ),
            true,
        );
    }
}

#[test]
fn a_k_off_the_read_chunk_takes_the_host_fallback_and_is_still_right() {
    // k = 96 is a multiple of the 32-element block but not of the 64-column read chunk.
    for rows in [1, 5] {
        check(
            &format!("q4_0 fallback rows={rows}"),
            &run(QuantType::Q4_0, quantize::<BlockQ4_0>(40, 96), rows, 40, 96),
            false,
        );
    }
}

#[test]
fn q4_k_super_blocks() {
    for rows in [1, 6] {
        check(
            &format!("q4_k rows={rows}"),
            &run(
                QuantType::Q4_K_M,
                quantize::<BlockQ4K>(32, 256),
                rows,
                32,
                256,
            ),
            true,
        );
    }
}

/// The point of the binding (row 105): a prefill-sized QMatMul on the device is faster than the CPU
/// kernel ALONE, which is the floor of the host route it replaces (that route also copies both
/// operands down and the result up). One warm-up call per route first, so allocation and kernel
/// load are not timed.
#[test]
fn the_device_route_beats_the_cpu_kernel_alone_on_a_prefill_shape() {
    use std::time::Instant;
    let (rows, n, k) = (140usize, 2048usize, 2048usize);
    // Taken first, so no other case runs while either route is timed.
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (w_bytes, _) = quantize::<BlockQ4_0>(n, k);
    let x = data(rows * k, 5);

    let cpu = cpu_kernel_time(&x, &w_bytes, (rows, n, k));

    fuel_test_support::require_gpu_run_lock();
    let (dev, kernel) = cuda_kernel();
    let act = upload(&dev, bytemuck::cast_slice(&x), DType::F32);
    let w = upload(&dev, &w_bytes, DType::U32);
    let out = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Cuda(CudaStorageBytes::alloc(&dev, rows * n * 4).expect("alloc")),
        DType::F32,
    )));
    let params = OpParams::QMatMul {
        quant_type: QuantType::Q4_0,
        batch_count: 1,
        m: rows,
        n,
        k,
    };
    let device_once = || {
        let t = Instant::now();
        kernel(&[act.clone(), w.clone()], &mut [out.clone()], &[], &params).expect("qmatmul call");
        // The result must be read back for the call to be complete on the stream.
        if let BackendStorage::Cuda(c) = &out.read().unwrap().inner {
            let _ = c.to_cpu_bytes().expect("d2h");
        }
        t.elapsed()
    };
    let _ = device_once();
    let device = device_once();
    eprintln!(
        "qmatmul Q4_0 rows={rows} n={n} k={k}: device {device:?} (incl. readback) vs cpu kernel {cpu:?}"
    );
    assert!(
        device < cpu,
        "device route {device:?} is not faster than the CPU kernel alone {cpu:?}"
    );
}
