// SPDX-License-Identifier: MIT OR Apache-2.0
//! Accuracy audit of the CUDA `QMatMul` binding (GAP-154, ROADMAP row 128): the error of
//! `baracuda_dispatch::qmatmul::qmatmul_f32` over a spread of shapes, formats and seeds, against an
//! f64 reference built from the dequantized weights, and against the CPU kernel it replaces.
//!
//! The declared bound lives in the contract (`kernel-contracts/cuda/qmatmul.fkc.md`,
//! `precision.max_relative`) and this test reads it from there, so the claim cannot drift from the
//! audit: the worst measured error must be within it, and an injected error (every block scale of
//! the weights nudged by 1%, "a wrong scale") must exceed it.
//!
//! The metric is SCALE-RELATIVE: `max |out - ref|` over a call, divided by `max |ref|` of the same
//! call. A per-element relative error is unstable near zero crossings and would grow with output
//! size for reasons unrelated to accuracy. The contract's notes say the same.
//!
//! Both routes of the wrapper are audited and reported separately (the device MMVQ route, and the
//! host fallback for a `k` off the read chunk), since one contract claim covers both.
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-dispatch --features cuda
//! --test baracuda_qmatmul_audit -- --nocapture` (built through `scripts/cuda-build.ps1`).

#![cfg(feature = "cuda")]

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{
    baracuda_dispatch::{qmatmul, register_baracuda_cuda_kernels},
    dispatch::register_cuda_kernels,
    fkc::parse_file,
    kernel::{KernelBindingTable, KernelRef, OpParams},
};
use fuel_graph::QuantType;
use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_quantized::{BlockQ4_0, BlockQ4K, BlockQ8_0, k_quants::GgmlType};

const KEY: [DType; 3] = [DType::F32, DType::U32, DType::F32];
const CONTRACT: &str = include_str!("../kernel-contracts/cuda/qmatmul.fkc.md");

/// Deterministic generator (xorshift64*), so a failing seed reproduces.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Roughly normal (sum of four uniforms), mean 0, unit-ish scale.
    fn normal(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.next_f32()).sum();
        (s - 2.0) * 1.732
    }
}

fn randn(len: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    (0..len).map(|_| r.normal()).collect()
}

/// Quantize `n` rows of `k` weights (seeded); returns block bytes and the dequantized f32 weights.
fn quantize<T: GgmlType>(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let w = randn(n * k, seed ^ 0xA5A5);
    let mut blocks = vec![T::zeros(); n * k / T::BLCK_SIZE];
    T::from_float(&w, &mut blocks);
    let mut deq = vec![0.0f32; n * k];
    T::to_float(&blocks, &mut deq);
    (bytemuck::cast_slice(&blocks).to_vec(), deq)
}

fn upload(dev: &CudaDevice, bytes: &[u8], dt: DType) -> Arc<RwLock<Storage>> {
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(cuda), dt)))
}

fn cuda_kernel() -> (CudaDevice, KernelRef) {
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

/// The route counters are process-wide, so the cases take turns.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One call of the CUDA wrapper; returns the output and whether it ran on the device (else host).
fn run_cuda(
    q: QuantType,
    w_bytes: &[u8],
    x: &[f32],
    (rows, n, k): (usize, usize, usize),
) -> (Vec<f32>, bool) {
    fuel_test_support::require_gpu_run_lock();
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (dev, kernel) = cuda_kernel();
    let out = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Cuda(CudaStorageBytes::alloc(&dev, rows * n * 4).expect("alloc")),
        DType::F32,
    )));
    let before = (
        qmatmul::DEVICE_CALLS.load(Relaxed),
        qmatmul::HOST_CALLS.load(Relaxed),
    );
    kernel(
        &[
            upload(&dev, bytemuck::cast_slice(x), DType::F32),
            upload(&dev, w_bytes, DType::U32),
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
    let device = qmatmul::DEVICE_CALLS.load(Relaxed) - before.0 == 1
        && qmatmul::HOST_CALLS.load(Relaxed) == before.1;
    let bytes = match &out.read().unwrap().inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("output not on CUDA"),
    };
    (bytemuck::cast_slice(&bytes).to_vec(), device)
}

/// The CPU kernel the CUDA binding replaces, on the same bytes.
fn run_cpu(
    q: QuantType,
    w_bytes: &[u8],
    x: &[f32],
    (rows, n, k): (usize, usize, usize),
) -> Vec<f32> {
    use fuel_cpu_backend::{CpuStorageBytes, byte_kernels as bk};
    let act = CpuStorageBytes::from_slice(x);
    let w = CpuStorageBytes::from_bytes(w_bytes);
    let mut out = CpuStorageBytes::from_zero_bytes(rows * n * 4);
    match q {
        QuantType::Q4_0 => bk::qmatmul_q4_0_f32(&act, &w, &mut out, 1, rows, n, k),
        QuantType::Q8_0 => bk::qmatmul_q8_0_f32(&act, &w, &mut out, 1, rows, n, k),
        QuantType::Q4_K_M => bk::qmatmul_q4_k_m_f32(&act, &w, &mut out, 1, rows, n, k),
        other => panic!("audit does not cover {other:?}"),
    }
    .expect("cpu qmatmul");
    bytemuck::cast_slice(out.bytes()).to_vec()
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

/// `max |got - want| / max |want|`, and the flat index where the absolute error peaks.
fn scale_rel(got: &[f32], want: &[f64]) -> (f64, usize) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-30);
    let (mut worst, mut at) = (0.0f64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (f64::from(*g) - w).abs();
        if e > worst {
            (worst, at) = (e, i);
        }
    }
    (worst / scale, at)
}

/// The declared claim, read from the contract so it cannot drift from this audit.
fn declared_max_relative() -> Option<f64> {
    let file = parse_file(CONTRACT).expect("contract parses");
    let k = file
        .kernels
        .iter()
        .find(|k| k.kernel == "qmatmul_f32")
        .expect("qmatmul_f32 kernel");
    k.precision
        .as_ref()
        .and_then(|p| p.max_relative.as_ref())
        .and_then(|v| v.as_f64())
}

struct Worst {
    err: f64,
    label: String,
}
impl Worst {
    fn new() -> Self {
        Worst {
            err: 0.0,
            label: String::new(),
        }
    }
    fn see(&mut self, err: f64, label: String) {
        if err > self.err {
            *self = Worst { err, label };
        }
    }
}

/// (rows, n, k): small, ragged m, large k, prefill-sized, and k off the 64-column read chunk.
const SHAPES: [(usize, usize, usize); 11] = [
    (1, 64, 128),
    (1, 2048, 2048),
    (1, 96, 8192),
    (3, 40, 256),
    (7, 33, 512),
    (17, 96, 1024),
    (64, 128, 4096),
    (140, 512, 2048),
    (5, 200, 8192),
    (1, 40, 96),
    (9, 17, 160),
];
const SEEDS: [u64; 4] = [1, 2, 3, 4];

fn audit_format<T: GgmlType>(q: QuantType, name: &str, worst: &mut [Worst; 2]) {
    let block = T::BLCK_SIZE;
    for &(rows, n, k) in &SHAPES {
        if k % block != 0 {
            continue;
        }
        for seed in SEEDS {
            let (w_bytes, w_deq) = quantize::<T>(n, k, seed);
            let x = randn(rows * k, seed);
            let want = reference(&x, &w_deq, (rows, n, k));
            let (got, device) = run_cuda(q, &w_bytes, &x, (rows, n, k));
            let (err, at) = scale_rel(&got, &want);
            let cpu = run_cpu(q, &w_bytes, &x, (rows, n, k));
            let (cpu_err, _) = scale_rel(&cpu, &want);
            let (vs_cpu, _) =
                scale_rel(&got, &cpu.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
            let route = if device { "device" } else { "host" };
            eprintln!(
                "{name} rows={rows} n={n} k={k} seed={seed} {route}: cuda_vs_f64 {err:.3e} (worst at {at}) | cpu_vs_f64 {cpu_err:.3e} | cuda_vs_cpu {vs_cpu:.3e}"
            );
            worst[usize::from(!device)].see(
                err,
                format!("{name} rows={rows} n={n} k={k} seed={seed} at flat index {at}"),
            );
        }
    }
}

#[test]
fn the_declared_bound_holds_over_the_audited_spread() {
    let mut worst = [Worst::new(), Worst::new()];
    audit_format::<BlockQ4_0>(QuantType::Q4_0, "q4_0", &mut worst);
    audit_format::<BlockQ8_0>(QuantType::Q8_0, "q8_0", &mut worst);
    audit_format::<BlockQ4K>(QuantType::Q4_K_M, "q4_k", &mut worst);
    for (route, w) in ["device", "host"].iter().zip(&worst) {
        eprintln!("WORST {route}: {:.3e} ({})", w.err, w.label);
    }
    assert!(worst[0].err > 0.0, "no case ran on the device route");
    let declared = declared_max_relative()
        .expect("the contract declares precision.max_relative (scale-relative; see the audit)");
    let overall = worst[0].err.max(worst[1].err);
    assert!(
        overall <= declared,
        "worst measured error {overall:e} exceeds the declared bound {declared:e}"
    );
}

/// Positive control: the same audit machinery must see an injected error. Every block scale of the
/// Q4_0 weights is nudged by 1% (a wrong scale) on the device only; the f64 reference keeps the
/// true weights, so the measured error must exceed the declared bound.
#[test]
fn an_injected_wrong_scale_exceeds_the_declared_bound() {
    let declared = declared_max_relative().expect("the contract declares precision.max_relative");
    let (rows, n, k) = (7, 64, 512);
    let (mut w_bytes, w_deq) = quantize::<BlockQ4_0>(n, k, 9);
    let x = randn(rows * k, 9);
    let want = reference(&x, &w_deq, (rows, n, k));
    // Q4_0 block: f16 scale `d` in the first two bytes, then 16 bytes of nibbles.
    for blk in w_bytes.chunks_exact_mut(18) {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32() * 1.01;
        blk[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    let (got, device) = run_cuda(QuantType::Q4_0, &w_bytes, &x, (rows, n, k));
    assert!(device, "the control must run on the device route");
    let (err, _) = scale_rel(&got, &want);
    assert!(
        err > declared,
        "an injected 1% scale error ({err:e}) did not exceed the declared bound {declared:e}: the bound is too loose to mean anything"
    );
}

/// How much the host fallback costs, on the shapes that take it (type-0/1 formats with
/// `k % 64 == 32`). Prints; asserts only that the route really was the host one. The wall time of
/// the wrapper includes uploading the operands and reading the result back, the CPU kernel's time
/// is the kernel alone (the old fallback also paid the copies), so the ratio is a LOWER bound on
/// the old fallback's cost relative to the new one.
#[test]
fn host_fallback_cost_on_the_shapes_that_take_it() {
    for (rows, n, k) in [(1usize, 2048usize, 2080usize), (140, 2048, 2080)] {
        let (w_bytes, _) = quantize::<BlockQ4_0>(n, k, 5);
        let x = randn(rows * k, 5);
        let _ = run_cuda(QuantType::Q4_0, &w_bytes, &x, (rows, n, k));
        let t = std::time::Instant::now();
        let (_, device) = run_cuda(QuantType::Q4_0, &w_bytes, &x, (rows, n, k));
        let new = t.elapsed();
        assert!(!device, "k={k} must take the host route");
        let _ = run_cpu(QuantType::Q4_0, &w_bytes, &x, (rows, n, k));
        let t = std::time::Instant::now();
        let _ = run_cpu(QuantType::Q4_0, &w_bytes, &x, (rows, n, k));
        let old_kernel = t.elapsed();
        eprintln!(
            "FALLBACK q4_0 rows={rows} n={n} k={k}: new host route {new:?} (incl. copies) vs old CPU kernel alone {old_kernel:?}"
        );
    }
}
