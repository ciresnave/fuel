// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 34: wall time of the CPU decode-shaped (m = 1) matmul with an f32 weight against
//! the native bf16 / f16 weight kernels, same shapes, same process. Run with
//! `cargo run --release -p fuel-cpu-backend --example half_matvec_speed`.

use fuel_cpu_backend::CpuStorageBytes;
use fuel_cpu_backend::byte_kernels::{matmul_f32, matmul_f32_bf16, matmul_f32_f16};
use half::{bf16, f16};
use std::time::Instant;

/// `(k, n)` of Qwen3-0.6B decode projections: q/o, MLP up/gate, MLP down.
const SHAPES: &[(usize, usize)] = &[(1024, 2048), (1024, 3072), (3072, 1024)];
const RUNS: usize = 15;

fn median_ms(mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    t[RUNS / 2]
}

fn main() {
    for &(k, n) in SHAPES {
        let a: Vec<f32> = (0..k).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i % 31) as f32 * 0.003 - 0.04).collect();
        let wb: Vec<bf16> = w.iter().map(|&x| bf16::from_f32(x)).collect();
        let wh: Vec<f16> = w.iter().map(|&x| f16::from_f32(x)).collect();
        let lhs = CpuStorageBytes::from_slice(&a);
        let mut out = CpuStorageBytes::from_zero_bytes(n * 4);
        let (r32, rb, rh) = (
            CpuStorageBytes::from_slice(&w),
            CpuStorageBytes::from_slice(&wb),
            CpuStorageBytes::from_slice(&wh),
        );
        let f32_ms =
            median_ms(|| matmul_f32(&lhs, &r32, &mut out, &[], &[], 1, n, k).expect("f32"));
        let bf_ms =
            median_ms(|| matmul_f32_bf16(&lhs, &rb, &mut out, &[], &[], 1, n, k).expect("bf16"));
        let f16_ms =
            median_ms(|| matmul_f32_f16(&lhs, &rh, &mut out, &[], &[], 1, n, k).expect("f16"));
        println!(
            "k={k} n={n}: f32 {f32_ms:.3} ms ({:.1} GB/s) | bf16 {bf_ms:.3} ms ({:.1} GB/s, {:.2}x) | f16 {f16_ms:.3} ms ({:.2}x)",
            (k * n * 4) as f64 / f32_ms / 1e6,
            (k * n * 2) as f64 / bf_ms / 1e6,
            bf_ms / f32_ms,
            f16_ms / f32_ms,
        );
    }
}
