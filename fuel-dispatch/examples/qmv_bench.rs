// SPDX-License-Identifier: MIT OR Apache-2.0
//! Speed of the Vulkan quantized decode matmul (`m == 1`) on real Qwen3.5-4B shapes (ROADMAP row 119).
//!
//! Q8_0 goes through `VulkanBackend::matmul_q8_0_bytes`: on a build before the fused gemv that is
//! dequantize-the-whole-weight-to-f32 then `matmul_f32`; on a build with it, the fused `qmatvec_q8_0`.
//! Run the SAME file on both builds and compare. Q6_K only exists fused (no earlier Vulkan path).
//!
//! Method: weights are synthetic quantized bytes (a valid f16 scale, pseudo-random quants), because
//! the kernels' time does not depend on the values. Per case: 5 warm-up calls, then `REPEATS`
//! repeats of `ITERS` back-to-back calls ended by one download of the output (forces completion);
//! each repeat reports microseconds per call. The table gives min / median / max over the repeats.
//! The calls submit and flush per call, so a per-call figure includes submit overhead on both builds.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 2 --release \
//!     -p fuel-dispatch --features vulkan --example qmv_bench
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its ceiling).

use std::sync::Arc;
use std::time::Instant;

use fuel_vulkan_backend::VulkanBackend;

const REPEATS: usize = 9;
const ITERS: usize = 30;

fn pattern(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(131).wrapping_add(seed).wrapping_add(i >> 8) % 251) as u8)
        .collect()
}

/// `rows * (k / 32)` Q8_0 blocks: f16 scale 0.0625 then 32 pseudo-random i8 quants.
fn q8_blocks(n: usize, k: usize) -> Vec<u8> {
    let mut v = pattern(n * (k / 32) * 34, 7);
    for b in v.as_chunks_mut::<34>().0 {
        b[0] = 0x00;
        b[1] = 0x2C; // f16 0x2C00 = 0.0625
    }
    v
}

/// `rows * (k / 256)` Q6_K super-blocks (210 bytes): quants random, f16 `d` at the end = 0.0625.
fn q6_blocks(n: usize, k: usize) -> Vec<u8> {
    let mut v = pattern(n * (k / 256) * 210, 11);
    for b in v.as_chunks_mut::<210>().0 {
        b[208] = 0x00;
        b[209] = 0x2C;
    }
    v
}

fn stats(mut us: Vec<f64>) -> (f64, f64, f64) {
    us.sort_by(|a, b| a.total_cmp(b));
    (us[0], us[us.len() / 2], us[us.len() - 1])
}

/// One table row: min / median / max microseconds per call and effective GB/s of weight read.
fn report(label: &str, n: usize, k: usize, weight_bytes: usize, per_call_us: Vec<f64>) {
    let (lo, med, hi) = stats(per_call_us);
    let mb = weight_bytes as f64 / 1e6;
    println!(
        "{label:<8} {n:>6} {k:>6} {mb:>10.1} {lo:>9.0} {med:>8.0} {hi:>8.0} {:>10.1}",
        weight_bytes as f64 / (med * 1e-6) / 1e9
    );
}

fn print_header() {
    println!("# repeats={REPEATS} iters/repeat={ITERS}; microseconds per call");
    println!("kernel    n      k      weight_MB   min_us   med_us   max_us   GB/s(med)");
}

fn main() {
    let backend = Arc::new(VulkanBackend::new().expect("a Vulkan device"));
    print_header();
    let run = |label: &str, n: usize, k: usize, w_bytes: Vec<u8>, q6: bool| {
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011).sin() * 0.5).collect();
        let a_s = backend
            .upload_bytes_handle(bytemuck::cast_slice(&a))
            .expect("upload a");
        let w_s = backend.upload_bytes_handle(&w_bytes).expect("upload w");
        let mut out = backend.alloc_bytes_handle(n * 4).expect("alloc out");
        let call = |out: &mut fuel_vulkan_backend::VulkanStorageBytes| {
            let r = if q6 {
                // Q6K-BEGIN
                backend.matmul_q6_k_bytes(&a_s, &w_s, out, 1, 1, k, n)
                // Q6K-END
                // Q6K-STUB: unreachable!()
            } else {
                backend.matmul_q8_0_bytes(&a_s, &w_s, out, 1, 1, k, n)
            };
            r.expect("matmul");
        };
        for _ in 0..5 {
            call(&mut out);
        }
        let _ = backend.download_bytes(&out).expect("sync");
        let mut per_call_us = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let t = Instant::now();
            for _ in 0..ITERS {
                call(&mut out);
            }
            let _ = backend.download_bytes(&out).expect("sync");
            per_call_us.push(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
        }
        report(label, n, k, w_bytes.len(), per_call_us);
    };

    // Real Qwen3.5-4B shapes: (n = output rows, k = contracted dim).
    for (n, k) in [(32usize, 2560usize), (2560, 4096), (9216, 2560)] {
        run("q8_0", n, k, q8_blocks(n, k), false);
    }
    // Q6K-BEGIN
    for (n, k) in [
        (8192usize, 2560usize),
        (9216, 2560),
        (2560, 9216),
        (248_320, 2560),
    ] {
        run("q6_k", n, k, q6_blocks(n, k), true);
    }
    // Q6K-END
}
