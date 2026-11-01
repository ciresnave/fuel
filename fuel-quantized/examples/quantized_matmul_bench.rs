// SPDX-License-Identifier: MIT OR Apache-2.0
//! Speed and agreement of the quantized CPU matmul (ROADMAP row 140, board 307).
//!
//! Run the SAME file on three builds and compare: the build before the removal with its default
//! (integer block dot, activation quantised to Q8), the same build with `--features float-dot`, and the
//! build after the removal (float route only). The first column says which one ran.
//!
//! Method. Weights are `from_float` of a deterministic pseudo-random matrix; activations likewise.
//! Reference: the dequantized weights (`to_float`) dotted with the f32 activations in f64, so the
//! reference carries the weight-quantisation error of the FORMAT and none of the route's. Agreement is
//! the largest absolute difference from that reference divided by the reference row's RMS, over the
//! whole output. Speed: 3 warm-up calls, then `REPEATS` timed calls; min / median / max milliseconds.
//!
//! ```sh
//! cargo run -j 2 --release -p fuel-quantized --example quantized_matmul_bench
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its ceiling).

use fuel_quantized::{BlockQ4_0, BlockQ4K, BlockQ8_0, GgmlType, matmul};
use std::time::Instant;

const REPEATS: usize = 9;
/// Replaced by a one-line sed per build: which route this binary is.
const ARM: &str = "ARM_PLACEHOLDER";

fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len as u32)
        .map(|i| {
            let x = i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503))
                >> 8;
            (x % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

fn run<T: GgmlType>(label: &str, (m, k, n): (usize, usize, usize)) {
    let w = fill(n * k, 7);
    let x = fill(m * k, 11);
    let blk = T::BLCK_SIZE;
    let mut blocks = vec![T::zeros(); n * k / blk];
    T::from_float(&w, &mut blocks);
    let mut deq = vec![0f32; n * k];
    T::to_float(&blocks, &mut deq);
    let mut got = vec![0f32; m * n];
    for _ in 0..3 {
        matmul((m, k, n), &x, &blocks, &mut got).expect("matmul");
    }
    let mut ms = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t = Instant::now();
        matmul((m, k, n), &x, &blocks, &mut got).expect("matmul");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let (lo, med, hi) = median(ms);
    let mut worst = 0f64;
    for r in 0..m {
        let refs: Vec<f64> = (0..n)
            .map(|c| {
                (0..k)
                    .map(|i| f64::from(x[r * k + i]) * f64::from(deq[c * k + i]))
                    .sum()
            })
            .collect();
        let rms = (refs.iter().map(|v| v * v).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-12);
        for (c, rv) in refs.iter().enumerate() {
            worst = worst.max((f64::from(got[r * n + c]) - rv).abs() / rms);
        }
    }
    println!(
        "{ARM:<22} {label:<6} m={m:<2} k={k} n={n:<5} min {lo:>8.2} med {med:>8.2} max {hi:>8.2} ms   max|err|/rms {worst:.2e}"
    );
}

fn main() {
    println!(
        "# repeats={REPEATS}; milliseconds per matmul call; agreement vs the f64 dequantized reference"
    );
    for shape in [
        (1usize, 4096usize, 4096usize),
        (8, 4096, 4096),
        (1, 4096, 11008),
    ] {
        run::<BlockQ4_0>("Q4_0", shape);
        run::<BlockQ8_0>("Q8_0", shape);
        run::<BlockQ4K>("Q4_K", shape);
    }
}
