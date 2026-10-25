// SPDX-License-Identifier: MIT OR Apache-2.0
//! The CPU f32 matmul (`matmul_f32`, `matmul_f32_capacity`) runs through the `gemm` crate above a
//! work threshold and keeps the plain loop below it. Both paths must agree with an f64 reference
//! to the f32 accumulation bound `k * 2^-24 * sum |a w|` (any reduction order stays inside it), and
//! the capacity rows beyond `m_compute` must stay zero.
//!
//! The tests print the measured worst `error / bound`; `TOL` is sabotage-calibrated: a transposed
//! right-hand side and a dropped last k-block both land far above it.

use fuel_cpu_backend::CpuStorageBytes;
use fuel_cpu_backend::byte_kernels::{matmul_f32, matmul_f32_capacity};

const TOL: f64 = 0.1;

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// f64 `a[m,k] @ w[k,n]` and the per-element f32 accumulation bound.
fn reference(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; m * n];
    let mut mag = vec![0f64; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk] as f64;
            for j in 0..n {
                let t = av * w[kk * n + j] as f64;
                out[i * n + j] += t;
                mag[i * n + j] += t.abs();
            }
        }
    }
    let bound = mag
        .iter()
        .map(|&s| (k as f64 * 2f64.powi(-24) * s).max(1e-30))
        .collect();
    (out, bound)
}

fn worst(got: &[f32], want: &[f64], bound: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .zip(bound)
        .map(|((&g, &w), &b)| (g as f64 - w).abs() / b)
        .fold(0.0, f64::max)
}

fn run(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let (lhs, rhs) = (
        CpuStorageBytes::from_slice(a),
        CpuStorageBytes::from_slice(w),
    );
    let mut out = CpuStorageBytes::from_zero_bytes(m * n * 4);
    matmul_f32(&lhs, &rhs, &mut out, &[], &[], m, n, k).expect("matmul_f32");
    out.as_slice::<f32>().expect("f32").to_vec()
}

/// `(m, n, k)`: below the threshold (plain loop), decode (m = 1) on both sides of it, and prefill
/// shapes well above it (`gemm`, single- and multi-threaded).
const SHAPES: &[(usize, usize, usize)] = &[
    (3, 17, 33),
    (1, 64, 128),
    (1, 2048, 1024),
    (17, 300, 129),
    (140, 576, 576),
    (64, 1536, 576),
];

#[test]
fn both_paths_match_the_f64_reference() {
    let mut max_ratio = 0f64;
    for (idx, &(m, n, k)) in SHAPES.iter().enumerate() {
        let a = values(m * k, 10 + idx as u64);
        let w = values(k * n, 20 + idx as u64);
        let (want, bound) = reference(&a, &w, m, n, k);
        let ratio = worst(&run(&a, &w, m, n, k), &want, &bound);
        println!("({m},{n},{k}): worst error / bound = {ratio:.4}");
        assert!(ratio <= TOL, "({m},{n},{k}): {ratio}");
        max_ratio = max_ratio.max(ratio);
    }
    println!("max over shapes = {max_ratio:.4}");
}

#[test]
fn the_tolerance_fails_a_transposed_rhs_and_a_dropped_k_block() {
    let (m, n, k) = (140, 96, 192);
    let a = values(m * k, 1);
    let w = values(k * n, 2);
    let (want, bound) = reference(&a, &w, m, n, k);
    // Transposed rhs: same values read as [n, k].
    let mut wt = vec![0f32; k * n];
    for kk in 0..k {
        for j in 0..n {
            wt[j * k + kk] = w[kk * n + j];
        }
    }
    assert!(worst(&run(&a, &wt, m, n, k), &want, &bound) > TOL * 100.0);
    // Dropped last 16 of k: the product over k - 16.
    let (a2, w2): (Vec<f32>, Vec<f32>) = (
        (0..m)
            .flat_map(|i| a[i * k..i * k + k - 16].to_vec())
            .collect(),
        w[..(k - 16) * n].to_vec(),
    );
    assert!(worst(&run(&a2, &w2, m, n, k - 16), &want, &bound) > TOL * 100.0);
}

#[test]
fn capacity_tail_rows_stay_zero_above_the_gemm_threshold() {
    let (m_cap, m_compute, n, k) = (160, 100, 576, 576);
    let a = values(m_cap * k, 5);
    let w = values(k * n, 6);
    let (lhs, rhs) = (
        CpuStorageBytes::from_slice(&a),
        CpuStorageBytes::from_slice(&w),
    );
    let mut out = CpuStorageBytes::from_zero_bytes(m_cap * n * 4);
    matmul_f32_capacity(&lhs, &rhs, &mut out, &[], &[], m_cap, m_compute, n, k).expect("capacity");
    let got = out.as_slice::<f32>().expect("f32").to_vec();
    assert!(
        got[m_compute * n..].iter().all(|&v| v == 0.0),
        "tail rows written"
    );
    let (want, bound) = reference(&a[..m_compute * k], &w, m_compute, n, k);
    assert!(worst(&got[..m_compute * n], &want, &bound) <= TOL);
}
