// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 34: the CPU mixed-precision matmul kernels `matmul_f32_bf16` and `matmul_f32_f16`
//! (f32 activation x half weight -> f32). The weight stays half-width in memory and widens to f32
//! as it is read, so the result must be the f32 matmul over the WIDENED weight, to f32 rounding.
//!
//! Reference: an `f64` host matmul over the same f32 activations and the same widened weights.
//! The tests print their measured worst absolute error; the bound is the f32 accumulation error
//! of the inner product, `k * 2^-24 * sum |a w|` per element, which is the claim the kernel
//! makes (f32 accumulate) and which a half-precision accumulate or a dropped widening would break.

use fuel_cpu_backend::CpuStorageBytes;
use fuel_cpu_backend::byte_kernels::{matmul_f32_bf16, matmul_f32_f16};
use half::{bf16, f16};

/// Allowed worst `error / bound`. The bound is the worst-case f32 accumulation error, so real error
/// sits far below it (measured worst 0.021 over these shapes); 0.1 leaves ~5x and
/// the two controls below land above 1, which is the sabotage calibration.
const TOL: f64 = 0.1;

/// `(m, n, k)`: decode (m=1), a skinny batch, prefill-like, and a wide-k case.
const SHAPES: &[(usize, usize, usize)] =
    &[(1, 64, 256), (3, 32, 100), (17, 48, 512), (4, 16, 2048)];

/// Deterministic values in `[-1, 1)` (xorshift; no dependency).
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

/// `f64` reference over `a` (f32) and `w` (already widened); returns `(values, per-element bound)`.
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
    // Sequential f32 accumulation: |err| <= k * u * sum|a w|, u = 2^-24. A factor 1 keeps the bound
    // tight enough that dropping the widening (error ~ 2^-9 relative) is far outside it.
    let bound = mag.iter().map(|&s| k as f64 * 2f64.powi(-24) * s).collect();
    (out, bound)
}

fn run_bf16(a: &[f32], w: &[bf16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let (lhs, rhs) = (
        CpuStorageBytes::from_slice(a),
        CpuStorageBytes::from_slice(w),
    );
    let mut out = CpuStorageBytes::from_zero_bytes(m * n * 4);
    matmul_f32_bf16(&lhs, &rhs, &mut out, &[], &[], m, n, k).expect("matmul_f32_bf16");
    out.as_slice::<f32>().expect("f32 output").to_vec()
}

fn run_f16(a: &[f32], w: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let (lhs, rhs) = (
        CpuStorageBytes::from_slice(a),
        CpuStorageBytes::from_slice(w),
    );
    let mut out = CpuStorageBytes::from_zero_bytes(m * n * 4);
    matmul_f32_f16(&lhs, &rhs, &mut out, &[], &[], m, n, k).expect("matmul_f32_f16");
    out.as_slice::<f32>().expect("f32 output").to_vec()
}

/// Worst `|got - want|` and the worst `|got - want| / bound` over the output.
fn worst(got: &[f32], want: &[f64], bound: &[f64]) -> (f64, f64) {
    let mut abs = 0f64;
    let mut ratio = 0f64;
    for ((&g, &w), &b) in got.iter().zip(want).zip(bound) {
        let d = (g as f64 - w).abs();
        abs = abs.max(d);
        ratio = ratio.max(d / b.max(f64::MIN_POSITIVE));
    }
    (abs, ratio)
}

#[test]
fn bf16_weight_matches_the_f64_reference_of_the_widened_weight() {
    for (idx, &(m, n, k)) in SHAPES.iter().enumerate() {
        let a = values(m * k, 10 + idx as u64);
        let w: Vec<bf16> = values(k * n, 20 + idx as u64)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let widened: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
        let (want, bound) = reference(&a, &widened, m, n, k);
        let (abs, ratio) = worst(&run_bf16(&a, &w, m, n, k), &want, &bound);
        println!(
            "[row-34] bf16 m={m} n={n} k={k}: worst abs {abs:.3e}, worst err/bound {ratio:.3}"
        );
        assert!(
            ratio <= TOL,
            "m={m} n={n} k={k}: error {ratio} x the f32 bound"
        );
    }
}

#[test]
fn f16_weight_matches_the_f64_reference_of_the_widened_weight() {
    for (idx, &(m, n, k)) in SHAPES.iter().enumerate() {
        let a = values(m * k, 30 + idx as u64);
        let w: Vec<f16> = values(k * n, 40 + idx as u64)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let widened: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
        let (want, bound) = reference(&a, &widened, m, n, k);
        let (abs, ratio) = worst(&run_f16(&a, &w, m, n, k), &want, &bound);
        println!("[row-34] f16 m={m} n={n} k={k}: worst abs {abs:.3e}, worst err/bound {ratio:.3}");
        assert!(
            ratio <= TOL,
            "m={m} n={n} k={k}: error {ratio} x the f32 bound"
        );
    }
}

/// Swap control: the same bytes fed through the OTHER half kernel (bf16 bits read as f16, and
/// the reverse) must fall outside the bound, so the tests above can tell a wrong widening from a
/// right one.
#[test]
fn a_swapped_weight_dtype_is_outside_the_bound() {
    let (m, n, k) = (3, 32, 100);
    let a = values(m * k, 50);
    let w: Vec<bf16> = values(k * n, 51).into_iter().map(bf16::from_f32).collect();
    let widened: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
    let (want, bound) = reference(&a, &widened, m, n, k);
    // Same 16-bit patterns, read as f16.
    let as_f16: Vec<f16> = w.iter().map(|x| f16::from_bits(x.to_bits())).collect();
    let (_, ratio) = worst(&run_f16(&a, &as_f16, m, n, k), &want, &bound);
    println!("[row-34] swap control: err/bound {ratio:.2}");
    assert!(ratio > 1.0, "swap not detected: err/bound {ratio}");
}

/// Swap control for the accumulate: a reference built from the UNROUNDED f32 weights differs from
/// the widened-weight reference by the bf16 weight rounding, far outside the f32 bound, so the
/// tests above do check that the kernel multiplies by the rounded weight it was given.
#[test]
fn the_unrounded_weight_is_outside_the_bound() {
    let (m, n, k) = (3, 32, 100);
    let a = values(m * k, 60);
    let exact_w = values(k * n, 61);
    let w: Vec<bf16> = exact_w.iter().map(|&x| bf16::from_f32(x)).collect();
    let widened: Vec<f32> = w.iter().map(|x| x.to_f32()).collect();
    let (_, bound) = reference(&a, &widened, m, n, k);
    let (want_exact, _) = reference(&a, &exact_w, m, n, k);
    let (_, ratio) = worst(&run_bf16(&a, &w, m, n, k), &want_exact, &bound);
    println!("[row-34] unrounded-weight control: err/bound {ratio:.2}");
    assert!(
        ratio > 1.0,
        "weight rounding not visible: err/bound {ratio}"
    );
}

/// Batched with a GQA-divisible rhs batch (lhs batch 4, rhs batch 2): each lhs slot reads rhs slot
/// `b / 2`.
#[test]
fn gqa_batch_reads_the_matching_rhs_slot() {
    let (m, n, k) = (2, 8, 16);
    let a = values(4 * m * k, 70);
    let w: Vec<bf16> = values(2 * k * n, 71)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let (lhs, rhs) = (
        CpuStorageBytes::from_slice(&a),
        CpuStorageBytes::from_slice(&w),
    );
    let mut out = CpuStorageBytes::from_zero_bytes(4 * m * n * 4);
    matmul_f32_bf16(&lhs, &rhs, &mut out, &[4], &[2], m, n, k).expect("batched");
    let got = out.as_slice::<f32>().unwrap().to_vec();
    for b in 0..4 {
        let widened: Vec<f32> = w[(b / 2) * k * n..(b / 2 + 1) * k * n]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let (want, bound) = reference(&a[b * m * k..(b + 1) * m * k], &widened, m, n, k);
        let (_, ratio) = worst(&got[b * m * n..(b + 1) * m * n], &want, &bound);
        assert!(ratio <= TOL, "batch {b}: err/bound {ratio}");
    }
}

/// A byte length that does not match the shape is a typed error, never a read past the buffer.
#[test]
fn a_wrong_sized_weight_is_a_typed_error() {
    let a = values(2 * 4, 80);
    let w: Vec<bf16> = values(4 * 3 - 1, 81)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let (lhs, rhs) = (
        CpuStorageBytes::from_slice(&a),
        CpuStorageBytes::from_slice(&w),
    );
    let mut out = CpuStorageBytes::from_zero_bytes(2 * 3 * 4);
    assert!(matmul_f32_bf16(&lhs, &rhs, &mut out, &[], &[], 2, 3, 4).is_err());
}
