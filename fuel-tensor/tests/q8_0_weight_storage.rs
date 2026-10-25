// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 34 (TTFT): a GGUF Q8_0 weight stays 8-bit and runs through `Op::QMatMul`
//! (`WeightStorage::Q8_0`) instead of being dequantized to an f32 copy. The CPU kernel quantizes
//! the f32 activations to Q8_0 blocks first (llama.cpp's `vec_dot`), so the result differs from the
//! f32 matmul over the dequantized weight by the activation rounding error only.
//!
//! Tolerance `TOL` is sabotage-calibrated, not the raw bound: measured worst `error / bound` is 0.184,
//! a 5% output scale error lands at 2.4 and a wrong weight far above, so `TOL = 0.5` keeps 2.7x
//! over the measured error and fails both controls.
//!
//! Bound: each activation block of 32 is rounded to a step `amax/127` (error <= half a step), so per
//! output `|err| <= sum_k |w_k| * amax_block/254` plus the f32 accumulation error. The tests print
//! the measured worst `error / bound`; the controls land above 1 (sabotage calibration).

use fuel_ir::Shape;
use fuel_quantized::{BlockQ8_0, GgmlType};
use fuel_tensor::lazy::{Tensor, WeightStorage};
use std::sync::Arc;

/// Allowed worst `error / bound` (see the module doc for its calibration).
const TOL: f64 = 0.5;
const IN: usize = 96; // 3 blocks of 32
const OUT: usize = 20;
const ROWS: usize = 5;

fn values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `[OUT, IN]` f32 -> the Q8_0 `WeightStorage` (GGUF layout, blocks along `IN`) and its dequantized
/// `[OUT, IN]` values.
fn quantized_weight(w: &[f32]) -> (WeightStorage, Vec<f32>) {
    let mut blocks = vec![BlockQ8_0::zeros(); OUT * IN / 32];
    BlockQ8_0::from_float(w, &mut blocks);
    let mut deq = vec![0.0f32; OUT * IN];
    BlockQ8_0::to_float(&blocks, &mut deq);
    let bytes_len = std::mem::size_of_val(blocks.as_slice());
    assert_eq!(bytes_len, OUT * IN / 32 * 34);
    // SAFETY: `BlockQ8_0` is `repr(C)` plain data; viewing its bytes is sound.
    let bytes = unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), bytes_len) };
    let words: Vec<u32> = bytes
        .chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect();
    (
        WeightStorage::Q8_0 {
            words: Arc::from(words),
            bytes_len,
            in_features: IN,
            out_features: OUT,
        },
        deq,
    )
}

fn run(ws: &WeightStorage, x: &[f32]) -> Vec<f32> {
    let anchor = Tensor::from_f32(
        x.to_vec(),
        Shape::from_dims(&[ROWS, IN]),
        &fuel_tensor::Device::cpu(),
    )
    .expect("x");
    ws.apply_linear(&anchor, IN, OUT)
        .expect("apply_linear")
        .realize_f32()
        .to_vec()
}

/// f64 reference `x @ deq^T` and the per-output rounding bound.
fn reference(x: &[f32], deq: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; ROWS * OUT];
    let mut bound = vec![0f64; ROWS * OUT];
    for r in 0..ROWS {
        for o in 0..OUT {
            let (mut acc, mut b) = (0f64, 0f64);
            for blk in 0..IN / 32 {
                let amax = x[r * IN + blk * 32..r * IN + blk * 32 + 32]
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs())) as f64;
                for k in blk * 32..blk * 32 + 32 {
                    let w = deq[o * IN + k] as f64;
                    acc += x[r * IN + k] as f64 * w;
                    b += w.abs() * amax / 254.0;
                }
            }
            out[r * OUT + o] = acc;
            // f32 accumulation slack on top of the activation rounding bound.
            bound[r * OUT + o] = b * 1.05 + 1e-5;
        }
    }
    (out, bound)
}

fn worst_ratio(got: &[f32], want: &[f64], bound: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .zip(bound)
        .map(|((&g, &w), &b)| (g as f64 - w).abs() / b)
        .fold(0.0, f64::max)
}

#[test]
fn q8_0_weight_matches_the_dequantized_matmul_within_activation_rounding() {
    let w = values(OUT * IN, 11, 0.5);
    let x = values(ROWS * IN, 12, 2.0);
    let (ws, deq) = quantized_weight(&w);
    assert_eq!(ws.dtype(), fuel_ir::DType::U32);
    let (want, bound) = reference(&x, &deq);
    let ratio = worst_ratio(&run(&ws, &x), &want, &bound);
    println!("worst error / bound = {ratio:.3}");
    assert!(ratio <= TOL, "ratio {ratio}");
}

#[test]
fn the_bound_can_fail_wrong_scale_and_swapped_weight_controls() {
    let w = values(OUT * IN, 11, 0.5);
    let x = values(ROWS * IN, 12, 2.0);
    let (ws, deq) = quantized_weight(&w);
    let (want, bound) = reference(&x, &deq);
    // Control 1: a reference built from a differently seeded weight must land far above the bound.
    let other = values(OUT * IN, 99, 0.5);
    let (_, other_deq) = quantized_weight(&other);
    let (wrong, _) = reference(&x, &other_deq);
    assert!(
        worst_ratio(&run(&ws, &x), &wrong, &bound) > TOL,
        "swapped weight not detected"
    );
    // Control 2: an output scaled by 5% must exceed it too (1% would not: 0.44).
    let scaled: Vec<f32> = run(&ws, &x).iter().map(|v| v * 1.05).collect();
    assert!(
        worst_ratio(&scaled, &want, &bound) > TOL,
        "5% scale error not detected"
    );
}

#[test]
fn a_shape_mismatch_is_a_typed_error_not_a_panic() {
    let w = values(OUT * IN, 11, 0.5);
    let (ws, _) = quantized_weight(&w);
    let anchor = Tensor::from_f32(
        values(ROWS * 64, 3, 1.0),
        Shape::from_dims(&[ROWS, 64]),
        &fuel_tensor::Device::cpu(),
    )
    .expect("x");
    let err = ws.apply_linear(&anchor, 64, OUT).expect_err("must refuse");
    assert!(err.to_string().contains("Q8_0 shape mismatch"), "{err}");
}
