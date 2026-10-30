// SPDX-License-Identifier: MIT OR Apache-2.0
//! The CUDA `QMatMul` accuracy measurement (GAP-154, ROADMAP row 114): the registered
//! `qmatmul_f32` binding driven through the same invoker as the other CUDA claims, against an
//! `f64` product of the f32 activations and the DEQUANTIZED weights.
//!
//! The reference is the exact product, not the CPU `QMatMul` kernel: the CPU kernel quantizes the
//! activations to 8 bits (as ggml does) and is itself about 3e-3 to 8e-3 off the exact product, so
//! it cannot judge a kernel that consumes the activations in f32. Cases cover Q4_0, Q8_0 and
//! Q4_K, rows 1 to 140 (ragged rows included), `k` up to 8192, and both routes of the wrapper
//! (the baracuda kernels and the host fallback for a `k` off the format's read chunk). `Q8_1` is
//! not measured: it has no dequantization (GAP-125) and keeps the CPU kernel's result.
//!
//! Metric, shared with every other claim in the ledger: `abs = |out - ref|`,
//! `rel = |out - ref| / max(|ref|, 1.0)`, inputs uniform in `[-1, 1)`.
//! `tests/baracuda_qmatmul_audit.rs` reports the same cases scale-relatively and with the CPU
//! kernel alongside.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_quantized::{BlockQ4_0, BlockQ4K, BlockQ8_0, k_quants::GgmlType};

use super::accuracy_common::{Claim, OpAccuracy, Runner, case, f32_tensor, scaled};
use super::bit_stability::{HostTensor, VerifyError};
use crate::kernel::{BindingEntry, OpParams};
use fuel_graph::QuantType;

/// `(rows, n, k)`: decode, ragged rows, large `k`, prefill-sized, and `k` off the 64-column chunk.
pub(crate) const QMATMUL_SHAPES: [(usize, usize, usize); 11] = [
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

/// Quantize `n x k` uniform weights to `T`; returns the block bytes and the dequantized weights.
fn quantize<T: GgmlType>(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let w = scaled(n * k, 0x3000 + seed, 1.0);
    let mut blocks = vec![T::zeros(); n * k / T::BLCK_SIZE];
    T::from_float(&w, &mut blocks);
    let mut deq = vec![0.0f32; n * k];
    T::to_float(&blocks, &mut deq);
    (bytemuck::cast_slice(&blocks).to_vec(), deq)
}

/// `x[rows, k]` times the dequantized `w[n, k]` transposed, in f64.
fn reference(x: &[f32], w: &[f32], (rows, n, k): (usize, usize, usize)) -> Vec<f64> {
    let mut want = vec![0.0f64; rows * n];
    for r in 0..rows {
        for o in 0..n {
            want[r * n + o] = (0..k)
                .map(|c| f64::from(x[r * k + c]) * f64::from(w[o * k + c]))
                .sum();
        }
    }
    want
}

fn measure_format<T: GgmlType>(
    run: &Runner,
    entry: &BindingEntry,
    (q, name): (QuantType, &str),
    cases: &mut Vec<super::accuracy_common::CaseError>,
) -> Result<(), VerifyError> {
    for &(rows, n, k) in &QMATMUL_SHAPES {
        if k % T::BLCK_SIZE != 0 {
            continue;
        }
        for seed in SEEDS {
            let (w_bytes, w_deq) = quantize::<T>(n, k, seed);
            let x = scaled(rows * k, 0x4000 + seed, 1.0);
            let weights = HostTensor {
                dtype: DType::U32,
                shape: vec![w_bytes.len().div_ceil(4)],
                bytes: w_bytes,
            };
            let params = OpParams::QMatMul {
                quant_type: q,
                batch_count: 1,
                m: rows,
                n,
                k,
            };
            let got = run(
                entry,
                params,
                &[f32_tensor(vec![rows, k], &x), weights],
                vec![rows, n],
            )?;
            let want = reference(&x, &w_deq, (rows, n, k));
            cases.push(case(
                format!("{name} rows={rows} n={n} k={k} seed={seed}"),
                &got,
                &want,
            ));
        }
    }
    Ok(())
}

pub(crate) fn measure_qmatmul(
    run: &Runner,
    entry: &BindingEntry,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let mut cases = Vec::new();
    measure_format::<BlockQ4_0>(run, entry, (QuantType::Q4_0, "q4_0"), &mut cases)?;
    measure_format::<BlockQ8_0>(run, entry, (QuantType::Q8_0, "q8_0"), &mut cases)?;
    measure_format::<BlockQ4K>(run, entry, (QuantType::Q4_K_M, "q4_k"), &mut cases)?;
    Ok(OpAccuracy::new(
        OpKind::QMatMul,
        &[DType::F32, DType::U32, DType::F32],
        claim,
        entry,
        cases,
        "f64 product of the f32 activations and the dequantized weights",
        "uniform [-1, 1) activations and weights (Q4_0, Q8_0, Q4_K)",
    ))
}
