// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measures the registered f16 and bf16 CUDA `MatMul`, `RmsNormLastDim` and `SoftmaxLastDim`
//! kernels against `f64` host references of the same half-precision inputs (ROADMAP row 39).
//!
//! Same cases, shapes and references as the f32 measurements in [`super::accuracy_common`]; the
//! inputs are rounded to the kernel dtype first (the reference sees the rounded values), and the
//! kernel output is decoded back to `f32`. The output itself is rounded to the dtype, so the
//! error floor is half an ulp of the result: that, not the kernel, can exceed a bound.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;

use super::accuracy_attention::to_half;
use super::accuracy_common::{
    Claim, MATMUL_SHAPES, NORM_EPS, NORM_SHAPES, OpAccuracy, Runner, SOFTMAX_SHAPES, case,
    reference_matmul, reference_rms_norm, reference_softmax, scaled,
};
use super::bit_stability::VerifyError;
use crate::kernel::{BindingEntry, MatmulM, OpParams};

/// Shapes beyond the f32 set that separate the gemm routes by `k`: the tiled route (`m >= 32`)
/// at `n = 1024` leaves the output-rounding floor once `k >= 4096`, which the f32 set never
/// reaches. Measured on the RTX 4070 Laptop; see the GAP row for the finding.
const HALF_EXTRA_SHAPES: &[(usize, usize, usize)] = &[
    (127, 1024, 1024),
    (127, 1024, 2048),
    (64, 1024, 4096),
    (32, 1024, 4096),
    (127, 64, 4096),
    (127, 1024, 8192),
];

pub(crate) fn measure_half_matmul(
    run: &Runner,
    entry: &BindingEntry,
    dtype: DType,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let mut cases = Vec::new();
    for (idx, &(m, n, k)) in MATMUL_SHAPES.iter().chain(HALF_EXTRA_SHAPES).enumerate() {
        let (at, a) = to_half(dtype, vec![m, k], &scaled(m * k, 0x1000 + idx as u64, 1.0));
        let (wt, w) = to_half(dtype, vec![k, n], &scaled(k * n, 0x2000 + idx as u64, 1.0));
        let params = OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m,
            n,
            k,
            m_compute: MatmulM::All,
        };
        let got = run(entry, params, &[at, wt], vec![m, n])
            .map_err(|e| VerifyError::Backend(format!("{dtype:?} m={m} n={n} k={k}: {e:?}")))?;
        let want = reference_matmul(&a, &w, m, n, k);
        cases.push(case(format!("m={m} n={n} k={k}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::MatMul,
        &[dtype; 3],
        claim,
        entry,
        cases,
        "f64 host matmul of the same half-precision inputs",
        "uniform [-1, 1) rounded to the kernel dtype",
    ))
}

pub(crate) fn measure_half_rms_norm(
    run: &Runner,
    entry: &BindingEntry,
    dtype: DType,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in NORM_SHAPES.iter().enumerate() {
        let (xt, x) = to_half(
            dtype,
            vec![rows, cols],
            &scaled(rows * cols, 0x3000 + idx as u64, 2.0),
        );
        let params = OpParams::NormLastDim {
            outer_count: rows,
            last_dim: cols,
            eps: NORM_EPS,
        };
        let got = run(entry, params, &[xt], vec![rows, cols])?;
        let want = reference_rms_norm(&x, rows, cols, NORM_EPS);
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::RmsNormLastDim,
        &[dtype; 2],
        claim,
        entry,
        cases,
        "f64 host x / sqrt(mean(x^2) + eps) of the same half-precision inputs",
        "uniform [-2, 2) rounded to the kernel dtype",
    ))
}

pub(crate) fn measure_half_softmax(
    run: &Runner,
    entry: &BindingEntry,
    dtype: DType,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in SOFTMAX_SHAPES.iter().enumerate() {
        let (xt, x) = to_half(
            dtype,
            vec![rows, cols],
            &scaled(rows * cols, 0x4000 + idx as u64, 4.0),
        );
        let params = OpParams::SoftmaxLastDim {
            outer_count: rows,
            last_dim: cols,
        };
        let got = run(entry, params, &[xt], vec![rows, cols])?;
        let want = reference_softmax(&x, rows, cols);
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::SoftmaxLastDim,
        &[dtype; 2],
        claim,
        entry,
        cases,
        "f64 host exp(x - max) / sum of the same half-precision inputs",
        "uniform [-4, 4) rounded to the kernel dtype",
    ))
}
