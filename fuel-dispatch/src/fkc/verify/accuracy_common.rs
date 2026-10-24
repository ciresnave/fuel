// SPDX-License-Identifier: MIT OR Apache-2.0
//! The parts of the accuracy-claim seeders that do not depend on a backend: the claim and error
//! types, the `f64` host references, the deterministic inputs, and the ledger-record shape.
//! Used by [`super::seed_vulkan_matmul_accuracy`] and [`super::seed_cuda_matmul_accuracy`]
//! (board 240, option D), so the two backends are measured by the same metric and the same
//! reference.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;

use super::bit_stability::{HostTensor, VerifyError, fill_deterministic};
use super::ledger::LedgerRecord;
use crate::kernel::{BindingEntry, KernelBindingTable, MatmulM, OpParams};

/// A declared claim: ceilings on the kernel's `max_relative` and `max_absolute`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Claim {
    pub max_relative: f64,
    pub max_absolute: f64,
}

/// The floor under `|ref|` in the relative metric.
pub(crate) const REL_FLOOR: f64 = 1.0;

/// One case's measured error.
#[derive(Debug, Clone)]
pub(crate) struct CaseError {
    pub label: String,
    pub abs: f64,
    pub rel: f64,
}

/// One op's measurement against its claim.
#[derive(Debug)]
pub(crate) struct OpAccuracy {
    pub op: OpKind,
    pub dtypes: Vec<DType>,
    pub claim: Claim,
    pub cases: Vec<CaseError>,
    pub worst_abs: f64,
    pub worst_rel: f64,
    pub reference: &'static str,
    pub input_range: &'static str,
    pub kernel_source: String,
    pub kernel_revision_hash: u64,
}

impl OpAccuracy {
    pub(crate) fn new(
        op: OpKind,
        dtypes: &[DType],
        claim: Claim,
        entry: &BindingEntry,
        cases: Vec<CaseError>,
        reference: &'static str,
        input_range: &'static str,
    ) -> Self {
        let worst_abs = cases.iter().map(|c| c.abs).fold(0.0, f64::max);
        let worst_rel = cases.iter().map(|c| c.rel).fold(0.0, f64::max);
        Self {
            op,
            dtypes: dtypes.to_vec(),
            claim,
            cases,
            worst_abs,
            worst_rel,
            reference,
            input_range,
            kernel_source: entry.kernel_source.to_string(),
            kernel_revision_hash: entry.kernel_revision_hash,
        }
    }

    /// The two ledger records (`max_absolute`, `max_relative`) for this measurement.
    pub(crate) fn records(&self, backend: &str, harness: &str, adapter: &str) -> Vec<LedgerRecord> {
        let cases_json: Vec<serde_json::Value> = self
            .cases
            .iter()
            .map(|c| serde_json::json!({"case": c.label, "abs": c.abs, "rel": c.rel}))
            .collect();
        [
            ("max_absolute", self.worst_abs, self.claim.max_absolute),
            ("max_relative", self.worst_rel, self.claim.max_relative),
        ]
        .into_iter()
        .map(|(claim, worst, bound)| LedgerRecord {
            kernel_ref: self.kernel_source.clone(),
            backend: backend.to_string(),
            dtypes: self.dtypes.iter().map(|d| format!("{d:?}")).collect(),
            kernel_revision_hash: self.kernel_revision_hash,
            claim: claim.to_string(),
            result: if worst <= bound { "pass" } else { "fail" }.to_string(),
            verified_at: verified_at_string(),
            protocol_version: 1,
            evidence: serde_json::json!({
                "op": format!("{:?}", self.op),
                "bound": bound,
                "measured_worst": worst,
                "reference": self.reference,
                "metric": "abs = |out-ref|; rel = |out-ref| / max(|ref|, 1.0)",
                "input_range": self.input_range,
                "adapter": adapter,
                "cases": cases_json,
                "harness": harness,
            }),
        })
        .collect()
    }
}

/// Worst per-element `(abs, rel)` error of `got` against `want`.
pub(crate) fn worst_error(got: &[f32], want: &[f64]) -> (f64, f64) {
    let mut abs = 0f64;
    let mut rel = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = (g as f64 - w).abs();
        abs = abs.max(d);
        rel = rel.max(d / w.abs().max(REL_FLOOR));
    }
    (abs, rel)
}

/// `f64` host reference of `[m, k] @ [k, n]` over `f32` inputs.
pub(crate) fn reference_matmul(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk] as f64;
            for j in 0..n {
                out[i * n + j] += av * w[kk * n + j] as f64;
            }
        }
    }
    out
}

/// `fill_deterministic` yields `[-0.5, 0.5)`; scale it to `[-half_range, half_range)`.
pub(crate) fn scaled(len: usize, seed: u64, half_range: f32) -> Vec<f32> {
    fill_deterministic(len, seed)
        .into_iter()
        .map(|x| x * 2.0 * half_range)
        .collect()
}

pub(crate) fn f32_tensor(shape: Vec<usize>, data: &[f32]) -> HostTensor {
    HostTensor {
        dtype: DType::F32,
        shape,
        bytes: bytemuck::cast_slice(data).to_vec(),
    }
}

pub(crate) fn f32s(t: &HostTensor) -> Vec<f32> {
    t.bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

pub(crate) fn find_entry<'a>(
    table: &'a KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    backend: BackendId,
) -> Result<&'a BindingEntry, VerifyError> {
    table
        .iter_entries()
        .find(|(o, d, b, _)| *o == op && *d == dtypes && *b == backend)
        .map(|(_, _, _, e)| e)
        .ok_or_else(|| VerifyError::Backend(format!("no {backend:?} {op:?} {dtypes:?} entry")))
}

pub(crate) fn case(label: String, got: &[f32], want: &[f64]) -> CaseError {
    let (abs, rel) = worst_error(got, want);
    CaseError { label, abs, rel }
}

/// `(m, n, k)`: all three routes of the matmul wrapper (`M == 1` matvec, `1 < M < 32`
/// register-tiled, `M >= 32` shared-memory tiled); `k` up to 4096 (Qwen3-0.6B hidden and FFN
/// widths are 1024 and 3072; the 8192-wide output stands in for the vocabulary projection).
pub(crate) const MATMUL_SHAPES: &[(usize, usize, usize)] = &[
    (1, 1024, 1024),
    (1, 3072, 1024),
    (1, 1024, 3072),
    (1, 8192, 1024),
    (1, 64, 4096),
    (4, 1024, 1024),
    (16, 3072, 1024),
    (31, 512, 2048),
    (32, 1024, 1024),
    (64, 3072, 1024),
    (127, 1024, 4096),
    (21, 1024, 1024),
];

pub(crate) fn verified_at_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

/// Runs a registered kernel on a device: `(entry, params, inputs, out_shape) -> f32 output`.
pub(crate) type Runner<'a> = dyn Fn(&BindingEntry, OpParams, &[HostTensor], Vec<usize>) -> Result<Vec<f32>, VerifyError>
    + 'a;

/// The reference a kernel is compared with when no `f64` host formula exists:
/// `(params, inputs, out_shape) -> f32 output`.
pub(crate) type Reference<'a> =
    dyn Fn(OpParams, &[HostTensor], Vec<usize>) -> Result<Vec<f32>, VerifyError> + 'a;

/// `eps` the norm cases run with (the probe recipes' value).
pub(crate) const NORM_EPS: f64 = 1.0e-5;

/// `f64` host reference of RMS norm without affine: `x / sqrt(mean(x^2) + eps)` per row.
pub(crate) fn reference_rms_norm(x: &[f32], rows: usize, cols: usize, eps: f64) -> Vec<f64> {
    let mut out = vec![0f64; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / cols as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = row[c] as f64 * inv;
        }
    }
    out
}

/// `f64` host reference of softmax along the last dim.
pub(crate) fn reference_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
        let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
        for c in 0..cols {
            out[r * cols + c] = (row[c] as f64 - max).exp() / sum;
        }
    }
    out
}

/// `(rows, cols)` for the last-dim ops: decode rows (1 token, 16 q heads x head_dim 128),
/// prefill rows (21 tokens), wide rows up to 4096.
pub(crate) const NORM_SHAPES: &[(usize, usize)] = &[
    (1, 1024),
    (16, 1024),
    (21, 1024),
    (1, 3072),
    (8, 128),
    (16, 128),
    (448, 128),
    (1, 4096),
    (32, 64),
];

/// `(rows, cols)` for softmax: attention rows over a growing KV length, and a wide row.
pub(crate) const SOFTMAX_SHAPES: &[(usize, usize)] = &[
    (1, 22),
    (16, 22),
    (16, 128),
    (28, 1024),
    (16, 2048),
    (8, 4096),
    (1, 8192),
];

/// `(outer, seq, head_dim)`: decode (seq 1), prefill (seq 21), and wider.
pub(crate) const ROPE_SHAPES: &[(usize, usize, usize)] = &[
    (16, 1, 128),
    (8, 1, 128),
    (16, 21, 128),
    (1, 64, 128),
    (4, 512, 64),
];

pub(crate) fn measure_matmul(
    run: &Runner,
    entry: &BindingEntry,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32, DType::F32];
    let mut cases = Vec::new();
    for (idx, &(m, n, k)) in MATMUL_SHAPES.iter().enumerate() {
        let a = scaled(m * k, 0x1000 + idx as u64, 1.0);
        let w = scaled(k * n, 0x2000 + idx as u64, 1.0);
        let params = OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m,
            n,
            k,
            m_compute: MatmulM::All,
        };
        let got = run(
            entry,
            params,
            &[f32_tensor(vec![m, k], &a), f32_tensor(vec![k, n], &w)],
            vec![m, n],
        )?;
        let want = reference_matmul(&a, &w, m, n, k);
        cases.push(case(format!("m={m} n={n} k={k}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::MatMul,
        &dt,
        claim,
        entry,
        cases,
        "f64 host matmul of the same f32 inputs",
        "uniform [-1, 1)",
    ))
}

pub(crate) fn measure_rms_norm(
    run: &Runner,
    entry: &BindingEntry,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32];
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in NORM_SHAPES.iter().enumerate() {
        let x = scaled(rows * cols, 0x3000 + idx as u64, 2.0);
        let params = OpParams::NormLastDim {
            outer_count: rows,
            last_dim: cols,
            eps: NORM_EPS,
        };
        let got = run(
            entry,
            params,
            &[f32_tensor(vec![rows, cols], &x)],
            vec![rows, cols],
        )?;
        let want = reference_rms_norm(&x, rows, cols, NORM_EPS);
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::RmsNormLastDim,
        &dt,
        claim,
        entry,
        cases,
        "f64 host x / sqrt(mean(x^2) + eps)",
        "uniform [-2, 2)",
    ))
}

pub(crate) fn measure_softmax(
    run: &Runner,
    entry: &BindingEntry,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32];
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in SOFTMAX_SHAPES.iter().enumerate() {
        let x = scaled(rows * cols, 0x4000 + idx as u64, 4.0);
        let params = OpParams::SoftmaxLastDim {
            outer_count: rows,
            last_dim: cols,
        };
        let got = run(
            entry,
            params,
            &[f32_tensor(vec![rows, cols], &x)],
            vec![rows, cols],
        )?;
        let want = reference_softmax(&x, rows, cols);
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::SoftmaxLastDim,
        &dt,
        claim,
        entry,
        cases,
        "f64 host exp(x - max) / sum",
        "uniform [-4, 4)",
    ))
}

/// `cos`/`sin` tables `[seq, head_dim]` with the rotate-half layout (the angle repeats over
/// the two halves), position `p`, frequency `10000^(-2j / head_dim)`.
pub(crate) fn rope_tables(seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let mut cos = vec![0f32; seq * hd];
    let mut sin = vec![0f32; seq * hd];
    for p in 0..seq {
        for j in 0..hd {
            let freq = 10_000f64.powf(-2.0 * (j % half) as f64 / hd as f64);
            let theta = p as f64 * freq;
            cos[p * hd + j] = theta.cos() as f32;
            sin[p * hd + j] = theta.sin() as f32;
        }
    }
    (cos, sin)
}

pub(crate) fn measure_rope(
    run: &Runner,
    reference: &Reference,
    entry: &BindingEntry,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32, DType::F32, DType::F32];
    let mut cases = Vec::new();
    for (idx, &(outer, seq, hd)) in ROPE_SHAPES.iter().enumerate() {
        let x = scaled(outer * seq * hd, 0x5000 + idx as u64, 1.0);
        let (cos, sin) = rope_tables(seq, hd);
        let inputs = [
            f32_tensor(vec![outer, seq, hd], &x),
            f32_tensor(vec![seq, hd], &cos),
            f32_tensor(vec![seq, hd], &sin),
        ];
        let params = OpParams::Rope {
            outer_count: outer,
            seq,
            head_dim: hd,
        };
        let got = run(entry, params.clone(), &inputs, vec![outer, seq, hd])?;
        let want: Vec<f64> = reference(params, &inputs, vec![outer, seq, hd])?
            .into_iter()
            .map(f64::from)
            .collect();
        cases.push(case(
            format!("outer={outer} seq={seq} hd={hd}"),
            &got,
            &want,
        ));
    }
    Ok(OpAccuracy::new(
        OpKind::Rope,
        &dt,
        claim,
        entry,
        cases,
        "registered portable-cpu Rope kernel (f32)",
        "x uniform [-1, 1); cos/sin of position * 10000^(-2j/hd)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_matmul_matches_a_hand_computed_product() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let r = reference_matmul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(r, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn reference_rms_norm_matches_a_hand_computed_row() {
        // row [3, 4]: mean(x^2) = 12.5; eps 0 -> x / sqrt(12.5)
        let r = reference_rms_norm(&[3.0, 4.0], 1, 2, 0.0);
        let inv = 1.0 / 12.5f64.sqrt();
        assert!((r[0] - 3.0 * inv).abs() < 1e-12 && (r[1] - 4.0 * inv).abs() < 1e-12);
    }

    #[test]
    fn reference_softmax_matches_a_hand_computed_row() {
        // row [0, ln 3]: probabilities 1/4 and 3/4
        let r = reference_softmax(&[0.0, 3f32.ln()], 1, 2);
        assert!((r[0] - 0.25).abs() < 1e-6 && (r[1] - 0.75).abs() < 1e-6);
        assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
