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
use crate::kernel::{BindingEntry, KernelBindingTable};

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
