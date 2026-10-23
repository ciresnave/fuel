// SPDX-License-Identifier: MIT OR Apache-2.0
//! Earns the Vulkan f32 accuracy claims (`max_relative`, `max_absolute`) in the verification
//! ledger for the kernels a decode step needs: `MatMul`, `RmsNormLastDim`, `SoftmaxLastDim` and
//! `Rope` (board 240, option D).
//!
//! The default placement rule ([`crate::ranker::filters::AccuracyPreferenceFilter`]) lets a
//! non-bit-stable kernel win placement on cost only if its accuracy claim meets
//! [`crate::ranker::filters::DEFAULT_ACCURACY_BOUND`]. The Vulkan contracts declare such claims,
//! and [`super::gate_precision`] collapses a numeric claim that has no passing ledger record to
//! `UNAUDITED`; this module measures the claims and writes the records.
//!
//! Method: drive the REGISTERED Vulkan kernel (the binding-table entry, through
//! [`super::VulkanInvoker`]) over shapes taken from a Qwen3-0.6B decode and prefill step plus
//! wider ones, with deterministic inputs (range per op below). Per element:
//! `abs = |out - ref|`, `rel = |out - ref| / max(|ref|, 1.0)`. The reference is an `f64` host
//! computation for `MatMul`, `RmsNormLastDim` and `SoftmaxLastDim`, and the registered
//! `portable-cpu` kernel for `Rope` (a pointwise rotation: the CPU kernel is the oracle the
//! fixtures compare against, and re-deriving its rotation convention here would be a second
//! place to be wrong). A record is `pass` only if the worst `abs` and `rel` are within the
//! declared claim; a miss writes `fail`, never a pass. The adapter is recorded in the evidence:
//! a ledger record has no adapter field, so the evidence is the only place the hardware is named.
//!
//! The claim values here are the ones in `kernel-contracts/vulkan/{matmul,norm,rope}.fkc.md`;
//! the `claim_constants_match_the_contracts` test pins them together.

use std::sync::Arc;

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_vulkan_backend::VulkanBackend;

use super::accuracy_common::{
    Claim, MATMUL_SHAPES, OpAccuracy, case, f32_tensor, f32s, find_entry, reference_matmul, scaled,
};
use super::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use super::invoker_cpu::CpuInvoker;
use super::invoker_vulkan::VulkanInvoker;
use super::ledger::LedgerRecord;
use crate::kernel::{BindingEntry, KernelBindingTable, MatmulM, OpParams};

/// The `backend` field of the records this module writes.
const BACKEND: &str = "Vulkan";
/// The `harness` field of their evidence.
const HARNESS: &str = "board-240/seed_vulkan_matmul_accuracy";

/// `matmul_f32` (`kernel-contracts/vulkan/matmul.fkc.md`), unit-scale inputs.
pub(crate) const MATMUL_F32_CLAIM: Claim = Claim {
    max_relative: 1.0e-3,
    max_absolute: 1.0e-3,
};
/// `rms_norm_last_dim_f32` (`kernel-contracts/vulkan/norm.fkc.md`).
pub(crate) const RMS_NORM_F32_CLAIM: Claim = Claim {
    max_relative: 1.0e-5,
    max_absolute: 1.0e-5,
};
/// `softmax_last_dim_f32` (`kernel-contracts/vulkan/norm.fkc.md`).
pub(crate) const SOFTMAX_F32_CLAIM: Claim = Claim {
    max_relative: 1.0e-5,
    max_absolute: 1.0e-5,
};
/// `rope_f32` (`kernel-contracts/vulkan/rope.fkc.md`).
pub(crate) const ROPE_F32_CLAIM: Claim = Claim {
    max_relative: 1.0e-5,
    max_absolute: 1.0e-5,
};

/// `eps` the norm cases run with (the probe recipes' value).
const NORM_EPS: f64 = 1.0e-5;

/// `f64` host reference of RMS norm without affine: `x / sqrt(mean(x^2) + eps)` per row.
fn reference_rms_norm(x: &[f32], rows: usize, cols: usize, eps: f64) -> Vec<f64> {
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
fn reference_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
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

/// Run `entry` on the Vulkan device with `inputs`/`params`, returning the f32 output.
fn run_vulkan(
    backend: &Arc<VulkanBackend>,
    entry: &BindingEntry,
    params: OpParams,
    inputs: &[HostTensor],
    out_shape: Vec<usize>,
) -> Result<Vec<f32>, VerifyError> {
    let inv = VulkanInvoker::new(backend.clone(), DType::F32, out_shape).with_params(params);
    Ok(f32s(&inv.invoke(entry, inputs)?))
}

/// `(rows, cols)` for the last-dim ops: decode rows (1 token, 16 q heads x head_dim 128),
/// prefill rows (21 tokens), wide rows up to 4096.
const NORM_SHAPES: &[(usize, usize)] = &[
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
const SOFTMAX_SHAPES: &[(usize, usize)] = &[
    (1, 22),
    (16, 22),
    (16, 128),
    (28, 1024),
    (16, 2048),
    (8, 4096),
    (1, 8192),
];

/// `(outer, seq, head_dim)`: decode (seq 1), prefill (seq 21), and wider.
const ROPE_SHAPES: &[(usize, usize, usize)] = &[
    (16, 1, 128),
    (8, 1, 128),
    (16, 21, 128),
    (1, 64, 128),
    (4, 512, 64),
];

fn measure_matmul(
    backend: &Arc<VulkanBackend>,
    table: &KernelBindingTable,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32, DType::F32];
    let entry = find_entry(table, OpKind::MatMul, &dt, BackendId::Vulkan)?;
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
        let got = run_vulkan(
            backend,
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
        MATMUL_F32_CLAIM,
        entry,
        cases,
        "f64 host matmul of the same f32 inputs",
        "uniform [-1, 1)",
    ))
}

fn measure_rms_norm(
    backend: &Arc<VulkanBackend>,
    table: &KernelBindingTable,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32];
    let entry = find_entry(table, OpKind::RmsNormLastDim, &dt, BackendId::Vulkan)?;
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in NORM_SHAPES.iter().enumerate() {
        let x = scaled(rows * cols, 0x3000 + idx as u64, 2.0);
        let params = OpParams::NormLastDim {
            outer_count: rows,
            last_dim: cols,
            eps: NORM_EPS,
        };
        let got = run_vulkan(
            backend,
            entry,
            params,
            &[f32_tensor(vec![rows * cols], &x)],
            vec![rows * cols],
        )?;
        let want = reference_rms_norm(&x, rows, cols, NORM_EPS);
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::RmsNormLastDim,
        &dt,
        RMS_NORM_F32_CLAIM,
        entry,
        cases,
        "f64 host x / sqrt(mean(x^2) + eps)",
        "uniform [-2, 2)",
    ))
}

fn measure_softmax(
    backend: &Arc<VulkanBackend>,
    table: &KernelBindingTable,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32];
    let entry = find_entry(table, OpKind::SoftmaxLastDim, &dt, BackendId::Vulkan)?;
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in SOFTMAX_SHAPES.iter().enumerate() {
        let x = scaled(rows * cols, 0x4000 + idx as u64, 4.0);
        let params = OpParams::SoftmaxLastDim {
            outer_count: rows,
            last_dim: cols,
        };
        let got = run_vulkan(
            backend,
            entry,
            params,
            &[f32_tensor(vec![rows * cols], &x)],
            vec![rows * cols],
        )?;
        let want = reference_softmax(&x, rows, cols);
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        OpKind::SoftmaxLastDim,
        &dt,
        SOFTMAX_F32_CLAIM,
        entry,
        cases,
        "f64 host exp(x - max) / sum",
        "uniform [-4, 4)",
    ))
}

/// `cos`/`sin` tables `[seq, head_dim]` with the rotate-half layout (the angle repeats over
/// the two halves), position `p`, frequency `10000^(-2j / head_dim)`.
fn rope_tables(seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
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

fn measure_rope(
    backend: &Arc<VulkanBackend>,
    table: &KernelBindingTable,
    cpu_table: &KernelBindingTable,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32, DType::F32, DType::F32];
    let entry = find_entry(table, OpKind::Rope, &dt, BackendId::Vulkan)?;
    let cpu_entry = find_entry(cpu_table, OpKind::Rope, &dt, BackendId::Cpu)?;
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
        let got = run_vulkan(
            backend,
            entry,
            params.clone(),
            &inputs,
            vec![outer, seq, hd],
        )?;
        let cpu = CpuInvoker::new(DType::F32, vec![outer, seq, hd]).with_params(params);
        let want: Vec<f64> = f32s(&cpu.invoke(cpu_entry, &inputs)?)
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
        ROPE_F32_CLAIM,
        entry,
        cases,
        "registered portable-cpu Rope kernel (f32)",
        "x uniform [-1, 1); cos/sin of position * 10000^(-2j/hd)",
    ))
}

/// Everything one run measured.
#[derive(Debug)]
pub(crate) struct AccuracyRun {
    pub device_name: String,
    pub ops: Vec<OpAccuracy>,
}

impl AccuracyRun {
    pub(crate) fn records(&self) -> Vec<LedgerRecord> {
        self.ops
            .iter()
            .flat_map(|o| o.records(BACKEND, HARNESS, &self.device_name))
            .collect()
    }
}

/// Measure the registered Vulkan f32 `MatMul`, `RmsNormLastDim`, `SoftmaxLastDim` and `Rope`.
pub(crate) fn measure_vulkan_f32_accuracy(
    selection: fuel_vulkan_backend::DeviceSelection,
) -> Result<AccuracyRun, VerifyError> {
    let mut table = KernelBindingTable::new();
    crate::vulkan_dispatch::register_vulkan_kernels(&mut table);
    let mut cpu_table = KernelBindingTable::new();
    crate::dispatch::register_cpu_kernels(&mut cpu_table);
    let backend = VulkanBackend::with_selection(selection)
        .map(Arc::new)
        .map_err(|e| VerifyError::Backend(format!("no Vulkan device: {e}")))?;
    let device_name = backend.device_name.clone();
    let ops = vec![
        measure_matmul(&backend, &table)?,
        measure_rms_norm(&backend, &table)?,
        measure_softmax(&backend, &table)?,
        measure_rope(&backend, &table, &cpu_table)?,
    ];
    Ok(AccuracyRun { device_name, ops })
}

fn print_run(run: &AccuracyRun) {
    println!("[board-240] adapter: {}", run.device_name);
    for o in &run.ops {
        for c in &o.cases {
            println!(
                "[board-240] {:?} {:32} abs {:.3e}  rel {:.3e}",
                o.op, c.label, c.abs, c.rel
            );
        }
        println!(
            "[board-240] {:?} WORST abs {:.3e} (claim {}) rel {:.3e} (claim {})",
            o.op, o.worst_abs, o.claim.max_absolute, o.worst_rel, o.claim.max_relative
        );
    }
}

/// Measure on the discrete adapter, print the tables and WRITE the records through the one
/// merging writer. Needs a live Vulkan device. Run via
/// `cargo run -p fuel-dispatch --features dev-tools,vulkan --bin fuel-dev-tools -- seed-vulkan-accuracy`.
#[cfg(feature = "dev-tools")]
pub(crate) fn run_seed_vulkan_matmul_accuracy() {
    let run = measure_vulkan_f32_accuracy(fuel_vulkan_backend::DeviceSelection::PreferDiscrete)
        .expect("vulkan f32 accuracy measurement runs");
    print_run(&run);
    let records = run.records();
    for r in &records {
        println!(
            "[board-240] {} {} -> {}",
            r.evidence["op"].as_str().unwrap_or("?"),
            r.claim,
            r.result
        );
    }
    assert!(
        records.iter().all(|r| r.result == "pass"),
        "a measurement above the declared claim must not be written as a pass; fix the kernel or \
         raise the claim in the contract AND in this module"
    );
    let summary = super::ledger::write_merged_ledger(&records);
    println!(
        "[board-240] merged {} record(s) into {} existing -> {} total, written to {}",
        summary.fresh,
        summary.before,
        summary.after,
        summary.path.display()
    );
}

/// Measure and print against the claims, WRITING NOTHING. `FUEL_VULKAN_TEST_ADAPTER` (a
/// case-insensitive adapter-name substring) selects the adapter; unset means the discrete card.
/// Exits non-zero when any measurement exceeds its claim. Run via
/// `cargo run -p fuel-dispatch --features dev-tools,vulkan --bin fuel-dev-tools -- check-vulkan-accuracy`.
#[cfg(feature = "dev-tools")]
pub(crate) fn run_check_vulkan_accuracy() {
    let selection = match std::env::var("FUEL_VULKAN_TEST_ADAPTER") {
        Ok(name) => fuel_vulkan_backend::DeviceSelection::ByName(name),
        Err(_) => fuel_vulkan_backend::DeviceSelection::PreferDiscrete,
    };
    let run = measure_vulkan_f32_accuracy(selection).expect("vulkan f32 accuracy measurement runs");
    print_run(&run);
    let failing: Vec<_> = run
        .records()
        .into_iter()
        .filter(|r| r.result != "pass")
        .collect();
    for r in &failing {
        println!(
            "[board-240] EXCEEDS CLAIM: {} {}",
            r.evidence["op"], r.claim
        );
    }
    assert!(
        failing.is_empty(),
        "{} measurement(s) exceed their claim",
        failing.len()
    );
}

#[cfg(test)]
mod tests {
    use super::super::accuracy_common::worst_error;
    use super::*;

    /// Sabotage sibling: a corrupted element must exceed the claim, so a green measurement
    /// cannot come from a comparison that never looks at the values.
    #[test]
    fn worst_error_sees_a_corrupted_element() {
        let want = vec![1.0f64, 2.0, 3.0];
        let mut got = vec![1.0f32, 2.0, 3.0];
        let (abs, rel) = worst_error(&got, &want);
        assert!(abs <= ROPE_F32_CLAIM.max_absolute && rel <= ROPE_F32_CLAIM.max_relative);
        got[1] += 0.01;
        let (abs, rel) = worst_error(&got, &want);
        assert!(abs > MATMUL_F32_CLAIM.max_absolute || rel > MATMUL_F32_CLAIM.max_relative);
    }

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

    /// Each claim constant and its contract section say the same numbers: a claim edited in one
    /// place only would be earned against a bound the contract does not declare.
    #[test]
    fn claim_constants_match_the_contracts() {
        fn section<'a>(text: &'a str, kernel: &str) -> &'a str {
            let marker = format!("kernel: {kernel}");
            let mut start = 0;
            loop {
                let at = text[start..]
                    .find(&marker)
                    .unwrap_or_else(|| panic!("section `{kernel}` present"));
                let after = start + at + marker.len();
                // `kernel: rms_norm_last_dim` is a prefix of `kernel: rms_norm_last_dim_f32`:
                // require the name to END at the line.
                if matches!(text.as_bytes().get(after), Some(b'\r' | b'\n')) {
                    let begin = start + at;
                    let rest = &text[begin..];
                    return &rest[..rest.find("```").expect("section ends")];
                }
                start = after;
            }
        }
        let cases = [
            (
                include_str!("../../../kernel-contracts/vulkan/matmul.fkc.md"),
                "matmul_f32",
                MATMUL_F32_CLAIM,
            ),
            (
                include_str!("../../../kernel-contracts/vulkan/norm.fkc.md"),
                "rms_norm_last_dim_f32",
                RMS_NORM_F32_CLAIM,
            ),
            (
                include_str!("../../../kernel-contracts/vulkan/norm.fkc.md"),
                "softmax_last_dim_f32",
                SOFTMAX_F32_CLAIM,
            ),
            (
                include_str!("../../../kernel-contracts/vulkan/rope.fkc.md"),
                "rope_f32",
                ROPE_F32_CLAIM,
            ),
        ];
        for (text, kernel, claim) in cases {
            let s = section(text, kernel);
            assert!(
                s.contains(&format!("max_relative: {}", claim.max_relative)),
                "{kernel}: contract max_relative differs from the claim constant"
            );
            assert!(
                s.contains(&format!("max_absolute: {}", claim.max_absolute)),
                "{kernel}: contract max_absolute differs from the claim constant"
            );
        }
    }

    /// Every claim is within the default placement bound, or earning it would not let the
    /// kernel win by default.
    #[test]
    fn every_claim_is_within_the_default_bound() {
        use crate::ranker::filters::DEFAULT_ACCURACY_BOUND;
        for c in [
            MATMUL_F32_CLAIM,
            RMS_NORM_F32_CLAIM,
            SOFTMAX_F32_CLAIM,
            ROPE_F32_CLAIM,
        ] {
            assert!(c.max_relative <= DEFAULT_ACCURACY_BOUND.max_relative);
            assert!(c.max_absolute <= DEFAULT_ACCURACY_BOUND.max_absolute);
        }
    }
}
