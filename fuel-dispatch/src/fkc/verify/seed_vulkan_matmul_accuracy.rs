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
    Claim, OpAccuracy, f32s, find_entry, measure_matmul, measure_rms_norm, measure_rope,
    measure_softmax,
};
use super::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use super::invoker_cpu::CpuInvoker;
use super::invoker_vulkan::VulkanInvoker;
use super::ledger::LedgerRecord;
use crate::kernel::{BindingEntry, KernelBindingTable, OpParams};

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
    let run = |entry: &BindingEntry, params: OpParams, inputs: &[HostTensor], shape: Vec<usize>| {
        run_vulkan(&backend, entry, params, inputs, shape)
    };
    let rope_dt = [DType::F32; 4];
    let cpu_rope = find_entry(&cpu_table, OpKind::Rope, &rope_dt, BackendId::Cpu)?;
    let cpu_reference = |params: OpParams, inputs: &[HostTensor], shape: Vec<usize>| {
        let cpu = CpuInvoker::new(DType::F32, shape).with_params(params);
        Ok(f32s(&cpu.invoke(cpu_rope, inputs)?))
    };
    let entry = |op: OpKind, dt: &[DType]| find_entry(&table, op, dt, BackendId::Vulkan);
    let ops = vec![
        measure_matmul(
            &run,
            entry(OpKind::MatMul, &[DType::F32; 3])?,
            MATMUL_F32_CLAIM,
        )?,
        measure_rms_norm(
            &run,
            entry(OpKind::RmsNormLastDim, &[DType::F32; 2])?,
            RMS_NORM_F32_CLAIM,
        )?,
        measure_softmax(
            &run,
            entry(OpKind::SoftmaxLastDim, &[DType::F32; 2])?,
            SOFTMAX_F32_CLAIM,
        )?,
        measure_rope(
            &run,
            &cpu_reference,
            entry(OpKind::Rope, &rope_dt)?,
            ROPE_F32_CLAIM,
        )?,
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
