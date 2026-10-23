// SPDX-License-Identifier: MIT OR Apache-2.0
//! Earns the CUDA f32 `MatMul` accuracy claims (`max_relative`, `max_absolute`) in the
//! verification ledger (ROADMAP row 39, board 240 option D).
//!
//! Today a CUDA kernel is placed by the default rule only because its `bit_stable_on_same_hardware`
//! pass makes it "reference-grade" ([`crate::ranker::filters::AccuracyPreferenceFilter`]); 140 CUDA
//! kernels have that pass and none has a numeric accuracy claim. Row 39 narrows that clause to
//! the `portable-cpu` reference, so every other kernel needs a claim within
//! [`crate::ranker::filters::DEFAULT_ACCURACY_BOUND`]. This module earns the first one, for the
//! dense f32 matmul that carries a decode step.
//!
//! Method (the same as [`super::seed_vulkan_matmul_accuracy`], whose helpers it shares through
//! [`super::accuracy_common`]): drive the REGISTERED CUDA kernel (the binding-table entry,
//! through [`super::CudaInvoker`]) over the shapes Qwen3-0.6B decode and prefill use plus wider
//! ones, with deterministic inputs uniform in `[-1, 1)`, and compare per element against an
//! `f64` host matmul of the same `f32` inputs. A record is `pass` only if the worst `abs` and
//! `rel` are within the declared claim; a miss writes `fail`, never a pass. The device name is
//! recorded in the evidence: a ledger record has no device field.
//!
//! The claim values are the ones in `kernel-contracts/cuda/gemm_dense.fkc.md`; the
//! `claim_constants_match_the_contract` test pins them together.

use fuel_cuda_backend::CudaDevice;
use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;

use super::accuracy_common::{
    Claim, MATMUL_SHAPES, OpAccuracy, case, f32_tensor, f32s, find_entry, reference_matmul, scaled,
};
use super::bit_stability::{KernelInvoker, VerifyError};
use super::invoker_cuda::CudaInvoker;
use super::ledger::LedgerRecord;
use crate::kernel::{KernelBindingTable, MatmulM, OpParams};

/// The `backend` field of the records this module writes.
const BACKEND: &str = "Cuda";
/// The `harness` field of their evidence.
const HARNESS: &str = "row-39/seed_cuda_matmul_accuracy";

/// `matmul_f32` (`kernel-contracts/cuda/gemm_dense.fkc.md`), unit-scale inputs.
pub(crate) const MATMUL_F32_CLAIM: Claim = Claim {
    max_relative: 2.0e-4,
    max_absolute: 2.0e-4,
};

/// The CUDA device's marketing name, from the driver's own tool; `"unknown"` when it cannot be
/// read. The hardware is evidence for the claim, so a guess is worse than a gap.
fn device_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader", "--id=0"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn measure_matmul(
    device: &CudaDevice,
    table: &KernelBindingTable,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32, DType::F32];
    let entry = find_entry(table, OpKind::MatMul, &dt, BackendId::Cuda)?;
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
        let inv = CudaInvoker::new(device.clone(), DType::F32, vec![m, n]).with_params(params);
        let got = f32s(&inv.invoke(
            entry,
            &[f32_tensor(vec![m, k], &a), f32_tensor(vec![k, n], &w)],
        )?);
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

/// Measure the registered CUDA f32 `MatMul` on device 0. Returns the device name and the
/// measurement.
pub(crate) fn measure_cuda_f32_accuracy() -> Result<(String, Vec<OpAccuracy>), VerifyError> {
    let mut table = KernelBindingTable::new();
    crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
    let device =
        CudaDevice::new(0).map_err(|e| VerifyError::Backend(format!("no CUDA device: {e}")))?;
    Ok((device_name(), vec![measure_matmul(&device, &table)?]))
}

fn records(name: &str, ops: &[OpAccuracy]) -> Vec<LedgerRecord> {
    ops.iter()
        .flat_map(|o| o.records(BACKEND, HARNESS, name))
        .collect()
}

fn print_run(name: &str, ops: &[OpAccuracy]) {
    println!("[row-39] device: {name}");
    for o in ops {
        for c in &o.cases {
            println!(
                "[row-39] {:?} {:32} abs {:.3e}  rel {:.3e}",
                o.op, c.label, c.abs, c.rel
            );
        }
        println!(
            "[row-39] {:?} WORST abs {:.3e} (claim {}) rel {:.3e} (claim {})",
            o.op, o.worst_abs, o.claim.max_absolute, o.worst_rel, o.claim.max_relative
        );
    }
}

/// Measure, print and WRITE the records through the one merging writer. Needs a live CUDA
/// device. Run via `pwsh scripts/gpu-run.ps1 -Project fuel -- <fuel-dev-tools exe> seed-cuda-accuracy`.
#[cfg(feature = "dev-tools")]
pub(crate) fn run_seed_cuda_matmul_accuracy() {
    // The dtype list is part of a section's revision hash, so splitting the matmul section
    // re-keyed every MatMul entry: earn `bit_stable_on_same_hardware` for the entries that lack
    // a pass at their current hash (`force = false` leaves every other kernel's record alone).
    let (ledger, log) =
        super::seed_cuda_ledger::run_cuda_verification(Some(&[OpKind::MatMul]), false)
            .expect("cuda matmul bit-stability seeding runs");
    for a in &log {
        println!(
            "[row-39] {} {:?} (rev={}): {}",
            a.op, a.dtypes, a.kernel_revision_hash, a.outcome
        );
    }
    assert!(
        log.iter().all(|a| !a.outcome.starts_with("fail")),
        "a MatMul kernel failed bit-stability; see the log above"
    );
    let (name, ops) = measure_cuda_f32_accuracy().expect("cuda f32 accuracy measurement runs");
    print_run(&name, &ops);
    let recs = records(&name, &ops);
    assert!(
        recs.iter().all(|r| r.result == "pass"),
        "a measurement above the declared claim must not be written as a pass; fix the kernel or \
         raise the claim in the contract AND in this module"
    );
    // ONE write: the merging writer merges into the ledger EMBEDDED in this binary, not the file
    // on disk, so a second call would drop the first call's records.
    let mut all = ledger.records().to_vec();
    all.extend(recs);
    let summary = super::ledger::write_merged_ledger(&all);
    println!(
        "[row-39] merged {} record(s) into {} existing -> {} total, written to {}",
        summary.fresh,
        summary.before,
        summary.after,
        summary.path.display()
    );
}

/// Measure and print against the claims, WRITING NOTHING. Exits non-zero when any measurement
/// exceeds its claim.
#[cfg(feature = "dev-tools")]
pub(crate) fn run_check_cuda_accuracy() {
    let (name, ops) = measure_cuda_f32_accuracy().expect("cuda f32 accuracy measurement runs");
    print_run(&name, &ops);
    let failing = records(&name, &ops)
        .into_iter()
        .filter(|r| r.result != "pass")
        .count();
    assert!(failing == 0, "{failing} measurement(s) exceed their claim");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim constant and its contract section say the same numbers: a claim edited in one
    /// place only would be earned against a bound the contract does not declare.
    #[test]
    fn claim_constants_match_the_contract() {
        let text = include_str!("../../../kernel-contracts/cuda/gemm_dense.fkc.md");
        let at = text
            .find("kernel: matmul_f32")
            .expect("matmul_f32 section present");
        let rest = &text[at..];
        let prec = &rest[rest.find("precision:").expect("precision block")..];
        let prec = &prec[..prec.find("audited:").expect("precision block ends")];
        assert!(
            prec.contains(&format!("max_relative: {}", MATMUL_F32_CLAIM.max_relative)),
            "contract max_relative differs from the claim constant"
        );
        assert!(
            prec.contains(&format!("max_absolute: {}", MATMUL_F32_CLAIM.max_absolute)),
            "contract max_absolute differs from the claim constant"
        );
    }

    /// What the import gate leaves on the registered entries once the ledger backs the claim:
    /// the F32 matmul carries its numeric claim AND keeps `bit_stable`, and the F16 matmul (a
    /// different section, no numeric claim) is not collapsed to UNAUDITED by it.
    #[test]
    fn the_ledger_backs_the_f32_claim_and_leaves_the_other_dtypes_audited() {
        let mut table = KernelBindingTable::new();
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
        let f32e = find_entry(&table, OpKind::MatMul, &[DType::F32; 3], BackendId::Cuda)
            .expect("f32 entry");
        assert!(f32e.precision.bit_stable_on_same_hardware);
        assert_eq!(
            f32e.precision.max_relative,
            Some(MATMUL_F32_CLAIM.max_relative)
        );
        assert_eq!(
            f32e.precision.max_absolute,
            Some(MATMUL_F32_CLAIM.max_absolute)
        );
        let f16e = find_entry(&table, OpKind::MatMul, &[DType::F16; 3], BackendId::Cuda)
            .expect("f16 entry");
        assert!(f16e.precision.bit_stable_on_same_hardware);
        assert_eq!(f16e.precision.max_relative, None);
    }

    /// The claim is within the default placement bound, or earning it would not let the kernel
    /// win by default.
    #[test]
    fn the_claim_is_within_the_default_bound() {
        use crate::ranker::filters::DEFAULT_ACCURACY_BOUND;
        for c in [MATMUL_F32_CLAIM] {
            assert!(c.max_relative <= DEFAULT_ACCURACY_BOUND.max_relative);
            assert!(c.max_absolute <= DEFAULT_ACCURACY_BOUND.max_absolute);
        }
    }
}
