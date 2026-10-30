// SPDX-License-Identifier: MIT OR Apache-2.0
//! Earns the CUDA f32 `MatMul`, `RmsNormLastDim` and `SoftmaxLastDim` accuracy claims (`max_relative`, `max_absolute`) in the
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

use super::accuracy_attention::{bf16s, f16s, measure_flash_decoding};
use super::accuracy_common::{
    Claim, OpAccuracy, Runner, f32s, find_entry, measure_matmul, measure_rms_norm, measure_softmax,
};
use super::accuracy_elementwise::{BINARY_SPECS, UNARY_SPECS, measure_binary, measure_unary};
use super::accuracy_half::{measure_half_matmul, measure_half_rms_norm, measure_half_softmax};
use super::accuracy_qmatmul::measure_qmatmul;
use super::accuracy_reduce::{Fold, measure_reduce};
use super::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use super::invoker_cuda::CudaInvoker;
use super::ledger::LedgerRecord;
use crate::kernel::{BindingEntry, KernelBindingTable, OpParams};

/// The `backend` field of the records this module writes.
const BACKEND: &str = "Cuda";
/// The `harness` field of their evidence.
const HARNESS: &str = "row-39/seed_cuda_accuracy";

/// The ops this module claims for; their sections are split by dtype in the contracts, which
/// re-keys their entries (the dtype list is part of the revision hash).
fn measured_ops() -> Vec<OpKind> {
    [
        OpKind::MatMul,
        OpKind::RmsNormLastDim,
        OpKind::SoftmaxLastDim,
    ]
    .into_iter()
    .chain(UNARY_CLAIMS.iter().map(|c| c.0))
    .chain(BINARY_CLAIMS.iter().map(|c| c.0))
    .chain(REDUCE_CLAIMS.iter().map(|c| c.0))
    .collect()
}

const fn both(bound: f64) -> Claim {
    Claim {
        max_relative: bound,
        max_absolute: bound,
    }
}

/// `(op, f32 section, claim)` of each unary op, one entry per `UNARY_SPECS` spec. The claim is
/// the same number in `kernel-contracts/cuda/unary.fkc.md` (`claim_constants_match_the_contracts`).
pub(crate) const UNARY_CLAIMS: &[(OpKind, &str, Claim)] = &[
    (OpKind::NegElementwise, "neg_f32", both(0.0)),
    (OpKind::AbsElementwise, "abs_f32", both(0.0)),
    (OpKind::SqrElementwise, "sqr_f32", both(1.0e-5)),
    (OpKind::SqrtElementwise, "sqrt_f32", both(1.0e-5)),
    (OpKind::RecipElementwise, "recip_f32", both(1.0e-5)),
    // The absolute ceiling is looser: results reach 148 on the measured domain, where one f32 ulp is 1.5e-5.
    (
        OpKind::ExpElementwise,
        "exp_f32",
        Claim {
            max_relative: 1.0e-5,
            max_absolute: 1.0e-4,
        },
    ),
    (OpKind::LogElementwise, "log_f32", both(1.0e-5)),
    (OpKind::SinElementwise, "sin_f32", both(1.0e-5)),
    (OpKind::CosElementwise, "cos_f32", both(1.0e-5)),
    (OpKind::TanhElementwise, "tanh_f32", both(1.0e-5)),
    (OpKind::ReluElementwise, "relu_f32", both(0.0)),
    (OpKind::GeluTanhElementwise, "gelu_tanh_f32", both(1.0e-5)),
    (OpKind::GeluElementwise, "gelu_f32", both(1.0e-5)),
    (OpKind::StepElementwise, "step_f32", both(0.0)),
    (OpKind::SiluElementwise, "silu_f32", both(1.0e-5)),
    (OpKind::SigmoidElementwise, "sigmoid_f32", both(1.0e-5)),
    (OpKind::RsqrtElementwise, "rsqrt_f32", both(1.0e-5)),
    (OpKind::FloorElementwise, "floor_f32", both(0.0)),
    (OpKind::CeilElementwise, "ceil_f32", both(0.0)),
    (OpKind::RoundElementwise, "round_f32", both(0.0)),
    (OpKind::ErfElementwise, "erf_f32", both(1.0e-5)),
    (OpKind::SignElementwise, "sign", both(0.0)),
];

/// `(op, f32 section, claim)` of each binary op, one entry per `BINARY_SPECS` spec; the
/// contract is `kernel-contracts/cuda/binary.fkc.md`.
pub(crate) const BINARY_CLAIMS: &[(OpKind, &str, Claim)] = &[
    (OpKind::AddElementwise, "add_f32", both(1.0e-5)),
    (OpKind::SubElementwise, "sub_f32", both(1.0e-5)),
    (OpKind::MulElementwise, "mul_f32", both(1.0e-5)),
    (OpKind::DivElementwise, "div_f32", both(1.0e-5)),
    (OpKind::MaximumElementwise, "maximum_f32", both(0.0)),
    (OpKind::MinimumElementwise, "minimum_f32", both(0.0)),
    (OpKind::PowElementwise, "pow_f32", both(1.0e-5)),
    (OpKind::RemElementwise, "rem_f32", both(1.0e-5)),
];

/// `(op, f32 section, claim)` of each axis reduction; the contract is
/// `kernel-contracts/cuda/reduce.fkc.md`.
pub(crate) const REDUCE_CLAIMS: &[(OpKind, &str, Claim)] = &[
    (OpKind::SumReduce, "sum_f32", both(1.0e-4)),
    (OpKind::MaxReduce, "max_f32", both(0.0)),
    (OpKind::MinReduce, "min_f32", both(0.0)),
    (OpKind::MeanReduce, "mean_f32", both(1.0e-5)),
];

fn claim_of(table: &[(OpKind, &str, Claim)], op: OpKind) -> Claim {
    table
        .iter()
        .find(|c| c.0 == op)
        .unwrap_or_else(|| panic!("{op:?} has a spec but no claim"))
        .2
}

/// `matmul_f32` (`kernel-contracts/cuda/gemm_dense.fkc.md`), unit-scale inputs.
pub(crate) const MATMUL_F32_CLAIM: Claim = Claim {
    max_relative: 2.0e-4,
    max_absolute: 2.0e-4,
};
/// `rms_f32` (`kernel-contracts/cuda/norm.fkc.md`). Measured worst 2.4e-7 abs / 1.6e-7 rel.
pub(crate) const RMS_NORM_F32_CLAIM: Claim = Claim {
    max_relative: 1.0e-5,
    max_absolute: 1.0e-5,
};
/// `softmax_f32` (`kernel-contracts/cuda/softmax.fkc.md`). Measured worst 7.9e-8 abs and rel.
pub(crate) const SOFTMAX_F32_CLAIM: Claim = Claim {
    max_relative: 1.0e-5,
    max_absolute: 1.0e-5,
};
/// `qmatmul_f32` (`kernel-contracts/cuda/qmatmul.fkc.md`), unit-scale uniform inputs, Q4_0 / Q8_0 /
/// Q4_K, `k` up to 8192, both routes of the wrapper. Measured worst 3.8e-5 abs / 3.2e-5 rel (at
/// k = 8192, growing with k).
pub(crate) const QMATMUL_F32_CLAIM: Claim = Claim {
    max_relative: 4.0e-4,
    max_absolute: 4.0e-4,
};
/// ONE CUDA device for the whole process. baracuda's cuBLAS handle pool is keyed by the context
/// ADDRESS (fixed in baracuda alpha.16), so dropping a context and creating the next can revive a
/// handle bound to the dead one: `status 5` from `gemm_dense_{f16,bf16}` (GAP-370). Every
/// measurement shares this device's context.
fn shared_device() -> Result<CudaDevice, VerifyError> {
    static DEVICE: std::sync::OnceLock<Result<CudaDevice, String>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| CudaDevice::new(0).map_err(|e| format!("no CUDA device: {e}")))
        .clone()
        .map_err(VerifyError::Backend)
}

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

/// Measure the registered CUDA f32 `MatMul`, `RmsNormLastDim` and `SoftmaxLastDim` on
/// device 0. Returns the device name and the measurements.
pub(crate) fn measure_cuda_f32_accuracy() -> Result<(String, Vec<OpAccuracy>), VerifyError> {
    let mut table = KernelBindingTable::new();
    crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
    let device = shared_device()?;
    let run = |entry: &BindingEntry, params: OpParams, inputs: &[HostTensor], shape: Vec<usize>| {
        let inv = CudaInvoker::new(device.clone(), DType::F32, shape).with_params(params);
        Ok(f32s(&inv.invoke(entry, inputs)?))
    };
    let entry = |op: OpKind, dt: &[DType]| find_entry(&table, op, dt, BackendId::Cuda);
    let mut ops = vec![
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
    ];
    ops.push(measure_qmatmul(
        &run,
        entry(OpKind::QMatMul, &[DType::F32, DType::U32, DType::F32])?,
        QMATMUL_F32_CLAIM,
    )?);
    for spec in UNARY_SPECS {
        let e = entry(spec.op, &[DType::F32; 2])?;
        ops.push(measure_unary(
            &run,
            e,
            spec,
            claim_of(UNARY_CLAIMS, spec.op),
        )?);
    }
    for spec in BINARY_SPECS {
        let e = entry(spec.op, &[DType::F32; 3])?;
        ops.push(measure_binary(
            &run,
            e,
            spec,
            claim_of(BINARY_CLAIMS, spec.op),
        )?);
    }
    for fold in [Fold::Sum, Fold::Max, Fold::Min, Fold::Mean] {
        let e = entry(fold.op(), &[DType::F32; 2])?;
        ops.push(measure_reduce(
            &run,
            e,
            fold,
            claim_of(REDUCE_CLAIMS, fold.op()),
        )?);
    }
    Ok((device_name(), ops))
}

/// The registered f16 or bf16 CUDA `FlashAttn` decode kernel against the default placement
/// bound. Evidence for `flash_decoding` only: no claim is earned, so `seed-cuda-accuracy` does
/// not write it (`check-cuda-accuracy` prints and asserts it).
fn measure_cuda_half_attention(dtype: DType) -> Result<OpAccuracy, VerifyError> {
    use crate::ranker::filters::DEFAULT_ACCURACY_BOUND;
    let mut table = KernelBindingTable::new();
    crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
    let device = shared_device()?;
    let run = |entry: &BindingEntry, params: OpParams, inputs: &[HostTensor], shape: Vec<usize>| {
        let inv = CudaInvoker::new(device.clone(), dtype, shape).with_params(params);
        let out = inv.invoke(entry, inputs)?;
        Ok(if dtype == DType::BF16 {
            bf16s(&out)
        } else {
            f16s(&out)
        })
    };
    let entry = find_entry(&table, OpKind::FlashAttn, &[dtype; 4], BackendId::Cuda)?;
    let bound = Claim {
        max_relative: DEFAULT_ACCURACY_BOUND.max_relative,
        max_absolute: DEFAULT_ACCURACY_BOUND.max_absolute,
    };
    measure_flash_decoding(&run, entry, dtype, bound)
}

/// Decode a half-precision kernel output (F16 or BF16) to `f32`.
fn decode_half(dtype: DType, t: &HostTensor) -> Vec<f32> {
    if dtype == DType::BF16 {
        bf16s(t)
    } else {
        f16s(t)
    }
}

/// The three decode-path ops of `dtype` through `run`, against `bound`.
fn measure_half_ops(
    run: &Runner,
    table: &KernelBindingTable,
    dtype: DType,
    bound: Claim,
) -> Result<Vec<OpAccuracy>, VerifyError> {
    let entry =
        |op: OpKind, arity: usize| find_entry(table, op, &vec![dtype; arity], BackendId::Cuda);
    Ok(vec![
        measure_half_matmul(run, entry(OpKind::MatMul, 3)?, dtype, bound)?,
        measure_half_rms_norm(run, entry(OpKind::RmsNormLastDim, 2)?, dtype, bound)?,
        measure_half_softmax(run, entry(OpKind::SoftmaxLastDim, 2)?, dtype, bound)?,
    ])
}

/// The registered f16 or bf16 CUDA `MatMul`, `RmsNormLastDim` and `SoftmaxLastDim` against the
/// default placement bound. Evidence only, like [`measure_cuda_half_attention`].
fn measure_cuda_half_decode_ops(dtype: DType) -> Result<Vec<OpAccuracy>, VerifyError> {
    use crate::ranker::filters::DEFAULT_ACCURACY_BOUND;
    let mut table = KernelBindingTable::new();
    crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
    let device = shared_device()?;
    let run = |entry: &BindingEntry, params: OpParams, inputs: &[HostTensor], shape: Vec<usize>| {
        let inv = CudaInvoker::new(device.clone(), dtype, shape).with_params(params);
        Ok(decode_half(dtype, &inv.invoke(entry, inputs)?))
    };
    let bound = Claim {
        max_relative: DEFAULT_ACCURACY_BOUND.max_relative,
        max_absolute: DEFAULT_ACCURACY_BOUND.max_absolute,
    };
    measure_half_ops(&run, &table, dtype, bound)
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
pub(crate) fn run_seed_cuda_accuracy() {
    // The dtype list is part of a section's revision hash, so splitting a section re-keyed every
    // entry of its op: earn `bit_stable_on_same_hardware` for the entries that lack
    // a pass at their current hash (`force = false` leaves every other kernel's record alone).
    let (ledger, log) =
        super::seed_cuda_ledger::run_cuda_verification(Some(&measured_ops()), false)
            .expect("cuda bit-stability seeding runs");
    for a in &log {
        println!(
            "[row-39] {} {:?} (rev={}): {}",
            a.op, a.dtypes, a.kernel_revision_hash, a.outcome
        );
    }
    assert!(
        log.iter().all(|a| !a.outcome.starts_with("fail")),
        "a kernel failed bit-stability; see the log above"
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
    let attention: Vec<OpAccuracy> = [DType::F16, DType::BF16]
        .into_iter()
        .map(|d| measure_cuda_half_attention(d).expect("cuda half attention measurement runs"))
        .collect();
    print_run(&name, &attention);
    // Printed, NOT asserted: no half-precision claim is earned (bf16 rounding alone exceeds the
    // 1e-3 bound for MatMul and RmsNormLastDim; the f16/bf16 tiled gemm is far past its rounding
    // floor at k >= 4096, GAP-369). The numbers are the evidence a later claim starts from.
    for d in [DType::F16, DType::BF16] {
        println!("[row-39] dtype {d:?} decode-path ops");
        // A half gemm launch failure (GAP-370; the shared device avoids the known cause) is
        // reported, and the run goes on, since this evidence is not asserted.
        match measure_cuda_half_decode_ops(d) {
            Ok(ops) => print_run(&name, &ops),
            Err(e) => println!("[row-39] {d:?} decode-path ops NOT MEASURED (GAP-370): {e:?}"),
        }
    }
    let failing = records(&name, &ops)
        .into_iter()
        .chain(records(&name, &attention))
        .filter(|r| r.result != "pass")
        .count();
    assert!(failing == 0, "{failing} measurement(s) exceed their claim");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(contract text, kernel name, claim, op, dtypes of the F32 entry)` for each claimed op.
    fn cases() -> Vec<(&'static str, &'static str, Claim, OpKind, usize)> {
        let mut v = vec![
            (
                include_str!("../../../kernel-contracts/cuda/gemm_dense.fkc.md"),
                "matmul_f32",
                MATMUL_F32_CLAIM,
                OpKind::MatMul,
                3,
            ),
            (
                include_str!("../../../kernel-contracts/cuda/norm.fkc.md"),
                "rms_f32",
                RMS_NORM_F32_CLAIM,
                OpKind::RmsNormLastDim,
                2,
            ),
            (
                include_str!("../../../kernel-contracts/cuda/softmax.fkc.md"),
                "softmax_f32",
                SOFTMAX_F32_CLAIM,
                OpKind::SoftmaxLastDim,
                2,
            ),
        ];
        let unary = include_str!("../../../kernel-contracts/cuda/unary.fkc.md");
        let binary = include_str!("../../../kernel-contracts/cuda/binary.fkc.md");
        v.extend(UNARY_CLAIMS.iter().map(|&(op, k, c)| (unary, k, c, op, 2)));
        v.extend(
            BINARY_CLAIMS
                .iter()
                .map(|&(op, k, c)| (binary, k, c, op, 3)),
        );
        let reduce = include_str!("../../../kernel-contracts/cuda/reduce.fkc.md");
        v.extend(
            REDUCE_CLAIMS
                .iter()
                .map(|&(op, k, c)| (reduce, k, c, op, 2)),
        );
        v
    }

    /// Every claimed op has a measurement recipe and every recipe a claim: a spec added without
    /// a claim would be measured against a panic, and a claim without a spec would never be
    /// earned.
    #[test]
    fn specs_and_claims_cover_the_same_ops() {
        let spec_ops: Vec<OpKind> = UNARY_SPECS.iter().map(|s| s.op).collect();
        let claim_ops: Vec<OpKind> = UNARY_CLAIMS.iter().map(|c| c.0).collect();
        assert_eq!(spec_ops, claim_ops, "unary");
        let spec_ops: Vec<OpKind> = BINARY_SPECS.iter().map(|s| s.op).collect();
        let claim_ops: Vec<OpKind> = BINARY_CLAIMS.iter().map(|c| c.0).collect();
        assert_eq!(spec_ops, claim_ops, "binary");
    }

    /// The claim constants and their contract sections say the same numbers: a claim edited in
    /// one place only would be earned against a bound the contract does not declare.
    #[test]
    fn claim_constants_match_the_contracts() {
        for (text, kernel, claim, _, _) in cases() {
            let text = text.replace("\r\n", "\n");
            let marker = format!("kernel: {kernel}\n");
            let at = text
                .find(&marker)
                .unwrap_or_else(|| panic!("section `{kernel}` present"));
            let rest = &text[at..];
            let prec = &rest[rest.find("precision:").expect("precision block")..];
            let prec = &prec[..prec.find("audited:").expect("precision block ends")];
            assert!(
                prec.contains(&format!("max_relative: {}", claim.max_relative)),
                "{kernel}: contract max_relative differs from the claim constant"
            );
            assert!(
                prec.contains(&format!("max_absolute: {}", claim.max_absolute)),
                "{kernel}: contract max_absolute differs from the claim constant"
            );
        }
    }

    /// `qmatmul_f32` is not bit-stable, so it is outside `cases()` (which asserts that). Its claim
    /// constant must equal the contract's numbers, and once the ledger backs them the registered
    /// entry must carry them and meet the planner's default accuracy bound; otherwise the importer
    /// downgrades it to UNAUDITED and the planner prefers the CPU kernel (ROADMAP row 128).
    #[test]
    fn the_qmatmul_claim_matches_the_contract_and_reaches_the_entry() {
        let text =
            include_str!("../../../kernel-contracts/cuda/qmatmul.fkc.md").replace("\r\n", "\n");
        let at = text.find("kernel: qmatmul_f32\n").expect("section present");
        let rest = &text[at..];
        let prec = &rest[rest.find("precision:").expect("precision block")..];
        let prec = &prec[..prec.find("audited:").expect("precision block ends")];
        assert!(prec.contains(&format!("max_relative: {}", QMATMUL_F32_CLAIM.max_relative)));
        assert!(prec.contains(&format!("max_absolute: {}", QMATMUL_F32_CLAIM.max_absolute)));
        let mut table = KernelBindingTable::new();
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
        let e = find_entry(
            &table,
            OpKind::QMatMul,
            &[DType::F32, DType::U32, DType::F32],
            BackendId::Cuda,
        )
        .expect("QMatMul entry");
        assert_eq!(
            e.precision.max_relative,
            Some(QMATMUL_F32_CLAIM.max_relative)
        );
        assert_eq!(
            e.precision.max_absolute,
            Some(QMATMUL_F32_CLAIM.max_absolute)
        );
        assert!(crate::ranker::filters::meets_accuracy_bound(
            &e.precision,
            &crate::ranker::filters::DEFAULT_ACCURACY_BOUND
        ));
    }

    /// What the import gate leaves on the registered entries once the ledger backs the claims:
    /// each F32 entry carries its numeric claim AND keeps `bit_stable`, and the F16 entry of the
    /// same op (a different section, no numeric claim) is not collapsed to UNAUDITED by it.
    #[test]
    fn the_ledger_backs_the_f32_claims_and_leaves_the_other_dtypes_audited() {
        let mut table = KernelBindingTable::new();
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
        for (_, kernel, claim, op, arity) in cases() {
            let f32e = find_entry(&table, op, &vec![DType::F32; arity], BackendId::Cuda)
                .unwrap_or_else(|e| panic!("{kernel}: {e:?}"));
            assert!(f32e.precision.bit_stable_on_same_hardware, "{kernel}");
            assert_eq!(
                f32e.precision.max_relative,
                Some(claim.max_relative),
                "{kernel}"
            );
            assert_eq!(
                f32e.precision.max_absolute,
                Some(claim.max_absolute),
                "{kernel}"
            );
            if kernel == "sign" {
                continue; // F32-only op: there is no other-dtype section to keep audited
            }
            let f16e = find_entry(&table, op, &vec![DType::F16; arity], BackendId::Cuda)
                .unwrap_or_else(|e| panic!("{kernel} f16: {e:?}"));
            assert!(f16e.precision.bit_stable_on_same_hardware, "{kernel} f16");
            assert_eq!(f16e.precision.max_relative, None, "{kernel} f16");
        }
    }

    /// Every claim is within the default placement bound, or earning it would not let the kernel
    /// win by default.
    #[test]
    fn every_claim_is_within_the_default_bound() {
        use crate::ranker::filters::DEFAULT_ACCURACY_BOUND;
        for (_, _, c, _, _) in cases() {
            assert!(c.max_relative <= DEFAULT_ACCURACY_BOUND.max_relative);
            assert!(c.max_absolute <= DEFAULT_ACCURACY_BOUND.max_absolute);
        }
    }
}
