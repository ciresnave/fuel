//! CapturedRun 4b-resume Step 3 — empirical seeding of the CUDA verification
//! ledger.
//!
//! The CUDA analogue of Task 4.5b's [`super::seed_cpu_ledger`]. Every CUDA
//! kernel contract in `docs/kernel-contracts/cuda/*.fkc.md` declares
//! `bit_stable_on_same_hardware: true`, but the V-FKC-9 import gate
//! ([`super::gate_precision`], wired at `register.rs:363`) downgrades any such
//! claim lacking a matching `pass` entry in the git-checked-in
//! `docs/kernel-contracts/.fkc-verified-ledger.json` to
//! `PrecisionGuarantee::UNAUDITED`. An unaudited CUDA candidate then loses its
//! placement to the (Task-4.5b-seeded, audited) CPU alternative under
//! [`crate::ranker::filters::BitStablePreferenceFilter`], so the decode runs on
//! CPU and never enters a CapturedRun capture scope. This harness earns the
//! CUDA `bit_stable_on_same_hardware` ledger entries so the decode-path CUDA
//! kernels are audited and win their placements.
//!
//! Mechanism (mirrors [`super::harness`], generalized from the single
//! rope-apply contract to the WHOLE production CUDA binding table): register
//! every CUDA kernel via [`crate::baracuda_dispatch::register_baracuda_cuda_kernels`],
//! iterate `table.iter_entries()`, synthesize a per-`OpKind` probe, drive
//! [`super::verify_bit_stability`] through a real [`super::CudaInvoker`] for
//! `ITERS` repeat calls, and `upsert` a `pass`/`fail` record keyed on the
//! entry's `kernel_revision_hash`. `upsert` (not `push`) keeps re-seeding
//! idempotent — the `include_str!`-embedded ledger recompiles after the first
//! write, so a naive `push` re-run would append duplicates.
//!
//! Never fabricates a pass: an op with no probe recipe, a kernel `Err`, or a
//! panic (caught via `catch_unwind`) contributes NO ledger entry and is logged
//! as skipped/failed. `#[cfg(feature = "cuda")]` throughout — needs a live
//! `CudaDevice`; its seeding test is `#[ignore]`'d.

use std::panic::{catch_unwind, AssertUnwindSafe};

use fuel_cuda_backend::CudaDevice;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::DType;

use super::{
    fill_deterministic, verify_bit_stability, CpuInvoker, CudaInvoker, HostTensor, KernelInvoker,
    LedgerRecord, ProbeInputs, VerificationLedger, VerifyError, VerifyOutcome,
};
use crate::kernel::{KernelBindingTable, MatmulM, OpParams};

/// CUDA ops whose contract declares a numeric `max_ulp` bound (only
/// `indexing.fkc.md` today — exact copies, bound 0). The V-FKC-9 gate collapses
/// the WHOLE guarantee if ANY declared machine-checkable claim is unbacked, so
/// these need a `max_ulp` ledger entry IN ADDITION to `bit_stable`. Verified
/// CUDA-candidate-vs-CPU-reference (0 ULP = byte-identical for an exact copy).
const MAX_ULP_OPS: &[(OpKind, u32)] = &[
    (OpKind::IndexSelect, 0),
    (OpKind::Gather, 0),
    (OpKind::MaskedFill, 0),
];

/// `true` iff every `f32` element of `cand` is within `bound` ULP of `refr`
/// (mirrors `verify_precision_bound`'s `MaxUlp` arm). `Err` on a length/align
/// mismatch rather than a panic.
fn max_ulp_ok(cand: &[u8], refr: &[u8], bound: u32) -> std::result::Result<bool, String> {
    if cand.len() % 4 != 0 || refr.len() != cand.len() {
        return Err(format!("output length/align mismatch (cand {} ref {})", cand.len(), refr.len()));
    }
    let a: &[f32] = bytemuck::cast_slice(cand);
    let b: &[f32] = bytemuck::cast_slice(refr);
    for (x, y) in a.iter().zip(b.iter()) {
        // Shared total-order ULP distance — correct across the sign/zero
        // boundary; never drifts from `verify_precision_bound`'s `MaxUlp` arm.
        if super::ulp::ulp_distance(*x, *y) > bound as u64 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Repeat-call count per probe for `bit_stable_on_same_hardware` (≥16 floor,
/// same as the CPU seeder + the rope-apply harness).
const ITERS: usize = 16;

/// The verified claim this harness earns. CUDA contracts also carry
/// `max_ulp`/`max_relative`/`max_absolute` slots, but the decode-path
/// contracts leave them `~` (null → not gated); only `indexing` sets
/// `max_ulp: 0` (handled as a follow-on, needs a CPU-reference ULP check).
const CLAIM: &str = "bit_stable_on_same_hardware";

/// A synthesized, safe, valid probe for one `(OpKind, dtypes)` CUDA
/// registration: real inputs the kernel runs on without crashing, the
/// `OpParams` it needs, and the output dtype/shape `CudaInvoker` allocates.
struct Probe {
    inputs: ProbeInputs,
    params: OpParams,
    out_dtype: DType,
    out_shape: Vec<usize>,
}

/// Encode `vals` into `dt`'s byte representation (float dtypes only — the
/// decode path is F32; other float widths fan the same recipes). `None` for a
/// dtype this harness can't encode (integer index dtypes, quantized, etc.),
/// so the caller skips rather than guesses.
fn to_bytes(dt: DType, vals: &[f32]) -> Option<Vec<u8>> {
    Some(match dt {
        DType::F32 => bytemuck::cast_slice(vals).to_vec(),
        DType::F64 => bytemuck::cast_slice(&vals.iter().map(|&x| x as f64).collect::<Vec<_>>()).to_vec(),
        DType::BF16 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| half::bf16::from_f32(x)).collect::<Vec<_>>()).to_vec()
        }
        DType::F16 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>()).to_vec()
        }
        _ => return None,
    })
}

fn ht(dt: DType, shape: Vec<usize>, vals: &[f32]) -> Option<HostTensor> {
    Some(HostTensor { dtype: dt, shape, bytes: to_bytes(dt, vals)? })
}

/// Build a real, valid probe for a CUDA primitive `op` at the registered
/// `dtypes`. `None` ⇒ this harness has no recipe for that op yet (logged +
/// skipped, never a fabricated entry). The recipe (inputs + `OpParams`) is
/// backend-agnostic — the same shapes the CPU wrappers accept — because the
/// CUDA wrappers pull identical `OpParams` (verified by reading
/// `baracuda_dispatch.rs`'s wrapper macros).
fn build_cuda_probe(op: OpKind, dtypes: &[DType], seed: u64) -> Option<Probe> {
    let dt = *dtypes.first()?;
    // Only recipes over encodable float dtypes; integer/quant fan-outs skip.
    to_bytes(dt, &[0.0])?;

    match op {
        // --- Binary elementwise (2 inputs, params ignored) -----------------
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise
        | OpKind::DivElementwise
        | OpKind::MaximumElementwise
        | OpKind::MinimumElementwise => {
            let a = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let b = ht(dt, vec![4], &fill_deterministic(4, seed ^ 0x9E37_79B9))?;
            Some(Probe { inputs: vec![a, b], params: OpParams::None, out_dtype: dt, out_shape: vec![4] })
        }

        // --- Unary elementwise (1 input, params ignored) -------------------
        OpKind::NegElementwise
        | OpKind::ReluElementwise
        | OpKind::SqrElementwise
        | OpKind::SqrtElementwise
        | OpKind::RecipElementwise
        | OpKind::RsqrtElementwise
        | OpKind::AbsElementwise
        | OpKind::TanhElementwise
        | OpKind::ExpElementwise
        | OpKind::LogElementwise
        | OpKind::SinElementwise
        | OpKind::CosElementwise
        | OpKind::SigmoidElementwise
        | OpKind::SiluElementwise
        | OpKind::GeluElementwise
        | OpKind::GeluErfElementwise
        | OpKind::ErfElementwise
        | OpKind::StepElementwise
        | OpKind::SignElementwise
        | OpKind::FloorElementwise
        | OpKind::CeilElementwise
        | OpKind::RoundElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::None, out_dtype: dt, out_shape: vec![4] })
        }

        // --- MatMul (2 inputs) ---------------------------------------------
        OpKind::MatMul => {
            let (m, n, k) = (2usize, 2usize, 2usize);
            let lhs = ht(dt, vec![m * k], &fill_deterministic(m * k, seed))?;
            let rhs = ht(dt, vec![k * n], &fill_deterministic(k * n, seed ^ 0x1234))?;
            Some(Probe {
                inputs: vec![lhs, rhs],
                params: OpParams::Matmul {
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                    m,
                    n,
                    k,
                    m_compute: MatmulM::All,
                },
                out_dtype: dt,
                out_shape: vec![m * n],
            })
        }

        // --- RmsNorm / LayerNorm last-dim (1 input) ------------------------
        OpKind::RmsNormLastDim | OpKind::LayerNormLastDim => {
            let (outer, last) = (2usize, 4usize);
            let x = ht(dt, vec![outer * last], &fill_deterministic(outer * last, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::NormLastDim { outer_count: outer, last_dim: last, eps: 1e-5 },
                out_dtype: dt,
                out_shape: vec![outer * last],
            })
        }

        // --- Softmax / LogSoftmax last-dim (1 input) -----------------------
        OpKind::SoftmaxLastDim | OpKind::LogSoftmaxLastDim => {
            let (outer, last) = (2usize, 4usize);
            let x = ht(dt, vec![outer * last], &fill_deterministic(outer * last, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::SoftmaxLastDim { outer_count: outer, last_dim: last },
                out_dtype: dt,
                out_shape: vec![outer * last],
            })
        }

        // --- Affine y = mul*x + add (1 input) ------------------------------
        OpKind::Affine => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Affine { mul: 2.0, add: 1.0 },
                out_dtype: dt,
                out_shape: vec![4],
            })
        }

        // --- IndexSelect (embed): src[outer, source_dim, inner] + U32 idx --
        OpKind::IndexSelect => {
            let (outer, source_dim, n_idx, inner) = (1usize, 4usize, 2usize, 3usize);
            let src = ht(dt, vec![outer * source_dim * inner], &fill_deterministic(outer * source_dim * inner, seed))?;
            // Indices are ALWAYS U32 (the CUDA wrapper hard-requires it); 0,1
            // are in-bounds for source_dim=4.
            let indices = HostTensor {
                dtype: DType::U32,
                shape: vec![n_idx],
                bytes: bytemuck::cast_slice(&[0u32, 1u32]).to_vec(),
            };
            Some(Probe {
                inputs: vec![src, indices],
                params: OpParams::IndexSelect {
                    outer_count: outer,
                    source_dim_size: source_dim,
                    n_indices: n_idx,
                    inner_count: inner,
                },
                out_dtype: dt,
                out_shape: vec![outer * n_idx * inner],
            })
        }

        // --- Concat along a single axis (2 inputs) — rope decompose --------
        OpKind::Concat => {
            let a = ht(dt, vec![2], &fill_deterministic(2, seed))?;
            let b = ht(dt, vec![2], &fill_deterministic(2, seed ^ 0x5555))?;
            Some(Probe {
                inputs: vec![a, b],
                params: OpParams::Concat {
                    outer_count: 1,
                    input_dim_sizes: vec![2, 2],
                    inner_count: 1,
                    axis: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
            })
        }

        _ => None,
    }
}

/// One attempt outcome, kept even for skips/failures so the report shows
/// exactly what did and didn't verify — never silently.
#[derive(Debug)]
pub struct CudaSeedAttempt {
    pub op: String,
    pub dtypes: Vec<DType>,
    pub kernel_revision_hash: u64,
    pub outcome: String,
}

/// `epoch:<unix seconds>` — dependency-free timestamp (house convention).
fn verified_at_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("epoch:{secs}")
}

/// Empirically verify every CUDA primitive registration this harness has a
/// probe recipe for, and return a ledger seeded from the EMBEDDED (checked-in)
/// records plus any freshly-earned CUDA `pass` entries (never discards prior
/// entries — the CPU 4.5b records + the rope-apply record survive), together
/// with a full attempt log (every skip/failure) for the report.
///
/// `only` restricts to a subset of `OpKind`s (e.g. the decode-path set); pass
/// `None` to sweep every registered CUDA kernel. Requires a live CUDA device.
pub fn run_cuda_verification(
    only: Option<&[OpKind]>,
    force: bool,
) -> std::result::Result<(VerificationLedger, Vec<CudaSeedAttempt>), VerifyError> {
    let mut table = KernelBindingTable::new();
    crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);

    let mut ledger =
        VerificationLedger::from_records(VerificationLedger::embedded().records().to_vec());
    let device = CudaDevice::new(0)
        .map_err(|e| VerifyError::Backend(format!("no CUDA device: {e}")))?;
    let mut log = Vec::new();

    for (op, dtypes, backend, entry) in table.iter_entries() {
        if backend != BackendId::Cuda {
            continue;
        }
        if let Some(set) = only {
            if !set.contains(&op) {
                continue;
            }
        }
        let dtypes_vec = dtypes.to_vec();
        let rev = entry.kernel_revision_hash;

        if !force && ledger.has_pass(BackendId::Cuda, dtypes, rev, CLAIM) {
            log.push(CudaSeedAttempt {
                op: format!("{op:?}"),
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "skip: already has a pass for this revision".to_string(),
            });
            continue;
        }

        // Deterministic per-(op,dtype) seed so a re-run is byte-identical.
        let seed = 0x2545_F491_4F6C_DD1D_u64
            ^ (op as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (dtypes.len() as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let probe = match build_cuda_probe(op, dtypes, seed) {
            Some(p) => p,
            None => {
                log.push(CudaSeedAttempt {
                    op: format!("{op:?}"),
                    dtypes: dtypes_vec,
                    kernel_revision_hash: rev,
                    outcome: "skip: no probe recipe for this op/dtype".to_string(),
                });
                continue;
            }
        };

        let inv = CudaInvoker::new(device.clone(), probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let inputs = probe.inputs.clone();
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            verify_bit_stability(&inv, entry, std::slice::from_ref(&inputs), ITERS)
        }));

        let (result, outcome, evidence) = match attempt {
            Ok(Ok(VerifyOutcome::Pass)) => (
                Some("pass"),
                "pass".to_string(),
                serde_json::json!({ "repeat_calls": ITERS, "harness": "capturedrun-4b/seed_cuda_ledger" }),
            ),
            Ok(Ok(VerifyOutcome::Fail { detail })) => {
                (Some("fail"), format!("fail: {detail}"), serde_json::json!({ "detail": detail }))
            }
            Ok(Ok(VerifyOutcome::NoReference)) => {
                (None, "skip: no reference".to_string(), serde_json::Value::Null)
            }
            Ok(Err(e)) => (
                Some("fail"),
                format!("fail: invoke error {e:?}"),
                serde_json::json!({ "invoke_error": format!("{e:?}") }),
            ),
            Err(_) => (
                Some("fail"),
                "fail: kernel invocation panicked".to_string(),
                serde_json::json!({ "panic": true }),
            ),
        };

        if let Some(result) = result {
            ledger.upsert(LedgerRecord {
                kernel_ref: entry.kernel_source.to_string(),
                backend: "Cuda".to_string(),
                dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                kernel_revision_hash: rev,
                claim: CLAIM.to_string(),
                result: result.to_string(),
                verified_at: verified_at_string(),
                protocol_version: 1,
                evidence,
            });
        }
        log.push(CudaSeedAttempt {
            op: format!("{op:?}"),
            dtypes: dtypes_vec,
            kernel_revision_hash: rev,
            outcome,
        });
    }

    // ---- Second pass: max_ulp claims (CUDA candidate vs CPU reference) ----
    // The gate needs these for the few ops declaring a numeric bound (only the
    // exact-copy indexing family). Reference is the registered CPU kernel for
    // the SAME (op, dtypes); 0 ULP means byte-identical.
    let mut cpu_table = KernelBindingTable::new();
    crate::dispatch::register_cpu_kernels(&mut cpu_table);
    for (op, dtypes, backend, cuda_entry) in table.iter_entries() {
        if backend != BackendId::Cuda {
            continue;
        }
        if let Some(set) = only {
            if !set.contains(&op) {
                continue;
            }
        }
        let Some(&(_, bound)) = MAX_ULP_OPS.iter().find(|(o, _)| *o == op) else {
            continue;
        };
        let dtypes_vec = dtypes.to_vec();
        let rev = cuda_entry.kernel_revision_hash;
        if !force && ledger.has_pass(BackendId::Cuda, dtypes, rev, "max_ulp") {
            continue;
        }
        let seed = 0x2545_F491_4F6C_DD1D_u64
            ^ (op as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (dtypes.len() as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let Some(probe) = build_cuda_probe(op, dtypes, seed) else {
            log.push(CudaSeedAttempt {
                op: format!("{op:?}"),
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "max_ulp skip: no probe recipe".to_string(),
            });
            continue;
        };
        let cpu_entry = cpu_table
            .iter_entries()
            .find(|(o, d, b, _)| *o == op && *d == dtypes && *b == BackendId::Cpu)
            .map(|(_, _, _, e)| e);
        let Some(cpu_entry) = cpu_entry else {
            log.push(CudaSeedAttempt {
                op: format!("{op:?}"),
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "max_ulp skip: no CPU reference entry".to_string(),
            });
            continue;
        };

        let cand = CudaInvoker::new(device.clone(), probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let refr = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let inputs = probe.inputs.clone();
        let attempt = catch_unwind(AssertUnwindSafe(|| -> std::result::Result<String, VerifyError> {
            let a = cand.invoke(cuda_entry, &inputs)?;
            let b = refr.invoke(cpu_entry, &inputs)?;
            Ok(match max_ulp_ok(&a.bytes, &b.bytes, bound) {
                Ok(true) => "pass".to_string(),
                Ok(false) => format!("fail: exceeds max_ulp {bound}"),
                Err(e) => format!("fail: {e}"),
            })
        }));

        let (result, outcome) = match attempt {
            Ok(Ok(s)) if s == "pass" => (Some("pass"), "max_ulp pass".to_string()),
            Ok(Ok(s)) => (Some("fail"), format!("max_ulp {s}")),
            Ok(Err(e)) => (Some("fail"), format!("max_ulp invoke error {e:?}")),
            Err(_) => (Some("fail"), "max_ulp panicked".to_string()),
        };
        if let Some(result) = result {
            ledger.upsert(LedgerRecord {
                kernel_ref: cuda_entry.kernel_source.to_string(),
                backend: "Cuda".to_string(),
                dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                kernel_revision_hash: rev,
                claim: "max_ulp".to_string(),
                result: result.to_string(),
                verified_at: verified_at_string(),
                protocol_version: 1,
                evidence: serde_json::json!({ "bound": bound, "reference": "cpu", "harness": "capturedrun-4b/seed_cuda_ledger" }),
            });
        }
        log.push(CudaSeedAttempt {
            op: format!("{op:?}"),
            dtypes: dtypes_vec,
            kernel_revision_hash: rev,
            outcome,
        });
    }

    Ok((ledger, log))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// CapturedRun 4b-resume Step 3 — sweep the production CUDA binding table,
    /// empirically verify `bit_stable_on_same_hardware` for every op this
    /// harness has a recipe for, and WRITE the merged ledger back to the
    /// git-checked-in `docs/kernel-contracts/.fkc-verified-ledger.json`.
    ///
    /// Requires a live CUDA device (`#[ignore]`'d). Run:
    ///   `cargo test -p fuel-dispatch --features cuda --lib \
    ///    seed_cuda_verified_ledger -- --ignored --nocapture`
    #[test]
    #[ignore = "re-seeding tool: writes the verified ledger; needs a live CUDA device + --features cuda"]
    fn seed_cuda_verified_ledger() {
        let (ledger, log) = run_cuda_verification(None, true).expect("cuda seeding runs");
        for a in &log {
            println!("[step3] {} {:?} (rev={}): {}", a.op, a.dtypes, a.kernel_revision_hash, a.outcome);
        }
        let passed = log.iter().filter(|a| a.outcome == "pass").count();
        let failed = log.iter().filter(|a| a.outcome.starts_with("fail")).count();
        let skipped = log.iter().filter(|a| a.outcome.starts_with("skip")).count();
        println!("[step3] {passed} passed, {failed} failed, {skipped} skipped, {} attempts", log.len());
        assert!(passed > 0, "expected at least one CUDA kernel to verify bit-stable; got 0 — see log");

        let cuda_passes = ledger
            .records()
            .iter()
            .filter(|r| r.backend == "Cuda" && r.claim == CLAIM && r.result == "pass")
            .count();
        println!("[step3] ledger now holds {cuda_passes} CUDA bit-stable pass records");

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/kernel-contracts/.fkc-verified-ledger.json");
        let json = serde_json::to_string_pretty(ledger.records()).expect("serialize ledger");
        let mut f = std::fs::File::create(&path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
        f.write_all(json.as_bytes()).expect("write ledger");
        f.write_all(b"\n").expect("write newline");
        println!("[step3] wrote {} records to {}", ledger.records().len(), path.display());
    }
}
