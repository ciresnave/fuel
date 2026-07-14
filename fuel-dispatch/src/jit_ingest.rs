//! Candidate-kernel ingestion (Spec B) — the seam types this module starts
//! from. Task 2 defined the types a provider's rejection/adoption feedback
//! flows through: [`RejectionReport`], [`ProviderFeedback`], and
//! [`IngestOutcome`]. Task 3 (this slice) adds [`CandidateKernel`] — the
//! not-yet-verified offer a provider hands Fuel, bundling the kernel fn
//! pointer with the operand/dtype shape facts and the precision claims it
//! *declares* (unverified until [`crate::fkc::verify`] empirically checks
//! them — see `jit_ingest_probe`'s `probe_from_operands` for the probe-input
//! synthesis step that feeds that verification). No consumer yet
//! (Task 5/6 wire ingest + verify around it) — `dead_code` is expected.

use crate::fkc::verify::LedgerRecord;

#[cfg(feature = "cuda")]
use crate::fkc::verify::{
    verify_bit_stability, verify_precision_bound, Bound, CudaInvoker, HostTensor, KernelInvoker,
    ProbeInputs, VerificationLedger, VerifyError, VerifyOutcome,
};
#[cfg(feature = "cuda")]
use crate::kernel::BindingEntry;
#[cfg(feature = "cuda")]
use crate::runtime_fused_kernels::adopt_runtime_fused;
#[cfg(feature = "cuda")]
use fuel_cuda_backend::CudaDevice;

/// A not-yet-verified kernel a provider has offered Fuel: the callable
/// itself, the region it claims to replace (`decompose`, `None` if the
/// provider synthesized from scratch rather than fusing an existing
/// subgraph), the exact operand/dtype shapes it was built for, and the
/// precision guarantees it *declares* (unverified — [`crate::fkc::verify`]
/// is what turns a declared claim into an empirically-checked one before
/// `adopt_runtime_fused` ever sees it).
pub struct CandidateKernel {
    pub entry_point: String,
    pub kernel: crate::kernel::KernelRef,
    pub op_params: crate::kernel::OpParams,
    pub decompose: Option<fuel_graph::jit::PatternNode>,
    pub operands: Vec<baracuda_kernels_types::OperandDesc>,
    pub dtypes: Vec<fuel_ir::DType>,
    pub kernel_revision_hash: u64,
    pub declared: crate::fused::PrecisionGuarantee,
    pub backend: fuel_ir::probe::BackendId,
}

/// Why Fuel refused a candidate kernel a provider offered — handed to
/// [`ProviderFeedback::on_rejected`] so the provider can stop re-offering or
/// log the reason. `ledger_record` carries the empirical verification result
/// (if the rejection came from a failed [`crate::fkc::verify`] claim) rather
/// than a synthetic value.
pub struct RejectionReport {
    pub entry_point: String,
    pub failed_claim: &'static str,
    pub detail: String,
    pub ledger_record: Option<LedgerRecord>,
}

/// The callback surface a candidate-kernel provider implements to learn the
/// outcome of an offer. `on_rejected` is required (the whole point of the
/// report); `on_adopted` is optional telemetry, default no-op.
pub trait ProviderFeedback: Send + Sync {
    fn on_rejected(&self, report: &RejectionReport);
    fn on_adopted(&self, _entry_point: &str, _id: fuel_graph::registry::FusedOpId) {}
}

/// The result of ingesting one candidate kernel: adopted (with the
/// `FusedOpId` it registered under) or rejected (with the report explaining
/// why).
pub enum IngestOutcome {
    Adopted(fuel_graph::registry::FusedOpId),
    Rejected(RejectionReport),
}

// ---------------------------------------------------------------------------
// Task 5 — candidate-kernel verification (`verify_candidate`).
// ---------------------------------------------------------------------------

/// The verdict of empirically verifying a [`CandidateKernel`] against the
/// reference realized from its `decompose`: either every declared,
/// machine-checkable precision claim held ([`VerifyVerdict::Pass`]) or the
/// FIRST one that didn't ([`VerifyVerdict::Fail`], naming the claim + why).
///
/// Returned alongside the earned [`LedgerRecord`]s (one per checked claim). The
/// records are built in a FRESH, candidate-local [`VerificationLedger`] via
/// `upsert` — the git-checked-in embedded ledger is never mutated here; Task
/// 6's `ingest_one` is what merges an adopted candidate's records into the real
/// ledger.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum VerifyVerdict {
    /// Every DECLARED claim was empirically backed.
    Pass,
    /// A claim failed (or its reference couldn't be produced). `claim` is the
    /// stage/claim id: `"probe"` / `"invoke"` / `"bit_stable_on_same_hardware"`
    /// / `"max_ulp"` / `"max_relative"` / `"max_absolute"` / `"panic"`.
    Fail { claim: &'static str, detail: String },
}

/// A [`KernelInvoker`] that returns a pre-computed [`HostTensor`] regardless of
/// the entry/inputs it's handed. Lets [`verify_candidate`] reuse
/// [`verify_precision_bound`] (written against two invokers) to check numeric
/// bounds on the ALREADY-computed candidate and reference outputs — without
/// re-invoking the kernel or re-realizing the reference (wasteful, and for a
/// non-bit-stable kernel could drift between calls).
#[cfg(feature = "cuda")]
struct FixedOutput(HostTensor);

#[cfg(feature = "cuda")]
impl KernelInvoker for FixedOutput {
    fn invoke(
        &self,
        _entry: &BindingEntry,
        _inputs: &[HostTensor],
    ) -> std::result::Result<HostTensor, VerifyError> {
        Ok(self.0.clone())
    }
}

/// `epoch:<unix seconds>` — dependency-free timestamp (house convention,
/// mirrors `seed_cuda_ledger::verified_at_string`).
#[cfg(feature = "cuda")]
fn verified_at_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("epoch:{secs}")
}

/// The first declared numeric claim (checked order: ulp → relative → absolute).
/// Only called when at least one is declared, so it always returns a real name.
#[cfg(feature = "cuda")]
fn first_numeric_claim(g: &crate::fused::PrecisionGuarantee) -> &'static str {
    if g.max_ulp.is_some() {
        "max_ulp"
    } else if g.max_relative.is_some() {
        "max_relative"
    } else {
        "max_absolute"
    }
}

/// Check one numeric [`Bound`] on the already-computed candidate vs reference
/// outputs by reusing [`verify_precision_bound`] through [`FixedOutput`]
/// adapters (no re-invoke). `Ok(())` = within bound; `Err(detail)` = out of
/// bound (or a length/align issue `verify_precision_bound` reports).
#[cfg(feature = "cuda")]
fn check_numeric_bound(
    cand_out: &HostTensor,
    reference: &HostTensor,
    entry: &BindingEntry,
    probe: &ProbeInputs,
    bound: Bound,
) -> std::result::Result<(), String> {
    let cand_fx = FixedOutput(cand_out.clone());
    let ref_fx = FixedOutput(reference.clone());
    match verify_precision_bound(&cand_fx, &ref_fx, entry, std::slice::from_ref(probe), bound) {
        Ok(VerifyOutcome::Pass) => Ok(()),
        Ok(VerifyOutcome::Fail { detail }) => Err(detail),
        Ok(VerifyOutcome::NoReference) => {
            Err("verify_precision_bound returned NoReference".to_string())
        }
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Empirically verify a received [`CandidateKernel`] on a synthetic probe:
/// compare it to the reference realized from its `decompose` and check every
/// DECLARED, machine-checkable precision claim (bit-stability + the numeric
/// bounds). Returns the [`VerifyVerdict`] plus the earned [`LedgerRecord`]s
/// (one per checked claim), in a fresh candidate-local ledger — the embedded
/// ledger is never touched.
///
/// Never panics: the whole body runs inside `catch_unwind`, so a candidate
/// kernel that panics (or a reference realize that does) becomes a
/// `Fail { claim: "panic", .. }`, never a process crash (Fuel never-panic
/// production invariant).
#[cfg(feature = "cuda")]
pub fn verify_candidate(
    cand: &CandidateKernel,
    device: &CudaDevice,
) -> (VerifyVerdict, Vec<LedgerRecord>) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_candidate_impl(cand, device)
    })) {
        Ok(result) => result,
        Err(_) => (
            VerifyVerdict::Fail {
                claim: "panic",
                detail: "candidate verification panicked (kernel invoke or reference realize)"
                    .to_string(),
            },
            Vec::new(),
        ),
    }
}

#[cfg(feature = "cuda")]
fn verify_candidate_impl(
    cand: &CandidateKernel,
    device: &CudaDevice,
) -> (VerifyVerdict, Vec<LedgerRecord>) {
    // A fresh, candidate-local ledger — NEVER the git-checked-in embedded one.
    // `upsert` (not `push`): each claim is checked at most once here, but the
    // discipline is uniform (re-checking a key replaces, never dups).
    let mut ledger = VerificationLedger::default();

    let make_record = |claim: &str, result: &str, evidence: serde_json::Value| -> LedgerRecord {
        LedgerRecord {
            kernel_ref: cand.entry_point.clone(),
            backend: "Cuda".to_string(),
            dtypes: cand.dtypes.iter().map(|d| format!("{d:?}")).collect(),
            kernel_revision_hash: cand.kernel_revision_hash,
            claim: claim.to_string(),
            result: result.to_string(),
            verified_at: verified_at_now(),
            protocol_version: 1,
            evidence,
        }
    };

    // (1) Probe synthesis. A candidate carrying an operand we can't faithfully
    // encode (e.g. a non-float dtype `to_bytes` rejects) yields NO probe — a
    // "probe" fail, never a fabricated input. Seed derived from the revision
    // hash so a re-run is byte-identical.
    let seed = 0x5EED_C0DE_1234_5678_u64 ^ cand.kernel_revision_hash;
    let probe = match crate::jit_ingest_probe::probe_from_operands(&cand.operands, seed) {
        Some(p) if !p.is_empty() => p,
        Some(_) => {
            return (
                VerifyVerdict::Fail {
                    claim: "probe",
                    detail: "candidate declares no operands to probe".to_string(),
                },
                Vec::new(),
            )
        }
        None => {
            return (
                VerifyVerdict::Fail {
                    claim: "probe",
                    detail:
                        "candidate carries an operand whose dtype cannot be encoded as a probe input"
                            .to_string(),
                },
                Vec::new(),
            )
        }
    };

    // Output dtype/shape: derived from the first probe operand. This matches
    // the elementwise candidates that are Task 6's only live consumer (Add's
    // output shape/dtype == its operands'); a candidate whose output geometry
    // differs from operand[0] is out of scope for this slice.
    let out_dtype = probe[0].dtype;
    let out_shape = probe[0].shape.clone();

    // Numeric claims (checked ahead of the invoke below, for the f32-only
    // guard right after) — the same declared-claim test the numeric block
    // uses further down.
    let numeric_declared = cand.declared.max_ulp.is_some()
        || cand.declared.max_relative.is_some()
        || cand.declared.max_absolute.is_some();

    // (1a) Non-f32 numeric-claim guard. `verify_precision_bound` (`ulp.rs`)
    // unconditionally reinterprets BOTH outputs' raw bytes as `f32`
    // (`bytemuck::cast_slice`) — for a non-F32 `out_dtype` (BF16/F16/F64/...)
    // that reinterpretation reads the wrong element count/values from bytes
    // that were never f32 in the first place. A kernel computing the WRONG
    // function could then land within `max_ulp` of the reinterpreted
    // reference purely by accident and wrongly PASS — exactly the "wrong
    // candidate adopted" defect this whole module exists to prevent. Refuse
    // honestly instead: no numeric claim can be verified for a non-f32
    // candidate yet. Placed here (before the GPU invoke below) so a
    // candidate that can't be numerically verified doesn't waste GPU work
    // being invoked at all — this short-circuits bit-stability too, matching
    // the existing first-failure-wins posture (e.g. the probe checks above).
    if numeric_declared && out_dtype != fuel_ir::DType::F32 {
        let claim = first_numeric_claim(&cand.declared);
        let detail =
            "numeric bound verification is f32-only; non-f32 candidate cannot be numerically verified yet"
                .to_string();
        ledger.upsert(make_record(
            claim,
            "fail",
            serde_json::json!({ "detail": detail.clone(), "out_dtype": format!("{out_dtype:?}") }),
        ));
        return (VerifyVerdict::Fail { claim, detail }, ledger.records().to_vec());
    }

    // (2) Candidate output via a real CUDA invoke. The `BindingEntry` mirrors
    // `invoker_cuda.rs`'s wiring, carrying the candidate's DECLARED precision
    // (unverified until this fn checks it) + revision hash.
    let entry = BindingEntry {
        kernel: cand.kernel,
        caps: crate::kernel::KernelCaps::empty(),
        precision: cand.declared,
        cost: crate::kernel::unknown_cost,
        kernel_source: "candidate",
        is_generic: false,
        kernel_revision_hash: cand.kernel_revision_hash,
        cost_expr: None,
    };
    let inv = CudaInvoker::new(device.clone(), out_dtype, out_shape.clone())
        .with_params(cand.op_params.clone());
    let cand_out = match inv.invoke(&entry, &probe) {
        Ok(o) => o,
        Err(e) => {
            return (
                VerifyVerdict::Fail {
                    claim: "invoke",
                    detail: format!("candidate kernel invoke failed: {e:?}"),
                },
                Vec::new(),
            )
        }
    };

    // (3) Bit-stability — only when DECLARED. A candidate that makes no
    // bit-stability claim isn't held to it, and no ledger entry is earned for
    // an unclaimed property (matching `gate_precision`'s declared-only gate).
    if cand.declared.bit_stable_on_same_hardware {
        match verify_bit_stability(&inv, &entry, std::slice::from_ref(&probe), 16) {
            Ok(VerifyOutcome::Pass) => ledger.upsert(make_record(
                "bit_stable_on_same_hardware",
                "pass",
                serde_json::json!({ "repeat_calls": 16 }),
            )),
            Ok(VerifyOutcome::Fail { detail }) => {
                ledger.upsert(make_record(
                    "bit_stable_on_same_hardware",
                    "fail",
                    serde_json::json!({ "detail": detail.clone() }),
                ));
                return (
                    VerifyVerdict::Fail { claim: "bit_stable_on_same_hardware", detail },
                    ledger.records().to_vec(),
                );
            }
            // `verify_bit_stability` never returns NoReference; treat defensively.
            Ok(VerifyOutcome::NoReference) => {}
            Err(e) => {
                let detail = format!("bit-stability invoke failed: {e:?}");
                ledger.upsert(make_record(
                    "bit_stable_on_same_hardware",
                    "fail",
                    serde_json::json!({ "detail": detail.clone() }),
                ));
                return (
                    VerifyVerdict::Fail { claim: "bit_stable_on_same_hardware", detail },
                    ledger.records().to_vec(),
                );
            }
        }
    }

    // (4)+(5) Numeric claims. These need a REFERENCE. Resolution (Task-5;
    // deviates from the plan's literal "CPU-reference path"):
    //   - `decompose.is_some()` → realize it on the same probe (`reference_output`).
    //   - `decompose.is_none()` → there is NO reference. We do NOT attempt a
    //     CPU-op lookup: `CandidateKernel` carries no `OpKind`, so the plan's
    //     "look up the CPU kernel for the same op" is infeasible here. Any
    //     declared numeric claim then fails honestly (bit-stability above stays
    //     checkable). This branch is defensive — Task 6 requires a decompose to
    //     adopt at all, so it has no live consumer.
    if numeric_declared {
        let reference = match &cand.decompose {
            Some(dec) => match crate::jit_ingest_probe::reference_output(
                dec,
                &probe,
                out_dtype,
                out_shape.clone(),
                device,
            ) {
                Ok(r) => r,
                Err(e) => {
                    let claim = first_numeric_claim(&cand.declared);
                    let detail = format!("reference realize from decompose failed: {e:?}");
                    ledger.upsert(make_record(
                        claim,
                        "fail",
                        serde_json::json!({ "detail": detail.clone() }),
                    ));
                    return (VerifyVerdict::Fail { claim, detail }, ledger.records().to_vec());
                }
            },
            None => {
                let claim = first_numeric_claim(&cand.declared);
                let detail =
                    "no decompose: cannot verify numeric claim against a reference".to_string();
                ledger.upsert(make_record(
                    claim,
                    "fail",
                    serde_json::json!({ "detail": detail.clone() }),
                ));
                return (VerifyVerdict::Fail { claim, detail }, ledger.records().to_vec());
            }
        };

        // Check each declared numeric bound in order; FIRST failure returns.
        if let Some(b) = cand.declared.max_ulp {
            match check_numeric_bound(&cand_out, &reference, &entry, &probe, Bound::MaxUlp(b)) {
                Ok(()) => {
                    ledger.upsert(make_record("max_ulp", "pass", serde_json::json!({ "bound": b })))
                }
                Err(detail) => {
                    ledger.upsert(make_record(
                        "max_ulp",
                        "fail",
                        serde_json::json!({ "detail": detail.clone(), "bound": b }),
                    ));
                    return (VerifyVerdict::Fail { claim: "max_ulp", detail }, ledger.records().to_vec());
                }
            }
        }
        if let Some(b) = cand.declared.max_relative {
            match check_numeric_bound(&cand_out, &reference, &entry, &probe, Bound::MaxRelative(b)) {
                Ok(()) => ledger.upsert(make_record(
                    "max_relative",
                    "pass",
                    serde_json::json!({ "bound": b }),
                )),
                Err(detail) => {
                    ledger.upsert(make_record(
                        "max_relative",
                        "fail",
                        serde_json::json!({ "detail": detail.clone(), "bound": b }),
                    ));
                    return (
                        VerifyVerdict::Fail { claim: "max_relative", detail },
                        ledger.records().to_vec(),
                    );
                }
            }
        }
        if let Some(b) = cand.declared.max_absolute {
            match check_numeric_bound(&cand_out, &reference, &entry, &probe, Bound::MaxAbsolute(b)) {
                Ok(()) => ledger.upsert(make_record(
                    "max_absolute",
                    "pass",
                    serde_json::json!({ "bound": b }),
                )),
                Err(detail) => {
                    ledger.upsert(make_record(
                        "max_absolute",
                        "fail",
                        serde_json::json!({ "detail": detail.clone(), "bound": b }),
                    ));
                    return (
                        VerifyVerdict::Fail { claim: "max_absolute", detail },
                        ledger.records().to_vec(),
                    );
                }
            }
        }
    }

    (VerifyVerdict::Pass, ledger.records().to_vec())
}

// ---------------------------------------------------------------------------
// Task 6 — sync verify → adopt / reject-with-feedback (`ingest_one`).
// ---------------------------------------------------------------------------

/// Ingest one [`CandidateKernel`]: verify it ([`verify_candidate`]) against
/// the reference realized from its `decompose`, then either adopt it as a
/// runtime-fused kernel ([`adopt_runtime_fused`]) or build a
/// [`RejectionReport`] a provider's [`ProviderFeedback::on_rejected`] can act
/// on.
///
/// Never panics on the production path: `verify_candidate` is already
/// `catch_unwind`-internal, and [`adopt_verified`] wraps its own registration
/// call in `catch_unwind` too, so a candidate that panics invoking its
/// kernel, realizing its reference, or registering with `fuel-graph`'s
/// runtime-fused registry becomes a `Rejected(..)`, never a process crash —
/// this deliberately deviates from the plan's literal
/// `.expect("fused candidate has a decompose")`, which is a production panic
/// the constitution's never-panic invariant forbids.
#[cfg(feature = "cuda")]
pub fn ingest_one(cand: &CandidateKernel, device: &CudaDevice) -> IngestOutcome {
    let (verdict, records) = verify_candidate(cand, device);
    match verdict {
        VerifyVerdict::Pass => adopt_verified(cand),
        VerifyVerdict::Fail { claim, detail } => {
            // The ledger record for the FAILED claim, if verify_candidate
            // earned one for it — it does for every fail path except the
            // very earliest (probe synthesis / invoke / top-level panic),
            // which return before any record is upserted.
            let ledger_record = records.into_iter().find(|r| r.claim == claim);
            IngestOutcome::Rejected(RejectionReport {
                entry_point: cand.entry_point.clone(),
                failed_claim: claim,
                detail,
                ledger_record,
            })
        }
    }
}

/// The `Pass`-verdict half of [`ingest_one`]: adopt the candidate as a
/// runtime-fused kernel bound for `cand.backend`. Never panics: a candidate
/// that verified `Pass` but carries no `decompose` (nothing to register a
/// region from) is `Rejected` rather than unwrapped, and the
/// `adopt_runtime_fused` call itself runs inside `catch_unwind` — a `None`
/// (region not registrable, e.g. a non-decomposable/shape-changing pattern)
/// or a caught panic both become `Rejected`, never a crash.
#[cfg(feature = "cuda")]
fn adopt_verified(cand: &CandidateKernel) -> IngestOutcome {
    let Some(region) = cand.decompose.clone() else {
        return IngestOutcome::Rejected(RejectionReport {
            entry_point: cand.entry_point.clone(),
            failed_claim: "no_decompose",
            detail: "Pass verdict but candidate has no decompose region to adopt".to_string(),
            ledger_record: None,
        });
    };
    let entry_point = cand.entry_point.clone();
    let kernel = cand.kernel;
    let dtypes = cand.dtypes.clone();
    let backend = cand.backend;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adopt_runtime_fused(entry_point, region, kernel, dtypes, backend)
    })) {
        Ok(Some(id)) => IngestOutcome::Adopted(id),
        Ok(None) => IngestOutcome::Rejected(RejectionReport {
            entry_point: cand.entry_point.clone(),
            failed_claim: "adopt_failed",
            detail: "adopt_runtime_fused returned None (region not registrable)".to_string(),
            ledger_record: None,
        }),
        Err(_) => IngestOutcome::Rejected(RejectionReport {
            entry_point: cand.entry_point.clone(),
            failed_claim: "adopt_failed",
            detail: "adopt_runtime_fused panicked during registration".to_string(),
            ledger_record: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_feedback_receives_the_report() {
        use std::sync::Mutex;
        struct Rec(Mutex<Vec<String>>);
        impl ProviderFeedback for Rec {
            fn on_rejected(&self, r: &RejectionReport) {
                self.0.lock().unwrap().push(r.failed_claim.into());
            }
        }
        let rec = Rec(Mutex::new(vec![]));
        rec.on_rejected(&RejectionReport {
            entry_point: "k".into(),
            failed_claim: "max_ulp",
            detail: "d".into(),
            ledger_record: None,
        });
        assert_eq!(rec.0.lock().unwrap().as_slice(), &["max_ulp".to_string()]);
    }

    /// Spec-B Task-5 acceptance (live GPU): a candidate whose kernel is the
    /// CUDA `add_f32` wrapper, carrying a 2-input `Add` decompose and matching
    /// F32 operands, verifies `Pass` — its output is byte-identical to the
    /// reference realized from the decompose (elementwise sum), so it's
    /// bit-stable AND meets every `PrecisionGuarantee::REFERENCE` numeric bound
    /// (0 ULP / 0 relative / 0 absolute). Earns one `pass` `LedgerRecord` per
    /// declared claim (bit_stable + the 3 numeric bounds = 4). `#[ignore]`'d
    /// (needs a live CUDA device). Candidate/probe construction mirrors
    /// `invoker_cuda.rs`'s `cuda_invoker_runs_add_elementwise_f32_end_to_end`.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn verify_candidate_add_f32_passes_against_its_decompose() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "add_f32".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x00AD_DF32,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        };

        let (verdict, records) = verify_candidate(&cand, &dev);
        assert!(matches!(verdict, VerifyVerdict::Pass), "expected Pass, got {verdict:?}");
        // REFERENCE declares 4 machine-checkable claims → 4 pass records.
        assert_eq!(records.len(), 4, "one pass record per declared claim: {records:?}");
        assert!(records.iter().all(|r| r.result == "pass"), "all pass: {records:?}");
        assert!(records.iter().all(|r| r.backend == "Cuda"));
        assert!(records.iter().all(|r| r.kernel_revision_hash == 0x00AD_DF32));
        assert!(records.iter().any(|r| r.claim == "bit_stable_on_same_hardware"));
        assert!(records.iter().any(|r| r.claim == "max_ulp"));
        assert!(records.iter().any(|r| r.claim == "max_relative"));
        assert!(records.iter().any(|r| r.claim == "max_absolute"));
    }

    /// Review-fix regression: `verify_precision_bound` (`ulp.rs`)
    /// unconditionally reinterprets output bytes as `f32`
    /// (`bytemuck::cast_slice::<f32>`), so a candidate whose real output
    /// dtype is NOT f32 (BF16 here) must never reach that comparison — it
    /// would silently reinterpret BF16 bytes as f32 and could wrongly PASS a
    /// wrong-function kernel. A BF16 candidate declaring a numeric claim
    /// (`max_ulp`) must get an honest `Fail` naming the f32-only limitation,
    /// not a GPU invoke + mis-compare. `#[ignore]`'d (needs a live CUDA
    /// device) — the guard fires before any GPU work, so the `kernel` fn
    /// pointer is never actually called; reusing `add_f32` here is just a
    /// valid `KernelRef` value, not a claim it computes anything meaningful
    /// for BF16.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn verify_candidate_refuses_numeric_claim_for_non_f32() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::Bf16, 16);
        let cand = CandidateKernel {
            entry_point: "add_bf16".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::BF16, DType::BF16],
            kernel_revision_hash: 0x00AD_DF16,
            declared: crate::fused::PrecisionGuarantee {
                bit_stable_on_same_hardware: false,
                max_ulp: Some(1),
                max_relative: None,
                max_absolute: None,
                notes: "test-only, f32-only guard regression",
            },
            backend: BackendId::Cuda,
        };

        let (verdict, records) = verify_candidate(&cand, &dev);
        match verdict {
            VerifyVerdict::Fail { claim, detail } => {
                assert_eq!(claim, "max_ulp", "first (only) declared numeric claim");
                assert!(
                    detail.contains("f32-only"),
                    "expected an f32-only refusal detail, got: {detail}"
                );
            }
            other => panic!("expected Fail (f32-only guard), got {other:?}"),
        }
        assert_eq!(records.len(), 1, "one fail record for the refused claim: {records:?}");
        assert_eq!(records[0].claim, "max_ulp");
        assert_eq!(records[0].result, "fail");
    }

    /// Spec-B Task-6 acceptance (live GPU), REJECTION leg.
    ///
    /// The plan's original Step-1 test (interleaved `rope_apply_f32` candidate
    /// vs a rotate-half `decompose`) is infeasible here: rotate-half rope
    /// isn't expressible as a `PatternNode` (elementwise-only — `emit`/
    /// `register_runtime_fused` reject non-representable/shape-changing ops),
    /// and `rope_apply` is a reverted registration, not a wired `KernelRef`
    /// (see `~/.claude/projects/.../rope-convention-mismatch-baracuda-fuel.md`).
    /// This test exercises the SAME essential property — `ingest_one` must
    /// reject a candidate that computes a DIFFERENT function than the region
    /// it claims to replace — with an elementwise substitute: `kernel` is
    /// CUDA `mul_f32` (elementwise product) offered for a 2-input `Add`
    /// region. The candidate declares `PrecisionGuarantee::REFERENCE` (a
    /// NUMERIC claim, not just bit-stability) so `verify_candidate` actually
    /// compares candidate-vs-reference output — mul is just as deterministic
    /// as add, so a bit-stable-only claim would wrongly pass without ever
    /// comparing values. `1+2=3 != 1*2=2` on the synthetic probe, so the
    /// numeric bound fails and `ingest_one` must return `Rejected`, never
    /// `Adopted`.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn ingest_rejects_mul_candidate_for_the_add_region() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "test::ingest::mul_for_add_region".to_string(),
            kernel: crate::baracuda_dispatch::binary::mul_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_0001,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        };

        match ingest_one(&cand, &dev) {
            IngestOutcome::Rejected(r) => {
                assert!(
                    r.failed_claim.contains("max") || r.failed_claim == "vs_decompose",
                    "expected a precision claim naming the mismatch, got: {:?}",
                    r.failed_claim
                );
            }
            IngestOutcome::Adopted(_) => {
                panic!("mul_f32 must NOT be adopted for an Add region — it computes a different function")
            }
        }
    }

    /// Spec-B Task-6 acceptance (live GPU), ADOPTION leg — the counterpart to
    /// [`ingest_rejects_mul_candidate_for_the_add_region`]: a candidate whose
    /// kernel genuinely matches its `decompose` (CUDA `add_f32` for a 2-input
    /// `Add` region) verifies `Pass` and is `Adopted`. After adoption, the
    /// adopted id's kernel must be visible to the capability gate on Cuda via
    /// [`crate::runtime_fused_kernels::fused_kernel_available_in`] — the same
    /// predicate `offer_runtime_fused_arm` gates on before offering the fused
    /// arm. `entry_point` carries a distinctive per-test-run suffix (unlike
    /// the shared `add_f32`/`mul_f32` entry points above) because
    /// `adopt_runtime_fused` registers into the PROCESS-GLOBAL runtime-fused
    /// registry + binding table, which other `#[ignore]` tests in this binary
    /// share — a colliding name across runs/tests would risk resolving a
    /// stale/different registration rather than this call's own.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn ingest_adopts_add_candidate_for_the_add_region() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "test::ingest::add_for_add_region::run_1nge_5702".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_0002,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        };

        match ingest_one(&cand, &dev) {
            IngestOutcome::Adopted(id) => {
                let table = crate::dispatch::global_bindings();
                assert!(
                    crate::runtime_fused_kernels::fused_kernel_available_in(
                        &table,
                        id,
                        BackendId::Cuda
                    ),
                    "the adopted candidate's kernel must be visible to the capability gate on Cuda",
                );
            }
            IngestOutcome::Rejected(r) => panic!(
                "add_f32 for an Add region must be Adopted, got Rejected: {} / {}",
                r.failed_claim, r.detail
            ),
        }
    }
}
