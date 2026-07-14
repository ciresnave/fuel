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

use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::Duration;

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

// ---------------------------------------------------------------------------
// Task 7 — IngestionService: bounded queue + idle-aware concurrency-1 worker.
// ---------------------------------------------------------------------------

/// Tunables for [`IngestionService`]. `Default` matches the plan's defaults.
#[derive(Debug, Clone, Copy)]
pub struct IngestionConfig {
    /// Bounded `sync_channel` capacity. Once full, [`IngestionService::enqueue`]
    /// returns [`Backpressure`] instead of growing unbounded — candidate
    /// ingestion must never be able to out-race verification and pile up
    /// memory behind it. Default 32.
    pub queue_bound: usize,
    /// How many verifies may run concurrently. **Only `1` (the default) is
    /// implemented**: [`IngestionService::start_with_verify`] always spawns
    /// exactly one worker thread, so a candidate is always verified strictly
    /// serially with respect to any other candidate. A value other than `1`
    /// is currently advisory — building a real bounded worker pool is
    /// deferred until something actually needs concurrent verification
    /// (YAGNI; see the Task-7 brief). Default 1.
    pub max_concurrent: usize,
    /// The worker's idle-gate threshold: before starting the next verify,
    /// the worker waits (best-effort, bounded short sleeps — never
    /// unbounded) while `inflight_count` for the CUDA device is `>=` this,
    /// so candidate verification defers to live inference load rather than
    /// competing with it. Default 1 (wait for the device to be fully idle).
    pub idle_load_threshold: u32,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self { queue_bound: 32, max_concurrent: 1, idle_load_threshold: 1 }
    }
}

/// The bounded queue was full (or the worker is gone) when
/// [`IngestionService::enqueue`] was called — the candidate was NOT
/// accepted. If a `feedback` was supplied to `enqueue`, it already received
/// a synchronous `on_rejected` (`failed_claim == "queue_full"`) on the
/// caller's own thread before `enqueue` returned this.
#[derive(Debug)]
pub struct Backpressure;

/// The [`RejectionReport`] `enqueue` hands a provider's `feedback` when the
/// bounded queue is full.
fn queue_full_report(cand: &CandidateKernel) -> RejectionReport {
    RejectionReport {
        entry_point: cand.entry_point.clone(),
        failed_claim: "queue_full",
        detail: format!(
            "ingestion queue is at capacity; candidate '{}' dropped under backpressure",
            cand.entry_point
        ),
        ledger_record: None,
    }
}

/// One queued offer: the candidate plus the (optional) feedback sink its
/// eventual outcome should be reported to.
type IngestItem = (CandidateKernel, Option<Arc<dyn ProviderFeedback>>);

/// Bounded candidate-kernel ingestion queue + a single idle-aware background
/// verify worker (Spec B Task 7). Candidates offered via [`Self::enqueue`]
/// are handed to ONE worker thread that verifies them one at a time —
/// against a live CUDA device in production ([`Self::start`]), or an
/// injected closure in tests ([`Self::start_with_verify`], the test/
/// production seam) — deferring to `inflight_count`-observed live GPU load
/// between items, so a burst of candidate offers never competes with live
/// inference for the device. A full queue backpressures the caller
/// ([`Backpressure`]) instead of growing unbounded.
pub struct IngestionService {
    sender: Option<SyncSender<IngestItem>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl IngestionService {
    /// Production entry point: verify each candidate with [`ingest_one`]
    /// against `device`. `device` is captured by the verify closure handed
    /// to [`Self::start_with_verify`] — this is a thin wrapper, all the
    /// queue/worker/idle-gate/never-panic machinery lives there and is
    /// exercised directly (no GPU) by this module's tests.
    #[cfg(feature = "cuda")]
    pub fn start(device: CudaDevice, cfg: IngestionConfig) -> Self {
        Self::start_with_verify(move |cand| ingest_one(cand, &device), cfg)
    }

    /// Test/production seam: build the service around an injected verify
    /// step rather than requiring a live CUDA device, so the queue,
    /// backpressure, idle-gate, and panic-survival behavior are all
    /// testable with NO GPU. Spawns the single worker thread that
    /// `max_concurrent == 1` (the only implemented value — see
    /// [`IngestionConfig::max_concurrent`]) requires.
    pub fn start_with_verify<F>(verify: F, cfg: IngestionConfig) -> Self
    where
        F: Fn(&CandidateKernel) -> IngestOutcome + Send + 'static,
    {
        let (sender, receiver) = sync_channel::<IngestItem>(cfg.queue_bound);
        // Clamp to >= 1: `inflight_count(..) >= 0` is always true for a `u32`
        // count, so an unclamped `idle_load_threshold: 0` would have the
        // worker's idle-gate spin-sleep forever and never reach `verify` —
        // 0 is treated as 1 (wait for the device to be fully idle).
        let idle_load_threshold = cfg.idle_load_threshold.max(1);
        let worker =
            std::thread::spawn(move || worker_loop(receiver, verify, idle_load_threshold));
        Self { sender: Some(sender), worker: Some(worker) }
    }

    /// Offer a candidate for background verification. `Ok(())` means it was
    /// accepted into the bounded queue — NOT that it verified or adopted;
    /// that happens asynchronously on the worker thread and is reported via
    /// `feedback`. `Err(Backpressure)` means the queue was full (or the
    /// worker is gone); when `feedback` was supplied it already received a
    /// synchronous `on_rejected("queue_full")` before this call returns.
    ///
    /// Never panics: a full or disconnected channel degrades to
    /// `Backpressure`, never an unwrap. Note the asymmetry with the worker
    /// loop: the synchronous `queue_full` `on_rejected` call below runs
    /// UNGUARDED on the caller's own thread (unlike the worker's
    /// `catch_unwind`-guarded callbacks) — intentionally, so a panicking
    /// `feedback` implementation surfaces to the caller (their bug, their
    /// thread) rather than being silently swallowed.
    pub fn enqueue(
        &self,
        cand: CandidateKernel,
        feedback: Option<Arc<dyn ProviderFeedback>>,
    ) -> Result<(), Backpressure> {
        let Some(sender) = &self.sender else {
            return Err(Backpressure);
        };
        match sender.try_send((cand, feedback)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full((cand, feedback))) => {
                if let Some(fb) = &feedback {
                    fb.on_rejected(&queue_full_report(&cand));
                }
                Err(Backpressure)
            }
            Err(TrySendError::Disconnected(_)) => Err(Backpressure),
        }
    }

    /// Stop accepting new candidates and wait for the worker to drain
    /// whatever is already queued/in-flight, then exit. Dropping the sender
    /// (rather than sending a sentinel) is what makes the worker's blocking
    /// `recv()` return `Err` once the queue empties — that's its exit
    /// signal, so `join()` below is guaranteed to return once any in-flight
    /// verify finishes.
    pub fn shutdown(mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            // A worker that itself panicked shouldn't happen (`verify` is
            // catch_unwind-wrapped in `worker_loop`), but `join()`'s Err is
            // swallowed defensively either way — shutdown must never panic.
            let _ = worker.join();
        }
    }
}

/// The single background worker: pull one `(CandidateKernel, feedback)` off
/// `receiver` at a time, idle-gate on live CUDA load (best-effort), verify
/// without ever letting a panicking `verify` kill the thread, and report the
/// outcome to `feedback`. Returns once `receiver.recv()` errors — i.e. once
/// [`IngestionService::shutdown`] drops the sender and the queue drains.
fn worker_loop<F>(receiver: Receiver<IngestItem>, verify: F, idle_load_threshold: u32)
where
    F: Fn(&CandidateKernel) -> IngestOutcome,
{
    while let Ok((cand, feedback)) = receiver.recv() {
        // Idle-gate (best-effort, bounded sleeps): defer verification until
        // live CUDA inference isn't busy. `inflight_count` is
        // backend-agnostic and reads 0 for a location that's never
        // submitted async work, so this never blocks when there's no CUDA
        // device at all (no-GPU tests / CPU-only builds) or when CUDA is
        // simply idle.
        while crate::dispatch::inflight_count(fuel_ir::DeviceLocation::Cuda { gpu_id: 0 })
            >= idle_load_threshold
        {
            std::thread::sleep(Duration::from_millis(5));
        }

        // The ENTIRE per-item body — `verify` AND the resulting
        // on_adopted/on_rejected callback dispatch — runs inside ONE
        // `catch_unwind`. A `ProviderFeedback` implementation is
        // third-party, injected code (a provider bug), the same trust
        // boundary as `verify` itself; a panic from EITHER must not kill
        // the worker thread, or every future `enqueue` would silently
        // degrade to `Backpressure` for the process lifetime.
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| match verify(&cand) {
            IngestOutcome::Adopted(id) => {
                if let Some(fb) = &feedback {
                    fb.on_adopted(&cand.entry_point, id);
                }
            }
            IngestOutcome::Rejected(report) => {
                if let Some(fb) = &feedback {
                    fb.on_rejected(&report);
                }
            }
        }));

        if outcome.is_err() {
            // Never-panic: a candidate whose verify step OR whose feedback
            // callback panics is logged and skipped, never allowed to kill
            // the worker thread — a subsequent enqueue must still be
            // processed (see `worker_survives_a_panicking_verify` and
            // `worker_survives_a_panicking_feedback_callback`).
            eprintln!(
                "jit_ingest: verify or feedback callback panicked for candidate '{}'; skipping (worker continues)",
                cand.entry_point
            );
        }
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

    // -----------------------------------------------------------------
    // Hardening — `adopt_verified`'s never-panic reject branches. Both are
    // testable WITHOUT a live device: they return before (or without ever
    // performing) any GPU work, so these are non-`#[ignore]` (run in the
    // `cuda,jit` suite with no device attached).
    // -----------------------------------------------------------------

    /// `adopt_verified` never-panic branch: a `Pass`-verdict candidate with
    /// `decompose: None` (nothing to register a region from) must be
    /// `Rejected` with `failed_claim == "no_decompose"`, never unwrapped.
    /// This branch returns before `adopt_runtime_fused` (or any GPU call) is
    /// ever reached, so it needs no live CUDA device.
    #[test]
    #[cfg(feature = "cuda")]
    fn adopt_verified_rejects_a_candidate_without_a_decompose() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "test::adopt_verified::no_decompose".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: None,
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_9001,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        };

        match adopt_verified(&cand) {
            IngestOutcome::Rejected(r) => {
                assert_eq!(r.failed_claim, "no_decompose");
            }
            IngestOutcome::Adopted(_) => {
                panic!("a candidate with no decompose must never be adopted")
            }
        }
    }

    /// `adopt_verified` never-panic branch: a `Pass`-verdict candidate whose
    /// `decompose` region has non-contiguous binds (`{0, 2}`, missing `1`) is
    /// one `register_runtime_fused` refuses with `NonContiguousBinds`, so
    /// `adopt_runtime_fused` returns `None` (region not registrable) —
    /// `adopt_verified` must surface that as `Rejected` with
    /// `failed_claim == "adopt_failed"`, never panic. The rejection is
    /// decided inside `register_runtime_fused`'s bind-index check, before any
    /// GPU kernel is invoked, so this needs no live CUDA device.
    /// `entry_point` carries a distinctive suffix (process-global registry)
    /// though this path never actually registers anything.
    #[test]
    #[cfg(feature = "cuda")]
    fn adopt_verified_rejects_when_the_region_is_not_registrable() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        // Bind indices {0, 2} — missing 1 — register_runtime_fused rejects
        // this as NonContiguousBinds.
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 2 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "test::adopt_verified::non_contiguous_binds::run_9002".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(region),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_9002,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        };

        match adopt_verified(&cand) {
            IngestOutcome::Rejected(r) => {
                assert_eq!(r.failed_claim, "adopt_failed");
            }
            IngestOutcome::Adopted(_) => {
                panic!("a non-contiguous-bind region must never be adopted")
            }
        }
    }

    // -----------------------------------------------------------------
    // Task 7 — IngestionService (NO GPU: `start_with_verify` is the seam;
    // these run under `--features jit` alone).
    // -----------------------------------------------------------------

    /// A [`crate::kernel::KernelRef`]-shaped no-op — `IngestionService`'s
    /// worker never actually calls `cand.kernel` (verification is injected
    /// via `start_with_verify`), so this only needs to type-check as a valid
    /// function pointer to build a [`CandidateKernel`] with no CUDA/GPU
    /// dependency at all.
    fn dummy_kernel(
        _inputs: &[Arc<std::sync::RwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<std::sync::RwLock<fuel_memory::Storage>>],
        _layouts: &[fuel_ir::Layout],
        _params: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }

    /// A minimal, GPU-free [`CandidateKernel`] — `decompose: None` because
    /// none of the Task-7 tests ever reach real verification (it's always
    /// mocked out via `start_with_verify`), so there's nothing to realize a
    /// reference from.
    fn test_candidate(entry_point: &str) -> CandidateKernel {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        CandidateKernel {
            entry_point: entry_point.to_string(),
            kernel: dummy_kernel,
            op_params: crate::kernel::OpParams::None,
            decompose: None,
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0xDEAD_BEEF,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        }
    }

    /// Records every `on_rejected`/`on_adopted` callback it receives, and
    /// (if built with `with_notify`) fires a one-shot signal on the first
    /// callback — the deterministic "the worker finished processing an
    /// item" synchronization point the async tests below wait on instead of
    /// sleeping.
    #[derive(Default)]
    struct RecordingFeedback {
        rejected: std::sync::Mutex<Vec<String>>,
        adopted: std::sync::Mutex<Vec<(String, fuel_graph::registry::FusedOpId)>>,
        notify: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>>,
    }

    impl RecordingFeedback {
        fn with_notify(tx: std::sync::mpsc::Sender<()>) -> Self {
            Self { notify: std::sync::Mutex::new(Some(tx)), ..Default::default() }
        }

        fn fire_notify(&self) {
            if let Some(tx) = self.notify.lock().unwrap().take() {
                let _ = tx.send(());
            }
        }
    }

    impl ProviderFeedback for RecordingFeedback {
        fn on_rejected(&self, report: &RejectionReport) {
            self.rejected.lock().unwrap().push(report.failed_claim.to_string());
            self.fire_notify();
        }

        fn on_adopted(&self, entry_point: &str, id: fuel_graph::registry::FusedOpId) {
            self.adopted.lock().unwrap().push((entry_point.to_string(), id));
            self.fire_notify();
        }
    }

    /// A bounded queue of 1 + a worker deterministically held mid-verify (it
    /// signals `started_tx` the instant it's invoked, then blocks on
    /// `release_rx`) means: enqueue #1 is taken by the worker (buffer empty
    /// once `started_rx` fires), enqueue #2 fills the now-empty 1-slot
    /// buffer, and enqueue #3 — issued while the worker is STILL blocked and
    /// the buffer is STILL full — must deterministically backpressure. No
    /// sleep anywhere: `started_rx.recv()` is the only synchronization point
    /// needed before the buffer-state assertions become deterministic.
    #[test]
    fn enqueue_backpressures_and_notifies_when_full() {
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let cfg = IngestionConfig { queue_bound: 1, max_concurrent: 1, idle_load_threshold: 1 };
        let svc = IngestionService::start_with_verify(
            move |_cand| {
                started_tx.send(()).ok();
                // Block here until the test releases us — holds the worker
                // out of `recv()` so the queue's buffer state is fully
                // under the test's control.
                release_rx.recv().ok();
                IngestOutcome::Rejected(RejectionReport {
                    entry_point: "mock".to_string(),
                    failed_claim: "mock_verify_result",
                    detail: "mock verify (Task 7 backpressure test)".to_string(),
                    ledger_record: None,
                })
            },
            cfg,
        );

        // #1: taken by the worker's recv() (may buffer briefly, but the
        // worker will drain it as soon as it's scheduled).
        svc.enqueue(test_candidate("c1"), None).expect("first enqueue is accepted");
        // Deterministic sync point: by the time this returns, the worker
        // has already called `recv()` (removing c1 from the buffer) and
        // entered the verify closure.
        started_rx.recv().expect("worker started processing the first candidate");

        // #2: the buffer is now guaranteed empty (0/1) — this fills it.
        svc.enqueue(test_candidate("c2"), None).expect("second enqueue fills the 1-slot buffer");

        // #3: the buffer is guaranteed full (1/1) and the worker is
        // guaranteed still blocked in verify (it hasn't been released yet)
        // — this must backpressure.
        let fb = Arc::new(RecordingFeedback::default());
        let result = svc.enqueue(test_candidate("c3"), Some(fb.clone()));
        assert!(matches!(result, Err(Backpressure)), "queue is full; expected Backpressure");
        assert_eq!(
            fb.rejected.lock().unwrap().as_slice(),
            &["queue_full".to_string()],
            "a full queue must synchronously notify the provided feedback with queue_full"
        );

        // Drain the worker (c1's verify, then c2's) so `shutdown` can join
        // rather than hang on a still-blocked thread.
        release_tx.send(()).ok();
        release_tx.send(()).ok();
        svc.shutdown();
    }

    /// A mock verify that returns `Adopted` must fire the feedback's
    /// `on_adopted` — synchronized deterministically via `RecordingFeedback`'s
    /// one-shot notify channel (no sleep).
    #[test]
    fn worker_fires_on_adopted_for_adopted_outcome() {
        let cfg = IngestionConfig::default();
        let adopted_id = fuel_graph::registry::FusedOpId(0x8001);
        let svc =
            IngestionService::start_with_verify(move |_cand| IngestOutcome::Adopted(adopted_id), cfg);

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(RecordingFeedback::with_notify(tx));
        svc.enqueue(test_candidate("adopted-one"), Some(fb.clone())).expect("enqueue accepted");

        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker should report the adopted outcome");

        assert_eq!(
            fb.adopted.lock().unwrap().as_slice(),
            &[("adopted-one".to_string(), adopted_id)],
            "on_adopted must fire with the entry_point and the adopted id"
        );
        assert!(fb.rejected.lock().unwrap().is_empty());

        svc.shutdown();
    }

    /// Hardening regression: `idle_load_threshold == 0` must NOT starve the
    /// worker forever. `inflight_count(..) >= 0` is always true for a `u32`
    /// count, so an unclamped read of the config would have the worker's
    /// idle-gate spin-sleep indefinitely and never reach `verify` — proven
    /// here by asserting the callback still fires within a bounded timeout.
    /// Mirrors `worker_fires_on_adopted_for_adopted_outcome`'s no-sleep
    /// synchronization style, just with `idle_load_threshold: 0` in the
    /// config.
    #[test]
    fn worker_does_not_stall_when_idle_threshold_is_zero() {
        let cfg = IngestionConfig { idle_load_threshold: 0, ..Default::default() };
        let adopted_id = fuel_graph::registry::FusedOpId(0x8003);
        let svc =
            IngestionService::start_with_verify(move |_cand| IngestOutcome::Adopted(adopted_id), cfg);

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(RecordingFeedback::with_notify(tx));
        svc.enqueue(test_candidate("zero-threshold"), Some(fb.clone())).expect("enqueue accepted");

        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker must not stall forever with idle_load_threshold == 0");

        assert_eq!(
            fb.adopted.lock().unwrap().as_slice(),
            &[("zero-threshold".to_string(), adopted_id)],
            "on_adopted must fire once the worker processes the item"
        );

        svc.shutdown();
    }

    /// A verify that panics must NOT crash the worker thread — proven by
    /// enqueueing a SECOND candidate afterward (with a non-panicking mock
    /// outcome) and observing that it still gets processed. Synchronized
    /// via the second item's notify channel; nothing waits on the panicking
    /// first item beyond letting the worker move on to the second.
    #[test]
    fn worker_survives_a_panicking_verify() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_worker = call_count.clone();
        let cfg = IngestionConfig::default();
        let svc = IngestionService::start_with_verify(
            move |_cand| {
                let n = call_count_worker.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    panic!("mock verify panics on purpose (Task 7 panic-survival test)");
                }
                IngestOutcome::Rejected(RejectionReport {
                    entry_point: "after-panic".to_string(),
                    failed_claim: "mock_after_panic",
                    detail: "mock verify, second call, after the first panicked".to_string(),
                    ledger_record: None,
                })
            },
            cfg,
        );

        // First candidate: its verify panics. No feedback attached — there
        // is nothing to observe from this call directly; its only job is to
        // try to kill the worker.
        svc.enqueue(test_candidate("panics"), None).expect("enqueue accepted");

        // Second candidate: only processed if the worker survived the first
        // panic and looped back to `recv()`. Wait on ITS notification —
        // deterministic proof the worker is alive and serial-processing.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(RecordingFeedback::with_notify(tx));
        svc.enqueue(test_candidate("after-panic"), Some(fb.clone())).expect("enqueue accepted");

        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker must survive the panic and process the next item");

        assert_eq!(
            fb.rejected.lock().unwrap().as_slice(),
            &["mock_after_panic".to_string()],
            "the post-panic item must be processed normally"
        );
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "both candidates reached verify");

        svc.shutdown();
    }

    /// A panicking `ProviderFeedback` CALLBACK (not `verify` itself) must
    /// also NOT crash the worker thread — the same trust boundary as a
    /// panicking `verify` (a third-party provider bug), and the whole
    /// per-item body (verify + callback dispatch) now runs under one
    /// `catch_unwind`. Mirrors `worker_survives_a_panicking_verify`'s
    /// two-item structure: item 1's `verify` SUCCEEDS (`Adopted`) but its
    /// feedback's `on_adopted` panics; item 2 is only reached if the worker
    /// survived that and looped back to `recv()`, observed deterministically
    /// via its own notify channel (no sleep).
    #[test]
    fn worker_survives_a_panicking_feedback_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct PanickingFeedback;
        impl ProviderFeedback for PanickingFeedback {
            fn on_rejected(&self, _report: &RejectionReport) {
                panic!("mock on_rejected panics on purpose (Task 7 callback-panic-survival test)");
            }
            fn on_adopted(&self, _entry_point: &str, _id: fuel_graph::registry::FusedOpId) {
                panic!("mock on_adopted panics on purpose (Task 7 callback-panic-survival test)");
            }
        }

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_worker = call_count.clone();
        let adopted_id = fuel_graph::registry::FusedOpId(0x8002);
        let cfg = IngestionConfig::default();
        let svc = IngestionService::start_with_verify(
            move |_cand| {
                call_count_worker.fetch_add(1, Ordering::SeqCst);
                IngestOutcome::Adopted(adopted_id)
            },
            cfg,
        );

        // First candidate: verify SUCCEEDS (Adopted), but the feedback it's
        // paired with panics inside `on_adopted` — the callback dispatch,
        // not verify. Proves Fix 1's widened guard, not just the
        // pre-existing verify-panic guard.
        let panicking_fb = Arc::new(PanickingFeedback);
        svc.enqueue(test_candidate("panics-in-callback"), Some(panicking_fb))
            .expect("enqueue accepted");

        // Second candidate: only reached if the worker survived the
        // callback panic and looped back to `recv()`. Wait on ITS notify
        // channel — deterministic proof of liveness, no sleep.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(RecordingFeedback::with_notify(tx));
        svc.enqueue(test_candidate("after-callback-panic"), Some(fb.clone()))
            .expect("enqueue accepted");

        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker must survive the on_adopted panic and process the next item");

        assert_eq!(
            fb.adopted.lock().unwrap().as_slice(),
            &[("after-callback-panic".to_string(), adopted_id)],
            "the post-panic item must be processed normally"
        );
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "both candidates reached verify");

        svc.shutdown();
    }

    // -----------------------------------------------------------------
    // Task 8 — end-to-end GPU wiring: `IngestionService::start` (the
    // PRODUCTION constructor, verifying against a real `CudaDevice` via
    // `ingest_one`) driven through the whole stack: enqueue → worker →
    // ingest_one → verify_candidate → reference_output → adopt_verified /
    // rejection → ProviderFeedback callback.
    // -----------------------------------------------------------------

    /// A [`ProviderFeedback`] that fires a one-shot notify on EITHER
    /// callback (unlike `RecordingFeedback` above, which is Task-7-local
    /// and mock-verify-only) — the deterministic sync point Task 8's e2e
    /// tests wait on instead of sleeping, mirroring `RecordingFeedback::
    /// with_notify`'s pattern but against the real `ingest_one` path.
    #[cfg(feature = "cuda")]
    #[derive(Default)]
    struct E2eFeedback {
        adopted: std::sync::Mutex<Vec<(String, fuel_graph::registry::FusedOpId)>>,
        rejected: std::sync::Mutex<Vec<RejectionReportSnapshot>>,
        notify: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>>,
    }

    /// An owned snapshot of a [`RejectionReport`] — the report itself
    /// borrows nothing but `ledger_record` isn't `Clone`-needed here, so
    /// this just lifts the two fields the assertions below check into an
    /// owned, `'static` value the callback can push into a `Mutex` from
    /// inside the borrowed `&RejectionReport`.
    #[cfg(feature = "cuda")]
    #[derive(Debug)]
    struct RejectionReportSnapshot {
        failed_claim: &'static str,
        detail: String,
    }

    #[cfg(feature = "cuda")]
    impl E2eFeedback {
        fn with_notify(tx: std::sync::mpsc::Sender<()>) -> Self {
            Self { notify: std::sync::Mutex::new(Some(tx)), ..Default::default() }
        }

        fn fire_notify(&self) {
            if let Some(tx) = self.notify.lock().unwrap().take() {
                let _ = tx.send(());
            }
        }
    }

    #[cfg(feature = "cuda")]
    impl ProviderFeedback for E2eFeedback {
        fn on_rejected(&self, report: &RejectionReport) {
            self.rejected.lock().unwrap().push(RejectionReportSnapshot {
                failed_claim: report.failed_claim,
                detail: report.detail.clone(),
            });
            self.fire_notify();
        }

        fn on_adopted(&self, entry_point: &str, id: fuel_graph::registry::FusedOpId) {
            self.adopted.lock().unwrap().push((entry_point.to_string(), id));
            self.fire_notify();
        }
    }

    /// Build the `Add`-region `add_f32` candidate (matching function) with a
    /// DISTINCTIVE `entry_point` so it doesn't collide with any other
    /// `#[ignore]` test's process-global registration in this binary (same
    /// discipline as `ingest_adopts_add_candidate_for_the_add_region`).
    #[cfg(feature = "cuda")]
    fn e2e_add_candidate() -> CandidateKernel {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        CandidateKernel {
            entry_point: "test::e2e::add_for_add_region::run_e2e_8801".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_8801,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        }
    }

    /// Build the `mul_f32`-vs-`Add`-region MISMATCHED candidate — the plan's
    /// original Step-1 interleaved-rope rejection leg is infeasible (rotate-
    /// half rope isn't a `PatternNode`; `rope_apply` is a reverted
    /// registration, not a wired `KernelRef` — see
    /// `ingest_rejects_mul_candidate_for_the_add_region`'s doc comment above
    /// for the full rationale). `declared: REFERENCE` is load-bearing: it
    /// declares a NUMERIC claim so the mul-vs-add mismatch is actually
    /// compared and caught, not just skipped as an unclaimed property.
    #[cfg(feature = "cuda")]
    fn e2e_mul_candidate() -> CandidateKernel {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        CandidateKernel {
            entry_point: "test::e2e::mul_for_add_region::run_e2e_8802".to_string(),
            kernel: crate::baracuda_dispatch::binary::mul_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_8802,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
        }
    }

    /// Spec-B Task-8 acceptance (live GPU): drive the WHOLE ingestion
    /// service end-to-end through its PRODUCTION constructor
    /// (`IngestionService::start`, not the `start_with_verify` test seam
    /// Task 7's tests use) — enqueue → worker thread → `ingest_one` →
    /// `verify_candidate` → `reference_output` → `adopt_verified` →
    /// `ProviderFeedback::on_adopted`. Asserts BOTH that the callback fired
    /// AND that the adopted op is genuinely visible to the capability gate
    /// (`fused_kernel_available_in`) — the same check
    /// `ingest_adopts_add_candidate_for_the_add_region` makes directly
    /// against `ingest_one`, now exercised through the async service.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn ingestion_service_adopts_a_matching_add_candidate_e2e() {
        use fuel_ir::probe::BackendId;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let svc = IngestionService::start(dev, IngestionConfig::default());

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(E2eFeedback::with_notify(tx));
        svc.enqueue(e2e_add_candidate(), Some(fb.clone()))
            .expect("enqueue accepted (queue starts empty)");

        rx.recv_timeout(Duration::from_secs(30))
            .expect("worker should report the adopted outcome within 30s");

        let adopted = fb.adopted.lock().unwrap();
        assert_eq!(adopted.len(), 1, "exactly one on_adopted callback: {adopted:?}");
        let (entry_point, id) = &adopted[0];
        assert_eq!(entry_point, "test::e2e::add_for_add_region::run_e2e_8801");
        assert!(
            fb.rejected.lock().unwrap().is_empty(),
            "the matching add_f32 candidate must not be rejected"
        );

        let table = crate::dispatch::global_bindings();
        assert!(
            crate::runtime_fused_kernels::fused_kernel_available_in(&table, *id, BackendId::Cuda),
            "the adopted candidate's kernel must be visible to the capability gate on Cuda"
        );
        drop(table);

        svc.shutdown();
    }

    /// Spec-B Task-8 acceptance (live GPU), rejection leg — the counterpart
    /// to [`ingestion_service_adopts_a_matching_add_candidate_e2e`]: the SAME
    /// production service, driven end-to-end, must reject a candidate whose
    /// kernel computes a different function than the region it claims to
    /// replace (`mul_f32` offered for an `Add` region), reporting a
    /// PRECISION claim (`failed_claim` contains "max") via `on_rejected` —
    /// never `on_adopted`.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn ingestion_service_rejects_a_mismatched_mul_candidate_e2e() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let svc = IngestionService::start(dev, IngestionConfig::default());

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(E2eFeedback::with_notify(tx));
        svc.enqueue(e2e_mul_candidate(), Some(fb.clone()))
            .expect("enqueue accepted (queue starts empty)");

        rx.recv_timeout(Duration::from_secs(30))
            .expect("worker should report the rejected outcome within 30s");

        let rejected = fb.rejected.lock().unwrap();
        assert_eq!(rejected.len(), 1, "exactly one on_rejected callback: {rejected:?}");
        assert!(
            rejected[0].failed_claim.contains("max"),
            "expected a precision claim naming the mismatch, got: {} / {}",
            rejected[0].failed_claim,
            rejected[0].detail
        );
        assert!(
            fb.adopted.lock().unwrap().is_empty(),
            "the mismatched mul_f32 candidate must not be adopted"
        );

        svc.shutdown();
    }
}
