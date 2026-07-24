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

use crate::fkc::verify::{is_transcendental, LedgerRecord};

#[cfg(feature = "cuda")]
use crate::fkc::verify::{
    region_contains_transcendental, verify_bit_stability, verify_precision_bound,
    widen_bound_for_transcendental, Bound, CudaInvoker, HostTensor, KernelInvoker, ProbeInputs,
    VerificationLedger, VerifyError, VerifyOutcome,
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
    /// The op-identity this candidate asserts it implements. `Some(id)` →
    /// verify against Fuel's registered recipe for `id` as the reference
    /// (Task 5); `None` → the Spec B behavior (verify against the
    /// candidate's own `decompose`) is retained. No consumer yet — Tasks
    /// 4/5 are what resolve this into a verification reference.
    pub claimed_op: Option<fuel_graph::registry::FusedOpId>,
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

/// Escalation record for a `Flagged` ingest — a non-authoritative reference
/// (kiss-ref) disagreed, or was the only reference available, on an input
/// beyond corpus coverage. NOT a rejection: per KISS-CONFORM §6.6-0007 a live
/// kiss-ref outcome flags/escalates, never verdicts, in EITHER direction.
pub struct FlagReport {
    pub entry_point: String,
    pub claim: &'static str,
    pub detail: String,
    /// Compact kiss-ref `DiffReport` summary, if a diff was run.
    pub diff_summary: Option<String>,
    /// Always true today: this flag should escalate to mint a corpus vector.
    pub escalate: bool,
}

/// The callback surface a candidate-kernel provider implements to learn the
/// outcome of an offer. `on_rejected` is required (the whole point of the
/// report); `on_adopted` / `on_flagged` are optional telemetry, default no-op.
pub trait ProviderFeedback: Send + Sync {
    fn on_rejected(&self, report: &RejectionReport);
    fn on_adopted(&self, _entry_point: &str, _id: fuel_graph::registry::FusedOpId) {}
    /// A candidate was flagged for escalation (non-authoritative reference
    /// disagreement, or no authoritative reference available). Default no-op.
    fn on_flagged(&self, _report: &FlagReport) {}
}

/// The result of ingesting one candidate kernel: adopted (with the
/// `FusedOpId` it registered under), rejected (with the report explaining
/// why), or flagged for escalation (a non-authoritative reference could not
/// render a verdict — §6.6-0007).
pub enum IngestOutcome {
    Adopted(fuel_graph::registry::FusedOpId),
    Rejected(RejectionReport),
    Flagged(FlagReport),
}

// ---------------------------------------------------------------------------
// Flag-not-verdict (kiss-ref reference) — CPU-side, cuda-independent.
//
// Per KISS-CONFORM §6.6-0007 the frozen corpus is the sole verdict authority;
// kiss-ref (live) FLAGS, never verdicts, for EVERY op class (provenance rule,
// not a precision rule). Until Fuel consumes a populated corpus, recipe-realize
// stays the interim verdict; kiss-ref is an advisory cross-check whose only live
// behaviour change is turning a no-authoritative-reference case into an escalate
// (`Inconclusive`) instead of a hard reject. These types + `classify_floor_verdict`
// are pure so they are unit-tested without a CUDA device.
// ---------------------------------------------------------------------------

/// Determinism class of a floor op. Consumed by the wiring (Task 8) to pick the
/// advisory diff's tolerance band (`Exact` vs 2×-widened for transcendentals);
/// it does NOT gate the verdict (§6.6-0007: kiss-ref never verdicts, any class).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    Exact,
    Transcendental,
}

/// Fuel-side summary of a kiss-ref differential (no kiss-ref types cross this
/// boundary, keeping `classify_floor_verdict` cuda- and dependency-independent).
#[derive(Debug, Clone)]
pub struct DiffOutcome {
    pub within: bool,
    pub max_ulp: Option<u64>,
    pub detail: String,
}

/// Fuel-side summary of the recipe-realize reference verdict (today's interim
/// authority for every op class).
#[derive(Debug, Clone)]
pub struct RefOutcome {
    pub pass: bool,
    pub claim: &'static str,
    pub detail: String,
}

/// Fuel-side summary of a corpus verdict. Produced only when a covering frozen
/// corpus cell exists — which is never, today (`corpus_verdict` returns None).
#[derive(Debug, Clone)]
pub struct CorpusOutcome {
    pub adopt: bool,
    pub claim: &'static str,
    pub detail: String,
}

/// Select the verdict for a floor-op candidate from the available references.
/// Pure + CPU-testable; the cuda `verify_candidate_impl` builds the outcomes and
/// calls this. Precedence (§6.6-0007): corpus (authoritative, dormant) →
/// recipe-realize (interim, all classes) → kiss-only ⇒ escalate → none ⇒ fail.
/// kiss-ref NEVER produces a Pass/Fail here: agreement is not Adopt and a
/// discrepancy is not Reject — it can only escalate.
pub fn classify_floor_verdict(
    kiss: Option<&DiffOutcome>,
    recipe: Option<&RefOutcome>,
    corpus: Option<&CorpusOutcome>,
) -> VerifyVerdict {
    // (1) Corpus is authoritative (dormant: corpus_verdict returns None today).
    if let Some(c) = corpus {
        return if c.adopt {
            VerifyVerdict::Pass
        } else {
            VerifyVerdict::Fail { claim: c.claim, detail: c.detail.clone() }
        };
    }
    // (2) Recipe-realize is the interim verdict for every class. kiss-ref, if
    // present, was already advisory-recorded by the caller; it does not gate.
    if let Some(r) = recipe {
        return if r.pass {
            VerifyVerdict::Pass
        } else {
            VerifyVerdict::Fail { claim: r.claim, detail: r.detail.clone() }
        };
    }
    // (3) No authoritative reference, but kiss-ref could compare ⇒ escalate:
    // never Adopt (agreement ≠ Adopt), never hard-Reject (kiss-ref ≠ verdict).
    if let Some(k) = kiss {
        let detail = format!(
            "no authoritative reference; kiss-ref {} (escalate to corpus): {}",
            if k.within { "agrees" } else { "disagrees" },
            k.detail
        );
        return VerifyVerdict::Inconclusive { claim: "max_ulp", detail };
    }
    // (4) Nothing to compare against.
    VerifyVerdict::Fail {
        claim: "no_reference",
        detail: "no reference available".to_string(),
    }
}

/// The dormant corpus-verdict seam. When Fuel consumes a populated frozen
/// corpus, a covering cell flips Adopt authority to the corpus (§6.6-0007).
///
/// STILL DORMANT after A4b (2026-07-23), by design — NOT for lack of a corpus.
/// KISS's v1 exact-byte golden corpus now EXISTS and is vendored + parsed:
/// `fuel-dispatch/fixtures/kiss-corpus/` (KISS `main` @ `c9153b2`), read by
/// [`crate::kiss_corpus`]. It is a per-`(op, dtype, input-vector)` **oracle**
/// (fixed inputs → one correct output), NOT an `(op, dtype) → adopt/reject`
/// table. This seam's signature carries no candidate output, and its `seed`
/// selects a *random probe* disjoint from the corpus's fixed inputs — so it
/// CANNOT turn the oracle into a candidate verdict without re-running the
/// candidate on the corpus's own inputs (a seam change out of A4b scope).
/// Returning any `Some` here would adopt/reject a candidate the corpus never
/// checked. So it stays `None` and recipe-realize remains the interim
/// authority. Activation needs the seam correction recorded in
/// `docs/design-notes/2026-07-23-kiss-corpus-verdict-seam-mismatch.md` — the
/// precedence wiring in `verify_candidate_impl` is already live for it.
pub fn corpus_verdict(
    _op: fuel_graph::jit::OpTag,
    _dtype: fuel_ir::DType,
    _seed: u64,
) -> Option<CorpusOutcome> {
    None
}

/// Map a non-`Pass` verdict to its ingest outcome. Pure so the new
/// `Inconclusive → Flagged` escalate path is tested without a device. `Pass`
/// never reaches here (it adopts, which needs cuda) — a defensive `Rejected`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn outcome_from_nonadopt_verdict(
    verdict: VerifyVerdict,
    records: Vec<LedgerRecord>,
    entry_point: &str,
) -> IngestOutcome {
    match verdict {
        VerifyVerdict::Pass => IngestOutcome::Rejected(RejectionReport {
            entry_point: entry_point.to_string(),
            failed_claim: "internal",
            detail: "Pass routed to non-adopt mapping".to_string(),
            ledger_record: None,
        }),
        VerifyVerdict::Fail { claim, detail } => {
            let ledger_record = records.into_iter().find(|r| r.claim == claim);
            IngestOutcome::Rejected(RejectionReport {
                entry_point: entry_point.to_string(),
                failed_claim: claim,
                detail,
                ledger_record,
            })
        }
        VerifyVerdict::Inconclusive { claim, detail } => {
            // (D8) Thread a compact kiss-ref summary into the flag report,
            // lifted from the `kiss_ref_advisory` ledger record the advisory
            // cross-check produced — so `on_flagged` learns WHAT kiss-ref saw
            // (result, max ULP, mismatch count, op count) without re-running
            // the diff. `None` when no advisory record was earned (a
            // Fail-arm-only escalate, or the advisory declined the region).
            let diff_summary = records
                .iter()
                .find(|r| r.claim == "kiss_ref_advisory")
                .map(|r| kiss_advisory_diff_summary(r));
            IngestOutcome::Flagged(FlagReport {
                entry_point: entry_point.to_string(),
                claim,
                detail,
                diff_summary,
                escalate: true,
            })
        }
    }
}

/// A compact one-line summary of a `kiss_ref_advisory` [`LedgerRecord`] for a
/// [`FlagReport::diff_summary`]: the advisory result plus the diff evidence
/// (max ULP, mismatch count, op count). Missing evidence keys render as `?`,
/// never panic. Pure + CPU-testable.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn kiss_advisory_diff_summary(r: &LedgerRecord) -> String {
    let field = |k: &str| r.evidence.get(k).map(|v| v.to_string()).unwrap_or_else(|| "?".to_string());
    format!(
        "kiss_ref_advisory={} max_ulp={} mismatches={} op_count={}",
        r.result,
        field("max_ulp"),
        field("mismatches"),
        field("op_count"),
    )
}

/// The single primitive `OpTag` of a decompose that is exactly one `Op` node over
/// identity-ordered `Bind` leaves (`Bind{0}`, `Bind{1}`, …) — the only shape the
/// kiss-ref advisory diff can align a probe against. `None` for a multi-node/fused
/// decompose, a reordered/gapped binding, or no decompose.
// The generalized (region-based) advisory block (T3) no longer consults this;
// it is re-consumed by the dormant corpus consult in the numeric region (Task
// 4/5) and by its CPU unit test. Unconditionally allowed until that re-wiring
// lands (dead in the plain cuda build meanwhile).
#[allow(dead_code)]
fn single_primitive_optag(
    dec: Option<&fuel_graph::jit::PatternNode>,
) -> Option<fuel_graph::jit::OpTag> {
    use fuel_graph::jit::PatternNode;
    let PatternNode::Op { op, operands, .. } = dec? else {
        return None;
    };
    for (i, operand) in operands.iter().enumerate() {
        match operand {
            PatternNode::Bind { index } if *index as usize == i => {}
            _ => return None,
        }
    }
    Some(*op)
}

/// Reinterpret little-endian `f32` bytes as an owned `Vec<f32>`. Safe:
/// `chunks_exact(4)` never yields a short chunk, so the array build can't panic.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Reinterpret little-endian `f64` bytes as an owned `Vec<f64>`. Mirrors
/// [`bytes_to_f32`]: `chunks_exact(8)` never yields a short chunk.
// Consumed by the cuda advisory block's dtype dispatch (`run_region_diff`);
// dead only in the CPU-only (`jit` without `cuda`) build.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// Reinterpret little-endian `f16` bytes as an owned `Vec<half::f16>`. Mirrors
/// [`bytes_to_f32`]: `chunks_exact(2)` never yields a short chunk.
// See `bytes_to_f64`: consumed by `run_region_diff`, dead only CPU-only.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn bytes_to_f16(bytes: &[u8]) -> Vec<half::f16> {
    bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect()
}

/// Reinterpret little-endian `bf16` bytes as an owned `Vec<half::bf16>`.
/// Mirrors [`bytes_to_f32`]: `chunks_exact(2)` never yields a short chunk.
// See `bytes_to_f64`: consumed by `run_region_diff`, dead only CPU-only.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn bytes_to_bf16(bytes: &[u8]) -> Vec<half::bf16> {
    bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]])).collect()
}

/// Fallback per-op ULP ceiling for a transcendental whose kiss `Op` declares no
/// §6.8 ceiling: kiss non-primitives (`Tanh`, `Sigmoid`, `Silu`, `GeluTanh`,
/// `Gelu`, `Rsqrt`) inherit their decomposition's tolerance and return `None`
/// from `Op::ulp_ceiling`. Per the kiss-ref band refinement (2026-07-23): treat
/// a mapped op with no exposed ceiling as 4 ULP.
const ADVISORY_FALLBACK_TRANSCENDENTAL_ULP_CEILING: u64 = 4;

/// The per-op §6.8 ULP-ceiling contribution to the advisory band:
/// `Some(ceiling)` for a transcendental op, `None` for an exact-class op
/// (which contributes to the exact-rounding `n-1` term instead).
///
/// Classification delegates to `fkc/verify/ulp.rs`'s [`is_transcendental`] —
/// the single source of the transcendental set (IEEE-correctly-rounded
/// `Sqrt`/`Recip` are exact-class there, so they carry no ceiling here).
/// Ceilings are read from kiss-ref's OWN API (`kiss_ops_vocab::Op::ulp_ceiling`,
/// verified at the pinned rev `1f3981f`; the `b75a748..1f3981f` bump left the
/// kiss-ops-vocab crate byte-unchanged, so the ceilings did not move) through a
/// mapping arm that mirrors `fuel-kiss-ref-backend::mapping::op_to_kiss`
/// restricted to that set — the
/// adapter crate itself is cuda-gated, and this helper must stay CPU-testable
/// (`--features jit` alone). Keep the two mappings in lockstep.
fn advisory_op_ulp_ceiling(op: fuel_graph::jit::OpTag) -> Option<u64> {
    use fuel_graph::jit::OpTag as T;
    use kiss_ops_vocab::Op as K;
    if !is_transcendental(op) {
        return None;
    }
    let kiss = match op {
        T::Exp => Some(K::Exp),
        T::Log => Some(K::Log),
        T::Sin => Some(K::Sin),
        T::Cos => Some(K::Cos),
        T::Erf => Some(K::Erf),
        T::Tanh => Some(K::Tanh),
        T::Sigmoid => Some(K::Sigmoid),
        T::Silu => Some(K::Silu),
        T::Gelu => Some(K::GeluTanh), // Fuel Gelu = tanh-approx
        T::GeluErf => Some(K::Gelu),  // Fuel GeluErf = exact erf
        T::Rsqrt => Some(K::Rsqrt),
        // A transcendental tag with no kiss mapping (none today — the match is
        // total over `is_transcendental`'s set) takes the fallback ceiling.
        _ => None,
    };
    let declared = kiss.and_then(|k| k.ulp_ceiling());
    // `ceil() as u64` saturates (never-panic); declared ceilings are small
    // integers (2/4/8) today.
    Some(declared.map_or(ADVISORY_FALLBACK_TRANSCENDENTAL_ULP_CEILING, |c| c.ceil() as u64))
}

/// The advisory comparison band for a region, per the kiss-ref tolerance
/// refinement (2026-07-23) — `None` ⇒ compare `Tolerance::Exact`, `Some(n)` ⇒
/// `Tolerance::Ulp(n)`:
///
/// * single exact op → exact (`None`);
/// * multi-node exact-only region → `n_ops - 1` (each intermediate rounding
///   contributes at most ~1 ULP; the final rounding matches the reference's);
/// * transcendental-containing region → `Σ per-op §6.8 ceilings over the
///   region's transcendental ops + (n_exact_ops - 1)`, the exact term
///   saturating at 0 (a lone transcendental keeps exactly its own ceiling).
///
/// CANCELLATION CAVEAT (pinned): linear ULP addition is a first-order model —
/// cancellation-heavy regions (e.g. subtraction of nearby intermediates) can
/// exceed the band and flag spuriously. The label stays advisory-only per
/// KISS-CONFORM §6.6-0007 (kiss-ref flags, never verdicts), and the raw
/// `max_ulp` is always recorded alongside any flag.
///
/// `None` also for an op-free region (nothing to band; callers gate on the
/// adapter's `region_supported`, which requires ≥1 op node, before diffing).
/// `SeeThrough` is traversed through (structural metadata, mirroring
/// [`region_contains_transcendental`]); `Bind`/`Any` are leaves.
// Consumed by the cuda advisory block (its band selection); dead only CPU-only.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn advisory_ulp_band(region: &fuel_graph::jit::PatternNode) -> Option<u64> {
    use fuel_graph::jit::PatternNode;
    fn walk(node: &PatternNode, n_ops: &mut u64, n_exact: &mut u64, trans_sum: &mut Option<u64>) {
        match node {
            PatternNode::Op { op, operands, .. } => {
                *n_ops += 1;
                match advisory_op_ulp_ceiling(*op) {
                    Some(c) => {
                        *trans_sum = Some(trans_sum.unwrap_or(0).saturating_add(c));
                    }
                    None => *n_exact += 1,
                }
                for o in operands {
                    walk(o, n_ops, n_exact, trans_sum);
                }
            }
            PatternNode::SeeThrough { then } => walk(then, n_ops, n_exact, trans_sum),
            PatternNode::Bind { .. } | PatternNode::Any => {}
        }
    }
    let (mut n_ops, mut n_exact, mut trans_sum) = (0u64, 0u64, None);
    walk(region, &mut n_ops, &mut n_exact, &mut trans_sum);
    if n_ops == 0 {
        return None;
    }
    match trans_sum {
        // Exact-only region: single op compares exact; n-node accumulates n-1.
        None => {
            if n_ops == 1 {
                None
            } else {
                Some(n_ops - 1)
            }
        }
        Some(sum) => Some(sum.saturating_add(n_exact.saturating_sub(1))),
    }
}

/// The region the kiss-ref advisory cross-check runs against (design D1,
/// derive-from-registry): the candidate's own submitted `decompose` when it
/// carries one, else — for a runtime-registered `claimed` op — Fuel's OWN
/// registered region for that claim
/// ([`runtime_region`](fuel_graph::runtime_fused::runtime_region)). A STATIC
/// claimed id carries no `PatternNode` region until the registry's decomposes
/// migrate to PatternNode data, so it declines (`None` — no static table);
/// an unregistered id likewise resolves to `None`, never a panic.
// Consumed by the cuda advisory block (region derivation); dead only CPU-only.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn advisory_region(
    decompose: Option<&fuel_graph::jit::PatternNode>,
    claimed: Option<fuel_graph::registry::FusedOpId>,
) -> Option<fuel_graph::jit::PatternNode> {
    decompose.cloned().or_else(|| claimed.and_then(fuel_graph::runtime_fused::runtime_region))
}

/// Pre-invoke eligibility for the Plan-T5 non-f32 numeric-claim ESCALATE path:
/// a numeric-claim candidate whose output dtype is one of the kiss-coverable
/// non-f32 floats (F64/F16/BF16) AND whose advisory region kiss-ref can evaluate
/// (`region_kiss_supported`, computed by the caller through the cuda-gated
/// adapter's `region_supported`) proceeds to invoke + the advisory diff and
/// escalates to [`VerifyVerdict::Inconclusive`] in the numeric region, rather
/// than the pre-invoke f32-only hard Fail. F32 is NEVER escalated — its numeric
/// authority (`verify_precision_bound`) IS f32, so it is verified inline. Any
/// dtype outside {F64,F16,BF16}, or a region kiss-ref cannot cover (unmapped
/// op/dtype, non-default attrs, no advisory region ⇒ `region_kiss_supported ==
/// false`), keeps the honest f32-only hard Fail — same claim, same detail bytes.
///
/// Pure (the adapter support bool is threaded in) so the combining logic is
/// unit-tested under `--features jit` alone, without the adapter or a device.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn nonf32_escalate_eligible(out_dtype: fuel_ir::DType, region_kiss_supported: bool) -> bool {
    region_kiss_supported
        && matches!(
            out_dtype,
            fuel_ir::DType::F64 | fuel_ir::DType::F16 | fuel_ir::DType::BF16
        )
}

#[cfg(test)]
mod flag_not_verdict_tests {
    use super::*;

    #[test]
    fn new_outcome_types_construct_and_match() {
        let flag = FlagReport {
            entry_point: "k".into(),
            claim: "max_ulp",
            detail: "kiss-ref discrepancy".into(),
            diff_summary: Some("max_ulp=3".into()),
            escalate: true,
        };
        assert!(matches!(IngestOutcome::Flagged(flag),
            IngestOutcome::Flagged(ref r) if r.escalate && r.claim == "max_ulp"));
        let v = VerifyVerdict::Inconclusive { claim: "max_ulp", detail: "x".into() };
        assert!(matches!(v, VerifyVerdict::Inconclusive { claim, .. } if claim == "max_ulp"));
    }

    #[test]
    fn classify_corpus_wins_when_present() {
        let c = CorpusOutcome { adopt: true, claim: "max_ulp", detail: "corpus".into() };
        assert!(matches!(classify_floor_verdict(None, None, Some(&c)), VerifyVerdict::Pass));
        let cr = CorpusOutcome { adopt: false, claim: "max_ulp", detail: "corpus".into() };
        assert!(matches!(classify_floor_verdict(None, None, Some(&cr)),
            VerifyVerdict::Fail { claim, .. } if claim == "max_ulp"));
    }

    #[test]
    fn classify_recipe_is_interim_verdict_kiss_advisory() {
        // kiss-ref disagrees but recipe passes: recipe verdict stands, kiss never gates.
        let kiss = DiffOutcome { within: false, max_ulp: Some(5), detail: "disagree".into() };
        let recipe = RefOutcome { pass: true, claim: "max_ulp", detail: "recipe ok".into() };
        assert!(matches!(classify_floor_verdict(Some(&kiss), Some(&recipe), None), VerifyVerdict::Pass));
    }

    #[test]
    fn classify_no_reference_but_kiss_is_inconclusive() {
        let agree = DiffOutcome { within: true, max_ulp: Some(0), detail: "agree".into() };
        assert!(matches!(classify_floor_verdict(Some(&agree), None, None),
            VerifyVerdict::Inconclusive { .. }), "kiss agreement != Adopt");
        let off = DiffOutcome { within: false, max_ulp: Some(4), detail: "disagree".into() };
        assert!(matches!(classify_floor_verdict(Some(&off), None, None),
            VerifyVerdict::Inconclusive { .. }), "kiss discrepancy != Reject");
    }

    #[test]
    fn classify_all_none_fails() {
        assert!(matches!(classify_floor_verdict(None, None, None),
            VerifyVerdict::Fail { claim, .. } if claim == "no_reference"));
    }

    #[test]
    fn corpus_verdict_is_dormant_returns_none() {
        assert!(corpus_verdict(fuel_graph::jit::OpTag::Add, fuel_ir::DType::F32, 0).is_none());
    }

    #[test]
    fn map_fail_to_rejected_and_inconclusive_to_flagged() {
        let out = outcome_from_nonadopt_verdict(
            VerifyVerdict::Fail { claim: "max_ulp", detail: "off".into() }, vec![], "k");
        assert!(matches!(out, IngestOutcome::Rejected(ref r) if r.failed_claim == "max_ulp"));
        let out = outcome_from_nonadopt_verdict(
            VerifyVerdict::Inconclusive { claim: "max_ulp", detail: "esc".into() }, vec![], "k");
        assert!(matches!(out, IngestOutcome::Flagged(ref r) if r.escalate && r.claim == "max_ulp"));
    }

    /// D8 (Task 4): mapping an `Inconclusive` verdict to `Flagged` threads a
    /// compact kiss-ref summary into `FlagReport.diff_summary`, lifted from the
    /// `kiss_ref_advisory` ledger record the advisory cross-check produced —
    /// so a provider's `on_flagged` learns WHAT kiss-ref saw (max ULP,
    /// mismatch count, advisory result) without re-running the diff. With no
    /// advisory record present the summary stays `None` (a Fail-arm-only or
    /// advisory-declined escalate).
    #[test]
    fn map_inconclusive_carries_diff_summary_from_advisory_record() {
        let advisory = LedgerRecord {
            kernel_ref: "k".into(),
            backend: "Cuda".into(),
            dtypes: vec!["F64".into(), "F64".into()],
            kernel_revision_hash: 0,
            claim: "kiss_ref_advisory".into(),
            result: "flag".into(),
            verified_at: "epoch:0".into(),
            protocol_version: 1,
            evidence: serde_json::json!({
                "dtype": "F64",
                "op_count": 1,
                "max_ulp": 3,
                "mismatches": 2,
            }),
        };
        let out = outcome_from_nonadopt_verdict(
            VerifyVerdict::Inconclusive { claim: "max_ulp", detail: "esc".into() },
            vec![advisory],
            "k",
        );
        match out {
            IngestOutcome::Flagged(r) => {
                let summary =
                    r.diff_summary.expect("diff_summary threaded from the advisory record");
                assert!(summary.contains("max_ulp=3"), "names the max ULP: {summary}");
                assert!(summary.contains("mismatches=2"), "and the mismatch count: {summary}");
                assert!(summary.contains("flag"), "and the advisory result: {summary}");
            }
            _ => panic!("expected Flagged"),
        }

        // No advisory record present -> diff_summary stays None (non-regression
        // with the vec![] cases in map_fail_to_rejected_and_inconclusive_to_flagged).
        let out = outcome_from_nonadopt_verdict(
            VerifyVerdict::Inconclusive { claim: "max_ulp", detail: "esc".into() },
            vec![],
            "k",
        );
        match out {
            IngestOutcome::Flagged(r) => assert!(r.diff_summary.is_none()),
            _ => panic!("expected Flagged"),
        }
    }

    #[test]
    fn single_primitive_optag_extracts_and_declines() {
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        let add = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        assert!(matches!(single_primitive_optag(Some(&add)), Some(OpTag::Add)));
        let reordered = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 1 }, PatternNode::Bind { index: 0 }],
        };
        assert!(single_primitive_optag(Some(&reordered)).is_none());
        let nested = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![add.clone(), PatternNode::Bind { index: 1 }],
        };
        assert!(single_primitive_optag(Some(&nested)).is_none());
        assert!(single_primitive_optag(None).is_none());
    }

    #[test]
    fn bytes_to_f32_roundtrips() {
        let v = [1.0f32, -2.5, 3.25];
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes_to_f32(&bytes), vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn bytes_to_f64_f16_bf16_roundtrip() {
        let v64 = [1.0f64, -2.5, 1e300];
        let bytes: Vec<u8> = v64.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes_to_f64(&bytes), v64.to_vec());

        let v16: Vec<half::f16> =
            [1.0f32, -2.5, 0.5].iter().map(|&x| half::f16::from_f32(x)).collect();
        let bytes: Vec<u8> = v16.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes_to_f16(&bytes), v16);

        let vb: Vec<half::bf16> =
            [1.0f32, -2.5, 0.5].iter().map(|&x| half::bf16::from_f32(x)).collect();
        let bytes: Vec<u8> = vb.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes_to_bf16(&bytes), vb);
    }

    /// The advisory band per the kiss-ref tolerance refinement (2026-07-23):
    /// single exact op → Exact (`None`); multi-node exact-only → `Ulp(n-1)`;
    /// transcendental-containing → `Ulp(Σ per-op §6.8 ceilings + (n_exact-1))`,
    /// exact term saturating at 0. Ceilings are read from kiss-ref's own
    /// `Op::ulp_ceiling` (declared 4 for exp; fallback 4 for kiss
    /// non-primitives like tanh).
    #[test]
    fn advisory_ulp_band_selects_by_region_shape() {
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        let bind = |i: u8| PatternNode::Bind { index: i };
        let node = |op: OpTag, operands: Vec<PatternNode>| PatternNode::Op {
            op,
            operands,
            attrs: OpAttrs::default(),
        };
        // Single exact op -> Exact (None).
        let add = node(OpTag::Add, vec![bind(0), bind(1)]);
        assert_eq!(advisory_ulp_band(&add), None);
        // Multi-node exact-only -> Ulp(n_ops - 1).
        let relu_add = node(OpTag::Relu, vec![add.clone()]);
        assert_eq!(advisory_ulp_band(&relu_add), Some(1));
        // Lone transcendental keeps exactly its own §6.8 ceiling (exp: kiss
        // declares 4).
        let exp = node(OpTag::Exp, vec![bind(0)]);
        assert_eq!(advisory_ulp_band(&exp), Some(4));
        // Transcendental + exact: ceiling sum + (n_exact - 1) = 4 + 0.
        let exp_add = node(OpTag::Exp, vec![add.clone()]);
        assert_eq!(advisory_ulp_band(&exp_add), Some(4));
        // Two transcendentals (exp declared 4; tanh is a kiss non-primitive ->
        // fallback 4) + one exact: 8 + (1 - 1) = 8.
        let tanh_exp_add = node(
            OpTag::Tanh,
            vec![node(OpTag::Add, vec![node(OpTag::Exp, vec![bind(0)]), bind(1)])],
        );
        assert_eq!(advisory_ulp_band(&tanh_exp_add), Some(8));
        // All-transcendental region: the exact term saturates at 0, never
        // underflows.
        let exp_tanh = node(OpTag::Exp, vec![node(OpTag::Tanh, vec![bind(0)])]);
        assert_eq!(advisory_ulp_band(&exp_tanh), Some(8));
        // Sqrt is IEEE correctly-rounded -> exact class (mirrors
        // fkc/verify/ulp.rs `is_transcendental`): sqrt(a+b) is 2 exact ops.
        let sqrt_add = node(OpTag::Sqrt, vec![add.clone()]);
        assert_eq!(advisory_ulp_band(&sqrt_add), Some(1));
        // Op-free region: nothing to band.
        assert_eq!(advisory_ulp_band(&bind(0)), None);
    }

    /// D1 (derive-from-registry): the advisory region is the candidate's own
    /// submitted decompose when present; else a runtime-registered claimed id
    /// resolves to Fuel's OWN registered region; a static id (no `PatternNode`
    /// region until the registry decomposes migrate) and an unregistered id
    /// decline.
    #[test]
    fn advisory_region_resolves_runtime_claimed_id_and_declines_static() {
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_graph::registry::{FusedOpId, FusedOps};
        let bind = |i: u8| PatternNode::Bind { index: i };
        let region = PatternNode::Op {
            op: OpTag::Mul,
            attrs: OpAttrs::default(),
            operands: vec![
                PatternNode::Op {
                    op: OpTag::Add,
                    attrs: OpAttrs::default(),
                    operands: vec![bind(0), bind(1)],
                },
                bind(0),
            ],
        };
        let id = fuel_graph::runtime_fused::register_runtime_fused(
            "test::advisory_region::mul_add",
            region.clone(),
        )
        .expect("runtime registration");
        // Claimed runtime id (no submitted decompose) -> the registered region.
        assert_eq!(advisory_region(None, Some(id)), Some(region.clone()));
        // A submitted decompose wins over the claimed id (D1 precedence).
        let dec = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![bind(0), bind(1)],
        };
        assert_eq!(advisory_region(Some(&dec), Some(id)), Some(dec.clone()));
        // A STATIC id has no runtime region -> decline.
        assert_eq!(advisory_region(None, Some(FusedOps::ROPE)), None);
        // An unregistered runtime-range id -> decline (never panic).
        let unregistered = FusedOpId(FusedOpId::RUNTIME_FUSED_BASE + 0x7FF0);
        assert_eq!(advisory_region(None, Some(unregistered)), None);
        // Nothing submitted, nothing claimed.
        assert_eq!(advisory_region(None, None), None);
    }

    /// Plan T5: the pre-invoke non-f32 escalate eligibility predicate is the AND
    /// of the dtype gate (F64/F16/BF16 — the kiss-coverable non-f32 floats) and
    /// the adapter's region-support bool (threaded in, so this stays CPU-testable
    /// without the cuda-gated adapter or a device). A kiss-coverable non-f32
    /// candidate escalates (invoke → advisory diff → Inconclusive); an
    /// unsupported region, F32 (verified inline), or any dtype outside the set
    /// keeps the pre-T5 hard Fail.
    #[test]
    fn nonf32_escalate_eligible_gates_on_dtype_and_region_support() {
        use fuel_ir::DType;
        // Coverable non-f32 floats WITH a kiss-supported region -> escalate.
        assert!(nonf32_escalate_eligible(DType::F64, true));
        assert!(nonf32_escalate_eligible(DType::F16, true));
        assert!(nonf32_escalate_eligible(DType::BF16, true));
        // Region kiss cannot cover -> not eligible (hard Fail), every dtype.
        assert!(!nonf32_escalate_eligible(DType::F64, false));
        assert!(!nonf32_escalate_eligible(DType::F16, false));
        assert!(!nonf32_escalate_eligible(DType::BF16, false));
        // F32 is never escalated — its numeric authority IS f32 (verified inline).
        assert!(!nonf32_escalate_eligible(DType::F32, true));
        // A non-f32 dtype outside the kiss-coverable float set stays hard Fail
        // even with a (hypothetically) supported region.
        assert!(!nonf32_escalate_eligible(DType::I32, true));
        assert!(!nonf32_escalate_eligible(DType::F8E4M3, true));
    }

    /// Live-GPU: the kiss-ref advisory cross-check fires for an f32 Add candidate
    /// with a single-primitive Add decompose — `verify_candidate` records a
    /// `kiss_ref_advisory` "pass" (CUDA add_f32 === kiss-ref add, 0 ULP).
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn kiss_ref_advisory_records_for_add_f32() {
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
            entry_point: "test::kiss_advisory::add_f32".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_ADD1,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: None,
        };
        let (_verdict, records) = verify_candidate(&cand, &dev);
        let advisory = records.iter().find(|r| r.claim == "kiss_ref_advisory");
        assert!(
            advisory.is_some(),
            "kiss-ref advisory record must be present for a supported f32 Add"
        );
        assert_eq!(
            advisory.unwrap().result,
            "pass",
            "CUDA add_f32 must match kiss-ref add exactly (0 ULP)"
        );
    }

    /// Live-GPU (T3): the generalized, REGION-based advisory cross-check reaches
    /// a MULTI-NODE decompose. The candidate carries a `relu(add(a, b))` region
    /// (2 op nodes — the pre-T3 single-primitive path could never align it) but
    /// its kernel is plain `add_f32`, so kiss-ref's composed `relu(add)`
    /// reference disagrees on rows where the sum is negative. `verify_candidate`
    /// records a `kiss_ref_advisory` entry with `op_count == 2` and result
    /// `"flag"`. Born-red pre-T3: the old block only recorded for a
    /// single-primitive f32 decompose, so no advisory record was earned at all.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn multi_node_region_advisory_flags_add_kernel_for_relu_add() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        // relu(add(bind0, bind1)) — a 2-op region.
        let decompose = PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        };
        let od = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 32);
        let cand = CandidateKernel {
            entry_point: "test::multi_node_advisory::relu_add".to_string(),
            // Plain add kernel offered for a relu(add) region — it disagrees
            // with the composed reference wherever the sum is negative.
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_4EAD,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: None,
        };
        let (_verdict, records) = verify_candidate(&cand, &dev);
        let advisory = records
            .iter()
            .find(|r| r.claim == "kiss_ref_advisory")
            .expect("a multi-node region must reach the generalized advisory cross-check");
        assert_eq!(
            advisory.evidence["op_count"],
            serde_json::json!(2),
            "advisory ran over the 2-op relu(add) region: {advisory:?}"
        );
        assert_eq!(
            advisory.result, "flag",
            "an add kernel disagrees with kiss-ref's relu(add) reference: {advisory:?}"
        );
    }

    /// Live-GPU (T3): a candidate that carries NO submitted decompose but claims
    /// a runtime-registered op is reached by the advisory cross-check through
    /// Fuel's OWN registered region for that claim (D1 derive-from-registry). The
    /// claimed op's region is a plain `add` and the kernel is `add_f32`, so the
    /// advisory `"pass"`es, with `source == "claimed_recipe"`. Born-red pre-T3:
    /// the old block only consulted `cand.decompose` (here `None`), so a
    /// claimed-op-only candidate earned no advisory record.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn claimed_op_candidate_reaches_advisory_via_runtime_region() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        // Register an `add(bind0, bind1)` region under a runtime FusedOpId; the
        // candidate claims it WITHOUT submitting its own decompose.
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let claimed = fuel_graph::runtime_fused::register_runtime_fused(
            "test::claimed_op_advisory::add",
            region,
        )
        .expect("runtime registration");
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "test::claimed_op_advisory::add_f32".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: None,
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_C1A1,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: Some(claimed),
        };
        let (_verdict, records) = verify_candidate(&cand, &dev);
        let advisory = records
            .iter()
            .find(|r| r.claim == "kiss_ref_advisory")
            .expect("a claimed-op candidate must reach advisory via its runtime region");
        assert_eq!(
            advisory.evidence["source"],
            serde_json::json!("claimed_recipe"),
            "the region came from the claimed op's registered recipe, not a decompose: {advisory:?}"
        );
        assert_eq!(
            advisory.result, "pass",
            "add_f32 matches kiss-ref add exactly (0 ULP): {advisory:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 5 (Increment 1) — recipe-IDENTITY gate (structural, jit-level, NO CUDA).
//
// `recipe_identity_matches` answers one yes/no: does a candidate's submitted
// decompose lower to the SAME primitive base map as FUEL's own registered
// recipe for the op it CLAIMS to implement? It's the cheap, device-free,
// structural pre-check that pairs with the numeric registered-recipe reference
// (`reference_from_registered_recipe`, Task 4) — run FIRST, before any GPU
// work, so a candidate that computes a different function than the op it claims
// is rejected structurally rather than only numerically.
// ---------------------------------------------------------------------------

/// Number of leading input `Op::Const` leaves a region needs so `emit_region`'s
/// positional `inputs[index]` bind resolution can never index out of bounds:
/// one past the largest bind index the region references (0 for a bind-free
/// region).
fn region_arity(region: &fuel_graph::jit::PatternNode) -> usize {
    region.bind_indices().iter().max().map(|m| *m as usize + 1).unwrap_or(0)
}

/// Push `arity` uniform placeholder leaves (`Op::Const`, F32 `[1]`, NO storage)
/// onto `g` and return their ids. Uniform + storage-free is load-bearing: two
/// independently-built graphs' leaves must hash IDENTICALLY under
/// [`base_map_hash`](fuel_graph::opt::base_map_hash) (which folds a const's
/// shape/dtype and silently no-ops on an unpopulated storage slot) for a
/// cross-graph base-map comparison to be meaningful.
fn push_placeholder_leaves(
    g: &mut fuel_graph::Graph,
    arity: usize,
) -> Vec<fuel_graph::NodeId> {
    (0..arity)
        .map(|_| {
            g.push(fuel_graph::Node {
                op: fuel_graph::Op::Const,
                inputs: vec![],
                shape: fuel_ir::Shape::from_dims(&[1]),
                dtype: fuel_ir::DType::F32,
            })
        })
        .collect()
}

/// Lower a `PatternNode` region to its primitive base map on placeholder leaves
/// and return its [`base_map_hash`](fuel_graph::opt::base_map_hash). `None` on
/// any structural failure (a non-re-emittable `OpTag` panics inside
/// `emit_region` — caught by [`recipe_identity_matches`]'s wrapper — a poisoned
/// lock, or an empty lowering result); the caller treats `None` as "not a
/// match" (conservative reject).
fn base_map_hash_of_region(region: &fuel_graph::jit::PatternNode) -> Option<u64> {
    let graph = std::sync::Arc::new(std::sync::RwLock::new(fuel_graph::Graph::new()));
    let sink = {
        let mut g = graph.write().ok()?;
        let inputs = push_placeholder_leaves(&mut g, region_arity(region));
        fuel_graph::runtime_fused::emit_region(&mut g, region, &inputs, &[])
    };
    let roots = fuel_graph::opt::lower_to_base_map(&graph, &[sink]);
    let root = *roots.first()?;
    let g = graph.read().ok()?;
    Some(fuel_graph::opt::base_map_hash(&g, root))
}

/// Lower Fuel's registered recipe for a STATIC `claimed_op` — built as a fresh
/// `Op::Fused(claimed_op, ..)` over `arity` placeholder leaves, dissolved in
/// place by [`lower_to_base_map`](fuel_graph::opt::lower_to_base_map) — to its
/// base map and hash it. Used only when `claimed_op` is NOT a runtime op
/// (runtime ops resolve via their region, the symmetric
/// [`base_map_hash_of_region`] path). `arity` mirrors the submitted region's
/// bind count so the leaf hashes line up; a genuine same-op match has equal
/// arity, a mismatch merely yields a different base map (conservative
/// non-match). `None` on any structural failure.
fn base_map_hash_of_fused(
    claimed_op: fuel_graph::registry::FusedOpId,
    arity: usize,
) -> Option<u64> {
    let graph = std::sync::Arc::new(std::sync::RwLock::new(fuel_graph::Graph::new()));
    let fused = {
        let mut g = graph.write().ok()?;
        let inputs = push_placeholder_leaves(&mut g, arity);
        g.push(fuel_graph::Node {
            op: fuel_graph::Op::Fused(claimed_op, fused_params_for(claimed_op)),
            inputs,
            shape: fuel_ir::Shape::from_dims(&[1]),
            dtype: fuel_ir::DType::F32,
        })
    };
    let roots = fuel_graph::opt::lower_to_base_map(&graph, &[fused]);
    let root = *roots.first()?;
    let g = graph.read().ok()?;
    Some(fuel_graph::opt::base_map_hash(&g, root))
}

/// Structural recipe-identity: does `submitted` lower to the SAME primitive
/// base map as Fuel's registered recipe for `claimed_op`? Both sides are
/// lowered to primitives ([`lower_to_base_map`](fuel_graph::opt::lower_to_base_map))
/// and compared via the `NodeId`-independent
/// [`base_map_hash`](fuel_graph::opt::base_map_hash); equal hash ⇒ same op.
///
/// The registered recipe resolves two ways, BOTH ending in the same
/// lower-then-hash comparison: a RUNTIME-registered op via its `PatternNode`
/// region ([`runtime_region`](fuel_graph::runtime_fused::runtime_region)),
/// emitted exactly like the submitted side (fully symmetric); a STATIC registry
/// op via a fresh `Op::Fused` node dissolved by its registered `decompose`.
///
/// SCOPE (elementwise-now, by design): the submitted side is materialized with
/// `emit_region`, which is elementwise-only today, so this fires only for
/// elementwise-expressible submitted decomposes. A non-elementwise claim (e.g.
/// rope) carries no submittable `PatternNode` decompose, skips this check
/// entirely, and rests on the numeric registered-recipe reference (Task 4/6).
///
/// CONSERVATIVE FALSE + NEVER-PANIC: any inability to resolve / emit / lower /
/// hash EITHER side (a missing region, a non-re-emittable op, a poisoned lock,
/// an arity that would panic `emit`) returns `false` = "not a match". A
/// candidate whose recipe identity cannot be ESTABLISHED must not silently pass
/// the gate; rejecting it is the safe direction. The whole body is
/// `catch_unwind`-wrapped so it never panics even when called directly (no
/// outer guard) from unit tests.
fn recipe_identity_matches(
    claimed_op: fuel_graph::registry::FusedOpId,
    submitted: &fuel_graph::jit::PatternNode,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(submitted_hash) = base_map_hash_of_region(submitted) else {
            return false;
        };
        let registered_hash = match fuel_graph::runtime_fused::runtime_region(claimed_op) {
            Some(region) => base_map_hash_of_region(&region),
            None => base_map_hash_of_fused(claimed_op, region_arity(submitted)),
        };
        registered_hash == Some(submitted_hash)
    }))
    .unwrap_or(false)
}

/// The [`FusedOpParams`](fuel_graph::registry::FusedOpParams) to instantiate an
/// `Op::Fused(id, ..)` with when lowering / realizing Fuel's registered recipe
/// for `id`.
///
/// Increment 1 (the "Rope Oracle") has one static consumer — `FusedOps::ROPE`
/// → `FusedOpParams::Rope`. Runtime-registered ops lower their region
/// INDEPENDENT of the param payload (`runtime_lowering_decompose` ignores it),
/// so the value is irrelevant for them. Every id reachable here today therefore
/// wants `Rope`; a future STATIC op with real params (e.g. RmsNorm's eps) adds
/// a match arm keyed on `id`. Anything unmapped falling through to `Rope` at
/// worst yields a non-matching base map or a wrong realize that fails the
/// numeric bound — a conservative reject, never a wrong adopt.
fn fused_params_for(
    id: fuel_graph::registry::FusedOpId,
) -> fuel_graph::registry::FusedOpParams {
    // No per-id branch needed yet — see doc above. `id` is bound for the
    // future match arm and to document intent.
    let _ = id;
    fuel_graph::registry::FusedOpParams::Rope
}

/// Task-6 carry-forward guard predicate: does a candidate CLAIM an op
/// (`claimed_op.is_some()`) while declaring NO numeric bound (`max_ulp` /
/// `max_relative` / `max_absolute` all `None`)?
///
/// This is the latent-bypass check that pairs with `verify_candidate_impl`'s
/// claimed-op gates. Those gates (the structural recipe-identity pre-check and
/// the numeric registered-recipe reference) both live INSIDE the
/// `numeric_declared` block — so a `claimed_op = Some` candidate that declares
/// no numeric bound would skip both and fall through to the trailing `Pass`,
/// adopting a claimant whose OP IDENTITY was never checked against Fuel's
/// registered recipe. Bit-stability alone doesn't rescue it: it only proves
/// the kernel is DETERMINISTIC, never that it computes the CLAIMED op (that
/// needs a numeric comparison against the reference realized from Fuel's
/// registered recipe) — so `bit_stable_on_same_hardware` is deliberately NOT
/// an exemption here, unlike the (retired) all-guarantees version of this
/// guard. A claimed-op candidate must declare a numeric bound; if it declares
/// none, `verify_candidate_impl` refuses it up front (`Fail { claim:
/// "no_guarantee" }`). Device-free and NOT `cuda`-gated so the guard is
/// unit-testable under `--features jit` alone.
// The sole non-test caller (`verify_candidate_impl`) is `cuda`-gated; in a
// non-`cuda` build only the `jit` unit test uses it, so silence dead-code there.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn claimed_op_lacks_numeric_bound(
    claimed_op: Option<fuel_graph::registry::FusedOpId>,
    declared: &crate::fused::PrecisionGuarantee,
) -> bool {
    claimed_op.is_some()
        && declared.max_ulp.is_none()
        && declared.max_relative.is_none()
        && declared.max_absolute.is_none()
}

/// The exact input arity Fuel's registered recipe for `id` positionally indexes
/// its probe by — so a candidate whose operand count wouldn't satisfy the
/// recipe's `decompose` is refused (as `probe_arity`) BEFORE it panics indexing
/// a short `Vec`. `Some(n)` for ops known here (Increment 1: ROPE = 3 → x, cos,
/// sin); `None` = unknown → the caller falls back to the candidate's own
/// operand count. Cuda-gated: only the numeric reference path (which needs a
/// live device) consults it.
#[cfg(feature = "cuda")]
fn expected_input_arity(id: fuel_graph::registry::FusedOpId) -> Option<usize> {
    if id == fuel_graph::registry::FusedOps::ROPE {
        Some(3)
    } else {
        None
    }
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
// Un-gated (the enum itself is cuda-independent; only its cuda producers in
// `verify_candidate_impl` stay gated) so the CPU-testable `classify_floor_verdict`
// / `outcome_from_nonadopt_verdict` can build + match it without a device.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug)]
pub enum VerifyVerdict {
    /// Every DECLARED claim was empirically backed.
    Pass,
    /// A claim failed (or its reference couldn't be produced). `claim` is the
    /// stage/claim id: `"probe"` / `"invoke"` / `"bit_stable_on_same_hardware"`
    /// / `"max_ulp"` / `"max_relative"` / `"max_absolute"` / `"panic"`.
    Fail { claim: &'static str, detail: String },
    /// No authoritative verdict was possible (only a non-authoritative live
    /// reference was available, or the reference realize failed): escalate,
    /// neither Adopt nor Reject. §6.6-0007 (kiss-ref flags, never verdicts).
    Inconclusive { claim: &'static str, detail: String },
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

/// Dtype-dispatched region differential for the advisory cross-check (design
/// D6): decode the candidate output bytes and each probe input column into
/// `out_dtype`'s scalar lattice, then diff the candidate against kiss-ref's
/// composed-region reference under `tol`. Delegates per dtype to the (cuda-only)
/// adapter's `diff_region_{f32,f64,f16,bf16}`. `UnsupportedDtype` for a dtype
/// outside {F32,F64,F16,BF16}; the caller gates on that set (and on
/// `region_supported`) first, so that arm is unreachable in the live path — it
/// is kept total for never-panic. Probe columns are matched positionally to the
/// region's `Bind` indices, exactly as the synthesized probe is 1:1 with the
/// candidate's operands.
#[cfg(feature = "cuda")]
fn run_region_diff(
    region: &fuel_graph::jit::PatternNode,
    out_dtype: fuel_ir::DType,
    cand_bytes: &[u8],
    probe: &ProbeInputs,
    tol: fuel_kiss_ref_backend::Tolerance,
) -> std::result::Result<fuel_kiss_ref_backend::DiffReport, fuel_kiss_ref_backend::KissRefError> {
    use fuel_ir::DType;
    macro_rules! dispatch {
        ($decode:path, $diff:path) => {{
            let cand = $decode(cand_bytes);
            let cols: Vec<_> = probe.iter().map(|t| $decode(&t.bytes)).collect();
            let refs: Vec<_> = cols.iter().map(|v| v.as_slice()).collect();
            $diff(region, &cand, &refs, tol)
        }};
    }
    match out_dtype {
        DType::F32 => dispatch!(bytes_to_f32, fuel_kiss_ref_backend::diff_region_f32),
        DType::F64 => dispatch!(bytes_to_f64, fuel_kiss_ref_backend::diff_region_f64),
        DType::F16 => dispatch!(bytes_to_f16, fuel_kiss_ref_backend::diff_region_f16),
        DType::BF16 => dispatch!(bytes_to_bf16, fuel_kiss_ref_backend::diff_region_bf16),
        other => Err(fuel_kiss_ref_backend::KissRefError::UnsupportedDtype(other)),
    }
}

/// Resolve a numeric-region reference-realize failure (or absent reference)
/// into a verdict (Task 4, design D4). Fuel's interim numeric authority — the
/// recipe reference realized from the registered recipe / the candidate's own
/// decompose — is unusable at the call site. But if the advisory kiss-ref
/// cross-check already produced a live (non-authoritative) [`DiffOutcome`],
/// there IS a reference to compare against, so [`classify_floor_verdict`]
/// escalates to [`VerifyVerdict::Inconclusive`] (flag, never verdict —
/// §6.6-0007) rather than hard-Fail: the candidate is neither adopted
/// (kiss agreement ≠ adopt) nor rejected (kiss discrepancy ≠ reject), it
/// escalates to mint a corpus vector.
///
/// With NO kiss outcome present, `classify_floor_verdict(None, None, None)`
/// yields the `no_reference` Fail, which this maps back to today's EXACT
/// realize-failed Fail (`fail_claim` + `fail_detail` — a numeric-claim
/// rejection naming the realize error): the no-advisory path is byte-for-byte
/// unchanged from the pre-Task-4 arms it replaces.
#[cfg(feature = "cuda")]
fn resolve_reference_failure(
    kiss_outcome: Option<&DiffOutcome>,
    fail_claim: &'static str,
    fail_detail: String,
    ledger: &mut VerificationLedger,
    make_record: &dyn Fn(&str, &str, serde_json::Value) -> LedgerRecord,
) -> (VerifyVerdict, Vec<LedgerRecord>) {
    match classify_floor_verdict(kiss_outcome, None, None) {
        VerifyVerdict::Inconclusive { claim, detail } => {
            ledger.upsert(make_record(
                claim,
                "inconclusive",
                serde_json::json!({
                    "detail": detail.clone(),
                    "realize_error": fail_detail,
                    "note": "recipe reference unusable; kiss-ref advisory escalates \
                             (flag, never verdict — §6.6-0007)",
                }),
            ));
            (VerifyVerdict::Inconclusive { claim, detail }, ledger.records().to_vec())
        }
        // kiss absent: classify returns the `no_reference` Fail; keep today's
        // exact realize-failed Fail (same claim + detail + ledger record as the
        // pre-Task-4 arm) so the no-advisory path is unchanged.
        _ => {
            ledger.upsert(make_record(
                fail_claim,
                "fail",
                serde_json::json!({ "detail": fail_detail.clone() }),
            ));
            (
                VerifyVerdict::Fail { claim: fail_claim, detail: fail_detail },
                ledger.records().to_vec(),
            )
        }
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

/// The claimed-op STRUCTURAL pre-checks shared by the f32 numeric registered-
/// recipe reference and the Plan-T5 non-f32 escalate path. A submitted decompose
/// whose primitive base map differs from Fuel's registered recipe for `claimed`
/// is simply not the same op (`recipe_identity` Fail); a probe whose arity
/// doesn't match the claimed op's expected input count would index-panic the
/// recipe realize (`probe_arity` Fail). Both are genuine op-identity rejections
/// kiss-ref cannot rescue, so they gate BEFORE any escalate — a wrong-identity
/// non-f32 claimant hard-Fails here rather than escalating on the advisory.
/// `Ok(())` when the candidate claims no op, submits no decompose to
/// identity-check, or passes both checks; `Err((verdict, records))` is a
/// ready-to-return early hard Fail whose ledger record is already upserted.
#[cfg(feature = "cuda")]
fn claimed_op_structural_gate(
    cand: &CandidateKernel,
    probe: &ProbeInputs,
    ledger: &mut VerificationLedger,
    make_record: &dyn Fn(&str, &str, serde_json::Value) -> LedgerRecord,
) -> Result<(), (VerifyVerdict, Vec<LedgerRecord>)> {
    let Some(claimed) = cand.claimed_op else {
        return Ok(());
    };
    // (i) Structural recipe-identity pre-check — only fires when the candidate
    // carries a submitted (elementwise-expressible) decompose. A submitted recipe
    // whose primitive base map differs from Fuel's registered recipe for
    // `claimed` is not the same op; reject before spending any GPU work.
    if let Some(dec) = &cand.decompose {
        if !recipe_identity_matches(claimed, dec) {
            let detail = "submitted recipe's base map differs from Fuel's registered \
                 recipe for the claimed op — not the same op"
                .to_string();
            ledger.upsert(make_record(
                "recipe_identity",
                "fail",
                serde_json::json!({ "detail": detail.clone() }),
            ));
            return Err((
                VerifyVerdict::Fail { claim: "recipe_identity", detail },
                ledger.records().to_vec(),
            ));
        }
    }

    // (ii) Probe-arity guard: the registered recipe's `decompose` indexes the
    // probe POSITIONALLY, so a probe whose length doesn't match the claimed op's
    // expected input count would panic that `Vec` indexing (caught by the outer
    // `catch_unwind` and surfaced as a generic `panic`, but an explicit named
    // refusal is cleaner). Expected count comes from the claimed op when known
    // (Increment 1: ROPE = 3); otherwise the candidate's own operand count —
    // probe synthesis is 1:1 with operands, so that fallback only guards the
    // empty-probe degenerate (no per-op arity table exists in the registry).
    let expected_inputs = expected_input_arity(claimed).unwrap_or(cand.operands.len());
    if probe.len() != expected_inputs {
        let detail = format!(
            "probe arity {} does not match the claimed op's expected input count {}",
            probe.len(),
            expected_inputs
        );
        ledger.upsert(make_record(
            "probe_arity",
            "fail",
            serde_json::json!({ "detail": detail.clone() }),
        ));
        return Err((
            VerifyVerdict::Fail { claim: "probe_arity", detail },
            ledger.records().to_vec(),
        ));
    }
    Ok(())
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

    // (0) Claimed-op-lacks-numeric-bound guard (Task-6 carry-forward). Every
    // claimed-op gate below (recipe-identity + the numeric registered-recipe
    // reference) lives inside `if numeric_declared` — so a `claimed_op = Some`
    // candidate declaring NO numeric bound would skip both gates and reach the
    // trailing `Pass`, adopting a claimant whose op identity was never checked
    // against Fuel's registered recipe. Bit-stability alone does NOT rescue
    // it: it only proves the kernel is deterministic, never that it computes
    // the CLAIMED op (that needs a numeric comparison against the reference
    // realized from Fuel's registered recipe). Refuse it before any probe/GPU
    // work: a claimed-op candidate MUST declare a numeric bound.
    if claimed_op_lacks_numeric_bound(cand.claimed_op, &cand.declared) {
        let detail = "a claimed-op candidate must declare a numeric bound (max_ulp/max_relative/\
             max_absolute) to verify against the registered recipe"
            .to_string();
        ledger.upsert(make_record(
            "no_guarantee",
            "fail",
            serde_json::json!({ "detail": detail.clone() }),
        ));
        return (VerifyVerdict::Fail { claim: "no_guarantee", detail }, ledger.records().to_vec());
    }

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

    // Numeric claims (checked ahead of the invoke below, for the escalate-or-
    // fail guard right after) — the same declared-claim test the numeric block
    // uses further down.
    let numeric_declared = cand.declared.max_ulp.is_some()
        || cand.declared.max_relative.is_some()
        || cand.declared.max_absolute.is_some();

    // The advisory region (design D1, derive-from-registry): the candidate's own
    // submitted `decompose`, else Fuel's OWN registered region for a
    // runtime-registered `claimed` op. Bound HERE (pre-invoke) so the T5 guard
    // just below can test kiss-ref coverage for a non-f32 candidate; the advisory
    // diff further down reuses this SAME binding (no re-derivation).
    let advisory = advisory_region(cand.decompose.as_ref(), cand.claimed_op);

    // (1a) Non-f32 numeric-claim guard (Plan T5 — escalate OR fail).
    // `verify_precision_bound` (`ulp.rs`) unconditionally reinterprets BOTH
    // outputs' raw bytes as `f32` (`bytemuck::cast_slice`) — for a non-F32
    // `out_dtype` (BF16/F16/F64/...) that reads the wrong element count/values
    // from bytes that were never f32, so a kernel computing the WRONG function
    // could land within `max_ulp` of the reinterpreted reference by accident and
    // wrongly PASS — exactly the "wrong candidate adopted" defect this module
    // exists to prevent. A non-f32 numeric claim is therefore NEVER checked
    // through that f32 path; the two honest outcomes are:
    //   * kiss-COVERABLE (`nonf32_escalate_eligible`: dtype ∈ {F64,F16,BF16} ∧
    //     the advisory region is kiss-supported) ⇒ fall through: invoke, run the
    //     kiss-ref advisory diff, and ESCALATE to `Inconclusive` in the numeric
    //     region below (flag, never verdict — §6.6-0007), the live kiss-ref diff
    //     standing in as the only (non-authoritative) reference.
    //   * kiss-UNCOVERABLE (any other non-f32 cell — unmapped op/dtype,
    //     non-default attrs, or no advisory region) ⇒ the EXACT pre-T5 hard Fail,
    //     same claim + same detail bytes: nothing can verify it yet, so refuse
    //     honestly before any GPU work (short-circuits the invoke + bit-stability
    //     too, matching the first-failure-wins posture of the probe checks).
    if numeric_declared && out_dtype != fuel_ir::DType::F32 {
        let region_kiss_supported = advisory
            .as_ref()
            .is_some_and(|r| fuel_kiss_ref_backend::region_supported(r, out_dtype));
        if !nonf32_escalate_eligible(out_dtype, region_kiss_supported) {
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
        // Eligible: fall through to invoke + the advisory diff; the numeric
        // region escalates to Inconclusive rather than checking an f32 bound.
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

    // kiss-ref advisory cross-check (§6.6-0007: kiss-ref FLAGS, never verdicts).
    // GENERALIZED (T3) from the shipped single-primitive f32 path to arbitrary
    // elementwise regions, all four float dtypes, and claimed-op reach:
    //   * REGION derivation (D1): the candidate's own submitted `decompose`, else
    //     — for a runtime-registered `claimed` op with no decompose — Fuel's OWN
    //     registered region for that claim (`advisory_region`);
    //   * DTYPE dispatch (D6): F32/F64/F16/BF16 via `run_region_diff`, gated by
    //     the adapter's `region_supported(region, out_dtype)` (every op node
    //     mapped, default attrs, ≥1 op node);
    //   * BAND (D3, kiss-ref refinement): `advisory_ulp_band` — single exact op →
    //     Exact; multi-node exact → Ulp(n−1); transcendental region → Ulp(Σ
    //     per-op §6.8 ceilings + (n_exact−1)); raw `max_ulp` always recorded.
    // Diff the candidate output against kiss-ref's INDEPENDENT composed-region
    // reference and record an advisory ledger entry. This does NOT gate the
    // verdict — recipe-realize stays the interim authority until Fuel consumes a
    // corpus; kiss-ref is the independent Fuel-floor-vs-spec drift detector.
    //
    // NON-REGRESSION: an f32 single-primitive `Add` decompose derives
    // region=Add (op_count 1), band=None ⇒ Exact, run_region_diff→diff_region_f32
    // over the same kiss add — byte/ledger-identical to the shipped path (result
    // "pass" at 0 ULP; one `kiss_ref_advisory` record via `upsert`).
    // `advisory` was derived pre-invoke (above the T5 non-f32 guard); reuse it.
    // `kiss_outcome` is threaded into the numeric-claim verdict below (Task 4/5
    // consult it as the advisory input to `classify_floor_verdict`: escalate,
    // never gate — §6.6-0007). Bound here so T3 owns the one construction site.
    #[allow(unused_variables)]
    let kiss_outcome: Option<DiffOutcome> = advisory.as_ref().and_then(|region| {
        let dtype_supported = matches!(
            out_dtype,
            fuel_ir::DType::F32 | fuel_ir::DType::F64 | fuel_ir::DType::F16 | fuel_ir::DType::BF16
        );
        if !dtype_supported || !fuel_kiss_ref_backend::region_supported(region, out_dtype) {
            return None;
        }
        let band = advisory_ulp_band(region);
        let tol = band
            .map_or(fuel_kiss_ref_backend::Tolerance::Exact, fuel_kiss_ref_backend::Tolerance::Ulp);
        let report = run_region_diff(region, out_dtype, &cand_out.bytes, &probe, tol).ok()?;
        let op_count = fuel_kiss_ref_backend::region_op_count(region);
        let source = if cand.decompose.is_some() { "decompose" } else { "claimed_recipe" };
        ledger.upsert(make_record(
            "kiss_ref_advisory",
            if report.conforms() { "pass" } else { "flag" },
            serde_json::json!({
                "dtype": format!("{out_dtype:?}"),
                "op_count": op_count,
                "max_ulp": report.max_ulp,
                "mismatches": report.mismatches,
                "advisory_band_ulp": band,
                "source": source,
                "note": "advisory only; kiss-ref flags, never verdicts (§6.6-0007)"
            }),
        ));
        Some(DiffOutcome {
            within: report.conforms(),
            max_ulp: Some(report.max_ulp),
            detail: format!(
                "kiss-ref region diff (op_count={op_count}, max_ulp={}, mismatches={})",
                report.max_ulp, report.mismatches
            ),
        })
    });

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

    // (4)+(5) Numeric claims. These need a REFERENCE. Resolution (Task-5):
    //   - `claimed_op.is_some()` → verify against FUEL's REGISTERED recipe for
    //     the claimed op (`reference_from_registered_recipe`) — the candidate
    //     is checked against what Fuel says the op computes, not against its
    //     own (possibly-wrong) decompose. When the candidate ALSO carries a
    //     submitted decompose, a structural recipe-identity pre-check
    //     (`recipe_identity_matches`) first confirms it lowers to the SAME
    //     primitive base map as Fuel's recipe (else it's simply not the same
    //     op — a `recipe_identity` fail, before any GPU work).
    //   - `claimed_op.is_none()` → the UNCHANGED Spec B path: realize the
    //     candidate's OWN `decompose` on the same probe (`reference_output`);
    //     no decompose ⇒ no reference ⇒ the declared numeric claim fails
    //     honestly (bit-stability above stays checkable). This branch is
    //     defensive — Task 6 requires a decompose to adopt at all.
    if numeric_declared {
        // (D5) Corpus precedence. The frozen golden corpus is the SOLE verdict
        // authority (§6.6-0007), so consult it FIRST, ahead of the recipe
        // reference. A single-primitive region maps to the `OpTag` the corpus
        // keys on; a multi-node region has no single-op corpus cell
        // (`single_primitive_optag` → None → corpus None). `corpus_verdict` is
        // DORMANT today (always `None` — see its doc), so this block is
        // wired-but-inert: the PRECEDENCE is live for when Fuel consumes a
        // populated corpus, at which point a covering cell short-circuits the
        // recipe reference below via `classify_floor_verdict(None, None,
        // corpus)` — corpus adopt ⇒ Pass, corpus non-adopt ⇒ Fail. The kiss-ref
        // advisory record (if any) was already upserted above; corpus does not
        // consult it (kiss-ref never verdicts).
        let corpus = advisory
            .as_ref()
            .and_then(|region| single_primitive_optag(Some(region)))
            .and_then(|op| corpus_verdict(op, out_dtype, seed));
        if let Some(c) = &corpus {
            ledger.upsert(make_record(
                c.claim,
                if c.adopt { "pass" } else { "fail" },
                serde_json::json!({ "detail": c.detail.clone(), "source": "corpus" }),
            ));
            return (classify_floor_verdict(None, None, Some(c)), ledger.records().to_vec());
        }

        // (T5) Non-f32 numeric-claim ESCALATE. Reached only for a kiss-coverable
        // non-f32 candidate the (1a) guard let through (dtype ∈ {F64,F16,BF16},
        // advisory region kiss-supported). Fuel's numeric authority
        // (`verify_precision_bound`) is f32-only, so there is NO usable
        // recipe-realize reference here: realizing a non-f32 reference and then
        // f32-reinterpreting it is exactly the unsafe mis-compare the (1a) guard
        // exists to prevent. But the claimed-op STRUCTURAL gates still apply (a
        // wrong-identity claimant is a real rejection kiss-ref cannot rescue),
        // and the live kiss-ref advisory IS a non-authoritative reference — so
        // after those gates this ESCALATES to Inconclusive rather than the pre-T5
        // hard Fail (§6.6-0007: flag, never verdict). `corpus` was consulted-and-
        // None just above, so `resolve_reference_failure`'s
        // `classify_floor_verdict(kiss, None, None)` is identical to the plan's
        // `(kiss, None, corpus)` here: Inconclusive when the advisory diff ran,
        // else the honest f32-only Fail (the rare case the advisory declined
        // despite eligibility — e.g. a `run_region_diff` eval error).
        if out_dtype != fuel_ir::DType::F32 {
            if let Err(early) =
                claimed_op_structural_gate(cand, &probe, &mut ledger, &make_record)
            {
                return early;
            }
            let claim = first_numeric_claim(&cand.declared);
            let detail = "numeric bound verification is f32-only; non-f32 candidate escalated \
                 to the kiss-ref advisory (flag, never verdict — §6.6-0007)"
                .to_string();
            return resolve_reference_failure(
                kiss_outcome.as_ref(),
                claim,
                detail,
                &mut ledger,
                &make_record,
            );
        }

        let reference = match cand.claimed_op {
            Some(claimed) => {
                // (i)+(ii) Claimed-op STRUCTURAL gates (recipe-identity +
                // probe-arity), shared with the T5 non-f32 escalate path via
                // `claimed_op_structural_gate`. A wrong-identity or wrong-arity
                // claimant is a genuine rejection, refused BEFORE any GPU work.
                if let Err(early) =
                    claimed_op_structural_gate(cand, &probe, &mut ledger, &make_record)
                {
                    return early;
                }

                // (iii) Reference = FUEL's registered recipe for `claimed`,
                // lowered to primitives and realized on the same probe.
                let params = fused_params_for(claimed);
                match crate::jit_ingest_probe::reference_from_registered_recipe(
                    claimed,
                    &params,
                    &probe,
                    out_dtype,
                    out_shape.clone(),
                    device,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        // (D4) The registered-recipe reference is unusable.
                        // Escalate to Inconclusive when the advisory kiss-ref
                        // diff ran (a live non-authoritative reference exists);
                        // else the same Fail as today.
                        let claim = first_numeric_claim(&cand.declared);
                        let detail =
                            format!("reference realize from registered recipe failed: {e:?}");
                        return resolve_reference_failure(
                            kiss_outcome.as_ref(),
                            claim,
                            detail,
                            &mut ledger,
                            &make_record,
                        );
                    }
                }
            }
            None => match &cand.decompose {
                Some(dec) => match crate::jit_ingest_probe::reference_output(
                    dec,
                    &probe,
                    out_dtype,
                    out_shape.clone(),
                    device,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        // (D4) Candidate's own decompose reference is unusable —
                        // escalate to Inconclusive when the advisory diff ran,
                        // else the same Fail as today.
                        let claim = first_numeric_claim(&cand.declared);
                        let detail = format!("reference realize from decompose failed: {e:?}");
                        return resolve_reference_failure(
                            kiss_outcome.as_ref(),
                            claim,
                            detail,
                            &mut ledger,
                            &make_record,
                        );
                    }
                },
                None => {
                    // (D4) No reference of any kind. If the advisory kiss-ref
                    // diff ran (e.g. a claimed-op region derived from the
                    // registry), escalate to Inconclusive; else the same Fail
                    // as today.
                    let claim = first_numeric_claim(&cand.declared);
                    let detail =
                        "no decompose: cannot verify numeric claim against a reference".to_string();
                    return resolve_reference_failure(
                        kiss_outcome.as_ref(),
                        claim,
                        detail,
                        &mut ledger,
                        &make_record,
                    );
                }
            },
        };

        // Happy path (bound checks below): a usable recipe reference WAS
        // produced. This is exactly `classify_floor_verdict` arm (2) — corpus
        // already consulted-and-None above, recipe-realize is the interim
        // authority, and the kiss-ref advisory (if any) is recorded but does
        // NOT gate (§6.6-0007). The bound checks realize that arm inline: each
        // declared numeric bound is the recipe verdict; a pass ⇒ Pass, the
        // first out-of-bound ⇒ Fail. Kept inline (not routed through
        // `classify_floor_verdict`) so the per-bound ledger evidence + the
        // transcendental band-widening stay first-class.

        // Transcendental band-widening (KISS, 2026-07-18): kiss-ref and Fuel's
        // CPU oracle are BOTH hardware-precision (§6.5-0007), so on this LIVE
        // candidate-vs-reference path a transcendental-containing region gets
        // ~2× the declared ULP ceiling — two impls each within the ceiling of
        // the wide-precision truth can differ from each other by up to twice
        // it. Tight transcendental truth lives in the frozen wide-precision
        // corpus, not here. Non-transcendental regions keep the tight bound.
        let transcendental =
            cand.decompose.as_ref().is_some_and(|r| region_contains_transcendental(r));
        let widen = |b: Bound| if transcendental { widen_bound_for_transcendental(b) } else { b };

        // Check each declared numeric bound in order; FIRST failure returns.
        if let Some(b) = cand.declared.max_ulp {
            match check_numeric_bound(&cand_out, &reference, &entry, &probe, widen(Bound::MaxUlp(b))) {
                Ok(()) => ledger.upsert(make_record(
                    "max_ulp",
                    "pass",
                    serde_json::json!({ "bound": b, "transcendental_band": transcendental }),
                )),
                Err(detail) => {
                    ledger.upsert(make_record(
                        "max_ulp",
                        "fail",
                        serde_json::json!({ "detail": detail.clone(), "bound": b, "transcendental_band": transcendental }),
                    ));
                    return (VerifyVerdict::Fail { claim: "max_ulp", detail }, ledger.records().to_vec());
                }
            }
        }
        if let Some(b) = cand.declared.max_relative {
            match check_numeric_bound(&cand_out, &reference, &entry, &probe, widen(Bound::MaxRelative(b))) {
                Ok(()) => ledger.upsert(make_record(
                    "max_relative",
                    "pass",
                    serde_json::json!({ "bound": b, "transcendental_band": transcendental }),
                )),
                Err(detail) => {
                    ledger.upsert(make_record(
                        "max_relative",
                        "fail",
                        serde_json::json!({ "detail": detail.clone(), "bound": b, "transcendental_band": transcendental }),
                    ));
                    return (
                        VerifyVerdict::Fail { claim: "max_relative", detail },
                        ledger.records().to_vec(),
                    );
                }
            }
        }
        if let Some(b) = cand.declared.max_absolute {
            match check_numeric_bound(&cand_out, &reference, &entry, &probe, widen(Bound::MaxAbsolute(b))) {
                Ok(()) => ledger.upsert(make_record(
                    "max_absolute",
                    "pass",
                    serde_json::json!({ "bound": b, "transcendental_band": transcendental }),
                )),
                Err(detail) => {
                    ledger.upsert(make_record(
                        "max_absolute",
                        "fail",
                        serde_json::json!({ "detail": detail.clone(), "bound": b, "transcendental_band": transcendental }),
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
        // Fail ⇒ Rejected, Inconclusive ⇒ Flagged(escalate) — the pure mapper
        // handles both (and finds the earned ledger record for a Fail claim).
        other => outcome_from_nonadopt_verdict(other, records, &cand.entry_point),
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
            IngestOutcome::Flagged(report) => {
                if let Some(fb) = &feedback {
                    fb.on_flagged(&report);
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

    /// Task 3: `CandidateKernel.claimed_op` — the op-identity a candidate
    /// asserts it implements. `Some(id)` round-trips; later tasks (4/5) use
    /// this to resolve Fuel's registered recipe as the verification
    /// reference instead of the candidate's own `decompose`. No CUDA/GPU
    /// needed — this only exercises the struct field, so the kernel fn-ptr
    /// is `dummy_kernel` (the same no-op used by the Task-7
    /// `IngestionService` tests below, which run under `--features jit`
    /// alone) rather than a real cuda-gated kernel.
    #[test]
    fn candidate_kernel_carries_claimed_op() {
        let c = CandidateKernel {
            entry_point: "k".into(),
            kernel: dummy_kernel,
            op_params: crate::kernel::OpParams::None,
            decompose: None,
            operands: vec![],
            dtypes: vec![],
            kernel_revision_hash: 0,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: fuel_ir::probe::BackendId::Cuda,
            claimed_op: Some(fuel_graph::registry::FusedOps::ROPE),
        };
        assert_eq!(c.claimed_op, Some(fuel_graph::registry::FusedOps::ROPE));
    }

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

    /// Task 6 (Increment 1) carry-forward — the claimed-op-lacks-numeric-bound
    /// guard, structural + device-free (`--features jit` alone). Both
    /// claimed-op verification gates (recipe-identity + the numeric
    /// registered-recipe reference) live INSIDE `if numeric_declared` in
    /// `verify_candidate_impl`, so a `claimed_op = Some` candidate declaring NO
    /// numeric bound (`max_ulp`/`max_relative`/`max_absolute` all `None`) would
    /// slip both gates and reach the trailing `Pass` — silently adopting a
    /// claimant whose op identity was never checked. Declaring
    /// `bit_stable_on_same_hardware` alone does NOT rescue it: bit-stability
    /// only proves the kernel is deterministic, never that it computes the
    /// CLAIMED op — verifying THAT requires a numeric comparison against the
    /// reference realized from Fuel's registered recipe.
    /// `claimed_op_lacks_numeric_bound` is the predicate that closes that
    /// bypass (→ `Fail { claim: "no_guarantee" }` in `verify_candidate_impl`);
    /// this exercises it directly with no CUDA device (the guard fires before
    /// any probe/invoke/GPU work).
    #[test]
    fn claimed_op_without_a_numeric_bound_is_refused() {
        use fuel_graph::registry::FusedOps;

        // The bypass case: claims ROPE but declares NO numeric bound (even
        // though it claims bit-stability — determinism isn't op identity).
        let empty = crate::fused::PrecisionGuarantee {
            bit_stable_on_same_hardware: true,
            max_ulp: None,
            max_relative: None,
            max_absolute: None,
            notes: "test-only: bit-stable claim but no numeric bound",
        };
        assert!(
            claimed_op_lacks_numeric_bound(Some(FusedOps::ROPE), &empty),
            "a claimed-op candidate declaring only bit-stability (no numeric bound) must be refused"
        );

        // Any single numeric claim rescues it (nothing to bypass).
        assert!(!claimed_op_lacks_numeric_bound(
            Some(FusedOps::ROPE),
            &crate::fused::PrecisionGuarantee { max_ulp: Some(1), ..empty }
        ));
        assert!(!claimed_op_lacks_numeric_bound(
            Some(FusedOps::ROPE),
            &crate::fused::PrecisionGuarantee { max_relative: Some(1e-3), ..empty }
        ));
        assert!(!claimed_op_lacks_numeric_bound(
            Some(FusedOps::ROPE),
            &crate::fused::PrecisionGuarantee { max_absolute: Some(1e-3), ..empty }
        ));

        // No claimed op → not this guard's concern (the `claimed_op = None`
        // Spec-B path verifies against the candidate's own decompose).
        assert!(!claimed_op_lacks_numeric_bound(None, &empty));
    }

    /// Task 5 (Increment 1) — recipe-IDENTITY gate, structural, NO GPU
    /// (`--features jit` alone). Register a small elementwise `Add` region as a
    /// known runtime op, then assert `recipe_identity_matches`:
    ///   - TRUE for the SAME region (the submitted decompose that IS the
    ///     registered recipe lowers to the same primitive base map), and
    ///   - FALSE for a `Mul` region (a different function → a different base
    ///     map → not the same op).
    /// This exercises the genuine lower-both-and-compare path (emit → lower to
    /// base map → `base_map_hash`), not a stub.
    #[test]
    fn recipe_identity_rejects_a_mismatched_submitted_decompose() {
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_graph::runtime_fused::register_runtime_fused;

        let add_region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let mul_region = PatternNode::Op {
            op: OpTag::Mul,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };

        // Register the Add region → a known claimed id whose registered recipe
        // is exactly `add_region`.
        let claimed_id =
            register_runtime_fused("test::recipe_identity::add", add_region.clone())
                .expect("register add region");

        assert!(
            recipe_identity_matches(claimed_id, &add_region),
            "the submitted decompose that IS the registered recipe must match"
        );
        assert!(
            !recipe_identity_matches(claimed_id, &mul_region),
            "a Mul-region decompose must NOT match an Add-region registered recipe"
        );
    }

    /// Spec-B Task-5 acceptance (live GPU): a candidate whose kernel is the
    /// CUDA `add_f32` wrapper, carrying a 2-input `Add` decompose and matching
    /// F32 operands, verifies `Pass` — its output is byte-identical to the
    /// reference realized from the decompose (elementwise sum), so it's
    /// bit-stable AND meets every `PrecisionGuarantee::REFERENCE` numeric bound
    /// (0 ULP / 0 relative / 0 absolute). Earns one `pass` `LedgerRecord` per
    /// declared claim (bit_stable + the 3 numeric bounds = 4) PLUS the
    /// `kiss_ref_advisory` record (f32 single-primitive Add is on the kiss-ref
    /// floor, so the advisory cross-check fires — see
    /// `kiss_ref_advisory_records_for_add_f32`). `#[ignore]`'d (needs a live
    /// CUDA device). Candidate/probe construction mirrors
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
            claimed_op: None,
        };

        let (verdict, records) = verify_candidate(&cand, &dev);
        assert!(matches!(verdict, VerifyVerdict::Pass), "expected Pass, got {verdict:?}");
        // REFERENCE declares 4 machine-checkable claims → 4 pass records,
        // plus the kiss-ref advisory record (f32 floor Add) = 5.
        assert_eq!(
            records.len(),
            5,
            "one pass record per declared claim + the kiss-ref advisory: {records:?}"
        );
        assert!(records.iter().all(|r| r.result == "pass"), "all pass: {records:?}");
        assert!(records.iter().all(|r| r.backend == "Cuda"));
        assert!(records.iter().all(|r| r.kernel_revision_hash == 0x00AD_DF32));
        assert!(records.iter().any(|r| r.claim == "kiss_ref_advisory"));
        assert!(records.iter().any(|r| r.claim == "bit_stable_on_same_hardware"));
        assert!(records.iter().any(|r| r.claim == "max_ulp"));
        assert!(records.iter().any(|r| r.claim == "max_relative"));
        assert!(records.iter().any(|r| r.claim == "max_absolute"));
    }

    /// Review-fix regression, NARROWED by Plan T5: `verify_precision_bound`
    /// (`ulp.rs`) unconditionally reinterprets output bytes as `f32`
    /// (`bytemuck::cast_slice::<f32>`), so a candidate whose real output dtype is
    /// NOT f32 must never reach that comparison — it would silently reinterpret
    /// non-f32 bytes as f32 and could wrongly PASS a wrong-function kernel. T5
    /// splits the non-f32 guard: a kiss-COVERABLE non-f32 candidate now ESCALATES
    /// (see `verify_candidate_add_f64_is_inconclusive_not_failed`); a kiss-
    /// UNCOVERABLE non-f32 candidate keeps the honest f32-only hard `Fail`. This
    /// test pins that retained hard-Fail arm: a BF16 candidate whose advisory
    /// region is an UNMAPPED op (`MatMul`) that kiss-ref cannot evaluate
    /// (`region_supported` declines) ⇒ `nonf32_escalate_eligible` is false ⇒
    /// pre-invoke `Fail` naming the f32-only limitation, same claim + detail
    /// bytes as pre-T5. `#[ignore]`'d (needs a live CUDA device) — the guard
    /// fires before any GPU work, so the `kernel` fn pointer is never called;
    /// reusing `add_f32` is just a valid `KernelRef` value.
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

        // An UNMAPPED-op region (MatMul): kiss-ref has no equivalent, so
        // `region_supported` declines and this BF16 candidate is NOT escalate-
        // eligible — it keeps the pre-T5 hard Fail. (The region is never realized;
        // the guard fires pre-invoke.)
        let decompose = PatternNode::Op {
            op: OpTag::MatMul,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::Bf16, 16);
        let cand = CandidateKernel {
            entry_point: "matmul_bf16".to_string(),
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
                notes: "test-only, f32-only guard regression (kiss-uncoverable region)",
            },
            backend: BackendId::Cuda,
            claimed_op: None,
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
            other => panic!("expected Fail (f32-only guard, kiss-uncoverable), got {other:?}"),
        }
        assert_eq!(records.len(), 1, "one fail record for the refused claim: {records:?}");
        assert_eq!(records[0].claim, "max_ulp");
        assert_eq!(records[0].result, "fail");
    }

    /// Live-GPU (Plan T5): a kiss-COVERABLE non-f32 numeric-claim candidate
    /// ESCALATES to `Inconclusive` instead of the pre-T5 hard Fail. An `add_f64`
    /// kernel carrying a single-primitive `Add` decompose over F64 operands,
    /// declaring `PrecisionGuarantee::REFERENCE` (a numeric claim), is exactly the
    /// case the old (1a) guard hard-Failed `Fail{"max_ulp", "…f32-only…"}`
    /// pre-invoke. Under T5 it is escalate-eligible (dtype F64 ∈ {F64,F16,BF16},
    /// the Add region is kiss-supported), so it is invoked, the kiss-ref advisory
    /// diff runs (`add_f64` === kiss-ref add over f64 ⇒ 0 ULP, an advisory
    /// "pass"), and — Fuel's f32-only numeric authority yielding no usable
    /// reference — it escalates to `Inconclusive` on the live kiss-ref advisory
    /// (§6.6-0007: flag, never verdict). NOT a Fail, NOT a Pass. Born-red pre-T5
    /// (the guard returned `Fail`). `#[ignore]`'d (needs a live CUDA device).
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn verify_candidate_add_f64_is_inconclusive_not_failed() {
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
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F64, 32);
        let cand = CandidateKernel {
            entry_point: "test::t5_escalate::add_f64".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f64,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F64, DType::F64],
            kernel_revision_hash: 0x00AD_DF64,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: None,
        };

        let (verdict, records) = verify_candidate(&cand, &dev);
        match verdict {
            VerifyVerdict::Inconclusive { claim, detail } => {
                assert_eq!(claim, "max_ulp", "escalate names the evidence claim");
                assert!(
                    detail.contains("kiss-ref"),
                    "escalate detail references the kiss-ref advisory: {detail}"
                );
            }
            other => panic!("expected Inconclusive (T5 escalate), got {other:?}"),
        }
        // The advisory cross-check ran and agreed (add_f64 === kiss add, 0 ULP).
        let advisory = records
            .iter()
            .find(|r| r.claim == "kiss_ref_advisory")
            .expect("a kiss-coverable f64 Add must reach the advisory cross-check");
        assert_eq!(advisory.evidence["dtype"], serde_json::json!("F64"));
        assert_eq!(advisory.result, "pass", "add_f64 matches kiss-ref add exactly: {advisory:?}");
        // And the escalate is recorded as an inconclusive max_ulp entry.
        assert!(
            records.iter().any(|r| r.claim == "max_ulp" && r.result == "inconclusive"),
            "the escalate earns an inconclusive max_ulp record: {records:?}"
        );
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
            claimed_op: None,
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
            IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
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
            claimed_op: None,
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
            IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
        }
    }

    // -----------------------------------------------------------------
    // Task 6 (Increment 1) — the ROPE ORACLE. A candidate claiming
    // `FusedOps::ROPE` is verified against FUEL's registered ROTATE-HALF rope
    // recipe (Task 4's `reference_from_registered_recipe`, wired into
    // `verify_candidate` in Task 5) — NOT against its own decompose. The
    // headline is the REJECTION leg: an INTERLEAVED rope candidate (GPT-J
    // (2k, 2k+1) pairing) that claims ROPE must be REJECTED, because
    // interleaved rope is a DIFFERENT function than Fuel's rotate-half
    // (j, j+head_dim/2) ROPE — the real convention-mismatch bug the oracle
    // exists to catch (see the memory note
    // `rope-convention-mismatch-baracuda-fuel.md`). A matching ELEMENTWISE
    // claimant is Adopted (positive leg).
    // -----------------------------------------------------------------

    /// TEST-ONLY [`crate::kernel::KernelRef`]: wraps the STAGED interleaved
    /// `rope_apply` driver (`fuel_cuda_backend::baracuda::attention::
    /// rope_apply_fused_f32_into`) as a candidate kernel. Takes the 3
    /// rope-shaped inputs `(x, cos, sin)` + 1 output and `OpParams::Rope` —
    /// exactly the arity Fuel's rotate-half ROPE reference uses. The driver
    /// narrows Fuel's FULL-width cos/sin to baracuda's half-width tables
    /// internally, then applies the INTERLEAVED (2k, 2k+1) rotation. This is
    /// the faithful interleaved-vs-rotate-half bug (the plan's PREFER path):
    /// the candidate invokes cleanly and deterministically, and its output
    /// differs from Fuel's rotate-half recipe on the SAME probe → a numeric
    /// ("max_*") rejection, never an adopt.
    #[cfg(feature = "cuda")]
    fn interleaved_rope_apply_candidate_kernel(
        inputs: &[Arc<std::sync::RwLock<fuel_memory::Storage>>],
        outputs: &mut [Arc<std::sync::RwLock<fuel_memory::Storage>>],
        _layouts: &[fuel_ir::Layout],
        params: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        use crate::dispatch::{cuda_input, cuda_output, read_storage, write_storage};
        use crate::kernel::OpParams;

        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "interleaved_rope_apply_candidate_kernel: expected 3 inputs (x, cos, sin) + 1 \
                 output, got {} + {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let (outer_count, seq, head_dim) = match params {
            OpParams::Rope { outer_count, seq, head_dim } => (*outer_count, *seq, *head_dim),
            other => {
                return Err(fuel_ir::Error::Msg(format!(
                    "interleaved_rope_apply_candidate_kernel: expected OpParams::Rope, got {other:?}"
                )))
            }
        };
        let x_guard = read_storage(&inputs[0])?;
        let cos_guard = read_storage(&inputs[1])?;
        let sin_guard = read_storage(&inputs[2])?;
        let mut out_guard = write_storage(&outputs[0])?;
        let x_cuda = cuda_input(&x_guard)?;
        let cos_cuda = cuda_input(&cos_guard)?;
        let sin_cuda = cuda_input(&sin_guard)?;
        let out_cuda = cuda_output(&mut out_guard)?;
        fuel_cuda_backend::baracuda::attention::rope_apply_fused_f32_into(
            x_cuda, cos_cuda, sin_cuda, outer_count, seq, head_dim, out_cuda,
        )
    }

    /// Spec-B Task-6 acceptance (live GPU), the ROPE ORACLE headline —
    /// REJECTION leg. A candidate whose kernel is the STAGED INTERLEAVED
    /// `rope_apply` driver (see [`interleaved_rope_apply_candidate_kernel`]),
    /// claiming `FusedOps::ROPE` with rope-shaped F32 operands (x
    /// `[1, seq, head_dim]`, cos/sin FULL-width `[seq, head_dim]`) and
    /// `PrecisionGuarantee::REFERENCE`, must be REJECTED: `ingest_one` realizes
    /// FUEL's registered ROTATE-HALF ROPE recipe on the SAME probe
    /// (`reference_from_registered_recipe`), and the interleaved candidate's
    /// output differs → the numeric bound fails (`failed_claim` contains
    /// "max"). Never `Adopted` — a wrong-ROPE claimant must not enter the
    /// binding table.
    ///
    /// APPROACH — the plan's PREFER path (faithful interleaved rope), NOT the
    /// elementwise fallback: the candidate wraps the real
    /// `rope_apply_fused_f32_into` driver (the genuine interleaved-vs-rotate-
    /// half convention bug), which wraps cleanly here because its
    /// `(x, cos, sin)` arity matches Fuel's rope reference and it tolerates
    /// Fuel's full-width cos/sin (narrows to half-width internally). No
    /// submitted `decompose` (rope isn't a `PatternNode`), so the structural
    /// recipe-identity pre-check is correctly skipped and the NUMERIC
    /// registered-recipe reference is what rejects it.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn rope_oracle_rejects_interleaved_rope_claiming_rope() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::registry::FusedOps;
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let (seq, head_dim) = (4usize, 8usize); // head_dim even (baracuda rope)
        let x_od = OperandDesc::new(
            3,
            &[1, seq as i64, head_dim as i64],
            &[(seq * head_dim) as i64, head_dim as i64, 1],
            ElementKind::F32,
            16,
        );
        // cos/sin FULL-width [seq, head_dim] — Fuel's rotate-half reference
        // convention; the interleaved candidate narrows them internally.
        let trig_od = OperandDesc::new(
            2,
            &[seq as i64, head_dim as i64],
            &[head_dim as i64, 1],
            ElementKind::F32,
            16,
        );
        let cand = CandidateKernel {
            entry_point: "test::rope_oracle::interleaved_rope_apply_claims_rope".to_string(),
            kernel: interleaved_rope_apply_candidate_kernel,
            op_params: crate::kernel::OpParams::Rope { outer_count: 1, seq, head_dim },
            decompose: None,
            operands: vec![x_od, trig_od, trig_od],
            dtypes: vec![DType::F32, DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_0003,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: Some(FusedOps::ROPE),
        };

        match ingest_one(&cand, &dev) {
            IngestOutcome::Rejected(r) => {
                assert!(
                    r.failed_claim.contains("max") || r.failed_claim == "recipe_identity",
                    "expected a precision/identity rejection (interleaved != rotate-half), \
                     got: {} / {}",
                    r.failed_claim,
                    r.detail
                );
                // Pin the REASON, not just the claim id: `first_numeric_claim`
                // also names a `max_*` claim when the REFERENCE realize itself
                // errors (`verify_candidate_impl`'s
                // "reference realize from registered recipe failed: {e:?}"
                // detail), which would ALSO satisfy `contains("max")` above for
                // the WRONG reason (a broken reference, not a genuine
                // interleaved-vs-rotate-half numeric mismatch). Rule that out
                // explicitly so a future regression that breaks rotate-half
                // reference realization fails LOUDLY here instead of being
                // mistaken for this oracle working.
                assert!(
                    !r.detail.contains("reference realize"),
                    "rejection must be a genuine numeric mismatch vs the realized rotate-half \
                     reference, not a reference-realize failure: {}",
                    r.detail
                );
            }
            IngestOutcome::Adopted(_) => panic!(
                "interleaved rope claiming ROPE must NOT be adopted — it computes a different \
                 function than Fuel's rotate-half rope"
            ),
            IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
        }
    }

    /// Spec-B Task-6 acceptance (live GPU), the ROPE ORACLE — ADOPTION leg
    /// (positive counterpart). No CUDA ROTATE-HALF rope kernel exists as a
    /// single callable `KernelRef` (Fuel's rotate-half rope runs via the
    /// primitive decompose; every callable CUDA rope kernel is interleaved), so
    /// — per the plan's documented FALLBACK — this adopts a MATCHING
    /// ELEMENTWISE claimant, exercising the SAME claimed-op path the rope
    /// rejection does (structural recipe-identity + the numeric
    /// registered-recipe reference), just on an op that HAS a matching CUDA
    /// kernel. A runtime-registered `Add` region is the claimed op; the
    /// candidate's kernel is CUDA `add_f32` and its submitted decompose IS the
    /// `Add` region → recipe-identity matches AND `add_f32`'s output equals the
    /// registered recipe realized on the same probe (0 ULP) → `Adopted`, and
    /// the adopted kernel is visible to the capability gate on Cuda. A
    /// per-run-distinct id suffix avoids process-global registry collisions
    /// with other `#[ignore]` tests in this binary.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn rope_oracle_adopts_matching_elementwise_claimant() {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_graph::runtime_fused::register_runtime_fused;
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        // A runtime-registered Add op is the CLAIMED op; the candidate submits
        // the very same Add region as its decompose (recipe-identity matches).
        let add_region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let claimed_id = register_runtime_fused(
            "test::rope_oracle::adopt::claimed_add::run_r0pe_0004",
            add_region.clone(),
        )
        .expect("register the runtime Add op the candidate claims");

        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let cand = CandidateKernel {
            entry_point: "test::rope_oracle::adopt::add_f32::run_r0pe_0004".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f32,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(add_region),
            operands: vec![od, od],
            dtypes: vec![DType::F32, DType::F32],
            kernel_revision_hash: 0x1_9E57_0004,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: Some(claimed_id),
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
                    "the adopted claimant's kernel must be visible to the capability gate on Cuda"
                );
            }
            IngestOutcome::Rejected(r) => panic!(
                "a matching add_f32 claimant (recipe-identity + 0-ULP numeric) must be Adopted, \
                 got Rejected: {} / {}",
                r.failed_claim, r.detail
            ),
            IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
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
            claimed_op: None,
        };

        match adopt_verified(&cand) {
            IngestOutcome::Rejected(r) => {
                assert_eq!(r.failed_claim, "no_decompose");
            }
            IngestOutcome::Adopted(_) => {
                panic!("a candidate with no decompose must never be adopted")
            }
            IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
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
            claimed_op: None,
        };

        match adopt_verified(&cand) {
            IngestOutcome::Rejected(r) => {
                assert_eq!(r.failed_claim, "adopt_failed");
            }
            IngestOutcome::Adopted(_) => {
                panic!("a non-contiguous-bind region must never be adopted")
            }
            IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
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
            claimed_op: None,
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
        /// `(claim, escalate, diff_summary)` for every `on_flagged` — the
        /// escalate route the T6 service-level pin asserts.
        flagged: std::sync::Mutex<Vec<(String, bool, Option<String>)>>,
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

        fn on_flagged(&self, report: &FlagReport) {
            self.flagged.lock().unwrap().push((
                report.claim.to_string(),
                report.escalate,
                report.diff_summary.clone(),
            ));
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

    /// T6 service-level routing PIN — **expected BORN-GREEN, not born-red**
    /// (the labeled exception to this thread's TDD red-first discipline). The
    /// worker already routes `IngestOutcome::Flagged => on_flagged` (see
    /// `worker_loop`'s match arm); this test only adds the
    /// `RecordingFeedback::on_flagged` sink and pins the route at the SERVICE
    /// level, so there is no production change that flips it red→green. A mock
    /// verify that returns `Flagged` must reach the feedback's `on_flagged`
    /// (carrying `escalate` + the threaded `diff_summary`), never
    /// `on_rejected`/`on_adopted`. Synchronized deterministically via the
    /// one-shot notify channel — no sleep, same style as
    /// `worker_fires_on_adopted_for_adopted_outcome`.
    #[test]
    fn worker_routes_flagged_to_on_flagged() {
        let cfg = IngestionConfig::default();
        let svc = IngestionService::start_with_verify(
            move |_cand| {
                IngestOutcome::Flagged(FlagReport {
                    entry_point: "flagged-one".to_string(),
                    claim: "max_ulp",
                    detail: "kiss-ref advisory escalate (mock, Task 7 seam)".to_string(),
                    diff_summary: Some("result=flag max_ulp=3 mismatches=2".to_string()),
                    escalate: true,
                })
            },
            cfg,
        );

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(RecordingFeedback::with_notify(tx));
        svc.enqueue(test_candidate("flagged-one"), Some(fb.clone())).expect("enqueue accepted");

        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker should route the flagged outcome to on_flagged");

        let flagged = fb.flagged.lock().unwrap();
        assert_eq!(flagged.len(), 1, "exactly one on_flagged callback: {flagged:?}");
        let (claim, escalate, summary) = &flagged[0];
        assert_eq!(claim, "max_ulp", "the flag names the evidence claim");
        assert!(*escalate, "the flag escalates to mint a corpus vector");
        assert_eq!(
            summary.as_deref(),
            Some("result=flag max_ulp=3 mismatches=2"),
            "the threaded kiss-ref diff summary reaches the provider"
        );
        assert!(fb.rejected.lock().unwrap().is_empty(), "a Flagged outcome must not route to on_rejected");
        assert!(fb.adopted.lock().unwrap().is_empty(), "a Flagged outcome must not route to on_adopted");

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
        flagged: std::sync::Mutex<Vec<FlagReportSnapshot>>,
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

    /// An owned snapshot of a [`FlagReport`] — the escalate counterpart of
    /// [`RejectionReportSnapshot`], lifting the fields the T6 e2e assertions
    /// check (claim/detail/diff_summary/escalate) into an owned `'static`
    /// value the borrowed `&FlagReport` callback can push into a `Mutex`.
    #[cfg(feature = "cuda")]
    #[derive(Debug)]
    struct FlagReportSnapshot {
        claim: &'static str,
        detail: String,
        diff_summary: Option<String>,
        escalate: bool,
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

        fn on_flagged(&self, report: &FlagReport) {
            self.flagged.lock().unwrap().push(FlagReportSnapshot {
                claim: report.claim,
                detail: report.detail.clone(),
                diff_summary: report.diff_summary.clone(),
                escalate: report.escalate,
            });
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
            claimed_op: None,
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
            claimed_op: None,
        }
    }

    /// Build the T5 escalate candidate — an `add_f64` kernel carrying a
    /// single-primitive `Add` decompose over F64 operands and declaring a
    /// NUMERIC claim (`PrecisionGuarantee::REFERENCE`) — with a DISTINCTIVE
    /// `entry_point`/hash so it doesn't collide with the T5 unit
    /// (`verify_candidate_add_f64_is_inconclusive_not_failed`,
    /// `test::t5_escalate::add_f64`) or the other e2e candidates (8801/8802)
    /// under this binary's process-global registration. This is the
    /// service-level (v) counterpart of the T5 escalate: a kiss-coverable
    /// non-f32 numeric claim Fuel's f32-only authority can't verdict.
    #[cfg(feature = "cuda")]
    fn e2e_add_f64_candidate() -> CandidateKernel {
        use baracuda_kernels_types::{ElementKind, OperandDesc};
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
        use fuel_ir::probe::BackendId;
        use fuel_ir::DType;

        let decompose = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F64, 32);
        CandidateKernel {
            entry_point: "test::e2e::add_f64_escalate::run_e2e_8803".to_string(),
            kernel: crate::baracuda_dispatch::binary::add_f64,
            op_params: crate::kernel::OpParams::None,
            decompose: Some(decompose),
            operands: vec![od, od],
            dtypes: vec![DType::F64, DType::F64],
            kernel_revision_hash: 0x1_9E57_8803,
            declared: crate::fused::PrecisionGuarantee::REFERENCE,
            backend: BackendId::Cuda,
            claimed_op: None,
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

    /// T6 (v) service-level acceptance (live GPU), ESCALATE leg — the
    /// FLAGGED counterpart of the adopt/reject e2e legs above. The SAME
    /// production service (`IngestionService::start`, verifying against a
    /// real `CudaDevice` via `ingest_one`), driven end-to-end, must ESCALATE
    /// a kiss-coverable non-f32 numeric-claim candidate (`add_f64` with an
    /// `Add` decompose over F64, declaring `REFERENCE`) — Fuel's f32-only
    /// numeric authority yields no usable verdict, the kiss-ref advisory
    /// agrees (add_f64 === kiss add over f64), so `verify_candidate` returns
    /// `Inconclusive`, `ingest_one` maps it to `IngestOutcome::Flagged`, and
    /// the worker routes it to `on_flagged` (NEVER `on_rejected`/`on_adopted`)
    /// carrying `escalate` + the threaded kiss-ref `diff_summary` (D8). This
    /// is the service-level unit of `verify_candidate_add_f64_is_inconclusive_
    /// not_failed`. `#[ignore]`'d — needs a live CUDA device; it first RUNS at
    /// landing (the T5 escalate wiring it exercises is already on this branch,
    /// so it lands GREEN). Synchronized deterministically via the notify
    /// channel — no sleep.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn ingestion_service_flags_an_f64_add_candidate_e2e() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };

        let svc = IngestionService::start(dev, IngestionConfig::default());

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let fb = Arc::new(E2eFeedback::with_notify(tx));
        svc.enqueue(e2e_add_f64_candidate(), Some(fb.clone()))
            .expect("enqueue accepted (queue starts empty)");

        rx.recv_timeout(Duration::from_secs(30))
            .expect("worker should report the flagged outcome within 30s");

        let flagged = fb.flagged.lock().unwrap();
        assert_eq!(flagged.len(), 1, "exactly one on_flagged callback: {flagged:?}");
        assert_eq!(flagged[0].claim, "max_ulp", "the escalate names the evidence claim");
        assert!(flagged[0].escalate, "the flag escalates to mint a corpus vector");
        assert!(
            flagged[0].diff_summary.is_some(),
            "the escalate threads the kiss-ref advisory summary (D8): {:?}",
            flagged[0]
        );
        assert!(
            flagged[0].detail.contains("kiss-ref"),
            "the escalate detail references the kiss-ref advisory: {}",
            flagged[0].detail
        );
        assert!(
            fb.adopted.lock().unwrap().is_empty(),
            "a non-f32 escalate must not be adopted (Fuel renders no verdict)"
        );
        assert!(
            fb.rejected.lock().unwrap().is_empty(),
            "a kiss-coverable escalate must not be rejected — §6.6-0007: flag, never verdict"
        );

        svc.shutdown();
    }
}
