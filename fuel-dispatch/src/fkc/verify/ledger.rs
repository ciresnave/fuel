//! The FKC verification ledger (`V-FKC-9`, empirical precision verification).
//!
//! A git-checked-in JSON ledger of empirical verification results: which
//! `(kernel_revision_hash, backend, dtypes, claim)` tuples have actually been
//! measured and passed, vs. merely *asserted* by a kernel-contract author.
//! The embedded copy (`include_str!`) is baked into every build so the
//! import-time gate (a later task) can run in hardware-free `cargo test` —
//! no filesystem access, no network, no live device required.
//!
//! This task (4.1) ships only the ledger foundation: the record/ledger
//! types, the `embedded()` loader, and the `has_pass` lookup. The bit-
//! stability / ULP / accept-coverage verifiers and the invoker back ends
//! that actually *produce* ledger entries are later tasks (4.4/4.5); they
//! extend `verify/mod.rs`'s module declarations when they land.
//!
//! Never-panic: a malformed embedded ledger parses to an *empty* ledger
//! (via `unwrap_or_default()`), never panics. Empty is the conservative
//! outcome — every claim looks unverified, so a downstream gate (built in
//! a later task) downgrades everything rather than trusting a claim that
//! was never checked.

use fuel_ir::{probe::BackendId, DType};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// One empirical verification result for a single kernel/backend/dtype/claim
/// combination, as recorded by the (external, later-task) verification
/// harness and checked in to `docs/kernel-contracts/.fkc-verified-ledger.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerRecord {
    /// The kernel's `entry_point` / ref name, e.g. `"rope_apply_f32"`. Purely
    /// informational for lookups (`has_pass` keys on the hash, not this name)
    /// — carried so the ledger is human-auditable without cross-referencing
    /// hashes back to contracts.
    pub kernel_ref: String,
    /// Backend label: `"Cpu"` | `"Cuda"` | `"Vulkan"` | `"Metal"`.
    pub backend: String,
    /// `DType` `Debug` names, e.g. `"F32"`. Order-sensitive: must match the
    /// query's dtype list positionally (see `dtypes_match`).
    pub dtypes: Vec<String>,
    /// The kernel-contract revision hash (`fkc::compute_revision`) this
    /// result was measured against. `u64` (not `f64`): a plain JSON-number
    /// ledger loaded through an f64-based parser (e.g. YAML via `serde_yml`)
    /// would silently round revision hashes above 2^53, corrupting the
    /// lookup key — this is why the ledger is JSON (`serde_json`), which
    /// parses `u64` natively, and not YAML like the rest of FKC.
    pub kernel_revision_hash: u64,
    /// Claim identifier, e.g. `"bit_stable_on_same_hardware"` | `"max_ulp"`
    /// | `"max_relative"` | `"max_absolute"` | `"accept_coverage"`.
    pub claim: String,
    /// `"pass"` | `"fail"` | `"no_reference"`. Only `"pass"` satisfies
    /// `has_pass`.
    pub result: String,
    /// ISO-8601 timestamp of when the verification ran. Informational.
    pub verified_at: String,
    /// Ledger schema/protocol version, for forward-compatible parsing.
    pub protocol_version: u32,
    /// Free-form verifier-specific evidence (repeat-call counts, measured
    /// ULP distances, etc.). Defaults to `Value::Null` if absent.
    #[serde(default)]
    pub evidence: serde_json::Value,
}

/// A parsed collection of [`LedgerRecord`]s, with a `(backend, dtypes,
/// revision, claim)` lookup (`has_pass`).
#[derive(Debug, Clone, Default)]
pub struct VerificationLedger {
    records: Vec<LedgerRecord>,
}

/// The git-checked-in verification ledger, embedded at compile time so the
/// gate runs in every hardware-free `cargo test` with no filesystem access.
/// Must exist and parse as a JSON array (an empty ledger is `[]`) before
/// `fuel-dispatch` compiles at all.
const LEDGER_JSON: &str =
    include_str!("../../../../docs/kernel-contracts/.fkc-verified-ledger.json");

impl VerificationLedger {
    /// Parse a ledger from a JSON array of [`LedgerRecord`]s.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        Ok(Self {
            records: serde_json::from_str(s)?,
        })
    }

    /// Build a ledger directly from records (e.g. for tests or programmatic
    /// construction, ahead of the invoker back ends that will append to the
    /// checked-in file).
    pub fn from_records(records: Vec<LedgerRecord>) -> Self {
        Self { records }
    }

    /// The ledger's records, in file order.
    pub fn records(&self) -> &[LedgerRecord] {
        &self.records
    }

    /// Append a record.
    pub fn push(&mut self, r: LedgerRecord) {
        self.records.push(r);
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True iff the ledger has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The embedded (compile-time, `include_str!`) ledger, parsed once and
    /// cached. Never panics: a malformed embedded file parses to an empty
    /// ledger (`unwrap_or_default()`) — the conservative outcome, since an
    /// empty ledger fails every `has_pass` lookup and so downgrades every
    /// claim, rather than trusting one that was never actually verified.
    pub fn embedded() -> &'static VerificationLedger {
        static L: OnceLock<VerificationLedger> = OnceLock::new();
        L.get_or_init(|| VerificationLedger::from_json(LEDGER_JSON).unwrap_or_default())
    }

    /// True iff the ledger has a `"pass"` record matching all four
    /// components: `backend`, `dtypes` (positional), `kernel_revision_hash`,
    /// and `claim`. Any single mismatched component is a miss — the ledger
    /// is deliberately narrow (revision-hash-keyed) so a kernel edit that
    /// changes the hash invalidates all prior verification for it.
    pub fn has_pass(&self, backend: BackendId, dtypes: &[DType], rev: u64, claim: &str) -> bool {
        self.records.iter().any(|r| {
            r.result == "pass"
                && r.kernel_revision_hash == rev
                && r.claim == claim
                && backend_label(backend) == r.backend
                && dtypes_match(&r.dtypes, dtypes)
        })
    }
}

fn backend_label(b: BackendId) -> &'static str {
    match b {
        BackendId::Cpu => "Cpu",
        BackendId::Cuda => "Cuda",
        BackendId::Vulkan => "Vulkan",
        BackendId::Metal => "Metal",
        _ => "Unknown",
    }
}

fn dtypes_match(rec: &[String], want: &[DType]) -> bool {
    rec.len() == want.len() && rec.iter().zip(want).all(|(s, d)| *s == format!("{d:?}"))
}

use crate::fkc::ImportWarning;
use crate::fused::PrecisionGuarantee;

/// A query key for [`gate_precision`]: identifies the kernel/backend/dtype/
/// revision combination whose declared [`PrecisionGuarantee`] must be
/// checked against the [`VerificationLedger`].
pub struct LedgerQuery<'a> {
    /// Diagnostic-only — NOT part of the match key (`has_pass` matches on
    /// `backend`/`dtypes`/`kernel_revision_hash`/`claim` alone). Carried so
    /// warnings can name the kernel without a second lookup.
    pub kernel_ref: &'a str,
    /// Backend the claim was declared for.
    pub backend: BackendId,
    /// Dtypes the claim was declared for (order-sensitive; see `dtypes_match`).
    pub dtypes: &'a [DType],
    /// The kernel-contract revision hash (`fkc::compute_revision`) the
    /// declared guarantee is being checked against.
    pub kernel_revision_hash: u64,
}

/// V-FKC-9 precision gate. Any machine-checkable claim in `declared`
/// (`bit_stable_on_same_hardware` / `max_ulp` / `max_relative` /
/// `max_absolute`) must have a matching `pass` ledger record for the
/// CURRENT `kernel_revision_hash`, else the WHOLE guarantee collapses to
/// [`PrecisionGuarantee::UNAUDITED`] plus one [`ImportWarning`] naming every
/// unbacked claim. An audited-none (no machine-checkable bounds) guarantee
/// passes through untouched — there's nothing for the ledger to back.
pub fn gate_precision(
    declared: PrecisionGuarantee,
    q: &LedgerQuery,
    ledger: &VerificationLedger,
    warnings: &mut Vec<ImportWarning>,
) -> PrecisionGuarantee {
    let mut unbacked: Vec<&'static str> = Vec::new();
    let check =
        |c: &'static str| ledger.has_pass(q.backend, q.dtypes, q.kernel_revision_hash, c);
    if declared.bit_stable_on_same_hardware && !check("bit_stable_on_same_hardware") {
        unbacked.push("bit_stable_on_same_hardware");
    }
    if declared.max_ulp.is_some() && !check("max_ulp") {
        unbacked.push("max_ulp");
    }
    if declared.max_relative.is_some() && !check("max_relative") {
        unbacked.push("max_relative");
    }
    if declared.max_absolute.is_some() && !check("max_absolute") {
        unbacked.push("max_absolute");
    }
    if unbacked.is_empty() {
        return declared;
    }
    warnings.push(ImportWarning {
        section: q.kernel_ref.to_string(),
        message: format!(
            "precision claim(s) {unbacked:?} for kernel `{}` ({:?}, dtypes {:?}, rev {}) have no passing \
            verification-ledger entry — downgraded to UNAUDITED (run the fkc_verify harness to earn them)",
            q.kernel_ref, q.backend, q.dtypes, q.kernel_revision_hash
        ),
    });
    PrecisionGuarantee::UNAUDITED
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{probe::BackendId, DType};

    #[test]
    fn ledger_from_json_roundtrips_and_has_pass_matches_on_revision_and_claim() {
        let json = r#"[{
            "kernel_ref": "rope_apply_f32", "backend": "Cuda", "dtypes": ["F32"],
            "kernel_revision_hash": 1234567890123456789, "claim": "bit_stable_on_same_hardware",
            "result": "pass", "verified_at": "2026-07-11T00:00:00Z", "protocol_version": 1,
            "evidence": {"repeat_calls": 150}
        }]"#;
        let ledger = VerificationLedger::from_json(json).expect("parses");
        assert!(ledger.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456789, "bit_stable_on_same_hardware"));
        assert!(!ledger.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456788, "bit_stable_on_same_hardware"));
        assert!(!ledger.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456789, "max_ulp"));
        assert!(!ledger.has_pass(BackendId::Cpu, &[DType::F32], 1234567890123456789, "bit_stable_on_same_hardware"));
        assert!(!ledger.has_pass(BackendId::Cuda, &[DType::F16], 1234567890123456789, "bit_stable_on_same_hardware"));
        let failing = VerificationLedger::from_json(&json.replace("\"pass\"", "\"fail\"")).unwrap();
        assert!(!failing.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456789, "bit_stable_on_same_hardware"));
        // Task 4.1 shipped this as `assert_eq!(embedded().len(), 0)` — the
        // ledger was an intentional Task-4.1 placeholder (`[]`). Task 4.5b
        // (2026-07-12) populated it with REAL empirically-verified CPU
        // fused-op `bit_stable_on_same_hardware` records (see
        // `seed_cpu_ledger.rs`), so a bare emptiness check would now be
        // false BY DESIGN. What must still hold — an unrelated, made-up
        // revision hash never spuriously matches — is exactly the property
        // `has_pass`'s revision-keying exists to guarantee, so assert that
        // invariant against the embedded ledger instead of its length.
        assert!(!VerificationLedger::embedded().has_pass(
            BackendId::Cuda,
            &[DType::F32],
            0xDEAD_BEEF_u64,
            "bit_stable_on_same_hardware",
        ));
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use fuel_ir::{probe::BackendId, DType};

    fn claim() -> PrecisionGuarantee {
        PrecisionGuarantee { bit_stable_on_same_hardware: true, max_ulp: Some(0), max_relative: None, max_absolute: None, notes: "audited exact f32 add" }
    }
    fn q() -> LedgerQuery<'static> {
        LedgerQuery { kernel_ref: "rope_apply_f32", backend: BackendId::Cuda, dtypes: &[DType::F32], kernel_revision_hash: 42 }
    }
    fn pass(c: &str) -> LedgerRecord {
        LedgerRecord { kernel_ref: "rope_apply_f32".into(), backend: "Cuda".into(), dtypes: vec!["F32".into()],
            kernel_revision_hash: 42, claim: c.into(), result: "pass".into(), verified_at: "t".into(), protocol_version: 1, evidence: serde_json::Value::Null }
    }

    #[test]
    fn no_ledger_entry_downgrades_to_unaudited_and_warns() {
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &VerificationLedger::default(), &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
        assert!(!g.bit_stable_on_same_hardware);
        assert!(g.max_ulp.is_none());
        assert_eq!(w.len(), 1);
        assert!(w[0].message.contains("rope_apply_f32") && w[0].message.contains("bit_stable_on_same_hardware") && w[0].message.contains("max_ulp"));
    }
    #[test]
    fn matching_pass_entries_for_every_claim_are_honored() {
        let ledger = VerificationLedger::from_records(vec![pass("bit_stable_on_same_hardware"), pass("max_ulp")]);
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &ledger, &mut w);
        assert!(g.bit_stable_on_same_hardware && g.max_ulp == Some(0) && w.is_empty());
    }
    #[test]
    fn partial_backing_still_downgrades_the_whole_claim() {
        let ledger = VerificationLedger::from_records(vec![pass("bit_stable_on_same_hardware")]);
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &ledger, &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
        assert!(w[0].message.contains("max_ulp"));
        assert!(!g.bit_stable_on_same_hardware, "whole-collapse: even the backed bit_stable claim is dropped");
        assert!(g.max_ulp.is_none(), "whole-collapse: the unbacked max_ulp bound is dropped");
    }
    #[test]
    fn stale_hash_downgrades_even_with_a_pass_for_the_old_hash() {
        let mut old = pass("bit_stable_on_same_hardware"); old.kernel_revision_hash = 41;
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &VerificationLedger::from_records(vec![old]), &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
    }
    #[test]
    fn no_verifiable_bound_passes_through_untouched() {
        let declared = PrecisionGuarantee::none("audited; no static bound applies");
        let mut w = Vec::new();
        let g = gate_precision(declared, &q(), &VerificationLedger::default(), &mut w);
        assert_eq!(g.notes, declared.notes);
        assert!(w.is_empty());
    }
}
