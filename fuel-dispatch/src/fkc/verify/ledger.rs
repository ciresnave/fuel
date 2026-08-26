// SPDX-License-Identifier: MIT OR Apache-2.0
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

use fuel_ir::{DType, probe::BackendId};
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
    /// ledger loaded through an f64-based parser (e.g. YAML via `serde_yaml_ng`)
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
///
/// NOTE: `include_str!` makes this a compile-time snapshot. When the ledger is
/// re-seeded (e.g. `seed_vulkan_verified_ledger`), THIS crate must be recompiled
/// for the new records to take effect — a downstream test binary relinking
/// against a cached `fuel-dispatch` lib will still see the old ledger.
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

    /// Insert `r`, first removing any existing record with the same
    /// verification key `(backend, dtypes, kernel_revision_hash, claim)` — the
    /// same tuple `has_pass` matches on. This makes re-verification idempotent:
    /// a re-run of a seeding/acceptance harness UPDATES an op's entry in place
    /// rather than appending a duplicate (a verification ledger records the
    /// latest verdict per key, not a history of runs).
    pub fn upsert(&mut self, r: LedgerRecord) {
        self.records.retain(|e| {
            !(e.backend == r.backend
                && e.dtypes == r.dtypes
                && e.kernel_revision_hash == r.kernel_revision_hash
                && e.claim == r.claim)
        });
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

/// Where the checked-in ledger lives, relative to this crate's manifest.
///
/// **Deliberately private to this module, not `pub(crate)`.** The scan below
/// is a textual backstop and will drift; narrowing visibility is the
/// structural half. A would-be fifth writer can no longer ASK for the path —
/// it has to retype the relative path itself, which is a conspicuous act
/// rather than an innocent-looking call. Callers that want to report where
/// the ledger went read [`LedgerWriteSummary::path`].
#[cfg(test)]
fn checked_in_ledger_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/kernel-contracts/.fkc-verified-ledger.json")
}

/// Merge `fresh` into the CHECKED-IN ledger, WITHOUT writing anything.
///
/// Every record already in the file survives; a `fresh` record whose
/// verification key `(backend, dtypes, kernel_revision_hash, claim)` matches
/// an existing one REPLACES it in place (see [`VerificationLedger::upsert`]),
/// so re-running a seeder updates its own verdicts and touches nobody else's.
///
/// Split out from [`write_merged_ledger`] purely so the merge can be tested:
/// the writer targets the real repo file, so a guard test cannot call it
/// without destroying the thing it is guarding.
#[cfg(test)]
fn merge_into_checked_in(fresh: &[LedgerRecord]) -> VerificationLedger {
    let mut ledger =
        VerificationLedger::from_records(VerificationLedger::embedded().records().to_vec());
    for r in fresh {
        ledger.upsert(r.clone());
    }
    ledger
}

/// Summary of a ledger write, for the seeder's `--nocapture` report.
#[cfg(test)]
pub(crate) struct LedgerWriteSummary {
    /// Records in the checked-in ledger BEFORE this run.
    pub(crate) before: usize,
    /// Records in the file after the merge.
    pub(crate) after: usize,
    /// Records this run contributed (some of which may have replaced
    /// same-key entries rather than adding new ones).
    pub(crate) fresh: usize,
    /// Where it was written. Returned rather than exposing the path helper,
    /// so reporting where the file is does not hand out the ability to open
    /// it.
    pub(crate) path: std::path::PathBuf,
}

/// Merge `fresh` into the checked-in ledger and write the union back.
///
/// **This is the ONE writer, and it merges, because the alternative was in
/// the tree and it silently destroys other backends' work.** FOUR call sites
/// write this single file — the CPU / CUDA / Vulkan seeders and the
/// rope-apply acceptance test in `harness.rs` — and each earns records the
/// others cannot: CUDA's 142 need a forge slot and a live GPU, Vulkan's 530
/// need a live device, and there is no GPU runner in CI — so a truncating
/// write is not a regeneratable inconvenience, it is unrecoverable outside
/// this machine. `seed_cpu_verified_ledger` did exactly that
/// (`serde_json::to_string_pretty(&records)` over a FRESH `Vec` into a
/// `File::create`), and would have dropped 672 of 749 records on its next
/// manual run (GAP-210).
///
/// The fix is deliberately structural rather than a note in three places:
/// the merge is not a discipline the caller has to remember, it is the only
/// path to the file — and `no_seeder_writes_the_ledger_behind_this_writer`
/// is what keeps that sentence true, since nothing in the language stops a
/// future seeder from reaching for `File::create` itself.
///
/// `#[cfg(test)]` is the right scope, not an accident: every one of those
/// four call sites is an `#[ignore]`d manual tool, so this file is only ever
/// written from a test binary.
///
/// The count is FOUR and not three on purpose: `harness.rs` was a writer
/// nobody had counted, and it is the reason the backstop scan checks a
/// family of spellings rather than `File::create` alone.
#[cfg(test)]
pub(crate) fn write_merged_ledger(fresh: &[LedgerRecord]) -> LedgerWriteSummary {
    use std::io::Write as _;

    let before = VerificationLedger::embedded().len();
    let ledger = merge_into_checked_in(fresh);
    let path = checked_in_ledger_path();
    let json = serde_json::to_string_pretty(ledger.records()).expect("serialize ledger records");
    // (`path` is moved into the summary below; open by reference.)
    let mut f = std::fs::File::create(&path)
        .unwrap_or_else(|e| panic!("failed to open ledger at {path:?} for writing: {e}"));
    f.write_all(json.as_bytes()).expect("write ledger json");
    f.write_all(b"\n").expect("write trailing newline");
    LedgerWriteSummary {
        before,
        after: ledger.len(),
        fresh: fresh.len(),
        path,
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
    let check = |c: &'static str| ledger.has_pass(q.backend, q.dtypes, q.kernel_revision_hash, c);
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
    use fuel_ir::{DType, probe::BackendId};

    #[test]
    fn ledger_from_json_roundtrips_and_has_pass_matches_on_revision_and_claim() {
        let json = r#"[{
            "kernel_ref": "rope_apply_f32", "backend": "Cuda", "dtypes": ["F32"],
            "kernel_revision_hash": 1234567890123456789, "claim": "bit_stable_on_same_hardware",
            "result": "pass", "verified_at": "2026-07-11T00:00:00Z", "protocol_version": 1,
            "evidence": {"repeat_calls": 150}
        }]"#;
        let ledger = VerificationLedger::from_json(json).expect("parses");
        assert!(ledger.has_pass(
            BackendId::Cuda,
            &[DType::F32],
            1234567890123456789,
            "bit_stable_on_same_hardware"
        ));
        assert!(!ledger.has_pass(
            BackendId::Cuda,
            &[DType::F32],
            1234567890123456788,
            "bit_stable_on_same_hardware"
        ));
        assert!(!ledger.has_pass(
            BackendId::Cuda,
            &[DType::F32],
            1234567890123456789,
            "max_ulp"
        ));
        assert!(!ledger.has_pass(
            BackendId::Cpu,
            &[DType::F32],
            1234567890123456789,
            "bit_stable_on_same_hardware"
        ));
        assert!(!ledger.has_pass(
            BackendId::Cuda,
            &[DType::F16],
            1234567890123456789,
            "bit_stable_on_same_hardware"
        ));
        let failing = VerificationLedger::from_json(&json.replace("\"pass\"", "\"fail\"")).unwrap();
        assert!(!failing.has_pass(
            BackendId::Cuda,
            &[DType::F32],
            1234567890123456789,
            "bit_stable_on_same_hardware"
        ));
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
    use fuel_ir::{DType, probe::BackendId};

    fn claim() -> PrecisionGuarantee {
        PrecisionGuarantee {
            bit_stable_on_same_hardware: true,
            max_ulp: Some(0),
            max_relative: None,
            max_absolute: None,
            notes: "audited exact f32 add",
        }
    }
    fn q() -> LedgerQuery<'static> {
        LedgerQuery {
            kernel_ref: "rope_apply_f32",
            backend: BackendId::Cuda,
            dtypes: &[DType::F32],
            kernel_revision_hash: 42,
        }
    }
    fn pass(c: &str) -> LedgerRecord {
        LedgerRecord {
            kernel_ref: "rope_apply_f32".into(),
            backend: "Cuda".into(),
            dtypes: vec!["F32".into()],
            kernel_revision_hash: 42,
            claim: c.into(),
            result: "pass".into(),
            verified_at: "t".into(),
            protocol_version: 1,
            evidence: serde_json::Value::Null,
        }
    }

    #[test]
    fn no_ledger_entry_downgrades_to_unaudited_and_warns() {
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &VerificationLedger::default(), &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
        assert!(!g.bit_stable_on_same_hardware);
        assert!(g.max_ulp.is_none());
        assert_eq!(w.len(), 1);
        assert!(
            w[0].message.contains("rope_apply_f32")
                && w[0].message.contains("bit_stable_on_same_hardware")
                && w[0].message.contains("max_ulp")
        );
    }
    #[test]
    fn matching_pass_entries_for_every_claim_are_honored() {
        let ledger = VerificationLedger::from_records(vec![
            pass("bit_stable_on_same_hardware"),
            pass("max_ulp"),
        ]);
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
        assert!(
            !g.bit_stable_on_same_hardware,
            "whole-collapse: even the backed bit_stable claim is dropped"
        );
        assert!(
            g.max_ulp.is_none(),
            "whole-collapse: the unbacked max_ulp bound is dropped"
        );
    }
    #[test]
    fn stale_hash_downgrades_even_with_a_pass_for_the_old_hash() {
        let mut old = pass("bit_stable_on_same_hardware");
        old.kernel_revision_hash = 41;
        let mut w = Vec::new();
        let g = gate_precision(
            claim(),
            &q(),
            &VerificationLedger::from_records(vec![old]),
            &mut w,
        );
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

    /// A seeding run must ADD to the checked-in ledger, never REPLACE it.
    ///
    /// Born-red against the exact code that was in the tree: the CPU seeder
    /// wrote `serde_json::to_string_pretty(&records)` — a FRESH `Vec` — into a
    /// `File::create`, so one manual re-seed would have left the file holding
    /// only what that run earned. Both assertions below fail under that
    /// behaviour (`after` collapses to `fresh.len()`, and the survivor lookup
    /// finds nothing).
    ///
    /// This guards the MERGE, not the write: `write_merged_ledger` targets the
    /// real repo file, so a test that exercised the writer end-to-end would
    /// destroy the artifact it exists to protect.
    #[test]
    fn a_seeding_run_merges_into_the_checked_in_ledger_rather_than_replacing_it() {
        let embedded = VerificationLedger::embedded();
        // Non-triviality: with a near-empty checked-in ledger the assertions
        // below would hold for a truncating implementation too.
        assert!(
            embedded.len() > 100,
            "checked-in ledger holds only {} records — this guard cannot distinguish a merge from a truncation against a population that small. If the ledger really did shrink that far, that is the finding.",
            embedded.len()
        );
        let witness = embedded.records()[0].clone();

        // A record whose key collides with nothing in the file.
        let fresh = LedgerRecord {
            kernel_ref: "guard-test-synthetic".to_string(),
            backend: "Cpu".to_string(),
            dtypes: vec!["F32".to_string()],
            kernel_revision_hash: 0xDEAD_BEEF_DEAD_BEEF,
            claim: "bit_stable_on_same_hardware".to_string(),
            result: "pass".to_string(),
            verified_at: "epoch:0".to_string(),
            protocol_version: 1,
            evidence: serde_json::Value::Null,
        };

        let merged = merge_into_checked_in(std::slice::from_ref(&fresh));
        assert_eq!(
            merged.len(),
            embedded.len() + 1,
            "a fresh record with a new key must ADD one row, leaving the other {} untouched — a seeder that rewrites the file from its own results destroys the two backends' records it cannot re-earn (GAP-210)",
            embedded.len()
        );
        assert!(
            merged.records().contains(&witness),
            "record {:?}/{:?} present before the merge is missing after it",
            witness.backend,
            witness.kernel_ref
        );

        // ...and a COLLIDING key updates in place rather than duplicating,
        // so a re-run of the same seeder is idempotent.
        let mut again = witness.clone();
        again.verified_at = "epoch:1".to_string();
        let remerged = merge_into_checked_in(std::slice::from_ref(&again));
        assert_eq!(
            remerged.len(),
            embedded.len(),
            "re-verifying an existing key must replace its row, not append a second one"
        );
        assert!(
            remerged
                .records()
                .iter()
                .any(|r| r.verified_at == "epoch:1"),
            "the replacement row did not take effect"
        );
    }

    /// The merging writer is only "the one writer" for as long as nobody
    /// opens the ledger behind it. Nothing in the language enforces that, so
    /// this does: no file-opening call may appear in `verify/` outside
    /// `ledger.rs`.
    ///
    /// Deliberately a source scan and not a type-system trick — the thing
    /// being prevented is a future author writing five ordinary lines of
    /// `std::fs`, which no API design can make unspellable.
    ///
    /// **The predicate scans for a FAMILY of spellings, and that is not
    /// belt-and-braces.** The first version of this test looked for
    /// `File::create` alone and would have passed while
    /// `harness.rs` wrote the same file through `std::fs::write` — a fourth
    /// writer, invisible to the gate, found by reading the tree rather than
    /// by the gate that existed to find it. Counting one spelling of a
    /// construct is not counting the construct.
    ///
    /// **What this scan CANNOT see, stated so nobody reads its green as more
    /// than it is:** a write reached through a helper defined elsewhere; a
    /// write through an alias or re-export (`use std::fs::write as emit;`);
    /// and any writer in a file outside `verify/*.rs` — which is precisely
    /// how the fourth writer was missed, since widening the predicate does
    /// nothing about a scope that is a glob somebody chose. The primary
    /// control is therefore STRUCTURAL — `checked_in_ledger_path` is private
    /// to this module, so a fifth writer must retype the path rather than ask
    /// for it — and this scan is the backstop, not the guarantee.
    #[test]
    fn no_seeder_writes_the_ledger_behind_this_writer() {
        // Every way this crate could open the ledger for writing. A new
        // spelling here is cheaper than the fourth writer it would catch.
        const WRITE_SPELLINGS: &[&str] =
            &["File::create", "fs::write", "OpenOptions", "File::options"];

        // Drop whole-line comments, so a doc comment that NAMES a forbidden
        // spelling does not trip the scan. Deliberately only whole-line: a
        // trailing comment on a code line stays in, because the error
        // direction that matters here is loud-and-wrong over silent-and-wrong,
        // and truncating code lines at `//` would also swallow a real call
        // sitting after a `https://` inside a string.
        fn strip_line_comments(src: &str) -> String {
            src.lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        }

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fkc/verify");
        let mut scanned = 0usize;
        let mut offenders: Vec<String> = Vec::new();
        let entries =
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {dir:?}: {e}"));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
            let src = strip_line_comments(&src);
            scanned += 1;
            if name != "ledger.rs" {
                for spelling in WRITE_SPELLINGS {
                    if src.contains(spelling) {
                        offenders.push(format!("{name} ({spelling})"));
                    }
                }
            }
        }
        // Positive control: a wrong directory would make the scan above pass
        // by finding nothing at all. These are the files that must be in it.
        assert!(
            scanned >= 8,
            "only scanned {scanned} .rs files under {dir:?} — this scan is looking in the wrong place and would pass vacuously"
        );
        // Positive control on the PREDICATE, not just the path: `ledger.rs`
        // is the one file that must trip it, so if it doesn't, the predicate
        // is broken and every `offenders.is_empty()` below is meaningless.
        let ledger_src = strip_line_comments(
            &std::fs::read_to_string(dir.join("ledger.rs")).expect("ledger.rs must be readable"),
        );
        assert!(
            WRITE_SPELLINGS.iter().any(|s| ledger_src.contains(s)),
            "the scan cannot see a file-opening call even in ledger.rs, where one certainly is — the predicate is broken, not the tree"
        );
        assert!(
            offenders.is_empty(),
            "{offenders:?} open the verification ledger directly instead of going through `write_merged_ledger`. A seeder that writes its own results over this file destroys the CUDA and Vulkan records it cannot re-earn — that is GAP-210, and it is the reason there is exactly one writer."
        );
    }
}
