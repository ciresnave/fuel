//! GAP-036 increment 1 — **derived, bidirectional clause→test coverage for the
//! FKC enforcement surface.**
//!
//! Adopts KISS-Conform §6.1's discipline in the shape that actually fits Rust:
//! every normative clause resolves to ≥1 test, every citation resolves to a
//! live clause, and the authoritative clause set is **derived, never
//! hand-authored** (§6.1-0004).
//!
//! # What is a "clause" here, and why it is not a prose section number
//!
//! KISS anchors clauses on numbered spec prose (`6.1-0002`). Fuel's FKC spec
//! (`docs/specs/kernel-contract-format.md`) is 2480 lines whose normative
//! statements are **not atomic** — its own numbered validator rules bundle
//! several independent checks per paragraph (rule 5 alone carries four).
//! Numbering that prose is real work and is tracked as increment 2.
//!
//! This increment anchors on the **`FkcError` variant set** instead, because
//! for the defect class GAP-036 exists to kill it is strictly the better
//! anchor:
//!
//!   * It is **derived from the code that enforces**, so it cannot drift from
//!     enforcement the way a prose clause list can. A validator that stops
//!     being invoked does not silently keep its clause number — its variant is
//!     right here, and this gate asks whether anything exercises it.
//!   * It is already **atomic and machine-readable** — one variant, one
//!     rejectable condition, no prose parsing, no semantic judgement.
//!
//! The tradeoff, stated rather than hidden: this covers the normative surface
//! that **produces a typed rejection**. MUSTs that bind something other than
//! the validator — e.g. "the planner MUST insert `Op::Contiguize`" (§4.3) —
//! have no `FkcError` and are invisible here. Those are increment 2's job, and
//! this file's green says nothing about them.
//!
//! # Anti-vacuity (read before trusting a green)
//!
//! A coverage scanner that silently goes blind reports **full coverage**, which
//! is the worst possible failure mode — it is indistinguishable from success.
//! The first version of this scanner did exactly that: it disarmed its
//! `#[cfg(test)]` region tracker on the `#[cfg(test)]` line itself, before the
//! module's opening brace was ever counted, so `in_test` reset immediately and
//! it reported **0 of 47 variants tested**. A hand count found 41 citations in
//! one file. It was caught by a positive control, not by reading the code.
//!
//! So this gate positive-controls **itself**, every run (`POSITIVE CONTROL`
//! below): it asserts the derivation found a plausible number of clauses, and
//! that specific variants known to be cited are in fact detected as cited. If
//! the scanner breaks, these fail loudly instead of reporting a clean sweep.
//! Do not remove them to "simplify" — they are the reason a green here means
//! anything.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Deliberate, reasoned exemptions from clause→test totality.
///
/// Every entry needs a reason a reader can check, and a registry id where one
/// exists. This table is the *visible* form of an uncovered clause — the
/// alternative is not "no exemptions", it is exemptions nobody wrote down.
/// KISS-Conform's own framing (and the GAP-046 precedent): a declined thing
/// with reasoning beats a silently omitted one.
///
/// An entry naming a variant that no longer exists is a **hard failure**, not a
/// no-op — see `no_orphan_exemptions`. That is the test→clause direction
/// (§6.1-0003): a citation must resolve to a live clause.
const EXEMPT: &[(&str, &str)] = &[
    (
        "BlurbMismatch",
        "RESERVED BY DESIGN, not a gap. `validate.rs:40` records that the \
         prose/structured blurb equality check (§10.11) is performed by the \
         layer holding the raw markdown, and that this variant `stays reserved` \
         for it. Nothing constructs it, so there is nothing to provoke.",
    ),
    (
        "UnauditedPrecision",
        "DEAD CLAUSE — genuinely unreachable, and this gate is how it became \
         visible. The variant appears NOWHERE outside `error.rs`: not \
         constructed, not matched, not even named in a doc comment. This is the \
         precision half of V-FKC-9, recorded as a wiring DECISION rather than a \
         defect. It is exempted because no test can provoke an error no code \
         emits — writing one would require wiring the validator first.",
    ),
    (
        "Io",
        "Infrastructure, not a contract rule. Constructed on filesystem read \
         failure in the loader. Provoking it in-process means an unreadable \
         path, which is a harness-portability problem rather than a contract \
         assertion. Low value, non-zero cost; deliberately deferred.",
    ),
    (
        "Yaml",
        "Infrastructure, not a contract rule. Wraps the YAML parser's own \
         error. The FKC-specific YAML restrictions that DO carry contract \
         meaning have their own variants (`TabIndentation`, `AnchorDisallowed`, \
         `AliasDisallowed`, `MergeKeyDisallowed`, `NorwayToken`) and every one \
         of those IS covered — so the meaningful surface here is tested, and \
         this variant is the uninteresting remainder.",
    ),
];

/// Minimum plausible clause count. A derivation that collapses (bad regex, moved
/// enum) yields a tiny set and would otherwise report a clean sweep over
/// nothing.
const MIN_EXPECTED_CLAUSES: usize = 40;

/// Variants known to be cited by tests. If the scanner stops seeing these it is
/// blind, and its green is meaningless.
const MUST_DETECT_AS_COVERED: &[&str] = &["LayoutIncoherent", "QuantIncoherent", "MissingRequiredField"];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Derive the authoritative clause set from the enum that defines it.
///
/// This is the §6.1-0004 property: the set is read out of the enforcing code,
/// so it cannot be hand-maintained into disagreement with reality.
fn derive_clauses() -> BTreeSet<String> {
    let path = crate_root().join("src/fkc/error.rs");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read the clause source {}: {e}", path.display()));
    let start = src
        .find("pub enum FkcError")
        .unwrap_or_else(|| panic!("`pub enum FkcError` not found in {} — the clause set moved; \
             this gate must be repointed, NOT deleted", path.display()));
    let body = &src[start..];

    let mut out = BTreeSet::new();
    for line in body.lines() {
        // Variants sit at exactly one indent level, and are followed by `{`
        // (struct-like), `(` (tuple-like) or `,` (unit).
        let Some(rest) = line.strip_prefix("    ") else { continue };
        if rest.starts_with(' ') || rest.starts_with("//") || rest.starts_with('#') {
            continue;
        }
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric()).collect();
        if name.is_empty() || !name.starts_with(|c: char| c.is_ascii_uppercase()) {
            continue;
        }
        let after = rest[name.len()..].trim_start();
        if after.starts_with('{') || after.starts_with('(') || after.starts_with(',') {
            out.insert(name);
        }
    }
    out
}

fn rust_sources() -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if p.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                walk(&p, files);
            } else if p.extension().is_some_and(|e| e == "rs") {
                files.push(p);
            }
        }
    }
    walk(&crate_root().join("src"), &mut files);
    walk(&crate_root().join("tests"), &mut files);
    files
}

/// Which clauses are cited from **test** regions.
///
/// Test-region detection is brace-depth tracking armed on `#[cfg(test)]` /
/// `#[test]`. Note the `entered` flag: without it the tracker disarms on the
/// arming line itself (depth has not yet passed the opening brace) and the
/// scanner sees no tests at all. That exact bug produced a false "0 of 47".
fn cited_by_tests(clauses: &BTreeSet<String>) -> BTreeMap<String, Vec<String>> {
    let mut cited: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in rust_sources() {
        let Ok(src) = std::fs::read_to_string(&file) else { continue };
        let label = file
            .strip_prefix(crate_root())
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");

        let (mut depth, mut in_test, mut arm, mut entered) = (0i32, false, 0i32, false);
        for line in src.lines() {
            let trimmed = line.trim_start();
            if !in_test && (trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("#[test]")) {
                in_test = true;
                arm = depth;
                entered = false;
            }
            if in_test && !trimmed.starts_with("//") {
                for c in clauses {
                    if line.contains(&format!("FkcError::{c}")) || line.contains(&format!("Self::{c}")) {
                        let entry = cited.entry(c.clone()).or_default();
                        if !entry.contains(&label) {
                            entry.push(label.clone());
                        }
                    }
                }
            }
            depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
            if in_test {
                if depth > arm {
                    entered = true;
                } else if entered {
                    in_test = false;
                    entered = false;
                }
            }
        }
    }
    cited
}

/// **§6.1-0002 direction — clause → test.** Every derived clause must be cited
/// by at least one test, or carry a written exemption.
#[test]
fn every_clause_is_cited_by_a_test() {
    let clauses = derive_clauses();
    let cited = cited_by_tests(&clauses);

    // ---- POSITIVE CONTROL. A blind scanner reports full coverage; these make
    // it report failure instead. Do not remove.
    assert!(
        clauses.len() >= MIN_EXPECTED_CLAUSES,
        "clause DERIVATION collapsed: found only {} clauses (expected >= {}). \
         The scanner is broken, not the tree — a green here would have been vacuous.",
        clauses.len(),
        MIN_EXPECTED_CLAUSES
    );
    for probe in MUST_DETECT_AS_COVERED {
        assert!(
            cited.contains_key(*probe),
            "SCANNER IS BLIND: `{probe}` is cited by tests in this crate, but the \
             test-region scan did not see it. Every 'uncovered' result below is \
             therefore untrustworthy. Suspect the #[cfg(test)] brace tracking."
        );
    }
    // ---- end positive control

    let exempt: BTreeSet<&str> = EXEMPT.iter().map(|(v, _)| *v).collect();
    let uncovered: Vec<&String> = clauses
        .iter()
        .filter(|c| !cited.contains_key(*c) && !exempt.contains(c.as_str()))
        .collect();

    assert!(
        uncovered.is_empty(),
        "FKC clause->test totality FAILED — {} clause(s) can be raised by the \
         validator but no test provokes them:\n{}\n\n\
         This is the 'existence != enforcement' class: a rejection nothing \
         proves is reachable. Either add a test that provokes it, or add an \
         entry to EXEMPT with a reason a reader can check.",
        uncovered.len(),
        uncovered.iter().map(|c| format!("  - FkcError::{c}")).collect::<Vec<_>>().join("\n")
    );
}

/// **§6.1-0003 direction — citation → clause.** An exemption naming a clause
/// that no longer exists is a build-visible failure, not a silent no-op.
///
/// Without this, deleting a variant leaves a stale exemption behind that
/// silently pre-authorises a *future* variant of the same name to skip
/// coverage. That is precisely the hand-maintained-matrix drift §6.1-0004
/// forbids.
#[test]
fn no_orphan_exemptions() {
    let clauses = derive_clauses();
    assert!(clauses.len() >= MIN_EXPECTED_CLAUSES, "clause derivation collapsed");

    let orphans: Vec<&str> = EXEMPT
        .iter()
        .map(|(v, _)| *v)
        .filter(|v| !clauses.contains(*v))
        .collect();

    assert!(
        orphans.is_empty(),
        "orphan exemption(s) naming clauses that no longer exist: {orphans:?}\n\
         Remove them. A stale exemption silently pre-authorises a future clause \
         of the same name to skip coverage."
    );
}

/// **Over-inclusion control + staleness ratchet.** Every exemption must still
/// be *necessary*: an exempt clause that has since acquired a test is a stale
/// exemption and must be deleted.
///
/// This is deliberately the mirror of the positive control in
/// `every_clause_is_cited_by_a_test`, and it exists because that control has a
/// blind spot. `MUST_DETECT_AS_COVERED` catches a scanner that sees *nothing*
/// — but a scanner that over-includes (e.g. mis-tracking regions so production
/// code counts as test code) marks **everything** covered, reports zero
/// uncovered clauses, and sails through every check above. That failure mode is
/// indistinguishable from success by any assert that only looks for coverage.
///
/// The exempt set is the natural negative control: these are known *not* to be
/// cited, so if the scanner claims they are, it is over-including. The same
/// assert keeps the table honest as the tree changes — an exemption that
/// quietly stops being needed gets removed instead of accumulating.
#[test]
fn exemptions_are_still_necessary_and_the_scanner_discriminates() {
    let clauses = derive_clauses();
    let cited = cited_by_tests(&clauses);
    assert!(clauses.len() >= MIN_EXPECTED_CLAUSES, "clause derivation collapsed");

    let now_covered: Vec<&str> = EXEMPT
        .iter()
        .map(|(v, _)| *v)
        .filter(|v| cited.contains_key(*v))
        .collect();

    assert!(
        now_covered.is_empty(),
        "exemption(s) no longer needed — these are now cited by tests: {now_covered:?}\n\n\
         Two possible causes, and they need opposite responses:\n\
         (a) someone wrote the test — DELETE the entry from EXEMPT so the clause \
         is held to real coverage from now on; or\n\
         (b) the scanner is OVER-INCLUDING (counting production code as test \
         code), in which case every 'covered' verdict in this file is worthless \
         and the region tracking is broken. Rule out (b) before assuming (a)."
    );
}

/// Every exemption must carry a real reason. A blank or placeholder reason
/// re-creates the hand-waved matrix this whole gate exists to prevent.
#[test]
fn every_exemption_states_a_reason() {
    for (variant, reason) in EXEMPT {
        let r = reason.trim();
        assert!(
            r.len() >= 40,
            "exemption for `{variant}` has no substantive reason ({} chars). \
             State why it cannot or should not be provoked.",
            r.len()
        );
        assert!(
            !r.to_ascii_uppercase().contains("TODO") && !r.to_ascii_uppercase().contains("FIXME"),
            "exemption for `{variant}` is a placeholder, not a decision"
        );
    }
}
