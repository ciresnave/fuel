//! GAP-036 increment 2 — **clause IDs on the FKC spec PROSE, derived by
//! parsing the spec, bidirectionally gated.**
//!
//! Increment 1 (`tests/fkc_clause_coverage.rs`) anchors on the `FkcError`
//! variant set and answers *"is every ENFORCED rule tested?"*. It is blind by
//! construction to a normative claim that raises no typed error — a MUST that
//! binds the **planner**, the **corpus**, the **graph registry**, or a **Rust
//! API shape** rather than contract validation. This file covers exactly that
//! remainder and nothing else.
//!
//! # One surface, one anchor
//!
//! The FKC spec's *validator* rules are deliberately NOT numbered here. They
//! are already anchored on `FkcError` variants and gated by increment 1;
//! numbering them too would double-anchor one surface, and two anchors on one
//! surface is how a coverage matrix starts disagreeing with itself. If you are
//! about to add a `[FKC-…]` token to a rule that produces a typed error, don't
//! — increment 1 already owns it.
//!
//! # The §6.1-0004 hazard this file had to design around
//!
//! Clause IDs on prose *reintroduce* the exact thing KISS-Conform §6.1-0004
//! forbids: a hand-authored list that can silently diverge from the document it
//! claims to describe. A sidecar written **alongside** the spec is a
//! hand-maintained matrix wearing a derived matrix's clothes — worse than no
//! matrix, because it carries the authority of the mechanism without the
//! property.
//!
//! So there is **no sidecar file**. The authoritative clause set is obtained by
//! parsing the spec for `[FKC-<section>-<NNNN>]` tokens at test time. There is
//! nothing to maintain, nothing to forget to update, and no second copy to
//! drift. Deleting a clause from the prose deletes it from the set; a citation
//! left behind then fails `no_orphan_citations`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Clauses whose enforcement genuinely lives outside this crate. Recorded with
/// the location so "not cited here" never silently reads as "unenforced".
const ENFORCED_ELSEWHERE: &[(&str, &str)] = &[(
    "FKC-12.5-0001",
    "G1 recipe totality is a GRAPH-side invariant, not an FKC-import one: the \
     `decompose`/`pattern` pair and the primitive-basis closure live in \
     fuel-graph, with parity + gap-posture tests in fuel-core/src/lazy.rs. \
     Citing it from fuel-dispatch would assert coverage this crate does not \
     provide.",
)];

/// **Divergences: the spec asserts a MUST the implementation deliberately does
/// not satisfy.** Recorded, not silently tolerated and not silently "fixed".
///
/// This category is why increment 2 earns its keep. Increment 1 cannot see
/// these — no typed error is involved — and neither can a reader, because the
/// spec reads as though the rule holds. `docs/architecture` already calls
/// doc-vs-code drift a defect; this is the mechanism that detects it.
const DIVERGENCES: &[(&str, &str)] = &[(
    "FKC-10.10-0001",
    "SPEC SAYS: `register_full_with_source` MUST become `Result` (never-panic), \
     a 'firm prerequisite' of the FKC import path. CODE DOES: it returns `()`, \
     and so does `register_full_with_source_generic`. \
     RULED (architect, 2026-08-08): amend the SPEC, not the signature. The \
     never-panic goal IS met, by a different route — registration is \
     append-only and genuinely infallible; the duplicate-`KernelRef` case is \
     detected once by `finalize()`, which returns `Result`, and the FKC \
     importer maps it to a typed `FkcError::DuplicateKernelRef` and propagates \
     with `?` (register.rs, 'never-panic on the import path'). VERIFIED against \
     the code rather than inherited from the ruling. Forcing a `Result` here \
     would make every caller handle an error that cannot occur, and an \
     always-`Ok` Result trains readers to stop reading Results. \
     SCOPE LIMIT, stated because the claim is narrower than it looks: the \
     never-panic property holds for the IMPORT path. The hand-written \
     static-table init path still `.expect()`s on `finalize()` (two sites), a \
     deliberate init-boundary fail-fast on a programmer error in checked-in \
     tables. The defect is the DRIFT, not the signature.",
)];

/// A parse that finds far fewer clauses than the spec carries is broken, and a
/// broken parse reports **full coverage over an empty set** — a green that
/// means nothing. Raise this as clauses are added.
const MIN_EXPECTED_CLAUSES: usize = 5;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn spec_path() -> PathBuf {
    repo_root().join("docs/specs/kernel-contract-format.md")
}

/// Derive the authoritative clause set **by parsing the spec**, never from a
/// checked-in list. This is the whole §6.1-0004 property; if you are tempted to
/// cache this into a file, re-read the module docs.
fn derive_clauses_from_prose() -> BTreeSet<String> {
    let src = std::fs::read_to_string(spec_path())
        .unwrap_or_else(|e| panic!("cannot read the FKC spec at {}: {e}", spec_path().display()));
    // `match_indices` yields BYTE offsets. An earlier version walked a
    // `Vec<char>` index and sliced the byte-indexed `str` with it, which
    // panics the moment the spec contains a multi-byte character — and this
    // spec is full of em dashes. Byte offsets throughout, no mixing.
    let mut out = BTreeSet::new();
    for (start, _) in src.match_indices("[FKC-") {
        let after = &src[start + 1..];
        let Some(end) = after.find(']') else { continue };
        let tok = &after[..end];
        // Shape: FKC-<section>-<4 digits>, section being digits and dots.
        let Some(rest) = tok.strip_prefix("FKC-") else { continue };
        let Some((sec, num)) = rest.rsplit_once('-') else { continue };
        if !sec.is_empty()
            && sec.chars().all(|c| c.is_ascii_digit() || c == '.')
            && num.len() == 4
            && num.chars().all(|c| c.is_ascii_digit())
        {
            out.insert(tok.to_string());
        }
    }
    out
}

/// Scan the crate for `FKC-CLAUSE:` citations, returning clause -> files.
fn citations() -> BTreeMap<String, Vec<String>> {
    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut stack = vec![
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests"),
    ];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&p) else { continue };
            // Skip this file: it NAMES every clause (in DIVERGENCES /
            // ENFORCED_ELSEWHERE) without testing any of them. Counting itself
            // would make the gate self-satisfying — the purest vacuity.
            if p.file_name().is_some_and(|n| n == "fkc_prose_clause_coverage.rs") {
                continue;
            }
            let label = p.file_name().unwrap().to_string_lossy().to_string();
            for line in src.lines() {
                let Some(idx) = line.find("FKC-CLAUSE:") else { continue };
                for tok in line[idx + "FKC-CLAUSE:".len()..].split_whitespace() {
                    let tok = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '.');
                    if tok.starts_with("FKC-") {
                        let entry = found.entry(tok.to_string()).or_default();
                        if !entry.contains(&label) {
                            entry.push(label.clone());
                        }
                    }
                }
            }
        }
    }
    found
}

/// **clause → test.** Every numbered prose clause resolves to a citing test, an
/// enforced-elsewhere record, or a recorded divergence. Nothing is silent.
#[test]
fn every_prose_clause_is_cited_or_recorded() {
    let clauses = derive_clauses_from_prose();
    let cited = citations();

    assert!(
        clauses.len() >= MIN_EXPECTED_CLAUSES,
        "prose clause parse collapsed: {} found, expected >= {}. The spec moved \
         or the token shape changed — a green here would be coverage over an \
         empty set.",
        clauses.len(),
        MIN_EXPECTED_CLAUSES
    );

    let elsewhere: BTreeSet<&str> = ENFORCED_ELSEWHERE.iter().map(|(c, _)| *c).collect();
    let diverged: BTreeSet<&str> = DIVERGENCES.iter().map(|(c, _)| *c).collect();

    let unaccounted: Vec<&String> = clauses
        .iter()
        .filter(|c| {
            !cited.contains_key(*c)
                && !elsewhere.contains(c.as_str())
                && !diverged.contains(c.as_str())
        })
        .collect();

    assert!(
        unaccounted.is_empty(),
        "prose clause->test totality FAILED — {} numbered clause(s) with no test, \
         no enforced-elsewhere record, and no divergence entry:\n{}\n\n\
         Add `FKC-CLAUSE: <id>` to the test that enforces it, or record why it \
         is not enforced here.",
        unaccounted.len(),
        unaccounted.iter().map(|c| format!("  - {c}")).collect::<Vec<_>>().join("\n")
    );
}

/// **citation → clause (§6.1-0003).** A citation naming a clause the spec no
/// longer carries is a hard failure.
///
/// This is what makes deleting prose safe: remove a clause and its stale
/// citations surface immediately, instead of sitting in the tree asserting
/// coverage of a rule that no longer exists.
#[test]
fn no_orphan_citations() {
    let clauses = derive_clauses_from_prose();
    assert!(clauses.len() >= MIN_EXPECTED_CLAUSES, "prose clause parse collapsed");

    let mut orphans: Vec<String> = Vec::new();
    for (c, files) in citations() {
        if !clauses.contains(&c) {
            orphans.push(format!("{c} (cited in {})", files.join(", ")));
        }
    }
    for (c, _) in ENFORCED_ELSEWHERE.iter().chain(DIVERGENCES.iter()) {
        if !clauses.contains(*c) {
            orphans.push(format!("{c} (recorded in this file's tables)"));
        }
    }

    assert!(
        orphans.is_empty(),
        "citation(s) naming clauses the spec does not carry:\n{}\n\n\
         Either the clause ID was removed/renamed in the spec, or the citation \
         has a typo. A citation that resolves to nothing asserts coverage of a \
         rule that does not exist.",
        orphans.join("\n")
    );
}

/// Divergence and enforced-elsewhere entries must state something checkable.
/// These categories exist to make an uncomfortable fact visible; a one-word
/// reason re-hides it.
#[test]
fn recorded_exceptions_state_their_reasoning() {
    for (clause, reason) in DIVERGENCES.iter().chain(ENFORCED_ELSEWHERE.iter()) {
        assert!(
            reason.trim().len() >= 80,
            "record for `{clause}` is too thin ({} chars) to be checkable",
            reason.trim().len()
        );
        assert!(
            !reason.to_ascii_uppercase().contains("TODO"),
            "record for `{clause}` is a placeholder, not a decision"
        );
    }
    // A divergence must say what the spec claims AND what the code does —
    // otherwise a later reader cannot tell which side to change.
    for (clause, reason) in DIVERGENCES {
        let up = reason.to_ascii_uppercase();
        assert!(
            up.contains("SPEC SAYS") && up.contains("CODE DOES"),
            "divergence `{clause}` must state both sides: what the spec asserts \
             and what the code actually does"
        );
    }
}
