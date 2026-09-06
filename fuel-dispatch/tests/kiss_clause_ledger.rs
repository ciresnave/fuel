// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The KISS clause ledger — a DECLARED obligation per cited clause, gated.**
//!
//! Fuel cites KISS clause ids in source comments, contracts, registry rows and
//! outreach. Until now **nothing checked any of it**: a citation could name a
//! clause Fuel no longer honours, or a renumbered clause, and it would sit
//! there indefinitely. That is GAP-283's shape — a false line surviving because
//! no detector exists.
//!
//! # Why this ASSERTS rather than INFERS
//!
//! The obvious instrument is *"count clauses named in a test"*. **It diverges in
//! both directions, and the under-count is the dangerous one.**
//!
//! Measured 2026-09-05 while designing this file: a census scoped to
//! `*/tests/*.rs` reported **1 of 16** clauses "named in a test", because Fuel's
//! unit tests live in `src/` under `#[cfg(test)]`. `KISS-OPS-6.16-0009` scored
//! **zero** — on the same day it was implemented with a born-red observed twice.
//! **The author of the instrument, measuring their own known-good work, got zero
//! and was structurally positioned to believe it.**
//!
//! The repair attempt was no better: *"cited in a file that contains `#[test]`"*
//! is **co-location, not linkage** — `byte_kernels.rs` has over 200 tests and
//! the citation is a module comment. **Proximity is the naming metric with a
//! longer reach.**
//!
//! So this ledger does not infer coverage from where a string appears. Each row
//! **declares** what the citation is for and, where it is an obligation, **names
//! the test that discharges it**.
//!
//! # The population is DERIVED, never hand-authored
//!
//! [`LEDGER`] rows are written by hand, but **which rows are REQUIRED is
//! computed from the citations in the tree** (`cited_clauses`). A citation with
//! no row fails `every_cited_clause_has_a_ledger_row`; a row citing nothing
//! fails `no_orphan_ledger_rows`. **An author cannot omit a row for a clause
//! they cited, because the citation is what demands it** — the same §6.1-0004
//! property `fkc_clause_coverage.rs` relies on, applied to a surface it does not
//! cover.
//!
//! # ⚠️ WHAT THIS LEDGER CANNOT CLAIM
//!
//! **It verifies the discharging test EXISTS and RUNS. It CANNOT verify the
//! clause still exists in KISS, or that its text still says what the row says.**
//! There is no KISS clause set in this repo to check against.
//!
//! A pinned KISS clause index is being prepared upstream. When it is vendored,
//! `Row::exists_at` stops being `None` and this paragraph becomes a check rather
//! than a disclaimer. **Do not delete the disclaimer before the check runs** — a
//! capability arriving is a reason to drop a caveat; the caveat being
//! inconvenient is not.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// What a citation of a KISS clause is DOING in this repo.
///
/// # ⚠️ Four values, deliberately, REPLACING a two-valued instruction
///
/// This gate was commissioned as *"every citation names a live test **or is
/// deleted**"*. **That is a forced choice between the two dispositions that
/// DESTROY EVIDENCE**, on a population that has four.
///
/// A `docs/outreach/` letter recording what Fuel asked KISS for in July is
/// **historical**: demanding a test for it is impossible, and deleting it
/// erases the record. Fuel's own rule for old names in prose already says this
/// — a mention is STALE (fix it), HISTORICAL (rewriting destroys the record and
/// self-erases), or PINNED (rewriting makes it false) — and **a swept
/// historical mention reads as correct while the evidence is gone.**
///
/// [`Reference`] and [`Declined`] are the two that would not exist under the
/// original instruction, and both are load-bearing: `KISS-OPS-6.16-0003` is
/// cited to say what is **permitted** (rounding a *computed* narrow result), so
/// there is nothing for Fuel to discharge; `KISS-CLASSIFY-6.6-0019` is a
/// deliberate **non**-conformance with a tracked gap.
///
/// The widening is enforced in **both** directions:
/// `obligation_and_declined_rows_name_a_test_that_exists_and_runs` also rejects
/// a `Reference` or `Record` row that NAMES a test — **a row that cannot be
/// discharged by a test must not be permitted to claim one**, or a `Record`
/// row quietly acquires a fake discharge and starts reading as parity evidence.
///
/// Ratified 2026-09-05. **If you are reading a four-valued gate against a
/// two-valued ruling, the mismatch is deliberate and this is why.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    /// Fuel claims to conform. **Must name a discharging test.**
    Obligation,
    /// Fuel deliberately does NOT conform, with a tracked reason.
    /// **Must name the test that keeps the decline honest.**
    Declined,
    /// Cited to explain why some behaviour is PERMITTED, or to locate a design
    /// in the standard. Nothing to discharge; a reason is still required.
    Reference,
    /// Correspondence, provenance, a decisions-log entry, a registry row, a
    /// plan. **Historical or administrative** — renaming or testing it would be
    /// wrong. A reason is still required.
    Record,
}
use Disposition::{Declined, Obligation, Record, Reference};

struct Row {
    clause: &'static str,
    disposition: Disposition,
    /// `"<repo-relative file>::<test fn name>"`, required for `Obligation` and
    /// `Declined`, forbidden otherwise.
    test: Option<&'static str>,
    /// Why this disposition. Required on every row without exception.
    reason: &'static str,
    /// Reserved for the vendored KISS clause index: the pinned ref at which the
    /// clause was confirmed to exist. `None` everywhere until the index lands.
    exists_at: Option<&'static str>,
}

/// One row per KISS clause id cited anywhere in the tree.
///
/// Adding a citation without a row is a BUILD FAILURE, which is the point.
const LEDGER: &[Row] = &[
    Row {
        clause: "KISS-CLASSIFY-6.1-0001",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row plus an outreach letter. The row TRACKS a \
                 classification divergence; the obligation is the gap's, not the code's.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CLASSIFY-6.1-0004",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CLASSIFY-6.3-0009",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CLASSIFY-6.6-0019",
        disposition: Declined,
        test: Some(
            "fuel-dispatch/tests/kiss_structure_key_byte_match.rs::the_i4_exclusion_still_has_its_reason",
        ),
        reason: "Fuel CANNOT express this cell: it pins <wdt> for an i4 weight and there is \
                 no `DType::I4` (GAP-097). The decline is recorded as a byte-match exclusion \
                 carrying its reason, and the named test asserts that reason survives.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CLASSIFY-6.7-0013",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CONFORM-6.4-0002",
        disposition: Record,
        test: None,
        reason: "fixtures/kiss-corpus/PROVENANCE.md — records WHICH clause the vendored \
                 corpus was drawn under. Provenance, not a conformance claim.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CONFORM-6.5-0008",
        disposition: Record,
        test: None,
        reason: "fixtures/kiss-corpus/PROVENANCE.md — provenance of the vendored corpus.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CONTRACT-6.4-0011",
        disposition: Obligation,
        test: Some("fuel-kernel-seam-types/src/shape_expr.rs::out_differs_from_operands"),
        reason: "Declared/computed output-shape consistency, with Gap never counting as a \
                 mismatch. Fuel implements it in `matmul_shape` and the seam's shape checker.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CONTRACT-6.7-0006",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
    Row {
        clause: "KISS-CONTRACT-6.9-0001",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row (GAP-038). Records what KISS PINS: the Provenance section is exactly five fields with no crate-version field, which is why Fuel needs no new front-matter field for the second version axis. A fact about KISS, not an obligation Fuel discharges.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.0-0003",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.15-0001",
        disposition: Obligation,
        test: Some("fuel-core/src/lazy.rs::fmax_fmin_ieee_suppress_nan_where_prop_propagates"),
        reason: "The four minmax ops must not be merged/aliased/substituted. Fuel has the \
                 NaN-PROPAGATING pair natively (Maximum/Minimum, torch parity) and RESOLVES the \
                 NaN-SUPPRESSING pair (fmax_ieee/fmin_ieee) through KISS's §6.13 decomposition; \
                 the named test pins that they SUPPRESS NaN where maximum/minimum propagate \
                 (GAP-048).",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.15-0002",
        disposition: Obligation,
        test: Some("fuel-cpu-backend/src/byte_kernels.rs::relu_f32_propagates_nan"),
        reason: "`relu` is `select(x<0, 0, x)`, so it PRESERVES a NaN rather than returning \
                 zero. Fuel matches torch here and the named test pins it.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.15-0003",
        disposition: Obligation,
        test: Some("fuel-core/src/lazy.rs::rem_trunc_diverges_from_floored_rem_on_opposite_signs"),
        reason: "rem_floor and rem_trunc must not be merged. Fuel has floored `rem` natively \
                 and RESOLVES rem_trunc through KISS's §6.13 decomposition `a - trunc(a/b)*b`; \
                 the named test pins the sign-of-dividend divergence from floored rem (GAP-048).",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.16-0003",
        disposition: Reference,
        test: None,
        reason: "Cited to say what is PERMITTED, not what Fuel owes: rounding a COMPUTED \
                 narrow result is licensed, which is why the compute ops keep the promoting \
                 path. There is no obligation here to discharge -- the obligation for the \
                 no-arithmetic ops is 6.16-0009, which carries its own row.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.16-0009",
        disposition: Obligation,
        test: Some(
            "fuel-cpu-backend/src/chassis/binary.rs::minmax_move_a_nan_operand_without_quieting_it",
        ),
        reason: "An op whose decomposition contains no arithmetic returns the MOVED operand, \
                 bits preserved exactly. `half` quiets a signalling NaN on BOTH conversion \
                 legs, so the narrow paths must never construct the f32. A second test, \
                 `dyn_impl.rs::cpu_unary_abs_moves_a_signalling_nan_without_quieting_it`, \
                 covers the dyn path the chassis tests cannot reach.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.16-0010",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.19-0005",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row (GAP-287). Cited to identify the clause a gap is about — the FDX flag semantics the validator was measured against.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.20-0002",
        disposition: Record,
        test: None,
        reason: "An outreach letter filing a registry extension, plus a superseded plan \
                 document. Both are HISTORICAL: they record what Fuel asked KISS for on a \
                 date. Rewriting or testing them would destroy the record.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.3-0002",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row (GAP-300). Records an obligation Fuel has \
                 NOT yet discharged: a consumer MUST be able to EVALUATE the pinned \
                 semantics of every primitive-floor op, and `trunc` is in KISS's \
                 43-op floor while fuel-graph::Op has no native one. The obligation \
                 is BEHAVIOURAL -- an expansion satisfies it -- but §6.3-0003 leaves \
                 nothing to resolve through, so the expansion is Fuel's own and no \
                 KISS vector can catch an error in it. Record, not Obligation: the \
                 row tracks the liability; the code change is GAP-048's PR.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.3-0003",
        disposition: Reference,
        test: None,
        reason: "Cited in `trunc`'s doc to explain WHY Fuel's trunc expansion is Fuel's own \
                 liability: a §6.3 floor op MUST NOT carry an in-standard reference \
                 decomposition, so §6.14-0004 resolution is EMPTY for a floor op and no KISS \
                 conformance vector catches an error in Fuel's expansion (GAP-300). Nothing to \
                 discharge — a Reference, not an Obligation.",
        exists_at: None,
    },
    Row {
        clause: "KISS-OPS-6.8-0001",
        disposition: Record,
        test: None,
        reason: "docs/gaps.md registry row. Cited to identify the clause a gap is about.",
        exists_at: None,
    },
];

// ---- derivation ----------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-dispatch must sit under the workspace root")
        .to_path_buf()
}

/// This file states every clause id in its own `LEDGER`, so scanning itself
/// would make every row self-justifying. Excluded, and
/// `the_scan_excludes_exactly_one_file` pins that the exclusion is this file
/// alone.
///
/// **Measured: removing this exclusion reddens THREE arms, not one**, and all
/// three are correct rather than redundant:
///
/// - `the_scan_excludes_exactly_one_file` — the direct claim;
/// - `the_scanner_discriminates` — its constructed negative
///   (`KISS-NOSUCH-9.9-9999`) lives in this file, so a self-scan turns the
///   fabricated id into a real citation and the "must be absent" assertion is
///   correctly violated;
/// - `every_cited_clause_has_a_ledger_row` — that same fabricated id then has
///   no row.
///
/// **The negative control is therefore self-defending: it cannot be quietly
/// neutralised by deleting the exclusion, because deleting the exclusion is
/// exactly what makes it fire.** Do not "fix" the over-detection by moving the
/// fabricated id out of this file — that would buy isolation by removing the
/// coupling that makes the control robust.
const SELF_PATH: &str = "fuel-dispatch/tests/kiss_clause_ledger.rs";

fn scanned_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        // A discovery step that can fail must FAIL, not silently shrink its
        // corpus: `else { return }` would truncate the file list and the gate
        // would assert completeness over a corpus that had quietly lost files.
        let rd = std::fs::read_dir(dir).unwrap_or_else(|e| {
            panic!(
                "scanned_files: read_dir({}) failed: {e} -- the citation corpus must FAIL, \
                 not silently truncate",
                dir.display()
            )
        });
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let skip = p
                    .file_name()
                    .is_some_and(|n| n == "target" || n == ".git" || n == "node_modules");
                if !skip {
                    walk(&p, files);
                }
            } else if p.extension().is_some_and(|e| e == "rs" || e == "md") {
                files.push(p);
            }
        }
    }
    walk(&repo_root(), &mut files);
    files
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo_root())
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Advance past a run of bytes satisfying `pred`.
fn scan(b: &[u8], mut j: usize, pred: impl Fn(u8) -> bool) -> usize {
    while j < b.len() && pred(b[j]) {
        j += 1;
    }
    j
}

/// Consume the byte `c` at `j`, or fail.
fn eat(b: &[u8], j: usize, c: u8) -> Option<usize> {
    (j < b.len() && b[j] == c).then_some(j + 1)
}

/// End index of the clause id beginning at `start`, which must sit on `KISS-`.
///
/// Split out of `clause_ids_in` rather than compressed: the grammar is four
/// guarded segments and expressing it as one nested condition chain is what
/// made the scanner unreadable (and tripped a complexity gate).
fn clause_id_end(b: &[u8], start: usize) -> Option<usize> {
    let mut j = start + "KISS-".len();

    let area = j;
    j = scan(b, j, |c| c.is_ascii_uppercase());
    if j == area {
        return None;
    }
    j = eat(b, j, b'-')?;

    let sec = j;
    j = scan(b, j, |c| c.is_ascii_digit() || c == b'.');
    // A section must be `<n>.<n>`: digits alone are not a clause id.
    if j == sec || !b[sec..j].contains(&b'.') {
        return None;
    }
    j = eat(b, j, b'-')?;

    let num = j;
    j = scan(b, j, |c| c.is_ascii_digit());
    (j > num).then_some(j)
}

/// Scan for `KISS-<AREA>-<n>.<n>-<nnnn>` without a regex crate.
fn clause_ids_in(src: &str) -> BTreeSet<String> {
    let b = src.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while let Some(off) = src[i..].find("KISS-") {
        let start = i + off;
        if let Some(end) = clause_id_end(b, start) {
            out.insert(src[start..end].to_string());
        }
        i = start + "KISS-".len();
    }
    out
}

/// clause id -> the repo-relative files citing it. **The authoritative
/// population**: every key here demands a ledger row.
fn cited_clauses() -> BTreeMap<String, Vec<String>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in scanned_files() {
        let label = rel(&f);
        if label == SELF_PATH {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&f) else {
            continue;
        };
        for id in clause_ids_in(&src) {
            map.entry(id).or_default().push(label.clone());
        }
    }
    map
}

/// Is `file::test_name` a real `#[test]` that is not `#[ignore]`d?
fn named_test_is_live(spec: &str) -> Result<(), String> {
    let (file, name) = spec
        .split_once("::")
        .ok_or_else(|| format!("`{spec}` is not `<file>::<test fn>`"))?;
    let path = repo_root().join(file);
    let src = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let lines: Vec<&str> = src.lines().collect();
    let needle = format!("fn {name}(");
    let at = lines
        .iter()
        .position(|l| l.trim_start().starts_with(&needle))
        .ok_or_else(|| format!("no `fn {name}(` in {file}"))?;
    let lo = at.saturating_sub(6);
    let attrs = &lines[lo..at];
    if !attrs.iter().any(|l| l.trim() == "#[test]") {
        return Err(format!("`{name}` in {file} is not a `#[test]`"));
    }
    if attrs.iter().any(|l| l.trim_start().starts_with("#[ignore")) {
        return Err(format!(
            "`{name}` in {file} is `#[ignore]`d -- it does not RUN"
        ));
    }
    Ok(())
}

fn row(clause: &str) -> Option<&'static Row> {
    LEDGER.iter().find(|r| r.clause == clause)
}

// ---- the scanner's blind spot, ENUMERATED rather than assumed away -------
//
// `clause_ids_in` anchors on the literal `KISS-`, so a citation written bare
// -- `6.15-0001` rather than `KISS-OPS-6.15-0001` -- is INVISIBLE to it. The
// gate below therefore asserts completeness over a corpus it does not fully
// see, and nothing in a green run says so.
//
// ## Why the obvious fix is WRONG, not merely expensive
//
// The intended remedy was "report matched/skipped and assert skipped == 0".
// Measured first, at `80def246`:
//
// ```text
// qualified   KISS-<AREA>-<n>.<n>-<nnnn>          48
// bare        <n>.<n>-<nnnn> with no KISS- prefix  570   <- 12x larger
// ```
//
// **A blanket `skipped == 0` is not merely infeasible at 570 -- it is wrong.**
// A bare section number is AMBIGUOUS ACROSS KISS DOCUMENTS: `6.4-0011` is
// KISS-Grammar, `6.1-0004` is KISS-ANNOUNCE, `7.4-0001` is KISS-Contract. The
// Ops ledger correctly does not track those, so most of the 570 are not its
// business, and a gate demanding they carry an Ops prefix would demand a
// falsehood.
//
// ## What IS checkable
//
// The narrow hazard is a clause number cited BOTH as `KISS-OPS-<n>` somewhere
// AND bare elsewhere: there the bare site is provably an Ops citation the
// scanner cannot see. Measured: 9 such numbers across 36 bare sites. All 9
// already have ledger rows -- the qualified citation is what earned them --
// so today they are redundant rather than dangerous.
//
// **The residual risk is a KISS-Ops clause cited ONLY bare.** That is
// UNDETECTABLE by any scanner, because the bare form carries no document
// identity. The remedy is a WRITING RULE (cite `KISS-OPS-<n>`, never bare,
// for an Ops clause) rather than a gate -- and this test keeps the rule
// honest by failing when the bare-cited SET GROWS.
//
// The assertion is on the SET OF CLAUSE IDS, never on a site count: ids do
// not rot, counts do.

/// Clause numbers (the `<n>.<n>-<nnnn>` tail) that the ledger tracks.
fn ledger_clause_numbers() -> BTreeSet<String> {
    LEDGER
        .iter()
        .filter_map(|r| r.clause.strip_prefix("KISS-OPS-"))
        .map(str::to_string)
        .collect()
}

/// End index of a bare `<n>.<n>-<nnnn>` beginning at `start`, or `None`.
fn bare_id_end(b: &[u8], start: usize) -> Option<usize> {
    let mut j = scan(b, start, |c| c.is_ascii_digit());
    if j == start {
        return None;
    }
    j = eat(b, j, b'.')?;
    let minor = j;
    j = scan(b, j, |c| c.is_ascii_digit());
    if j == minor {
        return None;
    }
    j = eat(b, j, b'-')?;
    let num = j;
    j = scan(b, j, |c| c.is_ascii_digit());
    (j - num == 4).then_some(j)
}

/// A bare `<n>.<n>-<nnnn>` starting exactly at `i`, or `None`.
///
/// Split out for the same reason `clause_id_end` was split out of
/// `clause_ids_in`: the guards are what push a single scanning function past
/// the complexity limit, and a nested chain is what made the original scanner
/// unreadable.
fn bare_candidate_at(b: &[u8], i: usize) -> Option<usize> {
    if !b[i].is_ascii_digit() {
        return None;
    }
    // A preceding `-` or `.` means this is the tail of a qualified
    // `KISS-OPS-...` id or of a longer number, not a bare citation.
    if i > 0 && (b[i - 1] == b'-' || b[i - 1] == b'.') {
        return None;
    }
    bare_id_end(b, i)
}

/// Every tracked clause number appearing BARE in one file's text.
fn bare_ids_in(src: &str, tracked: &BTreeSet<String>) -> BTreeSet<String> {
    let b = src.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0usize;
    while i < b.len() {
        match bare_candidate_at(b, i) {
            Some(end) => {
                let id = &src[i..end];
                if tracked.contains(id) {
                    out.insert(id.to_string());
                }
                i = end;
            }
            None => i += 1,
        }
    }
    out
}

/// Ledger clause numbers cited in BARE form, mapped to the files doing it.
fn bare_cited_ledger_clauses() -> BTreeMap<String, Vec<String>> {
    let tracked = ledger_clause_numbers();
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in scanned_files() {
        let label = rel(&f);
        if label == SELF_PATH {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&f) else {
            continue;
        };
        // A per-file set, so no de-duplication branch is needed here.
        for id in bare_ids_in(&src, &tracked) {
            out.entry(id).or_default().push(label.clone());
        }
    }
    out
}

/// Ledger clauses KNOWN to be cited bare somewhere, measured at `80def246`.
///
/// A DECLARATION of the scanner's blind spot, not an approval of it. A new
/// entry means someone cited a ledger clause without its `KISS-OPS-` prefix,
/// where the citation gate cannot see it -- qualify the citation rather than
/// extending this list.
const BARE_CITED_LEDGER_CLAUSES: &[&str] = &[
    "6.0-0003",
    "6.15-0001",
    "6.15-0002",
    "6.15-0003",
    "6.19-0005",
    "6.20-0002",
    "6.3-0002",
    "6.3-0003",
    "6.8-0001",
];

// ---- the gate ------------------------------------------------------------

#[test]
fn every_cited_clause_has_a_ledger_row() {
    let cited = cited_clauses();
    assert!(
        !cited.is_empty(),
        "the scanner found NO KISS clause citations at all -- it is broken, not the tree clean"
    );
    let missing: Vec<String> = cited
        .iter()
        .filter(|(id, _)| row(id).is_none())
        .map(|(id, files)| format!("{id}  cited in: {}", files.join(", ")))
        .collect();
    assert!(
        missing.is_empty(),
        "{} KISS clause id(s) are cited with NO ledger row. Add a row stating the \
         disposition (Obligation / Declined / Reference / Record) and its reason, or \
         delete the citation:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// The citation gate's blind spot must stay ENUMERATED.
///
/// `every_cited_clause_has_a_ledger_row` scans for `KISS-`-prefixed ids only.
/// This pins the set of ledger clauses that are ALSO cited bare, so the gap
/// cannot grow silently. It asserts on the ID SET rather than a site count --
/// ids do not rot and counts do.
#[test]
fn the_scanners_blind_spot_stays_enumerated() {
    // FOUNDATION: if the ledger yields no clause numbers, everything below is
    // vacuously empty and would pass while measuring nothing.
    let tracked = ledger_clause_numbers();
    assert!(
        !tracked.is_empty(),
        "ledger_clause_numbers() is EMPTY -- strip_prefix(\"KISS-OPS-\") matched no \
         row, so the bare-citation probe ranges over nothing and its silence is \
         meaningless"
    );

    let bare = bare_cited_ledger_clauses();
    let found: BTreeSet<&str> = bare.keys().map(String::as_str).collect();

    // FOUNDATION 2: the probe has been seen to find things. An empty result
    // here is a broken scanner, not a clean tree -- measured 9 at `80def246`,
    // asserted as a PREDICATE (non-empty) rather than a VALUE (== 9).
    assert!(
        !found.is_empty(),
        "the bare-citation probe found NOTHING across {} tracked clause \
         number(s) -- it is broken, not the tree clean",
        tracked.len()
    );

    let declared: BTreeSet<&str> = BARE_CITED_LEDGER_CLAUSES.iter().copied().collect();

    let undeclared: Vec<String> = found
        .difference(&declared)
        .map(|id| {
            let files = bare.get(*id).map(|v| v.join(", ")).unwrap_or_default();
            format!("{id}  cited bare in: {files}")
        })
        .collect();
    assert!(
        undeclared.is_empty(),
        "{} ledger clause(s) are cited WITHOUT their `KISS-OPS-` prefix, where \
         `every_cited_clause_has_a_ledger_row` CANNOT SEE THEM. Qualify the \
         citation (`KISS-OPS-<n>`) rather than adding to \
         BARE_CITED_LEDGER_CLAUSES -- a bare section number is ambiguous across \
         KISS documents and no scanner can resolve it:\n  {}",
        undeclared.len(),
        undeclared.join("\n  ")
    );

    let stale: Vec<&str> = declared.difference(&found).copied().collect();
    assert!(
        stale.is_empty(),
        "{} entr(ies) in BARE_CITED_LEDGER_CLAUSES no longer correspond to any \
         bare citation -- the citations were qualified or removed. Delete them; \
         a declaration that outlives its subject reads as a live exemption:\n  {}",
        stale.len(),
        stale.join("\n  ")
    );
}

#[test]
fn no_orphan_ledger_rows() {
    let cited = cited_clauses();
    let orphans: Vec<&str> = LEDGER
        .iter()
        .map(|r| r.clause)
        .filter(|c| !cited.contains_key(*c))
        .collect();
    assert!(
        orphans.is_empty(),
        "{} ledger row(s) name a clause NOTHING cites -- the citation was deleted and the \
         row outlived it. Delete the row:\n  {}",
        orphans.len(),
        orphans.join("\n  ")
    );
}

#[test]
fn obligation_and_declined_rows_name_a_test_that_exists_and_runs() {
    let mut bad = Vec::new();
    for r in LEDGER {
        let needs = matches!(r.disposition, Obligation | Declined);
        match (needs, r.test) {
            (true, None) => bad.push(format!(
                "{}: {:?} MUST name a discharging test",
                r.clause, r.disposition
            )),
            (false, Some(t)) => bad.push(format!(
                "{}: {:?} must NOT name a test (names `{t}`) -- a row that cannot be \
                 discharged by a test must not imply it was",
                r.clause, r.disposition
            )),
            (true, Some(t)) => {
                if let Err(e) = named_test_is_live(t) {
                    bad.push(format!("{}: {e}", r.clause));
                }
            }
            (false, None) => {}
        }
    }
    assert!(
        bad.is_empty(),
        "ledger rows are wrong:\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn every_row_states_a_reason() {
    let thin: Vec<&str> = LEDGER
        .iter()
        .filter(|r| r.reason.trim().len() < 40)
        .map(|r| r.clause)
        .collect();
    assert!(
        thin.is_empty(),
        "a ledger of bare ids degrades into noise and takes the gate's signal with it. \
         These rows state no usable reason:\n  {}",
        thin.join("\n  ")
    );
}

#[test]
fn no_row_yet_claims_the_clause_exists_upstream() {
    // The disclaimer in this file's header is load-bearing: nothing here can
    // check KISS. When the pinned clause index is vendored, THIS test is what
    // must be rewritten -- deliberately -- alongside the header.
    let claimed: Vec<&str> = LEDGER
        .iter()
        .filter(|r| r.exists_at.is_some())
        .map(|r| r.clause)
        .collect();
    assert!(
        claimed.is_empty(),
        "row(s) claim upstream existence with no vendored KISS clause index to check \
         against:\n  {}",
        claimed.join("\n  ")
    );
}

#[test]
fn the_scan_excludes_exactly_one_file() {
    let all = scanned_files();
    let self_hits = all.iter().filter(|p| rel(p) == SELF_PATH).count();
    assert_eq!(
        self_hits, 1,
        "the walker must SEE this file (it is excluded at scan time, not by being unreachable)"
    );
    for files in cited_clauses().values() {
        assert!(
            !files.iter().any(|f| f == SELF_PATH),
            "the ledger scanned itself -- every row would then justify its own existence"
        );
    }
}

#[test]
fn the_scanner_discriminates() {
    // Constructed negative: well-formed and absent. Not sourced from the tree,
    // so it cannot expire by the tree changing.
    let absent = clause_ids_in("see KISS-NOSUCH-9.9-9999 for details");
    assert_eq!(
        absent.iter().next().map(String::as_str),
        Some("KISS-NOSUCH-9.9-9999"),
        "the scanner must parse a well-formed id it has never seen"
    );
    assert!(
        !cited_clauses().contains_key("KISS-NOSUCH-9.9-9999"),
        "a fabricated clause id must not appear in the derived population"
    );

    // Malformed shapes must NOT parse -- otherwise the population inflates with
    // prose that merely starts with the prefix.
    for bad in ["KISS-", "KISS-OPS", "KISS-OPS-6-0009", "KISS-ops-6.1-0001"] {
        assert!(
            clause_ids_in(bad).is_empty(),
            "`{bad}` is not a clause id and must not parse as one"
        );
    }

    // Positive: a clause that IS cited, chosen because it appears in more than
    // one file -- a control on a token cited exactly once cannot see a scan that
    // stops at the first hit per id.
    let cited = cited_clauses();
    let multi = cited
        .iter()
        .find(|(_, files)| files.len() > 1)
        .map(|(id, files)| (id.clone(), files.len()));
    let (id, n) = multi.expect(
        "no clause is cited by more than one file -- the multi-file control cannot arm, \
         so this test would not detect a scanner that stops at the first hit",
    );
    assert!(n > 1, "{id} must be seen in {n} files");
}

#[test]
fn the_ledger_is_sorted_by_clause_id() {
    // The ledger is APPEND-ONLY by construction: every new KISS citation
    // anywhere in the tree obliges a new row here. Nothing enforced the order
    // until now, and it decayed the first time two lanes appended at once --
    // PR #106 and PR #120 both inserted after `KISS-CONTRACT-6.7-0006`, which
    // is the correct sorted position for exactly one of them. They conflicted
    // for no semantic reason: five rows apart in sort order, adjacent on disk.
    //
    // Sorting is therefore not tidiness. It makes a row's position a FUNCTION
    // OF ITS ID rather than of whichever anchor the author happened to be
    // editing near, so two lanes adding unrelated clauses cannot collide.
    //
    // `>=` rather than `>` is deliberate and does a second job: in a sorted
    // list, rejecting `w[0] >= w[1]` on adjacent pairs proves GLOBAL
    // UNIQUENESS, because equal ids could only ever be adjacent. That matters
    // because `row()` resolves with `.find()`, which silently returns the
    // first of a duplicated pair -- a shadowed row would otherwise never be
    // reported by anything in this file.
    let out_of_order: Vec<String> = LEDGER
        .windows(2)
        .filter(|w| w[0].clause >= w[1].clause)
        .map(|w| {
            if w[0].clause == w[1].clause {
                format!(
                    "{} appears twice (the second row is shadowed by `row()`)",
                    w[0].clause
                )
            } else {
                format!("{} must not precede {}", w[0].clause, w[1].clause)
            }
        })
        .collect();
    assert!(
        out_of_order.is_empty(),
        "LEDGER rows are sorted by clause id, so a row's position is decided by its id rather than by whichever anchor its author happened to edit near. Insert each new row where it sorts:\n  {}",
        out_of_order.join("\n  ")
    );
}
