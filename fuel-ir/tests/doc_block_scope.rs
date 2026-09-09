// SPDX-License-Identifier: MIT OR Apache-2.0
//! **GAP-302, increment 1: the BLOCK-AWARE SCOPE CLASSIFIER, and nothing else.**
//!
//! A doc/code drift gate over `docs/architecture/` flags a backticked identifier
//! that no `*.rs` contains. **A line-scoped version of that gate is wrong, and
//! wrong in the direction that destroys evidence.**
//!
//! # The incident this exists to prevent, which nearly happened
//!
//! `docs/architecture/02-layers.md:62` reads *"As-built note (2026-07-29):
//! `fuel-nn` is a planned crate, not a shipped one… there is no `fuel-nn`
//! directory… the NN surface currently lives in `fuel-core` as `lazy_nn_*`
//! modules."* **Every clause of that is false today and inverted: `fuel-nn`
//! shipped, and the `lazy_nn_*` modules were deleted.**
//!
//! A line-scoped scan reports nine drifted names and prescribes renaming them.
//! **That would have been a catastrophe**, because four lines above it sits a
//! `SUPERSEDED` marker and a sentence saying the note is *retained* — it records
//! a real consumer injury, and it is the only evidence in the corpus that this
//! defect class ever cost anyone. **Correcting it deletes the evidence, and the
//! result looks like a document getting tidier.**
//!
//! The reader-side lesson generalises past markdown: **a `grep` returns a LINE,
//! and a line is a RENDERING of the block that gives it its meaning.** Going to
//! the line is not going to the artefact.
//!
//! # Two classes, one mechanism
//!
//! ```text
//! QUOTED-HISTORICAL   the name was true WHEN QUOTED; the quote is the evidence
//! CITED-AS-ABSENT     the sentence's POINT is that the name is missing
//!                     (`10-decisions-log.md`: "obvious names (`const_fold` /
//!                      `constant_fold` / `fold_const`); the only hit,")
//! ```
//!
//! **Both are invisible to any rule keyed on the NAME**, because the
//! discriminator is the surrounding block. **And an allowlist is the wrong
//! instrument for both**: it would accumulate a list of names that are
//! *correct*, which is the worst possible thing to keep — every entry a
//! permanent claim that a true statement is an exception.
//!
//! # Scope of THIS file
//!
//! **The classifier and its fixtures. Not the drift arm.** The drift arm needs a
//! marker population that does not exist yet (`**UNBUILT** (GAP-NNN)`, zero
//! conforming instances at `64e5f46e`), and a gate landed ahead of its
//! population is unfalsifiable at exactly the moment everyone is looking at it.
//!
//! # What this classifier CANNOT see, stated on its face
//!
//! **Markdown LAZY CONTINUATION.** CommonMark lets a wrapped paragraph inside a
//! blockquote omit the `>` on its continuation lines. This classifier reads
//! such a line as depth 0 and would end the block early — mis-dispositioning
//! retained material as live prose, the dangerous direction. **Measured at
//! `64e5f46e`: ZERO instances in `docs/architecture/`** (no depth-0 non-blank
//! line directly follows a depth-1 non-blank one). **Deciding it properly needs
//! a real markdown parser, so it is recorded as a limit rather than guessed
//! at** — and if the count ever moves off zero, this is the arm to build.
//!
//! **Fixtures are CONSTRUCTED, never sampled from the live corpus**, per the
//! standing rule that a gate must not source its cases from data the work is
//! actively changing: the marker pass is about to rewrite every disclosure line
//! in `docs/architecture/`, so a classifier calibrated on them **expires by that
//! pass succeeding**.

/// What the drift arm may say about a name found on this line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Live prose. A name here is the gate's business.
    InScope,
    /// Depth >= 2 — a quote inside a quote. Retained history by construction.
    OutNested,
    /// Depth 1, but the block it belongs to is marked superseded/retained.
    OutRetired,
}

const RETIRED_MARKERS: &[&str] = &[
    "SUPERSEDED",
    "STRUCK",
    "retained below",
    "RETAINED BELOW",
    "DISCHARGED",
];

/// Leading blockquote depth: the number of `>` markers before any content.
fn depth(line: &str) -> usize {
    let mut d = 0;
    for ch in line.chars() {
        match ch {
            '>' => d += 1,
            c if c.is_whitespace() => {}
            _ => break,
        }
    }
    d
}

/// Strip a line's blockquote markers and leading space, leaving its content.
fn strip_quote_prefix(line: &str) -> &str {
    line.trim_start_matches(|c: char| c == '>' || c.is_whitespace())
}

/// Classify every line by the scope its BLOCK gives it.
///
/// A block is a maximal run of consecutive lines at depth >= 1; a depth-0 line
/// ends it. **Retirement is a property of the BLOCK, not the line** — the
/// marker sits at the top and the retained material below it, which is exactly
/// why a line-scoped reading gets `02-layers.md:62` wrong.
pub fn classify(lines: &[&str]) -> Vec<Scope> {
    let d: Vec<usize> = lines.iter().map(|l| depth(l)).collect();
    let mut out = vec![Scope::InScope; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        if d[i] == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < lines.len() && d[i] >= 1 {
            i += 1;
        }
        // ⚠️ JOINED, NOT PER-LINE. A multi-word marker split across a hard
        // wrap -- `retained` ending one line and `below` starting the next --
        // is invisible to a per-line `contains`, and it fails in the DANGEROUS
        // direction: the block reads as NOT retired, so retained history is
        // classified IN SCOPE and the drift arm prescribes editing it.
        //
        // Measured at `64e5f46e`: 9 blockquote blocks in docs/architecture/,
        // ZERO where per-line and joined disagree. That null has a positive
        // control -- a constructed block with `retained below` spanning a wrap
        // gives per_line=false, joined=true -- so the zero is a real absence
        // and not a broken comparison. The join lands anyway: the corpus is
        // about to be rewritten by the marker pass, and a latent hazard in a
        // classifier is cheaper to remove than to remember.
        let joined = lines[start..i]
            .iter()
            .map(|l| strip_quote_prefix(l))
            .collect::<Vec<_>>()
            .join(" ");
        let retired = RETIRED_MARKERS.iter().any(|m| joined.contains(m));
        for k in start..i {
            out[k] = if d[k] >= 2 {
                Scope::OutNested
            } else if retired {
                Scope::OutRetired
            } else {
                Scope::InScope
            };
        }
    }
    out
}

// ---- fixtures -------------------------------------------------------------
//
// One fixture per decision, per the ruling. The FOURTH is the discriminating
// one and the one a hand-built set gets wrong by omission: depth 1 with no
// supersession above it is LIVE PROSE and fully in scope. Two real sites live
// exactly there (`05-backend-contract.md:401` and `:446`), so it is not an
// academic row.

#[test]
fn depth_zero_live_prose_is_in_scope() {
    let f = ["Ordinary prose naming `Gap302FixtureAlpha`.", "More prose."];
    assert_eq!(classify(&f), vec![Scope::InScope, Scope::InScope]);
}

#[test]
fn a_nested_quote_is_out_of_scope() {
    let f = [
        "> Outer block, live.",
        ">",
        "> > **As-built note (2026-07-29): `fuel-nn` is a planned crate.**",
    ];
    let got = classify(&f);
    assert_eq!(got[2], Scope::OutNested, "depth 2 must be OUT: {got:?}");
}

#[test]
fn a_block_marked_superseded_is_out_of_scope() {
    let f = [
        "> ⚠️ **AS-BUILT NOTE SUPERSEDED 2026-08-27 — `fuel-nn` HAS SHIPPED.**",
        ">",
        "> The original note is retained below.",
    ];
    assert_eq!(
        classify(&f),
        vec![Scope::OutRetired, Scope::OutRetired, Scope::OutRetired]
    );
}

#[test]
fn depth_one_without_a_marker_is_in_scope() {
    // THE DISCRIMINATING CASE. Modelled on `05-backend-contract.md:401`/`:446`,
    // which are depth-1 as-built notes with no supersession above them: live
    // prose that the gate must read, not retained history it must skip.
    let f = [
        "> ⚠️ **AS-BUILT 2026-08-28 — GENUINELY UNBUILT. FILE AS WORK.**",
        "> Measured at head: ZERO `fn` definitions for `Gap302FixtureDelta`.",
    ];
    assert_eq!(classify(&f), vec![Scope::InScope, Scope::InScope]);
}

// ---- over-scope arms ------------------------------------------------------
//
// These are the arms actually under test today. The classifier's population is
// constructed, so "it fires" proves little; what needs proving is that it does
// NOT reach past the block it is deciding about. A retirement that leaks is the
// failure that silently shrinks the gate's corpus, and a shrunken corpus reads
// as a clean gate.

#[test]
fn retirement_does_not_leak_into_the_next_block() {
    let f = [
        "> **SUPERSEDED — retained below.**",
        "> > the quoted history",
        "",
        "> A LATER, UNRELATED quoted block naming `Gap302FixtureBeta`.",
    ];
    let got = classify(&f);
    assert_eq!(got[0], Scope::OutRetired);
    assert_eq!(got[1], Scope::OutNested);
    assert_eq!(
        got[3],
        Scope::InScope,
        "a blank line ends the block; retirement must not carry over: {got:?}"
    );
}

#[test]
fn retirement_does_not_leak_into_following_prose() {
    let f = [
        "> **SUPERSEDED 2026-08-27.**",
        "> > the quoted history",
        "",
        "Ordinary prose naming `Gap302FixtureGamma`, which the gate must read.",
    ];
    assert_eq!(classify(&f)[3], Scope::InScope);
}

#[test]
fn a_marker_below_still_retires_the_block_it_is_in() {
    // Retirement is a property of the BLOCK, so the marker need not be first.
    // Written as its own arm because the obvious implementation -- decide on
    // the first line and carry forward -- passes every arm above and fails this.
    let f = [
        "> An opening line with no marker at all.",
        "> **SUPERSEDED 2026-08-27.**",
    ];
    assert_eq!(classify(&f), vec![Scope::OutRetired, Scope::OutRetired]);
}

#[test]
fn a_marker_split_across_a_hard_wrap_still_retires_the_block() {
    // ⚠️ THE WRAP-WIDTH ARM. A line-anchored scan measures the AUTHOR'S WRAP
    // WIDTH, not the document -- the Claim Auditor measured exactly this on
    // `docs/` (10 files line-anchored vs 16 paragraph-joined, four of five
    // defects straddling a wrap). Here `retained below` spans the break, so no
    // single line contains it, and a per-line implementation calls the block
    // LIVE. That is the dangerous direction: retained history would be handed
    // to the drift arm as prose to edit.
    let f = [
        "> **The original note is retained",
        "> below because its recorded injury is the point.**",
        "> > **As-built note (2026-07-29): `fuel-nn` is a planned crate.**",
    ];
    let got = classify(&f);
    assert_eq!(
        got[0],
        Scope::OutRetired,
        "wrapped marker must still retire: {got:?}"
    );
    assert_eq!(got[1], Scope::OutRetired);
    assert_eq!(got[2], Scope::OutNested);
}

#[test]
fn depth_counts_markers_not_indentation() {
    assert_eq!(depth("no quote"), 0);
    assert_eq!(depth("> one"), 1);
    assert_eq!(depth("> > two"), 2);
    assert_eq!(depth(">> two, unspaced"), 2);
    assert_eq!(depth("    > still one, indented"), 1);
}

// ===========================================================================
// GAP-302 increment 2: the POPULATION RATCHET.
//
// The drift arm's full predicate is "backticked, absent from *.rs, AND
// UNDISCLOSED". ⚠️ THE DISCLOSURE HALF IS A DECLARED UNBUILT SEAM, NOT AN
// OVERSIGHT: measured at 64e5f46e, the undisclosed count is 17 under the
// marker vocabulary I assembled by reading and 9 under an equally natural
// wider one -- `future`, `planned`, `will become`, `proposed`, `candidate`.
// EIGHT of 26 names flip on that choice. A gate whose flag set halves on an
// unruled word list would ship the exact defect GAP-302 exists to catch: a
// number that looks measured and is a choice.
//
// !! A GATE BORN INERT BY DECLARATION, WITH ITS MISSING HALF NAMED, IS A
// DIFFERENT OBJECT FROM ONE THAT IS ACCIDENTALLY VACUOUS. This is the first.
//
// TRIGGER THAT FILLS THE SEAM: the architect's marker pass makes the predicate
// mechanical -- `**UNBUILT** (GAP-NNN)` present or absent -- at which point the
// vocabulary question DISAPPEARS rather than being answered, and this ratchet
// is replaced by the real arm.
//
// WHAT IS ASSERTED IN THE MEANTIME, and it is not nothing: the POPULATION is
// pinned. Any name entering or leaving reddens. That catches a NEW drift the
// day it lands and a FIX the day it lands, neither of which needs a disclosure
// predicate.
//
// !! THIS IS A RATCHET, NOT AN ALLOWLIST, and the difference is load-bearing.
// An allowlist says "these are permitted exceptions" and grows silently. A
// ratchet says "this is the measured population; any change needs a look" and
// reddens in BOTH directions. Same shape as check-gaps-table.py's arity
// ratchet. Every member here is ADJUDICATED in the registry (3 real drift,
// 3 external, 4 cited-as-absent, 1 concept-not-type, 15 proposed-future);
// this file carries the SET, the registry carries the CLASSES, and they are
// deliberately not both -- two sources of truth for one adjudication would
// diverge.
// ===========================================================================

/// The gate's own path. ⚠️ EXCLUDED FROM THE `*.rs` SCAN, AND THIS IS NOT
/// TIDINESS -- IT IS THE DIFFERENCE BETWEEN A WORKING GATE AND A BLIND ONE.
///
/// ⚠️ THIS EXCLUSION AND THE DATA FILE SOLVE **DIFFERENT** PROBLEMS FOR
/// **DIFFERENT** BENEFICIARIES, AND NEITHER REPLACES THE OTHER.
///
/// * `gap302_known_absent.txt` keeps the population OUT of `*.rs`, so
///   EVERY OTHER INSTRUMENT -- the GAP-302 census, any future scan, tools that
///   do not exist yet -- reads the corpus correctly with no exclusion to
///   remember. That is the structural fix.
/// * `SELF_PATH` protects THIS gate against whatever a future edit puts back.
///   It is defence in depth, and it is why the probe below is CONSTRUCTED
///   rather than sampled from the population.
///
/// THE HISTORY, KEPT BECAUSE THE MECHANISM RECURS: the fixtures used to name
/// real drifted identifiers, and the population used to be a `&[&str]` here.
/// Measured with the exclusion: 26 names; without it: 17 -- NINE hidden by the
/// fixtures alone. It was FOUR when found; writing the explanation added five,
/// **because explaining which names drift requires naming them.** One of the
/// hidden was the name this entire gate exists because of -- a researcher built
/// a proposal on it -- so **the gate would have been permanently blind to its
/// own motivating example.**
///
/// ⚠️ AND THE NINE WAS NOT THE WHOLE STORY, WHICH IS THE PART WORTH KEEPING.
/// Nine is the count an instrument WITH this exclusion hides. An `*.rs`-wide
/// census that does NOT self-exclude sees all **26**, because the list literal
/// counted too -- and that is what took the GAP-302 census to 0 on main. **Two
/// correct numbers about two different scans; the first did not transfer,
/// because it was reported without naming which scan it ranged over.**
///
/// **Note the direction throughout: self-poisoning made the population
/// SMALLER.** A gate that quietly under-reports is the flattering failure, and
/// nothing in a green run would have contradicted it.
const SELF_PATH: &str = "fuel-ir/tests/doc_block_scope.rs";

/// A CONSTRUCTED witness that `SELF_PATH` still matches a file on disk.
///
/// It appears in exactly one place in the tree -- this line -- and in no
/// document, so it can never acquire a real referent and can never be a
/// population member. That is the point: the arm proving the self-exclusion is
/// load-bearing must not take its evidence from the defect the exclusion exists
/// to fix, or fixing the defect silently makes the arm vacuous.
const SELF_EXCLUSION_PROBE: &str = "Gap302SelfExclusionProbe";

/// The adjudicated in-scope absent set, read from a NON-RUST data file.
///
/// ⚠️ THE NAMES USED TO BE STRING LITERALS HERE, AND THAT MADE EVERY ONE OF
/// THEM RESOLVE IN ANY `*.rs` SCAN. They are absent from `*.rs` BY DEFINITION
/// -- that is what makes them members -- so writing them into a `.rs` file
/// falsified the property the list exists to record. Measured: the GAP-302
/// census went 25 -> 0 when this file landed on main (`f2bfcb04`), with all 25
/// resolving here.
///
/// `SELF_PATH` protected THIS gate and did nothing for anyone else's
/// instrument. Moving the names out of `*.rs` DISSOLVES the coupling instead
/// of asking every other tool to remember an exclusion.
///
/// The data file also carries the prose that has to name members (the R5
/// derivation), which is what made the list grow 4 -> 9 in its first hour.
fn known_absent() -> Vec<&'static str> {
    include_str!("gap302_known_absent.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}
fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must sit under the workspace root")
        .to_path_buf()
}

fn rel(p: &std::path::Path, root: &std::path::Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Build output and VCS metadata: never source, and `target/` alone would
/// dwarf the corpus.
fn is_pruned_dir(p: &std::path::Path) -> bool {
    p.file_name()
        .is_some_and(|n| n == "target" || n == ".git" || n == "node_modules")
}

/// A `*.rs` path that belongs in the scanned corpus. The `skip_self` arm is
/// the self-exclusion; see `SELF_PATH` for why it is load-bearing.
fn is_scanned_source(p: &std::path::Path, root: &std::path::Path, skip_self: bool) -> bool {
    p.extension().is_some_and(|e| e == "rs") && !(skip_self && rel(p, root) == SELF_PATH)
}

/// Every `*.rs` under the workspace root, EXCEPT this file when `skip_self`.
fn rust_files(skip_self: bool) -> Vec<std::path::PathBuf> {
    let root = repo_root();
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        // A discovery step that can fail must FAIL, not silently shrink its
        // corpus: a truncated file list makes every name look ABSENT, which is
        // this gate's false-positive direction.
        let rd = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir({}) failed: {e}", dir.display()));
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if !is_pruned_dir(&p) {
                    stack.push(p);
                }
            } else if is_scanned_source(&p, &root, skip_self) {
                out.push(p);
            }
        }
    }
    out
}

/// Every identifier token in every `*.rs`, EXCEPT this file. See `SELF_PATH`.
fn rust_tokens(skip_self: bool) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for p in rust_files(skip_self) {
        if let Ok(src) = std::fs::read_to_string(&p) {
            for t in idents(&src) {
                out.insert(t);
            }
        }
    }
    out
}

/// `[A-Za-z_][A-Za-z0-9_]*` runs.
fn idents(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(src[s..i].to_string());
        } else {
            i += 1;
        }
    }
    out
}

/// R5: leading capital, NO underscore, at least one lowercase-or-digit, and at
/// least two capitals.
///
/// Keeps `DType` / `SType` / `QMatMul` (leading acronyms). Drops `MINOR` and
/// `NORMATIVE` (prose caps), `CUBLAS_WORKSPACE_CONFIG` and
/// `VK_EXT_memory_budget` (underscored). Chosen by measuring four candidate
/// rules over one population, not by fitting to the examples that prompted it.
///
/// !! THERE IS DELIBERATELY NO "at least two capitals" CLAUSE, and the first
/// version of this function had one. That constraint is exactly what R5 exists
/// to REMOVE -- it drops five real doc-only identifier claims, one of them a
/// documented enum whose variants do not exist. The rule was measured in one
/// language and re-implemented in another, and the constraint came back
/// silently in the port; the population ratchet caught it on its FIRST RUN,
/// naming all five.
///
/// ⚠️ THE FIVE ARE NAMED IN `gap302_known_absent.txt`, NOT HERE, AND THAT
/// IS THE POINT: they are population members, so naming them in a `*.rs` file
/// would make them resolve and drop them from the population. Explaining a
/// drift requires naming it, which is exactly how this file poisoned its own
/// corpus the first time.
fn is_identifier_claim(t: &str) -> bool {
    t.len() > 1
        && t.starts_with(|c: char| c.is_ascii_uppercase())
        && t.chars().all(|c| c.is_ascii_alphanumeric())
        && t.chars()
            .any(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
}

fn backtick_parity_is_inside(line: &str, byte_at: usize) -> bool {
    line[..byte_at].chars().filter(|c| *c == BACKTICK).count() % 2 == 1
}

const BACKTICK: char = '`';

/// In-scope, absent, R5-shaped names across `docs/architecture/*.md`.
///
/// MEMBERSHIP IS BY BACKTICK PARITY, NOT BY A SPAN REGEX. A regex pairing
/// single backticks cannot pair a RUN of three (a fenced-language marker
/// written inline), so the pairing shifts and every later span on the line is
/// offset. Measured: that manufactured a 392-character "span" out of prose and
/// admitted the English word `Refines` as an identifier. Parity is what
/// markdown actually does.
fn in_scope_absent() -> std::collections::BTreeSet<String> {
    let root = repo_root();
    let present = rust_tokens(true);
    let mut out = std::collections::BTreeSet::new();
    let dir = root.join("docs/architecture");
    let rd = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir({}) failed: {e}", dir.display()));
    for entry in rd.flatten() {
        let p = entry.path();
        if !p.extension().is_some_and(|e| e == "md") {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&p) else {
            continue;
        };
        let lines: Vec<&str> = src.lines().collect();
        let scope = classify(&lines);
        for (i, line) in lines.iter().enumerate() {
            if scope[i] != Scope::InScope {
                continue;
            }
            let b = line.as_bytes();
            let mut j = 0;
            while j < b.len() {
                if b[j].is_ascii_alphabetic() || b[j] == b'_' {
                    let s = j;
                    while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                        j += 1;
                    }
                    let t = &line[s..j];
                    if backtick_parity_is_inside(line, s)
                        && is_identifier_claim(t)
                        && !present.contains(t)
                    {
                        out.insert(t.to_string());
                    }
                } else {
                    j += 1;
                }
            }
        }
    }
    out
}

#[test]
fn the_absent_population_is_exactly_the_adjudicated_set() {
    let found = in_scope_absent();
    let known: std::collections::BTreeSet<String> =
        known_absent().iter().map(|s| (*s).to_string()).collect();
    let new: Vec<&String> = found.difference(&known).collect();
    let gone: Vec<&String> = known.difference(&found).collect();
    assert!(
        new.is_empty(),
        "NEW doc/code drift -- {} backticked name(s) in docs/architecture/ with no \
         referent in *.rs and no entry here: {new:?}. Adjudicate each in \
         docs/gaps.md (GAP-302), then add it to gap302_known_absent.txt.",
        new.len()
    );
    assert!(
        gone.is_empty(),
        "{} name(s) in the adjudicated set now HAVE a referent, or left the corpus: \
         {gone:?}. If the drift was fixed, delete the entry. A ratchet reddens in \
         BOTH directions on purpose -- a stale entry is a claim nobody re-derives.",
        gone.len()
    );
}

#[test]
fn excluding_this_file_is_load_bearing_and_changes_the_population() {
    // THE OVER-SCOPE ARM THAT MATTERS -- now on a CONSTRUCTED probe.
    //
    // ⚠️ IT USED TO SAMPLE ITS NEGATIVE CASE FROM THE POPULATION, asserting
    // that four real drifted names appear in `*.rs` when this file is
    // scanned. That held ONLY BECAUSE THE FILE WAS POISONING THE CORPUS --
    // the arm's evidence WAS the defect, so fixing the defect would have
    // made it vacuous. A gate cannot source its negative case from the thing
    // it exists to fix.
    //
    // `SELF_EXCLUSION_PROBE` exists for no other purpose, appears in no
    // document, and can never acquire a real referent -- so this arm keeps
    // discriminating now that the population has left `*.rs`.
    let with_self = rust_tokens(false);
    let without = rust_tokens(true);
    assert!(
        with_self.contains(SELF_EXCLUSION_PROBE),
        "the probe must appear in *.rs when THIS file is scanned; if not, it \
         was renamed and this arm no longer tests anything"
    );
    assert!(
        !without.contains(SELF_EXCLUSION_PROBE),
        "the probe must vanish once this file is excluded; if it survives, \
         SELF_PATH is not matching this file and the exclusion is inert"
    );
    assert!(
        without.len() < with_self.len(),
        "excluding one file must remove tokens; it removed none, so SELF_PATH no \
         longer matches any file on disk"
    );
}

#[test]
fn parity_membership_rejects_a_name_outside_every_span() {
    // The `Refines` case, CONSTRUCTED rather than sampled: a name in bare prose
    // on a line that also carries inline code and a three-backtick run.
    let line = "a `Foo` and a ```rust``` marker. Refines an already-stated claim.";
    let at = line.find("Refines").unwrap();
    assert!(
        !backtick_parity_is_inside(line, at),
        "an even backtick count before the token means OUTSIDE every span"
    );
    let at2 = line.find("Foo").unwrap();
    assert!(
        backtick_parity_is_inside(line, at2),
        "control: a genuinely backticked name must read as INSIDE"
    );
}

/// ⚠️ THE GATE MUST NOT NAME ITS OWN SUBJECTS -- THIS IS THE ARM THAT PROTECTS
/// EVERY INSTRUMENT THAT IS NOT THIS ONE.
///
/// `SELF_PATH` makes THIS gate correct while it poisons the corpus. It does
/// nothing for a `*.rs`-wide census that has no self-exclusion, and that is not
/// hypothetical: when this file first landed on main (`f2bfcb04`) the GAP-302
/// census went from 25 absent names to **0**, with all 25 resolving here.
///
/// So the population lives in `gap302_known_absent.txt` and this arm keeps
/// it there. It reddens the moment a future edit writes a member back into the
/// Rust -- as a fixture, an example, or a helpful explanation -- which is
/// exactly how the list grew 4 -> 9 the first time.
///
/// It cannot be satisfied by adding an exclusion somewhere else, which is the
/// point: an exclusion is a coupling someone must remember, and this is a
/// property the file either has or does not.
#[test]
fn this_file_names_none_of_the_population() {
    let path = repo_root().join(SELF_PATH);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()));
    let mut offenders: Vec<&str> = Vec::new();
    for name in known_absent() {
        if idents(&src).iter().any(|t| t == name) {
            offenders.push(name);
        }
    }
    assert!(
        offenders.is_empty(),
        "{} adjudicated name(s) are written into this *.rs file: {offenders:?}.          Every one of them RESOLVES in any *.rs scan, which falsifies the very          property that makes them population members -- and SELF_PATH hides that          from this gate while leaving every other instrument wrong. Put the name          in gap302_known_absent.txt, where prose may reference it freely.",
        offenders.len()
    );

    // POSITIVE CONTROL: the check must be able to FIND a name in this file at
    // all. The probe is in this file by construction, so a run that cannot see
    // it is a broken reader, not a clean result.
    assert!(
        idents(&src).iter().any(|t| t == SELF_EXCLUSION_PROBE),
        "the tokenizer cannot see a token known to be in this file, so the          emptiness above is a reader defect, not a finding"
    );
}
