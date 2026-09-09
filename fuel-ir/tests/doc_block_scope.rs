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
    let f = ["Ordinary prose naming `OptimizationMap`.", "More prose."];
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
        "> Measured at head: ZERO `fn` definitions for `ReferenceFactory`.",
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
        "> A LATER, UNRELATED quoted block naming `CostRegistry`.",
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
        "Ordinary prose naming `PrecisionFloor`, which the gate must read.",
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
