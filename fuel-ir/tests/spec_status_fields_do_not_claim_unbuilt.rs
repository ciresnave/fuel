// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A spec's `**Status:**` field must not say the work is unbuilt when it is built.**
//!
//! This is GAP-283's class, and GAP-283 is the case this guard exists because of:
//! three specs read *"Design pass — no code yet"* against live FKC implementation.
//! An auditor found that on 2026-08-20, wrote an accurate notice into each file, and
//! **left the false sentence in the Status line.** Sixteen days later the KISS
//! architect read the Status line, reported FKC unimplemented, and scoped a ratified
//! cross-project parity gate against it.
//!
//! # Why this checks the STATUS FIELD and not the file
//!
//! ⚠️ **A file-wide ban would fire on its own remedy.** The three published specs were
//! corrected on 2026-09-05, and their `DISCHARGED` notes *quote* the retired wording as
//! the record of what was wrong. Banning the phrase everywhere would redden the fix and
//! push the next person to delete the history instead of the falsehood.
//!
//! **The defect was never the phrase. It was the phrase IN THE POSITION THAT CLAIMS
//! AUTHORITY** — a reader who stops at `Status:` never reaches the correction below it.
//! So the scope is exactly one paragraph per file: the one beginning `**Status:**`.
//!
//! # What this does NOT cover — read before trusting a green
//!
//! - **Only `docs/specs/`.** `ROADMAP.md`, `docs/architecture/` and `docs/gaps.md` carry
//!   the same phrase and are NOT checked here; they were held by open PRs when this was
//!   written. **A green here is not "no spec in fuel claims to be unbuilt".**
//! - **Only this vocabulary.** *"Design pass"* and *"no code yet"*. A status line saying
//!   *"not started"* against shipped code passes this and is the same defect.
//! - **It cannot tell whether the subject is built.** It bans the CLAIM from the
//!   authority position; it does not verify implementation. A genuinely unbuilt spec
//!   should say so somewhere that is not the Status field, or this guard is wrong for it.
//!
//! Population at authoring time: 5 violations — 4 in `_drafts/` plus one orphaned
//! `"Design pass —"` fragment left in `storage-encoding.md` BY the 2026-09-05 correction
//! itself. ⚠️ **The 2026-09-05 sweep fixed the three instances it was shown and did not
//! search for siblings. A finding that names one instance of a shape is a SAMPLE.**

use std::path::{Path, PathBuf};

/// The phrases that assert unbuilt-ness. Deliberately short; see the scope note above.
/// ⚠️ **"not started" was tried here and REMOVED: it does not discriminate.**
/// `gap-029-persistent-decode-trait.md` reads *"Increment 1 landed; increments 2+ not
/// started"* — precise, true, and exactly the kind of honest partial status a guard must
/// not punish. A vocabulary that fires on true statements is a nag, not a detector.
/// The dead-branch half of the same defect is caught structurally by the second test.
const UNBUILT_CLAIMS: &[&str] = &[
    "design pass",
    "no code yet",
    "before code lands",
    "pending implementation plan",
];

/// The directories this guard ranges over.
///
/// ⚠️ **`docs/architecture/`, `ROADMAP.md` and `docs/gaps.md` are NOT here and they
/// carry the same defects.** They were held by open PRs when this was written. **A green
/// here is a fact about two directories, not about fuel's documentation.**
const SCAN_DIRS: &[&str] = &["docs/specs", "docs/session-prompts", "docs/superpowers"];

/// `docs/` itself, NON-recursively — its subdirectories are either listed above or
/// deliberately out of scope.
///
/// ⚠️ `docs/gaps.md` and `docs/method-rules.md` live here and are held by open PRs.
/// They are not excluded: measured at `16577dc1`, neither carries a status FIELD this
/// guard recognises (0 each), so they are out of the population on the facts rather
/// than by an exemption. If either grows one, this guard will start checking it.
const SCAN_ROOT: &str = "docs";

fn scan_dirs() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must have a parent directory");
    SCAN_DIRS.iter().map(|d| root.join(d)).collect()
}

fn all_status_fields() -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    for d in scan_dirs() {
        markdown_files(&d, &mut files);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must have a parent directory")
        .join(SCAN_ROOT);
    for e in std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("control: {} must be readable: {e}", root.display()))
        .flatten()
    {
        let p = e.path();
        if p.is_file() && p.extension().is_some_and(|x| x == "md") {
            files.push(p);
        }
    }
    assert!(
        files.len() >= 20,
        "only {} markdown files across {SCAN_DIRS:?} — the scanner is broken, not the docs",
        files.len()
    );
    let mut out = Vec::new();
    let mut superseded = 0usize;
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        // ⚠️ Order matters. Counting the exemption BEFORE asking whether the file has a
        // status field at all inflated it from 4 to 10: files with no field and an
        // incidental "superseded" blockquote were being counted as exemptions.
        let Some(s) = status_paragraph(&text) else {
            continue;
        };
        if has_supersession_banner(&text) {
            superseded += 1;
            continue;
        }
        out.push((f.clone(), s));
    }
    assert!(
        superseded <= 6,
        "{superseded} files now carry a supersession banner and are exempt. A banner is not \
         a way to retire a status field from scrutiny — re-read them before raising this."
    );
    assert!(
        out.len() >= 15,
        "only {} status fields found in {} files — the extractor is broken, and an empty \
         population passes every check below having examined nothing",
        out.len(),
        files.len()
    );
    out
}

fn markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| {
        panic!(
            "control: {} must be readable, or this test scans nothing: {e}",
            dir.display()
        )
    });
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            markdown_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "md") {
            out.push(p);
        }
    }
}

/// Is this paragraph a status FIELD?
///
/// ⚠️ The first version required the literal `**Status:**`. Fuel writes the field four
/// ways — `**Status:**`, `**Status:** x`, `**Status: x**`, `**Status**: x` — and the
/// `**Status: x**` form is the one all four session-prompt files in this PR use. **So the
/// guard's population silently excluded the very files the PR was fixing**, and it would
/// have reported green over them forever.
fn is_status_field(para: &str) -> bool {
    let s = para
        .trim_start()
        .trim_start_matches([' ', '>', '*', '_', '-', '\t']);
    let low = s.to_ascii_lowercase();
    let Some(rest) = low.strip_prefix("status") else {
        return false;
    };
    rest.trim_start_matches('*').trim_start().starts_with(':')
}

/// Does this document announce, up front, that it has been superseded?
///
/// ⚠️ **Scoped on the BANNER, not on the path — and the difference is not cosmetic.**
/// A rule reading `_drafts/` is out of scope would have been right by accident and wrong
/// by mechanism: two of the four drafts carried this banner and two did not, and the two
/// WITHOUT it were exactly the ones whose status fields were misleading readers. **A
/// path-scoped exemption would have excused them permanently and silently, because an
/// exemption reads as deliberate.** The banner is the property that actually varies.
///
/// Vocabulary deliberately matches the block-aware scanner in fuel's doc-drift work, so
/// two independently-built detectors do not drift apart on the same discriminator.
fn has_supersession_banner(text: &str) -> bool {
    // ⚠️ CRLF FIRST, and this bit me twice. `status_paragraph` normalised and this did
    // not, so on a checkout where git had written CRLF the paragraph split here found
    // nothing, `cut` became the whole file, and every incidental "superseded" blockquote
    // counted as a banner. The guard passed on an LF tree and failed on a Windows one:
    // an ENVIRONMENT-DEPENDENT gate, which is worse than no gate because CI would have
    // been green while a local run was red, and the disagreement would have read as
    // flakiness rather than as a defect.
    let text = &text.replace("\r\n", "\n");
    // A BANNER is a blockquote, ABOVE the status field, announcing the document is
    // superseded. Both halves are load-bearing.
    //
    // ⚠️ The first version of this asked only whether a keyword appeared in the first
    // 1200 characters. That exempted 10 of 50 files instead of 4 — and TWO of the
    // extra six were exempted BY THE VERY CORRECTIONS THIS PR MAKES: a status line
    // reading "Retained as the implementation plan it was" and another reading
    // "SUPERSEDED BY ITS OWN ROLLOUT" both matched. **The remedy silently switched the
    // guard off for the file it had just fixed.** Caught only by the bound on the
    // exemption count, which is the entire reason that bound exists.
    //
    // Incidental prose about supersession is not a banner. Structure is the test.
    let cut = text
        .split("\n\n")
        .position(is_status_field)
        .map(|i| {
            text.split("\n\n")
                .take(i)
                .map(|p| p.len() + 2)
                .sum::<usize>()
        })
        .unwrap_or(text.len())
        .min(text.len());
    text[..cut]
        .lines()
        .filter(|l| l.trim_start().starts_with('>'))
        .any(|l| {
            let u = l.to_ascii_uppercase();
            ["SUPERSEDED", "STRUCK", "RETAINED AS", "RETAINED BELOW"]
                .iter()
                .any(|m| u.contains(m))
        })
}

/// The paragraph beginning `**Status:**`, with wrapping collapsed.
///
/// ⚠️ Paragraph-joined on purpose: a line-anchored scan measures the author's wrap
/// width, not the document. Four of the five violations this was written against
/// straddle a hard wrap, and a `grep` for the phrase found 10 files where a
/// paragraph-joined search found 16.
fn status_paragraph(text: &str) -> Option<String> {
    // These files are CRLF on a Windows checkout, so splitting on two bare newlines
    // finds NOTHING. The first run reported `0 of 15 files have a Status field` —
    // caught only by the non-vacuity floor below, which is why that floor exists.
    let text = text.replace("\r\n", "\n");
    text.split("\n\n")
        .find(|p| is_status_field(p))
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
}

#[test]
fn no_spec_status_field_claims_the_work_is_unbuilt() {
    let mut bad = Vec::new();
    for (f, para) in all_status_fields() {
        let lower = para.to_ascii_lowercase();
        for claim in UNBUILT_CLAIMS {
            if lower.contains(claim) {
                bad.push(format!("{}: Status field says {claim:?}", f.display()));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "a spec's **Status:** field claims the work is unbuilt:\n  {}\n\n\
         This is GAP-283's class. The Status field is the highest-authority text in the \
         file, so a reader who stops there never reaches a correction placed below it.\n\n\
         FIX: correct the Status field itself. Keep the old wording lower down, marked \
         DISCHARGED, as the record of when it was found.",
        bad.join("\n  ")
    );
}

/// **A status field must not point at a branch as the live location of the work.**
///
/// GAP-283's other half, and the one that cost the most: ten documents named
/// `feat/kernel-contracts-dlpack`, a branch that does not exist on origin. Following
/// such a line costs a failed checkout and **the natural conclusion that the work was
/// abandoned** — when in every case the subject was live on `main`.
///
/// A field may still MENTION a dead branch, as history. It must say it is dead.
#[test]
fn no_status_field_points_at_a_branch_as_the_live_location() {
    // ⚠️ Vocabulary, therefore a FLOOR and not a census. Each entry is an instance that was
    // found; "shipped on `" was added after `fused-op-registry.md` named a dead branch in a
    // phrasing the first three did not cover.
    const POINTERS: &[&str] = &[
        "branch `",
        "wip lands on `",
        "lands on branch `",
        "shipped on `",
    ];
    const DISCLAIMERS: &[&str] = &[
        "does not exist",
        "no longer exists",
        "used to name",
        "previously",
        "is not on origin",
    ];
    let mut bad = Vec::new();
    for (f, para) in all_status_fields() {
        let lower = para.to_ascii_lowercase();
        if POINTERS.iter().any(|p| lower.contains(p))
            && !DISCLAIMERS.iter().any(|d| lower.contains(d))
        {
            bad.push(format!("{}: Status field points at a branch", f.display()));
        }
    }
    assert!(
        bad.is_empty(),
        "a status field names a branch as where the work lives, with nothing saying \
         whether that branch still exists:\n  {}\n\n\
         A branch pointer rots silently: the branch is deleted on merge and the document \
         keeps sending readers to it. Say where the work IS, and keep the old pointer only \
         as history that names itself as history.",
        bad.join("\n  ")
    );
}
