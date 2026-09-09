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
const UNBUILT_CLAIMS: &[&str] = &["design pass", "no code yet"];

fn specs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must have a parent directory")
        .join("docs/specs")
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
        .find(|p| p.trim_start().starts_with("**Status:**"))
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
}

#[test]
fn no_spec_status_field_claims_the_work_is_unbuilt() {
    let dir = specs_dir();
    let mut files = Vec::new();
    markdown_files(&dir, &mut files);

    // Non-vacuity: a wrong path or a moved directory would pass this test having
    // examined nothing, and would look identical to a clean tree.
    assert!(
        files.len() >= 10,
        "only {} markdown files under {} — the scanner is broken, not the specs",
        files.len(),
        dir.display()
    );

    let with_status: Vec<_> = files
        .iter()
        .filter_map(|f| status_paragraph(&std::fs::read_to_string(f).unwrap()).map(|s| (f, s)))
        .collect();
    assert!(
        with_status.len() >= 5,
        "only {} of {} spec files have a **Status:** field — the extractor is broken; \
         an empty population passes every check below having examined nothing",
        with_status.len(),
        files.len()
    );

    let mut bad = Vec::new();
    for (f, para) in &with_status {
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
         file, so a reader who stops there never reaches a correction placed below it — \
         the label does not merely fail to protect the truth, it defends the falsehood \
         against the correct statement elsewhere in the same document.\n\n\
         FIX: correct the Status field itself. Keep the old wording lower down, marked \
         DISCHARGED, as the record of when it was found. Do not add a third annotation \
         layer above the falsehood.",
        bad.join("\n  ")
    );
}
