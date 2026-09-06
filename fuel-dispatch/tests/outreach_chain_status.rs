// SPDX-License-Identifier: MIT OR Apache-2.0
//! **An outreach reply cannot still be unsent once its own successor exists.**
//!
//! `docs/outreach/` carries ask/reply chains — `<base>-ask.md`, `<base>-reply.md`,
//! `<base>-reply-2.md`, `<base>-reply-3.md` — and each document leads with a
//! `**Status:**` line. Nothing checked those lines against the chain they sit in.
//!
//! Measured 2026-09-06 at `0656f2c6`:
//!
//! ```text
//! reply.md    **Status:** DRAFT for CireSnave review before it goes to Baracuda.
//! reply-2.md  **Status:** RELAYED to Baracuda (CireSnave, 2026-07-15)
//! reply-3.md  **Status: item closed, mutual.** ... no remaining open items.
//! ```
//!
//! The first line had been false for seven weeks. **A `-reply-2` cannot exist before
//! `-reply` was sent** — its own text names `-reply` as its predecessor — so the
//! contradiction is decidable from the directory listing alone, without reading a
//! word of prose or asking anyone what happened.
//!
//! ⚠️ The external refutation was stronger still and is why this is worth a gate
//! rather than a one-off fix: the document is *in Baracuda's repository*, and the
//! recipient **prepended a receipt line** — *"Fuel-authored reply, received on
//! Baracuda's side + approved by CireSnave"* — four lines above a status saying it
//! had not yet gone to them. **Two contradictory claims about one event, written by
//! the two parties to it.** No detector inside Fuel could see that; this one sees the
//! local half, which is the half Fuel can keep true.
//!
//! This is GAP-283's shape — a false line surviving because nothing checked it —
//! narrowed to the one case that is mechanically decidable.
//!
//! # Why blockquotes are excluded
//!
//! A `DISCHARGED` note has to QUOTE the wording it retires, or a reader cannot tell
//! what was corrected. A whole-file scan would fire on the retraction itself and
//! could only be satisfied by deleting the history, so only non-blockquote lines
//! count as the document's own claim.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-dispatch must sit under the workspace root")
        .to_path_buf()
}

fn outreach_dir() -> PathBuf {
    repo_root().join("docs").join("outreach")
}

/// Lines outside a blockquote — the document's own claims, not the history it quotes.
fn live_body(doc: &str) -> String {
    doc.lines()
        .filter(|l| !l.trim_start().starts_with('>'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `**Status` line of a document's live body, if it has one.
///
/// `None` rather than an empty string, so a caller distinguishes "no status field"
/// from "a status field that says nothing".
fn status_line(doc: &str) -> Option<String> {
    live_body(doc)
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("**Status"))
        .map(str::to_string)
}

/// Does this status claim the document has NOT yet reached its counterparty?
///
/// Deliberately narrow. `DRAFT` alone is not enough — a document can legitimately be
/// a draft of something never intended to be sent. What is contradictory is a claim
/// of *not yet delivered* sitting in a chain that demonstrably continued.
fn claims_unsent(status: &str) -> bool {
    let s = status.to_ascii_lowercase();
    s.contains("before it goes to")
        || s.contains("not yet sent")
        || s.contains("not sent")
        || s.contains("unsent")
        || s.contains("before sending")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("cannot read {} — {e}", p.display()))
}

/// Chains present in `docs/outreach/`: a `<base>-reply.md` that has at least one
/// `<base>-reply-<n>.md` successor.
fn chains_with_successors() -> Vec<(PathBuf, Vec<String>)> {
    let dir = outreach_dir();
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| {
        panic!(
            "cannot read {} — {e}; this test scanned nothing",
            dir.display()
        )
    });

    let names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".md"))
        .collect();

    let mut out = Vec::new();
    for n in &names {
        let Some(base) = n.strip_suffix("-reply.md") else {
            continue;
        };
        let successors: Vec<String> = names
            .iter()
            .filter(|m| {
                m.strip_prefix(base)
                    .and_then(|r| r.strip_prefix("-reply-"))
                    .and_then(|r| r.strip_suffix(".md"))
                    .is_some_and(|d| d.chars().all(|c| c.is_ascii_digit()) && !d.is_empty())
            })
            .cloned()
            .collect();
        if !successors.is_empty() {
            out.push((dir.join(n), successors));
        }
    }
    out
}

#[test]
fn an_outreach_reply_with_a_successor_does_not_claim_to_be_unsent() {
    let chains = chains_with_successors();

    // Positive control on the premise: if no chain has a successor, the loop below
    // asserts nothing and passes. That must fail loudly instead — the directory is
    // supposed to contain multi-round chains, and an empty result means the naming
    // convention moved, not that everything is well.
    assert!(
        !chains.is_empty(),
        "no `<base>-reply.md` with a `<base>-reply-<n>.md` successor found in {} — \
         the outreach naming convention changed and this check now verifies nothing",
        outreach_dir().display()
    );

    let mut violations = Vec::new();
    for (path, successors) in &chains {
        let Some(status) = status_line(&read(path)) else {
            continue; // no status field is a different question, not this one
        };
        if claims_unsent(&status) {
            violations.push(format!(
                "{}\n      status:     {status}\n      successors: {}",
                path.strip_prefix(repo_root()).unwrap_or(path).display(),
                successors.join(", ")
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "OUTREACH REPLY CLAIMS TO BE UNSENT WHILE ITS OWN SUCCESSOR EXISTS:\n\n  {}\n\n\
         A `-reply-2` cannot precede `-reply` being delivered — it names `-reply` as its \
         predecessor. Update the earlier document's status to what the chain records. \
         Quoting the retired wording inside a DISCHARGED blockquote is intended and is \
         not counted.",
        violations.join("\n\n  ")
    );
}

#[test]
fn the_predicates_can_fail_rather_than_returning_a_false_clean_result() {
    assert_eq!(status_line("no status here"), None);
    assert_eq!(
        status_line("**Status:** SENT").as_deref(),
        Some("**Status:** SENT")
    );
    // A status inside a blockquote is quoted history, not the document's claim.
    assert_eq!(
        status_line("> **Status:** DRAFT before it goes to X\n\nbody"),
        None
    );

    assert!(claims_unsent(
        "**Status:** DRAFT before it goes to Baracuda"
    ));
    assert!(claims_unsent("**Status:** not yet sent"));
    // ⚠️ `DRAFT` alone must NOT trip it: a document can be a draft of something never
    // meant to be sent, and over-firing here would push people to delete honest
    // status fields rather than keep them accurate.
    assert!(!claims_unsent("**Status:** DRAFT for review"));
    assert!(!claims_unsent(
        "**Status:** SENT AND CLOSED. Relayed 2026-07-15"
    ));

    assert_eq!(live_body("> quoted\n> more").trim(), "");
    assert_eq!(live_body("> quoted\nlive").trim(), "live");
}

#[test]
fn the_outreach_directory_this_check_reads_is_where_it_expects() {
    let dir = outreach_dir();
    assert!(
        dir.is_dir(),
        "{} is missing — this check has lost its subject",
        dir.display()
    );
}
