//! **GAP-236 PRECONDITION TRIPWIRE — a candidate provider may not flow kernels
//! into Fuel until the divergence-reaching admission probe is wired.**
//!
//! # Why this guard fires on the TRIGGER, not on the ABSENCE
//!
//! The obvious guard is "assert increment 2 has landed". That guard is RED
//! from the moment it is written until the work is done, so it must be
//! `#[ignore]`d to keep the suite green — and **an ignored test fires never**.
//! `jit_ingest_probe.rs::gap_236_the_probe_set_is_actually_consumed` is exactly
//! that test, and it is exactly that shape: correct, ignored, silent.
//!
//! This guard inverts the polarity. It asserts the *dangerous combination*:
//!
//! > something OUTSIDE `fuel-dispatch` refers to `IngestionService` /
//! > `CandidateKernel` (a provider can now flow candidates)
//! > **AND** `verify_candidate_impl` still does not call `admission_probes`
//! > (admission still runs on the seeded `[-0.5, 0.5)` fill alone).
//!
//! Today the first half is false — measured: both symbols occur only in
//! `fuel-dispatch/src/{jit_ingest.rs, jit_adopt.rs, lib.rs}`. So this test is
//! **GREEN today**, adds nothing to the existing red, and goes **RED at the
//! exact moment the risk becomes live** — when the approved Unpopped kernel
//! handback wires a provider before the probe wiring lands.
//!
//! That is the difference between a precondition that is *remembered* and one
//! that *fires*. The deadline "wired and forge-verified before the first
//! candidate can reach an `IngestionService`" is only enforceable if something
//! observes the event. This is that something.
//!
//! # What this guard deliberately does NOT do
//!
//! It does not verify that the wiring is *correct* — only that it is *present*.
//! Correctness is the forge-verified job of increment 2 itself. A guard that
//! claimed more than a source scan can establish would be a false guard with a
//! green test behind it.
//!
//! Known limitation, stated rather than smoothed: the scan skips `//` comment
//! lines but does not parse block comments, so a `/* ... */` mention outside
//! `fuel-dispatch` would read as a code reference. That direction is a false
//! RED — loud, and fixed by an allowlist entry when it happens. The silent
//! direction (a real provider that this scan cannot see) is the one that would
//! matter, and a plain textual reference cannot hide from it.

use std::path::{Path, PathBuf};

/// Symbols whose appearance outside `fuel-dispatch` means a provider can flow
/// candidates into the admission gate.
const TRIGGER_SYMBOLS: [&str; 2] = ["IngestionService", "CandidateKernel"];

/// The anchor proving the wiring scan is reading the admission block, not some
/// unrelated part of the file. Without it, "no call found" is a wrong-path
/// verdict wearing the costume of a missing wire.
const ADMISSION_BLOCK_ANCHOR: &str = "(1) Probe synthesis";

/// The call that closes GAP-236: the divergence-reaching probe set, consumed.
const WIRED_MARKER: &str = "admission_probes(";

/// This file's own basename. It names the trigger symbols in string literals
/// and lives outside `fuel-dispatch`, so it must exclude itself or it flags
/// itself. If this skip ever stops matching, the test goes RED on its own
/// path — loud and self-correcting, never silent.
const SELF: &str = "gap236_precondition.rs";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must have a parent workspace directory")
        .to_path_buf()
}

/// Recursively collect `.rs` files under `dir`, skipping build output, VCS
/// metadata, and any path component in `skip_dirs`.
fn rust_files(dir: &Path, skip_dirs: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if name == "target" || name == ".git" || name == ".claude" {
                continue;
            }
            if skip_dirs.contains(&name.as_str()) {
                continue;
            }
            rust_files(&path, skip_dirs, out);
        } else if name.ends_with(".rs") && name != SELF {
            out.push(path);
        }
    }
}

/// Lines that reference a trigger symbol as CODE (not in a `//` comment).
fn trigger_refs(files: &[PathBuf], root: &Path) -> Vec<String> {
    let mut hits = Vec::new();
    for path in files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        for (i, line) in src.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            if TRIGGER_SYMBOLS.iter().any(|s| line.contains(s)) {
                let rel = path.strip_prefix(root).unwrap_or(path);
                hits.push(format!("{}:{}", rel.display(), i + 1));
            }
        }
    }
    hits
}

/// The whole decision, as a pure function so both directions can be proven on
/// every run rather than only at authoring time.
///
/// `Err` is the dangerous state: a provider can flow candidates while
/// admission still runs on the seeded fill alone.
fn verdict(outside_refs: &[String], ingest_src: &str) -> Result<(), String> {
    if outside_refs.is_empty() {
        return Ok(());
    }
    if ingest_src.contains(WIRED_MARKER) {
        return Ok(());
    }
    Err(format!(
        "GAP-236 PRECONDITION VIOLATED — a candidate provider now exists OUTSIDE \
         `fuel-dispatch`, but `verify_candidate_impl` still does not call \
         `{WIRED_MARKER}`, so candidate admission runs on the seeded `[-0.5, 0.5)` \
         fill alone.\n\n\
         A kernel that lifts `fmaxf` (IEEE `maxNum`, NaN-SUPPRESSING) while honestly \
         claiming `Max` (NaN-PROPAGATING) is BIT-IDENTICAL to a correct one on every \
         input that probe can reach — measured across 256 seeds — and would be \
         ADMITTED.\n\n\
         Land GAP-236 increment 2 (wire `admission_probes` into the \
         `#[cfg(feature = \"cuda\")]` admission call site, forge-verified) BEFORE \
         letting candidates flow.\n\n\
         Trigger sites found ({} total):\n  {}",
        outside_refs.len(),
        outside_refs.join("\n  ")
    ))
}

#[test]
fn gap236_no_candidate_provider_before_the_probe_wiring() {
    let root = workspace_root();

    // POSITIVE CONTROL 1 — runs unconditionally, BEFORE any early exit, so this
    // test can never report `ok` having asserted nothing. It proves the walker
    // and the matcher are live by finding the trigger symbols where they are
    // known to be: inside `fuel-dispatch`.
    let mut inside_files = Vec::new();
    rust_files(&root.join("fuel-dispatch"), &[], &mut inside_files);
    let inside = trigger_refs(&inside_files, &root);
    assert!(
        !inside.is_empty(),
        "positive control FAILED: the scan found no reference to {TRIGGER_SYMBOLS:?} \
         inside `fuel-dispatch`, where they are known to live. The walker or the \
         matcher is broken, and a zero from the real scan below would be a broken \
         query rather than genuine absence."
    );

    // POSITIVE CONTROL 2 — the wiring check must be reading the admission block.
    let ingest_path = root.join("fuel-dispatch/src/jit_ingest.rs");
    let ingest_src = std::fs::read_to_string(&ingest_path)
        .unwrap_or_else(|e| panic!("cannot read {ingest_path:?}: {e}"));
    assert!(
        ingest_src.contains(ADMISSION_BLOCK_ANCHOR),
        "positive control FAILED: `{ADMISSION_BLOCK_ANCHOR}` not found in \
         {ingest_path:?}. The wiring scan is looking in the wrong place, so its \
         verdict would be meaningless."
    );

    // The real population: the whole workspace except `fuel-dispatch` (where
    // these symbols legitimately live) and this file (which names them).
    let mut outside_files = Vec::new();
    rust_files(&root, &["fuel-dispatch"], &mut outside_files);
    let outside = trigger_refs(&outside_files, &root);

    if let Err(msg) = verdict(&outside, &ingest_src) {
        panic!("{msg}");
    }
}

/// **RETAINED SABOTAGE SIBLING.** The test above is green today because no
/// provider exists yet. That green proves the precondition holds; it does NOT
/// prove the guard would still notice if it stopped holding.
///
/// These cases feed `verdict` the states the workspace does not currently
/// occupy, so the guard's discrimination is re-proven on EVERY run rather than
/// once at authoring time. Without this, a later edit that broke the detector
/// would leave the real test passing for the wrong reason, silently.
#[cfg(test)]
mod sabotage {
    use super::*;

    #[test]
    fn fires_when_a_provider_appears_before_the_wiring() {
        let refs = vec!["some-consumer/src/main.rs:42".to_string()];
        let unwired = "fn verify_candidate_impl() { probe_from_operands(); }";
        let v = verdict(&refs, unwired);
        assert!(v.is_err(), "a provider with no wiring MUST fail the guard");
        let msg = v.unwrap_err();
        assert!(
            msg.contains("GAP-236") && msg.contains("some-consumer/src/main.rs:42"),
            "the failure must name the gap AND the trigger site, or it reads as a \
             flaky scan rather than a precondition violation: {msg}"
        );
    }

    #[test]
    fn passes_once_the_wiring_lands() {
        let refs = vec!["some-consumer/src/main.rs:42".to_string()];
        let wired = "fn verify_candidate_impl() { let p = admission_probes(&ops); }";
        assert!(
            verdict(&refs, wired).is_ok(),
            "once `admission_probes` is wired, a provider is exactly what this \
             system is for — the guard must stop objecting"
        );
    }

    #[test]
    fn passes_while_no_provider_exists() {
        let unwired = "fn verify_candidate_impl() { probe_from_operands(); }";
        assert!(
            verdict(&[], unwired).is_ok(),
            "no provider means the precondition is not yet engaged; the unwired \
             state alone is tracked by GAP-236, not by this tripwire"
        );
    }

    #[test]
    fn comment_lines_are_not_trigger_references() {
        let dir = std::env::temp_dir().join("gap236_tripwire_probe");
        let _ = std::fs::create_dir_all(&dir);
        let f = dir.join("commented.rs");
        std::fs::write(&f, "// mentions IngestionService in prose\nfn x() {}\n").unwrap();
        let hits = trigger_refs(std::slice::from_ref(&f), &dir);
        assert!(
            hits.is_empty(),
            "a `//` comment mentioning the symbol must not read as a provider: {hits:?}"
        );

        // Opposite-outcome control: the SAME symbol on a code line IS a hit, so
        // the emptiness above is the comment rule working, not the matcher
        // being dead.
        std::fs::write(&f, "let s: IngestionService = build();\n").unwrap();
        let hits = trigger_refs(&[f], &dir);
        assert_eq!(
            hits.len(),
            1,
            "the same symbol on a CODE line must be found, or the comment-skip \
             result above proves nothing"
        );
    }
}
