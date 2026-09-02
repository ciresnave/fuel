//! No tracked file may begin with a UTF-8 BOM (`EF BB BF`).
//!
//! WHY THIS IS A GATE AND NOT A ONE-TIME SWEEP: 136 tracked files carried a BOM,
//! written in one go by `815abd04` ("chore: globally rename project from candle
//! to fuel") — a tool of ours, not something inherited from the fork. A tool
//! wrote them, so a tool can write them again, and a 136-file cleanup with no
//! gate is an instance-fix that decays the moment anyone runs a formatter that
//! defaults to `utf-8-sig`.
//!
//! WHAT A BOM ACTUALLY BREAKS, so the cost is not taken on faith: it is three
//! bytes BEFORE the first character, so `#!/bin/sh` stops being a shebang,
//! `[workspace]` stops being the first TOML key, a leading `#` stops opening a
//! markdown heading, and a `^`-anchored grep silently misses the first line of
//! the file. That last one is why this matters here specifically — several of
//! this repo's gates are line-anchored scanners, and a BOM makes line 1
//! invisible to them while everything still looks right.
//!
//! ENUMERATION IS `git ls-files`, DELIBERATELY: it is the same population the
//! census measured, it respects `.gitignore`, and it cannot pick up untracked
//! scratch files and red the build for something nobody committed. If git is
//! unavailable the test FAILS rather than passing over an empty list — a gate
//! that cannot enumerate must never report clean, which is the `0 passed` trap.
//!
//! SELF-MATCHING IS IMPOSSIBLE BY CONSTRUCTION: the marker appears in this file
//! only as the ASCII text `\xef\xbb\xbf` inside a Rust byte-string literal, so
//! this source cannot match its own pattern. It is nonetheless scanned like any
//! other tracked file, which is the correct relationship — the gate is subject
//! to itself.

use std::path::PathBuf;
use std::process::Command;

const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// Files that MUST keep their BOM, with the measured reason for each.
///
/// ⚠️ THESE ARE NOT EXEMPTIONS, THEY ARE FINDINGS. Windows PowerShell 5.1 reads
/// a BOM-less file as Windows-1252, so a UTF-8 multi-byte sequence is mangled
/// into cp1252 bytes. Where the mangled bytes land somewhere the parser cares
/// about, the script stops parsing — and the failure mode of a broken guard is
/// SILENCE, which this project has already paid for once with `gpu-run.ps1`.
///
/// MEASURED 2026-09-02, both directions, under 5.1 specifically:
///   with BOM:     cuda-build 0 errors · gpu-run 0 errors
///   without BOM:  cuda-build 10 errors · gpu-run 4 errors
///
/// Note `scripts/aarch64-cross-check.ps1` has 19 non-ASCII lines and parses
/// clean at 0 errors WITHOUT a BOM, so it is not listed: non-ASCII content is
/// necessary but not sufficient, and the entries below were chosen by measuring
/// each file rather than by extension.
///
/// THE PROPER FIX IS TO REMOVE THE NON-ASCII CHARACTERS (they are em-dashes in
/// comments), which would retire both entries. That is a content change to a
/// machine-wide guard and belongs in its own change, not in an encoding sweep.
const BOM_REQUIRED: &[(&str, &str)] = &[
    (
        "scripts/cuda-build.ps1",
        "PowerShell 5.1 parses it with 10 errors when the BOM is stripped (0 with it)",
    ),
    (
        "scripts/gpu-run.ps1",
        "PowerShell 5.1 parses it with 4 errors when the BOM is stripped (0 with it); \
         this is the machine-wide GPU mutex and a parse error deletes the guard silently",
    ),
];

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file()
            && std::fs::read_to_string(&manifest).is_ok_and(|s| s.contains("[workspace]"))
        {
            return dir;
        }
        assert!(
            dir.pop(),
            "no Cargo.toml with [workspace] above CARGO_MANIFEST_DIR"
        );
    }
}

/// Does this byte slice start with a UTF-8 BOM?
///
/// The one definition used by both the scan and its self-test. Testing a copy
/// of the predicate would prove nothing about the predicate that runs.
fn starts_with_bom(bytes: &[u8]) -> bool {
    bytes.starts_with(&BOM)
}

/// Every tracked file, from `git ls-files`.
///
/// Returns `Err` rather than an empty list when git cannot answer, so a broken
/// enumeration reds the gate instead of silently passing it.
fn tracked_files(root: &PathBuf) -> Result<Vec<PathBuf>, String> {
    let out = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run `git ls-files`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`git ls-files` failed with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let files: Vec<PathBuf> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| root.join(String::from_utf8_lossy(s).as_ref()))
        .collect();
    if files.is_empty() {
        return Err("`git ls-files` returned nothing — refusing to report clean".into());
    }
    Ok(files)
}

#[test]
fn no_tracked_file_begins_with_a_utf8_bom() {
    let root = workspace_root();
    let files = tracked_files(&root).expect("enumerate tracked files");

    let allowed: Vec<String> = BOM_REQUIRED.iter().map(|(p, _)| (*p).to_string()).collect();

    let mut offenders = Vec::new();
    for path in &files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string()
            .replace('\\', "/");
        if allowed.contains(&rel) {
            continue;
        }
        // A file that cannot be read is skipped, not failed: `git ls-files`
        // lists entries that may be absent from the working tree (sparse
        // checkouts, submodule gitlinks). A missing file has no first byte.
        if let Ok(bytes) = std::fs::read(path) {
            if starts_with_bom(&bytes) {
                offenders.push(
                    path.strip_prefix(&root)
                        .unwrap_or(path)
                        .display()
                        .to_string()
                        .replace('\\', "/"),
                );
            }
        }
    }

    offenders.sort();
    assert!(
        offenders.is_empty(),
        "{} tracked file(s) begin with a UTF-8 BOM (EF BB BF). A BOM sits BEFORE the first \
         character, so a `^`-anchored scanner cannot see line 1, a shebang stops being a \
         shebang, and the first TOML key stops being first. Strip it by writing the file with \
         plain utf-8 (never `utf-8-sig`), then RE-READ the file and assert the first three \
         bytes are not EF BB BF — a stripper that writes with the wrong codec puts the BOM \
         back and reports success. Scanned {} tracked files. Offenders: {:?}",
        offenders.len(),
        files.len(),
        offenders,
    );
}

/// RETAINED SABOTAGE. The assertion above goes green the moment the sweep lands
/// and stays green forever, so from that point it has NO evidence from live data
/// that it can still discriminate — a born-red proves the gate worked ONCE, at
/// authoring time. This proves it still works on every run.
///
/// If someone later "simplifies" `starts_with_bom`, or the scan starts reading
/// the wrong thing, THIS test reds while the one above stays quietly green.
#[test]
fn the_scanner_detects_a_planted_bom_and_ignores_a_clean_file() {
    let mut dirty = Vec::from(BOM);
    dirty.extend_from_slice(b"# heading\n");
    assert!(
        starts_with_bom(&dirty),
        "the BOM scanner failed to detect a planted EF BB BF — the gate above is inert and \
         its green means nothing"
    );

    assert!(
        !starts_with_bom(b"# heading\n"),
        "the BOM scanner flagged a clean file: it would red the build for every file in the \
         tree"
    );

    // A prefix of the marker is NOT a marker. Guards against a scanner that
    // compares too few bytes and starts flagging innocent Latin-1 content.
    assert!(
        !starts_with_bom(&[0xEF, 0xBB]),
        "the BOM scanner matched a two-byte prefix of the marker"
    );

    // ...and the marker must be at the START, not merely present.
    let mut late = Vec::from(&b"x"[..]);
    late.extend_from_slice(&BOM);
    assert!(
        !starts_with_bom(&late),
        "the BOM scanner matched a marker that is not at offset 0"
    );
}

/// Every allowlist entry must still EXIST, still CARRY a BOM, and still contain
/// the non-ASCII content that is the reason it needs one.
///
/// ⚠️ THIS IS WHAT STOPS THE ALLOWLIST BECOMING A JUNK DRAWER, and each clause
/// catches a different rot:
///   * MISSING FILE  — a deletion sweep orphans the entry (this repo has done
///     exactly that: retiring `fuel-wasm-examples` left four dead entries in a
///     different allowlist and reddened a guard for unrelated reasons).
///   * NO BOM        — someone stripped it anyway; the entry is now a lie.
///   * ALL-ASCII     — the em-dashes were removed, so PowerShell 5.1 no longer
///     needs the BOM and THE ENTRY SHOULD BE DELETED ALONG WITH THE BOM. This
///     clause makes the allowlist SELF-RETIRING: fixing the root cause reds this
///     test, which is the prompt to remove both.
#[test]
fn every_bom_allowlist_entry_still_needs_its_bom() {
    let root = workspace_root();
    for (rel, reason) in BOM_REQUIRED {
        let path = root.join(rel);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "BOM allowlist names `{rel}` which cannot be read ({e}). Reason on file: \
                    {reason}. If the file was deleted or moved, DELETE THE ENTRY — an \
                    allowlist pointing at nothing reds this guard for reasons unrelated to \
                    any BOM."
            )
        });
        assert!(
            starts_with_bom(&bytes),
            "BOM allowlist names `{rel}`, but it has NO BOM. Either the BOM was stripped (and \
             PowerShell 5.1 can no longer parse it — re-check) or the entry is stale and \
             should be removed. Reason on file: {reason}"
        );
        assert!(
            bytes.iter().any(|b| *b >= 0x80),
            "BOM allowlist names `{rel}`, but it is now pure ASCII — so PowerShell 5.1 would \
             read it identically with or without a BOM, and the entry has outlived its \
             reason. DELETE THE BOM AND THIS ENTRY TOGETHER. Reason on file: {reason}"
        );
    }
}

/// The enumeration must refuse to report clean when it cannot enumerate.
///
/// Without this, a gate that loses its file list passes silently — the exact
/// shape of `running 0 tests; N filtered out` being read as a pass.
#[test]
fn an_empty_enumeration_is_an_error_not_a_pass() {
    let empty = PathBuf::from(
        std::env::temp_dir()
            .join("fuel-bom-gate-not-a-repo")
            .to_string_lossy()
            .to_string(),
    );
    std::fs::create_dir_all(&empty).ok();
    // Outside any git work tree (or in one that lists nothing), `tracked_files`
    // must return Err. It must never return Ok(vec![]).
    match tracked_files(&empty) {
        Ok(v) => assert!(
            !v.is_empty(),
            "tracked_files returned Ok with an EMPTY list — a gate that cannot enumerate \
             would report clean"
        ),
        Err(_) => {}
    }
}
