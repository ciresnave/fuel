//! **A ratchet on live-GPU tests that acquire a device without the lock (GAP-224).**
//!
//! [`fuel_test_support::require_gpu_run_lock`] makes a missing GPU mutex loud,
//! but a guard nobody calls guards nothing — and there are **59 unguarded device
//! acquisitions across 35 `*_live.rs` files**, which is not a migration anyone
//! finishes in one change.
//!
//! So this is the other half, and it is the half that stops the bleeding:
//! **you cannot make 59 remaining sites call a helper today, but you can make a test fail
//! when the 60th appears.** An unbounded migration becomes a ratchet.
//!
//! Pattern adopted from Vulkane's `gpu-run` scanner, **including the bug they
//! found in it**: their first version exempted any *file* that mentioned a
//! guarded helper, which waved through all twelve direct sites in one file
//! because that file used the helper elsewhere. It reported 1 unguarded site
//! out of 20 and passed — **covered in name, covering almost nothing, green
//! either way.** So attribution here is **per site, via its enclosing `fn`**,
//! never per file.
//!
//! # What this test can NOT see
//!
//! The population is **naming and syntax**. Measured at the time of writing,
//! `*_live.rs` files acquire a device by exactly two spellings —
//! `CudaDevice::new(` and `new_device()` — and those are what
//! [`ACQUISITION_SPELLINGS`] lists. **A device acquired by any other spelling
//! is invisible here.** This is a lower bound, not a proof: it cannot certify
//! the tree, it can only stop the count rising.
//!
//! Deliberately not listed: spellings the code does not currently use. Claiming
//! a reach the scan does not have would be the same defect as the missing
//! guard, pointed at the documentation.
//!
//! And it over-counts in the OTHER direction: the scan skips lines beginning
//! with `//`, but it CANNOT distinguish a code site from a STRING LITERAL or
//! macro argument containing the same spelling. A `panic!`/`assert!` message
//! mentioning `CudaDevice::new(` is counted as an acquisition. Found 2026-08-21
//! by a lane whose assert message tripped it. The remedy is to reword the
//! message, never to "guard" it — guarding a string satisfies the scanner
//! instead of the property, which is the defect this scanner exists to catch,
//! aimed at itself. So `BUDGET` is a count of MATCHES, not of acquisitions, and
//! the two differ by an unknown small number.

use std::path::{Path, PathBuf};

/// How a `*_live.rs` test gets a device. See the module docs' population note.
const ACQUISITION_SPELLINGS: [&str; 2] = ["CudaDevice::new(", "new_device()"];

/// The call that marks a site guarded.
const GUARD_CALL: &str = "require_gpu_run_lock()";

/// Unguarded device acquisitions at the time this ratchet was installed.
///
/// **This is DEBT, not an allowance.** It exists so the number can only move
/// down; every reduction should come with this constant reduced in the same
/// change.
///
/// Set to 60 after the scanner DISAGREED with the `git grep -o` count that
/// preceded it (61) and turned out to be right: the extra hit was
/// `tri_backend_device_group_live.rs:24`, a `//!` doc comment mentioning
/// `new_device()` in prose. **grep counted a STRING; the scanner counts a
/// SITE**, and those are different populations — the same distinction that
/// makes this file's own population claim worth stating.
///
/// Lowered 59 -> 58 on 2026-08-21: guarding `jit_synth_kernel_live.rs`'s device
/// helper (`dev`) with `require_gpu_run_lock` removed one real site. The same
/// change also removed a FALSE +1 that a lane's assert-message string had added
/// (see the over-count note in the module docs above), so the net is a genuine
/// one-site reduction, not a wash. FIRST time this constant has moved down.
///
/// Lowered 58 -> 54 on 2026-08-21: guarded the four `fuel-cuda-backend/tests`
/// `*_live.rs` device helpers (`pinned`, `byte_storage`, `backend_runtime`,
/// `baracuda_contiguize`) — each `dev_or_skip -> dev` with `require_gpu_run_lock`
/// + `required_ok`. Four real acquisition sites now guarded (all GUARDS, no
///   string-corrections this batch), verified per site (none remain unguarded).
const BUDGET: usize = 54;

fn repo_root() -> PathBuf {
    // fuel-test-support/ -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent")
        .to_path_buf()
}

fn live_test_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            // `target` is build output; `.git` is not source.
            if name != "target" && name != ".git" {
                live_test_files(&path, out);
            }
        } else if name.ends_with("_live.rs") {
            out.push(path);
        }
    }
}

/// The line of the enclosing `fn` for `line`, and whether that function body
/// calls the guard.
///
/// Attribution is per site: a guard call elsewhere in the file does not cover
/// this one. That is the whole point — see the module docs.
fn enclosing_fn_is_guarded(lines: &[&str], site: usize) -> bool {
    let indent = |s: &str| s.len() - s.trim_start().len();

    let mut start = None;
    for i in (0..=site).rev() {
        if lines[i].trim_start().starts_with("fn ") || lines[i].trim_start().starts_with("pub fn ")
        {
            start = Some(i);
            break;
        }
    }
    let Some(start) = start else { return false };
    let my_indent = indent(lines[start]);

    let mut end = lines.len();
    for j in start + 1..lines.len() {
        let t = lines[j].trim_end();
        if t.trim_start() == "}" && indent(lines[j]) == my_indent {
            end = j + 1;
            break;
        }
    }

    lines[start..end].iter().any(|l| l.contains(GUARD_CALL))
}

#[test]
fn the_guard_still_refuses_an_unheld_lock() {
    // FOUNDATION. If `require_gpu_run_lock` stopped refusing — or stopped
    // existing — every site this ratchet counts as "guarded" would be guarded
    // by nothing, and **the site count would not move by one**. The ratchet
    // cannot see its own foundation being removed, so it is asserted here.
    use fuel_test_support::GpuRunLock;
    let verdict = fuel_test_support::gpu_run_lock::classify(None, None, None, None);
    assert!(
        matches!(verdict, GpuRunLock::Absent { .. }),
        "the guard no longer refuses an unheld lock ({verdict:?}); every `{GUARD_CALL}` in the \
         tree is now decorative and this ratchet's count means nothing"
    );
}

#[test]
fn unguarded_gpu_acquisitions_do_not_increase() {
    let mut files = Vec::new();
    live_test_files(&repo_root(), &mut files);

    // NON-TRIVIALITY. A scan that stopped finding files would report 0
    // unguarded sites and read as complete success.
    assert!(
        files.len() >= 30,
        "found only {} `*_live.rs` files (expected >= 30) — the walk is not reaching the tree, \
         so any count below is vacuous",
        files.len()
    );

    let mut unguarded: Vec<String> = Vec::new();
    let mut total_sites = 0usize;

    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            let hits = ACQUISITION_SPELLINGS
                .iter()
                .filter(|s| line.contains(**s))
                .count();
            if hits == 0 {
                continue;
            }
            total_sites += hits;
            if !enclosing_fn_is_guarded(&lines, i) {
                let rel = path.strip_prefix(repo_root()).unwrap_or(path);
                for _ in 0..hits {
                    unguarded.push(format!("  {}:{}", rel.display(), i + 1));
                }
            }
        }
    }

    // NON-TRIVIALITY, second axis: the spellings must still match something.
    assert!(
        total_sites >= 30,
        "matched only {total_sites} device acquisitions across {} files — ACQUISITION_SPELLINGS \
         has stopped matching the code, so a low unguarded count would be an artifact",
        files.len()
    );

    // TWO-DIRECTIONAL. Failing only when the count RISES turns the budget into
    // a ceiling nobody lowers: a run that beat it would pass silently and the
    // constant would never move. Vulkane's first run failed on exactly this
    // assertion, which is how they found their file-level exemption bug.
    assert_eq!(
        unguarded.len(),
        BUDGET,
        "unguarded live-GPU device acquisitions = {} (budget {BUDGET}).\n\
         {}\n\
         Sites:\n{}",
        unguarded.len(),
        if unguarded.len() > BUDGET {
            format!(
                "A NEW unguarded site appeared. Call `{GUARD_CALL}` in the test that acquires the \
                 device — see fuel_test_support::gpu_run_lock. Do not raise BUDGET."
            )
        } else {
            "GOOD NEWS, and the constant must move with it: lower BUDGET to this number in the \
             same change. A budget that tolerates being beaten stops being a ratchet."
                .to_string()
        },
        unguarded.join("\n")
    );
}
