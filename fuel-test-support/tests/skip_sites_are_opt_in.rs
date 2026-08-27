// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The guard that makes `Hardware::default_policy`'s proxy checkable.**
//!
//! `default_policy` returns `Fatal` for CUDA and Vulkan. The honest reason is
//! *not* "building `--features cuda` proves you have a GPU" — that is false, and
//! its docs say so. The real reason is that **every CUDA and Vulkan call site is
//! inside an `#[ignore]`d test**, so running one takes an explicit
//! `-- --ignored`, and *that* is the declaration the `Fatal` default relies on.
//!
//! `#[ignore]` is a libtest attribute no runtime code can observe, so the
//! default keys on the *family*, which merely correlates. **This file is what
//! stops that correlation from being an assumption nobody checks.**
//!
//! # What it would catch
//!
//! A `Fatal`-default family used outside an `#[ignore]`d test: the default then
//! hard-fails on any machine without the device, with no `-- --ignored` opt-in
//! anywhere to justify it.
//!
//! # Day-one count: ZERO — and the predicted number was wrong
//!
//! This guard was expected to report **exactly 1** violation when written:
//! `fuel-core/src/test_utils.rs::assert_cuda_matches_reference`, a `pub fn`
//! assertion helper, not a test, with zero callers. **It reports 0.** That site
//! never adopted the mechanism — it was deliberately excluded from the GAP-243
//! sweep as needing a different remedy — so it is not a call site, and a scan
//! for call sites cannot see it. The prediction described a site that would
//! have violated *had it been converted*.
//!
//! Recorded because the number was written into this comment from a prediction
//! **before it was measured**, which is its own defect: a wrong number with a
//! plausible reason attached becomes a reason.
//!
//! A zero day-one count means the born-red had to come from a **sabotage**
//! rather than from existing code — see
//! [`a_removed_ignore_is_detected`], which is that sabotage kept permanently
//! rather than performed once and described in prose.
//!
//! # Why a zero here is not vacuous
//!
//! A guard whose count is permanently zero is indistinguishable from a guard
//! that stopped working. Three controls run on every invocation, and each fails
//! loudly rather than silently returning nothing:
//!
//! 1. the scan finds **at least `MIN_SITES` call sites** — if the anchor ever
//!    stops matching (a rename, a re-export, a formatting change), the floor
//!    fails instead of reporting a clean zero;
//! 2. it finds at least one `Fatal`-family site it classifies as **ignored** —
//!    proving the `#[ignore]` detection can return *true*;
//! 3. it finds at least one **`Permissive`-family** site — proving the family
//!    classifier discriminates rather than answering `Cuda` to everything.
//!
//! Without (2) especially, a detector that never sees `#[ignore]` would report
//! every site as a violation, and one that always sees it would report none.
//!
//! # Self-reference, which is the standard way a source scan lies
//!
//! This file searches for text that would otherwise appear *in this file*. The
//! anchor is therefore **built at runtime** from fragments, and this file is
//! **excluded from its own search space** by name. Both, deliberately: either
//! alone has failed in this repo before.

use std::path::{Path, PathBuf};

/// Floor for total discovered call sites. Set below the count at authoring time
/// (44) so ordinary churn does not trip it, but far enough above zero that a
/// broken anchor cannot pass.
const MIN_SITES: usize = 30;

/// Families whose default is `Fatal`, so their sites must be opt-in.
const FATAL_FAMILIES: [&str; 2] = ["Cuda", "Vulkan"];

#[derive(Debug)]
struct Site {
    file: String,
    line: usize,
    family: String,
    in_ignored_test: bool,
}

fn rust_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            // `target` dwarfs the tree and holds generated copies; `.git` holds
            // packed objects that are not source at all.
            if name != "target" && name != ".git" && !name.starts_with('.') {
                rust_files(&p, out);
            }
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// Classify every call site in one file's text. **Pure**, so the detector can
/// be exercised against synthetic source that contains a known violation —
/// which is the only way to prove on every run that it still discriminates.
fn sites_in(text: &str, file: &str) -> Vec<Site> {
    // Built at runtime so the literal never appears in this file's own bytes.
    let anchor = format!("hardware::{}(", "skip");
    let ignore_attr = format!("#[{}", "ignore");

    let lines: Vec<&str> = text.lines().collect();
    let mut sites = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if !l.contains(&anchor) {
            continue;
        }
        // The family is named within a few lines of the call.
        let family = lines[i..(i + 6).min(lines.len())]
            .iter()
            .find_map(|w| {
                w.split("Hardware::")
                    .nth(1)
                    .map(|r| r.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            })
            .unwrap_or_else(|| "UNKNOWN".to_string());

        // Walk back to the enclosing `fn`, then read its attribute block. The
        // gate can sit on the fn, so only the fn's own attributes matter for
        // opt-in-ness.
        let mut in_ignored_test = false;
        for j in (0..i).rev() {
            let s = lines[j].trim_start();
            if s.starts_with("fn ") || s.starts_with("pub fn ") || s.starts_with("async fn ") {
                for k in (0..j).rev() {
                    let a = lines[k].trim_start();
                    if a.starts_with("#[") {
                        if a.starts_with(&ignore_attr) {
                            in_ignored_test = true;
                        }
                    } else if !a.starts_with("//") && !a.is_empty() {
                        break;
                    }
                }
                break;
            }
        }

        sites.push(Site {
            file: file.to_string(),
            line: i + 1,
            family,
            in_ignored_test,
        });
    }
    sites
}

fn scan() -> Vec<Site> {
    let anchor = format!("hardware::{}(", "skip");

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-test-support has a parent directory")
        .to_path_buf();

    // This file is excluded from its own search space.
    let own = Path::new(file!())
        .file_name()
        .expect("file!() has a file name")
        .to_string_lossy()
        .to_string();

    let mut files = Vec::new();
    rust_files(&root, &mut files);

    let mut sites = Vec::new();
    for path in files {
        let fname = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        if fname == own {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !text.contains(&anchor) {
            continue;
        }
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        sites.extend(sites_in(&text, &rel));
    }
    sites
}

/// **The permanent sabotage.** A born-red expires the moment it goes green;
/// this is the same discrimination check kept as a sibling that runs every
/// time.
///
/// The real guard above scans the tree and currently finds **zero** violations.
/// A permanently-zero count is indistinguishable from a detector that stopped
/// working — the controls in the main test cover the *scan*, and this covers the
/// *classification*: fed source containing one violation and two compliant
/// sites, the detector must pick out exactly the violation.
///
/// The fixture is synthetic rather than a real file because a real file can be
/// edited out from under it, at which point the check silently stops testing
/// what it names. It was validated against the real thing once: removing the
/// single `#[ignore]` from `invoker_cuda.rs` made the main guard fail naming
/// that exact site and family, with all three controls still passing.
#[test]
fn a_removed_ignore_is_detected() {
    let fixture = "\
mod tests {
    #[test]
    #[ignore = \"requires a live CUDA device\"]
    fn compliant_ignored_cuda() {
        let Ok(d) = CudaDevice::new(0) else {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Cuda,
                fuel_test_support::hardware::Missing::device(\"x\"),
            );
        };
    }

    #[test]
    fn violating_unignored_cuda() {
        let Ok(d) = CudaDevice::new(0) else {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Cuda,
                fuel_test_support::hardware::Missing::device(\"x\"),
            );
        };
    }

    #[test]
    fn compliant_unignored_aocl() {
        if !aocl_present() {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Aocl,
                fuel_test_support::hardware::Missing::device(\"x\"),
            );
        }
    }
}
";
    let sites = sites_in(fixture, "fixture.rs");
    assert_eq!(
        sites.len(),
        3,
        "the fixture has three call sites: {sites:?}"
    );

    let violations: Vec<&Site> = sites
        .iter()
        .filter(|s| FATAL_FAMILIES.contains(&s.family.as_str()) && !s.in_ignored_test)
        .collect();

    assert_eq!(
        violations.len(),
        1,
        "the detector must find exactly the one un-`#[ignore]`d CUDA site. \
         Finding 0 means it can no longer tell a missing `#[ignore]` from a \
         present one, and the main guard's zero above means nothing. Finding 3 \
         means it never sees `#[ignore]` at all. Sites: {sites:?}"
    );
    assert_eq!(violations[0].family, "Cuda");
    assert!(
        !sites[2].in_ignored_test && sites[2].family == "Aocl",
        "the un-ignored AOCL site must be classified Permissive-family, not \
         swept in as a violation -- otherwise the guard would demand \
         `#[ignore]` on families whose default is already permissive: {:?}",
        sites[2]
    );
}

#[test]
fn a_fatal_default_family_is_only_used_inside_an_ignored_test() {
    let sites = scan();

    // --- CONTROL 1: the anchor still matches something. ---
    assert!(
        sites.len() >= MIN_SITES,
        "found only {} call sites, expected at least {MIN_SITES}. The scan is \
         broken, not the code -- most likely the call spelling changed and this \
         guard is now silently checking nothing. A clean zero from a broken \
         anchor is indistinguishable from compliance, which is why this floor \
         exists.",
        sites.len()
    );

    // --- CONTROL 2: `#[ignore]` detection can return TRUE. ---
    assert!(
        sites
            .iter()
            .any(|s| FATAL_FAMILIES.contains(&s.family.as_str()) && s.in_ignored_test),
        "no Fatal-family site was classified as being inside an `#[ignore]`d \
         test. Either every such site really is a violation, or the `#[ignore]` \
         detection never returns true -- and a detector stuck on `false` would \
         report EVERY site as a violation below, which reads as a real finding."
    );

    // --- CONTROL 3: the family classifier discriminates. ---
    assert!(
        sites
            .iter()
            .any(|s| !FATAL_FAMILIES.contains(&s.family.as_str()) && s.family != "UNKNOWN"),
        "every site classified into a Fatal family. The family parser is not \
         discriminating, so this guard cannot tell which default applies and \
         its verdict means nothing. Sites seen: {:?}",
        sites.iter().map(|s| &s.family).collect::<Vec<_>>()
    );

    // --- THE SUBJECT. ---
    let violations: Vec<&Site> = sites
        .iter()
        .filter(|s| FATAL_FAMILIES.contains(&s.family.as_str()) && !s.in_ignored_test)
        .collect();

    assert!(
        violations.is_empty(),
        "{} site(s) use a Fatal-default family OUTSIDE an `#[ignore]`d test:\n{}\n\n\
         `Hardware::default_policy` returns `Fatal` for these families because \
         every call site is opt-in -- an explicit `-- --ignored` is what makes \
         \"the device was promised\" true. A site with no opt-in inherits a \
         default that hard-fails on any machine lacking the device, with \
         nothing to justify it.\n\n\
         Fix by making the test `#[ignore]`d, or -- if it is not a test at all \
         -- pass the policy explicitly via `skip_with` with a written reason. \
         DO NOT add an allowlist here: an exception list is where violations go \
         to live, and this guard's value is that its count reaches zero for a \
         reason.",
        violations.len(),
        violations
            .iter()
            .map(|s| format!("  {}:{}  [{}]", s.file, s.line, s.family))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
