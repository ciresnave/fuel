// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Every tracked `.rs` file carries an SPDX licence identifier.**
//!
//! WHY THIS IS A GATE AND NOT A SWEEP, AND THE REASON IS MEASURED RATHER THAN
//! ASSUMED. `62b0e36b` stamped 795 files on 2026-08-19. By 2026-09-10 ten files
//! carried no identifier — and **not one of them was missed by that sweep**:
//! every one was ADDED between 2026-08-20 and 2026-09-02. The gap was not a hole
//! in the sweep, it was REGROWTH, and 98% was on its way back down.
//!
//! **A SWEEP IS NOT THE FIX FOR A PROPERTY THAT REGROWS.** A one-time cleanup of
//! a property that every new file must satisfy decays at exactly the rate the
//! repository grows, and nothing reports the decay. This test is the ratchet the
//! sweep needed.
//!
//! WHAT AN ABSENT IDENTIFIER ACTUALLY COSTS, so it is not taken on faith: the
//! root manifest declares `license = "MIT OR Apache-2.0"` and every crate takes
//! it with `license.workspace = true`, so an unstamped file is not *unlicensed* —
//! it is **licensed with no in-file evidence of it**, which is what automated
//! licence scanners, SBOM generators and downstream vendoring tools read. The
//! manifest binds the crate; the header is what survives a file being copied out
//! of it.
//!
//! ## ⚠️ THE ONE EXEMPTION, AND IT DEFENDS ITSELF
//!
//! `fuel-examples/src/bs1770.rs` carries `Apache-2.0` **only**. It is a verbatim
//! copy of a third-party Apache-2.0 work whose author never granted an MIT
//! option, and the ORIGINAL sweep stamped the dual licence on it before
//! `ef59fc23` corrected it — that commit's own subject reads *"my SPDX sweep
//! asserted a grant that does not exist"*. **That is a legal claim, not a
//! formatting one.**
//!
//! So this gate does not merely SKIP that file. It asserts that the file's SPDX
//! IDENTIFIER LINE still says `Apache-2.0` and still does not say `MIT`, **so a
//! future sweep that
//! "corrects" it back to the dual licence reddens this test.** An exemption that
//! only names its subject decays into a hole; one that asserts the property its
//! subject was exempted FOR is a second gate.
//!
//! ## ENUMERATION IS `git ls-files`, DELIBERATELY
//!
//! Same population the census measured, respects `.gitignore`, and cannot pick
//! up untracked scratch files. **If git is unavailable the test FAILS rather
//! than passing over an empty list** — a gate that cannot enumerate must never
//! report clean, which is the `0 passed` trap.
//!
//! ## SELF-MATCHING
//!
//! This file carries a real identifier on line 1 and is scanned like any other
//! tracked file, which is the correct relationship: **the gate is subject to
//! itself.** The prose below mentions the token repeatedly, which is harmless —
//! the check reads only the head of each file and asks whether the token is
//! present, never how many times.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Bytes of each file inspected. Generous enough for a shebang-like preamble or
/// a short block comment above the identifier, small enough that the token can
/// not be satisfied by something buried in the body.
const HEAD_BYTES: usize = 512;

const TOKEN: &str = "SPDX-License-Identifier";

/// The single file licensed differently, and WHY. Named here so the reason
/// travels with the exemption instead of living in a commit message.
const APACHE_ONLY: &str = "fuel-examples/src/bs1770.rs";

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

/// Tracked `.rs` files. Fails loudly if git cannot answer — an empty list must
/// never be mistaken for a clean repository.
fn tracked_rs_files(root: &Path) -> Vec<String> {
    let out = Command::new("git")
        .arg("ls-files")
        .arg("*.rs")
        .current_dir(root)
        .output()
        .expect("git ls-files could not be run; this gate cannot enumerate and must not pass");
    assert!(
        out.status.success(),
        "git ls-files failed ({}); a gate that cannot enumerate must never report clean",
        out.status
    );
    let files: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    assert!(
        files.len() > 500,
        "git ls-files returned only {} .rs files, which is far below this workspace's \
         known size — the enumeration is broken, not the repository",
        files.len()
    );
    files
}

fn head_of(root: &Path, rel: &str) -> String {
    let bytes = std::fs::read(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    String::from_utf8_lossy(&bytes[..bytes.len().min(HEAD_BYTES)]).into_owned()
}

#[test]
fn every_tracked_rs_file_carries_an_spdx_identifier() {
    let root = workspace_root();
    let files = tracked_rs_files(&root);

    let missing: Vec<&String> = files
        .iter()
        .filter(|f| !head_of(&root, f).contains(TOKEN))
        .collect();

    assert!(
        missing.is_empty(),
        "{} tracked .rs file(s) carry no `{TOKEN}` in their first {HEAD_BYTES} bytes.\n\
         The root manifest licenses this workspace, so these files are not unlicensed — they \
         carry no IN-FILE evidence of it, which is what licence scanners, SBOM generators and \
         downstream vendoring read.\n\
         Add `// SPDX-License-Identifier: MIT OR Apache-2.0` as the first line. If a file is \
         VENDORED from elsewhere, do NOT stamp the dual licence on it: that asserts a grant its \
         author may never have made. See the {APACHE_ONLY} exemption in this file.\n\
         Missing: {missing:#?}",
        missing.len()
    );
}

/// ⚠️ THE ASSERTION READS THE IDENTIFIER LINE, NOT THE HEAD.
///
/// The first version of this arm asserted `!head.contains("MIT")` and **failed
/// at rest** — because the file explains its own exemption in prose directly
/// beneath the identifier: *"stamping `MIT OR Apache-2.0` here would assert a
/// licence grant that does not exist."* **The explanation of why the file is not
/// MIT contains the string MIT**, and a head-wide search cannot tell an
/// ASSERTION from an EXPLANATION of the same token.
///
/// The licence claim lives in the identifier line and nowhere else, so that is
/// the only line entitled to answer for it.
fn spdx_line_of(head: &str) -> String {
    head.lines()
        .find(|l| l.contains(TOKEN))
        .unwrap_or_else(|| panic!("{APACHE_ONLY} carries no `{TOKEN}` line at all"))
        .trim()
        .to_string()
}

/// The exemption asserts the property it was exempted FOR, so a future sweep that
/// "corrects" the vendored file back to the dual licence reddens here.
#[test]
fn the_vendored_file_is_still_apache_only() {
    let root = workspace_root();
    let line = spdx_line_of(&head_of(&root, APACHE_ONLY));

    assert!(
        line.contains("Apache-2.0"),
        "{APACHE_ONLY} must carry `Apache-2.0`. Its identifier line reads: {line:?}"
    );
    assert!(
        !line.contains("MIT"),
        "{APACHE_ONLY}'s identifier line now claims MIT: {line:?}\n\
         It is a verbatim copy of a third-party Apache-2.0 work and its author never granted an \
         MIT option. Commit ef59fc23 exists precisely because an earlier sweep made this mistake, \
         with the subject \"my SPDX sweep asserted a grant that does not exist\". This is a LEGAL \
         claim, not a formatting one. Revert it."
    );
}

/// Proof the scanner can SEE an absent identifier. Without this, a gate that
/// silently matched everything would pass forever and look identical to a clean
/// repository — the failure mode the `0 passed` trap describes.
#[test]
fn the_scanner_can_see_a_missing_identifier() {
    let with = format!("// {TOKEN}: MIT OR Apache-2.0\nfn main() {{}}\n");
    let without = "fn main() {}\n".to_string();

    assert!(with.contains(TOKEN), "control: a stamped head must match");
    assert!(
        !without.contains(TOKEN),
        "control: an unstamped head must NOT match — if this fires, the predicate matches \
         everything and the population test above is vacuous"
    );
}
