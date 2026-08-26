//! **A citation in `CLAUDE.md` may not use a bare line number as its anchor.**
//!
//! WHY THIS EXISTS, MEASURED RATHER THAN ASSUMED (2026-08-25, doc-currency
//! program). A sweep of every citation in `CLAUDE.md` found three populations
//! with three decay rates:
//!
//! ```text
//! file paths / symbols / GAP ids     0% defective  (19, 25 checked)
//! line numbers                      67% defective  (9 checked)
//! commands                           0% dead       (10 checked)
//! ```
//!
//! A path survives because moving a file is a loud event someone notices; a
//! registry row is never renumbered. A line number rots on **any insertion
//! above it**, silently, and nothing in the artifact shows it. The worst
//! observed drift landed a reader in a function signature, where the honest
//! conclusion is *"the cited panic was fixed"* — a stale citation that looks
//! plausible at the destination is worse than one that obviously misses.
//!
//! THE FIX IS DELETION, NOT UPDATING. Every one of the six anchors this gate
//! was built against **already named its target** — a symbol, a quoted panic
//! string, a dependency line. The number was redundant *and* the only
//! rot-prone half, so updating it would reset a clock that runs out again.
//!
//! ## Why the allowlist is empty, and why that matters
//!
//! `GAP-141`'s prose-hedge guard establishes that an allowlist is where the
//! distinction a pattern cannot make has to live — so entries are load-bearing
//! and each needs a reason. **This gate needs none: all six real instances were
//! fixable.** An allowlist of six with reasons is tractable; one of 157 is a
//! shredder that degrades to noise and takes the guard's signal with it. Zero
//! is better than both.
//!
//! ## Self-matching: solved by SCOPE, not by escaping
//!
//! A source-scanning check placed inside its own scan target matches its own
//! anchors and goes red for the wrong reason — and only the *message*
//! distinguishes that from a real red. **This scanner reads `CLAUDE.md` only,
//! a markdown file, and is itself Rust.** It therefore cannot see its own doc
//! comments or examples *by construction*. Nothing here is escaped or
//! obfuscated to dodge the pattern; the scope simply excludes it.

use std::path::PathBuf;

/// Extensions whose `name.ext:123` form is a line-anchored citation.
const ANCHORED: &[&str] = &[".rs:", ".toml:", ".ps1:", ".yml:", ".md:"];

/// `(anchor, reason it is legitimate)`. **Empty on purpose** — see module docs.
/// An entry here must name why the line number cannot be replaced by the thing
/// it points at; "it is correct today" is not a reason, since correctness is
/// what rots.
const ALLOWLIST: &[(&str, &str)] = &[];

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

/// Every `path.ext:NNN` (optionally `:~NNN`) in `text`, as `(line_no, anchor)`.
fn find_line_anchors(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        for ext in ANCHORED {
            let mut from = 0usize;
            while let Some(rel) = line[from..].find(ext) {
                let at = from + rel;
                let after_colon = at + ext.len();
                let rest = &line[after_colon..];
                let digits = rest.strip_prefix('~').unwrap_or(rest);
                let n: String = digits.chars().take_while(char::is_ascii_digit).collect();
                if !n.is_empty() {
                    let start = line[..at]
                        .rfind(|c: char| !(c.is_alphanumeric() || "_./-".contains(c)))
                        .map_or(0, |i| i + 1);
                    let end = after_colon + (rest.len() - digits.len()) + n.len();
                    out.push((idx + 1, line[start..end].to_string()));
                }
                from = after_colon;
            }
        }
    }
    out
}

#[test]
fn claude_md_carries_no_bare_line_number_anchor() {
    let root = workspace_root();
    let path = root.join("CLAUDE.md");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let offenders: Vec<_> = find_line_anchors(&text)
        .into_iter()
        .filter(|(_, a)| !ALLOWLIST.iter().any(|(allowed, _)| a == allowed))
        .collect();

    assert!(
        offenders.is_empty(),
        "CLAUDE.md carries {} bare line-number citation anchor(s). A line number rots on any \
         insertion above it, silently — measured at 67% defective against 0% for paths and GAP \
         ids. DELETE the number and keep the name: every anchor this gate was built against \
         already named its target, so the number was redundant AND the only rot-prone half. \
         If a number genuinely cannot be replaced, add it to ALLOWLIST with the reason. \
         Offenders: {offenders:?}",
        offenders.len(),
    );
}

/// Positive control. Without this, the gate above passes identically whether the
/// file is clean or the scanner is blind, and a blind scanner is indistinguishable
/// from a clean corpus.
#[test]
fn the_scanner_can_see_an_anchor_when_one_exists() {
    let hits = find_line_anchors("see `fuel-graph/src/lib.rs:3931` and `a/b.toml:41` for detail");
    assert_eq!(
        hits.len(),
        2,
        "scanner must find both anchors, got {hits:?}"
    );
    assert!(
        hits.iter().any(|(_, a)| a.ends_with(":3931")),
        "got {hits:?}"
    );
}

/// Negative control: a plain path is the FIX, so flagging it would make the gate
/// reject its own remedy and train people to allowlist reflexively.
#[test]
fn the_scanner_does_not_flag_a_plain_path_or_a_bare_number() {
    assert!(find_line_anchors("see `fuel-graph/src/lib.rs` and `a/b.toml`").is_empty());
    assert!(find_line_anchors("157 flags at 90% false positive").is_empty());
    assert!(find_line_anchors("docs/architecture/10-decisions-log.md is an index").is_empty());
}
