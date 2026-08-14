//! GAP-141 Increment 2 — CI guard: no *new* prose-hedge admission of unfinished
//! work in a COMMENT on a production path. Increment 1 (`gap_refs.rs`) covers the
//! macro class (`todo!`/`unimplemented!`); this is the INVERSE on comments — the
//! hedges are prose, so we look *inside* comments — and it is the class that
//! caused GAP-001.
//!
//! # Scope — read before extending
//! Scans **comment lines only** (a line whose trimmed start is `//`, covering
//! `//`, `///`, `//!`) on **production paths** (any path component in
//! {tests, examples, benches, target} is excluded; `#[cfg(test)]`-gated regions
//! are excluded via the same naive brace-depth tracking Increment 1 uses). A
//! comment line is a **hedge** if it contains any of these 14 patterns:
//!   - Case-INSENSITIVE substrings: `for now`, `not yet`, `doesn't yet`,
//!     `shortcut`, `bails`, `would need`, `stopgap`, `workaround`, `naive`,
//!     `temporary` (checked against the lowercased line).
//!   - Case-SENSITIVE substrings (uppercase-marker convention): `TODO`, `FIXME`,
//!     `XXX`, `HACK` (checked against the original line).
//!
//! No regex crate — std only.
//!
//! # The GAP-001 near-miss (why the pattern set leans inclusive)
//! The founding case (jit_cuda_load.rs `//!` block) is caught by exactly ONE of
//! the fourteen patterns — `shortcut` — and NOT by `would need` (the text says
//! 'would read', not 'would need'). The margin is one word. Preserved so anyone
//! pruning the pattern set for noise sees the stakes.
//!
//! Decision rule that follows: a false positive costs one allowlist line; a
//! false negative costs a GAP-001. So the set leans inclusive and the allowlist
//! absorbs the noise.
//!
//! # Why `placeholder` is NOT a pattern
//! `placeholder` has ~237 hits in the tree, and FKC uses `~` placeholders as a
//! documented, legitimate construct — flagging `placeholder` would flag correct
//! behaviour, not a gap. It is deliberately excluded.
//!
//! # Allowlist mechanism (`gap_hedge_allowlist.txt`) — the core of inc-2
//! A checked-in baseline of every hedge that already existed at authoring time,
//! so the gate fails only on *new* untracked hedges. Each entry is a STABLE,
//! line-number-independent key: `<workspace-relative path, forward slashes>` +
//! TAB + `<trimmed comment text>`, one per line, **sorted** for stable diffs.
//! Per production comment-line hedge:
//!   - contains `GAP(GAP-` → **tracked** (OK; NOT added to the allowlist).
//!   - else key in the allowlist → **grandfathered** (OK).
//!   - else → **VIOLATION** (a new untracked hedge; CI fails until it is
//!     GAP-referenced, reworded, or the line is deliberately allowlisted).
//!
//! # "May only shrink" teeth
//! After scanning, any allowlist key NOT found in the current tree is **STALE**
//! and FAILS the gate, listing the stale keys. This forces removal of a hedge's
//! allowlist line when the hedge itself is removed, so the allowlist cannot rot
//! into permanent amnesty and its size is a real, monotonically-shrinking metric
//! of outstanding prose debt. Regenerate deliberately (never to paper over a new
//! hedge) via the ignored `regenerate_allowlist` test below.
//!
//! # Policy decisions (documented so a later disagreement sees a decision)
//!   - **`examples/` excluded — a JUDGMENT CALL.** An example program is not a
//!     *test* path, so by the letter of the rule it is arguably in scope; it is
//!     excluded anyway because a hedge in a demo is honest "this sample doesn't
//!     cover that" behaviour, not a hidden production gap. (`tests/`, `benches/`,
//!     `target/` excluded for the ordinary reason: not production source.)
//!   - **`#[cfg(test)]` regions are test code.** Skipped via naive brace-depth
//!     tracking (count `{` minus `}` per line), same as Increment 1.
//!   - **Bare `#[test]` fns are test code too.** A `#[test]`-attributed fn is a
//!     test even when NOT wrapped in a `#[cfg(test)] mod`; its body is skipped
//!     by the same brace-depth mechanism, armed on the `#[test]` line.
//!     (`"#[test]"` is not a substring of `"#[cfg(test)]"`, so no double-fire.)
//!   - **Files included via a parent's `#[cfg(test)] mod <stem>;` are test
//!     code.** Such a file has no top-level `#[cfg(test)]`; the whole-tree loop
//!     reads the candidate parent-module sources (`.../mod.rs`, the sibling
//!     `.../<dir>.rs`, and `lib.rs`/`main.rs` for a file directly under `src/`)
//!     and excludes the file when a parent gates it. Real cases:
//!     `dlpack/header_check.rs` and `dlpack/validate/tests.rs`.
//!   - **Comments only.** A hedge phrase inside a string literal on a code line
//!     (`let s = "not yet ready";`) is NOT scanned — only lines whose trimmed
//!     start is `//`.
//!
//! # Self-reporting (coverage is stated, not silent)
//! The whole-tree test prints the tally (hedges seen / tracked / grandfathered /
//! new violations / allowlist size / stale count) AND a per-phrase breakdown
//! (one count per pattern across the whole tree), so a green result advertises
//! not just that the ratchet held but *which* pattern class is being paid down
//! versus which is inert — an evidence base for any future pruning argument.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// The 14 hedge patterns, `(substring, case_sensitive)`. Order is STABLE — the
/// per-phrase breakdown prints in this order. Case-insensitive patterns are
/// stored lowercase and matched against the lowercased line; case-sensitive
/// patterns (the uppercase-marker convention) are matched against the original.
const PATTERNS: [(&str, bool); 14] = [
    ("for now", false),
    ("not yet", false),
    ("doesn't yet", false),
    ("shortcut", false),
    ("bails", false),
    ("would need", false),
    ("stopgap", false),
    ("workaround", false),
    ("naive", false),
    ("temporary", false),
    ("TODO", true),
    ("FIXME", true),
    ("XXX", true),
    ("HACK", true),
];

/// Indices of every pattern that matches `line`. Empty ⇒ not a hedge.
fn matched_patterns(line: &str) -> Vec<usize> {
    let lower = line.to_lowercase();
    let mut out = Vec::new();
    for (i, (pat, case_sensitive)) in PATTERNS.iter().enumerate() {
        let hit = if *case_sensitive {
            line.contains(pat)
        } else {
            lower.contains(pat)
        };
        if hit {
            out.push(i);
        }
    }
    out
}

/// One production comment-line hedge.
struct HedgeHit {
    /// Stable allowlist key: `<rel-path>\t<trimmed comment text>`.
    key: String,
    /// Carries a same-or-inline `GAP(GAP-` reference ⇒ tracked, not allowlisted.
    tracked: bool,
    /// Human-readable violation description (`path:line: text — …`).
    desc: String,
    /// Indices into `PATTERNS` that this line matched (for the breakdown).
    patterns: Vec<usize>,
}

/// Per-file scan: every production comment-line hedge in `text`, applying the
/// comment-line and `#[cfg(test)]`-region exclusions. `rel_path` is the
/// workspace-relative, forward-slashed path used to build stable keys. This is
/// the single source of truth; `scan_hedges` and the whole-tree gate both use
/// it, so the enforced logic and the generated baseline can never diverge.
///
fn scan_file_hedges(rel_path: &str, text: &str) -> Vec<HedgeHit> {
    let mut hits = Vec::new();

    // Naive brace-depth tracking to find the extent of `#[cfg(test)]` blocks —
    // the same scheme Increment 1 uses.
    let mut depth: i32 = 0;
    let mut pending_cfg_test = false;
    // When `Some(d)`, we are inside a cfg(test) block and skip until brace depth
    // returns to `d` (the depth just before the block opened).
    let mut skip_until_depth: Option<i32> = None;

    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();
        let is_comment = line.trim_start().starts_with("//");

        // Detect the start of a test region. `#[cfg(test)]` gates a whole
        // module/block; a bare `#[test]` marks a single test fn whose body is
        // likewise test code (real case: dlpack/header_check.rs, a `#[test]` fn
        // not wrapped in a `#[cfg(test)] mod`). `"#[test]"` is NOT a substring of
        // `"#[cfg(test)]"`, so a `#[cfg(test)]` line never double-triggers here.
        // Guard against re-arming while already inside a region so a nested
        // attribute cannot corrupt the depth we must return to (naive safety).
        if skip_until_depth.is_none()
            && (line.contains("#[cfg(test)]") || line.contains("#[test]"))
        {
            pending_cfg_test = true;
        }

        let opens = line.matches('{').count() as i32;
        let closes = line.matches('}').count() as i32;

        // A pending cfg(test) attribute's gated block opens on the first line
        // that introduces a brace. Skip from here until depth unwinds.
        if pending_cfg_test && opens > 0 {
            skip_until_depth = Some(depth);
            pending_cfg_test = false;
        }

        let in_cfg_test = skip_until_depth.is_some();

        if is_comment && !in_cfg_test {
            let patterns = matched_patterns(line);
            if !patterns.is_empty() {
                let tracked = line.contains("GAP(GAP-");
                let key = format!("{rel_path}\t{trimmed}");
                let desc = format!(
                    "{rel_path}:{line_no}: {trimmed} — new untracked prose hedge \
                     (add // GAP(GAP-NNN), reword, or allowlist)"
                );
                hits.push(HedgeHit { key, tracked, desc, patterns });
            }
        }

        depth += opens - closes;
        if let Some(d) = skip_until_depth
            && depth <= d {
                skip_until_depth = None;
            }
    }

    hits
}

/// Violation descriptions for each un-allowlisted, un-tracked prose hedge in
/// `text`. Pure function of its inputs — the unit tests below pin BOTH
/// directions (flags a new hedge AND passes an allowlisted / GAP-referenced one).
fn scan_hedges(path: &str, text: &str, allowlist: &HashSet<String>) -> Vec<String> {
    scan_file_hedges(path, text)
        .into_iter()
        .filter(|h| !h.tracked && !allowlist.contains(&h.key))
        .map(|h| h.desc)
        .collect()
}

/// Allowlist keys that were NOT produced by any hedge in the current tree — the
/// "may only shrink" teeth. `keys_seen` is every production hedge key found.
///
fn stale_allowlist_keys(allowlist: &HashSet<String>, keys_seen: &HashSet<String>) -> Vec<String> {
    allowlist
        .iter()
        .filter(|k| !keys_seen.contains(*k))
        .cloned()
        .collect()
}

/// Walk up from `CARGO_MANIFEST_DIR` until a `Cargo.toml` containing
/// `[workspace]` is found; that directory is the workspace root.
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists()
            && let Ok(txt) = std::fs::read_to_string(&manifest)
                && txt.contains("[workspace]") {
                    return dir;
                }
        if !dir.pop() {
            panic!("could not find a Cargo.toml containing [workspace] above CARGO_MANIFEST_DIR");
        }
    }
}

/// True if any component of `path` is in {tests, examples, benches, target}.
fn excluded_component(path: &Path) -> Option<&'static str> {
    for c in path.components() {
        match c.as_os_str().to_str() {
            Some("tests") => return Some("tests"),
            Some("examples") => return Some("examples"),
            Some("benches") => return Some("benches"),
            Some("target") => return Some("target"),
            _ => {}
        }
    }
    None
}

/// Recursively collect `.rs` files under `root`, skipping `target/` and `.git/`.
fn collect_rs_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name == "target" || name == ".git" {
                    continue;
                }
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    out
}

/// Workspace-relative path with forward slashes (the key's path component).
fn workspace_relative(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

/// True if `parent_text` declares an external `mod <stem>;` (semicolon form, so
/// `<stem>` lives in its own file) attributed `#[cfg(test)]` on the same line or
/// the immediately-preceding line. Pure text function — unit-tested both ways.
fn parent_declares_mod_cfg_test(parent_text: &str, stem: &str) -> bool {
    let needle = format!("mod {stem};");
    let lines: Vec<&str> = parent_text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains(&needle) {
            if line.contains("#[cfg(test)]") {
                return true;
            }
            if i > 0 && lines[i - 1].contains("#[cfg(test)]") {
                return true;
            }
        }
    }
    false
}

/// True if `path`'s parent module includes it via a `#[cfg(test)]`-gated
/// `mod <stem>;`, making the WHOLE file test code even though the file itself
/// carries no top-level `#[cfg(test)]`. Candidate parent-module sources for
/// `.../a/b.rs`: `.../a/mod.rs`, the sibling `.../a.rs`, and — for a file
/// directly under `crate/src/` — `.../lib.rs` / `.../main.rs`. Real cases:
/// `dlpack/header_check.rs` (via `mod.rs`) and `dlpack/validate/tests.rs` (via
/// the sibling `validate.rs`).
fn parent_gates_as_cfg_test(path: &Path) -> bool {
    let stem = match path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return false,
    };
    let parent = match path.parent() {
        Some(p) => p,
        None => return false,
    };
    let mut candidates: Vec<PathBuf> = vec![parent.join("mod.rs")];
    if let Some(dir_name) = parent.file_name().and_then(|s| s.to_str()) {
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join(format!("{dir_name}.rs")));
        }
        if dir_name == "src" {
            candidates.push(parent.join("lib.rs"));
            candidates.push(parent.join("main.rs"));
        }
    }
    for cand in candidates {
        if cand == path {
            continue; // a file is never its own parent module
        }
        if let Ok(txt) = std::fs::read_to_string(&cand)
            && parent_declares_mod_cfg_test(&txt, stem) {
                return true;
            }
    }
    false
}

/// Path to the checked-in baseline allowlist.
fn allowlist_path(root: &Path) -> PathBuf {
    root.join("fuel-ir").join("tests").join("gap_hedge_allowlist.txt")
}

/// Load the allowlist as a set of keys. Missing file ⇒ empty set.
fn load_allowlist(root: &Path) -> HashSet<String> {
    let txt = std::fs::read_to_string(allowlist_path(root)).unwrap_or_default();
    txt.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

#[cfg(test)]
mod unit {
    use super::*;

    fn allowlist(keys: &[&str]) -> HashSet<String> {
        keys.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flags_unallowlisted_hedge() {
        let v = scan_hedges("x.rs", "    // this is a shortcut for now", &allowlist(&[]));
        assert_eq!(v.len(), 1, "a new un-allowlisted hedge must be flagged: {v:?}");
    }

    #[test]
    fn passes_allowlisted_hedge() {
        let key = "x.rs\t// this is a shortcut for now";
        let v = scan_hedges("x.rs", "    // this is a shortcut for now", &allowlist(&[key]));
        assert_eq!(v.len(), 0, "an allowlisted hedge must pass: {v:?}");
    }

    #[test]
    fn passes_gap_referenced_hedge() {
        let v = scan_hedges("x.rs", "    // shortcut for now GAP(GAP-001)", &allowlist(&[]));
        assert_eq!(v.len(), 0, "a GAP-referenced hedge must pass: {v:?}");
    }

    #[test]
    fn ignores_non_comment_line() {
        // The phrase lives in a string on a code line, not a comment.
        let v = scan_hedges("x.rs", "    let s = \"not yet ready\";", &allowlist(&[]));
        assert_eq!(v.len(), 0, "only comment lines are scanned: {v:?}");
    }

    #[test]
    fn skips_cfg_test_hedge() {
        let text = "#[cfg(test)]\nmod tests {\n  // for now\n}\n";
        let v = scan_hedges("x.rs", text, &allowlist(&[]));
        assert_eq!(v.len(), 0, "hedges inside #[cfg(test)] are test code: {v:?}");
    }

    #[test]
    fn flags_uppercase_marker() {
        let v = scan_hedges("x.rs", "    // TODO: wire this", &allowlist(&[]));
        assert_eq!(v.len(), 1, "an uppercase TODO marker must be flagged: {v:?}");
        // Lowercase "todo" is NOT one of the case-insensitive phrases, and the
        // case-sensitive TODO marker requires uppercase — so this must not match.
        let v = scan_hedges("x.rs", "    // todo lowercase prose", &allowlist(&[]));
        assert_eq!(v.len(), 0, "lowercase 'todo' prose must not match the marker: {v:?}");
    }

    #[test]
    fn skips_bare_test_fn_hedge() {
        // A hedge inside a bare `#[test] fn` (NOT wrapped in a `#[cfg(test)] mod`)
        // is test code. Real case: dlpack/header_check.rs.
        let text = "#[test]\nfn t() {\n  // for now\n}\n";
        let v = scan_hedges("x.rs", text, &allowlist(&[]));
        assert_eq!(v.len(), 0, "a hedge inside a bare #[test] fn must be excluded: {v:?}");
    }

    #[test]
    fn still_flags_production_hedge_outside_test() {
        // Control (other direction): a hedge NOT inside any test construct must
        // still be flagged — the fix must not over-broaden into ordinary code.
        let text = "fn real() {\n  // for now\n}\n";
        let v = scan_hedges("x.rs", text, &allowlist(&[]));
        assert_eq!(v.len(), 1, "a production hedge must still be flagged: {v:?}");
    }

    #[test]
    fn parent_cfg_test_detection_both_directions() {
        // A `#[cfg(test)]`-attributed `mod foo;` in a parent gates foo.rs as test.
        assert!(parent_declares_mod_cfg_test("#[cfg(test)]\nmod foo;", "foo"));
        // A plain `mod foo;` (no attribute) does not.
        assert!(!parent_declares_mod_cfg_test("mod foo;", "foo"));
    }

    #[test]
    fn detects_stale_allowlist_key() {
        let al = allowlist(&["a.rs\t// foo"]);
        // Key absent from the tree ⇒ stale.
        let seen: HashSet<String> = HashSet::new();
        assert_eq!(stale_allowlist_keys(&al, &seen), vec!["a.rs\t// foo".to_string()]);
        // Same key present ⇒ not stale.
        let seen: HashSet<String> = ["a.rs\t// foo".to_string()].into_iter().collect();
        assert!(stale_allowlist_keys(&al, &seen).is_empty());
    }
}

/// Whole-tree gate: no NEW untracked prose hedge on any production `.rs` path,
/// and no STALE allowlist key. Prints the tally + a per-phrase breakdown first,
/// so a green result states its own coverage rather than silently bounding it.
#[test]
fn no_new_prose_hedge_in_workspace() {
    let root = workspace_root();
    let allowlist = load_allowlist(&root);
    let files = collect_rs_files(&root);

    let mut violations: Vec<String> = Vec::new();
    let mut tracked = 0usize;
    let mut grandfathered = 0usize;
    let mut seen = 0usize;
    let mut pattern_counts = [0usize; 14];
    let mut keys_seen: HashSet<String> = HashSet::new();
    let mut production_files = 0usize;
    let mut parent_cfg_test_files = 0usize;

    for path in &files {
        // Never scan this enforcer file itself (it is packed with literal hedge
        // phrases used as fixtures). It also lives under tests/ and is excluded
        // by path, but self-exclude explicitly per the Increment-1 pattern.
        if path.file_name().and_then(|f| f.to_str()) == Some("gap_hedges.rs") {
            continue;
        }
        if excluded_component(path).is_some() {
            continue;
        }
        // The WHOLE file is test code, included via a parent module's
        // `#[cfg(test)] mod <stem>;` even though the file itself carries no
        // top-level `#[cfg(test)]`. Real cases: dlpack/header_check.rs (via
        // mod.rs) and dlpack/validate/tests.rs (via the sibling validate.rs).
        if parent_gates_as_cfg_test(path) {
            parent_cfg_test_files += 1;
            continue;
        }
        production_files += 1;
        let rel = workspace_relative(&root, path);
        let text = std::fs::read_to_string(path).unwrap_or_default();
        for hit in scan_file_hedges(&rel, &text) {
            seen += 1;
            for &p in &hit.patterns {
                pattern_counts[p] += 1;
            }
            keys_seen.insert(hit.key.clone());
            if hit.tracked {
                tracked += 1;
            } else if allowlist.contains(&hit.key) {
                grandfathered += 1;
            } else {
                violations.push(hit.desc);
            }
        }
    }

    let stale = stale_allowlist_keys(&allowlist, &keys_seen);

    let breakdown = PATTERNS
        .iter()
        .enumerate()
        .map(|(i, (p, _))| format!("{} {}", p, pattern_counts[i]))
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "gap-hedge check: scanned {} production .rs files \
         (+{} excluded as parent-#[cfg(test)] modules); {} comment hedge(s) seen — \
         {} NEW violation(s), {} tracked (GAP-referenced), {} grandfathered (allowlisted); \
         allowlist size {}, {} stale.",
        production_files,
        parent_cfg_test_files,
        seen,
        violations.len(),
        tracked,
        grandfathered,
        allowlist.len(),
        stale.len(),
    );
    println!("gap-hedge per-phrase (whole tree): {breakdown}");

    let mut failure = String::new();
    if !violations.is_empty() {
        violations.sort();
        failure.push_str(&format!(
            "{} new untracked prose hedge(s) — add // GAP(GAP-NNN), reword, or \
             deliberately add the line to fuel-ir/tests/gap_hedge_allowlist.txt:\n{}\n",
            violations.len(),
            violations.join("\n"),
        ));
    }
    if !stale.is_empty() {
        let mut stale = stale;
        stale.sort();
        failure.push_str(&format!(
            "{} stale allowlist key(s) — the hedge was removed, so delete the line \
             (the allowlist may only shrink):\n{}\n",
            stale.len(),
            stale.join("\n"),
        ));
    }
    assert!(failure.is_empty(), "{failure}");
}

/// Regenerate the baseline allowlist from the current tree. Ignored by default;
/// run deliberately (NEVER to paper over a new hedge) with:
///   `GAP_HEDGE_REGEN=1 cargo test -p fuel-ir --test gap_hedges regenerate_allowlist -- --ignored --nocapture`
/// Shares `scan_file_hedges` with the gate, so the generated baseline can never
/// diverge from what the gate enforces. GAP-referenced hedges are tracked and
/// deliberately NOT written to the allowlist.
#[test]
#[ignore]
fn regenerate_allowlist() {
    if std::env::var("GAP_HEDGE_REGEN").is_err() {
        eprintln!("set GAP_HEDGE_REGEN=1 to actually rewrite gap_hedge_allowlist.txt");
        return;
    }
    let root = workspace_root();
    let files = collect_rs_files(&root);
    let mut keys: Vec<String> = Vec::new();
    for path in &files {
        if path.file_name().and_then(|f| f.to_str()) == Some("gap_hedges.rs") {
            continue;
        }
        if excluded_component(path).is_some() {
            continue;
        }
        // Mirror the gate's parent-`#[cfg(test)]` exclusion exactly, so the
        // regenerated baseline can never diverge from what the gate enforces.
        if parent_gates_as_cfg_test(path) {
            continue;
        }
        let rel = workspace_relative(&root, path);
        let text = std::fs::read_to_string(path).unwrap_or_default();
        for hit in scan_file_hedges(&rel, &text) {
            if !hit.tracked {
                keys.push(hit.key);
            }
        }
    }
    keys.sort();
    keys.dedup();
    let body = if keys.is_empty() {
        String::new()
    } else {
        format!("{}\n", keys.join("\n"))
    };
    std::fs::write(allowlist_path(&root), body).unwrap();
    eprintln!("wrote {} allowlist keys to gap_hedge_allowlist.txt", keys.len());
}
