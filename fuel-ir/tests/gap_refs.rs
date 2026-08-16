//! GAP-141 Increment 1 — CI guard: every `todo!()` / `unimplemented!()` on a
//! non-test path must carry a `// GAP(GAP-NNN)` registry reference, so an
//! incomplete-work marker cannot silently drop off the schedule. This is the
//! seam that stops incomplete-work markers from slipping off the roadmap.
//!
//! # Scope — read before extending
//! This check covers **`todo!(` and `unimplemented!(` ONLY**. It deliberately
//! does **NOT** cover:
//!   - `panic!(` — there are ~600+ uncounted `panic!` sites in the tree, which
//!     are their own class (tracked separately as GAP-016), not this gate.
//!   - **Prose hedges** — a `//!` / `///` doc comment (or an inline `//`) that
//!     admits a gap *in words* rather than as a macro call. GAP-001's founding
//!     case is exactly such a prose hedge (a `//!` doc comment). Prose hedges
//!     belong to **Increment 2**, so this check does **NOT** close that class.
//!
//! Do not read a green result here as "the tree has no unfinished work"; read
//! it as "no un-referenced `todo!`/`unimplemented!` macro on a production path".
//!
//! # Honest billing (what this gate did, and did not, accomplish)
//! On the tree at authoring time, **36 candidate `todo!`/`unimplemented!` sites
//! were triaged down to exactly ONE genuine production gap** (k_quants.rs's
//! `to_float` for Q8_1, already tracked as **GAP-125**). The other 35 are
//! test code (`#[cfg(test)]`), doc-comment mentions, or example programs. So
//! the `todo!`/`unimplemented!` macro surface was already essentially clean —
//! this gate **caught nothing historical**. Its value is entirely
//! **prospective**: it is a *ratchet* that fails CI the moment a *future*
//! untracked `todo!`/`unimplemented!` lands on a production path. The real
//! backlog of unfinished-work admissions lives in prose (Increment 2) and in
//! the `panic!` surface (GAP-016), neither of which this gate sees.
//!
//! # v1 policy decisions (documented so a later disagreement sees a decision,
//! not an oversight)
//!   - **Reference placement — same line.** A macro is considered tracked only
//!     when the `GAP(GAP-` marker appears on the SAME line as the macro
//!     (`unimplemented!(...) // GAP(GAP-NNN)`). Adjacent-comment references are
//!     out of scope for v1; keep the marker on the call line.
//!   - **Comment lines are prose, not calls.** A line whose trimmed start is
//!     `//` (covering `//`, `///`, `//!`) is skipped — a `todo!()` written
//!     inside a doc/line comment is documentation, not an invocation. (Known
//!     v1 simplification: a macro sitting in a *trailing* comment after real
//!     code — `let x = 1; // todo!() later` — would still be seen; no such
//!     line exists on a scanned production path today, so v1 does not special-
//!     case it.)
//!   - **`#[cfg(test)]` regions are test code.** A `#[cfg(test)]`-gated module
//!     or block is skipped via naive brace-depth tracking (count `{` minus `}`
//!     per line). The motivating real case is
//!     `fuel-backend-contract/src/storage.rs`'s `#[cfg(test)] mod stype_attach`
//!     whose `MockStorage` impl has 30 `unimplemented!()` stubs that must NOT
//!     be flagged.
//!   - **Bare `#[test]` fns are test code too.** A `#[test]`-attributed fn is a
//!     test even when it is NOT wrapped in a `#[cfg(test)] mod` — its body is
//!     skipped by the same brace-depth mechanism, armed on the `#[test]` line.
//!     (`"#[test]"` is not a substring of `"#[cfg(test)]"`, so this never
//!     double-fires.) Real case: `fuel-ir/src/dlpack/header_check.rs`.
//!   - **Files included via a parent's `#[cfg(test)] mod <stem>;` are test
//!     code.** Such a file has no top-level `#[cfg(test)]`, so a per-file scan
//!     would wrongly treat it as production; the whole-tree loop instead reads
//!     the candidate parent-module sources (`.../mod.rs`, the sibling
//!     `.../<dir>.rs`, and `lib.rs`/`main.rs` for a file directly under `src/`)
//!     and excludes the file when a parent gates it. Real case:
//!     `fuel-ir/src/dlpack/validate/tests.rs` (via the sibling `validate.rs`).
//!   - **`examples/` is excluded as a deliberate JUDGMENT CALL.** Strictly, an
//!     example program is not a *test* path, so by the letter of the rule an
//!     `examples/**` file is arguably in scope. It is excluded anyway, on
//!     purpose: an example that declines to handle a case (`todo!()`) is honest
//!     *demo* behaviour — "this sample doesn't cover that" — not a hidden
//!     production gap that could ship and surprise a user. Excluding it keeps
//!     the gate pointed at production code. This is a decision, not an
//!     oversight; revisit it if example-path gaps ever start masking real work.
//!     (`tests/`, `benches/`, and `target/` are excluded for the ordinary
//!     reason: not production code / not source.)
//!
//! # Self-reporting (coverage is stated, not silent)
//! The whole-tree test prints per-category tallies of every
//! `todo!`/`unimplemented!` occurrence it saw (violation / tracked-with-ref /
//! excluded-by-reason), so a green result advertises *what it excluded* rather
//! than silently bounding its own coverage. A check that quietly narrows its
//! own scope makes "clean" read as "clean everywhere" when it means "clean in
//! the part I looked at" — the exact silent-truncation failure this registry
//! exists to prevent.

use std::path::{Path, PathBuf};

/// Per-file categorisation of every `todo!`/`unimplemented!` occurrence found
/// on a *production* (non-path-excluded) file.
#[derive(Default)]
struct FileScan {
    /// Un-referenced, non-comment, non-`cfg(test)` sites — the failures.
    violations: Vec<String>,
    /// Sites carrying a same-line `GAP(GAP-` reference.
    tracked: usize,
    /// Occurrences inside a `//` / `///` / `//!` comment line.
    excluded_comment: usize,
    /// Occurrences inside a `#[cfg(test)]`-gated block.
    excluded_cfg_test: usize,
}

/// Count `todo!(` + `unimplemented!(` textual occurrences on a single line.
fn macro_occurrences(line: &str) -> usize {
    line.matches("todo!(").count() + line.matches("unimplemented!(").count()
}

/// Categorise every `todo!(`/`unimplemented!(` occurrence in `text`, applying
/// the comment-line and `#[cfg(test)]`-region exclusions. This is the single
/// source of truth; `scan_source` is a thin wrapper that returns only the
/// violation descriptions.
fn scan_production_file(path: &str, text: &str) -> FileScan {
    let mut scan = FileScan::default();

    // Naive brace-depth tracking to find the extent of `#[cfg(test)]` blocks.
    let mut depth: i32 = 0;
    let mut pending_cfg_test = false;
    // When `Some(d)`, we are inside a cfg(test) block and skip until brace
    // depth returns to `d` (the depth just before the block opened).
    let mut skip_until_depth: Option<i32> = None;

    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim_start();
        let is_comment = trimmed.starts_with("//");

        // Detect the start of a test region. `#[cfg(test)]` gates a whole
        // module/block; a bare `#[test]` marks a single test fn whose body is
        // likewise test code (real case: dlpack/header_check.rs, a `#[test]` fn
        // not wrapped in a `#[cfg(test)] mod`). `"#[test]"` is NOT a substring of
        // `"#[cfg(test)]"`, so a `#[cfg(test)]` line never double-triggers here.
        // Guard against re-arming while already inside a region so a nested
        // attribute cannot corrupt the depth we must return to (naive safety).
        if skip_until_depth.is_none() && (line.contains("#[cfg(test)]") || line.contains("#[test]"))
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
        let occ = macro_occurrences(line);
        if occ > 0 {
            if is_comment {
                scan.excluded_comment += occ;
            } else if in_cfg_test {
                scan.excluded_cfg_test += occ;
            } else if line.contains("GAP(GAP-") {
                scan.tracked += occ;
            } else {
                // One violation record per offending line; on this tree each
                // macro sits on its own line so line-count == occurrence-count.
                scan.violations.push(format!(
                    "{}:{}: {} — missing // GAP(GAP-NNN) reference",
                    path,
                    line_no,
                    trimmed.trim_end()
                ));
            }
        }

        depth += opens - closes;
        if let Some(d) = skip_until_depth
            && depth <= d
        {
            skip_until_depth = None;
        }
    }

    scan
}

/// Return a violation description for each un-referenced `todo!(`/
/// `unimplemented!(` in `text`, applying the comment-line and `#[cfg(test)]`
/// exclusions. Pure function of its inputs — the unit tests below pin both
/// directions (flag AND pass).
fn scan_source(path: &str, text: &str) -> Vec<String> {
    scan_production_file(path, text).violations
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
            && parent_declares_mod_cfg_test(&txt, stem)
        {
            return true;
        }
    }
    false
}

/// Walk up from `CARGO_MANIFEST_DIR` until a `Cargo.toml` containing
/// `[workspace]` is found; that directory is the workspace root.
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists()
            && let Ok(txt) = std::fs::read_to_string(&manifest)
            && txt.contains("[workspace]")
        {
            return dir;
        }
        if !dir.pop() {
            panic!("could not find a Cargo.toml containing [workspace] above CARGO_MANIFEST_DIR");
        }
    }
}

/// The first path component in {tests, examples, benches, target} that appears
/// in `path`, if any. OS-agnostic: `Path::components` splits on the platform
/// separator.
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

/// Recursively collect `.rs` files under `root`, descending into everything
/// except `target/` and `.git/` (build output / VCS — never source). Path-based
/// exclusions (tests/examples/benches) are applied by the caller so those files
/// can still be *tallied*.
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

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn flags_unreferenced_todo() {
        assert_eq!(scan_source("x.rs", "    todo!();").len(), 1);
    }

    #[test]
    fn passes_referenced_todo() {
        assert_eq!(scan_source("x.rs", "    todo!(); // GAP(GAP-042)").len(), 0);
    }

    #[test]
    fn passes_referenced_unimplemented() {
        assert_eq!(
            scan_source("x.rs", "    unimplemented!(); // GAP(GAP-010)").len(),
            0
        );
    }

    #[test]
    fn skips_comment_line() {
        // A `todo!()` inside a doc comment is prose, not a call.
        assert_eq!(scan_source("x.rs", "    /// example: todo!()").len(), 0);
    }

    #[test]
    fn skips_cfg_test_block() {
        let text = "#[cfg(test)]\nmod tests {\n  fn m() { unimplemented!() }\n}\n";
        assert_eq!(scan_source("x.rs", text).len(), 0);
    }

    #[test]
    fn skips_bare_test_fn_body() {
        // A `todo!()` inside a bare `#[test] fn` (NOT wrapped in a
        // `#[cfg(test)] mod`) is test code. Real case: dlpack/header_check.rs.
        let text = "#[test]\nfn t() {\n  todo!()\n}\n";
        assert_eq!(
            scan_source("x.rs", text).len(),
            0,
            "a bare #[test] fn body must be excluded"
        );
    }

    #[test]
    fn still_flags_production_todo_outside_test() {
        // Control (other direction): a genuine production todo!() that is NOT
        // inside any test construct must still be flagged — the fix must not
        // over-broaden into ordinary code.
        let text = "fn real() {\n  todo!()\n}\n";
        assert_eq!(
            scan_source("x.rs", text).len(),
            1,
            "a production todo!() must still be flagged"
        );
    }

    #[test]
    fn parent_cfg_test_detection_both_directions() {
        // A `#[cfg(test)]`-attributed `mod foo;` in a parent gates foo.rs as test.
        assert!(parent_declares_mod_cfg_test(
            "#[cfg(test)]\nmod foo;",
            "foo"
        ));
        // A plain `mod foo;` (no attribute) does not.
        assert!(!parent_declares_mod_cfg_test("mod foo;", "foo"));
    }
}

/// Whole-tree gate: no un-referenced `todo!`/`unimplemented!` on any production
/// `.rs` path anywhere in the workspace. Prints per-category tallies first so a
/// green result states its own coverage rather than silently bounding it.
#[test]
fn no_unreferenced_todo_or_unimplemented_in_workspace() {
    let root = workspace_root();
    let files = collect_rs_files(&root);

    let mut violations: Vec<String> = Vec::new();
    let mut tracked = 0usize;
    let mut excluded_comment = 0usize;
    let mut excluded_cfg_test = 0usize;
    let mut excluded_examples = 0usize;
    let mut excluded_tests = 0usize;
    let mut excluded_benches = 0usize;
    let mut production_files = 0usize;
    let mut path_excluded_files = 0usize;
    let mut parent_cfg_test_files = 0usize;

    for path in &files {
        // Never scan this enforcer file itself (it is packed with literal
        // `todo!(`/`unimplemented!(` strings used as test fixtures).
        if path.file_name().and_then(|f| f.to_str()) == Some("gap_refs.rs") {
            continue;
        }

        let text = std::fs::read_to_string(path).unwrap_or_default();
        let display = path.to_string_lossy().to_string();

        match excluded_component(path) {
            Some("target") => {
                // Build output — not source; ignore entirely.
            }
            Some(bucket) => {
                // Path-excluded (tests/examples/benches): tally, do not enforce.
                path_excluded_files += 1;
                let occ: usize = text.lines().map(macro_occurrences).sum();
                match bucket {
                    "examples" => excluded_examples += occ,
                    "tests" => excluded_tests += occ,
                    "benches" => excluded_benches += occ,
                    _ => {}
                }
            }
            None if parent_gates_as_cfg_test(path) => {
                // The WHOLE file is test code, included via a parent module's
                // `#[cfg(test)] mod <stem>;` (the file itself carries no
                // top-level `#[cfg(test)]`). Exclude like a path-excluded file.
                parent_cfg_test_files += 1;
            }
            None => {
                // Production file — categorise and enforce.
                production_files += 1;
                let scan = scan_production_file(&display, &text);
                tracked += scan.tracked;
                excluded_comment += scan.excluded_comment;
                excluded_cfg_test += scan.excluded_cfg_test;
                violations.extend(scan.violations);
            }
        }
    }

    let total_seen = violations.len()
        + tracked
        + excluded_comment
        + excluded_cfg_test
        + excluded_examples
        + excluded_tests
        + excluded_benches;

    println!(
        "gap-ref check: scanned {} production .rs files (+{} excluded by path, \
         +{} excluded as parent-#[cfg(test)] modules); \
         {} todo!/unimplemented! occurrences seen — \
         {} untracked violation(s), {} tracked (GAP-referenced), \
         {} in #[cfg(test)], {} doc/line-comment mention(s), \
         {} in examples/, {} in tests/, {} in benches/.",
        production_files,
        path_excluded_files,
        parent_cfg_test_files,
        total_seen,
        violations.len(),
        tracked,
        excluded_cfg_test,
        excluded_comment,
        excluded_examples,
        excluded_tests,
        excluded_benches,
    );

    assert!(
        violations.is_empty(),
        "untracked todo!/unimplemented! (add // GAP(GAP-NNN) or move to increment-2 policy):\n{}",
        violations.join("\n")
    );
}
