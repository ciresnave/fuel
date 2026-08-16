//! GAP-049 — marker-integrity guard for the `EXHAUSTIVE-BY-DESIGN` convention.
//!
//! # What this enforces
//! An enum whose definition carries the `EXHAUSTIVE-BY-DESIGN` marker (a comment
//! in the attribute/doc block directly above `pub enum …`) declares that it is a
//! *closed* set: every consumer is expected to `match` it exhaustively, so adding
//! a variant is a deliberate breaking change that the compiler must surface at
//! every match site. Such an enum must therefore **never** also be
//! `#[non_exhaustive]` — that attribute forces downstream *wildcard* arms, which
//! converts the very compile error the marker relies on into a silent runtime
//! fallthrough. The two declarations are contradictory; this test makes them
//! **mutually exclusive by test rather than by discipline** (seams beat
//! vigilance).
//!
//! Origin: GAP-049. `feff38ed` added `Scalar::F8E8M0(u8)` and left two exhaustive
//! consumer matches un-updated; `main` went RED on a default build. The ruling
//! (architect, 2026-08-07) was that the compile break is the *correct* outcome
//! for an exhaustive-by-design enum, and that the extension model should be an
//! explicit per-enum declaration recorded at the definition site. This guard is
//! the seam that keeps that declaration honest.
//!
//! # Why two assertions (the vacuity trap)
//! A guard that can only ever pass is worthless — it asserts a real claim while
//! its name promises coverage it never had (the GAP-028 lesson: a sabotage that
//! "caught nothing" because the guarded path was never reached). So this test
//! ALSO asserts it actually *found* the enums we know are marked. If the scan
//! silently matches nothing — marker renamed, parser drift, files moved — the
//! reach assertion fails instead of passing vacuously. Both assertions were
//! observed to go RED under deliberate sabotage before this landed (see the
//! GAP-049 handoff): the invariant reddens when a marked enum is given
//! `#[non_exhaustive]`; the reach assertion reddens when a marker is removed.
//!
//! # Coverage is stated, not silent
//! The whole-tree test prints the set of marked enums it found, so a green result
//! advertises what it covered rather than silently bounding itself — same
//! discipline as the GAP-141 gate in `gap_refs.rs`, whose workspace-scan helpers
//! this file mirrors.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The marker token. Kept as a bare token (no backticks) so the human-readable
/// tail of a marker line is free-form; the scan keys only on this substring.
const MARKER: &str = "EXHAUSTIVE-BY-DESIGN";

/// Enums we KNOW carry the marker. The reach assertion requires every one of
/// these to be found, which is what makes the whole-tree scan non-vacuous.
/// Extending the marker to a new enum means adding it here in the same change —
/// deliberately, so a new closed-enum declaration is a reviewed act, not a
/// silent one.
const EXPECTED_MARKED: &[&str] = &["Scalar", "DType", "Op"];

/// One enum definition whose preceding attribute/doc block carries [`MARKER`].
#[derive(Debug, PartialEq, Eq)]
struct MarkedEnum {
    name: String,
    line: usize,
    /// Whether that same contiguous block also carries a `#[non_exhaustive]`
    /// attribute line — the contradiction this guard exists to catch.
    non_exhaustive: bool,
}

/// Parse the enum name out of a declaration line's trimmed head, or `None` if the
/// line is not an enum declaration. Handles `enum X`, `pub enum X`,
/// `pub(crate) enum X`, and generics (`enum X<T>` → `X`). Rejects `pub struct`,
/// `fn enumerate`, comments, and ordinary code — the `enum` keyword must sit at a
/// token boundary at the start of the (visibility-stripped) line.
fn enum_decl_name(trimmed: &str) -> Option<String> {
    let mut rest = trimmed;
    if let Some(after_pub) = rest.strip_prefix("pub") {
        // Only treat "pub" as the visibility keyword when a boundary follows
        // (`pub `, `pub(`), so "public"/"pubfoo" are not mistaken for it.
        if after_pub.starts_with(char::is_whitespace) || after_pub.starts_with('(') {
            rest = after_pub.trim_start();
            if rest.starts_with('(') {
                // Skip a `pub(crate)` / `pub(in path)` restriction.
                let close = rest.find(')')?;
                rest = rest[close + 1..].trim_start();
            }
        }
    }
    let after_enum = rest.strip_prefix("enum")?;
    // Require whitespace immediately after `enum` (so `enumerate` is rejected).
    if !after_enum.starts_with(char::is_whitespace) {
        return None;
    }
    let name: String = after_enum
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

/// Return every enum in `text` whose contiguous attribute/comment block directly
/// above the declaration contains [`MARKER`], flagged with whether that block
/// also carries a `#[non_exhaustive]` attribute.
///
/// The `#[non_exhaustive]` check is EXACT line-equality, not a substring search:
/// a marker line is free to *mention* `#[non_exhaustive]` in prose (e.g. "never
/// add #[non_exhaustive] here") without the guard mistaking that prose for the
/// real attribute. Pure function of its input — unit-tested both directions.
fn marked_enums(text: &str) -> Vec<MarkedEnum> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let name = match enum_decl_name(line.trim_start()) {
            Some(n) => n,
            None => continue,
        };
        // Walk upward over the contiguous block of attribute (`#…`) and comment
        // (`//…`) lines directly above the declaration; stop at the first blank
        // or code line. Both the marker and any `#[non_exhaustive]` attribute
        // that apply to this enum live in exactly this block.
        let mut has_marker = false;
        let mut non_exhaustive = false;
        let mut j = i;
        while j > 0 {
            j -= 1;
            let t = lines[j].trim();
            let is_block_line = t.starts_with('#') || t.starts_with("//");
            if !is_block_line {
                break;
            }
            if t.contains(MARKER) {
                has_marker = true;
            }
            if t == "#[non_exhaustive]" {
                non_exhaustive = true;
            }
        }
        if has_marker {
            out.push(MarkedEnum {
                name,
                line: i + 1,
                non_exhaustive,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Workspace-scan helpers — mirrored from `gap_refs.rs` (each integration-test
// file is its own crate, so the helpers are duplicated rather than shared).
// ---------------------------------------------------------------------------

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

/// Recursively collect `.rs` files under `root`, skipping `target/` and `.git/`
/// (build output / VCS — never source).
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

// ---------------------------------------------------------------------------
// Pure-function unit tests for the parser — both directions pinned.
// ---------------------------------------------------------------------------

#[test]
fn parses_enum_names() {
    assert_eq!(
        enum_decl_name("pub enum Scalar {").as_deref(),
        Some("Scalar")
    );
    assert_eq!(enum_decl_name("enum Bar {").as_deref(), Some("Bar"));
    assert_eq!(
        enum_decl_name("pub(crate) enum Baz {").as_deref(),
        Some("Baz")
    );
    assert_eq!(enum_decl_name("pub enum Gen<T> {").as_deref(), Some("Gen"));
    assert_eq!(enum_decl_name("pub struct NotAnEnum {"), None);
    assert_eq!(enum_decl_name("fn enumerate() {"), None);
    assert_eq!(enum_decl_name("// enum InAComment {"), None);
}

#[test]
fn marked_clean_enum_is_found_not_flagged() {
    let text = "\
/// doc
// EXHAUSTIVE-BY-DESIGN: closed set — gate every consumer.
#[derive(Debug)]
pub enum Foo {
    A,
}
";
    let got = marked_enums(text);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].name, "Foo");
    assert!(
        !got[0].non_exhaustive,
        "clean marked enum must not be flagged"
    );
}

#[test]
fn marked_and_non_exhaustive_is_flagged() {
    // The exact contradiction the whole-tree guard rejects.
    let text = "\
// EXHAUSTIVE-BY-DESIGN: closed set.
#[derive(Debug)]
#[non_exhaustive]
pub enum Foo {
    A,
}
";
    let got = marked_enums(text);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].non_exhaustive,
        "marker + #[non_exhaustive] must be flagged"
    );
}

#[test]
fn unmarked_enum_is_ignored_even_if_non_exhaustive() {
    // We only track MARKED enums; a plain #[non_exhaustive] enum is fine.
    let text = "\
#[non_exhaustive]
pub enum Foo {
    A,
}
";
    assert!(marked_enums(text).is_empty());
}

#[test]
fn prose_mention_of_attribute_does_not_false_positive() {
    // The marker line *mentions* the attribute in words; the real attribute is
    // absent. Exact line-equality detection must not treat the prose as the attr.
    let text = "\
// EXHAUSTIVE-BY-DESIGN: never add #[non_exhaustive] here — see GAP-049.
pub enum Foo {
    A,
}
";
    let got = marked_enums(text);
    assert_eq!(got.len(), 1);
    assert!(
        !got[0].non_exhaustive,
        "a prose mention of #[non_exhaustive] must not be read as the attribute"
    );
}

#[test]
fn blank_line_breaks_the_block() {
    // A marker separated from the enum by a blank line is NOT associated — the
    // convention requires the marker to sit in the contiguous block. This both
    // documents the placement rule and guards the parser against over-reach.
    let text = "\
// EXHAUSTIVE-BY-DESIGN: stray, detached from any enum.

pub enum Foo {
    A,
}
";
    assert!(marked_enums(text).is_empty());
}

// ---------------------------------------------------------------------------
// The whole-tree guard.
// ---------------------------------------------------------------------------

/// Every enum carrying the `EXHAUSTIVE-BY-DESIGN` marker anywhere in the
/// workspace must NOT also be `#[non_exhaustive]`, and every enum we expect to be
/// marked must actually be found (non-vacuity). Prints the found set first so a
/// green result states its own coverage.
#[test]
fn marked_enums_are_never_non_exhaustive() {
    let root = workspace_root();
    let files = collect_rs_files(&root);

    let mut all_marked: Vec<(String, String, usize, bool)> = Vec::new();
    for path in &files {
        // Never scan THIS file: it contains the marker token as fixtures and in
        // its own documentation, which would pollute the census.
        if path.file_name().and_then(|f| f.to_str()) == Some("exhaustive_by_design_marker.rs") {
            continue;
        }
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let display = path.to_string_lossy().to_string();
        for m in marked_enums(&text) {
            all_marked.push((m.name, display.clone(), m.line, m.non_exhaustive));
        }
    }

    let found_names: BTreeSet<&str> = all_marked.iter().map(|(n, _, _, _)| n.as_str()).collect();
    println!(
        "exhaustive-by-design marker check: found {} marked enum(s): {:?}",
        all_marked.len(),
        found_names
    );

    // (1) Reach / non-vacuity: every expected marked enum must be found. If the
    // scan silently matched nothing, this fails instead of passing vacuously.
    let missing: Vec<&str> = EXPECTED_MARKED
        .iter()
        .copied()
        .filter(|e| !found_names.contains(e))
        .collect();
    assert!(
        missing.is_empty(),
        "expected EXHAUSTIVE-BY-DESIGN marker on {missing:?} but did not find it — \
         the marker was removed, moved, or the scan is broken (this would make the \
         invariant check below vacuous)"
    );
    assert!(
        !all_marked.is_empty(),
        "no EXHAUSTIVE-BY-DESIGN markers found at all — scan is vacuous"
    );

    // (2) Invariant: no marked enum is also #[non_exhaustive].
    let violations: Vec<String> = all_marked
        .iter()
        .filter(|(_, _, _, ne)| *ne)
        .map(|(name, file, line, _)| format!("{file}:{line}: enum {name}"))
        .collect();
    assert!(
        violations.is_empty(),
        "these enums are marked EXHAUSTIVE-BY-DESIGN AND #[non_exhaustive], which is \
         contradictory — an exhaustive-by-design enum must stay exhaustive so variant \
         additions break every consumer at compile time (see GAP-049):\n{}",
        violations.join("\n")
    );
}
