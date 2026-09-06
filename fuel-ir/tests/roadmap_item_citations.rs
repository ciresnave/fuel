//! ROADMAP frontier item NUMBERS are load-bearing citations. This refuses a
//! renumber instead of reminding someone not to do one.
//!
//! # Why this exists
//!
//! `ROADMAP.md`'s "Active frontier" is a numbered list, and 31 files cite its
//! entries by position — `ROADMAP item 8 (II)` alone appears 76 times, mostly
//! in `fuel-transformers/src/models/*.rs`. Measured at `785b6ecc`:
//!
//! ```text
//! git grep -lIE 'ROADMAP item [0-9]+' -- '*.md' '*.rs'
//!   31 files, 90 citations;  by item: 8 x76, 9 x7, 12 x3, 7 x2, 10 x2
//! ```
//!
//! A 2026-09-05 currency audit marked five items SHIPPED / RECORD / STRUCK and
//! told a reader to MOVE them. Executing that removes entries from a numbered
//! list, renumbering every later item: 2,3,5,6,7,8,11 become 1..7, and **item 8
//! becomes item 6**, repointing seventy-six citations at the wrong item. Three
//! of the five marked items are themselves cited (9, 10, 12 — twelve citations
//! between them), so moving those breaks the references pointing *at* them.
//!
//! # ⚠️ Why prose was not enough
//!
//! The citations live in **code comments**. Nothing checks them, and nothing
//! would fail: a wrong item number does not dangle, it points at a REAL item
//! that is not the one meant, and reads as correct. A wrong path fails loudly;
//! a wrong number misleads quietly.
//!
//! `ROADMAP.md` already carries a DO-NOT-RENUMBER note. That note is a
//! *reminder*, and a reminder is read by whoever is already looking for it.
//! This file *refuses*.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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

/// This file's own path, relative to the workspace root, in `/` form.
///
/// Excluded from the scan because the fixtures below contain deliberately
/// fabricated citations. Which file is skipped is asserted by a test, so the
/// exclusion cannot silently widen.
const SELF_PATH: &str = "fuel-ir/tests/roadmap_item_citations.rs";

/// The numbers of the entries in `ROADMAP.md`'s Active frontier, in file order.
///
/// Bounded by the `### Active frontier` heading and the next `### ` heading:
/// `ROADMAP.md` contains other numbered lists (measured: nine more `N. **`
/// openers outside this range), and a document-wide match would sweep them in.
fn frontier_items(roadmap: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in roadmap.lines() {
        if line.starts_with("### Active frontier") {
            inside = true;
            continue;
        }
        if inside && line.starts_with("### ") {
            break;
        }
        if !inside {
            continue;
        }
        // `N. **` — an item opener, not a numbered line inside a body.
        let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && line[digits.len()..].starts_with(". **") {
            if let Ok(n) = digits.parse::<u32>() {
                out.push(n);
            }
        }
    }
    out
}

/// Every `ROADMAP item <N>` / `ROADMAP frontier item <N>` citation in `text`.
///
/// Both spellings are matched deliberately: the count is not stable under
/// rephrasing, and a gate that knew only one form would pass a corpus that had
/// drifted to the other.
fn cited_in(text: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for prefix in ["ROADMAP item ", "ROADMAP frontier item "] {
        let mut rest = text;
        while let Some(i) = rest.find(prefix) {
            rest = &rest[i + prefix.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<u32>() {
                out.insert(n);
            }
        }
    }
    out
}

/// (item number -> the files citing it), over every `.rs` and `.md` in the tree.
fn citations(root: &Path) -> BTreeMap<u32, Vec<String>> {
    let mut out: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name == "target" || name == ".git" || name == "node_modules" {
                    continue;
                }
                stack.push(p);
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "rs" && ext != "md" {
                continue;
            }
            let rel = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            if rel == SELF_PATH {
                continue;
            }
            if let Ok(txt) = std::fs::read_to_string(&p) {
                for n in cited_in(&txt) {
                    out.entry(n).or_default().push(rel.clone());
                }
            }
        }
    }
    out
}

#[test]
fn every_cited_roadmap_item_exists() {
    let root = workspace_root();
    let roadmap = std::fs::read_to_string(root.join("ROADMAP.md")).expect("ROADMAP.md");
    let items: BTreeSet<u32> = frontier_items(&roadmap).into_iter().collect();
    assert!(
        items.len() > 5,
        "only {} frontier items parsed — the ROADMAP scan is broken, and a broken \
         scan makes this whole file vacuous",
        items.len()
    );

    let cited = citations(&root);
    assert!(
        !cited.is_empty(),
        "zero citations found across the tree — the file walk is broken. This gate \
         is only meaningful while citations exist to be broken."
    );

    let dangling: Vec<String> = cited
        .iter()
        .filter(|(n, _)| !items.contains(n))
        .map(|(n, files)| {
            format!(
                "item {n} cited by {} file(s): {}",
                files.len(),
                files.join(", ")
            )
        })
        .collect();
    assert!(
        dangling.is_empty(),
        "a citation names a frontier item that does not exist. Either the item was \
         removed (do not remove items — see the DO-NOT-RENUMBER note in ROADMAP.md) \
         or the citation is wrong:\n  {}",
        dangling.join("\n  ")
    );
}

#[test]
fn the_frontier_numbering_is_contiguous_from_one() {
    let root = workspace_root();
    let roadmap = std::fs::read_to_string(root.join("ROADMAP.md")).expect("ROADMAP.md");
    let items = frontier_items(&roadmap);
    let expected: Vec<u32> = (1..=items.len() as u32).collect();
    assert_eq!(
        items, expected,
        "the Active frontier must be numbered 1..N with no gaps and no reordering.\n\
         THIS IS THE ARM THAT CATCHES A REMOVAL NOBODY CITED. `every_cited_roadmap_item_exists` \
         only fires when a citation is left dangling, so deleting an uncited item would slip \
         past it while silently renumbering every later item — and the citations that then \
         resolve to the WRONG item still resolve, so nothing else would ever complain."
    );
}

#[test]
fn the_scan_excludes_exactly_one_file() {
    let root = workspace_root();
    assert!(
        root.join(SELF_PATH).exists(),
        "the excluded path {SELF_PATH} does not exist — the exclusion has drifted and \
         this file is now being scanned, or is skipping something else"
    );
    // The exclusion must be exactly this file, so a widened skip is visible.
    assert_eq!(SELF_PATH, "fuel-ir/tests/roadmap_item_citations.rs");
}

// FOUNDATION CHECK, THREE ARMS. Each arm asserts a DIFFERENT failure can be
// seen; a gate with one arm cannot distinguish "correct" from "asleep".
#[test]
fn the_detector_discriminates() {
    // (1) a dangling citation must be visible
    let items: BTreeSet<u32> = [1, 2, 3].into_iter().collect();
    let cited = cited_in("see ROADMAP item 99 for the rest");
    assert!(
        cited.contains(&99) && !items.contains(&99),
        "a citation to a nonexistent item was not detected"
    );

    // (2) the alternate spelling must be seen too — the corpus uses both
    assert!(
        cited_in("per ROADMAP frontier item 4, the seam is open").contains(&4),
        "the `ROADMAP frontier item N` spelling is not matched, so a corpus that \
         drifted to it would read as having zero citations"
    );

    // (3) a gap in the numbering must be visible, and a clean list must not fire
    let holed = "### Active frontier\n1. **a**\n2. **b**\n4. **d**\n### next\n";
    assert_eq!(
        frontier_items(holed),
        vec![1, 2, 4],
        "the gap was not parsed"
    );
    let clean = "### Active frontier\n1. **a**\n2. **b**\n3. **c**\n### next\n";
    assert_eq!(
        frontier_items(clean),
        vec![1, 2, 3],
        "a clean list mis-parsed"
    );

    // (4) the section bound must hold: a numbered list AFTER the frontier is
    // not a frontier item. ROADMAP.md has nine such openers elsewhere.
    let bounded = "### Active frontier\n1. **a**\n### Deferred\n2. **not an item**\n";
    assert_eq!(
        frontier_items(bounded),
        vec![1],
        "the scan ran past the section boundary and swept in an unrelated list"
    );
}
