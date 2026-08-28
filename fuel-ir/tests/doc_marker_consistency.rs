// SPDX-License-Identifier: MIT OR Apache-2.0
//! CI guard: once an artifact name is marked absent in an architecture doc, EVERY
//! claim site for that name in that doc must carry the marker.
//!
//! # What this catches, and — read this first — what it CANNOT
//! This is a **consistency** guard, not a **discovery** one. It fires only for
//! names that already carry the absence marker (†) somewhere in the same
//! document. A name that is absent from the code and marked NOWHERE is
//! structurally invisible to it.
//!
//! **Measured against the real history rather than asserted.** The 2026-08-28
//! audit found six unmarked claim sites in `02-layers.md`. Run against the
//! pre-fix commit (`d56cfd0f`), this rule finds **two of the six**:
//! `fuel-autograd:80` and `fuel-loaders:83` — both cases of a name marked earlier
//! on the same line and unmarked later. The other four (`fuel-interchange-weights`
//! and `fuel-interchange-graph`, in the diagram and in their bullet definitions)
//! carried no marker anywhere, so no consistency rule could see them. Finding
//! those needs the absence check over prose, which is deliberately not automated:
//! it runs at roughly 10% precision and its false positives include satisfied
//! non-goals, where an absence is the constitution being obeyed and "fixing" it
//! breaks a deliberate decision.
//!
//! So the honest claim is narrow and worth stating plainly: **this prevents
//! regression of an absence someone has already acknowledged. It does not
//! discover new ones.**
//!
//! # Why that narrow class earns a guard anyway
//! It is the class that would not stop recurring. `02-layers.md` was hand-fixed
//! across v0.7, v0.8, v0.9, v0.10 and v0.11 — five passes, each correct about
//! what it touched, none able to establish it had touched everything. v0.10
//! missed a second occurrence of a name on a line containing it twice; v0.11
//! fixed sites the v0.9 pass had walked past. A reader cannot establish
//! completeness over a claim that appears in a diagram, in prose, in a bullet
//! list, and twice in one sentence. A scan can.
//!
//! # The exclusion rule is where this guard's judgment lives
//! Two line kinds name an absent artifact *because they are recording that it is
//! absent*, and marking them would put a dagger on the sentence that explains
//! what the dagger means:
//!   - the **Status/changelog line** (`**Status**: …`) — a version history, not a
//!     claim;
//!   - **blockquoted as-built notes** (`> …`) — disclosures.
//!
//! Without these exclusions the rule fires 32 times on `02-layers` for ~6 real
//! sites (~19% precision). With them it is exact. At head, 10 marked names span
//! 43 occurrences: 22 marked, 21 excluded as disclosure, 0 violations.
//!
//! **The excluded count is larger than the marked count.** That is not slack in
//! the rule — it is what a well-documented absence looks like: one marked claim
//! site, several sentences explaining it.
//!
//! # Marker placement
//! The marker goes immediately after the closing backtick when the name is in a
//! code span (`` `name`† ``), immediately after the name otherwise — typographic
//! ruling, 2026-08-28. **The scanner must therefore accept an optional
//! intervening backtick.** Without that, four `fuel-format-*` leaves read as
//! unmarked and a green turns into a false report of finished work.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// The absence marker used in the architecture docs.
const MARKER: char = '\u{2020}'; // DAGGER

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

/// One occurrence of a candidate artifact name on one line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Occurrence {
    line: usize,
    name: String,
    marked: bool,
    /// True when the line is a disclosure (Status/changelog, or a blockquoted
    /// as-built note) rather than a claim.
    disclosure: bool,
}

/// True when the line records an absence rather than claiming existence.
fn is_disclosure_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('>') || t.starts_with("**Status**:")
}

/// Every artifact-name occurrence in one document.
///
/// A name is either a backticked identifier (`` `FusedOpEntry` ``) or a bare
/// `fuel-*` crate-style token, which is how the docs write crate names inside
/// ASCII diagrams where backticks would break the box drawing.
fn scan_occurrences(text: &str) -> Vec<Occurrence> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let disclosure = is_disclosure_line(line);
        for (start, end, name) in find_names(line) {
            let _ = start;
            out.push(Occurrence {
                line: idx + 1,
                name,
                marked: is_marked_at(line, end),
                disclosure,
            });
        }
    }
    out
}

/// Marked iff the marker follows the name, allowing one intervening backtick
/// because the docs write `` `name`† `` — the marker sits OUTSIDE the code span.
fn is_marked_at(line: &str, end: usize) -> bool {
    let tail = &line[end..];
    let mut chars = tail.chars();
    match chars.next() {
        Some(MARKER) => true,
        Some('`') => chars.next() == Some(MARKER),
        _ => false,
    }
}

/// Backticked identifiers and bare `fuel-*` tokens, as (start, end, name).
///
/// Indexes via `char_indices` throughout: the docs contain multi-byte characters
/// (the dagger itself, em dashes, box-drawing glyphs), so byte-stepping panics on
/// a non-boundary slice. Every index handed out here is a char boundary.
fn find_names(line: &str) -> Vec<(usize, usize, String)> {
    let idx: Vec<(usize, char)> = line.char_indices().collect();
    let mut out = Vec::new();
    let mut k = 0usize;
    while k < idx.len() {
        let (bi, c) = idx[k];
        if c == '`' {
            // a backticked span: take it only if its whole content is one identifier
            if let Some(rel) = line[bi + 1..].find('`') {
                let close = bi + 1 + rel;
                let inner = &line[bi + 1..close];
                if !inner.is_empty()
                    && inner
                        .chars()
                        .all(|ch| ch.is_alphanumeric() || ch == '_' || ch == '-')
                    && inner.chars().next().is_some_and(|ch| ch.is_alphabetic())
                {
                    out.push((bi + 1, close, inner.to_string()));
                    while k < idx.len() && idx[k].0 <= close {
                        k += 1;
                    }
                    continue;
                }
                // NOT a single identifier — do NOT skip the span. A multi-word
                // span still carries crate tokens that matter: `cargo add
                // fuel-model-llama`† marks `fuel-model-llama`, and skipping the
                // span would hide both the name and its marker. Step past the
                // opening backtick and let normal scanning find the token; the
                // marker check already tolerates the intervening backtick.
                k += 1;
                continue;
            }
            k += 1;
            continue;
        }
        // a bare `fuel-*` token (how crate names appear inside ASCII diagrams,
        // where backticks would break the box drawing), only at a word boundary
        if line[bi..].starts_with("fuel-")
            && (k == 0 || {
                let p = idx[k - 1].1;
                !(p.is_alphanumeric() || p == '_' || p == '-')
            })
        {
            let mut j = k + 5; // "fuel-" is five ASCII chars
            while j < idx.len() {
                let ch = idx[j].1;
                if ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' {
                    j += 1;
                } else {
                    break;
                }
            }
            while j > k && idx[j - 1].1 == '-' {
                j -= 1;
            }
            if j > k + 5 {
                let end = if j < idx.len() { idx[j].0 } else { line.len() };
                out.push((bi, end, line[bi..end].to_string()));
                k = j;
                continue;
            }
        }
        k += 1;
    }
    out
}

/// The gate, factored off the filesystem. Returns (violations, stale, reasonless).
fn evaluate(
    doc: &str,
    occurrences: &[Occurrence],
    allowlist: &HashMap<(String, String, usize), String>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let marked: HashSet<&str> = occurrences
        .iter()
        .filter(|o| o.marked)
        .map(|o| o.name.as_str())
        .collect();
    let mut violations = Vec::new();
    let mut seen = HashSet::new();
    for o in occurrences {
        if o.marked || o.disclosure || !marked.contains(o.name.as_str()) {
            continue;
        }
        let key = (doc.to_string(), o.name.clone(), o.line);
        seen.insert(key.clone());
        if !allowlist.contains_key(&key) {
            violations.push(format!(
                "{doc}:{} `{}` is unmarked here but marked elsewhere in the same \
                 document — every claim site for an acknowledged absence must \
                 carry the marker",
                o.line, o.name
            ));
        }
    }
    let mut stale: Vec<String> = allowlist
        .keys()
        .filter(|k| k.0 == doc && !seen.contains(*k))
        .map(|(d, n, l)| format!("{d}\t{n}\t{l}"))
        .collect();
    stale.sort();
    let mut reasonless: Vec<String> = allowlist
        .iter()
        .filter(|((d, _, _), r)| d == doc && r.is_empty())
        .map(|((d, n, l), _)| format!("{d}\t{n}\t{l}"))
        .collect();
    reasonless.sort();
    (violations, stale, reasonless)
}

fn allowlist_path(root: &Path) -> PathBuf {
    root.join("fuel-ir")
        .join("tests")
        .join("doc_marker_consistency_allowlist.txt")
}

/// (doc, name, line) -> reason. Line is part of the key here (unlike the trait
/// guard) because the same name legitimately recurs many times in one document.
fn load_allowlist(root: &Path) -> HashMap<(String, String, usize), String> {
    let mut map = HashMap::new();
    let Ok(txt) = std::fs::read_to_string(allowlist_path(root)) else {
        return map;
    };
    for line in txt.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(4, '\t');
        let (Some(doc), Some(name), Some(lineno)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let Ok(lineno) = lineno.trim().parse::<usize>() else {
            continue;
        };
        let reason = parts.next().unwrap_or("").trim().to_string();
        map.insert((doc.to_string(), name.to_string(), lineno), reason);
    }
    map
}

fn arch_docs(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("docs").join("architecture");
    let mut out: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(e) => e
            .flatten()
            .map(|x| x.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("md"))
            .collect(),
        Err(_) => Vec::new(),
    };
    out.sort();
    out
}

#[test]
fn marked_absence_is_marked_at_every_claim_site() {
    let root = workspace_root();
    let docs = arch_docs(&root);
    assert!(
        !docs.is_empty(),
        "no docs/architecture/*.md found — a clean result and a broken scan are \
         byte-identical here"
    );
    let allowlist = load_allowlist(&root);
    let (mut violations, mut stale, mut reasonless) = (Vec::new(), Vec::new(), Vec::new());
    let mut total_marked = 0usize;
    for path in &docs {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Ok(txt) = std::fs::read_to_string(path) else {
            continue;
        };
        let occ = scan_occurrences(&txt);
        total_marked += occ.iter().filter(|o| o.marked).count();
        let (v, s, r) = evaluate(&name, &occ, &allowlist);
        violations.extend(v);
        stale.extend(s);
        reasonless.extend(r);
    }
    // POSITIVE CONTROL. If marker parsing breaks, nothing is ever "marked", the
    // rule has no names to check, and this gate passes vacuously while reporting
    // success. The corpus is known to carry markers, so zero means broken.
    assert!(
        total_marked > 0,
        "scanned {} architecture docs and found NO marked names — the marker \
         parser is broken, not the corpus (the marker sits outside the code \
         span: `name`†)",
        docs.len()
    );
    let mut msg = String::new();
    if !violations.is_empty() {
        msg.push_str(&format!("\n{} unmarked claim site(s):\n", violations.len()));
        for v in &violations {
            msg.push_str(&format!("  - {v}\n"));
        }
        msg.push_str(
            "\nMark the site (the marker goes after the closing backtick when the \
             name is in a code span), or — if the line is recording the absence \
             rather than claiming existence — make it a disclosure. If neither \
             fits, add an entry to \
             fuel-ir/tests/doc_marker_consistency_allowlist.txt as\n  \
             <doc>\\t<name>\\t<line>\\t<why this site is legitimately unmarked>\n",
        );
    }
    if !stale.is_empty() {
        msg.push_str(&format!(
            "\n{} STALE allowlist entr(ies) — no such unmarked site exists now, \
             so the line must be deleted (the list may only shrink):\n",
            stale.len()
        ));
        for s in &stale {
            msg.push_str(&format!("  - {s}\n"));
        }
    }
    if !reasonless.is_empty() {
        msg.push_str(&format!(
            "\n{} allowlist entr(ies) carry no reason:\n",
            reasonless.len()
        ));
        for r in &reasonless {
            msg.push_str(&format!("  - {r}\n"));
        }
    }
    assert!(msg.is_empty(), "{msg}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn occ(text: &str) -> Vec<Occurrence> {
        scan_occurrences(text)
    }

    /// PERMANENT SABOTAGE SIBLING, arm 1: an unmarked occurrence of a name that
    /// is marked elsewhere in the same doc must be flagged.
    #[test]
    fn flags_an_unmarked_occurrence_of_a_marked_name() {
        let doc = "The `fuel-tensor`\u{2020} crate is planned.\nLater prose mentions fuel-tensor again.\n";
        let o = occ(doc);
        assert!(
            o.iter().any(|x| x.name == "fuel-tensor" && x.marked),
            "marker after the closing backtick must be recognised: {o:?}"
        );
        let (v, _, _) = evaluate("d.md", &o, &HashMap::new());
        assert_eq!(v.len(), 1, "the unmarked repeat must be flagged: {v:?}");
        assert!(v[0].contains(":2"), "and it must name the line: {v:?}");
    }

    /// PERMANENT SABOTAGE SIBLING, arm 2 — THE LOAD-BEARING ONE. A guard that
    /// cannot tell a claim from a disclosure is the failure mode here, not a
    /// missed occurrence: it would demand a dagger on the sentence that explains
    /// what the dagger means.
    #[test]
    fn does_not_flag_a_disclosure_line() {
        let doc = "The `fuel-tensor`\u{2020} crate is planned.\n\
                   > **AS-BUILT NOTE.** Measured at head: fuel-tensor is ABSENT.\n\
                   **Status**: v0.9 — marks fuel-tensor in the diagram.\n";
        let o = occ(doc);
        assert!(
            o.iter().filter(|x| x.disclosure).count() >= 2,
            "both the blockquote and the Status line must be disclosures: {o:?}"
        );
        let (v, _, _) = evaluate("d.md", &o, &HashMap::new());
        assert!(
            v.is_empty(),
            "a disclosure names the artifact BECAUSE it is recording the \
             absence; it must never be flagged: {v:?}"
        );
    }

    #[test]
    fn marker_directly_after_a_bare_name_also_counts() {
        let doc =
            "In a diagram: fuel-loaders\u{2020} (transport adapters)\nand fuel-loaders here.\n";
        let o = occ(doc);
        assert!(o.iter().any(|x| x.name == "fuel-loaders" && x.marked));
        let (v, _, _) = evaluate("d.md", &o, &HashMap::new());
        assert_eq!(v.len(), 1, "{v:?}");
    }

    #[test]
    fn an_unmarked_name_marked_nowhere_is_not_flagged() {
        // The documented scope limit: this is a CONSISTENCY guard. Discovering a
        // never-acknowledged absence needs the prose absence check, which is not
        // automated because its false positives include satisfied non-goals.
        let doc = "Prose naming fuel-nevermarked twice: fuel-nevermarked.\n";
        let (v, _, _) = evaluate("d.md", &occ(doc), &HashMap::new());
        assert!(v.is_empty(), "out of scope by design: {v:?}");
    }

    #[test]
    fn allowlisted_site_passes_and_stale_entry_is_reported() {
        let doc = "`fuel-x`\u{2020} here.\nfuel-x again.\n";
        let a: HashMap<(String, String, usize), String> = [(
            ("d.md".to_string(), "fuel-x".to_string(), 2usize),
            "deliberate: quoting the pre-fix text verbatim".to_string(),
        )]
        .into_iter()
        .collect();
        let (v, stale, _) = evaluate("d.md", &occ(doc), &a);
        assert!(v.is_empty(), "allowlisted: {v:?}");
        assert!(
            stale.is_empty(),
            "not stale while the site exists: {stale:?}"
        );
        let (_, stale2, _) = evaluate("d.md", &occ("`fuel-x`\u{2020} only.\n"), &a);
        assert_eq!(stale2.len(), 1, "the repeat is gone, so the entry is stale");
    }

    #[test]
    fn reasonless_allowlist_entry_is_reported() {
        let doc = "`fuel-x`\u{2020} here.\nfuel-x again.\n";
        let a: HashMap<(String, String, usize), String> = [(
            ("d.md".to_string(), "fuel-x".to_string(), 2usize),
            String::new(),
        )]
        .into_iter()
        .collect();
        let (_, _, reasonless) = evaluate("d.md", &occ(doc), &a);
        assert_eq!(
            reasonless.len(),
            1,
            "a bare entry is amnesty with no record"
        );
    }

    #[test]
    fn backticked_prose_span_is_not_mistaken_for_a_name() {
        // Only a span whose entire content is one identifier counts, so ordinary
        // inline code does not manufacture candidate names.
        let doc = "`cargo add fuel-model-llama`\u{2020} and `let x = 1;` here.\n";
        let found = occ(doc);
        let names: Vec<&str> = found.iter().map(|o| o.name.as_str()).collect();
        assert!(
            !names.contains(&"cargo add fuel-model-llama"),
            "a multi-word span is not an identifier: {names:?}"
        );
        assert!(
            names.contains(&"fuel-model-llama"),
            "but the bare crate token inside it is still seen: {names:?}"
        );
    }
}
