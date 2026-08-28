// SPDX-License-Identifier: MIT OR Apache-2.0
//! CI guard: a `pub trait` declared in a rust fence in `docs/architecture/` must
//! exist in the code, or carry an allowlist entry stating its disposition.
//!
//! # Why this class, and why traits first
//! The docs-vs-code audit (2026-08-28) found `05-backend-contract` declaring
//! three traits that do not exist — `BackendIdentity`, `BackendPressureSignals`,
//! `BackendDiagnostics` — one of them under "Tier 1 — Mandatory (every backend
//! must implement)", above the sentence "A backend that cannot satisfy Tier 1
//! cannot participate in dispatch". That is the document external backend
//! authors are pointed at. An implementer searches for `trait BackendIdentity`,
//! finds nothing, and **cannot distinguish "I missed a mandatory requirement"
//! from "the doc is stale"**. The same injury is already recorded in
//! `02-layers`: a consumer read the layer diagram, wrote `use
//! fuel_nn::VarBuilder`, and hit a wall (the Lightbulb port, 2026-07-29).
//!
//! A section heading plus a rust fence is the strongest existence claim a doc
//! can make short of code — **it looks like an interface definition, so nobody
//! re-derives it** — and it is the format most likely to rot silently, because
//! nothing compiles it.
//!
//! # Why a guard rather than another careful read
//! The same defect was hand-fixed four times in one file in one night. v0.7
//! struck a name; v0.8 appended a note and the diagram went on making the claim;
//! v0.9 marked the diagram and the prose went on making it; v0.10 marked the
//! prose and **missed a second occurrence of `fuel-format-onnx` on a line that
//! contained it twice** — caught by re-counting, not by reading. A hand pass
//! cannot establish completeness over a claim that appears in a diagram, in
//! prose, in a bullet list, and twice in one sentence. A grep does not have that
//! failure mode.
//!
//! # Scope — deliberately narrow, read before widening
//! Only `pub trait <Name>` inside a rust fence in `docs/architecture/*.md`. NOT
//! prose mentions: a backticked type name in a sentence is a judgment call (it
//! may be historical, pinned to a commit, or a satisfied non-goal — a doc
//! section whose job is to say what Fuel does NOT have inverts the polarity of
//! "named but missing" entirely), whereas a declared trait body is a **clean
//! binary**: the name either exists or it does not. The audit measured the wider
//! prose population at ~27 surviving absences across 9 docs and 7 distinct
//! false-positive classes; that population needs human disposition and is
//! deliberately NOT in scope here.
//!
//! # The allowlist REQUIRES a reason, and that is enforced, not conventional
//! Entry format, TAB-separated, sorted:
//!   `<doc file name>` TAB `<TraitName>` TAB `<disposition + why>`
//! The **key** is the first two fields only, so editing a reason never breaks an
//! entry. **An entry whose reason field is missing or blank FAILS the gate.**
//! The `gap_hedges` guard records the lesson this enforces: an allowlist of bare
//! entries degrades into noise and takes the guard's signal with it. Here the
//! reason is load-bearing — it is the only place the disposition (unbuilt work /
//! doc drift / renamed / deliberately aspirational) is written down.
//!
//! # "May only shrink" teeth
//! An allowlist entry whose (doc, trait) pair is no longer declared in the docs
//! is **STALE** and fails the gate, so the allowlist cannot rot into permanent
//! amnesty. Its size is a real, monotonically-shrinking metric of doc debt.
//!
//! # The sabotage sibling is permanent, not scaffolding
//! `guard_flags_a_synthetic_absent_trait` feeds the scanner a doc naming a trait
//! that cannot exist. A born-red proves the gate discriminated ONCE, at authoring
//! time; the sibling proves it still discriminates on EVERY run. Without it, a
//! later edit that breaks fence parsing makes this gate pass vacuously and
//! nothing reports it — the suite stays green and the assertion silently stops
//! being about anything.

use std::collections::{HashMap, HashSet};
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

/// Every `pub trait <Name>` declared inside a rust fence, as (line, name).
///
/// Fence tracking is deliberate: a `pub trait` in prose or in a non-rust fence
/// is not an interface claim. Only a rust-tagged fence opens one.
fn scan_doc_traits(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut in_rust_fence = false;
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_rust_fence {
                in_rust_fence = false;
            } else {
                let lang = trimmed.trim_start_matches('`');
                in_rust_fence = lang.starts_with("rust");
            }
            continue;
        }
        if !in_rust_fence {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("pub trait ") {
            let name = leading_ident(rest);
            if !name.is_empty() {
                out.push((idx + 1, name));
            }
        }
    }
    out
}

/// The identifier at the start of `s` (stops at the first non-ident char).
fn leading_ident(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Pull `trait <Name>` out of one source text. Split out so a unit test can
/// exercise it without touching the filesystem.
///
/// Intentionally matches bare `trait` as well as `pub trait`: the question is
/// "does this name exist as a trait", not "is it public". A doc-declared trait
/// satisfied by a private one is doc drift, not a missing artifact, and the
/// allowlist reason is where that distinction gets written down.
fn collect_traits_from_source(txt: &str, out: &mut HashSet<String>) {
    for line in txt.lines() {
        let t = line.trim_start();
        if t.starts_with("//") {
            continue;
        }
        // `unsafe trait` and `pub(crate) trait` are declarations too.
        let rest = t
            .strip_prefix("pub trait ")
            .or_else(|| t.strip_prefix("unsafe trait "))
            .or_else(|| t.strip_prefix("pub unsafe trait "))
            .or_else(|| {
                t.split_once(") trait ")
                    .filter(|(head, _)| head.starts_with("pub("))
                    .map(|(_, tail)| tail)
            })
            .or_else(|| t.strip_prefix("trait "));
        if let Some(rest) = rest {
            let name = leading_ident(rest);
            if !name.is_empty() {
                out.insert(name);
            }
        }
    }
}

/// Every `trait <Name>` declared anywhere in a `.rs` file under `root`.
fn collect_declared_traits(root: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
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
                if name == "target" || name == ".git" {
                    continue;
                }
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs")
                && let Ok(txt) = std::fs::read_to_string(&p)
            {
                collect_traits_from_source(&txt, &mut out);
            }
        }
    }
    out
}

/// The architecture docs, sorted for stable output.
fn collect_arch_docs(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("docs").join("architecture");
    let mut out: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
            .collect(),
        Err(_) => Vec::new(),
    };
    out.sort();
    out
}

fn allowlist_path(root: &Path) -> PathBuf {
    root.join("fuel-ir")
        .join("tests")
        .join("doc_declared_trait_allowlist.txt")
}

/// (doc, trait) -> reason. A blank or missing reason is preserved as an empty
/// string so the gate can reject it explicitly rather than silently treating the
/// entry as absent.
fn load_allowlist(root: &Path) -> HashMap<(String, String), String> {
    let mut map = HashMap::new();
    let Ok(txt) = std::fs::read_to_string(allowlist_path(root)) else {
        return map;
    };
    for line in txt.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let (Some(doc), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        let reason = parts.next().unwrap_or("").trim().to_string();
        map.insert((doc.to_string(), name.to_string()), reason);
    }
    map
}

/// The gate, factored out of the filesystem so unit tests drive it directly.
/// Returns (violations, stale_entries, reasonless_entries).
fn evaluate(
    declared_in_docs: &[(String, usize, String)],
    exists_in_code: &HashSet<String>,
    allowlist: &HashMap<(String, String), String>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut violations = Vec::new();
    let mut seen = HashSet::new();
    for (doc, line, name) in declared_in_docs {
        if exists_in_code.contains(name) {
            continue;
        }
        let key = (doc.clone(), name.clone());
        seen.insert(key.clone());
        match allowlist.get(&key) {
            Some(_) => {}
            None => violations.push(format!(
                "{doc}:{line}  `pub trait {name}` is declared in the docs but no \
                 `trait {name}` exists in any .rs file"
            )),
        }
    }
    let mut stale: Vec<String> = allowlist
        .keys()
        .filter(|k| !seen.contains(*k))
        .map(|(d, n)| format!("{d}\t{n}"))
        .collect();
    stale.sort();
    let mut reasonless: Vec<String> = allowlist
        .iter()
        .filter(|(_, reason)| reason.is_empty())
        .map(|((d, n), _)| format!("{d}\t{n}"))
        .collect();
    reasonless.sort();
    (violations, stale, reasonless)
}

#[test]
fn doc_declared_trait_must_exist_in_code() {
    let root = workspace_root();
    let docs = collect_arch_docs(&root);
    assert!(
        !docs.is_empty(),
        "found no docs/architecture/*.md — the scan cannot be trusted; a clean \
         result and a broken query are byte-identical here"
    );
    let mut declared = Vec::new();
    for path in &docs {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Ok(txt) = std::fs::read_to_string(path) else {
            continue;
        };
        for (line, t) in scan_doc_traits(&txt) {
            declared.push((name.clone(), line, t));
        }
    }
    // POSITIVE CONTROL, in-corpus: the docs declare traits that DO exist
    // (BackendCapabilityProvider, BackendRuntime, BackendStreams). If the fence
    // parser breaks, `declared` empties and this gate passes vacuously — so
    // require that the scan found something before trusting any verdict.
    assert!(
        !declared.is_empty(),
        "scanned {} architecture docs and found no `pub trait` in any rust \
         fence — the fence parser is broken, not the corpus",
        docs.len()
    );
    let in_code = collect_declared_traits(&root);
    assert!(
        in_code.len() > 50,
        "only {} traits found across the workspace — the source walk is broken",
        in_code.len()
    );
    let allowlist = load_allowlist(&root);
    let (violations, stale, reasonless) = evaluate(&declared, &in_code, &allowlist);

    let mut msg = String::new();
    if !violations.is_empty() {
        msg.push_str(&format!(
            "\n{} doc-declared trait(s) do not exist in code:\n",
            violations.len()
        ));
        for v in &violations {
            msg.push_str(&format!("  - {v}\n"));
        }
        msg.push_str(
            "\nFix the doc, build the trait, or add an allowlist entry to \
             fuel-ir/tests/doc_declared_trait_allowlist.txt as\n  <doc>\\t<Trait>\\t<disposition + why>\n",
        );
    }
    if !stale.is_empty() {
        msg.push_str(&format!(
            "\n{} STALE allowlist entr(ies) — the doc no longer declares them, \
             so the line must be deleted (the allowlist may only shrink):\n",
            stale.len()
        ));
        for s in &stale {
            msg.push_str(&format!("  - {s}\n"));
        }
    }
    if !reasonless.is_empty() {
        msg.push_str(&format!(
            "\n{} allowlist entr(ies) carry no reason. A bare entry is amnesty \
             with no record of which disposition it is:\n",
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

    fn al(entries: &[(&str, &str, &str)]) -> HashMap<(String, String), String> {
        entries
            .iter()
            .map(|(d, n, r)| ((d.to_string(), n.to_string()), r.to_string()))
            .collect()
    }

    /// PERMANENT SABOTAGE SIBLING. The real gate above is allowlisted and so is
    /// green; this proves the machinery still DISCRIMINATES on every run. If it
    /// ever passes, the gate above is vacuous regardless of what it reports.
    #[test]
    fn guard_flags_a_synthetic_absent_trait() {
        let doc = "```rust\npub trait TotallyAbsentTraitXyz {\n    fn f(&self);\n}\n```\n";
        let found = scan_doc_traits(doc);
        assert_eq!(
            found,
            vec![(2, "TotallyAbsentTraitXyz".to_string())],
            "the fence parser must find a trait declared in a rust fence"
        );
        let declared = vec![("d.md".to_string(), 2, "TotallyAbsentTraitXyz".to_string())];
        let (v, _, _) = evaluate(&declared, &HashSet::new(), &HashMap::new());
        assert_eq!(v.len(), 1, "an absent, unallowlisted trait must be flagged");
    }

    #[test]
    fn passes_a_trait_that_exists_in_code() {
        let declared = vec![("d.md".to_string(), 1, "RealTrait".to_string())];
        let code: HashSet<String> = ["RealTrait".to_string()].into_iter().collect();
        let (v, _, _) = evaluate(&declared, &code, &HashMap::new());
        assert!(v.is_empty(), "an existing trait must not be flagged: {v:?}");
    }

    #[test]
    fn passes_an_allowlisted_absent_trait() {
        let declared = vec![("d.md".to_string(), 1, "Gone".to_string())];
        let a = al(&[("d.md", "Gone", "doc drift — filed as ROADMAP item N")]);
        let (v, _, _) = evaluate(&declared, &HashSet::new(), &a);
        assert!(v.is_empty(), "an allowlisted absence must pass: {v:?}");
    }

    #[test]
    fn allowlist_entry_without_a_reason_is_rejected() {
        let declared = vec![("d.md".to_string(), 1, "Gone".to_string())];
        let a = al(&[("d.md", "Gone", "")]);
        let (v, _, reasonless) = evaluate(&declared, &HashSet::new(), &a);
        assert!(v.is_empty(), "it is allowlisted, so not a violation");
        assert_eq!(
            reasonless.len(),
            1,
            "a reasonless entry must be reported — a bare allowlist degrades \
             into noise and takes the guard's signal with it"
        );
    }

    #[test]
    fn stale_allowlist_entry_is_reported() {
        // The doc no longer declares `Gone`, so its amnesty line must go.
        let a = al(&[("d.md", "Gone", "some reason")]);
        let (_, stale, _) = evaluate(&[], &HashSet::new(), &a);
        assert_eq!(stale, vec!["d.md\tGone".to_string()]);
    }

    #[test]
    fn ignores_a_trait_outside_a_rust_fence() {
        let doc = "Prose mentioning pub trait NotAClaim in text.\n\
                   ```text\npub trait AlsoNotAClaim {}\n```\n";
        assert!(
            scan_doc_traits(doc).is_empty(),
            "only a rust fence declares an interface; prose is a judgment call"
        );
    }

    #[test]
    fn source_scan_finds_the_declaration_forms_that_exist() {
        let mut got = HashSet::new();
        collect_traits_from_source(
            "pub trait A {}\ntrait B {}\npub(crate) trait C {}\nunsafe trait D {}\n// trait E {}\n",
            &mut got,
        );
        assert!(got.contains("A") && got.contains("B"), "got {got:?}");
        assert!(got.contains("C"), "pub(crate) trait must count: {got:?}");
        assert!(got.contains("D"), "unsafe trait must count: {got:?}");
        assert!(!got.contains("E"), "a commented-out trait must NOT count");
    }
}
