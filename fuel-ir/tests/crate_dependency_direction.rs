// SPDX-License-Identifier: MIT OR Apache-2.0
//! **GAP-265: dependencies must flow downward, and this is what enforces it.**
//!
//! `docs/architecture/02-layers.md` Rule 1 says dependencies flow downward only
//! and claims it is *"enforced via Cargo's dep graph"*. It is not. **Cargo
//! enforces ACYCLICITY, not DIRECTION** -- a crate may depend on anything at all
//! so long as no cycle results, and every layering violation this project has
//! shipped was acyclic.
//!
//! # Why the tier file is a SOURCE and not a transcription of the diagram
//!
//! Measured 2026-09-02: taking Rule 1 literally against 02-layers' drawing makes
//! **every backend a violator** -- 6 of 6 backends depend on `fuel-ir`, which the
//! drawing places above them. That is the backend contract working as designed.
//! The drawing's vertical axis is CONSUMPTION / ABSTRACTION, not dependency; its
//! own prose (*"higher layers consume the Foundation surface but don't shape
//! it"*) is about authority. **Rule 1 constrains a direction on the wrong axis,
//! which is why it is unenforceable as written rather than merely unenforced.**
//!
//! `crate_dependency_tiers.txt` therefore states the dependency order, which
//! 02-layers does not state anywhere, and inverts the drawing across the
//! Foundation/Backends boundary deliberately.
//!
//! # What a green here does and does not mean
//!
//! The gate prints its own coverage every run -- how many edges it judged, and
//! how many crates carry no tier. **A crate with no tier makes every edge it
//! touches unjudgeable, and a silent 40%-coverage green is the exact failure
//! this project keeps recording.** Read the printed coverage, not just the pass.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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
            panic!("no Cargo.toml containing [workspace] above CARGO_MANIFEST_DIR");
        }
    }
}

/// Workspace member directory names, read from the root manifest's `members`.
fn workspace_members(root: &Path) -> BTreeSet<String> {
    let txt = std::fs::read_to_string(root.join("Cargo.toml")).expect("root manifest");
    let mut out = BTreeSet::new();
    let mut in_members = false;
    for line in txt.lines() {
        let t = line.trim();
        if t.starts_with("members") && t.contains('[') {
            in_members = true;
            continue;
        }
        if in_members {
            if t.starts_with(']') {
                in_members = false;
                continue;
            }
            let cleaned: String = t.chars().filter(|c| *c != '"' && *c != ',').collect();
            let name = cleaned.trim().split('/').next().unwrap_or("").to_string();
            if !name.is_empty()
                && !name.starts_with('#')
                && root.join(&name).join("Cargo.toml").exists()
            {
                out.insert(name);
            }
        }
    }
    out
}

/// `(from, to)` for every NORMAL intra-workspace dependency edge.
///
/// Leading whitespace is allowed on the declaration. A column-0 anchor misses
/// indented declarations and returns a clean, confident, low answer.
///
/// MEASURED, with the population named because the bare ratio is wider than
/// the evidence: across every workspace `Cargo.toml` at `ca398d5c`, counting
/// DECLARATION LINES FOR `fuel*` CRATES ONLY (`^[[:space:]]+fuel[a-z-]*=` vs
/// `^fuel[a-z-]*=`), 68 were indented against 103 at column 0. That is 40% of
/// THOSE lines -- not of all dependency declarations, which nobody measured.
/// It cost the lane that measured it two published claims before they swept
/// the anchor bug backwards.
/// dev- and build-dependencies are excluded: they do not constrain layering.
fn dependency_edges(root: &Path, members: &BTreeSet<String>) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for m in members {
        let Ok(txt) = std::fs::read_to_string(root.join(m).join("Cargo.toml")) else {
            continue;
        };
        let mut in_normal_deps = false;
        for line in txt.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                in_normal_deps = t == "[dependencies]"
                    || (t.starts_with("[target.") && t.ends_with(".dependencies]"));
                continue;
            }
            if !in_normal_deps || t.starts_with('#') || t.is_empty() {
                continue;
            }
            let name = t
                .split(['=', '.'])
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('"')
                .to_string();
            if members.contains(&name) && name != *m {
                out.insert((m.clone(), name));
            }
        }
    }
    out
}

/// crate -> (tier, layer name)
fn load_tiers(root: &Path) -> BTreeMap<String, (u32, String)> {
    let p = root.join("fuel-ir/tests/crate_dependency_tiers.txt");
    let txt =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let mut out = BTreeMap::new();
    for line in txt.lines() {
        let t = line.trim_end();
        if t.starts_with('#') || t.trim().is_empty() {
            continue;
        }
        let mut f = t.split('\t');
        let (Some(name), Some(tier)) = (f.next(), f.next()) else {
            continue;
        };
        let layer = f.next().unwrap_or("").trim().to_string();
        if let Ok(n) = tier.trim().parse::<u32>() {
            out.insert(name.trim().to_string(), (n, layer));
        }
    }
    out
}

/// (from, to) -> reason. A blank or missing reason is preserved as empty so the
/// gate can REJECT it rather than silently accepting amnesty with no record.
fn load_allowlist(root: &Path) -> BTreeMap<(String, String), String> {
    let p = root.join("fuel-ir/tests/crate_dependency_tier_allowlist.txt");
    let txt =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let mut out = BTreeMap::new();
    for line in txt.lines() {
        let t = line.trim_end();
        if t.starts_with('#') || t.trim().is_empty() {
            continue;
        }
        let mut f = t.split('\t');
        let (Some(a), Some(b)) = (f.next(), f.next()) else {
            continue;
        };
        out.insert(
            (a.trim().to_string(), b.trim().to_string()),
            f.next().unwrap_or("").trim().to_string(),
        );
    }
    out
}

#[test]
fn dependencies_flow_downward_only() {
    let root = workspace_root();
    let members = workspace_members(&root);
    let edges = dependency_edges(&root, &members);
    let tiers = load_tiers(&root);
    let allow = load_allowlist(&root);

    // FOUNDATION CHECKS. Without these the gate passes vacuously when its own
    // data goes missing -- which is the failure mode it exists to prevent.
    assert!(
        members.len() >= 30,
        "found only {} workspace members -- the manifest scan is broken, and a \
         broken scan and a clean tree are byte-identical here",
        members.len()
    );
    assert!(
        edges.len() >= 50,
        "found only {} intra-workspace dependency edges -- the manifest parse is \
         broken. A column-0 anchor misses indented declarations (68 of 171 \
         `fuel*` declaration LINES in this repo -- that population, not all \
         dependencies) and returns a confident low number",
        edges.len()
    );
    assert!(
        tiers.len() >= 25,
        "crate_dependency_tiers.txt assigns only {} crates. If this file were \
         emptied the gate would judge NOTHING and still pass, so the count is \
         asserted: an unassigned crate makes every edge it touches unjudgeable",
        tiers.len()
    );

    let judged: Vec<&(String, String)> = edges
        .iter()
        .filter(|(a, b)| tiers.contains_key(a) && tiers.contains_key(b))
        .collect();
    let unassigned: Vec<&String> = members.iter().filter(|m| !tiers.contains_key(*m)).collect();

    // COVERAGE, printed every run. A green over 40% of the edges is not the same
    // claim as a green over all of them, and only this line says which it is.
    println!(
        "[dep-direction] judged {} of {} edges; {} of {} crates assigned; {} unassigned: {:?}",
        judged.len(),
        edges.len(),
        tiers.len(),
        members.len(),
        unassigned.len(),
        unassigned
    );

    let violations: Vec<String> = judged
        .iter()
        .filter(|(a, b)| tiers[b].0 > tiers[a].0)
        .filter(|(a, b)| !allow.contains_key(&(a.clone(), b.clone())))
        .map(|(a, b)| {
            format!(
                "{a} (tier {}, {}) -> {b} (tier {}, {})",
                tiers[a].0, tiers[a].1, tiers[b].0, tiers[b].1
            )
        })
        .collect();

    let reasonless: Vec<String> = allow
        .iter()
        .filter(|(_, r)| r.is_empty())
        .map(|((a, b), _)| format!("{a} -> {b}"))
        .collect();

    let stale: Vec<String> = allow
        .keys()
        .filter(|(a, b)| !edges.contains(&(a.clone(), b.clone())))
        .map(|(a, b)| format!("{a} -> {b}"))
        .collect();

    assert!(
        reasonless.is_empty(),
        "allowlist entries with a BLANK reason: {reasonless:?}\n\
         A bare allowlist is amnesty with no record of which disposition it is, \
         and it takes the guard's signal with it."
    );
    assert!(
        stale.is_empty(),
        "STALE allowlist entries -- the edge no longer exists: {stale:?}\n\
         This list may only SHRINK. Delete the entry; the repoint landed."
    );
    assert!(
        violations.is_empty(),
        "dependency edges pointing UPWARD through the tier order:\n  {}\n\n\
         A crate may depend only on its own tier or below \
         (fuel-ir/tests/crate_dependency_tiers.txt). Cargo will NOT catch this: \
         it enforces acyclicity, not direction. Either the edge is wrong or the \
         tier assignment is -- and if the edge is deliberate, add it to \
         crate_dependency_tier_allowlist.txt WITH A REASON.",
        violations.join("\n  ")
    );
}
