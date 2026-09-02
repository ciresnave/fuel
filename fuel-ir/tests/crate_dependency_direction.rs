// SPDX-License-Identifier: MIT OR Apache-2.0
//! **GAP-265: dependencies must flow downward, and this is what enforces it.**
//!
//! `docs/architecture/02-layers.md` Rule 1 says dependencies flow downward only
//! and claimed it is *"enforced via Cargo's dep graph"*. It is not. **Cargo
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
//! # Why this is FIVE tests and not one
//!
//! The checks were verified by independent sabotages, so they must FAIL as
//! independent arms. Folded into one `#[test]`, the first assertion to fire
//! hides every later one: a direction failure would tell you nothing about
//! whether the stale-allowlist check still discriminates. That is the fail-fast
//! problem in miniature, and this project has measured what it costs -- fixing
//! one defect took a macOS CI run from 21 executed suites to 72.
//!
//! The foundation assertions live in [`load`], not in one arm, so **no arm can
//! pass vacuously over missing data**. A gate whose own inputs went missing and
//! stayed green is the exact failure this file exists to prevent.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The parsed workspace: members, normal dependency edges, tiers, allowlist.
struct Workspace {
    members: BTreeSet<String>,
    edges: BTreeSet<(String, String)>,
    tiers: BTreeMap<String, (u32, String)>,
    allow: BTreeMap<(String, String), String>,
    root: PathBuf,
}

impl Workspace {
    /// Edges both of whose endpoints carry a tier. An unassigned crate makes
    /// every edge it touches unjudgeable, in BOTH directions.
    fn judged(&self) -> Vec<&(String, String)> {
        self.edges
            .iter()
            .filter(|(a, b)| self.tiers.contains_key(a) && self.tiers.contains_key(b))
            .collect()
    }

    fn unassigned(&self) -> Vec<&String> {
        self.members
            .iter()
            .filter(|m| !self.tiers.contains_key(*m))
            .collect()
    }
}

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

/// The directory name a `members` entry refers to, or the empty string.
///
/// Extracted so [`workspace_members`] reads as the scan and this reads as the
/// grammar. It changes no condition.
fn member_dir_name(entry: &str) -> String {
    let cleaned: String = entry.chars().filter(|c| *c != '"' && *c != ',').collect();
    cleaned.trim().split('/').next().unwrap_or("").to_string()
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
        if !in_members {
            continue;
        }
        if t.starts_with(']') {
            in_members = false;
            continue;
        }
        let name = member_dir_name(t);
        if !name.is_empty()
            && !name.starts_with('#')
            && root.join(&name).join("Cargo.toml").exists()
        {
            out.insert(name);
        }
    }
    out
}

/// Whether a section header opens NORMAL dependencies.
///
/// dev- and build-dependencies are excluded deliberately: they do not constrain
/// layering. `fuel-test-support` is the worked example -- all six of its
/// consumers take it as a dev-dependency, so it participates in no edge here.
fn opens_normal_deps(header: &str) -> bool {
    header == "[dependencies]"
        || (header.starts_with("[target.") && header.ends_with(".dependencies]"))
}

/// The crate a dependency line declares, or the empty string.
///
/// **Leading whitespace is allowed.** A column-0 anchor misses indented
/// declarations and returns a clean, confident, LOW answer. Measured across
/// every workspace `Cargo.toml` at `ca398d5c`, counting DECLARATION LINES FOR
/// `fuel*` CRATES ONLY, 68 were indented against 103 at column 0 -- 40% of THOSE
/// lines, not of all dependency declarations, which nobody measured. It cost the
/// lane that found it two published claims before they swept the anchor bug
/// backwards.
fn declared_dep_name(line: &str) -> String {
    line.split(['=', '.'])
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('"')
        .to_string()
}

/// Every NORMAL intra-workspace dependency edge, as `(from, to)`.
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
                in_normal_deps = opens_normal_deps(t);
                continue;
            }
            if !in_normal_deps || t.starts_with('#') || t.is_empty() {
                continue;
            }
            let name = declared_dep_name(t);
            if members.contains(&name) && name != *m {
                out.insert((m.clone(), name));
            }
        }
    }
    out
}

/// Tab-separated records, comments and blank lines skipped.
fn tsv_records(path: &Path) -> Vec<Vec<String>> {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    txt.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split('\t').map(|f| f.trim().to_string()).collect())
        .collect()
}

/// Parse everything once, and REFUSE to return a degenerate graph.
fn load() -> Workspace {
    let root = workspace_root();
    let members = workspace_members(&root);
    let edges = dependency_edges(&root, &members);

    let mut tiers = BTreeMap::new();
    for r in tsv_records(&root.join("fuel-ir/tests/crate_dependency_tiers.txt")) {
        if let (Some(name), Some(tier)) = (r.first(), r.get(1))
            && let Ok(n) = tier.parse::<u32>()
        {
            tiers.insert(name.clone(), (n, r.get(2).cloned().unwrap_or_default()));
        }
    }

    let mut allow = BTreeMap::new();
    for r in tsv_records(&root.join("fuel-ir/tests/crate_dependency_tier_allowlist.txt")) {
        if let (Some(a), Some(b)) = (r.first(), r.get(1)) {
            allow.insert(
                (a.clone(), b.clone()),
                r.get(2).cloned().unwrap_or_default(),
            );
        }
    }

    // FOUNDATION. Here rather than in one arm, so NO arm can pass vacuously
    // over missing data -- the failure this gate exists to prevent.
    assert!(
        members.len() >= 30,
        "found only {} workspace members -- the manifest scan is broken, and a \
         broken scan and a clean tree are byte-identical here",
        members.len()
    );
    assert!(
        edges.len() >= 50,
        "found only {} intra-workspace dependency edges -- the manifest parse is \
         broken. A column-0 anchor misses indented declarations and returns a \
         confident low number",
        edges.len()
    );
    assert!(
        tiers.len() >= 25,
        "crate_dependency_tiers.txt assigns only {} crates. If this file were \
         emptied every arm below would judge NOTHING and still pass",
        tiers.len()
    );

    Workspace {
        members,
        edges,
        tiers,
        allow,
        root,
    }
}

#[test]
fn dependencies_flow_downward_only() {
    let w = load();
    let violations: Vec<String> = w
        .judged()
        .iter()
        .filter(|(a, b)| w.tiers[b].0 > w.tiers[a].0)
        .filter(|(a, b)| !w.allow.contains_key(&(a.clone(), b.clone())))
        .map(|(a, b)| {
            format!(
                "{a} (tier {}, {}) -> {b} (tier {}, {})",
                w.tiers[a].0, w.tiers[a].1, w.tiers[b].0, w.tiers[b].1
            )
        })
        .collect();
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

#[test]
fn every_allowlist_entry_carries_a_reason() {
    let w = load();
    let reasonless: Vec<String> = w
        .allow
        .iter()
        .filter(|(_, r)| r.is_empty())
        .map(|((a, b), _)| format!("{a} -> {b}"))
        .collect();
    assert!(
        reasonless.is_empty(),
        "allowlist entries with a BLANK reason: {reasonless:?}\n\
         A bare allowlist is amnesty with no record of which disposition it is, \
         and it takes the guard's signal with it."
    );
}

/// An allowlist entry may not outlive the edge it exempts.
///
/// ⚠️ **THIS CHECKS THE EDGE, NEVER THE REASON, AND A REASON ROTS FIRST.**
/// Worked example, 2026-09-02: Stage 2 (`131e8b84`) left the `fuel-nn -> fuel`
/// entry perfectly valid while making BOTH halves of its reason false — it
/// named `fuel-transformers` as a dependent (now 0) and predicted the repoint
/// would unblock "until Stage 2 moves it", when the one blocking symbol,
/// `fuel::lazy_latent_cache::LatentCache`, is precisely the file Stage 2
/// carved OUT by design. **The entry was correct; its justification had
/// become fiction, and this arm is structurally blind to that.**
///
/// No prose guard is built for it deliberately. A reason is free text, so any
/// checker would be pattern-matching English — the shape that scores ~10%
/// precision and starts flagging correct statements, taking the guard's signal
/// with it. **The honest remedy is a human re-reading reasons when the graph
/// moves, and this comment naming the exposure so it is not mistaken for
/// covered.**
#[test]
fn no_allowlist_entry_outlives_its_edge() {
    let w = load();
    let stale: Vec<String> = w
        .allow
        .keys()
        .filter(|(a, b)| !w.edges.contains(&(a.clone(), b.clone())))
        .map(|(a, b)| format!("{a} -> {b}"))
        .collect();
    assert!(
        stale.is_empty(),
        "STALE allowlist entries -- the edge no longer exists: {stale:?}\n\
         This list may only SHRINK. Delete the entry; the repoint landed."
    );
}

/// Coverage is a RATCHET, not a report.
///
/// A green over 92% of the edges is not the same claim as a green over all of
/// them, and a shrinking denominator is invisible in a pass. The exemption count
/// may only fall: a new crate with no tier fails HERE rather than silently
/// widening the blind spot.
#[test]
fn unjudged_coverage_may_only_shrink() {
    let w = load();
    let unassigned = w.unassigned();
    let judged = w.judged();
    println!(
        "[dep-direction] judged {} of {} edges; {} of {} crates assigned; {} unassigned: {:?}",
        judged.len(),
        w.edges.len(),
        w.tiers.len(),
        w.members.len(),
        unassigned.len(),
        unassigned
    );
    assert!(
        unassigned.len() <= 5,
        "{} crates carry no tier (5 at f9bdfb20): {:?}\n\
         Every edge they touch is UNJUDGED, in both directions. If you added a \
         crate, give it a tier. This bound may only SHRINK -- the header of \
         crate_dependency_tiers.txt lists what the current exemptions cost.",
        unassigned.len(),
        unassigned
    );
}

/// Where a crate states its own layer in a `**Layer**:` doc line, the tier file
/// must agree with it.
///
/// **This is a CROSS-CHECK between two independently-maintained sources, not a
/// source of truth.** The declared vocabulary is partial (6 of 39 crates with a
/// `lib.rs`) and differently spelled from both the diagram and this gate, so the
/// mapping below is stated explicitly rather than inferred -- the mapping IS the
/// interesting part. `Inference` and `Training` are not diagram band names; the
/// diagram calls that band Use-Case Orchestration.
///
/// One crate is a DECLARED DISAGREEMENT rather than a mapping, and it is
/// recorded instead of forced: `fuel-onnx` self-declares `IO`, while
/// `02-layers.md:76` places it at `Interchange (as-built form of
/// fuel-format-interchange-onnx)`, and this gate's dependency order puts it with
/// the model libraries. **Three sources, three names, no ruling.** Its violation
/// verdict is unaffected either way, so nothing is blocked on resolving it.
///
/// # The escape hatch can empty the population it exempts from
///
/// `disputed` is a hatch, and a hatch that grows silently hollows out the arm:
/// were it to cover all six declaring crates, this test would compare NOTHING
/// and still pass its `declared.len() >= 6` floor, because that floor asserts
/// the PARSE worked, not that anything was CHECKED. **The guard and the escape
/// hatch were measuring different quantities and only one was asserted.** So
/// the count that MOVES if this arm becomes a no-op is asserted instead --
/// comparisons actually made. Same shape as an allowlist that grows until
/// nothing is checked: whenever you add an exemption mechanism, assert the size
/// of the set that SURVIVES it, never the size of the set going in.
#[test]
fn self_declared_layers_agree_with_the_tier_file() {
    let w = load();
    let map: &[(&str, &str)] = &[
        ("Use-Case Orchestration", "use-case orchestration"),
        ("Inference", "use-case orchestration"),
        ("Training", "use-case orchestration"),
        ("Models", "models / libraries"),
    ];
    let disputed: &[(&str, &str)] = &[(
        "fuel-onnx",
        "declares IO; 02-layers:76 says Interchange; tiered with the model \
         libraries by dependency order. Three sources, three names, no ruling.",
    )];

    let mut declared: Vec<(String, String)> = Vec::new();
    for m in &w.members {
        let Ok(txt) = std::fs::read_to_string(w.root.join(m).join("src/lib.rs")) else {
            continue;
        };
        let Some(raw) = txt.lines().find_map(|l| l.split("**Layer**:").nth(1)) else {
            continue;
        };
        // The label ends at the first em-dash, pipe, or sentence stop; several
        // declarations run prose afterwards. It is NOT split on '-', because the
        // longest label in use is "Use-Case Orchestration" -- a hyphen split
        // yields "Use", which is a plausible-LOOKING label that no assertion
        // would reject, so the tidier parser fails SILENTLY. The correct
        // delimiter set is longer and less obvious than the wrong one.
        let label = raw
            .split(['\u{2014}', '|'])
            .next()
            .unwrap_or("")
            .split('.')
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('*')
            .trim()
            .to_string();
        if !label.is_empty() {
            declared.push((m.clone(), label));
        }
    }

    // FOUNDATION for this arm specifically: a broken doc-parse yields zero
    // declarations and then agrees with everything.
    assert!(
        declared.len() >= 6,
        "parsed only {} self-declared layers (6 at f9bdfb20) -- the doc-comment \
         parse is broken, and a broken parse agrees with the tier file about \
         nothing at all while reporting PASS. Declarations found: {declared:?}",
        declared.len()
    );

    let mut mismatches = Vec::new();
    let mut compared = 0usize;
    for (krate, label) in &declared {
        if disputed.iter().any(|(d, _)| d == krate) {
            continue;
        }
        let Some((_, tier_layer)) = w.tiers.get(krate) else {
            continue; // unassigned crates are reported by the coverage arm
        };
        compared += 1;
        match map.iter().find(|(d, _)| d == label).map(|(_, t)| *t) {
            Some(e) if e == tier_layer => {}
            Some(e) => mismatches.push(format!(
                "{krate}: declares {label:?} (maps to {e:?}) but is tiered {tier_layer:?}"
            )),
            None => mismatches.push(format!(
                "{krate}: declares {label:?}, which has no entry in this arm's \
                 vocabulary map. Add the mapping, or record it as disputed WITH \
                 A REASON -- do not force it."
            )),
        }
    }
    // THE COUNT THAT MOVES IF THIS ARM BECOMES A NO-OP. `declared.len()` does
    // not -- it asserts the PARSE worked. Only this asserts anything was
    // CHECKED, and the `disputed` hatch is exactly what could empty it.
    assert!(
        compared >= 4,
        "only {compared} self-declared layers were actually COMPARED (4 at          f9bdfb20). The parse floor above says the doc comments were READ; this          says they were CHECKED. If the `disputed` list grew, it hollowed out          this arm -- assert the set that SURVIVES an exemption, never the set          going in."
    );

    assert!(
        mismatches.is_empty(),
        "a crate's self-declared layer disagrees with its tier:\n  {}\n\n\
         Two independently-maintained sources have drifted. Fix whichever is \
         wrong; this project treats doc-vs-code drift as a defect.",
        mismatches.join("\n  ")
    );
}
