// SPDX-License-Identifier: MIT OR Apache-2.0
//! Facade glob-disjointness gate (restructure Stage 2).
//!
//! The facade re-exports TWO globs — `pub use fuel_core::*` (Foundation) and
//! `pub use fuel_transformers::models::*` (the 146 moved model modules). If a name
//! is exported by BOTH, `fuel::<name>` becomes ambiguous and breaks at a
//! CONSUMER's use site — a defect the facade itself compiles through (glob
//! conflicts are lazy in Rust) and that surfaces only downstream, possibly months
//! later, on a name nobody has used yet.
//!
//! Today the sets are disjoint by the STAY-LIST, not by construction: fuel-core
//! keeps `lazy` (the tensor API) and `lazy_latent_cache` (the sole carve-out) —
//! neither a `lazy_<model>` name — while the movers are all `lazy_<model>`. That
//! is a runtime fact about two file listings. A model added to fuel-transformers
//! named `lazy`/`lazy_latent_cache`, or a new fuel-core root module named
//! `lazy_<x>`, would collide. This test fails loudly the moment the two
//! module-name sets intersect.
//!
//! SCOPE / KNOWN HOLE — named rather than papered over: this compares `pub mod`
//! DECLARATIONS, but `pub use fuel_core::*` re-exports the crate's ROOT public
//! NAME SET — root `pub use` re-exports and root `pub struct`/`fn`/`const` too,
//! not only modules. The realistic collision is a MODULE name (the movers are all
//! `lazy_<X>` modules), which this catches. The construct that SLIPS PAST: a
//! `pub use some::path as lazy_foo;` at fuel-core's root — a NON-module export
//! spelled like a model — would collide in the facade while this test stayed
//! green. Symbol-level resolution is hard in Rust and not worth building for that
//! narrow case; the hole is stated so a reader sees it instead of trusting a gate
//! that looks total. Parsed from the sources at compile time (`include_str!`),
//! so hermetic.

use std::collections::BTreeSet;

const CORE_LIB: &str = include_str!("../../fuel-core/src/lib.rs");
const MODELS_MOD: &str = include_str!("../../fuel-transformers/src/models/mod.rs");

/// Every root-level `pub mod <name>;` name in a source file (not `pub(crate) mod`,
/// which a glob does not re-export).
fn pub_mod_names(src: &str) -> BTreeSet<String> {
    src.lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("pub mod ")
                .and_then(|r| r.strip_suffix(';'))
                .map(|n| n.trim().to_string())
        })
        .collect()
}

/// Every name introduced at the crate root via `pub use <path>::NAME;` or
/// `pub use <path>::{NAME, ...};` (single- or multi-line) -- the same
/// root-name-set effect as `pub mod` for THIS test's purposes. Added for
/// board #109: `lazy`/`lazy_latent_cache` moved from direct `pub mod`
/// declarations to `pub use fuel_tensor::{...}` re-exports, so the test's
/// previous `pub_mod_names`-only check stopped seeing them even though
/// `fuel_core::lazy`/`fuel::lazy` still resolve identically at runtime --
/// the LAYOUT changed, the invariant this test protects did not.
///
/// Line/brace-based, not a real Rust parser (same stated-scope tradeoff as
/// `pub_mod_names` above): it doesn't filter by source crate, so it also
/// picks up names from unrelated re-exports (`Layout`, `Storage`, `probe`,
/// ...). That's safe here -- the caller only intersects this set against
/// `lazy_<model>`-shaped names, so a non-colliding extra name is inert.
fn pub_use_names(src: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut buf = String::new();
    let mut in_use = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if !in_use {
            match trimmed.strip_prefix("pub use ") {
                Some(rest) => {
                    in_use = true;
                    buf.clear();
                    buf.push_str(rest);
                }
                None => continue,
            }
        } else {
            buf.push(' ');
            buf.push_str(trimmed);
        }
        if let Some((stmt, _)) = buf.split_once(';') {
            if let Some(path_part) = stmt.rsplit("::").next() {
                let path_part = path_part.trim();
                if let Some(inner) = path_part
                    .strip_prefix('{')
                    .and_then(|s| s.strip_suffix('}'))
                {
                    for item in inner.split(',') {
                        // `as ALIAS` introduces ALIAS at the root, not the
                        // original name -- `.last()` takes the alias when
                        // present (sabotage-verified: `pub use X::lazy as
                        // lazy_llama2c;` must register `lazy_llama2c`, not
                        // `lazy`, or a real collision slips past this gate).
                        let name = item.trim().split(" as ").last().unwrap_or("").trim();
                        if !name.is_empty() {
                            names.insert(name.to_string());
                        }
                    }
                } else if !path_part.is_empty() {
                    let name = path_part.split(" as ").last().unwrap_or("").trim();
                    if !name.is_empty() {
                        names.insert(name.to_string());
                    }
                }
            }
            in_use = false;
            buf.clear();
        }
    }
    names
}

#[test]
fn facade_globs_are_disjoint() {
    let mut core = pub_mod_names(CORE_LIB);
    core.extend(pub_use_names(CORE_LIB));
    let models = pub_mod_names(MODELS_MOD);

    // Positive controls: an empty parse would make the disjointness vacuous.
    assert!(
        core.len() >= 5,
        "parsed only {} fuel-core pub mods — parser or include path broken",
        core.len()
    );
    assert!(
        models.len() >= 100,
        "parsed only {} model pub mods — parser or include path broken",
        models.len()
    );

    // Stay-list controls: the two carve-outs live in fuel-core, NOT in models.
    assert!(
        core.contains("lazy"),
        "fuel-core must keep `lazy` (the tensor API)"
    );
    assert!(
        core.contains("lazy_latent_cache"),
        "fuel-core must keep `lazy_latent_cache` (the sole Stage-2 stay-list member)"
    );
    assert!(
        !models.contains("lazy_latent_cache"),
        "lazy_latent_cache must NOT be in fuel-transformers::models — it stayed in fuel-core"
    );
    assert!(
        models.contains("lazy_bert"),
        "sanity: a known moved model must be in the models set"
    );

    // The load-bearing check: the two module globs must not share a name, or
    // `fuel::<name>` is ambiguous at every consumer use site.
    let collision: Vec<&String> = core.intersection(&models).collect();
    assert!(
        collision.is_empty(),
        "facade glob COLLISION — `pub use fuel_core::*` and \
         `pub use fuel_transformers::models::*` both export: {collision:?}. \
         `fuel::<name>` would be ambiguous downstream.",
    );
}
