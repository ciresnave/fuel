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

#[test]
fn facade_globs_are_disjoint() {
    let core = pub_mod_names(CORE_LIB);
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
