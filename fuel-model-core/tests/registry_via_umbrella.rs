// SPDX-License-Identifier: MIT OR Apache-2.0
//! **MEASURED: 02-layers' registration promise holds on the UMBRELLA path and
//! fails only on the granular one.**
//!
//! `tests/registry.rs` measured that a leaf which is only a dependency never
//! reaches the registry. 02-layers offers a second consumer path:
//!
//! > `fuel-transformers` — an optional umbrella **re-exporting** the
//! > `fuel-model-*` crates behind features, so a consumer gets *either*
//! > granular *or* batteries-included
//!
//! A `pub use` is a reference, and it turns out to be enough.
//!
//! ```text
//! consumer -> leaf directly, no reference       leaf ABSENT   (tests/registry.rs)
//! consumer -> umbrella -> leaf via `pub use`    leaf PRESENT  (this file)
//! consumer references nothing at all            registry []   (control below)
//! ```
//!
//! Measured 2026-08-27, rustc 1.98.0 / `inventory` 0.3 / Windows MSVC.
//!
//! **So the correction to 02-layers is a SCOPING one, not a retraction:**
//! *"no feature gates works"* survives on the batteries-included path, which
//! is the path most consumers take. Granular consumers
//! (`cargo add fuel-model-llama`) need one link-forcing line. And the reason a
//! scaffolder cannot paper over the granular case is that the line goes in the
//! **consumer**, not in the model crate — so the umbrella is the answer, not
//! codegen.
//!
//! ## Why the consumer references the umbrella
//!
//! It must — otherwise the umbrella is itself an unreferenced dependency, does
//! not link, and this file measures nothing while looking like a clean result.
//! `use .. as _;` stands in for "the consumer uses the umbrella somehow",
//! which is the only reason anyone depends on one.
//!
//! **Nothing from the leaf is named here** — no `use`, no path, no identifier.
//! The leaf reaches the registry through the umbrella's re-export or not at
//! all.
//!
//! ## The control, and it was run in both directions
//!
//! The umbrella registers its OWN architecture, asserted first: if that is
//! absent the umbrella did not link, and an absent leaf would be a setup
//! failure wearing the shape of a negative result.
//!
//! **And the discriminating direction was measured, not assumed:** removing
//! the `use .. as _;` below makes the registry come back **empty — `[]`**,
//! both umbrella and leaf gone. That is what rules out "everything links in
//! this binary regardless" as the explanation for the positive result.

// The consumer's reference to the UMBRELLA. Not to the leaf.
use fuel_model_umbrella_fixture as _;

use fuel_model_core::registered_architectures;

/// An umbrella `pub use` forces its leaf to link, so the leaf registers.
#[test]
fn an_umbrella_reexport_makes_its_leaf_register() {
    let known = registered_architectures();

    // CONTROL — the umbrella itself must have linked.
    assert!(
        known.contains(&"umbrella-fixture"),
        "the UMBRELLA's own registration is missing: {known:?}. The umbrella \
         did not link, so nothing below is a measurement of the re-export -- \
         an absent leaf here would be a broken setup, not a negative result."
    );

    // SUBJECT — the umbrella's `pub use` drags the leaf's registration in.
    assert!(
        known.contains(&"fixture-never-referenced"),
        "an umbrella `pub use` of a leaf NO LONGER forces the leaf to link. \
         Registry: {known:?} -- umbrella present, leaf absent.\n\n\
         That would mean 02-layers' promise is false on BOTH consumer paths, \
         and the scoping correction recorded in this file's docs is wrong: \
         neither granular nor batteries-included gets a leaf into the registry \
         without a link-forcing line. DO NOT add a reference to the leaf to \
         make this pass -- that converts the gate into a tautology. Flip it to \
         an asserted absence with this text as the finding, as \
         tests/registry.rs does, and tell the architect the constitution needs \
         the stronger correction."
    );
}
