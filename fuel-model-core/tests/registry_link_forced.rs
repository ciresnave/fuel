// SPDX-License-Identifier: MIT OR Apache-2.0
//! The other half of arm 3, and the reason it is a separate file.
//!
//! Each integration test is its own crate and its own link, so
//! `tests/registry.rs` and this file give two independent link environments
//! over the same dependency graph. **They differ by exactly one line** — the
//! `use ... as _;` below — which is what isolates the effect to linkage rather
//! than to anything about the registry itself.
//!
//! ```text
//! tests/registry.rs              no link-forcing line   -> fixture ABSENT
//! tests/registry_link_forced.rs  one link-forcing line  -> fixture PRESENT
//! ```
//!
//! Measured 2026-08-27 on rustc 1.98.0 / `inventory` 0.3 / Windows MSVC.
//! `extern crate fuel_model_fixture;` behaves identically; `use .. as _;` is
//! the edition-2024 idiom for "link this, I name nothing from it".
//!
//! ⚠️ **This file does NOT rescue the 02-layers promise — it measures the
//! price of it.** The claim was *"merely depending"*; the truth is *"depending
//! plus one line"*. Keeping the two files apart is what stops that one line
//! from quietly becoming invisible inside a passing gate.

// The whole point. No item from this crate is named anywhere below.
use fuel_model_fixture as _;

use fuel_model_core::{ModelSpec, lookup, registered_architectures};

/// **With linkage forced, the dependency's registration survives and is
/// usable.**
///
/// Asserts both halves, because listing and resolving are different failures:
/// a registry that lists an architecture whose builder was dropped is worse
/// than one that omits it, since the omission is at least visible in the list.
#[test]
fn a_link_forced_dependency_registers_and_builds() {
    let known = registered_architectures();
    assert!(
        known.contains(&"fixture-never-referenced"),
        "one `use .. as _;` was not enough to keep the dependency's \
         registration alive: {known:?}. If this fails while tests/registry.rs \
         still passes, link-time registration does not work here AT ALL and \
         `AutoModel` cannot rely on it — that is a finding for 02-layers, not \
         a test to adjust."
    );

    let reg = lookup("fixture-never-referenced").expect("listed, so it must resolve");
    assert_eq!(
        reg.architecture, "fixture-never-referenced",
        "identity, not presence: a registry matching loosely would satisfy the \
         assertion above unchanged"
    );

    let model = (reg.build)(&ModelSpec {
        path: std::path::PathBuf::from("<no artifact needed>"),
        architecture: "fixture-never-referenced".to_string(),
    })
    .expect("the builder must have survived linking too, not just the entry");
    assert_eq!(model.architecture(), "fixture-never-referenced");
}
