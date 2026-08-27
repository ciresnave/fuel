// SPDX-License-Identifier: MIT OR Apache-2.0
//! A stand-in `fuel-model-*` leaf, existing for exactly one purpose: to be
//! **depended on and never named**.
//!
//! ⚠️ **Nothing in `fuel-model-core`'s tests may `use` this crate, name this
//! type, or mention `fuel_model_fixture` in source.** The moment it does, the
//! reference itself keeps the crate alive and arm 3 of `tests/registry.rs`
//! passes for a reason that has nothing to do with link-time registration —
//! Convention 18, *a check can pass because something ELSE guarantees its
//! outcome.*
//!
//! This crate is the consumer-side half of the promise 02-layers makes:
//! *"merely depending on a `fuel-model-*` crate makes it appear in the
//! registry."* If a linker drops it for lack of references, the registry comes
//! up short and nothing errors — which is precisely the failure arm 3 exists
//! to make loud.

use fuel_model_core::{Model, ModelError, ModelRegistration, ModelSpec};

/// The fixture model. Carries the architecture it was built for so a test can
/// assert on **identity** rather than on "something came back".
pub struct FixtureModel {
    architecture: String,
}

impl Model for FixtureModel {
    fn architecture(&self) -> &str {
        &self.architecture
    }
}

/// The registered builder. Never called by name from a test — only reached
/// through the registry.
fn build(spec: &ModelSpec) -> Result<Box<dyn Model>, ModelError> {
    Ok(Box::new(FixtureModel {
        architecture: spec.architecture.clone(),
    }))
}

inventory::submit! {
    ModelRegistration {
        // Deliberately not a real architecture name: if a future real
        // `fuel-model-*` leaf ever claims this key, the collision is a bug in
        // the fixture, not a silent shadowing of production behaviour.
        architecture: "fixture-never-referenced",
        build,
    }
}
