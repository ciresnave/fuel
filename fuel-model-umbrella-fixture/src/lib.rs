// SPDX-License-Identifier: MIT OR Apache-2.0
//! A stand-in for `fuel-transformers`: the **umbrella** shape 02-layers
//! describes as *"an optional umbrella re-exporting the `fuel-model-*` crates
//! behind features, so a consumer gets either granular or
//! batteries-included."*
//!
//! It exists to answer one question: **does an umbrella's `pub use` of a leaf
//! force that leaf to link, so a batteries-included consumer gets the leaf's
//! registration without writing a link line?**
//!
//! Two things live here and they play different roles:
//!
//! - the **re-export** below is the SUBJECT — the thing under test;
//! - its **own registration** is the CONTROL. Without it, a consumer that
//!   somehow failed to link the umbrella at all would produce an absent
//!   fixture that looks like a negative result but is really a broken setup.

/// The subject. 02-layers' umbrella re-exports its leaves; this is that line.
pub use fuel_model_fixture;

use fuel_model_core::{Model, ModelError, ModelRegistration, ModelSpec};

/// The umbrella's own model, so a consumer can observe that the UMBRELLA
/// linked independently of whether the leaf did.
pub struct UmbrellaModel {
    architecture: String,
}

impl Model for UmbrellaModel {
    fn architecture(&self) -> &str {
        &self.architecture
    }
}

fn build(spec: &ModelSpec) -> Result<Box<dyn Model>, ModelError> {
    Ok(Box::new(UmbrellaModel {
        architecture: spec.architecture.clone(),
    }))
}

inventory::submit! {
    ModelRegistration { architecture: "umbrella-fixture", build }
}
