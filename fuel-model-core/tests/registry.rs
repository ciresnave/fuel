// SPDX-License-Identifier: MIT OR Apache-2.0
//! The registry gate for ROADMAP item 8 increment (I).
//!
//! ⚠️ **"Compiles + registers" is not a gate.** *Compiles* is the precondition
//! for having one, and *registers* as usually written is satisfied by a
//! registry that accepts everything — indistinguishable from one that resolves
//! correctly until someone asks it for something that should not be there.
//!
//! So three arms, and only the third tests the claim 02-layers actually makes:
//!
//! ```text
//! 1. registered-and-referenced       resolves, asserted by IDENTITY
//! 2. never-registered                fails, and the arm is OBSERVED executing
//! 3. registered-but-NEVER-referenced still resolves   <- the real test
//! ```
//!
//! ⚠️⚠️ **THIS FILE MUST NEVER *REFERENCE* `fuel_model_fixture` IN CODE** --
//! no `use`, no `extern crate`, no path, no identifier. Prose mentions like
//! this one are fine and are why the rule is stated as "reference" rather than
//! "name": doc comments are not code and do not force linkage, which the
//! passing state of this file demonstrates while mentioning the crate three
//! times. Naming it would create the reference that makes link-time
//! registration unnecessary, and arm 3 would then pass because of the `use`
//! statement rather than because of the registry — Convention 18 in its purest
//! form. If a future edit needs to name it to compile, the test has stopped
//! testing the thing and the edit is wrong.

use fuel_model_core::{
    AutoModel, Model, ModelError, ModelRegistration, ModelSpec, architecture_of, lookup,
    registered_architectures,
};

// ---------------------------------------------------------------------------
// Arm 1's own model, registered from THIS crate and referenced by name here.
// That reference is fine: arm 1 is about resolution and identity, not about
// link survival.
// ---------------------------------------------------------------------------

struct LocalModel {
    architecture: String,
}

impl Model for LocalModel {
    fn architecture(&self) -> &str {
        &self.architecture
    }
}

fn build_local(spec: &ModelSpec) -> Result<Box<dyn Model>, ModelError> {
    Ok(Box::new(LocalModel {
        architecture: spec.architecture.clone(),
    }))
}

inventory::submit! {
    ModelRegistration { architecture: "local-test-arch", build }
}

// `build` is named separately so the submit above reads as data; keeping the
// fn item and the submission adjacent is what makes a leaf crate copy-able.
use build_local as build;

// ---------------------------------------------------------------------------
// ARM 1 — a registered model resolves, and WHAT it resolves to is asserted.
// ---------------------------------------------------------------------------

/// **Identity, not presence.**
///
/// `assert!(lookup(..).is_some())` would pass against a registry that returns
/// the first entry for every key, or one that accepts everything. Asserting
/// the returned registration's own `architecture` — and then the built model's
/// — is what distinguishes a resolver from an acceptor.
#[test]
fn a_registered_architecture_resolves_to_its_own_builder() {
    let reg = lookup("local-test-arch").expect("registered architecture must resolve");

    assert_eq!(
        reg.architecture, "local-test-arch",
        "the registry returned a registration for a DIFFERENT architecture — \
         it is matching loosely, and every `is_some()` assertion in this file \
         would still pass"
    );

    let spec = ModelSpec {
        path: std::path::PathBuf::from("<arm 1 has no artifact>"),
        architecture: "local-test-arch".to_string(),
    };
    let model = (reg.build)(&spec).expect("the registered builder must build");
    assert_eq!(
        model.architecture(),
        "local-test-arch",
        "the builder produced a model that disagrees about what it is"
    );
}

// ---------------------------------------------------------------------------
// ARM 2 — the negative arm, and it must be OBSERVED executing.
// ---------------------------------------------------------------------------

/// **A decision branch nobody has watched run is a mechanism, not a guarantee.**
///
/// This arm exists because a registry that accepts everything passes arm 1
/// unchanged. It asserts the *typed* failure and that the error names the
/// registered set — a bare "not found" cannot answer the question that
/// actually gets asked when this fires, which is *"is the crate linked in at
/// all?"*
#[test]
fn an_unregistered_architecture_fails_to_resolve() {
    assert!(
        lookup("architecture-that-was-never-registered").is_none(),
        "the registry resolved an architecture nobody registered — it is \
         accepting everything, which makes arm 1 vacuous"
    );

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "architecture-that-was-never-registered"}"#,
    )
    .expect("write config");

    match AutoModel::from_path(dir.path()) {
        Err(ModelError::UnknownArchitecture { key, known }) => {
            assert_eq!(key, "architecture-that-was-never-registered");
            assert!(
                known.contains(&"local-test-arch".to_string()),
                "the error must report the registry it searched; got {known:?}"
            );
        }
        other => panic!(
            "an unregistered architecture must fail as UnknownArchitecture; got {:?}",
            other.map(|m| m.architecture().to_string())
        ),
    }
}

// ---------------------------------------------------------------------------
// ARM 3 — THE REAL TEST. Nothing below names the fixture crate.
// ---------------------------------------------------------------------------

/// **MEASURED: a dependency that is never referenced does NOT reach the
/// registry — so 02-layers' promise, as worded, is FALSE on this toolchain.**
///
/// 02-layers v0.5 says *"merely depending on a `fuel-model-*` crate makes it
/// appear in the registry, so 'no feature gates' works."* This test is the arm
/// that checks it, and it found the opposite.
///
/// Measured 2026-08-27, rustc 1.98.0, `inventory` 0.3, Windows MSVC:
///
/// ```text
/// this file, no link-forcing line   -> registry = ["local-test-arch"]      FIXTURE ABSENT
/// tests/registry_link_forced.rs     -> registry includes the fixture       PRESENT
/// ```
///
/// The two files differ by **one line** — `use fuel_model_fixture as _;` —
/// and nothing else. `extern crate fuel_model_fixture;` was measured to work
/// identically. So the true rule is **"depend PLUS one link-forcing line in
/// the consumer"**, not "merely depend": Rust does not link an rlib whose
/// symbols nothing references, and the registration section goes with it.
///
/// ⚠️ **This is asserted as ABSENCE deliberately, and it is NOT the gate
/// passing by giving up.** Asserting the measured behaviour means this test
/// goes RED the day the behaviour changes — a newer `inventory`, a different
/// linker, `-C link-dead-code`, or a toolchain that keeps the section. That
/// red would be *good news* and the correct response is to restore the
/// original claim, not to update this assertion. An `#[ignore]` here would
/// have recorded the same fact while detecting nothing.
///
/// **The practical consequence is smaller than the wording suggests and
/// should be stated fairly:** one additive line per model crate in the
/// consumer, which a scaffolder can emit, is still materially better than
/// editing a central dispatch file — which is the property 02-layers was
/// actually reaching for. What does NOT hold is the zero-consumer-code form.
#[test]
fn a_dependency_that_is_never_referenced_does_not_register() {
    let known = registered_architectures();

    assert!(
        known.contains(&"local-test-arch"),
        "positive control: this file's OWN registration must be present, or \
         the assertion below proves nothing about linkage — an empty registry \
         would satisfy it for the wrong reason. got {known:?}"
    );

    assert!(
        !known.contains(&"fixture-never-referenced"),
        "the unreferenced dependency DID reach the registry: {known:?}. That \
         contradicts the measurement this test records, which means link-time \
         registration now delivers 02-layers' promise as originally worded. \
         That is GOOD NEWS: restore the original claim (assert PRESENCE), \
         delete tests/registry_link_forced.rs's justification, and correct \
         02-layers' status. Do not simply flip this assertion."
    );
}

// ---------------------------------------------------------------------------
// The key reader, which `AutoModel::from_path` depends on.
// ---------------------------------------------------------------------------

/// `architecture_of` declines by NAME rather than guessing.
#[test]
fn an_artifact_without_a_model_type_is_declined_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), r#"{"hidden_size": 4096}"#)
        .expect("write config");

    match architecture_of(dir.path()) {
        Err(ModelError::UnreadableArtifact { detail, .. }) => assert!(
            detail.contains("model_type"),
            "the decline must name the field it looked for; got {detail}"
        ),
        other => panic!("a config without `model_type` must be declined; got {other:?}"),
    }
}
