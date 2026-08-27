// SPDX-License-Identifier: MIT OR Apache-2.0
//! # fuel-model-core
//!
//! The model tier's core, ratified in
//! [02-layers](../../docs/architecture/02-layers.md) v0.5 as
//! *"the `Model` trait, the `model_type`/`general.architecture` → builder
//! **registry**, and `AutoModel::from_path`"*.
//!
//! ## Why this crate exists, measured
//!
//! 02-layers v0.5 names this layer. Until this commit it did not exist —
//! `grep -c fuel-model-core Cargo.toml` returned **0** against a positive
//! control of **3** for `fuel-core`. A ratified layer with nothing behind it.
//!
//! The measured price of its absence, per ROADMAP item 8: **399 `*Weights`
//! structs · 148 `*Config` structs · 27 `from_hf_json_str` definitions**.
//! `LlamaFullConfig` and `build_llama3_model` are not an *example* of the
//! pattern — they are the pattern, 148 times.
//!
//! ## What this crate is NOT, and the boundary is deliberate
//!
//! ROADMAP item 8 decomposes into three independently-landable increments.
//! This is **(I) only**: the registry and `AutoModel`.
//!
//! - **(II) the config core** — 27 hand-rolled `from_hf_json_str` collapsing
//!   to a common core plus per-model extension. **Not here.** [`ModelSpec`] is
//!   therefore deliberately thin: it carries the path and the architecture key
//!   and *nothing parsed*, because parsing is (II)'s subject and inventing a
//!   config shape here would pre-empt it.
//! - **(III) the weight-source abstraction** — `WeightSource` across 399
//!   structs. **Not here, and explicitly held until after migration Stage 2.**
//!
//! A builder registered today therefore receives a path and a name. That is
//! the honest surface for (I), and it is why [`ModelSpec`] has no `config` and
//! no `weights` field: those would be shapes guessed ahead of the increments
//! that own them.
//!
//! ## Registration is link-time distributed, and that promise is gated
//!
//! Per 02-layers: *"merely depending on a `fuel-model-*` crate makes it appear
//! in the registry, so 'no feature gates' works and the scaffolder can emit a
//! self-registering crate without editing a central dispatch file."*
//!
//! ⚠️ **That promise is exactly what link-time registration breaks silently.**
//! A linker may drop a crate whose symbols nothing references, taking its
//! registration section with it. The failure is not a compile error and not a
//! panic — the model is simply not there. `tests/registry.rs` arm 3 is the
//! only arm that tests it, and it is constructed so that naming the fixture
//! crate anywhere in the test source would destroy what it measures.

use std::path::{Path, PathBuf};

/// A constructed model.
///
/// Deliberately minimal for increment (I). `forward()` is **not** here and is
/// not coming here: 02-layers assigns the architecture to the model leaves,
/// and ROADMAP item 8's own analysis answers *"what should be generic?"* with
/// **"(iv) `forward()` — NO. That is the architecture."**
pub trait Model: Send + Sync {
    /// The architecture key this model was built for — the same string the
    /// registry resolved. Used by [`crate::tests`]-style identity assertions
    /// and by diagnostics; it is the model's own account of what it is.
    fn architecture(&self) -> &str;
}

/// What a registered builder receives.
///
/// Thin **on purpose** — see the crate docs. It carries the path and the
/// resolved architecture key. It does **not** carry a parsed config (that is
/// increment II) or a weight source (increment III).
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The path `AutoModel` was asked about — a directory holding
    /// `config.json`, or a single-file artifact.
    pub path: PathBuf,
    /// The architecture key resolved from the artifact, e.g. `"llama"`.
    pub architecture: String,
}

/// Errors from resolving or building a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The artifact named an architecture no registered crate claims.
    ///
    /// Carries the full registered set, because the useful question when this
    /// fires is *"is the crate linked in at all?"* and a bare "unknown
    /// architecture" cannot answer it.
    #[error("no registered model for architecture `{key}`; registered: {known:?}")]
    UnknownArchitecture {
        /// The key read from the artifact.
        key: String,
        /// Every architecture currently in the registry.
        known: Vec<String>,
    },

    /// The path exists but carries no architecture key this crate can read.
    ///
    /// Increment (I) reads HuggingFace `config.json`'s `model_type` only.
    /// GGUF's `general.architecture` needs the interchange tier and is a
    /// declined-by-name case rather than a silent miss.
    #[error("cannot read an architecture key from {path}: {detail}")]
    UnreadableArtifact {
        /// The artifact inspected.
        path: PathBuf,
        /// Why the key could not be read.
        detail: String,
    },

    /// The registered builder itself failed.
    #[error("builder for `{key}` failed: {detail}")]
    BuildFailed {
        /// The architecture whose builder ran.
        key: String,
        /// The builder's own message.
        detail: String,
    },
}

/// One crate's claim on an architecture.
///
/// A `fuel-model-*` leaf submits one of these; nothing edits a central list.
pub struct ModelRegistration {
    /// The `model_type` / `general.architecture` string this builder answers to.
    pub architecture: &'static str,
    /// Construct the model from a [`ModelSpec`].
    pub build: fn(&ModelSpec) -> Result<Box<dyn Model>, ModelError>,
}

inventory::collect!(ModelRegistration);

/// Every architecture currently registered, sorted.
///
/// Sorted so a diagnostic is stable and diffable; link order is not.
pub fn registered_architectures() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = inventory::iter::<ModelRegistration>
        .into_iter()
        .map(|r| r.architecture)
        .collect();
    v.sort_unstable();
    v
}

/// Resolve one architecture key to its registration.
///
/// Returns the registration itself rather than a bool, so a caller can assert
/// on **identity** — which builder answered — instead of on presence. A test
/// that only checks `is_some()` cannot tell a correct registry from one that
/// accepts everything.
pub fn lookup(architecture: &str) -> Option<&'static ModelRegistration> {
    inventory::iter::<ModelRegistration>
        .into_iter()
        .find(|r| r.architecture == architecture)
}

/// Read the architecture key from an artifact without building anything.
///
/// Increment (I) supports HuggingFace layout: a directory containing
/// `config.json` with a `model_type` field, or that file directly. Anything
/// else is [`ModelError::UnreadableArtifact`] **by name** — a declined case,
/// never a guess.
pub fn architecture_of(path: &Path) -> Result<String, ModelError> {
    let cfg = if path.is_dir() {
        path.join("config.json")
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read_to_string(&cfg).map_err(|e| ModelError::UnreadableArtifact {
        path: cfg.clone(),
        detail: format!("cannot read: {e}"),
    })?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| ModelError::UnreadableArtifact {
            path: cfg.clone(),
            detail: format!("not JSON: {e}"),
        })?;
    v.get("model_type")
        .and_then(|m| m.as_str())
        .map(str::to_owned)
        .ok_or_else(|| ModelError::UnreadableArtifact {
            path: cfg,
            detail: "no string `model_type` field (GGUF `general.architecture` \
                     needs the interchange tier and is not read by increment I)"
                .to_string(),
        })
}

/// The architecture-dispatching entry point.
pub struct AutoModel;

impl AutoModel {
    /// Build whichever registered model claims this artifact's architecture.
    pub fn from_path(path: &Path) -> Result<Box<dyn Model>, ModelError> {
        let key = architecture_of(path)?;
        let reg = lookup(&key).ok_or_else(|| ModelError::UnknownArchitecture {
            key: key.clone(),
            known: registered_architectures()
                .into_iter()
                .map(str::to_owned)
                .collect(),
        })?;
        (reg.build)(&ModelSpec {
            path: path.to_path_buf(),
            architecture: key,
        })
    }
}
