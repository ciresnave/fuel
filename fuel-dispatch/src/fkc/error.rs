//! FKC importer error type.
//!
//! Every parse / §3.8-enforcement / structural failure surfaces as a typed
//! [`FkcError`] variant — the importer never panics on a production path
//! (constitution: never-panic; FKC G9, §10). Each variant carries enough
//! context (section name, line number where the YAML layer can provide it) to
//! be actionable.
//!
//! This first slice covers the **parse + restricted-YAML (§3.8) + schema**
//! failure modes. The lowering / validation / registration variants the full
//! `V-FKC-*` battery needs (`UnknownOpKind`, `DuplicateKernelRef`,
//! `ScaleDoubleDeclared`, `ShapeRuleMismatch`, …) land with their respective
//! steps; placeholders for the prose-vs-structured blurb check (§10.11) are
//! provided here so the variant exists ahead of its consumer.

use thiserror::Error;

/// A typed FKC import failure (`Result`-returning, never a panic).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FkcError {
    // ===== §3.8 restricted-YAML pre-pass =====
    /// Tab indentation inside a YAML chunk (front-matter or an fkc block).
    /// YAML forbids tab indentation; §3.8 makes this a hard error with a line
    /// rather than a confusing downstream parse error.
    #[error("FKC §3.8: tab indentation is forbidden (line {line})")]
    TabIndentation { line: usize },

    /// A YAML anchor (`&name`) was used. §3.8 disables anchors so a diff
    /// reviewer sees every value literally and the importer never resolves a
    /// reference graph.
    #[error("FKC §3.8: YAML anchors (`&name`) are disallowed (line {line})")]
    AnchorDisallowed { line: usize },

    /// A YAML alias (`*name`) was used. Disallowed for the same reason as
    /// anchors (§3.8).
    #[error("FKC §3.8: YAML aliases (`*name`) are disallowed (line {line})")]
    AliasDisallowed { line: usize },

    /// A YAML merge key (`<<:`) was used. Disallowed (§3.8).
    #[error("FKC §3.8: YAML merge keys (`<<:`) are disallowed (line {line})")]
    MergeKeyDisallowed { line: usize },

    /// An unquoted Norway-problem token (`no`/`yes`/`on`/`off`/`n`/`y`) appeared
    /// in a scalar value position. §3.8 disarms the Norway problem by requiring
    /// such tokens to be quoted so they cannot silently coerce to a bool.
    #[error(
        "FKC §3.8: unquoted token `{token}` in a scalar position would coerce to a bool under \
         YAML 1.1 (the Norway problem); quote it (line {line})"
    )]
    NorwayToken { token: String, line: usize },

    // ===== markdown / file anatomy (§3.1) =====
    /// The file-level `---`-fenced YAML front-matter is missing or malformed.
    #[error("FKC §3.1: malformed or missing front-matter: {0}")]
    MalformedFrontMatter(String),

    /// A `## ` kernel section has no fenced ` ```fkc ` block (§3.1 requires
    /// exactly one).
    #[error("FKC §3.1: section `{section}` has no ```fkc block")]
    MissingFkcBlock { section: String },

    /// A `## ` kernel section has more than one fenced ` ```fkc ` block (§3.1
    /// requires exactly one).
    #[error("FKC §3.1: section `{section}` has {count} ```fkc blocks (expected exactly 1)")]
    MultipleFkcBlocks { section: String, count: usize },

    // ===== deserialization / schema =====
    /// `serde_yml` failed to deserialize a YAML chunk into the schema. The
    /// string carries the underlying error (and the section name when known).
    #[error("FKC: YAML deserialize error: {0}")]
    Yaml(String),

    /// A required schema field is absent (V-FKC-1, §10). `field` names the
    /// missing key; `section` the kernel section it belongs to.
    #[error("FKC §10 (V-FKC-1): kernel `{section}` is missing required field `{field}`")]
    MissingRequiredField { section: String, field: String },

    // ===== §10.11 blurb equality (variant reserved ahead of its consumer) =====
    /// The structured `blurb:` does not equal the prose blurb (V-FKC-2,
    /// §10.11). Reserved here; the prose-vs-structured comparison is wired in a
    /// later step.
    #[error(
        "FKC §10.11 (V-FKC-2): kernel `{section}` structured blurb does not match prose blurb \
         (structured={structured:?}, prose={prose:?})"
    )]
    BlurbMismatch {
        section: String,
        structured: String,
        prose: String,
    },
}

impl FkcError {
    /// Convenience: wrap a `serde_yml` error (optionally tagged with the
    /// section name) into [`FkcError::Yaml`].
    pub(crate) fn yaml(section: Option<&str>, err: impl std::fmt::Display) -> Self {
        match section {
            Some(s) => FkcError::Yaml(format!("section `{s}`: {err}")),
            None => FkcError::Yaml(err.to_string()),
        }
    }
}
