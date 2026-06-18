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

    // ===== lowering (this slice) — string → typed dispatch records =====
    /// Neither `op_kind` nor `fused_op` is present, or both are present.
    /// Exactly one must be declared (a primitive vs fused contract; §3.7,
    /// V-FKC-5). The structural check happens at lower time.
    #[error(
        "FKC §3.7 (V-FKC-5): kernel `{section}` must declare exactly one of `op_kind` / \
         `fused_op` (op_kind={op_kind:?}, fused_op={fused_op:?})"
    )]
    OpTargetAmbiguous {
        section: String,
        op_kind: Option<String>,
        fused_op: Option<String>,
    },

    /// An `op_kind` string did not match any as-built `OpKind` variant
    /// (§2.1; the lower-time `match` is exhaustive over `OpKind`).
    #[error("FKC §2.1: kernel `{section}` names unknown op_kind `{op_kind}`")]
    UnknownOpKind { section: String, op_kind: String },

    /// A `fused_op` string did not match any registered `FusedOpId` name
    /// (the name table is built from `fuel-graph`'s `default_registry()`;
    /// §2.2).
    #[error("FKC §2.2: kernel `{section}` names unknown fused_op `{fused_op}`")]
    UnknownFusedOp { section: String, fused_op: String },

    /// A dtype token (or a `dtype_class` shorthand expansion) did not
    /// match any as-built `DType`/shorthand (§3.4). The explicit match
    /// makes a bad token a typed error rather than a silent skip.
    #[error("FKC §3.4: kernel `{section}` operand `{operand}` has bad dtype token `{token}`")]
    BadScalarType {
        section: String,
        operand: String,
        token: String,
    },

    /// A `backend` string did not match any as-built `BackendId` (§2.1).
    #[error("FKC §2.1: kernel `{section}` names unknown backend `{backend}`")]
    UnknownBackend { section: String, backend: String },

    /// The `entry_point` symbol was absent from the provider link
    /// registry (§12.6; P9 — the importer never fabricates a pointer).
    #[error(
        "FKC §12.6: kernel `{section}` entry_point `{entry_point}` is not in the link registry"
    )]
    UnknownEntryPoint {
        section: String,
        entry_point: String,
    },

    /// A `layout` tri-state flag carried a value outside the legal set
    /// (`required` / `accepted` / `rejected` / `n/a`); §4.1, §6.
    #[error(
        "FKC §4.1: kernel `{section}` operand `{operand}` layout flag `{flag}` has illegal \
         value `{value}` (expected required|accepted|rejected|n/a)"
    )]
    BadLayoutFlag {
        section: String,
        operand: String,
        flag: String,
        value: String,
    },

    /// A cost expression failed to parse against the §2.3 grammar
    /// (V-FKC-8, §4.4). Carries the offending field + the raw expression.
    #[error(
        "FKC §4.4 (V-FKC-8): kernel `{section}` cost field `{field}` expression `{expr}` does \
         not parse: {reason}"
    )]
    CostExprParse {
        section: String,
        field: String,
        expr: String,
        reason: String,
    },

    /// A precision block is a bare placeholder (no `audited`, no bounds,
    /// no notes) — there is nothing to lower (§4.8, `PlaceholderPrecision`).
    #[error(
        "FKC §4.8: kernel `{section}` precision block is a bare placeholder (no audited flag, \
         no bounds, no notes)"
    )]
    PlaceholderPrecision { section: String },

    // ===== registration (this slice) — Resolved* → the two registries =====
    /// A registration drove the same `KernelRef` function pointer onto one
    /// `(op, dtypes, backend)` decision point twice (§3 / V-FKC-3). The
    /// importer never panics on this; it surfaces the
    /// [`crate::kernel::KernelBindingTable::finalize`] gate as a typed
    /// error after all inserts. The string carries the dispatch-layer
    /// message (which names the offending `(op, dtypes, backend)` key).
    #[error("FKC §10 (V-FKC-3): duplicate KernelRef on a single decision point: {0}")]
    DuplicateKernelRef(String),

    /// `import_glob` collected files whose front-matter disagrees: the
    /// `provider.name` / `provider.backend` / `provider.kernel_source`
    /// must match across every file merged into one provider (§9.2).
    #[error(
        "FKC §9.2: provider front-matter mismatch merging globbed files — field `{field}` is \
         `{found}` in `{file}` but the first file declared `{expected}`"
    )]
    ProviderMismatch {
        field: String,
        expected: String,
        found: String,
        file: String,
    },

    /// A glob / file-read I/O failure (no readable file, an unreadable
    /// path, a bad glob pattern). Carries the offending path/pattern + the
    /// underlying error text.
    #[error("FKC: I/O error for `{path}`: {reason}")]
    Io { path: String, reason: String },
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
