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

    /// A `` ```fkc `` fenced block appears outside any `## ` section (before the
    /// first heading, or in a file with no `## ` headings). Such a block is
    /// silently dropped by section scanning, so the import would succeed while
    /// adopting nothing — a no-op that looks like success (§3.1).
    #[error(
        "FKC §3.1: a ```fkc block (line {line}) is not under a `## ` heading — every kernel block \
         must belong to a `## ` section (a block outside a section would be silently ignored)"
    )]
    OrphanFkcBlock { line: usize },

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

    /// Two operands that both **vary** (enumerate >1 dtype) in one section
    /// disagree on their fan-out dtype list (§3.4 multi-dtype fan-out). ALL
    /// varying operands must enumerate the SAME dtype list in the SAME order
    /// so the importer can fan the section into one per-dtype binding without
    /// silently picking one operand's list over another's.
    #[error(
        "FKC §3.4: kernel `{section}` operand `{operand}` varies over `{found}` but a prior \
         varying operand enumerated `{expected}` — all varying operands must share one dtype \
         list (and order) for multi-dtype fan-out"
    )]
    FanoutDtypeMismatch {
        section: String,
        operand: String,
        expected: String,
        found: String,
    },

    /// An `optional: true` operand is NOT the LAST input (§3.4 optional-operand
    /// fan-out). Only the last input may be optional — omitting the last input
    /// yields a CONTIGUOUS valid key ending just before the outputs, whereas an
    /// earlier optional operand would leave a hole in the middle of the key and
    /// mis-align every following operand. This is a typed error, never a silent
    /// mis-key.
    #[error(
        "FKC §3.4: kernel `{section}` operand `{operand}` is `optional: true` but is not the \
         last input — only the last input may be optional (omitting it yields a contiguous key)"
    )]
    OptionalOperandNotLast { section: String, operand: String },

    /// A `passthrough(role)` output names the OPTIONAL operand as its dtype
    /// source (§3.4 / §5.1). When the optional operand is absent the passthrough
    /// cannot resolve, so its two keys (with / without the operand) would carry
    /// DIFFERENT output dtypes — an output may not derive its dtype from an
    /// operand that can be omitted. A typed error, never a silent mis-key.
    #[error(
        "FKC §3.4/§5.1: kernel `{section}` output `passthrough({role})` derives its dtype from \
         the optional operand `{role}` — an output may not passthrough an optional operand"
    )]
    PassthroughNamesOptionalOperand { section: String, role: String },

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

    /// A `cost.cost_fn` NAME was absent from the provider link registry's
    /// cost-fn table (§4.4 cost-fn trampoline; the cost-model analogue of
    /// [`Self::UnknownEntryPoint`]). The importer never fabricates a cost-fn
    /// pointer nor silently falls back to `unknown_cost` — a named-but-unknown
    /// cost fn is a typed error.
    #[error(
        "FKC §4.4: kernel `{section}` cost_fn `{cost_fn}` is not in the link registry's cost-fn table"
    )]
    UnknownCostFn { section: String, cost_fn: String },

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

    // ===== §10 build-time validators (V-FKC-*) =====
    /// `fkc_version` exceeds the importer's supported maximum (V-FKC §10.1).
    #[error("FKC §10.1: fkc_version {found} exceeds supported max {max}")]
    UnsupportedVersion { found: u32, max: u32 },

    /// Layout flags are mutually incoherent (V-FKC §10.4): neither
    /// `contiguous` nor `strided` is acceptable; or `broadcast_stride0` /
    /// `reverse_strides` accepted without `strided: accepted`.
    #[error(
        "FKC §10.4 (V-FKC-4): kernel `{section}` operand `{operand}` layout is incoherent: {reason}"
    )]
    LayoutIncoherent {
        section: String,
        operand: String,
        reason: String,
    },

    /// A per-operand (or kernel-wide) `awkward_layout_strategy` contradicts
    /// the operand's own layout flags, or carries an unknown value
    /// (V-FKC §10.5; an unknown value is meaning-bearing per §11.1).
    #[error(
        "FKC §10.5 (V-FKC-5): kernel `{section}` operand `{operand}` awkward_layout_strategy \
         `{strategy}` is incoherent: {reason}"
    )]
    AwkwardStrategyIncoherent {
        section: String,
        operand: String,
        strategy: String,
        reason: String,
    },

    /// A quant block is internally incoherent (V-FKC §10.6): a `GGML_BLOCK`
    /// with a `ScalePair`/`granularity`, an `MX` without `granularity:
    /// PerBlock`, an `AFFINE_*` with a bad granularity, an `AFFINE_BLOCK`
    /// missing its block geometry, a `ggml_dtype` that is not a real
    /// `GgmlDType` variant, a sub-byte operand with no `fdx.quant`, etc.
    #[error("FKC §10.6 (V-FKC-6): kernel `{section}` operand `{operand}` quant is incoherent: {reason}")]
    QuantIncoherent {
        section: String,
        operand: String,
        reason: String,
    },

    /// A scale was declared in two places at once (V-FKC §10.6): both an
    /// `fdx.quant.scale_operand` (separate-input scale) and a sidecar-bundled
    /// scale for the same scale. Each scale lives in exactly one place.
    #[error(
        "FKC §10.6 (V-FKC-6): kernel `{section}` operand `{operand}` declares a scale in two \
         places (scale_operand AND a sidecar scale): {reason}"
    )]
    ScaleDoubleDeclared {
        section: String,
        operand: String,
        reason: String,
    },

    /// A contract is describable but NOT yet registrable on today's types
    /// (V-FKC §10.6): `MX` (no `ScaleGranularity::PerBlock`) or `AFFINE_BLOCK`
    /// (no block-quant descriptor target). The contract parse-validates; this
    /// gates it at registration time (§6).
    #[error(
        "FKC §6/§10.6: kernel `{section}` quant family `{family}` parse-validates but is not yet \
         registrable on today's types ({reason})"
    )]
    MxNotYetRegistrable {
        section: String,
        family: String,
        reason: String,
    },

    /// An op-param `variant` is not a real variant **in the correct
    /// namespace** (V-FKC §10.7): `OpParams` for an `op_kind` contract,
    /// `FusedOpParams` for a `fused_op` contract (§3.7).
    #[error(
        "FKC §10.7 (V-FKC-7): kernel `{section}` op_params variant `{variant}` is not a real \
         {namespace} variant"
    )]
    BadOpParamsVariant {
        section: String,
        variant: String,
        namespace: String,
    },

    /// The required `cost.provenance` marker is absent or not one of
    /// `{declared, judge_measured}` (V-FKC §10.8a; the COST_RULE).
    #[error(
        "FKC §10.8a: kernel `{section}` cost.provenance is missing or invalid (got {found:?}; \
         expected `declared` or `judge_measured`)"
    )]
    CostProvenanceMissing { section: String, found: Option<String> },

    /// A cost block is a bare / placeholder / zero-sentinel cost under either
    /// provenance marker (V-FKC §10.8a) — not the honest `class: free`
    /// metadata-only case.
    #[error("FKC §10.8a: kernel `{section}` ships a placeholder/zero cost ({reason})")]
    PlaceholderCost { section: String, reason: String },

    /// An audited:false + all-null (UNAUDITED) precision on a non-reference
    /// contract (V-FKC §10.9). (Reserved; the lint treats UNAUDITED as a
    /// non-fatal note unless a stricter mode is requested.)
    #[error("FKC §10.9 (V-FKC-9): kernel `{section}` ships UNAUDITED precision")]
    UnauditedPrecision { section: String },

    /// A bundle slot's shape rule yields rank > 6 (V-FKC §10.13), which the
    /// serialized `FDXOutputView` (`[u64; 6]`) cannot represent.
    #[error("FKC §10.13 (V-FKC-7): kernel `{section}` bundle slot `{slot}` has rank {rank} > 6")]
    BundleSlotRankExceeded {
        section: String,
        slot: String,
        rank: usize,
    },

    /// A `fused_op` contract's declared §5.1/§5.2 return rule disagrees with the
    /// real registered `FusedOpEntry` fn at a probe shape (V-FKC-7, Finding 5.1).
    /// `expected`/`actual` render either a shape or a dtype.
    #[error(
        "FKC §5 (V-FKC-7): kernel `{section}` output `{role}` declared return rule disagrees with \
         the registered fused fn (declared {expected}, real {actual})"
    )]
    ShapeRuleMismatch { section: String, role: String, expected: String, actual: String },

    /// A `return.bundle` slot count disagrees with the registered
    /// `output_views` arity (V-FKC-7, Finding 5.2).
    #[error("FKC §5.5 (V-FKC-7): kernel `{section}` declares {actual} bundle slots but output_views has {expected}")]
    BundleArityMismatch { section: String, expected: usize, actual: usize },

    /// A `shape_constraint:` segment committed to §3.5 vocabulary but its
    /// argument is malformed (`rank=banana`, an unclosed `divisible(`, an empty
    /// `dim[0]=`). Non-vocabulary segments degrade to free text (a warning), not
    /// this error — this fires only on a real authoring mistake in the grammar.
    #[error(
        "FKC §3.5: kernel `{section}` operand `{operand}` shape_constraint segment `{raw}` \
         uses vocabulary but is malformed"
    )]
    UnparseableShapeConstraint { section: String, operand: String, raw: String },

    /// A paged/gather operand is incoherent (V-FKC §10.14): `kind:
    /// paged_blocks` without `requires_ext: true` / `symbolic_extent:
    /// required`, or `block_table`/`context_lens` naming a non-existent
    /// `accept.inputs` role.
    #[error("FKC §10.14 (V-FKC): kernel `{section}` operand `{operand}` gather is incoherent: {reason}")]
    GatherIncoherent {
        section: String,
        operand: String,
        reason: String,
    },

    /// A gather-bearing operand reached registration before the FDX gather
    /// codes / `Capability::DlpackExtGather` landed (V-FKC §10.14; the
    /// `MxNotYetRegistrable` discipline for gather).
    #[error("FKC §3.9.1/§10.14: kernel `{section}` operand `{operand}` uses gather, not yet supported")]
    GatherNotYetSupported { section: String, operand: String },

    /// A dtype / quant family / granularity / ggml_dtype token is not a
    /// member of FDX's normative code table (V-FKC §10.16 drift-guard) — so
    /// FKC's token set stays a subset of FDX's.
    #[error(
        "FKC §10.16 (rule 16): kernel `{section}` token `{token}` (field `{field}`) is not in the \
         FDX normative code table"
    )]
    FdxTokenNotInTable {
        section: String,
        field: String,
        token: String,
    },

    /// An admissibility-affecting enum (`extent_kind`, `gather.kind`, quant
    /// `family`) carried an unknown value (V-FKC §10.5/§10.14/§10.15;
    /// meaning-bearing per §11.1 — a typed error, never a silent default).
    #[error(
        "FKC §11.1: kernel `{section}` field `{field}` has unknown meaning-bearing value `{value}`"
    )]
    UnknownAdmissibilityEnum {
        section: String,
        field: String,
        value: String,
    },

    /// The structured `blurb:` does not match the prose blurb when the prose
    /// is available (V-FKC §10.11). Distinct from [`Self::BlurbMismatch`]
    /// (which is reserved for the prose-vs-structured comparison wired with
    /// the prose-bearing parse).
    #[error(
        "FKC §10.11 (V-FKC-2): kernel `{section}` is missing a required blurb (must be a non-empty \
         one-line string)"
    )]
    MissingBlurb { section: String },
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
