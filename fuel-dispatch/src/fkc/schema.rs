//! `serde::Deserialize` structs mirroring the FKC structured schema
//! (§3.1 file anatomy + §3.3 per-kernel block + §3.2 tensor descriptors).
//!
//! Design decisions for this deserialization layer:
//!
//! - **Tokens are `String`.** Every dtype / family / granularity / ggml_dtype /
//!   role / determinism token deserializes as a `String` (or `Option<String>`).
//!   §3.8 mandates quoted scalars, and keeping these as strings is also what
//!   disarms the YAML-1.1 "Norway problem" (a `family: no` cannot silently
//!   become a bool when the target type is `String`). The string → typed
//!   conversion (dtype → `DType`, op_kind → `OpKind`, …) is the **lowering**
//!   step's job (a later slice), not this parse step's.
//! - **Cost / shape EXPRESSION fields stay `String`.** FKC has its own
//!   expression mini-parser (§4.4); YAML never tokenizes `"2 * m * n * k"`.
//!   They are carried verbatim here as `Option<String>`.
//! - **`return` is a reserved word** → `#[serde(rename = "return")]` on the
//!   field, exposed as `return_`.
//! - **Booleans are literal `true`/`false`** only (`audited`, `in_place`,
//!   `requires_ext`, the layout flags' coherence helpers carry their own
//!   string-valued tri-states so they are NOT bools here — see [`LayoutSpec`]).
//!
//! The structs are intentionally permissive (`#[serde(default)]`, `Option`,
//! and `deny_unknown_fields` deliberately NOT set so the additive-versioning
//! posture of FKC G7 holds — unknown forward-looking fields are ignored). The
//! required-field enforcement (V-FKC-1) is a validation concern, not a parse
//! concern.

use serde::Deserialize;

// ===========================================================================
// File-level front-matter (§3.1)
// ===========================================================================

/// The file-level `---`-fenced YAML front-matter (§3.1).
#[derive(Debug, Clone, Deserialize)]
pub struct FkcFrontMatter {
    /// Format version (e.g. `1`).
    pub fkc_version: u32,
    /// Provider identity block.
    pub provider: FkcProvider,
}

/// Provider identity (front-matter `provider:` mapping, §3.1).
#[derive(Debug, Clone, Deserialize)]
pub struct FkcProvider {
    /// Provider name (e.g. `fuel-cpu-backend`).
    pub name: String,
    /// Backend tag (e.g. `Cpu`) — maps to a `BackendId` at lower time.
    pub backend: String,
    /// The `BindingEntry.kernel_source` tag (e.g. `"portable-cpu"`).
    pub kernel_source: String,
    /// `link_registry` symbol path (e.g. `fuel_cpu_backend::fkc::ENTRY_POINTS`).
    /// Optional so a structure-only lint corpus can omit it.
    #[serde(default)]
    pub link_registry: Option<String>,
    /// Provider build id folded into `kernel_revision_hash` (e.g. `"git:f41137b4"`).
    #[serde(default)]
    pub revision_base: Option<String>,
}

// ===========================================================================
// Top-level parsed file
// ===========================================================================

/// A fully parsed FKC file: front-matter + one record per `## ` kernel section.
#[derive(Debug, Clone)]
pub struct FkcFile {
    /// File-level front-matter.
    pub front_matter: FkcFrontMatter,
    /// One per `## ` section, in source order.
    pub kernels: Vec<FkcKernel>,
}

// ===========================================================================
// Per-kernel structured block (§3.3)
// ===========================================================================

/// The serde default for [`FkcKernel::registrable`] — `true` so a section
/// that omits the field registers exactly as before (§3.10, additive §11).
fn default_registrable() -> bool {
    true
}

/// The per-kernel ` ```fkc ` structured contract (§3.3).
///
/// Exactly one of `op_kind` / `fused_op` is present (a primitive vs fused
/// contract) **unless** the section is describe-only (`registrable: false`,
/// §3.10), in which case `op_kind`/`fused_op` need not resolve to a real
/// dispatch target; the structural check is a validation step, not a parse
/// step.
#[derive(Debug, Clone, Deserialize)]
pub struct FkcKernel {
    // ----- identity -----
    /// Unique-within-file diagnostic kernel name.
    pub kernel: String,
    /// Describe-only marker (§3.10). When `false` this section is
    /// **documentation-only**: it is NOT registered and is NOT required to
    /// name a real dispatch `op_kind`/`fused_op` (the op may be `~` or a
    /// descriptive non-dispatch token). Its descriptive fields (dtypes,
    /// layout, quant) are STILL validated as docs. Defaults to `true` so
    /// every existing contract registers exactly as before (additive
    /// versioning, §11).
    #[serde(default = "default_registrable")]
    pub registrable: bool,
    /// The Fuel `OpKind` this kernel implements (primitive contract).
    #[serde(default)]
    pub op_kind: Option<String>,
    /// OR a `FusedOpId` name (fused contract). Exactly one of these two.
    #[serde(default)]
    pub fused_op: Option<String>,
    /// One-line blurb; MUST equal the prose blurb (§10.11, checked later).
    #[serde(default)]
    pub blurb: Option<String>,
    /// Backend (inherited from front-matter unless overridden).
    #[serde(default)]
    pub backend: Option<String>,
    /// `kernel_source` tag (inherited unless overridden).
    #[serde(default)]
    pub kernel_source: Option<String>,
    /// Symbolic ref into the provider link registry → `KernelRef` (P9, §12.6).
    #[serde(default)]
    pub entry_point: Option<String>,
    /// Revision hash hex, OR `"auto"` to derive (§4.7). String per §3.8.
    #[serde(default)]
    pub kernel_revision_hash: Option<String>,
    /// OPAQUE specialization-variant tag (Baracuda `variant:`, e.g.
    /// `"splitk_partial"`). Retained verbatim so a contract-declared variant
    /// survives lowering and can ride into a telemetry record as an opaque
    /// annotation; the entry point remains the true kernel identity. Purely a
    /// string on both sides — Fuel never parses or validates it beyond being a
    /// string (additive §11, no `deny_unknown_fields`). Distinct from
    /// [`OpParamsSchema::variant`], which names the `OpParams` Rust variant.
    #[serde(default)]
    pub variant: Option<String>,

    // ----- accept / return -----
    /// Accept contract (§3.6).
    #[serde(default)]
    pub accept: Option<AcceptBlock>,
    /// Return contract (§3.6). `return` is a reserved word.
    #[serde(rename = "return", default)]
    pub return_: Option<ReturnBlock>,

    // ----- capability + cost + precision + determinism (§4) -----
    /// Capability block (§4.1–§4.6).
    #[serde(default)]
    pub caps: Option<CapsBlock>,
    /// Cost block (§4.4).
    #[serde(default)]
    pub cost: Option<CostBlock>,
    /// Precision block (§4.8) → `PrecisionGuarantee`.
    #[serde(default)]
    pub precision: Option<PrecisionBlock>,
    /// Determinism (§4.9): `bitwise` | `same_hardware_bitwise` | `nondeterministic`.
    #[serde(default)]
    pub determinism: Option<String>,
}

// ===========================================================================
// Accept (§3.6 / §3.7)
// ===========================================================================

/// The `accept:` block.
#[derive(Debug, Clone, Deserialize)]
pub struct AcceptBlock {
    /// Ordered list of input tensor descriptors (§3.2).
    #[serde(default)]
    pub inputs: Vec<TensorDesc>,
    /// Op-param schema (§3.7).
    #[serde(default)]
    pub op_params: Option<OpParamsSchema>,
}

/// Op-param schema (§3.7): names the params variant + per-field constraints.
#[derive(Debug, Clone, Deserialize)]
pub struct OpParamsSchema {
    /// The `OpParams` (primitive) / `FusedOpParams` (fused) variant name.
    #[serde(default)]
    pub variant: Option<String>,
    /// Per-field constraint specs (kept loose; the field map is free-form so
    /// the importer can validate constraints in a later step). Keys are field
    /// names; values carry `kind` / `constraint` / `note`.
    #[serde(default)]
    pub fields: std::collections::BTreeMap<String, OpParamFieldSpec>,
}

/// A single op-param field spec (`{ kind, constraint?, note? }`, §3.7).
#[derive(Debug, Clone, Deserialize)]
pub struct OpParamFieldSpec {
    /// The field's Rust type token (e.g. `usize`, `QuantType`, `DynScalar`).
    #[serde(default)]
    pub kind: Option<String>,
    /// A free-text constraint expression (validated by FKC's own parser later).
    #[serde(default)]
    pub constraint: Option<String>,
    /// Free-text note.
    #[serde(default)]
    pub note: Option<String>,
}

// ===========================================================================
// Return (§5)
// ===========================================================================

/// The `return:` block.
#[derive(Debug, Clone, Deserialize)]
pub struct ReturnBlock {
    /// Ordered list of output descriptors + return rules (§5).
    #[serde(default)]
    pub outputs: Vec<OutputDesc>,
    /// OR a list of bundle slot specs for multi-output ops (§5.5). Kept opaque
    /// for this slice (the rank/name validation is a later step).
    #[serde(default)]
    pub bundle: Option<serde_yml::Value>,
}

/// An output descriptor + return rules (§5.1–§5.4). Rule fields are `String`
/// expressions parsed by FKC later, not YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputDesc {
    /// Output role name (§5.5).
    #[serde(default)]
    pub name: Option<String>,
    /// Dtype rule, e.g. `passthrough(lhs)` / `fixed(F32)` (§5.1).
    #[serde(default)]
    pub dtype_rule: Option<String>,
    /// Shape rule, e.g. `same_as(lhs)` / `from_params(batch, m, n)` (§5.2).
    #[serde(default)]
    pub shape_rule: Option<String>,
    /// Layout guarantee, e.g. `contiguous` / `preallocated` (§5.3).
    #[serde(default)]
    pub layout_guarantee: Option<String>,
    /// Aliasing, e.g. `none` (§5.4).
    #[serde(default)]
    pub aliasing: Option<String>,
}

// ===========================================================================
// Tensor descriptor (§3.2)
// ===========================================================================

/// A tensor descriptor (operand or, with return rules, an output; §3.2).
#[derive(Debug, Clone, Deserialize)]
pub struct TensorDesc {
    /// Operand role name (diagnostic + maps to an FDX view name, §5.5).
    #[serde(default)]
    pub name: Option<String>,
    /// `true` ⇒ this INPUT operand is OPTIONAL (§3.4 optional-operand support):
    /// production registers the op BOTH with and without it (e.g. conv's
    /// `bias`). The importer's key-builder fans each dtype variant into two
    /// keys — one omitting this operand, one including it — when set. Only the
    /// LAST input may be optional. Defaults to `false` (a required operand);
    /// `#[serde(default)]` so every existing contract that omits the field
    /// keeps building exactly one key per dtype variant (additive §11).
    #[serde(default)]
    pub optional: bool,
    /// Accepted DLPack dtype names (Fuel `DType` names; FDX §6.1). Strings;
    /// the string → `DType` conversion is a lowering concern.
    #[serde(default)]
    pub dtypes: Vec<String>,
    /// Optional shorthand `int|uint|float|any` that expands at lower time (§3.4).
    #[serde(default)]
    pub dtype_class: Option<String>,
    /// Layout capability flags (§4.1).
    #[serde(default)]
    pub layout: Option<LayoutSpec>,
    /// Rank: exact int, `"any"`, or a range `"2..=4"`. Kept as a raw scalar so
    /// `2` and `"any"` both parse (the FKC parser interprets it later).
    #[serde(default)]
    pub rank: Option<serde_yml::Value>,
    /// Free-predicate shape constraint (§3.5). Carried verbatim.
    #[serde(default)]
    pub shape_constraint: Option<String>,
    /// DLPack-extension (FDX) requirements (§3.2 `fdx:`).
    #[serde(default)]
    pub fdx: Option<FdxSpec>,
    /// Placement device (inherited from front-matter unless overridden).
    #[serde(default)]
    pub device: Option<String>,
    /// FDX substrate class.
    #[serde(default)]
    pub substrate: Option<String>,
}

/// The five-flag layout capability set (§4.1). Each flag is a string tri-state
/// (`required`/`accepted`/`n/a` for `contiguous`; `accepted`/`rejected` for the
/// rest) carried verbatim — projection onto today's `KernelCaps.strided_input`
/// is a lowering concern (§6).
#[derive(Debug, Clone, Deserialize)]
pub struct LayoutSpec {
    /// `required` | `accepted` | `n/a`.
    #[serde(default)]
    pub contiguous: Option<String>,
    /// `accepted` | `rejected`.
    #[serde(default)]
    pub strided: Option<String>,
    /// `accepted` | `rejected`.
    #[serde(default)]
    pub broadcast_stride0: Option<String>,
    /// `accepted` | `rejected`.
    #[serde(default)]
    pub start_offset: Option<String>,
    /// `accepted` | `rejected` — NEGATIVE strides (§4.1.1).
    #[serde(default)]
    pub reverse_strides: Option<String>,
    /// Optional per-operand override of `caps.awkward_layout_strategy` (§4.3.1).
    #[serde(default)]
    pub awkward_layout_strategy: Option<String>,
}

/// DLPack-extension (FDX) requirements on an operand (§3.2 `fdx:`).
#[derive(Debug, Clone, Deserialize)]
pub struct FdxSpec {
    /// `true` ⇒ this operand's meaning needs an FDX sidecar.
    #[serde(default)]
    pub requires_ext: Option<bool>,
    /// Quant facts (§3.2 `fdx.quant:`).
    #[serde(default)]
    pub quant: Option<QuantSpec>,
    /// Sub-byte logical_dtype code when the base carries opaque uint8 (FDX §6.1).
    #[serde(default)]
    pub sub_byte: Option<String>,
    /// `rejected`|`tolerated`|`required` symbolic-extent tolerance (§4.5).
    #[serde(default)]
    pub symbolic_extent: Option<String>,
    /// `rejected`|`scalar`|`range`|`affine` (§3.9.2).
    #[serde(default)]
    pub extent_kind: Option<String>,
    /// Paged / indexed-residency (gather) operand spec (§3.9.1).
    #[serde(default)]
    pub gather: Option<GatherSpec>,
}

/// Quant facts (§3.2 `fdx.quant:`). All tokens are `String` (disarms Norway;
/// the typed conversion is a lowering concern).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantSpec {
    /// FDXQuant.family symbol: `none|GGML_BLOCK|MX|AFFINE_INT|AFFINE_FLOAT|AFFINE_BLOCK` (FDX §6.2).
    #[serde(default)]
    pub family: Option<String>,
    /// `GgmlDType` variant NAME when family=GGML_BLOCK (e.g. `Q4_0`, `Q4K`); §3.4.
    #[serde(default)]
    pub ggml_dtype: Option<String>,
    /// FDXScaleGranularity symbol: `PerTensor|PerToken|PerChannel|PerBlock` (FDX §6.2).
    #[serde(default)]
    pub granularity: Option<String>,
    /// Per-output-row per-block grain (e.g. `[block_size]`) for AFFINE_BLOCK (§3.9.3 / NF4).
    #[serde(default)]
    pub block_shape: Option<serde_yml::Value>,
    /// `activation`|`weight` (FDX ScalePair role).
    #[serde(default)]
    pub role: Option<String>,
    /// Role of the SEPARATE-GRAPH-INPUT scale operand when the ABI takes the
    /// scale as its own input (§3.9.3). Mutually exclusive with a sidecar
    /// `scale_buffer` (the `ScaleDoubleDeclared` check, a later step).
    #[serde(default)]
    pub scale_operand: Option<String>,
}

/// Paged / indexed-residency (gather) operand spec (§3.9.1).
#[derive(Debug, Clone, Deserialize)]
pub struct GatherSpec {
    /// `~` | `paged_blocks` — FDX FDXIndexedResidency.kind symbol.
    #[serde(default)]
    pub kind: Option<String>,
    /// Role of the block-table operand (a separate accept.input).
    #[serde(default)]
    pub block_table: Option<String>,
    /// Role of the per-sequence live-length operand (a separate accept.input).
    #[serde(default)]
    pub context_lens: Option<String>,
}

// ===========================================================================
// caps / cost / precision (§4)
// ===========================================================================

/// Capability block (§4.1–§4.6).
#[derive(Debug, Clone, Deserialize)]
pub struct CapsBlock {
    /// `requires_contiguous` | `handles_strided` | `contiguize_internally` (§4.3).
    #[serde(default)]
    pub awkward_layout_strategy: Option<String>,
    /// Declared fast-path predicates (§4.2). Kept opaque for this slice.
    #[serde(default)]
    pub fast_paths: Option<serde_yml::Value>,
    /// `in_place` (§4.6).
    #[serde(default)]
    pub in_place: Option<bool>,
    /// Mirrors `BackendCapabilities.required_alignment`.
    #[serde(default)]
    pub alignment_bytes: Option<u64>,
    /// Access granularity in bits.
    #[serde(default)]
    pub access_granularity_bits: Option<u64>,
}

/// Cost block (§4.4). Coefficient EXPRESSION fields stay `String` — FKC's own
/// parser reads them later, NOT YAML. `~` (YAML null) deserializes to `None`.
#[derive(Debug, Clone, Deserialize)]
pub struct CostBlock {
    /// `declared` | `judge_measured` — REQUIRED, both first-class (§4.4).
    #[serde(default)]
    pub provenance: Option<String>,
    /// Coarse relative cost class bucket (§4.4).
    #[serde(default)]
    pub class: Option<String>,
    /// Symbolic FLOPs expression over shape/param symbols (§4.4).
    #[serde(default)]
    pub flops: Option<String>,
    /// Symbolic bytes-moved expression (§4.4).
    #[serde(default)]
    pub bytes_moved: Option<String>,
    /// Launch overhead. May be a literal number or `~`; kept as a raw scalar.
    #[serde(default)]
    pub overhead_ns: Option<serde_yml::Value>,
    /// Per-tier memory footprint (§4.4) — [consumer-ahead] beyond device_bytes.
    #[serde(default)]
    pub memory: Option<CostMemory>,
}

/// Per-tier memory footprint (§4.4). Each tier is an expression `String` (or a
/// literal `0`); kept as a raw scalar so `0` and `"n * 4"` both parse.
#[derive(Debug, Clone, Deserialize)]
pub struct CostMemory {
    /// Output device alloc (executor pre-allocates).
    #[serde(default)]
    pub device_bytes: Option<serde_yml::Value>,
    /// Host-tier bytes.
    #[serde(default)]
    pub host_bytes: Option<serde_yml::Value>,
    /// Disk-tier bytes.
    #[serde(default)]
    pub disk_bytes: Option<serde_yml::Value>,
}

/// Precision block (§4.8) → `PrecisionGuarantee`. Bound fields are raw scalars
/// (`0`, a float, or `~`); the typed conversion is a lowering concern.
#[derive(Debug, Clone, Deserialize)]
pub struct PrecisionBlock {
    /// Whether bit-stable on the same hardware.
    #[serde(default)]
    pub bit_stable_on_same_hardware: Option<bool>,
    /// Max ULP error (`~` ⇒ none).
    #[serde(default)]
    pub max_ulp: Option<serde_yml::Value>,
    /// Max relative error (`~` ⇒ none).
    #[serde(default)]
    pub max_relative: Option<serde_yml::Value>,
    /// Max absolute error (`~` ⇒ none).
    #[serde(default)]
    pub max_absolute: Option<serde_yml::Value>,
    /// `false` ⇒ UNAUDITED; `true` + all-null ⇒ none(reason) audited-no-bound.
    #[serde(default)]
    pub audited: Option<bool>,
    /// Free-text precision notes.
    #[serde(default)]
    pub notes: Option<String>,
}
