# Kernel-Contract (FKC) adoption plan — moving Fuel's dispatch layers onto importable contracts

**Status:** PLAN (2026-06-17). WIP lands on branch `feat/kernel-contracts-dlpack` (same branch the
two specs live on); nothing in this plan touches `main` until the rollout gate in §11.
**Goal:** make *importing a provider's FKC contract file(s) auto-register all of that provider's
kernels* onto Fuel's existing dispatch surface — the `KernelBindingTable` (primitive ops) and the
`FusedKernelRegistry` (`Op::Fused` ops) — with **zero hand-written registration glue**, validated at
import time, never panicking, leaving the optimizer/planner the sole decision-maker.

**Authoritative inputs (read before touching this plan):**
- FKC spec: `docs/specs/kernel-contract-format.md` (the advertisement axis — this is what we import).
- FDX spec: `docs/specs/dlpack-extension.md` (the tensor axis — the vocabulary FKC tensor descriptors
  reference; the importer references FDX codes, it does not re-list them).
- Constraints digest: `docs/specs/_research/architecture-constraints.md`.
- As-built dispatch types: `fuel-dispatch/src/kernel.rs` (`KernelRef`, `KernelCaps`,
  `KernelBindingTable`, `BindingEntry`, `CostFn`, `OpParams`, `unknown_cost`),
  `fuel-dispatch/src/fused.rs` (`BackendImpl`, `CostEstimate`, `PrecisionGuarantee`,
  `KernelRevisionHash`, `FusedKernelRegistry`, the fused-op cost-fn signature).
- Graph-side fused metadata: `fuel-graph/src/registry.rs` (`FusedOpId`, `FusedOps::*` id constants,
  `FusedOpParams`, `FusedOpEntry { shape_rule, dtype_rule, output_views, … }`).
- Per-backend kernel inventories already drafted (the seed for the contract files):
  `docs/kernel-contracts/_inventory/{cpu,vulkan,conv-attn,fused,quantized,reference,metal,mkl-aocl,dispatch}.md`.

**Resolved decisions this plan applies consistently** (from the design pass — do not relitigate):
DLPack floor v1.0 versioned / v1.2+ explicit strides / validate vs v1.3; **negative strides are
first-class** (FDX describes signed strides; OOB via the signed touched-range; per-kernel
`reverse_strides` capability; planner normalizes only for incapable consumers, never universally);
interior-axis live-prefix = materialized copy w/ `IS_COPIED`; **`register_full_with_source` must
become `Result`** (never-panic prerequisite, §3); CostFn capacity-only for v1 (SymEnv + per-tier
memory are consumer-ahead); sub-byte/quant logical shape carried explicitly; the **quant scale
single-place rule** (separate-graph-input scale ⇒ FKC `accept.inputs` operand, NOT also an FDX
`scale_buffer`); inline `[;6]` shape/stride mirroring `DimVec`, explicit error beyond rank 6;
sidecar sub-structs frozen-size + size-assertion tests; `kernel_revision_hash` over a canonicalized
parse, shared with FDX `name_hash`, bundle slot names in a side-table; cross-runtime transport via
explicit sidecar param wherever we control the signature, `manager_ctx` only as opportunistic
fallback on the pure `__dlpack__` path; little-endian v1; FDX owns the numeric code table, FKC uses
string names.

---

## 0. Outcome, scope, and the shape of "import = registration"

### 0.1 The end state (what "done" looks like)

A provider ships one or more markdown FKC files. Today's hand-written Rust registration functions
(`register_cpu_kernels`, `register_aocl_cpu_kernels`, `register_vulkan_kernels`,
`register_default_fused_kernels`, the baracuda dispatch wiring) are **replaced or backed** by a call
of the form:

```rust
// New public API in fuel-dispatch::fkc.
let provider = fkc::import_bundle(CPU_CONTRACTS_MD)?;      // single bundle file
provider.register_into(&mut table, &mut fused_registry)?;  // populates BOTH registries
```

or, for a multi-file provider:

```rust
let provider = fkc::import_glob("fuel-vulkan-kernels/contracts/*.fkc.md")?;
provider.register_into(&mut table, &mut fused_registry)?;
```

`import_*` does **all parsing + build-time validation** and returns a `Result`; `register_into`
does the actual `KernelBindingTable::register_full_with_source(...)` /
`FusedKernelRegistry::register(...)` calls, also `Result`. Both paths are `Result` end-to-end — no
panic, no `try_*` sibling (digest §3, constitution).

The link from a contract's `entry_point: <symbol-id>` string to a concrete `KernelRef`/`BackendImpl`
function pointer is resolved through a **provider link registry** (FKC §12.6): a
`&'static [(&'static str, KernelRef)]` (and the fused analog) the provider crate exports. The
importer never fabricates a function pointer; it looks up the symbol and errors
(`UnknownEntryPoint`) if absent. This keeps P9 (serializable, no pointers in the contract) honest.

### 0.2 What stays exactly as-is

- `KernelRef` / `OpParams` / `FusedOpParams` ABI — unchanged. FKC is a *serialization + authoring
  surface* over the types that already exist (FKC §0). No kernel signature changes.
- The two-registry split (`KernelBindingTable` for primitives, `FusedKernelRegistry` for fused) —
  unchanged. FKC `op_kind` contracts map to the former, `fused_op` contracts to the latter (FKC §3.7,
  §12). The importer routes on which of `op_kind`/`fused_op` a contract declares.
- The graph-side `FusedOpEntry` (shape_rule/dtype_rule/decompose/backward) — **NOT** imported from
  FKC. Those are graph-side metadata authored in Rust (`fuel-graph/src/registry.rs`); FKC's
  `shape_rule`/`dtype_rule` strings are **validated against** the registered entry (§5, §10.7), not
  used to generate it. FKC adopts the *kernel/dispatch* layers, not the graph-rewrite layer.
- The executor hot path, the route picker, the cost-vector ranker — unchanged. They read the same
  `BindingEntry`/`BackendImpl` fields; FKC just fills those fields from a file instead of from a
  hand-written `register_*` call.

### 0.3 The one CONSTITUTION-CONFLICT this plan must resolve first

`KernelBindingTable::register_full_with_source` **panics** on a duplicate `KernelRef`
(`fuel-dispatch/src/kernel.rs:910-918`). The importer cannot be allowed to drive a panicking
registration path — duplicate detection on an *imported* contract must be a typed `Result` error,
not a process abort. §3 is the prerequisite step that converts this to `Result`. This is the only
existing-type change the adoption strictly requires; everything else is additive.

---

## 1. New module: `fuel-dispatch/src/fkc/` (the importer)

All new code lands under a new module tree in the crate that already owns both registries
(`fuel-dispatch` — see `lib.rs` `pub mod` list). No new crate; no new path dep. The module is gated
behind a new default-off cargo feature `fkc` until §11's flip, so `main` keeps building while the
importer is WIP.

```
fuel-dispatch/src/fkc/
  mod.rs          // public API: import_bundle / import_glob / ImportedProvider / register_into
  parse.rs        // markdown + fenced ```fkc YAML block extraction (§3.1, §3.8)
  schema.rs       // serde structs mirroring FKC §3.3 (RawContract, RawTensorDescriptor, …)
  lower.rs        // RawContract -> resolved (OpKind|FusedOpId, KernelDTypes, caps, cost, precision)
  caps_map.rs     // FKC five-flag layout set -> today's KernelCaps (the lossy projection, §6)
  cost_expr.rs    // the cost-expression mini-parser/evaluator (FKC §4.4, capacity-eval for v1)
  precision.rs    // FKC precision block -> PrecisionGuarantee (incl. UNAUDITED / none(reason))
  link.rs         // entry_point symbol -> KernelRef / fused BackendImpl, via provider link registry
  revhash.rs      // kernel_revision_hash over canonicalized parse (shared with FDX name_hash, §8)
  validate.rs     // build-time validators V-FKC-* (§10) returning Result
  error.rs        // FkcError enum (thiserror) — every failure mode is a typed variant
```

### 1.1 Dependencies

Add to `fuel-dispatch/Cargo.toml` under the `fkc` feature:
- `serde` + `serde_yaml` (or `serde_yml`) for the fenced-block YAML in the restricted core-schema mode
  FKC §3.8 mandates (quoted-scalar, no anchors/aliases/merge-keys, tabs-are-error). The parser layer
  wraps the YAML lib and enforces §3.8's restrictions *before* deserializing.
- `pulldown-cmark` (or a minimal hand-rolled section/fence scanner) to split the markdown into
  `## `-delimited kernel sections and extract the single ` ```fkc ` fence per section (§3.1).
- A stable hash for `revhash.rs` — reuse whatever FDX `name_hash` uses (decision: shared stable hash;
  pick a non-SipHash, endianness-stable function, e.g. FNV-1a or xxhash with a fixed seed, so the
  hash round-trips across processes/persistence per digest §7). Document the choice once in
  `revhash.rs` and assert it in a test against a frozen fixture.

### 1.2 Public API surface (signatures)

```rust
// fuel-dispatch/src/fkc/mod.rs

/// A parsed + validated provider bundle. Holds the resolved per-kernel
/// records ready to register; construction already ran all build-time
/// validators (§10). `register_into` only does the table inserts.
pub struct ImportedProvider {
    pub name: String,                 // provider.name (front-matter)
    pub backend: BackendId,           // provider.backend
    pub kernel_source: &'static str,  // provider.kernel_source (interned; see §1.3)
    primitives: Vec<ResolvedPrimitive>,   // op_kind contracts
    fused:      Vec<ResolvedFused>,        // fused_op contracts
}

/// Parse + validate a single bundle markdown file's bytes. Pure; no I/O
/// of its own beyond what the caller hands in (so tests pass &str).
pub fn import_bundle_str(
    src: &str,
    link: &dyn LinkRegistry,
) -> Result<ImportedProvider, FkcError>;

/// Convenience: read a file path, then import_bundle_str.
pub fn import_bundle(
    path: impl AsRef<std::path::Path>,
    link: &dyn LinkRegistry,
) -> Result<ImportedProvider, FkcError>;

/// Glob multiple per-kernel files into one provider. Each file is one
/// `## ` section (FKC §9.2); front-matter must agree across files
/// (same provider/backend/kernel_source) or `ProviderMismatch`.
pub fn import_glob(
    pattern: &str,
    link: &dyn LinkRegistry,
) -> Result<ImportedProvider, FkcError>;

impl ImportedProvider {
    /// Register every primitive contract into `table` and every fused
    /// contract into `fused`. Result-returning: duplicate KernelRef,
    /// missing fused-op metadata entry, etc. all surface as FkcError.
    pub fn register_into(
        &self,
        table: &mut KernelBindingTable,
        fused: &mut FusedKernelRegistry,
    ) -> Result<(), FkcError>;
}

/// entry_point symbol -> concrete function pointer. Implemented by each
/// provider crate (a thin wrapper over its `&'static [(&str, KernelRef)]`).
pub trait LinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef>;
    fn resolve_fused(&self, symbol: &str) -> Option<KernelRef>;
}
```

`FkcError` is `#[derive(thiserror::Error)]` with one variant per failure mode named in §10
(`YamlTabIndent`, `YamlAnchorDisallowed`, `BadScalarType`, `UnknownOpKind`, `UnknownFusedOp`,
`UnknownEntryPoint`, `DuplicateKernelRef`, `QuantIncoherent`, `ScaleDoubleDeclared`,
`RankExceedsSix`, `BlurbMismatch`, `PlaceholderPrecision`, `PlaceholderCost`, `CostExprParse`,
`OpParamsVariantMismatch`, `MxNotYetRegistrable`, `GatherNotYetSupported`, `ProviderMismatch`,
`ShapeRuleMismatch`, …). Every variant carries enough context (kernel name, file, line/col where the
YAML layer can provide it) to be actionable.

### 1.3 `kernel_source` lifetime

`BindingEntry.kernel_source` and `BackendImpl` registrations use `&'static str`. A contract's
`kernel_source` is read from a file at runtime, so it is not `'static`. Resolution: intern it.
`ImportedProvider` holds the owned `String`; `register_into` interns via a small
`OnceLock<Mutex<HashSet<&'static str>>>` string-interner (`Box::leak` of first sighting) and passes
the `&'static` handle to the registration calls. This is a process-lifetime leak bounded by the
number of distinct provider source tags (a handful) — acceptable and documented. (Alternative if a
leak is unwanted: widen `kernel_source` to `Cow<'static, str>` in a follow-up; out of scope here.)

---

## 2. Mapping a contract onto the as-built dispatch types (the heart)

This is the concrete `RawContract -> registry call` mapping. Two targets, per FKC §12.

### 2.1 Primitive (`op_kind`) contract -> `KernelBindingTable`

| FKC field | As-built target | How |
|-----------|-----------------|-----|
| `op_kind` | `OpKind` (key.0) | string -> `OpKind` via an explicit `match` table in `lower.rs` (`UnknownOpKind` on miss). NOT `FromStr`-by-discriminant — an exhaustive match so a new `OpKind` forces a compile error to extend the table. |
| `accept.inputs[*].dtypes` + outputs | `KernelDTypes` (key.1 = `SmallVec<[DType; 8]>`) | per-operand dtype list, inputs in order then outputs (matches `kernel.rs` key shape). dtype strings -> `DType` via explicit match (FKC §3.4); `dtype_class` shorthands expand here. Quant facts enrich the operand slot for the key (FKC §3.2, §12.1). |
| `backend` | `BackendId` (key.2) | explicit match. |
| `entry_point` | `KernelRef` (`BindingEntry.kernel`) | `LinkRegistry::resolve_primitive` (`UnknownEntryPoint` on miss). |
| `caps.*` + `layout.*` | `KernelCaps` (`BindingEntry.caps`) | the five-flag layout set projected onto today's single `strided_input` bool — see §6. |
| `cost.*` | `CostFn` (`BindingEntry.cost`) | compiled to a `CostFn` — see §2.3. |
| `precision.*` | `PrecisionGuarantee` (`BindingEntry.precision`) | direct field map; `audited: false` + all-null -> `UNAUDITED`; `audited: true` + all-null -> `none(notes)`; bounds present -> populated struct. |
| `kernel_source` | `BindingEntry.kernel_source` | interned `'static` (§1.3). |
| `kernel_revision_hash` | (consumer-ahead) | `KernelRevisionHash` lives on `BackendImpl` (fused) today, not on `BindingEntry` (primitive). For primitives the hash is computed and stored on the `ResolvedPrimitive` for the persistence layer to read later; it is NOT dropped, but there is no `BindingEntry` slot for it yet. Note this as a forward-looking field (digest §7) and add a `BindingEntry.revision` slot in a follow-up if/when the primitive persistence path needs it. |

The actual call is exactly today's:
```rust
table.register_full_with_source(op, &dtypes, backend, kernel, caps, precision, cost, source)?;
//                                                                                          ^ now Result (§3)
```

### 2.2 Fused (`fused_op`) contract -> `FusedKernelRegistry`

| FKC field | As-built target | How |
|-----------|-----------------|-----|
| `fused_op` | `FusedOpId` | string -> `FusedOpId` via a name table built from `FusedOps::*` constants (FKC §3.7; `UnknownFusedOp` on miss). The name table is generated from `fuel-graph`'s `default_registry()` entries' `name` field so it cannot drift (one source of truth). |
| `accept.inputs[*].dtypes` + outputs | `BackendImpl.dtypes: &'static [DType]` | resolved dtype slice, interned `'static` like `kernel_source`. |
| `backend` | `BackendId` (the `(id, backend)` registry key) | explicit match. |
| `entry_point` | `BackendImpl.kernel` | `LinkRegistry::resolve_fused`. |
| `cost.*` | `BackendImpl.cost: fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate` | compiled to the **fused** cost-fn shape — note `&FusedOpParams` and NO `&[DType]` (the BLOCKER fix in FKC §4.4 / resolved-critique). The cost-expr compiler emits two different fn signatures depending on primitive-vs-fused; see §2.3. |
| `precision.*` | `BackendImpl.precision` | same mapping as primitives. |
| `caps.*` + `layout.*` | `BackendImpl.caps` (`KernelCaps`) | same §6 projection. |
| `kernel_revision_hash` | `BackendImpl.revision: KernelRevisionHash` | from `revhash.rs`; `"auto"` derives from `entry_point` + `revision_base` (FKC §4.7). This slot exists today — the fused path is the one that already round-trips a revision hash. |
| `op_params.variant` | (validated, not stored) | checked against the `FusedOpParams` namespace (§10.7); not a registry field. |
| `return.outputs` / `bundle` | (validated against `FusedOpEntry`) | the bundle slot specs are validated against the graph-side `output_views` (§5.5); FKC carries rank ≤ 6 + slot-name side-table. |

The actual call is exactly today's:
```rust
fused.register(id, backend, BackendImpl { kernel, dtypes, cost, precision, caps, revision });
```
(`FusedKernelRegistry::register` already appends and never panics — no change needed there.)

### 2.3 The cost-expression compiler (`cost_expr.rs`) — two compile targets

FKC §4.4 / §5 / §12.3 (the BLOCKER fix): a cost expression compiles to a `fn`, but **primitive and
fused need different signatures**. The compiler therefore has two entry points:

- `compile_primitive_cost(&CostBlock) -> CostFn`
  (`fn(&[Shape], &[DType], &OpParams, &BackendCapabilities) -> CostEstimate`).
- `compile_fused_cost(&CostBlock) -> fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate`.

Because `CostFn`/`BackendImpl.cost` are bare `fn` pointers (not closures), and a contract's cost
coefficients are runtime data, the compiler **cannot** emit a monomorphized `fn` per contract. Two
viable strategies — pick **(A)** for v1:

- **(A) Interpreter-backed `fn` with a side table (v1).** The compiler parses each cost expression
  (`flops`, `bytes_moved`, `overhead_ns`, `memory.device_bytes`) into a small RPN/AST and stores it
  in a process-wide `OnceLock<Vec<CompiledCostExpr>>` keyed by an index baked into a generated
  trampoline. Since we cannot close over the index in a bare `fn`, v1 instead registers a **fixed
  family of trampolines** that look the expression up by the `(OpKind|FusedOpId, dtypes, backend)`
  key from a global `OnceLock` cost-table the importer populates. i.e. the `CostFn` we store is a
  single generic `fkc_cost_primitive` that, when called, re-derives its key from its inputs and
  evaluates the stored AST. This keeps the `fn`-pointer ABI intact. The eval is capacity-only for v1
  (`Extent::bound()`), which is exactly what today's `CostFn` already receives (shapes carry capacity
  via `dims()`), so no signature change (FKC §4.4 [consumer-ahead] note, digest §5).
- (B) Generate Rust source for each contract at build time (proc-macro/build.rs). Rejected for v1:
  heavier, and contracts are runtime-imported, not compile-time.

The cost-expr vocabulary the parser accepts: shape symbols (`m`, `n`, `k`, `n` = element count,
`dtype_bytes`), the §3.5 shape predicates, integer/float literals, `+ - * / %`, `==`, and the
`fast_paths` multipliers. `CostExprParse` on anything outside the grammar. The per-tier `memory`
block beyond `device_bytes` and any `SymEnv`-resolved live-extent term are **parsed and retained but
not evaluated** in v1 (size-prefixed tail, FKC §11) — they degrade to today's three
`CostEstimate` scalars (`flops`, `bytes_moved`, `kernel_overhead_ns`) plus `device_bytes` informing
nothing today. Document this degradation explicitly at the `CostEstimate` construction site.

> **Sentinel handling.** A contract with no cost block (or `class` only) maps to `unknown_cost`
> (the existing sentinel, `kernel.rs:724`) so the existing `fill_unset_*_cost` passes can still fill
> it — but the CI lint (§10.9) flags a non-reference contract that ships `unknown_cost` as
> `PlaceholderCost`, matching the existing "every Vulkan registration has a real CostFn" lint
> (digest §3).

---

## 3. PREREQUISITE — `register_full_with_source` panic -> `Result` (CONSTITUTION-CONFLICT)

This is **step 1 of the implementation** (born-red, lands before the importer). It is the only change
to an existing type the adoption strictly requires, and it is a standing constitution violation
regardless of FKC (never-panic, digest §3).

### 3.1 The change

`fuel-dispatch/src/kernel.rs:895-920`: change `register_full_with_source` to return
`Result<(), Error>` and replace the `panic!` on duplicate `KernelRef` with
`return Err(Error::DuplicateKernelRef { op, dtypes, backend }.bt())`. Add the `DuplicateKernelRef`
variant to `fuel-core-types/src/error.rs` (it already hosts `NoBackendForOp` with the same shape —
mirror it). Thread the `Result` through the thin wrappers (`register`, `register_with_caps`,
`register_with_precision`, `register_with_caps_and_precision`, `register_full`) — each now returns
`Result` and `?`-propagates.

> Duplicate detection is at the **resolved `KernelRef` function-pointer** level (FKC §10.10 MAJOR
> fix), which is exactly what the existing pointer-equality check at `kernel.rs:910` already does —
> two distinct `entry_point` strings that resolve to the same `fn` (an alias / shared generic) are
> caught here. The importer surfaces this as `FkcError::DuplicateKernelRef` (wrapping the dispatch
> `Error`).

### 3.2 Blast radius (callers to update — all in-repo)

From the grep sweep, the registration callers are:
`fuel-dispatch/src/{dispatch.rs, fused.rs, baracuda_dispatch.rs, vulkan_dispatch.rs}`,
`fuel-aocl-cpu-backend/src/binding_table.rs`, `fuel-mkl-cpu-backend/src/binding_table.rs`,
`fuel-core/src/pipelined_bridge.rs`, and the test files. The bulk callers are
`register_cpu_kernels` (`dispatch.rs:3880`) and the GPU `register_*` functions, which call
`register*` hundreds of times in a sequence.

Migration shape for bulk callers: they themselves become `-> Result<(), Error>` and `?`-propagate;
their callers (the `OnceLock` initializers in `dispatch.rs:4840`/`global_bindings`,
`extend_global_bindings` at `dispatch.rs:5091`) `.expect()` **once** at the process-init boundary
(an init-time panic at startup on a *programmer-error* duplicate is acceptable per the existing
doc-comment rationale — the difference is the importer path no longer routes through a panic; the
hand-written always-built tables may still assert at init, since a duplicate there is a build bug).
This preserves "fail fast at startup" for the hand-written tables while giving the **importer** a
`Result` to surface. Document this split in the `register_full_with_source` doc-comment.

### 3.3 Born-red test (step 1)

`fuel-dispatch/src/kernel.rs` tests: replace `step_9a_duplicate_kernel_ref_panics`
(`#[should_panic]`) with `step_9a_duplicate_kernel_ref_errors` that asserts
`register(...)` of the same `fn` twice returns `Err(DuplicateKernelRef{..})`. Write it first, watch
it fail to compile (signature still `()`), then make the change green. This is the canonical
born-red gate for step 1.

---

## 4. Importer parse + lower pipeline (steps, born-red at each)

Each sub-step ships with the failing test first.

### 4.1 Parse layer (`parse.rs`, `schema.rs`)
- Extract front-matter YAML + each `## ` section's ` ```fkc ` fenced block (FKC §3.1).
- Enforce the §3.8 restricted subset *before* deserialize: reject tab indentation (`YamlTabIndent`
  with line/col), anchors/aliases/merge keys (`YamlAnchorDisallowed`), and read all token fields as
  strings (disarm the Norway problem). `BadScalarType` on a type mismatch vs the schema's declared
  scalar type.
- **Born-red:** a fixture file with a tab inside the fkc block -> `Err(YamlTabIndent)`; a fixture
  using `<<` merge -> `Err(YamlAnchorDisallowed)`; a `family: no` operand -> the string `"no"`, not
  a bool.

### 4.2 Lower layer (`lower.rs`)
- `op_kind`/`fused_op` string -> `OpKind`/`FusedOpId` (exactly one of the two present, else
  `OpParamsVariantMismatch`/structural error).
- dtype + `dtype_class` expansion -> `KernelDTypes` / `&'static [DType]`.
- `entry_point` -> `KernelRef`/fused `KernelRef` via `LinkRegistry`.
- **Born-red:** a contract naming a nonexistent op -> `UnknownOpKind`; a contract whose
  `entry_point` is absent from the link registry -> `UnknownEntryPoint`.

### 4.3 Caps / cost / precision lowering (`caps_map.rs`, `cost_expr.rs`, `precision.rs`)
- See §6 (caps), §2.3 (cost), §2.1 (precision).
- **Born-red:** the five-flag `{strided: accepted, broadcast_stride0: accepted}` projects to
  `KernelCaps { strided_input: true }`; `{strided: rejected}` -> `false`; `start_offset: accepted`
  does NOT flip `strided_input` (it routes through auto-Contiguize today, §6). A cost expr
  `"2 * m * n * k"` evaluates correctly at a fixed shape; `"k_len <= sk garbage("` ->
  `Err(CostExprParse)`.

### 4.4 Register layer (`register_into`)
- The §2.1 / §2.2 calls, `?`-propagating the now-`Result` registration.
- **Born-red:** importing a two-kernel bundle populates `table.len()`/`fused.len()` by the right
  counts; importing a bundle whose two kernels resolve to the same `fn` -> `Err(DuplicateKernelRef)`.

---

## 5. Return-contract validation against graph-side metadata (§5 of FKC)

FKC return rules (`dtype_rule: passthrough(lhs)`, `shape_rule: same_as(lhs)`, `layout_guarantee`,
`aliasing`, `bundle`) are **validated, not generated**. For a `fused_op` contract the importer looks
up the matching `FusedOpEntry` in `fuel-graph`'s `default_registry()` by name and checks:

- The FKC `shape_rule`/`dtype_rule` are *consistent with* the registered `shape_rule`/`dtype_rule`
  fns at one or more probe shapes (a sampled-equivalence check, since the FKC rule is a string and
  the registered rule is a fn — evaluate both on a small set of representative input shapes/dtypes
  and require agreement; `ShapeRuleMismatch` on disagreement). This catches a contract that
  mis-describes an op's output.
- The `bundle` slot specs (FKC §5.5): rank ≤ 6 per slot (`RankExceedsSix` — mirrors `[u64;6]` /
  `DimVec`), slot count matches the registered `output_views` arity, and slot **names** round-trip
  via the side-table (FKC keeps the name, not only the `name_hash`). For a multi-output fused op the
  importer cross-checks `output_views(...)[0]` against `shape_rule`/`dtype_rule` (the existing
  `FusedOpEntry` contract).

For a primitive `op_kind` contract there is no `FusedOpEntry`; the return rules are validated for
internal coherence only (dtype rule references a real operand role; shape rule predicate parses).

**Born-red:** a flash-attn contract whose `shape_rule` string disagrees with `flash_attn::entry()`'s
`shape_rule` at a probe shape -> `Err(ShapeRuleMismatch)`; a bundle slot at rank 7 -> `RankExceedsSix`.

---

## 6. The five-flag layout set vs today's single `strided_input` bool (lossy projection)

FKC §4.1 / §12.2: the contract carries five independent layout facts
(`contiguous`, `strided`, `broadcast_stride0`, `start_offset`, `reverse_strides`); today's
`KernelCaps` (`kernel.rs:66`) has exactly one bool, `strided_input`. The importer projects:

```
KernelCaps.strided_input = (layout.strided == accepted) && (layout.broadcast_stride0 == accepted)
```

per FKC §4.1 ("Today's importer projects `(strided && broadcast_stride0)` onto the one
`KernelCaps.strided_input` bool"). The other three flags are handled per as-built behavior:

- `start_offset: accepted` is **not** projected — a non-zero `byte_offset` operand still routes
  through auto-Contiguize today (`kernel.rs:70-73` doc-comment is explicit), so the importer records
  it on the resolved record for forward use but does not flip any current cap.
- `reverse_strides: accepted` is treated as **not declared** at the `KernelCaps` level (the flag does
  not exist yet) — a negative-stride operand is normalized to a non-negative copy by the planner
  until `KernelCaps` grows the flag (FKC §4.1, §12.2). The importer **retains** the parsed value on
  the resolved record (so nothing is lost) and emits it once `KernelCaps` gains the field.
- `awkward_layout_strategy` (FKC §4.3: `requires_contiguous` | `handles_strided` |
  `contiguize_internally`) is validated for coherence against the layout flags (§10.4) and retained;
  its planner consequence (insert `Op::Contiguize` + sum its FKC cost) is the planner's job and reads
  the same `strided_input` projection today.

**Forward-extension hook.** Add a `KernelCaps` follow-up (separate, after the importer lands) that
grows the struct with `reverse_strides: bool`, `start_offset_capable: bool`, and an
`awkward_layout_strategy` enum — the importer is written to fill them the moment they exist (the
parsed values are already on the resolved record). This is a faithful realization of the digest's
"KernelCaps must be an open/extensible set" (digest §2a) without churning the executor now.

**Born-red:** the projection truth table above, as a parameterized test over all five-flag combos.

---

## 7. Quant, sub-byte, and the scale single-place rule

- **Dispatch-key enrichment (FKC §3.2, §12.1).** For a quant kernel, `fdx.quant`
  (`family`/`ggml_dtype`/`granularity`/`role`) enriches the per-operand dtype slot so the key
  distinguishes e.g. `(QMatMul, A=F32×PerToken, W=Q4_0)` from `(QMatMul, A=F32×PerTensor, W=Q8_0)`.
  In as-built terms the GGML format already rides `OpParams::QMatMul { quant_type, .. }` /
  `FusedOpParams::QMatMul { quant_type, .. }` and the flat `Capability` tokens; the importer maps the
  FKC `ggml_dtype` **variant name** to the as-built `GgmlDType` **by numeric code** (FKC §3.4 — there
  is no `Q4_K_M` variant; `Q4_K_M` -> `GgmlDType::Q4K` (12) -> `Capability::MatMulQ4KM`). A contract
  writing `Q4_K_M` in `ggml_dtype` -> `QuantIncoherent`.
- **Scale single-place rule (FKC §3.9.3, resolved decision).** If a kernel's ABI takes the scale as a
  **separate graph input**, the scale is an FKC `accept.inputs` operand and the consuming operand's
  `fdx.quant.scale_operand` names that role — and there is **no** sidecar `FDXQuant.scale_buffer` for
  it. The importer rejects a contract declaring **both** `scale_operand` and a sidecar-bundled scale
  for the same scale (`ScaleDoubleDeclared`, §10.6). Sidecar-bundled scales (GGML INLINE, MX
  separate-buffer F8E8M0) leave `scale_operand: ~`.
- **Sub-byte logical shape carried explicitly (resolved decision).** Sub-byte dtypes
  (`F4`/`F6E2M3`/`F6E3M2`, `size_in_bytes()==0`) MUST pair an `fdx.sub_byte` code with an `fdx.quant`
  block; the importer never sizes off `size_in_bytes` (FKC §3.4, digest §8). The logical shape is
  read from the contract, not derived.
- **`PerBlock` / MX not yet registrable (FKC §6, §10.6).** `ScaleGranularity` has no `PerBlock`
  arm in `fuel-core-types` today, so an MX contract **parse-validates** but returns
  `MxNotYetRegistrable` at the register step (not at parse) — the contract is legal, the consumer is
  behind. Same posture for `gather`-bearing operands before the FDX gather codes land
  (`GatherNotYetSupported`, FKC §3.9.1).

**Born-red:** a Q4K QMatMul contract registers and its key matches the as-built QMatMul key; a
contract with `scale_operand` + sidecar scale -> `ScaleDoubleDeclared`; an MX contract parses then
-> `MxNotYetRegistrable` on register.

---

## 8. `kernel_revision_hash` (shared stable hash with FDX `name_hash`)

`revhash.rs` computes the hash over a **canonicalized parse** of the ` ```fkc ` block (FKC §4.7,
resolved decision), not the raw bytes — so insignificant whitespace/comment edits do not invalidate
a persisted plan, but a semantic change does. Canonicalization: deserialize, drop comments + the
prose, re-serialize the structured fields in a fixed key order, hash that. `"auto"` derives from
`entry_point` + `revision_base` (front-matter). The hash function is the **same** stable function FDX
uses for `name_hash` (decision: shared) so a `(backend, op, dtypes, kernel_revision_hash)` persisted
tuple (digest §7) re-resolves consistently. **Bundle slot names** are kept in a side-table for
round-trip (decision; FKC §5.5 MAJOR fix) — the hash covers the structured contract, the side-table
preserves the human-readable slot names the `[u64;6]`/`name_hash` form would otherwise lose.

**Born-red:** two contracts differing only in a comment hash equal; differing in `flops` hash
differ; `"auto"` is deterministic given fixed `entry_point` + `revision_base`. A frozen-fixture test
pins the exact hash so the function choice cannot silently change (mirrors FDX's build-time mapping
test).

---

## 9. Single-bundle vs globbed-multi-file import

Both modes produce the same `ImportedProvider` (FKC §9.1/§9.2: the importer treats a bundle's many
`## ` sections and a per-file single section identically):
- `import_bundle_str` / `import_bundle`: one file, front-matter once, N sections.
- `import_glob`: many files; each contributes its sections; **front-matter must agree** across files
  (`provider.name`/`backend`/`kernel_source`/`link_registry`/`revision_base`) or `ProviderMismatch`.
  Glob order is sorted for determinism (so the revision hash + duplicate-detection order are stable).

**Born-red:** a 2-file glob with matching front-matter yields the union of kernels; a 2-file glob
with mismatched `backend` -> `ProviderMismatch`.

---

## 10. Build-time validation + CI lint (`validate.rs`)

Every check runs at import time, `Result`-returning, no `try_*` (digest §3, FKC §10, G6). The
validators (named `V-FKC-*` to parallel FDX's `V*`):

1. **V-FKC-1 required fields** — `kernel`, exactly one of `op_kind`/`fused_op`, `blurb`,
   `entry_point`, ≥1 `accept.inputs`, ≥1 `return.outputs`. (`MissingField`.)
2. **V-FKC-2 blurb equality** — structured `blurb:` equals the prose blurb (FKC §10.11,
   `BlurbMismatch`).
3. **V-FKC-3 non-overlapping keys** — within a provider, no two contracts resolve to the same
   `(op|id, dtypes, backend, kernel_source)` *and* the same `KernelRef` (`DuplicateKernelRef`; the
   resolved-pointer level, §3).
4. **V-FKC-4 layout coherence** — `reverse_strides: accepted` requires `strided: accepted` is not
   *forced* (they are independent — §4.1.1), but `contiguous: n/a` with `strided: rejected` is
   incoherent; `awkward_layout_strategy` consistent with the flags (`contiguize_internally` requires
   `strided: accepted`). (`LayoutIncoherent`.)
5. **V-FKC-5 op-param namespace** — `op_kind` contract -> `OpParams` variant; `fused_op` contract ->
   `FusedOpParams` variant (FKC §3.7, §10.7; `OpParamsVariantMismatch`).
6. **V-FKC-6 quant coherence** — `ggml_dtype` is a real `GgmlDType` variant by code; scale
   single-place (`ScaleDoubleDeclared`); sub-byte has `fdx.quant` (`QuantIncoherent`).
7. **V-FKC-7 return coherence + rank** — bundle rank ≤ 6 (`RankExceedsSix`); shape/dtype rule vs
   graph metadata (§5, `ShapeRuleMismatch`).
8. **V-FKC-8 cost expr parses** — every cost field parses in the §4.4 grammar (`CostExprParse`).
9. **V-FKC-9 non-placeholder precision/cost** — a non-reference contract may not ship `UNAUDITED`
   precision or `unknown_cost` cost (`PlaceholderPrecision`/`PlaceholderCost`). Mirrors the two
   existing lints (every Vulkan registration has a non-UNAUDITED `PrecisionGuarantee` + real
   `CostFn`; the always-built backend has a `bit_stable` kernel per primitive op — digest §3).
10. **V-FKC-10 cross-spec token subset** — FKC's accepted dtype/quant/granularity token set is a
    subset of FDX's (FKC §0, §10.12) — a unit test, run as part of the lint.

### 10.1 The CI lint

A new test target `fuel-dispatch/tests/fkc_lint.rs` (runs in normal `cargo test -p fuel-dispatch`,
no GPU) that:
- imports every checked-in contract file under `docs/kernel-contracts/` (or wherever the contracts
  live — see §11.2) against a **stub `LinkRegistry`** that maps every `entry_point` to a no-op
  `KernelRef` (so the lint validates *structure + coherence* without needing the real kernels), and
  asserts `import_*` returns `Ok` for every well-formed contract and the documented `Err` for the
  negative fixtures.
- runs V-FKC-10 (the cross-spec subset test) and the §8 revision-hash frozen-fixture test.

This is the constitution's "statically verifiable at registration, CI-lintable" requirement
(digest §3, FKC G6) made concrete.

---

## 11. Rollout — keep `main` building, ship in verifiable increments

> **STATUS UPDATE (2026-07-02) — strategy superseded for the endgame.** Steps 1–7 shipped
> (importer + parser + lowering + `register_into` + validators + the authored CPU corpus + CI lint).
> The **first production consumer landed** (WIP branch): the CPU **elementwise-binary** family (8 ops
> × 4 dtypes = 32 bindings) is now registered by importing its contract in `register_cpu_kernels`,
> and its hand-written `table.register(...)` calls are **DELETED** (behavior-preserving; born-red
> `global_bindings_registers_binary_family_from_contract`). Crucially, the maintainer **superseded the
> "parallel path behind `--features fkc`, flip the default later" plan (old steps 6/7/9)**: the `fkc`
> cargo feature is **REMOVED** and FKC is now **unconditional core infrastructure** (serde/serde_yml
> always compiled; `pub mod fkc` unconditional). Rationale: once a family's hand-written registration
> is deleted on migration, a build that could disable the importer would silently lose that family —
> a doomed config. No gate ⇒ no such config, and no dual registration paths to drift. So the remaining
> work (step 8: widen family-by-family) is **plain deletion of each migrated family's hand-written
> glue**, not a `cfg(not(fkc))` fallback, and there is no "flip the default" step 9 to do.

Ordered. Each numbered item is a separate commit (or small commit cluster) on
`feat/kernel-contracts-dlpack`, each with its born-red test observed to go red then green. ~~**`main`
keeps building because the importer is feature-gated until 11.6.**~~ (Superseded — FKC is now
unconditional; see the status update above.)

1. **Prerequisite — `register_full_with_source` -> `Result` (§3).** Born-red: the duplicate-errors
   test. Thread `Result` through wrappers + bulk callers + the three sibling backends + tests. Gate:
   `cargo test -p fuel-dispatch -p fuel-aocl-cpu-backend -p fuel-mkl-cpu-backend` green; the
   `OnceLock` init `.expect()`s preserve fail-fast for the hand-written tables. **No FKC code yet** —
   this is a standalone never-panic fix that is independently valuable and lands first.

2. **`fkc` feature + module skeleton (§1).** Empty modules, `FkcError` enum, `LinkRegistry` trait,
   feature off by default. Gate: `cargo check -p fuel-dispatch` compiles the skeleton. (The `fkc`
   feature has since been REMOVED — FKC is unconditional; see the §11 status update.)

3. **Parse + schema (§4.1).** Born-red parse fixtures (tab/anchor/Norway/blurb). Gate:
   `cargo test -p fuel-dispatch --lib fkc::parse`.

4. **Lower + caps/cost/precision/link (§4.2–4.3, §2, §6).** Born-red: op/dtype/entry-point resolution
   + the caps projection truth table + cost-expr eval. Gate: the `fkc::lower` / `fkc::cost_expr`
   tests.

5. **register_into + revision hash + validators (§2.4, §5, §7, §8, §10).** Born-red: round-trip a
   2-kernel bundle into a fresh `KernelBindingTable` + `FusedKernelRegistry` and assert
   `lookup`/`lookup_by_dtypes` returns the imported kernels; the negative-fixture battery
   (`DuplicateKernelRef`, `ScaleDoubleDeclared`, `MxNotYetRegistrable`, `ShapeRuleMismatch`,
   `RankExceedsSix`). Gate: `fkc::register` + `fkc::validate` tests.

6. **Author the first real contract file + wire one provider (§11.2).** Convert the **CPU primitive**
   inventory (`docs/kernel-contracts/_inventory/cpu.md`) into a real `cpu.fkc.md` bundle, add a
   `LinkRegistry` for `fuel-cpu-backend`'s wrappers, and add a **parallel** import path:
   `register_cpu_kernels_via_fkc(table)` that imports the bundle and registers — guarded behind
   `--features fkc`. The hand-written `register_cpu_kernels` stays the default. A test asserts the
   two paths produce an **equivalent** binding table (same keys, same caps projection, same
   precision/cost where the contract is authored to match). This is the "ship -> verify" gate: the
   importer must reproduce the hand-written registrations before it can replace them.

7. **The CI lint (§10.1)** over all authored contracts. Gate: `cargo test -p fuel-dispatch` (the lint
   runs without the `fkc` feature? — no; it needs the importer, so the lint test is `#[cfg(feature =
   "fkc")]` and CI runs `-p fuel-dispatch --features fkc`). Document that CI must add the
   `--features fkc` job.

8. **Widen to the remaining providers** one bundle at a time (Vulkan, fused, quantized, conv-attn,
   mkl/aocl, metal), each with its own equivalence test (item 6's pattern). Negative-stride
   `reverse_strides` contracts author cleanly but stay [consumer-ahead] until the `KernelCaps`
   follow-up (§6).

9. **Flip the default (only after every provider has an equivalence-tested contract).** Make `fkc` a
   default feature and switch the global init (`dispatch.rs` `OnceLock` initializers) to import the
   checked-in bundles instead of calling the hand-written `register_*`. The hand-written functions
   remain (as the `LinkRegistry`-referenced wrappers ARE the kernels) but the *registration glue*
   becomes the contract import. This is the "import = registration, zero hand-written glue" end state.
   Gate: full `-p fuel-dispatch` suite + a live-GPU smoke (one suite at a time per environment
   discipline) confirming an imported plan still executes.

### 11.2 Where contract files live

Author the contracts beside the existing inventories: promote
`docs/kernel-contracts/_inventory/*.md` into validated `docs/kernel-contracts/*.fkc.md` bundles (the
inventories are the prose seed; the `.fkc.md` adds the ` ```fkc ` blocks). They drop straight into the
`fuel-book` mdBook (FKC G1). The CI lint and the runtime importer both read from there. The
provider's `link_registry` front-matter names the Rust symbol path
(e.g. `fuel_cpu_backend::fkc::ENTRY_POINTS`) the `LinkRegistry` impl wraps.

### 11.3 Docs to update in the same change (per CLAUDE.md "docs are part of every material change")

- `docs/architecture/05-backend-contract.md` — note that kernel registration is now contract-import;
  bump its version + add a `10-decisions-log.md` entry on the MAJOR bump (registration mechanism
  change is an interface change).
- `ROADMAP.md` frontier — add the FKC-adoption program and its phase position.
- The two specs' §0 "Current status / handoff" — flip "Nothing here is implemented" to point at the
  landed importer module + lint as each step lands.

---

## 12. Risks / open items (flagged, not deferred)

- **Cost-fn `fn`-pointer vs runtime coefficients (§2.3).** Strategy (A) routes every imported cost
  through a global side-table keyed by `(op|id, dtypes, backend)`. If a provider registers two
  alternatives at one key with *different* cost expressions, the side-table must key by the resolved
  `KernelRef` too (or by `kernel_source`) to disambiguate. Resolve by keying the cost side-table on
  `(op|id, dtypes, backend, kernel_source)` — the same tuple the persistence layer uses. Note this in
  `cost_expr.rs`.
- **`kernel_revision_hash` for primitives has no `BindingEntry` slot** (§2.1). v1 retains it on the
  resolved record but cannot persist it through the primitive binding table until `BindingEntry`
  grows a `revision` field — a small follow-up, sequenced behind the persistence consumer (digest §7;
  "no consumer is a reason to sequence behind, not skip").
- **Equivalence tests (item 6) define "equivalent" generously.** Cost/precision authored in a
  contract must *match* the hand-written values to pass — which means the first contract authoring
  pass is also an audit of the hand-written cost/precision. Treat divergences as findings (some
  hand-written costs may be wrong; the contract is the chance to fix them, with a test).
- **`serde_yaml` is unmaintained upstream.** If that matters for the dependency policy, the §3.8
  restricted subset is small enough to hand-parse; decide at step 3.

---

## 13. One-screen summary (the ordered spine)

1. `register_full_with_source` panic -> `Result` (+ `DuplicateKernelRef` error), thread through all
   callers — the never-panic prerequisite. (born-red: duplicate-errors test)
2. `fuel-dispatch/src/fkc/` module + `fkc` feature + `FkcError` + `LinkRegistry`. (skeleton compiles)
3. Parse the markdown + restricted-YAML ` ```fkc ` blocks. (born-red: tab/anchor/Norway/blurb)
4. Lower to `(OpKind|FusedOpId, KernelDTypes, BackendId, KernelRef, KernelCaps, CostFn, Precision)`;
   five-flag -> `strided_input` projection; two-target cost compiler. (born-red: projection table +
   cost eval)
5. `register_into` the two registries; revision hash; quant/scale/sub-byte; all V-FKC validators.
   (born-red: round-trip + negative battery)
6. Author `cpu.fkc.md`, wire a `LinkRegistry`, prove import == hand-written registration. (born-red:
   equivalence test)
7. CI lint over all contracts (`--features fkc`). 
8. Widen provider-by-provider, each with an equivalence test.
9. Flip `fkc` to default; global init imports the bundles; hand-written glue retired. (gate: full
   suite + one live-GPU smoke)

Everything above 6 is feature-gated and off by default, so `main` builds throughout; step 1 is an
independently-valuable constitution fix that lands even if the rest slips.
