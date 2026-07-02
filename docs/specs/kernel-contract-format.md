# Fuel Kernel Contract Format (FKC) — how a kernel provider advertises to Fuel

**Status:** DRAFT FOR REVIEW (2026-06-17; reconciled 2026-06-20), on branch
`feat/kernel-contracts-dlpack`. Design pass — no code yet. This is the final-after-critique revision
of the `_drafts/` v0.1; the "Resolved critique" note below records what changed and why. **Reconciled
2026-06-20** to the adaptive-runtime-fusion decision ([10-decisions-log](../architecture/10-decisions-log.md),
G1/G4/G5): the §1 / §9.4 "not a runtime-extensible registry / freezes the registry" claims are
re-scoped (Tier-2 trusted, Fuel-orchestrated, cost-gated runtime fused-op registration is now a goal;
the kernel binding table is already Tier-1 runtime-extensible); a `fused_op` contract is required to
carry its **recipe** (`decompose` + `pattern`) or it is an opaque island; and §4.11/§4.12's structural
miss is distinguished from the closed-world `FusionMissRecord` (the v1 telemetry headline) — see the
dated 2026-06-20 note at the end.
**Scope:** a versioned, **markdown + structured-block hybrid** file format in which a kernel
provider declares, *per kernel*, everything Fuel's optimizer needs to choose, cost, admit, and
dispatch that kernel: dispatch key, accept-contract (dtypes / layouts / shape / DLPack-extension
requirements), return-contract (dtype / shape / layout / aliasing), and the
**capability + cost + precision + determinism** advertisement that lets the planner pick
contiguize-vs-strided-vs-materialize. Importing a provider's contract file(s) **auto-registers
every kernel** onto Fuel's dispatch surface.
**Audience:** kernel providers (Baracuda CUDA, `fuel-vulkan-kernels` Slang, `fuel-metal-kernels`,
`fuel-cpu-backend`, MKL/AOCL, `fuel-quantized`, external third parties) and Fuel's
`fuel-dispatch` / `fuel-graph` import path. Drops straight into the `fuel-book` mdBook.

Authoritative inputs: the architecture-constraints digest
(`docs/specs/_research/architecture-constraints.md`); the as-built dispatch types in
`fuel-dispatch/src/{kernel.rs, fused.rs}` (`KernelRef`, `KernelDTypes`, `KernelCaps`,
`BindingEntry`, `KernelBindingTable`, `CostFn`, `CostEstimate`, `PrecisionGuarantee`,
`KernelRevisionHash`, `OpParams`); the graph-side fused registry
(`fuel-graph/src/registry.rs`: `FusedOp { shape_rule, dtype_rule, output_views, … }`); the
core types in `fuel-core-types/src/{dtype.rs, shape.rs, symbol.rs, quant_scale.rs, quantized.rs,
capability.rs, backend.rs, probe.rs, storage.rs}`; and the **sibling FDX spec**
(`docs/specs/dlpack-extension.md`) plus its two 2026-06-17 additions —
`docs/specs/_drafts/fdx-addition-gather.md` (paged / indexed-residency `FDXIndexedResidency`) and
`docs/specs/_drafts/fdx-addition-affine.md` (affine `FDXExtent`) — for all tensor (DLPack +
extension) vocabulary. When this draft and the constitution (`docs/architecture/`) conflict, the
constitution wins; flagged conflicts are called out inline as **CONSTITUTION-CONFLICT** callouts.

---

## Resolved critique (what changed from `_drafts/` v0.1)

This revision incorporates an adversarial verification pass against the real `dlpack.h`
(DLPACK 1.3 / `DLManagedTensorVersioned`) and the as-built `fuel-core-types` / `fuel-dispatch` /
`fuel-graph` sources. The load-bearing fixes:

- **[BLOCKER] Fused-op cost-fn signature.** v0.1 said *every* cost expression compiles to the
  primitive `CostFn = fn(&[Shape], &[DType], &OpParams, &BackendCapabilities) -> CostEstimate`.
  That is wrong for `fused_op` contracts: the as-built `BackendImpl.cost`
  (`fuel-dispatch/src/fused.rs:63`) is `fn(&[Shape], &FusedOpParams, &BackendCapabilities) ->
  CostEstimate` — **no `&[DType]`**, and `FusedOpParams` not `OpParams`. §4.4 / §5 / §12.3 now
  define **two distinct compile targets** (primitive vs fused) and state which return rules
  compile to functions vs are checked against the key.
- **[MAJOR] Contiguize/Dequantize are themselves FKC kernels.** The planner's
  contiguize-vs-strided comparison was costable "by assertion." §4.3 / §4.4 now state explicitly
  that `Op::Contiguize` and `Op::Dequantize` are ordinary FKC-described kernels (CPU + each GPU
  backend ships a contract for them), and add a worked two-candidate example showing the planner
  **summing** the two `CostEstimate`s.
- **[MAJOR] Duplicate detection at the resolved-`KernelRef` level.** v0.1 deduped by
  `entry_point` *string*, which does not pre-empt the as-built panic — that panics on duplicate
  `KernelRef` *function pointer* (`kernel.rs:910-918`), and two distinct `entry_point` strings can
  resolve to the same `KernelRef` via the link registry (an alias / shared generic fn). §10.10 now
  dedupes **after** `entry_point → KernelRef` resolution and adds a `DuplicateKernelRef` error,
  plus a **CONSTITUTION-CONFLICT** callout that the underlying `register_full_with_source` must
  lose its `panic!` to honor never-panic.
- **[MAJOR] Bundle round-trip is rank-capped and name-lossy.** `FDXOutputView` uses `shape:
  [u64; 6]` / `name_hash: u64`; the as-built `OutputView` (`storage.rs:46`) has `shape: Shape`
  (arbitrary rank, symbolic-capable) and `name: Option<&'static str>`. §5.5 now states the
  rank ≤ 6 limit with an explicit validation error, keeps the slot **name** in a side-table (not
  only a hash) so it round-trips, and downgrades the "lossless" claim accordingly.
- **[MAJOR] YAML hand-authoring landmines.** §3.8 (new) pins the YAML subset, the
  norway-problem / sexagesimal / tab pitfalls, and the canonical scalar forms, so a developer can
  author by hand without silent type coercion.
- **[MINOR] `PerBlock` has no `ScaleGranularity` counterpart yet.** §6 / §10.6 now state plainly
  that `PerBlock` is an FDX/FKC-only value; an MX contract parse-validates but is **not
  registrable** until `ScaleGranularity` (or a block-quant descriptor) gains `PerBlock`.
- **[MINOR] `Q4_K_M` vs `Q4K` naming.** §3.4 / §8 now use the exact as-built `GgmlDType::Q4K`
  (code 12); `Q4_K_M` is documented as the file-format name and matched **by numeric code**.
- **[MINOR] Single normative source of shared codes.** §0 / §3.2 now designate **FDX** as the
  normative source for all shared dtype / quant / granularity / substrate codes; FKC references
  FDX section numbers and the generated `fuel-core-types` constants rather than re-listing values.
- **[MINOR] Live-extent cost re-eval needs a signature change.** §4.4 / §12.3 now scope live-`k_len`
  re-evaluation as forward-looking (the as-built `CostFn` takes no `&SymEnv`); the
  capacity-evaluated path maps onto today's `CostFn` cleanly and is all that is claimed for v1.
- **[MINOR] Unknown-enum-value policy reconciled with FDX.** §11 now splits unknown values:
  correctness/admissibility-affecting fields (`awkward_layout_strategy`, quant `family`) → typed
  **error / drop the kernel** (matching FDX §14); only advisory fields (`cost.class` bucket,
  tiling hints) warn-and-default.
- **[MINOR] Parallelism budget (`slot_capacity`).** §0 / §4.10 now state that per-backend slot
  capacity is a static fact on `BackendCapabilities` (not per-kernel, not telemetry) and point at
  where the optimizer reads it.

Findings deliberately **not** addressed here (and why) are listed at the end of §13.

---

## 0. Current status / handoff

- This is the **advertisement (capability/cost-axis) half** of the kernel boundary. Its sibling,
  **FDX** (`dlpack-extension.draft.md`), is the **tensor/storage-axis** half. FKC *describes a
  kernel*; FDX *describes a tensor handed to that kernel*. They share the dtype / quant /
  symbolic-extent vocabularies but are kept separate concerns (13-interchange: weight ⊥ graph).
- **FDX is the single normative source for all shared codes** (dtype / quant `family` /
  `granularity` / `pack_order` / substrate). FKC **never re-lists** their numeric values; it names
  them by symbol and cites the FDX section + the generated `fuel-core-types` constants. The dtype
  vocabulary is the Fuel `DType` set (FDX §6.1); the quant vocabulary is `FDXQuant` (FDX §6.2);
  the granularity vocabulary is `FDXScaleGranularity` (FDX §6.2). A cross-spec consistency test
  (§10 rule 16) asserts FKC's accepted token set is a subset of FDX's, so the two cannot drift.
- Nothing here is implemented. The contract maps onto existing `fuel-dispatch` types (§12); the
  importer is a new `fuel-dispatch` module (`fkc`) that parses the structured blocks and calls
  the existing `KernelBindingTable::register_full_with_source(...)` / `FusedKernelRegistry`
  registration paths. **One** dispatch primitive must change to honor never-panic (the importer
  must not have to drive a panicking registration path — see the CONSTITUTION-CONFLICT in §10.10);
  otherwise FKC is a *serialization + authoring surface* over types that already exist.
- Targets the **Intended** architecture (Phase D symbolic extents + per-tier cost vector +
  sessions) while staying loadable by today's code: a v1 contract may declare fields whose
  *consumers* are still ahead (per-tier memory, **symbolic live-extent cost**, MX/`PerBlock`
  quant); today's importer reads what today's types model and ignores the forward-looking tail
  (size-prefixed, §11). Each such field is flagged **[consumer-ahead]** at its definition.
- **Parallelism budget is not per-kernel.** The optimizer's wall-clock model
  (`max(parallel_branches) + serial`, digest §5) needs a per-backend parallelism width
  (max concurrent execution contexts / "slots"). That is a **static device fact**, distinct from
  live slot-count *telemetry* (which is out of scope, §1). It lives on `BackendCapabilities`
  (`slot_capacity`, §4.10), read by the optimizer's scheduler — **not** in a per-kernel contract.

> **2026-06-18 — describe-only (non-registrable) sections (§3.10).** Added a section-level
> `registrable: false` marker so a `## ` section can be **documentation-only**: it is parsed and
> its descriptive facts (dtypes, layout, quant) are still validated, but it is **not registered**
> and is **not required** to name a real dispatch `op_kind`/`fused_op`. This serves two needs the
> corpus already had: (1) **chassis/family umbrellas** (`## binary`, `## unary`) that document a
> shared algorithm backing many per-`(op, dtype)` registrable thunks — previously these had to
> name one op arbitrarily as a "representative"; and (2) **ops with no dispatch `OpKind`**
> (`Im2Col`/`Im2Col1d`/`Col2Im1d`, pools, `Upsample*`, `Transpose`/`Permute`, `ArgSort`,
> `Conv2dSimple`, `Dequantize*`/`QuantizeQ8_0`, `AddAssignScaled`, `ConvTranspose1D`, …) that
> Fuel performs as graph rewrites / views / lowerings. The discipline (§3.10): a token that is a
> *typo* for a **real** `OpKind` (e.g. `Cumsum→CumSum`, `Clamp→ClampInplace`, `PowI→PowIInplace`,
> `ConvTranspose2D`) is RENAMED to the exact variant, not marked describe-only; only a token with
> **no** real `OpKind` becomes describe-only. Never invent an `OpKind`; never relax a validator to
> hide a defect. Importer landed in `fuel-dispatch/src/fkc/{schema,validate,register}.rs` (the
> corpus is unchanged this round).

---

## 1. Overview & goals

A kernel in Fuel is **pure description consumed by the optimizer** — it never makes a strategic
choice (no internal placement, fusion, kernel-variant swap, result caching, or silent
layout/dequant materialization; that is the constitution's governing principle and the 01
enforcement gate). The kernel-contract format exists to make that description **authorable,
reviewable, diff-able, doc-publishable, and machine-importable**, so that:

- a developer who knows markdown can read a provider's kernels as prose + a precise table;
- a parser can extract an unambiguous structured contract per kernel;
- importing the file(s) **auto-registers every kernel** onto Fuel's dispatch surface
  (`KernelBindingTable` key → `BindingEntry { caps, precision, cost, kernel_source }`, or the
  `FusedKernelRegistry` for fused ops), with **zero hand-written registration glue**;
- every fact the optimizer needs is *visible* (the enforcement gate: a change is admissible only
  if it makes more cost/precision/layout facts flow to the optimizer). Hiding a cost, a precision
  bound, or a layout capability behind backend code is a constitutional failure; FKC's job is to
  surface them.

### Goals

- **G1 — Hybrid human/machine format.** One markdown file is *both* readable docs (prose blurb +
  long description) *and* a precisely-schema'd structured block per kernel. mdBook renders it;
  the importer parses it.
- **G2 — Complete per-kernel advertisement.** Name, blurb, long description, dispatch key,
  per-operand accept-contract, per-output return-contract, capability + cost + precision +
  determinism. Nothing the optimizer needs lives outside the contract.
- **G3 — Layout richer than one boolean.** Accept-layouts are an explicit enumerated capability
  set (contiguous-only / strided / broadcast-via-stride-0 / non-zero-start-offset-capable /
  in-place), not the single `strided_input: bool` of today — designed so the planner can cost
  contiguize-vs-strided-vs-materialize honestly, and so today's one bool is a faithful *lossy
  projection* of the richer set (§12.2).
- **G4 — Cost as a vector contribution, precision as a hard pre-filter.** Cost yields the
  per-node contribution to the optimizer's cost *vector* (compute / bandwidth / overhead +
  per-tier memory); precision is a structured `PrecisionGuarantee` that the planner's
  precision-filter pass applies **before** cost ranking.
- **G5 — Import = registration.** A provider becomes available by importing their contract
  file(s) — one bundle file *or* many globbed files. The mapping onto `(OpKind, [DType…],
  BackendId, kernel_source)` + `KernelCaps` + `CostFn`/fused-cost-fn + `PrecisionGuarantee` +
  `OpParams`/`FusedOpParams` is fully specified (§12).
- **G6 — Build-time validatable.** Every consistency check that can run at registration runs at
  registration, `Result`-returning, no `try_*` siblings; a CI lint verifies required fields,
  non-overlapping keys, non-placeholder precision, and **cost provenance** — a cost must be
  **author-`declared` OR explicitly `judge_measured`** (both first-class), and a bare/placeholder/
  omitted-without-marker cost still fails (§4.4, §10.8a).
- **G7 — Versioned, additively extensible.** A format-version field; new fields are additive and
  ignored by older importers; enums are `#[non_exhaustive]`-spirited (§11).
- **G8 — Tensors in DLPack + FDX terms.** Every operand/output is described in standard DLPack
  (`DLDataType` code/bits/lanes, device, layout) plus the FDX extension codes for sub-byte /
  quant / symbolic / substrate facts. FKC never invents a parallel tensor vocabulary and never
  re-numbers an FDX code (§0).
- **G9 — Never panic; pure description.** Importer errors are typed `Result` (§10). A contract can
  *describe* a kernel that handles awkward layouts internally, but it **declares** that it does
  so (so the planner costs the fused contiguize+op honestly) — it can never hide a decision.

### Non-goals

- Not a transport for **telemetry** (live slot count, per-tier memory pressure, queue depth).
  Telemetry is queried live via the Tier-1 `BackendRuntime` trait; FKC carries only the *static*
  advertisement the optimizer bakes at plan time. (Static *slot capacity* — the parallelism
  width — is a `BackendCapabilities` fact, §4.10, distinct from the live slot count.)
- Not a tensor-interchange format (that is FDX) nor a graph-interchange format (that is the base
  map; 13-interchange).
- Not a place to encode within-kernel concurrency (the backend's business — the principled
  exception to "backends don't decide") or device placement (the planner's).
- Not an *untrusted* runtime-extensible registry: arbitrary user-hot-loaded ops/rules and new
  *primitives* stay out (the build-time-closed `Op` enum; [09-non-goals](../architecture/09-non-goals.md)).
  But the blanket "built at startup and frozen thereafter" claim is **re-scoped** by the 2026-06-20
  adaptive-fusion decision ([10-decisions-log](../architecture/10-decisions-log.md), G4): the **kernel
  binding table** (implementations) is **already runtime-extensible** (`extend_global_bindings`,
  Tier 1), and **trusted, Fuel-orchestrated, cost-gated runtime registration of a new *fused-op
  identity*** (Tier 2, via the declarative recipe form — append-only, stable `FusedOpId`s) is now an
  architectural goal. FKC is read at import/registration time, but registration time is no longer
  only "process startup."

---

## 2. Design principles

- **P1 — Description, never decision.** Mirrors the constitution. A contract says *what a kernel
  accepts, returns, costs, and guarantees*; it never says *use me here* or *I'll fix it
  silently*. The one nuance: a kernel that contiguizes awkward layouts internally must **declare
  `awkward_layout_strategy = contiguize_internally`**, turning a hidden behavior into a costed,
  visible fact (§4.3).
- **P2 — Maximize optimizer visibility (the 01 gate).** Every field exists because the optimizer
  reasons over it. If a kernel knows a fact relevant to placement/cost/precision/layout, the
  contract has a slot for it.
- **P3 — Hybrid, lossless both ways (within stated limits).** The prose is for humans; the
  structured block is authoritative for the importer. The two must not disagree — a lint can
  re-render the structured block's blurb and diff it against the prose blurb (§10.11). "Lossless"
  is bounded: see the bundle rank/name limits in §5.5.
- **P4 — Tensors in DLPack + FDX terms (G8).** dtype = `DLDataType` + optional FDX
  `logical_dtype`; layout facts = explicit capability flags backed by FDX layout vocabulary;
  quant = FDX `family`/`ggml_dtype`/granularity codes; symbolic = FDX `SymId`/capacity vocabulary.
  All by FDX symbol, never by a re-listed numeric value (§0).
- **P5 — Key is richer than `(OpKind, [DType])`.** The dispatch key carries per-operand dtype
  **and** quant granularity / format (digest §9: `ScalePair` and GGML format are part of the
  key), plus `BackendId` and `kernel_source`. This is exactly the as-built
  `(OpKind, KernelDTypes, BackendId)` key + the `kernel_source` tag, with the quant facts encoded
  into the operand descriptors (§3.2, §12.1).
- **P6 — Precision before cost.** The contract makes `PrecisionGuarantee` a first-class block
  with the audited-vs-unaudited distinction, because the planner's admissibility filter runs on
  it before any cost comparison (digest §4).
- **P7 — Cost is a vector, memory is per-tier.** The cost block yields compute/bandwidth/overhead
  for the time axis and a **per-tier (disk/host/device) memory footprint** for the memory axis
  (digest §5). Today's `CostEstimate` is the scalar-ish projection (§12.3); FKC carries the full
  vector and degrades to today's three scalars on import.
- **P8 — Cost may be symbolic (at capacity for v1).** A cost expression may reference a symbolic
  extent (`SymId`) so a symbolic-shape graph plans once. **v1 evaluates the expression at
  capacity** (`Extent::bound()`), which maps cleanly onto today's `CostFn`. Re-evaluation at the
  *resolved live* extent at route-pick time is **[consumer-ahead]** — it requires a `CostFn`
  signature extension (§4.4) and is scoped as forward-looking, not claimed for v1.
- **P9 — Serializable, no pointers.** Everything in a contract is data: op name, dtype codes,
  capability flags, cost coefficients, precision bounds, a **`kernel_revision_hash`**, and a
  *symbolic kernel entry-point reference* (a string id resolved against the provider's link
  registry at import) — **never a function pointer or process-absolute address**. This is what
  lets a persisted plan store `(backend_id, op_kind, dtypes, kernel_revision_hash)` and
  re-resolve on load (digest §7).
- **P10 — Build-time validatable, additively versioned (G6/G7).**

---

## 3. The hybrid format

### 3.1 File anatomy

An FKC file is a normal markdown document. It begins with a **file-level YAML front-matter**
block (between `---` fences) declaring the format version and provider identity, then one
**`## ` section per kernel**, each containing free prose (blurb + long description) followed by
exactly one fenced **` ```fkc ` typed block** carrying the structured contract in YAML.

```
---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan            # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"   # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:8f3c1a"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — kernel contracts

Prose intro to the bundle (optional). Rendered by mdBook; ignored by the importer.

## binary  (Add / Sub / Mul / Div / Max / Min)

One-line blurb here … (this prose blurb MUST equal the structured `blurb:` — §10.11.)

Long description in prose: the algorithm, the numerics (all math at f32, half narrows on
store), perf notes (both-contig fast path; per-operand strided path masks packed-u32 lanes for
bf16), known limitations (NOT offset-capable; bf16 out_size must be even). A developer reads
this; the importer ignores it and reads the block below.

```fkc
kernel: binary
# … the structured contract (schema in §3.3) …
```

## affine  (y = x*mul + add)
…
```

Why this shape:

- **Front-matter** is the single place provider-wide defaults live (backend, `kernel_source`,
  the link registry, the revision base). Per-kernel blocks inherit and may override.
- **One `## ` per kernel** gives mdBook a heading + anchor and gives the importer a stable record
  boundary. A bundle file is just many such sections (§9.1); a per-kernel file is one section
  (§9.2). The importer treats them identically.
- **The ` ```fkc ` block is authoritative.** Anything outside it is documentation. The block is
  YAML (chosen over TOML for nested-list ergonomics and over a bespoke grammar for tooling
  ubiquity; the schema in §3.3 is the contract, YAML is the carrier). The exact YAML subset and
  the hand-authoring pitfalls it avoids are pinned in §3.8.

### 3.2 Tensor descriptors (shared by accept + return)

Every operand and output is a **tensor descriptor**, expressed in DLPack + FDX terms (P4). Every
dtype / quant / granularity token below is an FDX symbol; FKC does not assign it a value (§0). It
is a YAML mapping:

```yaml
# A tensor descriptor.
name: lhs                 # operand role name (diagnostic + maps to FDX view name; §5.5)
optional: false           # true ⇒ OPTIONAL input operand (e.g. conv `bias`, flash `alibi_slopes`).
                          # The importer's key-builder then registers the op BOTH with AND without
                          # this operand (two operand-count keys, same entry_point/kernel — presence
                          # is implicit in inputs.len()). Only the LAST input may be optional
                          # (an earlier optional operand is an OptionalOperandNotLast error, never a
                          # silent mis-key); an output may not passthrough() an optional operand.
                          # Defaults to false (a required operand). Ignored on outputs.
dtypes: [F32, F64, BF16, F16]   # accepted DLPack dtypes (Fuel DType names; FDX §6.1 / §3.4)
dtype_class: float        # optional shorthand: int|uint|float|any (expands per §3.4)
# --- layout capability (richer than one bool — §4.1) ---
layout:
  contiguous: required          # required | accepted | n/a
  strided: accepted             # accepted | rejected   (walks arbitrary strides)
  broadcast_stride0: rejected   # accepted | rejected   (stride-0 axis = broadcast)
  start_offset: rejected        # accepted | rejected   (non-zero byte_offset / view base)
  reverse_strides: rejected     # accepted | rejected   (NEGATIVE strides, e.g. Op::Flip; §4.1.1)
  awkward_layout_strategy: ~    # OPTIONAL per-operand override of caps.awkward_layout_strategy
                                # (requires_contiguous|handles_strided|contiguize_internally; §4.3.1).
                                # ~ ⇒ inherit the kernel-wide caps default. Lets a kernel walk
                                # strided q/k/v while requiring contiguous aux operands.
# --- shape / rank ---
rank: any                       # exact int, "any", or a range "2..=4"
shape_constraint: same_as=out   # free predicate vocabulary, §3.5
# --- DLPack-extension (FDX) requirements ---
fdx:
  requires_ext: false           # true ⇒ this operand's meaning needs an FDX sidecar
  quant:
    family: none                # FDXQuant.family symbol: none|GGML_BLOCK|MX|AFFINE_INT|AFFINE_FLOAT|AFFINE_BLOCK (FDX §6.2)
    ggml_dtype: ~               # GgmlDType variant NAME when family=GGML_BLOCK (e.g. Q4_0, Q4K); §3.4
    granularity: ~              # FDXScaleGranularity symbol: PerTensor|PerToken|PerChannel|PerBlock (FDX §6.2)
    role: ~                     # activation|weight (FDX ScalePair role)
    scale_operand: ~            # role of the SEPARATE-GRAPH-INPUT scale operand, when the ABI
                                # takes the scale as its own input (§3.9). MUTUALLY EXCLUSIVE with
                                # a sidecar-bundled FDXQuant.scale_buffer — each scale in ONE place.
  sub_byte: ~                   # logical_dtype code when base carries opaque uint8 (FDX §6.1)
  # --- symbolic / affine live extent (uses FDX FDXExtent — §4.5; FDX affine addition) ---
  symbolic_extent: tolerated    # rejected|tolerated|required (uses FDX FDXExtent Scalar/Range/Affine)
  extent_kind: ~                # rejected|scalar|range|affine — the FDXExtentKind this operand
                                # tolerates on its symbolic axis (FDX §6.4 / affine addition).
                                # `affine` ⇒ the live extent is an FDXAffine (c0 + Σ cᵢ·SymIdᵢ);
                                # the kernel reads the resolved value from the SymEnv (§4.5).
  # --- paged / indexed-residency (gather) operand (FDX gather addition; §3.9) ---
  gather:
    kind: ~                     # ~ | paged_blocks — FDX FDXIndexedResidency.kind symbol
                                # (FDX_GATHER_NONE | FDX_GATHER_PAGED_BLOCKS); §6.9 (FDX gather addition)
    block_table: ~             # role of the block-table operand (the [B, max_blocks_per_seq] U32
                                # input the FDX FDXBlockTable references); a SEPARATE accept.input
    context_lens: ~            # role of the per-sequence live-length operand ([B] U32); a SEPARATE
                                # accept.input, OR ~ when the live length is one SymEnv symbol
# --- placement ---
device: Vulkan                  # inherited from provider front-matter unless overridden
substrate: VulkanBuffer         # FDX substrate class (HostBytes|CudaUntyped|VulkanBuffer|MetalBuffer)
```

The **dispatch key** (P5) is derived from `(op_kind, [each operand's dtype + quant facts in
order, then outputs], backend, kernel_source)`. For a plain kernel the quant facts are empty and
the key collapses to today's `(OpKind, [DType…], BackendId)` + `kernel_source` (§12.1). For a
quant kernel, `fdx.quant` enriches the per-operand dtype slot so the key distinguishes
`(QMatMul, A=F32×PerToken, W=Q4_0)` from `(QMatMul, A=F32×PerTensor, W=Q8_0)` — matching the flat
GGML `Capability` tokens and the affine `(op, lhs_dtype, lhs_gran, rhs_dtype, rhs_gran)` key
(digest §9). **The key codes are FDX's** (§0): FKC encodes the same `(family, ggml_dtype |
(granularity, role))` form FDX §6.2 defines, by referencing FDX, not by transcribing a second
table.

### 3.3 The per-kernel structured schema (` ```fkc ` block)

```yaml
# ===== identity =====
kernel: <string>            # unique within the file; the diagnostic kernel name
op_kind: <OpKind>           # the Fuel OpKind this kernel implements (e.g. AddElementwise, MatMul)
fused_op: ~                 # OR a FusedOpId name (e.g. FLASH_ATTN); exactly one of op_kind|fused_op
blurb: <string>             # one line; MUST equal the prose blurb (§10.11)
backend: <BackendId>        # inherited from front-matter unless overridden
kernel_source: <string>     # inherited; the BindingEntry.kernel_source tag
entry_point: <symbol-id>    # symbolic ref into provider link_registry → KernelRef (P9, §12.6)
kernel_revision_hash: <hex> # OR "auto" to derive from entry_point + revision_base (§4.7)

# ===== accept contract (§3.6) =====
accept:
  inputs:                   # ordered list of tensor descriptors (§3.2)
    - { name: lhs, ... }
    - { name: rhs, ... }
  op_params: <OpParamsSchema>   # which OpParams|FusedOpParams variant + field constraints (§3.7)

# ===== return contract (§3.6) =====
return:
  outputs:                  # ordered list of OUTPUT descriptors (§3.2) + return rules (§5)
    - name: out
      dtype_rule: passthrough(lhs)     # §5.1
      shape_rule: same_as(lhs)         # §5.2
      layout_guarantee: contiguous     # §5.3
      aliasing: none                   # §5.4
  bundle: ~                 # OR a list of bundle slot specs for multi-output (§5.5)

# ===== capability + cost + precision + determinism (§4) =====
caps:
  awkward_layout_strategy: requires_contiguous   # §4.3
  fast_paths: [ ... ]       # §4.2 declared fast-path predicates
  in_place: false           # §4.6
  alignment_bytes: 16       # mirrors BackendCapabilities.required_alignment
  access_granularity_bits: 32

cost:
  provenance: declared      # declared | judge_measured — REQUIRED, both first-class (§4.4)
  class: cheap_elementwise  # §4.4 relative cost class (coarse bucket)
  flops: "n"                # symbolic expr over shape/param symbols (§4.4)
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: 4000         # launch overhead (Vulkan command buffer submit)
  memory:                   # per-tier footprint (§4.4) — [consumer-ahead] beyond device_bytes
    device_bytes: "n * dtype_bytes"   # output alloc (executor pre-allocates)
    host_bytes: 0
    disk_bytes: 0

precision:                  # §4.8 → PrecisionGuarantee
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false            # false ⇒ UNAUDITED; true + all-null ⇒ none(reason) audited-no-bound
  notes: "all math f32; f16/bf16 narrow on store; NOT bit-stable cross-hardware."

determinism: same_hardware_bitwise  # §4.9: bitwise | same_hardware_bitwise | nondeterministic
```

### 3.4 Dtype vocabulary

Dtype names are the Fuel `DType` set, one-to-one with the FDX `logical_dtype` table
(`dlpack-extension.draft.md` §6.1; **FDX owns the numeric codes** — §0): `U8, I8, U32, I16, I32,
I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0`. Shorthands expand at import:

| `dtype_class` | expands to |
|---------------|------------|
| `float` | `[BF16, F16, F32, F64]` (+ `F8E4M3` only when the kernel lists it) |
| `int` | `[I8, I16, I32, I64]` |
| `uint` | `[U8, U32]` |
| `any` | every dtype the kernel's `accept.inputs[i].dtypes` enumerates |

Sub-byte dtypes (`F4`, `F6E2M3`, `F6E3M2` — `size_in_bytes()==0`) MUST be paired with an
`fdx.sub_byte` code and an `fdx.quant` block (bit-width + packing come from FDX, never from
`size_in_bytes`). `F8E8M0` is the canonical MX block-scale dtype (digest §8).

**Multi-dtype fan-out.** A section whose operand(s) *vary* — enumerate more than one dtype (a
compact family chassis like elementwise-unary's `[F32, F64, BF16, F16]`), or a `dtype_class` that
expands to more than one — **fans out at import into one binding per fanned dtype** (NOT a single
"first-dtype representative"; a section keyed on a dtype *list* registers the whole family). The
rules the importer applies (`fuel-dispatch/src/fkc/lower.rs::assemble_dtype_variants`):

- The **fan-out dtype set** is the enumerated list of the operand(s) that vary. **All varying
  operands MUST enumerate the same list in the same order**; if they disagree it is a typed
  `FanoutDtypeMismatch` error — the importer never silently picks one operand's list. (A genuine
  *multi-axis* contract — a mixed-precision matmul, or an indexing op with an independent
  data-dtype axis and index-dtype axis — is therefore *describable but not-yet-fannable* by this
  uniform importer: it surfaces the typed error and is deferred, awaiting a cartesian / per-axis
  fan-out follow-up.)
- Per fanned dtype `dt`, each **input** operand contributes its dtype at that variant — a *fixed*
  (single-enumerated) operand its one dtype (e.g. `where`'s `cond` = `U8`), a *varying* operand
  `dt`. Then **outputs**: `fixed(D)` → `D`; `passthrough(role)` → the dtype of the input operand
  *named* `role` at that variant (so `where`'s `passthrough(a)` mirrors operand `a` = `dt`, NOT the
  first input `cond`). The binding key stays inputs-in-order then outputs.
- A section may return `return.outputs` **or** a `return.bundle` (Option C multi-output: one
  packed buffer, §5.5). A `return.bundle` contributes its **primary (first) slot's** dtype to the
  key tail — the one dtype the single output buffer is tagged by (per the `KernelRef` multi-output
  contract: the key "describes inputs + the bundle's primary dtype only") — derived through the
  *same* `dtype_rule`/`passthrough` path as a regular output. So a 5-input `selective_scan` whose
  bundle is `passthrough(u)` keys `[T; 6]` (5 inputs + the one bundled output slot), matching the
  as-built `SelectiveScan` binding.
- A **fanning** section's `entry_point` is a **BASE** symbol (no dtype suffix, e.g.
  `fuel_cpu_backend::byte_kernels::relu`); per fanned `dt` the importer resolves
  `<base>_<dtype_suffix>` through the `LinkRegistry`, where `<dtype_suffix>` is the canonical
  lowercase `DType` spelling (`F32`→`f32`, `BF16`→`bf16`, `U8`→`u8`, `F8E4M3`→`f8e4m3`, … — the
  same spelling the byte-kernel `ep!` macro uses). A **non-fanning** (all-fixed, single-variant)
  section keeps its specific `entry_point` and resolves it AS-IS (so a per-`(op,dtype)` thunk like
  `add_f32` stays `add_f32`). Revision/cost/precision/caps are per-section (shared across variants);
  only the per-variant `dtypes` + resolved kernel differ.
- Fused (`fused_op`) sections do **not** fan out in this slice: a multi-dtype fused section
  registers only its representative (first) dtype today (a follow-up fans the fused registry).

**GGML dtype names are the as-built `GgmlDType` variant names, matched by numeric code.** The
canonical set (`fuel-core-types/src/quantized.rs`) is:
`F32(0), F16(1), Q4_0(2), Q4_1(3), Q5_0(6), Q5_1(7), Q8_0(8), Q8_1(9), Q2K(10), Q3K(11),
Q4K(12), Q5K(13), Q6K(14), Q8K(15), BF16(30)`. **There is no `Q4_K_M` / `Q4KM` `GgmlDType`
variant.** `Q4_K_M` is the *file-format (GGUF) name* for a mixed-precision K-quant whose storage
dtype is `GgmlDType::Q4K` (code 12); the "medium" mixed variant is a kernel/dispatch distinction,
not a separate storage dtype. The op-level distinction is carried by the `Capability` token
`MatMulQ4KM` / `DequantizeQ4KM` (`capability.rs`), **not** by a `GgmlDType`. The mapping to keep
straight:

> `Q4_K_M` (GGUF file-format name) → `GgmlDType::Q4K` (code 12) → op-level `Capability::MatMulQ4KM`.

`fdx.quant.ggml_dtype` therefore takes a `GgmlDType` **variant name** (`Q4K`, never `Q4_K_M`),
and the importer matches it to the as-built variant by its numeric code (FDX §6.2). A contract
that writes `Q4_K_M` in the `ggml_dtype` slot fails §10.6 (`QuantIncoherent`).

### 3.5 Shape/rank constraint vocabulary (`shape_constraint`)

A small predicate language, all build-time checkable (§10):

- `same_as=<role>` — element-shape equal to another operand/output.
- `same_rank=<role>` — rank equality only.
- `rank=<n>` / `rank=2..=4` — exact / range.
- `broadcast_to=<role>` — NumPy/PyTorch broadcasting compatible with `<role>`.
- `last_dim_eq=<role>` / `dim[i]=<expr>` — per-axis equalities (e.g. matmul `k`).
- `divisible(dim[i], <expr>)` — e.g. GQA `hq % hkv == 0`, block-quant `k % block == 0`.
- `capacity_ge(dim[i], sym)` — for symbolic axes: capacity ≥ a symbol's `min` (FDX §6.4).
- free text in `notes:` for anything not yet vocabularized (importer warns, does not reject).

### 3.6 Accept vs return semantics — summary

- **Accept** (per input operand): accepted **dtypes**; accepted **layouts** (the five-flag set incl.
  `reverse_strides`, §4.1 / §4.1.1) plus an optional per-operand **awkward-layout strategy**
  (§4.3.1, overriding the kernel-wide `caps.awkward_layout_strategy`); **shape/rank** constraints
  (§3.5); **FDX requirements** (quant family/format/granularity, sub-byte dtype, symbolic-extent
  tolerance incl. affine, paged/gather); **device/substrate**; the **op-param schema** (§3.7).
  Field-by-field semantics in §4.1, §4.3.1, §4.5.
- **Return** (per output): **dtype rule** incl. passthrough (§5.1); **shape rule** (§5.2);
  **layout guarantee** (§5.3); **aliasing / in-place** (§5.4); and for multi-output ops the
  **bundle slot specs** (§5.5).

### 3.7 Op-param schema (`OpParamsSchema`)

Names the params variant the kernel consumes plus per-field constraints. **For a primitive
`op_kind` contract the variant is a `fuel-dispatch::kernel::OpParams` variant; for a `fused_op`
contract it is a `fuel_graph::registry::FusedOpParams` variant.** The two namespaces are
distinct; §10.7 checks the variant against the correct namespace depending on which of
`op_kind` / `fused_op` the contract declares. Example for matmul (primitive):

```yaml
op_params:
  variant: Matmul          # OpParams::Matmul
  fields:
    m:  { kind: usize }
    n:  { kind: usize }
    k:  { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
    lhs_batch_dims: { kind: "Vec<usize>" }
    rhs_batch_dims: { kind: "Vec<usize>", constraint: "GQA-divisible vs lhs_batch_dims" }
```

For a symbolic dynamic scalar param (FlashAttn `k_len`, WriteSlice offset, Rope position) the
field declares `kind: DynScalar` so the importer knows it rides the `SymEnv` (FDX §6.4), and
whether the kernel takes a *resolved scalar* vs a *mask tensor* (length-vs-mask is a lowering
decision, never a tensor property — digest §6).

### 3.8 YAML subset and hand-authoring rules (avoid the landmines)

The ` ```fkc ` block is YAML 1.2 **core schema**, parsed in a restricted mode so that
hand-authoring never silently changes a value's type. The importer enforces:

- **Quoted scalars for all enum / token / dtype / symbol values.** `family: none` is the
  *string* `"none"`, not YAML-null; `granularity: PerChannel` is a token; `ggml_dtype: Q4_0` is a
  string, **never** parsed as a number. The schema declares each field's expected scalar type and
  the importer rejects a type mismatch (`BadScalarType`) rather than coercing.
- **The Norway problem is disarmed.** YAML 1.1 turns `no`, `off`, `n`, `y`, `yes`, `on` into
  booleans. FKC parses in YAML-1.2-core mode where only `true`/`false` are booleans, and *all
  token fields are read as strings regardless*. A `name: n` operand role stays the string
  `"n"`; a hypothetical `country: NO` stays `"NO"`.
- **No sexagesimal / no implicit numeric coercion of versions or hashes.**
  `kernel_revision_hash` and any hex literal are **quoted strings** (`"8f3c1a"`); a bare `1:30`
  is rejected, never read as 90. Cost coefficients live inside quoted expression strings
  (`flops: "2 * m * n * k"`), so YAML never tokenizes the operators.
- **Tabs are a hard error.** YAML forbids tab indentation; the importer reports
  `YamlTabIndent` with line/column rather than producing a confusing parse error.
- **Booleans are literal `true`/`false`** for `audited`, `in_place`, `causal`,
  `requires_ext`, and the layout flags' coherence checks; everything else enum-valued is a
  quoted token.
- **Expressions are strings, validated by FKC's own parser** (§4.4 / §10.8), not by YAML. This
  keeps `n % 4 == 0` and `k_len <= sk` out of YAML's hands entirely.
- **Anchors / aliases / merge keys (`<<`) are disabled.** A contract is flat data; YAML
  reference machinery is rejected (`YamlAnchorDisallowed`) so a diff reviewer sees every value
  literally and the importer never has to resolve a graph.

A linter (`fkc fmt --check`) re-emits the canonical quoted form and diffs it against the source,
so a hand-authored file that *parsed* but used a risky unquoted form is flagged in CI before it
can bite.

### 3.9 Paged / indexed-residency (gather) operands and the scale single-place rule

Two FDX additions (2026-06-17) expand the tensor vocabulary an operand can declare. Both are
described **in FDX terms by symbol** — FKC re-numbers nothing (§0); FDX is the normative owner of
every `FDX_GATHER_*` / `FDXExtentKind` code and the gather buffer roles (FDX gather addition §6.0;
FDX affine addition §6.0).

#### 3.9.1 Declaring a paged-attention (indexed-residency) operand

A vLLM-style paged / blocked KV cache is, in FDX, a **single tensor**: an honest contiguous
`uint8` block-pool base + an `FDXIndexedResidency` sidecar block that re-interprets the pool via a
per-sequence block table (FDX gather addition §4; `FDX_FLAG_HAS_GATHER`, `kind =
FDX_GATHER_PAGED_BLOCKS`). The pool, the `[B, max_blocks_per_seq]` U32 block table, and the `[B]`
U32 context-lens are FDX buffer-table entries with roles `FDX_BUFFER_POOL` /
`FDX_BUFFER_BLOCK_TABLE` / `FDX_BUFFER_CONTEXT_LENS` (FDX gather addition §5).

An FKC contract for a kernel that consumes such a cache declares, on the **pool** operand. PagedAttn
is a **FUSED op**, so the param carrier is **`FusedOpParams::PagedAttn`** (`FusedOpId(13)`,
`fuel-graph/src/registry.rs:241`), **not** an `OpParams::PagedAttn` variant (there is none — the
`PagedAttn` shape data lives on `KernelRef::PagedAttn`, `fuel-dispatch/src/kernel.rs:320`). The
worked descriptor therefore declares `fused_op: PAGED_ATTN` with `op_params.variant: PagedAttn` in
the `FusedOpParams` namespace (the same shape as the FlashAttn fused example, §8) — never
`op_kind`/`OpParams` (an `op_kind` carrier would fail the §10.7 namespace check). (`OpKind::PagedAttn`
at `fuel-core-types/src/dispatch.rs:168` exists as the op-kind tag but is **not** the param carrier.)

```yaml
- name: k_cache
  dtypes: [F16, BF16, F32]      # the TRUE per-token pool element type (FDX FDXDTypeExt)
  layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted }
  rank: 4                        # physical pool shape [num_blocks, block_size, Hkv, D]
  fdx:
    requires_ext: true           # MEANING_REQUIRES_EXT is mandatory for a paged pool (FDX gather V19)
    symbolic_extent: required    # the per-seq live length is symbolic (context_lens)
    extent_kind: affine          # OR range, when the live length is one SymId (§3.9.2)
    gather:
      kind: paged_blocks         # FDX FDX_GATHER_PAGED_BLOCKS
      block_table: block_table   # ← the role of the SEPARATE block-table accept.input (below)
      context_lens: context_lens # ← the role of the SEPARATE context-lens accept.input (below)
```

and lists the **block table** and **context lengths as ordinary separate `accept.inputs`**, because
the as-built PagedAttn ABI takes them as their own graph inputs — the `KernelRef::PagedAttn` operand
order is `[q, k_cache, v_cache, block_table, context_lens, alibi?]`
(`fuel-dispatch/src/kernel.rs:314-331`, matching FDX §6.9.3 / V21(b); the earlier
`fuel-core-types/src/dispatch.rs:165-168` operand-order citation was wrong and is corrected here):

```yaml
- name: block_table
  dtypes: [U32]
  rank: 2                        # [B, max_blocks_per_seq]
- name: context_lens
  dtypes: [U32]
  rank: 1                        # [B]
  fdx: { symbolic_extent: required }   # per-seq live lengths (data-determined sym)
```

**Single-place rule (this is the load-bearing discipline — it mirrors the scale rule, §3.9.3).**
The block table and context-lens are described in **exactly one authoritative place**. Because the
kernel ABI takes them as **separate graph inputs**, they are **FKC `accept.inputs` operands**; the
operand's `fdx.gather.block_table` / `fdx.gather.context_lens` fields carry **the role names of
those same inputs**, not a duplicate copy of the data. The FDX `FDXBlockTable.table_buffer` /
`FDXIndexedResidency.context_lens_buffer` then point at the **same** buffers (FDX gather addition
§5, single-place rule; cross-checked by FDX validator V21(b) — same buffer-table slot by
index-equality, not a value comparison). FKC does not introduce a parallel
table. When a *future* fully-fused paged op carries the table sidecar-bundled inside the FDX tensor
(no separate operand), the same `fdx.gather` fields describe it with `block_table: ~` /
`context_lens: ~` and the FDX buffer-table entry is authoritative — one place either way.

**Capability gate.** A backend admits the paged operand directly only if it advertises the FDX
gather capability token (`Capability::DlpackExtGather`, FDX gather addition §8/§12); the planner
otherwise inserts an explicit materialize (dense un-paged copy, `IS_COPIED`) per FDX producer
policy §9.1 and prices it from the materialize kernel's FKC contract (§4.3, §9.3 — cost is FKC's
job, FDX carries none). `paged_attn` is its own kernel-contract section (the attention family file,
§9.2); a contract that declares `fdx.gather.kind: paged_blocks` and `extent_kind: affine` is the
canonical paged-attention advertisement.

**[consumer-ahead].** `OpKind::PagedAttn` exists today; the FDX gather descriptor and
`Capability::DlpackExtGather` are the 2026-06-17 FDX addition (no code yet). An importer that
reaches a `gather`-bearing operand before the FDX gather codes land returns `GatherNotYetSupported`
rather than fabricating a descriptor — exactly the `MxNotYetRegistrable` discipline (§6).

#### 3.9.2 Affine live extents on an operand

The FDX affine addition generalizes `FDXExtent` with a third kind, `Affine` (`FDXExtentKind`:
`0=Scalar, 1=Range, 2=Affine`), carrying a bounded combination `value = c0 + Σ cᵢ·resolve(SymIdᵢ)`
(integer coeffs, ≤ `FDX_AFFINE_MAX_TERMS = 4` terms; FDX affine addition §2). This lets the
persistent-decode `k_len = cached_len + new_tokens` be expressed **symbolically over base symbols**
— planned once, evaluated from the `SymEnv` every token, with **no per-pass recompute of a derived
sym** (FDX affine addition §0, §6).

An operand whose symbolic axis is an affine combination declares `fdx.extent_kind: affine`
(alongside `symbolic_extent: required`). Capacity stays a **concrete bound** keyed to the buffer
allocation (`cap_kind = EXPLICIT`, `capacity == base.shape[i]`, the only v1 form; FDX affine
addition §3.3); the affine expression describes only the **live prefix length**, bounded by
capacity and checked `min ≤ value ≤ capacity` at realize (FDX affine addition §3.5 / V14). A
contract states *that* the operand's live extent is affine (`extent_kind: affine`); the affine
coefficients/terms themselves live in the **FDX sidecar** `FDXExtent.affine`, not in the FKC
contract — FKC advertises the kernel's *tolerance* for an affine axis, FDX *describes* the specific
expression (the description/advertisement split of §0). `extent_kind: range` is the single-`SymId`
degenerate case (today's FlashAttn `k_len ≤ sk`, §8); a one-`SymId` axis SHOULD declare `range`,
not `affine` (FDX affine addition §4 canonicalization).

**Capability gate (affine extent vs affine quant — do not confuse the two tokens).** An affine
`extent_kind` is gated by the **same `Capability::DlpackExtSymbolic`** token as any symbolic extent
(FDX §12) — there is **no separate affine-*extent* capability; the kernel that honors symbolic
extents + `FDXSymEnv` also evaluates the `kind=Affine` extent's `c0 + Σ cᵢ·symᵢ`. The distinct
`Capability::DlpackExtAffine` token gates affine-int/float dynamic **quant** (§3.9.3), a different
concept. This is why §3.9.2/§4.5 gate `extent_kind: affine` on `symbolic_extent: required` and name
no capability of their own, while the gather operand (§3.9.1) names `Capability::DlpackExtGather`.

#### 3.9.3 The quant scale-operand single-place rule (RESOLVED 2026-06-17)

The same single-place discipline governs **quant scales**, and the resolved decision is explicit:

> **If a kernel's ABI takes the scale as a SEPARATE GRAPH INPUT, the scale is an FKC
> `accept.inputs` operand — NOT also an FDX `scale_buffer`.** The FDX `FDXQuant.scale_buffer`
> sidecar reference is **only** for sidecar-bundled scales (the GGML-INLINE / MX-separate-buffer
> cases the FDX tensor carries itself). Each scale is described in **exactly one place**, with a
> consistency check.

Concretely:

- **Separate-input scale (dynamic affine quant whose ABI passes the scale tensor as its own
  operand):** declare it as a normal `accept.inputs` entry and name its role in the consuming
  operand's `fdx.quant.scale_operand`. **Do not** also set a sidecar `FDXQuant.scale_buffer` for
  it — the operand is the authority. The importer's coherence check (§10.6) rejects a contract
  that declares both `fdx.quant.scale_operand` *and* a sidecar-bundled scale for the same scale
  (`ScaleDoubleDeclared`).
- **Sidecar-bundled scale (GGML INLINE block scales; MX separate-buffer F8E8M0 per-block scale):**
  the scale rides the FDX tensor (`FDXQuant.scale_placement = INLINE | SEPARATE_BUFFER`,
  `scale_buffer` valid; FDX §6.2). There is **no** FKC scale operand; `fdx.quant.scale_operand`
  stays `~`. This is the existing GGML/MX path (§6, §8).

This is the literal generalization of the gather single-place rule (§3.9.1) and FDX's own
scale-buffer discipline (FDX §6.3): a fact lives in one normative place, and the cross-form
consistency check guarantees the FKC operand and the FDX buffer reference (when both are present
for *different* scales) never disagree.

### 3.10 Describe-only (non-registrable) sections (ADDED 2026-06-18)

A `## ` section may be marked **describe-only** with a single section-level boolean field in its
` ```fkc ` block:

```yaml
registrable: false      # §3.10 — a documentation-only section; NOT registered, op_kind/fused_op
                        # is NOT required to resolve to a real dispatch target. Default: true.
```

`registrable:` is a section-level field on the per-kernel block (sibling of `kernel:` / `op_kind:`
/ `blurb:`), defaulting to **`true`** (every existing contract is registrable; the field is
additively versioned per §11). When a section sets `registrable: false`:

- **It is NOT registered.** The importer parses + validates it as documentation but does **not**
  insert it into the `KernelBindingTable` / `FusedKernelRegistry` (it never produces a
  `ResolvedPrimitive` / `ResolvedFused` and is excluded from `register_into`, §9.3 / §12.5). It
  carries no dispatch decision point, so it can never collide at a key (§10.10) and never
  contributes a `KernelRef`.
- **It does NOT require `op_kind` / `fused_op` to resolve to a real dispatch target.** The
  exactly-one-of-`op_kind`/`fused_op` structural rule (§10.2) and the "the named op resolves"
  rule (§10.3) are **skipped**. `op_kind` may be `~` (absent), or a **descriptive non-dispatch
  token** that names no real `OpKind` (a documentation umbrella name). The op-param namespace
  check (§10.7) is likewise skipped (a describe-only section may name a params variant for prose,
  or omit it).
- **It does NOT require `accept.inputs` (≥1).** The ≥1-input required-field rule (§10.2) is
  **exempted** for a describe-only section: a documentation-only **zero-operand op** is
  legitimate — e.g. a `rand_uniform` / `rand_normal` random fill whose only "input" is
  backend-private RNG seed state (not a graph tensor), so it declares `accept.inputs: []`. A
  zero-input op has no real dispatch carrier (there is no `OpKind` that consumes zero operands),
  which is exactly why it is describe-only in the first place; the exemption is therefore
  describe-only-scoped. **A *registrable* section still requires ≥1 input** — a real dispatch
  target must consume at least one graph operand, so the ≥1-input rule is NOT relaxed for
  registrable sections.
- **Its descriptive fields are STILL validated as documentation.** Everything that is a *fact
  about the described family* still runs: the FDX-subset drift-guard (§10.16 — every `dtypes:` /
  `fdx.quant.family` / `granularity` / `ggml_dtype` token must still be a real FDX-table member,
  so the docs cannot name a dtype that does not exist), layout coherence (§10.4 — the documented
  five-flag set must still be internally consistent), the awkward-layout coherence (§10.5), and
  quant coherence where a `fdx.quant` block is present (§10.6, including the MX/AFFINE_BLOCK
  not-yet-registrable note as a documentation finding). A describe-only section is *honest docs*,
  not an escape hatch from coherence — it only drops the *dispatch-resolution* checks that
  presuppose a real registrable kernel.

**When to use it.** Two cases, both about description without a dispatch target:

1. **Chassis / family umbrellas.** A section that documents a *shared algorithm* backing many
   per-`(op, dtype)` thunks — e.g. `## binary` (Add/Sub/Mul/Div/Max/Min/Pow/Rem) or `## unary` —
   where each concrete op is its own registrable section. The umbrella is prose + the shared
   layout/precision/cost description; it should not name one op arbitrarily as a "representative"
   (which would either double-register that op or mislead the optimizer). Mark it
   `registrable: false` and let `op_kind` be the descriptive umbrella name (`binary`) or `~`.
2. **Ops with no dispatch `OpKind`.** A token that documents an operation Fuel performs as a
   *graph rewrite / view / lowering*, not via a dispatch `OpKind` entry — e.g. `Im2Col` /
   `Im2Col1d` / `Col2Im1d`, the pooling ops, `Upsample*`, `Transpose` / `Permute` (metadata-only
   views), `ArgSort`, `Conv2dSimple`, `Dequantize*` / `QuantizeQ8_0`, `AddAssignScaled`, and any
   other token that has **no real `OpKind`** in the as-built `fuel-core-types::dispatch::OpKind`
   enum. These have descriptive contracts (dtypes, layout, precision) worth publishing, but there
   is no dispatch key to register against. Mark them `registrable: false`.

> **Decide describe-only vs RENAME by checking the as-built enum.** A token that is a *typo* for a
> real `OpKind` (the enum genuinely has the variant) is **renamed to the exact variant name**, not
> marked describe-only: `Cumsum → CumSum`; `Clamp → ClampInplace` / `ClampElementwise` (the real
> inplace/elementwise variant); `PowI → PowIInplace` / `PowIElementwise`; `ConvTranspose1D` /
> `ConvTranspose2D` → the real `OpKind` if one exists (`ConvTranspose2D` does;
> `ConvTranspose1D` does not — the latter is describe-only). Only a token with **no** real
> `OpKind` becomes describe-only. **Never invent an `OpKind`, and never relax a validator to hide
> a defect** — describe-only is a *typed, declared* posture, not a silent skip.

---

## 4. Capability + cost + precision + determinism (the linchpin)

This is the block that lets the planner choose **contiguize-vs-strided-vs-materialize** and rank
admissible candidates. It exists because backends advertise and the planner decides (governing
principle): the kernel states facts; the planner makes the choice.

### 4.1 Accepted layouts — the explicit five-flag set (richer than one bool)

Per input operand, `layout:` carries five independent facts (G3). This replaces today's single
`KernelCaps.strided_input: bool` with the dimensions the digest (§2) and the prompt require:

| flag | values | meaning |
|------|--------|---------|
| `contiguous` | `required` / `accepted` / `n/a` | the kernel can consume dense row-major input |
| `strided` | `accepted` / `rejected` | walks arbitrary (non-negative) strides (transpose/slice as metadata-only views) |
| `broadcast_stride0` | `accepted` / `rejected` | tolerates a stride-0 axis (broadcast without materialize) |
| `start_offset` | `accepted` / `rejected` | tolerates a non-zero `byte_offset` / view base |
| `reverse_strides` | `accepted` / `rejected` | tolerates **negative** strides (a reversed iteration order, e.g. `Op::Flip`) — zero-copy (§4.1.1) |

These five map directly onto FDX's layout vocabulary (a strided/broadcast/offset/reversed operand
is an FDX-describable `DLTensor` with non-trivial / signed `strides` and a `byte_offset` pointing
at the iteration-first element). The planner reads them to decide, **per operand**, whether to
insert an `Op::Contiguize` (or a non-negative-stride normalizing copy, §4.1.1) before the kernel —
and to *cost* that insertion (§4.3, §4.4). Today's importer projects `(strided &&
broadcast_stride0)` onto the one `KernelCaps.strided_input` bool, routes `start_offset` through
auto-Contiguize, and treats `reverse_strides` as **not declared** (so a negative-stride operand is
normalized to a non-negative copy until `KernelCaps` grows the flag; the as-built behavior, §12.2).
The full five-flag set is retained for forward use as `KernelCaps` grows.

These five flags state *what layouts the operand accepts*; the operand's **strategy** for a layout
it cannot fast-path (contiguize-vs-strided-vs-internal) is the separate, optional, per-operand
`layout.awkward_layout_strategy` field (§4.3.1), which overrides the kernel-wide
`caps.awkward_layout_strategy` for that operand.

> **Convention examples** (sanity-checked against the inventory):
> - fuel-cpu-backend `binary`: `{contiguous: required, strided: rejected, broadcast_stride0:
>   rejected, start_offset: rejected, reverse_strides: rejected}`.
> - Vulkan `unary` (f32): `{contiguous: accepted, strided: accepted, broadcast_stride0: accepted,
>   start_offset: rejected, reverse_strides: rejected}`.
> - Metal `unary_kernel_strided`: all `accepted` (offset-capable via `BufferOffset`; a
>   signed-stride walk handles `reverse_strides`).

#### 4.1.1 `reverse_strides` — negative strides are first-class (FDX), zero-copy when declared

A negative stride means an axis is walked **backwards**: the operand's `byte_offset` points at the
iteration-*first* element (the highest physical address along that axis) and the stride is negative
toward lower addresses. This is exactly what `Op::Flip` produces, and Fuel's `Layout::flip` already
maintains the invariant the OOB-safety argument needs — `byte_offset` is the iteration-first
element while the buffer's own `start_offset` stays **non-negative**, so the touched byte range
never precedes the allocation. Negative strides are therefore **first-class in the tensor
description**, reversing the earlier universal "no negative strides" rule:

- **FDX *describes* negative strides.** The base `DLTensor.strides` are `int64`; an FDX export of a
  flipped/reversed view carries the signed strides directly, with the OOB validator computing the
  touched `[min, max]` byte-address range over the **signed** strides (per axis: the max
  contribution is `(dim-1)*stride` when `stride > 0` else `0`; the min contribution is
  `(dim-1)*stride` when `stride < 0` else `0`) and checking it lies within `[data, data +
  size_bytes)`. This is the same invariant `Layout::flip` upholds.
  **[cross-spec dependency]:** the sibling FDX spec presently still bans negative strides on export
  (its validator V13 / §3.2 "non-negative on FDX export"). FDX is the normative tensor-axis owner
  (§0), so the V13 reversal — *FDX describes signed strides; the OOB validator uses the signed
  `[min,max]` range above* — is an **FDX-side edit** to be applied in the FDX pass; this FKC
  revision adds only the *kernel-side capability* to accept what FDX will describe. Until the FDX
  edit lands, a `reverse_strides: accepted` declaration parses and registers (it is a kernel fact),
  but no FDX producer will hand the kernel a negative-stride tensor — the same [consumer-ahead]
  posture as the gather/affine additions.

- **Acceptance is a per-kernel declared capability.** `layout.reverse_strides: accepted` states the
  kernel walks signed strides correctly (a backward axis walk costs the same as a forward one — no
  fixup). A kernel that omits it (or sets `rejected`) does **not** accept negative-stride operands.

- **Normalization is NOT universal — only for incapable consumers.** The planner normalizes a
  negative-stride operand to a non-negative materialized copy (an `Op::Contiguize`-class kernel,
  costed from its own FKC contract, exporting `IS_COPIED` on the FDX side) **only** when the chosen
  consumer cannot handle it: a kernel whose contract does **not** declare `reverse_strides`, **or**
  a bare standard-DLPack export to a non-cooperating *external* ecosystem. It is **never** inserted
  for a capable internal kernel. An internal zero-copy `Op::Flip` feeding a capable kernel
  (`reverse_strides: accepted`) is preserved as a metadata-only view — no copy, no cost beyond the
  flip's own (free, class `free`) view rewrite.

This is the same description-vs-decision split as everywhere else: FDX *describes* the signed
strides, the kernel *declares* whether it accepts them, and the planner *decides* (and *costs*) the
normalizing copy only when a consumer in the chosen route cannot. `reverse_strides` sits alongside
`strided` as an independent fact — a kernel may accept arbitrary non-negative strides
(`strided: accepted`) yet reject reversed ones (`reverse_strides: rejected`), so the two are not
collapsed (§10.4 coherence).

### 4.2 Declared fast-path predicates

`fast_paths:` is a list of predicates over shape/layout/params that hit a fast code path. Each
is a `{ when: <predicate>, effect: <relative cost multiplier or class> }` pair. This is the
linchpin for honest costing: the planner can tell whether a given concrete shape will hit the
fast path *before* dispatch.

```yaml
fast_paths:
  - { when: "all_inputs_contiguous", class: cheap_elementwise }   # both-contig path
  - { when: "n % 4 == 0", note: "vec4 packed store; ~1.0x" }
  - { when: "strided", class: strided_elementwise }               # slower path declared
```

Predicate vocabulary reuses §3.5 plus `all_inputs_contiguous`, `any_input_strided`,
`any_input_broadcast`, `any_input_reversed` (≥1 operand carries a negative stride — for a kernel
that accepts `reverse_strides` but wants to declare a distinct cost class for the backward walk),
`dtype == <D>`, `dim[i] % <k> == 0`, `k_len == sk` (the FlashAttn static-fast case),
`groups == 1`, `depthwise`.

**Structure-granularity predicates (the tight-vs-generic axis).** A fast-path predicate may also
express the three *structural* facts a structure-specialized kernel folds into constants
(coordinate-unravel, vectorization, broadcast-hoist; see §4.12) — so that such a kernel imports as
a **tight-predicate contract** and a "miss" (the planner's best match at a key is a *generic*
contract) falls out of normal predicate matching, no special case needed:

- **Divisibility buckets** on an operand's inner (fastest-varying) extent:
  `inner_div(<role>) % 16 == 0` / `% 8` / `% 4` / `% 2` / `any` — the bucket ladder
  `%16 ⊐ %8 ⊐ %4 ⊐ %2 ⊐ any` (a `%16` operand also satisfies `%8`, etc.; `any` is the
  no-constraint floor). A kernel whose contract carries `inner_div(out) % 16 == 0` is admissible
  *only* when the concrete inner extent divides 16, so the planner can tell **before dispatch**
  whether the shape lands in that tight cell.
- **Vec-width** the kernel requires/uses: `vec_width(<role>) >= v4` over the ladder
  `scalar < v2 < v4 < v8` (a vectorized store path is admissible only when align/stride/dtype
  permit that width — derived exactly as Baracuda's `VecWidth` is, §4.12).
- **Inner-contiguous:** `inner_contiguous(<role>)` — the fastest-varying axis has stride 1 even
  if outer axes are strided (Baracuda's `Contiguity::InnerContig`, distinct from fully `Contig`).
  This is the common "strided outer, packed inner" case a vectorized kernel still fast-paths.

A kernel that declares **none** of these (its widest predicate is `any` / generic strided) is a
**generic contract**; one that declares them is a **tight-predicate contract**. The planner
matches a concrete shape against the tightest admissible contract at the key; when the only
admissible contract is the generic one (the tight cell isn't registered), that *is* a structural
miss (§4.12) — surfaced by the same matching pass, not a bolt-on.

### 4.3 Awkward-layout strategy — and why Contiguize/Dequantize are themselves FKC kernels

`caps.awkward_layout_strategy` is the single most planner-relevant capability fact. It states
**which strategy the kernel uses for layouts it does not have a fast path for** — making the
contiguize-vs-strided-vs-materialize decision visible and costable (the prompt's core
requirement):

| value | meaning | planner consequence |
|-------|---------|---------------------|
| `requires_contiguous` | rejects non-contiguous; needs dense zero-offset input | planner MUST insert `Op::Contiguize` for any non-contiguous operand and **adds the Contiguize kernel's cost** to the candidate (see below) |
| `handles_strided` | walks strides directly; no fixup needed | planner passes the strided view through; no contiguize cost |
| `contiguize_internally` | accepts strided but **copies to dense inside the kernel** | planner attributes the contiguize cost to *this kernel* (it is part of the kernel's `bytes_moved` / `cost`), and MUST NOT insert a separate `Op::Contiguize` — the fused contiguize+op is costed honestly as one unit |

`contiguize_internally` is the constitutionally-careful case: a kernel *may* contiguize
internally (some do — MKL matmul relies on the executor's auto-Contiguize; some quant kernels
contiguify callers), but it must **declare** it so the optimizer is not blind to a hidden copy.
Silent internal contiguize without this declaration is a contract violation (digest §3: no silent
materialization).

> **`Op::Contiguize` and `Op::Dequantize` are ordinary FKC kernels.** The "planner adds the
> Contiguize cost" above is not a built-in magic number: it is **the cost of an actual kernel
> with its own FKC contract**. Every backend that has a `requires_contiguous` kernel also ships
> an `Op::Contiguize` contract (CPU + each GPU backend), and every dequant-on-the-way-in path is
> an `Op::Dequantize` contract. The planner's contiguize-vs-strided comparison is therefore a
> literal sum of two `CostEstimate`s drawn from two FKC contracts — see the worked example in
> §4.4. This is what makes the "costable" claim concrete rather than aspirational: there is no
> cost the planner has to know that is not in *some* kernel's contract.

#### 4.3.1 Awkward-layout strategy is PER OPERAND (a kernel may handle strided q/k/v while requiring contiguous aux)

`caps.awkward_layout_strategy` is the **kernel-wide default**; a kernel whose operands have
*different* layout tolerances declares the strategy **per operand**, on the operand descriptor, as
`layout.awkward_layout_strategy`. This is the load-bearing case for fused attention: a
flash-attention kernel walks **strided `q`/`k`/`v`** directly (`handles_strided`) but its **aux
operands are contiguous-only** — `alibi_slopes`, `seqlens`/`context_lens`, the block table — small
metadata vectors the kernel reads with a dense pointer (`requires_contiguous`). A single kernel-wide
`handles_strided` cannot describe that kernel, and a kernel-wide `requires_contiguous` would force a
needless contiguize of the (already-strided-capable) q/k/v; **per-operand strategy** is the only
honest description.

Resolution and precedence:

- **Per operand:** an operand descriptor (§3.2) may carry `layout.awkward_layout_strategy ∈
  {requires_contiguous, handles_strided, contiguize_internally}`. When present, it is authoritative
  **for that operand**; when absent, the operand inherits `caps.awkward_layout_strategy` (the
  kernel-wide default), which in turn defaults to `requires_contiguous` if the kernel declares
  neither. The kernel-wide field is retained (it is the common case — most kernels treat all
  operands alike) and is exactly the "all operands inherit this" shorthand.
- **Coherence is per operand, against that operand's own layout flags (§10.5):** an operand whose
  effective strategy is `handles_strided` MUST have `strided: accepted` on *that operand*; an
  operand whose effective strategy is `requires_contiguous` MUST have `contiguous: required` on
  *that operand*; `contiguize_internally` folds *that operand's* copy into the kernel's declared
  `bytes_moved` (§4.4). The check no longer demands a single tolerance across **every** input.
- **Planner consequence is per operand:** the planner inserts `Op::Contiguize` (and sums its
  FKC-contract cost, above) **only for the operands whose effective strategy is
  `requires_contiguous` and whose incoming view is non-contiguous** — e.g. it contiguizes a strided
  `alibi_slopes` but passes a strided `q` straight through. `contiguize_internally` operands never
  get a separate `Op::Contiguize` (the copy is the kernel's, §4.3); `handles_strided` operands never
  get one. The decision is made independently per operand from each operand's effective strategy ×
  its incoming layout — backends advertise (per operand), the planner decides (per operand).

**Worked: flash-attention mixed tolerance.**

```yaml
accept:
  inputs:
    - { name: q, dtypes: [F16, BF16], layout: { contiguous: accepted, strided: accepted, awkward_layout_strategy: handles_strided } }
    - { name: k, dtypes: [F16, BF16], layout: { contiguous: accepted, strided: accepted, awkward_layout_strategy: handles_strided } }
    - { name: v, dtypes: [F16, BF16], layout: { contiguous: accepted, strided: accepted, awkward_layout_strategy: handles_strided } }
    - { name: alibi_slopes, dtypes: [F32], layout: { contiguous: required, strided: rejected, awkward_layout_strategy: requires_contiguous } }
    - { name: seqlens,      dtypes: [U32], layout: { contiguous: required, strided: rejected, awkward_layout_strategy: requires_contiguous } }
caps:
  awkward_layout_strategy: handles_strided   # kernel-wide DEFAULT (q/k/v); aux operands override to requires_contiguous above
```

The kernel-wide `handles_strided` covers q/k/v; the two aux operands override to
`requires_contiguous`. The planner contiguizes only a non-contiguous `alibi_slopes`/`seqlens`,
never q/k/v — which a single kernel-wide flag could not express without a needless q/k/v copy.

### 4.4 Cost — a vector contribution, optionally symbolic, with TWO compile targets

**Cost provenance — `declared` or `judge_measured`, both first-class, never a hidden gap.** Every
cost block carries a required `provenance:` field with exactly one of two values, and **both are
first-class and visible**:

- `provenance: declared` — the cost coefficients are **author-declared**: the kernel provider's
  best estimate of `flops`/`bytes_moved`/`overhead_ns`/`memory` (pessimistic upper bound, "when in
  doubt round up"). An author-declared cost is a **prior** — a starting value the Judge later
  refines (it is *not* a final number, just as precision fields are seeds the Judge audits, §4.8).
- `provenance: judge_measured` — the cost coefficients were **populated/last-updated by empirical
  measurement** (the Judge's calibration), and the contract records that explicitly. A
  judge-measured cost is equally first-class: it is not "more official" than a declared one, it is
  simply a cost whose provenance is measurement rather than authorship, and the contract says so.

The two are not a quality ranking; they are a **provenance** label so that every cost the optimizer
reads carries its origin, and **no cost is silently a placeholder** (the 01 visibility gate — a
fact the optimizer reasons over must be visible, here including *where the number came from*). A
**bare, placeholder, or omitted-without-marker** cost is a **lint failure** (§10.8a): a contract
must either declare its cost (and mark it `declared`) or mark it `judge_measured`; it may not ship a
cost the optimizer cannot tell the origin of, and it may not ship a zero/sentinel "TODO" cost under
either marker. (A genuinely free metadata-only op declares `class: free` with zero coefficients and
`provenance: declared` — that is an honest declaration, not a placeholder.)

> **Judge-agnosticism caveat (the Judge is mid-rebuild — FKC stays agnostic to its internals).**
> FKC depends only on two facts about the Judge: **(1) the Judge exists, and (2) it refines /
> bootstraps cost** — an author-declared cost is a prior it improves, and a `judge_measured` cost is
> one it has populated. FKC does **not** depend on *how* the Judge measures, calibrates, or
> cross-checks: no specific mechanism is named or assumed here (the Judge is actively being
> rebuilt). The `declared`↔`judge_measured` distinction is the entire contract surface the Judge
> needs from FKC; the inverse direction (binding entries are re-readable and the Judge may overwrite
> a `declared` cost with a refined one, flipping the provenance to `judge_measured`, §11) is the
> only coupling, and it is stated in terms of "the Judge refines cost," never a particular
> algorithm. Precision is governed the same way: author-declared and Judge-audited (§4.8).

This `declared` → `judge_measured` provenance is the correct substrate for the adaptive optimizer's
ground-truth fitness signal (G7, [10-decisions-log](../architecture/10-decisions-log.md)): a JIT-
synthesized kernel arrives with a `declared` cost prior, and its empirical *winning* — entering an
optimized plan under cost-gated selection, the route picker's adopt/reject call — is exactly the
measured posterior the Judge records by flipping the binding to `judge_measured`. FKC supplies the
honest cost surface; Fuel-the-strategist makes the adoption decision.

The cost block yields the per-node contribution to the optimizer's **cost vector** (digest §5):

- **time axis:** `flops`, `bytes_moved`, `overhead_ns` — these three map onto today's
  `CostEstimate { flops, bytes_moved, kernel_overhead_ns }` (§12.3). Convention: **pessimistic
  upper bound** ("when in doubt, round up").
- **memory axis [consumer-ahead]:** `memory: { device_bytes, host_bytes, disk_bytes }` — the
  **per-tier** footprint (digest §5/§11). Today's `CostEstimate` has no per-tier memory field;
  the importer folds `device_bytes` into the cost-vector memory axis as the consumer lands, and
  ignores the rest until then (size-prefixed forward-compat).
- **`class:`** a coarse relative cost class (a discrete bucket for fast frontier pruning before
  the precise expression is evaluated). Suggested ladder: `free` (metadata-only views) <
  `cheap_elementwise` < `strided_elementwise` < `reduction` < `normalization` < `gemm_like` <
  `attention` < `conv`. The planner uses the class for coarse ordering and the expressions for
  the precise frontier.

**Symbolic cost (P8):** `flops`/`bytes_moved`/`memory.*` are *expressions* over named symbols.
Available symbols: shape dims by role (`lhs.dim[0]`, `m`, `n`, `k`), `n` (= product of output
elements), `dtype_bytes`, op-param fields, and **FDX `SymId`-bound extents** (e.g. `k_len` for a
KV-cache flash kernel). A cost symbol may name a single `SymId` (an `FDXExtent` `Range`) **or an
affine live extent** (`FDXExtent` `Affine` — FDX affine addition): a cost expression may reference
the *resolved* affine length by the operand's extent name (e.g. `k_len`), and the importer
substitutes the affine value `c0 + Σ cᵢ·SymIdᵢ` **evaluated at capacity** (each base sym at its
`Extent::bound()`), exactly as it does a single sym. The contract never re-spells the affine
coefficients in the cost string — it names the extent, and FDX owns the expression (§3.9.2). When a
graph is symbolic, **v1 evaluates the cost at capacity** (`Extent::bound()`) for plan-time frontier
pruning. Referencing a sym (single or affine) that the `SymEnv` will bind is legal at capacity.
Re-evaluating at the *resolved live* extent at route-pick time is **[consumer-ahead]** (see the
signature note below) — and applies identically to an affine extent (the same missing `&SymEnv`).

**Two compile targets** (this is the as-built reality; see §12.3):

- A **primitive `op_kind`** contract's cost expression compiles into a
  ```
  CostFn = fn(&[Shape], &[DType], &OpParams, &BackendCapabilities) -> CostEstimate
  ```
  (`fuel-dispatch/src/kernel.rs:713`). It receives shapes, dtypes, the `OpParams` payload, and
  backend caps.
- A **fused `fused_op`** contract's cost expression compiles into the **fused** cost fn
  ```
  fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate
  ```
  (`fuel-dispatch/src/fused.rs:63`, the `BackendImpl.cost` field). It receives shapes,
  `FusedOpParams` (**not** `OpParams`), and backend caps — **there is no `&[DType]` argument**.
  An importer that emitted the primitive `CostFn` shape for a `fused_op` would not compile; the
  contract therefore selects the compile target from which of `op_kind` / `fused_op` is declared.

In both cases the importer substitutes the named symbols from the shapes/params at call time. A
`Shape` carries capacity bounds and `DynAxis` sym ids, so capacity-evaluation needs nothing more
than today's signatures.

> **[consumer-ahead] Live-extent re-evaluation needs a signature change.** Neither cost-fn
> signature takes a `&SymEnv`, so neither can be evaluated at a *resolved live* extent today —
> only at capacity. The "re-evaluate at live `k_len` at route-pick without re-planning" path
> (digest §6) requires a future `CostFn`/fused-cost-fn extension (add `&SymEnv`, or a sibling
> `CostFnSym`). FKC v1 declares symbolic cost expressions; the importer evaluates them at
> capacity, which is exactly what today's signatures support. The live-prefix re-evaluation is
> scoped as forward-looking, like the per-tier memory field.

**Worked example — the two-candidate contiguize-vs-strided comparison.** Consider a matmul whose
`rhs` arrives as a transposed (strided) view of shape `[k, n]`. Two candidates compete at the
same decision point:

- **Candidate A — strided kernel, no copy.** A `handles_strided` matmul contract.
  Cost = `A.cost(shapes, params, caps)` = e.g. `CostEstimate { flops: 2*m*n*k, bytes_moved:
  (m*k + k*n + m*n)*4, kernel_overhead_ns: 5000 }`. No inserted op.
- **Candidate B — contiguous kernel + inserted contiguize.** A `requires_contiguous` matmul
  contract plus a separate `Op::Contiguize` on `rhs`. The planner sums **two** FKC `CostEstimate`s:
  - `Contiguize.cost([rhs_shape], …)` for the `[k, n]` copy = e.g. `{ flops: 0, bytes_moved:
    2*k*n*4, kernel_overhead_ns: 40 }` (the Contiguize kernel's own contract),
  - plus `B.cost(shapes, params, caps)` for the contiguous matmul.

The planner ranks `A.cost` against `Contiguize.cost + B.cost` on the cost vector. If B's
contiguous kernel is fast enough to pay back the copy (large `k*n` amortized over `m`), B wins;
otherwise A wins. Every term is read from a contract — no number is hidden in planner code. This
is the mechanism §4.3 makes possible.

### 4.5 Symbolic-extent tolerance (per operand) — including affine extents

Each input's `fdx.symbolic_extent` declares how the kernel treats a symbolic (live-vs-capacity)
axis:

- `rejected` — the kernel needs a concrete extent; the planner may only place it where the axis
  is `Scalar`.
- `tolerated` — the kernel works whether the axis is concrete or symbolic, but ignores liveness
  (reads the full capacity). Safe only when the capacity tail is harmless.
- `required`/`uses` — the kernel actively consumes the live extent (reads the `SymEnv`-resolved
  length and walks only the live prefix). This is the FlashAttn-over-KV-cache case: stride keyed
  to capacity `sk`, live count `k_len` from the `SymEnv` (FDX §6.4; `OpParams::FlashAttn` already
  carries `sk` physical + `k_len` logical).

`fdx.extent_kind` further declares **which `FDXExtentKind`** the kernel tolerates on that axis
(FDX §6.4 + the affine addition): `scalar` / `range` / `affine`.

- `range` — a single bounded `SymId` (today's FlashAttn `k_len ≤ sk`, §8). The kernel reads one
  resolved value from the `SymEnv`.
- `affine` — the live extent is an `FDXAffine` combination `c0 + Σ cᵢ·resolve(SymIdᵢ)` (≤ 4 terms;
  FDX affine addition §2). The kernel resolves it through the **same** `SymEnv` at the same point
  it would resolve a `Range` — the FDX side computes `value` (i128-accumulate, `min ≤ value ≤
  capacity` guard, FDX affine addition §3.2/§3.5) and hands the kernel a single resolved length.
  This is the persistent-decode `k_len = cached_len + new_tokens` case: capacity stays the
  concrete KV bound `K` (strides keyed to `K`), the affine expresses the live prefix, and the same
  contract + sidecar serve every token with **no recompute of a derived sym** (FDX affine
  addition §6). A kernel that declares `extent_kind: affine` MUST also accept `range` and `scalar`
  (affine subsumes them — FDX affine addition §4); the importer treats `affine` as the most
  permissive tolerance.

A kernel that only handles a single `SymId` (the as-built flash path) declares `extent_kind:
range`; declaring `affine` is a forward-looking advertisement that the kernel reads its live length
from the env regardless of how FDX derived it (which it already does — the affine value is just
resolved before the kernel sees it). **[consumer-ahead]:** the FDX affine `FDXExtent` is the
2026-06-17 addition; an importer predating the affine codes treats `extent_kind: affine` per the
unknown-meaning-bearing-value policy (§11.1, typed error / drop), never a silent demotion to
`range`.

### 4.6 In-place / aliasing capability

`caps.in_place: true` declares the kernel writes its output **into an input buffer** (the output
aliases input N). Rare and must be explicit — e.g. Vulkan `add_assign_scaled` (`dst[i] +=
src[i]*scale`, binding 0 RW dst aliases output), Metal `const_set` (mutates caller storage).
The return-contract `aliasing:` field (§5.4) names *which* input is aliased. The planner treats
an in-place kernel as consuming-and-producing the same Storage (no separate output alloc).

### 4.7 Kernel revision hash (cache-invalidation primitive)

`kernel_revision_hash` is a stable per-implementation-version hash (digest §7) mapped onto
`fuel-dispatch::fused::KernelRevisionHash(u64)`. `auto` derives it at import from
`hash(entry_point ++ provider.revision_base ++ structured-block-bytes)`, so editing the contract
or bumping the provider build changes the hash and scopes cache re-optimization to the affected
decision point. A persisted plan stores `(backend_id, op_kind, dtypes, kernel_revision_hash)` and
re-resolves the `KernelRef` on load via the link registry — **never a function pointer** (P9).

### 4.8 Precision guarantee (the pre-filter input)

Maps onto `fuel-dispatch::fused::PrecisionGuarantee`. The planner's precision-filter pass runs
**before** cost ranking (digest §4): a kernel failing the per-call precision floor or the
cumulative tolerance budget is not a candidate at all.

| FKC field | `PrecisionGuarantee` field |
|-----------|---------------------------|
| `bit_stable_on_same_hardware` | `bit_stable_on_same_hardware: bool` |
| `max_ulp` | `max_ulp: Option<u32>` |
| `max_relative` | `max_relative: Option<f64>` |
| `max_absolute` | `max_absolute: Option<f64>` |
| `notes` | `notes: &'static str` |

The **audited-vs-unaudited** distinction (digest §4) is carried by `audited:`:
- `audited: false` + all bounds null ⇒ `PrecisionGuarantee::UNAUDITED` (CI-lint flags it).
- `audited: true` + all bounds null ⇒ `PrecisionGuarantee::none(notes)` — "audited, no static
  bound applies, here's why" (e.g. Vulkan subgroup reductions with scheduler-dependent FADD
  order). The lint accepts any `notes` other than `UNAUDITED.notes`.
- `audited: true` + bounds present ⇒ a real bounded claim.

These are **starting values, overridable by the Judge** (digest §4): the contract's precision
fields seed the binding; empirical calibration refines `max_ulp`/`max_relative`/`max_absolute`
over time (the binding entry is re-readable/overridable post-load, §11).

> **Always-built coverage commitment.** fuel-cpu-backend's contract MUST declare ≥1
> `bit_stable_on_same_hardware: true` kernel for every primitive op; a new primitive's contract
> triggers the CI coverage lint until the bit-stable CPU kernel's contract exists (§10.9,
> digest §4). The CPU bulk-fill convention (`fill_unset_cpu_precision` →
> `PRIMITIVE_DETERMINISTIC_CPU`) is honored by leaving a CPU primitive kernel's precision block
> absent/`audited: false` and letting the importer apply the family default (§12.4).

### 4.9 Determinism

`determinism:` is a coarse summary distinct from (but consistent with) precision:

- `bitwise` — bit-identical across *any* compatible hardware (rare; pure-integer / exact-shuffle
  kernels like Flip/Roll/copy).
- `same_hardware_bitwise` — bit-identical re-run on the *same* hardware
  (`bit_stable_on_same_hardware: true`); the common deterministic case.
- `nondeterministic` — run-to-run variation possible (atomic FP accumulation, scheduler-dependent
  reduction order); `bit_stable_on_same_hardware` MUST be false and the validator requires
  `audited: true` with a `none(reason)` precision (no silent unaudited nondeterminism).

### 4.10 Parallelism budget is not in the contract — it is a `BackendCapabilities` fact

The optimizer's wall-clock model is `max(parallel_branches) + serial_composition` (digest §5),
which needs a **per-backend parallelism width**: how many kernel invocations the device can run
concurrently (CUDA streams / SM occupancy budget, Vulkan queues, CPU worker threads). This is a
**static device fact, not per-kernel**, and **not telemetry** (it is the *capacity*, not the live
free count). FKC intentionally does **not** carry it.

It belongs on `BackendCapabilities` (`fuel-core-types/src/backend.rs`), as a `slot_capacity:
u32` / `max_concurrent_contexts: u32` field read by the optimizer's scheduler. As of this draft
`BackendCapabilities` does **not** yet have that field (it has `required_alignment`,
`access_granularity_bits`, `transfer_paths`, `storage_substrate`, `op_dtype_support`); adding it
is a small `fuel-core-types` change, separate from FKC. **[consumer-ahead]** until then. FKC's
`caps.alignment_bytes` / `caps.access_granularity_bits` mirror the *existing* per-device
`BackendCapabilities` fields for cross-checking, but the parallelism width is a single per-device
value, so it does not get a per-kernel slot.

### 4.11 The canonical stable `ImplId` — the dispatch-telemetry / specialization basis

A contract already serializes everything a downstream consumer needs to name *the implementation
that was chosen at a dispatch* without a function pointer: **the tuple `(BackendId, op,
dtypes, kernel_source, kernel_revision_hash)` IS the canonical stable `ImplId`.** It is exactly
the persisted-plan re-resolution key (§4.7, §12.8) plus the `kernel_source` tag — every field is
already in the contract (`backend`, `op_kind`/`fused_op`, the ordered operand `dtypes`,
`kernel_source`, `kernel_revision_hash`) and every field is serializable data (P9). FKC defines
**no new identifier**; the `ImplId` is a *projection of the dispatch key + source + revision* that
already exist, so a dispatch-telemetry or kernel-specialization consumer can tag a record with a
stable, pointer-free implementation id and re-resolve it on another process/build.

This is what lets a **dispatch-telemetry / specialization feed** (the planner already times and
compares every candidate it admits — the route picker ranks sibling `BindingEntry`s, §12.5) name
"which implementation won this dispatch, and what it fell back from" in terms the kernel boundary
already owns:

- **Baracuda's `{ Baracuda(symbol) | Vendor(which) | FuelNative(which) }` `ImplId` maps directly
  from `BackendId + kernel_source`.** `BackendId::Cuda` + `kernel_source: "baracuda"` →
  `Baracuda(entry_point_symbol)` (the `entry_point` is the symbol, §12.6); `kernel_source:
  "cublas"`/`"cudnn"`/… → `Vendor(which)`; a `FuelNative`-class CPU/portable kernel →
  `FuelNative(which)`. The remaining `(op, dtypes, kernel_revision_hash)` distinguishes cells and
  pins the revision. No reconciliation table is needed: the discriminant is `kernel_source`, which
  the provider already declares per kernel (front-matter default + per-kernel override).
- The same tuple is the join token for a **structure-key miss record** (a desired specialized
  cell had no registered kernel, so the planner fell back to *F*): the `fallback`'s `ImplId` is
  this tuple, and "no registered kernel at the wanted key" is exactly the planner's own
  best-match-is-a-generic-contract outcome (§4.12).

> **[consumer-ahead: deferred Baracuda telemetry feed].** The *emission* of dispatch/miss records
> (a JSONL artifact keyed by this `ImplId` + Baracuda's externally-supplied `structure_key`) is a
> separate, opt-in Fuel feature with no consumer in this spec; it is **deferred**. FKC's role is
> only to establish that the stable `ImplId` already falls out of the contract's existing fields
> (no new identifier, no function pointer) and that the `kernel_source`→`{Baracuda|Vendor|
> FuelNative}` mapping is direct — so when the telemetry feed lands it tags records with facts the
> boundary already owns. Sources: `baracuda/docs/fuel-ask-telemetry-2026-06-17.md` (the
> `DispatchRecord`/`MissRecord`/`ImplId` ask) and its companion `kernel-specialization.md` (the
> canonical `structure_key`/`OperandKey`). The `ImplId` enum is co-defined with Baracuda and its
> encoding frozen jointly; FKC commits only to the basis tuple, not to the wire format.

### 4.12 Structure-key alignment — tight-predicate contracts and the structural miss

The structure-granularity predicates (§4.2) are deliberately the same structural vocabulary
Baracuda's **`StructureKey`** uses to name a specialized cell. A structure-specialized kernel is
one generated for a *known* layout class — it folds coordinate-unravel and stride math into
constants, vectorizes to a fixed width, hoists broadcasts, and drops remainder loops. Those four
facts are exactly the per-operand `OperandKey` axes Baracuda canonicalizes a live tensor into:

> `OperandKey { contig: Contig|InnerContig|Strided|Broadcast, bcast_mask, vec_width:
> Scalar|V2|V4|V8, inner_div: %16|%8|%4|%2|Any, flipped: bool }`
> (`baracuda/docs/fuel-ask-telemetry-2026-06-17.md`; full def in `kernel-specialization.md`
> §2/§5).

The FKC predicate vocabulary maps onto that key axis-for-axis, so a structure-specialized kernel's
admissibility predicate **is** its structure key, expressed as fast-path predicates:

| Baracuda `OperandKey` axis | FKC predicate (§4.2) / layout flag (§4.1) |
|----------------------------|--------------------------------------------|
| `inner_div: %16 \| %8 \| %4 \| %2 \| Any` | `inner_div(<role>) % {16,8,4,2} == 0` / `any` |
| `vec_width: Scalar \| V2 \| V4 \| V8` | `vec_width(<role>) >= {v2,v4,v8}` / `scalar` |
| `contig: Contig \| InnerContig \| Strided \| Broadcast` | `all_inputs_contiguous` / `inner_contiguous(<role>)` / `strided` / `broadcast_stride0` (§4.1) |
| `bcast_mask` | per-operand `broadcast_stride0` (§4.1) + `any_input_broadcast` |
| `flipped: bool` | **already maps to `reverse_strides`** (§4.1.1) — the negative-stride flag IS `flipped` |

Consequence for the planner (this is the whole point):

- A structure-specialized kernel **imports as a tight-predicate contract** — its admissibility is
  the conjunction of its structure predicates, so it is a candidate **only** for shapes in its
  cell. A generic strided kernel imports with the `any`/floor predicates and is admissible
  everywhere.
- A **structural miss** — "a specialized kernel for structure-key *K* would have been an exact fit
  here, but none is registered, so the planner's best match is a *generic* contract" — therefore
  **falls out of ordinary contract matching**: it is the case where, at a dispatch key, the
  tightest admissible contract is the generic one. No separate miss-detection mechanism is needed;
  the miss is observable as "best admissible match = generic contract" and is exactly the
  `MissRecord.wanted` demand signal a future telemetry feed (§4.11) would emit.

> **The structural miss above is distinct from the missing-*fusion* miss (G5,
> [10-decisions-log](../architecture/10-decisions-log.md)).** §4.11/§4.12 cover only the *structural
> specialization* miss — a *specialized* cell unregistered, best match a generic contract at the same
> op identity. The 2026-06-20 decision pins a separate sequencing for *fusion* misses: the **v1
> headline is the closed-world `FusionMissRecord`** — a recognized fusion-eligible chain realized as
> N primitives because the fused kernel was absent (reason `NoBackendKernel`, against a **known**
> `FusedOpId`); its consumer (append a `BindingEntry`, Tier 1) already exists, so it is built **first**.
> The **open-world** co-occurrence signal — a frequent realized op chain matching **no** known fused-op
> identity (`SequenceRecord{fused_as: None}`, by *observation*, not subgraph *enumeration*) — is
> **deferred**, because its consumer is the Tier-2 runtime declarative registration (§9.4). Fuel never
> enumerates the subgraph space. Canonical: `docs/session-prompts/baracuda-telemetry-plan.md` §9.

**[consumer-ahead].** The alignment with Baracuda's `StructureKey` / `OperandKey` is recorded so
the FKC predicate vocabulary and Baracuda's key stay byte-compatible in spirit — but the canonical
`structure_key(op_class, operands, arch) -> StructureKey` is **Baracuda's** to define and ship as
the single callable (Fuel calls it, does not reimplement it; `fuel-ask-telemetry` "the join
token"). FKC does not own the key; it owns the *predicate surface* a contract uses to advertise
its cell, designed to project onto that key without drift. Only `flipped`↔`reverse_strides` is
load-bearing **today** (the negative-stride capability, §4.1.1); the `inner_div` / `vec_width` /
`inner_contiguous` predicates are admitted into the vocabulary now so a structure-specialized
contract is expressible, but their *specialization consumer* (the AOT/JIT matrix-selection feed)
is deferred with the telemetry feed (§4.11).

---

## 5. Return-contract field semantics

### 5.1 `dtype_rule`

How the output dtype is computed.

- For a **`fused_op`** contract this compiles to `fuel_graph::registry::FusedOp.dtype_rule`
  (`fn(&[DType], &FusedOpParams) -> DType`, verified `registry.rs:108`).
- For a **primitive `op_kind`** contract **no function is registered**: the rule is *checked
  against the binding key's* output dtype slot at import (§12.7). The dtype is already pinned by
  the `(OpKind, KernelDTypes, BackendId)` key, so the rule is a validation predicate, not a
  compiled fn.

Vocabulary:
- `passthrough(<role>)` — same dtype as the named input (elementwise, unary, affine).
- `fixed(<DType>)` — a constant (FusedSoftmaxCrossEntropy → F32; comparisons → U8/bool).
- `cast(<param>)` — from an op-param target (Cast: output dtype is the output Storage's dtype).
- `dequant(<role>)` — the dequantized wide dtype (Q4_0 → F32).
- `bundle` — per-slot (see §5.5).

### 5.2 `shape_rule`

For a **`fused_op`** contract this compiles to `fuel_graph::registry::FusedOp.shape_rule`
(`fn(&[Shape], &FusedOpParams) -> Shape`, verified `registry.rs:104`). For a primitive `op_kind`
contract it is checked against the op's known shape semantics at import (no fn registered).

- `same_as(<role>)` — elementwise/unary.
- `broadcast(<roles…>)` — binary with broadcasting.
- `matmul(lhs, rhs)` — `[..batch.., m, n]`.
- `conv2d(params)` / `conv_transpose2d(params)` — geometry from `OpParams::Conv2D` etc.
- `reduce(input, dims, keepdim)` — reductions.
- `from_params(<expr>)` — explicit (FlashAttn → q's shape; Rope → input shape).
- **Symbolic preservation:** a `shape_rule` MUST preserve symbolic extents through to the output
  when the op is shape-preserving (the output's `Extent::Range`/`SymId` carries through), so the
  persistent-decode graph stays input-independent (digest §6).

### 5.3 `layout_guarantee`

What layout the *output* is in when the kernel returns:

- `contiguous` — dense row-major (the overwhelmingly common guarantee; e.g. every elementwise
  kernel writes a fresh contiguous buffer).
- `preallocated` — the executor pre-allocated the output buffer; the kernel only fills bytes
  (the `KernelRef` ABI hard rule — kernels never allocate). This is always true and is the
  default; `contiguous` is an additional guarantee about the *content* layout.
- `same_as(<input>)` — preserves an input's strides (rare).
- `bundle` — a packed multi-slot buffer (§5.5).

### 5.4 `aliasing`

- `none` — fresh output buffer, no overlap with any input (the default; full overwrite, no read
  of prior output content).
- `in_place(<input role>)` — output IS input N's buffer (requires `caps.in_place: true`, §4.6).
- `accumulate(<input role>)` — output is input N read-modified-written (e.g. IndexAdd/ScatterAdd
  accumulate into `base`; the kernel reads prior content).

### 5.5 Multi-output bundles (rank-limited, name round-trips)

A multi-output kernel emits **one** `KernelRef` with `outputs.len() == 1` (the bundle), per the
as-built ABI and 12-multi-output Option C. The contract declares the bundle's slot specs:

```yaml
return:
  bundle:                       # maps to FusedOp.output_views: fn(&[Shape],&[DType],&FusedOpParams)->Vec<OutputViewSpec>
    - { name: y,          dtype_rule: passthrough(u), shape_rule: same_as(u),     layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(state), layout_guarantee: contiguous }
```

Each slot compiles to an `OutputViewSpec` (`fuel-core-types/src/storage.rs`), which the executor
turns into an `OutputView { byte_offset, len_elements, dtype, shape: Shape, layout, name }` via
`allocate_bundled_storage`. **Two honest limits, stated rather than hidden** (the in-memory
`OutputView` is the authority; the on-the-wire `FDXOutputView` is its serialization, and it is
narrower):

- **Rank ≤ 6 per slot.** `FDXOutputView.shape` is `[u64; 6]` (matches Fuel's `DimVec` inline
  capacity). The in-memory `OutputView.shape` is an arbitrary-rank `Shape`. A bundle slot whose
  shape has rank > 6 is **not representable** in the serialized FDX form. §10 (rule 13) makes
  this an explicit `BundleSlotRankExceeded` validation error at import, never a silent
  truncation. (In-process FKC→registry registration uses `OutputViewSpec` directly and is not
  rank-capped; the cap bites only when a bundle is serialized through FDX, e.g. cross-process /
  persisted.)
- **Slot names round-trip via a side-table, not the hash.** `FDXOutputView.name_hash` is a
  one-way FNV-1a digest — you cannot reconstruct `Some("argmax_idx")` from it. To keep
  FKC→`OutputView` reconstruction faithful, the slot **name string** is carried in a side-table
  (the same string-table mechanism FDX uses for its buffer table), and `name_hash` is retained
  only as a fast diagnostic/lookup key. The importer reconstructs `OutputView.name:
  Option<&'static str>` from the side-table entry; `name_hash` alone is a *diagnostic*, not the
  source of truth.

> **Lossless claim, scoped.** A Fuel→Fuel managed export of a bundle round-trips losslessly
> **except** that (a) any slot of rank > 6 cannot be serialized through FDX (rejected, not
> truncated), and (b) if a deployment opts to carry only `name_hash` (no side-table), slot names
> become diagnostic-only. With the side-table present, names round-trip. This is the corrected
> form of FDX §15's "reconstructs the exact logical tensor (lossless)" claim — it is lossless for
> rank ≤ 6 with the name side-table, and the limits are documented, not silent. **[consumer-ahead]**:
> symbolic extents on bundle slots are not yet representable in `FDXOutputView` (plain `u64`
> dims); cross-referenced to FDX open-question #4, a slot that must preserve a symbolic axis is
> currently a documented limitation, not a supported case.

Slots are independent in dtype/shape/layout (e.g. F32 `y` + I64 `argmax_idx`).

---

## 6. Quantization vocabulary and the `PerBlock` gap

All quant facts are FDX symbols (§0, FDX §6.2). **Where a scale physically lives is governed by the
single-place rule (§3.9.3):** a scale passed as a separate graph input is an FKC `accept.inputs`
operand named in `fdx.quant.scale_operand` and is **never** also an FDX `scale_buffer`; a
sidecar-bundled scale (GGML INLINE, MX separate-buffer) rides the FDX tensor's `scale_buffer` and
has no FKC operand. Each scale is described once, with a cross-form consistency check
(`ScaleDoubleDeclared`, §10.6). The five families and their granularity rules:

| `family` | regime | `granularity` | as-built target type |
|----------|--------|---------------|----------------------|
| `none` | dense | — | — |
| `GGML_BLOCK` | static block-quant | **none** — baked INLINE; `ggml_dtype` is the format (no `granularity`, no `PerBlock`, no separate scale operand) | `GgmlDType` (real type, §3.4) |
| `AFFINE_INT` | dynamic int affine | `PerTensor` / `PerToken` / `PerChannel` | `ScaleGranularity` (real type) |
| `AFFINE_FLOAT` | dynamic FP8 affine | `PerTensor` / `PerToken` / `PerChannel` | `ScaleGranularity` (real type) |
| `AFFINE_BLOCK` | static block-grained affine (nf4/QLoRA) | block-shaped via `block_shape` + a **separate** per-block scale operand (absmax); **not** `PerBlock` | **no as-built target yet** (block-quant descriptor; see below) |
| `MX` | static block-scaled (F8E8M0 scale) | `PerBlock` (MX-only) | **no as-built target yet** (see below) |

> **`PerBlock` is an FDX/FKC-only value with no `fuel-core-types::ScaleGranularity`
> counterpart yet.** The as-built `ScaleGranularity` (`quant_scale.rs`) is exactly
> `{ PerTensor, PerToken, PerChannel }` — there is **no `PerBlock`**. The `GGML_BLOCK` family's
> per-block layout is baked into `GgmlDType` (it does not use `ScaleGranularity` at all), and the
> `MX` family's `PerBlock` granularity (separate F8E8M0 per-block scale) has no
> `ScaleGranularity` variant to map onto. Consequently **an MX contract parse-validates (§10.6)
> but is NOT registrable** until `ScaleGranularity` gains `PerBlock` (or a separate block-quant
> descriptor type lands). This matches §0's "MX consumer is ahead." FKC does **not** imply
> `ScaleGranularity` already models `PerBlock`; an importer that reaches MX registration today
> returns `MxNotYetRegistrable` rather than fabricating a granularity. The dispatch-key form for
> `AFFINE_INT` / `AFFINE_FLOAT` (the `ScalePair`-in-key shape) is the part that maps onto today's
> types; `MX` is the forward-looking part.
>
> **`AFFINE_BLOCK` (FDX code 4, FDX §6.2) is likewise describable-but-not-yet-registrable.** The
> nf4/QLoRA-style block-grained affine family carries low-bit data + a **separate block-shaped
> scale operand** (absmax), with the block grain expressed by `block_shape` — it does **NOT** use
> the `PerBlock` granularity code (that stays MX-only, FDX §6.2). It has no as-built target type
> yet (it needs a block-quant descriptor distinct from the per-tensor/token/channel
> `ScaleGranularity`), so an `AFFINE_BLOCK` contract **parse-validates (§10.6) but returns
> `MxNotYetRegistrable` at registration**, the same describe-now/register-later discipline as `MX`.
> FKC references the `AFFINE_BLOCK` symbol from FDX (§0) and re-numbers nothing.

---

## 7. Worked example A — elementwise binary (representative simple kernel)

fuel-cpu-backend `binary` (Add/Sub/Mul/Div/Max/Min), from the inventory sample.

````markdown
## binary  (Add / Sub / Mul / Div / Maximum / Minimum)

Element-wise binary arithmetic and extremum. Positional walk `out[i] = op(lhs[i], rhs[i])` over
contiguous, zero-offset, row-major buffers — no broadcasting (validated `lhs.len_bytes ==
rhs.len_bytes == out.len_bytes`). f32/f64 evaluate natively; bf16/f16 widen to f32 and narrow on
store (round-trip). Div follows IEEE inf/NaN; Maximum/Minimum use NaN-as-missing
(`f32::max`/`min`). Six op ids share this contract, selected by `OpKind`. Known limitation:
contiguous-only — any strided/broadcast/offset operand must be contiguized by the planner first.

```fkc
kernel: binary
op_kind: AddElementwise      # one contract per OpKind: also Sub/Mul/Div/Maximum/Minimum
blurb: "Elementwise binary arithmetic/extremum; contiguous same-shape; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::add_f32"   # one per (op,dtype); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected }
      rank: any
      shape_constraint: same_as=rhs
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected }
      rank: any
      shape_constraint: same_as=lhs
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous     # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: declared                # author prior; Judge refines (§4.4)
  class: cheap_elementwise
  flops: "n"                          # one op per element
  bytes_moved: "3 * n * dtype_bytes"  # read lhs+rhs, write out
  overhead_ns: 40                     # CPU nested-loop call overhead
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loop; F32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32/f64 native; bf16/f16 widen to f32 then narrow. Div IEEE inf/NaN; Max/Min NaN-as-missing."

determinism: same_hardware_bitwise
```
````

Because `binary` declares `requires_contiguous`, the planner that places it after a transposed
producer prices the inserted `Op::Contiguize` from the CPU `contiguize` contract (§4.4) and
compares that summed cost against any `handles_strided` sibling — concretely, not by assertion.

---

## 8. Worked example B — FlashAttn (complex: symbolic extent, dynamic scalar, GQA)

`fuel_graph::registry` `FLASH_ATTN` fused op, dispatched via `OpParams::FlashAttn`. Exercises:
GQA divisibility, a **symbolic live-vs-capacity KV axis** (`sk` capacity vs `k_len` live), a
**dynamic scalar param** on the `SymEnv`, and an attention cost class. Note this is a `fused_op`,
so its cost compiles to the **fused** cost-fn shape (`fn(&[Shape], &FusedOpParams,
&BackendCapabilities)`), not the primitive `CostFn` (§4.4).

````markdown
## flash_attn  (multi-head scaled-dot-product attention, KV-cache aware)

Streaming-softmax fused attention. `q [B, Hq, Sq, D]`, `k`/`v [B, Hkv, Sk, D]` with `Hkv ≤ Hq`,
GQA-divisible (`Hq % Hkv == 0`). Optional 4th input `alibi_slopes [Hq]`. The K/V `Sk` axis is the
**physical capacity** (strides + byte-length checks key off it); the kernel attends only the
first `k_len ≤ Sk` rows (the live prefix from a fixed-capacity KV-cache) and bottom-right-aligns
the causal mask at `k_len - Sq`. `k_len` is a dynamic scalar resolved per token via the SymEnv —
the static path sets `k_len == Sk` and is byte-identical to a plain `0..Sk` loop. f32
accumulation; `softmax_scale`, optional `softcap`, sliding-window `(left,right)`. Online-softmax
numerics; not bit-stable cross-hardware.

```fkc
kernel: flash_attn
fused_op: FLASH_ATTN
blurb: "Fused MHSA over a fixed-capacity KV cache; attends live prefix k_len <= Sk; GQA; causal/window/softcap."
backend: Cuda
kernel_source: "baracuda"
entry_point: "baracuda::flash_attn_fwd_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted }
      rank: 4                       # [B, Hkv, Sk, D]  (Sk = CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx: { symbolic_extent: required }   # ← uses live k_len from SymEnv, stride keyed to Sk
    - name: v
      dtypes: [F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required }   # k_len ≡ v_len ⇒ SAME SymId (FDX unification)
    - name: alibi_slopes          # optional; presence implicit in inputs.len()==4
      dtypes: [F32]
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      b:  { kind: usize }
      hq: { kind: usize }
      hkv:{ kind: usize, constraint: "hq % hkv == 0" }
      sq: { kind: usize }
      sk: { kind: usize, note: "physical K/V capacity" }
      d:  { kind: usize }
      k_len: { kind: DynScalar, note: "live attended length <= sk; rides SymEnv" }
      softmax_scale: { kind: f32 }
      causal: { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap: { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to 0..Sk loop" }
    - { when: "causal == false", note: "no mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: declared                       # author prior; Judge refines (§4.4)
  class: attention
  # fused cost-fn shape (no &[DType] arg). Symbolic over k_len; v1 evaluates at CAPACITY (sk).
  flops: "2 * b * hq * sq * k_len * d * 2"   # QK^T + PV; live-k_len re-eval is [consumer-ahead]
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: 5000
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # online-softmax + warp reductions: scheduler-dependent
  max_ulp: ~
  max_relative: 0.005
  max_absolute: ~
  audited: true
  notes: "online softmax, f32 accumulate; rel err vs reference < 5e-3; not bit-stable (warp reduction order)."

determinism: nondeterministic
```
````

This single contract carries everything the planner needs: GQA admissibility, the symbolic KV
axis (cost expressed over the *live* `k_len`, evaluated at capacity `sk` in v1, with live-prefix
re-evaluation scoped as [consumer-ahead]), the dynamic-scalar param routed through the `SymEnv`,
the precision bound for the admissibility pre-filter, and the per-tier memory for the cost vector.

> A **Conv2D** contract follows the same shape with `op_params.variant: Conv2D` (the
> `x_shape`/`w_shape`/`out_shape`/`stride`/`padding`/`dilation`/`groups` fields),
> `shape_rule: conv2d(params)`, `cost.class: conv`, `fast_paths: [{when: "groups==1"},
> {when: "depthwise"}]`, and `awkward_layout_strategy: requires_contiguous` (NCHW packed). The
> MKL conv wrapper would additionally declare `kernel_source: "mkl"` as a sibling alternative at
> the same key. A **Q4K QMatMul** contract uses `op_params.variant: QMatMul`, the weight
> operand's `fdx.quant: {family: GGML_BLOCK, ggml_dtype: Q4K}` (the GGUF `Q4_K_M` weight; §3.4),
> `dtype_rule: dequant` for the internal widen, and `cost.class: gemm_like`.

---

## 9. Import / registration model

### 9.1 Single bundle file

A provider ships **one** file with all its kernels: `fuel-vulkan-kernels.fkc.md` containing N
`## ` sections. Front-matter declares provider-wide defaults; each section overrides as needed.
One file, N registrations.

### 9.2 Globbed multi-file layout

A provider ships **many** files, one-or-more kernels each, discovered by glob:

```
contracts/
  vulkan/
    elementwise.fkc.md      # unary + binary + affine + clamp + powi
    cast.fkc.md             # all cast directions
    attention.fkc.md        # flash_attn + paged_attn
    _provider.fkc.md        # front-matter only: provider-wide defaults
```

The importer is pointed at a glob (`contracts/vulkan/**/*.fkc.md`). A bare `_provider.fkc.md`
(front-matter, no `## ` sections) supplies defaults inherited by every file in the tree; per-file
front-matter overrides. Files are processed in sorted path order for deterministic registration
ordering (which determines the order of sibling alternatives at a key — §12.5).

### 9.3 The import pipeline

```
glob → for each file:
  parse front-matter (provider defaults: backend, kernel_source, link_registry, revision_base)
  for each `## ` section:
    extract the ```fkc block (YAML, restricted subset §3.8) → FkcKernel struct
    validate (§10)                   → Result<(), FkcError>   [no panic]
    if registrable == false (§3.10): documentation-only — SKIP everything below
                                     (no lower, no entry_point resolution, not registered)
    resolve entry_point against provider.link_registry → KernelRef
    check resolved-KernelRef-level duplicate at the key (§10.10) → Result, NOT a panic
    compute kernel_revision_hash (auto | literal)
    select compile target by op_kind|fused_op (§4.4); derive key + KernelCaps + cost-fn + PrecisionGuarantee
    if op_kind:  KernelBindingTable::register_full_with_source(key, kernel, caps, precision, cost, source)
    if fused_op: FusedKernelRegistry::register(fused_id, backend, BackendImpl { kernel, dtypes, cost, precision, caps, revision })
```

Result: importing a provider's file(s) makes every kernel available with **zero hand-written
registration code**. A new backend (ROCm, TPU) ships an `.fkc.md` bundle and is dispatchable.

### 9.4 Discovery / manifest

A workspace lists its contract sources in a manifest (e.g. a `[fkc]` table in a crate's metadata,
or a `contracts.toml`): each entry is `{ provider, path-or-glob }`. The Fuel build/startup reads
the manifest and imports every source. **This is the bulk, up-front registration path, but it is no
longer the *only* one** (re-scoped 2026-06-20, [10-decisions-log](../architecture/10-decisions-log.md),
G4): the kernel binding table is append-only and **runtime-extensible** (`extend_global_bindings`,
Tier 1 — JIT-ing a kernel for an existing op identity lands here), and a new **fused-op identity** may
be registered at runtime through the trusted, Fuel-orchestrated, cost-gated declarative path (Tier 2,
append-only with stable `FusedOpId`s). What stays frozen is the **primitive `Op` enum** and any
*untrusted* user-injected ops/rules ([09-non-goals](../architecture/09-non-goals.md)). Providers in
sibling crates (Baracuda, vulkane) expose their `link_registry` symbol; the manifest binds the
contract files to it.

---

## 10. Build-time validation rules (all `Result`-returning, no `try_*`)

Run at import/registration; the principle is "every check that can run at build time must"
(digest §3). Typed `FkcError` on failure, never panic, never silent fix-up.

1. **Format version** supported (`fkc_version <= FKC_VERSION_MAX`); unknown trailing fields
   ignored (size-prefixed forward-compat, §11).
2. **Required fields present:** `kernel`, exactly one of `op_kind`/`fused_op`, `blurb`,
   `entry_point`, `accept.inputs` (≥1), `return.outputs` (≥1) or `return.bundle`, a `cost` block
   (including its required `provenance:`, §10.8a), a `precision` block, `determinism`.
   **Describe-only exception (§3.10):** a section with `registrable: false` SKIPS the
   exactly-one-of-`op_kind`/`fused_op` requirement — its `op_kind` may be `~` or a descriptive
   non-dispatch token — AND is EXEMPT from the `accept.inputs` (≥1) requirement (a documentation-
   only zero-operand op such as a `rand_uniform`/`rand_normal` random fill, whose only "input" is
   backend-private RNG state rather than a graph tensor, legitimately declares `accept.inputs: []`).
   All other required fields (`blurb`, `return.outputs`/`bundle`, `cost`, `precision`,
   `determinism`, etc.) still apply. A **registrable** section is NOT exempted from either rule.
3. **Dtype validity:** every dtype name is a real `DType`; sub-byte dtypes carry `fdx.sub_byte` +
   `fdx.quant` (no reliance on `size_in_bytes()==0`); every `ggml_dtype` is a real `GgmlDType`
   **variant name** matched by code (`Q4K`, never `Q4_K_M`; §3.4). The named op
   (`op_kind`/`fused_op`) must resolve to a real dispatch target **unless** the section is
   describe-only (§3.10), in which case the resolution check is skipped (the descriptive
   dtype/layout/quant checks below still run).
4. **Layout coherence:** at least one of `contiguous`/`strided` is acceptable per operand;
   `broadcast_stride0: accepted` ⇒ `strided: accepted` (broadcast is a stride-0 special case);
   `reverse_strides: accepted` ⇒ `strided: accepted` (a negative stride is still a strided walk —
   a kernel cannot accept reversed strides while rejecting strided ones; `LayoutIncoherent`
   otherwise). `reverse_strides` is independent of `broadcast_stride0` and `start_offset` (a kernel
   may accept any subset). `reverse_strides` is an FKC-only kernel capability; the FDX side that
   *describes* the negative strides is a [cross-spec dependency] (§4.1.1).
5. **Awkward-layout coherence — PER OPERAND (§4.3.1).** The strategy is resolved per operand:
   each operand's **effective** strategy is its own `layout.awkward_layout_strategy` when present,
   else the kernel-wide `caps.awkward_layout_strategy`, else `requires_contiguous`. The check runs
   **per operand against that operand's own layout flags**: an operand whose effective strategy is
   `handles_strided` MUST have `strided: accepted` on *that operand*; `requires_contiguous` ⇒ that
   operand has `contiguous: required`; `contiguize_internally` folds *that operand's* copy into the
   kernel's declared `bytes_moved` (§4.3). It does **not** require a single tolerance across every
   input — a kernel may declare `handles_strided` q/k/v alongside `requires_contiguous` aux operands
   (`alibi_slopes`/`seqlens`), the motivating flash-attention case (§4.3.1). A per-operand
   `awkward_layout_strategy` that contradicts its operand's layout flags (e.g. `handles_strided`
   with `strided: rejected`), or an unknown strategy value at either level, is
   `AwkwardStrategyIncoherent` (an unknown value is meaning-bearing per §11.1, never silently
   demoted).
6. **Quant coherence (mirrors FDX §8 / FDX §6.2):** `GGML_BLOCK` ⇒ `ggml_dtype` a real
   `GgmlDType` variant + no `ScalePair`; `AFFINE_INT` / `AFFINE_FLOAT` ⇒ `granularity ∈
   {PerTensor,PerToken,PerChannel}` (the real `ScaleGranularity` set); `MX` ⇒ scale dtype F8E8M0 +
   `granularity = PerBlock`; **`AFFINE_BLOCK`** ⇒ block geometry present (`block_shape` / a
   block-shaped **separate** scale operand) and `granularity` is **NOT** `PerBlock` (`PerBlock`
   stays MX-only — FDX §6.2) — its block grain rides `block_shape`, not a granularity code. The
   dispatch key derived from these MUST be coherent with FDX's dispatch-key form. **`MX` and
   `AFFINE_BLOCK` parse-validate but return `MxNotYetRegistrable` at registration** (no
   `ScaleGranularity::PerBlock` target for `MX`, and no as-built block-quant descriptor target type
   for `AFFINE_BLOCK` — §6). **Scale single-place (§3.9.3):** a scale is declared in exactly one
   place — `fdx.quant.scale_operand` (a separate `accept.inputs` role) **XOR** a sidecar-bundled
   `FDXQuant.scale_buffer`, never both for the same scale (`ScaleDoubleDeclared`);
   `fdx.quant.scale_operand` MUST name a real input role. For `AFFINE_BLOCK` the per-block scale is
   the `SEPARATE_BUFFER` form (FDX §6.2), so it is named once — either an `accept.inputs` operand in
   `fdx.quant.scale_operand` or the FDX `scale_buffer`, never both.
7. **Shape/param coherence:** `shape_constraint` predicates parse; the op-param `variant` is a
   real variant **in the correct namespace** — `OpParams` for an `op_kind` contract,
   `FusedOpParams` for a `fused_op` contract (§3.7); declared fields match the variant's fields.
   **Describe-only (§3.10) SKIPS the op-param namespace check** (there is no resolved dispatch
   target to bind a params variant against).
8. **Cost expressions** parse (FKC's own parser, not YAML — §3.8) and reference only in-scope
   symbols (operand dims by role, op-param fields, `n`, `dtype_bytes`, bound `SymId`s evaluated
   at capacity); no division-by-zero at capacity. The compile target (primitive `CostFn` vs
   fused cost-fn) is selected by `op_kind`/`fused_op` and the symbol set is checked against the
   chosen signature (a `fused_op` cost expression may not reference `&[DType]`-only facts).
8a. **Cost provenance (the COST_RULE — supersedes the old "non-placeholder cost required"
    lint).** A cost must be **author-`declared` OR explicitly `judge_measured`** — both are
    first-class and visible (§4.4). Concretely: `cost.provenance` is **required** and ∈
    `{declared, judge_measured}` (an absent or unknown value is `CostProvenanceMissing`); and a
    **bare / placeholder / omitted-without-marker cost still fails** — coefficients that are all
    zero/sentinel "TODO" under *either* marker (except the honest `class: free` metadata-only op,
    §4.4) are `PlaceholderCost`. There is **no hidden gap**: the optimizer can always tell whether
    a cost is an author prior or a measured value, and never reads a cost of unknown origin. This
    replaces the earlier "non-placeholder cost required" phrasing: the requirement is now "cost
    must be declared OR explicitly judge_measured; bare/placeholder still fails." (Precision is
    governed by §10.9 — author-declared + Judge-audited — unchanged.) FKC stays **agnostic to the
    Judge's internals** (§4.4): the lint checks only the provenance marker and non-placeholder-ness,
    never how the Judge produced a `judge_measured` value.
9. **Precision coverage lint:** `audited: false` + all-null ⇒ flagged UNAUDITED; the always-built
   CPU backend MUST have ≥1 `bit_stable_on_same_hardware: true` contract per primitive op (digest
   §4). `determinism: nondeterministic` ⇒ `bit_stable=false` + `audited: true`.
10. **Dispatch-key duplicate detection — at the resolved `KernelRef`, not the string.** After
    `entry_point → KernelRef` link-registry resolution, two contracts whose `entry_point`s
    resolve to the **same `KernelRef`** at one `(op, dtypes, backend)` key is a hard
    `DuplicateKernelRef` error returned as a typed `Result` **before** any registration call.
    String-level `entry_point` equality is *also* rejected (`DuplicateEntryPoint`), but it is not
    sufficient: two **distinct** `entry_point` strings can alias the same `KernelRef` (a shared
    generic fn reused for two op variants, or a deliberate alias in the link registry), which the
    string check would miss but the as-built `register_full_with_source` would *panic* on
    (`kernel.rs:910-918`, function-pointer identity). FKC therefore dedupes at the resolved
    pointer to genuinely pre-empt the panic. Distinct `entry_point`s resolving to **distinct**
    `KernelRef`s at one key are legal sibling alternatives.

    > **CONSTITUTION-CONFLICT (never-panic, G9).** The as-built
    > `KernelBindingTable::register_full_with_source` *panics* on a duplicate `KernelRef`
    > (`kernel.rs:911`). FKC's §10.10 pre-check makes the panic *unreachable on the import path*
    > by detecting the duplicate first and returning `Result`. But relying on "we never feed it a
    > duplicate" leaves a panicking primitive on a production path, which the constitution
    > forbids. **The underlying `register_full_with_source` must be changed to return
    > `Result<(), DuplicateKernelRef>` instead of `panic!`** so the never-panic guarantee holds
    > structurally, not by the importer's discipline. This is a small `fuel-dispatch` change that
    > FKC depends on; until it lands, the FKC importer's pre-check is the only thing standing
    > between an aliased entry-point and a startup panic. Flagged per CLAUDE.md (validate at
    > build time; never panic on production paths).
11. **Prose/structured agreement:** the prose blurb (first non-empty line of the section) MUST
    equal the structured `blurb:`. A re-render lint diffs them (P3).
12. **No pointers in the file:** the file contains only data + symbolic `entry_point` strings;
    any literal address is rejected (P9). YAML anchors/aliases/merge keys are rejected (§3.8).
13. **Bundle slot limits (§5.5):** every `return.bundle` slot's `shape_rule` must yield a shape
    of rank ≤ 6 when serialized through FDX (`BundleSlotRankExceeded` otherwise, never silent
    truncation); each slot's `name` is recorded in the name side-table so it round-trips.
14. **Gather (paged) coherence (§3.9.1; mirrors FDX gather V18/V19/V20/V21):** `fdx.gather.kind:
    paged_blocks` ⇒ (a) `fdx.requires_ext: true` (the paged pool is mandatorily meaning-bearing —
    FDX V19); (b) `fdx.symbolic_extent: required` (the per-seq live length is symbolic, the common
    case); (c) `fdx.gather.block_table` / `fdx.gather.context_lens`, **when non-`~`**, name real
    `accept.inputs` roles (the single-place rule — they are separate operands, not a duplicate
    table; cross-checked against the FDX buffer references by FDX V21(b)); (d) an unknown
    `fdx.gather.kind` value is a typed `UnknownAdmissibilityEnum` (§11.1, meaning-bearing — never a
    guess). `GatherIncoherent` on failure. **The FDX gather codes are the 2026-06-17 addition (no
    code yet): a `gather`-bearing operand that reaches registration before the codes/`DlpackExtGather`
    land returns `GatherNotYetSupported`** (the `MxNotYetRegistrable` discipline, §3.9.1 / §6).
15. **Affine / symbolic extent coherence (§3.9.2, §4.5):** `fdx.extent_kind` ∈
    `{rejected, scalar, range, affine}` and is consistent with `fdx.symbolic_extent`
    (`extent_kind: range|affine` ⇒ `symbolic_extent: required`); the affine *expression* is **not**
    authored in the FKC contract (it lives in the FDX `FDXExtent.affine`) so there is nothing to
    coefficient-check here — only the tolerance token's validity. An unknown `extent_kind` is a
    typed `UnknownAdmissibilityEnum` (§11.1, meaning-bearing). The FDX affine `FDXExtent` is the
    2026-06-17 addition; `extent_kind: affine` reaching registration before the affine codes land
    is treated per §11.1 (typed, never demoted to `range`).
16. **FDX-subset drift-guard — FKC's token set is a SUBSET of FDX's normative tables (the
    cross-spec consistency check; §0, §13, mirrors FDX §6.0).** A build-time check asserting that
    **every dtype / quant `family` / `granularity` / `ggml_dtype` token any FKC contract uses is a
    member of FDX's normative code table** — FDX being the single normative owner of those codes
    (§0; FDX §6.0/§6.1/§6.2). Concretely, the importer resolves each such token through the
    generated `fuel-core-types::dlpack` constants (the same constants FDX's own conversion fns and
    mapping test pin, FDX §6.0): a `dtypes:` name must resolve via `fdx_code(DType)` (FDX §6.1); an
    `fdx.quant.family` token must be a real `FDX_QUANT_*` symbol (FDX §6.2 family table —
    `NONE | GGML_BLOCK | MX | AFFINE_INT | AFFINE_FLOAT | AFFINE_BLOCK`); an `fdx.quant.granularity`
    token must be a real `FDXScaleGranularity` symbol (FDX §6.2 — `PerTensor | PerToken | PerChannel
    | PerBlock`); an `fdx.quant.ggml_dtype` name must be a real `GgmlDType` variant matched **by
    code** (FDX §6.2; §3.4). A token absent from the FDX table is a `FdxTokenNotInTable` error
    (returned as a typed `Result`, never a guess or a silent default — meaning-bearing per §11.1),
    so **FKC's accepted token set can only ever be a subset of FDX's, and the two specs cannot
    drift**: FDX is the producer of new codes (e.g. it appended `AFFINE_BLOCK` 2026-06-18, §6), and
    a contract that names a token FDX has not minted fails this check rather than introducing a
    parallel code. This is the build-time half of the §0 designation that "FDX is the single
    normative source for all shared codes"; its FDX-side counterpart is the §6.0 cross-spec test
    asserting FKC's dispatch-key codes **equal** the FDX constants — the two together pin the
    relation in both directions (FKC ⊆ FDX here; FKC-key-codes == FDX-constants on the FDX side).
    A token that is in the FDX table but **not yet registrable** by today's types (`MX` for lack of a
    `PerBlock` `ScaleGranularity`; `AFFINE_BLOCK` for lack of a block-quant descriptor target type —
    `AFFINE_BLOCK` does NOT use `PerBlock`, its block grain rides `block_shape`) still **passes this
    subset check** — it is a member of FDX's table —
    and is gated separately at registration by §10.6 (`MxNotYetRegistrable`), never by this rule.
17. **Describe-only sections (§3.10).** A section with `registrable: false` is documentation: the
    importer **skips the dispatch-resolution checks** — exactly-one-of-`op_kind`/`fused_op` (rule
    2), the `accept.inputs` (≥1) required-field check (rule 2; a zero-operand documentation op like
    a `rand_uniform`/`rand_normal` random fill may declare `accept.inputs: []`), the op resolving to
    a real `OpKind`/`FusedOpId` (rule 3), and the op-param namespace check
    (rule 7) — and **excludes the section from registration** (§9.3, §12.5): it never becomes a
    `ResolvedPrimitive`/`ResolvedFused`, never resolves an `entry_point`, and never reaches the
    binding table / fused registry or the duplicate-`KernelRef` gate (rule 10). All **descriptive**
    checks still run: the FDX-subset drift-guard (rule 16), layout coherence (rule 4),
    awkward-layout coherence (rule 5), and quant coherence (rule 6) — a describe-only section is
    *validated documentation*, not an unchecked skip. `registrable` defaults to `true` (additive,
    §11); a section that omits it is registered exactly as before. **Never invent an `OpKind` and
    never relax a validator to make a section "pass": a token with no real `OpKind` is marked
    describe-only (a declared, typed posture); a token that is a typo for a real `OpKind` is renamed
    to the exact variant (§3.10).**

`FkcError` variants: `UnsupportedVersion`, `MissingField`, `BadDType`, `BadScalarType`,
`YamlTabIndent`, `YamlAnchorDisallowed`, `LayoutIncoherent`, `AwkwardStrategyIncoherent`,
`QuantIncoherent`, `ScaleDoubleDeclared`, `MxNotYetRegistrable`, `ShapeConstraintParse`,
`BadOpParamsVariant`, `CostExprParse`, `CostTargetMismatch`, `CostProvenanceMissing`,
`PlaceholderCost`, `UnauditedPrecision`, `MissingBitStableCoverage`, `DuplicateEntryPoint`,
`DuplicateKernelRef`, `BlurbMismatch`, `EntryPointUnresolved`, `BundleSlotRankExceeded`,
`GatherIncoherent`, `GatherNotYetSupported`, `FdxTokenNotInTable`, `UnknownAdmissibilityEnum`
(§11).

---

## 11. Format versioning and the unknown-value policy

- **`fkc_version`** (file front-matter) is the single discriminator. v1 is this document.
- **Additive growth:** new optional fields go into the structured block and are ignored by older
  importers (a v1 importer reading a v2 block keeps the fields it knows, drops the rest).
- **Breaking changes** bump `fkc_version` and the matching `docs/architecture/` section + a
  `10-decisions-log.md` entry on a MAJOR bump (per CLAUDE.md doc discipline).
- **Re-readable post-load:** the binding entry derived from a contract is re-readable and its
  cost/precision overridable by background re-optimization against fresh Judge data (digest §11),
  so a contract's static values are *seeds*, not immutable literals. When the Judge refines an
  author-`declared` cost, the binding's provenance flips to `judge_measured` (§4.4) — the prior is
  replaced by a measured value and the entry records that, so the provenance label always reflects
  the *current* cost's origin, not just the contract's authored intent.

### 11.1 Unknown enum value policy (reconciled with FDX §14)

The v0.1 "always demote to a conservative default" rule was too loose: silently demoting an
unknown `awkward_layout_strategy` (a *newer* `handles_strided`-class value) to `rejected` would
make the planner think a strided-capable kernel needs a contiguize — a silent behavioral change
that sits uneasily with no-silent-degradation. FKC v1 **splits unknown values by whether they
affect correctness/admissibility**, matching FDX §14's "unknown code ⇒ typed error, never a
guess" for the meaning-bearing cases:

| field | unknown-value policy |
|-------|----------------------|
| `awkward_layout_strategy` (affects whether a contiguize is inserted) | **typed error / drop the kernel** (`UnknownAdmissibilityEnum`); never a silent demotion |
| quant `family` (affects dispatch admissibility) | **typed error** (matches FDX §14) |
| `fdx.gather.kind` (affects whether the operand is a paged pool) | **typed error** (`UnknownAdmissibilityEnum`; FDX gather addition §14, `#[non_exhaustive]` spirit) — never demote a paged operand to dense |
| `fdx.extent_kind` (affects how the live extent resolves) | **typed error** (`UnknownAdmissibilityEnum`; FDX affine addition §14) — never demote `affine` to `range` |
| dtype / `ggml_dtype` / `granularity` codes | **typed error** (FDX §14; FDX owns the codes, §0) |
| `cost.class` (advisory coarse bucket only) | **warn + default** to `gemm_like` upper bound (purely advisory; the precise cost expression still governs) |
| `fast_paths[].when` predicate unknown to this importer | **warn + ignore that fast path** (advisory; the slow-path cost still applies) |
| tiling / alignment hints | **warn + default** |

The rule, stated once and cited by both specs: **a value that changes a kernel's correctness or
admissibility is a typed error or makes the kernel unavailable on this importer; only purely
advisory fields warn-and-default.** FDX §14 is the normative statement of this rule for shared
codes; FKC §11.1 extends it to FKC-only advisory fields. A newer provider's *advisory* additions
stay loadable by an older Fuel; its *meaning-bearing* additions correctly make the affected
kernels unavailable rather than mis-costed.

---

## 12. Mapping onto the current Fuel dispatch types

This section makes the conversion path concrete against `fuel-dispatch/src/{kernel.rs, fused.rs}`
and `fuel-graph/src/registry.rs`.

### 12.1 Dispatch key → `(OpKind, KernelDTypes, BackendId)` + `kernel_source`

The contract's `op_kind` → `OpKind`; `backend` → `BackendId`; the ordered operand dtypes (inputs
then outputs) → `KernelDTypes` (`SmallVec<[DType; 8]>`); `kernel_source` → `BindingEntry
.kernel_source`. Quant facts (`fdx.quant`) are encoded into the operand dtype slots so a quant
key is distinct (today via the per-format `Capability` token / `QuantType` in `OpParams::QMatMul`;
the affine `ScalePair`-in-key form lands with the dynamic-quant work, digest §9). A multi-input
attention contract (q + 2 caches + block_table + context_lens + alibi + out = 7) fits the
inline-8 `KernelDTypes` capacity by design — and a **paged** attention contract (§3.9.1) keys the
same way: the pool, `block_table`, and `context_lens` are ordinary ordered operands in the key,
while the FDX gather descriptor (paged residency) and the `Capability::DlpackExtGather` token gate
*admissibility*, not the key shape. A separate-input quant scale (§3.9.3) is likewise just another
operand dtype slot in the key.

### 12.2 `caps` → `KernelCaps` (today one bool; FKC keeps the richer set)

Today `KernelCaps { strided_input: bool }`. The importer projects the five-flag layout set:
`strided_input = (any input has strided: accepted)`. `broadcast_stride0` is subsumed (broadcast
is stride-0 strided), `start_offset: accepted` is recorded but currently routed through
auto-Contiguize (the as-built note: "inputs with non-zero `start_offset` still go through
auto-Contiguize today"), and `reverse_strides` is **recorded but treated as not-yet-honored**: until
`KernelCaps` grows a `reverse_strides`/signed-stride bool, a negative-stride operand is normalized to
a non-negative copy regardless of the declaration (§4.1.1). As `KernelCaps` grows new fields (the
struct is explicitly forward-extensible — "Forward-extensible by adding fields"), the importer maps
the remaining four flags directly, at which point a `reverse_strides: accepted` kernel receives the
`Op::Flip` view zero-copy and the normalizing copy is inserted only for non-declaring consumers. `caps.in_place`, `alignment_bytes`, `access_granularity_bits` map onto the
in-place handling and `BackendCapabilities.{required_alignment, access_granularity_bits}`. The
parallelism width (`slot_capacity`) is **not** here — it is a single per-device
`BackendCapabilities` value (§4.10), not per-kernel.

### 12.3 `cost` → `CostFn` (primitive) / `BackendImpl.cost` (fused) / `CostEstimate`

Two compile targets, selected by `op_kind` vs `fused_op` (§4.4):

- **Primitive `op_kind`:** compile to `CostFn = fn(&[Shape], &[DType], &OpParams,
  &BackendCapabilities) -> CostEstimate` (`kernel.rs:713`).
- **Fused `fused_op`:** compile to `fn(&[Shape], &FusedOpParams, &BackendCapabilities) ->
  CostEstimate` (`fused.rs:63`, the `BackendImpl.cost` field) — **no `&[DType]` arg,
  `FusedOpParams` not `OpParams`**.

The importer substitutes the named symbols (operand dims, op-param fields, `n`, `dtype_bytes`,
`SymId`-bound extents evaluated **at capacity**) and returns `CostEstimate { flops, bytes_moved,
kernel_overhead_ns }` from `flops`/`bytes_moved`/`overhead_ns`. The **per-tier `memory` block**
has no slot in today's `CostEstimate` (it is the scalar-ish Layer-1 shape); the importer carries
it forward into the optimizer's per-tier cost vector as that consumer lands (digest §5), and
folds `device_bytes` into bandwidth pressure in the interim. The `provenance:` field (§4.4) is
import metadata on the binding: a `declared` cost seeds the entry as an author prior; a
`judge_measured` cost records that the coefficients came from calibration. The as-built
OpKind-family bulk-fill (`fill_unset_cost_for_backend`) still applies, but **only to a contract
that explicitly opts into it** — a contract with no cost coefficients is *not* silently
`unknown_cost`; it must declare an explicit family-default cost with `provenance: declared` (the
"use the family default" intent stated, not a bare gap), which the importer then fills via the
existing pass. A truly bare/placeholder cost fails the §10.8a lint at import rather than reaching
the bulk-fill. **Live-extent re-evaluation is not supported by either signature** (no `&SymEnv`);
it is [consumer-ahead] (§4.4).

### 12.4 `precision` → `PrecisionGuarantee`

Direct field map (§4.8). `audited: false` + all-null → `PrecisionGuarantee::UNAUDITED`;
`audited: true` + all-null → `PrecisionGuarantee::none(notes)`; bounds present → a populated
`PrecisionGuarantee`. A CPU primitive kernel that omits the precision block (or leaves it
`audited: false`) gets `PRIMITIVE_DETERMINISTIC_CPU` via the existing `fill_unset_cpu_precision`
pass — the importer leaves it UNAUDITED and lets the fill upgrade it, exactly as hand-written
registrations do today. The coverage lint (§10.9) reuses the as-built notes-equality detector.

### 12.5 Registration → `register_full_with_source` / `FusedKernelRegistry`

For an `op_kind` contract:
`table.register_full_with_source(op, &dtypes, backend, kernel, caps, precision, cost, source)` —
the exact as-built signature, including the sibling-alternative append semantics (distinct
`KernelRef`s at one key become `SmallVec<[BindingEntry; 2]>` siblings; the route picker ranks
them). Note: per §10.10, the importer's duplicate pre-check runs **before** this call so the
as-built duplicate-`KernelRef` panic is never reached, and the CONSTITUTION-CONFLICT callout
tracks changing that primitive to `Result`. For a `fused_op` contract:
`FusedKernelRegistry::register(fused_id, backend, BackendImpl { kernel, dtypes: &'static [DType],
cost, precision, caps, revision })`, joined to the graph-side `FusedOp { shape_rule, dtype_rule,
output_views, … }` by `FusedOpId`.

**A `fused_op` contract MUST carry its recipe (the G1 recipe principle,
[10-decisions-log](../architecture/10-decisions-log.md)).** The graph-side `FusedOp` half is required
to supply **both** inverse directions: a `decompose` (fused → primitive subgraph; *lowers* it onto the
base map) and a `pattern` (recognize that subgraph; *re-fuse*) — see the
[FKC fusion-patterns spec](fkc-fusion-patterns.md) for the declarative `pattern:` block and the
`decompose` contract. **Both are mandatory.** A `fused_op` registered with no recipe is an **opaque
island**: invisible to base-map analysis (the missing-fusion / co-occurrence telemetry, §4.11/§4.12,
cannot see across or inside it), impossible to re-fuse, and — because optimization *is*
lower-to-base-map + find-best-cover — un-lowerable by the optimizer at all. `decompose` is **total and
never-`panic!`s** (a primitive decomposes to itself; a non-basis op that fails to decompose is a
*surfaced opaque-op gap*, never a crash); the recipe **always ships with the op**, never deferred
"until intermediates fit." (The recipe is the *math* definition; the kernel this contract registers is
a faster, numerically-close implementation governed by the FKC `precision` tolerance, §4.8.)

### 12.6 `entry_point` → `KernelRef` (the no-pointer indirection)

A provider exposes a **link registry**: a static `&[(symbol_id: &str, KernelRef)]` (or a
`HashMap<&str, KernelRef>`) named in front-matter (`link_registry`). The importer resolves each
contract's `entry_point` string against it → a `KernelRef`. This keeps the *contract file*
pointer-free and serializable (P9), while the *live* `KernelRef` is recovered at import time —
exactly how a persisted plan re-resolves `(backend, op, dtypes, kernel_revision_hash)` →
`KernelRef` on load (digest §7). An unresolved `entry_point` is a typed `EntryPointUnresolved`
error. Because two `entry_point`s may alias one `KernelRef`, duplicate detection happens here, at
the resolved pointer (§10.10).

### 12.7 Return rules → `FusedOp.shape_rule` / `dtype_rule` / `output_views`

For a **`fused_op`** contract, `return.outputs[].shape_rule` / `dtype_rule` compile to the
graph-side `fn(&[Shape], &FusedOpParams) -> Shape` and `fn(&[DType], &FusedOpParams) -> DType`
(the as-built `FusedOp` fields, `registry.rs:104,108`). `return.bundle` compiles to
`output_views: Option<fn(&[Shape], &[DType], &FusedOpParams) -> Vec<OutputViewSpec>>` (the
as-built optional field, `registry.rs:133`). For **primitive `op_kind`** ops these rules are
**checked against the binding key** (no fn is registered) — the dtype is pinned by the key and
the shape by the op's known semantics (§5.1, §5.2).

### 12.8 `kernel_revision_hash` → `KernelRevisionHash`

`auto` → `KernelRevisionHash(hash(entry_point ++ revision_base ++ block_bytes))`; a literal hex →
`KernelRevisionHash(u64)`; absent → `KernelRevisionHash::UNTRACKED` (today's step-1 sentinel),
upgradable as the hashing function lands.

---

## 13. Summary — the format decisions that most shape FKC

1. **Hybrid markdown + ` ```fkc ` YAML block**, one `## ` section per kernel, file-level
   front-matter for provider defaults — readable in mdBook, unambiguously parseable; the YAML
   subset is pinned (§3.8) so hand-authoring never silently mis-types a value.
2. **Tensors in DLPack + FDX terms, FDX-numbered**: every operand/output descriptor reuses FDX's
   dtype / quant / symbolic / substrate codes by symbol; FDX is the single normative source of
   codes (§0), checked by the cross-spec consistency drift-guard (§10 rule 16) — every FKC dtype /
   quant `family` / `granularity` / `ggml_dtype` token must be a member of FDX's normative table, so
   FKC's token set is provably a **subset** of FDX's and the two specs cannot drift.
3. **Layout is a five-flag capability set** (`contiguous`/`strided`/`broadcast_stride0`/
   `start_offset`/`reverse_strides`), strictly richer than today's one `strided_input` bool, which
   the importer recovers as a lossy projection. `reverse_strides` makes **negative strides
   first-class** (e.g. `Op::Flip`): a declaring kernel receives them zero-copy; the planner inserts
   a non-negative normalizing copy only for a consumer that does not declare the capability or for a
   bare export to a non-cooperating external ecosystem — never universally (§4.1.1).
4. **`awkward_layout_strategy` is the linchpin — and Contiguize/Dequantize are themselves FKC
   kernels** — so the contiguize-vs-strided-vs-materialize choice is a literal sum of two
   contract `CostEstimate`s (§4.3, §4.4 worked example), not a hidden planner constant. Internal
   contiguize is *declared*, never hidden. The strategy is resolvable **per operand** (§4.3.1): a
   kernel-wide `caps.awkward_layout_strategy` default, optionally overridden by a per-operand
   `layout.awkward_layout_strategy`, so a flash-attention kernel can walk **strided q/k/v** while
   requiring **contiguous aux operands** (`alibi_slopes`/`seqlens`) — and §10.5 checks coherence per
   operand, never demanding one tolerance across every input.
5. **Cost is a vector contribution** with **two compile targets** (primitive `CostFn` vs fused
   `fn(&[Shape], &FusedOpParams, &BackendCapabilities)`), a coarse `class`, and may be
   **symbolic** — evaluated **at capacity** in v1; live-extent re-evaluation is [consumer-ahead]
   (needs a `&SymEnv` signature extension). Every cost carries a **`provenance`** marker
   (`declared` author-prior **or** `judge_measured`) — both first-class and visible; a
   bare/placeholder/unmarked cost is a lint failure (§4.4, §10.8a). FKC stays agnostic to the
   Judge's internals: it depends only on "the Judge exists and refines/bootstraps cost."
6. **Precision is a structured pre-filter** (`PrecisionGuarantee` with the audited /
   unaudited / audited-no-bound trichotomy), applied before cost ranking; values are
   Judge-overridable seeds.
7. **Import = registration**: a single bundle file or a glob of per-kernel files auto-registers
   every kernel via the as-built `register_full_with_source` / `FusedKernelRegistry`, with no
   hand-written glue.
8. **Pointer-free + serializable**: `entry_point` symbols (resolved via a provider link registry)
   and a `kernel_revision_hash` replace function pointers; duplicate detection is at the resolved
   `KernelRef`, pre-empting the as-built panic (and §10.10 flags that the panic itself must
   become a `Result`). The same serializable facts form the canonical stable **`ImplId`** =
   `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)` for dispatch-telemetry /
   specialization consumers; Baracuda's `{Baracuda|Vendor|FuelNative}` maps directly from
   `BackendId + kernel_source` (§4.11, [consumer-ahead: deferred Baracuda telemetry feed]). The
   fast-path predicate vocabulary expresses structure-granularity buckets (`inner_div`
   %16/%8/%4/%2/any, `vec_width` scalar/v2/v4/v8, `inner_contiguous`) so a structure-specialized
   kernel imports as a tight-predicate contract and a structural *miss* falls out of matching,
   aligned with Baracuda's `StructureKey`/`OperandKey` (`flipped`↔`reverse_strides`) (§4.2, §4.12).
9. **Build-time validated, never-panic**: 16 typed `Result` checks at import (version, fields,
   dtype/layout coherence incl. `reverse_strides`⇒`strided`, quant coherence incl. scale
   single-place, op-param-namespace + cost-target match, cost-expr scope, precision coverage,
   resolved-KernelRef key non-overlap, bundle rank, prose/structured agreement, paged-gather
   coherence, affine-extent coherence, and the FDX-subset drift-guard — every dtype/quant/
   granularity/family token FKC uses is a member of FDX's normative table, so the two specs cannot
   drift, §10 rule 16).
10. **Additively versioned, with a correctness-aware unknown-value policy**: `fkc_version`;
    unknown *advisory* fields/values warn-and-default, unknown *meaning-bearing* values
    (awkward-layout strategy, quant family, codes) are typed errors / drop the kernel — reconciled
    with FDX §14 (§11.1).

### Findings deliberately not fully closed here (and why)

- **`slot_capacity` / `max_concurrent_contexts` is documented, not added.** §4.10 states where it
  belongs (`BackendCapabilities`) and that it is not per-kernel/not telemetry, and marks it
  [consumer-ahead]. Actually adding the field is a `fuel-core-types` change outside this spec's
  surface; FKC's job here is to say it is *not* an FKC field and point at its home, which the
  finding accepted as a valid resolution.
- **The `register_full_with_source` panic → `Result` change is a firm prerequisite (RESOLVED
  2026-06-17), flagged here, performed with the implementation.** It is a `fuel-dispatch` code
  change (CONSTITUTION-CONFLICT in §10.10), not a spec edit. Per the resolved conversion-plan
  decision, `register_full_with_source` **MUST become `Result`** (never-panic) and is a
  **prerequisite** of the FKC import path — not merely a future cleanup. The spec makes it a
  tracked dependency and gives FKC an import-path pre-check so the panic is unreachable in the
  interim; the actual de-panic lands with the importer implementation.
- **Live-extent (`&SymEnv`) cost re-evaluation is scoped, not designed.** The signature extension
  (`CostFn` + `&SymEnv`, or a `CostFnSym` sibling) is named as the required future change but its
  exact shape is left to the implementation that adds the consumer, consistent with the
  [consumer-ahead] discipline (don't pin an interface ahead of its consumer).
- **`MX`/`PerBlock` registrability is documented as blocked, not unblocked.** §6 states an MX
  contract parse-validates but returns `MxNotYetRegistrable` until `ScaleGranularity` gains
  `PerBlock`. Adding that variant is a `fuel-core-types` decision (it may instead want a separate
  block-quant descriptor), so the spec records the gap rather than pre-deciding the type change.
- **The FDX negative-stride ban reversal (V13) is a firm cross-spec dependency, recorded not
  edited (RESOLVED 2026-06-17).** FKC now declares the `reverse_strides` kernel capability (§4.1.1,
  §3.2) so a kernel can accept negative-stride (`Op::Flip`) tensors zero-copy. The matching
  *description* side — FDX *describing* signed `int64` strides and its OOB validator computing the
  touched range over the **signed** strides (max contrib `= (dim-1)*stride` if `stride>0` else `0`;
  min contrib `= (dim-1)*stride` if `stride<0` else `0`), with the old universal "non-negative on
  export" rule (V13) **removed** — is an FDX-side edit (FDX owns the tensor axis, §0). It is called
  out here and in §4.1.1 so the FDX pass applies it; editing `dlpack-extension.md` is out of this
  FKC deliverable's scope. Negative-stride acceptance is thus [consumer-ahead] until both the FDX
  V13 reversal and the `KernelCaps` signed-stride flag (§12.2) land.
- **FDX-side prose fixes (the sibling spec's `Q4_K_M` §13.2 wording, its §15 "lossless" claim,
  its §14 cross-citation) are noted but not edited here.** This task writes the FKC final; the
  matching FDX edits (use `Q4K` by code in §13.2 prose; downgrade §15 to "lossless except slot
  names + rank ≤ 6"; cite the shared unknown-value rule) are called out in this spec's Resolved-
  critique and §11.1 so the FDX pass can apply them, but editing `dlpack-extension.draft.md`
  is out of this deliverable's scope.

---

## Note (2026-06-17) — FKC referenced to two new FDX features

This dated note records an additive revision folding the **two new FDX features** into the FKC
tensor vocabulary — and, per the same resolved-decision pass, a third tensor-layout capability
(negative strides) plus the resolved quant scale-operand rule. Every tensor fact is described by FDX
symbol only; **FDX remains the single normative code owner** (§0) — FKC references the codes/section
numbers and re-numbers nothing. Sources: `docs/specs/_drafts/fdx-addition-gather.md` (paged /
indexed-residency) and `docs/specs/_drafts/fdx-addition-affine.md` (affine symbolic extents); the
negative-stride / single-place / never-panic decisions are from the resolved-decisions pass
(2026-06-17).

1. **Paged / indexed-residency (gather) operand kind (§3.9.1).** Added the vocabulary for a
   contract to declare "this input is an FDX indexed-residency tensor": the tensor-descriptor
   `fdx.gather { kind: paged_blocks, block_table, context_lens }` block (§3.2), referencing FDX's
   `FDXIndexedResidency` / `FDXBlockTable` / `FDX_FLAG_HAS_GATHER` / `FDX_GATHER_PAGED_BLOCKS` and
   the `Capability::DlpackExtGather` admissibility token. The block-table and context-lens are
   ordinary **separate `accept.inputs`** (matching the as-built `OpKind::PagedAttn` ABI), named by
   role in `fdx.gather.*` — the **single-place rule** (the operand is the authority; the FDX buffer
   references point at the same buffers; consistency cross-checked by FDX V20). New validator §10.14
   (gather coherence) + `GatherIncoherent` / `GatherNotYetSupported` errors; §11.1 makes unknown
   `fdx.gather.kind` a meaning-bearing typed error; §12.1 confirms the operands fit the dispatch
   key while the gather descriptor gates admissibility.

2. **Affine-extent references in operand / cost expressions (§3.9.2, §4.4, §4.5).** Added
   `fdx.extent_kind: scalar|range|affine` to the tensor descriptor (§3.2), referencing the FDX
   `FDXExtent` `Affine` kind (`c0 + Σ cᵢ·SymIdᵢ`, ≤ 4 terms; `cap_kind = EXPLICIT`). §4.5 extends
   symbolic-extent tolerance with the affine arm (the kernel resolves the affine value through the
   same `SymEnv` as a `Range`; capacity stays a concrete bound; `min ≤ value ≤ capacity` guard).
   §4.4 lets a **cost expression** reference an affine live extent by its operand extent name,
   evaluated **at capacity** in v1 (live re-eval remains [consumer-ahead], same missing `&SymEnv`).
   The affine *coefficients* live in the FDX sidecar, never in the FKC contract (advertisement vs
   description split). New validator §10.15 (affine-extent coherence); §11.1 makes unknown
   `extent_kind` a meaning-bearing typed error (never silently demoted to `range`). This is the
   persistent-decode `k_len = cached_len + new_tokens` advertisement (FDX affine §6 / §13.7).

3. **`reverse_strides` (negative-stride) layout capability — the universal ban REVERSED (§4.1.1,
   §3.2, §4.1, §10.4).** Added a fifth per-operand layout flag, `reverse_strides`, alongside
   `strided` / `broadcast_stride0` / `start_offset`. A kernel declaring `reverse_strides: accepted`
   receives **negative-stride** tensors (e.g. `Op::Flip`) **zero-copy** — the operand's
   `byte_offset` points at the iteration-first element with a non-negative buffer `start_offset`,
   the invariant `Layout::flip` already maintains. The planner materializes a non-negative copy
   (`Op::Contiguize`-class, `IS_COPIED`) **only** for a consumer that does **not** declare the
   capability, or for a bare standard-DLPack export to a non-cooperating external ecosystem —
   **never universally, and never for a capable internal kernel** (an internal zero-copy `Op::Flip`
   between capable kernels is preserved). This reverses the earlier "negative strides forbidden on
   export" rule; the *description* side (FDX carrying signed `int64` strides + the signed-stride OOB
   `[min,max]` validator, removing FDX V13's non-negative-export ban) is a [cross-spec dependency]
   on the FDX pass (FDX owns the tensor axis, §0). New §10.4 coherence arm
   (`reverse_strides: accepted` ⇒ `strided: accepted`, `LayoutIncoherent` otherwise); §12.2 records
   the flag and notes the `KernelCaps` signed-stride field that honors it is [consumer-ahead].

4. **Quant scale-operand rule updated to the resolved single-place decision (§3.9.3, §6, §10.6).**
   If a kernel's ABI takes the scale as a **separate graph input**, it is an FKC `accept.inputs`
   operand named in `fdx.quant.scale_operand` — **not** also an FDX `scale_buffer`. The FDX
   `scale_buffer` is **only** for sidecar-bundled scales (GGML INLINE, MX separate-buffer). Each
   scale is described in exactly one place, enforced by the new §10.6 `ScaleDoubleDeclared` check.

5. **`register_full_with_source` → `Result` recorded as a firm prerequisite** (not merely flagged)
   per the resolved conversion-plan decision (never-panic): see the updated "Findings deliberately
   not fully closed" entry. The CONSTITUTION-CONFLICT in §10.10 stands; the de-panic lands with the
   importer implementation.

Items 1, 2, and 3 are **additive and [consumer-ahead]**: the FDX gather/affine structs and the FDX
signed-stride description (V13 reversal) are the 2026-06-17 FDX draft additions / edit (no code
yet). An importer that reaches a `gather`- or `affine`-bearing operand before those FDX codes land
returns the corresponding typed error (`GatherNotYetSupported`, or the §11.1 unknown-meaning-bearing
policy for `extent_kind: affine`) rather than fabricating a descriptor — the same
`MxNotYetRegistrable` discipline FKC already applies to MX; a `reverse_strides: accepted` kernel
registers (it is a kernel fact), but no negative-stride tensor reaches it until the FDX V13 reversal
and the `KernelCaps` signed-stride flag land.

---

## Note (2026-06-17) — three forward-compat touches (ImplId basis, cost provenance, structure-key predicates)

This dated note records a second additive 2026-06-17 revision: three forward-compatibility touches
that are **purely additive** (everything above is intact, no field renumbered, FDX remains the
single normative owner of dtype/quant/granularity/substrate codes — §0). They prepare FKC for two
ahead-of-it consumers (the Judge mid-rebuild, and Baracuda's deferred telemetry / kernel-
specialization feed) without coupling to either's internals. Sources:
`baracuda/docs/fuel-ask-telemetry-2026-06-17.md` (the `DispatchRecord`/`MissRecord`/`ImplId` ask and
the `StructureKey`/`OperandKey` shape) and its companion `kernel-specialization.md`; the cost-
provenance and Judge-agnosticism decisions are from the resolved-decisions pass.

1. **`ImplId` basis (new §4.11).** Stated that the tuple **`(BackendId, op, dtypes, kernel_source,
   kernel_revision_hash)` IS the canonical stable `ImplId`** for dispatch-telemetry /
   specialization consumers — a projection of the existing dispatch key + source + revision (no new
   identifier, no function pointer, P9). Baracuda's `{ Baracuda(symbol) | Vendor(which) |
   FuelNative(which) }` maps **directly from `BackendId + kernel_source`** (the discriminant is the
   already-declared `kernel_source`; `entry_point` is the Baracuda symbol). §13 summary item 8 and
   §4.12 reference it. Marked **[consumer-ahead: deferred Baracuda telemetry feed]** — the JSONL
   record emission is a separate opt-in feature with no consumer in this spec; FKC commits only to
   the basis tuple and the direct `kernel_source` mapping, not to a wire format or the enum's frozen
   encoding (co-defined with Baracuda).

2. **Cost provenance — `declared` | `judge_measured`, both first-class (§3.3, §4.4, §10.8a, §12.3,
   §11).** Revised the cost block to carry a required `provenance:` field: a contract's cost is
   **EITHER author-`declared` OR explicitly `judge_measured`**, and **both are first-class and
   visible**. An author-declared cost is a **prior the Judge refines**; a judge-measured cost
   records measurement provenance and is equally official. The cost lint was reworded from the old
   "non-placeholder cost required" to the COST_RULE: **"cost must be declared OR explicitly
   judge_measured; a bare/placeholder/omitted-without-marker cost still fails"** (new §10.8a,
   `CostProvenanceMissing` / `PlaceholderCost`; no hidden gaps). The §12.3 bulk-fill note was
   reconciled (a family-default cost is an explicit `declared` opt-in, not a silent `unknown_cost`),
   and §11 records that the Judge refining a `declared` cost flips the binding's provenance to
   `judge_measured`. **Precision stays author-declared + Judge-audited** (§4.8, unchanged). Added a
   **Judge-agnosticism caveat** (§4.4, §10.8a): FKC depends only on "the Judge exists and
   refines/bootstraps cost" — **no specific Judge mechanism is named or assumed** (no
   flash-vs-decomposed, no cross-check path); the Judge is mid-rebuild and FKC stays agnostic to its
   internals.

3. **Structure-key-granularity fast-path predicates (§4.2, new §4.12).** Extended the fast-path
   predicate vocabulary so a structure-specialized kernel's admissibility is expressible as
   predicates: **divisibility buckets** (`inner_div(<role>) % 16|8|4|2 == 0` / `any`), **vec-width**
   (`vec_width(<role>) >= v2|v4|v8` / `scalar`), and **inner-contiguous** (`inner_contiguous(<role>)`
   — packed inner axis under strided outer). Such a kernel **imports as a TIGHT-predicate contract**
   (admissible only in its cell); a kernel with only `any`/floor predicates is a **generic
   contract**. A structural **"miss"** — the planner's best admissible match at a key is a generic
   contract, the desired tight cell unregistered — therefore **falls out of ordinary contract
   matching** (no bolt-on mechanism), and is exactly the `MissRecord.wanted` demand signal the
   deferred feed (§4.11) would emit. New §4.12 records the axis-for-axis alignment with Baracuda's
   `StructureKey` **`OperandKey { contig, bcast_mask, vec_width, inner_div, flipped }`** as
   **[consumer-ahead]** — **`flipped` already maps to `reverse_strides`** (§4.1.1, the only
   load-bearing-today axis); `inner_div`/`vec_width`/`inner_contiguous` are admitted into the
   vocabulary now, but their specialization consumer (the AOT/JIT matrix-selection feed) is deferred
   with the telemetry feed. The canonical `structure_key(...)` callable is Baracuda's to ship; FKC
   owns only the predicate surface that projects onto it, and does not reimplement the key.

All three are additive: a v1 contract that omits the structure predicates is a valid generic
contract; the `ImplId` basis and structure-key alignment add no required field (only `cost.provenance`
is newly required — touch 2), and no existing field, code, or section number changed.

---

## Note (2026-06-18) — two FKC defect fixes (dangling §10.12 → real rule 16; per-operand awkward-layout) + FDX `AFFINE_BLOCK` reference

This dated note records two defect fixes against the final FKC draft plus the FKC-side reference to
FDX's newly-appended `AFFINE_BLOCK` quant family. All three are **additive / corrective**, change no
existing field or code, and keep FDX the single normative code owner (§0); FKC re-numbers nothing.

1. **Resolved the dangling `§10.12` cross-reference — the FDX-subset drift-guard is now a real §10
   rule (new §10 rule 16).** §0 and §13 cited a "§10.12" cross-spec consistency test ("FKC's
   accepted token set is a subset of FDX's, so the two cannot drift") that **did not exist** in the
   §10 validation list (which ran items 1–15). Added it as a real build-time rule: **§10 rule 16**
   asserts that every dtype / quant `family` / `granularity` / `ggml_dtype` token any FKC contract
   uses is a **member of FDX's normative code table** (resolved through the generated
   `fuel-core-types::dlpack` constants — the same ones FDX's §6.0 conversion fns + mapping test
   pin), returning a typed `FdxTokenNotInTable` on a token FDX has not minted. This makes FKC's
   token set provably a **subset** of FDX's (its FDX-side counterpart is the §6.0 cross-spec test
   that FKC's dispatch-key codes **equal** the FDX constants). The two `§10.12` citations (§0, §13)
   now resolve to **§10 rule 16**; §13's check count went **15 → 16**; `FdxTokenNotInTable` was
   added to the `FkcError` list.

2. **Per-operand awkward-layout granularity (§4.3.1, §3.2, §4.1, §10.5, §3.6, §13 item 4).** The
   §10.5 rule required that `awkward_layout_strategy: handles_strided` ⇒ **every** input have
   `strided: accepted` — which a flash-attention kernel (strided q/k/v + **contiguous-only** aux
   operands like `alibi_slopes` / `seqlens`) cannot satisfy. The strategy is now expressible **per
   operand**: `caps.awkward_layout_strategy` remains the **kernel-wide default**, optionally
   overridden by a per-operand `layout.awkward_layout_strategy` (§3.2). New §4.3.1 defines the
   resolution + precedence (per-operand override → kernel default → `requires_contiguous`), a worked
   flash-attention example (`handles_strided` q/k/v + `requires_contiguous` aux), and the
   per-operand planner consequence (contiguize only the operands whose effective strategy is
   `requires_contiguous` and whose incoming view is non-contiguous). **§10.5 now checks coherence
   PER OPERAND** against that operand's own layout flags, no longer demanding one tolerance across
   every input (`AwkwardStrategyIncoherent`; an unknown strategy at either level is meaning-bearing,
   §11.1). Backends advertise per operand, the planner decides per operand.

3. **FKC references FDX's new `AFFINE_BLOCK` quant family by symbol (§6 table, §3.2, §10.6, §10
   rule 16).** FDX appended `AFFINE_BLOCK` (family **code 4**, FDX §6.2) on 2026-06-18 — the
   nf4/QLoRA-style block-grained affine family (low-bit data + a **separate block-shaped** absmax
   scale operand; block grain via `block_shape`, **NOT** the `PerBlock` granularity, which stays
   MX-only). FKC's §6 quant family table now lists `AFFINE_BLOCK` **by symbol** (no FDX code
   transcribed, nothing re-numbered — `NONE`/`GGML_BLOCK`/`MX`/`AFFINE_INT`/`AFFINE_FLOAT` keep their
   codes), §3.2 adds it to the `fdx.quant.family` token enumeration, §10.6 gives it its own quant-
   coherence arm (block geometry present, granularity ≠ `PerBlock`, scale single-place via the
   `SEPARATE_BUFFER` form), and — like `MX` — it **parse-validates but returns `MxNotYetRegistrable`
   at registration** (no as-built block-quant descriptor target type yet). It is a member of FDX's
   normative table, so it **passes** the §10 rule 16 subset check and is gated only at registration.

---

## Note (2026-06-20) — reconciliation to the adaptive-runtime-fusion decision (G1/G4/G5/G7)

This dated note records the reconciliation of FKC to the 2026-06-20 adaptive-runtime-fusion decision
([10-decisions-log](../architecture/10-decisions-log.md) — recipe principle, two-tier extensibility,
the Fuel-strategist / backend-synthesizer JIT loop). All four touches are **corrective re-scopings or
clarifications** of existing text; no field is renumbered and FDX stays the single normative code owner.

1. **Two "frozen registry" non-goals re-scoped (§1 Non-goals, §9.4) — G4.** The blanket "Not a
   runtime-extensible registry: built at process startup and frozen thereafter" (§1) and "imports
   every source and freezes the registry" (§9.4) conflated *trusted/untrusted* and
   *primitive/fused-metadata*. They are re-scoped into the three-way split: the **primitive `Op` enum**
   and **untrusted user-injected ops/rules** stay closed
   ([09-non-goals](../architecture/09-non-goals.md)); the **kernel binding table** (implementations)
   is **already runtime-extensible** (`extend_global_bindings`, Tier 1); and **trusted,
   Fuel-orchestrated, cost-gated registration of a new *fused-op identity*** is now an architectural
   goal (Tier 2, via the declarative recipe form — append-only, stable never-reused `FusedOpId`s).
   "Registration time" is therefore no longer only "process startup."

2. **A `fused_op` contract MUST carry its recipe (§12.5) — G1.** Made explicit that the graph-side
   `FusedOp` half of a `fused_op` contract is required to supply **both** `decompose` (break-down →
   base map) and `pattern` (build-up → re-fuse); a fused op with no recipe is an **opaque island**
   (invisible to the missing-fusion telemetry, un-re-fusable, and un-lowerable by the optimizer, since
   optimization *is* lower-to-base-map + find-best-cover). `decompose` is **total + never-`panic!`s +
   primitive→self**, and the recipe **always ships with the op**. See the
   [FKC fusion-patterns spec](fkc-fusion-patterns.md) for the declarative form.

3. **The §4.11/§4.12 structural miss distinguished from the closed-world `FusionMissRecord` (§4.12) —
   G5.** §4.11/§4.12 cover only the *structural specialization* miss (a specialized cell unregistered,
   best match a generic contract). Added that the *fusion* miss has its own sequencing: the
   **closed-world `FusionMissRecord`** (recognized fusion-eligible chain, kernel absent, reason
   `NoBackendKernel`, against a known `FusedOpId`) is the **v1 headline**, built first (consumer = a
   `BindingEntry`, Tier 1); the **open-world** co-occurrence signal (`SequenceRecord{fused_as: None}`,
   by observation) is **deferred** (consumer = Tier-2 declarative registration). Canonical:
   `docs/session-prompts/baracuda-telemetry-plan.md` §9.

4. **Cost-provenance framed as the JIT-adoption substrate (§4.4) — G7, light cross-ref.** The existing
   `declared` → `judge_measured` provenance machinery (§4.4) is the correct substrate for the adaptive
   optimizer's explore/exploit loop: a JIT-synthesized kernel arrives with a `declared` cost prior, and
   its empirical *winning* under cost-gated selection (the route picker's adopt/reject call) is the
   measured posterior the Judge records by flipping to `judge_measured`. FKC supplies the honest cost
   surface; Fuel-the-strategist makes the adoption decision (the constitution holds — no backend-side
   opportunity-finding).
