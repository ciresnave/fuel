# Fuel Kernel Contract Format (FKC) — how a kernel provider advertises to Fuel

**Status:** DRAFT v0.1 (2026-06-17; reconciled 2026-06-20). Design pass — no code yet. **Reconciled
2026-06-20** to the adaptive-runtime-fusion decision
([10-decisions-log](../../architecture/10-decisions-log.md), G1/G4): the §1 / §9.4 "frozen registry"
non-goals are re-scoped (Tier-2 trusted, Fuel-orchestrated, cost-gated runtime fused-op registration
is now a goal; the kernel binding table is already Tier-1 runtime-extensible) and a `fused_op` contract
must carry its recipe (`decompose` + `pattern`) or it is an opaque island. (The published spec carries
the full set of touches; this draft mirrors the two freeze re-scopes + the recipe requirement.)
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
capability.rs, backend.rs, probe.rs}`; and the **sibling FDX spec**
(`docs/specs/_drafts/dlpack-extension.draft.md`) for all tensor (DLPack + extension) vocabulary.
When this draft and the constitution (`docs/architecture/`) conflict, the constitution wins;
flag the conflict.

---

## 0. Current status / handoff

- This is the **advertisement (capability/cost-axis) half** of the kernel boundary. Its sibling,
  **FDX** (`dlpack-extension.draft.md`), is the **tensor/storage-axis** half. FKC *describes a
  kernel*; FDX *describes a tensor handed to that kernel*. They share the dtype / quant /
  symbolic-extent vocabularies but are kept separate concerns (13-interchange: weight ⊥ graph).
  Wherever FKC names a tensor fact (sub-byte dtype, quant family, symbolic axis, substrate), it
  uses the FDX codes so a contract row and an FDX sidecar line up by construction.
- Nothing here is implemented. The contract maps onto existing `fuel-dispatch` types (§12); the
  importer is a new `fuel-dispatch` module (`fkc`) that parses the structured blocks and calls
  the existing `KernelBindingTable::register_full_with_source(...)` / `FusedKernelRegistry`
  registration paths. No new dispatch primitive is required for v1 — FKC is a *serialization +
  authoring surface* over types that already exist.
- Targets the **Intended** architecture (Phase D symbolic extents + per-tier cost vector +
  sessions) while staying loadable by today's code: a v1 contract may declare fields whose
  *consumers* are still ahead (per-tier memory, symbolic cost, MX quant); today's importer reads
  what today's types model and ignores the forward-looking tail (size-prefixed, §11).

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
  BackendId, kernel_source)` + `KernelCaps` + `CostFn` + `PrecisionGuarantee` + `OpParams` is
  fully specified (§12).
- **G6 — Build-time validatable.** Every consistency check that can run at registration runs at
  registration, `Result`-returning, no `try_*` siblings; a CI lint verifies required fields,
  non-overlapping keys, non-placeholder precision/cost (§10).
- **G7 — Versioned, additively extensible.** A format-version field; new fields are additive and
  ignored by older importers; enums are `#[non_exhaustive]`-spirited (§11).
- **G8 — Tensors in DLPack + FDX terms.** Every operand/output is described in standard DLPack
  (`DLDataType` code/bits/lanes, device, layout) plus the FDX extension codes for sub-byte /
  quant / symbolic / substrate facts. FKC never invents a parallel tensor vocabulary.
- **G9 — Never panic; pure description.** Importer errors are typed `Result`. A contract can
  *describe* a kernel that handles awkward layouts internally, but it **declares** that it does
  so (so the planner costs the fused contiguize+op honestly) — it can never hide a decision.

### Non-goals

- Not a transport for **telemetry** (live slot count, per-tier memory pressure, queue depth).
  Telemetry is queried live via the Tier-1 `BackendRuntime` trait; FKC carries only the *static*
  advertisement the optimizer bakes at plan time. The cost fields are written so telemetry can
  override them at pick time (§4.4), but the telemetry itself is out of scope.
- Not a tensor-interchange format (that is FDX) nor a graph-interchange format (that is the base
  map; 13-interchange).
- Not a place to encode within-kernel concurrency (the backend's business — the principled
  exception to "backends don't decide") or device placement (the planner's).
- Not an *untrusted* runtime-extensible registry: arbitrary user-hot-loaded ops/rules and new
  *primitives* stay out (the build-time-closed `Op` enum; [09-non-goals](../../architecture/09-non-goals.md)).
  But the blanket "built at startup and frozen thereafter" claim is **re-scoped** by the 2026-06-20
  adaptive-fusion decision ([10-decisions-log](../../architecture/10-decisions-log.md), G4): the
  **kernel binding table** (implementations) is **already runtime-extensible** (`extend_global_bindings`,
  Tier 1), and **trusted, Fuel-orchestrated, cost-gated runtime registration of a new *fused-op
  identity*** (Tier 2, via the declarative recipe form — append-only, stable `FusedOpId`s) is now an
  architectural goal. FKC is read at import/registration time, but registration time is no longer only
  "process startup."

---

## 2. Design principles

- **P1 — Description, never decision.** Mirrors the constitution. A contract says *what a kernel
  accepts, returns, costs, and guarantees*; it never says *use me here* or *I'll fix it
  silently*. The one nuance: a kernel that contiguizes awkward layouts internally must **declare
  `awkward_layout_strategy = ContiguizeInternally`**, turning a hidden behavior into a costed,
  visible fact (§4.3).
- **P2 — Maximize optimizer visibility (the 01 gate).** Every field exists because the optimizer
  reasons over it. If a kernel knows a fact relevant to placement/cost/precision/layout, the
  contract has a slot for it.
- **P3 — Hybrid, lossless both ways.** The prose is for humans; the structured block is
  authoritative for the importer. The two must not disagree — a lint can re-render the structured
  block's blurb and diff it against the prose blurb (§10.11).
- **P4 — Tensors in DLPack + FDX terms (G8).** dtype = `DLDataType` + optional FDX
  `logical_dtype`; layout facts = explicit capability flags backed by FDX layout vocabulary;
  quant = FDX `family`/`ggml_dtype`/granularity codes; symbolic = FDX `SymId`/capacity vocabulary.
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
- **P8 — Cost may be symbolic.** A cost expression may reference a symbolic extent (`SymId`) so a
  symbolic-shape graph plans once and the cost is evaluated at capacity or as a function of the
  resolved live extent — without re-planning per token (digest §6).
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
  ubiquity; the schema in §3.3 is the contract, YAML is the carrier).

### 3.2 Tensor descriptors (shared by accept + return)

Every operand and output is a **tensor descriptor**, expressed in DLPack + FDX terms (P4). It is
a YAML mapping:

```yaml
# A tensor descriptor.
name: lhs                 # operand role name (diagnostic + maps to FDX view name_hash)
dtypes: [F32, F64, BF16, F16]   # accepted DLPack dtypes (Fuel DType names; see §3.4 dtype table)
dtype_class: float        # optional shorthand: int|uint|float|any (expands per §3.4)
# --- layout capability (richer than one bool — §4.1) ---
layout:
  contiguous: required          # required | accepted | n/a
  strided: accepted             # accepted | rejected   (walks arbitrary strides)
  broadcast_stride0: rejected   # accepted | rejected   (stride-0 axis = broadcast)
  start_offset: rejected        # accepted | rejected   (non-zero byte_offset / view base)
# --- shape / rank ---
rank: any                       # exact int, "any", or a range "2..=4"
shape_constraint: same_as=out   # free predicate vocabulary, §3.5
# --- DLPack-extension (FDX) requirements ---
fdx:
  requires_ext: false           # true ⇒ this operand's meaning needs an FDX sidecar
  quant:
    family: none                # none|GGML_BLOCK|MX|AFFINE_INT|AFFINE_FLOAT  (FDX §6.2)
    ggml_dtype: ~               # e.g. Q4_0 when family=GGML_BLOCK
    granularity: ~              # PerTensor|PerToken|PerChannel|PerBlock (FDX §6.2)
    role: ~                     # activation|weight (FDX ScalePair role)
  sub_byte: ~                   # logical_dtype code when base carries opaque uint8 (FDX §6.1)
  symbolic_extent: tolerated    # rejected|tolerated|required (uses FDX FDXExtent — §4.5)
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
(digest §9).

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
  op_params: <OpParamsSchema>   # which OpParams variant + field constraints (§3.7)

# ===== return contract (§3.6) =====
return:
  outputs:                  # ordered list of OUTPUT descriptors (§3.2) + return rules (§5)
    - name: out
      dtype_rule: passthrough(lhs)     # §5.1
      shape_rule: same_as(lhs)         # §5.2
      layout_guarantee: contiguous     # §5.3
      aliasing: none                   # §5.4
  bundle: ~                 # OR a list of OutputViewSpec for multi-output (§5.5)

# ===== capability + cost + precision + determinism (§4) =====
caps:
  awkward_layout_strategy: requires_contiguous   # §4.3
  fast_paths: [ ... ]       # §4.2 declared fast-path predicates
  in_place: false           # §4.6
  alignment_bytes: 16       # mirrors BackendCapabilities.required_alignment
  access_granularity_bits: 32

cost:
  class: cheap_elementwise  # §4.4 relative cost class (coarse bucket)
  flops: "n"                # symbolic expr over shape/param symbols (§4.4)
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: 4000         # launch overhead (Vulkan command buffer submit)
  memory:                   # per-tier footprint (§4.4)
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
(`dlpack-extension.draft.md` §6.1): `U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3,
F6E2M3, F6E3M2, F4, F8E8M0`. Shorthands expand at import:

| `dtype_class` | expands to |
|---------------|------------|
| `float` | `[BF16, F16, F32, F64]` (+ `F8E4M3` only when the kernel lists it) |
| `int` | `[I8, I16, I32, I64]` |
| `uint` | `[U8, U32]` |
| `any` | every dtype the kernel's `accept.inputs[i].dtypes` enumerates |

Sub-byte dtypes (`F4`, `F6E2M3`, `F6E3M2` — `size_in_bytes()==0`) MUST be paired with an
`fdx.sub_byte` code and an `fdx.quant` block (bit-width + packing come from FDX, never from
`size_in_bytes`). `F8E8M0` is the canonical MX block-scale dtype (digest §8).

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

- **Accept** (per input operand): accepted **dtypes**; accepted **layouts** (the four-flag set,
  §4.1); **shape/rank** constraints (§3.5); **FDX requirements** (quant family/format/granularity,
  sub-byte dtype, symbolic-extent tolerance); **device/substrate**; the **op-param schema**
  (§3.7). Field-by-field semantics in §4.1, §4.5.
- **Return** (per output): **dtype rule** incl. passthrough (§5.1); **shape rule** (§5.2);
  **layout guarantee** (§5.3); **aliasing / in-place** (§5.4); and for multi-output ops the
  **bundle slot specs** (§5.5).

### 3.7 Op-param schema (`OpParamsSchema`)

Names the `OpParams` variant the kernel consumes plus per-field constraints. The variant set is
exactly the as-built `fuel-dispatch::kernel::OpParams` (and `fuel_graph::registry::FusedOpParams`
for fused ops). Example for matmul:

```yaml
op_params:
  variant: Matmul
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

---

## 4. Capability + cost + precision + determinism (the linchpin)

This is the block that lets the planner choose **contiguize-vs-strided-vs-materialize** and rank
admissible candidates. It exists because backends advertise and the planner decides (governing
principle): the kernel states facts; the planner makes the choice.

### 4.1 Accepted layouts — the explicit four-flag set (richer than one bool)

Per input operand, `layout:` carries four independent facts (G3). This replaces today's single
`KernelCaps.strided_input: bool` with the dimensions the digest (§2) and the prompt require:

| flag | values | meaning |
|------|--------|---------|
| `contiguous` | `required` / `accepted` / `n/a` | the kernel can consume dense row-major input |
| `strided` | `accepted` / `rejected` | walks arbitrary strides (transpose/slice as metadata-only views) |
| `broadcast_stride0` | `accepted` / `rejected` | tolerates a stride-0 axis (broadcast without materialize) |
| `start_offset` | `accepted` / `rejected` | tolerates a non-zero `byte_offset` / view base |

These four map directly onto FDX's layout vocabulary (a strided/broadcast/offset operand is an
FDX-describable `DLTensor` with non-trivial `strides`/`byte_offset`). The planner reads them to
decide, **per operand**, whether to insert an `Op::Contiguize` before the kernel — and to *cost*
that insertion (§4.4). Today's importer projects `(strided && broadcast_stride0)` onto the one
`KernelCaps.strided_input` bool and routes `start_offset` through auto-Contiguize (the as-built
behavior; §12.2); the full four-flag set is retained for forward use as `KernelCaps` grows.

> **Convention examples** (sanity-checked against the inventory):
> - fuel-cpu-backend `binary`: `{contiguous: required, strided: rejected, broadcast_stride0:
>   rejected, start_offset: rejected}`.
> - Vulkan `unary` (f32): `{contiguous: accepted, strided: accepted, broadcast_stride0: accepted,
>   start_offset: rejected}`.
> - Metal `unary_kernel_strided`: all four `accepted` (offset-capable via `BufferOffset`).

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
`any_input_broadcast`, `dtype == <D>`, `dim[i] % <k> == 0`, `k_len == sk` (the FlashAttn
static-fast case), `groups == 1`, `depthwise`.

### 4.3 Awkward-layout strategy (the decisive declaration)

`caps.awkward_layout_strategy` is the single most planner-relevant capability fact. It states
**which strategy the kernel uses for layouts it does not have a fast path for** — making the
contiguize-vs-strided-vs-materialize decision visible and costable (the prompt's core
requirement):

| value | meaning | planner consequence |
|-------|---------|---------------------|
| `requires_contiguous` | rejects non-contiguous; needs dense zero-offset input | planner MUST insert `Op::Contiguize` for any non-contiguous operand and **adds its cost** to the candidate |
| `handles_strided` | walks strides directly; no fixup needed | planner passes the strided view through; no contiguize cost |
| `contiguize_internally` | accepts strided but **copies to dense inside the kernel** | planner attributes the contiguize cost to *this kernel* (it is part of the kernel's `bytes_moved` / `cost`), and MUST NOT insert a separate `Op::Contiguize` — the fused contiguize+op is costed honestly as one unit |

`contiguize_internally` is the constitutionally-careful case: a kernel *may* contiguize
internally (some do — MKL matmul relies on the executor's auto-Contiguize; some quant kernels
contiguify callers), but it must **declare** it so the optimizer is not blind to a hidden copy.
Silent internal contiguize without this declaration is a contract violation (digest §3: no silent
materialization).

### 4.4 Cost — a vector contribution, optionally symbolic

The cost block yields the per-node contribution to the optimizer's **cost vector** (digest §5):

- **time axis:** `flops`, `bytes_moved`, `overhead_ns` — these three map onto today's
  `CostEstimate { flops, bytes_moved, kernel_overhead_ns }` (§12.3). Convention: **pessimistic
  upper bound** ("when in doubt, round up").
- **memory axis:** `memory: { device_bytes, host_bytes, disk_bytes }` — the **per-tier**
  footprint (digest §5/§11). Today's `CostEstimate` has no per-tier memory field; the importer
  folds `device_bytes` into the cost-vector memory axis as the consumer lands, and ignores the
  rest until then (size-prefixed forward-compat).
- **`class:`** a coarse relative cost class (a discrete bucket for fast frontier pruning before
  the precise expression is evaluated). Suggested ladder: `free` (metadata-only views) <
  `cheap_elementwise` < `strided_elementwise` < `reduction` < `normalization` < `gemm_like` <
  `attention` < `conv`. The planner uses the class for coarse ordering and the expressions for
  the precise frontier.

**Symbolic cost (P8):** `flops`/`bytes_moved`/`memory.*` are *expressions* over named symbols.
Available symbols: shape dims by role (`lhs.dim[0]`, `m`, `n`, `k`), `n` (= product of output
elements), `dtype_bytes`, op-param fields, and **FDX `SymId`-bound extents** (e.g. `k_len` for a
KV-cache flash kernel). When a graph is symbolic, the planner evaluates the cost **at capacity**
(`Extent::bound()`) for plan-time frontier pruning, and may re-evaluate at the resolved live
extent at route-pick time — **without re-planning** (digest §6). An expression referencing a sym
that the `SymEnv` will bind is legal and is the mechanism that lets one plan serve every token.

The importer compiles each expression into a `CostFn`
(`fn(&[Shape], &[DType], &OpParams, &BackendCapabilities) -> CostEstimate`) by substituting the
named symbols from the shapes/params at call time (§12.3).

### 4.5 Symbolic-extent tolerance (per operand)

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

---

## 5. Return-contract field semantics

### 5.1 `dtype_rule`

How the output dtype is computed. Maps onto `fuel_graph::registry::FusedOp.dtype_rule`
(`fn(&[DType], &FusedOpParams) -> DType`) for fused ops, and is checked against the binding key's
output dtype slot for primitive ops.

- `passthrough(<role>)` — same dtype as the named input (elementwise, unary, affine).
- `fixed(<DType>)` — a constant (FusedSoftmaxCrossEntropy → F32; comparisons → U8/bool).
- `cast(<param>)` — from an op-param target (Cast: output dtype is the output Storage's dtype).
- `dequant(<role>)` — the dequantized wide dtype (Q4_0 → F32).
- `bundle` — per-slot (see §5.5).

### 5.2 `shape_rule`

Maps onto `fuel_graph::registry::FusedOp.shape_rule` (`fn(&[Shape], &FusedOpParams) -> Shape`).

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

### 5.5 Multi-output bundles

A multi-output kernel emits **one** `KernelRef` with `outputs.len() == 1` (the bundle), per the
as-built ABI and 12-multi-output Option C. The contract declares the bundle's slot specs:

```yaml
return:
  bundle:                       # maps to FusedOp.output_views: fn(&[Shape],&[DType],&params)->Vec<OutputViewSpec>
    - { name: y,          dtype_rule: passthrough(u), shape_rule: same_as(u),     layout_guarantee: contiguous }
    - { name: last_state, dtype_rule: passthrough(u), shape_rule: from_params(state), layout_guarantee: contiguous }
```

Each slot maps to an FDX `FDXOutputView { byte_offset, len_elements, dtype, shape, strides,
name_hash }` (FDX §6.8). The importer derives the `output_views` rule that the executor calls to
pre-allocate the bundle via `allocate_bundled_storage`. Slots are independent in dtype/shape/
layout (e.g. F32 `y` + I64 `argmax_idx`).

---

## 6. Worked example A — elementwise binary (representative simple kernel)

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
  awkward_layout_strategy: requires_contiguous     # ← planner inserts Op::Contiguize + costs it
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
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

---

## 7. Worked example B — FlashAttn (complex: symbolic extent, dynamic scalar, GQA)

`fuel_graph::registry` `FLASH_ATTN` fused op, dispatched via `OpParams::FlashAttn`. Exercises:
GQA divisibility, a **symbolic live-vs-capacity KV axis** (`sk` capacity vs `k_len` live), a
**dynamic scalar param** on the `SymEnv`, and an attention cost class.

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
    variant: FlashAttn
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
  class: attention
  flops: "2 * b * hq * sq * k_len * d * 2"   # QK^T + PV, scored at LIVE k_len (symbolic, P8)
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
axis (cost scored at the *live* `k_len`, so the per-token cost is honest while the plan is built
once at capacity), the dynamic-scalar param routed through the `SymEnv`, the precision bound for
the admissibility pre-filter, and the per-tier memory for the cost vector.

> A **Conv2D** contract follows the same shape with `op_params.variant: Conv2D` (the
> `x_shape`/`w_shape`/`out_shape`/`stride`/`padding`/`dilation`/`groups` fields),
> `shape_rule: conv2d(params)`, `cost.class: conv`, `fast_paths: [{when: "groups==1"},
> {when: "depthwise"}]`, and `awkward_layout_strategy: requires_contiguous` (NCHW packed). The
> MKL conv wrapper would additionally declare `kernel_source: "mkl"` as a sibling alternative at
> the same key. A **Q4_0 QMatMul** contract uses `op_params.variant: QMatMul`, the weight
> operand's `fdx.quant: {family: GGML_BLOCK, ggml_dtype: Q4_0}`, `dtype_rule: dequant` for the
> internal widen, and `cost.class: gemm_like`.

---

## 8. Coverage: the format describes every kernel kind

Sanity-checked against the inventory sample. Each kind maps to a contract shape:

| kernel kind | distinguishing contract fields |
|-------------|-------------------------------|
| **elementwise unary/binary** | `op_params.variant: None`/`Affine`/`Clamp`/`PowI`; `cost.class: cheap_elementwise`; layout flag set per backend (CPU contiguous-only, Vulkan/Metal strided+broadcast) |
| **reductions** | `op_params: Reduce/ReduceSumTo/ReduceMaxTo`; `shape_rule: reduce(...)`; precision `bit_stable=false` for atomic-FP GPU reductions (`audited: true` + `none(reason)`) |
| **matmul / BLAS** | `op_params: Matmul`; `shape_rule: matmul`; GQA divisibility constraint; sibling alternatives via `kernel_source` (portable-cpu / mkl / aocl / cublas / cutlass) |
| **conv geometry** | `op_params: Conv2D/ConvTranspose2D/Conv1D`; `shape_rule: conv2d(params)`; `groups`/`depthwise` fast paths |
| **normalization** | `op_params: NormLastDim/SoftmaxLastDim/LogSoftmaxLastDim`; `cost.class: normalization` |
| **rope** | `op_params: Rope`; cos/sin broadcast operands; `shape_rule: same_as(input)` |
| **attention** | `op_params: FlashAttn/PagedAttn`; symbolic KV axis (§7); `cost.class: attention`; up to 6-7 operands |
| **gather/scatter/index** | `op_params: IndexSelect/Gather/IndexAdd/ScatterAdd`; U32 index operand; `aliasing: accumulate(base)` for the adds |
| **cast** | `op_params: Cast`; `dtype_rule: cast(param)`; cross-dtype key (`[src, dst]`) |
| **quantized (static GGML)** | `fdx.quant.family: GGML_BLOCK` + `ggml_dtype`; `dtype_rule: dequant`; `shape_constraint: divisible(k, block)`; flat per-format key |
| **quantized (dynamic affine)** | `fdx.quant.family: AFFINE_*` + `granularity` + `role`; `ScalePair` enriches the key |
| **MX / sub-byte** | `fdx.sub_byte` code + `fdx.quant.family: MX` + F8E8M0 block scale; bit-width from FDX, never `size_in_bytes` |
| **multi-output** | `return.bundle: [...]` → `output_views`; one `KernelRef`, `outputs.len()==1` (§5.5) |
| **in-place** | `caps.in_place: true` + `aliasing: in_place(dst)` (Vulkan `add_assign_scaled`, Metal `const_set`, `Op::WriteSlice`) |

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
    extract the ```fkc block (YAML)  → FkcKernel struct
    validate (§10)                   → Result<(), FkcError>   [no panic]
    resolve entry_point against provider.link_registry → KernelRef
    compute kernel_revision_hash (auto | literal)
    derive dispatch key + KernelCaps + CostFn + PrecisionGuarantee  (§12)
    if op_kind:  KernelBindingTable::register_full_with_source(key, kernel, caps, precision, cost, source)
    if fused_op: FusedKernelRegistry::register(fused_id, backend, BackendImpl { kernel, dtypes, cost, precision, caps, revision })
```

Result: importing a provider's file(s) makes every kernel available with **zero hand-written
registration code**. A new backend (ROCm, TPU) ships an `.fkc.md` bundle and is dispatchable.

### 9.4 Discovery / manifest

A workspace lists its contract sources in a manifest (e.g. a `[fkc]` table in a crate's metadata,
or a `contracts.toml`): each entry is `{ provider, path-or-glob }`. The Fuel build/startup reads
the manifest and imports every source. **This is the bulk, up-front registration path, but it is no
longer the *only* one** (re-scoped 2026-06-20, [10-decisions-log](../../architecture/10-decisions-log.md),
G4): the kernel binding table is append-only and **runtime-extensible** (`extend_global_bindings`,
Tier 1), and a new **fused-op identity** may be registered at runtime through the trusted,
Fuel-orchestrated, cost-gated declarative path (Tier 2, append-only with stable `FusedOpId`s). What
stays frozen is the **primitive `Op` enum** and any *untrusted* user-injected ops/rules
([09-non-goals](../../architecture/09-non-goals.md)). Providers in sibling crates (Baracuda, vulkane)
expose their `link_registry` symbol; the manifest binds the contract files to it.

---

## 10. Build-time validation rules (all `Result`-returning, no `try_*`)

Run at import/registration; the principle is "every check that can run at build time must"
(digest §3). Typed `FkcError` on failure, never panic, never silent fix-up.

1. **Format version** supported (`fkc_version <= FKC_VERSION_MAX`); unknown trailing fields
   ignored (size-prefixed forward-compat, §11).
2. **Required fields present:** `kernel`, exactly one of `op_kind`/`fused_op`, `blurb`,
   `entry_point`, `accept.inputs` (≥1), `return.outputs` (≥1) or `return.bundle`, a `cost` block,
   a `precision` block, `determinism`.
3. **Dtype validity:** every dtype name is a real `DType`; sub-byte dtypes carry `fdx.sub_byte` +
   `fdx.quant` (no reliance on `size_in_bytes()==0`).
4. **Layout coherence:** at least one of `contiguous`/`strided` is acceptable per operand;
   `broadcast_stride0: accepted` ⇒ `strided: accepted` (broadcast is a stride-0 special case).
5. **Awkward-layout coherence:** `awkward_layout_strategy: handles_strided` ⇒ every input has
   `strided: accepted`; `requires_contiguous` ⇒ every input has `contiguous: required` (and a
   `contiguize_internally` kernel folds the copy into its declared `bytes_moved`, §4.3).
6. **Quant coherence (mirrors FDX §8):** `GGML_BLOCK` ⇒ `ggml_dtype` valid + no `ScalePair`;
   `AFFINE_*` ⇒ `granularity ∈ {PerTensor,PerToken,PerChannel}`; `MX` ⇒ scale dtype F8E8M0 +
   `PerBlock`. The dispatch key derived from these MUST be coherent with FDX's dispatch-key form.
7. **Shape/param coherence:** `shape_constraint` predicates parse; op-param `variant` is a real
   `OpParams`/`FusedOpParams` variant; declared fields match the variant's fields.
8. **Cost expressions** parse and reference only in-scope symbols (operand dims by role, op-param
   fields, `n`, `dtype_bytes`, bound `SymId`s); no division-by-zero at capacity.
9. **Precision coverage lint:** `audited: false` + all-null ⇒ flagged UNAUDITED; the always-built
   CPU backend MUST have ≥1 `bit_stable_on_same_hardware: true` contract per primitive op (digest
   §4). `determinism: nondeterministic` ⇒ `bit_stable=false` + `audited: true`.
10. **Dispatch-key non-overlap:** registering the **same** `entry_point` (resolving to the same
    `KernelRef`) twice at one `(op, dtypes, backend)` key is a hard error (mirrors the as-built
    `register_full_with_source` duplicate-pointer panic — but FKC catches it at parse time as a
    typed `Result`, before the registration panic). Distinct `entry_point`s at one key are legal
    sibling alternatives.
11. **Prose/structured agreement:** the prose blurb (first non-empty line of the section) MUST
    equal the structured `blurb:`. A re-render lint diffs them (P3).
12. **No pointers in the file:** the file contains only data + symbolic `entry_point` strings;
    any literal address is rejected (P9).

`FkcError` variants: `UnsupportedVersion`, `MissingField`, `BadDType`, `LayoutIncoherent`,
`AwkwardStrategyIncoherent`, `QuantIncoherent`, `ShapeConstraintParse`, `BadOpParamsVariant`,
`CostExprParse`, `UnauditedPrecision`, `MissingBitStableCoverage`, `DuplicateEntryPoint`,
`BlurbMismatch`, `EntryPointUnresolved`.

---

## 11. Format versioning

- **`fkc_version`** (file front-matter) is the single discriminator. v1 is this document.
- **Additive growth:** new optional fields go into the structured block and are ignored by older
  importers (a v1 importer reading a v2 block keeps the fields it knows, drops the rest). Enums
  (`cost.class`, `awkward_layout_strategy`, quant `family`) are `#[non_exhaustive]`-spirited: an
  unknown enum value is a *typed warning* that demotes the kernel to a conservative default
  (e.g. unknown cost class → `gemm_like` upper bound; unknown layout flag → `rejected`), never a
  parse failure — so a newer provider stays loadable by an older Fuel (digest §7, decision #18).
- **Breaking changes** bump `fkc_version` and the matching `docs/architecture/` section + a
  `10-decisions-log.md` entry on a MAJOR bump (per CLAUDE.md doc discipline).
- **Re-readable post-load:** the binding entry derived from a contract is re-readable and its
  cost/precision overridable by background re-optimization against fresh Judge data (digest §11),
  so a contract's static values are *seeds*, not immutable literals.

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
inline-8 `KernelDTypes` capacity by design.

### 12.2 `caps` → `KernelCaps` (today one bool; FKC keeps the richer set)

Today `KernelCaps { strided_input: bool }`. The importer projects the four-flag layout set:
`strided_input = (any input has strided: accepted)`. `broadcast_stride0` is subsumed (broadcast
is stride-0 strided), and `start_offset: accepted` is recorded but currently routed through
auto-Contiguize (the as-built note: "inputs with non-zero `start_offset` still go through
auto-Contiguize today"). As `KernelCaps` grows new fields (the struct is explicitly
forward-extensible — "Forward-extensible by adding fields"), the importer maps the remaining
three flags directly. `caps.in_place`, `alignment_bytes`, `access_granularity_bits` map onto the
in-place handling and `BackendCapabilities.{required_alignment, access_granularity_bits}`.

### 12.3 `cost` → `CostFn` / `CostEstimate`

The importer compiles each cost expression into a `CostFn`
(`fn(&[Shape], &[DType], &OpParams, &BackendCapabilities) -> CostEstimate`): it substitutes the
named symbols (operand dims, op-param fields, `n`, `dtype_bytes`, `SymId`-bound extents evaluated
at capacity) and returns `CostEstimate { flops, bytes_moved, kernel_overhead_ns }` from
`flops`/`bytes_moved`/`overhead_ns`. The **per-tier `memory` block** has no slot in today's
`CostEstimate` (it is the scalar-ish Layer-1 shape); the importer carries it forward into the
optimizer's per-tier cost vector as that consumer lands (digest §5), and folds `device_bytes`
into bandwidth pressure in the interim. A kernel with no cost expression imports as
`unknown_cost`, then gets the OpKind-family default via the existing
`fill_unset_cost_for_backend` pass — preserving the as-built bulk-fill convention.

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
`entry_point`s at one key become `SmallVec<[BindingEntry; 2]>` siblings; the route picker ranks
them). For a `fused_op` contract: `FusedKernelRegistry::register(fused_id, backend, BackendImpl {
kernel, dtypes: &'static [DType], cost, precision, caps, revision })`, joined to the graph-side
`FusedOp { shape_rule, dtype_rule, output_views, … }` by `FusedOpId`.

**A `fused_op` contract MUST carry its recipe** (the G1 recipe principle,
[10-decisions-log](../../architecture/10-decisions-log.md)): the graph-side `FusedOp` half is required
to supply **both** a `decompose` (fused → primitive subgraph; lowers onto the base map) and a `pattern`
(recognize that subgraph; re-fuse) — see the [FKC fusion-patterns spec](../fkc-fusion-patterns.md).
**Both are mandatory.** A fused op with no recipe is an **opaque island**: invisible to the
missing-fusion telemetry, un-re-fusable, and un-lowerable by the optimizer (optimization *is*
lower-to-base-map + find-best-cover). `decompose` is **total + never-`panic!`s + primitive→self**, and
the recipe **always ships with the op**, never deferred.

### 12.6 `entry_point` → `KernelRef` (the no-pointer indirection)

A provider exposes a **link registry**: a static `&[(symbol_id: &str, KernelRef)]` (or a
`HashMap<&str, KernelRef>`) named in front-matter (`link_registry`). The importer resolves each
contract's `entry_point` string against it → a `KernelRef`. This keeps the *contract file*
pointer-free and serializable (P9), while the *live* `KernelRef` is recovered at import time —
exactly how a persisted plan re-resolves `(backend, op, dtypes, kernel_revision_hash)` → `KernelRef`
on load (digest §7). An unresolved `entry_point` is a typed `EntryPointUnresolved` error.

### 12.7 Return rules → `FusedOp.shape_rule` / `dtype_rule` / `output_views`

`return.outputs[].shape_rule` / `dtype_rule` compile to the graph-side
`fn(&[Shape], &FusedOpParams) -> Shape` and `fn(&[DType], &FusedOpParams) -> DType` (the as-built
`FusedOp` fields). `return.bundle` compiles to `output_views: fn(&[Shape], &[DType],
&FusedOpParams) -> Vec<OutputViewSpec>` (the as-built optional field). For primitive ops these
rules are checked against the binding key rather than registered as functions.

### 12.8 `kernel_revision_hash` → `KernelRevisionHash`

`auto` → `KernelRevisionHash(hash(entry_point ++ revision_base ++ block_bytes))`; a literal hex →
`KernelRevisionHash(u64)`; absent → `KernelRevisionHash::UNTRACKED` (today's step-1 sentinel),
upgradable as the hashing function lands.

---

## 13. Summary — the format decisions that most shape FKC

1. **Hybrid markdown + ` ```fkc ` YAML block**, one `## ` section per kernel, file-level
   front-matter for provider defaults — readable in mdBook, unambiguously parseable.
2. **Tensors in DLPack + FDX terms**: every operand/output descriptor reuses FDX's dtype / quant
   / symbolic / substrate codes, so a contract row and an FDX sidecar line up by construction.
3. **Layout is a four-flag capability set** (`contiguous`/`strided`/`broadcast_stride0`/
   `start_offset`), strictly richer than today's one `strided_input` bool, which the importer
   recovers as a lossy projection.
4. **`awkward_layout_strategy` is the linchpin** — `requires_contiguous` /`handles_strided` /
   `contiguize_internally` makes the contiguize-vs-strided-vs-materialize choice visible and
   costable; internal contiguize is *declared*, never hidden.
5. **Cost is a vector contribution** (compute/bandwidth/overhead + per-tier memory) with a coarse
   `class`, and may be **symbolic** (over `SymId` extents) so a symbolic graph plans once and
   costs honestly per token.
6. **Precision is a structured pre-filter** (`PrecisionGuarantee` with the audited /
   unaudited / audited-no-bound trichotomy), applied before cost ranking; values are Judge-overridable seeds.
7. **Import = registration**: a single bundle file or a glob of per-kernel files auto-registers
   every kernel via the as-built `register_full_with_source` / `FusedKernelRegistry`, with no
   hand-written glue.
8. **Pointer-free + serializable**: `entry_point` symbols (resolved via a provider link registry)
   and a `kernel_revision_hash` replace function pointers, matching the persisted-plan re-resolve
   model.
9. **Build-time validated, never-panic**: 12 typed `Result` checks at import (version, fields,
   dtype/layout/quant coherence, cost-expr scope, precision coverage, key non-overlap,
   prose/structured agreement).
10. **Additively versioned**: `fkc_version`; unknown fields/enum values demote to conservative
    defaults rather than failing, keeping a newer provider loadable by an older Fuel.
