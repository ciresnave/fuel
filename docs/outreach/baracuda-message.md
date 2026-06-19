# Fuel → Baracuda: the kernel boundary as a two-way contract (FDX + FKC + telemetry)

**From:** Fuel (the DAG-first ML framework; planner/executor + the `fuel-core-types` /
`fuel-dispatch` boundary).
**To:** Baracuda — the CUDA kernel library + the kernel-specialization / AOT-matrix team.

**This is both (a) Fuel's formal answer to your telemetry ask, and (b) a propose-first boundary
contract.** Nothing here has been written into the Baracuda repo. Everything you need to evaluate it
is inlined: the tensor-description ABI (full C header, **Appendix A**), the identity tuple and miss
signal, the proof that the timings you asked about are retained (§4), the **concrete telemetry wire
schema and how it fits the overall Fuel↔kernel loop** (§4.5–4.6), and the **complete FKC
kernel-contract format with a worked Baracuda example** (**Appendix B**). You should not need to
read Fuel's source to answer.

**The headline:** your **Open Question 1** ("do you retain per-(shape, impl) timings, or only the
winner?") is **answered: YES** — see §4, with the exact struct definitions. So your `candidates[]`
is feasible with no new retention work on our side.

---

## TL;DR

Your telemetry ask and Fuel's boundary specs describe the **same boundary from two ends**, and
they fit without a parallel identity scheme:

1. **Tensor description (inbound to your kernels) = FDX.** Standard DLPack for the ecosystem;
   **DLPack + an optional, nullable, versioned sidecar** when the caller is Fuel. The base
   `DLTensor` is *always* honest standard DLPack; the sidecar adds only what DLPack can't carry
   (sub-byte/quant dtypes, per-block scales, symbolic live-vs-capacity extents, paged residency,
   bundles). Minimum adoption: speak versioned standard DLPack externally; **accept a nullable
   `const FDXSidecar*` on the Fuel-facing ABI** and read the fields a kernel cares about.

2. **The telemetry/miss feed = your shapes, on shared identity tokens that already exist:**
   - **`StructureKey`** is computed by **your** shipped `structure_key(op_class, operands, arch)`
     from **FDX operand descriptions** — Fuel calls your function and never reimplements your key.
   - **`ImplId` = the tuple `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)`** — no
     new identifier; your `{ Baracuda | Vendor | FuelNative }` maps directly from
     `BackendId + kernel_source`.
   - **A "miss" is "the best admissible kernel at this dispatch key is a *generic* contract"** — it
     falls out of Fuel's planner matching; no separate detector. That **is** your `MissRecord`.

3. **Open Question 1 — RESOLVED: YES.** Fuel retains per-`(op, dtype, size_class, backend,
   kernel_source)` timings, **including losing alternatives**, as `u64` nanoseconds (§4). The
   earlier "Judge is mid-rebuild / f32-square latencies" framing was stale — the code is ground
   truth and it already retains per-impl timings. `candidates[]` is feasible.

4. **One Fuel decision makes your Ask 2 work: negative strides are first-class.** A flipped layout
   reaches a flip-capable kernel as flipped, and a non-capable kernel as a *miss* — so flip demand
   stays **visible** in the miss histogram instead of being normalized away (§3.4).

**What we ask of you is in §5** (six items, mostly "confirm compatibility / co-freeze a wire
encoding"). **What remains deferred on our side** is the telemetry *emission* layer (the JSONL
writer over the already-retained timings) — retention is done; emission is a self-contained Fuel
feature still to build.

---

## 1. Why this is one contract, not two asks

Your doc frames a data feed; our specs frame a tensor format and a kernel format. They are the
same boundary because **the feed's join tokens are facts the format already carries:**

| Your ask needs… | …which is already a fact in… |
|---|---|
| a structure key over operand layout | the **FDX operand description**: strides → contiguity; a stride-0 axis → broadcast; a stride's **sign** → flipped; dtype + the sub-byte/quant sidecar |
| a stable, pointer-free `ImplId` for `chosen` / `candidates` / `fallback` | the **kernel identity tuple** `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)` |
| the `flipped` axis to ever be observed | **Fuel's negative-strides-first-class decision** — the flip survives to the kernel instead of being normalized away |
| a "miss" detector | **Fuel planner matching** — best admissible match = a generic contract |
| dispatch timings | **Fuel's autotuner ("the Judge")** — per-impl retention **confirmed: YES** (§4) |

The payoff: **no new identity surface anywhere.** You canonicalize your key from the FDX operand
description Fuel hands to your shipped function; Fuel doesn't reimplement your key; and the records
you receive are tagged with the *same* tuple Fuel uses to re-resolve a persisted plan — so a record
is re-resolvable on another build, by construction.

---

## 2. Half one — tensor description (FDX)

### 2.1 The model

FDX = **standard DLPack** + **an optional versioned sidecar**, across two boundaries:

- **(a) The Fuel kernel ABI.** When Fuel launches a Baracuda kernel, the FDX sidecar is an explicit
  **nullable** parameter next to the `DLTensor` (`const FDXSidecar*`; `null` = "plain DLPack"). No
  capsule smuggling. A kernel that reads the sidecar gets Fuel's full meaning (quant params,
  symbolic live length via a `SymEnv`, paged block tables); one that ignores it sees an honest
  standard tensor.
- **(b) The external ecosystem.** Serving PyTorch / JAX / CuPy / TVM via `__dlpack__`, a kernel
  emits **standard DLPack** — versioned `DLManagedTensorVersioned`, explicit (non-NULL) strides,
  256-byte-aligned `data`. The sidecar rides `manager_ctx` *only* when the consumer advertised FDX
  **and** the producer's deleter identity is in force; otherwise only the standard part crosses.

**The honesty invariant (the load-bearing property):** the base `DLTensor` is *never a lie*. A
4-bit quantized weight appears to a sidecar-blind consumer as opaque `uint8` bytes of the correct
physical size — never a mislabeled `float16` over packed nibbles, never a buffer mis-sized by a
sub-byte dtype. So adopting FDX **loses nothing** with the broader ecosystem: a Baracuda kernel is
standard DLPack to everyone, and *additionally* rich to Fuel.

### 2.2 What the sidecar carries (and why a kernel provider wants it)

All of these are inlined as C structs in Appendix A; the relevant ones for dispatch are:

- **Sub-byte / microscaling dtypes** (`F4`, `F6E2M3`, `F6E3M2`, `F8E8M0`): bit width + packing in
  `FDXDTypeExt`, never via a native DLPack sub-byte path.
- **Parametric quant** in `FDXQuant` — block geometry + scale descriptor, by parameters not a
  hardcoded enum-per-format:
  - `GGML_BLOCK` (family 0): GGUF block layout, scale **baked inline**; `ggml_dtype` is the format
    (e.g. `Q4K` = 12). One self-contained buffer.
  - `MX` (family 1): OCP microscaling, F8E8M0 per-block scale (the sole `PerBlock` user).
  - `AFFINE_INT` / `AFFINE_FLOAT` (families 2/3): dynamic per-tensor/token/channel affine.
  - `AFFINE_BLOCK` (family 4): NF4/QLoRA — low-bit data **+ a separate per-block absmax scale
    operand** (`scale_placement = SEPARATE_BUFFER`, `scale_buffer` = a real index into the buffer
    table).
- **Symbolic live-vs-capacity extents** (`FDXExtent`): a KV-cache axis has a *capacity* K (which
  sets strides and allocation) and a *live* `k_len ≤ K` resolved per token via a `SymEnv` passed
  alongside the data. For attention-variant specialization, this is plan-once vs replan-per-token.
  The `AFFINE` extent kind expresses `k_len = cached_len + new_tokens` for persistent decode.
- **Paged / blocked residency** (`FDXIndexedResidency`): a vLLM-style KV cache as a *single* FDX
  tensor — an honest `uint8` block pool + a block-table sidecar that re-interprets it per sequence.
- **A plural buffer table** (`FDXBufferRef[]`): index 0 is the base data; a `role = SCALE` entry is
  a separate scale buffer; gather adds `POOL` / `BLOCK_TABLE` / `CONTEXT_LENS` entries.
- **256-byte data alignment + explicit signed strides** — the DLPack rules FDX obeys strictly,
  exactly what a CUDA consumer assumes.

None of this is wholesale-mandatory. Minimum viable adoption: **versioned standard DLPack on the
external boundary; accept a nullable `const FDXSidecar*` on the Fuel ABI and read the fields your
kernel cares about.** Everything else is opt-in per kernel.

### 2.3 FDX is description, never decision

FDX carries **no cost and no dispatch decision** — it describes a tensor; the planner decides. In
particular, **no `structure_key` field is added to FDX**: your key is a *derivation over* the
operand description, owned by your one shipped function, not a value baked into the description
struct. This keeps FDX's ABI decoupled from your keying scheme (and from ours).

---

## 3. Half two — the telemetry / miss feed on shared identity

This half *is* your Ask 1 + Ask 2. We accept the framing wholesale; what we add is **where each
join token comes from**, so neither side maintains a second copy.

### 3.1 `StructureKey` — your callable, fed FDX operand descriptions

You ship `structure_key(op_class, operands, arch) -> StructureKey`; **Fuel calls it** with FDX
operand descriptions as input and **never reimplements** it. FDX already carries every structural
fact your `OperandKey` axes need:

| Your `OperandKey` axis | FDX fact it derives from |
|---|---|
| `contig: Contig \| InnerContig \| Strided \| Broadcast` | base `DLTensor.strides` (row-major ⇒ Contig; inner stride 1 ⇒ InnerContig; else Strided) |
| `bcast_mask` | a **stride-0** axis on `DLTensor.strides` |
| `flipped: bool` | the **sign** of a stride (negative ⇒ flipped) |
| `inner_div` / `vec_width` | alignment via `FDXTiling.alignment_bytes` + the 256-byte data rule |
| `dtype` | base `DLTensor.dtype` + `FDXDTypeExt` for sub-byte/MX |

### 3.2 `ImplId` — the kernel identity tuple

Your `DispatchRecord.chosen` / `candidates[]` / `MissRecord.fallback` need a stable, pointer-free
implementation id. Fuel's kernel identity **is** exactly that: the tuple

```text
ImplId = (BackendId, op, dtypes, kernel_source, kernel_revision_hash)
```

every field is data, no function pointer. It is Fuel's persisted-plan re-resolution key plus the
`kernel_source` tag. Your enum maps **directly**, no reconciliation table:

- `BackendId::Cuda` + `kernel_source = "baracuda"` → `Baracuda(entry_point_symbol)` (the entry
  point IS the Baracuda symbol).
- `kernel_source = "cublas" | "cudnn" | …` → `Vendor(which)`.
- a portable CPU/native kernel → `FuelNative(which)`.

`kernel_source` is the discriminant the provider already declares per kernel; `(op, dtypes,
kernel_revision_hash)` distinguishes cells and pins the revision. **No new identifier is invented.**
This is the one piece your ask says you "can't specify without us": the **basis tuple is settled**
(it's our kernel identity), and we propose **co-defining and freezing its wire encoding jointly**.

### 3.3 The "miss" signal — falls out of planner matching

Your Ask 2 is the demand signal, hard because of the chicken-and-egg trap: Fuel routes around a
slow layout, so the layout never appears in traces, so the fast kernel is never built. We address
it structurally:

- A **structure-specialized** kernel registers as a **tight-predicate contract** (its admissibility
  is the conjunction of its structure predicates — `inner_div % 16 == 0`, `vec_width >= v4`,
  `inner_contiguous`, `reverse_strides`), so it's a candidate **only** for shapes in its cell.
- A **generic strided** kernel registers with `any`/floor predicates and is admissible everywhere.
- A **structural miss** is then *definitionally* "at this dispatch key, the tightest admissible
  contract is the **generic** one." **No miss-detection mechanism is needed** — the miss is
  observable as "best admissible match = generic contract," which **is** your `MissRecord.wanted`.

Our predicate vocabulary was built axis-for-axis onto your `OperandKey`, so a specialized
contract's admissibility predicate **is** its structure key. We don't own your key; we own the
predicate surface that projects onto it without drift.

### 3.4 Why negative-strides-first-class protects your Ask 2

Fuel **reversed** (2026-06-17) the earlier rule that banned negative strides on export and
normalized every flipped view to a copy. Under the current rule:

- **FDX describes negative strides as first-class** — a flipped view is a real zero-copy DLPack
  tensor with signed `int64` strides; out-of-bounds safety is a signed touched-range check.
- **Acceptance is a per-kernel capability** (`layout.reverse_strides: accepted`), not a blanket
  property — a kernel that walks signed strides declares it.
- **Normalization is the planner's choice, gated on the consumer — never universal.** A
  non-negative copy is inserted **only** when the chosen consumer can't take negatives (or for a
  bare external-DLPack handoff). Between capable internal kernels the flip stays zero-copy.

The consequence for you: **`flipped` is a live demand axis.** A flipped operand reaches a
flip-capable kernel as flipped, and a non-capable kernel as a *miss* (best match = generic, with a
normalizing copy). Either way the demand for a flip-specialized kernel is **visible in the miss
histogram**. Had Fuel kept the old normalize-everything rule, every flipped layout would have been
laundered into a copy before any kernel saw it, `flipped` would be permanently `false`, and your
Ask 2 would never surface flip demand.

---

## 4. Open Question 1 — ANSWERED: YES, per-impl timings are retained

> *Your Open-Q-1: "Do you retain per-(shape, impl) timings, or only the winner?"*

**Yes — per-alternative, including losers, as `u64` nanoseconds.** `candidates[]` is feasible with
no new retention work. The retention exists in two concrete artifacts; here are the exact
definitions (Rust, but the shapes are what matter):

### 4.1 Persistent per-alternative report (one entry per measured alternative, losers included)

```rust
pub struct ProfileEntry {
    pub op:            OpKind,
    pub dtype:         DType,
    pub size_class:    SizeClass,
    pub backend:       BackendId,
    pub device_index:  u32,
    pub latency_ns:    u64,      // median wall-clock per invocation — NOT "f32 squares"
    pub iterations:    u32,
    pub max_rel_error: f32,
    pub kernel_source: String,   // distinguishes sibling impls at the same (op,dtypes,backend) cell
}

pub struct ProfileReport {
    pub version: u32,            // PROFILE_REPORT_VERSION == 2
    pub entries: Vec<ProfileEntry>,
}
```

One `ProfileEntry` **per measured alternative including the ones that lost**; `kernel_source`
distinguishes siblings; it persists as atomic JSON. This is exactly the per-(cell, impl) shape your
`candidates[]` needs.

### 4.2 In-memory query surface, keyed per impl

```rust
// keyed on the same five axes; kernel_source is PART of the key, so siblings don't collide
pub struct HashMapJudge {
    entries: HashMap<(OpKind, DType, SizeClass, BackendId, String), u64>,
}

fn measured_latency_ns(
    &self, op: OpKind, dtype: DType, size_class: SizeClass,
    backend: BackendId, kernel_source: &str,
) -> Option<u64>;
```

A passing test asserts two impls at the identical `(op, dtype, size_class, backend)` cell resolve
to **distinct** latencies, and that an unmeasured sibling **misses** (`None`) rather than borrowing
a neighbour's number. That's the guarantee that makes an `ImplId`-keyed `candidates[]` honest: each
impl carries its own measured number.

### 4.3 The one caveat — coverage, not retention (and it's transient)

What is *retained* is fully keyed (above). What is *populated today* is a narrow profiling matrix:
**F32 only**, an offline square-matmul size ladder (no GEMV / decode-shaped cells), a fixed
primitive set, no online exploration. So today many decode-regime cells (GEMV, non-F32, quantized)
**miss** the oracle (`None` — the correct "no measurement" signal, never a fabricated number).

**This is explicitly transient.** The Judge is slated for extensive expansion — more dtypes
(it "will not be F32-only for long"), judging every op that supplies no declared cost, and
flash-vs-decomposed arm comparison. **Build the feed coverage-agnostic** — read whatever the oracle
holds — and `candidates[]` **densifies automatically** as the matrix grows, with **no telemetry
format or wire change**. Plan for a feed that starts sparse and fills in, not a fixed snapshot.
Crucially, the **miss histogram (§3.3) does not depend on Judge timings at all**, so it unblocks
your critical path regardless of coverage.

### 4.4 Your other open questions, now answerable

- **Granularity** — aggregated per-key histograms, not per-dispatch records. Fuel's decode dispatch
  rate is high; we store at cell granularity already, so aggregated histograms match both your
  preference and our storage shape. No per-dispatch retention offered.
- **`est_speedup`** — inferred from the *retained loser timings* (generic fallback's `latency_ns`
  vs the cell's best), not estimated at miss time. We'd rather drop the field than hold extra data
  to compute it; the retained losers make inference cheap if you want it.
- **Sampling** — feasible as a knob on the emission layer (rate-limit / reservoir over emitted
  records), since aggregation is over a bounded per-key store, not a per-dispatch log.

### 4.5 The concrete telemetry schema (the records you'd receive)

Here is Fuel's proposed wire schema, mirroring your `DispatchRecord` / `MissRecord` shapes. Format
is **JSONL** — one compact JSON object per line, newline-terminated (append-friendly: a long run
streams without rewriting). `ImplId` and `StructureKeyToken` are the join tokens from §3.1/§3.2.

```rust
/// One emitted dispatch decision. Serialized as one compact JSON line.
struct DispatchRecord {
    schema: u32,                              // telemetry wire-format version (NOT the report version)
    structure_key: Option<StructureKeyToken>, // YOUR structure_key over FDX operand descriptions;
                                              //   None until your callable is linked (Coarse can omit)
    chosen: ImplId,                           // the implementation that won this dispatch
    candidates: Vec<Candidate>,               // every admitted alternative + its measured latency
                                              //   (Detailed mode only; empty in Coarse)
    count: u64,                               // aggregated hits for (structure_key, chosen) since flush
}

/// One admitted alternative + its empirical latency (the "loser" rows).
struct Candidate {
    impl_id: ImplId,
    latency_ns: Option<u64>,                  // from the Judge oracle; None = unmeasured cell
                                              //   (oracle miss — never a fabricated 0)
}

/// A structural miss: the tightest admissible contract at this key was GENERIC —
/// a specialized cell would have fit, but none is registered. == MissRecord.wanted.
struct MissRecord {
    schema: u32,
    wanted: StructureKeyToken,                // the specialized cell that WOULD have fit (demand signal)
    fallback: ImplId,                         // the generic contract the planner actually used
    count: u64,
    // est_speedup is deliberately OMITTED — inferable from the fallback's DispatchRecord
    //   (the retained loser timings, §4.1), not estimated at miss time.
}

/// The stable, pointer-free impl id = FKC kernel identity (§3.2). Every field is data.
struct ImplId { backend: BackendId, op: OpKind, dtypes: Vec<DType>,
                kernel_source: String, kernel_revision_hash: u64 }

/// Opaque join token — YOU own the encoding (string or u64). Fuel never derives it.
struct StructureKeyToken(String);
```

As JSONL on the wire (illustrative; `chosen`/`fallback` are `ImplId` objects):

```jsonl
{"schema":1,"structure_key":"mm:innerdiv16:vec8:f16","chosen":{"backend":"Cuda","op":"MatMul","dtypes":["F16","F16","F16"],"kernel_source":"baracuda","kernel_revision_hash":"0x8f3c1a"},"candidates":[{"impl_id":{"...":"baracuda gemm"},"latency_ns":41230},{"impl_id":{"...":"cublas"},"latency_ns":48800}],"count":1024}
{"schema":1,"wanted":"mm:innerdiv16:vec8:flipped:f16","fallback":{"backend":"Cuda","op":"MatMul","kernel_source":"baracuda-generic-strided","kernel_revision_hash":"0x8f3c1a"},"count":37}
```

**Emission modes (opt-in, off by default):**

- **`Off`** (default) — nothing is written; zero overhead, no file opened.
- **`Coarse`** — `(structure_key, chosen)` + aggregated `count`, plus the miss histogram. **No
  `candidates[]`** (no Judge-oracle reads). This alone is enough to start ranking your build matrix,
  and it does **not** depend on Judge timings at all.
- **`Detailed`** — Coarse **plus** `candidates[]` with the per-impl retained latencies (§4.1–4.2).

**v1 = batch/offline** (aggregated counts flushed to a JSONL artifact at run/release cadence);
**v2 = live** (a line per dispatch) is forward-compat only — the `schema` field versions the line,
and v1's shape is a strict subset a live emitter extends (counts collapse to 1, `candidates[]`
unchanged), so adopting v1 does not strand you when v2 lands.

### 4.6 How telemetry fits the overall Fuel↔backend/kernel loop

Telemetry is the **feedback arc of a closed loop** — it is not a side-channel; it is how the
build/specialize half of the system learns what the dispatch half actually needed. The four arcs:

1. **DESCRIBE — FDX (§2).** Fuel hands every operand to a kernel as an honest standard `DLTensor`
   + an optional sidecar. This is the *input* to your `structure_key`: the operand description
   (strides → contiguity/broadcast/flipped, dtype, sub-byte/quant) is exactly what keys a dispatch
   site.
2. **ADVERTISE — FKC (Appendix B).** Every kernel (yours included) declares its dispatch key,
   accept-contract (which FDX-described operands it admits, with structure predicates), cost,
   precision, and identity. Importing the contract **registers** the kernel onto Fuel's dispatch
   surface. A *structure-specialized* kernel registers as a tight-predicate contract; a *generic*
   one registers with floor predicates.
3. **DECIDE — planner + Judge.** For a concrete shape, the planner matches the live FDX operand
   structure against the admissible FKC contracts, prefilters by precision, and costs each candidate
   (FKC `declared` priors, refined by the Judge's empirical per-impl latencies — §4). It picks a
   winner. The Judge **retains** every measured alternative (losers included), keyed by the same
   `kernel_source` that is part of `ImplId`.
4. **REPORT — telemetry (§4.5).** The winner (`chosen`), the admitted alternatives with their
   retained latencies (`candidates[]`), and the structural misses (`MissRecord` = "a tighter cell
   would have fit, but only a generic kernel was registered") are emitted, **keyed by `ImplId`
   (FKC identity) and `StructureKey` (your callable over FDX descriptions).**

The loop closes because arc 4 feeds **your AOT specialization matrix**: a miss histogram entry
(`wanted = mm:innerdiv16:vec8:flipped`) tells you precisely which specialized kernel to build; you
build it; it ships with an FKC contract (arc 2); the planner now admits it (arc 3); the next run's
telemetry shows the miss closing and the new kernel winning (arc 4). **Every join token in arc 4 is
a fact arcs 1–2 already carry** — `ImplId` is FKC identity, `StructureKey` is your function over FDX
descriptions — so nothing in the loop maintains a second copy of any identity, and a record captured
on one build re-resolves on another by construction. The negative-strides-first-class decision
(§3.4) is what keeps the `flipped` axis *alive* through arcs 1→4, so flip demand can ever reach the
miss histogram instead of being normalized away before any kernel sees it.

---

## 5. What Fuel asks of Baracuda

1. **Adopt FDX as the Fuel-facing tensor description, standard DLPack as the ecosystem-facing one.**
   Minimum: versioned standard DLPack on the external boundary; accept a nullable `const
   FDXSidecar*` on the Fuel ABI (Appendix A). Review the struct shapes **before** FDX freezes (it
   is DRAFT).
2. **Confirm `structure_key`'s input contract accepts FDX operand descriptions**, so Fuel never
   reimplements your key (§3.1).
3. **Co-define and freeze the `ImplId` wire encoding** on the basis tuple `(BackendId, op, dtypes,
   kernel_source, kernel_revision_hash)`. The basis is settled; the wire bytes are joint (§3.2).
4. **Confirm your `OperandKey.flipped` derivation matches FDX's signed-stride description**, and
   register the negative-strides-first-class decision (§3.4).
5. **Agree the miss signal is "best admissible match = generic contract"** so neither side builds a
   redundant detector (§3.3).
6. **Review the telemetry wire schema (§4.5) against what your AOT matrix consumes** — confirm the
   `DispatchRecord`/`MissRecord`/`Candidate` fields, the Off/Coarse/Detailed modes, and the v1
   batch / v2 live split capture your demand signal; and confirm the FKC contract format (Appendix
   B) is one your kernels can be advertised through (one contract per CUDA entry point).

The difference this message makes vs. an opening proposal: ask (3)'s `kernel_source` discriminant is
now demonstrably the **live** Judge key (§4.2), not just a spec proposal — so "no second identity
surface" is backed by running code.

---

## 6. What Fuel commits to vs what is deferred

**Committed now (the boundary shape):**

- FDX carries every structural fact your `structure_key` needs, with **no `structure_key` field
  added** (§2.3). The operand description is a complete input to your callable.
- `ImplId` = the kernel identity tuple; **no new identifier** (§3.2).
- The miss signal falls out of planner matching (§3.3); the predicate vocabulary projects
  axis-for-axis onto your `OperandKey`.
- Negative strides are first-class, preserving the `flipped` demand axis (§3.4).
- The honesty invariant: standard DLPack for the ecosystem is *guaranteed correct* even when the
  sidecar is ignored (§2.1).

**Resolved since our opening proposal:**

- **Judge timing-retention** — Open-Q-1 answered YES (§4). `candidates[]` feasible, no new
  retention.

**Still deferred on our side:**

- **The telemetry *emission* layer** — the JSONL writer for the `DispatchRecord`/`MissRecord`
  schema in §4.5, the opt-in flag (Off/Coarse/Detailed), and the `ImplId`/`StructureKey` join.
  Retention is done (§4); the schema and the loop it closes are specified (§4.5–4.6); only the
  emitter is left to build (a self-contained Fuel feature). The wire schema is offered for your
  review — it mirrors your `DispatchRecord`/`MissRecord` shapes, so confirm it captures what your
  AOT matrix consumes.

---

## 7. Process

This is a propose-first cross-project request: Fuel does not edit sibling projects without
proposing first, even though we share an author. Nothing here has been written into the Baracuda
repo; FDX and FKC are DRAFT on a Fuel branch. The struct shapes (Appendix A), the `ImplId` basis,
and the `structure_key` input contract are offered for your review **before** either side freezes
its half.

**Next steps:** your review of the answers above; the joint `ImplId` wire-encoding freeze; and our
build of the emission layer over the now-confirmed retention. The retention dependency that gated
this reply is closed.

---

## Appendix A — the complete FDX C ABI header

This is the full, language-neutral ABI you would accept. The Rust `#[repr(C)]` source and this
header are size-asserted against each other at build time, so the layouts are authoritative (64-bit
little-endian; v1). Index 0 of the buffer table is always the base `DLTensor.data`. Pointer fields
marked "live only" are never serialized (the serialized form replaces a pointer with a byte
offset).

```c
#include <stdint.h>
#include <stddef.h>

/* ===== Standard DLPack (consumed unchanged from dlpack.h v1.3) ===== */

/* DLPack standard flags on DLManagedTensorVersioned.flags. */
#define DLPACK_FLAG_BITMASK_READ_ONLY              (1UL << 0)
#define DLPACK_FLAG_BITMASK_IS_COPIED              (1UL << 1)
#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2)

typedef struct { int32_t device_type; int32_t device_id; } DLDevice;       /* 8  */
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType; /* 4  */

typedef struct {
  void*      data;        /* 256-byte aligned on export                         */
  DLDevice   device;
  int32_t    ndim;
  DLDataType dtype;
  int64_t*   shape;       /* length ndim; capacity bounds for symbolic axes      */
  int64_t*   strides;     /* length ndim; never NULL; may be negative            */
  uint64_t   byte_offset; /* logical start; never folded into `data`             */
} DLTensor;               /* 48 */

typedef struct { uint32_t major; uint32_t minor; } DLPackVersion; /* 8 */

typedef struct DLManagedTensorVersioned {
  DLPackVersion version;
  void*         manager_ctx;   /* FDX sidecar rides here at the capsule boundary  */
  void (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t      flags;
  DLTensor      dl_tensor;
} DLManagedTensorVersioned;    /* 80 */

/* ===== FDX constants ===== */

#define FDX_MAGIC       0x46445800u /* "FDX\0" */
#define FDX_VERSION_1   1u

#define FDX_SYM_NONE       0xFFFFFFFFu
#define FDX_DTYPE_NONE     0xFFFFu
#define FDX_BUFFER_INLINE  0xFFFFFFFFu  /* scale baked inline (GGML)             */
#define FDX_BUFFER_NONE    0xFFFFFFFEu  /* a separate scale not yet bound        */
#define FDX_BLOCK_UNMAPPED 0xFFFFFFFFu

/* FDXSidecar.flags */
#define FDX_FLAG_HAS_DTYPE_EXT        (1u << 0)
#define FDX_FLAG_HAS_QUANT            (1u << 1)
#define FDX_FLAG_HAS_SYMBOLIC         (1u << 2)
#define FDX_FLAG_HAS_TILING           (1u << 3)
#define FDX_FLAG_IS_BUNDLE            (1u << 4)
#define FDX_FLAG_MEANING_REQUIRES_EXT (1u << 5)  /* base bytes alone aren't usable */
#define FDX_FLAG_READ_ONLY            (1u << 6)
#define FDX_FLAG_HAS_GATHER           (1u << 7)
#define FDX_FLAG_HAS_AFFINE_EXTENT    (1u << 8)

/* FDXExtent.kind */
#define FDX_EXTENT_SCALAR 0u
#define FDX_EXTENT_RANGE  1u
#define FDX_EXTENT_AFFINE 2u
#define FDX_AFFINE_MAX_TERMS 4u
#define FDX_CAP_KIND_EXPLICIT   0u
#define FDX_CAP_KIND_AFFINE_MAX 1u

/* FDXIndexedResidency.kind (gather) */
#define FDX_GATHER_NONE         0u
#define FDX_GATHER_PAGED_BLOCKS 1u

/* FDXQuant.family — FDX is the normative owner of these codes */
#define FDX_QUANT_NONE         0xFFFFu
#define FDX_QUANT_GGML_BLOCK   0u   /* GGUF block, scale baked inline             */
#define FDX_QUANT_MX           1u   /* OCP microscaling, F8E8M0 per-block scale   */
#define FDX_QUANT_AFFINE_INT   2u
#define FDX_QUANT_AFFINE_FLOAT 3u
#define FDX_QUANT_AFFINE_BLOCK 4u   /* NF4/QLoRA: low-bit data + separate scale   */

/* FDXScaleGranularity */
#define FDX_SCALE_GRAN_PER_TENSOR  0u
#define FDX_SCALE_GRAN_PER_TOKEN   1u
#define FDX_SCALE_GRAN_PER_CHANNEL 2u
#define FDX_SCALE_GRAN_PER_BLOCK   3u   /* MX-only */

/* FDXScalePlacement */
#define FDX_SCALE_PLACEMENT_INLINE             0u
#define FDX_SCALE_PLACEMENT_SEPARATE_BUFFER    1u
#define FDX_SCALE_PLACEMENT_BROADCAST_PER_AXIS 2u

/* FDXBufferRef.role */
#define FDX_BUFFER_ROLE_DATA           0u   /* index 0 is always the base data    */
#define FDX_BUFFER_ROLE_SCALE          1u
#define FDX_BUFFER_ROLE_ZERO_POINT     2u
#define FDX_BUFFER_ROLE_BUNDLE_BACKING 3u
#define FDX_BUFFER_ROLE_AUX            4u
#define FDX_BUFFER_ROLE_POOL           5u   /* gather: physical block pool         */
#define FDX_BUFFER_ROLE_BLOCK_TABLE    6u   /* gather: per-sequence block ids       */
#define FDX_BUFFER_ROLE_CONTEXT_LENS   7u   /* gather: per-sequence live lengths    */

/* FDXResidency.tier / substrate / backend_id */
#define FDX_TIER_DEVICE 0u
#define FDX_TIER_HOST   1u
#define FDX_TIER_DISK_MMAP 2u
#define FDX_SUBSTRATE_HOST_BYTES 0u
#define FDX_SUBSTRATE_CUDA_UNTYPED 1u
#define FDX_SUBSTRATE_VULKAN_BUFFER 2u
#define FDX_SUBSTRATE_METAL_BUFFER 3u
#define FDX_BACKEND_CPU 0u
#define FDX_BACKEND_CUDA 1u
#define FDX_BACKEND_VULKAN 2u
#define FDX_BACKEND_METAL 3u

/* FDX ggml_dtype codes (mirror GgmlDType numbering) */
#define FDX_GGML_F32 0u
#define FDX_GGML_F16 1u
#define FDX_GGML_Q4_0 2u
#define FDX_GGML_Q4_1 3u
#define FDX_GGML_Q5_0 6u
#define FDX_GGML_Q5_1 7u
#define FDX_GGML_Q8_0 8u
#define FDX_GGML_Q8_1 9u
#define FDX_GGML_Q2K 10u
#define FDX_GGML_Q3K 11u
#define FDX_GGML_Q4K 12u
#define FDX_GGML_Q5K 13u
#define FDX_GGML_Q6K 14u
#define FDX_GGML_Q8K 15u
#define FDX_GGML_BF16 30u

/* FDX logical dtype codes */
#define FDX_DTYPE_U8 0u
#define FDX_DTYPE_I8 1u
#define FDX_DTYPE_U32 2u
#define FDX_DTYPE_I16 3u
#define FDX_DTYPE_I32 4u
#define FDX_DTYPE_I64 5u
#define FDX_DTYPE_BF16 6u
#define FDX_DTYPE_F16 7u
#define FDX_DTYPE_F32 8u
#define FDX_DTYPE_F64 9u
#define FDX_DTYPE_F8E4M3 10u
#define FDX_DTYPE_F6E2M3 11u
#define FDX_DTYPE_F6E3M2 12u
#define FDX_DTYPE_F4 13u
#define FDX_DTYPE_F8E8M0 14u

/* ===== FDX structs (64-bit LE layout) ===== */

/* Sub-byte / microscaling dtype descriptor. Valid iff FDX_FLAG_HAS_DTYPE_EXT. */
typedef struct {
  uint16_t logical_dtype;
  uint16_t bit_width;
  uint8_t  packing;
  uint8_t  lanes;
  uint8_t  sub_byte_bit_order;
  uint8_t  _pad;
  uint32_t reserved[2];
} FDXDTypeExt;            /* 16 */

/* Parametric quant block layout. Valid iff FDX_FLAG_HAS_QUANT.
 * Block count for AFFINE_BLOCK/MX is over the base LOGICAL element shape:
 * for a packed sub-byte payload, logical elems = base_bytes * 8 / bit_width. */
typedef struct {
  uint16_t family;
  uint16_t ggml_dtype;
  uint8_t  block_ndim;
  uint8_t  _pad0[3];
  uint32_t block_shape[4];   /* block extent (logical elements) per tiled axis    */
  int32_t  block_axes[4];    /* which base axes the block tiles; -1 unused         */
  uint8_t  pack_order;
  uint8_t  _pad1[3];
  uint8_t  scale_present;
  uint16_t scale_dtype;
  uint8_t  scale_placement;  /* INLINE | SEPARATE_BUFFER | BROADCAST_PER_AXIS      */
  uint8_t  scale_granularity;
  uint8_t  _pad2[3];
  uint32_t scale_buffer;     /* index into buffers[]; FDX_BUFFER_INLINE if inline  */
  uint8_t  zp_present;
  uint16_t zp_dtype;
  uint8_t  _pad3;
  uint32_t zp_buffer;
  uint8_t  scale_pair_act;
  uint8_t  scale_pair_weight;
  uint8_t  role;
  uint8_t  _pad4;
  uint32_t reserved[6];
} FDXQuant;              /* 100 */

/* Symbolic / dynamic extent (live-vs-capacity). One affine term = coeff*sym_id. */
typedef struct { int64_t coeff; uint32_t sym_id; uint32_t _pad; } FDXAffineTerm; /* 16 */
typedef struct {
  int64_t       c0;
  uint8_t       term_count;
  uint8_t       _pad[7];
  FDXAffineTerm terms[4];
} FDXAffine;             /* 80 */
typedef struct {
  uint8_t   kind;        /* SCALAR | RANGE | AFFINE                                */
  uint8_t   _pad[3];
  uint64_t  min;         /* RANGE: live lower bound; AFFINE: guaranteed minimum    */
  uint64_t  capacity;    /* == base shape[i]; strides key to this                  */
  uint32_t  sym_id;
  uint8_t   sym_scope;
  uint8_t   _pad2[3];
  uint8_t   cap_kind;
  uint8_t   _pad3[3];
  uint32_t  _pad4;
  FDXAffine affine;      /* k_len = cached_len + new_tokens, etc.                  */
  uint32_t  reserved[2];
} FDXExtent;             /* 128 */

typedef struct {
  uint32_t alignment_bytes;
  uint32_t access_granularity_bits;
  uint8_t  tile_ndim;
  uint8_t  _pad[7];
  uint32_t tile_shape[4];
  uint32_t reserved[4];
} FDXTiling;             /* 48 */

typedef struct {
  uint8_t  tier;
  uint8_t  substrate;
  uint8_t  backend_id;
  uint8_t  _pad;
  uint32_t device_index;
  uint8_t  is_mmap_view;
  uint8_t  _pad2[7];
  uint32_t reserved[4];
} FDXResidency;          /* 32 */

typedef struct {
  uint8_t  class_;       /* storage class: SHARED | SESSION | TRANSIENT           */
  uint8_t  _pad[3];
  uint32_t _pad_align;
  uint64_t session_id;
  uint32_t reserved[4];
} FDXStorage;            /* 32 */

/* Multi-output bundle sub-view. Valid iff FDX_FLAG_IS_BUNDLE. */
typedef struct {
  uint64_t byte_offset;
  uint64_t len_elements;
  uint16_t dtype;
  uint8_t  _pad[2];
  uint32_t ndim;
  uint64_t shape[6];
  int64_t  strides[6];
  uint64_t name_hash;
  uint32_t reserved[4];
} FDXOutputView;         /* 144 */

/* Logical→physical block mapping for a paged pool. */
typedef struct {
  uint32_t table_buffer;
  uint16_t id_dtype;
  uint16_t _pad0;
  uint32_t max_blocks_per_seq;
  uint32_t unmapped_sentinel;
  uint32_t layout_flags;
  uint32_t reserved[4];
} FDXBlockTable;         /* 36 */

/* GATHER descriptor: a contiguous physical BLOCK POOL re-interpreted as a
 * logically-gathered tensor (vLLM-style KV cache). Valid iff FDX_FLAG_HAS_GATHER. */
typedef struct {
  uint8_t       kind;
  uint8_t       _pad0[3];
  uint64_t      num_blocks;
  uint64_t      block_size;
  uint32_t      pool_buffer;
  uint32_t      _pad1;
  uint8_t       physical_ndim;
  uint8_t       _pad2[7];
  uint64_t      physical_shape[6];
  int64_t       physical_strides[6];
  uint16_t      element_dtype;
  uint8_t       _pad3[2];
  FDXBlockTable block_table;
  uint64_t      num_sequences;
  uint64_t      max_seq_capacity;
  uint8_t       logical_ndim;
  uint8_t       seq_axis;
  uint8_t       _pad4[6];
  uint64_t      logical_shape[6];
  uint8_t       logical_extents_count;
  uint8_t       _pad5[7];
  FDXExtent     logical_extents[6];
  uint32_t      context_lens_buffer;
  uint32_t      context_len_sym;
  uint8_t       context_len_scope;
  uint8_t       _pad6[3];
  uint32_t      reserved[6];
} FDXIndexedResidency;   /* 1064 */

/* A buffer reference in the side table. Index 0 is always the base data buffer. */
typedef struct {
  uint8_t  role;
  uint8_t  _pad[1];
  uint16_t dtype;
  uint32_t _pad2;
  void*    data;         /* live only: device pointer, NEVER serialized           */
  DLDevice device;
  uint64_t byte_offset;
  uint64_t size_bytes;
  uint32_t ndim;
  uint32_t _pad3;
  uint64_t shape[6];
  int64_t  strides[6];   /* signed                                                */
  uint32_t reserved[4];
} FDXBufferRef;          /* 160 */

/* One SymId -> u64 binding (per-pass call surface). Never serialized. */
typedef struct { uint32_t sym_id; uint32_t _pad; uint64_t value; } FDXSymBinding; /* 16 */
typedef struct {
  uint32_t             count;
  uint32_t             _pad;
  const FDXSymBinding* bindings;  /* live only */
} FDXSymEnv;             /* 16 */

/* The top-level optional, versioned sidecar carried ALONGSIDE a DLTensor.
 * A null FDXSidecar* means "plain DLPack". */
typedef struct {
  uint32_t            magic;          /* FDX_MAGIC                                  */
  uint32_t            version;        /* FDX_VERSION_1                              */
  uint32_t            struct_bytes;
  uint32_t            flags;
  FDXDTypeExt         dtype_ext;
  FDXQuant            quant;
  uint32_t            extents_count;
  uint32_t            _pad0;
  const FDXExtent*    extents;        /* serialized: byte offset                    */
  FDXTiling           tiling;
  FDXResidency        residency;
  FDXStorage          storage;
  uint32_t            buffers_count;  /* PLURAL buffer table                        */
  uint32_t            _pad1;
  const FDXBufferRef* buffers;        /* serialized: byte offset; index 0 = base    */
  uint32_t            views_count;
  uint32_t            _pad2;
  const FDXOutputView* views;         /* serialized: byte offset                    */
  FDXIndexedResidency gather;
  uint64_t            reserved[2];
} FDXSidecar;           /* 1376 */
```

---

## Appendix B — the full FKC kernel-contract format

This is the complete kernel-advertisement format, at the depth needed to support it fully. A
Baracuda kernel ideally ships an FKC contract per entry point; importing the contract **registers**
the kernel onto Fuel's dispatch surface — no hand-written glue. Two things in the body trace back
here: `ImplId` (the identity tuple, B.7) and the structural miss (the predicate ladder, B.4).

### B.0 The model

An FKC file is **markdown + a fenced ` ```fkc ` YAML block per kernel**. The prose is documentation
(rendered by mdBook); the ` ```fkc ` block is the authoritative contract. One `##` section per
kernel. **FDX is the single normative owner of all shared dtype / quant / granularity / substrate
codes** — FKC names them by symbol (the same symbols as Appendix A), never re-listing values, so
the two halves cannot drift. The contract declares, per kernel: identity + dispatch key, an
**accept-contract** (which FDX-described operands it admits), a **return-contract** (output
dtype/shape/layout/aliasing), and the **capability + cost + precision + determinism** advertisement
the planner uses to choose contiguize-vs-strided-vs-materialize.

### B.1 File anatomy + front-matter (provider-wide defaults)

```yaml
---
fkc_version: 1
provider:
  name: baracuda
  backend: Cuda                              # → BackendId::Cuda
  kernel_source: "baracuda"                  # the ImplId discriminant (per-kernel overridable)
  link_registry: baracuda::fkc::ENTRY_POINTS # symbol → KernelRef map (B.7)
  revision_base: "git:8f3c1a"                # provider build id, folded into kernel_revision_hash
---
```

Each `## kernel` section inherits these and may override per kernel.

### B.2 The per-kernel ` ```fkc ` schema (complete)

```yaml
# ===== identity / dispatch key =====
kernel: <string>            # diagnostic name, unique in file
op_kind: <OpKind>           # the Fuel primitive op (e.g. MatMul) — XOR fused_op
fused_op: <FusedOpId>       # OR a fused op id (e.g. FLASH_ATTN); exactly one of op_kind|fused_op
blurb: <string>             # one line; must equal the prose blurb
backend: <BackendId>        # inherited unless overridden
kernel_source: <string>     # inherited; the ImplId discriminant
entry_point: <symbol-id>    # ref into provider link_registry → the kernel function (B.7)
kernel_revision_hash: auto  # OR a quoted hex; "auto" derives from entry_point + revision_base

# ===== accept contract (per input operand) =====
accept:
  inputs:
    - { name: <role>, dtypes: [...], layout: {...}, rank: <n|any|range>,
        shape_constraint: <predicate>, fdx: {...}, optional: <bool> }   # descriptor, B.3
  op_params:                # which OpParams (primitive) / FusedOpParams (fused) variant + constraints
    variant: <Variant>
    fields: { <name>: { kind: <type>, constraint: <expr>, note: <str> } }

# ===== return contract (per output) =====
return:
  outputs:
    - name: out
      dtype_rule: passthrough(<role>) | fixed(<DType>) | promote(...)
      shape_rule: same_as(<role>) | from_params(...) | broadcast(...)
      layout_guarantee: contiguous | strided | same_as(<role>)
      aliasing: none | in_place(<role>) | view_of(<role>)
  bundle: ~                 # OR multi-output slot specs (rank ≤ 6; slot names kept in a side table)

# ===== capability + cost + precision + determinism =====
caps:
  awkward_layout_strategy: requires_contiguous | handles_strided | contiguize_internally  # B.3
  fast_paths: [ { when: <predicate>, class|note: ... }, ... ]   # B.4
  in_place: <bool>
  alignment_bytes: <int>
  access_granularity_bits: <int>

cost:                       # B.5
  provenance: declared | judge_measured     # REQUIRED; both first-class; a bare/placeholder cost is a lint failure
  class: free | cheap_elementwise | strided_elementwise | reduction | normalization | gemm_like | attention | conv
  flops: "<expr over shape/param/extent symbols>"
  bytes_moved: "<expr>"
  overhead_ns: <int>                        # launch overhead
  memory: { device_bytes: "<expr>", host_bytes: <expr>, disk_bytes: <expr> }   # per-tier [consumer-ahead beyond device]

precision:                  # B.6 → PrecisionGuarantee (planner prefilters on this BEFORE cost)
  bit_stable_on_same_hardware: <bool>
  max_ulp: <int|~>
  max_relative: <float|~>
  max_absolute: <float|~>
  audited: <bool>           # false+all-null ⇒ UNAUDITED (CI-flagged); true+all-null ⇒ none(reason)
  notes: <string>

determinism: bitwise | same_hardware_bitwise | nondeterministic    # B.6
```

YAML is parsed in **1.2-core restricted mode**: all enum/token/dtype/symbol values are quoted
strings (no Norway-problem coercion of `no`/`off`; `ggml_dtype: Q4_0` is never a number; hashes are
quoted hex), cost coefficients live inside quoted expression strings, and tabs are a hard error.

### B.3 The tensor descriptor (accept operand / return output)

Every operand/output is described in DLPack + FDX terms. The load-bearing part is **`layout` — five
independent flags, not one bool** (this is how the planner decides, per operand, whether to insert
and *cost* a normalizing copy):

```yaml
- name: lhs
  dtypes: [F16, BF16, F32]        # accepted DLPack dtypes (Appendix A dtype codes); or dtype_class: float|int|uint|any
  layout:
    contiguous: required          # required | accepted | n/a
    strided: accepted             # accepted | rejected  — walks arbitrary NON-negative strides
    broadcast_stride0: accepted   # accepted | rejected  — tolerates a stride-0 (broadcast) axis
    start_offset: accepted        # accepted | rejected  — tolerates a non-zero byte_offset / view base
    reverse_strides: accepted     # accepted | rejected  — tolerates NEGATIVE strides (Op::Flip), zero-copy
    awkward_layout_strategy: ~    # optional per-operand override of caps.awkward_layout_strategy
  rank: 2                         # exact int | "any" | range "2..=4"
  shape_constraint: "same_as=out" # predicate language: same_as / same_rank / rank= / broadcast_to /
                                  #   last_dim_eq / dim[i]= / divisible(dim[i],expr) / capacity_ge(dim[i],sym)
  fdx:                            # FDX (sidecar) requirements
    requires_ext: false           # true ⇒ this operand's meaning needs a sidecar
    quant: { family: none|GGML_BLOCK|MX|AFFINE_INT|AFFINE_FLOAT|AFFINE_BLOCK,
             ggml_dtype: ~|Q4_0|Q4K|..., granularity: ~|PerTensor|PerToken|PerChannel|PerBlock,
             role: ~|activation|weight, scale_operand: ~ }   # scale in ONE place (sidecar XOR separate input)
    sub_byte: ~                   # logical_dtype code when the base carries opaque uint8 (F4=13, ...)
    symbolic_extent: rejected|tolerated|required   # uses FDXExtent (Scalar/Range/Affine)
    extent_kind: ~|scalar|range|affine             # `affine` ⇒ live extent = c0 + Σ cᵢ·SymIdᵢ, read from SymEnv
    gather: { kind: ~|paged_blocks, block_table: ~, context_lens: ~ }   # paged-pool operands (separate inputs)
  optional: false                 # presence inferred from inputs.len()
```

The **dispatch key** is `(op_kind, [each operand's dtype + quant facts in order, then outputs],
backend, kernel_source)`. For a plain kernel the quant facts are empty and it collapses to `(OpKind,
[DType…], BackendId) + kernel_source`. For a quant kernel, `fdx.quant` enriches each operand slot so
`(QMatMul, A=F32×PerToken, W=Q4_0)` and `(QMatMul, A=F32×PerTensor, W=Q8_0)` are distinct keys.

`caps.awkward_layout_strategy` is the single most planner-relevant fact: `requires_contiguous` (the
planner inserts an `Op::Contiguize` — itself an FKC kernel, costed from its own contract — and
*sums* the two costs), `handles_strided` (the kernel walks strides itself), or
`contiguize_internally` (the kernel copies internally and the cost is folded in).

#### B.3.1 `reverse_strides` — negative strides are first-class

A negative stride walks an axis backwards (the operand's `byte_offset` points at the iteration-first
= highest-address element; the buffer's own allocation start stays non-negative, so the touched
range never precedes the allocation). FDX **describes** signed strides; the kernel **declares**
`reverse_strides: accepted` if it walks them (a backward walk costs the same as forward — no fixup);
the planner inserts a normalizing copy **only** for a consumer that did not declare it (or a bare
external-DLPack export), **never** between capable internal kernels. `strided` and `reverse_strides`
are independent — a kernel may accept arbitrary non-negative strides yet reject reversed ones.

### B.4 Structure predicates — the tight-vs-generic axis (maps to your `OperandKey`)

`fast_paths` predicates can express the three structural facts a structure-specialized kernel folds
into constants, so a specialized kernel imports as a **tight-predicate contract** and a "miss" falls
out of ordinary matching (no bolt-on detector):

- **Divisibility buckets** on an operand's inner (fastest-varying) extent:
  `inner_div(<role>) % 16 == 0` / `% 8` / `% 4` / `% 2` / `any` — the ladder
  `%16 ⊐ %8 ⊐ %4 ⊐ %2 ⊐ any`. A `%16` contract is admissible *only* when the inner extent divides
  16, so the planner can tell **before dispatch** whether a shape lands in that tight cell.
- **Vec-width:** `vec_width(<role>) >= v4` over `scalar < v2 < v4 < v8` (derived exactly as your
  `VecWidth` is — from align/stride/dtype).
- **Inner-contiguous:** `inner_contiguous(<role>)` — fastest-varying axis stride 1 even if outer
  axes are strided (your `Contiguity::InnerContig`, distinct from fully `Contig`).

Plus the layout predicates `all_inputs_contiguous`, `any_input_strided`, `any_input_broadcast`,
`any_input_reversed`, `dtype == <D>`, `dim[i] % k == 0`. A kernel declaring **none** of the
structure predicates (widest predicate `any`) is a **generic contract**; one declaring them is
**tight**. The planner matches a shape against the tightest admissible contract; **"the only
admissible contract at this key is the generic one" IS the structural miss** (§3.3) — `MissRecord`
falls out of the same matching pass. This vocabulary was built axis-for-axis onto your `OperandKey`,
so a specialized contract's admissibility predicate **is** its structure key.

### B.5 Cost — provenance + symbolic + two compile targets

Every cost block carries a required `provenance`: **`declared`** (author's prior — pessimistic upper
bound; the Judge refines it) or **`judge_measured`** (populated by empirical measurement). Both are
first-class; a bare/placeholder/zero-sentinel cost is a lint failure (no silent placeholders).
`flops` / `bytes_moved` / `memory.*` are **expressions** over named symbols (shape dims by role,
`m`/`n`/`k`, `n` = output element count, `dtype_bytes`, op-param fields, and **FDX `SymId`-bound
extents** like `k_len`). A symbolic graph evaluates cost **at capacity** (`Extent::bound()`) in v1;
re-evaluating at the resolved live extent is forward-looking. The expression compiles to one of two
targets: a **primitive** `op_kind` cost compiles to `fn(&[Shape], &[DType], &OpParams,
&BackendCapabilities) -> CostEstimate`; a **fused** `fused_op` cost to `fn(&[Shape], &FusedOpParams,
&BackendCapabilities) -> CostEstimate` (no `&[DType]`). `CostEstimate = { flops, bytes_moved,
kernel_overhead_ns }` today; the per-tier `memory` axis is forward-compat.

### B.6 Precision + determinism

`precision` maps to `PrecisionGuarantee` and is a **prefilter that runs before cost ranking** — a
kernel failing the per-call precision floor or the cumulative tolerance budget is not a candidate at
all. `audited: false` + all-null ⇒ `UNAUDITED` (CI-flagged); `audited: true` + all-null ⇒
`none(reason)` (audited, no static bound — e.g. warp-reduction order); bounds present ⇒ a real
bounded claim. These are seeds the Judge refines. `determinism` is a coarse summary: `bitwise`
(bit-identical on any hardware — Flip/Roll/copy), `same_hardware_bitwise` (the common deterministic
case), or `nondeterministic` (atomic FP accumulation / scheduler-dependent reductions — must be
`audited: true` with a `none(reason)` precision; no silent unaudited nondeterminism).

### B.7 Identity + revision hash

The dispatch-telemetry / specialization basis is the **canonical stable `ImplId`** =
`(BackendId, op, dtypes, kernel_source, kernel_revision_hash)` — every field serializable data, no
function pointer; **no new identifier**. `entry_point` is a symbol resolved through the provider's
`link_registry` to the actual kernel function (so the contract is pointer-free on disk).
`kernel_revision_hash` (`auto` ⇒ derived from `entry_point` + `revision_base`) pins the
implementation version, so a persisted plan re-resolves to the exact kernel build. Your
`{ Baracuda(symbol) | Vendor(which) | FuelNative(which) }` maps directly: `kernel_source ==
"baracuda"` → `Baracuda`, the vendor set → `Vendor`, else `FuelNative`.

### B.8 A complete worked example — a Baracuda flash-attention contract

This is a real FKC contract for a Baracuda CUDA kernel (`kernel_source: "baracuda"`,
`entry_point: "baracuda::flash_attn_fwd_f16"`): GQA admissibility, a symbolic KV axis (cost over the
live `k_len`, evaluated at capacity `sk` in v1), audited precision, declared determinism.

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
      fdx: { symbolic_extent: required }   # live k_len from SymEnv; stride keyed to Sk
    - name: v
      dtypes: [F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required }   # k_len ≡ v_len ⇒ SAME SymId
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace)
    fields:
      b: { kind: usize }
      hq: { kind: usize }
      hkv: { kind: usize, constraint: "hq % hkv == 0" }
      sq: { kind: usize }
      sk: { kind: usize, note: "physical K/V capacity" }
      d: { kind: usize }
      k_len: { kind: DynScalar, note: "live attended length <= sk; rides SymEnv" }
      softmax_scale: { kind: f32 }
      causal: { kind: bool }
      window_size_left: { kind: "Option<usize>" }
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
  provenance: declared                       # author prior; Judge refines
  class: attention
  flops: "2 * b * hq * sq * k_len * d * 2"   # QK^T + PV; live-k_len re-eval is forward-looking
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

### B.9 Import = registration

Importing a provider's contract file(s) parses each ` ```fkc ` block, resolves `entry_point`
through the `link_registry` to the kernel function, and **registers** the kernel onto Fuel's
dispatch surface (dispatch key + accept/return contracts + caps + cost + precision + determinism) —
no hand-written registration glue. A bundle file (many `##` sections) and a per-kernel file (one
section) import identically. Duplicate detection is at the resolved-function level (two distinct
`entry_point` strings that resolve to the same function are a typed error). For Baracuda, the
practical shape is: one contract per CUDA entry point, `kernel_source: "baracuda"`, the
`link_registry` mapping your symbols to the FFI functions — and every record the telemetry feed
emits for those kernels is tagged with the `ImplId` that contract defines.
