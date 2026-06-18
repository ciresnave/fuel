# Fuel → Baracuda — the kernel boundary as a two-way contract (FDX + FKC + telemetry)

**Status: DRAFT — not sent. PROPOSAL for a cross-project change.**
This is an *outbound proposal* from Fuel to the Baracuda team, framed per Fuel's working
agreement: cross-project changes are *proposed before they are made*, never landed as a
unilateral edit to a sibling repo. Nothing here is a decision Baracuda has agreed to; it is
Fuel's opening position on a shared boundary, written to be answered.

**Author side:** Fuel (`fuel-core-types`, `fuel-dispatch`, `fuel-graph`, planner/executor).
**Counterpart:** Baracuda CUDA kernels (`baracuda` on crates.io; `baracuda-kernels-sys` FFI
surface), kernel-specialization / AOT matrix team.

**This proposal answers, and is the mirror of, the inbound Baracuda ask**
`baracuda/docs/fuel-ask-telemetry-2026-06-17.md` (kernel telemetry + miss reporting;
companion `baracuda/docs/design/kernel-specialization.md`). That ask is *one half* of the
boundary — the telemetry/miss feed. This document proposes treating it as half of a single
coherent two-way contract whose other half is **how a tensor is described to a kernel** and
**how a kernel advertises itself to Fuel**.

**Read-with:** Fuel's two boundary specs (both DRAFT, branch `feat/kernel-contracts-dlpack`):
- `fuel/docs/specs/dlpack-extension.md` — **FDX**, the tensor/storage-description half.
- `fuel/docs/specs/kernel-contract-format.md` — **FKC**, the kernel-advertisement half.

**Scope note (what is committed vs deferred) is in §6.** The short version: Fuel commits to
the *boundary shape* (FDX + FKC carry every fact both halves need) now; the *telemetry
subsystem itself* and Fuel's *formal reply to your ask* are **DEFERRED**, pending a Judge
timing-retention check — the Judge is mid-rebuild, and the answer to your Open Question 1
("do you retain per-(shape, impl) timings?") depends on where that rebuild lands.

---

## 0. TL;DR

Your ask and our specs describe the **same boundary from two ends**, and they already fit:

1. **Tensor description (the inbound-to-Baracuda half).** We propose Baracuda adopt **FDX**
   (Fuel DLPack Extension) for describing the tensors a kernel touches: **standard DLPack for
   every kernel's tensors when serving any non-Fuel ecosystem** (PyTorch / JAX / CuPy / TVM),
   and **FDX = DLPack + an optional versioned sidecar** when serving Fuel. The base
   `DLTensor` is *always* honest standard DLPack; the sidecar only adds facts standard DLPack
   cannot carry (sub-byte / microscaling dtypes, parametric quant, per-axis scales, symbolic
   live-vs-capacity extents, paged/blocked residency, multi-output bundles). A Baracuda kernel
   that speaks standard DLPack already serves the ecosystem; a Baracuda kernel that *also*
   reads the FDX sidecar serves Fuel with no loss of meaning. (FDX spec §1–§4.)

2. **The telemetry / miss feed (your Ask 1 + Ask 2) built on shared identity.** The join
   tokens you need already fall out of the two specs — Fuel does **not** invent a parallel
   identity scheme, and Fuel does **not** reimplement your key:
   - **`StructureKey`** is computed from **FDX operand descriptions**. You ship the callable
     `structure_key(op_class, operands, arch) -> StructureKey`; Fuel **calls it** with FDX
     operand descriptions as input and **never reimplements** it. FDX already carries every
     structural fact the key needs (FDX §4.1).
   - **`ImplId`** derives from **FKC kernel identity**: the tuple `(BackendId, op, dtypes,
     kernel_source, kernel_revision_hash)`. Your
     `{ Baracuda(symbol) | Vendor(which) | FuelNative(which) }` maps **directly** from
     `BackendId + kernel_source` (FKC §4.11).
   - **The "miss" signal falls out of FKC planner matching** — no bolt-on detector. A
     structure-specialized kernel imports as a *tight-predicate* contract; a generic strided
     kernel imports with `any`/floor predicates. A miss is exactly "the best admissible match
     at this key is a *generic* contract" (FKC §4.2, §4.12). That **is** your
     `MissRecord.wanted`.

3. **One design decision of Fuel's makes your Ask 2 actually work.** Fuel's
   **negative-strides-first-class** decision (reversed 2026-06-17) keeps the *flipped-layout
   demand signal visible*. If Fuel normalized away negative strides universally (the old,
   now-withdrawn rule), a flipped operand would never reach a kernel as flipped — the
   `flipped: bool` axis of your `OperandKey` would be permanently `false`, and the demand for a
   flip-specialized kernel would be invisible. First-class negative strides preserve exactly
   the demand signal your chicken-and-egg trap (your Ask 2 rationale) is about. (FDX §3.2.1;
   FKC §4.1.1.)

---

## 1. Why this is one contract, not two asks

Your doc frames a data feed. Our specs frame a tensor format and a kernel format. They are the
same boundary because **the feed's join tokens are facts the format already carries**:

| Your ask needs… | …which is already a fact in… |
|---|---|
| a structure key over operand layout | **FDX operand description** (strides → contiguity; stride-0 → broadcast; stride sign → flipped; dtype + sub-byte/quant). FDX §4.1 |
| a stable, pointer-free `ImplId` for `chosen` / `candidates` / `fallback` | **FKC kernel identity** `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)`. FKC §4.11 |
| the `flipped` axis to ever be non-trivially observed | **Fuel's negative-strides-first-class decision** — the flip survives to the kernel instead of being normalized away. FDX §3.2.1 / FKC §4.1.1 |
| a "miss" detector | **FKC planner matching** — best admissible match = generic contract. FKC §4.12 |
| dispatch timings | **Fuel's existing autotuner / route picker** (the Judge — *retention is the deferred question*, §6) |

The payoff of treating it as one contract: **no new identity surface anywhere.** You don't
re-derive layout from raw shapes (you canonicalize from the FDX operand description we hand to
your shipped function); we don't reimplement your key; and the records you receive are tagged
with the *same* `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)` tuple Fuel uses
to re-resolve a persisted plan — so a record is re-resolvable on another build, by
construction.

---

## 2. Half one — tensor description: standard DLPack for the ecosystem, FDX for Fuel

### 2.1 The ask

We propose Baracuda kernels describe their tensors in **DLPack**, and accept the **FDX
sidecar** when the caller is Fuel. Concretely, two boundaries (FDX §10):

- **(a) The Fuel kernel ABI.** When Fuel launches a Baracuda kernel, the FDX sidecar is an
  explicit nullable parameter *next to* the `DLTensor` (`*const FDXSidecar`; `null` = "this is
  plain DLPack"). No smuggling, no capsule games. A Baracuda kernel that reads the sidecar gets
  Fuel's full meaning (quant params, symbolic live length via `SymEnv`, paged block tables);
  one that ignores it sees an honest `uint8`/standard tensor.
- **(b) The external ecosystem.** When a Baracuda kernel serves PyTorch / JAX / CuPy / TVM via
  the `__dlpack__` capsule protocol, it emits **standard DLPack** — versioned
  `DLManagedTensorVersioned`, explicit (non-NULL) strides, 256-byte-aligned `data`. The FDX
  sidecar rides `manager_ctx` *only* when the consumer advertised FDX *and* the producer's own
  deleter identity is in force; otherwise only the standard part crosses (FDX §10.2).

This is the whole point of FDX's design (FDX §3, the honesty invariant): **the base `DLTensor`
is never a lie.** A 4-bit MX-quant weight appears to a sidecar-blind consumer as opaque
`uint8` bytes of the correct physical size — never a mislabeled `float16` over packed nibbles,
never a buffer mis-sized by a `size_in_bytes()==0` sub-byte dtype. So a Baracuda kernel that
adopts FDX **loses nothing** with the broader ecosystem: it is standard DLPack to everyone, and
*additionally* rich to Fuel.

### 2.2 What FDX carries that standard DLPack cannot (and why Baracuda would want it)

- **Sub-byte / microscaling dtypes** (F4, F6E2M3, F6E3M2, F8E8M0): packing + block-scale,
  described in `FDXDTypeExt` + `FDXPacking`, never via the native DLPack sub-byte path (FDX §3.4,
  §6.1).
- **Parametric quant**: GGML block-quant (`GgmlDType` by numeric code), OCP-microscaling (MX),
  affine-int dynamic quant, expressed by parameters not a hardcoded enum-per-format (FDX §6.2).
  The scale single-place rule (FKC §3.9.3) keeps each scale in exactly one authoritative place.
- **Symbolic live-vs-capacity extent** (Phase D): a KV-cache axis has a *capacity* K (strides /
  alloc) and a *live* `k_len ≤ K` resolved per token via a `SymEnv` (FDX §3.1, §6.4). For your
  attention-variant specialization, this is the difference between planning once and re-planning
  per token.
- **Affine live extents** (`k_len = cached_len + new_tokens`) for persistent decode, planned
  once, evaluated from the `SymEnv` every token (FDX §6.4 affine addition).
- **Paged / blocked residency** (vLLM-style KV cache as a single FDX tensor: honest `uint8`
  block pool + block-table sidecar) (FDX §6.9 gather addition).
- **256-byte data alignment and explicit strides** — DLPack v1.2+ hard rules FDX obeys
  strictly on the external boundary (FDX §3.2, §3.3), exactly what a CUDA consumer assumes.

None of this is a *requirement* you adopt wholesale. The minimum viable adoption is: **speak
versioned standard DLPack on boundary (b); accept a nullable `*const FDXSidecar` on boundary
(a) and read the fields your kernel cares about.** Everything else is opt-in per kernel.

---

## 3. Half two — the telemetry / miss feed on shared identity

This half *is* your Ask 1 + Ask 2. We accept the framing wholesale; what we add is **where each
join token comes from**, so neither side maintains a second copy.

### 3.1 `StructureKey` — computed from FDX operand descriptions; Baracuda ships the callable

We agree with your "you should not reimplement it." We go one step further and state *why* you
don't have to: **an FDX operand description is the canonical input to your
`structure_key(op_class, operands, arch) -> StructureKey`.** Fuel calls your shipped function
and passes FDX operand descriptions; Fuel never derives the key itself (FDX §4.1).

FDX already carries every structural fact your `OperandKey` axes need:

| Baracuda `OperandKey` axis | FDX fact it is derived from |
|---|---|
| `contig: Contig \| InnerContig \| Strided \| Broadcast` | base `DLTensor.strides` (row-major ⇒ Contig; inner stride 1 ⇒ InnerContig; else Strided) — FDX §3.2 |
| `bcast_mask` | a **stride-0** axis on `DLTensor.strides` — FDX §4.1 |
| `flipped: bool` | the **sign** of a stride (negative ⇒ flipped) — FDX §3.2.1 |
| `inner_div` / `vec_width` | alignment via tiling hints (`FDXTiling.alignment_bytes`) + the 256-byte data rule — FDX §6.5, §3.3 |
| `dtype` | base `DLTensor.dtype` + `FDXDTypeExt` for sub-byte/MX — FDX §6.1 |

Deliberately, **no `structure_key` field is added to FDX** (FDX §4.1, principle P3:
description, never decision). The key is a *derivation over* the description, owned by your one
shipped function. Baking a downstream consumer's derived value into a pure-description struct
would (a) duplicate a fact the operand description already implies, (b) violate
description-vs-decision, and (c) couple FDX's ABI to your keying scheme. FDX *describes* the
structure; Baracuda *derives* the key.

### 3.2 `ImplId` — derives from FKC kernel identity

Your `DispatchRecord.chosen` / `candidates[]` / `MissRecord.fallback` need a stable,
pointer-free implementation id. FKC already serializes exactly that: the tuple **`(BackendId,
op, dtypes, kernel_source, kernel_revision_hash)` IS the canonical stable `ImplId`** (FKC
§4.11). It is the persisted-plan re-resolution key plus the `kernel_source` tag — every field
is data, no function pointer (FKC principle P9). Your enum maps **directly**, no reconciliation
table:

- `BackendId::Cuda` + `kernel_source: "baracuda"` → `Baracuda(entry_point_symbol)` (the
  `entry_point` IS the Baracuda symbol).
- `kernel_source: "cublas" | "cudnn" | …` → `Vendor(which)`.
- a portable CPU/native kernel → `FuelNative(which)`.

The discriminant is `kernel_source`, which the provider already declares per kernel
(front-matter default + per-kernel override). `(op, dtypes, kernel_revision_hash)`
distinguishes cells and pins the revision. FKC defines **no new identifier**.

This is the one piece your ask says you "can't specify without us" (your action item 2). Our
position: **the `ImplId` enum is co-defined with you and its wire encoding frozen jointly**, but
the *basis tuple* is settled — it is FKC kernel identity, and it falls out of facts the contract
already owns. We are not asking you to wait on a new identity scheme; we are telling you it
already exists.

### 3.3 The "miss" signal — falls out of FKC planner matching

Your Ask 2 is the demand signal, and it is the hard one because of the chicken-and-egg trap:
Fuel routes around a slow layout, so the layout never appears in traces, so the fast kernel is
never built. We address it structurally, not with a separate detector:

- A **structure-specialized** kernel imports as a **tight-predicate contract** — its
  admissibility is the conjunction of its structure predicates (`inner_div(role) % 16 == 0`,
  `vec_width(role) >= v4`, `inner_contiguous(role)`, `reverse_strides`), so it is a candidate
  **only** for shapes in its cell (FKC §4.2, §4.12).
- A **generic strided** kernel imports with `any`/floor predicates and is admissible
  everywhere.
- A **structural miss** is then *definitionally* "at this dispatch key, the tightest admissible
  contract is the **generic** one" — the desired tight cell isn't registered. **No
  miss-detection mechanism is needed**; the miss is observable as "best admissible match =
  generic contract," which **is** your `MissRecord.wanted` (FKC §4.12).

The FKC predicate vocabulary was deliberately built axis-for-axis onto your `OperandKey` (FKC
§4.12 mapping table), so a structure-specialized contract's admissibility predicate **is** its
structure key. We do not own your key; we own the *predicate surface* that projects onto it
without drift.

### 3.4 Why negative-strides-first-class is load-bearing for your Ask 2

This is the decision we most want you to register, because it directly protects your demand
signal. Fuel **reversed** (2026-06-17) the earlier rule that banned negative strides on export
and normalized every flipped view to a non-negative copy. Under the new rule (FDX §3.2.1, FKC
§4.1.1):

- **FDX describes negative strides as first-class.** A flipped/reversed view (an `Op::Flip`)
  is a real zero-copy DLPack tensor with signed `int64` strides; the OOB guarantee survives via
  a signed touched-range check (FDX validator V13), exactly the invariant `Layout::flip`
  already maintains.
- **Acceptance is a per-kernel FKC capability** (`layout.reverse_strides: accepted`), not a
  blanket property. A Baracuda kernel that walks signed strides declares it; one that doesn't,
  doesn't.
- **Normalization is the planner's choice, gated on the consumer — never universal.** The
  planner inserts a non-negative materialized copy **only** when the chosen consumer cannot
  take negatives, or for a bare external-DLPack handoff. Between capable internal kernels, the
  flip stays a zero-copy view.

The consequence for you: **`flipped` is a live demand axis.** A flipped operand reaches a
flip-capable kernel as flipped, and reaches a non-capable kernel as a *miss* (best match =
generic, with a normalizing copy). Either way the demand for a flip-specialized kernel is
**visible in the miss histogram**. Had Fuel kept the old normalize-everything rule, every
flipped layout would have been laundered into a contiguous copy before any kernel saw it,
`flipped` would be permanently `false`, and your Ask 2 would never surface flip demand. This is
`flipped` ↔ `reverse_strides` (FKC §4.12), the one structure-key axis that is **load-bearing
today**.

---

## 4. What this looks like concretely (the delivery, mapped to your asks)

We adopt your staging and your format wholesale where we can:

- **`StructureKey` token** — your string/`u64` encoding, computed by *your* shipped
  `structure_key` from FDX operand descriptions. (Your delivery §, your design doc §2/§5.)
- **`ImplId`** — `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)`, mapped onto
  your `{ Baracuda | Vendor | FuelNative }` per §3.2. Enum + wire encoding co-defined and
  frozen jointly.
- **`DispatchRecord` / `MissRecord`** — your shapes, your JSONL, one record per line, keys in
  the agreed encoding. v1 = batch (offline report at our release/run cadence); v2 = live,
  forward-compat only, designed later.
- **Privacy** — opt-in, off by default, single config flag; aggregated counts per key; coarse
  mode (key + counts) and detailed mode (+ timings). We endorse this fully — it matches Fuel's
  "telemetry is never baked into the static contract" line (FKC §1 non-goals: FKC carries only
  the static advertisement; live telemetry is queried separately).

On your specific Open Questions, our *current* answers (subject to the deferral in §6):

1. **Per-(shape, impl) timings retained, or only the winner?** — **DEFERRED.** This is the
   crux that gates our reply. The autotuner is the Judge, which is **mid-rebuild**; whether it
   retains per-candidate timings (making your `candidates[]` free) or only the winner depends
   on where the rebuild lands. We will not promise `candidates[]` until we know. Per your own
   note, **winner + time alone is still very useful**, and the **miss-key histogram alone
   unblocks you** — so the deferral does not block your critical path.
2. **Granularity** — we lean toward your aggregated histograms over per-dispatch records; Fuel
   dispatch rates in decode are high. Confirmable once the Judge retention shape is known.
3. **`est_speedup`** — likely inferred from the fallback's `DispatchRecord` rather than
   estimated at miss time; we'd rather drop the field than hold the dataset for it (your action
   item 4).
4. **Sampling** — open; depends on aggregation cost, again Judge-dependent.

---

## 5. What we ask of Baracuda (mirror of your "what we need from you")

1. **Adopt FDX as the Fuel-facing tensor description**, standard DLPack as the
   ecosystem-facing one (§2). Minimum: versioned standard DLPack on the external boundary;
   accept a nullable `*const FDXSidecar` on the Fuel ABI. FDX is a Fuel-owned spec — we are
   proposing you *consume* it, and we want your review of the struct shapes *before* they
   freeze (FDX is still DRAFT).
2. **Confirm `structure_key`'s input contract accepts FDX operand descriptions.** You ship the
   callable; we want to confirm the operand-description shape FDX produces is a complete input
   to it, so we never reimplement the key (§3.1, FDX §4.1).
3. **Co-define `ImplId`** on the basis tuple `(BackendId, op, dtypes, kernel_source,
   kernel_revision_hash)` and the direct `kernel_source → {Baracuda|Vendor|FuelNative}`
   mapping; freeze the wire encoding jointly (§3.2, your action item 2).
4. **Register the negative-strides-first-class decision** and that `flipped` ↔
   `reverse_strides` is the load-bearing-today structure-key axis (§3.4). Confirm your
   `OperandKey.flipped` derivation matches FDX's signed-stride description.
5. **Agree the miss signal is "best admissible match = generic contract"** (§3.3) so neither
   side builds a redundant detector.

## 6. What Fuel commits to vs what is DEFERRED

**Committed now (the boundary shape):**
- FDX carries every structural fact `structure_key` needs, with **no `structure_key` field
  added** (FDX §4.1). The operand description is a complete input to your callable.
- `ImplId` = FKC kernel identity tuple; **no new identifier**, the `kernel_source` mapping is
  direct (FKC §4.11).
- The miss signal falls out of FKC planner matching; the FKC predicate vocabulary projects
  axis-for-axis onto your `OperandKey` (FKC §4.12).
- Negative strides are first-class, preserving the `flipped` demand axis (FDX §3.2.1, FKC
  §4.1.1).
- FDX honesty invariant: standard DLPack for the ecosystem is *guaranteed correct* even when
  the sidecar is ignored (FDX §3).

**DEFERRED (explicitly not promised in this draft):**
- **The actual telemetry subsystem** — the JSONL emission, the opt-in flag, the
  `DispatchRecord`/`MissRecord` writer. FKC §4.11 marks this **[consumer-ahead: deferred
  Baracuda telemetry feed]**: the basis tuple and mapping exist, but the *emission* is a
  separate opt-in Fuel feature with no consumer in the current specs.
- **Fuel's formal reply to your ask** — this document is a *draft proposal*, not Fuel's
  answer. The formal reply is gated on:
- **A Judge timing-retention check.** Your Open Question 1 (per-candidate timings vs
  winner-only) is answerable only once the **Judge** (Fuel's autotuner / cost-refinement
  subsystem) settles its rebuild. The Judge is **mid-change**; FKC deliberately stays
  *agnostic* to the Judge's internals (FKC §4.4, the Judge-agnosticism caveat) and depends only
  on "the Judge exists and refines/bootstraps cost." Until the rebuild lands, Fuel will not
  promise `candidates[]`, the aggregation granularity, or the sampling answer.

Per your own doc, **none of this blocks your critical path**: the v1 miss-key histogram alone
is enough to start ranking your build matrix, and that histogram is exactly the
best-match-is-generic outcome of FKC matching — which does not depend on the Judge's timing
retention at all. The Judge dependency is only for the *richer* dispatch records (timings).

---

## 7. Process note (working-agreement framing)

Per Fuel's working agreement, **this is a proposal for a cross-project change, not a unilateral
edit.** Fuel does not modify sibling projects (baracuda, aocl, vulkane, lightbulb, mlmf)
without proposing first. Nothing here has been written into any Baracuda repo, and FDX/FKC
remain DRAFT on a Fuel branch (`feat/kernel-contracts-dlpack`). The struct shapes, the
`ImplId` basis, and the `structure_key`-input contract are offered for your review *before* any
side freezes its half. This document is the proposal; your reply, the joint `ImplId` freeze,
and Fuel's formal answer (post-Judge-check) are the next steps.

---

### References
- Inbound ask: `baracuda/docs/fuel-ask-telemetry-2026-06-17.md` (+ companion
  `baracuda/docs/design/kernel-specialization.md`).
- Fuel FDX spec: `fuel/docs/specs/dlpack-extension.md` (§3 honesty, §3.2.1 negative strides,
  §4.1 `structure_key` input, §6 codes/quant/extents/gather, §10 boundaries).
- Fuel FKC spec: `fuel/docs/specs/kernel-contract-format.md` (§4.1.1 `reverse_strides`, §4.2 /
  §4.12 structure-key predicates + miss, §4.11 `ImplId` basis).
