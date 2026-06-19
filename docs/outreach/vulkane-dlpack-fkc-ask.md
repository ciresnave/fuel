# Fuel → Vulkane — the FFI tensor boundary as a shared description (FDX; FKC if/when compute lands)

**Status: DRAFT — not sent. PROPOSAL for a cross-project change.**
This is an *outbound proposal* from Fuel to the Vulkane team, framed per Fuel's working
agreement: cross-project changes are *proposed before they are made*, never landed as a
unilateral edit to a sibling repo. Nothing here is a decision Vulkane has agreed to; it is
Fuel's opening position on a shared boundary, written to be answered.

**Author side:** Fuel (`fuel-vulkan-backend`, `fuel-core-types`, `fuel-graph`, planner/executor).
**Counterpart:** Vulkane (`github.com/ciresnave/vulkane`) — the Vulkan FFI layer that
`fuel-vulkan-backend` builds on (buffers, allocations, command submission, descriptor plumbing).
Same author (ciresnave) as Fuel; this proposal is the propose-first step, not an edit.

**The key difference from the Baracuda ask.** Vulkane is **not a kernel provider.** It does not
ship compute kernels — Fuel's Vulkan kernels are **internal Slang in `fuel-vulkan-kernels`**,
authored and contracted inside Fuel. So the two-half boundary that applies to Baracuda (FDX for
the tensor + FKC for the kernel) is **asymmetric here**: the relevant half for Vulkane is **FDX**
— how a tensor / buffer is *described* as it crosses the Vulkane FFI. **FKC is largely N/A**
for Vulkane today, and only becomes relevant *if* Vulkane ever exposes compute entry points of
its own (§4).

**Read-with:** Fuel's two boundary specs (both DRAFT, branch `feat/kernel-contracts-dlpack`):
- [`docs/specs/dlpack-extension.md`](../specs/dlpack-extension.md) — **FDX**, the
  tensor/storage-description half. *This is the relevant one for Vulkane.*
- [`docs/specs/storage-encoding.md`](../specs/storage-encoding.md) — **Self-describing storage**
  (`DType` + `SType`/`Encoding`): the internal type whose `to_fdx()` projection FDX carries. (This
  spec is written, and the `SType`/`Encoding` types + the `to_fdx()` projection are now implemented
  on this branch — see §7.)

**Scope note (committed vs deferred) is in §6.** The short version: Fuel commits to the
*description shape* — FDX as the honest, optional-sidecar description for any tensor/buffer
crossing the Vulkane FFI — and asks Vulkane to *preserve and pass through* the facts that
description depends on (signed strides, offsets, alignment, and an opaque sidecar pointer). FKC
for Vulkane is **deferred-conditional**: it exists only if/when Vulkane exposes compute.

---

## 0. TL;DR

1. **Tensor description across the Vulkane FFI = FDX.** We propose that any tensor or buffer
   crossing the Vulkane FFI boundary be describable as **FDX**: the base is *always* honest
   standard DLPack (`DLTensor` — explicit strides, 256-byte-aligned data, real dtype), plus an
   **optional versioned sidecar** (`FDXSidecar`) for the facts standard DLPack cannot carry.
   Vulkane never has to *interpret* the sidecar — it only has to **carry it intact** alongside the
   buffer descriptor it already passes. (FDX §3, §10.)

2. **Why this matters specifically for a Vulkan FFI.** A Vulkan buffer is `(VkBuffer handle,
   offset, range)` plus the binding plumbing — exactly the shape where a *byte offset*, *explicit
   strides*, and the **256-byte alignment** are first-class concerns rather than afterthoughts.
   DLPack's data-pointer-plus-`byte_offset` model and `int64` strides map onto a Vulkan buffer
   binding cleanly, and Fuel's **negative-strides-first-class** decision means a flipped/reversed
   view survives the FFI as a real zero-copy buffer binding instead of being forced into a
   materialized copy before it ever reaches a kernel (§3). For an FFI layer, the win is concrete:
   no description fact is lost or laundered as a buffer crosses the boundary.

3. **FKC is mostly N/A — but conditional.** Vulkane ships no kernels, so the kernel-advertisement
   half (FKC) does not apply to it today. *If* Vulkane ever exposes compute entry points
   (a Vulkane-owned compute pipeline Fuel could dispatch to), those entry points would carry **FKC
   contracts** at that point — and Fuel's own `fuel-vulkan-kernels` Slang contracts are the model
   to copy (§4). Until then, this proposal is FDX-only.

---

## 1. Why this is one description, not two asks

For Baracuda the boundary is genuinely two-sided: Baracuda *is* a kernel provider, so it both
*receives* FDX-described tensors and *advertises* itself via FKC. **Vulkane sits on only one
side.** It is the conduit through which Fuel's `(Storage, Layout)` becomes a GPU buffer binding,
and back. So there is exactly one thing to agree on: **the description that rides along with the
buffer must survive the crossing without loss.**

| What Fuel needs of the FFI | …which is a fact the description already carries |
|---|---|
| a buffer's element layout reach the kernel unaltered | **FDX base `DLTensor`** — explicit `int64` strides, `byte_offset`, 256-byte-aligned `data` (FDX §3.2, §3.3) |
| a flipped/reversed view cross without a forced copy | **FDX negative strides, first-class** — signed strides describe an `Op::Flip` as a zero-copy binding (FDX §3.2.1) |
| sub-byte / quant / symbolic facts cross intact | **FDX optional sidecar** (`FDXSidecar`) — carried, not interpreted, by the FFI (FDX §10) |

The payoff of treating this as one description: **Vulkane adds no parallel metadata scheme.** It
does not re-derive layout, it does not unpack the sidecar, it does not own any of FDX's
vocabulary. It carries the base `DLTensor` facts it already has reason to carry (a buffer needs an
offset, a range, and a stride story regardless), plus an opaque sidecar pointer it hands straight
back to Fuel.

---

## 2. The relevant half — tensor/buffer description across the Vulkane FFI

### 2.1 The ask

We propose that any tensor/buffer crossing the Vulkane FFI be described as **FDX**: an honest
standard `DLTensor` base, plus an **optional, nullable, versioned** `FDXSidecar` pointer carried
beside it. Two facets, mirroring the FDX boundary model (FDX §10):

- **(a) The Fuel ↔ Vulkane buffer handoff.** When `fuel-vulkan-backend` describes a tensor to
  Vulkane (to bind a `VkBuffer` for a Slang kernel, or to stage a transfer), the description is
  the base `DLTensor` — `data` pointer + `byte_offset` + `int64` strides + dtype — and an explicit
  **nullable** `*const FDXSidecar` *next to it* (`null` = "this is plain DLPack; the base says
  everything true about these bytes"). Vulkane's job is to **carry both**: bind the buffer from the
  base, and pass the sidecar pointer through to wherever Fuel's Slang kernel can read it. No
  capsule smuggling, no reinterpretation.

- **(b) Honesty when the sidecar is absent or ignored.** This is the load-bearing FDX invariant
  (FDX §3, the honesty invariant): **the base `DLTensor` is never a lie.** A 4-bit quant weight
  appears to any sidecar-blind reader as opaque `uint8` bytes of the correct physical size — never
  a mislabeled `float16` over packed nibbles, never a buffer mis-sized by a sub-byte dtype whose
  `size_in_bytes()` would be `0`. So if Vulkane only ever looks at the base (which is all an FFI
  conduit needs to do), it is **always correct** about the buffer's size and binding.

The minimum viable adoption for Vulkane is therefore *small*: **carry a nullable sidecar pointer
alongside the buffer descriptor, and preserve the base's signed strides, byte offset, and 256-byte
alignment as it builds the `VkBuffer` binding.** Vulkane does not parse the sidecar; Fuel's Slang
kernel does.

### 2.2 What FDX carries that standard DLPack cannot (the facts that must survive the FFI)

These are the facts that exist *in Fuel's graph* and must reach a `fuel-vulkan-kernels` Slang
kernel through Vulkane without being dropped. Vulkane carries them; it does not interpret them.

- **Sub-byte / microscaling dtypes** (`F4`, `F6E2M3`, `F6E3M2`, `F8E8M0` and general low-bit):
  packing + bit order described in `FDXDTypeExt` + `FDXPacking`, never via a native DLPack sub-byte
  path (FDX §6.1; dtype-ext code table at §6.1, e.g. `F4` = code 13, `bit_width 4`).
- **Parametric quant** — three regimes that partition cleanly (FDX §6.2):
  - **`GGML_BLOCK`** (family code 0): GGUF on-disk block layout, **scale baked INLINE** in the
    block struct; `ggml_dtype` IS the format (e.g. `Q4K` = code 12). One self-contained buffer.
  - **`AFFINE_BLOCK`** (family code 4, **the new separate-scale-operand regime**, added 2026-06-18):
    NF4 / QLoRA-style block-grained affine — low-bit data **plus a separate per-block absmax scale
    operand**. `scale_present == 1`, `scale_placement == SEPARATE_BUFFER`, `scale_buffer` a real
    buffer-table index (FDX §6.2 field-semantics, lines ~993–1013). This is the regime where the
    weight buffer and its scale buffer are *two distinct bindings* the FFI must both carry.
  - **`MX`** (OCP microscaling): F8E8M0 per-block scale, the sole user of `scale_granularity=PerBlock`.
- **Per-axis scales** — concrete scale buffers (`f32[1]` / `f32[rows]` / `f32[cols]` /
  `f8e8m0[n_blocks]`) are real entries in the FDX buffer table, each its own `VkBuffer` binding
  (FDX §6.3).
- **Symbolic live-vs-capacity extent** (Phase D, `FDXExtent`): a KV-cache axis has a *capacity* K
  (which sets strides and allocation size — and therefore the `VkBuffer` range) and a *live*
  `k_len ≤ K` resolved per token via a `SymEnv` supplied alongside the data (FDX §3.1, §6.4). For
  a Vulkan binding this is the difference between binding the full-capacity buffer once and
  re-binding per token; the capacity drives the range, the live length drives the dispatch bound.
- **Affine live extents** (`k_len = cached_len + new_tokens`, `FDXExtent.kind=Affine`) for
  persistent decode — planned once, evaluated from the `SymEnv` every token (FDX §6.4 affine
  addition).
- **Paged / blocked residency** (vLLM-style KV cache as a *single* FDX tensor: an honest dense
  `uint8` block-pool base + a block-table sidecar that re-interprets it per sequence) (FDX §6.9
  gather addition, `FDX_FLAG_HAS_GATHER`).
- **256-byte data alignment and explicit signed strides** — DLPack v1.2+ hard rules FDX obeys
  strictly (FDX §3.2, §3.3), and exactly the facts a Vulkan buffer binding is built from.

None of these require Vulkane to *understand* the encoding. They require Vulkane to **not destroy
the binding facts the base carries** and to **forward the sidecar pointer** so the Slang kernel
can read the rest.

### 2.3 Where the scales actually live (so the FFI carries the right number of buffers)

One design fact matters to an FFI layer because it determines **how many buffers cross the
boundary** for a quantized weight. Fuel decided (locked 2026-06-18, branch
`feat/kernel-contracts-dlpack`) that block-affine scales (NF4 / QLoRA) are **sibling operands**,
not embedded inside the weight's storage:

- The weight's encoding (`Encoding::AffineBlock`) declares only the **requirement** — "I need a
  per-block absmax operand of this dtype/granularity" (a `ScaleSpec`, a *descriptor*, not a
  pointer). The **consuming op** (dequant / matmul) binds the actual scale operand as a separate
  graph edge.
- At the kernel boundary, FDX **re-unites** them into one self-describing descriptor: `AFFINE_BLOCK`
  with `scale_placement == SEPARATE_BUFFER`, `scale_buffer` = a real index into the FDX buffer
  table (FDX §6.2).

Why this matters to Vulkane specifically: an `AFFINE_BLOCK` weight crosses the FFI as **two
bindings** — the low-bit data buffer *and* its separate scale buffer — both referenced from the
sidecar's buffer table. A `GGML_BLOCK` weight crosses as **one** self-contained binding (scale
baked inline). The FFI must be able to carry a buffer table with N entries, not assume one buffer
per tensor. (This is also why Fuel did **not** fold scales into the weight's storage: Fuel's
multi-output graph machinery is one-allocation-only, so folding a separate-source scale in would
force a load-time merge-copy and kill zero-copy mmap. Scales as sibling operands keep them a normal
placeable/transferable buffer the planner already handles. Source-format layout is preserved:
GGUF → interleaved/inline; NF4/GPTQ/bnb → separate. Full rationale in
[`docs/specs/storage-encoding.md`](../specs/storage-encoding.md), with the FDX projection in
`dlpack-extension.md` §6.2.)

For Vulkane the takeaway is narrow: **the buffer table is plural; carry every entry's binding
faithfully, scale buffers included.**

---

## 3. Why negative strides + offsets + alignment are load-bearing for a Vulkan FFI

This is the part of the proposal we most want Vulkane to register, because it is where an FFI layer
can silently destroy a fact Fuel depends on.

Fuel **reversed** (2026-06-17) the earlier rule that banned negative strides on export and
normalized every flipped view to a non-negative copy. Under the new rule (FDX §3.2.1):

- **FDX describes negative strides as first-class.** A flipped/reversed view (an `Op::Flip`) is a
  real zero-copy DLPack tensor with signed `int64` strides; the out-of-bounds guarantee survives
  via a signed *touched-range* check over `[min, max]` byte addresses (FDX validator V13), exactly
  the invariant `Layout::flip` already maintains internally.
- **Normalization is the planner's choice, gated on the consumer — never universal.** The planner
  inserts a non-negative materialized copy **only** when the chosen consumer cannot take negatives,
  or for a bare external-DLPack handoff. Between capable internal kernels, the flip stays a
  zero-copy view.

For a **Vulkan buffer binding**, the consequences are concrete and the failure modes are real:

- **Signed strides must survive.** A `VkBuffer` binding is `(handle, offset, range)`; the *stride
  story* is the kernel's, expressed in the descriptor. If Vulkane coerces strides to unsigned or
  silently materializes a "normalized" copy, a flipped view that Fuel intended to pass zero-copy is
  destroyed — and the flip demand signal Fuel's planner relies on is laundered away. **Fuel asks
  Vulkane to carry strides as signed `int64`, untouched.**
- **`byte_offset` must survive.** DLPack's model is a 256-byte-aligned `data` base pointer **plus**
  a `byte_offset` into it (FDX §3.3). A non-zero-offset view (a slice, a bundle slot) maps onto a
  Vulkan binding offset. The offset must reach the binding intact, not be folded into a recomputed
  base that breaks the 256-byte alignment guarantee.
- **256-byte alignment must be honored.** DLPack v1.2+ requires `data` aligned to 256 bytes (FDX
  §3.2). Vulkan has its own alignment requirements for buffer bindings; the two are compatible, but
  only if Vulkane treats the FDX 256-byte contract as a floor it preserves rather than a property
  it recomputes. (Fuel obeys this contract strictly on the boundary; the ask is that the FFI not
  break it.)

The single sentence: **an FFI that preserves signed strides, byte offsets, and 256-byte alignment
lets flipped/sliced/quantized views cross zero-copy; an FFI that doesn't forces a copy and erases
demand signal.**

---

## 4. FKC — mostly N/A for Vulkane (the conditional)

For Baracuda, the second half of the boundary is **FKC** (the Fuel Kernel Contract): how a kernel
advertises its dispatch key, accept/return contracts, and capability/cost surface so Fuel's
optimizer can choose it. **This half does not apply to Vulkane today**, for one structural reason:

> **Vulkane ships no kernels.** Fuel's Vulkan compute is **internal Slang in
> `fuel-vulkan-kernels`**, authored inside Fuel and contracted with Fuel-internal `.fkc.md` files.
> Vulkane is the FFI conduit (buffers, allocations, command submission), not a compute provider.
> There is nothing on the Vulkane side to advertise via FKC.

**The conditional.** *If* Vulkane ever exposes compute entry points of its own — a Vulkane-owned
compute pipeline, a built-in op, anything Fuel could dispatch to as a distinct implementation —
then **those entry points would carry FKC contracts at that point**, exactly as Baracuda's kernels
do and exactly as Fuel's own Slang kernels already do. The model to copy is **Fuel's internal
`fuel-vulkan-kernels` Slang contracts**: each kernel declares its dispatch key, its accept-contract
(what FDX-described tensors it admits, including `layout.reverse_strides: accepted` if it walks
signed strides), its return-contract, and its capability/cost/precision/determinism advertisement
(FKC, [`docs/specs/kernel-contract-format.md`](../specs/kernel-contract-format.md)). At that point
a Vulkane compute entry point would be just another contracted implementation on Fuel's dispatch
surface, and the negative-strides capability (§3) would be declared per-entry-point rather than
assumed.

Until that happens, **this proposal is FDX-only**, and FKC is listed here purely so the boundary is
fully mapped: the kernel-advertisement axis is reserved, not forgotten.

---

## 5. What Fuel asks of Vulkane

Narrow, and mostly "preserve what you already have reason to preserve":

1. **Accept and pass through a nullable `*const FDXSidecar`** alongside the buffer descriptor on
   the Fuel ↔ Vulkane handoff (§2.1). Vulkane does **not** interpret it — it carries it intact to
   where Fuel's Slang kernel reads it. `null` means "plain DLPack; the base is the whole truth."
2. **Preserve signed `int64` strides** on the buffer description — do not coerce to unsigned, do
   not normalize/materialize a flipped view into a copy (§3). The flip must cross zero-copy.
3. **Preserve `byte_offset`** into the 256-byte-aligned base, and **honor the 256-byte alignment**
   contract as a floor when building the `VkBuffer` binding (§3).
4. **Carry a *plural* buffer table** — an `AFFINE_BLOCK` weight crosses as data buffer **plus**
   separate scale buffer(s); a paged KV cache crosses as block pool **plus** block table. Do not
   assume one buffer per tensor (§2.2, §2.3).
5. **Review the FDX struct shapes before they freeze.** FDX is a Fuel-owned spec, still DRAFT
   (branch `feat/kernel-contracts-dlpack`). We are proposing Vulkane *carry* it, and we want
   Vulkane's review of the base-`DLTensor` + sidecar-pointer ABI shape — particularly anything
   that touches buffer binding, offset, stride, or alignment — *before* it freezes (§7).
6. **(Conditional, no action today) Register the FKC conditional** (§4): if Vulkane ever exposes
   compute entry points, they would carry FKC contracts then, modeled on `fuel-vulkan-kernels`.

---

## 6. What Fuel commits to vs what is DEFERRED

**Committed now (the description shape):**
- The base `DLTensor` crossing the Vulkane FFI is **always honest standard DLPack** — correct size,
  real dtype, explicit signed strides, 256-byte-aligned data — *even when the sidecar is absent or
  ignored* (FDX §3, the honesty invariant). An FFI conduit that reads only the base is always
  correct about the binding.
- The sidecar is **optional, nullable, versioned**; Vulkane carries it without interpreting it
  (FDX §10).
- Negative strides are **first-class** in the description; the flip survives the FFI as a zero-copy
  binding when both sides preserve signed strides (FDX §3.2.1).
- The buffer table is **plural** (separate scale buffers, block tables) and FDX is the normative
  owner of the shared dtype/quant/extent numeric codes (FDX §6.0).

**DEFERRED / CONDITIONAL (explicitly not promised in this draft):**
- **FKC for Vulkane** — deferred-conditional on Vulkane ever exposing compute entry points (§4).
  No consumer exists today; the axis is reserved, not built.
- **The concrete FFI ABI signatures** — the exact parameter shape of "buffer descriptor + nullable
  sidecar pointer" is offered for joint review (§5.5), not unilaterally frozen. FDX itself is DRAFT.
- **Per-slot `SType` on multi-output bundles.** The self-describing-storage design is **locked**
  (2026-06-18) and now **implemented** on this branch (`SType`/`Encoding` types, the `stype` field
  on both `Storage` structs, and the `to_fdx()` quant projection — see §7); what remains deferred is
  per-bundle-slot `SType` (v1 is primary-storage-only) and the consuming-op scale-buffer binding.
  Neither blocks Vulkane: Vulkane interacts with the *FDX projection*, specified in
  `dlpack-extension.md` and now produced by `SType::to_fdx()`.

---

## 7. Process note (working-agreement framing) + a divergence found

Per Fuel's working agreement, **this is a proposal for a cross-project change, not a unilateral
edit.** Fuel does not modify sibling projects (baracuda, aocl, vulkane, lightbulb, mlmf) without
proposing first — and Vulkane shares an author (ciresnave) with Fuel, which makes "propose, never
unilaterally edit" the discipline rather than a mere courtesy. **Nothing here has been written into
the Vulkane repo**, and FDX/FKC remain DRAFT on a Fuel branch (`feat/kernel-contracts-dlpack`). The
base-`DLTensor` + sidecar-pointer ABI shape and the stride/offset/alignment preservation contract
are offered for Vulkane's review *before* any side freezes its half.

**Ground-truth status (verified against the live tree, 2026-06-19):**

- **`docs/specs/storage-encoding.md` exists** (authored 2026-06-18 on this branch) as the canonical
  home of the `SType`/`Encoding` self-describing-storage type, so the FDX cross-reference
  (`dlpack-extension.md`:1017) and the read-with link above resolve to a live document.
- **`SType`/`Encoding` are now implemented (steps 1–3 shipped 2026-06-19 on this branch).** Both
  `fuel-memory::Storage` and `fuel-core-types::Storage` now carry a default-empty `stype: SType`
  field (`fuel-memory/src/lib.rs`, `fuel-core-types/src/storage.rs`), and `SType::to_fdx()` projects
  the encoding scheme into the FDX `FDXQuant` sidecar at the view boundary — `GGML_BLOCK` validates
  end-to-end, `AFFINE_BLOCK` emits the separate-scale-buffer descriptor (the scale-buffer index is
  bound by the consuming op). This **strengthens** the Vulkane ask: the FDX projection Vulkane is
  asked to carry is now produced by running code, not just a spec.
- **No `NF4` dtype-ext code; NF4 is the `F4` sub-byte code.** In the actual FDX dtype-ext vocabulary
  NF4 resolves to **`F4` (code 13, `bit_width 4`, MX4)** — there is no separate `NF4` code. FDX
  describes NF4 weights *as* the `F4` sub-byte code plus an `AFFINE_BLOCK` quant family with a
  separate absmax scale (FDX §6.1; §6.2 AFFINE_BLOCK semantics). This proposal uses `F4` accordingly.
- **Code confirmations:** `GgmlDType` and its block sizes (Q4_0 = 18 bytes/block, block_size 32) at
  `fuel-core-types/src/quantized.rs`:26–113. FDX family codes (`GGML_BLOCK` = 0, `AFFINE_BLOCK` = 4
  with `scale_placement == SEPARATE_BUFFER`) at `fuel-core-types/src/dlpack/codes.rs`:79–99 and
  `dlpack-extension.md` §6.2.

This document is the proposal; Vulkane's review of the FFI ABI shape and the stride/offset/alignment
preservation contract is the next step.

---

### References
- Fuel FDX spec: [`docs/specs/dlpack-extension.md`](../specs/dlpack-extension.md) (§3 honesty,
  §3.2.1 negative strides first-class, §3.3 256-byte alignment, §6.1 dtype-ext codes, §6.2 quant
  families incl. `AFFINE_BLOCK` code 4, §6.4 symbolic/affine extents, §6.9 gather/paged residency,
  §10 boundaries).
- Fuel storage-encoding spec (`SType`/`Encoding`):
  [`docs/specs/storage-encoding.md`](../specs/storage-encoding.md) — written; types + `to_fdx()`
  projection implemented on this branch (§7); design referenced from `dlpack-extension.md` §6.2.
- Fuel FKC spec (conditional, §4):
  [`docs/specs/kernel-contract-format.md`](../specs/kernel-contract-format.md).
- Internal Vulkan kernel contracts (the FKC model if Vulkane ever exposes compute): Fuel's
  `fuel-vulkan-kernels` Slang `.fkc.md` corpus (see [`docs/kernel-contracts/`](../kernel-contracts/)).
- Companion outbound proposal (the two-half, kernel-provider case):
  [`baracuda-dlpack-fkc-ask.md`](baracuda-dlpack-fkc-ask.md).
