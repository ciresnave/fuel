# Fuel DLPack Extension (FDX) — tensor interchange for Fuel, kernels, and the ecosystem

**Status:** DRAFT FOR REVIEW (2026-06-17), on branch `feat/kernel-contracts-dlpack`.
Design pass — no code yet. This is the final draft incorporating the correctness /
backward-compat critique sweep (see "Resolved critique" below); the struct shapes are designed
to be addable to `fuel-core-types` behind a `dlpack` feature.
**Scope:** a versioned, *optional sidecar* extension to standard DLPack that lets Fuel
describe tensors whose full meaning exceeds the standard `DLTensor` — sub-byte / microscaling
dtypes, parametric quantization, per-axis scales, symbolic (live-vs-capacity) extents,
multi-buffer quant payloads, multi-output bundles, fine-grained residency/substrate, storage
class — **without ever lying in the base `DLTensor`**.
**Audience:** Fuel (`fuel-core-types`, executor, planner), Baracuda (CUDA kernels), Vulkane /
`fuel-vulkan-kernels` (Slang), and any external DLPack consumer (PyTorch / JAX / CuPy / NumPy /
TVM).

**DLPack version targeted:** FDX rides the **versioned** struct family introduced in
**DLPack v1.0** (`DLManagedTensorVersioned`, `DLPackVersion {major, minor}`,
`DLPACK_FLAG_BITMASK_*`). It is written and validated against the **DLPack v1.3** header
(`DLPACK_MAJOR_VERSION == 1`, `DLPACK_MINOR_VERSION == 3`). Crucially, FDX assumes the
**v1.2+ strides rule**: *strides are never NULL on a versioned export* (see §3.2). The legacy
unversioned `DLManagedTensor` path is **not** an FDX export target; FDX never emits it.

Authoritative inputs: the architecture-constraints digest
(`docs/specs/_research/architecture-constraints.md`); the as-built core types in
`fuel-core-types/src/{dtype.rs, symbol.rs, shape.rs, quant_scale.rs, quantized.rs,
capability.rs, backend.rs, probe.rs, device.rs}`; the DLPack ABI (`dlpack.h` v1.3:
`DLDataType`, `DLDevice`, `DLTensor`, `DLManagedTensor`, `DLManagedTensorVersioned`,
`DLPackVersion`, the `DLPACK_FLAG_BITMASK_*` flags, and the `__dlpack__` /
`__dlpack_device__` Python protocol). When this draft and the constitution
(`docs/architecture/`) conflict, the constitution wins; flag the conflict.

---

## Resolved critique (what changed vs the v0.1 draft)

This final draft addresses every blocker and major finding from the DLPack-correctness review,
plus the sensible minors:

- **(blocker) NULL strides → explicit strides.** The v0.1 draft used `strides=NULL`
  ("contiguous") everywhere. The real `dlpack.h` forbids NULL strides on a v1.2+ versioned
  export ("Before DLPack v1.2, strides can be NULL ... This is NOT allowed in DLPack v1.2 and
  later"). FDX now **mandates a fully-populated strides array** (length `ndim`, `ndim != 0`) on
  every versioned export, reserves NULL for nothing FDX emits, adds the rule to §3.2, and adds
  validator check V11. Every §13 example now shows concrete strides.
- **(blocker) 256-byte data alignment.** Added the alignment contract (`dlpack.h`: "This pointer
  is always aligned to 256 bytes ... The byte_offset field should be used to point to the
  beginning of the data") to §3.3 and §11; intra-buffer starts ride `byte_offset`, never folded
  into `data`. Validator check V12.
- **(major) `DLPACK_FLAG_BITMASK_IS_COPIED`.** A producer that materializes (dequantize or
  live-prefix copy) for a blind consumer now MUST set the standard `IS_COPIED` flag. Added to
  §9.1 and §11.
- **(major) `manager_ctx` ownership / deleter identity.** Dropped the false claim that opacity
  is "contractually required." The sidecar is recovered **only** when the consumer advertised
  FDX *and* the live `deleter` function-pointer identity matches Fuel's own deleter — never on
  the magic word alone, so a forwarded/wrapped third-party capsule is never downcast (§10.2).
- **(major) negative strides — FIRST-CLASS (reversed 2026-06-17).** The v0.1/early-draft rule that
  **banned** negative strides on export is **withdrawn**. FDX strides are `int64` and FDX
  **describes negative strides as first-class** (a flip/reverse view is a real, zero-copy DLPack
  tensor). Validator check **V13** is now the **signed-stride OOB range check**: it computes the
  touched byte-address window `[min, max]` over the signed strides and requires it to lie within
  `[byte_offset, byte_offset + size_bytes)` — exactly the invariant `Layout::flip` already
  maintains (`byte_offset` points at the iteration-first element; `start_offset` stays
  non-negative). Acceptance of negative strides by a *consumer* is a per-kernel FKC-declared
  capability (`reverse_strides`); the **planner** — not FDX — normalizes to non-negative strides
  via a materialized copy (`IS_COPIED`) **only** when the chosen consumer cannot handle them (its
  FKC contract does not declare `reverse_strides`) or for a bare standard-DLPack handoff to a
  non-cooperating external ecosystem. Normalization is never universal and never applied between
  capable internal kernels; an internal zero-copy `Op::Flip` between capable kernels is preserved
  (§3.2.1, V13). See the changelog under "Resolved critique" for the reversal record.
- **(major) capacity-honesty over-stated.** §3.1 now qualifies the no-OOB guarantee: it holds
  *iff* the base buffer is physically allocated for the full capacity shape. mmap-of-live-region
  / partially-committed buffers MUST set `FDX_FLAG_MEANING_REQUIRES_EXT`. Validator check V8
  cross-checks `size_bytes` against capacity.
- **(major) live-prefix "slice" is a copy.** Reworded throughout (§3.1, §9.1, §13.4): a dense
  export of a non-leading symbolic axis is **always a materialized dense copy**, exported with
  `IS_COPIED`. The word "slice" no longer implies a zero-cost view.
- **(major) planner cost slot for contiguize/materialize.** Clarified the FDX↔kernel-contract
  boundary: FDX is pure description and deliberately carries **no** cost; the contiguize /
  materialize / strided alternatives and their declared costs are the **kernel-contract spec's**
  responsibility. §6.5 and §9.3 now name that explicitly so the gap is owned, not silent.
- **(minor) sub-byte padded flag.** §3.4 states FDX never emits a native DLPack sub-byte dtype
  in the base, so `DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED` is unused; `FDXPacking` is the
  sole packing authority.
- **(minor) stream sentinels.** §11 now pins the per-device stream-sentinel table instead of
  "honor DLPack stream semantics".
- **(minor) version axes.** §5.2 / §14 clarify FDX `version` is a major-only schema axis
  *independent* of `DLPackVersion`; feature detection is `(version, flags, struct_bytes)`
  jointly; additive `struct_bytes` growth never changes flag meanings.
- **(minor) hardcoded ordinals.** The dtype / backend / substrate code tables are now an
  **FDX-owned stable table** (§6.0) with an explicit conversion fn and a build-time mapping test,
  not a claimed structural mirror of `#[non_exhaustive]` source enums.
- **(minor) realize-time OOB guard + u64↔usize.** §6.4 / validator V14 add the
  `min <= value <= capacity` realize-time check (the missing OOB guard) and the 32-bit-host
  narrowing rule.
- **(minor, cross-spec) single source of codes.** §6.0 designates FDX as the normative source
  for shared dtype/quant/granularity/substrate codes; the kernel-contract (FKC) spec references
  FDX section numbers and the generated constants, with a cross-spec consistency test.

### Changelog — 2026-06-17 (two additions + the negative-stride reversal)

This dated entry records the three changes folded into this draft on 2026-06-17, after the
gather/affine integration critique sweep:

- **Addition 1 — GATHER / indexed-residency (`FDXIndexedResidency` + `FDXBlockTable`).** A vLLM-style
  paged / blocked KV cache is now a **single FDX tensor**: an honest dense `uint8` block pool base,
  plus a sidecar gather block that re-interprets it via a per-sequence block table. New flag bit 7
  (`FDX_FLAG_HAS_GATHER`), new sub-struct carved from `FDXSidecar.reserved`, new buffer roles
  (`POOL`/`BLOCK_TABLE`/`CONTEXT_LENS`), validators **V18–V21**, capability `DlpackExtGather`, and
  worked example **§13.8**. `MEANING_REQUIRES_EXT` is mandatory (V19). Geometry mirrors *derived*
  operand-shape facts cross-checked against the operand buffers (the op stores only 3 params —
  `FusedOpParams::PagedAttn { softmax_scale, block_size, softcap }`, `fuel-graph/src/registry.rs:241`;
  the geometry rides `KernelRef::PagedAttn`, `fuel-dispatch/src/kernel.rs:314-331`). **Critique
  fixes applied:** V19↔V20 reconciled to one strided "pool byte length" definition
  (`physical_strides[0]*num_blocks*elem_bytes`, padded-pool-safe); §13.8 base-pool capacity
  corrected `Scalar(134217728)`→`Scalar(8388608)` (it was 16× too large, failing its own V7);
  seq-axis lower bound corrected to `min=0` so an empty/finished/evicted sequence (`L=0`, the kernel
  `if ctx_len==0 {continue}` at `byte_kernels.rs:6876`) is legal and not spuriously rejected by V14;
  `physical_strides` unit pinned to *typed elements* with the byte-address composition scaling by
  `elem_bytes` exactly once (§6.9.4 step 3); added a runtime-`usize` narrowing guard
  (`GatherAddressOverflow`) for the flat block-table index and byte address; made the block-table
  column-index guard (`t/block_size < max_blocks_per_seq` via the `ceil(L/block_size)` check)
  explicit and ordered before the dereference; documented the data-determined per-seq path; V21(b)
  identity predicate pinned to buffer-table-slot index-equality (not value comparison, not
  serialized pointer-equality); V18's redundant `ceil` constraint dropped in favor of the exact
  `max_seq_capacity == max_blocks_per_seq * block_size` (capacity is the product, so it is always a
  multiple of `block_size`).
- **Addition 2 — AFFINE symbolic extents (`FDXExtent.kind=Affine`, `FDXAffine`/`FDXAffineTerm`).**
  `FDXExtent` carries a bounded affine combination `c0 + Σ cᵢ·symᵢ` (cap `FDX_AFFINE_MAX_TERMS=4`)
  evaluated through the `SymEnv` at realize, so persistent decode (`k_len = cached_len + new_tokens`)
  is planned once and served every token with no derived-sym recompute. New flag bit 8
  (`FDX_FLAG_HAS_AFFINE_EXTENT`), `FDXExtentKind`/`cap_kind` codes, validators **V16/V17** (V7/V14
  gain affine arms), worked example **§13.7**. Scalar/Range are the degenerate cases (V16 rejects
  degenerate affines). Layout frozen under the array-element-growth discipline (`sym_scope`@28,
  `cap_kind`@32, `affine`@40; `offset_of!`-per-field pins). Resolves §17.3. **Critique fixes
  applied:** the realize-eval pseudocode now widens *both* operands to i128 before the multiply
  (`(coeff as i128).checked_mul(s as i128)` — a u64 sym `> i64::MAX` would mis-cast negative if done
  in i64); the V17/V14 ordering clarified (V17's `>=0` gate rejects a negative result *before*
  narrowing, so V14 never compares a negative against `min`); negative-coefficient compositionality
  spelled out as a V16 cross-check *where the term sym carries its own in-sidecar extent*, with the
  residual (a fully-mechanical per-term bound for syms living only in the `SymEnv`) tracked as
  §17.3c; §5.4 notes `FDXExtent` is now *doubly* size-load-bearing (it restrides `base.extents[]`
  AND shifts `FDXIndexedResidency` fields after `logical_extents`), pinning
  `offset_of!(FDXIndexedResidency, context_lens_buffer/context_len_sym)`. A worked
  gather×affine composition variant was added under §13.8 (per-seq live length as an affine
  `cached_len + new_tokens`), closing the "do they compose" question with a concrete case.
- **Reversal — negative strides are FIRST-CLASS (the major-finding bullet above, withdrawn ban).**
  The early-draft V13 "negative strides forbidden on export" rule — its validator entry, the P9
  "non-negative-stride export" hard-rule claim, the DLTensor/`FDXBufferRef`/`FDXOutputView` strides
  doc-comments, and the §13/§16 prose — is **withdrawn**. FDX strides are `int64` and FDX
  **describes** negative strides as first-class (new §3.2.1). The reversed **V13** is now a
  *signed-stride OOB range check*: it computes the touched `[min,max]` byte window over signed
  strides (per axis: `(dim-1)*stride` as the positive max if `stride>0` else 0, and as the negative
  min if `stride<0` else 0) and requires it to lie within `[0, size_bytes)` — exactly the
  `Layout::flip` invariant (`byte_offset` at the iteration-first element, `start_offset`
  non-negative). Acceptance by a consumer is a per-kernel **FKC `reverse_strides`** layout
  capability (not a backend-wide token); the **planner** normalizes to non-negative via a
  materialized copy (`IS_COPIED`) **only** when the chosen consumer cannot take negatives or for a
  bare external-DLPack handoff — never universally, never between capable internal kernels (internal
  zero-copy `Op::Flip` preserved). The `FDXError` variant `NegativeStride` is replaced by
  `StrideRangeOutOfBounds`.
- **Forward-compat note — FDX operand description is the canonical input to Baracuda's
  `structure_key` (new §4.1).** Added an additive architecture-area subsection stating that an FDX
  operand description is the canonical INPUT to Baracuda's shipped
  `structure_key(op_class, operands, arch) -> StructureKey`; Fuel CALLS that shipped function and
  never reimplements it; FDX already carries every structural fact the key needs (contiguity via
  strides §3.2, broadcast via stride-0, flipped/reverse via stride sign §3.2.1, alignment via tiling
  hints §6.5 / 256-byte rule §3.3, dtype via `DLTensor.dtype` + `FDXDTypeExt` §6.1). **No
  `structure_key` field is added to FDX** (P3 — description, not decision; the key is derived
  externally by its one shipped owner). Marked **[consumer-ahead: deferred Baracuda telemetry
  feed]** — purely additive prose, no struct/validator/flag change.

### Changelog — 2026-06-19 (low-bit sidecar dtype codes + base-passthrough confirmation)

Both changes are **additive** (no struct field added/moved, no flag bit, no `version` bump), made
in response to the Baracuda dtype reconciliation (their 2026-06-19 reply):

- **Addition — `I4` / `U4` / `B1` sidecar logical-dtype codes (0x0102–0x0104, §6.1).** Names for
  packed 4-bit signed/unsigned int (2/byte, GPTQ/AWQ-style) and bitpacked binary (8/byte) in the
  sidecar logical-dtype namespace. They have **no Fuel `DType`** (so `fdx_to_dtype` returns `None`,
  like the `GENERIC_LOW_BIT_*` escapes); dedicated codes keep the structure-key dtype axis clean
  (`I4`/`U4`/`U8` distinct, no escape-flag parsing). C header + Rust constants + the no-Fuel-DType
  test added in lock-step.
- **Confirmation — base `DLTensor.dtype` honors any standard DLPack v1.3 code (§6.1.1).** The
  `FDX_DTYPE_*` namespace is **sidecar-only**; it never constrains the base. So `fp8e5m2`
  (`kDLFloat8_e5m2`), complex (`kDLComplex`; `bits` = total, 64/128), bool (`kDLBool`), and
  *unpacked* 4-bit int (`kDLInt`/`kDLUInt`, `bits = 4`) ride the base honestly with no FDX code and
  no sidecar. A sidecar `FDXDTypeExt` is required only for the *packed* sub-byte cases (base = `uint8`
  stand-in). `COMPLEX64`/`BOOL` (0x0200/0x0201) are documented as reserved placeholders that ride
  the base in practice.
- **Decision — Vulkan `data` = `VkDeviceAddress` (BDA), in answer to the Vulkane review (§3.3.1).**
  DLPack leaves `kDLVulkan` `data` opaque; FDX pins it to the buffer-device-address path (not a
  `VkBuffer` handle), so `void* data` is a real address on every backend, `byte_offset` is pure
  pointer arithmetic, negative-stride flips survive as raw addressing, and sub-256-aligned sliced
  offsets are a non-event (the descriptor-offset-alignment constraint vanishes). The sidecar needs
  **no** Vulkane ABI slot — it stays Fuel-side and decomposes into ordinary bindings + push
  constants. Spec-only (the Vulkan device-pointer extraction in the comm-layer is still
  `[consumer-ahead]`); records the frozen-ABI answer to Vulkane's one open question.

### Changelog — 2026-06-18 (AFFINE_BLOCK quant family + GGML_BLOCK regime-contradiction fix)

This dated entry records two changes, both **additive** (no struct field added or moved, no flag
bit allocated, no `version` bump). FDX remains the **normative owner** of the shared
dtype/quant/granularity/substrate codes (§6.0); the kernel-contract spec (FKC) continues to
reference these codes by symbol.

- **Addition — new quant family `AFFINE_BLOCK` (code 4) for block-grained affine quantization
  (nf4 / QLoRA-style).** A weight in NF4 (low-bit data, e.g. the `F4` sub-byte code) plus a
  **separate per-block scale operand** (an absmax / block scale), block-shaped, is now a
  first-class family distinct from MX and GGML. Added to the §6.2 family table (code 4); given its
  own `FDX_QUANT_AFFINE_BLOCK` constant in Appendix A; an explicit §6.2 field-semantics block and a
  V5 validator arm; a worked example §13.5a; an Appendix B mapping row; and noted under
  `Capability::DlpackExtAffine` (§12). **Field semantics:** `block_ndim >= 1` + `block_shape`
  present; the scale is a **separate graph input named exactly once** (`scale_present == 1`,
  `scale_placement == SEPARATE_BUFFER`, `scale_buffer` a real buffer-table index, **never**
  `FDX_BUFFER_INLINE` and **never** an inline baked scale — single-place rule); `scale_granularity`
  is **not** `PerBlock` (block grain rides `block_shape`). `AFFINE_FLOAT` stays
  `{PerTensor,PerToken,PerChannel}` only; MX keeps the F8E8M0 per-block scale only; `PerBlock`
  granularity stays **MX-only**.
- **Fix — GGML_BLOCK regime-separation internal contradiction resolved.** The §6.2 family
  table/regime callout previously asserted **both** that `GGML_BLOCK` carries baked scales
  (`ggml_dtype` only, no separate scale, no `PerBlock`) **and** (contradictorily) that
  `family=GGML_BLOCK` carries `scale_granularity=PerBlock`/INLINE. Resolved to the single
  consistent rule: **`GGML_BLOCK` carries `ggml_dtype` ONLY** — baked scales interleaved in the
  block struct, **no `scale_granularity`, no `PerBlock`, no separate scale operand, no `ScalePair`**.
  The contradictory "`scale_granularity=PerBlock`/INLINE" clause was removed from the regime callout;
  the family table row, the V5 GGML arm (now rejects `PerBlock`/`SEPARATE_BUFFER` under GGML via
  `QuantRegimeViolation`), and example §13.2 were reconciled. `PerBlock` is now pinned MX-only in
  one place (§6.2, V5, Appendix A), so no family but MX may set it.

Findings deliberately **not** fully resolved here (with rationale) are listed in §17.

### Changelog — 2026-06-18 (internal source of FDX quant families: Fuel `SType`/`Encoding`)

This dated entry is **purely a cross-spec provenance note** — it adds **no** struct field, flag bit,
code value, or validator, and does not bump `version`. It records *where the quant families described
by FDX come from internally*, so the projection direction is unambiguous.

- **The INTERNAL source of the FDX quant families is the Fuel `SType`/`Encoding` type**,
  canonically specified in [`docs/specs/storage-encoding.md`](storage-encoding.md). `SType` (an
  ordered stack of `Encoding` layers attached to a `Storage`) is the **internal source of truth** for
  how a tensor's bytes are encoded; the FDX `FDXQuant` sidecar (§6.2) is its **kernel-boundary
  projection** (the `SType::to_fdx()` direction). FDX remains the normative owner of the shared
  numeric *codes* (§6.0); `storage-encoding.md` owns the internal *type* that those codes project
  from.
- **Regime mapping (projection of the two block regimes):** the FDX `AFFINE_BLOCK` regime
  (separate-scale operand: `scale_present == 1`, `scale_placement == SEPARATE_BUFFER`,
  `scale_buffer` a real buffer-table index — §6.2) is the projection of `Encoding::AffineBlock`
  (the sibling-operand model **B**: the per-block absmax/scale is a separate first-class operand of
  the consuming op, not embedded in the weight's storage). The FDX `GGML_BLOCK` regime
  (`ggml_dtype` only, scales baked INLINE in the block struct — §6.2) is the projection of
  `Encoding::GgmlBlock` (inline-scale, forced by GGUF's interleaved on-disk struct-packing). This
  is consistent with FDX's existing regime separation (`AFFINE_BLOCK` = separate operand,
  `GGML_BLOCK` = inline-baked); the note only records that those two regimes are the boundary
  image of the two internal `Encoding` layers.

---

## 0. Current status / handoff

- This is the **interchange (weight/storage-axis) half** of the kernel boundary; the
  kernel-contract (advertisement-axis) format is a *sibling* spec (FKC). They share the
  dtype/quant/symbolic **codes defined here** (§6.0 — FDX is the normative owner) but are kept
  separate concerns (per 13-interchange: weight ⊥ graph; the node↔weight binding stays
  format-local).
- Nothing here is implemented. The struct shapes are designed to be addable to
  `fuel-core-types` as a new `dlpack` module behind a `dlpack` feature, plus a C header
  `fuel_dlpack_ext.h` co-generated/hand-maintained against the Rust `#[repr(C)]` types.
- Targets the **Intended** architecture (Phase D symbolic extents + sessions + mmap
  persistence) while staying loadable by today's code: the v1 struct carries fields whose
  *consumers* are still ahead (sessions, mmap residency, data-determined syms), but a today
  producer simply leaves them in their "absent/none" state.

---

## 1. Overview & goals

DLPack is the lingua franca for zero-copy tensor exchange across ML runtimes. Its `DLTensor`
is deliberately minimal: a base pointer, device, ndim, a `DLDataType` (code/bits/lanes),
`shape`, `strides`, and a `byte_offset`. That minimalism is its strength for dense, standard
dtypes — and exactly why it cannot, on its own, carry Fuel's load-bearing facts:

1. **Sub-byte / microscaling dtypes** — F4, F6E2M3, F6E3M2, F8E8M0 and general low-bit. Fuel's
   `DType::size_in_bytes()` returns **0** for the sub-byte ones; a buffer sized off that is
   mis-sized. DLPack's `bits/lanes` can *say* 4 bits, but cannot say "MX4, packed two-per-byte,
   block-scaled by an F8E8M0 per 32 elements."
2. **Parametric quantization** — GGUF/GGML block-quant (baked scales), OCP-microscaling (MX)
   block-scaled, and affine-int dynamic quant (`ScaleGranularity`/`ScalePair`) are three
   *different* layouts. DLPack has one buffer; quant is multi-buffer (data + scale sidecar(s)
   ± zero-points).
3. **Per-axis scales / granularity** — `PerTensor | PerToken | PerChannel`, paired
   activation×weight (`ScalePair`), part of the dispatch key.
4. **Symbolic / dynamic extent** — the live-vs-capacity split (Phase D). A KV-cache axis has a
   *capacity* K (for strides/alloc) and a *live* `k_len ≤ K` resolved per token via a `SymEnv`.
   DLPack's `shape[i]` is a single integer; it cannot say "this axis is bounded symbol
   `SymId(7)` ∈ [1, K], stride keyed to K."
5. **Tiling / alignment hints**, **fine residency** (disk-mmap / host / device + substrate
   class precise enough to decide vtable-swap vs copy), **storage class + SessionId**, and
   **multi-output bundles** (one allocation, N sub-views).

### Goals

- **G1 — Honest base, rich sidecar.** The base `DLTensor` is *always* valid standard DLPack on
  its own. All non-standard meaning lives in a separate, optional, versioned sidecar struct.
- **G2 — Universal interop preserved.** When the sidecar is absent (or unrecognized), the base
  is fully interoperable with any DLPack ecosystem, and Fuel produces tensors that are *honest*
  (a quant payload appears as opaque `uint8` bytes, never a mislabeled dtype) **and standards-
  conformant** (explicit strides, 256-byte-aligned data — §3.2, §3.3).
- **G3 — Carry every fact the kernel/planner needs** so the optimizer's pre-priced decisions
  survive the handoff intact (sub-byte/MX dtype, parametric quant, per-axis scales, symbolic
  extent, tiling/alignment, residency/substrate, storage class, bundles).
- **G4 — Versioned, additively extensible, forward/backward compatible.** A newer producer
  talks to an older consumer; a newer consumer reads an older sidecar. No ABI churn on field
  additions.
- **G5 — Stable C-ABI POD with a matching Rust `#[repr(C)]`.** No function pointers, no
  process-absolute pointers in any *serialized* form; deleter function pointers exist only in
  the live cross-runtime managed wrapper (never persisted).
- **G6 — Errors, never panics, never silent coercion.** Malformed/unrepresentable → typed
  `Result` error. Extension-blind consumer + meaning-bearing tensor → *refuse or dequantize*,
  never silent degradation.
- **G7 — Planner decides, not the backend.** The sidecar is *pure description*. It never
  encodes a choice (which kernel, whether to dequantize, where to place) **nor a cost**. Cost-
  bearing alternatives (contiguize/strided/materialize) live in the kernel-contract spec;
  capability negotiation drives the choices through the planner via `BackendProbe`.

### Non-goals

- Not a graph-interchange format (that is the base map; see 13-interchange). FDX is the
  *tensor/storage* axis only.
- Not a replacement for DLPack. FDX strictly *extends* it; the base stays canonical DLPack.
- Not a transport for telemetry, cost, or precision guarantees (those are the kernel-contract
  spec's concern). FDX carries only *what a tensor is*, not *what a kernel costs* and not the
  cost of a layout/precision alternative.
- Does not describe within-kernel concurrency or placement (backend/runtime concerns).

---

## 2. Design principles

- **P1 — Base-DLTensor honesty invariant (§3).** Load-bearing; everything else is built to
  preserve it. "Honest" means both *semantically honest* (no mislabeled dtype) and
  *standards-conformant* (explicit strides, 256-byte-aligned `data`).
- **P2 — Sidecar, not replacement.** State space is `{absent, v1, v2, …}` via an explicit
  version field, *not* a 2-state null/non-null. Absence = "this is plain DLPack."
- **P3 — Description, never decision, never cost.** Mirrors the constitution's governing
  principle (backends advertise, the planner decides). FDX says *what is*; it never says *what
  to do* or *what it costs*. No "dequantize-on-read" flag, no "preferred kernel," no cost slot.
- **P4 — Capacity for layout, symbol for liveness.** Strides and allocation key off the
  *capacity* bound (`dims()` / `Extent::bound()`); the *live* length is a `SymId` resolved per
  call via a `SymEnv` supplied alongside the data. One description plans once, serves every
  token/session/replay.
- **P5 — Bit-width + packing for sub-byte, never byte size.** Sub-byte dtypes carry an explicit
  bit-width and packing convention; `size_in_bytes()==0` is treated as a *flag*, not a size.
- **P6 — Quant is parametric and multi-buffer.** One open descriptor expresses GGML-style,
  MX-style, and affine-int by *parameters* (block shape, scale ref + dtype + placement,
  zero-point presence/dtype, packing order), not by a hardcoded enum-per-format.
- **P7 — No function/absolute pointers in serialized form.** Buffer references are
  *capability-relative handles* (an index into a side buffer table), not raw pointers, in any
  form that can be persisted. Live cross-runtime exchange uses DLPack's managed/deleter
  contract, whose pointers are never written to disk.
- **P8 — Additive versioning.** New fields go into reserved space or a higher version; old
  readers ignore unknown trailing fields (size-prefixed). `#[non_exhaustive]`-spirited enums.
- **P9 — Match external convention.** Reuse DLPack names/semantics (`DLDataType`, `DLDevice`,
  byte_offset, deleter, stream-exchange, the `DLPACK_FLAG_BITMASK_*` flags) where they exist;
  only extend where Fuel needs more. **Obey DLPack's hard rules** (explicit strides ≥ v1.2,
  256-byte data alignment) — they are not Fuel's to relax. (Negative strides are *permitted* by
  DLPack — `strides` is `int64` — and FDX treats them as first-class, §3.2.1.)
- **P10 — Build-time validatable.** Every consistency check is a `Result`-returning validation
  runnable at the boundary. No `try_*` siblings.

---

## 3. The base-DLTensor honesty invariant (load-bearing)

> **The base `DLTensor` must always be honestly interpretable as standard, *conformant* DLPack
> on its own.**

Concretely:

- **Never put a non-standard dtype code in the base `DLTensor.dtype`.** A quantized / sub-byte /
  microscaling payload is represented in the base as a **standard byte buffer**: `DLDataType {
  code: kDLUInt, bits: 8, lanes: 1 }` (opaque bytes), with the base `shape`/`strides`
  describing the *physical byte buffer* (the packed bytes), not the logical elements.
- The **true semantics live only in the sidecar.** A consumer that ignores the sidecar sees an
  honest `uint8` tensor of the correct physical byte size — never a mislabeled `float16` over
  4-bit data, never a buffer mis-sized by a `size_in_bytes()==0` dtype.
- For **standard, fully-representable dtypes** (F16/F32/I32/…), the base `DLTensor` is *fully
  faithful*: `dtype`, `shape`, `strides` are all exactly correct, the sidecar (if present) adds
  only orthogonal facts (symbolic extent, residency, storage class, tiling), and a sidecar-blind
  consumer loses *nothing semantic* — it just sees the capacity-shaped dense tensor (see §3.1).
- The honesty invariant is **one-directional about safety**: ignoring the sidecar can lose
  *meaning* (you get opaque bytes), but can never produce *wrong* numbers from a mislabeled
  dtype, a mis-sized buffer, NULL strides a strict consumer rejects, or a misaligned pointer.

### 3.1 The capacity-honesty corollary (symbolic extents)

For an axis with a symbolic live extent, the base `DLTensor.shape[i]` reports the **capacity**
(`Extent::bound()` / `Shape::dims()[i]` — the `Range`'s `max`), and `strides[i]` is keyed to
that capacity. This is *honest*, not a lie: the buffer truly has `capacity` slots; the live
count is a *different fact* carried in the sidecar.

- **No-OOB guarantee (qualified).** A sidecar-blind consumer that reads the whole
  capacity-shaped tensor reads **real, allocated memory (no OOB) iff the base buffer is
  physically allocated for the full capacity shape** — i.e. `buffers[0].size_bytes` covers
  `capacity * stride` on every axis (a dense, fully-committed allocation with no holes). For a
  KV cache backed by a dense `[n_heads, K, head_dim]` allocation, that holds: reading the full
  `K`-capacity buffer touches only allocated memory; the not-yet-written tail is whatever the
  allocation initialized it to.
- **Where it does NOT hold.** A buffer that is *not* physically committed to full capacity —
  an mmap that maps only the live region (`FDXResidency.is_mmap_view`), a partially-committed /
  growable allocation, or a multi-axis symbolic case where only a sub-region is backed —
  **MUST** set `FDX_FLAG_MEANING_REQUIRES_EXT`. Reading the capacity tail of such a buffer can
  fault; the flag forces the refuse-or-materialize path (§9.1), exactly as for quant. Validator
  check V8 cross-checks `buffers[0].size_bytes` against the capacity-shaped extent and rejects a
  symbolic tensor that claims full-capacity safety without the backing to prove it.
- **Producer policy (§9):** if reading the capacity tail instead of the live prefix would be
  *semantically wrong* for a given tensor (not merely wasteful), the producer MUST treat it
  like any other meaning-bearing-via-sidecar tensor: refuse the bare-DLPack export to an
  extension-blind consumer, or **materialize a dense copy of the live prefix**. A KV cache
  handed to PyTorch via bare `__dlpack__` is exported as a **materialized dense copy** of the
  resolved-length live region (a standard dense tensor with `IS_COPIED` set), never the raw
  capacity buffer. See §3.1.1 on why this is a copy, not a view.

#### 3.1.1 Live-prefix export is a COPY, not a view (when the symbolic axis is not leading)

For a KV cache `[n_heads, K_capacity, head_dim]`, the live region on the **middle** axis is
**not contiguous**: axis-0 stride is `K_capacity * head_dim`, so the live `[n_heads, k_len,
head_dim]` rows are interleaved with not-yet-live slots. A dense `[n_heads, k_len, head_dim]`
tensor therefore **cannot be a zero-copy view** of the capacity buffer — it requires a copy
(gather of the live rows into a fresh dense allocation).

- A dense export of a tensor whose symbolic axis is **not the leading (slowest-varying) axis**
  is **always a materialized copy** and MUST carry `DLPACK_FLAG_BITMASK_IS_COPIED` (§9.1, §11).
- A producer MAY instead export a **strided view** (same `data`, strides keyed to capacity,
  `shape[symbolic_axis] = k_len`). That is a legitimate zero-copy export *only* when the live
  region is exactly the leading `k_len` along the slowest axis and remains contiguous-prefix; in
  that case it is a view (no `IS_COPIED`) but the word "dense" does not apply — the consumer
  must honor the strides. FDX never labels a strided view as "dense".
- The spec uses **"materialized dense copy"** wherever a dense layout is promised, and
  **"strided view"** wherever zero-copy is promised. The two are never conflated.

### 3.2 Strides are always explicit (DLPack v1.2+ rule — load-bearing)

`dlpack.h` (v1.3) documents: *"Before DLPack v1.2, strides can be NULL to indicate contiguous
data. This is NOT allowed in DLPack v1.2 and later."* Because FDX targets the versioned struct
(v1.0+) and validates against v1.3:

- **A producer MUST write a fully-populated `strides` array of length `ndim`** for every
  versioned export with `ndim != 0`. NULL strides are **never** emitted by FDX. (`ndim == 0`,
  the scalar case, has no strides — the array is absent there per the base spec.)
- This applies to **both** boundaries: the internal kernel ABI (boundary a) and the external
  `__dlpack__` export (boundary b). An extension-blind, strict v1.2+ consumer must be able to
  accept any FDX export without rejecting it for NULL strides.
- For a contiguous tensor the strides are the row-major (C-contiguous) strides computed from
  `shape`; the producer writes them explicitly rather than relying on the legacy NULL
  convention. For the honesty-`uint8` base of a packed buffer, the strides describe the
  *physical byte buffer* (e.g. `[1]` for a 1-D byte buffer).
- Validator check **V11** rejects any versioned sidecar/export whose base `DLTensor.strides`
  is NULL while `ndim != 0`.

> Legacy unversioned `DLManagedTensor` (where NULL strides are still legal pre-v1.2) is **not an
> FDX export target**: FDX never produces it. A consumer importing a legacy capsule into Fuel
> normalizes NULL strides to explicit strides at the import boundary before any FDX reasoning.

### 3.2.1 Negative strides are first-class (reversed 2026-06-17 — load-bearing)

`DLPack`'s `strides` are `int64_t`; the spec does **not** forbid negative values, and a reversed /
flipped view (e.g. the result of an `Op::Flip`) is a perfectly valid zero-copy DLPack tensor.
**FDX describes negative strides as first-class** — the early-draft "non-negative-only on export"
ban is **withdrawn** (see the changelog under "Resolved critique"). What replaces it is a
*soundness* check, not a *prohibition*:

- **The OOB guarantee survives via a signed touched-range check (V13).** A buffer of `size_bytes`,
  read with strides `strides[i]` and capacity extents `shape[i]` from a logical start
  `byte_offset`, touches the byte window `[byte_offset + Σ min_i, byte_offset + Σ max_i]` where,
  **per axis**, the maximum positive contribution is `(shape[i] - 1) * strides[i]` if
  `strides[i] > 0` else `0`, and the minimum (most-negative) contribution is
  `(shape[i] - 1) * strides[i]` if `strides[i] < 0` else `0`. V13 requires that whole window to lie
  within `[0, size_bytes)` (after adding `byte_offset`, and accounting for the element byte width).
  This is **exactly the invariant `Layout::flip` already maintains**: `byte_offset` is moved to
  point at the *iteration-first* element of the flipped view and `start_offset` always stays
  non-negative, so the reachable window never escapes the allocation. The §3.1 no-OOB argument is
  thus preserved for negative strides without re-derivation — it is the same `[min, max] ⊆
  [0, size_bytes)` proof, now computed over signed contributions.
- **Acceptance is a per-kernel FKC capability (`reverse_strides`), not an FDX property.** Whether a
  given *consumer* can walk a negative-strided operand is declared in its **kernel-contract (FKC)**
  layout capability (`reverse_strides`); FDX merely *describes* the strides. FDX adds no flag for
  this (P3 — description, not decision).
- **Normalization is the PLANNER's choice, gated on the consumer — never universal.** When the
  chosen consumer's FKC contract does **not** declare `reverse_strides`, **or** when exporting a
  bare standard-DLPack tensor (boundary b) to a non-cooperating external ecosystem, the **planner**
  inserts an explicit materialized copy to non-negative strides and sets
  `DLPACK_FLAG_BITMASK_IS_COPIED` (§9.1). It does **not** normalize between capable internal
  kernels: an internal zero-copy `Op::Flip` handed to a `reverse_strides`-capable kernel stays a
  view. The decision lives in the planner from the capability advertisement (§12), never as an
  FDX-boundary trial.

> **Contrast with the withdrawn rule.** The old V13 *rejected* any negative stride on export and
> stated the OOB argument "for the non-negative case" only. The reversed V13 *accepts* negative
> strides and proves no-OOB over the signed touched range; the only thing that can still force a
> non-negative copy is a *consumer* that cannot take them — a planner decision, priced by FKC, not
> a blanket FDX ban.

### 3.3 Data-pointer alignment (256-byte rule — load-bearing)

`dlpack.h`: *"This pointer is always aligned to 256 bytes as in CUDA. The byte_offset field
should be used to point to the beginning of the data."* FDX obeys this on export:

- **The exported `DLTensor.data` pointer (and every exported `FDXBufferRef.data`) MUST be
  256-byte aligned.** Any intra-buffer start offset (a KV live-prefix start, a quant sub-buffer
  start, a bundle sub-view start) MUST be carried in `byte_offset` (or `FDXBufferRef.byte_offset`
  / `FDXOutputView.byte_offset`), **never folded into `data`**.
- A sub-view whose natural start is not 256-aligned is expressed as `data = aligned_base`,
  `byte_offset = start - aligned_base`. An extension-blind CUDA consumer that assumes
  256-aligned `data` then handles it correctly.
- Allocators feeding FDX exports SHOULD allocate to at least 256-byte alignment (it is also the
  floor for `FDXTiling.alignment_bytes` guidance, §6.5). When an internal buffer cannot meet
  256-byte alignment and must cross boundary (b), the producer materializes an aligned copy
  (sets `IS_COPIED`) rather than exporting a misaligned pointer.
- Validator check **V12**: on a boundary-(b) export, `(data as usize) % 256 == 0` for the base
  and every exported buffer; each sub-view start is expressed via `byte_offset`, and
  `byte_offset` for a buffer's logical start does not exceed `size_bytes`.

> Internal boundary (a) launches MAY relax 256-alignment to the backend's
> `required_alignment` (`FDXTiling.alignment_bytes`), since Fuel owns both ends and no external
> CUDA-256 assumption is in play; the 256-byte rule is enforced strictly only on boundary (b).

#### 3.3.1 What `data` holds per device — and the Vulkan `VkDeviceAddress` decision

`DLTensor.data` / `FDXBufferRef.data` is `void*`, but DLPack pins its *meaning* only loosely: it
is a host pointer on `kDLCPU`, a `CUdeviceptr` on `kDLCUDA`, and on `kDLVulkan` (= 7) the upstream
header says it "may be opaque on some device types" and gives **no** Vulkan-specific definition.
FDX must therefore pin it, because Vulkan offers two incompatible bindings.

**Decision (2026-06-19, in answer to the Vulkane review):** on a `kDLVulkan` device, FDX `data`
is a **`VkDeviceAddress`** — the buffer-device-address (BDA) path — **not** a `VkBuffer` handle.
A Vulkan FDX tensor's `data` is the device address of a buffer created with
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS`, and the kernel addresses it via Slang `buffer_reference`.
Rationale:

- **Honesty + uniformity.** `void* data = VkDeviceAddress` is a *real address*, so `data` means
  the same kind of thing on every backend (host pointer / CUdeviceptr / device address), and
  `byte_offset` is pure pointer arithmetic on all of them. A `VkBuffer` *handle* in `data` would be
  a handle-masquerading-as-pointer (the treatment DLPack reserves for Metal's `id<MTLBuffer>`) and
  would force every buffer-table entry to additionally carry a Vulkane buffer handle.
- **Negative strides / zero-copy flips are most direct under BDA.** Fuel's signed-stride,
  first-class-flip design (§3.2.1) wants the kernel doing raw signed-offset addressing — exactly
  what `buffer_reference` over a device address gives. This is the load-bearing reason to prefer
  BDA: it is the binding where a reversed view survives as pure pointer math.
- **The descriptor-offset-alignment constraint disappears.** A *descriptor*-bound storage buffer
  requires `VkDescriptorBufferInfo.offset` be a multiple of the device
  `min{Storage,Uniform}BufferOffsetAlignment` (whose spec-guaranteed maximum is 256 — so the §3.3
  256-byte base floor already dominates it). Under BDA, `byte_offset` is not a descriptor offset at
  all, so a **sub-256-aligned sliced / bundle-slot `byte_offset` is a non-event** — it is just
  added to the address. (On the descriptor path it would require binding at the aligned floor and
  passing the residual sub-offset as a push constant; BDA removes that case.) The §3.3 256-byte
  rule still governs the **base** buffer pointer on a boundary-(b) export; sub-view `byte_offset`s
  on the internal Vulkan path are unconstrained.

Vulkane supports this today (`Buffer::device_address()`), and confirms every preservation
guarantee holds by construction: on the compute path a binding is `(VkBuffer, offset, range)` with
no stride field, so signed strides never reach a binding and cannot be coerced; `byte_offset →`
descriptor `offset` (or pure arithmetic under BDA) verbatim; the buffer table is natively plural
(a descriptor set holds N bindings). **The FDX sidecar does not need a Vulkane-side ABI slot:** it
stays Fuel-side in `fuel-vulkan-backend`, which reads it and decomposes it into ordinary Vulkane
bindings (data + scale + block-table as N buffers) plus push-constant metadata; nothing in FDX
requires the sidecar to transit the FFI as a host pointer (consistent with §10 — the sidecar is
producer-side and the backend/kernel reads it). If a passthrough tag is ever wanted (tooling /
defrag bookkeeping), Vulkane's per-allocation `user_data: u64` is a zero-interpretation carrier.

> **Metal (`kDLMetal`)** follows the same per-substrate pinning: `data` carries the buffer
> identity the Metal compute path binds (DLPack's `id<MTLBuffer>` treatment). The general rule is
> that FDX pins `data`'s meaning **per substrate** (§6.6), and the Vulkan substrate is pinned to
> BDA here.

### 3.4 Sub-byte dtypes never use the native DLPack sub-byte path

DLPack v1.x has a native sub-byte mechanism (`bits < 8` plus
`DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED`, "whether a sub-byte type is packed or padded;
default for sub-byte types ex fp4/fp6 is packed"). **FDX deliberately does not use it.** A
sub-byte / microscaling payload always appears in the base as the `uint8` honesty stand-in
(§3), with packing described by `FDXDTypeExt` + `FDXPacking` in the sidecar. Therefore:

- **FDX never emits a native DLPack sub-byte `dtype` in the base**, and never sets
  `DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED`.
- `FDXPacking` + `sub_byte_bit_order` (§6.1) is the **sole packing authority** within FDX. A
  reader never has to reconcile a native-DLPack padded/packed flag against the sidecar, because
  the base carries no sub-byte dtype to be padded.

---

## 4. Architecture: two structs, one optional link

```
        ┌──────────────────────────────┐         ┌──────────────────────────────┐
        │  DLTensor  (STANDARD)        │         │  FDXSidecar  (FUEL EXTENSION) │
        │  - data: void* (256-aligned) │  ◄────  │  - version, flags            │
        │  - device: DLDevice          │  link   │  - dtype_ext (sub-byte/MX)   │
        │  - ndim, dtype(uint8!)       │         │  - quant (parametric)        │
        │  - shape[], strides[] (≠NULL)│         │  - extents[] (live-vs-cap)   │
        │  - byte_offset (intra-buf)   │         │  - residency, storage class  │
        └──────────────────────────────┘         │  - tiling/alignment hints    │
                                                  │  - buffer_table (sidecars)   │
                                                  │  - bundle views (multi-out)  │
                                                  └──────────────────────────────┘
```

- The link is **directional and optional**: a `DLTensor` may have *no* sidecar (it is then
  100% standard). The sidecar always refers back to exactly one base `DLTensor` whose `data`
  pointer is buffer index 0 of the sidecar's buffer table (§7.4).
- **Two boundaries** carry the link differently (§10):
  - **(a) Fuel kernel ABI** — the sidecar is an explicit nullable parameter *next to* the
    `DLTensor` (`*const FDXSidecar`, `null` = absent). Cleanest, no smuggling.
  - **(b) External `__dlpack__`** — the capsule signature is fixed (no extra-pointer slot), so
    the sidecar rides `DLManagedTensorVersioned.manager_ctx` *if and only if* the consumer
    advertised understanding **and** Fuel's own deleter is in force (§10.2); else it is **not
    carried** (only the standard part crosses, and producer policy §9 governs whether that is
    even allowed).

### 4.1 An FDX operand description is the canonical input to Baracuda's `structure_key` (forward-compat)

An FDX operand description is the **canonical input** to Baracuda's shipped
`structure_key(op_class, operands, arch) -> StructureKey` function: the per-operand structural
facts that key derives are exactly the facts an `FDXSidecar` + its base `DLTensor` already carry.
Fuel **calls Baracuda's shipped `structure_key`** with the FDX operand descriptions as input; it
**never reimplements** that derivation. Per the description-vs-decision principle (P3), the key is a
**derived, external** fact — FDX *describes* the structure, Baracuda *derives* the key from that
description — so it is computed downstream from FDX, not stored in it.

FDX already carries every structural fact `structure_key` needs:

- **contiguity** — via the base `DLTensor.strides` (a contiguous operand has row-major strides; a
  strided one does not, §3.2);
- **broadcast** — via a **stride-0** axis on `DLTensor.strides`;
- **flipped / reverse iteration** — via the **sign** of a stride (negative strides are first-class,
  §3.2.1);
- **alignment** — via the tiling hints (`FDXTiling.alignment_bytes`, §6.5) and the 256-byte data
  rule (§3.3);
- **dtype** — via the base `DLTensor.dtype` plus, when present, `FDXDTypeExt` (§6.1).

> **Do NOT add a `structure_key` field to FDX (P3 — description, not decision).** The key is a
> *derivation over* the FDX description, owned by Baracuda's shipped function, not a fact the
> producer states. Adding a `structure_key` slot to the sidecar would (a) duplicate a fact already
> fully implied by the operand description, (b) bake a *downstream consumer's derived value* into a
> pure-description struct — exactly the decision-in-description anti-pattern P3 forbids — and (c)
> couple FDX's ABI to Baracuda's keying scheme. FDX stays the description; the key is derived
> externally by the one shipped owner of that derivation.

**[consumer-ahead: deferred Baracuda telemetry feed]** — the call into Baracuda's `structure_key`
(and the telemetry/keying feed it backs) is a downstream consumer of the FDX description, not a part
of this spec's v1 surface. FDX v1 simply carries the structural facts above so that, when Fuel wires
the Baracuda call, the operand description is already a complete input — no FDX struct change is
required to feed it.

---

## 5. Concrete struct definitions

All multi-byte fields are **little-endian** in serialized form. All structs are
`#[repr(C)]` POD with explicit padding. Reserved fields are zero on write, ignored on read.
Every variable-length array is referenced by `(count, *const T)` for the live in-memory form;
the **serialized** form replaces pointers with byte offsets relative to the sidecar blob start
(P7).

### 5.1 Standard DLPack (reproduced for reference — NOT redefined by FDX)

```c
/* From dlpack.h v1.3 — FDX consumes these unchanged. */
typedef enum { kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLVulkan = 7,
               kDLMetal = 8, /* ... */ } DLDeviceType;
typedef struct { DLDeviceType device_type; int32_t device_id; } DLDevice;
typedef enum { kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBfloat = 4,
               kDLComplex = 5, kDLBool = 6 } DLDataTypeCode;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;

typedef struct {
  void*      data;        /* 256-byte aligned on export (§3.3); logical start via byte_offset */
  DLDevice   device;
  int32_t    ndim;
  DLDataType dtype;
  int64_t*   shape;       /* length ndim — CAPACITY bounds for symbolic axes */
  int64_t*   strides;     /* length ndim, NEVER NULL on a versioned export (§3.2); keyed to
                             capacity; int64 — negatives are FIRST-CLASS (§3.2.1 / V13), the
                             signed touched-range must lie within size_bytes (V13) */
  uint64_t   byte_offset; /* intra-buffer start; carries any non-256-aligned offset (§3.3) */
} DLTensor;

typedef struct { uint32_t major; uint32_t minor; } DLPackVersion;

/* DLPack standard flags (dlpack.h) — FDX uses the standard flags directly. */
#define DLPACK_FLAG_BITMASK_READ_ONLY              (1UL << 0)
#define DLPACK_FLAG_BITMASK_IS_COPIED              (1UL << 1)
#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2)  /* FDX never sets (§3.4) */

typedef struct {                       /* the cross-runtime managed form */
  DLPackVersion version;               /* DLPack ABI version — INDEPENDENT of FDX version */
  void         *manager_ctx;           /* FDX rides here at boundary (b), deleter-guarded */
  void        (*deleter)(struct DLManagedTensorVersioned *self);
  uint64_t      flags;                 /* DLPACK_FLAG_BITMASK_* */
  DLTensor      dl_tensor;
} DLManagedTensorVersioned;
```

### 5.2 FDX magic, version, and flags

```c
#define FDX_MAGIC      0x46445800u   /* "FDX\0" */
#define FDX_VERSION_1  1u            /* FDX SCHEMA major; INDEPENDENT of DLPackVersion */

/* FDXSidecar.flags — bitmask, additive. These are FDX-internal flags; the standard DLPack
   ownership/freshness flags (READ_ONLY, IS_COPIED) live on DLManagedTensorVersioned.flags. */
#define FDX_FLAG_HAS_DTYPE_EXT     (1u << 0)  /* dtype_ext is meaningful   */
#define FDX_FLAG_HAS_QUANT         (1u << 1)  /* quant block is meaningful */
#define FDX_FLAG_HAS_SYMBOLIC      (1u << 2)  /* >=1 axis is symbolic      */
#define FDX_FLAG_HAS_TILING        (1u << 3)  /* tiling block present      */
#define FDX_FLAG_IS_BUNDLE         (1u << 4)  /* multi-output bundle       */
#define FDX_FLAG_MEANING_REQUIRES_EXT (1u << 5)
        /* base bytes alone are NOT a usable tensor (quant/sub-byte/      */
        /* live<cap-where-tail-is-wrong-or-unbacked). Drives refuse-or-dequant. */
#define FDX_FLAG_READ_ONLY         (1u << 6)  /* mirrors DLPACK_FLAG_BITMASK_READ_ONLY  */
#define FDX_FLAG_HAS_GATHER        (1u << 7)  /* gather block (FDXIndexedResidency) is  */
        /* meaningful: the base bytes are a physical BLOCK POOL re-interpreted via a    */
        /* block table (§6.9). Implies FDX_FLAG_MEANING_REQUIRES_EXT (V18).            */
#define FDX_FLAG_HAS_AFFINE_EXTENT (1u << 8)  /* >=1 extent is kind=AFFINE (§6.4).      */
        /* Advisory / fast-reject; the per-extent `kind` byte is authoritative.        */
/* bits 9..63 reserved (0). */
```

> **Authoritative flag-bit allocation table (single owner — this is the only place a bit is
> assigned).** Every `FDX_FLAG_*` bit is allocated here, exactly once. A future addition picks the
> next free bit *from this table* — never "the next free bit" judged against a private copy of the
> struct — and a build-time test asserts no two `FDX_FLAG_*` constants share a bit (mechanically
> impossible to recollide). This table is the integration record for the two 2026-06-17 additions
> (gather + affine), which were drafted independently and **both originally claimed bit 7**; gather
> kept bit 7 and affine moved to bit 8 (see the changelog under "Resolved critique").

| bit | constant | meaning |
|-----|----------|---------|
| 0 | `FDX_FLAG_HAS_DTYPE_EXT` | `dtype_ext` meaningful |
| 1 | `FDX_FLAG_HAS_QUANT` | `quant` meaningful |
| 2 | `FDX_FLAG_HAS_SYMBOLIC` | ≥1 axis symbolic |
| 3 | `FDX_FLAG_HAS_TILING` | `tiling` present |
| 4 | `FDX_FLAG_IS_BUNDLE` | multi-output bundle |
| 5 | `FDX_FLAG_MEANING_REQUIRES_EXT` | base bytes not a usable tensor |
| 6 | `FDX_FLAG_READ_ONLY` | mirrors `DLPACK_FLAG_BITMASK_READ_ONLY` |
| 7 | `FDX_FLAG_HAS_GATHER` | `gather` (`FDXIndexedResidency`) meaningful (§6.9) |
| 8 | `FDX_FLAG_HAS_AFFINE_EXTENT` | ≥1 extent is `kind=AFFINE` (§6.4) |
| 9..63 | (reserved, 0) | next addition takes bit 9 from THIS table |

`FDX_FLAG_MEANING_REQUIRES_EXT` is the single most consumer-relevant flag: it is the producer's
explicit statement that handing the bare `DLTensor` to a sidecar-blind consumer would lose
meaning (quant/sub-byte) **or be unsafe** (capacity tail not backed, §3.1), and therefore the
producer policy in §9 (refuse-or-dequantize/materialize) applies.

> **Two independent version axes.** `FDXSidecar.version` (FDX_VERSION_*) selects the **FDX
> schema** and is *independent* of `DLManagedTensorVersioned.version` (`DLPackVersion
> {major,minor}`, the DLPack ABI version). A capsule can be DLPack v1.3 carrying an FDX v1
> sidecar; the two evolve separately. FDX uses a **major-only** schema version (no minor field):
> additive growth is detected via `struct_bytes` (§5.3) and `flags`, not a minor bump. Feature
> detection is therefore the **joint** `(version, flags, struct_bytes)` — and additive
> `struct_bytes` growth **never changes the meaning of an existing flag** (§14).

### 5.3 The top-level sidecar — Rust

```rust
/// Optional, versioned sidecar communicated ALONGSIDE a standard DLTensor.
/// `#[repr(C)]` POD. Absence (a null `*const FDXSidecar`) means "plain DLPack".
#[repr(C)]
pub struct FDXSidecar {
    /// FDX_MAGIC. Lets a consumer cheaply reject a misrouted pointer.
    pub magic: u32,
    /// FDX_VERSION_*. The {absent, v1, v2, …} discriminator (P2). Major-only;
    /// INDEPENDENT of DLPackVersion (§5.2).
    pub version: u32,
    /// Total byte size of this struct as written by the producer. Enables
    /// size-prefixed forward-compat: an older reader trusts fields up to
    /// `min(sizeof(known), struct_bytes)` and ignores the trailing tail (P8).
    /// Feature detection is (version, flags, struct_bytes) jointly (§5.2, §14).
    pub struct_bytes: u32,
    /// FDX_FLAG_* bitmask.
    pub flags: u32,

    /// Sub-byte / microscaling dtype descriptor (§6.1). Valid iff
    /// FDX_FLAG_HAS_DTYPE_EXT. When unset, the base DLTensor.dtype is the
    /// faithful dtype and this is `FDXDTypeExt::NONE`.
    pub dtype_ext: FDXDTypeExt,

    /// Parametric quantization layout (§6.2). Valid iff FDX_FLAG_HAS_QUANT.
    pub quant: FDXQuant,

    /// Per-axis symbolic extents (§6.4). `extents_count == 0` ⇒ all axes are
    /// concrete and equal to base `shape` (no symbolic axis). Otherwise
    /// `extents_count == base.ndim` and entry `i` annotates `shape[i]`.
    pub extents_count: u32,
    pub _pad0: u32,
    pub extents: *const FDXExtent,   // serialized: byte offset

    /// Tiling / alignment hints (§6.5). Valid iff FDX_FLAG_HAS_TILING.
    pub tiling: FDXTiling,

    /// Residency / substrate finer than DLDevice (§6.6).
    pub residency: FDXResidency,

    /// Storage class + session identity (§6.7).
    pub storage: FDXStorage,

    /// Side table of all buffers this logical tensor owns/references
    /// (data + scales + zero-points + bundle slots). Index 0 is ALWAYS the
    /// base DLTensor's data buffer (§7.4).
    pub buffers_count: u32,
    pub _pad1: u32,
    pub buffers: *const FDXBufferRef, // serialized: byte offset

    /// Multi-output bundle sub-views (§6.8). Valid iff FDX_FLAG_IS_BUNDLE.
    pub views_count: u32,
    pub _pad2: u32,
    pub views: *const FDXOutputView,  // serialized: byte offset

    /// Gather / indexed-residency descriptor — paged/blocked pool (§6.9).
    /// Valid iff FDX_FLAG_HAS_GATHER. When unset, this is
    /// `FDXIndexedResidency::NONE` (all-zero, kind = FDX_GATHER_NONE).
    /// Embedded by value (frozen-size sub-struct, like FDXQuant); carved from
    /// the former `reserved[8]` tail. The §5.4 size-assertion test pins
    /// sizeof(FDXSidecar).
    pub gather: FDXIndexedResidency,

    /// Reserved for additive growth without bumping `version`. Zero on write.
    /// Shrunk from `[u64; 8]` to keep FDXSidecar's size class stable across
    /// the gather addition; the §5.4 size-assertion test pins the new total.
    pub reserved: [u64; 2],
}
```

### 5.3 (cont.) The top-level sidecar — C

```c
typedef struct FDXSidecar {
  uint32_t       magic;          /* FDX_MAGIC */
  uint32_t       version;        /* FDX_VERSION_* (major-only; != DLPackVersion) */
  uint32_t       struct_bytes;   /* size-prefix for forward-compat */
  uint32_t       flags;          /* FDX_FLAG_* */

  FDXDTypeExt    dtype_ext;      /* §6.1 */
  FDXQuant       quant;          /* §6.2 */

  uint32_t       extents_count;  /* 0 or base.ndim */
  uint32_t       _pad0;
  const FDXExtent* extents;      /* serialized: int64 byte offset */

  FDXTiling      tiling;         /* §6.5 */
  FDXResidency   residency;      /* §6.6 */
  FDXStorage     storage;        /* §6.7 */

  uint32_t       buffers_count;
  uint32_t       _pad1;
  const FDXBufferRef* buffers;   /* §7.4 */

  uint32_t       views_count;
  uint32_t       _pad2;
  const FDXOutputView* views;    /* §6.8 */

  FDXIndexedResidency gather;    /* §6.9; valid iff FDX_FLAG_HAS_GATHER */

  uint64_t       reserved[2];    /* shrunk from [8] for `gather`; size pinned §5.4 */
} FDXSidecar;
```

### 5.4 Size discipline for embedded sub-structs (P8)

Every embedded sub-struct (`FDXDTypeExt`, `FDXQuant`, `FDXTiling`, `FDXResidency`, `FDXStorage`,
and now `FDXIndexedResidency`) is **frozen-size** and grows only via its own `reserved[]`. The
`gather` field was carved out of the former `FDXSidecar.reserved[8]` (P8: additive growth into
reserved space, guarded by `struct_bytes`), so `sizeof(FDXSidecar)` either stays in its existing
size class or grows by exactly one documented class — a layout decision, not a semantic one. The
implementation MUST pin every sub-struct size and the top-level total with build-time
assertions, e.g. `assert_eq!(size_of::<FDXSidecar>(), …)`, `size_of::<FDXIndexedResidency>()`,
`size_of::<FDXBlockTable>()`, `size_of::<FDXAffineTerm>() == 16`, `size_of::<FDXAffine>() == 80`,
and — because `FDXExtent` is an **array element** whose stride matters (§6.4) —
`size_of::<FDXExtent>()` plus an `offset_of!`-per-field assertion (so a future field-order edit
breaks the build, not the ABI; see §6.4 layout note). A `struct_bytes` round-trip test
complements the static size pins.

> **`FDXExtent` is now DOUBLY load-bearing on size.** Since the gather addition, `FDXExtent` is not
> only the element of the variable-length `FDXSidecar.extents[]` (where `sizeof(FDXExtent)` is the
> array stride, §6.4) but **also an inline array member** of `FDXIndexedResidency`
> (`logical_extents: [FDXExtent; 6]`, §6.9.2). Therefore any change to `sizeof(FDXExtent)` is
> doubly load-bearing: it **restrides `base.extents[]`** AND **shifts every `FDXIndexedResidency`
> field after `logical_extents`** (`context_lens_buffer`, `context_len_sym`, `context_len_scope`,
> `reserved`). The implementation MUST therefore also pin `offset_of!(FDXIndexedResidency,
> context_lens_buffer)` and `offset_of!(FDXIndexedResidency, context_len_sym)` (and the trailing
> fields) so a future `FDXExtent` growth breaks the build *there* too, not silently at the ABI.

---

## 6. Field-by-field semantics

### 6.0 FDX is the normative owner of shared codes (single source of truth)

All numeric codes for dtype, quant family, scale granularity/placement/order, packing,
substrate, residency tier, backend id, and storage class are **owned by FDX** as a standalone
stable table (the tables in §6.1–§6.8 and Appendix A). They are **not** a structural mirror of
the as-built `#[non_exhaustive]` source enums (`DType`, `BackendId`, `SubstrateClass`,
`ScaleGranularity`), which are explicitly allowed to change ordinal/variant set over time.

- **Conversion is explicit, not positional.** `fuel-core-types::dlpack` provides explicit
  `fn fdx_code(d: DType) -> Result<u16>` / `fn dtype_from_fdx(code: u16) -> Result<DType>` (and
  the analogous fns for `BackendId`, `SubstrateClass`, `ScaleGranularity`) implemented with a
  `match`, never `as u16` on the source enum's discriminant.
- **A build-time test pins the mapping.** A unit test asserts the full FDX-code ↔
  `DType`/`BackendId`/`SubstrateClass`/`ScaleGranularity` table by name, so reordering or
  retiring a source variant **breaks the build** (a compile/test failure on the exhaustive
  `match`) instead of silently shifting the ABI. This is the mechanism that keeps the FDX codes
  stable while the source enums stay free to evolve (e.g. `Aocl`/`Mkl` were already retired from
  `BackendId`).
- **Cross-spec.** The kernel-contract spec (FKC) does **not** re-list these codes; it references
  FDX section numbers and the generated `fuel-core-types::dlpack` constants. A cross-spec
  consistency test asserts FKC's dispatch-key codes equal the FDX constants, so the two specs
  cannot drift (the v0.1 hazard of two hand-maintained tables is removed).

### 6.1 Sub-byte / microscaling dtype descriptor (`FDXDTypeExt`)

Carries the *true* element type when the base `DLTensor.dtype` is the honesty-preserving
`uint8` stand-in. Maps Fuel's `DType` onto an explicit **bit-width + packing convention**, so a
`size_in_bytes()==0` sub-byte type sizes its buffer correctly (P5).

```rust
#[repr(C)]
pub struct FDXDTypeExt {
    /// FDX-owned stable code for the logical element type (§6.0 table below);
    /// also covers a "general low-bit int/float" escape. NOT a DType discriminant.
    pub logical_dtype: u16,
    /// True bit-width per logical element: 4, 6, 8, 16, 32, 64. NEVER 0.
    /// For sub-byte Fuel dtypes whose DType::size_in_bytes()==0, this is the
    /// authoritative size source.
    pub bit_width: u16,
    /// How sub-byte elements are packed into bytes (§6.1 packing table).
    pub packing: u8,    // FDXPacking
    /// Lanes (SIMD width within one logical element); 1 for scalar dtypes.
    pub lanes: u8,
    /// Endianness of multi-bit-but-sub-byte packing within a byte: 0 = LSB-first
    /// (element 0 in low nibble), 1 = MSB-first. Matches GGUF/MX conventions.
    pub sub_byte_bit_order: u8,
    pub _pad: u8,
    pub reserved: [u32; 2],
}
```

`FDXDTypeExt::NONE` is all-zero with `logical_dtype = FDX_DTYPE_NONE (0xFFFF)` and is the value
when `FDX_FLAG_HAS_DTYPE_EXT` is unset.

**`logical_dtype` codes** (FDX-owned stable table — §6.0; values match the as-built `DType`
*declaration order as of 2026-06-17*, but are pinned by the §6.0 conversion fn + test, not by
the discriminant):

| code | Fuel `DType` | `bit_width` | notes |
|------|--------------|-------------|-------|
| 0 | U8 | 8 | |
| 1 | I8 | 8 | int8 GEMM operand (added 2026-05-19) |
| 2 | U32 | 32 | |
| 3 | I16 | 16 | |
| 4 | I32 | 32 | |
| 5 | I64 | 64 | |
| 6 | BF16 | 16 | DLPack `kDLBfloat` faithful — sidecar usually absent |
| 7 | F16 | 16 | DLPack faithful — sidecar usually absent |
| 8 | F32 | 32 | faithful |
| 9 | F64 | 64 | faithful |
| 10 | F8E4M3 | 8 | 1 byte; has host type; base may carry as `uint8` |
| 11 | F6E2M3 | 6 | **sub-byte**, MX6; `size_in_bytes()==0` |
| 12 | F6E3M2 | 6 | **sub-byte**, MX6 |
| 13 | F4 | 4 | **sub-byte**, MX4 |
| 14 | F8E8M0 | 8 | MX block-scale dtype (8 exp, 0 mantissa) |
| 0x0100 | `GENERIC_LOW_BIT_INT` | n | escape: arbitrary `bit_width`, signed flag in `reserved[0]` |
| 0x0101 | `GENERIC_LOW_BIT_FLOAT` | n | escape: `(exp,mantissa)` packed in `reserved[0]` |
| 0x0102 | `I4` | 4 | **sub-byte** packed 4-bit signed int (2/byte, `DENSE_SUBBYTE`); no Fuel `DType` (added 2026-06-19) |
| 0x0103 | `U4` | 4 | **sub-byte** packed 4-bit unsigned int (2/byte, `DENSE_SUBBYTE`); no Fuel `DType` |
| 0x0104 | `B1` | 1 | **sub-byte** bitpacked binary (8/byte, `DENSE_SUBBYTE`); no standard DLPack repr; no Fuel `DType` |
| 0x0200 | `COMPLEX64` (reserved) | 64 | DLPack `kDLComplex`; rides the base honestly (§6.1.1), no sidecar code needed |
| 0x0201 | `BOOL` (reserved) | 8 | DLPack `kDLBool`; rides the base honestly (§6.1.1) |
| 0xFFFF | NONE | 0 | dtype_ext absent |

The `I4`/`U4`/`B1` codes (0x0102–0x0104) name sub-byte element types a producer/consumer carries
through the sidecar when the base is the `uint8` stand-in; they have **no Fuel `DType`**, so the
§6.0 `fdx_to_dtype` binding returns `None` for them (exactly like the `GENERIC_LOW_BIT_*` escapes).
They were added (2026-06-19) so a kernel provider's packed `S4`/`U4` (GPTQ/AWQ-style int4) and
bitpacked binary can be named in the sidecar logical-dtype namespace. A dedicated code (vs. the
`GENERIC_LOW_BIT_INT` escape) keeps the structure-key dtype axis clean — `I4` vs `U4` vs `U8` are
distinct codes, not an escape with a flag in `reserved[0]`.

#### 6.1.1 The base `DLTensor.dtype` honors any standard DLPack v1.3 code

The `FDX_DTYPE_*` namespace above governs **only the sidecar** — `FDXDTypeExt.logical_dtype`,
`FDXBufferRef.dtype`, and the `FDXQuant` scale/zero-point dtype fields. It **never constrains the
base `DLTensor.dtype`**, which is a plain standard-DLPack `DLDataType {code, bits, lanes}` and may
carry **any valid DLPack v1.3 code**. So any element type DLPack v1.3 can name **rides the base
honestly and needs no FDX code**, with **no sidecar at all** when nothing else is non-standard:

- `fp8` variants → `kDLFloat8_*` (e.g. `kDLFloat8_e5m2`); **complex** → `kDLComplex` (`bits` = total:
  64 for 2×f32, 128 for 2×f64 — DLPack counts total bits, not per-component); **bool** → `kDLBool`;
  **unpacked** (one-per-byte) 4-bit int → `kDLInt`/`kDLUInt` with `bits = 4`.

A sidecar `FDXDTypeExt` is required only when the base **cannot** faithfully name the type — i.e.
the *packed* sub-byte cases where the base must be the opaque `uint8` stand-in (P5): packed `F4` /
`F6*` / `I4` / `U4` / `B1` and the quant block formats. The honesty invariant (§3) guarantees a
sidecar-blind consumer reads the base correctly in every case. The `COMPLEX64`/`BOOL` rows above are
**reserved placeholders** for the sidecar namespace (kept for completeness / a future Fuel `DType`);
in practice complex and bool ride the base and a producer emits no `FDXDTypeExt` for them.

**`FDXPacking`** (`packing`):

| value | name | meaning |
|-------|------|---------|
| 0 | `BYTE_ALIGNED` | one logical element per `ceil(bit_width/8)` bytes (F8E4M3, F8E8M0) |
| 1 | `DENSE_SUBBYTE` | sub-byte elements packed back-to-back, no per-block framing (e.g. 2×F4/byte) |
| 2 | `MX_BLOCK` | OCP-microscaling: a packed sub-byte payload block + a separate F8E8M0 scale per block (block geometry in `quant`, §6.2) |
| 3 | `GGML_BLOCK` | ggml/GGUF block layout: scales/mins/quants interleaved inside one block struct (per-format byte layout in `quant`) |

> The packing convention plus `quant` (§6.2) together fully determine the byte layout. FDX does
> not enumerate one format; it parameterizes the three families. `FDXPacking` is the sole
> packing authority — the native DLPack `IS_SUBBYTE_TYPE_PADDED` flag is never used (§3.4).

### 6.2 Parametric quantization block layout (`FDXQuant`)

The heart of the "do not hardcode one format" constraint. Describes **GGUF/GGML-style**,
**OCP-microscaling (MX)-style**, *and* **affine-int** quant by parameters. Distinguishes the two
quant regimes Fuel keeps separate (digest §9): static block-quant (baked scales, no free
granularity → `GgmlDType`) vs dynamic affine quant (`ScaleGranularity`/`ScalePair`).

```rust
#[repr(C)]
pub struct FDXQuant {
    /// Quant family/regime (§6.2 table). FDX_QUANT_NONE when no quant.
    pub family: u16,
    /// For GGML family: the GgmlDType code (FDX-owned; mirrors GgmlDType::to_u32
    /// — Q4_0=2, Q4_1=3, Q5_0=6, … Q8K=15, F16=1, BF16=30). 0xFFFF if not GGML.
    pub ggml_dtype: u16,

    /// Block geometry, PARAMETRIC. block_shape[0..block_ndim] gives the block
    /// extent along each quantized axis (e.g. [32] legacy ggml, [256] K-quants,
    /// [1,32] MX along K). 0 ⇒ "not blocked / per-tensor".
    pub block_ndim: u8,
    pub _pad0: [u8; 3],
    pub block_shape: [u32; 4],
    /// Which base axes the block tiles, parallel to block_shape. -1 unused.
    pub block_axes: [i32; 4],

    /// Packing order of the quantized payload relative to the block
    /// (row-major within block, K-major, etc.). FDXPackOrder.
    pub pack_order: u8,
    pub _pad1: [u8; 3],

    /// SCALE descriptor.
    pub scale_present: u8,         // 0/1
    pub scale_dtype: u16,          // logical_dtype code (e.g. 8=F32, 7=F16, 14=F8E8M0 for MX)
    pub scale_placement: u8,       // FDXScalePlacement
    pub scale_granularity: u8,     // FDXScaleGranularity (mirrors ScaleGranularity; +PerBlock)
    pub _pad2: [u8; 3],
    /// Buffer-table index of the scale tensor (§7.4). FDX_BUFFER_INLINE if INLINE
    /// (scales interleaved in the data block, GGML family).
    pub scale_buffer: u32,

    /// ZERO-POINT descriptor (affine-int). zp_present=0 for symmetric quant.
    pub zp_present: u8,
    pub zp_dtype: u16,            // logical_dtype code (commonly I8/I32)
    pub _pad3: u8,
    pub zp_buffer: u32,          // buffer-table index, or FDX_BUFFER_INLINE if inline

    /// For dynamic affine quant: the activation×weight pairing this tensor
    /// participates in, when it is an operand of a quantized matmul. Mirrors
    /// ScalePair. role tells whether THIS tensor is the activation or weight.
    pub scale_pair_act: u8,      // FDXScaleGranularity
    pub scale_pair_weight: u8,   // FDXScaleGranularity
    pub role: u8,                // 0=unspecified, 1=activation, 2=weight
    pub _pad4: u8,

    pub reserved: [u32; 6],
}
```

**`family`** (`FDX_QUANT_*`):

| value | name | regime | scale source | granularity |
|-------|------|--------|--------------|-------------|
| 0xFFFF | `NONE` | — | — | — |
| 0 | `GGML_BLOCK` | static, load-time | baked, INLINE in block | **none** — `ggml_dtype` IS the format; no `scale_granularity`, no `PerBlock`, no separate scale operand |
| 1 | `MX` | static block-scaled | separate F8E8M0 per block | `PerBlock` (MX-only) |
| 2 | `AFFINE_INT` | dynamic, runtime | separate scale (± zero-point) | `PerTensor/PerToken/PerChannel` |
| 3 | `AFFINE_FLOAT` | dynamic FP8/etc. | separate scale | `PerTensor/PerToken/PerChannel` |
| 4 | `AFFINE_BLOCK` | static, block-grained affine (nf4/QLoRA) | **separate** per-block scale operand (absmax), block-shaped | block-shaped scale via `block_shape`; **NOT** `PerBlock` (that code stays MX-only) |

**`FDXScalePlacement`**: `0=INLINE` (interleaved in data block, GGML), `1=SEPARATE_BUFFER`
(`scale_buffer` valid), `2=BROADCAST_PER_AXIS` (scale tensor shape per granularity).

**`FDXScaleGranularity`** (FDX-owned; mirrors `ScaleGranularity` + a block value):
`0=PerTensor (f32[1]) | 1=PerToken (f32[rows]) | 2=PerChannel (f32[cols]) | 3=PerBlock (MX)`.

**`FDXPackOrder`**: `0=ROW_MAJOR_IN_BLOCK | 1=K_MAJOR | 2=GGML_NATIVE` (per-format, defined by
`ggml_dtype`).

**`AFFINE_BLOCK` field semantics (nf4/QLoRA-style block-grained affine).** `family=AFFINE_BLOCK`
carries low-bit data (e.g. NF4, described by `dtype_ext`) plus a **separate per-block scale
operand** (an absmax / block scale), block-shaped:

- **Block geometry is mandatory and parametric:** `block_ndim >= 1` and `block_shape[0..block_ndim]`
  give the block extent along each quantized axis (the QLoRA default is a 1-D `[64]` block along the
  flattened weight). The scale tensor is **block-shaped** — one scale per block — and its buffer
  shape MUST equal the per-axis block count derived from `base.shape` and `block_shape`.
- **The scale is a SEPARATE graph input, named once (single-place rule).** `scale_present == 1`,
  `scale_placement == SEPARATE_BUFFER` (never `INLINE`), and `scale_buffer` is a real buffer-table
  index (§7.4, role `Scale`) — **NOT** an inline baked scale and **NOT** `FDX_BUFFER_INLINE`. The
  absmax block-scale operand is named exactly once (the buffer-table entry); no second copy of the
  scale appears anywhere in the descriptor.
- **`scale_granularity` is NOT `PerBlock`.** `PerBlock` (granularity code 3) stays **MX-only**;
  `AFFINE_BLOCK`'s block grain is expressed by `block_shape` + the block-shaped separate scale, not
  by the `scale_granularity` byte. A consumer reads the block geometry from `block_shape`, not from
  a granularity code.
- **No `ScalePair`** (it is a stored weight format, not a dynamic act×weight matmul pairing):
  `scale_pair_act`/`scale_pair_weight` are unset and `role` is unspecified.
- **`zp_present`** may be set (NF4 is symmetric → `zp_present == 0`; a 4-bit affine-int block format
  with a zero-point sets `zp_present == 1` with `zp_buffer` a separate buffer-table index).

> **Cross-reference — internal source of this family.** The `AFFINE_BLOCK` family (and the
> `GGML_BLOCK` family above) is the **kernel-boundary projection** of the Fuel `SType`/`Encoding`
> type, canonically specified in [`docs/specs/storage-encoding.md`](storage-encoding.md):
> `AFFINE_BLOCK` projects `Encoding::AffineBlock` (sibling-operand model **B** — the per-block
> scale is a separate first-class operand of the consuming op) and `GGML_BLOCK` projects
> `Encoding::GgmlBlock` (inline-baked scale). The projection direction is one-way: `SType` is the
> **internal source of truth** for how a tensor's bytes are encoded; this FDX sidecar is its
> **kernel-boundary projection** (the `SType::to_fdx()` image). FDX stays the normative owner of the
> shared numeric codes (§6.0); `storage-encoding.md` owns the internal type those codes project from.
>
> **Regime separation (digest §9, do not unify):** the families partition cleanly with **no
> overlapping fields**:
>
> - `family=GGML_BLOCK` carries **`ggml_dtype` ONLY** — baked scales interleaved in the block
>   struct, **no `scale_granularity`, no `PerBlock`, no separate scale operand**, and **never** a
>   `ScalePair`. (`scale_present`/`scale_buffer` are not used as the scale source: the scale is baked
>   into the data block and recovered per-format from `ggml_dtype`, not from a granularity code or a
>   separate operand.)
> - `family=MX` carries an F8E8M0 per-block scale and is the **sole** user of
>   `scale_granularity=PerBlock`.
> - `family=AFFINE_BLOCK` carries low-bit data + a **separate block-shaped scale operand**
>   (`block_shape` + `scale_buffer`); it does **not** use `PerBlock`.
> - `family=AFFINE_{INT,FLOAT}` carries `scale_granularity ∈ {PerTensor,PerToken,PerChannel}` and
>   may carry a `ScalePair` when it is a matmul operand.
>
> A consumer keys its dispatch on `(family, ggml_dtype | block_shape | (scale_granularity, role))`,
> matching the flat per-format `Capability` tokens for GGML and the
> `(op, lhs_dtype, lhs_granularity, rhs_dtype, rhs_granularity)` key for affine quant.

### 6.3 Per-axis scales / granularity

Granularity is expressed two ways, deliberately:

1. **Coarse, op-level** — `FDXQuant.scale_granularity` + `scale_pair_*` + `role` (above) give
   the `ScaleGranularity`/`ScalePair` the planner keys dispatch on. This is the *dispatch-key*
   form.
2. **Concrete, buffer-level** — the scale buffer itself is a real entry in the buffer table
   (§7.4) with its own dtype and shape (`f32[1]` / `f32[rows]` / `f32[cols]` /
   `f8e8m0[n_blocks]`), so a consumer can read the scales directly. The two MUST be consistent;
   validation (§8) checks `scale_buffer.shape` against `scale_granularity` and the base shape.

### 6.4 Symbolic / dynamic extent — live-vs-capacity (`FDXExtent`)

The single biggest gap vs generic DLPack (digest §6, §13). One entry per base axis when
`extents_count == base.ndim`. Mirrors `fuel-core-types::shape::Extent` / `DynAxis` and the
`SymId` primitive. **v1 carries three kinds:** `Scalar (0)`, `Range (1)`, and **`Affine (2)`** —
the last carries a bounded affine combination `c0 + Σ cᵢ·symᵢ` over the `SymEnv`, so persistent
decode (`k_len = cached_len + new_tokens`) is expressed symbolically, planned once, and served
every token **without per-pass recompute of a derived symbol**. Scalar and Range are the
degenerate cases of the same descriptor (§4 of this section's subsumption table; §17.3 RESOLVED).

#### Affine term + combination (`FDXAffineTerm`, `FDXAffine`)

```rust
/// One affine term `coeff * sym_id`. `#[repr(C)]` POD, EXACTLY 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FDXAffineTerm {
    /// Signed integer coefficient (i64). Negative coeffs are allowed
    /// (e.g. `capacity - cached_len`); they do NOT relax the per-symbol
    /// bounds of the referenced syms (§6.4 "negative-coeff note").
    pub coeff: i64,
    /// Base `SymId(u32)` bound in the `FDXSymEnv`. `FDX_SYM_NONE` ⇒ unused slot
    /// (then `coeff == 0`). MUST be a BASE symbol — never another affine result
    /// (no nesting).
    pub sym_id: u32,
    pub _pad: u32,
}

/// `value = c0 + Σ_{i<term_count} terms[i].coeff * resolve(terms[i].sym_id)`.
/// Fixed-capacity (`FDX_AFFINE_MAX_TERMS = 4`), inline, POD. EXACTLY 80 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FDXAffine {
    /// Constant term (i64).
    pub c0: i64,
    /// Active term count, `0..=FDX_AFFINE_MAX_TERMS`.
    pub term_count: u8,
    pub _pad: [u8; 7],
    /// Slots `>= term_count` are zeroed (`sym_id = FDX_SYM_NONE`, `coeff = 0`).
    pub terms: [FDXAffineTerm; FDX_AFFINE_MAX_TERMS as usize],
}
```

`FDX_AFFINE_MAX_TERMS = 4` (Appendix A). The decode need is 2 terms
(`cached_len + 1·new_tokens`); 4 leaves headroom for ragged/chunked/MoE composites
(`prefix + chunk·chunks + tail`) while keeping the struct fixed-size. A producer needing > 4
terms gets a typed `AffineTooManyTerms` at **build time** (never a heap pointer, never a
variable-length blob), and falls back to a producer-precomputed composite `SymId` (still legal —
affine is an *additional* expressive option, not a mandate). Affine is the deliberate ceiling:
linear in the base symbols, integer-exact, fixed-size, bounds-checkable. It is **not** general
arithmetic (no `sym*sym`, no `max`, no `floordiv`) — those are `DynScalar`/kernel concerns.

#### The `FDXExtent` struct

> **Layout rule (load-bearing — preserves cross-version byte compatibility).** The leading bytes
> of `FDXExtent` are **frozen at the original §6.4 field order**: `kind` at offset 0, `min` at 8,
> `capacity` at 16, `sym_id` at 24, **`sym_scope` at offset 28** (with `_pad2[3]` at 29..31). The
> new affine machinery (`cap_kind`, the `affine` sub-block) is appended **strictly after offset
> 32**, into what was the original `reserved[u32; 2]` region and the struct's additive growth.
> `sym_scope` does **not** move; an earlier (pre-affine) extent and an affine-aware reader agree
> on every pre-affine field offset. Because `FDXExtent` is an **array element** (the `extents[]`
> stride is its `sizeof`), the implementation MUST pin `size_of::<FDXExtent>()` *and*
> `offset_of!` for **every** field, so a future field-order edit breaks the build instead of
> silently misaligning every entry after index 0 (§5.4, §14 array-element-growth note).

```rust
#[repr(C)]
pub struct FDXExtent {
    /// Axis kind: 0 = Scalar (concrete, == base shape[i]); 1 = Range (single
    /// bounded symbol); 2 = Affine (c0 + Σ cᵢ·symᵢ over the SymEnv). Offset 0.
    pub kind: u8,
    pub _pad: [u8; 3],
    /// Range only: the lower bound of the live value (Extent::min). Scalar:
    /// equals capacity. Affine: the GUARANTEED minimum the producer asserts
    /// (used by the V14 lower bound). Offset 8.
    pub min: u64,
    /// CAPACITY (== base DLTensor.shape[i] == Extent::bound() == Range.max).
    /// Strides in the base DLTensor are keyed to THIS (P4). For Affine this is
    /// the concrete bound the realized affine value is checked against — see
    /// `cap_kind`. Offset 16.
    pub capacity: u64,
    /// Range only: the SymId of the live value (resolved per call via SymEnv).
    /// Stable, serializable, session-independent, UNIFIABLE: two axes that
    /// must move together (KV K_len == V_len) carry the SAME sym_id. (Scalar &
    /// Affine: FDX_SYM_NONE = 0xFFFFFFFF — for Affine the symbols live in
    /// `affine.terms[]`, NOT here.) Widened to u32 to match SymId(u32). Offset 24.
    pub sym_id: u32,
    /// Scope hint for the symbol: 0=InputDetermined, 1=DataDetermined,
    /// 2=SessionScoped. Advisory; the SymEnv supplies the value regardless. For
    /// Affine: the most-constrained scope of its referenced symbols.
    /// FROZEN at offset 28 (do NOT move — cross-version byte compat).
    pub sym_scope: u8,
    pub _pad2: [u8; 3],   // offsets 29..31
    /// AFFINE only (kind==2): how `capacity` is determined (§6.4 "capacity for an
    /// affine axis"). 0 = EXPLICIT (the `capacity` field is the bound — the v1
    /// path, REQUIRED for decode); 1 = AFFINE_MAX (RESERVED, consumer-ahead).
    /// MUST be 0 for kind ∈ {Scalar, Range} (V7) and == 0 (EXPLICIT) in v1 for
    /// Affine. Occupies what was the original reserved[0] low byte. Offset 32.
    pub cap_kind: u8,
    pub _pad3: [u8; 3],   // offsets 33..35
    pub _pad4: u32,       // offsets 36..39 (8-byte align the affine sub-block)
    /// AFFINE only (kind==2): the combination. For Scalar/Range this is all-zero
    /// (`term_count == 0`) and ignored. Carried inline (POD, no pointer). Offset 40.
    pub affine: FDXAffine,
    pub reserved: [u32; 2],
}
```

> **Mapping to the source enum.** These map onto a future generalized
> `fuel-core-types::shape::Extent` as `Extent::Affine { min, capacity, c0, terms }` (or an
> `AffineExpr` helper), but the as-built `{Scalar, Range}` need not change for FDX to subsume them
> — the FDX encoding is wider than the source enum on purpose, exactly as §6.0 establishes the FDX
> code tables are wider/independent of the source enums.

Rules (load-bearing):

- **`capacity` MUST equal `base.shape[i]`.** `dims()` is the capacity, never the live value.
  Validation rejects a mismatch.
- **Strides are NOT in the sidecar** — they live in the base `DLTensor.strides` (always
  explicit, §3.2), keyed to capacity. A kernel walking the live prefix uses (base stride = how
  far per element) + (live extent = how many), exactly the two halves Fuel's `Layout` provides.
- **`sym_id` is transported, not resolved.** The live value comes from a **`SymEnv` supplied
  alongside the tensor at the realize boundary** (the FDX call surface passes a `*const FDXSymEnv`
  next to the sidecar, §7.3), never baked into the sidecar. This is what lets one description
  plan once and serve every token/session/replay (operand rebasing).
- **Realize-time OOB guard (load-bearing).** When a symbol is resolved, the consumer MUST verify
  `min <= value <= capacity`; a resolved value outside `[min, capacity]` is a typed error
  (`ExtentOutOfRange`), **not** silently clamped. This is the OOB-defining check: a `k_len`
  resolving above `K` would walk past the allocation. (Validator check V14; mirrors the
  write-once `SymEnv` contract but adds the bound check the v0.1 draft omitted.)
- **`SymEnv` value is `usize`; FDX widens to `u64`.** As-built `SymEnv` binds `SymId -> usize`
  and `SymId` is `u32`. The boundary form (`FDXSymBinding.value`) is `u64` (always wide enough).
  On a **32-bit host** (`usize == u32`), a binding `value > usize::MAX` is rejected with a typed
  error at the boundary rather than truncated (the narrowing policy).
- **Unification survives the boundary:** equal `sym_id` ⇒ same runtime value. Two distinct
  `Range`s with different syms are not interchangeable even at equal bounds.
- **`sym_scope`** anticipates the two scopes the architecture calls out (input-determined,
  bound up-front, e.g. `cached_len`; data-determined, filled mid-pass by a producer, e.g.
  NonZeroIndices/MoE counts) plus session-scoped (batch size). It is advisory in v1 (no
  consumer yet) but reserved so producers can start tagging.
- **Scalar dynamic op params** (KV write offset, RoPE position, flash `k_len`) are *not*
  dimensions and are NOT carried as `FDXExtent`. They ride the `SymEnv` as `DynScalar` and are a
  *kernel-param* concern, not a tensor-description one. FDX deliberately keeps "length vs mask"
  a lowering decision (masks are an op's job), not a tensor property.

#### 6.4.1 The generalized `FDXExtent` — C (mirrors the Rust above)

```c
typedef struct FDXAffineTerm {  /* sizeof == 16 */
  int64_t  coeff;     /* signed integer coefficient (i64). Negatives allowed.   */
  uint32_t sym_id;    /* SymId(u32). BASE symbol bound in the FDXSymEnv (NOT an  */
                      /* affine result). FDX_SYM_NONE marks an UNUSED slot       */
                      /* (coeff must then be 0).                                 */
  uint32_t _pad;      /* zero on write.                                          */
} FDXAffineTerm;

typedef struct FDXAffine {      /* sizeof == 8 + 8 + 4*16 == 80 */
  int64_t       c0;                          /* constant term (i64).             */
  uint8_t       term_count;                  /* 0..=FDX_AFFINE_MAX_TERMS.        */
  uint8_t       _pad[7];                     /* zero on write.                   */
  FDXAffineTerm terms[FDX_AFFINE_MAX_TERMS]; /* slots >= term_count zeroed.      */
} FDXAffine;

typedef struct FDXExtent {
  uint8_t   kind;        /* 0=Scalar, 1=Range, 2=Affine. Offset 0.              */
  uint8_t   _pad[3];
  uint64_t  min;         /* live lower bound. Offset 8.                          */
  uint64_t  capacity;    /* == base.shape[i]; strides keyed here (P4). Offset 16.*/
  uint32_t  sym_id;      /* Range only; else FDX_SYM_NONE. Offset 24.            */
  uint8_t   sym_scope;   /* advisory scope hint. FROZEN at offset 28.            */
  uint8_t   _pad2[3];
  uint8_t   cap_kind;    /* Affine only: 0=EXPLICIT (v1), 1=AFFINE_MAX (reserved)*/
                         /* MUST be 0 for Scalar/Range (V7). Offset 32.          */
  uint8_t   _pad3[3];
  uint32_t  _pad4;       /* align the affine sub-block. Offsets 36..39.          */
  FDXAffine affine;      /* Affine only; all-zero otherwise. Offset 40.          */
  uint32_t  reserved[2];
} FDXExtent;
```

#### 6.4.2 Affine kind — semantics and rules (load-bearing)

- **Kind discriminant.**
  - `kind = Scalar (0)` — `min == capacity == base.shape[i]`, `sym_id == FDX_SYM_NONE`,
    `cap_kind == 0`, `affine.term_count == 0`. Identical to the as-built `Extent::Scalar`.
  - `kind = Range (1)` — `min ≤ capacity == base.shape[i]`, `sym_id != FDX_SYM_NONE`,
    `cap_kind == 0`, `affine.term_count == 0`. Identical to the as-built `Extent::Range`. **The
    default for one-symbol axes** — a producer SHOULD emit `Range`, not a one-term affine
    (canonicalization, below).
  - `kind = Affine (2)` — the live value is the affine combination in `affine`, evaluated through
    the `SymEnv` at realize. `sym_id == FDX_SYM_NONE` (symbols are in `affine.terms[]`),
    `cap_kind == EXPLICIT (0)` in v1, `capacity == base.shape[i]`.

- **Affine evaluation (realize-time, §7.3 contract) — overflow-CHECKED per step.**

  ```text
  value: i128 = affine.c0 as i128
  for i in 0 .. affine.term_count:
      s: u64 = lookup(env, affine.terms[i].sym_id)   // typed UnboundSymbol if absent
      // BOTH operands widen to i128 BEFORE the multiply (coeff is i64, s is u64 —
      // s widens losslessly to i128, always >= 0). checked_* at EVERY step,
      // NOT a wrapping `+=`, NOT `coeff.checked_mul(s as i64)` (that overflows in
      // i64 far earlier and a u64 sym > i64::MAX would mis-cast negative):
      prod  = (affine.terms[i].coeff as i128).checked_mul(s as i128)   ?: AffineOverflow
      value = value.checked_add(prod)                                  ?: AffineOverflow
  // then narrow (below) and bound-check (V14)
  ```

  i128 is **not** unconditionally overflow-free: with `FDX_AFFINE_MAX_TERMS = 4` and pathological
  i64 coeffs × u64 bindings, a single term can reach ~`2^126` and four summed can exceed `2^127`,
  so the **running accumulation MUST use `checked_add`/`checked_mul` at each step** and any
  overflow ⇒ typed `AffineOverflow` (V17), never a wrap and never deferred to a single final
  check. (With realistic decode magnitudes it cannot overflow; the check is the never-silent-
  coercion guarantee, not a hot path.)

- **Every referenced `sym_i` MUST be bound** in the `FDXSymEnv`; an unbound symbol is typed
  `UnboundSymbol` (matching `SymEnv`'s write-once / presence contract, §7.3), **never a silent 0**
  (mirrors `Shape::resolve` erroring on an unbound dynamic axis).

- **u64 ↔ usize narrowing (V17 gates BEFORE V14 sees the value).** Resolve each `sym_i` to its
  `u64` binding, widen to i128, accumulate (checked), then **narrow once**. The **`>= 0` check is
  V17's**, applied to the i128 result *before* narrowing: a negative i128 cannot be narrowed to
  `u64`/`usize`, so a negative affine live length (a producer bug) is rejected as `ExtentOutOfRange`
  **by the `>= 0` gate, not by V14's `min ≤ value`** — V14 only ever sees a non-negative, narrowed
  value, so there is no "negative compared against `min`" case to reason about. On a **32-bit host**,
  a narrowed `value > usize::MAX` ⇒ typed `ExtentOutOfRange` — **never truncated** (the §6.4
  narrowing rule extended to the affine result). Only then does V14's `min ≤ value ≤ capacity` run
  on the safe, narrowed value.

- **Capacity for an affine axis (the bounds key).** Strides/allocation stay keyed to a **concrete
  capacity** (P4):
  - `cap_kind = EXPLICIT (0)` — **the v1 path, the only `cap_kind` a producer emits.** The
    `capacity` field *is* the concrete bound and MUST equal `base.shape[i]` (V7). For decode this
    is the KV buffer's fixed capacity `K`: the buffer is physically allocated for `K` slots,
    strides keyed to `K`, and `k_len = cached_len + new_tokens` is checked `min ≤ k_len ≤ K`. The
    honesty invariant (§3, §3.1) requires `base.shape[i] == capacity` so a sidecar-blind consumer
    reads a correctly-sized, fully-backed `[…, K, …]` tensor (V8). The affine expression describes
    the **live prefix length**, never the capacity; the §3.1.1 "live-prefix export is a COPY when
    the symbolic axis is not leading" reasoning is unchanged.
  - `cap_kind = AFFINE_MAX (1)` — **RESERVED (consumer-ahead).** Capacity would be the affine
    value at each symbol's per-binding maximum (growable/ragged buffers). **Not emitted in v1**;
    validators reject it as `UnsupportedVersion`-class until a consumer exists (V7/V16). The field
    is present so adding it later is additive (§14). The decode frontier does not need it.

- **Per-symbol bounds are NOT relaxed by the affine guard (no-OOB is compositional).** V14/V17
  bound the affine **result** only. Any base symbol that is **also** used elsewhere as an extent
  or as an offset into the same buffer (e.g. `cached_len` used both in `k_len = cached_len + seq`
  *and* as the persistent-decode write offset) MUST carry its **own** `FDXExtent`/bound — most
  naturally as its own `Range` extent on the consumed-prefix axis, or as a bounded `DynScalar` in
  the `SymEnv`. The unification-by-`sym_id` rule makes this automatic: the same `cached_len`
  symbol resolves identically wherever it appears. A **negative-coefficient** affine (e.g.
  `capacity - cached_len`) does **not** relax the per-symbol bounds of its terms — a binding that
  keeps the sum in `[min, capacity]` while a base term is itself out of its own range is rejected
  by that term's own bound, not by the sum. This keeps the no-OOB property compositional rather
  than only checking the final sum.

- **Determinism / unification.** Two axes that must move together carry the **same affine
  expression over the same base syms** (K-length and V-length both `cached_len + new_tokens`), so
  they resolve to the same value by construction — the as-built `Extent`/`SymEnv` unification-by-id
  lifted to expressions (an affine expression unifies when its term set + coeffs + `c0` match;
  the V16 no-duplicate-sym rule + a canonical sort by `sym_id` make Hash/Eq order-independent for
  plan-cache keying).

#### 6.4.3 Backward-compat: Scalar/Range are the degenerate cases (subsumption)

Every existing §13 example still validates byte-for-byte (they are all `Scalar`/`Range`); affine
is purely additive for the genuinely-composite case.

| as-built / pre-affine §6.4 | canonical FDX kind | equivalent affine form — *math only, NOT a legal encoding* |
|---|---|---|
| `Extent::Scalar(v)` | `kind=0` (`min=capacity=v`, `sym=NONE`) | `affine{c0=v, term_count=0}` — **V16 rejects; emit Scalar** |
| `Extent::Range{min,max,sym}` | `kind=1` (`min, capacity=max, sym`) | `affine{c0=0, term_count=1, [{coeff=1, sym}]}` — **V16 rejects; emit Range** |
| `k_len = cached_len + seq` (seq const) | `kind=2` | `affine{c0=seq, term_count=1, [{1, cached_len}]}` — **legal** (non-zero `c0`) |
| `k_len = cached_len + new_tokens` (both sym) | `kind=2` | `affine{c0=0, term_count=2, [{1,cached_len},{1,new_tokens}]}` — **legal** |

> The "equivalent affine form" column is **mathematical equivalence for understanding, NOT a legal
> encoding**: the first two rows are exactly the degenerate forms V16 **rejects** (a constant is
> always `Scalar`; a bare coeff-1 zero-`c0` symbol is always `Range`). Only genuinely-composite
> rows (`c0 != 0`, or `term_count >= 2`, or a non-unit coeff) are legal `kind=2` encodings.

**Canonicalization rule (producer policy, V16-checked):** a producer MUST emit the *lowest*
sufficient kind — `Scalar` for a constant, `Range` for a single coeff-1 zero-`c0` symbol, `Affine`
only for a genuinely multi-term / non-unit-coeff / non-zero-`c0` combination. V16 **rejects** the
degenerate affine forms so the two encodings never diverge for the same fact, keeping the simple
consumer path (`Scalar`/`Range`) untouched.

### 6.5 Tiling / alignment hints (`FDXTiling`)

Optional. The planner already decides repacks/`Op::Contiguize`; these hints let a producer
communicate the alignment/granularity a buffer *was* laid out for, so the boundary honors the
optimizer's repack decisions instead of silently violating them (digest §2).

```rust
#[repr(C)]
pub struct FDXTiling {
    /// Required base-address alignment in bytes (mirrors BackendCapabilities
    /// ::required_alignment). 0 = unspecified. NOTE: this is the *internal*
    /// layout alignment; a boundary-(b) export is additionally subject to the
    /// hard 256-byte DLPack data-pointer rule (§3.3), which is the floor.
    pub alignment_bytes: u32,
    /// Smallest addressable unit in bits (mirrors access_granularity_bits).
    /// 0 = unspecified (assume 8).
    pub access_granularity_bits: u32,
    /// Optional inner tile shape the buffer is blocked into (e.g. for a
    /// swizzled / tensor-core-friendly layout). tile_ndim==0 ⇒ no tiling.
    pub tile_ndim: u8,
    pub _pad: [u8; 7],
    pub tile_shape: [u32; 4],
    pub reserved: [u32; 4],
}
```

Hints, not commitments, **and not costs**: a consumer MAY ignore tiling and treat the buffer per
the base `strides`; if it cannot, it surfaces a typed error (it does NOT silently re-tile). The
*cost* of a contiguize / re-tile / materialize alternative — and the planner's choice among
contiguize-vs-strided-vs-materialize — is **out of FDX scope by design**: FDX is pure
description (P3). Those declared costs are the **kernel-contract (FKC) spec's** responsibility;
the planner reads FDX (what the buffer *is*) together with the FKC cost model (what each
alternative *costs* on the target backend) and commits a path (§9.3). FDX intentionally carries
no cost slot; the gap noted in the critique is owned by FKC, not FDX.

### 6.6 Residency / substrate (`FDXResidency`)

Finer than DLPack's `(device_type, device_id)`. Lets the planner decide same-buffer-vtable-swap
vs needs-a-copy — Vulkan and CUDA on the *same silicon* must not alias (digest §2, §13). Codes
are FDX-owned (§6.0), pinned by conversion fn + test against the `#[non_exhaustive]`
`BackendId` / `SubstrateClass` source enums.

```rust
#[repr(C)]
pub struct FDXResidency {
    /// Three-tier residency: 0=Device, 1=Host, 2=DiskMmap. Disk-mmap'd weights
    /// are zero-copy views, distinct from host RAM (digest §11).
    pub tier: u8,
    /// FDX substrate class (§6.0 table) — pins fuel SubstrateClass: 0=HostBytes,
    /// 1=CudaUntyped, 2=VulkanBuffer, 3=MetalBuffer. Decides aliasing. Not a
    /// SubstrateClass discriminant (that enum is #[non_exhaustive]).
    pub substrate: u8,
    /// FDX backend code (§6.0 table) — pins BackendId: 0=Cpu,1=Cuda,2=Vulkan,
    /// 3=Metal. For precise pointer-namespace identity beyond DLDevice. Not a
    /// BackendId discriminant (Aocl/Mkl already retired from that enum).
    pub backend_id: u8,
    pub _pad: u8,
    /// Device ordinal within the backend (mirrors DeviceLocation gpu_id).
    pub device_index: u32,
    /// Whether this buffer is a zero-copy view into a larger mmap'd region
    /// (1) vs an owned allocation (0). View ⇒ deleter must not free the region.
    /// A view that does NOT back the full capacity shape forces
    /// FDX_FLAG_MEANING_REQUIRES_EXT (§3.1).
    pub is_mmap_view: u8,
    pub _pad2: [u8; 7],
    pub reserved: [u32; 4],
}
```

> **Why finer than DLDevice:** DLPack says `kDLVulkan, id=0` and `kDLCUDA, id=0` are different
> device *types*, but cannot say "these two backends share the same physical silicon yet
> different pointer namespaces, so a vtable-swap is illegal and a copy is required." `substrate`
> + `backend_id` + `device_index` reproduce Fuel's `SubstrateClass`+`DeviceLocation`
> `shares_storage` predicate across the boundary.

### 6.7 Storage class + session (`FDXStorage`)

```rust
#[repr(C)]
pub struct FDXStorage {
    /// Storage class (digest §12, decision #6/#10): 0=Shared, 1=Session,
    /// 2=Transient. Session-state (KV-caches) is the durable-interop unit;
    /// Transient never crosses to disk.
    pub class: u8,
    pub _pad: [u8; 3],
    pub _pad_align: u32,
    /// Session identity for class=Session (KV-cache snapshotting / replay).
    /// FDX_SESSION_NONE (0) when not session-scoped.
    pub session_id: u64,
    pub reserved: [u32; 4],
}
```

### 6.8 Multi-output bundle views (`FDXOutputView`)

One allocation, N sub-views at byte offsets, each independent in dtype/shape/layout (digest
§10, 12-multi-output Option C). When `FDX_FLAG_IS_BUNDLE` is set, the base `DLTensor` describes
the *whole bundle buffer* as `uint8` and `views[0..views_count]` partition it.

```rust
#[repr(C)]
pub struct FDXOutputView {
    /// Byte offset into the bundle buffer (buffer-table index 0). Each slot's
    /// natural start; an external re-export of a single slot carries this in the
    /// DLTensor byte_offset over the 256-aligned bundle base (§3.3).
    pub byte_offset: u64,
    pub len_elements: u64,
    /// FDX logical dtype code for this slot (may itself be sub-byte → its own
    /// dtype_ext semantics; v1 keeps slot dtype simple/standard — see §17).
    pub dtype: u16,
    pub _pad: [u8; 2],
    pub ndim: u32,
    pub shape: [u64; 6],      // matches Fuel DimVec inline capacity (6)
    /// Slot strides, length ndim — ALWAYS explicit (§3.2); int64, negatives
    /// first-class (§3.2.1 / V13 signed touched-range OOB check); when this slot
    /// is re-exported as a standalone DLTensor they become its (never-NULL) strides.
    pub strides: [i64; 6],
    /// Stable slot name hash (FNV-1a of the slot name), for diagnostics /
    /// View resolution. 0 = anonymous.
    pub name_hash: u64,
    pub reserved: [u32; 4],
}
```

This mirrors `OutputView { byte_offset, len_elements, dtype, shape, layout, name }`. The kernel
ABI remains outputs-arity-1 (the bundle); `views` is the authoring contract for resolving slots
back to ordinary tensors via `Op::View` / `Op::ViewOwned`.

### 6.9 Gather / indexed-residency — paged/blocked KV cache (`FDXIndexedResidency`)

Models a vLLM-style **paged / blocked KV cache** — the cache `OpKind::PagedAttn` consumes — as a
**single FDX tensor**. The base `DLTensor` stays the honest contiguous block pool (dense `uint8`
bytes); the gather mapping (logical sequence position → physical block id) lives **only** in the
sidecar. It is a sub-block of `FDXSidecar` exactly like `FDXQuant`, not a third top-level struct.

**Grounding (as-built — verified 2026-06-17).** The op params actually stored are
`FusedOpParams::PagedAttn { softmax_scale: f32, block_size: usize, softcap: Option<f32> }`
(`fuel-graph/src/registry.rs:241-245`) — **three fields only**. The geometry the addition mirrors
(`b`, `hq`, `hkv`, `sq`, `d`, `max_blocks_per_seq`, `num_blocks`) is **not** stored as op params;
it is carried on the lowered `KernelRef::PagedAttn { b, hq, hkv, sq, d, block_size,
max_blocks_per_seq, num_blocks, softmax_scale, softcap }` (`fuel-dispatch/src/kernel.rs:314-331`,
all `usize`) and otherwise **derived at runtime from the operand shapes**: `k_cache`/`v_cache` =
`[num_blocks, block_size, Hkv, D]`, `block_table` = `[B, max_blocks_per_seq]` (U32),
`context_lens` = `[B]` (U32) — see the CPU kernel `fuel-cpu-backend/src/byte_kernels.rs:6789-6837`
(operand byte-size contract) and `:6900-6912` (per-accessed-slot block-id range check). The
`FDXIndexedResidency` fields below therefore **mirror DERIVED shape facts** (cross-checked against
the operand `FDXBufferRef` shapes in V21(a), and the pool backing in V19/V20), not nonexistent
op-param fields; only `block_size` (and `softmax_scale`/`softcap`, which are kernel params, not
tensor description) are actual param fields.

#### 6.9.1 Honesty invariant preserved (§3 — load-bearing)

The base `DLTensor` describes the **contiguous physical block pool as honest dense `uint8` bytes**
— never the logical (gathered, scattered) tensor:

- base `dtype = {kDLUInt, 8, 1}`, `shape = [pool_size_bytes]`, `strides = [1]` (explicit, §3.2),
  `data` 256-aligned (§3.3), `byte_offset = 0`.
- A **sidecar-blind consumer sees exactly the raw pool** — a correctly-sized opaque byte buffer —
  and never a mislabeled scattered tensor. There is no honest dense interpretation of a paged
  cache (logical rows are physically scattered across non-adjacent blocks — strictly worse than
  the §3.1.1 middle-axis case), so `FDX_FLAG_MEANING_REQUIRES_EXT` is **mandatory** (V18). A blind
  consumer goes through producer policy §9.1: refuse, or materialize a dense un-paged copy with
  `DLPACK_FLAG_BITMASK_IS_COPIED` set.
- The base is also honest about the **physical typed element**: the true per-token type
  (F16/BF16/F32) rides `FDXDTypeExt`, and the pool's physical typed shape
  `[num_blocks, block_size, Hkv, D]` rides `gather.physical_shape`. The `uint8` base is sized off
  the byte count, never off a `size_in_bytes()==0` dtype. **V19 requires the base byte length to
  exactly equal the pool allocation walk** — `base.shape[0] == physical_strides[0] * num_blocks *
  elem_bytes` (the same `stride·extent` definition V20 uses; for a dense gap-free pool this equals
  `num_blocks * block_size * intra_block_typed_count * elem_bytes`, and for a padded pool the
  strided form is larger). One definition of "pool byte length" is shared by V19 and V20, so a
  padded pool passes both; the honest-`uint8` cover is enforced, not just asserted in prose.

#### 6.9.2 `FDXIndexedResidency` + `FDXBlockTable` — Rust

```rust
/// GATHER / INDEXED-RESIDENCY descriptor: re-interprets a contiguous physical
/// BLOCK POOL (the honest uint8 base, §3) as a logically-gathered tensor via a
/// per-sequence block table. Models a vLLM-style paged KV cache as a SINGLE FDX
/// tensor. Description only (P3/G7): no cost, no decision. Frozen-size (§5.4);
/// grows only via `reserved`. All geometry fields mirror DERIVED operand-shape
/// facts (V21(a) cross-checks them against the operand FDXBufferRef shapes; V19/
/// V20 check pool backing), NOT stored op-param fields (FusedOpParams::PagedAttn
/// has only 3 fields, §6.9).
#[repr(C)]
pub struct FDXIndexedResidency {
    /// Gather kind: 0 = FDX_GATHER_NONE (absent), 1 = FDX_GATHER_PAGED_BLOCKS
    /// (vLLM/PagedAttn block-pool). Future kinds (ragged/CSR) are additive.
    pub kind: u8,
    pub _pad0: [u8; 3],

    /// ── PHYSICAL POOL geometry (the honest base; mirrors derived k/v_cache
    ///    shape [num_blocks, block_size, Hkv, D]) ───────────────────────────
    /// Number of fixed-size blocks in the pool (derived = k_cache.shape[0]).
    /// Widened to u64 to match the usize source and avoid author-side narrowing
    /// overflow; V18 checks it fits the runtime kernel's usize.
    pub num_blocks: u64,
    /// Tokens (logical positions) per physical block (= KernelRef block_size,
    /// = k_cache.shape[1]). NEVER 0 (V18). Pool token capacity = num_blocks *
    /// block_size. u64 (see num_blocks).
    pub block_size: u64,

    /// Buffer-table index (§7.4) of the physical block-pool buffer. Role =
    /// FDX_BUFFER_POOL. MUST be a valid index; conventionally 0 (the base data
    /// buffer). P7 — index, not pointer.
    pub pool_buffer: u32,
    pub _pad1: u32,

    /// PHYSICAL (pool) typed shape mirroring the as-built cache
    /// `[num_blocks, block_size, Hkv, D]`. physical_ndim gives the rank (<= 6,
    /// §5 inline rule). physical_shape[0] == num_blocks, physical_shape[1] ==
    /// block_size (V19). The base DLTensor's own shape stays the dense BYTE
    /// shape [pool_size_bytes].
    pub physical_ndim: u8,
    pub _pad2: [u8; 7],
    pub physical_shape: [u64; 6],
    /// Physical pool strides in **TYPED ELEMENTS** (NOT bytes), length
    /// physical_ndim, ALWAYS explicit (§3.2). The byte address composition
    /// (§6.9.4 step 3) multiplies the element-offset by `elem_bytes` EXACTLY ONCE
    /// — `byte_addr = pool.data + pool.byte_offset + elem_bytes * dot(coord,
    /// physical_strides)` — and V20/V19 size the pool the same way
    /// (`physical_strides[0] * num_blocks * elem_bytes`). The per-block (slowest)
    /// axis is physical_strides[0]. These are the HONEST strides of the actual
    /// allocation (V20): a padded pool (physical_strides[0] > the dense per-block
    /// stride) must be SIZED for its padding so block id num_blocks-1 never walks
    /// past size_bytes. (Negative strides are permitted in general — §3.2.1 / V13 —
    /// but a paged pool's physical strides are non-negative by construction.)
    pub physical_strides: [i64; 6],
    /// FDX logical dtype code of one pool element (the TRUE per-token type,
    /// e.g. 7=F16, 6=BF16, 8=F32). Mirrors FDXDTypeExt.logical_dtype when
    /// present; authoritative for sizing the typed pool (never size_in_bytes()==0).
    pub element_dtype: u16,
    pub _pad3: [u8; 2],

    /// ── BLOCK TABLE (logical → physical), see FDXBlockTable ────────────────
    pub block_table: FDXBlockTable,

    /// ── LOGICAL (gathered) shape & liveness ───────────────────────────────
    /// Batch size B (number of logical sequences; derived = block_table.shape[0]
    /// = context_lens.shape[0]). u64 (see num_blocks).
    pub num_sequences: u64,
    /// Per-sequence logical CAPACITY = max_blocks_per_seq * block_size (P4); the
    /// LIVE length is context_lens (symbolic, below). Computed in u64 WITHOUT
    /// overflow (V18: max_seq_capacity == max_blocks_per_seq * block_size in u64).
    pub max_seq_capacity: u64,

    /// LOGICAL (gathered) per-sequence shape, e.g. [Hkv, S_cap, D] or
    /// [S_cap, Hkv, D]. logical_ndim gives the rank. The symbolic (live) axis is
    /// seq_axis; its capacity == max_seq_capacity and its live extent is carried
    /// in `logical_extents[seq_axis]` (below) — NOT in the base `extents[]`,
    /// which annotate the 1-D byte pool (§6.9.4).
    pub logical_ndim: u8,
    /// Which axis of logical_shape is the per-sequence length (the gathered /
    /// symbolic axis). 0xFF if none / not applicable.
    pub seq_axis: u8,
    pub _pad4: [u8; 6],
    pub logical_shape: [u64; 6],

    /// Per-LOGICAL-axis live extents, parallel to logical_shape (NOT to the base
    /// DLTensor axes). logical_extents_count is 0 or logical_ndim (V21e). This is
    /// the home of the live seq length: logical_extents[seq_axis] is a Range (or
    /// Affine) extent over max_seq_capacity carrying the seq SymId, with min=0
    /// (empty/finished sequences are legal — V21d/e), and gets its OWN
    /// V7/V14/V16/V17 arms keyed to logical_shape/max_seq_capacity. Inline, fixed-
    /// capacity (6), so the gathered axis is NOT outside the extents machinery.
    pub logical_extents_count: u8,
    pub _pad5: [u8; 7],
    pub logical_extents: [FDXExtent; 6],

    /// ── CONTEXT LENGTHS (per-sequence LIVE extent; symbolic — P4) ─────────
    /// Buffer-table index (§7.4) of the context_lens buffer (role =
    /// FDX_BUFFER_CONTEXT_LENS), a [num_sequences] U32 tensor of true live
    /// lengths. FDX_BUFFER_NONE if the live length is carried purely
    /// symbolically (context_len_sym) with no buffer.
    pub context_lens_buffer: u32,
    /// When all sequences share ONE symbolic live length (the common batched-
    /// decode case), its SymId; logical_extents[seq_axis] carries the SAME
    /// sym_id (P4 unification). FDX_SYM_NONE if per-seq lengths differ (then
    /// context_lens_buffer is authoritative and seq_axis is data-determined).
    pub context_len_sym: u32,
    /// Scope hint for the context length symbol (matches FDXExtent.sym_scope).
    pub context_len_scope: u8,
    pub _pad6: [u8; 3],

    pub reserved: [u32; 6],
}

/// Logical → physical block mapping for a paged pool. Batched over sequences:
/// logical position `t` of sequence `s` lives in physical block
/// `id = block_ids[s * max_blocks_per_seq + (t / block_size)]` at intra-block
/// offset `t % block_size`. Frozen-size (§5.4); grows only via `reserved`.
#[repr(C)]
pub struct FDXBlockTable {
    /// Buffer-table index (§7.4) of the block-id table buffer (role =
    /// FDX_BUFFER_BLOCK_TABLE), a [num_sequences, max_blocks_per_seq] tensor of
    /// physical block ids. P7 — index, not ptr.
    pub table_buffer: u32,
    /// FDX logical dtype code of a block id. PINNED to U32 in v1 (code 2),
    /// matching the as-built U32 block_table (the kernel reads `&[u32]`). A
    /// non-U32 id_dtype ⇒ UnsupportedGatherKind-adjacent (V18). Block ids index
    /// [0, num_blocks).
    pub id_dtype: u16,
    pub _pad0: u16,
    /// Slots per sequence (columns) = max_blocks_per_seq = ceil(max_seq_capacity
    /// / block_size). Widened to u32 (was u16 in the draft — u16 capped at 65535
    /// blocks/seq ≈ 1.05M tokens at block_size 16, plausibly exceeded by long-
    /// context models, and a usize->u16 narrowing would silently wrap). V18
    /// guards the usize->u32 narrowing (overflow ⇒ GatherIncoherent, not a wrap).
    pub max_blocks_per_seq: u32,

    /// Sentinel block id meaning "unmapped / not yet allocated" (a row's tail
    /// past the sequence's allocated blocks). FDX_BLOCK_UNMAPPED (0xFFFFFFFF) by
    /// default. INVARIANT (V18): unmapped_sentinel MUST be representable in
    /// id_dtype AND MUST be >= num_blocks, so the single `id >= num_blocks`
    /// range check provably catches BOTH out-of-range and unmapped (the as-built
    /// kernel does range-only, byte_kernels.rs:6902). A consumer MUST NOT
    /// dereference an unmapped slot.
    pub unmapped_sentinel: u32,

    /// Layout flags: bit0 = ids sorted within a row (advisory); bit1 = table is
    /// shared/read-only across the call. 0 = no claims.
    pub layout_flags: u32,

    pub reserved: [u32; 4],
}
```

C mirrors both structs field-for-field with the same `#[repr(C)]` order; size pins per §5.4.

#### 6.9.3 `kind` codes + buffer roles (FDX-owned, §6.0)

| value | name | meaning |
|-------|------|---------|
| 0 | `FDX_GATHER_NONE` | gather block absent (HAS_GATHER clear) |
| 1 | `FDX_GATHER_PAGED_BLOCKS` | fixed-size block pool + per-seq block table (vLLM/PagedAttn) |
| 2.. | (reserved) | ragged/CSR gather, hierarchical paging — additive (§14) |

`FDXIndexedResidency::NONE` is all-zero with `kind = FDX_GATHER_NONE (0)`; V18 rejects `kind == 0`
while `HAS_GATHER` is set, and rejects an unknown `kind` as typed `UnsupportedGatherKind` (never a
guess — §14 `#[non_exhaustive]` spirit).

The `FDXBufferRef.role` enum (§7.2) gains three gather roles (additive): `5 = FDX_BUFFER_POOL`,
`6 = FDX_BUFFER_BLOCK_TABLE`, `7 = FDX_BUFFER_CONTEXT_LENS`. A new constant
`FDX_BUFFER_NONE = 0xFFFFFFFE` (distinct from `FDX_BUFFER_INLINE = 0xFFFFFFFF`) marks "no such
buffer" for `context_lens_buffer` when the live length is purely symbolic.

> **Single-place rule (parent RESOLVED DECISIONS, the FKC touch-point).** If the paged-attention
> kernel's ABI takes `block_table` / `context_lens` as **separate graph inputs** (it does — the
> `KernelRef::PagedAttn` operand order is `[q, k_cache, v_cache, block_table, context_lens,
> alibi?]`, `fuel-dispatch/src/kernel.rs:314-331`), they are **FKC `accept.inputs` operands**, and
> the FDX `pool_buffer` / `block_table.table_buffer` / `context_lens_buffer` indices point at the
> **same** buffers (V20b cross-check, not a copy). The FDX gather descriptor does not duplicate the
> data; it describes the indexing relationship the disjoint operands otherwise leave implicit. Each
> table is described in exactly one authoritative place + a consistency check.

#### 6.9.4 Indexing composition, strides, and the no-OOB argument (§3.1, §6.4)

The load-bearing part: how the gather composes with per-axis strides and capacity so the §3.1
no-OOB argument still holds.

1. **Capacity for layout (P4).** The pool buffer is sized for **full pool capacity** and
   `physical_strides` are the honest, gap-free strides of the actual allocation. V19 mirrors
   §3.1/V8 literally — `buffers[pool_buffer].size_bytes >= physical_strides[0] * num_blocks *
   elem_bytes` (and the analogous `stride * extent` on every physical axis), **not** just the
   dense element-count product — so a *padded* pool (where `physical_strides[0]` exceeds the dense
   per-block stride) is sized for its padding and block id `num_blocks-1` can never walk past
   `size_bytes`. The block table is sized for full capacity `[num_sequences, max_blocks_per_seq]`;
   unallocated tail slots carry `unmapped_sentinel`.

2. **Symbol for liveness (P4).** The per-sequence live length is `context_lens` — a symbolic
   extent, never folded into shape. When the batch advances together, `context_len_sym` is one
   `SymId` resolved per call; `logical_extents[seq_axis]` carries the **same** `sym_id`. When
   per-sequence lengths differ, `context_lens_buffer` is the authoritative `[B]` U32 buffer (a
   data-determined sym) and the kernel reads `context_lens[s]` per sequence (matching the as-built
   CPU kernel).

3. **Indexing composition (logical → physical address).** To read logical sequence `s`, position
   `t` (`0 <= t < L_s`), element coordinate `c` within the per-token shape:

   ```text
   physical_block = block_table.ids[s * max_blocks_per_seq + (t / block_size)]   // gather
   if (t / block_size) >= max_blocks_per_seq -> already excluded by V21d         // column guard
   if physical_block >= num_blocks  -> ERROR (BlockIdOutOfRange)                 // catches OOB
                                                                                 //  AND unmapped
                                                                                 //  (sentinel >=
                                                                                 //  num_blocks, V18)
   intra_block_token = t % block_size
   // physical_strides are in TYPED ELEMENTS; scale the whole element-offset by elem_bytes
   // exactly once (V20 and the §13.8 example multiply by elem_bytes the same way):
   byte_addr = pool.data + pool.byte_offset
             + elem_bytes * ( physical_block    * physical_strides[0]    // per-block (slowest)
                            + intra_block_token * physical_strides[1]    // per-token-in-block
                            + dot(c, physical_strides[2..]) )            // intra-token elements
   ```

   The per-axis strides are the honest pool strides (keyed to capacity); the gather only chooses
   *which block* (the slowest physical axis). The base pool's strides describe a dense, walkable
   buffer; the block table merely permutes the block axis — which is why the honesty invariant
   survives and why `MEANING_REQUIRES_EXT` is mandatory (the gathered logical tensor has no single
   set of dense strides). The single `physical_block >= num_blocks` test catches **both** an
   out-of-range id and the unmapped sentinel, because V18 forces `unmapped_sentinel >= num_blocks`
   — exactly what the as-built kernel relies on (`byte_kernels.rs:6902`).

4. **Three honest shapes.** `physical_shape` is the dense pool typed shape
   `[num_blocks, block_size, Hkv, D]`; `logical_shape` is the per-sequence gathered shape; the
   **base** `DLTensor.shape` is the honest byte shape `[pool_size_bytes]`. Never conflated (the
   §3.1.1 "never label a scatter as dense" discipline, generalized).

---

## 7. Buffer references, the call surface, and the buffer table

### 7.1 Why a buffer table, not raw pointers (P7)

A quantized tensor is *multi-buffer* (data + scale(s) ± zero-points), and a bundle has sub-views
into one buffer. Rather than scatter raw pointers through the sidecar (un-serializable,
un-mmap-able), all buffers are collected in one buffer table and referenced by **index**.
Index 0 is always the base `DLTensor.data` buffer.

### 7.2 `FDXBufferRef`

```rust
#[repr(C)]
pub struct FDXBufferRef {
    /// Role: 0=Data, 1=Scale, 2=ZeroPoint, 3=BundleBacking, 4=Aux,
    /// 5=FDX_BUFFER_POOL, 6=FDX_BUFFER_BLOCK_TABLE, 7=FDX_BUFFER_CONTEXT_LENS
    /// (the last three are the gather roles, §6.9.3).
    pub role: u8,
    pub _pad: [u8; 1],
    /// FDX logical dtype code of THIS buffer.
    pub dtype: u16,
    pub _pad2: u32,
    /// For the LIVE in-memory form only: device pointer (NEVER serialized).
    /// On export it is 256-byte aligned (§3.3). In serialized form this field
    /// is 0 and the byte location comes from the containing DLManagedTensor /
    /// file mapping.
    pub data: *mut core::ffi::c_void,
    pub device: DLDevice,
    /// Intra-buffer logical start (§3.3); MUST satisfy byte_offset <= size_bytes.
    pub byte_offset: u64,
    /// Physical allocated byte count of this buffer. Validator cross-checks this
    /// against the capacity-shaped extent for the no-OOB guarantee (§3.1 / V8).
    pub size_bytes: u64,
    pub ndim: u32,
    pub _pad3: u32,
    pub shape: [u64; 6],
    /// Length ndim, ALWAYS explicit (§3.2); int64 — negatives first-class
    /// (§3.2.1 / V13 signed touched-range OOB check). Never NULL.
    pub strides: [i64; 6],
    pub reserved: [u32; 4],
}
```

### 7.3 The Fuel kernel call surface (`FDXSymEnv`)

At boundary (a), a kernel receives, alongside `DLTensor*` + `FDXSidecar*`, an **optional**
`FDXSymEnv*` carrying the per-pass `SymId → u64` bindings (the realize-time resolution of every
symbolic extent / dynamic scalar). It is a flat sorted array for O(log n) lookup and is
**never serialized** (it is per-call data, the sibling of the tensor-data cache):

```rust
#[repr(C)]
pub struct FDXSymBinding {
    pub sym_id: u32,    // matches SymId(u32)
    pub _pad: u32,
    pub value: u64,     // SymEnv binds usize; widened to u64 (§6.4 narrowing policy)
}

#[repr(C)]
pub struct FDXSymEnv {
    pub count: u32,
    pub _pad: u32,
    pub bindings: *const FDXSymBinding,  // sorted by sym_id, write-once per pass
}
```

`FDXSymEnv` is the boundary form of `fuel-core-types::symbol::SymEnv`. A consumer resolves a
symbolic axis's live length as `lookup(env, extent.sym_id)`; an **unbound** symbol is a typed
error (matching `SymEnv`'s write-once / presence contract), never a silent 0, and a **bound but
out-of-range** value (`value < min || value > capacity`) is the typed `ExtentOutOfRange` error
(§6.4, V14).

### 7.4 The full Fuel-internal call signature (boundary a)

```c
/* Fuel's own kernel ABI: the extension is an explicit nullable parameter. */
FuelStatus fuel_kernel_launch(
    const DLTensor*    inputs,        size_t n_inputs,
    const FDXSidecar*  in_sidecars,   /* parallel to inputs; entries may be NULL */
    DLTensor*          outputs,       size_t n_outputs,  /* executor pre-allocated */
    const FDXSidecar*  out_sidecars,
    const FDXSymEnv*   env,           /* NULL when no symbolic axis/param */
    const FuelStream*  stream);       /* §11 */
```

Outputs are pre-allocated by the executor (kernels never allocate); the function returns a
status (`Result` on the Rust side), never panics (digest §3). A `NULL` sidecar entry means
"this operand is plain DLPack." On boundary (a), strides in the supplied `DLTensor`s are still
explicit (§3.2) but the 256-byte data rule is relaxed to the backend's `required_alignment`
(§3.3), since Fuel owns both ends.

---

## 8. Validation (build-time / boundary-time, Result-returning)

All checks are `Result`-returning and runnable at the boundary; there are no `try_*` siblings
(P10, digest §3). The reference validator MUST verify, in order:

1. **V1 — header:** `magic == FDX_MAGIC`; `version` is supported (`<= FDX_VERSION_MAX`);
   `struct_bytes >= sizeof(known prefix)`.
2. **V2 — flag/field coherence:** each `FDX_FLAG_HAS_*` set ⇒ the corresponding block is
   non-NONE, and vice-versa.
3. **V3 — honesty (dtype):** if `dtype_ext`/`quant` is meaning-bearing, the base
   `DLTensor.dtype` is the standard byte code (`kDLUInt/8/1`) and `base.shape`/`strides` size the
   *physical bytes*. The base never carries a native DLPack sub-byte dtype (§3.4).
4. **V4 — sub-byte sizing:** `bit_width != 0`; `packing` consistent with `bit_width` (e.g.
   `DENSE_SUBBYTE` ⇒ `bit_width < 8`); physical byte count derivable (never via
   `size_in_bytes()==0`).
5. **V5 — quant coherence (per family; the regimes do not overlap, §6.2):**
   - `GGML_BLOCK` ⇒ `ggml_dtype` valid + scales baked INLINE (`scale_placement == INLINE`,
     `scale_buffer == FDX_BUFFER_INLINE`) + **no `scale_granularity`** (the byte is not consulted;
     it MUST be the default `PerTensor`/0, never `PerBlock`) + **no separate scale operand** + no
     `ScalePair`. A `GGML_BLOCK` descriptor that sets `PerBlock` or `scale_placement ==
     SEPARATE_BUFFER` is rejected (`QuantRegimeViolation`).
   - `MX` ⇒ `scale_dtype == F8E8M0` + `scale_granularity == PerBlock` (MX is the **only** family
     for which `PerBlock` is legal) + `scale_placement == SEPARATE_BUFFER` + block geometry present.
   - `AFFINE_INT`/`AFFINE_FLOAT` ⇒ `scale_granularity ∈ {PerTensor,PerToken,PerChannel}` (never
     `PerBlock`) + `scale_buffer` valid (or BROADCAST).
   - `AFFINE_BLOCK` ⇒ `block_ndim >= 1` + `block_shape` present + `scale_present == 1` +
     `scale_placement == SEPARATE_BUFFER` + `scale_buffer != FDX_BUFFER_INLINE` (a real
     buffer-table index, the named-once block-scale operand) + `scale_granularity != PerBlock`
     (block grain rides `block_shape`, not the granularity byte; `PerBlock` stays MX-only) + no
     `ScalePair` (`role` unspecified). The block-shaped scale buffer's shape is cross-checked
     against `base.shape`/`block_shape` by V6.
   - Any `PerBlock` granularity under a non-`MX` family ⇒ `QuantRegimeViolation`.
6. **V6 — scale shape vs granularity / block geometry:** for `AFFINE_{INT,FLOAT}`,
   `scale_buffer.shape` matches `ScaleGranularity::scale_count(rows, cols)` against the base logical
   shape. For the **block-shaped** families (`MX`, `AFFINE_BLOCK`), `scale_buffer.shape` matches the
   **per-axis block count** derived from the base logical shape and `block_shape` (one scale per
   block: `ceil(dim_a / block_shape[a])` along each tiled axis `block_axes[a]`), not a
   `scale_count`. (`GGML_BLOCK` has no separate scale buffer to check — scales are baked INLINE.)
7. **V7 — extents:** `extents_count ∈ {0, base.ndim}`; each `capacity == base.shape[i]`;
   `min <= capacity`; `cap_kind == 0 (EXPLICIT)` for **every** kind (so a stray nonzero byte at
   the `cap_kind` offset from a mis-versioned blob is caught, not silently ignored — closes the
   cross-version `cap_kind` poisoning). `capacity == base.shape[i]` is also the **example
   self-consistency rule**: every §13 example's `extents[i].capacity` MUST equal that example's
   stated `base.shape[i]` (this is exactly V7), so a copy-error like a capacity that is a multiple
   of the true `base.shape[0]` is caught by V7 rather than mis-sizing a validator downstream. Per
   kind:
   - `Scalar` ⇒ `sym_id == FDX_SYM_NONE`, `min == capacity`, `affine.term_count == 0`.
   - `Range` ⇒ `sym_id != FDX_SYM_NONE`, `affine.term_count == 0`.
   - `Affine` ⇒ `sym_id == FDX_SYM_NONE`, `cap_kind == EXPLICIT (0)` in v1
     (`AFFINE_MAX` ⇒ `UnsupportedVersion`-class until a consumer exists), and **V16**
     well-formedness holds; `FDX_FLAG_HAS_AFFINE_EXTENT` set iff ≥1 axis is `kind=Affine`.
   - Any other `kind` ⇒ typed `UnsupportedVersion`-class error (no guess, §14).
   The same arms apply to `gather.logical_extents[]` (V21d) keyed to `logical_shape` /
   `max_seq_capacity` instead of `base.shape`.
8. **V8 — capacity backing:** for the no-OOB guarantee (§3.1), `buffers[0].size_bytes` must
   cover the **full capacity-shaped** extent (`capacity * stride` along every axis). If it does
   not — `is_mmap_view==1` mapping only the live region, or partial commit — the sidecar MUST set
   `FDX_FLAG_MEANING_REQUIRES_EXT`; absence of the flag with insufficient backing is rejected.
9. **V9 — buffer refs:** every referenced index (`scale_buffer`, `zp_buffer`, view backing) `<
   buffers_count`; index 0 role is `Data`; `byte_offset <= size_bytes` for each; no buffer
   overlaps another except declared bundle sub-views.
10. **V10 — bundle:** `IS_BUNDLE` ⇒ `views_count > 0`, sub-views in-bounds and non-overlapping
    within the bundle buffer.
11. **V11 — explicit strides:** the base `DLTensor.strides` is non-NULL when `ndim != 0`; every
    exported `FDXBufferRef`/`FDXOutputView` carries an explicit length-`ndim` strides array
    (§3.2). Reject NULL strides on any versioned export.
12. **V12 — 256-byte alignment (boundary b only):** `(data as usize) % 256 == 0` for the base
    and every exported buffer; any non-256-aligned logical start is carried in `byte_offset`,
    and `byte_offset <= size_bytes`.
13. **V13 — signed-stride OOB range (negatives FIRST-CLASS; §3.2.1):** strides are `int64` and
    negative strides are **permitted, not rejected** (the early-draft non-negative ban is
    withdrawn, 2026-06-17). For the base and every exported buffer/view, compute the touched byte
    window over the **signed** strides: per axis `i`, `hi_i = (shape[i]-1)*strides[i]` if
    `strides[i] > 0` else `0`, and `lo_i = (shape[i]-1)*strides[i]` if `strides[i] < 0` else `0`;
    the reachable window is `[byte_offset + Σ lo_i, byte_offset + Σ hi_i]` (scaled by the element
    byte width). V13 **requires that window to lie within `[0, size_bytes)`** ⇒ `StrideRangeOutOfBounds`
    otherwise. This is the `Layout::flip` invariant (`byte_offset` at the iteration-first element,
    `start_offset` non-negative), so the §3.1 no-OOB argument holds for negative strides without
    re-derivation. V13 does **not** force non-negative strides; *consumer* inability to take a
    negative stride is an FKC `reverse_strides` capability gap that the **planner** resolves by a
    materialized non-negative copy (`IS_COPIED`), never a blanket FDX rejection (§3.2.1, §9.1, §12).
14. **V14 — realize-time symbol bounds:** for **every** axis, compute the live `value` (Scalar:
    `capacity`; Range: `env.lookup(sym)`; **Affine: evaluate §6.4.2 — V17 runs first**), then
    enforce `extent.min <= value <= extent.capacity` ⇒ `ExtentOutOfRange` otherwise. An unbound
    symbol ⇒ `UnboundSymbol`. On a 32-bit host, a narrowed `value > usize::MAX` ⇒ typed error
    (no truncation). V14 bounds the affine **result** only; a base symbol used elsewhere as an
    extent/offset carries its own bound (§6.4.2 "per-symbol bounds").
15. **V15 — no raw pointers in serialized form:** in a serialized blob, all pointer-typed fields
    are 0 and replaced by offsets; reject a serialized blob with non-zero pointer fields.
16. **V16 — affine well-formedness (build/boundary time)** *(folded from the affine addition,
    2026-06-17)***.** For `kind=Affine`:
    `1 <= term_count <= FDX_AFFINE_MAX_TERMS`; each active term (`i < term_count`) has
    `sym_id != FDX_SYM_NONE`; each inactive slot (`i >= term_count`) is zeroed
    (`sym_id == FDX_SYM_NONE && coeff == 0`); **no duplicate `sym_id`** across active terms;
    **not degenerate** (reject `term_count==1 && c0==0 && coeff==1` ⇒ must be `Range`; reject the
    all-constant `term_count==0` form ⇒ must be `Scalar`); `cap_kind ∈ {EXPLICIT}` in v1. Failure
    ⇒ `AffineMalformed` / `AffineTooManyTerms` / `AffineDegenerate`.
    **Negative-coefficient compositionality (limit of mechanical enforcement).** Negative coeffs
    are type-allowed (i64; e.g. `capacity - cached_len`, §17.3c). V14/V17 bound only the affine
    **result** `[min, capacity]`; they do **not** prove each term's referenced sym stays within its
    *own* range. §6.4.2 calls this out as a producer responsibility. V16 mechanically enforces it
    **only when the term's `sym_id` also appears as a `Range`/`Affine` extent in the SAME sidecar**
    (then V16 cross-references that extent's own bound by `sym_id`); a sym bound *outside* the
    sidecar (a `DynScalar` carried only in the `SymEnv`) cannot be cross-checked here and its
    per-term bound is the producer's responsibility, enforced wherever that sym carries its own
    extent (§6.4.2). To keep the no-OOB property *fully* mechanical for negative coeffs in v1, a
    producer SHOULD reference, for every active negative-coeff term, a sym that also carries its own
    in-sidecar extent; the residual (a guaranteed-in-sidecar bound for every affine term sym, vs the
    producer-responsibility split) is tracked as §17.3c.
17. **V17 — affine evaluation safety (realize time, runs BEFORE V14).** The i128 accumulation
    (§6.4.2) widens **both** operands to i128 before each multiply (`(coeff as i128).checked_mul(s
    as i128)`, since a u64 sym `> i64::MAX` would mis-cast negative if multiplied in i64) and uses
    `checked_mul`/`checked_add` at **every** step; any overflow ⇒ `AffineOverflow` (i128 is NOT
    unconditionally safe for 4 terms — a single final check is insufficient). V17 then gates the
    final value `>= 0` (a negative i128 is rejected here as `ExtentOutOfRange` *before* narrowing —
    it cannot narrow to `u64`/`usize`, so V14 never compares a negative value against `min`); the
    host narrowing succeeds (32-bit `value > usize::MAX` ⇒ `ExtentOutOfRange`, no truncation).
18. **V18 — gather coherence** *(V18–V21 folded from the gather addition, 2026-06-17)***.**
    `FDX_FLAG_HAS_GATHER` set ⇔ `gather.kind != FDX_GATHER_NONE`.
    `kind == FDX_GATHER_PAGED_BLOCKS` ⇒ `block_size != 0`, `num_blocks != 0`, `num_sequences != 0`,
    `max_blocks_per_seq != 0`, `id_dtype == U32` (v1 pin), `physical_ndim ∈ [1,6]`,
    `logical_ndim ∈ [0,6]`, `physical_shape[0] == num_blocks`, `physical_shape[1] == block_size`,
    and `max_seq_capacity == max_blocks_per_seq * block_size` computed in **u64 without overflow**.
    `max_seq_capacity` is **defined as that product** (the per-seq capacity), so it is always an
    exact multiple of `block_size` and `max_blocks_per_seq == max_seq_capacity / block_size` is an
    exact division (no `ceil` ambiguity — v1 does not express ragged, non-multiple per-seq
    capacities; that is a `kind >= 2` future, §6.9.3 / §17 item 12). The author-side narrowings
    (`usize` source → the struct's `u64`/`u32` fields) are guarded: a source value exceeding the
    field width ⇒ `GatherIncoherent`, never a wrap. The `unmapped_sentinel` MUST be representable in
    `id_dtype` **and** `>= num_blocks` (so the single `id >= num_blocks` range check provably
    catches both OOB and unmapped). Unknown `kind` ⇒ `UnsupportedGatherKind`. → `GatherIncoherent`.
19. **V19 — MEANING_REQUIRES_EXT mandatory + base honesty.** `FDX_FLAG_HAS_GATHER` set ⇒
    `FDX_FLAG_MEANING_REQUIRES_EXT` set (a paged pool's logical tensor cannot be reconstructed
    from the base bytes). Absence ⇒ `DishonestBase` (sub-reason `GatherWithoutMeaningFlag`). The
    base `DLTensor` honesty (V3) still holds: base `dtype == {kDLUInt,8,1}`, base `strides == [1]`,
    and the base byte length **exactly equals the pool allocation walk** (the same definition of
    "pool byte length" V20 uses, so the two never disagree):
    `base.shape[0] == physical_strides[0] * num_blocks * elem_bytes(element_dtype)` — i.e. the
    `stride·extent` of the slowest physical axis. For a **dense, gap-free** pool this reduces to
    `num_blocks * block_size * intra_block_typed_count * elem_bytes` (where
    `intra_block_typed_count = product(physical_shape[2..physical_ndim])` and `physical_strides[0]
    == block_size * intra_block_typed_count`); for a **padded** pool (`physical_strides[0]` larger
    than the dense per-block stride) the strided form is larger, and the dense element-count product
    is NOT used — V19 and V20 share the one strided definition so a legitimately padded pool passes
    both. The honest-`uint8` cover is thus *enforced*, not merely asserted in prose. → `DishonestBase`.
20. **V20 — pool backing (stride·extent, not element-count).** Mirrors V8 literally for the pool:
    `buffers[pool_buffer].size_bytes >= physical_strides[0] * num_blocks * elem_bytes` **and** the
    analogous `stride * extent` on **every** physical axis (so a padded pool — `physical_strides[0]`
    larger than the dense per-block stride — is sized for its padding and block id `num_blocks-1`
    cannot walk past `size_bytes`). `physical_strides` MUST be the honest, gap-free strides of the
    actual allocation, the block axis being `physical_strides[0]`. A pool not backed to full
    capacity already sets `MEANING_REQUIRES_EXT` (V19). → `CapacityNotBacked`
    (sub-reason `PoolNotBacked`).
21. **V21 — gather ↔ operands ↔ symbol consistency (build + realize).**
    (a) `pool_buffer`, `block_table.table_buffer`, and `context_lens_buffer` (when not
    `FDX_BUFFER_NONE`) are valid buffer-table indices (`< buffers_count`) with matching declared
    roles; their shapes match the gather geometry (`block_table` is
    `[num_sequences, max_blocks_per_seq]`; `context_lens` is `[num_sequences]`).
    (b) When the FDX tensor coexists with the FKC operands carrying the same tables — the
    `KernelRef::PagedAttn` operand order `[q, k_cache, v_cache, block_table, context_lens, alibi?]`
    (`fuel-dispatch/src/kernel.rs:314-331`) — they MUST reference the **same buffer-table slot**,
    not two slots with equal contents. The identity predicate is concrete: **index-equality** of
    the buffer-table slot within one sidecar (the natural in-FDX form); for the live FKC cross-check
    where the operand and the gather descriptor are separate handles, identity is the tuple
    `(device, base data pointer, byte_offset, size_bytes)` of the underlying allocation. It is
    **never** a value/content comparison (that could silently pass on a copy) and **never**
    pointer-equality of the serialized `FDXBufferRef.data` (which is always 0, V15). This is the
    single-place rule (§6.9.3), a cross-check not a copy.
    (c) **Build / boundary FULL-table scan** (V18-class): every **mapped** entry
    (`id != unmapped_sentinel`) of the block-table buffer satisfies `0 <= id < num_blocks`.
    (d) **Realize-time LAZY per-ACCESSED slot** (matching the as-built kernel
    `byte_kernels.rs:6877,6900-6912`, NOT an eager pre-pass): for each sequence `s`, the live
    length `L` (from `context_len_sym` via `FDXSymEnv`, or `context_lens[s]`) satisfies
    **`0 <= L <= max_seq_capacity`** and `ceil(L/block_size) <= max_blocks_per_seq`. **`L == 0` is
    LEGAL** — a finished / evicted / not-yet-started sequence in a batched decode; the canonical
    handling is the as-built kernel's `if ctx_len == 0 { continue; }` (`byte_kernels.rs:6876`),
    which skips that sequence. The seq-axis live extent's lower bound is therefore **`min = 0`** for
    the gather case (V21e), distinguishing it from the dense single-`Range` KV of §13.4 where
    `min = 1` is right because there is one shared live length, not a per-seq vector. **Column-index
    guard:** the block-table dereference `block_table.ids[s*max_blocks_per_seq + (t/block_size)]`
    for `t < L` reads column `t/block_size < ceil(L/block_size) <= max_blocks_per_seq` — its
    in-bounds-ness rests on the `ceil(L/block_size) <= max_blocks_per_seq` check *above*, which a
    consumer MUST run **before** the dereference, not only the `id >= num_blocks` test after. At the
    moment each block id is dereferenced, `id >= num_blocks` ⇒ `BlockIdOutOfRange` (this single test
    catches both OOB and the unmapped sentinel, since V18 forces `unmapped_sentinel >= num_blocks`).
    **Runtime-`usize` narrowing (mirrors V17/§6.4.2):** on a host where `usize < u64`, the flat
    block-table index `s*max_blocks_per_seq + (t/block_size)`, the byte address
    `physical_block*physical_strides[0]*elem_bytes + …` (§6.9.4 step 3), and `max_seq_capacity`
    derivations are computed in **u64 / checked** and any `> usize::MAX` ⇒ `GatherAddressOverflow`
    — these are re-checked against the *runtime* `usize`, not only the author-side u64 fields (V18),
    so a large pool / long context cannot silently wrap to an in-bounds-looking but wrong address.
    **Data-determined per-seq case:** when `context_len_sym == FDX_SYM_NONE` (per-seq lengths differ,
    read from `context_lens_buffer`) there is no single sym for V14 to bound, so the consumer MUST
    validate `0 <= context_lens[s] <= max_seq_capacity` and
    `ceil(context_lens[s]/block_size) <= max_blocks_per_seq` **per sequence at the point each
    sequence is first accessed** (matching `byte_kernels.rs:6877`) — this is inherently per-seq, not
    a single env-resolved value, and is still lazy (it runs as each sequence is touched, not an
    eager pre-pass).
    (e) `context_len_sym != FDX_SYM_NONE` ⇒ `logical_extents_count == logical_ndim`, and
    `logical_extents[seq_axis]` carries the **same** `sym_id` (P4 unification) with
    `logical_extents[seq_axis].capacity == max_seq_capacity` and **`min == 0`** (empty sequences are
    legal — see (d); the seq-axis `Range`/`Affine` lower bound is 0 for a batched paged cache, so
    its own V14 arm uses `0 <= value <= max_seq_capacity` and never spuriously rejects `L == 0`).
    Its OWN V7/V14/V16/V17 arms run, keyed to `logical_shape`/`max_seq_capacity` — the live seq
    length is thus inside the extents machinery, NOT escaping it via the 1-D base `extents[]`.
    → `GatherIncoherent` / `BufferRefOutOfRange` / `ExtentOutOfRange` / `BlockIdOutOfRange` /
    `GatherAddressOverflow`.

Any failure → a typed `FDXError` (`UnsupportedVersion`, `DishonestBase`, `BadSubByte`,
`QuantIncoherent`, `ScaleShapeMismatch`, `ExtentMismatch`, `ExtentOutOfRange`,
`CapacityNotBacked`, `BufferRefOutOfRange`, `BundleOverlap`, `NullStrides`, `Misaligned`,
`StrideRangeOutOfBounds`, `PointerInSerializedForm`, `UnboundSymbol`, `AffineMalformed`,
`AffineTooManyTerms`, `AffineDegenerate`, `AffineOverflow`, `GatherIncoherent`,
`UnsupportedGatherKind`, `BlockIdOutOfRange`, `GatherAddressOverflow`; with `PoolNotBacked` a
sub-reason of `CapacityNotBacked` and `GatherWithoutMeaningFlag` a sub-reason of `DishonestBase`),
never a panic, never a silent fix-up.

> **Description-only re-affirmation (P3/G7).** None of V16–V21 priced anything or chose a path.
> The *cost* of materializing a dense un-paged copy for a blind consumer, the *decision* of
> paged-kernel-vs-materialize, and the contiguize/strided/materialize choice are the **FKC** cost
> model's job (§6.5/§9.3). The paged-attention kernel declares its acceptance of a paged cache in
> FKC; FDX only *describes* the tensor.

---

## 9. Producer / consumer policies

### 9.1 Producer policy — refuse-or-materialize (no silent degradation)

> A tensor whose meaning depends on the extension MUST be dequantized/materialized or refused
> when handed to an extension-blind consumer.

Concretely, a producer about to export across a boundary (b) to a consumer that did *not*
advertise FDX support (§12):

- If `FDX_FLAG_MEANING_REQUIRES_EXT` is **clear** (the base bytes are a faithful, fully-backed
  standard tensor — dense F16/F32, or a symbolic axis whose capacity-tail is harmless *and*
  physically backed): export the standard `DLTensor` as-is (explicit strides, 256-aligned data);
  the sidecar is simply not carried. For a symbolic axis where the *live prefix* is the intended
  content (KV cache), export a **materialized dense copy** of the resolved-length live region as
  a standard dense tensor (§3.1, §3.1.1) — **and set `DLPACK_FLAG_BITMASK_IS_COPIED`** on the
  managed wrapper.
- If `FDX_FLAG_MEANING_REQUIRES_EXT` is **set** (quant / sub-byte / meaning-bearing, or a
  symbolic capacity tail that is unsafe/unbacked): the producer MUST either (i)
  **dequantize/materialize** to a standard dtype/dense layout and export that **with
  `DLPACK_FLAG_BITMASK_IS_COPIED` set**, or (ii) **refuse** with a typed error naming the
  unrepresentable property. It MUST NOT export the raw bytes labeled as anything other than the
  honest `uint8` (which the consumer would misread). Silent degradation is banned.

**`IS_COPIED` rule (load-bearing).** Whenever the producer materializes for export — a dequant,
a live-prefix dense copy, an aligned copy to satisfy §3.3, or any copy made by the producer —
it **MUST** set `DLPACK_FLAG_BITMASK_IS_COPIED` on the `DLManagedTensorVersioned.flags`.
`dlpack.h`: a copied tensor "is considered solely owned throughout its lifetime by the consumer,
until the producer-provided deleter is invoked." Exporting a materialized copy *without*
`IS_COPIED` mislabels ownership/freshness; FDX forbids it. (FDX does not need a mirroring
internal flag — the standard DLPack flag is the contract on boundary b.)

The choice between dequantize/materialize and refuse is the producer's policy knob (a Fuel-side
configuration), not a property of FDX; FDX only *enables* the honest decision by flagging
meaning-dependence. The *cost* of the materialize alternative is the FKC cost model's concern
(§9.3).

### 9.2 Consumer policy

- A consumer that does not recognize `magic`/`version` MUST treat the sidecar as **absent** and
  fall back to pure DLPack (it then sees the honest base, §3). It MUST NOT guess.
- A consumer that recognizes the version but not a *set flag bit it does not understand* MUST
  NOT proceed as if the tensor were standard if `FDX_FLAG_MEANING_REQUIRES_EXT` is set; it
  errors (typed), per digest §3 "errors at dispatch, never silently materializes."
- A consumer MUST NOT silently dequantize/contiguize on read; layout/quant fixups are the
  planner's job (it inserts explicit ops). The consumer either handles the described layout or
  errors.
- A consumer reading a symbolic axis MUST resolve the live length from the supplied `FDXSymEnv`
  and apply the realize-time bound check (§6.4 / V14); it MUST NOT assume capacity == live
  unless the axis is `Scalar`.

### 9.3 Planner policy (the decision-maker) — and the FDX↔FKC cost split

Per the governing principle, the *planner* — not the producer or consumer — chooses the
extended path vs the dequant-to-standard path, from the `BackendProbe` capability advertisement
(§12), not by trial. **FDX carries description only (no cost).** The cost of each alternative —
contiguize, keep strided, dequantize/materialize — is supplied by the **kernel-contract (FKC)**
cost model, keyed on the same shared codes (§6.0). The planner reads FDX (*what the tensor is*)
+ FKC (*what each handling alternative costs on the target backend*) + the probe (*what the
backend understands*) and commits a path. This is the explicit home of the
"contiguize-vs-strided-vs-materialize via declared cost" decision the critique flagged: it lives
in FKC, not FDX, by the description/cost separation (P3).

- **Negative-stride normalization is one of these planner decisions (§3.2.1).** FDX *describes* a
  negative-strided operand (a flip/reverse view); the planner checks the chosen consumer's FKC
  `reverse_strides` capability and either keeps the zero-copy view (capable internal kernel) or
  inserts an explicit normalize-to-non-negative op (`Op::Flip`/contiguize, `IS_COPIED`, priced by
  FKC) when the consumer cannot take it or for a bare external handoff. It is never a universal
  normalization and never a boundary trial — exactly the contiguize-vs-view split above, applied to
  stride sign.

---

## 10. The two boundaries

### 10.1 Boundary (a): Fuel's own kernel ABI — explicit nullable parameter

Inside Fuel, the sidecar is an explicit `*const FDXSidecar` argument next to each `DLTensor`
(§7.4), `null` when absent. No smuggling, no managed wrapper needed for the common internal
launch: **Fuel owns the memory, so the bare `DLTensor` POD + sidecar POD suffice** (no
deleter). The `FDXSymEnv` rides alongside for symbolic resolution. This is the high-frequency,
zero-overhead path (one recorded run replayed with rebased operands + a fresh `SymEnv`).
Strides are explicit (§3.2); the 256-byte data rule relaxes to `required_alignment` (§3.3).

### 10.2 Boundary (b): external `__dlpack__` interchange — via `manager_ctx` or not at all

The Python `__dlpack__` capsule signature is fixed: it yields a `PyCapsule` named
`"dltensor_versioned"` (versioned) wrapping a `DLManagedTensorVersioned`. There is **no slot for
an extra pointer.** Therefore:

- **If the consumer advertised FDX support** (via the negotiation in §12), the producer attaches
  the sidecar through `DLManagedTensorVersioned.manager_ctx`: `manager_ctx` points to a
  Fuel-owned struct `FDXManagerCtx { u32 magic; FDXSidecar* sidecar; /* + the real manager
  context */ }`, and Fuel's own `deleter` frees both.
- **Recovery is deleter-identity-gated, NOT magic-gated.** A Fuel-aware consumer recovers the
  sidecar **only when both** hold: (1) the consumer advertised FDX (§12) **and** (2) the live
  `DLManagedTensorVersioned.deleter` function pointer **is identical to Fuel's own exported
  deleter** (`fuel_fdx_deleter`). Only then does it downcast `manager_ctx` to `FDXManagerCtx`
  and re-check the `magic` word as a second-line guard. **The magic word alone is never
  sufficient.** Rationale: the DLPack documentation does **not** restrict what a consumer may do
  with `manager_ctx`, and a non-Fuel library may legally re-export/forward/wrap the capsule.
  After such forwarding the `deleter` is the third party's, not Fuel's, and `manager_ctx` is
  whatever *they* put there — so a magic-word collision in their first word would otherwise cause
  an unsound downcast and a double-free / deleter mismatch.
- **On any deleter-identity mismatch, the consumer treats the tensor as plain DLPack** (sidecar
  absent), regardless of the magic word. This is the safe fallback for a forwarded/wrapped
  third-party capsule.
- **If the consumer did not advertise FDX support**, the sidecar is **not carried.** Only the
  standard `DLManagedTensorVersioned` crosses, governed by producer policy §9.1 (faithful base
  exported as-is, or dequantize/materialize/refuse for meaning-bearing tensors, with `IS_COPIED`
  set on any materialized copy). Riding `manager_ctx` is *only* attempted with a consenting,
  same-deleter peer because the deleter contract owns `manager_ctx` lifetime; we never assume a
  generic consumer will run *our* deleter logic on it.

> **Opacity is a convention, not a guarantee.** The v0.1 draft claimed a generic consumer is
> "contractually required" to treat `manager_ctx` as opaque. It is not — the DLPack spec is
> silent on consumer restrictions. FDX therefore relies on the *deleter identity*, not on an
> assumed opacity, for soundness.

> Rationale for two mechanisms: boundary (a) is the hot internal path and pays nothing; boundary
> (b) honors DLPack's fixed ABI and its ownership contract, degrading honestly when the peer is
> generic or when the capsule has been forwarded.

---

## 11. Ownership, lifetime, alignment, and stream semantics (cross-runtime)

- **Internal launches (boundary a):** bare POD; Fuel owns all memory; no deleter. The sidecar's
  live `data`/pointer fields point into Fuel-owned storage and are valid for the call. Strides
  explicit; 256-byte rule relaxed to `required_alignment`.
- **Cross-runtime (boundary b):** use `DLManagedTensorVersioned`'s managed/deleter contract.
  Ownership transfers per DLPack: the consumer, after consuming, calls `deleter(self)` exactly
  once; the producer's deleter releases both the tensor memory it owns and the
  `manager_ctx`-attached sidecar (when carried). The capsule rename dance
  (`"dltensor_versioned"` → `"used_dltensor_versioned"`) prevents double-free, unchanged by FDX.
- **Alignment (boundary b):** the exported `data` is 256-byte aligned and the logical start
  rides `byte_offset` (§3.3). A buffer that cannot meet this is exported as an aligned
  materialized copy (with `IS_COPIED`, below).
- **Freshness / copy (`IS_COPIED`):** any producer-materialized export (dequant, live-prefix
  dense copy, aligned copy) sets `DLPACK_FLAG_BITMASK_IS_COPIED` (§9.1). A non-copied export
  shares the producer's live buffer; a copied export is solely the consumer's until the deleter
  runs.
- **Read-only:** mirror `DLPACK_FLAG_BITMASK_READ_ONLY` into `FDX_FLAG_READ_ONLY`; a read-only
  tensor's buffers MUST NOT be written by the consumer.
- **Stream exchange (cross-runtime, async devices):** FDX adds **no new stream field** to the
  serialized sidecar. On boundary (b) the standard DLPack/array-API stream protocol governs:
  the consumer passes its stream to `__dlpack__(stream=...)`, and the producer **enqueues an
  ordering event so all writes to the tensor are ordered-before that stream before returning the
  capsule** (a `cudaStreamWaitEvent`-equivalent / Vulkan timeline-semaphore wait). The
  default-stream / no-sync handling is **pinned per device** (do not paraphrase as "honor stream
  semantics"):

  | `stream` argument | meaning (per array-API DLPack stream protocol) |
  |---|---|
  | `None` | producer may **skip** synchronization (consumer takes responsibility) |
  | `kDLCUDA` / `kDLCUDAHost`: integer `1` | the **legacy default** stream — producer syncs onto it |
  | `kDLCUDA` / `kDLCUDAHost`: integer `2` | the **per-thread default** stream — producer syncs onto it |
  | `kDLCUDA` / `kDLCUDAHost`: any other integer | an explicit stream handle — producer syncs onto it |
  | `kDLROCM`: `0`/`1`/`2` analogous to CUDA | per the array-API ROCm sentinels |
  | other device types | per that device's array-API stream entry; if unspecified, treat like an explicit handle (sync) |

  A missing sync where a non-`None` stream was supplied is a **correctness race**, not a perf
  issue; the producer MUST insert the ordering event before handing back the capsule.
  Boundary (a) uses the call-time `FuelStream*` (§7.4); symbolic resolution (`FDXSymEnv`) is
  likewise call-time, never persisted — operand rebasing for recorded-run replay re-supplies
  stream + env per replay while reusing the same sidecar.

---

## 12. Capability negotiation (via `BackendProbe` / `Capability`)

Consumers advertise "understands FDX vN" through Fuel's existing capability machinery; the
planner selects extended vs dequant-to-standard from the probe, **not by trial** (digest §2,
§11).

- **Extend the `Capability` enum** (`fuel-core-types::capability::Capability`, already
  `#[non_exhaustive]`) with FDX capability tokens, e.g.:
  - `DlpackExtV1` — understands the FDX v1 sidecar at all;
  - `DlpackExtMx` — can consume MX block-scaled tensors directly;
  - `DlpackExtGgml` — can consume GGML-family quant directly (paired with the existing per-format
    `MatMulQ4_0` / `MatMulQ4KM` / … tokens for the *op* that consumes them);
  - `DlpackExtAffine` — can consume the affine *quant* families: `AFFINE_INT`/`AFFINE_FLOAT`
    (dynamic, per-tensor/token/channel) **and** `AFFINE_BLOCK` (nf4/QLoRA — low-bit data + a
    separate block-shaped scale operand, §6.2). Distinct from affine *extents*, which ride under
    `DlpackExtSymbolic`;
  - `DlpackExtSymbolic` — honors symbolic live-vs-capacity extents + `FDXSymEnv`, **including the
    `kind=Affine` extent** (evaluates `c0 + Σ cᵢ·symᵢ` through the env, §6.4.2);
  - `DlpackExtGather` — can consume a paged/blocked FDX tensor (`FDX_FLAG_HAS_GATHER`,
    `FDXIndexedResidency`, §6.9) directly; a backend without it triggers producer policy
    (materialize a dense un-paged copy or refuse, §9.1).
- **Negative-stride acceptance is a per-kernel layout capability, NOT an FDX token** (§3.2.1).
  Whether a consumer can walk a negative-strided operand is the **kernel-contract (FKC)** layout
  capability `reverse_strides`, declared per kernel — *not* a backend-wide `Capability` token —
  because it varies kernel-by-kernel (a contiguous-only kernel cannot, a generic strided kernel
  can). FDX describes the strides (int64, §3.2.1); the **planner** reads each candidate kernel's
  `reverse_strides` declaration and, only when the chosen consumer lacks it (or for a bare external
  handoff), inserts a normalize-to-non-negative materialized copy (`Op::Flip`/contiguize with
  `IS_COPIED`, §9.1). Internal zero-copy `Op::Flip` between `reverse_strides`-capable kernels is
  preserved.
- **Surface in `BackendCapabilities`:** add the FDX tokens to the backend's advertised set (the
  same `op_dtype_support`/capability surface a backend declares at registration via
  `BackendCapabilityProvider`). A backend that does not list `DlpackExtV1` is treated as
  extension-blind for producer policy (§9.1).
- **Negotiation lives in the planner, not the boundary.** The planner reads the destination
  backend's advertised FDX tokens (from the probe/registration) and decides, per tensor: ride
  the sidecar (extended path), or insert an explicit `Op::Dequantize`/`Op::Contiguize` to
  produce a standard tensor the blind consumer can take. The boundary never trials-and-falls-
  back.
- **Version negotiation:** a backend advertising `DlpackExtV1` but not `DlpackExtV2` is handed a
  v1 sidecar (a v2 producer downgrades by emitting only v1-expressible facts, or dequantizes a
  v2-only feature). Newer-producer-to-older-consumer is the explicit forward-compat requirement
  (G4). Note this is the **FDX schema** version axis, independent of the DLPack ABI version on
  the capsule (§5.2).

---

## 13. Worked examples

In each, "base" = the standard `DLTensor`; "sidecar" = `FDXSidecar` (or absent). **All strides
are explicit** (§3.2); `data` is 256-byte aligned on any boundary-(b) export (§3.3).

### 13.1 Dense F16, null sidecar (pure DLPack)

- base: `data=ptr` (256-aligned), `device={kDLCUDA,0}`, `ndim=2`, `dtype={kDLFloat,16,1}`,
  `shape=[4096,4096]`, `strides=[4096,1]` (explicit C-contiguous — **not** NULL), `byte_offset=0`.
- sidecar: **NULL** (`*const FDXSidecar == null`).
- Interop: 100% standard and v1.2+-conformant; PyTorch/JAX/CuPy consume it directly. Nothing
  Fuel-specific.

### 13.2 GGUF/GGML Q4_K_M block-quant weight

A `[out=11008, in=4096]` weight in GGML `Q4K` (block_size 256, type_size 144B, scales baked
INLINE).

- base (honesty): `dtype={kDLUInt,8,1}`, `ndim=1`, `shape=[ n_bytes ]` where `n_bytes =
  (11008*4096/256)*144`, `strides=[1]` (explicit), `byte_offset=0`, `data` 256-aligned. A
  sidecar-blind consumer sees honest opaque bytes of the exact packed size.
- sidecar: `flags = HAS_DTYPE_EXT | HAS_QUANT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `packing=GGML_BLOCK`, `bit_width=4` nominal, `sub_byte_bit_order` per ggml.
  - `quant`: `family=GGML_BLOCK`, `ggml_dtype=12 (Q4K)`, `block_ndim=1`, `block_shape=[256]`,
    `block_axes=[1]` (along K=in), `scale_present=1`, `scale_placement=INLINE`,
    `scale_buffer=FDX_BUFFER_INLINE`, `pack_order=GGML_NATIVE`, no `ScalePair`.
  - `buffers`: index 0 = the packed data buffer (`size_bytes = n_bytes`, `strides=[1]`); scales
    are inline.
- Dispatch: planner keys on `(MatMul, GGML Q4K)` → the flat `Capability::MatMulQ4KM` token; the
  kernel owns the dequant. No silent dequant-on-read elsewhere. To a blind consumer:
  dequantize-and-export-with-`IS_COPIED`, or refuse (§9.1).

### 13.3 Sub-byte FP4 (MX4) activation tensor

A `[batch=8, hidden=4096]` tensor in F4, packed two-per-byte, dense (no block scale here — pure
F4 storage; the MX scale case is §13.5).

- base (honesty): `dtype={kDLUInt,8,1}`, `ndim=1`, `shape=[ 8 * 4096/2 ]` bytes, `strides=[1]`
  (explicit), 256-aligned `data`.
- sidecar: `flags = HAS_DTYPE_EXT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `logical_dtype=13 (F4)`, `bit_width=4`, `packing=DENSE_SUBBYTE`, `lanes=1`,
    `sub_byte_bit_order=0` (element 0 in low nibble).
  - `quant`: `family=NONE` (no scales — plain sub-byte storage).
  - `extents`: 0 (both axes concrete) — the **base byte shape** is `[8*2048]`; the logical-element
    shape is recovered from base byte shape × `bit_width` (or, optionally, an aux buffer ref; v1
    keeps it implicit — §17).
- Note the load-bearing point: `DType::size_in_bytes()` for F4 is **0**; the buffer is sized
  from `bit_width=4`, never that 0. The native DLPack sub-byte path is *not* used (§3.4).

### 13.4 KV-cache tensor with live < capacity (symbolic extent)

A per-layer K cache `[n_heads=32, K_capacity=4096, head_dim=128]` in F16, with live length
`k_len = SymId(7) ∈ [1, 4096]`, backed by a dense, fully-committed allocation.

- base: `dtype={kDLFloat,16,1}`, `ndim=3`, `shape=[32, 4096, 128]` (**capacity**),
  `strides=[4096*128, 128, 1]` (**explicit, keyed to capacity** — honest, the buffer truly has
  4096 slots; these happen to be positive, but negatives would be equally valid per §3.2.1/V13),
  `data` 256-aligned.
- sidecar: `flags = HAS_SYMBOLIC` (note: `MEANING_REQUIRES_EXT` *clear* — the base is a faithful
  F16 tensor and the allocation backs the full capacity, so reading the tail is harmless; per V8
  the dense backing justifies the no-OOB claim).
  - `extents` (count=3):
    - axis 0: `Scalar`, capacity=32, `sym_id=NONE`.
    - axis 1: `Range`, `min=1`, `capacity=4096`, `sym_id=7`, `sym_scope=InputDetermined`.
    - axis 2: `Scalar`, capacity=128, `sym_id=NONE`.
  - `storage`: `class=Session`, `session_id=<this session>` (KV cache is session-state, the
    durable-interop unit).
  - `residency`: `tier=Device`, `substrate=CudaUntyped`, `backend_id=Cuda`, `device_index=0`,
    `is_mmap_view=0`.
  - `buffers`: index 0, `size_bytes = 32*4096*128*2` (full capacity backing → V8 satisfied).
- Realize: the call passes `FDXSymEnv { (7 → live_k_len) }`; a flash-attention kernel reads
  `k_len = lookup(env, 7)`, applies the realize-time bound `1 <= k_len <= 4096` (V14), and walks
  the live prefix using base stride + live extent. The same sidecar serves every token (operand
  rebasing: re-supply data ptr + env, reuse description).
- Cross-runtime export to a generic consumer: because the live region on the **middle** axis is
  non-contiguous (§3.1.1), the producer exports a **materialized dense copy** `[32, k_len, 128]`
  as a standard dense F16 tensor with **`DLPACK_FLAG_BITMASK_IS_COPIED` set** (§9.1), never the
  raw capacity buffer and never a "slice" mislabeled as dense.

> Variant — mmap-only-live KV: if the K cache were an mmap view mapping only the live region
> (`is_mmap_view=1`, `size_bytes` < full capacity), V8 forces `MEANING_REQUIRES_EXT` to be set,
> and the blind-consumer export goes through refuse-or-materialize (§9.1).

### 13.5 MX / microscaling tensor (packed sub-byte payload + F8E8M0 per-block scale)

A `[out=4096, in=4096]` weight in MX4: F4 payload + one F8E8M0 scale per 32-element block along
K.

- base (honesty): `dtype={kDLUInt,8,1}`, `ndim=1`, `shape=[ 4096 * 4096/2 ]` bytes (the F4
  payload only), `strides=[1]` (explicit), 256-aligned `data`.
- sidecar: `flags = HAS_DTYPE_EXT | HAS_QUANT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `logical_dtype=13 (F4)`, `bit_width=4`, `packing=MX_BLOCK`.
  - `quant`: `family=MX`, `block_ndim=1`, `block_shape=[32]`, `block_axes=[1]` (along K),
    `scale_present=1`, `scale_dtype=14 (F8E8M0)`, `scale_placement=SEPARATE_BUFFER`,
    `scale_granularity=PerBlock`, `scale_buffer=1`.
  - `buffers`: index 0 = F4 payload (= base data, `strides=[1]`); index 1 = F8E8M0 scale buffer,
    role=Scale, `dtype=F8E8M0`, `ndim=2`, `shape=[4096, 4096/32]`, explicit strides
    `[128, 1]` (one scale per block), 256-aligned.
- Validation: family=MX ⇒ `scale_dtype==F8E8M0` + `PerBlock` + block geometry present (V5).
  Dispatch: planner needs an MX-aware consumer (`Capability::DlpackExtMx`); else
  dequantize-to-standard (with `IS_COPIED`) or refuse (§9.1).

### 13.5a AFFINE_BLOCK weight (nf4/QLoRA — NF4 data + separate per-block absmax scale)

A `[out=11008, in=4096]` weight in **NF4** (QLoRA): a 4-bit data payload plus a **separate**
per-block absmax scale operand, block size 64 along the flattened weight. This is the
block-grained affine regime — distinct from MX (no F8E8M0 / no `PerBlock`) and from GGML
(scales are a separate operand, not baked).

- base (honesty): `dtype={kDLUInt,8,1}`, `ndim=1`, `shape=[ 11008*4096/2 ]` bytes (the NF4
  payload only, two nibbles/byte), `strides=[1]` (explicit), 256-aligned `data`. A sidecar-blind
  consumer sees honest opaque bytes of the exact packed size.
- sidecar: `flags = HAS_DTYPE_EXT | HAS_QUANT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `logical_dtype=13 (F4)` (NF4 rides the F4 sub-byte code; the NF4 codebook is the
    kernel's), `bit_width=4`, `packing=DENSE_SUBBYTE`, `sub_byte_bit_order` per format.
  - `quant`: `family=AFFINE_BLOCK (4)`, `block_ndim=1`, `block_shape=[64]`, `block_axes=[0]`
    (along the flattened `out*in`), `scale_present=1`, `scale_dtype=8 (F32)`,
    `scale_placement=SEPARATE_BUFFER`, `scale_buffer=1` (a real index — **named once**, NOT
    `FDX_BUFFER_INLINE`), `scale_granularity` left at default `PerTensor` and **not consulted**
    (block grain rides `block_shape`; `PerBlock` stays MX-only), `zp_present=0` (NF4 is
    symmetric), no `ScalePair` (`role` unspecified).
  - `buffers`: index 0 = NF4 payload (= base data, `strides=[1]`); index 1 = the absmax block-scale
    buffer, role=Scale, `dtype=F32`, `ndim=1`, `shape=[ 11008*4096/64 ]` (one scale per 64-element
    block), `strides=[1]`, 256-aligned. The scale operand is a separate graph input named exactly
    once by this buffer-table entry (single-place rule).
- Validation: family=AFFINE_BLOCK ⇒ `block_ndim>=1` + `block_shape` present + `scale_present==1` +
  `scale_placement==SEPARATE_BUFFER` + `scale_buffer != FDX_BUFFER_INLINE` + `scale_granularity !=
  PerBlock` + no `ScalePair` (V5); the scale buffer shape `[ 11008*4096/64 ]` is cross-checked
  against `base.shape`/`block_shape` (V6). Dispatch: planner needs an AFFINE-aware consumer
  (`Capability::DlpackExtAffine`); else dequantize-to-standard (with `IS_COPIED`) or refuse (§9.1).

### 13.6 Multi-output bundle (softmax + argmax)

A node producing F32 `y [B,V]` and I64 `argmax_idx [B]` in one allocation.

- base: `dtype={kDLUInt,8,1}`, `ndim=1`, `shape=[ total_bytes ]` (the whole bundle),
  `strides=[1]` (explicit), 256-aligned `data`.
- sidecar: `flags = IS_BUNDLE`.
  - `views` (count=2):
    - view 0: `byte_offset=0`, `dtype=F32`, `ndim=2`, `shape=[B,V]`, `strides=[V,1]` (explicit),
      name_hash("y").
    - view 1: `byte_offset = B*V*4`, `dtype=I64`, `ndim=1`, `shape=[B]`, `strides=[1]`,
      name_hash("argmax_idx").
  - `buffers`: index 0 = the bundle backing buffer (`size_bytes = total_bytes`).
- The kernel emits one `KernelRef` writing both slots by offset; `Op::View`/`Op::ViewOwned`
  resolve slots back to ordinary tensors. Each slot is independent in dtype/shape/layout. A
  single slot re-exported standalone to a generic consumer carries its `byte_offset` over the
  256-aligned bundle base (§3.3) and its explicit `strides`.

### 13.7 Persistent-decode KV with an AFFINE live extent

The affine analogue of §13.4 (single-`Range` KV); §13.4 stays as the simple one-sym case. A
per-layer K cache `[n_heads=32, K_capacity=4096, head_dim=128]` in F16, in the **persistent
decode graph** (built once, re-realized per token). The live length is
**`k_len = cached_len + new_tokens`**, where `cached_len = SymId(7)` (input-determined, bound
up-front each pass) and `new_tokens = SymId(8)` (1 in pure decode; the chunk size in chunked
prefill — carried as a base sym so the *same* graph serves decode **and** prefill chunks). The
buffer is a dense, fully-committed `K_capacity=4096` allocation.

- base `DLTensor`: `dtype={kDLFloat,16,1}`, `ndim=3`, `shape=[32, 4096, 128]` (**capacity**),
  `strides=[4096*128, 128, 1]` (explicit, keyed to capacity — §3.2; positive here, negatives are
  permitted per §3.2.1/V13), `data` 256-aligned. A sidecar-blind consumer sees an honest,
  fully-backed F16 `[32,4096,128]` tensor.
- sidecar: `flags = HAS_SYMBOLIC | HAS_AFFINE_EXTENT` (note: **`MEANING_REQUIRES_EXT` clear** —
  the base is a faithful F16 tensor and the allocation backs the full capacity, so reading the
  tail is harmless; V8 satisfied by the dense backing, exactly as §13.4).
  - `extents` (count = 3):
    - axis 0: `kind=Scalar`, `min=32`, `capacity=32`, `sym_id=NONE`, `cap_kind=0`.
    - **axis 1 (the live length): `kind=Affine`**, `min=1`, `capacity=4096`, `sym_id=NONE`,
      `cap_kind=EXPLICIT (0)`, `sym_scope=InputDetermined`,
      `affine = { c0=0, term_count=2, terms=[ {coeff=1, sym_id=7 /*cached_len*/},
      {coeff=1, sym_id=8 /*new_tokens*/} ] }`.
    - axis 2: `kind=Scalar`, `min=128`, `capacity=128`, `sym_id=NONE`, `cap_kind=0`.
  - `storage`: `class=Session`, `session_id=<this session>` (KV cache is session-state).
  - `residency`: `tier=Device`, `substrate=CudaUntyped`, `backend_id=Cuda`, `device_index=0`,
    `is_mmap_view=0`.
  - `buffers`: index 0, `size_bytes = 32*4096*128*2` (full-capacity backing ⇒ V8 satisfied).
- **Realize (per token), no recompute of a derived sym:** the call passes
  `FDXSymEnv { 7 → cached_len, 8 → new_tokens }` (both base symbols already in the pass's
  `SymEnv`). A flash-attention kernel: (1) evaluates
  `k_len = c0 + 1·lookup(env,7) + 1·lookup(env,8)` with i128 **checked** accumulate then narrow
  (V17); (2) applies `1 ≤ k_len ≤ 4096` (V14) — a `cached_len + new_tokens > 4096` is
  `ExtentOutOfRange` *before* the kernel touches memory; (3) walks the live prefix using capacity
  strides + the affine live extent. The **same sidecar serves every token** (re-supply data ptr +
  `FDXSymEnv`; no producer-side recompute of a composite `k_len` symbol).
- **Per-symbol bound (no-OOB compositional).** Because `cached_len (SymId 7)` is *also* the
  persistent-decode write offset, it carries its **own** bound — either as the consumed-prefix
  axis's `Range` extent or as a bounded `DynScalar` in the `SymEnv` — so the affine `k_len` guard
  (which bounds only the sum) does not let `cached_len` itself walk OOB (§6.4.2).
- **Unification:** the V cache for the same layer carries the *identical* affine expression, so
  K-length and V-length resolve to the same value by construction.
- **Cross-runtime export to a generic consumer:** because the live region on the **middle** axis
  is non-contiguous (§3.1.1), the producer exports a **materialized dense copy** `[32, k_len, 128]`
  (`k_len` evaluated from the env) as a standard dense F16 tensor with
  `DLPACK_FLAG_BITMASK_IS_COPIED` set (§9.1); the affine expression is resolved away at the
  boundary, affine identity preserved only across a Fuel→Fuel managed export.

> **Degenerate-`seq`-constant variant:** in pure single-token decode where `new_tokens` is a
> build-time constant 1, the same axis is `kind=Affine`,
> `affine={ c0=1, term_count=1, terms=[{1, sym=7}] }` (`k_len = cached_len + 1`) — still affine
> (non-zero `c0` ⇒ not reducible to `Range` per §6.4.3/V16), still no recompute.

### 13.8 Batched paged KV cache (gather / indexed residency)

A decode batch of `B = 4` sequences sharing one paged K pool, F16, `Hkv = 8` heads, `head_dim
D = 128`, `block_size = 16` tokens/block, `num_blocks = 256` physical blocks,
`max_blocks_per_seq = 64` (so `max_seq_capacity = 64*16 = 1024`). Live lengths advance together
this step: `context_len = SymId(11) ∈ [1, 1024]`. Pool typed shape (as-built):
`[num_blocks=256, block_size=16, Hkv=8, D=128]` F16; `pool_size_bytes = 256*16*8*128*2 =
8,388,608` (8 MiB).

- **base (honest pool):** `dtype = {kDLUInt, 8, 1}`, `ndim = 1`, `shape = [8388608]`,
  `strides = [1]` (explicit, §3.2), `byte_offset = 0`, `data` 256-aligned. **A sidecar-blind
  consumer sees exactly this raw 8 MiB byte pool — never a scattered cache.**
- **sidecar:** `flags = HAS_GATHER | HAS_SYMBOLIC | MEANING_REQUIRES_EXT` (V19 satisfied;
  optionally `| HAS_DTYPE_EXT` to carry the F16 element type explicitly).
  - `gather` (`FDXIndexedResidency`):
    - `kind = FDX_GATHER_PAGED_BLOCKS (1)`, `num_blocks = 256`, `block_size = 16`, `pool_buffer = 0`.
    - `physical_ndim = 4`, `physical_shape = [256, 16, 8, 128]`,
      `physical_strides = [16384, 1024, 128, 1]` (typed F16 **elements** — the byte address scales
      these by `elem_bytes=2` once, §6.9.4 step 3; dense & gap-free, explicit, positive by
      construction — V20), `element_dtype = 7 (F16)`.
    - `block_table`: `table_buffer = 1`, `id_dtype = 2 (U32 — v1 pin)`, `max_blocks_per_seq = 64`,
      `unmapped_sentinel = 0xFFFFFFFF` (≥ num_blocks=256 ⇒ caught by the range check — V18),
      `layout_flags = 0`.
    - `num_sequences = 4`, `max_seq_capacity = 1024` (= 64*16, u64 no-overflow — V18).
    - `logical_ndim = 3`, `seq_axis = 1`, `logical_shape = [8, 1024, 128]` (per-seq `[Hkv, S_cap, D]`).
    - `logical_extents_count = 3`: axis 0 `Scalar(8)`, **axis 1 (`seq_axis`) `Range`**
      `min=0`, `capacity=1024`, `sym_id=11` (P4 unification with `context_len_sym`), axis 2
      `Scalar(128)` — the live seq length is **inside** the extents machinery (V21e), not on the
      1-D base `extents[]`. **`min=0` (not 1)** because a per-seq live length of 0 is legal here
      (a finished/evicted sequence in the batch; the kernel skips it — `byte_kernels.rs:6876`),
      unlike the dense single-shared-length §13.4 KV where `min=1` is right (V21d/e).
    - `context_lens_buffer = 2`, `context_len_sym = 11`, `context_len_scope = 0 (InputDetermined)`.
  - `extents` (count = base.ndim = 1): axis 0 is `Scalar(8388608)` (the byte pool is concrete; this
    MUST equal `base.shape[0]` per V7, and equals the V19 honest-`uint8` cover
    `num_blocks*block_size*intra_block_typed_count*elem_bytes = 256*16*(8*128)*2 = 8388608`).
  - `storage`: `class = Session`, `session_id = <this batch's session>`.
  - `residency`: `tier = Device`, `substrate = CudaUntyped`, `backend_id = Cuda`,
    `device_index = 0`, `is_mmap_view = 0`.
  - `buffers`:
    - index 0 — role `FDX_BUFFER_POOL`, `dtype = F16` (typed view; base DLTensor still uint8),
      `size_bytes = 8388608` (V20: dense gap-free pool, so `physical_strides[0]*num_blocks*elem =
      16384*256*2 = 8,388,608 == 256*16*8*128*2` — full pool backed, gap-free; a *padded* pool
      would need `size_bytes` sized for the larger `physical_strides[0]`), `ndim = 4`,
      `shape = [256,16,8,128]`, `strides = [16384,1024,128,1]`.
    - index 1 — role `FDX_BUFFER_BLOCK_TABLE`, `dtype = U32`, `size_bytes = 4*64*4 = 1024`,
      `ndim = 2`, `shape = [4, 64]`, `strides = [64, 1]`.
    - index 2 — role `FDX_BUFFER_CONTEXT_LENS`, `dtype = U32`, `size_bytes = 4*4 = 16`,
      `ndim = 1`, `shape = [4]`, `strides = [1]`.
- **Realize (lazy, matching the as-built kernel):** the call passes `FDXSymEnv { 11 → live_len }`;
  the PagedAttn kernel reads `L = lookup(env, 11)`, applies `0 ≤ L ≤ 1024` and `ceil(L/16) ≤ 64`
  (V21d). **An `L == 0` sequence is legal and skipped** (`if ctx_len == 0 { continue; }`,
  `byte_kernels.rs:6876`) — this is why the seq-axis extent is `min=0`, not `min=1`. For each
  accessed `(s, t)` with `t < L`, the column index `t/16 < ceil(L/16) ≤ 64 = max_blocks_per_seq`
  is in-bounds **by the `ceil(L/16) ≤ 64` check above** (run *before* the dereference, V21d), then
  it gathers `block = block_table[s, t/16]` and asserts `block < 256` **at dereference** (V21d; the
  single `>= num_blocks` test catches both OOB and the sentinel). The flat block-table index and
  the byte address are computed in u64/checked against the runtime `usize` (V21d
  `GatherAddressOverflow` guard) so an 8 MiB+ pool / long context cannot silently wrap. No eager
  pre-pass — the build-time full-table scan is V21c. The same sidecar serves every decode step.
- **Data-determined variant (per-seq lengths differ):** set `context_len_sym = FDX_SYM_NONE`; the
  authoritative live lengths are read from `context_lens_buffer` (index 2) and the kernel validates
  `0 ≤ context_lens[s] ≤ 1024` and `ceil(context_lens[s]/16) ≤ 64` **per sequence as each is first
  accessed** (matching `byte_kernels.rs:6877`), not via a single env-resolved value (V21d
  data-determined case). The seq-axis `logical_extents[seq_axis]` then has no unified sym to carry;
  it stays a `Range{min=0, capacity=1024, sym_id=NONE-or-the-data-determined-sym}` and V14 cannot
  bound it from the env — the per-seq buffer check is the bound owner.
- **Cross-runtime export to a generic (blind) consumer:** because the logical cache is a scatter
  over physical blocks, `MEANING_REQUIRES_EXT` forces producer policy §9.1: the producer
  **materializes a dense un-paged copy** `[4, 8, L, 128]` of the live region with
  `DLPACK_FLAG_BITMASK_IS_COPIED` set, or refuses — never the raw pool labeled as a cache.

> Variant — **AFFINE per-seq live length (gather × affine compose).** The composition of the two
> 2026-06-17 additions is concrete, not just asserted: replace the seq-axis `Range` above with a
> `kind=Affine` extent and set `flags |= HAS_AFFINE_EXTENT`. For a persistent-decode paged batch
> whose per-seq live length is `context_len = cached_len + new_tokens`:
> `logical_extents[seq_axis] = { kind=Affine, min=0, capacity=1024 (== max_seq_capacity),
> sym_id=NONE, cap_kind=EXPLICIT(0), affine={ c0=0, term_count=2, terms=[{1, cached_len},
> {1, new_tokens}] } }`. At realize the seq extent runs its **own** V16 (well-formed) then V17
> (checked-i128 eval of `cached_len + new_tokens`) **before** V21(d) applies `0 ≤ L ≤ 1024` and
> `ceil(L/16) ≤ 64` — i.e. the gather's per-seq bound governs an affine-derived `L` exactly as it
> governs a `Range`-derived one (V21e routes `logical_extents[seq_axis]` through the full
> V7/V14/V16/V17 machinery keyed to `max_seq_capacity`). `min=0` keeps empty sequences legal. This
> closes the "do gather and affine compose" question with a worked case.
>
> Variant — **quantized paged cache:** add `flags |= HAS_QUANT`; the `quant` block describes the
> *within-block* packing and the `gather` block the *block-level* scatter; the two block
> geometries are kept consistent (V18/§6.9 interplay; a worked quant-paged example is an open item
> before v1 freeze).
>
> The K and V caches are described as **two FDX tensors sharing the same**
> `FDX_BUFFER_BLOCK_TABLE` / `FDX_BUFFER_CONTEXT_LENS` buffers (single-place rule). A bundled
> two-pool descriptor is a v2 candidate if a fully-fused KV pool emerges (open item).

---

## 14. Versioning rules

- **State space `{absent, v1, v2, …}`** via the explicit `version` field, never null/non-null
  (P2). Absence (null pointer) is "plain DLPack." The FDX `version` is a **major-only schema
  axis, independent of the DLPack ABI `DLPackVersion`** on the capsule (§5.2).
- **Feature detection is `(version, flags, struct_bytes)` jointly.** A consumer that wants to
  detect feature level without parsing the whole struct reads `version` (schema major), `flags`
  (which optional blocks are present), and `struct_bytes` (how far the struct extends). **Additive
  `struct_bytes` growth never changes the meaning of an existing flag bit.**
- **Additive within a version (P8):** new facts go into `reserved` space (zero-default) or a
  per-block `reserved` array; old readers, guarded by `struct_bytes`, read the known prefix and
  ignore the trailing tail. This covers the common case with *no* version bump.
- **Array-element growth is NOT covered by tail-ignore (load-bearing caveat).** The
  `struct_bytes` tail-ignore rule works for *trailing scalar fields* of a top-level struct, but
  **not** for the per-element stride of a variable-length array: growing `sizeof(FDXExtent)` would
  make an older reader (using its smaller element size to stride `extents[]`) misalign *every*
  entry after index 0, not cleanly ignore a tail. FDX therefore handles array-element evolution
  by **freezing the element's leading field offsets** and appending only into the element's own
  reserved region with an `offset_of!`-per-field build assertion (§5.4, §6.4 layout note): an
  affine-aware producer and an affine-aware consumer agree on `sizeof(FDXExtent)` and every
  pre-affine offset, and an older (pre-affine) reader that encounters a `kind=2` element hits the
  **unknown-`kind` typed-error rule** below (it does not silently truncate or mis-stride). A
  genuine *incompatible* element-layout change requires a **version bump**, never an
  "additive-within-v1" claim.
- **The two 2026-06-17 additions are additive within v1.** *Gather* (`FDXIndexedResidency` +
  `FDXBlockTable`) is carved from `FDXSidecar.reserved` and gated by `FDX_FLAG_HAS_GATHER` (bit 7);
  an older v1 reader guarded by `struct_bytes` simply does not see it (and reads the honest pool —
  safe). *Affine* extends `FDXExtent` with `kind=2` + the appended `affine` sub-block (flag bit 8),
  under the array-element-growth discipline above. A new `gather.kind` (ragged/CSR) or a new
  `FDXExtent.kind` value is itself additive; an unknown value is a typed
  `UnsupportedGatherKind` / `UnsupportedVersion`-class error, never a guess.
- **Version bump** only when a field's *meaning* changes incompatibly or a new mandatory block
  is introduced. A bump adds new `Capability` tokens (`DlpackExtV2`, …) so negotiation (§12) can
  distinguish.
- **Newer producer → older consumer (G4, required):** the producer queries the consumer's
  advertised max FDX version (§12) and emits a sidecar at that version, expressing only
  facts the older version can represent; v2-only features are either downgraded (e.g. a v2 tiling
  hint dropped — hints are optional) or, if meaning-bearing, dequantized/refused (§9.1).
- **Newer consumer → older sidecar:** trivially supported — the consumer recognizes the lower
  `version`, reads the corresponding prefix.
- **Enums are `#[non_exhaustive]`-spirited:** unknown `logical_dtype`/`family`/`packing` codes →
  typed error (`UnsupportedVersion`-class), never a guess. New dtypes (the MX set was recent; I8
  was added 2026-05-19) extend the FDX code table additively via the §6.0 conversion fn + test.
- **Multi-version read policy:** aligns with the DAG-format policy (decision #18) — Fuel reads
  the previous N FDX versions; additions are backward-compatible where feasible.

---

## 15. Interop & backward-compatibility rules

- **Absent sidecar ⇒ canonical DLPack.** Any DLPack ecosystem (PyTorch/JAX/CuPy/NumPy/TVM)
  consumes the base unchanged. This is the default for dense standard-dtype tensors. The base is
  v1.2+-conformant (explicit strides, 256-aligned data), so even a *strict* consumer accepts it.
- **Unrecognized magic/version ⇒ treat as absent.** A non-Fuel consumer that receives a
  `manager_ctx`-attached sidecar never reaches it: recovery is deleter-identity-gated (§10.2), so
  a generic consumer (different deleter) never dereferences `manager_ctx`. Safety is preserved by
  the honesty invariant either way.
- **Honesty is the backstop:** even a buggy/blind consumer reading the base sees correctly-sized
  `uint8` bytes with explicit strides and aligned data — never a mislabeled dtype, a
  `size_in_bytes()==0`-mis-sized buffer, NULL strides, or a misaligned pointer.
- **No silent coercion either direction:** producers never silently degrade meaning-bearing
  tensors (refuse-or-materialize with `IS_COPIED`, §9.1); consumers never silently
  dequant/contiguize on read (error instead, §9.2). Layout/quant fixups are explicit planner ops
  priced by FKC (§9.3).
- **Round-trip:** a Fuel→Fuel managed export carries the full sidecar via `manager_ctx`
  (deleter-identity-matched) and reconstructs the exact logical tensor (lossless). A
  Fuel→generic→Fuel round-trip through a blind intermediary loses the sidecar (the intermediary
  saw only the standard part, possibly forwarded under its own deleter); this is *stated, not
  hidden*, and meaning-bearing tensors are dequantized/materialized before such a trip.
- **Symbolic across boundaries:** sym identity (`sym_id`) is preserved across a Fuel→Fuel
  managed export so the consumer's unification survives; across a generic boundary the tensor is
  exported resolved (materialized dense copy, `IS_COPIED`).

---

## 16. Quick reference — DLPack conformance checklist (FDX exports)

A boundary-(b) FDX export is conformant iff:

- [ ] Versioned capsule (`DLManagedTensorVersioned`, `"dltensor_versioned"`), `DLPackVersion`
      set to the producer's DLPack ABI version (≥ 1.2, validated against 1.3).
- [ ] `strides` non-NULL, length `ndim` (§3.2, V11); negatives are permitted (int64, §3.2.1) — the
      signed touched-range lies within `size_bytes` (V13); the planner has normalized to
      non-negative only if the external peer cannot take negative strides (§3.2.1, §9.1).
- [ ] `data` 256-byte aligned; logical start in `byte_offset` (§3.3, V12).
- [ ] base `dtype` is standard (never a native sub-byte dtype; honesty-`uint8` for packed data)
      (§3, §3.4, V3).
- [ ] `IS_COPIED` set on any producer-materialized export (§9.1, §11).
- [ ] `READ_ONLY` mirrored from `FDX_FLAG_READ_ONLY` where applicable.
- [ ] stream ordering event enqueued per the §11 sentinel table when a non-`None` stream is
      supplied.
- [ ] sidecar on `manager_ctx` only with a consenting peer **and** Fuel's own deleter (§10.2).

---

## 17. Open questions / future work (incl. critique items deferred with rationale)

1. **Planner cost model for layout alternatives (critique, "both").** The contiguize-vs-strided-
   vs-materialize decision-by-declared-cost is *acknowledged and placed* in §6.5 / §9.3 as the
   FKC spec's responsibility (FDX is description-only by P3). **Not added to FDX as a cost slot
   by design** — adding cost to FDX would violate the description/decision separation. The
   sibling FKC spec must define the cost slot; this spec only pins the shared codes (§6.0) so the
   two compose. (Owned, not silently dropped.)
2. **Logical vs physical shape for sub-byte.** v1 carries the *physical byte* shape in the base
   and the bit-width in the sidecar; the logical-element shape is implicit (derived) or via an
   aux buffer ref. Should v2 carry an explicit logical-shape array, at the cost of a
   redundancy-consistency check? (Deferred — v1 derivation is unambiguous.)
3. **Affine sym expressions — RESOLVED (2026-06-17).** `FDXExtent` now carries a bounded affine
   form (`kind=Affine`, `c0 + Σ cᵢ·symᵢ`, cap `FDX_AFFINE_MAX_TERMS=4`) evaluated through the
   `SymEnv` at realize (§6.4.2, V16/V17, example §13.7), so persistent decode
   (`k_len = cached_len + new_tokens`) is planned once and served every token with no derived-sym
   recompute. Scalar/Range are the degenerate cases (V16 rejects degenerate affines so encodings
   never diverge). **Residual (deferred):** (a) generalizing the *source*
   `fuel-core-types::shape::Extent` with an `Extent::Affine` variant (vs transport-only) — sequence
   behind the first persistent-decode consumer, lazy-only norm favors building the primitive;
   (b) `cap_kind=AFFINE_MAX` (growable/ragged capacity) is reserved until the ragged-batch program
   is its consumer; (c) **negative coefficients** are type-allowed (i64) and the affine *result* is
   bounded by V14/V17; per-term compositional soundness is mechanically enforced by V16 **only when
   the term's sym also carries its own `Range`/`Affine` extent in the same sidecar** (V16
   cross-references it by `sym_id`). A negative-coeff term whose sym is bounded *only* in the
   `SymEnv` (a `DynScalar`, no in-sidecar extent) is a producer responsibility, not a mechanical
   check — confirm whether a future consumer needs a fully-mechanical per-term bound (a guaranteed
   in-sidecar extent for every affine term sym) or whether the producer-responsibility split
   suffices (no decode use needs negative coeffs today, so this stays deferred).
4. **Data-determined sym dependency edges.** `sym_scope=DataDetermined` is advisory in v1; the
   build-time producer→consumer dependency edge (NonZeroIndices/MoE counts) is a graph concern,
   not a tensor-description one. Confirm FDX needs no more than the scope tag.
5. **Bundle slot sub-byte/quant.** v1 keeps bundle slots simple/standard-dtype. Do any real
   multi-output ops emit a *quantized* slot? If so, `FDXOutputView` needs its own
   `dtype_ext`/`quant` sub-block (its `reserved` leaves room).
6. **Zero-point scale shape conventions.** Affine-int zero-points: per-tensor vs per-channel
   zero-point shapes; confirm they mirror the scale granularity exactly (v1 assumes yes; V6
   currently checks the scale buffer only — a zp-shape check would be additive).
7. **`Capability` token granularity.** Is one `DlpackExtV1` + per-feature tokens the right cut,
   or should negotiation be a bitmask field? (Enum chosen for consistency with the existing flat
   `Capability` design.)
8. **Stream object portability.** Boundary (a) passes a `FuelStream*`; mapping that onto each
   backend's native stream/queue (CUDA stream, Vulkan queue/timeline-semaphore) is a kernel-ABI
   detail to finalize with Baracuda/Vulkane (cross-project — propose before editing siblings).
   The boundary-(b) sentinel table (§11) is pinned; the boundary-(a) object mapping is not.
9. **Endianness across heterogeneous hosts.** Serialized form is little-endian; a big-endian
   host consumer is out of scope for v1 — confirm Fuel never targets one, or add a byte-order
   mark.
10. **Interaction with the `.fuel` mmap persistence.** The serialized sidecar (pointers →
    offsets) must be embeddable in the whole-graph `.fuel` mmap; confirm the offset base and
    alignment match the persistence layout (Phase E). The 256-byte data rule (§3.3) and the
    persistence alignment should be reconciled there.
11. **Complex / bool dtypes.** DLPack has `kDLComplex`/`kDLBool`; Fuel's `DType` has neither
    yet. Codes `0x0200`/`0x0201` are reserved (§6.1) so adding them later is additive.
12. **Paged-cache (gather) residuals — RESOLVED core (2026-06-17), residual deferred.** The paged /
    blocked KV cache is now a single FDX tensor (`FDX_FLAG_HAS_GATHER`, §6.9, V18–V21, example
    §13.8). **Residual (deferred):** (a) a full per-logical-axis `FDXExtent` array beyond the
    single seq axis (`logical_extents[]` already provides the seq axis; a richer multi-symbolic
    logical shape is a v2 candidate); (b) ragged/CSR gather kinds (`gather.kind >= 2`); (c) a
    bundled two-pool (K+V) descriptor if a fully-fused KV pool emerges (v1 uses two FDX tensors
    sharing the block-table/context-lens buffers); (d) the precise MX/GGML packing interleave
    *inside* a paged block (`HAS_QUANT | HAS_GATHER` is permitted and V18 checks geometry
    consistency, but a worked quant-paged example is wanted before v1 freeze).

---

## Appendix A — constant reference

```c
#define FDX_MAGIC                 0x46445800u
#define FDX_VERSION_1             1u
#define FDX_VERSION_MAX           1u
#define FDX_SYM_NONE              0xFFFFFFFFu
#define FDX_SESSION_NONE          0u
#define FDX_DTYPE_NONE            0xFFFFu
#define FDX_BUFFER_INLINE         0xFFFFFFFFu
#define FDX_BUFFER_NONE           0xFFFFFFFEu   /* gather: no such buffer (§6.9.3)      */
#define FDX_BLOCK_UNMAPPED        0xFFFFFFFFu   /* gather: unmapped block id (§6.9)     */

/* FDXExtentKind (FDXExtent.kind, §6.4). */
#define FDX_EXTENT_SCALAR         0u
#define FDX_EXTENT_RANGE          1u
#define FDX_EXTENT_AFFINE         2u
#define FDX_AFFINE_MAX_TERMS      4u            /* term-count cap; overflow -> typed err */
/* FDXExtent.cap_kind (affine): 0=EXPLICIT (v1), 1=AFFINE_MAX (reserved). */

/* FDXIndexedResidency.kind (gather, §6.9.3). */
#define FDX_GATHER_NONE           0u
#define FDX_GATHER_PAGED_BLOCKS   1u
/* FDXBufferRef.role gather additions: 5=POOL, 6=BLOCK_TABLE, 7=CONTEXT_LENS. */

/* FDXQuant.family (FDX_QUANT_*, §6.2). FDX is the normative owner of these codes. */
#define FDX_QUANT_NONE            0xFFFFu
#define FDX_QUANT_GGML_BLOCK      0u            /* baked scales, ggml_dtype ONLY (no PerBlock)  */
#define FDX_QUANT_MX              1u            /* F8E8M0 per-block scale; sole PerBlock user    */
#define FDX_QUANT_AFFINE_INT      2u
#define FDX_QUANT_AFFINE_FLOAT    3u
#define FDX_QUANT_AFFINE_BLOCK    4u            /* nf4/QLoRA: low-bit data + SEPARATE block scale */
/* FDXScaleGranularity: 0=PerTensor,1=PerToken,2=PerChannel,3=PerBlock (MX-only, §6.2). */

/* FDX_FLAG_* — see §5.2 (authoritative bit-allocation table). The flag bits:
     FDX_FLAG_HAS_DTYPE_EXT        (1u << 0)
     FDX_FLAG_HAS_QUANT            (1u << 1)
     FDX_FLAG_HAS_SYMBOLIC         (1u << 2)
     FDX_FLAG_HAS_TILING           (1u << 3)
     FDX_FLAG_IS_BUNDLE            (1u << 4)
     FDX_FLAG_MEANING_REQUIRES_EXT (1u << 5)
     FDX_FLAG_READ_ONLY            (1u << 6)
     FDX_FLAG_HAS_GATHER           (1u << 7)   // gather (§6.9)
     FDX_FLAG_HAS_AFFINE_EXTENT    (1u << 8)   // affine extent (§6.4)
   A build-time test asserts no two FDX_FLAG_* share a bit.
   FDXPacking / FDXScalePlacement / FDXScaleGranularity / FDXPackOrder / FDX_QUANT_* — see §6.
   Standard DLPack flags FDX relies on (dlpack.h), NOT redefined by FDX:
     DLPACK_FLAG_BITMASK_READ_ONLY              (1UL << 0)
     DLPACK_FLAG_BITMASK_IS_COPIED              (1UL << 1)   // set on materialize (§9.1)
     DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2)   // FDX never sets (§3.4)
*/
```

## Appendix B — mapping table (Fuel type ⇄ FDX field)

> The FDX codes are **owned by FDX** (§6.0), not a positional mirror; the conversion fns in
> `fuel-core-types::dlpack` + the build-time mapping test are the binding, so a source-enum
> reorder breaks the build rather than shifting the ABI.

| Fuel as-built type | FDX carrier |
|---|---|
| `DType` (incl. sub-byte F4/F6E2M3/F6E3M2/F8E8M0) | `FDXDTypeExt.logical_dtype` (FDX code) + `bit_width` + `packing` |
| `DType::size_in_bytes()==0` (flag) | superseded by `FDXDTypeExt.bit_width` (never 0) |
| `GgmlDType` (Q4_0…Q8K) | `FDXQuant.family=GGML_BLOCK` + `ggml_dtype` (FDX code); baked scales, no separate operand |
| nf4/QLoRA block-affine weight | `FDXQuant.family=AFFINE_BLOCK` + `block_shape` + separate `scale_buffer` (block-shaped absmax, named once) |
| `ScaleGranularity` | `FDXQuant.scale_granularity` / `scale_pair_*` (FDX code); `PerBlock` is MX-only |
| `ScalePair` (act × weight) | `FDXQuant.scale_pair_act`/`scale_pair_weight` + `role` |
| `Extent::{Scalar,Range{min,max,sym}}` | `FDXExtent{kind=0/1, min, capacity, sym_id}` |
| affine extent (`k_len = cached_len + new_tokens`) | `FDXExtent{kind=2, cap_kind=EXPLICIT, affine={c0, terms[(coeff,sym)]}}` (§6.4.2) |
| `DynAxis{axis,min,sym}` | `FDXExtent` entries (axis = array index) |
| `SymId(u32)` | `FDXExtent.sym_id` (u32) / `FDXSymBinding.sym_id` (u32) |
| `SymEnv` (`SymId -> usize`, per-pass) | `FDXSymEnv` (`u32 -> u64`, call-time, never serialized; §6.4 narrowing) |
| `DynScalar` (offset/pos/k_len) | rides `FDXSymEnv` as a binding (NOT an `FDXExtent`) |
| `SubstrateClass` (`#[non_exhaustive]`) | `FDXResidency.substrate` (FDX code, pinned by fn+test) |
| `DeviceLocation`/`BackendId` (`#[non_exhaustive]`) | `FDXResidency.{backend_id,device_index}` (FDX codes) + base `DLDevice` |
| mmap residency (digest §11) | `FDXResidency.tier=DiskMmap` + `is_mmap_view` |
| storage class (shared/session/transient) | `FDXStorage.class` |
| `SessionId` | `FDXStorage.session_id` |
| `OutputView{byte_offset,len,dtype,shape,layout,name}` | `FDXOutputView` (explicit strides) |
| `BackendCapabilities::required_alignment` | `FDXTiling.alignment_bytes` (floor 256 on boundary b) |
| `access_granularity_bits` | `FDXTiling.access_granularity_bits` |
| `Capability` (FDX tokens) | `DlpackExtV1`/`Mx`/`Ggml`/`Affine`/`Symbolic` (§12) |
| DLPack ownership/freshness | `DLManagedTensorVersioned.flags`: `READ_ONLY`, `IS_COPIED` (§9.1, §11) |
