# Fuel Self-Describing Storage Encoding — `DType` + `SType` / `Encoding`

**Status:** DRAFT FOR REVIEW (2026-06-18), on branch `feat/kernel-contracts-dlpack`. Design pass —
no code yet. This is the third boundary spec, sibling to
[`dlpack-extension.md`](dlpack-extension.md) (FDX, the kernel-boundary tensor projection) and
[`kernel-contract-format.md`](kernel-contract-format.md) (FKC, the kernel advertisement). It is
the **source of truth for the internal `SType` / `Encoding` types**; FDX is `SType`'s *projection*
at the kernel boundary (§7). When this spec and FDX disagree on the boundary mapping, the two must
be reconciled — but the internal type shape is owned **here**, and the boundary code shape is owned
by FDX.

**Scope.** How a `Storage` describes *the scheme by which its bytes encode logical elements*, on
the tensor itself, without an op consulting op-params. Two orthogonal axes:

- **`DType`** (existing, [`fuel-core-types/src/dtype.rs:14`](../../fuel-core-types/src/dtype.rs)) —
  the **logical** element type ("what is a value"). Unchanged.
- **`SType`** (new) — the **physical** encoding stack ("how is a logical element stored"). A named
  newtype over an ordered stack of `Encoding` layers; empty = plain dense `DType`.

**Audience.** `fuel-core-types` (where `DType`, `GgmlDType`, `ScaleGranularity`, `Storage` live),
`fuel-memory` (the closed-enum `Storage`), `fuel-graph` / `fuel-dispatch` (operand binding, plan
caches, the multi-output machinery), and `fuel-memory::dlpack_view` (the FDX projection point).

**Authoritative inputs (verified against code, file:line cited inline).** The as-built core types
in [`fuel-core-types/src/dtype.rs`](../../fuel-core-types/src/dtype.rs),
[`quantized.rs`](../../fuel-core-types/src/quantized.rs),
[`quant_scale.rs`](../../fuel-core-types/src/quant_scale.rs),
[`storage.rs`](../../fuel-core-types/src/storage.rs); the closed-enum `Storage` in
[`fuel-memory/src/lib.rs`](../../fuel-memory/src/lib.rs); the multi-output graph machinery in
[`fuel-graph/src/lib.rs`](../../fuel-graph/src/lib.rs),
[`fuel-dispatch/src/pipelined.rs`](../../fuel-dispatch/src/pipelined.rs), and
[`docs/architecture/12-multi-output.md`](../architecture/12-multi-output.md); and the sibling FDX
spec ([`dlpack-extension.md`](dlpack-extension.md), §6.1–§6.2). When this spec and the constitution
([`../architecture/`](../architecture/)) conflict, the constitution wins; flag the conflict.

---

## 0. Status, thesis, and the locked decision

**Date:** 2026-06-18. **Branch:** `feat/kernel-contracts-dlpack`. **Status:** design, no code.

**Thesis.** Today, "how to interpret a tensor's bytes" is split between the tensor view and
op-params: a quantized weight's *bytes* live in `Storage`, but the *scheme* that gives those bytes
meaning (e.g. that they are NF4 block-affine, and where the per-block scales are) is carried as op
parameters or implied by which kernel was chosen. This spec makes the **encoding scheme
self-describing on the tensor**: any op holding the tensor knows how its bytes are encoded without
consulting op-params. The crucial split — and the load-bearing architectural decision (§4) — is:
**the scheme moves onto the tensor; the scale *values* (bulk data) stay a sibling operand in the
graph; FDX re-unites the two into one descriptor at the kernel boundary.**

> **LOCKED DECISION (2026-06-18) — Self-describing Storage: `DType` + `SType`/`Encoding`.**
> Today "how to interpret a tensor's bytes" is split between the tensor view and op-params (e.g.
> quant scales passed as op parameters). Goal: make the **encoding scheme** self-describing **on
> the tensor**, so any op holding the tensor knows how its bytes are encoded without consulting
> op-params. The scale **values** (bulk data) stay a sibling operand; only the **scheme** moves
> onto the tensor.
>
> - **`DType`** = the **logical** element type ("what is a value"). UNCHANGED, stays logical: an
>   NF4 weight's `DType` is the logical float it represents (F16/F32), not the 4-bit storage.
> - **`SType`** = an ordered stack of encoding layers describing **how** logical elements are
>   physically stored. A **named newtype** (not a bare field). Default = empty = plain.
> - **`Encoding`** = ONE layer. Holds **only static descriptors** (geometry, scheme, dtype codes,
>   scale REQUIREMENTS) — NEVER bulk data (weights) and NEVER scale VALUES. Keeps `Encoding` small
>   and `Eq + Hash` so it can feed structure keys / plan caches.
> - **Attachment:** `Storage` gains `stype: SType` (default empty). v1: `SType` lives on the
>   PRIMARY `Storage` only; bundle slots keep `dtype` only (per-slot `SType` is a FUTURE addition).
> - **The load-bearing decision:** decided AGAINST embedding the scale buffer inside the weight's
>   `Storage`/`Encoding` ("composite-by-reference", model A). Decided FOR **(B) graph layer =
>   sibling operands** PLUS **kernel boundary = FDX sidecar composite projection**.

The remaining sections expand this verbatim decision into a self-contained spec.

---

## 1. `DType` vs `SType` — logical vs physical, orthogonal

Fuel keeps two independent axes for "what these bytes mean". They are **orthogonal**: any `DType`
may carry any (compatible) `SType`, and the two are read for different questions.

| | **`DType`** (logical) | **`SType`** (physical) |
|---|---|---|
| Question it answers | *What is a value?* | *How is a logical value stored?* |
| Example value | `F16`, `F32`, `I8` | `[]` (plain) · `[GgmlBlock { Q4K }]` · `[AffineBlock { F4, [64], … }]` |
| Status | **existing**, unchanged | **new** |
| Home | [`dtype.rs:14`](../../fuel-core-types/src/dtype.rs) | new, this spec |
| Derives | `Copy, Eq, Hash` ([`dtype.rs:12`](../../fuel-core-types/src/dtype.rs)) | `Eq, Hash` (§2) |

**`DType` stays logical.** An NF4 (block-affine 4-bit) weight whose values *represent* F16 floats
has `DType::F16` — **not** a hypothetical "4-bit" `DType`. The 4-bit packing is an `Encoding`
*layer*, not a logical element type. This is why `DType` is unchanged by this spec: it already
answers the logical question, and forcing storage facts into it would conflate the two axes. Note
that `DType` *does* already include sub-byte logical floats (`F4`, `F6E2M3`, `F6E3M2`, `F8E8M0`,
[`dtype.rs:39-46`](../../fuel-core-types/src/dtype.rs)) — those are genuine *logical* element types
(an MX payload's logical element really is `F4`), and `DType::size_in_bytes()` returns `0` for them
([`dtype.rs:123-125`](../../fuel-core-types/src/dtype.rs)). The *packing* of those sub-byte
elements is, again, an `Encoding` concern, not a `DType` one.

**Why orthogonal and not one fused enum.** A fused `DType × packing` enum would multiply: every
logical float × every packing scheme × every block geometry. Keeping them orthogonal means the
~15-variant `DType` and the small `Encoding` set compose freely, and ops that only care about the
logical type (broadcast-shape inference, autograd dtype rules) read `DType` and ignore `SType`
entirely — exactly as they do today, since `SType` defaults empty.

---

## 2. The `SType` newtype and the `Encoding` enum

`SType` is a **named newtype**, not a bare `SmallVec` field, because it needs a home for: the
layer-ordering invariant (§8), the `to_fdx()` projection method (§7), construction invariants, and
room for the representation to evolve. Default (empty) = **plain**: a dense `DType` buffer with no
extra interpretation — byte-identical in behavior to today's `Storage`.

```rust
use smallvec::SmallVec;
use crate::dtype::DType;
use crate::quantized::GgmlDType;       // existing, quantized.rs:26
use crate::quant_scale::ScaleGranularity; // existing, quant_scale.rs:38

/// An ordered stack of encoding layers describing HOW the logical
/// elements of a `Storage` are physically stored. Empty = plain
/// (a dense `DType` buffer, no extra interpretation).
///
/// Named newtype (not a bare field) so it owns the layer-ordering
/// invariant (§8), the `to_fdx()` projection (§7), and construction
/// validation, and so the representation can evolve. The `[Encoding; 1]`
/// inline size keeps the common single-layer case allocation-free.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SType(pub SmallVec<[Encoding; 1]>);

impl SType {
    /// The plain (dense, no-encoding) `SType`. `SType::default()` is
    /// identical; both are the empty stack.
    pub const fn plain() -> Self { SType(SmallVec::new_const()) }

    /// True if this is the plain encoding (no layers).
    pub fn is_plain(&self) -> bool { self.0.is_empty() }

    /// Project to the FDX sidecar quant/dtype-ext descriptors at the
    /// kernel boundary (§7). The op supplies the bound scale operand's
    /// buffer-table index for any `ScaleSpec` requirement.
    pub fn to_fdx(&self /* , scale_buffers: &ScaleBufferBinding */) -> FdxEncodingProjection {
        // §7 — variant-by-variant mapping. The op binds concrete
        // scale_buffer indices; SType carries only the requirement.
        todo!("§7 projection")
    }
}

/// ONE encoding layer. Holds ONLY static descriptors — geometry, scheme,
/// dtype codes, and scale REQUIREMENTS. It NEVER holds bulk data (the
/// weight bytes live in `Storage::inner`) and NEVER holds scale VALUES
/// (those are a sibling operand, §4). This keeps `Encoding` small and
/// `Eq + Hash`, so it can key structure / plan caches.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// GGML / GGUF block format. The scale is baked INLINE in each block
    /// struct (Q4_0 = {f16 d; u8 qs[16]} = 18 bytes/block); one
    /// self-contained buffer, no separate scale operand. The `GgmlDType`
    /// IS the format (block size + byte layout). Projects to FDX
    /// GGML_BLOCK, scale_placement=INLINE (§5, §7).
    GgmlBlock {
        /// The ggml block-format dtype. Existing type, quantized.rs:26;
        /// block_size()/type_size() at quantized.rs:87-113 are the
        /// authoritative geometry.
        ggml_dtype: GgmlDType,
    },

    /// Block-grained affine quant: NF4 / QLoRA-style. Low-bit packed
    /// data plus a SEPARATE per-block absmax scale operand (the scale is
    /// a graph sibling, NOT baked, NOT a pointer here — §4). Maps to FDX
    /// AFFINE_BLOCK (family code 4), scale_placement=SEPARATE_BUFFER (§7).
    AffineBlock {
        /// The sub-byte logical element code of the packed payload, drawn
        /// from the existing `DType` sub-byte set (NF4 rides `DType::F4`,
        /// dtype.rs:44 — the NF4 codebook is the kernel's; FDX projects
        /// this to logical_dtype=13 (F4), §7 / FDX §6.1, dtype-ext line 870).
        packed: DType,
        /// Block extent along each quantized axis. QLoRA default is the
        /// 1-D `[64]` block along the flattened weight. PARAMETRIC, never
        /// hardcoded to one format.
        block_shape: SmallVec<[u32; 2]>,
        /// REQUIREMENT for a sibling per-block absmax scale operand —
        /// dtype + granularity only, NOT an operand pointer. The
        /// consuming OP binds the actual operand (§4); FDX fills the
        /// concrete `scale_buffer` index at projection (§7).
        scale: ScaleSpec,
        /// Asymmetric-affine zero-point requirement. `None` for
        /// symmetric formats (NF4 is symmetric → `None`).
        zero_point: Option<ScaleSpec>,
    },

    /// OCP-microscaling (MX). RESERVED placeholder — declared so the
    /// layer-stack vocabulary is forward-compatible, but NOT wired in v1.
    /// Shape will follow FDX MX (F4/F6 payload + one F8E8M0 scale per
    /// block, FDX §6.2 family code 1). Do not construct in v1.
    Mx {
        // reserved; fields land with MX support. See §6, §8.
    },

    // ---- Reserved for LATER (listed for shape; do NOT implement now) ----
    //   AffineInt   — dynamic int affine (per-tensor/token/channel),
    //                 FDX AFFINE_INT (family 2). Needs a runtime-scale
    //                 operand, not a static descriptor.
    //   AffineFloat — dynamic FP8/etc. affine, FDX AFFINE_FLOAT (family 3).
    //   Compressed  — sparse / entropy-coded payloads (no FDX family yet).
}

/// A REQUIREMENT descriptor for a scale (or zero-point) operand: its
/// dtype and granularity. It is NOT an operand pointer and NOT the scale
/// values. It says "I need an absmax operand of this dtype/granularity";
/// the consuming OP binds the actual operand (§4); FDX fills the concrete
/// `scale_buffer` index at projection (§7).
///
/// The per-block scale SHAPE is DERIVED, not stored: it is computed from
/// the base tensor's `shape` and the layer's `block_shape` (one scale per
/// block). Storing it would duplicate a derivable fact and risk drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScaleSpec {
    /// The scale operand's element dtype (e.g. `DType::F32` for an NF4
    /// absmax, `DType::F8E8M0` for an MX block scale).
    pub dtype: DType,
    /// The scale's granularity. For `AffineBlock` the operative grain is
    /// `block_shape` (the block-shaped separate scale); `granularity`
    /// records the coarse op-level form for dispatch keying. Existing
    /// type, quant_scale.rs:38 (`PerTensor`/`PerToken`/`PerChannel`).
    pub granularity: ScaleGranularity,
}
```

**Why `Encoding` is `Eq + Hash` and data-free.** `Encoding` feeds **structure keys and plan
caches** — two graphs with the same op + same encoding scheme must hash equal so a compiled plan is
reused. Bulk data (weight bytes, scale values) is *not* part of that identity: two NF4 weights with
identical geometry but different values share a plan. Hence `Encoding` holds **only** the static
scheme. (This mirrors the FDX rule that the boundary descriptor carries no values either — FDX
`FDXQuant` carries `block_shape` + a `scale_buffer` *index*, never the scale bytes, FDX §6.2.)

**Verified type bindings (ground truth, not the sketch):**

- `GgmlDType` — [`quantized.rs:26`](../../fuel-core-types/src/quantized.rs); 15 variants
  `F32 … Q8K, BF16`. `block_size()` / `type_size()` at
  [`quantized.rs:87-113`](../../fuel-core-types/src/quantized.rs) (Q4_0 = 18 bytes/block, etc.).
- `DType` (for `AffineBlock.packed`) — [`dtype.rs:14`](../../fuel-core-types/src/dtype.rs); the
  sub-byte members `F4`/`F6E2M3`/`F6E3M2`/`F8E8M0` exist
  ([`dtype.rs:39-46`](../../fuel-core-types/src/dtype.rs)). **NF4 reuses `DType::F4`** — there is no
  separate `NF4` `DType`, and this spec does **not** invent one (the locked decision forbids
  inventing dtype variants; the NF4 codebook is the kernel's, exactly as FDX §13.5a states).
- `ScaleGranularity` — [`quant_scale.rs:38`](../../fuel-core-types/src/quant_scale.rs);
  `PerTensor`/`PerToken`/`PerChannel`. **Divergence note:** there is **no `PerBlock` variant** in
  the as-built enum (its module doc, [`quant_scale.rs:24-30`](../../fuel-core-types/src/quant_scale.rs),
  explicitly says block-quant formats "don't expose `ScaleGranularity`"). This matches FDX, which
  keeps `PerBlock` an **FDX/FKC-only** code (MX-only) that is **not** mirrored back into
  `ScaleGranularity` (FDX §6.2, FKC "Resolved critique" `PerBlock` note). Therefore `AffineBlock`'s
  block grain is carried by `block_shape`, and `ScaleSpec.granularity` records only the coarse
  op-level form — it is **not consulted** for the block grain (FDX §13.5a: `scale_granularity` "left
  at default `PerTensor` and **not consulted**"). Do not add `PerBlock` to `ScaleGranularity` for
  `AffineBlock`.

---

## 3. Attachment to `Storage`

`Storage` gains one field, `stype: SType`, defaulting to empty (plain). There are **two** `Storage`
definitions in the tree (the type-erased `fuel-core-types` one and the closed-enum `fuel-memory`
one); both gain the field.

**As-built today (verified):**

- [`fuel-core-types/src/storage.rs:216-224`](../../fuel-core-types/src/storage.rs):
  ```rust
  pub struct Storage {
      pub(crate) inner:  Box<dyn DynBackendStorage>,
      pub(crate) bundle: Option<Arc<[OutputView]>>,
  }
  ```
  (here the dtype is read from `inner.dtype_dyn()`, [`storage.rs:396-398`](../../fuel-core-types/src/storage.rs)).
- [`fuel-memory/src/lib.rs:89-101`](../../fuel-memory/src/lib.rs):
  ```rust
  pub struct Storage {
      pub inner: BackendStorage,
      pub dtype: DType,
      pub bundle: Option<Arc<[OutputView]>>,
  }
  ```

> **Divergence note vs the locked sketch.** The sketch describes "Today `Storage = { inner,
> dtype, bundle }`". That is exactly the **`fuel-memory`** shape; the **`fuel-core-types`** shape
> has no explicit `dtype` field (it delegates to `inner.dtype_dyn()`). Both still gain `stype`; the
> `fuel-core-types` one simply has `{ inner, bundle, stype }`.

**After (both crates):**

```rust
// fuel-memory/src/lib.rs
pub struct Storage {
    pub inner:  BackendStorage,
    pub dtype:  DType,
    pub bundle: Option<Arc<[OutputView]>>,
    /// How the bytes are ENCODED (the scheme), self-describing on the
    /// tensor. Default empty = plain dense `dtype`. v1: PRIMARY-only —
    /// bundle slots keep `dtype` only (§3, per-slot SType is future).
    pub stype:  SType,
}
```

```rust
// fuel-core-types/src/storage.rs
pub struct Storage {
    pub(crate) inner:  Box<dyn DynBackendStorage>,
    pub(crate) bundle: Option<Arc<[OutputView]>>,
    pub(crate) stype:  SType,   // default empty
}
```

**Default-empty is byte-identical.** Every existing single-output `Storage` constructor
(`Storage::new`, `from_dyn`, `Storage::new` in fuel-memory) sets `stype: SType::default()`. An empty
`SType` means "plain dense `dtype`" — the exact behavior every existing `Storage` has today — so no
existing path changes. New constructors (`Storage::with_stype(self, SType) -> Result<Self>`, mirroring
`with_bundle`) attach a non-empty encoding; construction validates §8's invariants.

**v1 is PRIMARY-only.** `SType` lives on the **primary** `Storage` only. Bundle slots
(`OutputView`, [`storage.rs:46-68`](../../fuel-core-types/src/storage.rs)) keep **`dtype` only** —
they carry no `SType` in v1. The realistic v1 producers of encoded storage are *single-output*
quantized weights (a loaded NF4 / GGUF weight is one tensor, not a bundle); a *multi-output* node
emitting an encoded slot is not a v1 need. **Per-slot `SType` is a FUTURE addition** — note it, do
not build it. When it lands it will add an optional `stype: SType` to `OutputView` and an FDX
per-view dtype-ext (FDX already reserves this: FDX §13.6 view dtype "may itself be sub-byte → its
own dtype_ext semantics; v1 keeps slot dtype simple/standard").

---

## 4. THE load-bearing decision — sibling operand (B) + FDX sidecar composite at the boundary

This is the decision the whole spec turns on: **where do the scale (absmax) VALUES live?** The
scheme is on the tensor (§2–§3); the bulk scale data is the open question.

**Decided FOR (the two-layer answer):**

- **(B) Graph layer = SIBLING OPERANDS.** The per-block scale (absmax) is a **separate first-class
  tensor / graph edge**, an operand of the *consuming op* (dequant / matmul). The weight's
  `Encoding` declares only the **requirement** (`AffineBlock { …, scale: ScaleSpec, … }`); the **op
  binds the actual scale operand**. The weight's `Storage` holds only the packed weight bytes.
- **PLUS Kernel boundary = FDX SIDECAR COMPOSITE PROJECTION.** The `DlpackView` / FDX sidecar
  **re-unites** `{weight scheme, scale-buffer reference}` into ONE self-describing descriptor for
  the kernel: FDX `AFFINE_BLOCK` with `scale_placement = SEPARATE_BUFFER` and `scale_buffer = a
  buffer-table index` (§7).

**Decided AGAINST — model A, "composite-by-reference":** embedding the scale buffer *inside* the
weight's `Storage` / `Encoding` (an `Encoding::AffineBlock` that owns an `Arc` to the scale
storage, so the weight tensor is a self-contained composite). **Rejected** for the three verified
facts below.

### Why B, not A — three verified facts plus the property-preserved argument

**Fact 1 — multi-output graph machinery is ONE-BUFFER ONLY.** A multi-output node allocates **one**
bundled `Storage` (one alloc, one `Arc`) plus `OutputView` offset-slots; `Op::View { slot }` clones
that `Arc` and bakes `byte_offset` into the layout's `start_offset` — a zero-copy **window** into
the **one** buffer — while `Op::ViewOwned { slot }` **memcpys** a slot into a fresh allocation.
Verified:

- `Op::View` / `Op::ViewOwned { slot: u32 }` —
  [`fuel-graph/src/lib.rs:962-976`](../../fuel-graph/src/lib.rs); the View builder bakes the slot's
  `byte_offset` into the layout at [`fuel-graph/src/lib.rs:2690-2718`](../../fuel-graph/src/lib.rs).
- `Op::View` shares the producer's `Arc` (zero-copy window), `Op::ViewOwned` memcpys
  `producer.bytes[byte_offset .. byte_offset + len_bytes]` into a fresh contiguous `Storage` —
  [`fuel-dispatch/src/pipelined.rs:273-295`](../../fuel-dispatch/src/pipelined.rs) and the
  `SlotOwn` work item at [`pipelined.rs:1272-1295`](../../fuel-dispatch/src/pipelined.rs).
- The one-allocation contract: [`docs/architecture/12-multi-output.md`](../architecture/12-multi-output.md)
  and `allocate_bundled_storage` ([`storage.rs:187-201`](../../fuel-core-types/src/storage.rs)),
  which sizes **one** backing buffer covering all slots.

So the graph supports **"many NODES sharing ONE allocation"**, **not** "one node owning many
*separate* allocations". A weight's separately-sourced absmax scale is a *different* allocation
(NF4 / bnb ships the scale as its own tensor). Folding it into a bundle alongside the weight would
require a **load-time merge-copy** of two separate source buffers into one — which **kills zero-copy
load** for no benefit. Model A, generalized, runs into the same wall: it would want the weight
`Storage` to *own* the scale `Storage`, i.e. one node owning two separate allocations — the shape
the machinery deliberately does not provide.

**Fact 2 — FDX ALREADY specifies the scale as a SEPARATE operand.** The boundary spec is already
model-B at the boundary:

- `AFFINE_BLOCK` (family 4): `scale_present == 1`, `scale_placement == SEPARATE_BUFFER` (**never**
  `INLINE`), `scale_buffer` = a real buffer-table index — FDX
  [§6.2 line 958](dlpack-extension.md) and the field semantics at
  [lines 969-989](dlpack-extension.md).
- `GGML_BLOCK` (family 0): scale **baked inline**, **no separate scale operand** — FDX
  [§6.2 line 954](dlpack-extension.md).

So model B at the graph layer is **already what the shipped boundary spec assumes**. Model A would
**contradict** FDX: the projection would have to *invent* a `scale_buffer` from an `Arc` the FDX
buffer-table does not know about, or fake an inline placement that the `AFFINE_BLOCK` family forbids.

**Fact 3 — B is cheaper AND honest.** With B, the scale is a **normal operand**: placeable,
transferable, and costable by the **existing** planner machinery (the planner already handles
operands). No recursive `Storage` (a `Storage` owning a `Storage`), no new planner introspection to
discover a hidden inner allocation. **Weight+scale co-location falls out automatically** — both feed
the *one* consuming op, so the planner lands both on the device where that op runs. It matches
external convention (GPTQ, HuggingFace, bitsandbytes all pass scales as separate tensors). And it
keeps `Encoding` a small `Eq + Hash` POD (Fact: an `Arc<Storage>` inside `Encoding` would break
`Eq`/`Hash`-by-value and bloat plan keys — §2).

**Fact 4 — the self-describing property that MATTERS is still delivered.** The **scheme** is
self-describing on the tensor: an op holding the weight reads `stype = [AffineBlock { F4, [64], … }]`
and knows it is NF4 block-affine **without any op-param**. The scale **values** are bulk data —
correctly a sibling operand. FDX re-unites scheme + scale-reference at the boundary, where the
kernel needs exactly one descriptor. This was a **revision of an earlier lean toward A**; the
multi-output check (Fact 1) plus the FDX re-read (Fact 2) settled it as **B**.

> **Rejected: model A (composite-by-reference).** `Encoding::AffineBlock` does **not** hold an
> `Arc<Storage>` (or any pointer) to the scale buffer. Rationale: it would (a) demand "one node
> owning many separate allocations", a shape the one-buffer multi-output machinery does not provide
> (Fact 1); (b) contradict FDX `AFFINE_BLOCK`'s `SEPARATE_BUFFER` requirement (Fact 2); (c) force
> recursive `Storage` + new planner introspection and break `Encoding`'s `Eq`/`Hash` (Fact 3). The
> only thing A would "buy" — physical co-location of weight and scale — falls out of B for free
> (Fact 3). `Encoding` carries the **requirement** (`ScaleSpec`), never the operand.

---

## 5. GGML stays inline; the efficiency rule (layout follows source)

**GGML stays inline — forced, not a choice.** GGUF on-disk is **interleaved struct-packed**: each
block bakes its scale(s) into one struct (Q4_0 = `{ f16 d; u8 qs[16] }` = 18 bytes/block; the
per-format byte budget is `GgmlDType::type_size()`,
[`quantized.rs:87-105`](../../fuel-core-types/src/quantized.rs), and the block size is
`GgmlDType::block_size()`, [`quantized.rs:107-113`](../../fuel-core-types/src/quantized.rs)). The
GGUF format, the k-quants math, and the ~40 quantized kernels (the `DynQuantizedStorage` family,
[`quantized.rs:124-198`](../../fuel-core-types/src/quantized.rs)) all assume that interleaving, and
**zero-copy mmap requires it**. Therefore `Encoding::GgmlBlock` is **inline**: one self-contained
buffer, **no** sibling scale operand, projecting to FDX `GGML_BLOCK` / `scale_placement = INLINE`
(§7). There is no `ScaleSpec` on `GgmlBlock` — the scale is recovered per-format from `ggml_dtype`,
not from a requirement.

**Do NOT generalize interleaving to NF4.** Interleaving NF4's absmax into the weight buffer would
force a **repack on load** from bnb's separate-tensor format (bnb ships weight and absmax as two
tensors) — killing zero-copy — for **no kernel-locality win**: the absmax array is tiny and
block-indexed, so the kernel reads it from a separate buffer at negligible cost. `AffineBlock` is
therefore **separate** (the §4 sibling operand).

**Efficiency rule (general): layout follows source.** Match the **source format's native layout**
to preserve zero-copy on load. There is **no universal winner**:

| Source format | Native scale layout | Fuel `Encoding` | FDX placement |
|---|---|---|---|
| GGUF / GGML | interleaved (baked in block) | `GgmlBlock` (inline) | `GGML_BLOCK` / `INLINE` |
| NF4 / QLoRA / bnb | separate absmax tensor | `AffineBlock` (sibling) | `AFFINE_BLOCK` / `SEPARATE_BUFFER` |
| GPTQ | separate scales + zeros | `AffineBlock` (sibling, future zp) | `AFFINE_BLOCK` / `SEPARATE_BUFFER` |
| MX (OCP) | separate F8E8M0 per block | `Mx` (reserved, §6) | `MX` / `SEPARATE_BUFFER` |

The principle: the `Encoding` chosen for a loaded weight is the one whose physical layout **equals
the source file's**, so the loader can `mmap` (or zero-copy adopt) the bytes with no repack.

---

## 6. Dynamic vs static

`Encoding` holds **only static** descriptors. The static/dynamic split maps cleanly onto the §4
decision and needs **no new `Storage` capability**:

- **Static scheme / block-size** → `Encoding` parameters. The format (`GgmlDType`), the packed
  sub-byte code (`DType::F4`), and the block geometry (`block_shape`) are known at graph-build time
  and live in the layer. These are the v1 cases (`GgmlBlock`, `AffineBlock`).
- **Dynamic block sizes / dynamic scales** → **additional operands** (or runtime-produced tensors
  feeding the op). A dynamically-quantized activation whose scales are computed at runtime is
  expressed as a runtime tensor bound as a sibling operand of the consuming op — exactly the §4-B
  shape, just with the scale produced by an upstream node instead of loaded. The reserved
  `AffineInt` / `AffineFloat` families (§2) are the dynamic-affine cases; they will carry a
  *runtime* scale operand (and pair with FDX's `ScalePair` / `role` machinery, FDX §6.2) rather
  than a static `ScaleSpec` requirement. This is why they are reserved, not v1: v1 ships the static
  formats whose scales are known at load time.

The takeaway: **everything dynamic is expressible under B** because B already makes the scale an
ordinary operand — a *runtime-produced* operand is no different to the planner than a *loaded* one.

---

## 7. The `SType` → FDX projection mapping

`SType::to_fdx()` is the bridge from the internal type to the kernel boundary. FDX is `SType`'s
**projection**: each `Encoding` variant maps to an FDX quant family + dtype-ext + scale placement.
The op supplies the concrete `scale_buffer` index (the buffer-table position of the bound scale
operand, §4); `SType` carries only the scheme.

| `Encoding` variant | FDX `family` | FDX `dtype_ext.logical_dtype` / packing | `scale_placement` | `scale_buffer` | Source of scale |
|---|---|---|---|---|---|
| *(empty `SType`)* | `NONE` (0xFFFF) | faithful base dtype; sidecar usually absent | — | — | — |
| `GgmlBlock { ggml_dtype }` | `GGML_BLOCK` (0) | `packing = GGML_BLOCK`; `ggml_dtype` code (Q4K=12, …) | `INLINE` (0) | `FDX_BUFFER_INLINE` | baked in block |
| `AffineBlock { packed: F4, block_shape, scale, zero_point }` | `AFFINE_BLOCK` (4) | `logical_dtype = 13 (F4)`, `bit_width = 4`, `packing = DENSE_SUBBYTE` | `SEPARATE_BUFFER` (1) | op-bound buffer-table index (≥ 1; never `FDX_BUFFER_INLINE`) | sibling operand (§4) |
| `Mx { … }` *(reserved)* | `MX` (1) | F4/F6 payload + `packing = MX_BLOCK` | `SEPARATE_BUFFER` (1) | op-bound index | sibling F8E8M0 per-block |

**Projection field details (verified against FDX):**

- **`GgmlBlock` → GGML_BLOCK, INLINE.** FDX `family = GGML_BLOCK` carries `ggml_dtype` **only**,
  scale **baked inline**, no separate operand, no `scale_granularity` (FDX
  [§6.2 line 954, 994-998](dlpack-extension.md)). `Encoding::GgmlBlock.ggml_dtype` maps directly to
  `FDXQuant.ggml_dtype` (the FDX code mirrors `GgmlDType::to_u32` — Q4_0=2 … Q8K=15, BF16=30, FDX
  [§6.2 lines 903-905](dlpack-extension.md), matching
  [`quantized.rs:67-85`](../../fuel-core-types/src/quantized.rs)). `dtype_ext.packing = GGML_BLOCK`
  (FDX [§6.1 line 885](dlpack-extension.md)).
- **`AffineBlock` → AFFINE_BLOCK, SEPARATE_BUFFER.** `scale_present = 1`, `scale_placement =
  SEPARATE_BUFFER`, `scale_buffer` = a real index (named once — single-place rule), `block_ndim ≥
  1`, `block_shape` from the layer, `scale_dtype` from `ScaleSpec.dtype`, `scale_granularity` left
  default and **not consulted** (block grain rides `block_shape`; `PerBlock` stays MX-only),
  `zp_present` from `zero_point.is_some()` (FDX [§6.2 lines 969-989](dlpack-extension.md), worked
  example §13.5a [lines 2354-2381](dlpack-extension.md)). The packed payload projects via
  `dtype_ext`: `AffineBlock.packed = DType::F4` → `logical_dtype = 13`, `bit_width = 4`,
  `packing = DENSE_SUBBYTE` (FDX [§6.1 line 870, 883](dlpack-extension.md)).
- **`Mx` (reserved).** Will project to FDX `MX` (family 1), the sole user of
  `scale_granularity = PerBlock` and `packing = MX_BLOCK` with an F8E8M0 per-block scale (FDX
  [§6.2 line 955](dlpack-extension.md)). **Not wired in v1** (§8).

**Where the buffer index comes from.** `SType` does **not** know the `scale_buffer` index — that is
the position of the *bound scale operand* in the op's FDX buffer table. At projection time the op
(holding both the weight and the scale operand) supplies the binding; `to_fdx()` writes the index
into the `AFFINE_BLOCK` descriptor. This is exactly the §4 boundary re-union: the *scheme* (from
`SType`) and the *scale reference* (from the op's operand binding) compose into one FDX descriptor.

---

## 8. Invariants and decisions-log entry

### Invariants

1. **`DType` is logical, `SType` is physical, and they are orthogonal.** An encoded tensor's
   `DType` is the *logical* element type it represents (NF4 → `F16`/`F32`; GGML block → the logical
   float). `SType` never duplicates `DType` information.
2. **`Encoding` is data-free.** No bulk weight bytes, no scale values, no operand pointers — only
   static descriptors (scheme, geometry, dtype codes, `ScaleSpec` requirements). This is what keeps
   `Encoding` `Eq + Hash` and small enough to key plan caches.
3. **`SType` default = empty = plain.** Every existing single-output `Storage` is byte-identical in
   behavior; `is_plain()` ⇔ no extra interpretation.
4. **Layer ordering is meaningful and validated.** `SType.0` is an ordered stack (outermost-first
   convention); v1 has at most one layer, but the ordering invariant is owned by the `SType` newtype
   so a future composite (e.g. a compression layer over a block layer) is well-defined. Construction
   validates the stack (no duplicate incompatible layers; reserved variants rejected).
5. **v1 SType is PRIMARY-only.** `SType` attaches to the primary `Storage`; bundle `OutputView`
   slots carry `dtype` only. Per-slot `SType` is a future addition (§3).
6. **`GgmlBlock` is inline and scale-operand-free.** It has no `ScaleSpec`; it projects to
   `GGML_BLOCK` / `INLINE` (§5, §7). No code path attaches a sibling scale to a `GgmlBlock` tensor.
7. **`AffineBlock` requires a sibling scale operand.** `ScaleSpec` is a *requirement*, not a
   pointer; the consuming op MUST bind a scale operand whose dtype = `ScaleSpec.dtype` and whose
   shape = the per-block count derived from `base.shape` + `block_shape`. It projects to
   `AFFINE_BLOCK` / `SEPARATE_BUFFER` with `scale_buffer ≥ 1` (never `FDX_BUFFER_INLINE`).
8. **No invented dtype variants.** `AffineBlock.packed` reuses an existing `DType` sub-byte code
   (NF4 → `DType::F4`); `GgmlBlock.ggml_dtype` reuses `GgmlDType`. This spec introduces **no** new
   `DType` and **no** `ScaleGranularity::PerBlock`.
9. **Reserved variants are declared but not constructible in v1.** `Encoding::Mx` and the listed-only
   `AffineInt` / `AffineFloat` / `Compressed` exist for vocabulary/forward-compat; v1 construction
   rejects them. They are wired when their consumers and FDX families are ready.
10. **The boundary projection is total and FDX-consistent.** Every constructible v1 `Encoding` has a
    defined `to_fdx()` mapping (§7) that agrees with the shipped FDX families; a projection that
    cannot produce a valid FDX descriptor is a bug, not a silent fallback.

### Decisions-log entry

> **2026-06-18 — Self-describing storage encoding: `DType` + `SType`/`Encoding` (NEW spec).**
> Branch `feat/kernel-contracts-dlpack`. Established the third boundary spec
> ([`storage-encoding.md`](storage-encoding.md)), sibling to FDX (boundary tensor) and FKC (kernel
> advertisement), as the source of truth for the internal `SType` / `Encoding` types; FDX is its
> projection.
>
> - Added `SType` (named newtype over `SmallVec<[Encoding; 1]>`, default empty = plain) and
>   `Encoding` (data-free, `Eq + Hash`) with v1 variants `GgmlBlock` (inline) and `AffineBlock`
>   (NF4/QLoRA, sibling scale), a reserved `Mx`, and listed-only `AffineInt` / `AffineFloat` /
>   `Compressed`. `ScaleSpec` is a scale *requirement* (dtype + granularity), never an operand.
> - `Storage` gains `stype: SType` (default empty, primary-only in v1); default-empty keeps every
>   existing `Storage` byte-identical.
> - **LOAD-BEARING:** decided **B (sibling operand) + FDX sidecar composite at the boundary** over
>   **A (composite-by-reference)**. Verified rationale: (1) multi-output machinery is one-buffer
>   only (`fuel-graph/src/lib.rs:962-976`, `fuel-dispatch/src/pipelined.rs:273-295`,
>   `12-multi-output.md`); (2) FDX already specifies `AFFINE_BLOCK` scales as `SEPARATE_BUFFER`
>   (`dlpack-extension.md` §6.2 line 958); (3) B is cheaper and honest (scales are ordinary
>   operands, no recursive `Storage`, matches GPTQ/HF/bnb). The self-describing property that
>   matters (the *scheme*) is delivered on the tensor; the scale *values* stay a sibling operand;
>   FDX re-unites them at the kernel boundary. This was a revision of an earlier lean toward A.
> - GGML stays **inline** (forced by GGUF struct-packing + ~40 kernels + zero-copy mmap;
>   `quantized.rs:87-113`); interleaving is **not** generalized to NF4 (would kill zero-copy load
>   for no win). General efficiency rule: **layout follows source** — no universal winner.
> - **Divergences from the design sketch, reconciled against code:** (a) the `fuel-core-types`
>   `Storage` has no explicit `dtype` field (delegates to `inner.dtype_dyn()`), unlike the
>   `fuel-memory` one the sketch quoted — both still gain `stype`; (b) `ScaleGranularity` has **no**
>   `PerBlock` variant (`quant_scale.rs:38`), consistent with FDX keeping `PerBlock` MX/FDX-only —
>   so `AffineBlock` block grain rides `block_shape`, not a granularity code; (c) NF4 has no
>   dedicated `DType` — it reuses `DType::F4` (`dtype.rs:44`), as FDX §13.5a already assumes.
