# FDX addition — GATHER / INDEXED-RESIDENCY descriptor (paged / blocked KV cache)

> **SUPERSEDED / FOLDED INTO `docs/specs/dlpack-extension.md` as of 2026-06-17 — retained as
> rationale only; do NOT re-integrate.** The parent spec (`dlpack-extension.md`) is the single
> source of truth and already contains this addition in full (§5.2 bit 7, §6.9, validators
> V18–V21, Appendix A, example §13.8, plus the 2026-06-17 critique fixes folded after integration).
> Read the parent for the authoritative form; this file explains the *why*.

**Status:** COMPANION RATIONALE — FOLDED INTO the parent (2026-06-17). Design pass — no code yet.
This file is the pre-integration rationale for the gather addition; the post-integration
authoritative form lives in `docs/specs/dlpack-extension.md`. (Validator numbers here use V18–V21,
matching the parent; the earlier pre-integration V16–V20 draft is obsolete — see §"Numbering note".)
**USER DECISION (carried):** FDX v1 **WILL** include a GATHER / INDEXED-RESIDENCY descriptor so a
paged/blocked KV cache (vLLM-style; the cache `OpKind::PagedAttn` / `FusedOpParams::PagedAttn`
consumes) is describable as a **SINGLE FDX tensor**.

**Scope:** add a new FDX sub-descriptor, **`FDXIndexedResidency`** (with its companion
**`FDXBlockTable`**), embedded by value in `FDXSidecar` exactly like `FDXQuant`. The base
`DLTensor` stays the honest contiguous block pool (dense `uint8` bytes); the gather mapping
(logical sequence position → physical block id) lives **only** in the sidecar.

---

## 0. Grounding (as-built — verified 2026-06-17, do not trust stale citations)

The op params **actually stored** are `FusedOpParams::PagedAttn { softmax_scale: f32,
block_size: usize, softcap: Option<f32> }` — **three fields only**
(`fuel-graph/src/registry.rs:241-244`; the `FusedOpParamsKey` tag is 13, `:524-527`; the op id is
`PAGED_ATTN = FusedOpId(13)`, `:887`). The earlier draft's citation of a nine-field
`OpParams::PagedAttn { b, hq, hkv, sq, d, block_size, max_blocks_per_seq, num_blocks, … }` and the
path `fuel-core-types/src/dispatch.rs:165-168` were **wrong** and are corrected here.

The geometry this addition mirrors (`b`, `hq`, `hkv`, `sq`, `d`, `max_blocks_per_seq`,
`num_blocks`) is **not** stored as op params; it is carried on the lowered
`KernelRef::PagedAttn { b, hq, hkv, sq, d, block_size, max_blocks_per_seq, num_blocks,
softmax_scale, softcap }` (`fuel-dispatch/src/kernel.rs:314-331`, all `usize`) and otherwise
**derived at runtime from the operand shapes**:

- `q` = `[B, Hq, Sq, D]`
- `k_cache` / `v_cache` = `[num_blocks, block_size, Hkv, D]`
- `block_table` = `[B, max_blocks_per_seq]` (U32 — logical→physical block index)
- `context_lens` = `[B]` (U32 — true context length per sequence)
- optional `alibi_slopes` = `[Hq]`

See the CPU kernel `fuel-cpu-backend/src/byte_kernels.rs:6789-6837` (operand byte-size contract:
`k_cache.len_bytes() == num_blocks * block_size * hkv * d * elem`, etc.) and `:6900-6912` (the
per-accessed-slot block-id range check — a **single** `physical_block >= num_blocks` test, line
6902). The kernel reads `context_lens[s]` per sequence ("true context length per seq",
`:6797`).

**Consequence for the struct design.** The `FDXIndexedResidency` geometry fields therefore
**mirror DERIVED shape facts**, cross-checked against the operand `FDXBufferRef` shapes (V21), not
nonexistent op-param fields. Only `block_size` (and `softmax_scale`/`softcap`, which are kernel
params, **not** tensor description) are real param fields. This addition makes that exact tensor a
single FDX object instead of five disjoint `DLTensor`s with the indexing relationship lost at the
boundary.

This addition obeys every parent-spec principle:

- **P1** — the base `DLTensor` is an honest dense `uint8` pool; the gather/scatter mapping lives
  ONLY in the sidecar; a sidecar-blind consumer sees the raw pool, never a mislabeled scattered
  tensor. `FDX_FLAG_MEANING_REQUIRES_EXT` is **mandatory** (V19).
- **P3 / G7** — description only: NO cost, NO decision. The kernel that consumes a paged cache
  declares it in **FKC**; FDX only describes the tensor.
- **P4** — capacity for layout, symbol for liveness: the per-sequence live length
  (`context_lens`) is a symbolic / affine extent.
- **P7** — the pool / block-table / context-lens are **capability-relative buffer-table indices**,
  never raw pointers in serialized form.
- **P8** — `FDXIndexedResidency` and `FDXBlockTable` are frozen-size sub-structs that grow only
  via their own `reserved[]`; `gather` is carved from `FDXSidecar.reserved`, guarded by
  `struct_bytes`.

### Numbering note (the affine ↔ gather co-integration)

Two additions were drafted independently on 2026-06-17 — **affine extents** and **gather** — and
both originally claimed flag bit 7 and validator numbers V16/V17. On integration:

- **Flag bits:** gather kept **bit 7** (`FDX_FLAG_HAS_GATHER`); affine moved to **bit 8**
  (`FDX_FLAG_HAS_AFFINE_EXTENT`). The §5.2 authoritative bit-allocation table is the single owner;
  a build-time test asserts no two `FDX_FLAG_*` share a bit.
- **Validators:** the affine addition took **V16/V17**; gather took **V18–V21** (this file). The
  earlier gather draft's V16–V20 are superseded by the V18–V21 numbers used below and in the
  parent §8.

---

## 1. Architecture — where this slots in (parent §4, new §6.9)

Parent §4 ("Architecture: two structs, one optional link") shows one `DLTensor` + one
`FDXSidecar`. The gather descriptor adds **no third top-level struct**: it is a sub-block of
`FDXSidecar` exactly like `FDXQuant`, `FDXResidency`, and the `extents[]` / `views[]` arrays. It
is documented as the new parent **§6.9** (`FDXIndexedResidency`).

The conceptual layering for a paged KV cache:

```text
   ┌───────────────────────────────────────┐
   │ DLTensor (STANDARD, HONEST)           │   the PHYSICAL block pool, dense uint8 bytes:
   │  data: void* (256-aligned)            │   dtype = {kDLUInt,8,1}
   │  dtype = {kDLUInt, 8, 1}              │   shape = [ pool_size_bytes ]   strides = [1]
   │  shape = [pool_size_bytes]            │   (a sidecar-blind consumer sees ONLY this:
   │  strides = [1]   byte_offset = 0      │    a raw contiguous byte pool, never a
   └───────────────────────────────────────┘    mislabeled scattered tensor)
                   ▲  link (manager_ctx / boundary-a param)
   ┌───────────────────────────────────────┐
   │ FDXSidecar                            │   flags |= HAS_GATHER | HAS_SYMBOLIC
   │  ...                                   │            | MEANING_REQUIRES_EXT   (mandatory)
   │  gather: FDXIndexedResidency  ◄────── NEW (§6.9)
   │   .block_table  (logical→physical)     │   capability-relative indices (P7)
   │   .logical_extents[seq_axis] = Range   │   live per-seq length lives HERE (P4)
   │  buffers[]  (pool, block_table, ctx)   │   capability-relative indices (P7)
   └───────────────────────────────────────┘
```

The gather descriptor is **purely a re-interpretation layer over the honest pool**: it states
"the physical buffer `buffers[pool_buffer]` is a pool of `num_blocks` fixed-size blocks of
`block_size` tokens each; to read logical sequence `s` position `t`, look up physical block
`block_table[s, t / block_size]` and offset `t % block_size` within it." That re-interpretation
**cannot be reconstructed from the base bytes alone**, so — per the parent's honesty rule — it is
gated behind `FDX_FLAG_MEANING_REQUIRES_EXT` (mandatory; validator V19).

### 1.1 Honesty invariant preserved (parent §3 / §6.9.1 — load-bearing)

This is the hard requirement and the design's spine. The base `DLTensor` describes the
**contiguous physical block pool as honest, dense `uint8` bytes** — never the logical (gathered,
scattered) tensor:

- base `dtype = {kDLUInt, 8, 1}` (the §3 honesty stand-in), `shape = [pool_size_bytes]`,
  `strides = [1]` (explicit, §3.2), `data` 256-aligned (§3.3), `byte_offset = 0`.
- A **sidecar-blind consumer sees exactly the raw pool** — a correctly-sized opaque byte buffer —
  and **never a mislabeled scattered tensor**. It cannot accidentally read it as a dense
  `[B, n_heads, S, head_dim]` cache, because the base never claims that shape or that dtype. This
  is the §3 one-directional safety property applied to gather: ignoring the sidecar loses
  *meaning* (you get opaque pool bytes), but can never produce *wrong numbers from a mislabeled
  scattered tensor*.
- The gather/scatter mapping (the `block_table`, `block_size`, the logical shape) lives **only in
  the sidecar**. There is no honest dense interpretation of a paged cache (the logical rows are
  physically scattered across non-adjacent blocks — strictly worse than the §3.1.1 middle-axis
  case), so `FDX_FLAG_MEANING_REQUIRES_EXT` is **mandatory** (V19). A blind consumer therefore
  goes through producer policy §9.1: refuse, or **materialize a dense un-paged copy** of the live
  region with `DLPACK_FLAG_BITMASK_IS_COPIED` set.

The base pool is also honest about **the physical typed element**: although the base `DLTensor` is
`uint8` per §3, the *true* per-token element type (F16/BF16/F32 in the as-built PagedAttn) rides
`FDXDTypeExt` exactly as for any other meaning-bearing tensor, and the pool's physical typed shape
`[num_blocks, block_size, Hkv, D]` rides `gather.physical_shape`. The `uint8` base is sized off
the byte count, never off a `size_in_bytes()==0` dtype. **V19 requires the base byte length to
exactly cover the typed pool** (`base.shape[0] == num_blocks * block_size * intra_block_typed_count
* elem_bytes`), so the honest-`uint8` cover is *enforced*, not merely asserted in prose.

---

## 2. New flag (parent §5.2)

Add one bit to the `FDXSidecar.flags` table in §5.2 (it took the bit-7 slot in the authoritative
bit-allocation table; affine moved to bit 8):

```c
#define FDX_FLAG_HAS_GATHER        (1u << 7)
        /* gather block (FDXIndexedResidency) is meaningful: the base bytes are a
           physical BLOCK POOL, re-interpreted via a block table (§6.9). Implies the
           logical tensor cannot be reconstructed from the base alone, so
           FDX_FLAG_MEANING_REQUIRES_EXT MUST also be set (V19). */
```

Authoritative bit-allocation row (the single place a bit is assigned — §5.2):

| bit | constant | meaning |
|-----|----------|---------|
| 7 | `FDX_FLAG_HAS_GATHER` | `gather` (`FDXIndexedResidency`) meaningful (§6.9) |
| 8 | `FDX_FLAG_HAS_AFFINE_EXTENT` | ≥1 extent is `kind=Affine` (the sibling addition) |
| 9..63 | (reserved, 0) | next addition takes bit 9 from THIS table |

Interaction with the existing flags (validators V18/V19/V21):

- `FDX_FLAG_HAS_GATHER` set ⇔ `gather.kind != FDX_GATHER_NONE` (V18).
- `FDX_FLAG_HAS_GATHER` set ⇒ `FDX_FLAG_MEANING_REQUIRES_EXT` set (mandatory, V19).
- `FDX_FLAG_HAS_GATHER` set ⇒ `FDX_FLAG_HAS_SYMBOLIC` set **iff** any per-sequence live length
  (`context_lens`) is symbolic (it normally is — P4); a fully-static paged cache (all seqs at a
  fixed, equal length) MAY clear `HAS_SYMBOLIC`, but the common batched-decode case sets both.
- `FDX_FLAG_HAS_GATHER` is **orthogonal** to `FDX_FLAG_HAS_QUANT` and `FDX_FLAG_HAS_DTYPE_EXT`: a
  paged cache MAY also be quantized (KV-cache quant); the gather layer sits *over* the quant layer
  (the block pool's per-block bytes are then quant-packed). v1 permits the combination; the quant
  block describes the within-block packing and the gather block describes the block-level scatter
  (V18 keeps the two block geometries consistent). A worked quant-paged example is an open item
  before v1 freeze.

---

## 3. Top-level sidecar integration (parent §5.3 / §5.4)

`FDXIndexedResidency` is a **frozen-size** sub-struct embedded by value, like `FDXQuant`. To keep
the parent struct's field order stable and additive, it is carved out of the existing
`FDXSidecar.reserved: [u64; 8]` tail (P8: additive growth into reserved space, guarded by
`struct_bytes`). An older reader, guarded by `struct_bytes`, simply does not see it (and reads the
honest pool — safe).

Rust (added after `views`, consuming reserved space):

```rust
    /// Gather / indexed-residency descriptor — paged/blocked pool (§6.9).
    /// Valid iff FDX_FLAG_HAS_GATHER. When unset, this is
    /// `FDXIndexedResidency::NONE` (all-zero, kind = FDX_GATHER_NONE).
    /// Embedded by value (frozen-size sub-struct, like FDXQuant); carved from
    /// the former `reserved[8]` tail. The §5.4 size-assertion test pins
    /// sizeof(FDXSidecar).
    pub gather: FDXIndexedResidency,

    /// Reserved for additive growth without bumping `version`. Zero on write.
    /// Shrunk from `[u64; 8]` to keep FDXSidecar's size class stable across the
    /// gather addition; the §5.4 size-assertion test pins the new total.
    pub reserved: [u64; 2],
```

C (mirrors):

```c
  FDXIndexedResidency gather;    /* §6.9; valid iff FDX_FLAG_HAS_GATHER */
  uint64_t            reserved[2];   /* shrunk from [8] for `gather`; size pinned §5.4 */
```

> **Size discipline (P8, parent §5.4).** Whether `gather` fits inside the old `reserved[8]` or
> grows `FDXSidecar` by one documented size class is a layout decision, not a semantic one. Either
> way build-time assertions pin it: `assert_eq!(size_of::<FDXSidecar>(), …)`,
> `size_of::<FDXIndexedResidency>()`, `size_of::<FDXBlockTable>()`, plus the existing
> `size_of::<FDXExtent>()` + `offset_of!`-per-field pins (because `logical_extents` is an
> **array** of `FDXExtent`, whose element stride matters). A `struct_bytes` round-trip test
> complements the static pins.

---

## 4. `FDXIndexedResidency` — struct definition & semantics (new parent §6.9.2)

The gather descriptor. It names the physical pool, the per-block geometry, the block table, the
per-sequence live lengths, and the logical (gathered) shape. **No pointers** — every buffer is a
capability-relative index into the §7.4 buffer table (P7).

> **Field-width policy (corrected from the earlier draft).** `num_blocks`, `block_size`,
> `num_sequences`, `max_seq_capacity` are **u64** (matching the `usize` source — the kernel
> geometry is all `usize`; u64 avoids author-side narrowing overflow, and V18 checks the value
> fits the runtime kernel's `usize`). `block_table.max_blocks_per_seq` is **u32** (the earlier u16
> capped at 65535 blocks/seq ≈ 1.05M tokens at block_size 16 — plausibly exceeded by long-context
> models, and a `usize → u16` narrow would silently wrap). The boundary widenings are guarded:
> a source value exceeding a field's width ⇒ typed `GatherIncoherent`, never a wrap (V18).

### 4.1 Rust (`#[repr(C)]`)

```rust
/// GATHER / INDEXED-RESIDENCY descriptor: re-interprets a contiguous physical
/// BLOCK POOL (the honest uint8 base, §3) as a logically-gathered tensor via a
/// per-sequence block table. Models a vLLM-style paged KV cache as a SINGLE FDX
/// tensor. Description only (P3/G7): no cost, no decision. Frozen-size (§5.4);
/// grows only via `reserved`. All geometry fields mirror DERIVED operand-shape
/// facts (V20/V21 cross-check them against the operand FDXBufferRef shapes), NOT
/// stored op-param fields (FusedOpParams::PagedAttn has only 3 fields, §0).
#[repr(C)]
pub struct FDXIndexedResidency {
    /// Gather kind: 0 = FDX_GATHER_NONE (absent), 1 = FDX_GATHER_PAGED_BLOCKS
    /// (vLLM/PagedAttn block-pool). Future kinds (ragged/CSR) are additive.
    pub kind: u8,
    pub _pad0: [u8; 3],

    /// ── PHYSICAL POOL geometry (the honest base; mirrors derived k/v_cache
    ///    shape [num_blocks, block_size, Hkv, D]) ───────────────────────────
    /// Number of fixed-size blocks in the pool (derived = k_cache.shape[0]).
    /// u64 to match the usize source; V18 checks it fits the runtime usize.
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
    /// Physical pool strides (typed elements), length physical_ndim, ALWAYS
    /// explicit & non-negative (§3.2 / V13), and the HONEST strides of the
    /// actual allocation. The per-block (slowest) axis is physical_strides[0].
    /// MUST be the dense, gap-free strides (V20): a padded pool must be SIZED
    /// for its padding so block id num_blocks-1 never walks past size_bytes.
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
    /// Affine) extent over max_seq_capacity carrying the seq SymId, and gets its
    /// OWN V7/V14 arms keyed to logical_shape/max_seq_capacity. Inline, fixed-
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
    /// Scope hint for the context length symbol (matches FDXExtent.sym_scope:
    /// 0=InputDetermined, 1=DataDetermined, 2=SessionScoped). Advisory (v1).
    pub context_len_scope: u8,
    pub _pad6: [u8; 3],

    pub reserved: [u32; 6],
}
```

### 4.2 `FDXBlockTable` — struct definition & semantics

The logical-position → physical-block mapping. **Per-sequence and batched**: a dense
`[num_sequences, max_blocks_per_seq]` table of physical block ids, exactly the as-built
`block_table = [B, max_blocks_per_seq]` (U32). It is itself a buffer-table reference (P7), with
its geometry mirrored inline for self-containment and validation.

```rust
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
    /// / block_size). u32 (was u16 in the earlier draft — u16 capped at 65535
    /// blocks/seq ≈ 1.05M tokens at block_size 16, and a usize->u16 narrowing
    /// would silently wrap). V18 guards the usize->u32 narrowing (overflow ⇒
    /// GatherIncoherent, not a wrap).
    pub max_blocks_per_seq: u32,

    /// Sentinel block id meaning "unmapped / not yet allocated" (a row's tail
    /// past the sequence's allocated blocks). FDX_BLOCK_UNMAPPED (0xFFFFFFFF) by
    /// default. INVARIANT (V18): unmapped_sentinel MUST be representable in
    /// id_dtype AND MUST be >= num_blocks, so the single `id >= num_blocks`
    /// range check provably catches BOTH out-of-range and unmapped (the as-built
    /// kernel does range-only — byte_kernels.rs:6902). A consumer MUST NOT
    /// dereference an unmapped slot.
    pub unmapped_sentinel: u32,

    /// Layout flags: bit0 = ids sorted within a row (advisory); bit1 = table is
    /// shared/read-only across the call. 0 = no claims.
    pub layout_flags: u32,

    pub reserved: [u32; 4],
}
```

C mirrors both structs field-for-field with the same `#[repr(C)]` order; size pins per §5.4.

### 4.3 `kind` codes + buffer roles (FDX-owned, parent §6.0 / §6.9.3)

| value | name | meaning |
|-------|------|---------|
| 0 | `FDX_GATHER_NONE` | gather block absent (HAS_GATHER clear) |
| 1 | `FDX_GATHER_PAGED_BLOCKS` | fixed-size block pool + per-seq block table (vLLM/PagedAttn) |
| 2.. | (reserved) | ragged/CSR gather, hierarchical paging — additive (§"Versioning") |

`FDXIndexedResidency::NONE` is all-zero with `kind = FDX_GATHER_NONE (0)`; V18 rejects `kind == 0`
while `HAS_GATHER` is set, and rejects an unknown `kind` as typed `UnsupportedGatherKind` (never a
guess — §14 `#[non_exhaustive]` spirit). (Gather's NONE is `0`, unlike the `0xFFFF` NONE sentinels
elsewhere, because `kind` is a small additive enum, not a code-table dtype; it is distinguished by
the gating flag — same convention as `FDXResidency.tier` / `FDXStorage.class` using `0` as a real
first value.)

The `FDXBufferRef.role` enum (§7.2) gains three gather roles (additive): `5 = FDX_BUFFER_POOL`,
`6 = FDX_BUFFER_BLOCK_TABLE`, `7 = FDX_BUFFER_CONTEXT_LENS`. A new constant
`FDX_BUFFER_NONE = 0xFFFFFFFE` (distinct from `FDX_BUFFER_INLINE = 0xFFFFFFFF`) marks "no such
buffer" for `context_lens_buffer` when the live length is purely symbolic.

> **Single-place rule (parent RESOLVED DECISIONS, the FKC touch-point).** The paged-attention
> kernel's ABI takes `block_table` / `context_lens` as **separate graph inputs** — the
> `KernelRef::PagedAttn` operand order is `[q, k_cache, v_cache, block_table, context_lens,
> alibi?]` (`fuel-dispatch/src/kernel.rs:314-331`) — so they are **FKC `accept.inputs` operands**,
> and the FDX `pool_buffer` / `block_table.table_buffer` / `context_lens_buffer` indices point at
> the **same** buffers (V21b cross-check, not a copy). The FDX gather descriptor does not duplicate
> the data; it describes the indexing relationship the disjoint operands otherwise leave implicit.
> Each table is described in exactly one authoritative place + a consistency check — exactly the
> parent's scale-buffer discipline.

---

## 5. Indexing composition, strides, and the no-OOB argument (new parent §6.9.4)

The load-bearing part: how the gather composes with **per-axis strides** and **capacity** so the
parent §3.1 no-OOB argument still holds.

1. **Capacity for layout (P4).** The pool buffer is sized for **full pool capacity** and
   `physical_strides` are the honest, gap-free strides of the actual allocation. V20 mirrors
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
   CPU kernel, byte_kernels.rs:6797).

3. **Indexing composition (logical → physical address).** To read logical sequence `s`, position
   `t` (`0 <= t < L_s`), element coordinate `c` within the per-token shape:

   ```text
   physical_block = block_table.ids[s * max_blocks_per_seq + (t / block_size)]   // gather
   if physical_block >= num_blocks  -> ERROR (BlockIdOutOfRange)                 // catches OOB
                                                                                 //  AND unmapped
                                                                                 //  (sentinel >=
                                                                                 //  num_blocks, V18)
   intra_block_token = t % block_size
   byte_addr = pool.data + pool.byte_offset
             + physical_block    * physical_strides[0]    // per-block (slowest) stride
             + intra_block_token * physical_strides[1]    // per-token-in-block stride
             + dot(c, physical_strides[2..])              // intra-token element strides
   ```

   The per-axis strides are the honest pool strides (keyed to capacity); the gather only chooses
   *which block* (the slowest physical axis). The base pool's strides describe a dense, walkable
   buffer; the block table merely permutes the block axis — which is why the honesty invariant
   survives and why `MEANING_REQUIRES_EXT` is mandatory (the gathered logical tensor has no single
   set of dense strides). The single `physical_block >= num_blocks` test catches **both** an
   out-of-range id and the unmapped sentinel, because V18 forces `unmapped_sentinel >= num_blocks`
   — exactly what the as-built kernel relies on (byte_kernels.rs:6902).

4. **Three honest shapes.** `physical_shape` is the dense pool typed shape
   `[num_blocks, block_size, Hkv, D]`; `logical_shape` is the per-sequence gathered shape; the
   **base** `DLTensor.shape` is the honest byte shape `[pool_size_bytes]`. Never conflated (the
   §3.1.1 "never label a scatter as dense" discipline, generalized).

---

## 6. Buffer-table roles (parent §7.2)

Extend the `FDXBufferRef.role` enum with three gather roles (additive):

| value | name | meaning |
|-------|------|---------|
| 0 | `Data` | (existing) base data buffer |
| 1 | `Scale` | (existing) |
| 2 | `ZeroPoint` | (existing) |
| 3 | `BundleBacking` | (existing) |
| 4 | `Aux` | (existing) |
| 5 | `FDX_BUFFER_POOL` | the physical block pool (the honest base bytes; conventionally index 0) |
| 6 | `FDX_BUFFER_BLOCK_TABLE` | the `[num_sequences, max_blocks_per_seq]` block-id table (U32) |
| 7 | `FDX_BUFFER_CONTEXT_LENS` | the `[num_sequences]` U32 live-length table |

The pool, block table, and context-lens buffers are **first-class buffer-table entries** with
their own `dtype`, `shape`, `strides`, and `size_bytes` (P7) — so a consumer reads them directly
by index, and the validator can range-check them. `FDX_BUFFER_NONE = 0xFFFFFFFE` marks "no such
buffer" for `context_lens_buffer` when the live length is purely symbolic.

---

## 7. Validators (parent §8 — V18–V21, Result-returning, no `try_*`)

Append to the §8 list (all typed-`Result`, runnable at the boundary, P10). V16/V17 are the affine
checks; gather is V18–V21.

- **V18 — gather coherence.** `FDX_FLAG_HAS_GATHER` set ⇔ `gather.kind != FDX_GATHER_NONE`.
  `kind == FDX_GATHER_PAGED_BLOCKS` ⇒ `block_size != 0`, `num_blocks != 0`, `num_sequences != 0`,
  `max_blocks_per_seq != 0`, `id_dtype == U32` (v1 pin), `physical_ndim ∈ [1,6]`,
  `logical_ndim ∈ [0,6]`, `physical_shape[0] == num_blocks`, `physical_shape[1] == block_size`,
  `block_table.max_blocks_per_seq == ceil(max_seq_capacity / block_size)`, and
  `max_seq_capacity == max_blocks_per_seq * block_size` computed in **u64 without overflow**. The
  author-side narrowings (`usize` source → the struct's `u64`/`u32` fields) are guarded: a source
  value exceeding the field width ⇒ `GatherIncoherent`, never a wrap. The `unmapped_sentinel` MUST
  be representable in `id_dtype` **and** `>= num_blocks` (so the single `id >= num_blocks` range
  check provably catches both OOB and unmapped). Unknown `kind` ⇒ `UnsupportedGatherKind`.
  → `GatherIncoherent`.
- **V19 — MEANING_REQUIRES_EXT mandatory + base honesty.** `FDX_FLAG_HAS_GATHER` set ⇒
  `FDX_FLAG_MEANING_REQUIRES_EXT` set (a paged pool's logical tensor cannot be reconstructed from
  the base bytes). Absence ⇒ `DishonestBase` (sub-reason `GatherWithoutMeaningFlag`). The base
  `DLTensor` honesty (V3) still holds: base `dtype == {kDLUInt,8,1}`, base `strides == [1]`, and
  the base byte length **exactly covers the typed pool**:
  `base.shape[0] == num_blocks * block_size * intra_block_typed_count * elem_bytes(element_dtype)`
  where `intra_block_typed_count = product(physical_shape[2..physical_ndim])` — enforcing the
  honest-`uint8` cover, not merely asserting it. → `DishonestBase`.
- **V20 — pool backing (stride·extent, not element-count).** Mirrors V8 literally for the pool:
  `buffers[pool_buffer].size_bytes >= physical_strides[0] * num_blocks * elem_bytes` **and** the
  analogous `stride * extent` on **every** physical axis (so a padded pool — `physical_strides[0]`
  larger than the dense per-block stride — is sized for its padding and block id `num_blocks-1`
  cannot walk past `size_bytes`). `physical_strides` MUST be the honest, gap-free strides of the
  actual allocation, the block axis being `physical_strides[0]`. A pool not backed to full
  capacity already sets `MEANING_REQUIRES_EXT` (V19). → `CapacityNotBacked`
  (sub-reason `PoolNotBacked`).
- **V21 — gather ↔ operands ↔ symbol consistency (build + realize).**
  (a) `pool_buffer`, `block_table.table_buffer`, and `context_lens_buffer` (when not
  `FDX_BUFFER_NONE`) are valid buffer-table indices (`< buffers_count`) with matching declared
  roles; their shapes match the gather geometry (`block_table` is
  `[num_sequences, max_blocks_per_seq]`; `context_lens` is `[num_sequences]`).
  (b) When the FDX tensor coexists with the FKC operands carrying the same tables — the
  `KernelRef::PagedAttn` operand order `[q, k_cache, v_cache, block_table, context_lens, alibi?]`
  (`fuel-dispatch/src/kernel.rs:314-331`) — the referenced buffers are *identical* (single-place
  rule, §6.9.3), a cross-check not a copy.
  (c) **Build / boundary FULL-table scan** (V18-class): every **mapped** entry
  (`id != unmapped_sentinel`) of the block-table buffer satisfies `0 <= id < num_blocks`.
  (d) **Realize-time LAZY per-ACCESSED slot** (matching the as-built kernel
  byte_kernels.rs:6900-6912, NOT an eager pre-pass): for each sequence `s`, the live length `L`
  (from `context_len_sym` via `FDXSymEnv`, or `context_lens[s]`) satisfies
  `0 <= L <= max_seq_capacity` and `ceil(L/block_size) <= max_blocks_per_seq`; and at the moment
  each block id is dereferenced, `id >= num_blocks` ⇒ `BlockIdOutOfRange` (this single test
  catches both OOB and the unmapped sentinel, since V18 forces `unmapped_sentinel >= num_blocks`).
  (e) `context_len_sym != FDX_SYM_NONE` ⇒ `logical_extents_count == logical_ndim`, and
  `logical_extents[seq_axis]` carries the **same** `sym_id` (P4 unification) with
  `logical_extents[seq_axis].capacity == max_seq_capacity` (its OWN V7/V14 arms run, keyed to
  `logical_shape`/`max_seq_capacity` — the live seq length is thus inside the extents machinery,
  NOT escaping it via the 1-D base `extents[]`).
  → `GatherIncoherent` / `BufferRefOutOfRange` / `ExtentOutOfRange` / `BlockIdOutOfRange`.

Add the new typed errors to the §8 `FDXError` set: `GatherIncoherent`, `UnsupportedGatherKind`,
`BlockIdOutOfRange`, with `PoolNotBacked` a sub-reason of the existing `CapacityNotBacked` and
`GatherWithoutMeaningFlag` a sub-reason of `DishonestBase`.

> **Description-only re-affirmation (P3/G7).** None of V18–V21 priced anything or chose a path. The
> *cost* of materializing a dense un-paged copy for a blind consumer, the *decision* of
> paged-kernel-vs-materialize, and the contiguize/strided/materialize choice are the **FKC** cost
> model's job (parent §6.5/§9.3). The paged-attention kernel declares its acceptance of a paged
> cache in FKC; FDX only *describes* the tensor.

---

## 8. Worked example — batched paged KV cache (new parent §13.8)

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
      `physical_strides = [16384, 1024, 128, 1]` (typed F16 elements, dense & gap-free, explicit,
      non-negative — V13/V20), `element_dtype = 7 (F16)`.
    - `block_table`: `table_buffer = 1`, `id_dtype = 2 (U32 — v1 pin)`, `max_blocks_per_seq = 64`,
      `unmapped_sentinel = 0xFFFFFFFF` (≥ num_blocks=256 ⇒ caught by the range check — V18),
      `layout_flags = 0`.
    - `num_sequences = 4`, `max_seq_capacity = 1024` (= 64*16, u64 no-overflow — V18).
    - `logical_ndim = 3`, `seq_axis = 1`, `logical_shape = [8, 1024, 128]` (per-seq `[Hkv, S_cap, D]`).
    - `logical_extents_count = 3`: axis 0 `Scalar(8)`, **axis 1 (`seq_axis`) `Range`**
      `min=1`, `capacity=1024`, `sym_id=11` (P4 unification with `context_len_sym`), axis 2
      `Scalar(128)` — the live seq length is **inside** the extents machinery (V21e), not on the
      1-D base `extents[]`.
    - `context_lens_buffer = 2`, `context_len_sym = 11`, `context_len_scope = 0 (InputDetermined)`.
  - `extents` (count = base.ndim = 1): axis 0 is `Scalar(8388608)` (the byte pool is concrete).
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
  the PagedAttn kernel reads `L = lookup(env, 11)` (or `context_lens[s]` per seq), applies
  `0 ≤ L ≤ 1024` and `ceil(L/16) ≤ 64` (V21d), and for each accessed `(s, t)` gathers
  `block = block_table[s, t/16]` and asserts `block < 256` **at dereference** (V21d; the single
  `>= num_blocks` test catches both OOB and the sentinel). No eager pre-pass — the build-time
  full-table scan is V21c. The same sidecar serves every decode step.
- **Cross-runtime export to a generic (blind) consumer:** because the logical cache is a scatter
  over physical blocks, `MEANING_REQUIRES_EXT` forces producer policy §9.1: the producer
  **materializes a dense un-paged copy** `[4, 8, L, 128]` of the live region with
  `DLPACK_FLAG_BITMASK_IS_COPIED` set, or refuses — never the raw pool labeled as a cache.

> Variant — **quantized paged cache:** add `flags |= HAS_QUANT`; the `quant` block describes the
> *within-block* packing and the `gather` block the *block-level* scatter; the two block
> geometries are kept consistent (V18/§6.9 interplay; a worked quant-paged example is an open item
> before v1 freeze).
>
> The K and V caches are described as **two FDX tensors sharing the same**
> `FDX_BUFFER_BLOCK_TABLE` / `FDX_BUFFER_CONTEXT_LENS` buffers (single-place rule). A bundled
> two-pool descriptor is a v2 candidate if a fully-fused KV pool emerges (open item).

---

## 9. Versioning, alignment, codes (parent §14, §11/§3.3, §6.0)

- **Versioning (§14).** `FDXIndexedResidency` / `FDXBlockTable` are additive within FDX v1:
  introduced via `FDX_FLAG_HAS_GATHER` (bit 7) + `gather` carved from `FDXSidecar.reserved`,
  guarded by `struct_bytes`. An older v1 reader that predates the field reads the known prefix and
  treats it as absent (it then sees the honest pool — safe). Because `logical_extents` is an
  **array of `FDXExtent`**, it inherits the parent's "array-element growth is NOT covered by
  tail-ignore" caveat: `FDXExtent`'s leading field offsets are frozen and pinned by
  `offset_of!`-per-field build assertions (§5.4). New `gather.kind` values (ragged/CSR) are
  additive; an unknown `kind` is a typed `UnsupportedGatherKind`, never a guess
  (`#[non_exhaustive]` spirit).
- **Codes (§6.0).** `FDX_GATHER_*` kinds and the new buffer roles
  (`FDX_BUFFER_POOL/BLOCK_TABLE/CONTEXT_LENS`) are **FDX-owned stable codes** (§6.0), pinned by the
  conversion-fn + build-time mapping test, referenced (not re-listed) by FKC.
- **Alignment (§3.3/§11).** The pool buffer obeys the 256-byte data rule on boundary (b); each
  block's natural start is `physical_block * physical_strides[0] * elem_bytes` and is reached via
  `byte_offset`/stride arithmetic over the 256-aligned pool base, never a misaligned `data`. The
  block-table and context-lens buffers are likewise 256-aligned on export. Boundary (a) relaxes to
  `required_alignment` (§3.3).
- **Pointers (P7/V15).** `pool_buffer`, `block_table.table_buffer`, `context_lens_buffer` are
  **indices**, never pointers; the serialized form carries 0 in every `FDXBufferRef.data`
  (V15 unchanged).
- **Capability (§12).** Add `Capability::DlpackExtGather` — "I consume a paged FDX tensor
  directly". A backend without it triggers producer policy (materialize a dense un-paged copy or
  refuse, §9.1). The negotiation lives in the planner, not the boundary.

---

## 10. Exact FDX sections an integrator must touch

| Parent §  | What changes |
|-----------|--------------|
| **§4** (architecture / honesty) | add the gather layer to the two-struct diagram; reaffirm the base = honest dense `uint8` pool, gather lives only in the sidecar. |
| **§5.2** (magic/version/flags) | add `FDX_FLAG_HAS_GATHER (1u << 7)` to the authoritative bit-allocation table (gather = bit 7; affine = bit 8); note its implication of `MEANING_REQUIRES_EXT`. |
| **§5.3** (top-level sidecar, Rust + C) | add `gather: FDXIndexedResidency` after `views`; shrink `reserved` `[u64;8]`→`[u64;2]`. |
| **§5.4** (size discipline) | pin `size_of::<FDXIndexedResidency>()`, `size_of::<FDXBlockTable>()`, the new `FDXSidecar` total, and the `FDXExtent` array-element offsets. |
| **§6.0** (normative code owner) | register `FDX_GATHER_*` kinds + new buffer roles as FDX-owned codes (conversion fn + test). |
| **§6.4** (symbolic extents) | cross-reference: `context_lens` = the live extent; `context_len_sym` unifies with `gather.logical_extents[seq_axis]`; the V7 arms also apply to `logical_extents[]`. |
| **new §6.9** (field semantics) | `FDXIndexedResidency` + `FDXBlockTable` struct defs + §6.9.1 honesty + §6.9.2 structs + §6.9.3 codes/roles + §6.9.4 indexing composition. |
| **§7.2** (`FDXBufferRef`) | extend `role` with `5/6/7 = POOL/BLOCK_TABLE/CONTEXT_LENS`; add `FDX_BUFFER_NONE`. |
| **§8** (validators) | add **V18–V21** + the new `FDXError` variants (V16/V17 are affine). |
| **§9.1** (producer policy) | gather always `MEANING_REQUIRES_EXT` ⇒ refuse-or-materialize (dense un-paged copy, `IS_COPIED`). |
| **§11 / §3.3** (alignment) | pool/table/ctx buffers 256-aligned on boundary (b); per-block start via stride/`byte_offset`. |
| **§12** (capability negotiation) | add `Capability::DlpackExtGather`. |
| **new §13.8** (examples) | the batched paged KV cache worked example (§8 here). |
| **§14** (versioning) | gather is additive in v1 (guarded by `struct_bytes`; `logical_extents` under the array-element-growth rule); unknown `kind` typed-errors. |
| **Appendix A** (constants) | `FDX_GATHER_NONE/PAGED_BLOCKS`, `FDX_BUFFER_POOL/BLOCK_TABLE/CONTEXT_LENS`, `FDX_BUFFER_NONE (0xFFFFFFFE)`, `FDX_BLOCK_UNMAPPED (0xFFFFFFFF)`. |
| **Appendix B** (Fuel ⇄ FDX map) | rows: derived `[num_blocks,block_size,max_blocks_per_seq]` → `FDXIndexedResidency`; `block_table` → `FDXBlockTable`; `context_lens` → `gather.logical_extents[seq_axis]` / `context_len_sym`. |
| **§17** (open questions) | item 12: gather core RESOLVED; residuals (per-logical-axis extents beyond seq, ragged/CSR, two-pool bundle, quant-paged interleave) deferred. |

---

## 11. Open items (deferred with rationale; mirror parent §17 item 12)

1. **Per-logical-axis multi-symbolic shape.** v1 carries the seq-length symbol via
   `logical_extents[seq_axis]` (+ `context_len_sym`); a richer fully-multi-symbolic logical shape
   (more than one symbolic logical axis) is an additive v2 candidate (`reserved` leaves room). v1's
   single live axis matches the as-built PagedAttn (one live length per seq).
2. **Ragged / CSR gather (`kind ≥ 2`).** Variable block_size, hierarchical paging, or a CSR offset
   array (not a dense `[B, max_blocks]` table) are additive future kinds; the `kind` enum +
   `reserved[]` leave room.
3. **Separate K vs V pools.** PagedAttn has two caches (k_cache, v_cache) sharing one block table.
   v1 describes each as its own FDX tensor sharing the same `FDX_BUFFER_BLOCK_TABLE` /
   `FDX_BUFFER_CONTEXT_LENS` buffers (single-place rule). A bundled two-pool descriptor (one FDX
   tensor, two pool buffers) is a v2 candidate if a fully-fused KV pool emerges.
4. **Quant-over-gather block-geometry interplay.** v1 permits `HAS_QUANT | HAS_GATHER` and V18
   checks block-geometry consistency; the precise interleave of MX/GGML packing *inside* a paged
   block needs a worked quant-paged example before v1 freeze.
