# Fuel DLPack Extension (FDX) — tensor interchange for Fuel, kernels, and the ecosystem

**Status:** DRAFT v0.1 (2026-06-17). Design pass — no code yet.
**Scope:** a versioned, *optional sidecar* extension to standard DLPack that lets Fuel
describe tensors whose full meaning exceeds the standard `DLTensor` — sub-byte / microscaling
dtypes, parametric quantization, per-axis scales, symbolic (live-vs-capacity) extents,
multi-buffer quant payloads, multi-output bundles, fine-grained residency/substrate, storage
class — **without ever lying in the base `DLTensor`**.
**Audience:** Fuel (`fuel-core-types`, executor, planner), Baracuda (CUDA kernels), Vulkane /
`fuel-vulkan-kernels` (Slang), and any external DLPack consumer (PyTorch / JAX / CuPy / NumPy /
TVM).

Authoritative inputs: the architecture-constraints digest
(`docs/specs/_research/architecture-constraints.md`); the as-built core types in
`fuel-core-types/src/{dtype.rs, symbol.rs, shape.rs, quant_scale.rs, quantized.rs,
capability.rs, backend.rs, probe.rs, device.rs}`; the DLPack ABI (`dlpack.h` v1.x:
`DLDataType`, `DLDevice`, `DLTensor`, `DLManagedTensor`, `DLManagedTensorVersioned`,
`DLPackVersion`, the `DLPACK_FLAG_BITMASK_*` flags, and the `__dlpack__` /
`__dlpack_device__` Python protocol). When this draft and the constitution
(`docs/architecture/`) conflict, the constitution wins; flag the conflict.

---

## 0. Current status / handoff

- This is the **interchange (weight/storage-axis) half** of the kernel boundary; the
  kernel-contract (advertisement-axis) format is a *sibling* spec. They share the dtype/quant/
  symbolic vocabularies defined here but are kept separate concerns (per 13-interchange:
  weight ⊥ graph; the node↔weight binding stays format-local).
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
  (a quant payload appears as opaque `uint8` bytes, never a mislabeled dtype).
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
  encodes a choice (which kernel, whether to dequantize, where to place). Capability
  negotiation drives those choices through the planner via `BackendProbe`.

### Non-goals

- Not a graph-interchange format (that is the base map; see 13-interchange). FDX is the
  *tensor/storage* axis only.
- Not a replacement for DLPack. FDX strictly *extends* it; the base stays canonical DLPack.
- Not a transport for telemetry, cost, or precision guarantees (those are the kernel-contract
  spec's concern). FDX carries only *what a tensor is*, not *what a kernel costs*.
- Does not describe within-kernel concurrency or placement (backend/runtime concerns).

---

## 2. Design principles

- **P1 — Base-DLTensor honesty invariant (§3).** Load-bearing; everything else is built to
  preserve it.
- **P2 — Sidecar, not replacement.** State space is `{absent, v1, v2, …}` via an explicit
  version field, *not* a 2-state null/non-null. Absence = "this is plain DLPack."
- **P3 — Description, never decision.** Mirrors the constitution's governing principle
  (backends advertise, the planner decides). FDX says *what is*; it never says *what to do*.
  No "dequantize-on-read" flag, no "preferred kernel."
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
  *capability-relative handles* (an index into a side `DLBufferTable`), not raw pointers, in
  any form that can be persisted. Live cross-runtime exchange uses DLPack's managed/deleter
  contract, whose pointers are never written to disk.
- **P8 — Additive versioning.** New fields go into reserved space or a higher version; old
  readers ignore unknown trailing fields (size-prefixed). `#[non_exhaustive]`-spirited enums.
- **P9 — Match external convention.** Reuse DLPack names/semantics (`DLDataType`, `DLDevice`,
  byte_offset, deleter, stream-exchange) where they exist; only extend where Fuel needs more.
- **P10 — Build-time validatable.** Every consistency check (sidecar version supported, buffer
  refs in range, scale shape matches granularity, sub-byte packing matches bit-width, capacity
  ≥ min) is a `Result`-returning validation runnable at the boundary. No `try_*` siblings.

---

## 3. The base-DLTensor honesty invariant (load-bearing)

> **The base `DLTensor` must always be honestly interpretable as standard DLPack on its own.**

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
  dtype or a mis-sized buffer.

### 3.1 The capacity-honesty corollary (symbolic extents)

For an axis with a symbolic live extent, the base `DLTensor.shape[i]` reports the **capacity**
(`Extent::bound()` / `Shape::dims()[i]` — the `Range`'s `max`), and `strides[i]` is keyed to
that capacity. This is *honest*, not a lie: the buffer truly has `capacity` slots; the live
count is a *different fact* carried in the sidecar.

- A sidecar-blind consumer that reads the whole capacity-shaped tensor reads **real, allocated
  memory** (no OOB), just possibly including not-yet-live trailing slots. For a KV cache that
  means reading the full `K`-capacity buffer; the not-yet-written tail is whatever the
  allocation initialized it to.
- **Producer policy (§9):** if reading the capacity tail instead of the live prefix would be
  *semantically wrong* for a given tensor (not merely wasteful), the producer MUST treat it
  like any other meaning-bearing-via-sidecar tensor: refuse the bare-DLPack export to an
  extension-blind consumer, or materialize a live-prefix copy. A KV cache handed to PyTorch
  via bare `__dlpack__` is exported as the **live-prefix slice** (a standard dense tensor of
  the resolved length), never the raw capacity buffer.

---

## 4. Architecture: two structs, one optional link

```
        ┌─────────────────────────┐         ┌──────────────────────────────┐
        │  DLTensor  (STANDARD)    │         │  FDXSidecar  (FUEL EXTENSION) │
        │  - data: void*           │  ◄────  │  - version, flags            │
        │  - device: DLDevice      │  link   │  - dtype_ext (sub-byte/MX)   │
        │  - ndim, dtype(uint8!)   │         │  - quant (parametric)        │
        │  - shape[], strides[]    │         │  - extents[] (live-vs-cap)   │
        │  - byte_offset           │         │  - residency, storage class  │
        └─────────────────────────┘         │  - tiling/alignment hints    │
                                             │  - buffer_table (sidecars)   │
                                             │  - bundle views (multi-out)  │
                                             └──────────────────────────────┘
```

- The link is **directional and optional**: a `DLTensor` may have *no* sidecar (it is then
  100% standard). The sidecar always refers back to exactly one base `DLTensor` whose `data`
  pointer is buffer index 0 of the sidecar's `DLBufferTable` (§7.4).
- **Two boundaries** carry the link differently (§10):
  - **(a) Fuel kernel ABI** — the sidecar is an explicit nullable parameter *next to* the
    `DLTensor` (`*const FDXSidecar`, `null` = absent). Cleanest, no smuggling.
  - **(b) External `__dlpack__`** — the capsule signature is fixed (no extra-pointer slot), so
    the sidecar rides `DLManagedTensorVersioned.manager_ctx` *if and only if* the consumer
    advertised understanding (else it is **not carried**; only the standard part crosses, and
    producer policy §9 governs whether that is even allowed).

---

## 5. Concrete struct definitions

All multi-byte fields are **little-endian** in serialized form. All structs are
`#[repr(C)]` POD with explicit padding. Reserved fields are zero on write, ignored on read.
Every variable-length array is referenced by `(count, *const T)` for the live in-memory form;
the **serialized** form replaces pointers with byte offsets relative to the sidecar blob start
(P7).

### 5.1 Standard DLPack (reproduced for reference — NOT redefined by FDX)

```c
/* From dlpack.h — FDX consumes these unchanged. */
typedef enum { kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLVulkan = 7,
               kDLMetal = 8, /* ... */ } DLDeviceType;
typedef struct { DLDeviceType device_type; int32_t device_id; } DLDevice;
typedef enum { kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBfloat = 4,
               kDLComplex = 5, kDLBool = 6 } DLDataTypeCode;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;

typedef struct {
  void*      data;
  DLDevice   device;
  int32_t    ndim;
  DLDataType dtype;
  int64_t*   shape;      /* length ndim — CAPACITY bounds for symbolic axes */
  int64_t*   strides;    /* length ndim or NULL(=contiguous); keyed to capacity */
  uint64_t   byte_offset;
} DLTensor;

typedef struct { uint32_t major; uint32_t minor; } DLPackVersion;

typedef struct {                       /* the cross-runtime managed form */
  DLPackVersion version;
  void*         manager_ctx;           /* FDX rides here at boundary (b) */
  void        (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t      flags;                 /* DLPACK_FLAG_BITMASK_* */
  DLTensor      dl_tensor;
} DLManagedTensorVersioned;
```

### 5.2 FDX magic, version, and flags

```c
#define FDX_MAGIC      0x46445800u   /* "FDX\0" */
#define FDX_VERSION_1  1u

/* FDXSidecar.flags — bitmask, additive. */
#define FDX_FLAG_HAS_DTYPE_EXT     (1u << 0)  /* dtype_ext is meaningful   */
#define FDX_FLAG_HAS_QUANT         (1u << 1)  /* quant block is meaningful */
#define FDX_FLAG_HAS_SYMBOLIC      (1u << 2)  /* >=1 axis is symbolic      */
#define FDX_FLAG_HAS_TILING        (1u << 3)  /* tiling block present      */
#define FDX_FLAG_IS_BUNDLE         (1u << 4)  /* multi-output bundle       */
#define FDX_FLAG_MEANING_REQUIRES_EXT (1u << 5)
        /* base bytes alone are NOT a usable tensor (quant/sub-byte/      */
        /* live<cap-where-tail-is-wrong). Drives producer refuse-or-dequant. */
#define FDX_FLAG_READ_ONLY         (1u << 6)  /* mirrors DLPACK read-only  */
/* bits 7..63 reserved (0). */
```

`FDX_FLAG_MEANING_REQUIRES_EXT` is the single most consumer-relevant flag: it is the producer's
explicit statement that handing the bare `DLTensor` to a sidecar-blind consumer would lose
meaning, and therefore the producer policy in §9 (refuse-or-dequantize) applies.

### 5.3 The top-level sidecar — Rust

```rust
/// Optional, versioned sidecar communicated ALONGSIDE a standard DLTensor.
/// `#[repr(C)]` POD. Absence (a null `*const FDXSidecar`) means "plain DLPack".
#[repr(C)]
pub struct FDXSidecar {
    /// FDX_MAGIC. Lets a consumer cheaply reject a misrouted pointer.
    pub magic: u32,
    /// FDX_VERSION_*. The {absent, v1, v2, …} discriminator (P2).
    pub version: u32,
    /// Total byte size of this struct as written by the producer. Enables
    /// size-prefixed forward-compat: an older reader trusts fields up to
    /// `min(sizeof(known), struct_bytes)` and ignores the trailing tail (P8).
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

    /// Reserved for additive growth without bumping `version`. Zero on write.
    pub reserved: [u64; 8],
}
```

### 5.3 (cont.) The top-level sidecar — C

```c
typedef struct FDXSidecar {
  uint32_t       magic;          /* FDX_MAGIC */
  uint32_t       version;        /* FDX_VERSION_* */
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

  uint64_t       reserved[8];
} FDXSidecar;
```

---

## 6. Field-by-field semantics

### 6.1 Sub-byte / microscaling dtype descriptor (`FDXDTypeExt`)

Carries the *true* element type when the base `DLTensor.dtype` is the honesty-preserving
`uint8` stand-in. Maps Fuel's `DType` onto an explicit **bit-width + packing convention**, so a
`size_in_bytes()==0` sub-byte type sizes its buffer correctly (P5).

```rust
#[repr(C)]
pub struct FDXDTypeExt {
    /// Stable code for the logical element type. Mirrors Fuel `DType` ordinal
    /// (see §6.1 table); also covers a "general low-bit int/float" escape.
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

**`logical_dtype` codes** (stable; mirror `fuel-core-types::DType` plus escapes):

| code | Fuel `DType` | `bit_width` | notes |
|------|--------------|-------------|-------|
| 0 | U8 | 8 | |
| 1 | I8 | 8 | int8 GEMM operand |
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
| 0xFFFF | NONE | 0 | dtype_ext absent |

**`FDXPacking`** (`packing`):

| value | name | meaning |
|-------|------|---------|
| 0 | `BYTE_ALIGNED` | one logical element per `ceil(bit_width/8)` bytes (F8E4M3, F8E8M0) |
| 1 | `DENSE_SUBBYTE` | sub-byte elements packed back-to-back, no per-block framing (e.g. 2×F4/byte) |
| 2 | `MX_BLOCK` | OCP-microscaling: a packed sub-byte payload block + a separate F8E8M0 scale per block (block geometry in `quant`, §6.2) |
| 3 | `GGML_BLOCK` | ggml/GGUF block layout: scales/mins/quants interleaved inside one block struct (per-format byte layout in `quant`) |

> The packing convention plus `quant` (§6.2) together fully determine the byte layout. FDX does
> not enumerate one format; it parameterizes the three families.

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
    /// For GGML family: the GgmlDType code (mirrors GgmlDType::to_u32 — Q4_0=2,
    /// Q4_1=3, Q5_0=6, … Q8K=15, F16=1, BF16=30). 0xFFFF if not GGML.
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
    /// Buffer-table index of the scale tensor (§7.4). 0xFFFFFFFF if INLINE
    /// (scales interleaved in the data block, GGML family).
    pub scale_buffer: u32,

    /// ZERO-POINT descriptor (affine-int). zp_present=0 for symmetric quant.
    pub zp_present: u8,
    pub zp_dtype: u16,            // logical_dtype code (commonly I8/I32)
    pub _pad3: u8,
    pub zp_buffer: u32,          // buffer-table index, or 0xFFFFFFFF if inline

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
| 0 | `GGML_BLOCK` | static, load-time | baked, INLINE in block | none (baked); `ggml_dtype` is the format |
| 1 | `MX` | static block-scaled | separate F8E8M0 per block | `PerBlock` |
| 2 | `AFFINE_INT` | dynamic, runtime | separate scale (± zero-point) | `PerTensor/PerToken/PerChannel` |
| 3 | `AFFINE_FLOAT` | dynamic FP8/etc. | separate scale | `PerTensor/PerToken/PerChannel` |

**`FDXScalePlacement`**: `0=INLINE` (interleaved in data block, GGML), `1=SEPARATE_BUFFER`
(`scale_buffer` valid), `2=BROADCAST_PER_AXIS` (scale tensor shape per granularity).

**`FDXScaleGranularity`** (mirrors `ScaleGranularity` + a block value):
`0=PerTensor (f32[1]) | 1=PerToken (f32[rows]) | 2=PerChannel (f32[cols]) | 3=PerBlock (MX)`.

**`FDXPackOrder`**: `0=ROW_MAJOR_IN_BLOCK | 1=K_MAJOR | 2=GGML_NATIVE` (per-format, defined by
`ggml_dtype`).

> **Regime separation (digest §9, do not unify):** `family=GGML_BLOCK` carries `ggml_dtype` and
> `scale_granularity=PerBlock`/INLINE and **never** a `ScalePair`; `family=AFFINE_*` carries
> `scale_granularity ∈ {PerTensor,PerToken,PerChannel}` and may carry a `ScalePair` when it is a
> matmul operand. A consumer keys its dispatch on `(family, ggml_dtype | (scale_granularity,
> role))`, matching the flat per-format `Capability` tokens for GGML and the
> `(op, lhs_dtype, lhs_granularity, rhs_dtype, rhs_granularity)` key for affine quant.

### 6.3 Per-axis scales / granularity

Granularity is expressed two ways, deliberately:

1. **Coarse, op-level** — `FDXQuant.scale_granularity` + `scale_pair_*` + `role` (above) give
   the `ScaleGranularity`/`ScalePair` the planner keys dispatch on. This is the *dispatch-key*
   form.
2. **Concrete, buffer-level** — the scale buffer itself is a real entry in the `DLBufferTable`
   (§7.4) with its own dtype and shape (`f32[1]` / `f32[rows]` / `f32[cols]` / `f8e8m0[n_blocks]`),
   so a consumer can read the scales directly. The two MUST be consistent; validation (§8)
   checks `scale_buffer.shape` against `scale_granularity` and the base shape.

### 6.4 Symbolic / dynamic extent — live-vs-capacity (`FDXExtent`)

The single biggest gap vs generic DLPack (digest §6, §13). One entry per base axis when
`extents_count == base.ndim`. Mirrors `fuel-core-types::shape::Extent` /
`DynAxis` and the `SymId` primitive.

```rust
#[repr(C)]
pub struct FDXExtent {
    /// Axis kind: 0 = Scalar (concrete, == base shape[i]); 1 = Range (symbolic).
    pub kind: u8,
    pub _pad: [u8; 3],
    /// Range only: the lower bound of the live value (Extent::min). For Scalar,
    /// equals capacity.
    pub min: u64,
    /// CAPACITY (== base DLTensor.shape[i] == Extent::bound() == Range.max).
    /// Strides in the base DLTensor are keyed to THIS (P4).
    pub capacity: u64,
    /// Range only: the SymId of the live value (resolved per call via SymEnv).
    /// Stable, serializable, session-independent, UNIFIABLE: two axes that
    /// must move together (KV K_len == V_len) carry the SAME sym_id. (Scalar:
    /// FDX_SYM_NONE = 0xFFFFFFFF.)
    pub sym_id: u32,
    /// Scope hint for the symbol (§6.4): 0=InputDetermined, 1=DataDetermined,
    /// 2=SessionScoped. Advisory; the SymEnv supplies the value regardless.
    pub sym_scope: u8,
    pub _pad2: [u8; 3],
    pub reserved: [u32; 2],
}
```

Rules (load-bearing):

- **`capacity` MUST equal `base.shape[i]`.** `dims()` is the capacity, never the live value.
  Validation rejects a mismatch.
- **Strides are NOT in the sidecar** — they live in the base `DLTensor.strides`, keyed to
  capacity. A kernel walking the live prefix uses (base stride = how far per element) + (live
  extent = how many), exactly the two halves Fuel's `Layout` provides.
- **`sym_id` is transported, not resolved.** The live value comes from a **`SymEnv` supplied
  alongside the tensor at the realize boundary** (the FDX call surface passes a `*const FDXSymEnv`
  next to the sidecar, §7.3), never baked into the sidecar. This is what lets one description
  plan once and serve every token/session/replay (operand rebasing).
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

### 6.5 Tiling / alignment hints (`FDXTiling`)

Optional. The planner already decides repacks/`Op::Contiguize`; these hints let a producer
communicate the alignment/granularity a buffer *was* laid out for, so the boundary honors the
optimizer's repack decisions instead of silently violating them (digest §2).

```rust
#[repr(C)]
pub struct FDXTiling {
    /// Required base-address alignment in bytes (mirrors BackendCapabilities
    /// ::required_alignment). 0 = unspecified.
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

Hints, not commitments: a consumer MAY ignore tiling and treat the buffer per the base
`strides`; if it cannot, it surfaces a typed error (it does NOT silently re-tile).

### 6.6 Residency / substrate (`FDXResidency`)

Finer than DLPack's `(device_type, device_id)`. Lets the planner decide same-buffer-vtable-swap
vs needs-a-copy — Vulkan and CUDA on the *same silicon* must not alias (digest §2, §13).

```rust
#[repr(C)]
pub struct FDXResidency {
    /// Three-tier residency: 0=Device, 1=Host, 2=DiskMmap. Disk-mmap'd weights
    /// are zero-copy views, distinct from host RAM (digest §11).
    pub tier: u8,
    /// Substrate class — mirrors fuel SubstrateClass: 0=HostBytes,
    /// 1=CudaUntyped, 2=VulkanBuffer, 3=MetalBuffer. Decides aliasing.
    pub substrate: u8,
    /// Fuel BackendId of the owner (0=Cpu,1=Cuda,2=Vulkan,3=Metal), for precise
    /// pointer-namespace identity beyond DLDevice.
    pub backend_id: u8,
    pub _pad: u8,
    /// Device ordinal within the backend (mirrors DeviceLocation gpu_id).
    pub device_index: u32,
    /// Whether this buffer is a zero-copy view into a larger mmap'd region
    /// (1) vs an owned allocation (0). View ⇒ deleter must not free the region.
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
    /// Byte offset into the bundle buffer (buffer-table index 0).
    pub byte_offset: u64,
    pub len_elements: u64,
    /// Logical dtype code for this slot (may itself be sub-byte → its own
    /// dtype_ext semantics; v1 keeps slot dtype simple/standard).
    pub dtype: u16,
    pub _pad: [u8; 2],
    pub ndim: u32,
    pub shape: [u64; 6],      // matches Fuel DimVec inline capacity (6)
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

---

## 7. Buffer references, the call surface, and the buffer table

### 7.1 Why a buffer table, not raw pointers (P7)

A quantized tensor is *multi-buffer* (data + scale(s) ± zero-points), and a bundle has sub-views
into one buffer. Rather than scatter raw pointers through the sidecar (un-serializable,
un-mmap-able), all buffers are collected in one `DLBufferTable` and referenced by **index**.
Index 0 is always the base `DLTensor.data` buffer.

### 7.2 `FDXBufferRef`

```rust
#[repr(C)]
pub struct FDXBufferRef {
    /// Role: 0=Data, 1=Scale, 2=ZeroPoint, 3=BundleBacking, 4=Aux.
    pub role: u8,
    pub _pad: [u8; 3],
    /// Logical dtype of THIS buffer.
    pub dtype: u16,
    pub _pad2: [u8; 2],
    /// For the LIVE in-memory form only: device pointer (NEVER serialized).
    /// In serialized form this field is 0 and the byte location comes from the
    /// containing DLManagedTensor / file mapping.
    pub data: *mut core::ffi::c_void,
    pub device: DLDevice,
    pub byte_offset: u64,
    pub size_bytes: u64,
    pub ndim: u32,
    pub _pad3: u32,
    pub shape: [u64; 6],
    pub strides: [i64; 6],
    pub reserved: [u32; 4],
}
```

### 7.3 The Fuel kernel call surface (`FDXSymEnv`)

At boundary (a), a kernel receives, alongside `DLTensor*` + `FDXSidecar*`, an **optional**
`FDXSymEnv*` carrying the per-pass `SymId → usize` bindings (the realize-time resolution of
every symbolic extent / dynamic scalar). It is a flat sorted array for O(log n) lookup and is
**never serialized** (it is per-call data, the sibling of the tensor-data cache):

```rust
#[repr(C)]
pub struct FDXSymBinding { pub sym_id: u32, pub _pad: u32, pub value: u64 }

#[repr(C)]
pub struct FDXSymEnv {
    pub count: u32,
    pub _pad: u32,
    pub bindings: *const FDXSymBinding,  // sorted by sym_id, write-once per pass
}
```

`FDXSymEnv` is the boundary form of `fuel-core-types::symbol::SymEnv`. A consumer resolves a
symbolic axis's live length as `lookup(env, extent.sym_id)`; an unbound symbol is a typed error
(matching `SymEnv`'s write-once / presence contract), never a silent 0.

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
"this operand is plain DLPack."

---

## 8. Validation (build-time / boundary-time, Result-returning)

All checks are `Result`-returning and runnable at the boundary; there are no `try_*` siblings
(P10, digest §3). The reference validator MUST verify, in order:

1. `magic == FDX_MAGIC`; `version` is supported (`<= FDX_VERSION_MAX`); `struct_bytes >=
   sizeof(known prefix)`.
2. Flag/field coherence: each `FDX_FLAG_HAS_*` set ⇒ the corresponding block is non-NONE, and
   vice-versa.
3. **Honesty:** if `dtype_ext`/`quant` is meaning-bearing, the base `DLTensor.dtype` is a
   standard byte code (`kDLUInt/8/1`) and `base.shape`/`strides` size the *physical bytes*.
4. **Sub-byte sizing:** `bit_width != 0`; `packing` consistent with `bit_width` (e.g.
   `DENSE_SUBBYTE` ⇒ `bit_width < 8`); physical byte count derivable (never via
   `size_in_bytes()==0`).
5. **Quant coherence:** GGML family ⇒ `ggml_dtype` valid + scales INLINE + no `ScalePair`;
   AFFINE family ⇒ `scale_granularity ∈ {PerTensor,PerToken,PerChannel}` + `scale_buffer` valid
   (or BROADCAST) ; MX family ⇒ `scale_dtype == F8E8M0` + `PerBlock` + block geometry present.
6. **Scale shape vs granularity:** `scale_buffer.shape` matches
   `ScaleGranularity::scale_count(rows, cols)` against the base logical shape.
7. **Extents:** `extents_count ∈ {0, base.ndim}`; each `capacity == base.shape[i]`;
   `min <= capacity`; `Range` ⇒ `sym_id != FDX_SYM_NONE`; `Scalar` ⇒ `sym_id == FDX_SYM_NONE`
   and `min == capacity`.
8. **Buffer refs:** every referenced index (`scale_buffer`, `zp_buffer`, view backing) `<
   buffers_count`; index 0 role is `Data`; no buffer overlaps another except declared bundle
   sub-views.
9. **Bundle:** `IS_BUNDLE` ⇒ `views_count > 0`, sub-views in-bounds and non-overlapping within
   the bundle buffer.
10. **No raw pointers in serialized form:** in a serialized blob, all pointer-typed fields are
    0 and replaced by offsets; reject a serialized blob with non-zero pointer fields.

Any failure → a typed `FDXError` (`UnsupportedVersion`, `DishonestBase`, `BadSubByte`,
`QuantIncoherent`, `ScaleShapeMismatch`, `ExtentMismatch`, `BufferRefOutOfRange`,
`BundleOverlap`, `PointerInSerializedForm`, `UnboundSymbol`), never a panic, never a silent
fix-up.

---

## 9. Producer / consumer policies

### 9.1 Producer policy — refuse-or-dequantize (no silent degradation)

> A tensor whose meaning depends on the extension MUST be dequantized or refused when handed to
> an extension-blind consumer.

Concretely, a producer about to export across a boundary (b) to a consumer that did *not*
advertise FDX support (§12):

- If `FDX_FLAG_MEANING_REQUIRES_EXT` is **clear** (the base bytes are a faithful standard
  tensor — dense F16/F32, or a symbolic axis whose capacity-tail is harmless): export the
  standard `DLTensor` as-is; the sidecar is simply not carried. For a symbolic axis where the
  *live prefix* is the intended content (KV cache), export the **resolved live-prefix slice** as
  a standard dense tensor (§3.1).
- If `FDX_FLAG_MEANING_REQUIRES_EXT` is **set** (quant / sub-byte / meaning-bearing): the
  producer MUST either (i) **dequantize/materialize** to a standard dtype and export that, or
  (ii) **refuse** with a typed error naming the unrepresentable property. It MUST NOT export the
  raw bytes labeled as anything other than the honest `uint8` (which the consumer would
  misread). Silent degradation is banned.

The choice between dequantize and refuse is the producer's policy knob (a Fuel-side
configuration), not a property of FDX; FDX only *enables* the honest decision by flagging
meaning-dependence.

### 9.2 Consumer policy

- A consumer that does not recognize `magic`/`version` MUST treat the sidecar as **absent** and
  fall back to pure DLPack (it then sees the honest base, §3). It MUST NOT guess.
- A consumer that recognizes the version but not a *set flag bit it does not understand* MUST
  NOT proceed as if the tensor were standard if `FDX_FLAG_MEANING_REQUIRES_EXT` is set; it
  errors (typed), per digest §3 "errors at dispatch, never silently materializes."
- A consumer MUST NOT silently dequantize/contiguize on read; layout/quant fixups are the
  planner's job (it inserts explicit ops). The consumer either handles the described layout or
  errors.
- A consumer reading a symbolic axis MUST resolve the live length from the supplied `FDXSymEnv`;
  it MUST NOT assume capacity == live unless the axis is `Scalar`.

### 9.3 Planner policy (the decision-maker)

Per the governing principle, the *planner* — not the producer or consumer — chooses the
extended path vs the dequant-to-standard path, from the `BackendProbe` capability advertisement
(§12), not by trial. FDX carries description; the planner reads it plus the probe and commits a
path.

---

## 10. The two boundaries

### 10.1 Boundary (a): Fuel's own kernel ABI — explicit nullable parameter

Inside Fuel, the sidecar is an explicit `*const FDXSidecar` argument next to each `DLTensor`
(§7.4), `null` when absent. No smuggling, no managed wrapper needed for the common internal
launch: **Fuel owns the memory, so the bare `DLTensor` POD + sidecar POD suffice** (no
deleter). The `FDXSymEnv` rides alongside for symbolic resolution. This is the high-frequency,
zero-overhead path (one recorded run replayed with rebased operands + a fresh `SymEnv`).

### 10.2 Boundary (b): external `__dlpack__` interchange — via `manager_ctx` or not at all

The Python `__dlpack__` capsule signature is fixed: it yields a `PyCapsule` named `"dltensor"`
(or `"dltensor_versioned"`) wrapping a `DLManagedTensor[Versioned]`. There is **no slot for an
extra pointer.** Therefore:

- **If the consumer advertised FDX support** (via the negotiation in §12, e.g. an out-of-band
  capability handshake or a Fuel-aware import path), the producer attaches the sidecar through
  `DLManagedTensorVersioned.manager_ctx`: `manager_ctx` points to a Fuel-owned struct
  `{ FDXSidecar* sidecar; /* + the real manager context */ }`, and the `deleter` frees both. A
  Fuel-aware consumer downcasts `manager_ctx` (guarded by `magic`) to recover the sidecar; a
  generic consumer treats `manager_ctx` as opaque (which it is contractually required to do) and
  ignores it.
- **If the consumer did not advertise FDX support**, the sidecar is **not carried.** Only the
  standard `DLManagedTensorVersioned` crosses, governed by producer policy §9.1 (faithful base
  exported as-is, or dequantize/refuse for meaning-bearing tensors). Riding `manager_ctx` is
  *only* safe with a consenting consumer because a generic consumer's deleter contract owns
  `manager_ctx` lifetime; we never assume a generic consumer will run *our* deleter logic on it.

> Rationale for two mechanisms: boundary (a) is the hot internal path and pays nothing; boundary
> (b) honors DLPack's fixed ABI and its ownership contract, degrading honestly when the peer is
> generic.

---

## 11. Ownership, lifetime, and stream semantics (cross-runtime)

- **Internal launches (boundary a):** bare POD; Fuel owns all memory; no deleter. The sidecar's
  live `data`/pointer fields point into Fuel-owned storage and are valid for the call.
- **Cross-runtime (boundary b):** use `DLManagedTensorVersioned`'s managed/deleter contract.
  Ownership transfers per DLPack: the consumer, after consuming, calls `deleter(self)` exactly
  once; the producer's deleter releases both the tensor memory it owns and the
  `manager_ctx`-attached sidecar (when carried). The capsule rename dance (`"dltensor"` →
  `"used_dltensor"`) prevents double-free, unchanged by FDX.
- **Read-only:** mirror `DLPACK_FLAG_BITMASK_READ_ONLY` into `FDX_FLAG_READ_ONLY`; a read-only
  tensor's buffers MUST NOT be written by the consumer.
- **Stream exchange (cross-runtime, async devices):** honor DLPack stream semantics. The
  consumer passes its stream to `__dlpack__(stream=...)`; the producer ensures all writes to the
  tensor are ordered-before that stream (enqueues an event / `cudaStreamWaitEvent` equivalent).
  FDX adds **no new stream field to the serialized sidecar**; the `FuelStream*` is a *call-time*
  parameter on boundary (a) (§7.4) and the standard DLPack stream protocol on boundary (b).
  Symbolic resolution (`FDXSymEnv`) is likewise call-time, never persisted — operand rebasing
  for recorded-run replay re-supplies stream + env per replay while reusing the same sidecar.

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
  - `DlpackExtAffine` — can consume affine-int/float dynamic quant;
  - `DlpackExtSymbolic` — honors symbolic live-vs-capacity extents + `FDXSymEnv`.
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
  (G4).

---

## 13. Worked examples

In each, "base" = the standard `DLTensor`; "sidecar" = `FDXSidecar` (or absent).

### 13.1 Dense F16, null sidecar (pure DLPack)

- base: `data=ptr`, `device={kDLCUDA,0}`, `ndim=2`, `dtype={kDLFloat,16,1}`,
  `shape=[4096,4096]`, `strides=NULL` (contiguous), `byte_offset=0`.
- sidecar: **NULL** (`*const FDXSidecar == null`).
- Interop: 100% standard; PyTorch/JAX/CuPy consume it directly. Nothing Fuel-specific.

### 13.2 GGUF/GGML Q4_K_M block-quant weight

A `[out=11008, in=4096]` weight in GGML `Q4K` (block_size 256, type_size 144B, scales baked
INLINE).

- base (honesty): `dtype={kDLUInt,8,1}`, `shape=[ n_bytes ]` where `n_bytes =
  (11008*4096/256)*144`, `strides=NULL`, `byte_offset=0`. A sidecar-blind consumer sees honest
  opaque bytes of the exact packed size.
- sidecar: `flags = HAS_DTYPE_EXT | HAS_QUANT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `logical_dtype = GENERIC_LOW_BIT_INT`-ish via `packing=GGML_BLOCK`,
    `bit_width=4` nominal, `sub_byte_bit_order` per ggml.
  - `quant`: `family=GGML_BLOCK`, `ggml_dtype=12 (Q4K)`, `block_ndim=1`, `block_shape=[256]`,
    `block_axes=[1]` (along K=in), `scale_present=1`, `scale_placement=INLINE`,
    `scale_buffer=0xFFFFFFFF`, `pack_order=GGML_NATIVE`, no `ScalePair`.
  - `buffers`: index 0 = the packed data buffer only (scales are inline).
- Dispatch: planner keys on `(MatMul, GGML Q4K)` → the flat `Capability::MatMulQ4KM` token; the
  kernel owns the dequant. No silent dequant-on-read elsewhere.

### 13.3 Sub-byte FP4 (MX4) activation tensor

A `[batch=8, hidden=4096]` tensor in F4, packed two-per-byte, dense (no block scale here — pure
F4 storage; the MX scale case is §13.5).

- base (honesty): `dtype={kDLUInt,8,1}`, `shape=[ 8 * 4096/2 ]` bytes, `strides=NULL`.
- sidecar: `flags = HAS_DTYPE_EXT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `logical_dtype=13 (F4)`, `bit_width=4`, `packing=DENSE_SUBBYTE`, `lanes=1`,
    `sub_byte_bit_order=0` (element 0 in low nibble).
  - `quant`: `family=NONE` (no scales — plain sub-byte storage).
  - `extents`: 0 (both axes concrete) — but `shape` is *logical-element* count described via the
    sidecar; the **base byte shape** is `[8*2048]`. The sidecar's logical shape (if needed for a
    consumer) is recovered from base byte shape × `bit_width`. (For clarity producers MAY also
    populate a logical-shape buffer ref; v1 keeps it implicit.)
- Note the load-bearing point: `DType::size_in_bytes()` for F4 is **0**; the buffer is sized
  from `bit_width=4`, never that 0.

### 13.4 KV-cache tensor with live < capacity (symbolic extent)

A per-layer K cache `[n_heads=32, K_capacity=4096, head_dim=128]` in F16, with live length
`k_len = SymId(7) ∈ [1, 4096]`.

- base: `dtype={kDLFloat,16,1}`, `ndim=3`, `shape=[32, 4096, 128]` (**capacity**),
  `strides=[4096*128, 128, 1]` (**keyed to capacity** — honest, the buffer truly has 4096 slots).
- sidecar: `flags = HAS_SYMBOLIC` (note: `MEANING_REQUIRES_EXT` *clear* — the base is a faithful
  F16 tensor; the symbol only narrows the live region).
  - `extents` (count=3):
    - axis 0: `Scalar`, capacity=32, `sym_id=NONE`.
    - axis 1: `Range`, `min=1`, `capacity=4096`, `sym_id=7`, `sym_scope=InputDetermined`.
    - axis 2: `Scalar`, capacity=128, `sym_id=NONE`.
  - `storage`: `class=Session`, `session_id=<this session>` (KV cache is session-state, the
    durable-interop unit).
  - `residency`: `tier=Device`, `substrate=CudaUntyped`, `backend_id=Cuda`, `device_index=0`.
- Realize: the call passes `FDXSymEnv { (7 → live_k_len) }`; a flash-attention kernel reads
  `k_len = lookup(env, 7)` and walks the live prefix using base stride + live extent. The same
  sidecar serves every token (operand rebasing: re-supply data ptr + env, reuse description).
- Cross-runtime export to a generic consumer: producer exports the **live-prefix slice** `[32,
  k_len, 128]` as a standard dense F16 tensor (§3.1 / §9.1), never the raw capacity buffer.

### 13.5 MX / microscaling tensor (packed sub-byte payload + F8E8M0 per-block scale)

A `[out=4096, in=4096]` weight in MX4: F4 payload + one F8E8M0 scale per 32-element block along
K.

- base (honesty): `dtype={kDLUInt,8,1}`, `shape=[ 4096 * 4096/2 ]` bytes (the F4 payload only),
  `strides=NULL`.
- sidecar: `flags = HAS_DTYPE_EXT | HAS_QUANT | MEANING_REQUIRES_EXT`.
  - `dtype_ext`: `logical_dtype=13 (F4)`, `bit_width=4`, `packing=MX_BLOCK`.
  - `quant`: `family=MX`, `block_ndim=1`, `block_shape=[32]`, `block_axes=[1]` (along K),
    `scale_present=1`, `scale_dtype=14 (F8E8M0)`, `scale_placement=SEPARATE_BUFFER`,
    `scale_granularity=PerBlock`, `scale_buffer=1`.
  - `buffers`: index 0 = F4 payload (= base data); index 1 = F8E8M0 scale buffer, role=Scale,
    `dtype=F8E8M0`, `shape=[4096, 4096/32]` (one scale per block).
- Validation: family=MX ⇒ `scale_dtype==F8E8M0` + `PerBlock` + block geometry present
  (§8.5). Dispatch: planner needs an MX-aware consumer (`Capability::DlpackExtMx`); else
  dequantize-to-standard or refuse (§9.1).

### 13.6 Multi-output bundle (softmax + argmax)

A node producing F32 `y [B,V]` and I64 `argmax_idx [B]` in one allocation.

- base: `dtype={kDLUInt,8,1}`, `shape=[ total_bytes ]` (the whole bundle), `strides=NULL`.
- sidecar: `flags = IS_BUNDLE`.
  - `views` (count=2):
    - view 0: `byte_offset=0`, `dtype=F32`, `ndim=2`, `shape=[B,V]`, contiguous, name_hash("y").
    - view 1: `byte_offset = B*V*4`, `dtype=I64`, `ndim=1`, `shape=[B]`, name_hash("argmax_idx").
  - `buffers`: index 0 = the bundle backing buffer.
- The kernel emits one `KernelRef` writing both slots by offset; `Op::View`/`Op::ViewOwned`
  resolve slots back to ordinary tensors. Each slot is independent in dtype/shape/layout.

---

## 14. Versioning rules

- **State space `{absent, v1, v2, …}`** via the explicit `version` field, never null/non-null
  (P2). Absence (null pointer) is "plain DLPack."
- **Additive within a version (P8):** new facts go into `reserved` space (zero-default) or a
  per-block `reserved` array; old readers, guarded by `struct_bytes`, read the known prefix and
  ignore the trailing tail. This covers the common case with *no* version bump.
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
  typed error (UnsupportedVersion-class), never a guess. New dtypes (the MX set was recent; I8
  was added 2026-05-19) extend the code table additively.
- **Multi-version read policy:** aligns with the DAG-format policy (decision #18) — Fuel reads
  the previous N FDX versions; additions are backward-compatible where feasible.

---

## 15. Interop & backward-compatibility rules

- **Absent sidecar ⇒ canonical DLPack.** Any DLPack ecosystem (PyTorch/JAX/CuPy/NumPy/TVM)
  consumes the base unchanged. This is the default for dense standard-dtype tensors.
- **Unrecognized magic/version ⇒ treat as absent.** A non-Fuel consumer that happens to receive
  a `manager_ctx`-attached sidecar treats `manager_ctx` as opaque (DLPack contract) and never
  dereferences it. Safety is preserved by the honesty invariant.
- **Honesty is the backstop:** even a buggy/blind consumer reading the base sees correctly-sized
  `uint8` bytes, never a mislabeled dtype or a `size_in_bytes()==0`-mis-sized buffer.
- **No silent coercion either direction:** producers never silently degrade meaning-bearing
  tensors (refuse-or-dequantize, §9.1); consumers never silently dequant/contiguize on read
  (error instead, §9.2). Layout/quant fixups are explicit planner ops.
- **Round-trip:** a Fuel→Fuel managed export carries the full sidecar via `manager_ctx` and
  reconstructs the exact logical tensor (lossless). A Fuel→generic→Fuel round-trip through a
  blind intermediary loses the sidecar (the intermediary saw only the standard part); this is
  *stated, not hidden*, and meaning-bearing tensors are dequantized before such a trip.
- **Symbolic across boundaries:** sym identity (`sym_id`) is preserved across a Fuel→Fuel
  managed export so the consumer's unification survives; across a generic boundary the tensor is
  exported resolved (live-prefix slice).

---

## 16. Open questions / future work

1. **Logical vs physical shape for sub-byte.** v1 carries the *physical byte* shape in the base
   and the bit-width in the sidecar; the logical-element shape is implicit (derived) or via an
   aux buffer ref. Should v2 carry an explicit logical-shape array in the sidecar for
   ergonomics, at the cost of a redundancy-consistency check?
2. **Affine sym expressions.** `Extent` today is `min/max/sym`; the architecture anticipates
   affine sym expressions (`k_len = cached_len + seq`). Should `FDXExtent` carry a small affine
   form (`a*sym + b`), or keep resolution entirely in the `SymEnv` (producer pre-computes the
   composite sym)? v1 keeps it a single `sym_id`.
3. **Data-determined sym dependency edges.** `sym_scope=DataDetermined` is advisory in v1; the
   build-time producer→consumer dependency edge (NonZeroIndices/MoE counts) is a graph concern,
   not a tensor-description one. Confirm FDX needs no more than the scope tag.
4. **Bundle slot sub-byte/quant.** v1 keeps bundle slots simple/standard-dtype. Do any
   real multi-output ops emit a *quantized* slot? If so, `FDXOutputView` needs its own
   `dtype_ext`/`quant` sub-block.
5. **Zero-point scale shape conventions.** Affine-int zero-points: per-tensor vs per-channel
   zero-point shapes; confirm they mirror the scale granularity exactly (v1 assumes yes).
6. **`Capability` token granularity.** Is one `DlpackExtV1` + per-feature tokens
   (`DlpackExtMx`/`DlpackExtGgml`/`DlpackExtAffine`/`DlpackExtSymbolic`) the right cut, or should
   negotiation be a bitmask field rather than enum tokens? (Enum chosen for consistency with the
   existing flat `Capability` design.)
7. **Stream object portability.** Boundary (a) passes a `FuelStream*`; mapping that onto each
   backend's native stream/queue (CUDA stream, Vulkan queue/timeline-semaphore) is a kernel-ABI
   detail to finalize with Baracuda/Vulkane (cross-project — propose before editing siblings).
8. **Endianness across heterogeneous hosts.** Serialized form is little-endian; a big-endian
   host consumer is out of scope for v1 — confirm Fuel never targets one, or add a byte-order
   mark.
9. **Interaction with the `.fuel` mmap persistence.** The serialized sidecar (pointers → offsets)
   must be embeddable in the whole-graph `.fuel` mmap; confirm the offset base and alignment
   match the persistence layout (Phase E).
10. **Complex / bool dtypes.** DLPack has `kDLComplex`/`kDLBool`; Fuel's `DType` has neither
    yet. Reserve `logical_dtype` codes so adding them later is additive.

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

/* FDX_FLAG_* — see §5.2.  FDXPacking / FDXScalePlacement /
   FDXScaleGranularity / FDXPackOrder / FDX_QUANT_* — see §6. */
```

## Appendix B — mapping table (Fuel type ⇄ FDX field)

| Fuel as-built type | FDX carrier |
|---|---|
| `DType` (incl. sub-byte F4/F6E2M3/F6E3M2/F8E8M0) | `FDXDTypeExt.logical_dtype` + `bit_width` + `packing` |
| `DType::size_in_bytes()==0` (flag) | superseded by `FDXDTypeExt.bit_width` (never 0) |
| `GgmlDType` (Q4_0…Q8K) | `FDXQuant.family=GGML_BLOCK` + `ggml_dtype` |
| `ScaleGranularity` | `FDXQuant.scale_granularity` / `scale_pair_*` |
| `ScalePair` (act × weight) | `FDXQuant.scale_pair_act`/`scale_pair_weight` + `role` |
| `Extent::{Scalar,Range{min,max,sym}}` | `FDXExtent{kind,min,capacity,sym_id}` |
| `DynAxis{axis,min,sym}` | `FDXExtent` entries (axis = array index) |
| `SymId(u32)` | `FDXExtent.sym_id` / `FDXSymBinding.sym_id` |
| `SymEnv` (per-pass) | `FDXSymEnv` (call-time, never serialized) |
| `DynScalar` (offset/pos/k_len) | rides `FDXSymEnv` as a binding (NOT an `FDXExtent`) |
| `SubstrateClass` | `FDXResidency.substrate` |
| `DeviceLocation`/`BackendId` | `FDXResidency.{backend_id,device_index}` (+ base `DLDevice`) |
| mmap residency (digest §11) | `FDXResidency.tier=DiskMmap` + `is_mmap_view` |
| storage class (shared/session/transient) | `FDXStorage.class` |
| `SessionId` | `FDXStorage.session_id` |
| `OutputView{byte_offset,len,dtype,shape,layout,name}` | `FDXOutputView` |
| `BackendCapabilities::required_alignment` | `FDXTiling.alignment_bytes` |
| `access_granularity_bits` | `FDXTiling.access_granularity_bits` |
| `Capability` (FDX tokens) | `DlpackExtV1`/`Mx`/`Ggml`/`Affine`/`Symbolic` (§12) |
```
