# Fuel → Vulkane: carrying the FDX tensor description across the FFI

**From:** Fuel (the DAG-first ML framework that builds on Vulkane for its Vulkan backend).
**To:** Vulkane (`github.com/ciresnave/vulkane`) — the Vulkan FFI layer (buffers, allocations,
command submission, descriptor plumbing) that `fuel-vulkan-backend` sits on top of.

**This is a proposal, not a change.** Nothing has been written into the Vulkane repo. We are
asking you to review one boundary contract and confirm (or push back on) a small set of
preservation guarantees **before** Fuel freezes the ABI shape. Everything you need to evaluate
it is in this message — you should not have to read Fuel's source to answer.

---

## TL;DR — what we're asking of Vulkane

Fuel describes a tensor crossing the Vulkane FFI as **FDX**: an *honest standard DLPack*
`DLTensor` (a data pointer + byte offset + explicit signed `int64` strides + a real dtype, with
256-byte-aligned data) **plus an optional, nullable, opaque sidecar pointer** carried beside it.
Vulkane never has to *interpret* the sidecar — it only has to **carry it intact** and not destroy
the binding facts the base carries. Concretely we ask Vulkane to:

1. **Carry a nullable `const FDXSidecar*` alongside the buffer descriptor** on the Fuel↔Vulkane
   handoff. `null` means "plain DLPack — the base says everything true about these bytes." Vulkane
   passes it through to where Fuel's Slang kernel reads it; Vulkane does not parse it.
2. **Preserve signed `int64` strides** — do not coerce them to unsigned, and do not silently
   materialize a "normalized" copy of a reversed/flipped view. The flip must cross zero-copy.
3. **Preserve `byte_offset`** into a 256-byte-aligned base pointer, and **honor the 256-byte
   alignment** as a floor when building the `VkBuffer` binding (don't fold the offset into a
   recomputed base that breaks alignment).
4. **Carry a *plural* buffer table** — a quantized weight crosses as a data buffer **plus** a
   separate scale buffer; a paged KV cache crosses as a block pool **plus** a block table. Don't
   assume one `VkBuffer` per tensor.
5. **Review the ABI struct shapes below** (they are DRAFT) and tell us if anything about buffer
   binding, offset, stride, or alignment doesn't map cleanly onto a Vulkan buffer binding.

That's the whole ask. The rest of this message explains *why* and gives you the exact struct
definitions so you can evaluate the FFI shape without leaving this document.

---

## 1. The asymmetry: Vulkane is a conduit, not a kernel provider

Fuel's Vulkan compute kernels are **internal Slang shaders authored inside Fuel**
(`fuel-vulkan-kernels`). Vulkane does not ship compute kernels — it is the FFI through which
Fuel's `(storage, layout)` becomes a `VkBuffer` binding, and back. So unlike a kernel provider,
Vulkane sits on only **one** side of the boundary: there is exactly one thing to agree on — **the
description that rides along with a buffer must survive the crossing without loss.**

This means the kernel-advertisement half of Fuel's boundary work (how a kernel declares its
dispatch key / capabilities / cost — what we call FKC) is **N/A for Vulkane today**. It would
only become relevant *if* Vulkane ever exposed its own compute entry points that Fuel could
dispatch to (see §6). Until then this is **FDX-only**.

---

## 2. The model: an honest base + an opaque sidecar

FDX = **standard DLPack** (which you may already know) **+ an optional versioned sidecar**.

- The **base `DLTensor`** is *always* honest standard DLPack. A sidecar-blind reader — which is
  all an FFI conduit needs to be — sees a correctly-sized, real-dtype tensor with explicit
  strides. It is **never a lie**: a 4-bit quantized weight appears as opaque `uint8` bytes of the
  exact packed size, never as a mislabeled `float16` over packed nibbles, never as a buffer
  mis-sized by a sub-byte dtype. So **if Vulkane only ever looks at the base, it is always correct
  about the buffer's size and binding.**
- The **sidecar** (`FDXSidecar*`, nullable) carries the facts standard DLPack can't: sub-byte/
  quant dtypes, per-block scales, symbolic live-vs-capacity extents, paged residency, and the
  plural buffer table. **Vulkane carries it; Fuel's Slang kernel interprets it.** A `null` sidecar
  means "plain DLPack."

The single sentence: **an FFI that preserves signed strides, byte offsets, and 256-byte alignment
lets flipped/sliced/quantized views cross zero-copy; an FFI that doesn't forces a copy and erases
information Fuel depends on.**

---

## 3. The ABI you carry (inlined — this is the whole contract surface)

These are the exact C struct definitions (64-bit little-endian; Fuel maintains a co-checked C
header and Rust `#[repr(C)]` source that are size-asserted against each other at build time, so
these are authoritative, not sketches). Vulkane **binds the base + buffer table** and **carries
the sidecar pointer opaquely**.

### 3.1 Standard DLPack (you bind this)

```c
typedef struct { int32_t device_type; int32_t device_id; } DLDevice;       /* 8 bytes  */
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType; /* 4 bytes  */

/* The base tensor. `data` is 256-byte aligned on export; the logical start rides
 * `byte_offset` (NEVER folded into `data`). `strides` has length `ndim`, is never
 * NULL on a versioned export, and MAY BE NEGATIVE (a reversed/flipped view). */
typedef struct {
  void*      data;        /* device pointer; 256-byte aligned                       */
  DLDevice   device;
  int32_t    ndim;
  DLDataType dtype;
  int64_t*   shape;       /* length ndim; capacity bounds for symbolic axes         */
  int64_t*   strides;     /* length ndim; never NULL; may be negative (§flip)        */
  uint64_t   byte_offset; /* logical start, in bytes, into `data`                    */
} DLTensor;               /* 48 bytes */
```

### 3.2 The managed/capsule form (for the deleter-gated handoff)

When a tensor crosses as a managed capsule (e.g. to/from the ecosystem), the sidecar rides
`manager_ctx` and is only recovered when the live `deleter` identity matches Fuel's own. The
`deleter`/`manager_ctx` pointers are **never serialized**.

```c
typedef struct { uint32_t major; uint32_t minor; } DLPackVersion;  /* 8 bytes  */

typedef struct DLManagedTensorVersioned {
  DLPackVersion version;
  void*         manager_ctx;   /* FDX sidecar rides here at the capsule boundary */
  void (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t      flags;         /* DLPACK_FLAG_BITMASK_{READ_ONLY,IS_COPIED,...}   */
  DLTensor      dl_tensor;
} DLManagedTensorVersioned;    /* 80 bytes */
```

### 3.3 The sidecar (you carry this opaquely — but here it is in full)

You do **not** parse this. It is shown so you can see (a) it is a single fixed-size versioned
blob you can carry by pointer, and (b) it contains a **plural buffer table** (`buffers` /
`buffers_count`) that is the reason the FFI must bind more than one `VkBuffer` per tensor.

```c
typedef struct {
  uint32_t            magic;          /* 0x46445800 = "FDX\0"                       */
  uint32_t            version;        /* 1                                          */
  uint32_t            struct_bytes;   /* sizeof(FDXSidecar)                         */
  uint32_t            flags;          /* HAS_QUANT | HAS_SYMBOLIC | HAS_GATHER | …   */
  FDXDTypeExt         dtype_ext;      /* sub-byte / microscaling descriptor         */
  FDXQuant            quant;          /* parametric quant (block geometry + scales) */
  uint32_t            extents_count;
  uint32_t            _pad0;
  const FDXExtent*    extents;        /* symbolic live-vs-capacity axes             */
  FDXTiling           tiling;
  FDXResidency        residency;
  FDXStorage          storage;
  uint32_t            buffers_count;  /* >>> PLURAL: data + scale + … <<<           */
  uint32_t            _pad1;
  const FDXBufferRef* buffers;        /* the buffer table (index 0 = base data)     */
  uint32_t            views_count;
  uint32_t            _pad2;
  const FDXOutputView* views;         /* multi-output bundle sub-views              */
  FDXIndexedResidency gather;         /* paged/blocked KV-cache descriptor          */
  uint64_t            reserved[2];
} FDXSidecar;                         /* 1376 bytes */
```

### 3.4 The buffer table entry (you bind each of these as a `VkBuffer`)

This is the one sidecar sub-struct that matters to a binding layer: each entry is a buffer Fuel
needs bound. Index 0 is always the base `DLTensor.data`; a `role = SCALE` entry is a separate
scale buffer; gather adds `role = POOL` / `BLOCK_TABLE` / `CONTEXT_LENS` entries.

```c
/* role values: DATA=0, SCALE=1, ZERO_POINT=2, BUNDLE_BACKING=3, AUX=4,
 *               POOL=5, BLOCK_TABLE=6, CONTEXT_LENS=7 */
typedef struct {
  uint8_t  role;
  uint8_t  _pad[1];
  uint16_t dtype;
  uint32_t _pad2;
  void*    data;        /* device pointer (NEVER serialized)                        */
  DLDevice device;
  uint64_t byte_offset; /* offset into `data` — preserve it                         */
  uint64_t size_bytes;
  uint32_t ndim;
  uint32_t _pad3;
  uint64_t shape[6];
  int64_t  strides[6];  /* signed — may be negative                                 */
  uint32_t reserved[4];
} FDXBufferRef;         /* 160 bytes */
```

The takeaway for the FFI: **the buffer table is plural; bind every entry faithfully, scale and
block-table buffers included, with each entry's `byte_offset` and signed `strides` preserved.**

---

## 4. Why strides + offsets + alignment are load-bearing for a Vulkan binding

A `VkBuffer` binding is `(handle, offset, range)`; the stride story is the kernel's, expressed in
the descriptor. Three preservation guarantees protect facts Fuel depends on:

- **Signed strides must survive.** Fuel treats negative strides as **first-class**: a reversed/
  flipped view (e.g. reversing an axis) is a real zero-copy DLPack tensor with signed `int64`
  strides — Fuel does **not** normalize it to a copy on the internal path. If Vulkane coerces
  strides to unsigned or materializes a "normalized" copy, a view Fuel intended to pass zero-copy
  is destroyed. **Carry strides as signed `int64`, untouched.**
- **`byte_offset` must survive.** A non-zero-offset view (a slice, or a bundle slot) maps onto a
  binding offset. The offset must reach the binding intact, not be folded into a recomputed base
  pointer that breaks the 256-byte alignment guarantee.
- **256-byte alignment must be honored.** DLPack requires `data` aligned to 256 bytes. Vulkan has
  its own binding-alignment requirements; the two are compatible, but only if Vulkane treats the
  256-byte contract as a floor it preserves rather than a property it recomputes.

---

## 5. Where the scales live (so the FFI carries the right number of buffers)

One Fuel design decision determines **how many buffers cross** for a quantized weight, so it's
worth stating plainly:

- A **GGML/GGUF-style** block-quantized weight (`FDXQuant.family = GGML_BLOCK`, code 0) has its
  scales **baked inline** in the block layout — it crosses as **one** self-contained buffer.
- An **NF4 / QLoRA-style** block-affine weight (`FDXQuant.family = AFFINE_BLOCK`, code 4) is
  low-bit packed data **plus a separate per-block absmax scale operand** — it crosses as **two**
  bindings (data buffer + scale buffer), both referenced from the buffer table above
  (`scale_placement = SEPARATE_BUFFER`, `scale_buffer` = a real index into `buffers[]`).
- A **paged KV cache** (vLLM-style) crosses as a block pool **plus** a block table **plus** a
  per-sequence context-length buffer.

Fuel deliberately keeps separate-source scales as sibling buffers (not merged into the weight's
storage) to preserve zero-copy loading from formats that ship them separately. For Vulkane the
takeaway is narrow: **the buffer table is plural; carry every entry's binding faithfully.**

---

## 6. FKC — N/A today, conditional later

Fuel's kernel-advertisement format (FKC: how a kernel declares its dispatch key, accept/return
contracts, capability/cost/precision) **does not apply to Vulkane** today, because Vulkane ships
no kernels — Fuel's Vulkan kernels are internal Slang. **If** Vulkane ever exposes compute entry
points of its own that Fuel could dispatch to, those entry points would carry FKC contracts at
that point (modeled on Fuel's own internal Slang kernel contracts), and the negative-strides
acceptance would be declared per-entry-point rather than assumed. Until then there is nothing on
the Vulkane side to advertise, and this proposal is FDX-only.

---

## 7. What Fuel commits to vs what is open

**Committed now (the description shape):**

- The base `DLTensor` crossing the FFI is **always honest standard DLPack** — correct size, real
  dtype, explicit signed strides, 256-byte-aligned data — *even when the sidecar is absent or
  ignored*. An FFI conduit that reads only the base is always correct about the binding.
- The sidecar is **optional, nullable, versioned**; Vulkane carries it without interpreting it.
- Negative strides are **first-class** in the description; the flip survives the FFI as a
  zero-copy binding when both sides preserve signed strides.
- The buffer table is **plural** (separate scale buffers, block tables).

**Open / for joint review:**

- **The concrete FFI ABI signature** — the exact parameter shape of "buffer descriptor(s) +
  nullable sidecar pointer" on the Fuel↔Vulkane handoff. Offered for joint design, not
  unilaterally frozen. The struct shapes above are DRAFT and we want your review of anything
  touching buffer binding, offset, stride, or alignment **before** they freeze.
- **FKC for Vulkane** — conditional on Vulkane ever exposing compute entry points (§6). No action
  today.

---

## 8. Process

This is a propose-first cross-project request: Fuel does not edit sibling projects (Vulkane,
Baracuda, etc.) without proposing the change first, even though we share an author. Nothing here
has been written into the Vulkane repo. The base-`DLTensor` + sidecar-pointer ABI shape and the
stride/offset/alignment preservation guarantees are offered for your review before either side
freezes its half.

**The next step is your read of §3–§4 and a yes/no (or pushback) on the five asks in the TL;DR.**
If the buffer-descriptor + nullable-sidecar-pointer signature and the three preservation
guarantees (signed strides, byte offset, 256-byte alignment) are workable on the Vulkane side,
that's the green light we need to wire `fuel-vulkan-backend` against them.
