# Fuel → Baracuda: the kernel boundary as a two-way contract (FDX + FKC + telemetry)

**From:** Fuel (the DAG-first ML framework; planner/executor + the `fuel-core-types` /
`fuel-dispatch` boundary).
**To:** Baracuda — the CUDA kernel library + the kernel-specialization / AOT-matrix team.

**This is both (a) Fuel's formal answer to your telemetry ask, and (b) a propose-first boundary
contract.** Nothing here has been written into the Baracuda repo. Everything you need to evaluate
it — the tensor-description ABI, the identity tuple, the miss signal, and the proof that the
timings you asked about are retained — is inlined in this message (full C ABI header in Appendix
A). You should not need to read Fuel's source to answer.

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

**What we ask of you is in §5** (five items, mostly "confirm compatibility / co-freeze a wire
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

- **The telemetry *emission* layer** — the JSONL `DispatchRecord`/`MissRecord` writer over the
  already-retained timings, the opt-in flag, and the `ImplId`/`StructureKey` join. Retention is
  done; emission is a self-contained Fuel feature still to build. (Your `DispatchRecord`/`MissRecord`
  *shapes* are adopted as-is — your JSONL, one record per line, keys in the agreed encoding;
  v1 = batch/offline at our release cadence, v2 = live, designed later.)

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

## Appendix B — how Fuel advertises kernels (FKC), for the identity tuple

You do not have to author FKC contracts — Fuel authors them for every kernel on its dispatch
surface (including a thin facade for each Baracuda entry point). FKC matters to you only as the
source of two things in this message:

- **`ImplId`** — FKC serializes the kernel identity tuple `(BackendId, op, dtypes, kernel_source,
  kernel_revision_hash)` (§3.2). `kernel_source` is declared per kernel (a front-matter default
  plus per-kernel overrides), so a Baracuda entry point's records are tagged `kernel_source =
  "baracuda"` and re-resolve to the same symbol on any build.
- **The miss signal** — a kernel's FKC *accept-contract* is a conjunction of structure predicates
  (contiguity, `inner_div`, `vec_width`, `reverse_strides`, dtype/quant admissibility). A
  structure-specialized kernel is admissible only in its cell; a generic kernel is admissible
  everywhere; "best admissible match = generic" is the miss (§3.3). The predicate vocabulary maps
  axis-for-axis onto your `OperandKey`, which is why a specialized contract's predicate *is* its
  structure key.

If you want, the FKC contract schema and the full predicate vocabulary can be shared as a follow-up
— but they are not needed to answer the five asks in §5.
