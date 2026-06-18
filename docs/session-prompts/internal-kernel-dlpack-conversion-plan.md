# Converting every internal Fuel kernel to DLPack+extension tensors and FKC contracts

**Status:** PLAN FOR REVIEW (2026-06-17), to be executed on a branch
(`feat/kernel-contracts-dlpack` or a child). Design/sequencing pass — no code yet.
**Scope:** the ordered, test-gated migration of **all ~390 inventoried internal Fuel
kernels** so that (a) every kernel is described by an **FKC** contract
(`docs/specs/kernel-contract-format.md`) that auto-registers it onto Fuel's dispatch
surface, and (b) every tensor handed across the kernel boundary is an **FDX** tensor
(`docs/specs/dlpack-extension.md`) — an honest standard `DLTensor` base plus the
optional sidecar carrying sub-byte/quant/symbolic/residency/bundle facts.
**Authoritative inputs:** the two final specs (FDX/FKC); the constraints digest
(`docs/specs/_research/architecture-constraints.md`); the as-built dispatch types
(`fuel-dispatch/src/{kernel.rs,fused.rs}`, `fuel-graph/src/registry.rs`); the nine
per-crate inventories (`docs/kernel-contracts/_inventory/*.md`). When this plan and the
constitution (`docs/architecture/`) conflict, the constitution wins.

This plan **applies the resolved decisions** from the spec sweep consistently: DLPack
v1.0 versioned struct floor (validated vs v1.3, v1.2+ explicit-strides behavior);
negative strides **first-class** in FDX with the signed-touched-range OOB check (V13) and
a per-kernel FKC `reverse_strides` acceptance capability; interior-axis live-prefix is a
materialized `IS_COPIED` copy; `register_full_with_source` becomes `Result` (a hard
prerequisite); capacity-only costing for v1 (SymEnv/per-tier memory are consumer-ahead);
sub-byte/quant logical shape carried explicitly; the quant scale single-place rule
(separate graph-input scale ⇒ FKC `accept.inputs` operand, NOT also an FDX
`scale_buffer`); `[;6]` inline shape/stride mirroring DimVec with explicit >rank-6 error;
frozen-size sidecar sub-structs with size-assertion tests; `kernel_revision_hash` over a
canonicalized parse sharing the FDX `name_hash` stable hash; explicit sidecar param for
all controlled cross-runtime signatures with `manager_ctx` only as the opportunistic
fallback on the pure `__dlpack__` path; little-endian v1; FDX owns the stable numeric code
table while FKC uses string names.

---

## 0. The shape of the work — what "convert a kernel" actually means

A kernel today is a `KernelRef` (`fn(inputs, outputs, layouts, params) -> Result<()>`,
`kernel.rs:152`) registered into a `KernelBindingTable` via
`register*` calls (or a `BackendImpl` in the `FusedKernelRegistry`, `fused.rs:49`). It
carries `KernelCaps { strided_input: bool }`, a `CostFn`, a `PrecisionGuarantee`, and a
`kernel_source` tag. The dispatch key is `(OpKind, KernelDTypes, BackendId)`.

"Converting" a kernel has **two orthogonal axes**, matching the two specs:

1. **FKC (advertisement axis).** Author a `` ```fkc `` contract block describing the
   kernel's dispatch key, per-operand accept-contract (dtypes, the five-flag layout set
   incl. `reverse_strides`, shape/rank predicates, FDX requirements), per-output
   return-contract (dtype/shape/layout/aliasing rule), and the capability + cost +
   precision + determinism advertisement. Importing the file **registers** the kernel —
   replacing the hand-written `register_full_with_source(...)` call. This is a
   *serialization + authoring surface over types that already exist* (FKC §0): the FKC
   importer parses the block and calls the same registration path. **One** dispatch
   primitive must change first: `register_full_with_source` must lose its `panic!`
   (kernel.rs:910-918) and return `Result` (FKC §10.10 CONSTITUTION-CONFLICT).

2. **FDX (tensor axis).** Make the kernel's operands/outputs FDX-describable. For the
   **overwhelming majority** (contiguous standard-dtype kernels) this is *zero kernel
   code change* — the base `DLTensor` is fully faithful and the kernel keeps reading the
   same contiguous bytes. FDX only bites where the kernel handles a *non-standard* fact
   the base DLTensor cannot carry honestly: packed sub-byte/quant bytes (the `uint8`
   honesty stand-in + a `FDXQuant`/`FDXDTypeExt` sidecar), symbolic live extents (the
   `FDXExtent`/`SymEnv` path), negative strides (`reverse_strides`), paged/gather caches
   (`FDXIndexedResidency`), or multi-output bundles (`FDXOutputView`).

The critical realization from the inventories: **almost every Fuel kernel is
contiguous-only, zero-offset, row-major** (the cross-cutting fact in *every* inventory).
That makes FDX adoption mostly a *description* exercise, not a rewrite — the honest base
`DLTensor` already matches what these kernels consume. The work is dominated by **authoring
390 contracts**, not by 390 kernel rewrites. The kernels that need real code/wiring change
are a small, identifiable minority: strided GPU kernels (gain `strided`/`broadcast_stride0`
declarations + possibly `reverse_strides`), quant kernels (scale-operand wiring + the
`FDXQuant`/`uint8`-base description), attention/paged kernels (the symbolic-extent +
gather descriptor), and multi-output bundles (the `FDXOutputView` round-trip).

### 0.1 The inventory, bucketed by conversion category

The 390 kernels (per-crate counts from the prompt: cpu 71, reference 79, vulkan 61,
metal 39, quantized 40, mkl/aocl 4, fused registry 23, conv+flash-attn 5, dispatch 68)
fall into **five conversion categories** (defined in §3). The dispatch-crate 68 are the
*wrappers* that wire the cpu/cuda/vulkan typed kernels to the binding table — they are
where the FKC `entry_point` resolves and where `KernelCaps`/`CostFn`/`PrecisionGuarantee`
live today, so a dispatch-crate row and the backend typed kernel it wraps are **one
logical contract** (the contract describes the wrapper's `(op, dtypes, backend)` key and
points `entry_point` at the wrapper). The 79 reference-backend kernels are the
correctness oracle (`fuel-reference-backend`, no production path); they get contracts last
and at low priority (they never dispatch in production, but a contract documents the
oracle's accept/return shape for the Judge).

| Category | Definition | Approx. count | Kernel change needed |
|---|---|---|---|
| **A. Contiguous standard-dtype** | reads dense row-major bytes, standard `DType`, no quant/symbolic/bundle | ~300 (bulk of cpu/reference/vulkan-contiguous/metal-contiguous/dispatch + most fused) | **None** — author contract only; base `DLTensor` is faithful |
| **B. Strided / offset / reverse** | walks strides, broadcast (stride-0), non-zero offset, or negative strides | ~60 (vulkan strided elementwise/movement, metal `_strided`, baracuda `strided_input`, `strided_copy_signed`, flip/roll, matmul stride model) | Declare the five-flag layout set + `reverse_strides`; signed-stride OOB check on the FDX side |
| **C. Quant** | packed GGML/MX/NF4 bytes + scale(s); `size_in_bytes()==0` sub-byte | ~55 (quantized 40, qmatmul/nf4 across cpu/vulkan/metal, QMatMul fused) | `uint8` honest base + `FDXQuant`/`FDXDTypeExt` sidecar; scale-operand single-place wiring |
| **D. Attention / paged / symbolic** | FlashAttn (`k_len` over capacity), PagedAttn (block table), varlen | ~14 (flash/paged across cpu/vulkan/metal/cuda, fused FLASH_ATTN/PAGED_ATTN + 3 backward) | `FDXExtent`/`SymEnv` for `k_len`; `FDXIndexedResidency` for paged; `reverse_strides` n/a |
| **E. Multi-output bundle** | one allocation, N sub-views (SelectiveScan, SsdChunkScan; flash backward dQ/dK/dV) | ~8 | `FDXOutputView` per slot; rank≤6 + name side-table round-trip |

Categories are *per-kernel facets*, not exclusive: a paged-attention kernel is D (symbolic
+ gather) and could also be E if it bundled. The migration order (§2) sequences by
category because the **plumbing each category needs lands once and is then reused** by every
kernel in it.

---

## 1. Prerequisites (must land before any kernel converts)

These are the shared substrate. Each is a small, independently-testable change; together
they are the "the importer can register a contract and the boundary can describe a tensor"
floor. **Until these land, no contract can register a kernel** — so they gate the whole
program and ship first, on the branch, each born-red.

### P1 — `register_full_with_source` returns `Result` (never-panic prerequisite)

`KernelBindingTable::register_full_with_source` (kernel.rs:895-920) currently `panic!`s on
a duplicate `KernelRef` function pointer. FKC's importer cannot drive a panicking path
(FKC §10.10 CONSTITUTION-CONFLICT; constitution never-panic). Change the signature to
`-> Result<()>` returning a new `Error::DuplicateKernelRef { op, dtypes, backend }` instead
of panicking. Cascade the `Result` up through `register_full`, `register_with_caps`,
`register`, `register_with_precision`, `register_with_caps_and_precision`. The dedup is at
the resolved-`KernelRef`-pointer level (two distinct `entry_point` strings can alias one
`fn` via the link registry — FKC §10.10), so the check stays pointer-identity-based.

- **Born-red test:** `register_duplicate_kernel_ref_is_err` — register the same `fn` twice
  at one key, assert `Err(DuplicateKernelRef)` (replaces the existing
  `#[should_panic] step_9a_duplicate_kernel_ref_panics` test at kernel.rs:1352, which is
  rewritten to assert the `Err`). Watch the old panic-test fail to compile against the new
  signature first.
- **Touch:** every existing `register*` call site (cpu/cuda/vulkan/mkl/aocl dispatch
  registration fns) must now `?` or expect-once-at-startup-with-context. Bulk mechanical;
  the startup registration fns already return `Result` or run in a `OnceLock` initializer
  that can carry a `Result`.

### P2 — `fuel-core-types::dlpack` module (the FDX structs + code tables)

Add the `dlpack` feature and module to `fuel-core-types` carrying the `#[repr(C)]` POD
structs (`FDXSidecar`, `FDXDTypeExt`, `FDXQuant`, `FDXExtent`, `FDXTiling`, `FDXResidency`,
`FDXStorage`, `FDXBufferRef`, `FDXOutputView`, `FDXIndexedResidency`, `FDXBlockTable`,
`FDXAffine`/`FDXAffineTerm`) exactly as FDX §5-6 specify, plus the standard `DLTensor` /
`DLManagedTensorVersioned` mirrors (FDX §5.1) and the C header `fuel_dlpack_ext.h`. FDX is
the **normative owner of the shared numeric code table** (FDX §6.0): the explicit
`fdx_code(DType)->Result<u16>` / `dtype_from_fdx` (and `BackendId`/`SubstrateClass`/
`ScaleGranularity` analogs) implemented with a `match`, never `as u16`.

- **Born-red tests:** (1) the size-assertion suite (FDX §5.4) — `size_of::<FDXSidecar>()`,
  each sub-struct, `size_of::<FDXAffineTerm>()==16`, `size_of::<FDXAffine>()==80`, plus the
  `offset_of!`-per-field pins for `FDXExtent` and the gather fields shifted by it; (2) the
  code-table round-trip (FDX §6.0) — every `DType`/`BackendId`/`SubstrateClass`/
  `ScaleGranularity` variant maps to its FDX code and back by name; reordering a source
  enum breaks the exhaustive `match`; (3) the inline `[;6]` shape/stride mirror of `DimVec`
  with an explicit `RankExceedsSix` error beyond rank 6.

### P3 — The FDX OOB / honesty validator (V8, V11-V14, V16-V21)

Implement the FDX validator suite as `Result`-returning checks runnable at the boundary
(FDX §3, §6.4, §6.9; constitution build-time validation). The load-bearing ones for the
conversion: **V11** (strides never NULL on versioned export); **V13** the *signed*
touched-range OOB check (per axis `(dim-1)*stride` as positive max if `stride>0` else 0,
as negative min if `stride<0` else 0; window ⊆ `[0, size_bytes)`) — this is the
negative-strides-first-class reversal; **V12** (256-byte data alignment on boundary-b
export, intra-buffer starts via `byte_offset`); **V8** (`size_bytes` covers capacity-shape
or `MEANING_REQUIRES_EXT` set); **V14** (realize-time `min ≤ value ≤ capacity`); **V16/V17**
(affine extent); **V18-V21** (gather/paged).

- **Born-red tests:** the validator test matrix — a faithful contiguous tensor passes;
  NULL strides on a versioned export → `Err` (V11); a flip view (negative stride, window in
  bounds) → **passes** V13 (the reversal — this test would have been a *rejection* under the
  withdrawn rule, so it is the regression guard for the decision); a stride that escapes
  `size_bytes` → `Err(StrideRangeOutOfBounds)`; a misaligned boundary-b `data` → `Err`
  (V12); a symbolic tensor claiming full-capacity safety without backing → `Err` (V8).

### P4 — The FKC importer skeleton (`fuel-dispatch::fkc`)

A new `fuel-dispatch` module that parses the markdown + `` ```fkc `` YAML blocks (the
restricted YAML-1.2-core subset of FKC §3.8 — quoted tokens, no Norway/sexagesimal, tabs
are a hard error, anchors disabled), resolves `entry_point` → `KernelRef` against the
provider's link registry (FKC §12.6), computes `kernel_revision_hash` over the
canonicalized parse (sharing the FDX `name_hash` stable hash — resolved decision), and
calls `register_full_with_source` (now `Result`, P1) for primitive ops or the
`FusedKernelRegistry` for fused ops. Distinguishes the `OpParams` vs `FusedOpParams`
namespace by which of `op_kind`/`fused_op` the contract declares (FKC §10.7). Returns typed
errors for every consistency failure (FKC §10): `BadScalarType`, `YamlTabIndent`,
`DuplicateKernelRef`, `QuantIncoherent`, `ScaleDoubleDeclared`, `GatherNotYetSupported`,
`MxNotYetRegistrable`, `RankExceedsSix`.

- **Born-red test:** `import_one_trivial_contract_registers_kernel` — a minimal `binary`
  contract (the §4.1 worked example) imports and the resulting binding is
  lookup-resolvable with the right caps/precision/cost; a malformed block (tab indent,
  unquoted `ggml_dtype: Q4_0` parsed as a number) → the matching typed `Err`.

### P5 — A `KernelCaps` growth path for the five-flag layout set (forward-compatible)

Today `KernelCaps` is one bool (`strided_input`). FKC's five-flag set
(`contiguous/strided/broadcast_stride0/start_offset/reverse_strides`) is richer. v1 keeps
the *lossy projection* (FKC §12.2): the importer projects `(strided && broadcast_stride0)`
onto `strided_input`, routes `start_offset` through auto-Contiguize, and treats
`reverse_strides` as **not yet honored** (a negative-stride operand is normalized to a
non-negative copy until `KernelCaps` grows the flag). Add the remaining flags to
`KernelCaps` as `#[derive(Default)]` false fields (no enum churn, per the doc-comment's
"forward-extensible by adding fields") so the projection can tighten later without an ABI
break. **No behavior change in v1** beyond carrying the richer facts; the executor's
contiguize gate still reads `strided_input`.

- **Born-red test:** `kernel_caps_five_flags_default_false_and_project` — the new fields
  default false; the projection fn maps a declared `{strided, broadcast_stride0}` to
  `strided_input == true` and a `reverse_strides`-only kernel to `strided_input == false`
  (still normalized).

> **P1-P5 are the gate.** They are sequenced first and verified (each born-red) before any
> `OpKind`-family conversion. They also establish the **CI lints** FKC §10 requires
> (required fields present, non-overlapping keys, non-placeholder precision/cost,
> prose-blurb == structured-blurb, FKC token set ⊆ FDX token set) — these lints are what
> make every subsequent conversion test-gated rather than eyeballed.

---

## 2. Migration order (which crates / op-families first, and why)

The order is **plumbing-driven**: each category's shared substrate lands once, then every
kernel in that category converts behind it. Within a category, the always-built CPU backend
goes first (it is the coverage floor — the constitution requires ≥1 bit-stable CPU kernel
per primitive op, and the CI lint will fail until each converted op has its CPU contract),
then the GPU/BLAS backends as sibling alternatives at the same key.

**Phase 0 — prerequisites P1-P5** (§1). Gate for everything.

**Phase 1 — Category A, CPU elementwise + movement (the proof bulk).**
Convert the contiguous standard-dtype kernels first, on `fuel-cpu-backend` + its
`fuel-dispatch` wrappers. Why first: largest category (~300), simplest change (contract
only, base `DLTensor` faithful), and it *exercises the whole FKC pipeline end-to-end* — the
importer, the five-flag projection, the cost/precision/determinism blocks, the CI lints —
on the easiest possible kernels. Order within: elementwise binary → unary → compare/where →
affine/clamp/powi → cast → reductions/norms/softmax → indexing/gather/scatter →
shape-movement (flip/roll/concat/triangular/pad/cumsum) → rope → conv → matmul (dense
float/int) → the Mamba/SSM + FSCE forward kernels. This is also the order the digest's
"elementwise before quant before attention" guidance implies.

**Phase 2 — Category A, the other backends (Vulkan-contiguous, Metal-contiguous,
baracuda-contiguous, MKL/AOCL).** Same contracts, sibling alternatives at the same
`(op, dtypes, backend)` keys, each with its own `kernel_source` tag, `PrecisionGuarantee`
(Vulkan reductions/softmax/norm carry `PrecisionGuarantee::none` — scheduler-determined
FADD order), and cost (Vulkan command-buffer `overhead_ns` higher than CPU). MKL/AOCL are 4
kernels (matmul + conv2d × 2 crates) registering as CPU siblings tagged `"mkl"`/`"aocl"`;
their conv2d contracts must capture the **fallback boundary** (delegate to scalar CPU when
dilation≠(1,1) or `ConvShape::validate` fails) and flag the `.expect()` panic on the BLAS
gemm closure as a never-panic fix folded into the conversion.

**Phase 3 — Category B, strided / offset / reverse.** Lands after the `reverse_strides`
plumbing decision is wired (P3's signed V13 + P5's flag). Convert the strided GPU
elementwise/movement kernels (Vulkan `unary`/`binary`/`affine`/`clamp`/`powi`/`cumsum`/
`flip`/`roll`/`concat`/`rope`/matmul-stride-model/`arg_reduce_any_dim`/`strided_copy*`;
Metal `_strided` variants; baracuda `strided_input` registrations) by adding their
five-flag declarations. The `strided_copy_signed_b*` (Vulkan) and the Metal signed strider
get `reverse_strides: accepted` — they are the kernels that *prove* the negative-stride
path. `Op::Flip` feeding a `reverse_strides`-capable kernel stays a zero-copy view (the
planner does **not** normalize between capable internal kernels — resolved decision).

**Phase 4 — Category C, quant.** Lands after the `FDXQuant`/`FDXDTypeExt`/`uint8`-base
plumbing (an extension of P2) and the scale-operand single-place rule. Convert
`fuel-quantized` (the GGML block numerics: 12 dtypes × {to_float, from_float,
from_float_imatrix, vec_dot} + the two matmul drivers), then the per-backend quant matmul
kernels (cpu `qmatmul_*`/`nf4_matmul_*`, Vulkan `dequant_*`/`qmatvec_q4_0`/
`matmul_q4_0_tiled`/`quantize_q8_0`, Metal `kernel_mul_mv/mm_*`), then the fused `QMATMUL`
and `NF4_MATMUL`. NF4 is the worked scale-operand case: `absmax` is a **separate graph
input** (`OpParams::Nf4Matmul`/`FusedOpParams::Nf4Matmul`, 3 inputs incl. absmax), so it is
an FKC `accept.inputs` operand with `fdx.quant.scale_operand: absmax` and **no** FDX
`scale_buffer` (single-place rule). GGML block scales are **inline** in the block bytes →
`FDXQuant.scale_placement = INLINE`, no FKC scale operand.

**Phase 5 — Category D, attention / paged / symbolic.** Lands after `FDXExtent`/`SymEnv`
(symbolic, partly already in tree — Phase D steps 1-2B landed per git) and
`FDXIndexedResidency` (gather). Convert FlashAttn (CPU/Vulkan/Metal/CUDA) declaring
`k_len` as `extent_kind: range` (single-`SymId` live prefix over capacity `sk`), then
PagedAttn declaring `fdx.gather.kind: paged_blocks` with `block_table`/`context_lens` as
separate `accept.inputs` operands (single-place rule). The CUDA `fuel-flash-attn-cuda` ops
(fixed + varlen) are last in this phase: they are CUDA-only, last-dim-contiguous but
strided-outer + offset-capable, so they exercise the `start_offset: accepted` +
`strided: accepted` (outer axes only) declaration and the varlen `cu_seqlens` data-determined
sym path. `fuel-conv` (3 host primitives) converts here too — flag its
`ConvShape::validate().expect()` panic as a never-panic fix.

**Phase 6 — Category E, multi-output bundles.** SelectiveScan, SsdChunkScan (both bundle
`[y; last_state]`), and the flash-attn backward dQ/dK/dV. Lands after the `FDXOutputView`
round-trip (rank≤6 + name side-table, FKC §5.5). The contract declares `bundle:` slot specs;
the importer round-trips the slot **name** through the side-table (not just the `name_hash`).

**Phase 7 — Category A, the reference backend (oracle, last + low priority).** The 79
`fuel-reference-backend` kernels never dispatch in production, but a contract documents the
oracle's accept/return shape for the Judge's comparator selection (the oracle is
`bit_stable_on_same_hardware` by construction, fixed iteration order). These convert last
because they have no consumer and no urgency; "no consumer is not a reason to skip building
a capability, but it is a reason to sequence it behind things with consumers."

**Why this order (one line each):** A-first proves the pipeline on the easy bulk and
satisfies the CPU-coverage lint cheaply; B before C because strided plumbing is simpler than
quant and several quant kernels are strided; C before D because quant is a tensor-description
problem (sidecar) while attention adds the *symbolic + gather* descriptors on top; E last
among production paths because bundles are the rarest and the round-trip is independent;
reference dead-last because it has no production consumer.

---

## 3. Categories of per-kernel change (the conversion recipe per category)

For every kernel, the contract authoring + validation + `KernelRef` wiring is the same
shape (§5). What differs per category is **which FKC fields are non-trivial** and **whether
the kernel body changes**.

### 3.A Contiguous standard-dtype — contract only, no kernel change

The base `DLTensor` is fully faithful (FDX §3: standard dtype ⇒ `dtype`/`shape`/`strides`
exactly correct). The conversion is:

- **Accept layout:** `{contiguous: required, strided: rejected, broadcast_stride0:
  rejected, start_offset: rejected, reverse_strides: rejected}` (the cpu/reference/
  mkl/aocl/contiguous-Vulkan/contiguous-Metal default — matches the inventories'
  "contiguous-only, zero-offset, row-major" universal fact).
- **`caps.awkward_layout_strategy: requires_contiguous`** — the planner inserts
  `Op::Contiguize` (itself an FKC kernel, FKC §4.3) for any non-contiguous operand and adds
  its cost. **No silent internal contiguize** (the kernels genuinely require dense input;
  the executor's auto-Contiguize already guarantees it).
- **`fdx.requires_ext: false`** on every operand; no quant/symbolic/gather/sub_byte.
- **dtype list** from the inventory (`{F32, F64, BF16, F16}` for the float families; U8
  output for compares; U32 output for argmax/argmin; I64 targets for FSCE; etc.).
- **return rule:** `dtype_rule`/`shape_rule`/`layout_guarantee: contiguous`/`aliasing:
  none` (or `aliasing: in_place(0)` for the in-place families — InplaceAffine, the in-place
  unary/clamp/powi sets, WriteSlice/WriteSliceRotating which alias `outputs[0]`).
- **cost/precision/determinism** transcribed from the as-built `CostFn` and
  `PrecisionGuarantee` (most CPU = `PRIMITIVE_DETERMINISTIC_CPU` / bit-stable; Vulkan
  reductions = `none(reason)`).

**Kernel body: unchanged.** The wrapper still reads contiguous bytes. The only code change
is replacing the hand-written `register*` call with the imported contract.

### 3.B Strided / offset / reverse — declare the richer layout set

The kernel already walks strides (Vulkan rank-4 stride params, Metal `get_strided_index`,
baracuda stride-driven FFI). The conversion:

- **Accept layout:** declare the flags the kernel actually honors. E.g. Vulkan `unary`
  (f32): `{contiguous: accepted, strided: accepted, broadcast_stride0: accepted,
  start_offset: rejected, reverse_strides: rejected}` (FKC §4.1 example). Metal
  `unary_kernel_strided`: all `accepted` (offset-capable via `BufferOffset`; the signed
  walk handles `reverse_strides`). The Vulkan `strided_copy_signed_b*` and the signed
  Metal strider: `reverse_strides: accepted`.
- **`caps.awkward_layout_strategy: handles_strided`** (walks strides directly, no fixup).
- **`fast_paths:`** declare the contiguous fast path (`all_inputs_contiguous`) and the
  strided slow path so the planner costs honestly (FKC §4.2).
- **FDX side:** strided/broadcast/offset/reversed operands are FDX-describable `DLTensor`s
  with non-trivial / signed strides + a `byte_offset` at the iteration-first element. The
  **signed V13** check (P3) proves no-OOB; nothing in the kernel body changes — it already
  handles the strides. `reverse_strides: rejected` kernels get a planner-inserted
  non-negative copy (`IS_COPIED`) **only** when fed a negative stride (FKC §4.1.1) — never
  universal, never between capable internal kernels.

**Kernel body: unchanged** (the kernel is already strided). The change is *declaring* what
it accepts so the planner stops over-contiguizing (today everything is projected onto one
`strided_input` bool; v1 keeps that projection but the richer facts are now recorded for
the `KernelCaps` growth in P5).

### 3.C Quant — `uint8` honest base + sidecar + scale single-place

The kernel reads packed GGML/MX/NF4 bytes whose logical meaning exceeds the base
`DLTensor`. The conversion (FDX §3, §6.2; FKC §3.9.3, §6):

- **Base `DLTensor`:** the packed weight is `DLDataType { kDLUInt, 8, 1 }` over the
  *physical byte buffer* (FDX §3 honesty stand-in), `MEANING_REQUIRES_EXT` set (base bytes
  alone are not a usable tensor). Sub-byte dtypes (`F4`/`F6E2M3`/`F6E3M2`, `size_in_bytes()
  ==0`) carry their **logical shape explicitly** (resolved decision) via the sidecar — never
  sized off `size_in_bytes`.
- **Sidecar `FDXQuant`:** `family` (`GGML_BLOCK` / `MX` / `AFFINE_INT` / `AFFINE_FLOAT`),
  `ggml_dtype` by **variant name matched by numeric code** (`Q4K`, never `Q4_K_M` — FKC
  §3.4; `Q4_K_M` is the GGUF file-format name → `GgmlDType::Q4K(12)` → `Capability::
  MatMulQ4KM`), block size, `scale_placement`, packing order; `FDXDTypeExt` for the
  sub-byte bit-width + `FDXPacking` (FDX §6.1, the sole packing authority — FDX never uses
  the native DLPack sub-byte path, §3.4).
- **Scale single-place (resolved decision, the load-bearing rule):**
  - **GGML inline** block scales ride the block bytes → `FDXQuant.scale_placement =
    INLINE`, `scale_buffer` valid in the sidecar, **no** FKC scale operand
    (`fdx.quant.scale_operand: ~`). This is the GGML path.
  - **MX separate-buffer** F8E8M0 per-block scale rides the FDX tensor →
    `scale_placement = SEPARATE_BUFFER`, `scale_buffer` valid, no FKC operand.
  - **Dynamic affine quant / NF4 absmax whose ABI passes the scale as its own graph
    input** → the scale is an FKC `accept.inputs` operand named in `fdx.quant.scale_operand`,
    and **NOT** also an FDX `scale_buffer` (the §10.6 coherence check rejects declaring
    both — `ScaleDoubleDeclared`). NF4 (`absmax`) is exactly this case.
- **Dispatch key:** enriched per-operand with `(family, ggml_dtype | (granularity, role))`
  (FKC §3.2, P5) so `(QMatMul, A=F32×PerToken, W=Q4_0)` ≠ `(QMatMul, A=F32×PerTensor,
  W=Q8_0)`. `PerBlock` is FDX/FKC-only until `ScaleGranularity` gains it → such a contract
  **parse-validates but is not registrable** (`MxNotYetRegistrable`, FKC §6).
- **No silent dequant.** A backend without the quant capability errors at dispatch; the
  planner inserts an explicit `Op::Dequantize` (itself an FKC kernel) and prices it (FKC
  §4.3; digest §9).

**Kernel body:** mostly unchanged — the kernels already read packed bytes + scales as they
do today. The change is **describing** the packed payload honestly and **wiring the scale
operand to one authoritative place**. The one real code touch is NF4/dynamic-quant kernels
whose scale must be threaded as a declared operand rather than an implicit sidecar.

### 3.D Attention / paged / symbolic — the `FDXExtent` + `FDXIndexedResidency` descriptor

- **FlashAttn `k_len` (symbolic live prefix over a capacity KV cache):** the K/V operand's
  symbolic axis declares `symbolic_extent: required, extent_kind: range` — capacity is the
  concrete `sk` bound (strides keyed to capacity), the live `k_len ≤ sk` is one `SymId`
  resolved per call via the `SymEnv` (FDX §3.1, §6.4; FKC §3.9.2; matches
  `OpParams::FlashAttn { sk, k_len, ... }` and `FusedOpParams::FlashAttn { k_len:
  Option<DynScalar> }`). The CPU kernel already loops `0..k_len` with the `k_len - sq`
  causal offset — **no kernel change**, just the extent declaration. The static path sets
  `k_len == sk` (degenerate range). A dense export to a sidecar-blind consumer of an
  interior-axis live prefix is a **materialized `IS_COPIED` copy** (interior axis is not
  contiguous — FDX §3.1.1; resolved decision).
- **PagedAttn (block-table gather):** the pool operand declares `fdx.gather.kind:
  paged_blocks` with `requires_ext: true` (mandatory, FDX V19), and the
  `[B, max_blocks_per_seq]` U32 `block_table` + `[B]` U32 `context_lens` are **separate
  `accept.inputs` operands** named in `fdx.gather.block_table`/`context_lens` (single-place
  rule — the ABI takes them as graph inputs; `KernelRef::PagedAttn` operand order
  `[q, k_cache, v_cache, block_table, context_lens, alibi?]`, kernel.rs:314-331). The FDX
  `FDXBlockTable.table_buffer`/`context_lens_buffer` point at the **same** buffer-table
  slots (V21(b) index-equality). `Capability::DlpackExtGather` gates direct admission; else
  the planner materializes a dense un-paged copy (`IS_COPIED`) priced from the materialize
  kernel's contract.
- **Varlen (CUDA):** `cu_seqlens` is a data-determined sym path (`context_lens` declared
  `symbolic_extent: required`, FKC §3.9.1).
- **Capability gates:** `extent_kind: affine`/`range` gated by the same
  `Capability::DlpackExtSymbolic` (no separate affine-extent capability — FKC §3.9.2); the
  distinct `DlpackExtAffine` token gates affine **quant**, not extent.

**Kernel body: unchanged** for the symbolic case (the kernels already take `k_len`/block
tables). The change is the descriptor + the SymEnv plumbing (largely already in tree per the
Phase D git history). `[consumer-ahead]`: until the FDX gather codes land, a gather-bearing
operand returns `GatherNotYetSupported` rather than fabricating a descriptor (FKC §3.9.1).

### 3.E Multi-output bundle — `FDXOutputView` round-trip

The kernel writes N logical slots into one bundled `Storage` by byte offset
(`outputs.len() == 1`, kernel.rs:121-149; the executor pre-allocates via
`allocate_bundled_storage`). The conversion:

- **Return contract `bundle:`** declares each slot's `{name, dtype, shape, layout,
  byte_offset}`. The importer maps to the registry's `output_views: fn(&[Shape], &[DType],
  &FusedOpParams) -> Vec<OutputViewSpec>` (registry.rs:133). The as-built `OutputView` has
  arbitrary-rank symbolic `Shape` + `Option<&'static str>` name; FDX `FDXOutputView` is
  `shape: [u64; 6]` + `name_hash: u64`, so the contract is **rank ≤ 6** (explicit
  `RankExceedsSix` error, FKC §5.5) and keeps the slot **name** in a side-table so it
  round-trips (not lossy-to-hash).
- **`output_views(...)[0]` must equal `shape_rule`/`dtype_rule`** (registry invariant,
  registry.rs:124-125) — the importer cross-checks.

**Kernel body: unchanged** (already writes by byte offset). The change is declaring the slot
specs in the contract and round-tripping the name.

---

## 4. Worked conversions (three representative kernels)

### 4.1 Elementwise binary (Category A) — `add_f32` (`fuel-cpu-backend`)

As-built: `binary<T,U>` chassis (`chassis/binary.rs:78`), thunk `add_f32`
(`byte_kernels.rs:68`), registered at `dispatch.rs:3913` as
`(AddElementwise, [F32,F32,F32], Cpu)` with default caps (contiguous-only), bulk-upgraded to
`PRIMITIVE_DETERMINISTIC_CPU`, cost `default_cost_for_op_kind(AddElementwise)`. No
broadcasting (inputs must be same-shape contiguous; auto-Contiguize realizes broadcasts
upstream).

```fkc
kernel: add
op_kind: AddElementwise
blurb: "Elementwise add; all math in dtype, half via f32 round-trip; contiguous-only."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: cpu::add_f32        # resolves to the dispatch wrapper KernelRef
kernel_revision_hash: auto
accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected,
                start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=rhs"
      fdx: { requires_ext: false }
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected,
                start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=lhs"
      fdx: { requires_ext: false }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: "passthrough(lhs)"
      shape_rule: "same_as(lhs)"
      layout_guarantee: contiguous
      aliasing: none
caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: [ { when: "all_inputs_contiguous", class: cheap_elementwise } ]
  in_place: false
cost:
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"
  overhead_ns: 50
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }
precision:
  bit_stable_on_same_hardware: true
  audited: true
  notes: "deterministic positional walk; F32 accumulator for bf16/f16; bit-identical same-hw re-run."
determinism: same_hardware_bitwise
```

Importing this registers `(AddElementwise, [F32,F32,F32], Cpu)` (plus F64/BF16/F16 by
dtype expansion) with `KernelCaps::empty()` (contiguous-only projection),
`PRIMITIVE_DETERMINISTIC_CPU` precision, and the `cheap_elementwise` cost fn — **identical**
to today's hand-registration. **No kernel body change.** The Vulkan/baracuda strided
siblings register at the same key with `strided: accepted, broadcast_stride0: accepted`
(Category B) and their own `kernel_source`/precision/overhead.

- **Born-red test:** `fkc_add_f32_registers_and_matches_legacy` — import the contract;
  assert `lookup_with_caps(AddElementwise, [F32×3], Cpu)` returns the same `KernelRef` +
  caps the legacy `register_cpu_kernels` produced; run the kernel on a known input and
  assert the output bytes match the pre-conversion result. Watch it red (contract not yet
  imported / importer not yet wired) before green.

### 4.2 Quant matmul (Category C) — `qmatmul_q4_0_f32` (`fuel-cpu-backend` → `fuel-quantized`)

As-built: `qmatmul_q4_0_f32` (`byte_kernels.rs:4202`) → `qmatmul_generic_f32::<BlockQ4_0>`
→ `fuel_quantized::matmul`; registered as `(QMatMul, [F32, U32, F32], Cpu)` with
`OpParams::QMatMul { quant_type: Q4_0, batch_count, m, n, k }` (dispatch.rs:4561). The
weight is a flat U32-typed byte stream of `n*k/32` Q4_0 blocks with **inline f16 per-block
scales**. Activations F32, output F32.

```fkc
kernel: qmatmul_q4_0
op_kind: QMatMul
blurb: "A[F32] @ dequant(W[Q4_0]) -> F32; inline f16 block scales; contiguous-only."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: cpu::qmatmul_q4_0_f32
kernel_revision_hash: auto
accept:
  inputs:
    - name: a               # activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected,
                start_offset: rejected, reverse_strides: rejected }
      shape_constraint: "rank>=2"
      fdx: { requires_ext: false }
    - name: w_q             # packed Q4_0 weight, honest uint8 base
      dtypes: [U8]          # FDX honesty stand-in over the physical byte buffer
      layout: { contiguous: required, strided: rejected }
      fdx:
        requires_ext: true              # MEANING_REQUIRES_EXT: packed bytes, not a usable tensor
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0              # variant NAME, matched by code 2 (FKC §3.4)
          granularity: PerBlock         # block_size 32; scales inline
          scale_operand: ~              # INLINE scales -> no separate scale operand (single-place)
        sub_byte: F4                    # 4-bit logical element, carried explicitly
  op_params:
    variant: QMatMul                    # OpParams::QMatMul
    fields:
      quant_type: { kind: QuantType, constraint: "== Q4_0" }
      k: { kind: usize, constraint: "k % 32 == 0" }   # block boundary
      n: { kind: usize }
return:
  outputs:
    - name: out
      dtype_rule: "F32"
      shape_rule: "matmul(a, n)"        # [..., M, N]
      layout_guarantee: contiguous
      aliasing: none
caps:
  awkward_layout_strategy: requires_contiguous
cost:
  class: matmul
  flops: "2 * batch * m * n * k"
  bytes_moved: "batch*m*k*4 + n*(k/32)*18 + batch*m*n*4"   # 18-byte Q4_0 blocks
  overhead_ns: 50
precision:
  bit_stable_on_same_hardware: true
  audited: true
  notes: "i8/i16 integer dot then * (d_x . d_y) in f32; deterministic; must bit-match GPU dequant_q4_0."
determinism: same_hardware_bitwise
```

The **scale single-place rule** is exercised: Q4_0's scales are *inline* in the block
bytes, so `scale_operand: ~` and the FDX sidecar carries `scale_placement = INLINE` with
`scale_buffer` pointing at the inline region — **never** a separate FKC operand. Contrast
with the NF4 contract whose `absmax` *is* a separate `accept.inputs` operand with
`scale_operand: absmax` and no FDX `scale_buffer`. The §10.6 coherence check rejects any
contract that declares both for the same scale (`ScaleDoubleDeclared`). **Kernel body
unchanged**; the change is the honest `uint8` base + `FDXQuant` description.

- **Born-red tests:** (1) `fkc_qmatmul_q4_0_registers` — import registers
  `(QMatMul, [F32,U32,F32], Cpu)` matching legacy; (2) `fdx_q4_0_weight_is_honest_uint8` —
  the FDX base of a Q4_0 weight is `uint8` of the correct physical byte size with
  `MEANING_REQUIRES_EXT` set, and a sidecar-blind read sees opaque bytes (never a
  mislabeled F4); (3) `scale_double_declared_rejected` — a contract declaring both
  `scale_operand` and a sidecar `scale_buffer` for one scale → `Err(ScaleDoubleDeclared)`;
  (4) the existing CPU↔GPU dequant parity test stays green (the description change must not
  alter numerics).

### 4.3 Paged attention (Category D) — `paged_attn_f32` (`fuel-cpu-backend`)

As-built: `paged_attn_f32` (`byte_kernels.rs:6805`); fused `PAGED_ATTN`
(`FusedOpId(13)`, registry.rs:241); `FusedOpParams::PagedAttn { softmax_scale, block_size,
softcap }`; geometry on `KernelRef::PagedAttn` operand order `[q, k_cache, v_cache,
block_table, context_lens, alibi?]` (kernel.rs:314-331). Caches `[num_blocks, block_size,
Hkv, D]`; `block_table [B, max_blocks_per_seq]` U32; `context_lens [B]` U32. The kernel
already does `if ctx_len==0 { continue }` (so `L=0` is legal — V14 `min=0`).

```fkc
kernel: paged_attn
fused_op: PAGED_ATTN                     # FUSED op -> FusedOpParams namespace (FKC §3.9.1)
blurb: "Paged-cache attention; uint8 block pool re-interpreted via a per-seq block table."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: cpu::paged_attn_f32
kernel_revision_hash: auto
accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, start_offset: rejected }
      rank: 4                            # [B, Hq, Sq, D]
      fdx: { requires_ext: false }
    - name: k_cache                      # the block POOL (honest uint8 base in FDX)
      dtypes: [F16, BF16, F32]           # true per-token pool element type (FDXDTypeExt)
      layout: { contiguous: required, strided: rejected, start_offset: accepted }
      rank: 4                            # physical [num_blocks, block_size, Hkv, D]
      fdx:
        requires_ext: true               # MEANING_REQUIRES_EXT mandatory for a paged pool (V19)
        symbolic_extent: required        # per-seq live length is symbolic (context_lens)
        extent_kind: range               # one SymId live prefix; min=0 (empty seq legal, V14)
        gather:
          kind: paged_blocks             # FDX_GATHER_PAGED_BLOCKS
          block_table: block_table       # role of the SEPARATE block-table accept.input
          context_lens: context_lens     # role of the SEPARATE context-lens accept.input
    - name: v_cache
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, start_offset: accepted }
      rank: 4
      fdx: { requires_ext: true, gather: { kind: paged_blocks, block_table: block_table,
             context_lens: context_lens } }
    - name: block_table                  # SEPARATE graph input (single-place rule)
      dtypes: [U32]
      rank: 2                            # [B, max_blocks_per_seq]
    - name: context_lens                 # SEPARATE graph input
      dtypes: [U32]
      rank: 1                            # [B]
      fdx: { symbolic_extent: required } # per-seq live lengths (data-determined sym)
  op_params:
    variant: PagedAttn                   # FusedOpParams::PagedAttn
    fields:
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize }
      softcap:       { kind: "Option<f32>" }
return:
  outputs:
    - name: out
      dtype_rule: "passthrough(q)"
      shape_rule: "same_as(q)"           # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none
caps:
  awkward_layout_strategy: requires_contiguous
cost: { class: attention, flops: "2 * b * hq * sq * (max_blocks_per_seq*block_size) * d",
        bytes_moved: "...", overhead_ns: 50 }
precision:
  bit_stable_on_same_hardware: true
  audited: true
  notes: "naive SDPA reference; F32 accumulator; deterministic; GPU tiled-softmax differs (declared separately)."
determinism: same_hardware_bitwise
```

The **single-place rule** is exercised twice: `block_table` and `context_lens` are
*separate graph inputs* (the ABI takes them as operands), so they are `accept.inputs`
entries whose roles are *named* in `k_cache.fdx.gather`, and the FDX
`FDXBlockTable.table_buffer`/`context_lens_buffer` point at the **same** buffer-table slots
(V21(b) index-equality) — no duplicate data. `MEANING_REQUIRES_EXT` is mandatory (V19).
`Capability::DlpackExtGather` gates direct admission; without it the planner materializes a
dense un-paged copy (`IS_COPIED`) priced from the materialize kernel's contract. **Kernel
body unchanged** (it already consumes the block table + context lens + `block_size`); the
change is the gather descriptor + the geometry cross-check against the operand buffers (the
op stores only 3 params, geometry rides `KernelRef::PagedAttn`).

- **Born-red tests:** (1) `fkc_paged_attn_registers_under_fused_namespace` — the contract
  registers via the `FusedKernelRegistry` (not `OpParams::PagedAttn` — there is none; an
  `op_kind` carrier fails the §10.7 namespace check); (2) `paged_pool_requires_ext` — a
  paged pool contract without `requires_ext: true` → FDX V19 `Err`; (3)
  `block_table_single_place` — the FKC `gather.block_table` role and the FDX
  `FDXBlockTable.table_buffer` resolve to the **same** buffer-table slot index (V21(b));
  (4) `gather_not_yet_supported` — until the FDX gather codes land, importing the
  gather-bearing operand returns `GatherNotYetSupported` (the `[consumer-ahead]` posture),
  and the test flips to green-registration once P2's gather struct ships; (5)
  `empty_sequence_legal` — a `context_lens[b]=0` sequence is not rejected by V14 (`min=0`).

---

## 5. Authoring + validating + wiring each kernel's contract (the mechanical loop)

For every kernel, regardless of category, the loop is:

1. **Locate the as-built registration** (the inventory cites the exact `dispatch.rs` /
   backend line). The `(op, dtypes, backend)` key, current `KernelCaps`, `CostFn`, and
   `PrecisionGuarantee` are the ground truth the contract must reproduce **exactly** (a
   conversion must be behavior-preserving — the born-red test asserts byte-identical output
   and identical registered caps/precision/cost vs the legacy path).
2. **Author the `` ```fkc `` block** in the provider's bundle file (one
   `docs/kernel-contracts/<crate>.md` per crate, following the per-crate inventory
   structure; FKC §9.1 a bundle file is many `## ` sections). The prose blurb must equal the
   structured `blurb:` (FKC §10.11 lint).
3. **Set `entry_point`** to the symbol resolving to the dispatch wrapper `KernelRef` via the
   provider's `link_registry` (FKC §12.6). The wrapper is unchanged; the contract just
   references it. `kernel_revision_hash: auto` derives from `entry_point + revision_base`
   over the canonicalized parse (sharing the FDX `name_hash` stable hash — resolved
   decision).
4. **Run the importer** (P4) — it parses, validates (FKC §10 + FDX validators P3),
   resolves the entry point, and calls `register_full_with_source` (P1) / the fused
   registry. Any inconsistency is a typed `Err` at import (build-time validation; no
   `try_*` sibling).
5. **Delete the hand-written `register*` call** for that kernel once its contract imports
   green. The registration crate's bulk `register_cpu_kernels` / `register_vulkan_kernels` /
   etc. shrink to "import the bundle file(s)."
6. **The born-red test** (§6) is written *before* steps 2-5 and watched red, then green.

The **CI lints** (established in P1-P5) run over the whole imported set after each
conversion: required fields present, dispatch keys non-overlapping (no two distinct
`KernelRef`s collide except as intentional siblings — the `DuplicateKernelRef` guard),
precision/cost not placeholder (the always-built-CPU-has-a-bit-stable-kernel-per-op
coverage lint, digest §3/§4), prose-blurb == structured-blurb, FKC accepted-token-set ⊆
FDX token set (FKC §10.12 cross-spec consistency — the two specs cannot drift).

---

## 6. The born-red test per converted kernel (the gate)

Every converted kernel ships with a test that was **observed to fail before the conversion
and pass after** (constitution TDD; "born-red tests are the goal"). The standard shape:

- **Registration-equivalence test:** import the kernel's contract; assert the resulting
  binding (`lookup_with_caps` / `lookup_precision` / `lookup_cost` for primitives, or
  `lookup_by_dtypes` for fused) returns the **same** `KernelRef`, caps, precision, and cost
  the legacy hand-registration produced. This proves the contract is a faithful
  serialization of the as-built registration. Red before the contract/importer exists.
- **Numeric-parity test:** run the converted kernel on a fixed input and assert the output
  bytes are identical to the pre-conversion result (Category A/B/E — no numeric change) or
  within the declared `PrecisionGuarantee` of the oracle (Category C/D where a description
  change must provably not alter numerics — the CPU↔GPU dequant parity, the flash `k_len`
  static path being byte-identical to the old `0..sk` form).
- **FDX-description test (Categories B-E):** the kernel's operands are FDX-describable and
  pass the relevant validators — V13 signed-range for a strided/reversed operand (a flip
  view *passes*, the regression guard for the negative-stride reversal); the honest `uint8`
  base + `MEANING_REQUIRES_EXT` for a quant weight; V19 for a paged pool; the
  `FDXOutputView` round-trip (name survives, rank≤6) for a bundle.
- **Negative test (where a decision is load-bearing):** `ScaleDoubleDeclared` for a quant
  contract declaring a scale in two places; `GatherNotYetSupported` until the gather codes
  land; `StrideRangeOutOfBounds` for a stride that escapes `size_bytes`; `RankExceedsSix`
  for a rank-7 bundle slot.

The program is **not done** until every one of the ~390 kernels has its contract imported,
its hand-registration deleted, and its born-red test green — and the CI coverage lint passes
(every production primitive op has ≥1 bit-stable CPU contract; no UNAUDITED/placeholder
precision or `unknown_cost` on a converted entry).

---

## 7. End state — every internal kernel is contract-described

When this program completes:

- **Registration is entirely contract-driven.** `register_cpu_kernels` /
  `register_vulkan_kernels` / `register_baracuda_cuda_kernels` / the MKL/AOCL/Metal
  registration fns and the `register_default_fused_kernels` path are replaced by "import the
  provider's FKC bundle file(s)" (FKC §G5: import = registration, zero hand-written glue).
  The `KernelBindingTable` and `FusedKernelRegistry` are populated by the importer, frozen
  at startup as today.
- **Every tensor crossing the kernel boundary is FDX-describable** — an honest standard
  `DLTensor` base (faithful for standard dtypes; `uint8` honesty stand-in for quant/sub-byte)
  plus the optional sidecar for the non-standard facts. Internal boundary (a) passes the
  sidecar as an explicit nullable param next to the `DLTensor`; the external `__dlpack__`
  boundary (b) carries it via deleter-gated `manager_ctx` only as the opportunistic fallback,
  with explicit sidecar params everywhere Fuel controls the signature (incl. Fuel↔Baracuda/
  Vulkane native calls — resolved decision).
- **Every cost / precision / layout / quant / symbolic fact is visible to the optimizer**
  (the 01 enforcement gate, digest §1) — nothing is hidden behind backend code. The planner
  costs contiguize-vs-strided-vs-materialize honestly from declared facts, normalizes
  negative strides **only** for incapable consumers (never universally, never between capable
  internal kernels), and inserts explicit `Op::Contiguize`/`Op::Dequantize`/materialize
  kernels (themselves contract-described and priced).
- **Persisted plans round-trip** via `(backend_id, op_kind, dtypes, kernel_revision_hash)`
  tuples (digest §7); the `kernel_revision_hash` is now derived per-contract over the
  canonicalized parse (sharing the FDX stable hash), so a contract edit scopes
  re-optimization to the affected decision point.
- **The reference backend's 79 kernels are contract-documented** (the oracle's accept/return
  shape for Judge comparator selection) even though they never dispatch in production.
- **Forward-looking facts are declared but consumer-ahead:** per-tier memory, SymEnv-resolved
  live-extent cost, MX/`PerBlock` quant, affine extents, and gather are carried in the
  contracts where the spec models them and ignored by today's importer (size-prefixed,
  `[consumer-ahead]`), so a future planner reads them without a re-conversion.

### 7.1 Deferred to later phases (out of scope for this conversion, named so the gap is owned)

- **`.fuel` mmap alignment reconciliation** — deferred to Phase E (resolved decision;
  little-endian v1 here, mmap residency `is_mmap_view` described but its consumer is ahead).
- **SymEnv-through-realize for *cost*** — capacity-only costing in v1 (resolved decision);
  per-token live-extent cost re-eval needs a `CostFn` signature change (FKC §4.4
  `[consumer-ahead]`).
- **`KernelCaps` tightening to the full five flags** — v1 keeps the lossy `strided_input`
  projection (P5); tightening (honoring `reverse_strides`, `start_offset`,
  `broadcast_stride0` independently) is a follow-up once the executor's contiguize gate reads
  the richer flags.
- **`PerBlock` `ScaleGranularity`** — MX contracts parse-validate but are not registrable
  until the source enum gains `PerBlock` (FKC §6; `MxNotYetRegistrable`).
- **The unimplemented-numeric kernels** the inventories flag (`Q8_1::to_float`
  `unimplemented!()`; `from_float_imatrix` panics for non-K-quants; `fuel-conv`/MKL/AOCL
  `.expect()` panic surfaces) get their never-panic fixes folded into their conversion (a
  panicking kernel cannot honor the contract's `Result` ABI), but new numeric
  implementations are out of scope — a contract describes what exists.
