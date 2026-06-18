# Session prompt — DLPack + FDX-extension communication layer

**Status:** Plan (2026-06-17). Not started. WIP lands on `feat/kernel-contracts-dlpack`
(the same branch the two specs live on), never `main`.

**Goal.** Implement the tensor-handoff boundary the two design specs describe: a versioned
DLPack base (`DLTensor` / `DLManagedTensorVersioned`) plus the optional Fuel sidecar (`FDXSidecar`),
used to pass tensors to kernels. This is the *as-built realization* of the design — it constructs
the DLPack+FDX **view** at the call boundary from Fuel's existing `(Storage, Layout)` split with
**no storage rewrite**, wires the two boundaries (internal kernel ABI vs external `__dlpack__`),
enforces the honesty invariant, and migrates the `KernelRef` ABI onto DLPack-shaped operands.

**Authoritative inputs (read these first; they win on conflict, constitution highest):**
- `docs/specs/dlpack-extension.md` (FDX) — the tensor/storage-axis spec. Owns all shared codes.
- `docs/specs/kernel-contract-format.md` (FKC) — the advertisement-axis sibling. The `reverse_strides`
  and `gather`/`affine` operand vocabulary, the scale/gather single-place rules.
- `docs/specs/_research/architecture-constraints.md` — the digest these were designed against.
- `docs/architecture/` — the constitution. Validate at build time; never panic; lazy-only;
  backends advertise, the planner decides; docs+tests with every material change.

**As-built anchors (verified 2026-06-17 — every signature below was read, not assumed):**
- `fuel-memory/src/lib.rs:82` — `Storage { inner: BackendStorage, dtype: DType, bundle: Option<Arc<[OutputView]>> }`.
  **Storage owns only bytes + dtype; Layout lives separately on the consumer** (lib.rs:70-73). This
  split is exactly the DLPack shape: `Storage` ≈ `data`+`device`+`dtype`, `Layout` ≈ `shape`+`strides`+`byte_offset`.
- `fuel-core-types/src/layout.rs:24` — `Layout { shape: Shape, stride: StrideVec, start_offset: usize }`.
  **`stride` is already `isize` (signed)** (layout.rs:11-22) and `Layout::flip` (layout.rs:299) already
  maintains "`start_offset` is the byte offset of the iteration-first element, always non-negative."
  This is the precise invariant FDX V13 (signed-stride OOB) needs — no Layout change required.
- `fuel-core-types/src/layout.rs:84` — `Layout::resolve(&SymEnv) -> Result<Layout>` (strides/offset
  unchanged, only the symbolic extent becomes concrete). The realize-time half of P4.
- `fuel-core-types/src/symbol.rs:29,59,119` — `SymId(u32)`, `SymEnv { bind, get }`, `DynScalar { resolve }`.
- `fuel-dispatch/src/kernel.rs:152` — the `KernelRef` ABI: `fn(&[Arc<RwLock<Storage>>], &mut [Arc<RwLock<Storage>>], &[Layout], &OpParams) -> Result<()>`.
- `fuel-dispatch/src/kernel.rs:895` — `register_full_with_source(...)` **panics** on duplicate `KernelRef`
  (kernel.rs:910-918). This is the never-panic prerequisite below.
- `fuel-core-types/src/backend.rs:97,121,147` — `SubstrateClass`, `TransferPath`, `BackendCapabilities`
  (`op_dtype_support`, `required_alignment`, `access_granularity_bits`, `storage_substrate`).
- `fuel-core-types/src/probe.rs:63,182` — `BackendId`, `BackendProbe` (a *device-enumeration* marker
  convention, NOT a capability surface; see §6 — we extend the capability side, not this trait).
- `fuel-core-types/src/storage.rs:46` — `OutputView { byte_offset, len_elements, dtype, shape, layout, name }`
  (the bundle slot; maps to `FDXOutputView`, rank-capped + name side-table per FDX §6.8 / FKC §5.5).
- `fuel-graph/src/registry.rs:104,108,133,159` — `FusedOp.shape_rule` / `dtype_rule` / `output_views`;
  `FusedOpParams` (incl. `FlashAttn { k_len: Option<DynScalar> }`, `PagedAttn`, `QMatMul`).
- `fuel-dispatch/src/fused.rs:63` — fused `BackendImpl.cost: fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate`
  (no `&[DType]` — distinct from the primitive `CostFn` at kernel.rs:713).

---

## 0. Scope and non-scope

**In scope (this program):**
1. A new `fuel-core-types::dlpack` module behind a `dlpack` cargo feature: the `#[repr(C)]` POD
   structs from FDX §5 (`DLTensor`, `DLManagedTensorVersioned`, `DLPackVersion`, `FDXSidecar`,
   `FDXDTypeExt`, `FDXQuant`, `FDXExtent`, `FDXTiling`, `FDXResidency`, `FDXStorage`, `FDXBufferRef`,
   `FDXOutputView`, `FDXIndexedResidency`) + a co-maintained C header `fuel_dlpack_ext.h`.
2. The shared-code conversion layer (FDX §6.0): `fdx_code(DType) -> Result<u16>` etc., build-time mapping test.
3. View construction at the call boundary: `(&Storage, &Layout[, &SymEnv]) -> DlpackView<'a>` with **no copy**.
4. Both boundaries: (a) explicit nullable `*const FDXSidecar` next to the `DLTensor` on the internal
   kernel ABI; (b) the external `__dlpack__` capsule path with deleter-identity-gated `manager_ctx`.
5. The honesty-invariant enforcement points (a validator + the producer-policy gates).
6. Capability negotiation: extend the capability surface so the planner reads per-kernel FDX-extension
   acceptance (driven by FKC contracts) and decides materialize-vs-zero-copy.
7. The `KernelRef` ABI migration to DLPack-shaped operands (additive, behind the feature first).
8. The `register_full_with_source` never-panic conversion (prerequisite).

**Out of scope (sequenced behind, [consumer-ahead]):**
- `.fuel` mmap alignment reconciliation (FDX §"little-endian v1"; deferred to Phase E).
- Per-tier memory cost vectors + `SymEnv`-aware cost re-evaluation (FKC P8; capacity-only costing for v1).
- The FKC parser/importer itself (its own program). This program builds the *types and view* FKC
  references; the FKC importer consumes them later.
- Data-determined syms (mid-pass producer→consumer sym edges). The structs carry the fields; no consumer.

---

## 1. The new `fuel-core-types::dlpack` module (feature-gated)

### 1.1 Feature + module skeleton

- `fuel-core-types/Cargo.toml`: add `dlpack = []` (no deps; pure POD + conversions). The structs are
  `#[repr(C)]` and contain no Fuel types in their serialized form, so the feature is dependency-light.
  Gate it OFF by default — every other crate that wants the boundary turns it on (`fuel-memory`,
  `fuel-dispatch`, each backend wrapper crate).
- New files:
  - `fuel-core-types/src/dlpack/mod.rs` — re-exports; `#![cfg(feature = "dlpack")]` at the crate
    `lib.rs` mod declaration (`#[cfg(feature = "dlpack")] pub mod dlpack;`).
  - `fuel-core-types/src/dlpack/abi.rs` — the standard DLPack structs (FDX §5.1): `DLDevice`,
    `DLDataType`, `DLDeviceType`, `DLDataTypeCode`, `DLTensor`, `DLPackVersion`,
    `DLManagedTensorVersioned`, the `DLPACK_FLAG_BITMASK_*` consts. Reproduced from `dlpack.h` v1.3,
    validated against it (see §8 test gate). NOT redefined by Fuel — these are the canonical names.
  - `fuel-core-types/src/dlpack/sidecar.rs` — `FDXSidecar` + all embedded sub-structs (FDX §5.3, §6).
  - `fuel-core-types/src/dlpack/codes.rs` — the FDX-owned stable code tables + explicit conversions
    (FDX §6.0). This is the **only** place a numeric FDX code is assigned.
  - `fuel-core-types/src/dlpack/validate.rs` — the V1–V21 validators (FDX §"Validator checks"), all
    `Result`-returning, no `try_*` siblings.
  - `fuel-core-types/include/fuel_dlpack_ext.h` — the C header. Hand-maintained against the Rust
    `#[repr(C)]` types; a layout test (§8) pins parity.

### 1.2 The structs — discipline

- Every multi-byte field little-endian in serialized form; `#[repr(C)]` with explicit padding fields
  (`_pad0: u32` etc., exactly as FDX §5.3 lays out). Reserved arrays zero on write, ignored on read.
- **Variable-length arrays:** live in-memory form is `(count: u32, ptr: *const T)`; the serialized form
  replaces the pointer with a byte offset relative to the sidecar blob start (FDX P7). For v1 we build
  only the **live in-memory** form (kernel handoff); the serialized/persisted form is wired in the
  `.fuel` program (Phase E). Mark the offset-based serializer `[consumer-ahead]`.
- **Inline `[;6]` mirrors `DimVec`** (FDX §"Inline shape/stride [;6]"): `DLTensor` shape/strides for the
  kernel-ABI fast path use `[i64; 6]` inline arrays with an explicit `RankExceeds6` error beyond rank 6;
  the `*const i64` form is for the external/persisted path. Sub-structs (`FDXExtent`, `FDXBlockTable`,
  `FDXAffine`) are frozen-size, grow only via their own `reserved[]`.
- **Size discipline (FDX §5.4):** `const`-assert every sub-struct size and `offset_of!` per field on the
  array-element structs (`FDXExtent`, `FDXIndexedResidency` fields after `logical_extents`). A
  `struct_bytes` round-trip test complements the static pins. These are born-red: write the
  `assert_eq!(size_of::<FDXSidecar>(), N)` first against the spec's number, watch it fail until the
  struct is laid out, then green.

### 1.3 Shared codes (FDX §6.0 — single source of truth)

`codes.rs` is the normative owner. Conversions are **explicit `match`, never `as u16`** on a source
enum discriminant:
- `pub fn fdx_code(d: DType) -> Result<u16>` / `dtype_from_fdx(u16) -> Result<DType>`
- analogous for `BackendId` → substrate/backend code, `SubstrateClass`, and the quant `ScaleGranularity`.
- The build-time mapping test (§8) asserts the full table **by name**, so retiring/reordering a source
  variant (e.g. the already-retired `BackendId::Aocl`/`Mkl`) breaks the build on the exhaustive `match`
  rather than silently shifting the ABI. Sub-byte dtypes (`F4`/`F6E2M3`/`F6E3M2`, `size_in_bytes()==0`)
  map to a real bit-width + packing in `FDXDTypeExt`, never to a byte size (FDX §3.4, P5).

---

## 2. Constructing the DLTensor + FDXSidecar VIEW from `(Storage, Layout)` — no storage rewrite

This is the load-bearing realization step. **Nothing about `Storage` or `Layout` changes.** The view is
a thin, borrowed, zero-copy projection assembled at the call boundary.

### 2.1 The borrowed view type

New in `fuel-memory` (it is where `Storage` + the backend variants live), behind `dlpack`:
`fuel-memory/src/dlpack_view.rs`:

```rust
/// A borrowed DLPack+FDX view over a Fuel (Storage, Layout) pair. Holds
/// the DLTensor (with inline [i64;6] shape/strides), an optional owned
/// FDXSidecar, and PhantomData tying it to the borrowed Storage+Layout so
/// the `data` pointer cannot dangle. Constructed per kernel call; never
/// persisted; never owns the bytes.
pub struct DlpackView<'a> {
    pub dl: DLTensor,                 // data ptr borrowed from Storage; shape/strides from Layout
    pub sidecar: Option<FDXSidecar>,  // None ⇒ plain DLPack (P2 absence state)
    _shape: [i64; 6],                 // backing store for dl.shape (inline path)
    _strides: [i64; 6],               // backing store for dl.strides (signed)
    _marker: PhantomData<(&'a Storage, &'a Layout)>,
}
```

Constructor:
```rust
pub fn view<'a>(
    storage: &'a Storage,
    layout: &'a Layout,
    env: Option<&SymEnv>,        // Some for symbolic axes; resolves the live sidecar extents
) -> Result<DlpackView<'a>>;
```

### 2.2 Field-by-field mapping (the actual code path)

- **`dl.data`** — the raw base pointer of `storage.inner`. Per-backend, via the existing
  `dispatch_storage!` macro (fuel-memory/src/lib.rs:107): CPU → `CpuStorageBytes::bytes().as_ptr()`
  (byte_storage.rs:85); CUDA → the device pointer; Vulkan → the `VkBuffer`-relative base. The pointer
  is **never folded with an offset** — see `byte_offset`.
- **`dl.device`** — from `storage` backend variant → `DLDeviceType` (CPU=kDLCPU, Cuda=kDLCUDA,
  Vulkan=kDLVulkan) + `device_id` from the storage's `DeviceLocation`. (`SubstrateClass` precision —
  Vulkan-vs-CUDA-same-silicon — rides the FDX `FDXResidency` substrate code, FDX §6.6, because plain
  DLPack `device_type+device_id` is too coarse; digest §"residency + substrate".)
- **`dl.ndim` / `dl.shape`** — `layout.dims()` (= the **capacity** bounds; layout.rs:58, `Range::max`
  for symbolic axes). This is the capacity-honesty corollary (FDX §3.1): shape reports capacity,
  strides key to capacity, the live count is a *different fact* in the sidecar. Rank > 6 → `RankExceeds6`.
- **`dl.strides`** — `layout.stride()` (signed `&[isize]`, layout.rs:94) cast to `i64`. **Negative
  strides pass through unchanged** — they are first-class (FDX §3.2.1). Never NULL (FDX §3.2 / V11):
  for a contiguous layout we write the explicit row-major strides, not the legacy NULL convention.
- **`dl.byte_offset`** — `layout.start_offset() * dtype.size_in_bytes()` (start_offset is in elements,
  layout.rs:116; DLPack byte_offset is bytes). Carries the iteration-first element and **every** intra-
  buffer start (KV live-prefix start, quant sub-buffer start, bundle slot start) — never folded into
  `data` (FDX §3.3, the 256-byte rule).
- **`dl.dtype`** — for a standard dtype, the faithful `DLDataType` (F32→{kDLFloat,32,1}, etc.). **For a
  quant / sub-byte / MX payload, the honesty stand-in `{kDLUInt, 8, 1}`** with the true semantics only in
  the sidecar (FDX §3, P1). `dlpack::codes::dl_dtype(DType) -> Result<DLDataType>` decides; the quant
  case is driven by whether the consuming op marks the operand quant (the op/registry already knows —
  `OpParams::QMatMul`, `FusedOpParams::QMatMul`, `Nf4Matmul`).

### 2.3 The sidecar — built only when there is non-standard meaning

`sidecar = None` for a plain, fully-faithful standard tensor (P2: absence = "plain DLPack"). It is
`Some(FDXSidecar)` when **any** of these hold, with the matching `FDX_FLAG_*` bit set (FDX §5.2 table):
- the dtype needed the `uint8` honesty stand-in → `FDX_FLAG_HAS_DTYPE_EXT` + `FDXDTypeExt`;
- the operand is quantized → `FDX_FLAG_HAS_QUANT` + `FDXQuant` (block format/size/scale ref, from the
  op's quant params); for separate-input scales the scale is an FKC operand, NOT an `FDXQuant.scale_buffer`
  (the resolved single-place rule, FDX §"Quant scale operand rule" / FKC §3.9.3);
- `layout.has_dynamic()` (layout.rs:75) → `FDX_FLAG_HAS_SYMBOLIC` + `extents[]` from `Shape`'s `Extent`s:
  each carries `SymId`, `[min,max]` capacity, and `kind` (Scalar/Range/Affine). The `SymId`s transport
  the *symbol*, not the resolved value (digest §6 — plan once, serve every token). When `env` is `Some`,
  we additionally validate `min <= env.get(sym) <= capacity` (V14) but **never bake the resolved value
  into the description** — resolution stays at realize (P4).
- paged/blocked pool → `FDX_FLAG_HAS_GATHER` + `FDXIndexedResidency` (PagedAttn; FDX §6.9). Mandatory
  `FDX_FLAG_MEANING_REQUIRES_EXT` (V18/V19).
- `storage.is_bundled()` → `FDX_FLAG_IS_BUNDLE` + `views[]` from `storage.bundle()` (`OutputView` →
  `FDXOutputView`, rank≤6, name in the side-table; FDX §6.8 / FKC §5.5).
- the capacity tail is unbacked / mmap-of-live-region → `FDX_FLAG_MEANING_REQUIRES_EXT` (FDX §3.1, V8).

`buffers[0]` is **always** the base `DLTensor`'s data buffer (FDX §7.4); scale/zero-point/bundle-slot
buffers are appended. `magic = FDX_MAGIC`, `version = FDX_VERSION_1`, `struct_bytes = size_of::<FDXSidecar>()`.

### 2.4 Symbolic axes — capacity for layout, symbol for liveness (P4)

The view's `dl.shape` is capacity; the live length lives in `sidecar.extents[i]`. The realize boundary
calls `layout.resolve(env)` (layout.rs:84) to get a concrete-extent Layout for the *kernel's* own
indexing where it needs the live count, but the **view handed across the boundary keeps capacity shape +
symbolic sidecar** so one description is replay-stable (digest §11 operand rebasing: same recorded run,
new base pointer + new `SymEnv`). This is why the sidecar transports `SymId`, not `usize`.

---

## 3. The two boundaries

### Boundary (a) — internal Fuel kernel ABI: explicit nullable param

The cleanest carrier (FDX §10, no smuggling): the sidecar travels as a separate `*const FDXSidecar`
parameter *next to* the `DLTensor`, `null` = absent. This is the form the migrated `KernelRef` ABI uses
(§5). No `manager_ctx`, no deleter — Fuel owns both ends, lifetimes are the call scope. Internal
launches MAY relax 256-byte alignment to the backend's `required_alignment` (FDX §3.3 note), since no
external CUDA-256 assumption is in play.

### Boundary (b) — external `__dlpack__`: managed/deleter

The Python/C capsule signature is fixed — no extra-pointer slot. So:
- We export a `DLManagedTensorVersioned` (FDX §5.1). The base `DLTensor` is **always** valid standard
  DLPack on its own (honesty invariant).
- The FDX sidecar rides `manager_ctx` **iff** the consumer advertised FDX understanding **and** the live
  `deleter` function-pointer identity matches Fuel's own deleter (FDX §10.2 — deleter-identity gating). A
  forwarded/wrapped third-party capsule is never downcast on the magic word alone.
- Otherwise only the standard part crosses, and **producer policy (§4) governs whether that is even
  allowed**: a meaning-requires-ext tensor is refused or materialized (with `IS_COPIED`).
- **Strict mode here:** explicit strides (V11), 256-byte-aligned `data` (V12), signed-stride OOB in
  range (V13). A misaligned internal buffer is materialized to an aligned copy before export.
- `manager_ctx` is the **opportunistic fallback only** on the pure standard capsule path. Where we
  control the signature (Fuel↔Baracuda/Vulkane native calls), we use the explicit sidecar param
  (boundary a), never `manager_ctx` (the resolved cross-runtime-transport decision).

---

## 4. Honesty-invariant enforcement points

The invariant (FDX §3, P1): the base `DLTensor` is always honestly interpretable as standard, conformant
DLPack on its own — never a mislabeled dtype, mis-sized buffer, NULL strides, or misaligned pointer.
Enforcement is **mechanical at three points**, all `Result`, never panic:

1. **Construction (`view()` / sidecar build) — `dlpack::validate::validate_view(&DlpackView)`** runs the
   applicable V-checks before the view is handed to any consumer:
   - V11 strides non-NULL when ndim≠0; V13 signed touched-range ⊆ `[0, size_bytes)` (the `Layout::flip`
     invariant, computed over signed strides per FDX §3.2.1); V8 `buffers[0].size_bytes` ≥ capacity-shaped
     extent (else `MEANING_REQUIRES_EXT` must be set); V14 `min ≤ env value ≤ capacity` when `env` given;
     V16/V17 affine extent well-formedness; V18–V21 gather coherence; the dtype-honesty check (base dtype
     is `uint8` whenever `FDX_FLAG_HAS_QUANT`/`HAS_DTYPE_EXT`).
   - `FDXError`/`FdxError` variants: `RankExceeds6`, `StrideRangeOutOfBounds` (replaces the withdrawn
     `NegativeStride`), `NullStrides`, `CapacityNotBacked`, `BundleRankExceeds6`, `ScaleDoubleDeclared`,
     `GatherAddressOverflow`, `MagicMismatch`, `UnknownFdxCode`. One typed error per check.
2. **Boundary-(b) export — `export_managed()`** additionally runs V12 (256-byte alignment) and the
   producer policy: if `FDX_FLAG_MEANING_REQUIRES_EXT` is set and the consumer is sidecar-blind →
   **refuse-or-materialize** (FDX §9.1). Materialize sets `DLPACK_FLAG_BITMASK_IS_COPIED`. A KV cache to
   bare PyTorch becomes a *materialized dense copy* of the resolved-length live region, never the raw
   capacity buffer (FDX §3.1, §3.1.1 — a non-leading symbolic axis is a copy, not a view).
3. **Import — `import_managed()`** (consuming a foreign capsule into Fuel): legacy NULL strides are
   normalized to explicit strides at the boundary before any FDX reasoning (FDX §3.2 note); an
   unrepresentable tensor is a typed error naming the offending fact, never a silent coercion or drop
   (digest §3; constitution never-panic / no-silent-fallback).

> The invariant is one-directional about safety (FDX §3): ignoring the sidecar can lose *meaning* (opaque
> bytes) but can never produce *wrong numbers*. The validator is what makes that mechanical.

---

## 5. Capability negotiation — planner decides, backend advertises

FDX is pure description and carries **no** cost and **no** decision (P3/G7). The decision —
zero-copy-vs-materialize, accept-negative-strides-vs-normalize, accept-paged-vs-densify — is the
**planner's**, made from the **consumer's advertised capability**. The advertisement axis is FKC; the
capability tokens live on the existing capability surface.

### 5.1 What we extend (and what we do NOT)

- **Do NOT overload `probe::BackendProbe`** — it is device *enumeration*, not capability (probe.rs:182).
- **Extend the `Capability` token enum** (`fuel-core-types/src/capability.rs`, `#[non_exhaustive]`) with
  the FDX-extension acceptance tokens the specs name: `DlpackExtGather`, `DlpackExtSymbolic`,
  `DlpackExtAffine` (affine *quant*, distinct from the symbolic-extent token — FKC §3.9.2 warns these are
  two tokens), and the layout capability that backs `reverse_strides`. These are *facts a backend
  advertises*, consumed by the planner.
- **Grow `KernelCaps`** (kernel.rs:66, "forward-extensible by adding fields") toward the FKC five-flag
  layout set: today only `strided_input: bool`. Add `reverse_strides: bool` first (the resolved
  negative-stride decision needs it). Today's importer projects `(strided && broadcast_stride0)` onto
  `strided_input` and treats `reverse_strides` as not-declared → normalize until the flag lands (FKC
  §4.1, §12.2). The full five-flag set is retained for forward use.

### 5.2 The planner's negative-stride / paged / symbolic decision (the resolved rule)

When the view carries negative strides, a gather block, or a symbolic extent, the planner inserts a
**materialized normalizing copy** (an `Op::Contiguize`-class node, costed from *its own* FKC contract,
exporting `IS_COPIED`) **only** when the chosen consumer cannot handle it:
- the consumer kernel's `KernelCaps`/FKC contract does **not** declare the capability, **OR**
- it is a bare standard-DLPack handoff to a non-cooperating external ecosystem (boundary b).

It is **never** universal and **never** inserted between capable internal kernels — an internal zero-copy
`Op::Flip` (which `Layout::flip` already makes a metadata-only view) feeding a `reverse_strides`-capable
kernel stays a view (FDX §3.2.1, FKC §4.1.1). The decision lives in the planner from the capability
advertisement, never as an FDX-boundary trial.

---

## 6. KernelRef ABI migration to DLPack-shaped operands

The as-built ABI (kernel.rs:152) is `fn(&[Arc<RwLock<Storage>>], &mut [Arc<RwLock<Storage>>], &[Layout],
&OpParams) -> Result<()>` — inputs+outputs as `Storage`, layouts as a parallel `&[Layout]` side-channel,
`OpParams` for extras. **The DLPack view is exactly the `(Storage, Layout)` pairing this ABI already
splits** — so the migration is additive, not a rewrite.

### 6.1 Strategy — additive, feature-gated, born-red per backend

- **Step 1 (no ABI change):** ship `view()` + the validator + the new caps. Add a `#[cfg(feature =
  "dlpack")]` *adapter*: `fn dlpack_operands<'a>(inputs, outputs, layouts) -> Result<(Vec<DlpackView<'a>>,
  Vec<DlpackViewMut<'a>>)>` that zips the existing `(Storage, Layout)` slices into views. This proves the
  view round-trips against every live kernel's inputs **before** any kernel signature changes. Born-red:
  a test that builds views for a known op, asserts shape=capacity / strides=signed / byte_offset correct,
  fails until `view()` exists.
- **Step 2 (new ABI variant, opt-in):** introduce `KernelRefDlpack = fn(&[DlpackView], &mut [DlpackViewMut],
  *const FDXSidecar-per-operand-folded-into-the-view, &OpParams) -> Result<()>` — the sidecar rides
  *inside* each `DlpackView` (boundary a), so the signature stays four-ish args. Add a parallel
  `BindingEntry` field or a `KernelRef` enum `{ Legacy(KernelRef), Dlpack(KernelRefDlpack) }`; the executor
  dispatches whichever a binding registered. This keeps every un-migrated kernel working unchanged.
- **Step 3 (per-backend wrapper migration):** migrate one backend's wrappers at a time (CPU first — it is
  always-built and the precision-coverage anchor). Each migrated wrapper takes `DlpackView` instead of
  `(&Storage, &Layout)`, reads `dl.shape`/`dl.strides`/`dl.byte_offset` + the sidecar, and calls the same
  typed kernel underneath. **One backend's live-GPU suite at a time** (environment discipline: two
  concurrent live suites OOM the 4070).
- **Step 4 (retire the legacy arm):** once all wrappers are migrated, fold `KernelRef` to the DLPack
  shape and delete the adapter. This is the end state; sequence it last.

### 6.2 OpParams ↔ FDX division of labor

`OpParams`/`FusedOpParams` stay (they carry op math/geometry the *tensor* doesn't: reduce dims, conv
geometry, `softmax_scale`, `k_len: Option<DynScalar>`). FDX carries only *what a tensor is*. The overlap
to delete is **layout/shape duplication** already migrated to the `layouts` side-channel — that now flows
through the `DlpackView`'s `dl.shape`/`dl.strides`/`byte_offset`, so a migrated kernel reads geometry from
the view, not from re-derived shape fields. `k_len` stays in `FusedOpParams::FlashAttn` (it is a lowering
param, not a tensor property — digest §6 length-vs-mask) but its symbol is *also* discoverable in the
operand's sidecar extent (they must agree; cross-check at build).

---

## 7. Prerequisite — `register_full_with_source` must stop panicking (never-panic)

`register_full_with_source` (kernel.rs:895) **panics** on a duplicate `KernelRef` (kernel.rs:910-918).
The FKC importer (and any contract-driven registration) must not be forced to drive a panicking path
(constitution never-panic; FKC §10.10 CONSTITUTION-CONFLICT callout). This is a hard prerequisite in the
conversion plan:

- Change `register_full_with_source` (and the `register*` family that funnels into it) to **return
  `Result<()>`**, with a typed `Error::DuplicateKernelRef { op, dtypes, backend }` on the exact-duplicate
  case instead of `panic!`.
- Update the ~handful of in-crate callers (the `register_*_kernels` bulk passes in dispatch.rs,
  vulkan_dispatch.rs, baracuda_dispatch.rs, pipelined.rs, and the two CPU-extension binding tables in
  `fuel-mkl-cpu-backend` / `fuel-aocl-cpu-backend`) to propagate the `Result`. Module-init registration
  that wants fail-fast can `.expect()` at the *call site* (that is a startup-config error, an acceptable
  place), but the library function itself never panics.
- Born-red: a test that registers a duplicate `KernelRef` and asserts `Err(DuplicateKernelRef)` (replacing
  the current `#[should_panic(expected = "duplicate KernelRef")]` at kernel.rs:1351). Watch the old
  should_panic test go red against the new signature, rewrite it to assert the `Err`.

---

## 8. Test gates (born-red first, every one observed to run)

Run all with `-p <crate>` only (never workspace-wide; `tensor-tools` breaks bare `cargo check`). One
cargo invocation at a time.

| # | Gate | Crate | What it pins |
|---|------|-------|--------------|
| T1 | `size_of`/`offset_of!` assertions for every FDX sub-struct + `FDXSidecar` total | `fuel-core-types` (`--features dlpack`) | FDX §5.4 size discipline; born-red against the spec numbers |
| T2 | C-header ↔ Rust layout parity (`fuel_dlpack_ext.h` field offsets vs `offset_of!`) | `fuel-core-types` | header co-maintenance; FDX G5 |
| T3 | DLPack v1.3 conformance: a faithful F32 view is accepted by a strict v1.2+ consumer model (explicit strides, 256-align, ndim≠0) | `fuel-core-types` | FDX §3.2/§3.3, V11/V12 |
| T4 | `fdx_code`/`*_from_fdx` round-trip **by name** for every `DType`/`SubstrateClass`/`BackendId`/`ScaleGranularity` | `fuel-core-types` | FDX §6.0; breaks on a retired/reordered source variant |
| T5 | signed-stride OOB (V13): an `Op::Flip` view of `[a,b,c]` (stride `[-bc,c,1]`, offset at last row) is in-range; an out-of-range fabricated negative stride is `Err(StrideRangeOutOfBounds)` | `fuel-core-types` | the `Layout::flip` invariant ⇒ FDX V13 |
| T6 | capacity-honesty (V8): a symbolic KV view reports capacity shape; an under-backed buffer without `MEANING_REQUIRES_EXT` is `Err(CapacityNotBacked)` | `fuel-core-types` / `fuel-memory` | FDX §3.1 |
| T7 | `view()` builds a correct sidecar-`None` view for a plain F32 `(Storage, Layout)`, and a `uint8`+`FDX_FLAG_HAS_QUANT` view for a Q4_0 storage (dtype honesty) | `fuel-memory` (`--features dlpack`) | FDX §3, P1 |
| T8 | symbolic view transports `SymId` (not resolved value); `resolve(env)` only at the indexing site; `min≤env≤cap` enforced (V14), unbound sym → typed error | `fuel-memory` | digest §6, P4 |
| T9 | bundle view: a 2-slot bundled `Storage` → `FDX_FLAG_IS_BUNDLE` + 2 `FDXOutputView` with names in side-table; rank-7 slot → `Err(BundleRankExceeds6)` | `fuel-memory` | FDX §6.8 / FKC §5.5 |
| T10 | `register_full_with_source` returns `Err(DuplicateKernelRef)` (not panic) on exact-dup; distinct alts still append | `fuel-dispatch` | §7 never-panic prerequisite |
| T11 | KernelRef adapter: `dlpack_operands` zips a real op's inputs into views matching the legacy `(Storage, Layout)` semantics byte-for-byte | `fuel-dispatch` (`--features dlpack`) | §6.1 step 1 |
| T12 | planner inserts a normalizing copy (`IS_COPIED`) for a negative-stride operand into a `reverse_strides:false` kernel, and NOT into a `reverse_strides:true` kernel | `fuel-dispatch` | §5.2 resolved rule |
| T13 | boundary-(b) producer policy: a `MEANING_REQUIRES_EXT` KV export to a sidecar-blind consumer materializes a dense copy with `IS_COPIED`, never the raw capacity buffer | `fuel-memory`/`fuel-dispatch` | FDX §9.1, §3.1.1 |

---

## 9. Sequencing (ordered; each step ships with its gate, observed red→green)

1. **§7 never-panic prerequisite** — `register_full_with_source -> Result`, callers propagate. Gate T10.
   *(First, because it is small, independent, and unblocks contract-driven registration.)*
2. **§1 module + structs + codes** — `dlpack` feature, `abi.rs`/`sidecar.rs`/`codes.rs`, the header.
   Gates T1, T2, T4. (T3 once the structs exist.)
3. **§1.4 validators** — `validate.rs`, every V-check `Result`. Gates T5, T6 (with §2).
4. **§2 view construction** — `DlpackView` + `view()` in `fuel-memory`. Gates T7, T8, T9, T3.
5. **§5 capability surface** — `Capability` tokens + `KernelCaps.reverse_strides`. (Facts only; no
   behavior yet — the planner consumes them in step 7.)
6. **§6.1 step 1 adapter** — `dlpack_operands` zip; prove round-trip. Gate T11.
7. **§5.2 planner decision** — normalize-only-for-incapable-consumer. Gates T12, T13.
8. **§3 boundary (b) export/import** — `export_managed` / `import_managed`, deleter-identity gating.
   (Folds T13's export half.)
9. **§6.1 steps 2–4** — new ABI variant, per-backend wrapper migration (CPU first, then GPU one suite at
   a time), legacy-arm retirement. Each backend's migration is its own commit + its own live gate.

---

## 10. Docs to update with this change (constitution discipline)

- `docs/architecture/05-backend-contract.md` — the `KernelRef` ABI now has a DLPack-shaped operand form;
  bump version + `10-decisions-log.md` entry on the MAJOR (ABI) change. Note `register_full_with_source`
  is now `Result` (never-panic standing-violation closed for this path).
- `docs/architecture/13-interchange.md` — the tensor/storage-axis interchange now has a concrete
  realization (FDX view); cite this plan + the FDX spec.
- `ROADMAP.md` — move the FDX/FKC frontier from "design" to "in implementation"; this plan is the path.
- Flip both spec headers' "no code yet" / "Nothing here is implemented" lines as each step lands, and
  reconcile the FKC §4.1.1 `[cross-spec dependency]` note (it still says FDX bans negative strides on
  export — FDX has since reversed to first-class; the note's premise is stale and should be struck once
  the validator in step 3 ships V13 as the signed-range check).

---

## 11. Risks / watch-items

- **Pointer lifetime at boundary (a).** `DlpackView` borrows `&Storage`; the `Arc<RwLock<Storage>>` lock
  must be held for the kernel call's duration. The migrated ABI must take the read/write guard's lifetime,
  not a detached pointer — `PhantomData<&'a Storage>` enforces this at compile time. Verify no migrated
  wrapper leaks the `dl.data` pointer past the guard.
- **Sub-byte byte-sizing.** `dtype.size_in_bytes() == 0` for `F4`/`F6E2M3`/`F6E3M2` (digest §8). The
  `byte_offset = start_offset * size_in_bytes` formula must use the **physical packed byte width** from
  `FDXDTypeExt`/`FDXPacking` for these, not `size_in_bytes` — a sub-byte operand's base is `uint8` with
  the packing in the sidecar, so the byte view sizes off the physical buffer (FDX §3, P5). Cover in T7.
- **CUDA/Vulkan `data` is a device pointer, not host-addressable.** The 256-byte check (V12) is a
  numeric `(ptr as usize) % 256` on the device-pointer value — fine. But never *dereference* `dl.data` on
  the host; the view is metadata only. Backend wrappers translate to the backend's typed buffer handle.
- **Affine extent i128 widening (FDX §6.4 critique fix).** The realize-eval `c0 + Σ cᵢ·resolve(symᵢ)`
  must widen both operands to i128 before multiply (a u64 sym `> i64::MAX` mis-casts in i64). Pin in T8's
  affine arm.
