# Architecture constraints for a kernel-contract format + a DLPack extension

**Purpose.** This digest extracts the *forward-looking* constraints from Fuel's
architecture set, ROADMAP, and the symbolic-extents design that any **kernel-contract
format** (how a kernel advertises what it is, what it costs, how precise it is, what
layouts/dtypes it accepts) and any **DLPack-style tensor-interchange extension** (how a
tensor — its storage, dtype, shape, layout, quant metadata — is described across the
kernel/backend boundary) must plan for. It is research input, not a spec.

Sources read (all under `docs/architecture/` unless noted): 00-index, 01-identity,
03-ir, 04-optimization, 05-backend-contract, 06-runtime, 07-tolerance, 09-non-goals,
10-decisions-log, 11-persistence, 12-multi-output, 13-interchange, 14-lifecycle;
`ROADMAP.md`; `docs/session-prompts/symbolic-extents-and-persistent-decode.md`;
`docs/session-prompts/quantize-as-graph-op.md`; and the as-built types in
`fuel-core-types/src/{dtype.rs, quant_scale.rs, quantized.rs, capability.rs,
backend.rs, symbol.rs, shape.rs}` + `fuel-dispatch/src/kernel.rs`.

A note on doc-vs-code drift the reader must keep in mind: the architecture's **Intended**
state is partly aspirational. The current frontier (per 14-lifecycle and ROADMAP) is
**Phase D** — symbolic extents + persistent decode. Phases A–C have landed (multi-path
`Op::Branch`, per-device Pareto frontier + crowding cap, runtime route picker);
load-time build, sessions, and mmap persistence are still Intended. Both specs are being
designed *ahead of* the steady state, so they should target the Intended architecture
while staying loadable by today's code.

---

## 1. The governing principle: backends advertise; the planner decides

This is the single most load-bearing constraint, repeated across 01/04/05/09. **Every
strategic decision — placement, fusion, kernel-variant selection, slot assignment,
tolerance trade-off, dtype choice — lives at the DAG/optimizer level. Backends only
advertise capabilities/costs/precision/telemetry and execute what they are handed.**

Direct consequences for both formats:

- A kernel contract is a **pure description consumed by the optimizer**, never a place
  where a kernel makes a choice. It must surface *everything the optimizer needs to
  reason*, with no hidden behavior. The architecture explicitly rules out: backend-internal
  placement, backend-internal fusion, silent kernel-variant substitution, backend-internal
  result caching (05 §"What this rules out"). A contract format that lets a kernel say "I'll
  pick internally" violates the constitution.
- The enforcement gate (01 §"How this identity is enforced"): a change passes only if it
  makes *more decisions visible to the optimizer* / *more cost data flow to the optimizer* /
  *more algebraic rewrites reachable* / *more of the tolerance space reachable*, and none of
  these less true. **The kernel-contract format should be designed to maximize what the
  optimizer can see.** Hiding any cost/precision/layout fact behind a backend abstraction is
  a constitutional failure.
- The DLPack extension is the **data-description half of the same boundary**: it describes a
  tensor *as handed to a kernel*. It must therefore be expressive enough to carry every fact
  the kernel contract references (dtype incl. sub-byte + quant, layout incl. strided/
  symbolic, device/substrate, multi-output bundles) so the optimizer's pre-priced decisions
  survive the handoff intact.

---

## 2. Capability / cost / telemetry: the three advertisement surfaces

The contract is **static capability advertisement** (frozen at backend registration) +
**dynamic telemetry** (heartbeat/on-demand). The kernel-contract format must distinguish
these — they have different lifecycles and different consumers (optimizer vs route picker).

### 2a. Static, per-kernel (the heart of a kernel contract)

Registered once per kernel, frozen for process lifetime (05 §"Per-backend kernel
registration"; as-built `KernelRef` ABI in `fuel-dispatch/src/kernel.rs`):

- **Op + per-operand dtypes** the kernel implements — the dispatch key. As-built this is
  `(OpKind, [DType…], BackendId, kernel_source)`; the `op_dtype_support: HashSet<(OpKind,
  DType)>` set on `BackendCapabilities` is what routing checks via `contains`.
- **`KernelRef` function pointer** — the call surface:
  `fn(inputs: &[Arc<RwLock<Storage>>], outputs: &mut [Arc<RwLock<Storage>>],
  layouts: &[Layout], params: &OpParams) -> Result<()>`. **Hard rules baked into the ABI:**
  outputs are *pre-allocated by the executor* (kernels never allocate), and kernels **return
  `Result`, never panic** (§3). A contract format must encode this contract, not just the
  signature.
- **`CostEstimate { flops, bytes_moved, kernel_overhead_ns }`** as a function
  `cost(shapes, params, capabilities)`. Convention: **pessimistic upper bound** ("when in
  doubt, round up"). The optimizer composes these over a path as a **cost vector** (§5).
- **`PrecisionGuarantee`** — the precision contract (§4); the format's hardest-to-bolt-on
  field, so design it in from the start.
- **`KernelCaps`** — capability flags. Today only `strided_input: bool`, but the field is
  explicitly forward-extensible ("future capability flags"). The optimizer reads these to
  decide whether to insert an `Op::Contiguize` layout fixup. **The contract format must make
  KernelCaps an open/extensible set, not a fixed bitfield**, because new layout/quant/symbolic
  capabilities will be added.
- **Required allocation alignment (bytes) + access granularity (bits)** — the router pads/
  repacks to meet alignment and routes around granularity limits. (As-built on
  `BackendCapabilities`: `required_alignment`, `access_granularity_bits`.) **A DLPack
  extension must carry/honor alignment + granularity** so the optimizer's repack decisions are
  not silently violated at the boundary.
- **Outbound transfer paths** `(DeviceLocation, TransferPath)` — `SameDevice | Peer |
  SharedMemory | DeviceCopy | HostStaging` (as-built `TransferPath` enum). Plus
  **`SubstrateClass`** (`HostBytes | CudaUntyped | VulkanBuffer | MetalBuffer`,
  `#[non_exhaustive]`): two backends sharing a substrate class on the same device can
  interleave on one `Storage` handle *with no copy*. **The DLPack extension must expose
  substrate/device identity finely enough to decide "same buffer, vtable swap" vs "needs a
  copy."** This is a place generic DLPack (device_type + device_id) is too coarse: Vulkan and
  CUDA on the *same silicon* have different pointer namespaces and must not alias.
- **Slot capacity per device** — max concurrent execution contexts (CUDA streams, Vulkan
  queues, CPU pool size). Read by optimizer (parallelism budget) and route picker.
- **Kernel-revision hash** — stable per-implementation-version hash; the persistence layer
  detects when a cached plan referenced a since-updated kernel (§7). **A kernel-contract
  format must carry a revision hash as a first-class field**; it is the cache-invalidation
  primitive.

### 2b. Dynamic telemetry (route picker consumes; per-realize)

Per 05 §"Dynamic telemetry" + 06: currently-available slot count, **per-tier memory
pressure** (the route picker prefers a path whose binding tier has headroom), queue depth,
currently-resident weights, and accumulating local Judge profile data. Honesty contract:
`Option<u64>` returns (`available_bytes`/`total_bytes`) — `None` means "can't measure, fall
back to static cost," never "zero." `FitStatus { Comfortable | Tight | WontFit | Unknown }`.

**Constraint:** telemetry is *not* part of the kernel-contract format (it's queried live via
the Tier-1 `BackendRuntime` trait), but the kernel contract's cost/memory fields must be
expressed so telemetry can override them at pick time (§5's three-layer cost model). Keep
static cost and live telemetry as separate, composable surfaces.

### 2c. Tiered trait surface (pluggability target)

05 defines Tier 1 (mandatory: `BackendIdentity`, `BackendCapabilityProvider`,
`BackendRuntime`), Tier 2 (conditional: `BackendStreams`, `BackendPressureSignals`), Tier 3
(diagnostics). A backend is "Fuel-pluggable" at Tier 1 alone. **The kernel-contract format
must be Tier-1-complete on its own** (identity + capabilities + per-kernel static facts) and
treat stream/pressure/diagnostic facts as optional progressive enhancements — a new backend
(ROCm, Metal, TPU) should be able to ship a valid contract with Tier-1 fields only.

---

## 3. Build-time validation + never-panic (hard rules)

From CLAUDE.md, 01-identity, and the `KernelRef` ABI doc-comment:

- **Validate at graph-build time.** Every check that *can* run at build time *must*. No
  `try_*` siblings — the `Result`-returning version is the only version. → A kernel-contract
  format should be **statically verifiable at registration** (a CI lint already asserts the
  always-built backend has a `bit_stable` kernel for every primitive op; another asserts every
  Vulkan registration has a non-UNAUDITED `PrecisionGuarantee` + real `CostFn`). The format
  must support such lints: every required field present, dispatch keys non-overlapping,
  precision/cost not placeholder.
- **Never panic on production paths.** Kernels return `Result`. The interop boundary
  (DLPack extension) must surface malformed/incompatible tensors as typed errors, never panic
  or silently coerce. Interchange import already commits to "unsupported op → `Result` error
  naming the offending op, never a silent drop" (13 §"representation ≠ op") — the same posture
  applies to an unrepresentable tensor at the kernel boundary.
- **No silent fallback / no silent materialization.** The executor never silently
  materializes a layout (only the optimizer inserts `Op::Contiguize`); a chosen kernel failing
  (OOM/fault) surfaces the error rather than transparently switching paths (06 §"What this
  rules out"). → The contract must let a kernel declare *exactly* what it accepts so the
  optimizer inserts fixups explicitly; "I'll silently dequantize/contiguize for you" is
  forbidden (the quantize-as-graph-op prompt is explicit: a consumer that doesn't understand a
  quant format **errors at dispatch, never silently materializes a wide-dtype copy**).

---

## 4. Precision is a structured, per-kernel contract (not a mode)

`PrecisionGuarantee` (05 §"Per-kernel precision guarantees") is the precision half of the
kernel contract and must be a first-class field:

```rust
struct PrecisionGuarantee {
    bit_stable_on_same_hardware: bool, // strictest; deterministic same-hw bits
    max_ulp:      Option<u32>,
    max_relative: Option<f64>,
    max_absolute: Option<f64>,
    notes:        &'static str,
}
```

Constraints the format must honor:

- **Optionality is meaningful.** `None` = "uncharacterized, assume worst case"; tight bounds
  let a kernel participate in tighter routes. There is also an *audited-but-no-static-bound*
  state distinct from *unaudited* (as-built: `PrecisionGuarantee::none(reason)` vs
  `UNAUDITED`). The format must distinguish "no commitment yet" from "committed: no static
  bound exists, here's why."
- **Bit-stability is per-hardware, not cross-backend.** Two `bit_stable` kernels (CPU vs CUDA)
  are *not* byte-equal. Cross-backend comparison must use bounded-error (sum of declared
  bounds, or a coarse epsilon with an explicit "not rigorous" warning). → A DLPack extension
  that carries results across backends must not imply bit-equivalence; correctness tooling
  compares within declared tolerance.
- **Consumers:** the optimizer's **precision-filter pass runs *before* cost ranking** (04) —
  a kernel that fails the user's per-call precision floor or the cumulative tolerance budget is
  not a candidate, period. Cost is a tiebreaker among *admissible* kernels. Calibration tooling
  picks comparators by querying `bit_stable_on_same_hardware: true` + tight `max_ulp` (no
  privileged reference backend anymore — pairwise consensus / fixtures, 05).
- **Empirical refinement trajectory:** static `PrecisionGuarantee` values are starting points;
  the Judge measures actual error per cell and refines `max_ulp`/`max_relative`/`max_absolute`
  over time. **The contract format must allow these fields to be overridden by measured data**
  (don't bake them as immutable literals only).
- **The always-built coverage commitment:** fuel-cpu-backend must ship ≥1
  `bit_stable_on_same_hardware: true` kernel for *every* primitive op (CI-linted). A kernel
  contract for a new primitive triggers a coverage failure until the bit-stable kernel exists.

---

## 5. The cost model is a per-path **vector**, ranked by a bounded Pareto frontier

04 (v0.5) + the 2026-06-14 decisions-log entry are the binding shape here. The optimizer
ranks **paths**, not nodes, on a **cost vector**, and bounds survivors per device:

- **Axes (deliberately low-dimensional, to keep the frontier small + lossless):**
  - **time** — *one central metric* (median/avg for throughput, a tail percentile for
    latency-SLA). `t_min` / "fastest best case" is **explicitly dropped as a selection axis.**
  - **precision** (digits), **accuracy** (ULP / rounding / monotonicity) — discrete levels.
  - **memory** — a **per-tier vector** (disk / host-RAM / device-VRAM tracked *separately*;
    which tier binds depends on the target machine). Not a scalar.
- **Bounding:** survivors are the **per-ending-device Pareto frontier** over that vector, with
  an **NSGA-II crowding-distance cap (`keep` ≈ 32/device)** as the hard backstop. Never a
  single global N (that strands slow devices — the "scalar-top-N failure mode"). ≥1 path per
  device always survives; total ≤ `keep` × devices (prototype: ~10² paths over a deep model).
- **Wall-clock composition, not strict-serial sum:** `wall_clock ≈ max(parallel_branches) +
  serial_remainder`. Plans that expose parallelism rank higher.
- **Three composed layers:** (1) static annotations, *optionally refined by community-aggregated
  empirical priors*; (2) empirical **Judge** data (overrides static when present); (3) live
  telemetry at the route picker (per-tier memory pressure, slot availability). Layers 1+2 are
  baked at optimize time; layer 3 adapts at dispatch.

**Constraints for the kernel-contract format:**

- The per-kernel cost field must yield the *whole vector's per-node contribution* —
  flops/bytes/overhead for time, **per-tier memory footprint** for the memory axis, and tie
  into the precision/accuracy axes via `PrecisionGuarantee`. A scalar cost is insufficient.
- Memory must be reported **per tier** (host vs device, with disk for mmap'd weights), because
  dominance and runtime selection are per-tier. → A DLPack extension describing where a tensor
  *lives* must distinguish disk-mmap / host / device residency, not just a binary host/device
  flag.
- Cost is a *function of shapes/params/capabilities*, and shapes can now be **symbolic** (§6):
  the cost function must accept symbolic extents (cost at capacity, or cost as a function of
  the resolved live extent) without forcing the graph to re-plan per token.

---

## 6. Symbolic extents (Phase D): the live-vs-capacity split

This is the active frontier and the strongest new constraint on *tensor description*. From
`symbolic-extents-and-persistent-decode.md` (§0 wins on conflicts) + as-built
`fuel-core-types/src/{symbol.rs, shape.rs, layout.rs}`. **Steps 1a–1d have landed.**

The primitives:

- **`SymId(u32)`** — interned, **stable, serializable, session-independent** identity of a
  runtime value. Equal ids = the same value (unification by id equality). *Not* pointers:
  pointers can't serialize into the base map and would clobber across concurrent sessions.
- **`SymEnv: SymId → usize`** — a **per-forward-pass input**, sibling of the tensor-data
  `StorageCache`. **Write-once per pass** (rebinding to a different value is an error). The
  graph carries `SymId`s; the env is supplied per realize.
- **`Extent { Scalar(usize) | Range { min, max, sym: SymId } }`** — for *dimensions*. A
  runtime dim is always a `Range` (it always has a capacity); a `Scalar` carries no symbol.
- **`DynScalar { Concrete(usize) | Sym(SymId) }`** — for *scalar op params that aren't
  dimensions* (KV-write offset, RoPE position, flash `k_len`).

The **critical** tensor-description rule — **`dims()` is the capacity bound, not the live
value**:

- Annotated `Shape { dims: SmallVec<usize>, dynamic: Option<SmallVec<DynAxis>> }`. **`dims() ->
  &[usize]` is unchanged and returns the BOUNDS** (a `Scalar`'s value, or a `Range`'s
  `max`/capacity) — correct for the ~2,000 sizing/striding/iteration sites (allocate capacity,
  walk the buffer). It is *not* a lie; the live value is a *different fact*.
- **The live value comes from `extent(i) -> Extent` + `resolve(&env) -> Shape`** (reads
  `env.get(sym)` on demand, never cached).
- `Hash`/`Eq`/`Debug` **include** `dynamic` — a symbolic shape is a *distinct* shape that
  *plans once*; concrete shapes hash exactly as before.
- **`Layout` embeds `Shape`, so it inherits `Extent` for free.** For a symbolic axis the
  **stride stays concrete** (physical step in the fixed-capacity buffer, a build-time
  constant) while the **extent is symbolic** (the live count). These are the two halves a
  kernel needs to walk the *live prefix of a capacity buffer*: stride = how far per element,
  extent = how many live elements.

**Constraints for both formats — this is where generic DLPack falls short:**

- A tensor description must carry **both the capacity (for allocation/stride) and a live
  extent (a symbol resolved per call)** for any dynamic axis. DLPack's plain `shape: int64*`
  describes only one number per axis; the extension must add a per-axis "this is a bounded
  symbol; here is its `SymId` and its `[min, max]` capacity" annotation, **with strides keyed
  to the capacity** so the buffer is walked correctly at any live extent.
- The interchange must transport **`SymId`s** (serializable, unifiable), not resolved values,
  so one description plans once and serves every token/session. Resolution happens at the
  realize boundary via a `SymEnv` supplied alongside the data.
- **Unification matters:** K-length ≡ V-length ⇒ same `SymId`; two distinct `Range`s with
  different syms are *not* interchangeable even at equal bounds. The format must preserve sym
  identity across the boundary so the optimizer's unification survives.
- **Scalar dynamic params** (`DynScalar`) ride the same `SymEnv`. A kernel contract that
  consumes a length/offset/position must say whether it takes a *resolved scalar* (`flash`'s
  `k_len`) vs a *mask tensor* — the architecture treats "length vs mask" as a **lowering
  decision**, not a tensor property (masks are an *op's* job, not `Extent`'s).
- **Two scopes to anticipate** (design the API general, don't build consumers yet):
  *input-determined* syms bound up-front (all of decode: `cached_len`) and *data-determined*
  syms filled mid-pass by a producing op (NonZeroIndices/MoE counts), the latter needing a
  build-time producer→consumer dependency edge. Also session-scoped syms (batch size, bound
  once per session) and affine sym expressions (`k_len = cached_len + seq`). The format should
  not preclude these.

The payoff this enables (and which the formats must not block): the **persistent decode
graph** — build the decode-step graph once, re-bind data + `SymEnv` per token, re-realize the
*same* graph, skip `optimize_graph`. The graph being input-independent is what makes the plan
reusable; the ~1.8×/token win comes from the symbolic foundation, not from fusion.

---

## 7. Persistence / interop boundary constraints

The plan **is** the graph (03/04/06/11). The native `.fuel` serializes the whole graph
(base map + storage + optimized paths), mmap-backed. Kernel selections are persisted as
**`(backend_id, op_kind, dtypes, kernel_revision_hash)` tuples — never `KernelRef` function
pointers** (process-local), re-resolved on load via the binding-table catalog.

Constraints:

- **The kernel-contract format must round-trip through serialization**: every field the cache
  stores about a kernel choice (op_kind, per-operand dtypes, backend, **kernel_revision_hash**)
  must be serializable and re-resolvable. Function pointers and process-absolute pointers are
  forbidden in the persisted form (mmap-friendly: relative offsets only). This is exactly why
  `SymId` (not a pointer) is the symbolic primitive.
- **Cache invalidation keys** (11): `(arch_version, kernel_hashes, hw_fingerprint,
  judge_version, tolerance_set, model_hash)`. The kernel contract feeds `kernel_hashes` and the
  hw fingerprint; **a revision-hash mismatch invalidates only the affected decision point**
  (scoped re-optimization), so the contract's hash must be stable and granular.
- **Multi-version format support** (decision #18): newer Fuel reads the previous N DAG-format
  versions; additions should be backward-compatible (newer fields ignored by older readers)
  where feasible. → **Design both formats to be additively extensible** (new dtype/quant/
  capability/precision fields don't break old readers); reserve a version field.
- **Interchange hub is the base map** (13): Fuel's ~80–90 primitive `Op` vocabulary *is* the
  interchange vocabulary (no second neutral IR). A DLPack extension is a *tensor*-interchange
  surface (storage/dtype/shape/layout/quant), complementary to graph interchange — keep them
  separate concerns. Weight payload ⊥ graph payload (the two-axis decomposition); a tensor
  description belongs to the weight/storage axis. The node↔weight *binding* is format-local
  and must stay out of any shared core.

---

## 8. The dtype set, incl. sub-byte placeholders

As-built `DType` (`fuel-core-types/src/dtype.rs`) — the full set both formats must cover, with
safetensors interop already wired:

`U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0`.

Sharpest constraints:

- **Sub-byte types have `size_in_bytes() == 0`** (F6E2M3, F6E3M2, F4) by current convention —
  a *flag* that byte-count sizing does not apply, not a real size. **A DLPack extension MUST
  carry bit-width / packing, not byte size**, for these. Generic DLPack uses `bits` +
  `lanes` in its dtype struct; the extension should map Fuel sub-byte dtypes onto a real
  bit-width (4 / 6 / 8) + a packing convention (MX4/MX6 block formats), since `size_in_bytes()`
  returns 0 and would mis-size a buffer.
- **F8E4M3 is fully wired** (1 byte, has a Rust `float8` backing type, `FloatDType` impl,
  `WithDType`); **F6E2M3 / F6E3M2 / F4 / F8E8M0 are placeholders** (`F8E8M0` reports 1 byte but
  no `WithDType`; the 6/4-bit ones report 0 and have no host type). These are MX (microscaling)
  formats — they pair with a shared **block scale** (F8E8M0 is the canonical MX scale dtype:
  8 exponent bits, 0 mantissa). **The formats should anticipate MX block-scaled layouts** (a
  packed sub-byte payload + an F8E8M0 scale per block) as a first-class quant case, distinct
  from the per-tensor/token/channel scales of §9.
- `is_int` / `is_float` are defined for all; F8E4M3 is in `FloatDType`. The format's dtype
  enum should be `#[non_exhaustive]`-spirited (new dtypes arrive: I8 was added 2026-05-19 for
  int8 GEMM, the MX types are recent).

---

## 9. The quantization stack: two parallel systems

There are **two distinct quant systems**, and the formats must not conflate them:

### 9a. GGML block-quant (static, load-time) — `GgmlDType`

`fuel-core-types/src/quantized.rs`: `GgmlDType { F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q8_1, Q2K..Q8K }` with `type_size()` (per-block bytes) + `block_size()` (32 for legacy, 256 for
K-quants). Scales are **baked into the binary block format** (per-block along K) — **no free
granularity parameter.** Loaded pre-quantized; quantization is the *loader's* job. Dispatch
goes through the `DynQuantizedStorage` trait (`dtype/block_size/storage_size_in_bytes/quantize/
dequantize/fwd/indexed_moe_forward/...`) and `QuantizedDeviceKernels` (`qzeros`,
`load_quantized`). `Capability` enum has *flat per-format tokens* (`MatMulQ4_0`, `MatMulQ4KM`,
`MatMulQ8_0`, `QuantizeQ8_0`, `Dequantize…`) — intentionally not parameterized, because the
kernels are specialized per format.

**Constraint:** a quantized-tensor description must carry **block format + block size + packed
byte layout**, and the kernel contract for a QMatMul must key on the specific format (a flat
capability token), not a generic "quantized" flag. The kernel **owns the dequant**; there is no
generic dequant-on-read fallback.

### 9b. Dynamic quant (runtime, graph-op) — `ScaleGranularity` / `ScalePair`

`fuel-core-types/src/quant_scale.rs` + the `quantize-as-graph-op` prompt. For FP8/Int8/Int4
*dynamic* quant the caller chooses scale granularity:
`ScaleGranularity { PerTensor | PerToken | PerChannel }` (scale shapes `f32[1]` / `f32[rows]` /
`f32[cols]`), paired as `ScalePair { activation, weight }` (presets: `PRODUCTION_DECODE` =
per-token act × per-channel weight; `SIMPLE` = per-tensor × per-tensor). Intended end state:
`Op::Quantize { to: DType, granularity }` / `Op::Dequantize { to }` as **first-class graph
ops** the optimizer schedules. Storage is a **(data + sidecar scales) pair** (Option B, the
chosen design) carrying `ScaleGranularity` metadata. Dispatch keys on
`(op, lhs_dtype, lhs_granularity, rhs_dtype, rhs_granularity)` — i.e. **`ScalePair` is part of
the dispatch key.**

**Constraints for both formats:**

- A DLPack extension must be able to describe a **quantized tensor as data + scale sidecar(s)
  + a granularity/format tag** — i.e. *multiple correlated buffers* (packed data, scales, and
  for MX an F8E8M0 block-scale) under one logical tensor. Plain DLPack is single-buffer; the
  extension needs a way to bind the scale buffer(s) to the data buffer.
- The kernel-contract dispatch key for a quant op must include **granularity / scale-pair**,
  not just dtype. The contract format's key must be richer than `(OpKind, [DType])`.
- **Do not unify 9a and 9b.** Static GGUF/AWQ/Marlin/NF4 have no `ScaleGranularity` (it's
  baked); dynamic FP8/Int8 do. The tensor description must say which regime a tensor is in.
- Quant errors at dispatch (no silent dequant fallback) — §3.

---

## 10. Multi-output / bundled tensors

12-multi-output (Option C, shipped): a multi-output node produces **one bundled `Storage`** (one
allocation, one Arc) partitioned by a side-table of `OutputView { byte_offset, len_elements,
dtype, shape, layout, name }`. Slots are **independent in dtype, shape, and layout** (e.g. F32
`y` + I64 `argmax_idx` in one bundle). `Op::View` (zero-copy projection) and `Op::ViewOwned`
(independent copy) resolve slots back to ordinary tensors. **Multi-output kernels still emit ONE
`KernelRef` with `outputs.len() == 1`** (the bundle); the kernel writes per-slot bytes by
offset. Backends stay single-output at the trait level.

**Constraints:**

- A tensor-description format must be able to express a **bundle: one buffer, N sub-views at
  byte offsets, each with its own dtype/shape/layout.** DLPack is one-tensor-per-capsule; the
  extension must either describe sub-views into one allocation or carry a bundle wrapper. (This
  pairs with §9b's data+scales, which is itself a small bundle.)
- The kernel contract's "outputs" arity is **1 at the ABI level even for multi-output ops** —
  the contract describes the bundle's slot specs (`output_views: fn(&[Shape], &[DType],
  &params) -> Vec<OutputViewSpec>`), not N separate outputs. The format must carry this
  authoring contract.

---

## 11. Planned dispatch / ranker evolution (what the formats must not foreclose)

- **Two pickers:** "Picker 1" = plan-time ranker (`compile_plan`, baked kernel-variant choice);
  "Picker 2" = runtime route picker (chooses device/path at `Op::Branch` points from live
  per-tier memory). Kernel-variant choice is *largely baked at optimize time*; device/path
  choice adapts at runtime. → The kernel contract feeds Picker 1; the *path* structure +
  telemetry feed Picker 2. Keep static (contract) and dynamic (telemetry) cleanly separable.
- **Dispatch unit is a "run"** (the fixed op-sequence between two decision points), ideally a
  **pre-recorded CUDA Graph / Vulkan command buffer** replayed with rebased operands (capability
  built; wiring is Phase D). → A DLPack extension must support **operand rebasing** — describe a
  tensor such that the same recorded run can be replayed against new buffers (stable
  shapes/strides at capacity, only the base pointer + `SymEnv` change per replay). This is the
  same input-independence requirement as §6.
- **Three forms of parallelism** (pipeline / data / within-kernel). Within-kernel concurrency is
  the *backend's* business (the principled exception to "backends don't decide"); the contract
  must not try to describe it. Inter-run parallelism is the runtime's, sized by advertised slot
  capacity.
- **Concurrent optimize-and-execute** + **load-time + background re-optimization**: rules
  self-declare `Concurrent | WholeGraph` frontier-compatibility; kernel choices commit at the
  frontier and may be re-ranked in the background per decision point (atomic Arc swap). →
  Kernel-contract fields (cost/precision) must be **re-readable/overridable post-load** so
  background re-optimization can re-rank against fresh Judge data without rebuilding.
- **Future KernelCaps** (strided_input today; layout/quant/symbolic flags coming) and **future
  capability tokens** (the `Capability` enum is `#[non_exhaustive]`). The contract's capability
  set must be open.

---

## 12. Decisions-log items bearing directly on kernel boundaries / tensor description / interop

- **#4 (op-shape A):** closed `Op` enum + `Op::Fused(id, params)` arm; **#13:** reference backend
  dissolved into per-kernel `PrecisionGuarantee` + always-built bit-stable coverage. → Contract
  carries precision per kernel; no privileged oracle.
- **#5 / #12 / #23:** `KernelRef` is a planning-time *catalog* lookup; resolution is optional
  pre-resolve (throughput) vs lazy (TTFT); mmap'd cache + lazy resolution. → Contract entries are
  catalog rows keyed by `(backend, op, dtypes, source)` → `KernelRef`, re-resolved on load.
- **#7 superseded + 2026-06-14 redirection:** *the plan IS the graph*; per-node `AlternativeSet`
  is demoted; alternatives attach to `Op::Branch` branch points bounded by a **per-device Pareto
  frontier + crowding cap** over a **cost vector** (single time metric, per-tier memory, discrete
  precision/accuracy); `t_min` dropped. → Cost is a vector; memory is per-tier; design cost
  reporting accordingly (§5).
- **#6 + #10 (2026-06-14):** three storage classes (shared/session/transient) inferred from op +
  override, **session-indexed** for session-state; transient crosses devices (D2D) but never to
  disk. → A tensor description must carry/derive its **storage class** and (for session-state) a
  `SessionId`; the persistence/interop boundary treats the three classes differently.
- **#11 (2026-06-14):** **three-tier memory (disk/host/device) tracked as a vector**; the plan is
  the cross-tier prefetch schedule. **#11-storage:** `Storage` must support **mmap-backed
  zero-copy views**, not only owned buffers. → DLPack extension must describe disk-mmap residency
  + zero-copy views, not just host/device.
- **2026-06-08 (interchange):** weight ⊥ graph axes; base map is the hub; `Result`-error on
  unrepresentable, never silent drop; the node↔weight binding is format-local. → Tensor
  interchange (DLPack ext) is the weight/storage axis, kept separate from graph interchange.
- **2026-06-08 (L3 snapshot):** designated durable state (KV-caches via `Op::WriteSlice`,
  optimizer state) is snapshot-able; **not** all activations. → Session-class tensors are the
  durable-interop unit; transient activations are not an interop concern.
- **2026-06-13 (bundled Judge baseline):** empirical priors ship in-package; cost annotations are
  seeds refined by measurement — reinforces "contract cost/precision fields are overridable."

---

## 13. Summary — the constraints that most shape the two specs

**Kernel-contract format (highest-leverage constraints):**

1. **Pure advertisement, optimizer-consumed; zero hidden behavior** (no internal placement/
   fusion/variant-swap/cache). Designed to *maximize* what the optimizer sees (the 01
   enforcement gate).
2. **Per-kernel static fields, all first-class and serializable:** dispatch key
   `(OpKind, [DType…] + quant granularity, BackendId, kernel_source)`; `KernelRef` ABI
   (executor pre-allocates outputs; outputs arity 1 even for bundles; **never panic, return
   `Result`**); **`CostEstimate` → a per-tier cost-vector contribution** (flops/bytes/overhead +
   per-tier memory); **`PrecisionGuarantee`** (structured, optional bounds, audited-vs-unaudited
   distinction, empirically overridable); **open `KernelCaps`** (strided_input + future flags);
   alignment + access granularity; transfer paths + `SubstrateClass`; slot capacity;
   **`kernel_revision_hash`** (cache invalidation).
3. **Precision filter runs before cost** — admissibility is precision/tolerance first, cost as
   tiebreaker. The format must make precision a hard pre-filter input.
4. **Tier-1-complete standalone**, additively extensible (`#[non_exhaustive]`), statically
   verifiable at registration (CI-lintable). Re-readable/overridable post-load for background
   re-optimization.

**DLPack extension (highest-leverage constraints):**

1. **Symbolic extents:** per-axis carry **capacity bound (for stride/alloc) + a live symbol
   (`SymId` resolved per call via `SymEnv`) with `[min,max]`**, strides keyed to capacity.
   Transport `SymId`s (serializable, unifiable), not resolved values — so one description plans
   once and serves every token/session/replay. This is the single biggest gap vs generic DLPack.
2. **Sub-byte + MX dtypes:** carry **bit-width + packing**, never `size_in_bytes` (which is 0 for
   F4/F6E2M3/F6E3M2); anticipate MX block-scaled layouts (packed sub-byte payload + F8E8M0
   per-block scale).
3. **Quantized tensors are multi-buffer:** data + scale sidecar(s) + granularity/format tag;
   distinguish static GGML block-quant (baked scales, no granularity) from dynamic FP8/Int8
   (ScaleGranularity/ScalePair, part of the dispatch key). No silent dequant.
4. **Bundles / multi-output:** one allocation, N sub-views at byte offsets, each with own
   dtype/shape/layout.
5. **Residency + substrate finer than host/device:** disk-mmap / host / device, plus
   `SubstrateClass` + device identity precise enough to decide same-buffer-vtable-swap vs
   needs-a-copy (Vulkan vs CUDA on the same silicon do *not* alias); honor alignment +
   access-granularity; support operand rebasing for recorded-run replay.
6. **Storage class + SessionId:** carry/derive shared/session/transient; session-state is the
   durable-interop unit (KV-caches), transient never crosses to disk.
7. **Errors, never panics, never silent coercion** at the boundary; additively versioned for
   the multi-version DAG-format read policy.
