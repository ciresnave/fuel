# Fuel — Roadmap

This document describes the current state of this project, the structural and ergonomic
problems it aims to solve, and the planned order of work.

---

## Authoritative architecture: see `docs/architecture/`

The architecture documents in [`docs/architecture/`](docs/architecture/00-index.md)
are the constitutional description of fuel — what fuel is, how it's structured, what
makes it competitive, and the boundaries it commits to. **When this ROADMAP and the
architecture set conflict, the architecture set wins**; this ROADMAP is updated to
match. The architecture set was established at v1.0 on 2026-05-09 and captures 24
foundational architectural decisions in
[`docs/architecture/10-decisions-log.md`](docs/architecture/10-decisions-log.md).

This ROADMAP describes *the path* — phases, work items, sequencing, current state.
It anchors to architecture sections by cross-reference rather than by restating them.
Most phase entries below were drafted before the architecture set existed; where they
diverge from the architecture, the architecture is the source of truth and the phase
entry will be updated next time it's actively worked on.

---

## Current frontier (2026-05-22)

**Active phase**: Phase 7.6 step 9c — typed-storage retirement + Vulkan runtime
catch-up. Step 9a + 9b Track A shipped; the binding-table dispatch path is
load-bearing for CPU + CUDA + Vulkan.

**Active sub-thread**: Bridge cleanup post-typed-storage. `VulkanBackendDevice`
landed this session, closing the runtime `Device` gate for Vulkan. The
[bridge-retirement plan](#bridge-retirement-trajectory-post-9c) under
Phase 7.6 step 9c describes the path from the bridge to the architecture
v1.0 destination (graph-level `Op::Copy`/`Op::Alloc`, dispatch-erased
`Device` tag, retired `DynBackendStorage` trait).

**Recently shipped (last 30 days, summarized; see memory + commit log for full record)**:

- Phase 7.5 work item G + G2: graph owns Storage; `Op::Const` is a unit variant.
- Phase 7.5 work item B2: fuel-core eager `Tensor` factories produce node-handle tensors.
- Phase 7.6 steps 1-3 + step 6 + FusedLinear: registry skeleton + `Op::Fused` arm + first migrated op + `register_fused!` macro.
- Phase 7.6 step 9a + 9b Track A: KernelBindingTable multi-impl alternatives + ExecutionPlan + NodeKernelBinding.
- Phase 7b: AOCL + oneMKL CPU backends + Router empirical dispatch (shipped 2026-04-28 → 2026-05-15).
- CUDA Tier 1 migration to binding table (45 live tests; baracuda alpha.27).
- Vulkan V.2 + V.3 byte-storage fan-out (87 live RTX 4070 tests; QMatMul + Conv2D + f64 transcendentals + WriteSlice b1/b2/b4/b8).
- Vulkan runtime `Device` wiring (2026-05-22): `VulkanBackendDevice` + parity test against CPU through `forward_with_kv_context`.
- Vulkan binding-table key-shape audit (2026-05-22): every Vulkan registration cross-checked against its CPU peer; zero further mismatches beyond the Rope fix in `7a95001a`. Full coverage table in [`project_phase_7_6_step_9c_parity_audit.md`](.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md).
- **Bridge-retirement Phase 2 — `Op::Copy` D2H through the binding table** (2026-05-22): `OpKind::Copy` registered; CPU/CUDA/Vulkan each provide a `copy_to_cpu` wrapper at `(OpKind::Copy, [dt, dt], source_backend)`. `realize_one_as<T>` / `realize_many_as<T>` splice `Op::Copy { target: Cpu }` at every realize root; executor's `WorkItemKind::Copy { target_location }` arm allocates output on target while keying kernel lookup on source backend. First deletion of bridge code from `7a95001a`: `BackendStorage::read_to_cpu_bytes` is gone. Also fixed a pre-existing CPU bug where realize on a view (slice/permute) returned the parent's full bytes instead of the logical view's bytes — Op::Copy's `auto_contiguize` materializes view layouts uniformly. See [`project_phase_7_6_step_9c_parity_audit.md`](.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md) "Bridge-retirement Phase 2 shipped" follow-up.
- **Bridge-retirement Phase 3a — `Op::Alloc` (uninit) + `Op::ZeroFill` (explicit fill)** (2026-05-22 + 2026-05-23): two new graph IR primitives — `Op::Alloc { target: DeviceLocation }` produces uninit device memory; `Op::ZeroFill` does a destructive in-place zero-fill. Executor's `WorkItemKind::Alloc` arm uses true uninit allocators (CUDA: baracuda alpha.30's raw `cuMemAlloc`; Vulkan: `alloc_bytes_handle`); `WorkItemKind::ZeroFill` arm calls device-side fills (CUDA: `DeviceBuffer::zero_async` via `cuMemsetD8Async`; Vulkan: `VulkanBackend::fill_bytes_zero` via `vkCmdFillBuffer`, ~2× the bandwidth saved over the old host-staged zeros). `KvCache::with_capacity` emits 2N pairs of `Op::Alloc → Op::ZeroFill`. **Deleted**: `alloc_zeroed_on` (50-line per-`DeviceLocation` match in `fuel-core/src/inference_context.rs`) — second deletion of bridge code from `7a95001a`. **Residual**: `pipelined_bridge::device_seed_storage` (~15 lines), the small 0-byte-seed allocator that puts a device handle into the cache. **baracuda bumped to alpha.30** workspace-wide for `DeviceBuffer::zero_async`.
- **Bridge-retirement Phase 3b — H2D Const upload through `Op::Copy { target: device }`** (2026-05-23): Extended `copy_from_cpu_wrapper` (renamed from `copy_to_cpu_cpu_wrapper`) to switch on output variant — CPU→CPU memcpy, CPU→CUDA via new `CudaStorageBytes::write_from_host`, CPU→Vulkan via new `VulkanBackend::write_bytes` (staging buffer + `vkCmdCopyBuffer`). Executor's `WorkItemKind::Copy` arm extended to allocate non-CPU output (uninit). `pipelined_bridge::build_const_cache` for non-CPU realizes now builds a transient graph with `Op::Const → Op::Copy { target: device }` pairs (one per user Const) plus a device-handle anchor, realizes via `PipelinedExecutor::realize_many`. **Deleted**: `upload_host_buffer` (60-line per-`DeviceLocation` match in `fuel-core/src/pipelined_bridge.rs`) — third deletion of bridge code from `7a95001a`. All three branches from `7a95001a` are now retired.
- **Vulkan PrecisionGuarantee + cost annotations (session 1 of 2)** (2026-05-23): Per-kernel `PrecisionGuarantee` claims for every Vulkan registration (~120 entries). 8 family constants in `fuel-storage::fused` cover the precision space: `VULKAN_FLOAT_POINTWISE_PRECISION`, `VULKAN_HALF_POINTWISE_PRECISION`, `VULKAN_TRANSCENDENTAL_PRECISION`, `VULKAN_MATMUL_PRECISION`, `VULKAN_MATMUL_TENSORCORE_PRECISION`, `VULKAN_BYTE_LEVEL_PRECISION` (ULP=0, byte-identical), `VULKAN_CAST_PRECISION`, `VULKAN_QMATMUL_PRECISION`. Vulkan reductions (Sum/Max/Min/Mean/Softmax/RmsNorm) use `PrecisionGuarantee::UNAUDITED` (renamed from `UNKNOWN`) because the audit conclusion IS "no static bound" — these are listed in the lint's `KNOWN_GAPS` allowlist with documented reasons. Cost functions bulk-filled via new `fill_unset_cost_for_backend(Vulkan, default_cost_for_op_kind)`. `OpKind::WriteSlice` added to `default_cost_for_op_kind`. New `vulkan_dispatch_per_kernel_precision_and_cost_coverage` lint asserts every Vulkan registration has a non-UNAUDITED PrecisionGuarantee + non-`unknown_cost` CostFn (with structural detection + KNOWN_GAPS allowlist).
- **PrecisionGuarantee API cleanup** (2026-05-23, mid-session refactor through several iterations): renamed `PrecisionGuarantee::UNKNOWN` → `UNAUDITED` to make the placeholder nature explicit. Deleted `VULKAN_REDUCTION_PRECISION`. Added `PrecisionGuarantee::none(reason)` const fn for the "audited, no static guarantee could be established" case — the reason lives on the value via `notes`, captured at the registration site. Vulkan reductions (Sum/Max/Min/Mean/Softmax/RmsNorm) use `PrecisionGuarantee::none("reason text")`. Lint detector is notes-equality against `PrecisionGuarantee::UNAUDITED.notes` — robust to UNAUDITED's notes drifting because the lint reads from the const reference, not a hardcoded string. **No `audited: bool` field, no `KNOWN_GAPS` allowlist** — the audit state is encoded by which constructor was used (UNAUDITED literal vs `none(reason)`), and the audit reasoning lives in the value's notes at the registration site.
- **Phase β4 — `fuel-nn` crate retirement** (2026-06-06): all in-tree consumers migrated to `fuel_core::lazy_nn` (LazyTensor-native layers, losses, optimizers, init, VarMap, etc.). Workspace `Cargo.toml` no longer lists `fuel-nn` under `members`, `default-members`, or `[workspace.dependencies]`. Directory renamed `fuel-nn/` → `_fuel_nn_retired/` so the underscore prefix hides it from Cargo discovery while preserving the source on disk for diff/reference. **Not deleted** — future archaeology session can `git diff` against the retired tree before its final `git rm`. NN layer in the layer model is now provided entirely by the lazy stack inside `fuel-core`.
- **Vulkan KernelCaps (slot annotations) — session 2 of 2** (2026-05-24): per-kernel audit of every Vulkan registration's stride handling. **Stride-aware (opted into `KernelCaps::strided_input()`)**: binary f32/f16/bf16/f64 (24 regs), Concat f32, Rope f32 = 26 registrations. The executor's auto-Contiguize gate now passes broadcast / transpose / slice views directly to these kernels — no host-side materialization. **`// strided-input candidate:` markers** added near ~66 contiguous-only registrations that could be converted in a future perf sweep (unary across 4 dtypes, Affine/Clamp/PowI, casts, IndexSelect). **Contiguous-only by design** (no candidate marker): reductions / matmul / Conv2D / WriteSlice / Triu/Tril/Flip/Roll / Copy — family-level audit comments explain why each can't be extended. All 148 Vulkan registrations now have a per-kernel audit conclusion at the call site. RTX 4070 live sweep (89 ignored tests) green.

**Next 1-3 sessions** (in priority order):

1. **Vulkan strided-input candidate sweep** — convert the ~66 contiguous-only kernels currently marked `// strided-input candidate:` in `register_vulkan_kernels` (unary across 4 dtypes, Affine/Clamp/PowI, casts, IndexSelect) to walk per-input strides. Each conversion mirrors `binary.slang`'s pattern: extend the Slang kernel's Params with per-dim strides + a contiguous fast-path flag, extend the backend method to accept a `Layout` and pack its strides. Eliminates the auto-Contiguize materialization for broadcast / transpose / slice inputs feeding these kernels. ~2-3 sessions.
2. **Phase 7.6 step 9c Phase E.3 remainder** — `forward_with_cache_on` migration + `generate_*` + spec decoding migration to `PipelinedExecutor`. Each is its own session; the `KvCache` primitive from E.3.0 unblocks them.
3. **Bridge-retirement step 3** (per `docs/architecture` post-9c plan) — remove `*_dyn` storage methods from `DynBackendDevice` trait once nothing calls them from byte-storage callers. Trait shrinks to advertisement-only.
4. **Vulkan-flavored cost dispatcher** — replace `default_cost_for_op_kind` for Vulkan with a dispatcher that adjusts `kernel_overhead_ns` to reflect command-buffer submission cost (~5,000 ns vs CPU's 50 ns). Optional refinement; FLOP/bandwidth terms are already correct.

**Blockers**: none. Multi-GPU work (Phase 6c D2D + Phase 6d MoE placement) is parked pending multi-GPU hardware; not on the critical path.

**Architecture-alignment check**: every active workstream above moves at least
one of the four [identity-enforcement checks](docs/architecture/01-identity.md#how-this-identity-is-enforced)
*more* true. Bridge retirement moves checks 1 + 2; PrecisionGuarantee work
moves checks 2 + 4; E.3 caller migration moves check 1.

---

## Identity

Fuel is a **layered Rust ML framework**.

It aims to feel small at the bottom and powerful toward the top, without forcing
any particular use case on the layers below it. Someone doing tensor math should
not carry inference orchestration. Someone implementing a model architecture should
not need a runtime. Someone building a complete inference pipeline should have the
building blocks readily available.

The ecosystem should be easy to exit early. You should be able to stop at exactly
the layer you need.

---

## Layer Model

The ecosystem is organized into six conceptual layers. Dependencies within the
stack flow downward only. No lower layer may depend on a higher one.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│  Use-Case Orchestration                                                    │
│  fuel-inference, fuel-training  (leaf crates — nothing depends on     │
│  either of these)                                                          │
│                                                                            │
│  fuel-inference: sampling, logits processing, KV-cache policy,          │
│  token generation loops, speculative decoding, batching, streaming         │
│  decode, cancellation, inference session abstractions                      │
│                                                                            │
│  fuel-training: training loops, gradient accumulation, LR scheduling,   │
│  gradient clipping, mixed precision policy, run-time checkpointing,        │
│  training session abstractions                                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Models                                                                    │
│  fuel-transformers  (will be restructured internally)                   │
│  Architecture config structs, layer composition, forward passes,          │
│  weight name mapping. No serving logic, no decode loops, no sessions.     │
├────────────────────────────────────────────────────────────────────────────┤
│  IO                                                                        │
│  fuel-core (safetensors, npy, pickle), fuel-onnx                      │
│  Bidirectional data exchange across any boundary: files, network,         │
│  memory buffers. Checkpoint load and save, format translation, ONNX       │
│  import/export, HF Hub integration glue, config normalization,            │
│  tokenizer glue. To be consolidated.                                       │
├────────────────────────────────────────────────────────────────────────────┤
│  NN                                                                        │
│  fuel-nn                                                                 │
│  Layers, losses, optimizers, parameter utilities, initialization,         │
│  VarBuilder, VarMap. No model-family assumptions. No serving abstractions.│
├────────────────────────────────────────────────────────────────────────────┤
│  Foundation                                                                │
│  fuel-core                                                               │
│  Tensors, devices, dtypes, shapes, layouts, base ops, autograd,           │
│  storage backends, error types. No tokenization. No model-level concepts. │
├────────────────────────────────────────────────────────────────────────────┤
│  Backends / Kernels                                                        │
│  fuel-cuda-kernels, fuel-metal-kernels, fuel-flash-attn, fuel-ug  │
│  Hardware and runtime targets (CPU, CUDA, Metal) plus the concrete        │
│  mathematical kernel implementations for each: matrix multiply, conv,     │
│  flash attention, quantized dot products, SIMD/BLAS dispatch. This layer  │
│  knows tensors as shaped memory regions and operations as mathematical     │
│  functions over those regions. It has no concept of layers, models,       │
│  losses, tokens, or any other ML abstraction.                              │
│                                                                            │
│  Foundation: `BackendDevice` and `BackendStorage` traits already exist    │
│  in fuel-core. CUDA and Metal are behind feature flags. Phase 5         │
│  formalizes these as a published plugin contract and opens the type for   │
│  third-party backends.                                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Current State

### What is working well

- Dependency direction between published crates is already mostly correct.
  `fuel-core` does not depend on `fuel-nn`, which does not depend on
  `fuel-transformers`. The early-exit property is structurally present.
- `fuel-core` has a reasonable backend abstraction (CPU, CUDA, Metal).
- Quantization has a meaningful home in `fuel-core::quantized`, better
  centralized than most frameworks at a comparable stage.
- The breadth of model implementations in `fuel-transformers` is genuinely
  impressive and is a key asset.

### Identified problems

**Documentation**
The primary way users currently learn non-trivial usage patterns is by reading
examples. Examples are useful but they are poor architecture documentation. Most
public API items across all crates lack doc comments or runnable examples in the
documentation itself.

**Ergonomics / developer experience**
Using Fuel non-trivially requires understanding `Var`, `VarBuilder`, `VarMap`,
device management, and dtype handling simultaneously before anything works. There
is no convenience path for common cases. Error messages often carry the right
information but do not always present it in a form that immediately tells you what
went wrong and how to fix it.

**Inference and training concerns are scattered**
`fuel-nn` currently contains `kv_cache.rs` and `sampling.rs`.
`fuel-transformers` contains `generation/` with `LogitsProcessor` and the
`Sampling` enum, as well as a `pipelines/` directory intended for orchestration
logic. These are inference-specific tools with no natural home below the
orchestration layer. The consequence is that `fuel-nn` carries inference
weight that pure layer-building users never need.

**`fuel-transformers` is a flat namespace with no internal structure**
Over 100 model files coexist in a single `models/` directory alongside their
quantized variants, shared utilities, object detection helpers, and generation
logic. There is no enforced separation between architecture definitions and
runtime glue. This will worsen as more models are added.

**No top-level guide**
There is no document that routes a new user to the right crate based on their
intent. New users are expected to infer the architecture from the repository
layout and the README example list.

---

## Planned Work

Work is organized into ten phases. Later phases depend on earlier ones being
stable but phases within a group can proceed in parallel. Phase 9 is
extension hooks for downstream consumers (specifically: an out-of-tree
agentic library); not gated on the others, just gated on a real consumer
asking for them. Phase 10 is equivalence-rewrite search; gated on the
eager-retirement program finishing and the picker accumulating real
Judge telemetry.

---

### Phase 0 — Ecosystem compatibility ✅ shipped

*Prerequisite for everything else: ensure the fork builds cleanly with all
features enabled, on all supported platforms, without manual patching.*

**What it accomplished**: audited the entire fork ecosystem (fuel-optimisers,
fuel-layer-norm, fuel-bhop, fuel-einops, fuel-birnn, fuel-lstm, fuel-crf,
fuel-approx) for version compatibility against current workspace `fuel-core`;
fixed all build failures including the Windows + CUDA 13.0 / MSVC path
(`gen` reserved-keyword fix in `fuel-core/src/cuda_backend/device.rs`);
brought `fuel-cublaslt` and `fuel-cuda-vmm` into the workspace; extracted
a backend-agnostic `fuel-vmm` crate (8-method `VmmBackend` trait + generic
`VirtualMemoryPool<B>` / `SharedMemoryPool<B>`) from the CUDA-specific
implementation, unblocking future ROCm and CPU-mmap implementations
of the same elastic-KV-cache pattern at zero runtime cost
(monomorphized per backend). Established workspace-level dependency
version matrix (`[workspace.dependencies]` in `Cargo.toml`); added CI
coverage (`.github/workflows/rust-ci.yml` for CPU on Linux/Windows/macOS/AVX2/ARM;
`.github/workflows/ci_cuda.yaml` for CUDA 13.0 on a GPU runner).
Compatibility guide at workspace root: [`COMPATIBILITY.md`](COMPATIBILITY.md).

**Why it mattered**: the Candle ecosystem fragmented across ~12 crates that
required separately maintained forks. Phase 0 collapsed that fragmentation
into a single coherent workspace and matrix-tested build, raising the
contributor onboarding bar from "maintain your own multi-fork patchset"
to "`cargo build` works."

---

### Phase 1 — Documentation and clarity ✅ shipped

*Documentation pass across the public-API surface.*

**What it accomplished**: `# Example` doc blocks on every public API in
`fuel-core` / `fuel-nn` / `fuel-datasets` / `fuel-onnx`; per-crate "what
this is for / not for / use next" header in every `lib.rs`; per-layer
anti-goals documented (see the "Anti-goals by layer" table near the
bottom of this file); maturity labels (`stable` / `evolving` /
`experimental` / `example-only`) on major subsystems; top-level
[`GUIDE.md`](GUIDE.md) for intent-based routing; canonical
[`PATTERNS.md`](PATTERNS.md) for the five minimal-program archetypes
(tensor math, autograd module, pretrained-model load+forward,
inference+sampling, custom op extension).

---

### Phase 2 — Structural: use-case crate separation ✅ shipped

**What it accomplished**: created `fuel-inference` and `fuel-training` as
leaf crates with the architectural property that *nothing in the Fuel
ecosystem depends on either of them*. Both aggregate; neither defines.

`fuel-inference` owns the production-quality inference primitives
contributed from the Lightbulb engine: prefix caching, composable eviction
(`EvictionPolicy` + `VotingAggregator` + LRU + H2O), KV compression
(KIVI / R-KV / Low-Rank), segmented eviction (`SpanRegistry`), tiered
storage (GPU→CPU→Disk with position-ID preservation for RoPE re-injection),
streaming-LLM policy, memory-aware scheduler, speculative decoding,
chunked prefill, MoE routing, context compression, and tool-call
infrastructure. `fuel-training` owns training-loop abstractions:
LR schedulers (6 variants), gradient clipping (L2 + per-element),
gradient accumulation with averaged grads, checkpointing, and a composable
training-loop driver.

Mechanical follow-ons that shipped alongside: `kv_cache` + `sampling`
moved down to `fuel-core` (avoids circular deps); `fuel-inference`
re-exports from there; `fuel-transformers::generation` decoupled from
`fuel-nn`; `QuantizableLinear` (`Linear` × `QMatMul` behind one `Module`)
+ `LoraLinear` added to `fuel-nn` as first-class types so adapter-enabled
and quant-mixed inference needs no external crate.

**Why it mattered**: this is the cut that lets a tensor-math user, a
custom-model user, and an inference-pipeline user each carry only what
they need. The "early-exit property" (see Identity above) becomes
structurally enforceable rather than aspirational.

For *what's in each crate today*, read the corresponding `lib.rs` header
rather than this ROADMAP entry — the crates are the source of truth.

**Mixed-precision training policy** is deferred — requires `DType`
autocast hooks in `fuel-core` that don't yet exist. Picks up when
Phase 7.5 work items C/E (autograd + Tensor fission) make the autocast
seam natural.

<!-- BEGIN Phase 2 detail dropped 2026-05-22 (was 290+ lines of shipped tracking).
     See git log for the per-PR landings; see fuel-inference/src/lib.rs,
     fuel-training/src/lib.rs, fuel-nn/src/{quantizable_linear,lora}.rs
     for the current public surface. -->
---

### Phase 3 — Structural: model area organization ✅ shipped

**What it accomplished**: reorganized `fuel-transformers/src/models/` from
a flat namespace of 100+ files into category subdirectories
(`llm/` / `vision/` / `audio/` / `diffusion/` / `multimodal/` /
`encoders/` / `common/` / `quantized/`). Config structs and forward passes
stay in `models/`; KV-cache handling, decode loops, and sampling hooks live
in `fuel-inference`. Published API surface unchanged.

The Lightbulb multi-GPU primitives shipped as a new `fuel-parallel`
crate (5 modules, 58 tests): `topology` (device graph with interconnect
metadata, fastest-peer + transfer-time queries), `comm` (object-safe
`Communicator` trait — all-reduce / all-gather / reduce-scatter /
broadcast / barrier), `tensor_parallel` (column + row sharding,
`TensorShard`, `ColumnParallel` / `RowParallel<C>`), `pipeline_parallel`
(GPipe + 1F1B schedules with bubble-ratio analysis), `distributed_cache`
(per-rank-per-layer seq tracking + flush protocol).

<!-- BEGIN Phase 3 detail dropped 2026-05-22 (was 230+ lines of shipped tracking).
     See fuel-transformers/src/models/ for the directory layout; see
     fuel-parallel/src/lib.rs for the multi-GPU primitives. -->
---

### Phase 4 — Ergonomics ✅ shipped

**What it accomplished**: per-layer shape-context error wrapping
(`Linear`, `Conv{1,2}d`, `LayerNorm`, `RmsNorm`, `Embedding`, `BatchNorm`,
`GroupNorm`, `LSTM`, `GRU`, `ConvTranspose{1,2}d`) so errors carry
operation + shapes + expectations inline rather than requiring source
reads. `TrainingContext` (in `fuel-nn`) bundles
`Var`/`VarBuilder`/`VarMap`/optimizer behind `cpu_f32()` / `cpu_bf16()`
shorthands. Fluent builder methods (`.with_lr()`, `.with_stride()`,
`.no_bias()`, ...) on `ParamsAdamW`, `Conv1dConfig`, `Conv2dConfig`,
`LayerNormConfig`, `BatchNormConfig`. Naming-audit aliases for
the most-searched-for APIs (`transpose_last_two` → `t`, `matvec` →
`mv`, `scale_and_shift` → `affine`, `negative_log_likelihood` →
`nll`, `AdamWConfig` → `ParamsAdamW`).

`cargo doc` warnings: zero across `fuel-core` / `fuel-nn` /
`fuel-transformers` (was 67 combined). `cargo test --doc`: 1804 / 1804
passing across the workspace (was 167 failures in fuel-transformers
alone before the seven-root-cause sweep).

<!-- BEGIN Phase 4 detail dropped 2026-05-22 (was 75+ lines of shipped tracking).
     The per-layer shape-context wrappings + builder methods + alias names
     are documented in the corresponding crate's API docs. -->

---

### Phase 5 — Backend modularity and pluggable dispatch

*Large scope. Affects only the Backends/Kernels layer and fuel-core's Device
type. Layers 1–4 are untouched and do not need to wait for this phase.*

#### Starting point — what already exists

The seam is present. `fuel-core/src/backend.rs` defines `BackendDevice` and
`BackendStorage` as associated-type traits; CPU, CUDA, and Metal all implement
them. CUDA and Metal are already behind Cargo feature flags, meaning a
CPU-only user never compiles GPU code. The kernel crates (`fuel-cuda-kernels`,
`fuel-metal-kernels`, `fuel-flash-attn`, `fuel-ug`) are already separate
from `fuel-core`.

What is absent:

- The `Device` enum is closed: `Cpu`, `Cuda(CudaDevice)`, `Metal(MetalDevice)`.
  Adding a fourth backend means modifying `fuel-core`, which is a breaking
  change. Third parties cannot extend the type without forking.
- Routing is device-level. Once a tensor has a `Device`, every operation on it
  uses that device's single backend. There is no mechanism for per-operation
  backend selection.
- There is no probe or score mechanism. The framework does not measure backends
  against each other; users select them manually with no performance or
  correctness guidance.
- There is no correctness oracle. No reference backend exists to validate other
  backends against.

#### Reference point — faster-blaster

The faster-blaster project (sibling workspace) is a fully-realized modular
dispatch system for linear algebra. Studying it sharpens what Fuel's backend
story should eventually look like:

- Each backend is a plugin with a `probe-score-init` lifecycle. Plugins score
  themselves 0–100 based on hardware compatibility; the system selects the
  highest-scoring available plugin automatically.
- A **Judge module** runs once at startup, profiling every
  (operation, backend, device, size class, dtype) tuple for both latency and
  numerical precision. It uses a correctness oracle — the
  `faster-blaster-reference` pure-C reference implementation — as the ground
  truth, and stores accuracy curves ("N digits on X% of inputs") alongside
  timing statistics.
- **Ranked dispatch tables** store the top-N backends per operation, per
  criterion (fastest, most accurate, balanced). The router does O(1) lookups;
  the Judge never runs during normal dispatch.
- An **execution DAG planner** models sequences of operations as a layered DAG
  where each node is a (step, backend) pair and each edge carries a transfer
  cost (H2D, D2H, D2D). Forward dynamic programming finds the minimum-cost path
  across the DAG, allowing a sequence of operations to span CPU and GPU backends
  when the compute savings at each step outweigh the transfer cost to get there.
- Operations route **independently**: a dot product might go to AOCL-BLIS while
  the next GEMM goes to cuBLAS if the DAG planner determines the PCIe round trip
  is worth it.

This is the far end of what a modular backend system can look like. Fuel's
path arrives there through three well-sequenced tiers. None of them require
changing anything above the Backends/Kernels layer.

#### Tier 1 — Expose existing flags (now, cost: negligible)

This is a documentation-only change.

- [x] Document the `cuda` and `metal` Cargo feature flags prominently: in
  `fuel-core/src/lib.rs` (feature table added to crate header) and the top-level
  `README.md` (new "Cargo feature flags" section with table + code examples).
- [x] Add a one-line note to each feature confirming that omitting it produces a
  clean CPU-only build with no GPU code compiled in.
- Outcome: users who want CPU-only, or CUDA-only, or CUDA+Metal know how to
  express that today, without waiting for any structural change. ✅

#### Tier 2 — Formalize the plugin seam (near-term, non-breaking)

The `BackendDevice` and `BackendStorage` traits are currently internal
implementation details with no documentation. Promote them to a published,
stable interface:

- [x] Write full API documentation for both traits, covering every method's
  contract, preconditions, and expected error conditions. (`fuel-core/src/backend.rs`
  now documents all 30+ methods on both traits, including layout semantics,
  dtype contracts, safety requirements for `alloc_uninit`, synchronization
  guarantees for `to_cpu_storage` and `synchronize`, and the ordinal model
  for `BackendDevice::new`.)

**Backend extraction analysis** (completed — revised):

The original analysis said extraction was infeasible because `CpuStorage` is
the universal exchange type. That was corrected: `CpuStorage` as a *data type*
(an enum of typed Vec buffers) is distinct from the CPU *backend implementation*
(the code that executes BLAS, matmul, etc.). The data type belongs in a shared
foundation crate; the backend implementation is separable.

**Three-layer architecture** ✅ COMPLETE (2026-05-01):

```text
fuel-core-types       ← Shape, DType, Layout, Error, HostBuffer,
                          DynBackendStorage/DynBackendDevice traits,
                          InplaceOp1/2/3 traits, WithDType, SIMD kernels,
                          conv params, dispatch + capability + probe types
        ↑
fuel-cpu-backend      ← CpuStorage (newtype on HostBuffer),
fuel-cuda-backend       impl DynBackendStorage / DynBackendDevice on
fuel-metal-backend      concrete storage/device types directly
        ↑
fuel-core             ← Device/Storage backend-agnostic newtypes, Tensor,
                          Var, backprop, custom_op, lazy graph, factories
                          registry, thin re-export bridges in cuda_backend/
                          and metal_backend/
```

Static `BackendStorage`/`BackendDevice` traits dropped entirely;
polymorphism is `Box<dyn DynBackendStorage>` / `Arc<dyn DynBackendDevice>`.
Adding a backend = new crate impl-ing the dyn traits + a `BackendFactory`
entry in `fuel-core/src/factories.rs`.

Progress:

- [x] Created `fuel-core-types` crate with 21 source files extracting all
  foundational types, traits, and CPU SIMD infrastructure from `fuel-core`.
  Compiles standalone. Added to workspace members.
- [x] Verified full workspace builds with `fuel-core-types` present
  (`cargo check --workspace` passes).
- [x] Wire `fuel-core` to re-export from `fuel-core-types` — DONE.
  All foundational types now live in `fuel-core-types`: `shape`, `layout`,
  `strided_index`, `dummy_dtype`, `dtype` (with `DType`/`WithDType`/
  `IntDType`/`FloatDType`), `error` (with `Error`/`Result`/`Context`),
  `cpu_storage` (with `HostBuffer`/`HostBufferRef`/`CpuDevice`/
  `CpuStorage` alias), `dyn_backend`, `op`, `conv`, `quantized`,
  `inplace_op`, `capability`, `probe`, `dispatch`, `scalar`, `backend`
  (with `HostStorage`). The orphan-rule blockers got resolved when the
  static `BackendStorage`/`BackendDevice` traits were dropped (step 8 of
  the 15-step refactor): the impls that needed `fuel-core` types
  disappeared along with the traits. `MetalError` lives in
  `fuel-metal-backend` and is re-exported through the bridge module.
- [x] Create `fuel-cpu-backend` crate (extract cpu_backend module from
  fuel-core). Created with 6 source files: `lib.rs`, `ops.rs` (~1770 lines of
  CPU computation kernels — MatMul, pooling, convolution, reductions, index ops),
  `utils.rs` (Map traits + vectorised helpers), `conv2d.rs` (tiled/im2col Conv2D),
  `mkl.rs` (Intel MKL FFI), `accelerate.rs` (Apple Accelerate FFI). Compiles
  standalone against `fuel-core-types`. Added as workspace member and as
  dependency of `fuel-core` with `mkl`/`accelerate` feature forwarding.
- [x] Delegation: fuel-core's `cpu_backend/mod.rs` now delegates all major
  operations to `fuel-cpu-backend` via 5 macros (`cpu_map1!`, `cpu_map1any!`,
  `cpu_map2!`, `cpu_map2u8!`, `cpu_map2_in_place!`).
  Delegated: affine, avg_pool2d, max_pool2d, upsample_nearest1d/2d,
  upsample_bilinear2d, reduce_op (Sum/Min/Max/ArgMin/ArgMax), index_select,
  gather, where_cond, index_add, matmul, cmp, conv1d, conv2d,
  conv_transpose1d, conv_transpose2d, scatter_set, scatter_add_set.
  Type unification: CmpOp, ReduceOp re-exported from fuel-core-types;
  ParamsConv1D/2D/Transpose1D/Transpose2D + CudnnFwdAlgo re-exported from
  fuel-core-types (~250 lines of duplicate struct defs removed from conv.rs).
  Dead local helper structs removed (~1370 lines total: MatMul, Conv1D/2D,
  ConvTranspose1D/2D, Im2Col, Im2Col1D, Col2Im1D, Cmp, WCond, ReduceIndex,
  ReduceSum, Affine, AvgPool2D, MaxPool2D, UpsampleNearest1D/2D,
  UpsampleBilinear2D, Gather, IndexSelect, ElemUpdate, Set, Add,
  Scatter, IndexAdd, plus conv2d.rs submodule deleted).
  mod.rs reduced from 3284 → 1917 lines. Zero errors, zero test failures
  (462 fuel-core tests pass).
  Trait consolidation: UnaryOpT/BinaryOpT re-exported from fuel-core-types
  (eliminated ~65 lines of duplicate trait definitions in op.rs).
  `unary_dispatch<B>`/`binary_dispatch<B>` functions added to fuel-cpu-backend
  for standalone use. fuel-core's `unary_impl`/`binary_impl` retain thin enum
  dispatch calling fuel-cpu-backend's `unary_map`/`binary_map` helpers
  (full delegation blocked until CpuStorage is re-exported from fuel-core-types,
  which requires resolving dtype.rs/convert.rs orphan rule issues).
- [x] Extract cuda/metal backends into separate crates (DONE 2026-05-01).
  The 15-step backend-agnostic refactor (082d7ffa→f0c00233) moved all
  CUDA logic to `fuel-cuda-backend` (originally named `fuel-graph-cuda`)
  and all Metal logic to `fuel-metal-backend` (originally `fuel-metal`).
  fuel-core's `cuda_backend/`/`metal_backend/` modules collapsed to thin
  re-export shells. Static `BackendStorage`/`BackendDevice` traits dropped
  entirely; backends impl `DynBackendStorage`/`DynBackendDevice` (in
  `fuel-core-types`) directly on their concrete storage/device types.
  `BackendFactory` registry in `fuel-core/src/factories.rs` makes
  `judge.rs`/`probe.rs` backend-agnostic.

  **Phase B follow-on (2026-05-01)** — finished the symmetry pass that
  the 15-step plan deliberately deferred:
  - B1: deleted `Device::new_cuda`/`new_cuda_with_stream`/`new_metal`/
    `cuda_if_available`/`metal_if_available`/`as_cuda_device`/
    `as_metal_device`/`from_cuda_device`/`from_metal_device` from
    `fuel-core/src/device.rs`. Replacements (`cuda_backend::new_device`,
    `From<CudaDevice> for Device`, etc.) live in the bridge modules.
    71 caller sites migrated.
  - B2: split `UgIOp1` (the fuel-ug compiled-kernel bridge) into
    `fuel_cuda_backend::ug::CudaUgIOp1` + `fuel_metal_backend::ug::MetalUgIOp1`;
    moved `InplaceOp1/2/3` traits to `fuel-core-types` so backend crates
    can implement them without a cycle.
  - B3: renamed crates `fuel-graph-cuda` → `fuel-cuda-backend` and
    `fuel-metal` → `fuel-metal-backend` for naming consistency with
    `fuel-cpu-backend`.

  End state: fuel-core's CUDA/Metal awareness is confined to two places
  — the `BackendFactory` registry entries in `factories.rs`, and the
  thin bridge shells in `cuda_backend/`/`metal_backend/` (`From<XxxDevice>
  for Device` + a few free fns). `device.rs`/`custom_op.rs` no longer
  name any backend-specific types.

#### Tier 3 — Open Device for third-party backends ✅ COMPLETE

`Device::Custom(Arc<dyn DynBackendDevice>)` and `Storage::Custom(Box<dyn DynBackendStorage>)`
are now fully wired. Object-safe `DynBackendStorage` (31 methods) and `DynBackendDevice`
(11 methods) traits live in `fuel-core/src/dyn_backend.rs`.

**Implementation summary:**

- `Device::Custom` arm handled in all 16 match sites in `device.rs`
- `Storage::Custom` arm handled in all match sites across `storage.rs` (~30 methods),
  `tensor.rs` (4 sites), `safetensors.rs` (2 sites), `quantized/mod.rs` (3 sites),
  `quantized/ggml_file.rs` (1 site), and `fuel-pyo3` (1 site)
- `UnaryOp::from_name` / `BinaryOp::from_name` helpers bridge generic `UnaryOpT`/`BinaryOpT`
  to enum-based dynamic dispatch for custom backends
- `CustomOp1/2/3` and `InplaceOp1/2/3` return errors on custom backends (these use
  backend-specific `cpu_fwd`/`cuda_fwd`/`metal_fwd` that have no dynamic equivalent)
- Quantized operations bail on custom backends (Q-format is backend-specific)

**Design decisions:**

- `DynBackendStorage::device_arc_dyn()` returns `Arc<dyn DynBackendDevice>` so
  `Storage::device()` can reconstruct `Device::Custom(arc)` without redundant storage
- `_dyn` suffix on all trait methods avoids name collisions with `BackendStorage`
- `Cpu`/`Cuda`/`Metal` arms remain zero-overhead static dispatch; only `Custom` pays
  `dyn` overhead

**Usage example:**

```rust
use fuel_core::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_core::Device;
use std::sync::Arc;

struct MyDevice { /* ... */ }
impl DynBackendDevice for MyDevice { /* ... */ }

let device = Device::custom(Arc::new(MyDevice::new()));
```

#### Tier 3b (ABANDONED) — Full enum-to-trait-object migration

*Note: April 2026 Architectural Pivot — The `dyn` dispatch migration has been officially halted. While it solved the closed-enum problem, it created runtime overhead, required internal downcasting (`as_any().downcast_mut()`) violating strict type safety, and masked the physical reality of hardware memory boundaries.*

---

### Phase 6 — Lazy Execution & Autonomous Scheduling

*Large scope. Transitions Fuel from eager execution to a lazy computation
graph with an autonomous router that selects backends per operation, fuses
kernels, and minimizes cross-device transfer cost. The rewrite deliberately
severs upstream compatibility with HuggingFace Candle; models are re-ported
against Fuel's new semantics rather than kept in sync with upstream.
Multi-node execution is explicitly out of scope for this phase and is
deferred indefinitely.*

#### Why lazy execution is required

The end-state Fuel is aiming for — per-operation backend selection,
cross-device routing, kernel fusion, and transfer-cost-aware scheduling —
is fundamentally incompatible with eager execution. An eager op commits to
a backend the moment it runs, so there is nothing for a planner to analyze,
fuse, or re-route after the fact. Opt-in "compile this region" approaches
(PyTorch's `torch.compile`, for example) are notoriously leaky: graph
breaks, partial coverage, recompilation surprises, and a two-mode mental
model developers have to keep in their heads. Fuel commits to lazy-by-
default to keep the ceiling clean and the mental model single.

The cost of this decision is the loss of direct HF Candle model
portability. The ~100 models in `fuel-transformers` today will need to be
re-validated under lazy semantics; Phase 6 validates against a focused
anchor set rather than the full catalog (see Anchor models below). Models
outside the anchor set may be ported later as demand warrants or retired
where they are not earning their maintenance cost.

#### Prerequisites (hard gates before 6a can start)

- [x] **`fuel-reference-backend` crate.** Landed. 173 tests covering every
      op in the catalog as textbook-correct implementations. Used as the
      correctness oracle for `fuel-graph-cpu` via equivalence tests —
      every op the fast executor runs has a reference counterpart that
      the test suite verifies it agrees with.
- [x] **CI oracle gate.** Each Phase 6a anchor's synthetic-weights
      forward test now realizes output through *both*
      `realize_f32()` (fast) and `realize_f32_reference()` (oracle)
      and asserts elementwise `assert_allclose_f32(atol=1e-4,
      rtol=1e-3)` agreement. Since `cargo test --workspace` already
      runs in CI on Linux/Windows/macOS, the gate is live — 8 anchor
      tests cover BERT, ConvNeXt, Whisper (enc+dec), SD CLIP text
      encoder, SD VAE, Qwen2-MoE, YOLOv8. (SD UNet is not yet covered:
      it has no end-to-end forward test, building one would triple
      the test runtime, and the UNet uses the same primitives the
      VAE already exercises. Follow-up.)
- [~] **Debuggability requirements**, non-negotiable from day one:
  - [x] `Tensor::realize()` (originally drafted as `realize_eagerly()`) —
    forces immediate execution of any pending graph for a single
    tensor, for use in debug prints and interactive development.
    *Shipped as Phase 7.5 B1 (commit a8e192ff). Today it's an Arc-clone
    identity because every Tensor is eagerly realised at construction;
    becomes a real executor invocation once Phase 7.5 B3 wires op
    methods to build graph nodes.*
  - [~] Planner "why did I pick this" traces — `FUEL_TRACE=1`
    environment variable produces a Chrome-compatible trace of all
    realize-time operations (per-node spans, span attributes including
    op type and shape) via tracing/info_span! instrumentation in both
    CPU and CUDA executors. *Per-dispatch "candidates considered +
    cost-model inputs" attribution is not yet exposed; the dispatch
    table's `pick(op, dtype, size, criterion)` returns only the
    selection, not the rationale.*
  - [x] Shape mismatch errors report *where in the graph* the mismatch
    occurred. Build-time shape validation in the `Tensor::*` builders
    catches most of these with source locations via Rust panics; the
    structured graph-traversal version is also now in place — every
    per-node realize in both `fuel-reference-backend::exec` and
    `fuel-graph-cpu` is wrapped in `catch_unwind` + `resume_unwind`
    with a `Graph::describe_node(id)`-produced prefix, so realize-time
    panics carry the exact node id, op short name, output shape/dtype,
    and input shapes/dtypes.

#### Sub-phase 6a — Lazy frontend + single-backend (CPU) planner

*The smallest working version. Prove the architecture on the simplest
possible hardware configuration.*

- [x] Define `Tensor` as a pure handle (node ID) referencing a lazy
      computation graph. **Landed** as `fuel_graph::Tensor`. Tracks `Shape`,
      `DType`, and the pending op tree. No hardware generics. Lives in the
      `fuel-graph` crate.
- [x] Build the graph data structure, node types, and basic graph-walking
      utilities. **Landed**: `fuel_graph::Graph`, `fuel_graph::Op` with 60+
      variants, `fuel_graph::Node`, `fuel_graph::topo_order` as iterative
      post-order DFS (deep graphs up to 10k+ nodes tested). 64 unit tests
      in fuel-graph.
- [x] Implement `.realize()` as the boundary between graph-building and
      execution. **Landed**: `fuel_reference_backend::exec::realize_*` and
      `fuel_graph_cpu::realize_*` (fast path) both expose this boundary.
      `LazyTensor::realize_f32()` on top of them.
- [x] Implement a **minimal planner**. **Landed** as the topo-walk
      executor. Not yet ranked, profiled, or size-class-aware — that's
      Sub-phase 6b's Judge module. The interface is defined and later
      phases extend it.
- [x] Implement the **saved-tensors tracker**. **Landed** implicitly: the
      forward graph itself IS the saved-tensors store. Backward rules
      reference forward node IDs directly, and the executor's cache holds
      each node's result through the combined forward+backward walk.
- [x] Implement **unfused backward autograd**. **Landed**: `Tensor::backward`
      in `fuel-graph` with per-op gradient rules for every differentiable
      op, including MatMul, Transpose, Permute, Softmax, LayerNorm, RoPE
      (compositional), RmsNorm (compositional), and SwiGLU / SiLU. Full
      forward-and-backward tests on a 3-layer LLaMA-style transformer
      block passing as of this session.
- [x] Implement **sequence-length bucketing** for dynamic shapes.
      `fuel_core::seq_bucketing` module provides `pick_bucket(seq_len,
      &buckets)` and a `BucketedLen` wrapper carrying `(actual,
      bucket, padding)`. A `DEFAULT_BUCKETS` constant
      `[64, 128, 256, 512, 1024, 2048, 4096]` mirrors serving-side
      conventions. Integration into `generate()` is a follow-up
      (requires graph-memoization plumbing the executor doesn't have
      yet); the primitive stands alone. Phase 6d's paged attention
      supersedes this — tracked for removal at that point.
- [x] **KV cache for autoregressive decode.** `LlamaKVCache` /
      `LayerKVCache` types, `forward_with_cache` method that prepends
      cached K/V via concat, `realize_many_f32` (both executors) for
      single-walk multi-root evaluation. `generate()` uses prefill-then-
      decode: one O(prompt²) forward, then O(total_seq) per token.
      Correctness validated by `generate_with_cache_matches_non_cached_
      generate` test (token-for-token equivalence under greedy).
- [x] **Causal attention mask.** Additive lower-triangular mask applied to
      scores before softmax in both `apply_layer` and
      `apply_layer_with_cache`. During single-token decode the mask is
      all-zeros (no-op); during prefill it's the standard triangular.
      Without this, the model produced token salad.
- [x] **Streaming token output.** `generate_streaming()` takes an
      `FnMut(u32)` callback invoked per freshly sampled token. The
      `llama-lazy` and `qwen-lazy` binaries use delta-decoding (decode
      full sequence, print only the new text) for correct multi-byte
      UTF-8 streaming.
- [x] **Qwen2 architectural support (anchor model #2).** Optional Q/K/V
      attention biases on `LayerWeights`, auto-detected by the safetensors
      loader (LLaMA stores `None`). `apply_optional_bias` helper.
      `qwen-lazy` binary defaults to `Qwen/Qwen2-0.5B-Instruct`. EOS
      scan includes `<|im_end|>` for Qwen2 chat. Two new tests covering
      bias correctness and cached-vs-non-cached equivalence with biases.
- [x] **Performance: Arc-shared weights.** `ConstData` variants held
      `Arc<[T]>`, `RefTensor` stores `Arc<[T]>`, `LlamaWeights` fields
      are `Arc<[f32]>`. Const-node evaluation in both executors was a
      refcount bump, not a memcpy. Eliminated ~8 GB/call of allocation
      churn on TinyLlama. *(Phase 7.5 G2 retired `ConstData` itself —
      bytes now live in graph-Storage slots, with the Arc-shared
      perf property carried by the slot's `Arc<RwLock<Storage>>` and
      the executor's liveness-witnessed const_pool cache.)*
- [x] **Performance: zero-copy broadcast and reshape.** `broadcast_to`
      detects pure-padding cases (e.g. `[M,K]` → `[1,M,K]`) and Arc-
      shares the source buffer. General path preallocates scratch outside
      the inner loop (was allocating per-element). `reshape` shares via
      Arc since row-major reshape is metadata-only.
- [x] **Performance: gemm parallelism + shared RoPE tables.** fast_matmul
      uses `Parallelism::Rayon(0)` (all cores). `build_rope_tables` +
      `rope_with_tables` let all 22 layers share one pair of cos/sin
      const nodes per forward instead of each building its own.
- [x] **Fast CPU executor as a step beyond the original 6a plan.** Landed
      as `fuel-graph-cpu` — a gemm-backed executor that's ~50-200× faster
      than the reference path on matmul-bound workloads. Equivalence
      tested against reference on 6 matmul + transformer-block tests.
      `LazyTensor::realize_*` defaults to this fast path;
      `realize_f32_reference` is available for oracle/debugging use.
- [x] **HuggingFace Hub integration.** `LlamaModel::from_hub(repo_id)`
      and `LlamaTokenizer::from_hub(repo_id)` in `fuel_core::lazy` wrap
      `hf-hub` for one-call download + load. Supports both sharded and
      non-sharded safetensors layouts. Handles bf16/f16/f32/f64 loading
      with automatic conversion to f32 for the graph.
- [x] **`LazyTensor` bridge in `fuel-core`.** A wrapper around
      `fuel_graph::Tensor` that presents fuel-core-style method names
      (add, mul, matmul, softmax_last_dim, etc.) so existing model code
      has a gradual migration path onto the lazy backend. 38 lib tests
      in fuel-core covering the bridge, realization, the Llama end-to-end
      pipeline, and the `generate()` decode loop.
- [x] **Runnable end-to-end example**: `fuel-lazy-examples` crate with a
      `llama-lazy` binary that downloads a LLaMA-family model from HF
      Hub, tokenizes a prompt, runs the full forward pass through the
      lazy graph, samples tokens, and prints the decoded result.
      Defaults to TinyLlama for auth-free execution; works with Llama 3
      (requires HF token + license acceptance) via command-line override.
- [x] Port the **anchor model set** (see below) to the lazy API, one at a
      time, validating each against the reference backend via the oracle
      gate. All seven landed:
  - **LLaMA family** (Llama 1/2/3, TinyLlama, Mistral): end-to-end via
    `LlamaConfig`/`LlamaModel`. TinyLlama 1.1B generates coherent text
    at ~0.28 s/token on a modern desktop CPU (DRAM-bandwidth-bound at
    f32). Mistral is architecturally identical and runs without code
    changes.
  - **Qwen2**: end-to-end via the same model code with optional Q/K/V
    attention biases. `Qwen/Qwen2-0.5B-Instruct` generates coherent
    text at ~0.25 s/token. Bias detection is automatic from safetensors.
  - **BERT** (`fuel_core::lazy_bert`): encoder-only with real
    `bert-base-uncased` weights; both modern `.weight/.bias` and legacy
    `.gamma/.beta` LayerNorm conventions supported. Produces plausible
    token-embedding norms on sample inputs.
  - **Whisper** (`fuel_core::lazy_whisper`): encoder-decoder with
    cross-attention, mel-spectrogram input, greedy decode. Outputs
    `[BLANK_AUDIO]` on silence with the real `openai/whisper-tiny`
    checkpoint.
  - **ConvNeXt** (`fuel_core::lazy_convnext`): `timm/convnext_tiny.fb_in1k`
    end-to-end; top-1 class 722 ("ping-pong ball") on a centered bright
    spot. Uses the native `Op::Conv2D` for the depthwise 7×7 kernel
    (42s → 622ms after native-op wiring, ~67×).
  - **SD 1.5**: three components — CLIP text encoder
    (`fuel_core::lazy_sd_text_encoder`), VAE decoder
    (`lazy_sd_vae`), and UNet (`lazy_sd_unet`). All three run end-to-end
    against `stable-diffusion-v1-5/stable-diffusion-v1-5` weights with
    the tokenizer pulled from `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`.
    All conv helpers now dispatch to the native `Op::Conv2D`.
  - **Qwen2-MoE** (`fuel_core::lazy_qwen2_moe`): dense-routing MoE
    decoder. Shape-validated on synthetic weights (Qwen/Qwen2-57B-A14B
    is a 14 GB download held off real-weight validation).
  - **YOLOv8** (`fuel_core::lazy_yolov8`): backbone (C2f + SPPF) +
    PAN neck + 3-scale decoupled detect head with DFL decode and pure-
    Rust NMS postprocess. Shape-validated on synthetic zero weights
    (Ultralytics ships `.pt`; reliable safetensors mirrors are a
    follow-up).

**Exit criterion for 6a**: all seven anchor models run end-to-end on the
lazy CPU backend, produce output bit-equivalent to the reference backend
within tolerance, and training works on at least one anchor (suggested: a
small Llama 3 variant). Performance does not yet need to match eager CPU
Candle — correctness is the 6a gate. **Status**: 7 of 7 anchors landed
architecturally; 5 of 7 validated against real weights (Llama / Qwen2 /
BERT / Whisper / ConvNeXt / SD 1.5). Qwen2-MoE and YOLOv8 are shape-
validated only — real-weight validation is a clean follow-up when
reliable safetensors mirrors are identified. Training on LLaMA has been
validated on a 2-layer variant with full backward-pass gradient-descent
loops in tests. The remaining Phase 6a gates — CI oracle wiring,
structured graph-location shape-mismatch errors, and sequence-length
bucketing — are tracked below.

#### Sub-phase 6b — Multi-backend routing on one device

*Introduce backend choice. Prove the planner can pick between alternatives
on a single piece of hardware.*

- [x] Add CUDA and (future) Metal as selectable backends in the planner.
      CUDA + Vulkan already selectable via `fuel-graph-router`'s
      `DynBackend` (shipped in Router Phase 2/3). Metal remains future
      work — needs a `fuel-graph-metal` mirror of `fuel-cuda-backend`.
- [x] ~~Probe-score-init lifecycle~~ → **Probe lifecycle**. Dropped the
      self-reported 0-100 score — per design discussion (user call), the
      Judge's empirical numbers are strictly more informative than any
      score a backend could self-report, so the probe step is just
      "enumerate visible (backend, device) pairs." Shipped as
      `fuel_core::probe` + per-backend `enumerate_devices()` free
      functions. Unit of judgment is `(backend, device)` rather than
      backend alone — same silicon through CUDA vs Vulkan is two
      profile entries (different submission paths). Equivalence classes
      collapse identical hardware (four RTX 4090s → profile once, apply
      to all). Live-validated on a 5-device rig (cpu + cuda + ref +
      2×vulkan).
- [x] **Judge module** shipped as `fuel_core::judge`. Profiles
      `(op, dtype, size_class) × (backend, device)` for MatMul and
      AddElementwise at three sizes each, measures median wall-clock
      latency and max relative error vs the reference backend. Warmup
      iterations discard cold-cache effects; equivalence classes
      profiled once per class. Backends: cpu + reference + cuda wired;
      Vulkan pending a `realize_f32_vulkan` helper.
- [x] **Ranked dispatch tables** as `fuel_core::dispatch::DispatchTable`.
      Three criteria (Fastest / MostAccurate / Balanced), O(1)
      `pick(op, dtype, size_class, criterion)` lookup,
      `pick_nearest()` fallback for unprofiled size classes. Reference
      backend excluded from picks by default. Balanced criterion uses
      a default accuracy penalty coefficient of 100 (1% rel error ≈
      2× latency tax).
- [x] Cost model for 6b is simple: one device means no transfer costs,
      just per-op latency + accuracy from the dispatch table.
- [x] **Orchestrator** `fuel_core::scheduling::prepare_dispatch_table`
      does probe → load persisted Judge output → skip Judge if hardware
      unchanged → build dispatch. Cold start ~50s on the dev rig; warm
      start <1s (JSON load only). Both probe and profile reports persist
      to `%LOCALAPPDATA%\fuel\` / `$XDG_CACHE_HOME/fuel/`.
- [x] Validate anchor models on CUDA, oracle-gated. **Partial**:
      `fuel-core/tests/phase6b_cuda_anchor.rs` validates a single
      matmul (within 1e-4) and a 2-layer synthetic LLaMA forward
      (within 5e-3 — exercises matmul, RoPE, softmax, RMS norm,
      SwiGLU all in one graph). Composed-graph divergence root-caused
      to a one-liner uninitialized-shared-memory bug in
      rmsnorm/layernorm cross-warp reductions
      (`fuel-cuda-kernels/src/reduce.cu`); fix landed alongside the
      anchor test. The other 5 anchors (BERT, Whisper, ConvNeXt,
      SD 1.5×3, Qwen2-MoE, YOLOv8) are pending the Vulkan / Metal
      realize plumbing — same shape of work but not blocked on
      Phase 6b. Metal validation is future work — needs a
      `fuel-graph-metal` mirror first.

**Exit criterion for 6b**: anchor models run on CUDA and Metal with
per-operation backend selection, oracle-equivalent to the reference
backend, and measurably faster than the lazy-CPU baseline on workloads
where the GPU is an improvement. **Status**: probe/judge/dispatch
machinery shipped and live-validated. Single-anchor (LLaMA) on CUDA
oracle-equivalent within 5e-3. Router-level dispatch-aware routing is
the next commit; remaining anchors on CUDA are mostly mechanical
follow-ups now that the kernel-level numerical issue is fixed.

#### Sub-phase 6c — Multi-device routing on one node

*Introduce cross-device routing. This is where the DAG planner earns its
keep.*

- [x] Extend the cost model with **transfer costs**: H2D and D2H
      shipped (Phase 6c-A, `fuel_core::transfer_cost::BandwidthMatrix`
      with probe-time measurement). Live numbers on the dev rig
      surfaced an interesting asymmetry — Vulkan→CPU readback at
      ~0.21 GB/s (22× slower than CUDA's D2H at ~4.8 GB/s) due to
      vulkane's staging-buffer download path. D2D + cross-backend
      D2D parked pending multi-GPU hardware (see
      `project_phase6d_d2d_design.md` for the audit + design + 8
      enumerated gap items). Inter-GPU bandwidth from
      `fuel-parallel::DeviceTopology` is the natural integration
      point when D2D resumes.
- [x] Implement the **DAG planner** (Phase 6c-B,
      `fuel_core::scheduling::dp_plan`). Forward dynamic programming
      over the topo-sorted graph: for each `(node, backend)` pair,
      `best_cost = compute_cost + Σ_inputs min_{b_i} (input_cost +
      transfer_cost)`. Const nodes pinned to CPU; backtrack pass
      derives the per-node placement plan. Synthetic tests confirm
      it picks CPU when transfer dominates and the dispatch winner
      when transfer is cheap.
- [x] Insert **automatic transfer nodes** into the optimized graph.
      `fuel_core::scheduling::auto_place_and_route_with_transfer_cost`
      (Phase 6c-C) wraps the full pipeline: probe → load-or-judge →
      load-or-measure-bandwidth → `dp_plan` → `apply_placement_plan` →
      `fuel_graph::opt::insert_copies`. The existing `Op::Copy`
      mechanism handles the actual transfer.
- [ ] Multi-GPU cases (tensor parallelism, pipeline parallelism)
      use the existing `fuel-parallel` primitives rather than
      inventing new ones. **Blocked on D2D**, which is parked until
      multi-GPU hardware is available for live validation.
- [x] **Real-anchor validation** (Phase 6c follow-up,
      `fuel-lazy-examples/src/bin/dp_diff`). Compares
      `recommend_placement` (Phase 6b) vs `dp_plan` (Phase 6c) on
      every Phase 6a anchor. At synthetic-test sizes the planners
      agree 100% — every op falls below the dispatch table's
      CPU↔GPU crossover. At BERT-large stress scale (h=2048, seq=256,
      intermediate=8192) **35% of nodes route differently**: the DP
      planner clusters cheap surrounding ops onto CUDA after a
      matmul puts data there (BroadcastTo ×20, Permute ×10,
      Reshape ×9, etc.), and pulls a few CUDA-winner matmuls back
      to CPU when their inputs are CPU-pinned consts and the
      H2D+D2H round-trip exceeds the compute saving.

**Exit criterion for 6c**: at least one anchor model that does not fit on
a single GPU runs correctly and faster under the DAG planner than under
any hand-tuned placement. **Status**: machinery is shipped end-to-end
on a single-device rig. The OOM-forcing validation step requires a
specific hardware setup (e.g. a model deliberately sized larger than
the available VRAM) and is the natural follow-up when someone hits
that case in production. D2D primitives needed for true multi-GPU
routing are parked pending hardware.

#### Sub-phase 6d — Kernel fusion, symbolic autograd, paged attention

*Optimization phase. Everything here is an improvement on a working 6c
baseline, not a prerequisite for it. Sub-phase 6d may be broken into
parallel tracks; its pieces are independent.*

**Track-level architectural surfaces shipped 2026-04-30.** Each track
landed the IR variant, executor dispatch, and a working
reference/CPU impl. Backend-specific fast kernels (CUDA paged-attn,
Vulkan FusedLinear, real fused MoE routing) follow incrementally on
the same hooks.

- [x] **Paged attention** (Track 1). `Op::PagedAttn { softmax_scale,
  block_size, softcap }`. Inputs:
  `[q, k_cache, v_cache, block_table, context_lens, optional alibi]`.
  Reference impl `attention_paged_naive` in
  `fuel-reference-backend::attention`. Causal mask is implicit via
  `context_lens` — collapses every variable-length decode shape into
  the same kernel dispatch. `seq_bucketing` module retired; the
  bucket-and-pad primitive is gone. Native Vulkan / CUDA paged
  kernels are follow-ups on the `GraphBackend::paged_attn` trait
  hook.
- [x] **Symbolic autograd transform** (Track 2). `fuel_graph::grad`
  module: `GradientRule` trait + `dispatch_gradient` registry that
  `Tensor::backward` consults before the legacy inline `match`.
  Migrated rules ship for Add / Mul / Relu as the recipe; remaining
  ops migrate one at a time on the same trait. The backward graph
  is now constructed via the same lazy IR as the forward, so the
  planner sees both halves and the fusion / scheduling passes
  apply uniformly. Higher-order gradients fall out for free.
- [x] **Kernel fusion** (Track 3). `Op::FusedLinear` IR variant
  representing `(a @ b) + bias` as one node, plus
  `opt::fuse_linear` pattern-match-rewrite pass that detects
  `MatMul → BroadcastTo → Add(rank-1 bias)` sequences (with a
  single-consumer guard on the MatMul). Executor's `Op::FusedLinear`
  arm dispatches as `backend.matmul + backend.binary(Add)` so all
  backends benefit immediately; backends with a true fused kernel
  (cuBLAS gemm-with-bias-epilogue, hand-written Slang) opt in by
  adding a `GraphBackend::fused_linear` override.
- [x] **Scheduler integration** (Track 4).
  `fuel_inference::scheduler_bridge` module exports
  `MemoryPressureRule: SchedulerRule` — the pilot integration that
  consults `MemoryScheduler` pressure state to bias placement
  toward each op's primary input device when memory is tight. The
  pattern: lift inference-side runtime state into a `*Snapshot`
  struct, implement a `SchedulerRule` that consumes it, plug into
  the existing `RuleScheduler` pipeline. MoE-routing,
  speculative-decode, and tiered-storage rules follow the same
  shape; MoE specifically needs Phase 9a's per-tensor metadata to
  tag ops with expert IDs first.

**Exit criterion for 6d**: Fuel's performance ceiling matches or exceeds
the best hand-tuned execution on each anchor model at the time of writing.
This is the "we built what we set out to build" gate. **Status:** all
four architectural tracks shipped; per-backend specialization (real
fused kernels, native paged-attn, MoE-aware placement) is the
remaining work.

#### Explicitly out of scope for Phase 6

- **Multi-node execution.** Networking, cluster membership, distributed
  graph serialization, fault tolerance, and cross-node scheduling are a
  separate project sitting on top of a stable Phase 6, not part of it.
  Deferred for now.
- **Additional backends beyond CPU / CUDA / Metal.** A Vulkan or WebGPU
  backend is a natural future addition — once Phase 6 is stable it slots
  in as another implementation of the backend-engine contract and joins
  the Judge's profiling matrix like any other backend. Not a Phase 6
  deliverable.

#### Anchor models

The Phase 6 validation suite. Each is chosen to stress a different part of
the architecture; a bug that escapes all seven is unlikely to matter.

| Model     | Category           | Stresses                                                  |
| --------- | ------------------ | --------------------------------------------------------- |
| Llama 3   | Decoder-only LLM   | KV cache growth, grouped-query attention, bucketing       |
| Whisper   | Encoder-decoder    | Cross-attention, fixed-length mel input, separate encoder |
| ConvNeXt  | Pure conv vision   | Depthwise separable convs, channel-last LayerNorm         |
| SD 1.5    | Diffusion pipeline | UNet, VAE, text encoder, scheduler loop                   |
| YOLOv8    | Object detection   | NMS (data-dependent postprocessing), anchor box ops       |
| BERT      | Encoder-only       | Bidirectional attention, maximum checkpoint compatibility |
| Qwen2-MoE | Mixture of experts | Dynamic expert routing, per-token subgraph selection      |

Each anchor model is ported to the lazy API in 6a and is a Phase 6a
milestone. Sub-phases 6b, 6c, and 6d each re-run the full anchor suite as
a regression gate before the sub-phase can be declared complete.

SD 1.5 may be extended to SDXL after Phase 6 ships; SDXL shares ~90% of
the structure of SD 1.5 and is a follow-up rather than a Phase 6
requirement.

---

### Phase 7 — Storage hierarchy refactor and vendor-optimized CPU backends

*Architectural cleanup that pays for itself in three places: fixing a
long-standing layering inconsistency, unblocking the per-vendor CPU
backend crates, and establishing the foundation for future multi-host
execution. Sequenced deliberately — the storage refactor has to land
first because the vendor backends depend on it, and the multi-host
support builds on both.*

#### Why this phase exists

Today `CpuStorage` lives in `fuel-core-types`, one crate up from where
it conceptually belongs (`fuel-cpu-backend`). That inconsistency was
fine when there was only one CPU backend, but it becomes an active
blocker as soon as you want more than one: external code that wants
to implement a custom op against the CPU backend ends up depending on
both `fuel-core-types` (for `CpuStorage`) and `fuel-cpu-backend`
(for the `BackendStorage` trait's impls), and can't cleanly hold its
own `BackendStorage` type that wraps `HostStorage` without duplicating
the underlying enum.

There is also no trait-level abstraction for "data that lives in host
RAM." Today `CpuStorage` is the concrete type everything reaches for,
but the needs are broader: mmap'd safetensors for lazy weight loading,
pinned memory for GPU DMA, shared-memory regions for IPC, and
(eventually) remote host references for the networked-execution
future. All of these are conceptually "host storage" but none of them
are `Vec<T>`.

#### 7a — Trait hierarchy for storage

- [x] **`HostBuffer` / `HostBufferRef` as canonical names.** The enum
      previously called `CpuStorage` is now `HostBuffer` in
      `fuel-core-types`. `CpuStorage` and `CpuStorageRef` remain as
      transparent type aliases (pattern matching, `impl` blocks, and
      all 46 downstream files work unchanged).
- [x] **`HostStorage` trait.** Standalone capability trait in
      `fuel-core-types::backend` — orthogonal to the static-vs-dyn
      dispatch axis. Two methods: `as_host_buffer() -> &HostBuffer`
      (zero-copy borrow) and `into_host_buffer() -> HostBuffer` (owned
      extraction). Implemented by `CpuBackendStorage` in
      `fuel-cpu-backend`. Future impls slot in alongside: mmap, pinned,
      shared-mem, remote.
- [x] **New trait method names.** `BackendStorage::to_host_buffer`,
      `DynBackendStorage::to_host_buffer_dyn`,
      `BackendDevice::storage_from_host_buffer[_owned]`,
      `DynBackendDevice::storage_from_host_buffer[_owned]_dyn`. Old
      `*_cpu_storage*` names have default impls that delegate, so
      nothing breaks. Hot-path call sites in fuel-core (`storage.rs`,
      `device.rs`, `tensor.rs`, `safetensors.rs`) already migrated.
- [ ] **Relocate `HostBuffer` from `fuel-core-types` to
      `fuel-cpu-backend`.** The alias stays in `fuel-core-types` for
      path compatibility. Deferred until vendor backends land.
- [x] **Additional `HostStorage` impls** (unblocked by the trait):
  - `MmappedHostStorage` for zero-copy safetensors loading. *Shipped in
    `fuel-cpu-backend::host_storage::mmap` ([fuel-cpu-backend/src/host_storage/mmap.rs](fuel-cpu-backend/src/host_storage/mmap.rs#L53)).*
  - `PinnedHostStorage` for page-locked GPU DMA memory. *Shipped in
    `fuel-cuda-backend::PinnedHostStorage`; live tests in
    `fuel-cuda-backend/tests/pinned_live.rs`.*
  - `SharedMemHostStorage` for IPC across processes. *Shipped in
    `fuel-cpu-backend::host_storage::shared_mem` ([fuel-cpu-backend/src/host_storage/shared_mem.rs](fuel-cpu-backend/src/host_storage/shared_mem.rs#L54)).*
- [x] **Fix `fuel-core/tests/custom_op_tests.rs`** — no longer gated.
      Active integration test against the post-refactor public API
      (`CustomOp1` / `InplaceOp1` / `UgIOp1`), using
      `fuel_cpu_backend::dyn_impl::CpuBackendStorage` for the
      downcast pattern.

#### 7b — Vendor-optimized CPU backend crates

*Each is a self-contained crate implementing the backend trait against
a specific vendor library. Users opt in at their own Cargo.toml level,
so there are no feature-flag conflicts across transitive deps. The
pure-Rust `fuel-cpu-backend` stays the default; these are for users
who want the last 20-40% of performance out of specific hardware.*

- [x] **FFI / safe-wrapper crates live OUTSIDE the Fuel workspace.**
      Architectural decision (2026-04-27): unsafe bindings + safe
      Rust wrappers for each vendor library are user-maintained
      standalone projects, not Fuel sub-crates. This keeps Fuel
      backend-agnostic and lets the wrappers serve other consumers
      too. Released as of 2026-04-28:
  - **AOCL**: `aocl-blas-sys`, `aocl-blas`, plus 11 sibling crates
    for the rest of AOCL. On crates.io.
  - **oneMKL**: `onemkl-sys` 0.1.0, `onemkl` 0.1.0. On crates.io.
- [x] **`fuel-aocl-cpu-backend` crate.** Shipped 2026-04-28
      (commit `05d3adb9`). Wraps `aocl_blas::gemm` for matmul,
      delegates other ops to `CpuBackend`, runs a 2×2 sgemm probe at
      `try_new` time. `aocl` Cargo feature on `fuel-core` /
      `fuel-graph-router` enables it. Windows DLL auto-discovery
      added 2026-04-28 (commit `89bdbcca`) so `cargo run` works
      without manual PATH gymnastics.
- [x] **`fuel-mkl-cpu-backend` crate.** Shipped 2026-04-28
      (commit `edad5ccd`). Same shape as AOCL: wraps
      `onemkl::blas::level3::gemm` for matmul, delegates rest. Naming
      detail: feature flag is `onemkl` not `mkl` because the legacy
      `mkl` feature on `fuel-core` is the eager-backend
      `intel-mkl-src` linkage used by ~100 example files; couldn't
      reuse the name without a sweeping migration. When the eager
      backend deprecates, `mkl` can be reclaimed.
- [ ] **`fuel-accelerate-cpu-backend` crate.** Same pattern for
      Apple Accelerate (macOS). Awaiting a user-side `accelerate`
      binding crate.
- [ ] **`fuel-armpl-cpu-backend` crate.** Same pattern for ARM
      Performance Libraries. Awaiting user-side `armpl` binding
      crate.
- [ ] **`fuel-openblas-cpu-backend` crate.** Fallback for ARM and
      RISC-V where no vendor-tuned BLAS is available. Awaiting
      user-side `openblas` binding crate.
- [~] **Runtime CPU detection (`fuel-cpu-auto-backend`).** **Supplanted
      by Phase 6b/7b's empirical dispatch.** The original idea was a
      `raw-cpuid`-based startup picker (Intel → MKL, AMD → AOCL,
      unknown → pure-Rust). Phase 6b ships a stronger answer: the
      Judge profiles all loaded backends per `(op, dtype, size_class)`
      and the dispatch table picks the empirical winner. No
      heuristic-based picker needed; the "wrong" backend on a CPU
      just loses the profile race silently. Keeping the heuristic
      picker in addition would be redundant and could disagree with
      the empirical layer. Not building this crate.

**Naming note**: the crate names are library-named
(`fuel-mkl-cpu-backend`, `fuel-aocl-cpu-backend`) rather than
vendor-named (`fuel-intelcpu-backend`, `fuel-amdcpu-backend`) because
the crate is defined by the library it wraps, not by the CPU brand
running it. Users can in principle run MKL on an AMD CPU or AOCL on
an Intel CPU, so the library-based name stays accurate regardless of
execution hardware. If you want vendor aliases on top for
discoverability, they're easy to add as re-exports.

#### 7c — Multi-host foundation (deferred)

Eventually Fuel will execute across multiple machines. The storage
refactor in 7a lays the groundwork without yet committing to any
particular transport. The piece needed when the multi-host work
actually starts is:

- [ ] **`RemoteHostStorage` impl of `HostStorage`.** Holds a handle to
      data on another machine. `as_slice_*()` methods block on a
      network fetch the first time they're called, then cache the
      result locally. Any fuel code generic over `H: HostStorage`
      works uniformly on local and remote data; only the latency
      profile changes.
- [ ] **Cluster membership and routing** live in a separate future
      `fuel-cluster` crate — not Phase 7, not Phase 6. They depend on
      `RemoteHostStorage` being a working trait impl first.

Multi-host execution is explicitly out of scope for the current
roadmap, but Phase 7's trait-based storage design is chosen with it
in mind so that when it does land, no fundamental interface changes
are required in the frontend.

---

### CUDA stack restructure (2026-04, in progress)

*Not a numbered phase — backends/kernels-layer cleanup that
happened alongside the baracuda migration.*

Historical state: CUDA was split across `fuel-cuda` (cudarc wrapper
plus ML-layer dispatch code inherited from candle-cuda) and
`fuel-cuda-backend` (lazy-graph integration on top of it). Vulkan
parallel already looked different — `vulkane` (external FFI) feeding
`fuel-vulkan-backend` (ML-layer + graph) directly with no intermediate
wrapper crate.

With the cudarc → baracuda migration in flight, the two sides were
unified. `fuel-cuda`'s ML-layer content (`CudaStorage` /
`CudaStorageSlice`, `Map1`/`Map2`/`Map3` dispatch traits, kernel
launch scaffolding, cuBLAS / cuDNN / curand wiring, module cache for
`fuel-cuda-kernels`) moved into `fuel-cuda-backend` as internal
modules. `fuel-cuda` was deleted. Final stack:

```text
baracuda-*           (external, CUDA FFI)
   │
fuel-cuda-backend      (ML-layer + graph integration)
fuel-cuda-kernels    (PTX bundle)
   │
fuel-core cuda_backend (thin delegation to fuel-cuda-backend)
```

Parallel to:

```text
vulkane              (external, Vulkan FFI)
   │
fuel-vulkan-backend    (ML-layer + graph integration)
fuel-vulkan-kernels  (SPIR-V bundle)
   │
fuel-core vulkan_backend (future — same delegation shape)
```

Narrow-purpose crates kept separate (match the kernels crate
precedent): `fuel-cuda-vmm`, `fuel-cublaslt`, `fuel-flash-attn`,
`fuel-flash-attn-v3`.

#### Opportunities baracuda now unblocks (future work items)

These aren't blocking anything; they're capabilities baracuda exposes
that cudarc didn't, pitched for later roadmap consideration.

- [ ] **CUDA Graph capture + replay** for Phase 6's `realize()` hot
      path ([`baracuda-driver/src/graph.rs`]). Decode-heavy LLM
      inference runs the same attention + MLP sequence per token —
      prime territory for `cuGraphCapture`. Expected payoff: cuts
      per-token kernel launch overhead. Non-trivial to integrate
      because it changes the executor hot path; needs its own
      design pass.
- [ ] **Stream-ordered mempool allocation**
      ([`baracuda-driver/src/mempool.rs`]). Fuel today allocates
      a fresh `DeviceBuffer` per op output. A `CUmemoryPool` with
      trim / release policies would recycle within a stream. Needs
      buffer-lifetime analysis vs stream semantics — not a
      mechanical swap.
- [ ] **CUDA ↔ Vulkan P2P zero-copy** via
      `ExternalMemory::import` (baracuda) + `DeviceMemory::get_win32_handle`
      (vulkane). Was estimated ~2-3 days before baracuda existed;
      now closer to ~1-2 days since both sides expose the primitives.
      Gates on someone running a real multi-device model and finding
      the PCIe round-trip is the bottleneck.
- [ ] **Launch attributes** — cluster dims, programmatic stream
      serialization, priority ([`baracuda-driver/src/launch_attr.rs`]).
      Opportunistic tuning for specific kernels; measure before
      applying.
- [ ] **nvJitLink** for runtime kernel specialization. Matters when
      Fuel starts doing LoRA fusion, per-shape attention kernels,
      or other "generate a kernel for this exact problem" flows.
      Speculative.

#### Post-Phase-6 dead-op audit

- [ ] **Audit candle-heritage ops for unused code paths** once the
      Phase 6 anchor suite (Llama 3, Whisper, ConvNeXt, SD 1.5,
      YOLOv8, BERT, Qwen2-MoE) runs against the CUDA backend. Ops
      that no anchor exercises (suspected: `upsample_nearest1d`,
      `index_add`, `elu`, `const_set`) can be removed. Not done
      pre-emptively — too easy to delete something a model actually
      needs.

---

### Phase 7.5 — Core simplification: lazy-only execution, graph-rewrite autograd, and crate fissioning

*Structural cleanup that follows naturally from the now-complete
backend-agnostic refactor (the 15-step plan, branch tip f0c00233,
2026-05-01). Not urgent in the sense of blocking other work, but
high-leverage: each piece removes a tax that every consumer pays
today and unlocks downstream phases. Best done after Phase 7
stabilises and before Phase 8 lands new kernel-layer code that
would otherwise have to absorb the changes mid-flight.*

#### Why Phase 7.5 exists

The backend-agnostic refactor proved the architecture: `fuel-core`
no longer names any backend, every backend interacts through
`DynBackendStorage`, and the lazy stack (Phase 6) is the substrate
for empirical dispatch (Phase 6b), scheduler-driven residency
(PRs #1–#4), and multi-backend Router (PR #5). Three structural
debts remain that the previous architecture left in place:

1. **Two execution paths (eager + lazy) that do the same job.**
   `Tensor::matmul` runs immediately via `Storage::matmul`; the
   lazy stack builds a graph and dispatches via Router/Executor.
   The lazy path is strictly more capable (Judge dispatch,
   ResidencyEvictionRule, ConstLoweringRule, future fusion). Every
   op currently has to work in both modes — compile-time tax,
   test-matrix tax, source of subtle drift between paths.
2. **Autograd entangled with `Tensor` and the `Op` enum.**
   `Tensor_` carries an `op: BackpropOp` field that every
   inference path pays for, and `Op` does double duty as forward
   IR (used by the lazy graph) and backward tape entry (used by
   `.backward()`). Inference consumers (Lightbulb, embeddings,
   retrieval, oracle test runners, quantized-only paths) inherit
   the autograd cost for nothing.
3. **Inplace ops are a user-facing decision rather than an
   optimization concern.** `InplaceOp1/2/3` in `fuel-core-types`
   forces users to choose `relu_inplace` vs `relu` based on a
   correctness model they have to track manually, and in
   differentiated regions inplace can silently produce wrong
   gradients.

These three sit on top of a fourth pressure: `fuel-core` itself is
becoming a kitchen sink. `fuel-quantized` and `fuel-conv` already
fissioned out under real consumer pressure; the remaining
contents (Tensor + autograd + eager dispatch + loaders +
custom_op extension hook + indexer) split along clean
consumer-boundary lines.

#### Architectural decisions

**Single execution path: lazy-only + explicit `.realize()`.**
Drop eager mode. `Tensor::matmul` and every other op build a
graph node; values are produced when the user calls
`.realize()` / `.materialize()` / `.item()` / similar. The lazy
stack already has every capability eager has plus residency,
empirical dispatch, and (future) fusion. The cost of the change
is ergonomic — print-debug, dynamic control flow on tensor
values, and interop with non-Fuel code all need an explicit
materialisation call. JAX has demonstrated this idiom is
learnable. Single path also collapses the autograd story to
"is this graph differentiated?" — no "is autograd active and is
the op eager?" matrix.

**Option 2 with the lazy graph as the tape.** Autograd becomes a
graph rewrite over the forward IR, not a separate tape data
structure. The lazy graph already has every property a tape needs
(ordered nodes, input dependencies, op metadata). Backward is a
graph transformation that walks the forward graph in reverse and
emits backward nodes, then the unified graph is executed via the
same `fuel-graph-executor`. Backward implementations live in
`fuel-autograd` (or co-located per-op alongside their forward
definitions in their owning crate — `fuel-conv`, `fuel-quantized`,
etc.). `Tensor_` drops the `op: BackpropOp` field and the
`is_variable` flag. `Op` becomes pure forward IR; the lazy stack
and autograd both consume it.

This choice has strong synergy with what is already shipped:
- Phase 6b probe/judge/dispatch — backward ops are ordinary ops,
  dispatch through the same Judge/DispatchTable. Backward of
  `matmul(A, B)` is just `matmul(grad, Bᵀ)` and `matmul(Aᵀ, grad)`.
- PRs #1–#4 scheduler-driven residency — unified forward+backward
  graph means the scheduler sees full activation lifetimes and
  computes correct eviction. The destructive-input metadata on
  `Op::Release` already prevents forward eviction of tensors
  needed for backward; activation checkpointing falls out almost
  for free.
- P5 tiered residency — activations evicted during forward can
  be `fault_back`'d when backward consumes them; the planner has
  the dependency visible.
- ConstLoweringRule — backward graphs are also const-foldable.
- Higher-order gradients (`grad(grad(f))`) work because the
  backward graph is itself differentiable.

**Inplace as an optimization concern, not a user concern.** A
graph optimizer pass runs liveness analysis on the unified
forward+backward graph and rewrites in both directions:
- *Inplace-IN*: a non-inplace op whose input has no remaining
  consumers (no other forward use, no backward dependency) is
  swapped to its inplace variant. Free buffer reuse.
- *Inplace-OUT*: an inplace op whose input is needed elsewhere
  is swapped to its non-inplace variant. The original inplace
  marker becomes a hint the optimizer is free to ignore.

User consequence: the same source code is correct in inference
and training. Inplace is a perf hint, not a semantic constraint.
Inference paths get every inplace win the analysis can find;
differentiated paths get correctness for free; mixed regions
handled by the same liveness pass with no special-casing. This
generalises JAX's `donate_argnums` from a user annotation to an
optimizer-inferred property.

**Fissioning `fuel-core` along consumer boundaries.** Each split
is justified by a class of consumer that uses one side and not
the other:
- `fuel-tensor`: `Tensor` + eager-dispatch methods (now
  graph-builder methods) + indexer + custom_op + scalar helpers.
  Consumer: anyone who wants the tensor surface without autograd
  (Lightbulb, embedding/retrieval pipelines, oracle runners).
- `fuel-autograd`: tape-as-graph-rewrite + backward registration
  machinery + `.backward()` API. Consumer: training pipelines.
- `fuel-formats`: pure parsers for safetensors, pickle, GGUF,
  GGML, and imatrix wire formats. Operate on `impl Read` /
  `&[u8]` / `Cow<[u8]>` — knows about format structure, knows
  nothing about `Tensor`, `Device`, or `Storage`. Depends only
  on `fuel-core-types` (`DType`, `Shape`, `GgmlDType`).
  Consumer surface: anyone who needs to read or write these
  formats over *any* transport — file, mmap, HTTP, S3, Unix
  socket, shared-memory, network IPC. Splitting parsers from
  transport is the structural prerequisite for streaming weight
  load, inter-process tensor exchange (Fuel ↔ Lightbulb ↔ mlmf
  using safetensors as the wire schema), `RemoteHostStorage`
  (Phase 7c), and HF-ecosystem interop without bolting on
  adapters.
- `fuel-loaders`: file-transport adapters built on `fuel-formats`
  — `from_path`, `from_mmap`, `MmapedSafetensors`, etc. Builds
  `Tensor` / `QTensor` from parsed format output. Depends on
  `fuel-tensor` (post-E) and `fuel-formats`. Consumer: model-
  conversion tools and the initial-load path; not needed by
  inference-with-pre-loaded-weights or by network/IPC consumers
  that go directly through `fuel-formats`.
- `fuel-net` / `fuel-ipc` (out of scope for 7.5, natural
  follow-ons): same shape as `fuel-loaders` but over network /
  IPC transports respectively. Mentioned only to make clear that
  the `fuel-formats` / transport split is doing real work
  beyond breaking a circular dependency.
- `fuel-core` retains the umbrella facade role — re-exports the
  common API for ergonomics, like `tokio` re-exporting from
  `tokio-*`. Most users keep depending on `fuel-core` directly;
  internal consumers depend on the leaf crates.

The stopping rule: a crate boundary is justified only when there
is a class of consumer that uses one side and not the other.
Indexer and scalar helpers have no consumer asking for them
without `Tensor`, so they stay folded into `fuel-tensor`.

**Graph optimizer architecture: transactional rewrites on a single
primary graph.**

Optimization is a pipeline of rule-driven graph rewrites. Two rule
families:

- *Lowering*: high-level op → primitive subgraph (exposes fusion
  opportunities to later passes).
- *Fusion*: recognized primitive subgraph → fused op (recovers or
  improves on the original-flavour kernel).

Lowering and fusion are two halves of one machine. Rules ship as
`(matcher, rewriter)` in one registry. The lowered form is
intermediate IR, not an execution form — runs see the
post-optimization graph.

*When unpaired lowering rules are OK.* A lowering rule may ship
before its fusion partner only if the lowered form's intermediates
fit in memory at typical input sizes:

- *Lower now*, primitive intermediates linear in input:
  SoftmaxLastDim, RmsNorm, LayerNorm, NormLastDim, RoPE,
  FusedLinear, Affine, Clamp, PowI.
- *Wait for fusion partner*, intermediates blow up: MatMul →
  outer-product-then-reduce (×K), Conv2D → im2col+matmul
  (×Kh×Kw), FlashAttn → softmax(QKᵀ)V (materializes [N,N]
  attention matrix), QMatMul → dequant+matmul (eats the
  quantization memory win).

*Transaction model.*

- One primary graph in steady state.
- A working copy exists only during open transactions or briefly
  during commit-with-drain.
- Transaction = unit of consistency: at commit, all touched nodes
  are in a runnable state. No half-applied rules ever visible.
- Default granularity: one rule application. Coarser (per-pass,
  whole-pipeline) allowed when the optimizer can prove correctness
  across the larger atomic unit.
- Commit triggers: fixpoint (no more rules apply at current rule
  set) or budget exhaustion (deadline hit; used for cold-start
  TTFT).

*Switching semantics on commit.*

- New runs always start on the most-optimized version.
- In-flight runs switch at the next node-execution boundary if and
  only if the optimization is entirely ahead of the run's frontier.
  Otherwise the run finishes on the old graph; the optimization is
  preserved for subsequent runs.
- The conservative-ahead-of-frontier rule isn't just for
  approximate optimizations — lowering and fusion change node
  count and identity, so cached storage from already-executed
  nodes can't be remapped to the new graph in general. Switching
  backward across the frontier would require re-running upstream
  nodes to rebuild missing storage, which negates the in-flight
  optimization win.
- Multi-node device-queue case: when a backend queues N nodes'
  worth of ops asynchronously, the run's "currently executing"
  set is N nodes, not 1. Switching is gated on optimization being
  downstream of all queued nodes.
- Old graph lifetime ≤ max(longest queued-node duration,
  transaction duration) post-commit. Dropped once all in-flight
  runs have switched or finished.

*Concurrency.* Active graph as `Arc<Graph>` for lock-free runner
reads. Optimizer mutates a working copy uncontended. Commit =
atomic store on the active reference. No hand-rolled lock-free
machinery needed beyond `Arc` swap.

*Memory model.* Full-clone-then-mutate per transaction. CoW
between graph versions is profile-driven future work — only
attractive when graphs are 100K+ nodes and transactions touch
few-node deltas, neither of which fuel's typical inference graph
(low thousands of nodes) satisfies.

*Out of scope.* Approximate optimizations — mixed-precision
lowering (F32→BF16 hotspots), FP reassociation `(a+b)+c →
a+(b+c)`, fast/approximate intrinsics. These require explicit
approximation-budget semantics and don't fit the
strict-equivalence transaction model. Deferred until that
semantics layer exists.

*Phasing.*

- **PR 3 (next)**: rule-registry framework + first lowering/fusion
  rule pair (SoftmaxLastDim ↔ 7-node primitive subgraph) +
  synchronous "optimize-to-fixpoint, single graph" loop. No
  transactions, no snapshots, no concurrent optimization. Entry
  point factored cleanly so wrapping it in transactions later is
  mechanical.
- **Subsequent PR**: transaction snapshots, in-flight switching
  with ahead-of-frontier rule, multi-queued-node frontier
  accounting.
- **Later PR**: budget-exhaustion mode + cold-start TTFT path.
- **Future**: hot-path re-optimization triggered by execution-count
  or profiling; per-node optimization-tier tracking if needed for
  finer-grained scheduling decisions.

Work items D (inplace-as-optimization) and F (layout-tracking
pass) become rule families on this framework once it exists.

*Forward reference*: the rule registry's hand-written rules
(SoftmaxLastDim's lower/fuse pair) will become *auto-generated*
once Phase 7.6 (FusedOpRegistry) lands. Each FusedOpEntry's
`decompose` + `pattern` produce a lowering rule and a fusion rule
declaratively. The hand-written form remains as an escape hatch.
See [Phase 7.6](#phase-76--fusedopregistry-open-registry-for-fused-ops-closed-enum-for-primitives)
for the registry refactor that consumes this framework.

#### Work items

**A. `fuel-formats` extraction — transport-independent format
parser layer** (ships first, has zero `Tensor` coupling, unlocks
streaming / IPC / network use cases independent of the rest of
7.5).

The original framing here was "fission loaders for compile-time
leanness." Inspection in 2026-05-02 revealed the bigger seam:
loader files today couple format-parsing (header layout, block
decode, opcode interpretation) to transport (file path, mmap,
`Read`) to construction (`Tensor` / `QTensor` from parsed
metadata). Cutting only the construction join — what work item
A originally described as a `fuel-loaders` crate — would create
a circular dependency on `fuel-core` (loaders need `Tensor`;
`fuel-core` would re-export loaders for back-compat). Cutting
the parse-vs-construct join instead lifts a transport-agnostic
parser layer that has standalone value. See "Fissioning
fuel-core" above.

- [x] Create `fuel-formats` crate. Pure-Rust parsers for
      safetensors, pickle, GGUF (file + mmap), GGML, imatrix.
      API operates on `impl Read` / `impl Seek` / `&[u8]` /
      `Cow<[u8]>` and returns format-typed structs. Depends only
      on `fuel-core-types` (`DType`, `Shape`, `GgmlDType`).
      *Shipped 2026-05-02 (commits be7066f8 → 8f2614bb on branch
      refactor/step-11-quantized-kernels). Module surfaces:
      `imatrix::parse`, `ggml::{Header, RawTensor, read_one_raw_tensor}`,
      `gguf::{Content, TensorInfo, Value, ValueType, VersionedMagic}`,
      `pickle::{OpCode, Object, Stack, TensorInfo, read_pth_tensor_info}`,
      `safetensors::{SafeTensors, TensorView, MmapedFile}` (re-exports
      from upstream + the mmap convenience).*
- [x] Migrate the parser bodies out of
      `fuel-core/src/safetensors.rs`, `pickle.rs`,
      `quantized/gguf_file.rs`, `gguf_mmap.rs`, `ggml_file.rs`,
      `imatrix_file.rs`. Leave thin Tensor-construction wrappers
      in `fuel-core` (today's `safetensors::load(path, device)`,
      `pickle::read_all(path)`, etc.) that call `fuel-formats`
      to parse and then build Tensors. Public API of `fuel-core`
      unchanged.
      *Shipped — fuel-core's loader files now thin orchestrators
      that re-export format types and add Device-aware tensor
      construction. 126 fuel-core unit tests pass throughout.*
- [x] Add `fuel-formats` to the workspace. Verify the parser
      surface is complete by removing every byte-level read
      from `fuel-core` and confirming the wrappers don't
      reach for `byteorder` / `safetensors-rs` / etc. directly.
      *Shipped — workspace registration in commit be7066f8.
      fuel-core's remaining `byteorder`/`safetensors` imports are
      legitimate: NPY format (separate, not migrated), GGUF write
      path (Tensor-aware), and lazy_* materializers (Tensor-aware).
      No dead deps to remove from fuel-core/Cargo.toml.*
- [~] Round-trip test against the Phase 6 anchor weight set
      (BERT, ConvNeXt, Whisper, SD CLIP, SD VAE, Qwen2-MoE,
      YOLOv8) — same loaded tensors, byte-equivalent buffers,
      across both file and `Cursor<&[u8]>` paths.
      *Partial — `fuel-formats/tests/transport_independence.rs`
      exercises all 5 parsers with synthetic in-memory buffers and
      proves zero-filesystem operation. Real anchor-weight round-trip
      is gated on having those binary fixtures available in-tree;
      defer until Phase 6 anchor weights land in a test-data crate.*
- [x] Document the streaming / IPC / network use cases in
      `fuel-formats/README.md` so consumers know the parser
      surface is *the* public seam (file path is just one
      transport). *Shipped — README covers HTTP body parsing,
      inter-process tensor exchange via safetensors-on-the-wire,
      KV-cache handoff, RemoteHostStorage foundation, hot reload,
      and the pattern new transport adapters should follow.*

**A2. `fuel-loaders` finalization (post-E).** Once `Tensor`
lives in `fuel-tensor` (work item E below), the file-transport
wrappers currently in `fuel-core` migrate into a small
`fuel-loaders` crate that depends on `fuel-tensor` +
`fuel-formats`. `fuel-core` re-exports for back-compat. This
becomes ~one afternoon of mechanical extraction.

- [ ] Move `safetensors.rs` / `pickle.rs` Tensor-construction
      wrappers + `quantized/{gguf_file,gguf_mmap,ggml_file,
      imatrix_file}.rs` (now thin Tensor builders calling
      `fuel-formats`) into `fuel-loaders`.
- [ ] Decide whether `custom_op` extension hook stays with
      `fuel-tensor` (likely) or splits separately. If split,
      move to `fuel-custom-op`.
- [ ] Update `fuel-transformers` and `fuel-examples` to depend on
      `fuel-loaders` directly where weight loading is the only
      `fuel-core` API in use.

**B. Drop eager mode, introduce `.realize()`.**

Internal sub-phases (B1-B6) tracked in memory plan. B1 is shipped;
B2-B6 land *after* work items G + G2 below, both shipped
2026-05-02. G provides graph-owned Storage; G2 makes `Op::Const`
a slot-rooted unit variant. Together they're the substrate that
B's factory migration plugs into.

- [x] **B1.** Add `.realize()` / `.materialize()` / `.is_realized()`
      stubs to `Tensor`. Identity clones today; gain real semantics
      after G + B3.
      *Shipped 2026-05-02 (commit a8e192ff). 3 unit tests verify
      today-identity contract; full fuel-core test suite green.*
- [x] **B2.** Factories (`zeros`, `ones`, `from_slice`, `from_vec`,
      `from_iter`, `arange`, `arange_step`, `eye`, `full`, `rand`,
      `randn`, `meshgrid`) produce graph-rooted Tensors backed by
      `Op::Const` nodes whose Storage lives in the graph's
      storage map. *Shipped per `project_phase_7_5_work_item_b2_complete.md`:
      fuel-core eager `Tensor` factories produce node-handle tensors;
      8 view ops bridged through `realized_storage()`.*
- [ ] **B3.** Migrate every `Tensor::*` op method to build a graph
      node instead of calling `Storage::*` directly. One op family
      per commit (unary, binary, binary-scalar, cmp, reduce,
      reshape/transpose, slice, matmul, conv, qmatmul, misc).
      Dispatch becomes the lazy-stack's `realize_*` entry points,
      with a fast-path for one-node graphs to amortise per-op
      overhead.
- [ ] **B4.** Update `to_vec*`, `to_scalar`, `Display` impls, and
      any other "force value" entry points to call `.realize()`
      implicitly so users don't have to.
- [ ] **B5.** Migration pass through `fuel-nn`, `fuel-transformers`,
      `fuel-examples`: most code remains unchanged because op
      methods retain their signatures; only "inspect a value"
      sites need `.realize()`.
- [ ] **B6.** Drop eager dispatch entirely. `Storage::matmul`,
      `Storage::unary_impl`, `Storage::binary_impl`, etc. become
      dead code. Storage shrinks to a thin enum of typed buffers.
      Document the new idiom in `GUIDE.md` and `PATTERNS.md`.
      Particular care for the `if tensor.item::<f32>() > 0.5`
      case — this is the most user-visible difference from
      PyTorch eager.

**C. Sever `Op`-as-IR from `BackpropOp`-as-tape-entry; move
backward to `fuel-autograd`.**

- [ ] Confirm `Op` lives in `fuel-core-types` with no autograd
      coupling (already mostly there post-Phase 6).
- [ ] Drop `BackpropOp` and `is_variable` from `Tensor_`. Add a
      `Variable` concept that's just "a graph input the autograd
      pass differentiates with respect to" — data, not a type
      distinction.
- [ ] Create `fuel-autograd` crate. Define the
      `BackwardRule<Op>` registration trait and the
      `grad(graph, output, wrt)` graph-rewrite entry point.
- [ ] Move every existing backward closure into a `BackwardRule`
      impl. Co-locate per-op backward rules with their forward
      `Op` definitions in the owning crate where possible
      (`fuel-conv` owns Conv backward, `fuel-quantized` owns
      QMatMul backward). `fuel-autograd` provides only the
      traversal/transform machinery and the public API.
- [ ] Add a compile-time check that every `Op` variant has a
      registered `BackwardRule` (or is explicitly marked
      non-differentiable) — closes the "open enum" problem
      Option 2 normally has.
- [ ] Validate higher-order gradients work end-to-end on a small
      test case (`grad(grad(f))` for a simple function).

**D. Inplace-as-optimization graph rewrite.**

- [ ] Add `opt::inplace_rewrite` pass running before executor
      dispatch. Walks the unified graph, computes per-tensor
      liveness (forward consumers + backward dependencies),
      swaps non-inplace → inplace where the input has no
      remaining consumers, and swaps inplace → non-inplace
      where the input is needed.
- [ ] For each op that has both inplace and non-inplace forms,
      ensure the optimizer can pick freely. This is the
      shape-stable case; ops where inplace requires a different
      output shape don't qualify and the optimizer leaves them
      alone.
- [ ] Document that `*_inplace` op variants are now perf hints,
      not correctness primitives. Recommend users write the
      non-inplace form; the optimizer adds inplace where safe.
- [ ] Once the optimizer is shown to find every inplace win the
      hand-written `*_inplace` callers were getting, consider
      retiring the user-facing `*_inplace` API entirely and let
      the optimizer be the sole source of inplace decisions.

**E. Crate split: `fuel-tensor` and the umbrella facade.**

- [ ] Extract `Tensor`, eager-API methods (now graph builders),
      indexer, scalar helpers, and `custom_op` (if not split
      separately) into `fuel-tensor`.
- [ ] Reduce `fuel-core` to: re-export facade over
      `fuel-core-types`, `fuel-tensor`, `fuel-autograd`,
      `fuel-loaders`, `fuel-graph-*`, and the registered
      backends. Most public-API surface stays accessible via
      `fuel-core::*` for back-compat.
- [ ] Internal callers (`fuel-nn`, `fuel-transformers`,
      `fuel-examples`) keep depending on `fuel-core`. New
      lightweight consumers can depend on the smaller leaf
      crates directly.

**G. Graph owns Storage; `Tensor` becomes a thin handle.**

*Architectural prerequisite added 2026-05-02 between B1 (shipped)
and B2. Inserted after the design pass on B's design question
("how does `Tensor_` represent a graph-attached state?") concluded
that the long-term answer is "Tensor doesn't own Storage — the
Graph does," and that landing this before B2-B6 is cheaper than
migrating every consumer twice (once to add an `Option<GraphLink>`,
again to drop the Storage field).*

The model after G:

- `Graph` owns a `HashMap<NodeId, StorageSlot>` keyed per device.
  Each slot holds a `Box<dyn DynBackendStorage>` plus its realized
  `Layout`. Multi-device graphs (CPU↔Vulkan↔CUDA Router) keep
  working — each NodeId's slot lives on the device its placement
  side-table entry specifies.
- `Tensor` shrinks to `{ graph: SharedGraph, id: NodeId }`. The
  `Arc<RwLock<Storage>>` field on `Tensor_` goes away. (The
  `op: BackpropOp` field stays for now — its removal is work
  item C.)
- The executor's existing NodeId→Storage cache moves *as-is*
  into the Graph rather than living in executor scratch space.
  Residency machinery (`Op::Release`, `ResidencyEvictionRule`,
  `evict_from_candidates`) keeps working unchanged — it already
  operates by NodeId, so the cache's new home doesn't change its
  interface.
- `Op::Const` is a unit variant (post-G2). Bytes live in the
  graph's storage_map slot, populated when the constructor is
  called (`Tensor::from_f32`, `const_f32_like`, etc.). The
  executor's slot-first dispatch returns the slot's Arc on
  realize — no host-side payload rides on the node itself.
  Const-pool cache is liveness-witnessed via
  `Weak<RwLock<Storage>>` so slot Arc recycling can't produce
  stale cache hits.

Migration tactic — parallel-introduction-then-drop:

1. Add `StorageMap` to `Graph`. Add a "node-handle" mode to
   `Tensor` where the `storage` field is `Option<Arc<RwLock<Storage>>>`
   — `None` means "ask the graph." Existing eagerly-constructed
   Tensors stay as-is at first.
2. Migrate factories first (B2's actual work). The graph-side
   primitive (`fuel_graph::Tensor::from_storage`) is in place;
   B2 routes fuel-core's `Tensor::ones` / `::zeros` / `::from_slice`
   / etc. through it instead of the legacy `from_storage` (eager-
   mode) path. ~13 factory functions in `fuel-core/src/tensor.rs`
   plus a few callsites that use them; structural work, not
   trivially simple but not large either.
3. Migrate op methods family-by-family (B3 work, post-G). Each
   migrated family produces node-handle Tensors and removes one
   pin holding old-mode Tensors alive.
4. Once nothing produces old-mode Tensors, drop the `Option`
   wrapper and the legacy field. Tree compiles green throughout.

Sub-tasks (initial substrate 2026-05-02 + 5-commit fix-up sequence
that brought G into alignment with what was originally agreed):

- [x] Move `Storage` struct to `fuel-core-types`. *Shipped fix-up
      1/5 commit ffa9076e. Eager-dispatch methods that need
      `CustomOp1/2/3` (which transitively reference `Tensor`) stay
      in fuel-core via the `StorageApplyOps` trait extension; all
      other inherent methods moved with the struct. `Storage::device()`
      now returns `Arc<dyn DynBackendDevice>`; fuel-core wraps as
      `Device { inner: ... }` at use sites.*
- [x] `fuel_graph::Graph` owns the storage map directly:
      `HashMap<NodeId, Arc<RwLock<Storage>>>`. Sidecar
      (`fuel-core::graph_storage`) deleted. *Initial sidecar
      shipped 2026-05-02 commit 07691b97; fix-up 2/5 commit
      8c32b535 moved the map onto `fuel_graph::Graph` and dropped
      the fuel-core module entirely.*
- [x] Migrate `fuel_graph::SharedGraph` from `Rc<RefCell<>>` to
      `Arc<RwLock<>>` so `fuel_core::Tensor` retains Send+Sync
      after gaining `Option<fuel_graph::Tensor>`. *Shipped 2026-05-02
      commit e6c31614. ~100 mechanical borrow→read/write
      replacements across fuel-graph + fuel-graph-cpu/executor/router
      + cuda-backend + reference-backend + fuel-core
      lazy/scheduling. cudnn.rs's thread-local cache unrelated and
      unchanged.*
- [x] `Tensor_::link: Option<fuel_graph::Tensor>` — reuses the
      existing graph handle as the link payload (no separate
      `GraphLink` wrapper). *Initial commit 3c042bf8 introduced
      a separate `GraphLink`; fix-up 2/5 commit 8c32b535 dropped
      it in favor of `fuel_graph::Tensor` directly.*
- [x] `Tensor::realized_storage()` mode-agnostic read seam plus
      `has_graph_link()` / `graph_link()` accessors. *Initial commit
      3c042bf8; fix-up 4/5 commit f0f0df1d revised the seam to
      enforce the `(storage, link)` exactly-one-of invariant.*
- [x] Migrate every storage read in fuel-core + downstream
      (fuel-nn, fuel-flash-attn-cuda, …) through
      `realized_storage()`. ~85 sites bound the returned Arc into
      a named local + take `.read().unwrap()` /
      `.write().unwrap()`. *Shipped fix-up 3/5 commit 6e1e10db.*
- [x] `Tensor_::storage` becomes `Option<Arc<RwLock<Storage>>>`;
      "exactly one of `storage`, `link` is `Some`" invariant
      enforced at construction. `from_storage` produces
      legacy-mode tensors; new `from_link` constructor produces
      node-handle tensors (reads dtype/device/shape from the
      slot, errors cleanly when the slot is unpopulated). *Shipped
      fix-up 4/5 commit f0f0df1d.*
- [x] Smoke test: construct a node-handle Tensor end-to-end and
      verify `realized_storage()` returns the slot Arc.
      *Shipped 2026-05-02 commit 42a94c74; rewritten in fix-up 4/5
      to use the `from_link` constructor.*
- [x] Multi-device parity: parametric helper + gated CUDA/Metal
      tests verifying the slot mechanism preserves device
      identity. *Shipped 2026-05-02 commit ae87d92c — CUDA
      verified live on RTX 4070. Vulkan parity holds by
      construction (same trait inheritance) — re-enable an
      explicit Vulkan test once a device-construction shortcut
      for tests is added.*
- [x] Document the new model in `GUIDE.md` (architecture seam)
      and `PATTERNS.md` (runnable example). *Initial commit
      530cd371; fix-up 5/5 commit 56e109ca rewrote both to match
      the corrected architecture.*

Follow-on (post-G, ahead of CE):

- [x] **G2. Move `Op::Const` payload into graph-Storage.**
      Shipped 2026-05-02 as a 3-step sequence:
      1. Substrate (commit a4b836c9): `Op::Const(Option<ConstData>)`
         wraps the legacy host payload alongside a new slot-only
         `Op::Const(None)` mode; `Tensor::from_storage` primitive
         for slot-only construction; slot-first dispatch in
         fuel-graph-executor / fuel-graph-cpu / fuel-reference-backend's
         realize loops.
      2. Sweep (commit f0062c4f): public factories take an explicit
         `&Device` (`fuel_graph::Tensor` takes `&Arc<dyn DynBackendDevice>`,
         `fuel_core::LazyTensor` takes `&Device`); `const_*_like`
         methods stay 2-arg and derive device from `self`'s graph.
         ~700 callsites swept across ~50 files.
      3. Cleanup (commit a00e6738): `ConstData` enum dropped;
         `Op::Const` becomes a unit variant; gradient seeder
         `build_filled_const` slot-populates via
         `pick_device_from_graph`; `eval_const` arms in every
         backend become `unreachable!`; const_pool restored
         with `Weak<RwLock<Storage>>` liveness witness so slot
         pointer recycling across realize calls (fresh-graph-per-
         training-step pattern) can't cause stale cache hits.
         fuel-cuda-backend gained `try_adopt_slot_cuda` slot-first
         dispatch in all three realize loops.

Estimated scope: 1-2 focused weeks for G itself; G2 was about a
week (estimated half a week, plus the const_pool liveness fix and
the cuda slot-first wiring that surfaced during the work).

**F. Declared layout contracts and layout-tracking optimizer pass.**

*Placeholder — needs design-pass planning before sub-tasks are
written. Listed here so the idea isn't lost.*

The high-level idea: each op-on-each-backend declares the input
`Layout`s it can accept and the output `Layout` it produces. The
graph optimizer reads those contracts, matches consumer-input
against producer-output, and either inserts layout-conversion
ops where there's a mismatch or selects op variants whose
contract consumes the existing layout. Same kind of reasoning
XLA does for HLO sharding/layout, MLIR's linalg dialect does for
layout assignment, and cuDNN's plan-graph does for tensor format
selection.

Open design questions to resolve before this becomes actionable:

- Layout space is bigger than contiguous-vs-strided. NHWC vs
  NCHW for conv, blocked formats (cuDNN's `nchw_vect_c`, NHWC8),
  interleaved quant block layouts (Q4_0's 32-element packing
  isn't expressible as strides at all). Which axes of layout
  space does the optimizer reason about? A small closed set of
  named layouts plus an `Any` fallback for stride-aware kernels
  is the pragmatic answer, but the choice needs to be made
  explicitly.
- Most of Fuel's ops today implicitly accept any stride-aware
  Layout — their contract is `Any → Any`, which carries no
  signal for the optimizer. The ops where layout-contracts pay
  off are the rigid ones: cuBLAS gemm's lda/ldb/ldc rules, conv
  kernel format preferences, Q4_0 matmul's block-aligned input.
  Maybe 15-20 ops out of ~140. The cost-benefit of declaring
  contracts on the rest is real and needs a deliberate answer.
- Multi-device interaction: layout-on-device-A doesn't mean the
  same thing as layout-on-device-B. Per-device layout reasoning
  vs unified abstract layouts is itself a design choice.
- Interaction with G's storage slots: each slot already records
  a realized Layout. F's contract-reasoning operates on this
  metadata. F is gated on G having shipped.

Estimated scope: deferred — depends entirely on the design
choices above. Likely 2-4 weeks once scoped.

#### Sequencing

Revised after the 2026-05-02 design pass (see work item A
preamble for context). The original sequence put A first as a
cheap independent ship; closer inspection showed A's "loaders
fissioning" framing required a parse/construct seam workaround
because of the Tensor-coupling cycle. Re-framing A as
`fuel-formats` (parser layer) plus A2 (loaders finalization
after E) lets the parser layer ship now without compromise and
defers the Tensor-coupled file-transport extraction to where it
is mechanical.

Order (revised 2026-05-02 after G was added):

1. **A (`fuel-formats`)** ✅ shipped 2026-05-02. Parallel-safe with
   B because it touches the byte-decode bodies of loader files,
   not the Tensor construction call sites B is rewriting. No
   `Tensor` / `Storage` / `Device` coupling.
2. **B1 (`.realize()` stubs)** ✅ shipped 2026-05-02 (commit
   a8e192ff). Identity-clone stubs that stabilise the public API
   so downstream code can opt into the lazy idiom early.
3. **G (Graph owns Storage)** — architectural prerequisite for
   the rest of B. Lands the `(graph, NodeId)`-handle Tensor model
   and moves Storage ownership into the Graph. 1-2 focused weeks.
4. **G2 (`Op::Const` payload moves into graph-Storage)** ✅ shipped
   2026-05-02 across commits a4b836c9 / f0062c4f / a00e6738.
   Public factories (`Tensor::from_f32`, etc.) take an explicit
   `&Device`, slot-populate at construction, and emit `Op::Const`
   as a unit variant. ConstData is gone.
5. **B2-B6 (factories, op methods, force-value entry points,
   downstream migration, drop eager dispatch)** — much simpler
   on top of G. Each B sub-phase is an independently shippable
   landing; B3 ships op-family-by-op-family.
6. **C and E together** — once `Tensor_`'s `op: BackpropOp` and
   `is_variable` come out (C), `Tensor` is small enough that
   extracting it to `fuel-tensor` (E) is the same motion. C
   cannot finish without touching every site E needs to touch,
   and doing them together avoids a transitional state where
   `Tensor_` is half-shrunken.
7. **A2 (`fuel-loaders` finalization)** — afternoon of work once
   E lands. File-transport wrappers move from `fuel-core` to
   `fuel-loaders`; `fuel-core` re-exports for back-compat; no
   parse/construct seam to maintain.
8. **D (inplace-rewrite optimizer)** — depends on C+E producing
   the unified forward+backward graph to do liveness on.
9. **F (declared layout contracts)** — deferred, awaiting design
   pass. Gated on G being in place because F operates on the
   per-slot Layout metadata G introduces.

B and C/E are tightly coupled but should ship as separate
landings rather than one mega-PR — B first (so the eager-vs-
lazy duality is collapsed before autograd refactor), then CE.

Total estimated scope: A is one week (parser extraction is
self-contained); G is 1-2 weeks plus G2 about a week (estimated
half a week, plus the const_pool liveness fix and the cuda slot-
first wiring that surfaced during the work); B2-B6 is
two-to-three weeks of factory + op-method migration on top of
G; CE together is six-to-eight weeks (every op constructor
touched, plus mechanical Tensor extraction); A2 is half a day;
D is one-to-two weeks of optimizer-pass work; F is deferred.
Roughly three months end-to-end excluding F.

#### Success criteria

- `fuel-core` no longer carries byte-level format-parsing code;
  parser surface lives in `fuel-formats` and operates on
  arbitrary `Read` / `&[u8]` sources. File-transport wrappers
  live in `fuel-loaders` (post-E); `fuel-core` re-exports
  remain for back-compat.
- A streaming weight-load smoke test reads a safetensors file
  through `fuel-formats` directly off a network-style
  `impl Read` (e.g., `Cursor<&[u8]>`) without touching the
  filesystem — proves the parser surface is genuinely
  transport-independent.
- A new `fuel-tensor`-only program (no autograd, no loaders)
  builds and runs a forward pass with measurably smaller compile
  times than the current `fuel-core`-equivalent.
- `Tensor` has no `op: BackpropOp` field; inference paths show
  measurable reduction in per-op overhead and per-tensor memory
  vs. the pre-7.5 baseline.
- A training program written against `fuel-autograd` produces
  bit-equivalent gradients to the current in-tree autograd on
  the regression suite (CPU + at least one accelerator).
- Higher-order gradient test (`grad(grad(f))`) passes end-to-end.
- Inplace ops work correctly in differentiated regions without
  user intervention; benchmark suite shows inference paths
  picking up inplace wins on at least the activation functions
  in the Phase 6 anchor suite.
- Eager mode is removed; `.realize()` is the documented
  materialisation point; `GUIDE.md` and `PATTERNS.md` reflect
  the lazy-only idiom.

#### Honest caveats

This is the largest single phase since Phase 6 itself. The
biggest risk is C (autograd refactor): every op constructor in
the codebase is touched and any subtle change to gradient
semantics shows up as a training divergence that's expensive
to debug. Mitigation: bit-equivalence testing against the
pre-7.5 autograd at every step, on the CPU reference backend
where determinism is highest. The second risk is B: dropping
eager mode is a user-visible API change even if signatures
remain the same — anyone relying on "matmul executes now"
semantics has to learn `.realize()`. Mitigation: documentation,
an opt-in `FUEL_EAGER=1` env-flag during transition that forces
`.realize()` after every op, and a deprecation cycle before the
flag is removed.

The third risk is G (Graph owns Storage): every read path that
touches `tensor.storage()` today changes shape. Mitigation is
the parallel-introduction-then-drop tactic — old-mode and
node-handle Tensors coexist throughout the migration window so
the tree compiles green at every step, and a single mode-agnostic
read API (`tensor.realized_storage()`) gives consumers a stable
seam. The residency machinery (`Op::Release`,
`ResidencyEvictionRule`, `evict_from_candidates`) is purely
NodeId-based and rides the change without code edits, which is
a meaningful piece of evidence that the architectural cut is in
the right place.

This phase should not be attempted concurrently with Phase 8
(FlashAttention) or Phase 8.5 (sparsity); both add new
kernels/ops and would have to absorb the autograd-rewrite mid-
flight. Phase 9 (agentic hooks) is gated on a real consumer
and not in conflict.

---

### Phase 7.6 — FusedOpRegistry: open registry for fused ops, closed enum for primitives

*Architectural refactor that adds an open registry of fused ops accessible
through one arm of the closed `Op` enum (`Op::Fused(id, params)`). Enables
cross-backend cost-based placement and is the substrate the cost-aware
scheduler will consume. Touches every backend's kernel registration and
every consumer that pattern-matches on `Op`. ~2-3 weeks of focused work
against the architecture v1.0 design.*

**Architecture-set anchor**: this phase implements the commitments in
[`docs/architecture/03-ir.md`](docs/architecture/03-ir.md) (Op-shape A, fused-op registry, pre-resolved KernelRef),
[`docs/architecture/04-optimization.md`](docs/architecture/04-optimization.md) (per-decision-point alternatives, OptimizationMap),
and [`docs/architecture/05-backend-contract.md`](docs/architecture/05-backend-contract.md) (per-kernel `PrecisionGuarantee`).

**Phase design doc**: [`docs/fused-op-registry.md`](docs/fused-op-registry.md)
(refreshed against architecture v1.0). The design doc carries implementation
detail; this ROADMAP entry carries the work plan.

#### Why Phase 7.6 exists

PR 3 (rule registry) and PR 3.5 (Op::ReduceMaxTo, Unsqueeze, ReduceMaxToBackward)
surfaced a structural question: every fused op fuel adds today requires a new
`Op` variant + executor arms in every backend + autograd entry + op_short_name +
op_key + binding-table registration + hand-written lowering/fusion rules. Each
fused op multiplies the plumbing cost; the Op enum becomes the bottleneck.

The architectural answer (per [03-ir](docs/architecture/03-ir.md)): the `Op` enum has primitive variants
plus exactly one `Op::Fused(id, params)` arm. The `id` indexes a registry of
fused ops; the registry is open at build-time, frozen at startup. Adding a new
fused op is a registry entry + a kernel function — no `Op` enum edit, no
autograd edit, no per-backend executor arm.

The cross-backend payoff: the cost-aware scheduler (downstream phase work)
needs every backend's fusion catalog visible *before* placement decisions to
compare "matmul+bias+relu costs X on CUDA fused, Y on Vulkan unfused." A
registry is the natural shape for that visibility; backend-internal fusion
(XLA's model) couldn't satisfy this.

#### Architectural decisions (anchored to v1.0)

These decisions live in the architecture set; this phase implements them.

- **Op-shape A**: closed `Op` enum with primitive variants + one `Op::Fused(id, params)` arm. No separate `NodeKind` discriminator. Per [03-ir §How nodes carry their op identity](docs/architecture/03-ir.md#how-nodes-carry-their-op-identity).
- **Pre-resolved `KernelRef` per node** at planning time. The binding table is a planning-time catalog; the executor calls function pointers directly. Resolves audit Q-A. Per [03-ir §The optimized form](docs/architecture/03-ir.md#the-optimized-form-top-n-routes-with-pre-resolved-kernels).
- **Lazy KernelRef resolution** at decision-point pick time + mmap'd cache. Per [11-persistence §Re-resolution on use](docs/architecture/11-persistence.md#re-resolution-on-use-lazy-not-at-load).
- **Fused-op registry crate location**: metadata in fuel-graph; `BackendImpl` payload (which carries `KernelRef`) in fuel-storage. Avoids a circular dependency.
- **Per-kernel `PrecisionGuarantee` structure** on the registration surface, replacing the OracleGrade flag concept. Per [05-backend-contract §Per-kernel precision guarantees](docs/architecture/05-backend-contract.md#per-kernel-precision-guarantees).
- **PR 3's hand-written rules become auto-generated** from `FusedOpEntry.decompose` + `FusedOpEntry.pattern`. Hand-written remains an escape hatch.

#### Sub-tasks (revised against architecture v1.0)

- [x] **Step 1: registry skeleton.** *Shipped per [`project_phase_7_6_step_3_shipped.md`](MEMORY.md). `FusedOpId`, `FusedOpEntry`, `FusedOpParams`, `FusedOpRegistry` in fuel-graph; `BackendImpl`, `PrecisionGuarantee` in fuel-storage. See [`docs/fused-op-registry.md`](docs/fused-op-registry.md) v3 for the crate-split detail.*
- [x] **Step 2: extend `Op` enum with `Op::Fused(FusedOpId, FusedOpParams)` arm.** *Shipped (same memory). Coexists with legacy fused-op variants during migration; `op_short_name`/`op_key` handle the new arm.*
- [x] **Step 3: migrate first fused op (SoftmaxLastDim) end-to-end.** *Shipped (same memory). Auto-generated `LoweringRule` + `FusionRule` from the registry entry; PR 3's hand-written rules retired; live CUDA equivalence test green.*
- [~] **Step 4: migrate remaining 12 fused ops.** *Partial: FusedLinear shipped via [`project_phase_7_6_fused_linear_and_step_6_shipped.md`](MEMORY.md). RmsNormLastDim, LayerNormLastDim, Rope, Conv2D, ConvTranspose2D, FlashAttn, PagedAttn, QMatMul, plus the 4 backward-helper fused ops remain. Each is its own commit; ~half-day per op.*
- [ ] **Step 5: drop the per-fused-op `Op` variants.** Once nothing emits `Op::SoftmaxLastDim` etc., remove them from the enum. Mechanical: update `op_short_name`, `op_key`, autograd's match arms. Gated on Step 4.
- [x] **Step 6: backend registrations adopt `BackendImpl` shape.** *Shipped per [`project_phase_7_6_fused_linear_and_step_6_shipped.md`](MEMORY.md). `register_fused!` macro + `default_kernel_registry` populate `FusedOpEntry` → `BackendImpl` mapping; 4 CPU `FusedLinear` impls registered.*
- [ ] **Step 7: populate `PrecisionGuarantee` per registered kernel.** Bit_stable kernels (the always-built backend's coverage commitment) get the `bit_stable_on_same_hardware: true` flag; others declare what they can characterize. Per [05-backend-contract §Per-kernel precision guarantees](docs/architecture/05-backend-contract.md#per-kernel-precision-guarantees).
- [ ] **Step 8: populate cost estimates.** Each `BackendImpl`'s `cost` function gets a real implementation per backend. Initial: FLOP-counting + bandwidth model. Static-only for v1; community-aggregated empirical refinement (per [04-optimization §Cost model](docs/architecture/04-optimization.md#cost-model-static-annotations-refined-by-empirical-judge-data-accounting-for-parallelism)) follows when telemetry pipeline lands.
- [~] **Step 9: binding-table planning-time refactor.** *Steps 9a + 9b Track A shipped per [`project_phase_7_6_step_4_in_progress.md`](MEMORY.md). 9a: `KernelBindingTable` multi-impl alternatives per `(OpKind, dtypes, BackendId)` (commit `b9828f13`). 9b Track A: `NodeKernelBinding` + `compile_plan` + `resolve_kernel` + `TolerancePolicy` (commits `d60febc7`, `1251bb73`, `5b9f7ca3`, `700bb948`). Step 9c (typed-storage executor migration → see [Phase 7.6 step 9c](#phase-76-step-9c--typed-storage-retirement) below) is the next gate.*
- [ ] **Step 10: comparison family** (Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual) added to `Op` as primitive variants. Bit-exact equality on floats; non-differentiable backward (panic stub, ArgMaxDim precedent). Lands here because primitive-set completion belongs with this architectural cleanup. **Note**: also tracked in the [`fill-op-primitive-set.md`](docs/session-prompts/fill-op-primitive-set.md) session prompt which audits the broader missing-primitive surface.

#### Success criteria

- `Op` enum is primitive variants + one `Op::Fused(id, params)` arm. ~85 primitive variants. No per-fused-op variants.
- `FusedOpRegistry` populated with 13-14 entries (the migrated fused ops). Adding a new fused op is one entry + one kernel function.
- PR 3's hand-written SoftmaxLastDim rules deleted; auto-generated rules from registry entries produce equivalent behavior. Round-trip identity test still passes.
- Live CUDA equivalence test (`cuda_executor_matches_cpu_on_softmax_via_lowering`) still passes through the registry-dispatched path.
- `cost_estimate(SOFTMAX_LAST_DIM, [B, N, M], CUDA)` query returns a plausible estimate via the registry surface.
- Every registered kernel carries a `PrecisionGuarantee`; the always-built backend's coverage commitment (one `bit_stable` kernel per primitive op) is testable as a CI lint.
- All existing tests green throughout the migration. CSE / op_key handles `Op::Fused(id, params)` correctly.
- ROADMAP and architecture decisions-log updated post-migration.

#### Honest caveats

This refactor touches the deepest layer of fuel. Backends, autograd, executor, op_short_name, op_key, dispatch wrappers, CSE — all match on `Op`. The migration uses parallel-introduction-then-drop: existing variants and the new `Op::Fused` arm coexist through the migration window; per-fused-op variants drop in step 5. Each fused-op migration in step 4 is independently shippable.

The architecture's pre-resolved KernelRef commitment (step 9) is a meaningful refactor on its own — it changes where the binding table is consulted (planning time, not execution time). Lands in this phase because Phase 7.6's executor work is the natural place to also restructure the executor's per-node dispatch path.

Cost estimates registered with `BackendImpl`s are advisory; the cost-aware scheduler that consumes them is downstream. Initial cost models can be coarse; the community-aggregated empirical refinement framework (per [11-persistence §Cache generation and distribution](docs/architecture/11-persistence.md#cache-generation-and-distribution)) tightens them over time.

This phase should not run concurrently with Phase 8 (FlashAttention) or Phase 8.5 (sparsity); both add new fused ops mid-flight that would have to absorb the registry refactor. Phase 7.5 work items B/C/E (Tensor/autograd/fission refactor) are orthogonal — they can run before, after, or in parallel.

#### Phase 7.6 step 9c — typed-storage retirement

*Audit + multi-session plan to swap the legacy `GraphExecutor<B>` (typed-storage shape) for `PipelinedExecutor` (dispatch-erased shape) across all callers.*

**Full audit**: [`project_phase_7_6_step_9c_parity_audit.md`](.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md) (memory). 242 call sites across 34 files; ~12 PipelinedExecutor feature gaps; estimated 6-8 sessions / 30-50 commits.

**Status (2026-05-22)**:

- ✅ Phase A — multi-target realize (`realize_many` shipped 2026-05-19 in commit `c5ed169a`).
- ✅ Phase B — side-effect roots + destructive-input cleanup (shipped 2026-05-19 in commits `db89a283` + `f9ad93d0`).
- ✅ Phase C — CPU fallback shape decided: fail-fast (binding-table `lookup` returns `None` → typed error). Documented 2026-05-19.
- ✅ Phase D — optimization + rule-registry plumb-through: caller composes. Documented 2026-05-19.
- ✅ Phase E.1 + E.2 — fuel-core `pipelined_bridge` module shipped 2026-05-19 in commit `32d712f7`. `Tensor::realize_f32` migrated for CPU + CUDA; CUDA executor gained output allocation, auto-contiguize, layout-vs-storage-bytes mismatch handling.
- ✅ Phase E.3.0 — `InferenceContext` + `KvCache` primitives shipped 2026-05-20 in commit `a405e7c0`.
- ✅ Vulkan runtime `Device` wiring shipped 2026-05-22 (this session): `VulkanBackendDevice` + bridge module + parity test against CPU through `forward_with_kv_context`.
- ⏳ Phase E.3 remainder: pre-allocated buffers + `Op::WriteSlice` (E.3.2), `forward_with_cache_on` migration (E.3.3), `generate_*` + spec decoding (E.3.4).
- ⏳ Phase E.4: train.rs + factories.rs migration.
- ⏳ Phase F: backend-crate cleanup (remove `GraphExecutor::new` from backend crates' tests + examples).
- ⏳ Phase G: `fuel-graph-router` migration. `GraphBackend` retain-vs-retire decision.
- ⏳ Phase H: retire `fuel-graph-executor` crate (or keep as thin shim for Judge profiling).

#### Bridge-retirement trajectory post-9c

The Vulkan Device-wiring shipped this session uses a bridge pattern: `VulkanBackendDevice` wraps `Arc<VulkanBackend>` and implements `DynBackendDevice`, but the storage-returning `*_dyn` methods stub to errors because Vulkan storage lives on the byte-shape `VulkanStorageBytes` substrate, not on `DynBackendStorage`. The bridge is mirror-shaped across CUDA + CPU + Vulkan and follows the established pattern, but is not the architecture v1.0 destination.

The destination per [01-identity](docs/architecture/01-identity.md) + [05-backend-contract](docs/architecture/05-backend-contract.md) is: `Device` is a thin tag, storage allocation + transfer happen via graph-level `Op::Alloc` + `Op::Copy` primitives that the optimizer plans and the executor dispatches through the binding-table, and `DynBackendStorage` retires entirely.

Path from bridge to destination (each phase ~1 session, in dependency order):

1. ✅ **D2H through `Op::Copy`** — *shipped 2026-05-22.* `OpKind::Copy` registered in the binding table; CPU/CUDA/Vulkan each provide a `copy_to_cpu` wrapper at the canonical `[dt, dt]` key. `realize_one_as<T>` / `realize_many_as<T>` splice `Op::Copy { target: Cpu }` at every realize root (CPU + GPU) so the spliced node's `auto_contiguize` honors view-op layouts uniformly — the pre-9c "ignore strides, return full source bytes" CPU bug is fixed alongside. Executor uses a dedicated `WorkItemKind::Copy { target_location }` arm so output allocation goes on the target while the kernel lookup keys on the source backend. **Deleted**: the per-variant match in `BackendStorage::read_to_cpu_bytes` (including the Vulkan branch from `7a95001a`) — first deletion of bridge code from `7a95001a`. See [`project_phase_7_6_step_9c_parity_audit.md`](.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_6_step_9c_parity_audit.md) "Bridge-retirement Phase 2 shipped" follow-up.
2. **H2D + zero-alloc through `Op::Alloc` + `Op::Copy`** — split into Phase 3a (zero-alloc) + Phase 3b (H2D Const upload).
   - **Phase 3a — `Op::Alloc` (uninit) + `Op::ZeroFill` (explicit fill)** ✅ *shipped 2026-05-22 (initial) + 2026-05-23 (follow-up refactor).* `Op::Alloc { target: DeviceLocation }` is a new graph primitive (0 inputs, source op like `Op::Const`) producing **uninit** device memory; `Op::ZeroFill` is a destructive in-place fill primitive (paired in current callers). The executor's `WorkItemKind::Alloc` arm dispatches per-backend (CUDA `alloc_uninit` via baracuda alpha.30's raw `cuMemAlloc`; Vulkan `alloc_bytes_handle` truly uninit). The executor's `WorkItemKind::ZeroFill` arm calls baracuda `DeviceBuffer::zero_async` (cuMemsetD8Async, in-place) on CUDA and `VulkanBackend::fill_bytes_zero` (vkCmdFillBuffer, device-side, ~2× the bandwidth of the old host-staged zeros) on Vulkan. `KvCache::with_capacity` emits 2N pairs of `Op::Alloc → Op::ZeroFill`. **Deleted**: `alloc_zeroed_on` (50-line per-`DeviceLocation` match in `fuel-core/src/inference_context.rs`). **Residual**: `device_seed_storage` (~15-line 0-byte-seed allocator per backend). **baracuda bumped to alpha.30** (workspace-wide) to pick up `DeviceBuffer::zero_async`.
   - **Phase 3b — H2D Const upload through `Op::Copy { target: device }`** ✅ *shipped 2026-05-23.* Extended `copy_from_cpu_wrapper` (renamed from `copy_to_cpu_cpu_wrapper`) to switch on output variant — CPU→CPU memcpy, CPU→CUDA via `CudaStorageBytes::write_from_host`, CPU→Vulkan via `VulkanBackend::write_bytes` (new helper: staging buffer + `vkCmdCopyBuffer`). Executor's `WorkItemKind::Copy` arm extended to allocate non-CPU output (Cuda via `alloc_uninit`, Vulkan via `alloc_bytes_handle` uninit). `build_const_cache` (in `pipelined_bridge`) for non-CPU realizes now builds a transient graph with `Op::Const → Op::Copy { target: device }` pairs (one per user Const) plus a device-handle anchor, realizes via `PipelinedExecutor::realize_many`, and writes results back to the user StorageCache at the original Const NodeIds. The transient graph isn't observable from the user's graph. **Deleted**: `upload_host_buffer` (~60-line per-`DeviceLocation` match in `fuel-core/src/pipelined_bridge.rs`) — third deletion of bridge code from `7a95001a`.
3. **`*_dyn` storage methods removed from `DynBackendDevice` trait** once nothing calls them from byte-storage callers. Trait shrinks to advertisement-only (`location_dyn`, `same_device_dyn`, `synchronize_dyn`, `set_seed_dyn`, `get_current_seed_dyn`, `as_any`, `supports_bf16`, `as_quantized_kernels`). Gated on the typed-storage retirement (audit Phases F + H) being complete. **Deletes**: ~6 stub error-bodies per backend's `DynBackendDevice` impl, including the stubs in `VulkanBackendDevice`.
4. **Trait renamed** (`DynBackendDevice` → `BackendAdvertise` or merge into [`BackendCapabilityProvider`](docs/architecture/05-backend-contract.md#static-capability-advertisement-registered-at-startup)). Doc-only.
5. **`Device` becomes a tag, backend handles move to a registry**. `Device { backend_id, location }` is a pure value type. Backend handles live in a process-wide registry consulted by `Device::synchronize` / `Device::set_seed`. **Deletes**: `From<VulkanBackend> for Device` + `as_device(&Device) -> Arc<VulkanBackend>` helper in `fuel-core/src/vulkan_backend.rs`; matching CUDA/CPU/Metal equivalents; the `VulkanBackendDevice` newtype itself. *The bridge built this session is gone at this point.*
6. **`DynBackendStorage` trait retired entirely** once all callers migrate to byte-storage. Significant cleanup in `fuel-cpu-backend/src/dyn_impl.rs` (1365 LOC), `fuel-cuda-backend/src/dyn_impl.rs` (587 LOC), `fuel-metal-backend/src/dyn_impl.rs` (503 LOC).
7. **Router migration** (audit Phase G). `fuel-graph-router` consumes `BackendCapabilities` from the registry; no `Arc<dyn DynBackendDevice>` dependency.

**Architecture-alignment check**: every step makes [01-identity](docs/architecture/01-identity.md#how-this-identity-is-enforced) more true (decisions move to the DAG-level optimizer; cost data flows through binding-table). No step requires revisiting an architecture v1.0 commitment — this is implementation catch-up to the architecture, which is the expected shape since the architecture was drafted ahead.

---

### Phase 8 — FlashAttention tiered implementation

*Affects only the Backends/Kernels layer. Gated on two external
prerequisites landing first: the new Vulkane release (adds
external-memory / handle-export primitives among other things) and
Baracuda (Fuel-owned CUDA FFI crate replacing cudarc, exposing
functionality cudarc omits). Neither is ready as of this entry;
do not start Phase 8 work until both are integrated into Fuel.*

#### Why Phase 8 exists

FlashAttention reduces attention from O(N²) HBM traffic to O(N·d)
via tile-based online softmax. The math is backend-agnostic; the
kernel implementation is decidedly not. v3 and v4 in particular lean
hard on vendor-specific hardware (Hopper TMA + WGMMA, Blackwell
cooperative TMA + 5th-gen tensor cores). A direct port of the
upstream Dao-AILab kernels would be CUDA-only and leave Vulkan users
stuck on naive attention. The right shape is a tiered implementation
that shares the algorithm across backends and specializes only where
the perf justifies it.

A further question worth investigating within this phase: v4's
published speedups are partly algorithmic (warp specialization,
deeper pipeline depth, block-scaled low-precision) and partly raw
matrix-unit throughput. The algorithmic concepts extract cleanly and
can be re-expressed for non-Blackwell architectures; the throughput
component is hardware-gated. Tier 4 below is the place to ask "which
v4 ideas buy us something on Ampere / RDNA3 / Apple M / Intel Arc
and which don't."

#### Tier 0 — Audit existing FlashAttention crates

Fuel's workspace already contains `fuel-flash-attn` and
`fuel-flash-attn-v3` crates (see the Backends/Kernels layer box
above). Before writing anything new, determine what they contain,
which backends they target, and whether they can be refactored into
the tiered structure below.

- [x] Survey `fuel-flash-attn` — list the op surface, target
      backend(s), and parity-test coverage. *Shipped: see
      `docs/phase8_tier0_audit.md`.*
- [x] Survey `fuel-flash-attn-v3` — same. *Shipped in same audit.*
- [x] Decide whether Tier 2/3/4 below refactor these crates in place
      or supersede them. Document the decision. *Decision: rename
      to `fuel-flash-attn-cuda` / `fuel-flash-attn-v3-cuda`, extract
      `-sys` siblings to break the dep cycle, refactor in place.*

#### Tier 1 — CPU reference implementation

- [x] Pure-Rust FlashAttention forward in `fuel-flash-attn` (or
      wherever the audit in Tier 0 lands it). ~100 LOC. Slow by
      design; its job is to be the correctness oracle for every
      other tier. *Shipped as `fuel_reference_backend::attention::
      attention_flash` (~270 LOC; bigger than 100 because it also
      handles GQA, causal mask, sliding window, ALiBi, softcap —
      same surface the kernels target).*
- [x] Backward pass via recomputation — same approach as the
      upstream reference, matches the tier-2/3 kernels' expectations.
      *Shipped as `attention_flash_backward`.*
- [x] Parity tests against a naive-attention reference on small
      shapes (seq ≤ 256, head_dim ≤ 128, batch × heads ≤ 8). Tight
      tolerance (1e-5 in f32) — this tier has no excuse for drift.
      *Shipped: 7 parity tests + 1 finite-difference gradcheck in
      `fuel-reference-backend/tests/attention.rs`.*

#### Tier 2 — Portable GPU implementation in Slang

- [x] Single Slang source for FlashAttention v2 (tile-based,
      workgroup-parallel, online softmax, no warp specialization).
      Compile to SPIR-V for Vulkan; Slang's experimental CUDA PTX
      backend is a free bonus if it works, not a requirement.
      *Shipped as `fuel-kernels-source/kernels/flash_attention.slang`
      → `fuel-vulkan-kernels/spv/flash_attention.spv`.*
- [~] Targets VK_KHR_cooperative_matrix when the device advertises
      it; falls back to plain workgroup-shared-memory tiling on
      devices that don't. *Plain tiling shipped; coop_matrix path
      is a follow-up.*
- [x] Parity tests against Tier 1 across a matrix of
      (batch, heads, seq, head_dim, dtype) shapes. Start narrow
      (f32, contiguous, seq ≤ 1024) and widen once green.
      *Shipped: 4 parity tests in `fuel-core/tests/flash_attn_vulkan.rs`,
      green on RTX 4070 within 5e-4 of reference.*
- [ ] Performance notes: record ms/token on a handful of anchor
      shapes on the dev rig's Vulkan iGPU and on an RTX 4070. This
      tier's ceiling is roughly FA v2 perf on Hopper+ — v3/v4
      pipelining depth needs primitives Slang can't abstract.

#### Tier 3 — Hand-tuned backend kernels (opt-in per arch)

Only write these when Tier 2 benchmarks show meaningful perf left on
the table for a specific architecture.

- [x] **CUDA / Ampere (sm80)**: Dao-AILab FA-v2 kernels via
      `fuel-flash-attn-cuda-sys`. Wired through
      `CudaBackend::flash_attn` behind the `flash-attn` Cargo
      feature; validated on RTX 4070 within F16 precision (max
      abs 4.2e-5) of `attention_naive`.
- [x] **CUDA / Hopper (sm90)**: Dao-AILab FA-v3 kernels via
      `fuel-flash-attn-v3-cuda-sys`. Symbol renamed to `run_mha_v3`
      so both -sys crates link together cleanly. Behind the
      `flash-attn-v3` Cargo feature; dispatch chain prefers v3 and
      falls back to v2 on Err. Rust wiring complete; live-Hopper
      validation deferred to first user with sm90a hardware.
- [ ] **CUDA / Hopper+**: FA v2 or v3-equivalent using CUTLASS or
      hand-written PTX (would supersede the above port-only Tier 3
      entries with Fuel-native kernels). Requires Baracuda exposing
      `CUtensorMap`/`CUwgmma`/`cuTensorMemAcc` primitives.
- [ ] **AMD / RDNA3+**: WMMA + LDS prefetch, wavefront-specialized
      pipeline. Blocked on whatever Rust FFI we settle on for ROCm.
- [ ] **Apple Silicon**: simdgroup_matrix + AMX via Metal. Likely
      lives in `fuel-metal-kernels`.
- [ ] **Intel Arc / Xe-HPG+**: XMX via SYCL or a direct Level Zero
      binding. Lowest priority; revisit if an anchor model
      materially benefits.

Each arch lands independently. The Router + Tier 2 fallback means
users on untuned hardware still get FlashAttention, just not the
peak form.

#### Tier 4 — Extract v4 concepts for non-Blackwell architectures

This tier is research-flavoured and should be sized AFTER Tiers 1-3
are stable. Per-arch experiments to validate which v4 ideas
transfer:

- [ ] Deeper pipeline depth (3-4 stages vs v2's 2) on Ampere using
      `cp.async` + `__syncwarp()` — measure vs Tier 3 CUDA baseline.
- [ ] Warp specialization (producer / consumer split) on RDNA3 LDS
      prefetch — measure vs Tier 3 AMD baseline.
- [ ] Block-scaled low-precision path (MXFP4/MXFP6) — format is
      generic, so this lands as a Tier 2 Slang extension once the
      dtype plumbing exists in Fuel.

Honest caveat: on hardware with weaker matrix units the algorithmic
gains will not close the wall-clock gap with Blackwell; v4's
headline numbers are partly algorithm and partly hardware.

#### Success criteria for Phase 8

- Every backend Fuel targets runs FlashAttention (at minimum
  Tier 2-quality) on every model in the Phase 6 anchor suite.
- Parity with the CPU reference is verified per backend and included
  in the regression gate.
- The tiered structure is documented well enough that a contributor
  with a new backend (SYCL, WebGPU, whatever) can plug in at Tier 2
  without having to touch Tiers 1, 3, or 4.

---

### Phase 8.5 — Dynamic activation sparsity (research-flavoured)

*Affects only the Backends/Kernels and IR layers. Not urgent.
Research effort with model-specific calibration; queue after
Phase 6/7/8 are stable. Primarily benefits CPU inference; GPU
gains are model-dependent.*

#### Why Phase 8.5 exists

Older transformer FFN layers (ReLU-MLP, original Llama / OPT /
BLOOM) produce highly sparse intermediate activations — typically
70-90% of values are zero or below a meaningful threshold. Naive
dense compute on the down-projection wastes that work. Modern
SwiGLU/GeGLU models still have ~30-50% effective sparsity. The
**gather-compute-scatter** technique — extract active rows into a
dense subset, run the dense kernel on the subset, scatter back to
a zero-filled output — captures this win when the sparsity ratio
is high enough to amortize the predicate overhead.

The published name is *dynamic activation sparsity*. Production
references: DejaVu (Tri Dao et al., 2023), PowerInfer, TurboSparse.
DejaVu's headline result: ~2× on Llama 7B at 80% sparsity, with
negligible quality loss.

#### Where it wins (and doesn't)

- **Wins**: CPU inference (where dense GEMM is bandwidth-bound and
  sparsity directly saves work), older models with ReLU activations,
  FFN down-projection (the biggest dense matmul in the layer).
- **Marginal**: modern SwiGLU/GeGLU models — still positive, but
  smaller gain. Need higher sparsity ratios on GPU.
- **Doesn't apply**: attention output (no sparsity-producing
  activation), embedding lookup (already index-gather),
  normalization layers (no sparsity).

GPU dense GEMM is brutally hard to beat — cuBLAS sgemm hits ~98%
of peak on A100. Sparse alternatives need >70-80% real sparsity
*plus* cheap predicate overhead before they win on GPU. The
target hardware reality is: this technique should dominate on CPU
backends (AOCL, MKL, OpenBLAS) and earn its keep on GPU only for
very large FFN dims.

#### Building blocks (status today)

| Need | Status |
| --- | --- |
| Element gather (read indices → dense) | ✅ `Op::IndexSelect`, `Op::Gather` |
| Element scatter back | ✅ `Op::IndexAdd`, `Op::ScatterAdd` |
| Threshold→indices op (data-dependent count) | ❌ no `NonZero` / `Where` / `TopK` |
| Sparse-shaped matmul (variable batch dim) | ⚠️ `Op::MatMul` accepts variable `M`, but the IR's static-shape contract makes data-dependent shapes awkward |
| Gather-compute-scatter graph-rewrite pass | ❌ |

The two missing pieces are the work. The gather/scatter primitives
are already there from the lazy-graph IR.

#### Phase 8.5 work items

- [ ] **Add `Op::NonZeroIndices { threshold: f32 }`** to the IR.
      Returns `[active_count]` u32 indices. Data-dependent shape
      means the IR needs either a "ragged tensor" representation or
      a padded representation with a separate count. Padded is
      simpler; the down-projection sees padded zeros and the cost
      is bounded.
- [ ] **`opt::sparsify_ffn_down_projection`** rewrite pass.
      Detects `Activation(x) → MatMul(W_down)` (FFN down-projection)
      and rewrites to
      `IndexSelect(x, indices) → MatMul(IndexSelect(W_down, indices))
      → ScatterAdd(zeros, indices)`.
      Conservative single-consumer rule (don't fuse if the activation
      is consumed elsewhere) similar to `fuse_linear`'s.
- [ ] **Calibration harness**: per-layer sparsity profile for the
      Phase 6 anchor suite. Pick the threshold per layer per model.
      Offline, run once, stored as model metadata.
- [ ] **Quality gate**: token-equivalence vs the dense reference on
      each anchor model within a tolerance. Too-aggressive
      thresholds degrade output; this gate catches them.
- [ ] **Per-backend native sparse kernels**. Once the IR pattern
      stabilizes, hand-write CSR/dense gemv variants where the
      generic gather + dense matmul is leaving perf on the table.
      AOCL has sparse BLAS; oneMKL has IE-Sparse; cuSPARSE on CUDA;
      hand-written Slang on Vulkan.

#### Success criteria for Phase 8.5

- At least one anchor model (the ReLU-MLP archetype, e.g. original
  Llama 7B) shows ≥30% wall-clock speedup on CPU decode with
  sparsity enabled, without quality regression.
- The pass is opt-in via a flag/feature, never on by default —
  the threshold calibration is per-model.
- Modern SwiGLU models show neutral-to-positive perf (no regression
  even when the sparsity isn't there).
- Documentation explains *which* models benefit and how to find
  the threshold.

#### Honest caveats

This is a research effort, not a routine engineering task.
~2-3 weeks of focused work, ~60% of which is calibration and
benchmarking rather than IR plumbing. Not worth interrupting
current Phase 6/7/8 work for; do not pull forward.

---

### Phase 9 — Extension points for downstream agentic libraries

*Not urgent. Future-facing. Gated on a real downstream consumer
existing — i.e. when a separate agentic / cognitive-architecture
library on top of Fuel is far enough along to need these hooks. Do
not pre-build before that consumer exists.*

#### Why Phase 9 exists

A downstream "AGI library" — built on top of Fuel, not as part of
it — needs to make scheduling and execution decisions that are
*content-conditioned* (route based on tensor uncertainty, divert
work on prediction error, persist "inner monologue" state across
realize calls, distinguish self-state from sensory inputs). Today,
Fuel's surface lets you set placement hints and define custom ops,
but doesn't expose enough for an agent runtime to live cleanly above
without monkey-patching.

The right architectural cut: Fuel provides **theory-neutral
primitives** (metadata slots, runtime callbacks, persistent values).
The downstream library defines what GWT / IIT / Active-Inference /
hybrid semantics mean on top of those primitives. Fuel never ships
`enum SensoryBus` or `Op::DivertOnPredictionError` — those are the
agent library's concern, not Fuel's.

#### What this is NOT

- Fuel does not become an AGI framework. The AGI semantics live
  one layer up.
- No within-graph cycles. AGI's "inner monologue" is multi-step
  *streaming*, not a directed cyclic graph. Streaming is implemented
  by a Rust-level realize loop reading and writing persistent values
  between realize calls; the graph itself stays acyclic.
- No `Op::If` / `Op::Branch` in the graph. Conditional execution is
  Rust-level control flow above the realize loop, not a graph-level
  primitive. (Same reason every mature ML framework that experimented
  with in-graph control flow ended up regretting it.)

#### Three deliverables

**9a. Per-tensor user metadata slot.** A small additive change.
`LazyTensor::with_metadata(Arc<dyn Any + Send + Sync>)` builder +
`metadata() -> Option<&Arc<...>>` accessor. Metadata travels with
the lazy graph node, survives optimization passes (canonicalization
must preserve it), and is observable via `SchedulerRule` callbacks
and on the realized output. Fuel itself never reads or interprets
the contents — they're an opaque user payload. Sized: ~1-2 days
including survival-through-optimizer testing.

**9b. Runtime executor hooks.** Today's `SchedulerRule` runs at
plan time on the static graph. Add a sibling `RuntimeHook` trait
that fires after each node's realize:

```rust
pub trait RuntimeHook: Send + Sync {
    fn on_node_realized(
        &self,
        id: NodeId,
        op: &Op,
        output: &dyn DynBackendStorage,
    ) -> HookAction;
}

pub enum HookAction {
    Continue,                 // proceed with the planned next node
    Skip(NodeId),             // jump execution to a later node
    Inject(GraphFragment),    // splice new nodes into the plan
}
```

Lets the agent library steer execution mid-realize without
rewriting the executor. Output observation has to handle GPU
residency cleanly — either lazy host-readback on demand, or
shape-only metadata for the cheap path. Sized: 1-2 weeks for design
plus careful implementation. Real value beyond AGI: debugging,
tracing, checkpointing.

**9c. Named persistent values across realize calls.** Generalize
KVCache's "pre-populate + survive across realizes" pattern to
arbitrary user-named handles. `PersistentStore::write(name, tensor)`
plus `Graph::read_persistent(name)`. Each realize can read and write
across step boundaries; the agent library's outer realize loop
threads state by name. Covers "inner monologue," "world model
across observations," and any other multi-step state pattern.
Sized: ~1 week — KVCache is most of the work, generalization is
mostly API shape.

#### Success criteria for Phase 9

- A downstream cognitive-architecture library exists that uses 9a,
  9b, and 9c to implement at least one published theory of cognition
  (GWT, IIT, Active Inference, or a hybrid) without modifying Fuel
  source. The library's behaviour can be reasoned about purely in
  terms of those three primitives plus normal Rust control flow.
- Fuel's anti-goals (above) still hold: no AGI semantics, no theory
  of cognition, no agent abstractions inside Fuel itself.
- The hooks have at least one non-AGI consumer too (debugging,
  tracing, checkpointing) — pure single-purpose hooks tend to drift
  toward the consumer's specific needs over time, which we want to
  avoid.

#### Order of delivery

9a first (small, additive, immediately useful for diagnostic
metadata even pre-AGI). 9c next (KVCache generalization is its own
internal cleanup). 9b last (biggest investment, biggest design
risk, depends on the executor staying stable for the rest of Phase
6/7/8). Total: ~3-4 weeks across all three when a consumer needs
them.

---

### Phase 10 — Equivalence-rewrite search: device-shaped graph alternatives (research-flavoured)

*Not urgent. Future-facing. Do not pull forward. Sequenced after the
eager-retirement program completes and after the picker arc has
accumulated real Judge telemetry in production use. Builds entirely
on existing seams — rule registry, AlternativeSet, Judge, copy
insertion, SystemTopology — composing them rather than laying new
foundation. Inference-first; rewrites on the backward path are
explicitly out of scope for v1 (they can change training convergence
even inside tolerance).*

#### Why Phase 10 exists

The picker (fuel-dispatch ranker + Judge + selector chain) answers
"which kernel should run this node, on which device?" — but it takes
the graph's *shape* as given. For most op families that's correct:
GPU-optimal and CPU-optimal differ inside the kernel, not in the
graph. For a meaningful minority, the best *algorithm* differs per
target, and the graph shape should change with the placement:

- **Convolution**: im2col+GEMM vs direct vs Winograd — same math,
  opposite device preferences.
- **Attention**: FlashAttention's fusion is a GPU memory-hierarchy
  optimization; a cache-blocked CPU path may prefer the decomposed
  softmax chain it replaced.
- **Matmul reassociation**: `(AB)C` vs `A(BC)` — identical result,
  wildly different FLOPs/traffic depending on dims.
- **LoRA**: `Wx + B(Ax)` vs `(W+BA)x` depending on batch size.
- **Fuse-vs-stay-lowered**: fusion always wins today
  (`optimize_to_fixpoint` gives it the last word), but whether the
  fused form is best depends on whether the target backend has a
  good fused kernel — which the binding table already knows.

Generalizing the picker from "choose a kernel for this node" to
"choose among mathematically equivalent subgraphs for this region"
lets Fuel automatically retarget parts of a model GPU-shape ↔
CPU-shape (or NPU/TPU later) wherever the cost model — transfer
costs included — says it wins.

**Prior art**: this is graph-substitution search — TASO (SOSP '19),
PET (OSDI '21), Tensat (MLSys '21; equality saturation via the Rust
`egg` crate), and Unity (OSDI '22; joint rewrite + placement, the
closest analogue to Fuel's topology-aware version). Published wins:
1.3-3× on real models. Discovery of *new* equivalence rules is an
offline tool (TASO-style enumeration + verification), never a
realize-time activity.

#### Phase 10 building blocks (status today)

| Need | Status |
| --- | --- |
| Rewrite engine | ✅ `fuel-graph::opt` rule registry; `RuleFamily::Algebraic` exists |
| Equivalent forms in the IR | ✅ every lowering/fusion pair IS two equivalent designs — chosen globally + statically today |
| Per-device empirical cost | ✅ Judge measures real kernels per (op, dtype, size class, backend, device) |
| Transfer-aware placement cost | ✅ `insert_cross_device_copies` + SystemTopology transfer paths |
| Precision bookkeeping | ⚠️ `PrecisionGuarantee` is per-kernel; rewrite rules need per-rule deltas |
| Choice mechanism | ❌ `optimize_to_fixpoint` is one-way greedy, first-match-wins; nothing lets two equivalent forms coexist while a cost model picks |
| Equivalence rule library | ❌ one resident (cast fusion) |
| Offline rule discovery + verification | ❌ |

#### Work items (in delivery order)

- [ ] **10a — fuse-vs-lower as a picker decision.** The minimal
      version of the whole idea. Instead of fusion firing
      unconditionally, the fused form and the lowered composition
      become per-subgraph alternatives ranked by Judge data (it
      already profiles fused kernels against composed primitives).
      Reuses everything; no new theory. This alone captures the
      "backend lacks a good fused kernel" case.
- [ ] **10b — curated algebraic equivalence library.** Grow
      `RuleFamily::Algebraic` from 1 rule to dozens (hand-curated;
      this covers most of the published win). Every rule carries a
      declared precision delta feeding the existing precision-floor
      filters, and is cost-gated by Judge data instead of
      always-fire.
- [ ] **10c — search + joint placement.** Replace greedy rule
      application with search over rule applications, extracted
      against Judge costs + transfer costs jointly with device
      assignment. Escalation order: cost-gated greedy → backtracking
      over windows (TASO-scale graphs are fine) → e-graphs + ILP
      extraction (`egg`) only if backtracking hits combinatorial
      limits. Runs at `compile_plan` time with the plan cached —
      never per-realize.
- [ ] **10d — offline rule discovery.** TASO-style: enumerate small
      candidate graphs over the OpKind vocabulary, verify equivalence
      against `fuel-reference-backend` + `fuel-correctness-fixtures`
      as the oracle, emit rules into 10b's library. A build-once
      tool run per op-vocabulary change, not a runtime feature. This
      is the "automatically discover alternative designs" endpoint.

#### Hard constraints

- **Floating-point equivalence is tolerance-bounded, never exact.**
  Reassociation, Winograd, and factoring all change low bits. The
  gelu erf-vs-tanh incident (fixed `9b53da38`) is the cautionary
  tale: a ~1e-4 flavor divergence hid inside the 1e-3 consensus
  epsilon for two weeks. Per-rule precision deltas are mandatory,
  not decorative — the tolerance story is the actual hard part of
  this phase; the search is the easy 80%.
- **Transfers gate cross-device wins.** A CPU-shaped rewrite only
  wins if the subgraph amortizes 2× PCIe; the extraction objective
  must always include copy costs (it can — the pieces exist).
- **No rewrite fires without a cost-model win.** Equivalence alone
  is never sufficient justification.

#### Success criteria for Phase 10

- At least one anchor model where a device-conditional rewrite
  (fuse-vs-lower or a conv-algorithm choice) measurably beats the
  always-fuse pipeline on at least one backend, with the win
  attributable in the plan trace.
- Every rule in the library carries a verified precision delta; a
  rewritten graph's end-to-end tolerance is computable from the
  rules applied.
- Search cost is invisible in steady state (plan-cache hit) and
  bounded at cold compile.

#### Phase 10 honest caveats

This is a research-flavoured effort with strong prior art, not a
routine engineering task. 10a is days and worth doing early once the
gates clear; 10b-10c are multi-session; 10d is its own multi-week
tool. The phase exists in this document so the idea survives — the
2026-06-10 design discussion that produced it concluded Fuel is
unusually well-positioned (4 of 5 pieces already built,
device-agnostic and empirical by design), but that none of it should
interrupt the eager-retirement program or the picker arc.

---

## Eager-retirement follow-ups (post-Phase γ)

Phase γ (the Eager Tensor retirement program) shipped the bulk of the migration
off `fuel_core::Tensor` to `LazyTensor`, but a handful of items got quarantined
or deferred along the way rather than block the main sweep. Each bullet below
captures one such item with the minimum context needed to pick it up cold in a
future session. Group ordering mirrors the rough cost ladder — the binaries
need lazy ports of missing model families; the WASM crates need a workspace-
wide swap; the fuel-core integration tests are small mechanical fixes; the
fuel-book work is documentation; the lazy-side gaps are net-new primitives.

### 0. Closed (deleted)

These follow-ups were resolved by deleting the underlying binary directory
outright. Captured here so future readers see the decision rather than
wondering where the entry went.

- **mamba-minimal** — deleted 2026-06-07. The `_mamba-minimal_retired/` directory was supplanted by the full `lazy_mamba` + `lazy_mamba2` ports (both already migrated with working binaries), so the legacy minimal demo had no remaining consumer. No workspace member referenced it; the directory held only a stale README.
- **llama_multiprocess** — deleted 2026-06-07. The `_llama_multiprocess_retired/` directory was already emptied of `.rs` sources in Phase H, and `docs/session-prompts/lazy-multi-process-inference.md` explicitly recommends deferring a lazy multi-process driver until a real Fuel consumer needs multi-GPU tensor-parallel inference. Until that demand lands, there's no point keeping the empty directory around — when the work is picked up, the session prompt has everything needed to recreate `fuel-examples/examples/llama_multiprocess/{main.rs,model.rs}` from scratch against the lazy substrate.

### 1. Re-migrate the 10 quarantined `fuel-examples` binaries

Each binary was set aside because its target model family doesn't yet have a
lazy port. Restoring each one means landing the lazy port called out, then
doing the standard binary swap (lazy_X imports + lazy weight loader +
LazyTensor signatures).

- **debertav2** — needs `ForMaskedLM` + `ForSequenceClassification` heads in `lazy_debertav2` (encoder body already ports cleanly; the two task heads are the missing piece).
- **xlm-roberta** — needs `ForMaskedLM` + `ForSequenceClassification` heads in `lazy_xlm_roberta` (same shape as debertav2 — encoder ready, heads missing).
- **csm** — needs the autoregressive generation loop driver in `lazy_csm`; the underlying transformer blocks are already there, the AR decode harness is what's missing.
- **metavoice** — needs a `lazy_encodec` port (MetaVoice's neural audio codec dependency); the MetaVoice text-to-speech model itself can land once Encodec is available.
- **stable-diffusion-3** — needs the full `lazy_sd3` family: the triple-CLIP text-encoder composer (CLIP-L + CLIP-G + T5-XXL), the SD3 VAE, and the flow-match Euler sampler with SLG (Skip Layer Guidance).
- **llava** — needs `HFLLaVAConfig` + `LLaVAConfig` + `utils::select_best_resolution` in `lazy_llava` (the multi-resolution image preprocessing helper that picks the closest supported grid); the underlying CLIP + LLaMA ports are already lazy.
- **paddleocr-vl** — still needs an `HFConfig` helper in `lazy_paddleocr_vl` to bridge HuggingFace-style config JSON to the fuel-internal config struct; layer code is already lazy.
- **quantized-lfm2** — needs the base `lazy_lfm2` port to land first (LFM2 currently has no fp32/bf16 lazy port at all, so the quantized variant has nothing to specialize over).
- **rwkv** — needs the RWKV tokenizer ported (~95 LOC, inline-able) into `lazy_rwkv5` or a sibling tokenizer module; the model layers themselves are already lazy via `lazy_rwkv5` / `lazy_rwkv7`.
- **trocr** — needs `vit` + `trocr` ported into `lazy_vit` / `lazy_trocr` internals (the OCR-specific decoder is the trocr-specific part; the ViT encoder body is shared with the broader ViT port).

### 2. Re-migrate the 10 fuel-wasm-examples crates + fuel-wasm-tests

The entire WASM example tree is currently quarantined out of the workspace
(removed from `[workspace.members]`) because every crate depends on
`fuel_transformers::models::*` (retired in Phase H) and `fuel_nn::*` (retired
in Phase β4). Restoring the tree mirrors the `fuel-examples` program: per
crate, swap to the corresponding `lazy_X` module, and inline small helpers for
the handful of API points that don't have a 1:1 lazy equivalent yet (notably
`ops::softmax`, the `Linear` layer wrapper, and the VarMap-style loaders the
WASM binaries use for compact safetensors loading).

- `fuel-wasm-examples/bert`
- `fuel-wasm-examples/blip`
- `fuel-wasm-examples/llama2-c`
- `fuel-wasm-examples/moondream`
- `fuel-wasm-examples/phi`
- `fuel-wasm-examples/quant-qwen3`
- `fuel-wasm-examples/segment-anything`
- `fuel-wasm-examples/t5`
- `fuel-wasm-examples/whisper`
- `fuel-wasm-examples/yolo`
- `fuel-wasm-tests`

### 3. Fuel-core integration test fixups (pre-existing breakage, not retirement-related)

These three integration tests were already broken before Phase γ started, but
were left untouched during the sweep so the retirement diffs stayed focused.
They are small mechanical fixes against the current `LazyTensor` + storage-seam
API, not architectural follow-ups.

- `tests/phase6b_cuda_anchor.rs` — the `realize_f32_*` methods are now `Result`-returning; needs `?` insertion at the call sites. Separately, `ClipTextConfig` gained a required `activation` field that this test's fixture doesn't populate.
- `tests/cuda_composed_bisect.rs` — same `Result`-vs-`LazyTensor` mismatch across `realize_f32`, `matmul`, and `rms_norm_last_dim` call sites.
- `tests/tensor_tests.rs` — the storage seam now returns `Arc<RwLock<Storage>>` instead of a `RwLockReadGuard`; the test reaches through the old guard shape and needs to be retargeted at the `Arc<RwLock<...>>` API.

### 4. fuel-book doctest cleanup

`fuel-book/src/simplified.rs` is currently `mod`-gated off because it consumed
`fuel_nn::{Linear, VarMap, VarBuilder, SGD, Module, Optimizer, ops::log_softmax,
loss::nll}` — all retired in Phase β4. Port it to the lazy substrate:
`LazyTensor` + `LazyVar` + `lazy_nn_loss::nll` + `LazyAdamW` (SGD's lazy
equivalent is the AdamW family; for a strict SGD port, a `LazySGD` would be a
small additional follow-up).

### 5. fuel-book markdown docs

Five `.md` files under `fuel-book/src/guide` and `fuel-book/src/inference`
reference `fuel_nn` in prose and in inline code examples. Update both to use
the lazy substrate (`LazyTensor`, `lazy_nn::*`, `LazyVar`, `lazy_nn_loss::*`,
and `LazyAdamW`). Doc-only change; no fuel-core code touched.

### 6. Lazy-side primitive gaps surfaced during retirement (defer to follow-up)

These three primitives were tagged during Phase γ as "would have been nice to
have during the binary migrations" but were not load-bearing for any binary
that actually shipped — each one was worked around at the call site. They
warrant first-class lazy implementations when a downstream consumer needs them.

- **General-axis softmax on LazyTensor** — currently only `softmax_last_dim` is exposed; a general `softmax(axis)` would close a recurring port-time papercut for models that softmax over a non-trailing axis (typical in attention rewrites and some vision heads).
- **`max_pool2d` with `-inf` padding** — only the zero-padded variant is exposed today; some segmentation and detection heads expect `-inf` padding (so padded positions can never win the max). The shape is the same as the existing zero-padded kernel with a different fill constant; the lazy fanout is the work item.
- **`LazyConv2d::absorb_bn` helper** — for inference-time folding of a following BatchNorm into the conv's weight + bias (the standard "fuse BN" optimization). Small algebraic helper; deferred because no current lazy binary needs it at the API surface (the folding ports that do exist do it ad-hoc at load time).

---

## Anti-goals by layer

These are explicit rules. When a proposed addition fits one of these descriptions,
the answer is always no for that layer — find the right layer instead.

| Layer                                 | Will never contain                                                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Foundation (`fuel-core`)              | Tokenization, model-family assumptions, serving abstractions, HF Hub client code                                      |
| NN (`fuel-nn`)                        | Model-architecture implementations, inference session management, decode loops, training loops                        |
| Models (`fuel-transformers`)          | Serving infrastructure, batching schedulers, streaming decode loops, session lifecycle, training policy               |
| IO (`fuel-core` IO + `fuel-onnx`)     | Runtime policy, model architecture logic, serving abstractions                                                        |
| Inference (`fuel-inference`)          | New tensor primitives, new dtypes, new backend dispatch, training policy, anything that redefines foundation concepts |
| Training (`fuel-training`)            | New tensor primitives, new dtypes, inference-specific concerns (KV caches, sampling, decode loops)                    |
| Backends/Kernels                      | ML concepts, model logic, layer abstractions, training or inference policy, anything above shaped memory and math     |

---

## What will not change

- Published crate names will not be renamed speculatively. Renaming happens only
  after the new shape has proven itself, per the sequencing principle: define →
  document → reorganize → extract → rename.
- The early-exit property. A user who only wants tensor math must never be
  required to carry inference infrastructure.
- The breadth of model implementations. `fuel-transformers` is a genuine asset.
  The goal is to give it structure, not reduce its scope.
- Minimum viable complexity. Simple programs should stay simple. The framework
  should feel small from the bottom and powerful from the top.

---

## Dependency graph (target state)

```text
fuel-inference ─────────────────────────────────────────────────┐
fuel-training  ─────────────────────────────────────────────────┤
       │                                                      leaf crates
       │  both depend on                                          │
       ▼                                                          │
fuel-transformers ──────────────────────────────────────────────┤
       │                                                          │
       │  depends on                                          IO layer
       ▼                                                          │
fuel-nn ────────────────────────────────────────────────────────┘
       │
       │  depends on
       ▼
fuel-core  (eager path today; the lazy path in fuel_core::lazy
   │        wraps fuel-graph + fuel-graph-cpu + fuel-reference-backend)
   │
   │  depends on  [feature flags select which backend crates are compiled]
   ▼
fuel-cpu-backend          fuel-cuda-backend         fuel-metal-backend
    (always)                [feature = "cuda"]          [feature = "metal"]
                                   │                           │
                                   ▼                           ▼
                       fuel-cuda-kernels         fuel-metal-kernels
                       fuel-flash-attn
```

### Phase 6 sub-graph (the lazy layer)

```text
fuel-lazy-examples ─────────────────────┐
                                        │
                                      runnable
                                        │
                                        ▼
fuel-core::lazy (LazyTensor, LlamaModel, LlamaTokenizer, generate)
    │
    │  builds on
    ▼
fuel-graph-cpu (gemm-backed fast executor; `realize_*` entry points)
    │
    │  depends on, for non-matmul ops
    ▼
fuel-reference-backend (textbook-correct oracle; also provides RefTensor)
    │
    │  depends on
    ▼
fuel-graph (Op enum, Graph arena, Tensor handle, topo_order, backward)
    │
    │  depends on
    ▼
fuel-core-types (Shape, DType, Layout, BackendStorage trait, errors)
```

### Phase 7 sub-graph (vendor-optimized CPU backends)

```text
                Phase 6b dispatch table  (empirical per-op winner)
                              │ picks
       ┌──────────────────────┼──────────────────────────┐
       ▼                      ▼                          ▼
 fuel-aocl-cpu-backend  fuel-graph-cpu / fuel-cpu-     fuel-mkl-cpu-backend
       │                  backend (pure Rust;                │
       ▼                   gemm under                        ▼
   aocl-blas               the hood)                      onemkl
       │                                                    │
       ▼                                                    ▼
  AOCL BLIS runtime                                  Intel oneMKL runtime
   (external crate                                    (external crate
    aocl-blas-sys)                                     onemkl-sys)

All CPU backends implement the same GraphBackend trait surface and share
AnyRefTensor storage (so switching among them is a vtable swap, not a
transfer). Consumers enable backends via Cargo features (aocl, onemkl,
later accelerate / armpl / openblas). Phase 6b's Judge profiles each
loaded backend; the Router's pick_for_op consults the dispatch table per
op. No raw-cpuid heuristic picker — empirical wins on data, not on
vendor brand.
```

The backend crate split is the Tier 2 target state from Phase 5. Before that
landmark, the graph is the same but the backend code lives inside `fuel-core`
modules rather than separate crates.

Side dependencies: `fuel-onnx` and `fuel-datasets` depend downward as needed.
`fuel-pyo3` wraps whichever layers it needs without influencing them.
`fuel-dispatch` (Phase 5 Tier 4, long-term) sits between `fuel-core` and
new user-facing op-sequence APIs, with no effect on any layer above or below.
