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

## Current frontier (2026-06-25)

**Vision anchor.** Fuel is a DAG-first, lazy-only ML framework: the DAG is the source of truth for every decision, and the optimizer reading it is where the intelligence lives. The full statement is [`docs/architecture/01-identity.md`](docs/architecture/01-identity.md); this ROADMAP is *the path* toward it. Every item on the active frontier below moves at least one of the four [identity-enforcement checks](docs/architecture/01-identity.md#how-this-identity-is-enforced) *more* true and none less.

**The "plan IS the graph" redirection (2026-06-14, sharpened 2026-06-22).** The optimizer writes its decisions *into the graph* — chosen backend as a `Graph` side-table stamp, alternative execution paths as arms of `Op::Branch` decision-point nodes — and the executor picks among arms at runtime by live device load. `ExecutionPlan` is transitional scaffolding being removed; build new dispatch/fusion infrastructure into the **graph + the runtime-mutable kernel registry**, never into a side "plan." See [`docs/foundational-types.md`](docs/foundational-types.md) and the memory `plan-is-the-graph-architecture`.

### Active frontier (critical-path order)

1. **Dispatch-core cleanup** — move every strategic decision out of the realize-time bridge into `optimize_graph`, then have the executor read the graph. Step A (backend-stamping → optimizer) shipped on `dispatch-core-cleanup`. Remaining: B (residency + layout passes into the optimizer), C (arm-selection into the executor), D (delete `ExecutionPlan`), E (executor picks arms by **live device queue depth** — the near-term capability). *Unblocks Phase D integration + the §10 runtime/JIT refactor.*

2. **fuel-core / fuel-core-types retirement → `fuel-ir` + `fuel-hardware` + `fuel-backend-contract`** (foundation refactor; scoped, sibling-safe). `fuel-core-types` (the vocabulary) → `fuel-ir` (B0.1); hardware discovery (the `SystemTopology` split: discovery vs dispatch-overlay) → a new `fuel-hardware` crate (B0.2); the **backend-contract traits** (the `DynBackendStorage`/`DynBackendDevice` dyn-backend pair, `HostStorage`/`BackendStorage`/`BackendRuntime`/`BackendCapabilityProvider`, the quantized `DynQuantizedStorage`/`QuantizedDeviceKernels`, and `InplaceOp1/2/3`) + the type-erased `Storage` handle → a new `fuel-backend-contract` crate (B0.3, done 2026-06-27 — sits above fuel-ir, below the backends; capability *data* types stay in fuel-ir); the dispatch overlay stays in `fuel-dispatch`. **Remaining: B0.5** (storage-impl → fuel-memory: unify the closed-enum `fuel_memory::Storage` with the moved `Box<dyn DynBackendStorage>` handle, + drop the `VecOps` supertrait off `WithDType`). Sequenced inside the cleanup (Step B0) because both names collide on crates.io before publish. See the memory `fuel-core-retirement`.

3. **Phase D — symbolic extents + persistent decode.** Foundation shipped (SymId/SymEnv/DynScalar/Extent; WriteSlice runtime offset; FlashAttn runtime `k_len`; input-independent LlamaModel decode graph). Remaining: integrate symbolic extents into the unified `optimize_graph` post-cleanup; decode flash-vs-decomposed numerics; persistent plan-once decode (falls out once the graph is input-independent). Spec: `docs/session-prompts/symbolic-extents-and-persistent-decode.md`; memory `phase-d-symbolic-extents`. (Note the baracuda FA2 capacity-K constraint: valid for B·Hkv==1.)

4. **Tier-2 runtime fused-op registration (the JIT loop).** Envelope crate landed (`fuel-kernel-seam` / `fuel-kernel-seam-types`); FKC declarative patterns + structural matcher + CPU/Baracuda cost trampoline complete. Next: the §10 runtime/JIT integration onto plan-IS-the-graph (the sidecar becomes a graph annotation, not an executor-time lookup), then adopt into the executor. Baracuda coordination in flight. Memories `fusion-recipe-principle`, `kernel-contracts-dlpack-program`.

5. **Self-describing storage + kernel contracts.** SType/Encoding/ScaleSpec + DLPack view + FDX sidecar complete; kernel-boundary frozen. Next: finalize Baracuda/Vulkane coordination replies, land in a coordinated session. Memory `self-describing-storage`.

**Blockers**: none on the critical path. Multi-GPU work (Phase 6c D2D, 6d MoE placement) is parked pending hardware.

### Deferred backlog (behind the critical path)

Retained in detail below under Planned Work: Phase 7.5 C–F (graph-rewrite autograd, in-place-as-optimization, crate fission, layout contracts) + B3–B6 op-method sweep; Phase 7.6 steps 4/5/7/8/10 (fused-op migration sweep, Op-variant drops, PrecisionGuarantee/cost population, Comparison family); Phase 8 (FlashAttention tiers), 8.5 (activation sparsity), 9 (agentic extension hooks), 10 (equivalence-rewrite search); the eager-retirement follow-ups (binary re-migrations, test fixups). Sequenced *after* the active frontier; none is on the current critical path. One open design gap not yet phased: the **RNG / generator seam** — where a `Generator` lives (per-backend / per-device / per-graph), how it threads through realize and autograd, and how backends participate — which blocks dropout, sampling-as-a-graph-op, and stochastic training ops. Another backlog candidate: an **Apache Arrow `Tensor`** import/export leaf — *tensor*-level interchange (not a model format) for the columnar / data-engineering ecosystem (Arrow Flight distributed loading, polars/DuckDB feature pipelines, columnar feature stores). It is the host/serialization boundary, **complementary** to DLPack/FDX (which owns the device/kernel zero-copy boundary, `docs/specs/dlpack-extension.md`), not a competitor — sequence behind a real consumer, and lean on the existing DLPack↔Arrow-`ArrowDeviceArray` bridge first (FDX support already gives partial Arrow reach). Arrow's sparse layouts (COO/CSR/CSF) + `dim_names` are useful references if/when Fuel does sparse / named axes. See [13-interchange](docs/architecture/13-interchange.md) §Format posture.

### Shipped ledger

Phases 0–7 and the shipped portions of 7.5/7.6 + Phase C are complete; full detail is in git history (verbose phase blocks condensed 2026-06-25). Highlights:

- **Phases 0–4** — ecosystem compatibility, docs/clarity, use-case crate separation, model-area organization, ergonomics.
- **Phase 5 (Tier 1–3)** — backend modularity + pluggable empirical dispatch (CPU/CUDA/Vulkan/AOCL/MKL binding tables; the Judge).
- **Phase 6 (a–d)** — lazy frontend + single/multi-backend + multi-device routing; paged attention, symbolic autograd, kernel fusion, scheduler integration.
- **Phase 7 / CUDA restructure** — storage-hierarchy refactor; AOCL + oneMKL CPU backends; baracuda CUDA stack (CUTLASS B1–B4, flash, byte-storage fan-out).
- **Phase 7.5 A/B1/B2/G/G2** — graph owns Storage; `Op::Const` unit variant; lazy factories; realize() interface.
- **Phase 7.6 steps 1–3/6/9a–9c(A–E.3.0)** — FusedOpRegistry skeleton + `Op::Fused` arm; binding-table planning-time refactor; KvCache/InferenceContext; Vulkan runtime Device; multi-target realize; pipelined executor unification.
- **Phase C** — runtime route picker + command-buffer capture/replay (bounded per-device Pareto frontier).
- **Adaptive runtime fusion / FKC / self-describing storage** — recipe principle, two-tier extensibility, kernel-seam, declarative fusion engine, SType/Encoding + DLPack/FDX.
- **Dispatch-core cleanup Step A** — backend-stamping moved bridge→optimizer.

---

## Benchmarking

*Gated on Fuel running inference end-to-end without major speedbumps (the obviously
incomplete parts finished). Required as proof — not belief — before any non-alpha
release. This is the out-of-repo enforcement of the lazy-only performance bet stated in
[01-identity](docs/architecture/01-identity.md) and [09-non-goals](docs/architecture/09-non-goals.md):
since the in-repo eager path was retired in Phase 7.5, the comparator is now external.*

The thesis is that the lazy DAG should keep up with or **outperform every eager
framework**, because it picks the best available implementation of each op and adapts to
live device state — things eager code largely cannot do. That claim is currently
unproven; this program makes it falsifiable.

- **First yardstick — Candle eager** (fuel's near-unchanged fork parent; near-zero
  porting cost): same checkpoint, same machine, fuel lazy-realize vs Candle eager.
  Target: Candle's eager looks slow by comparison. This is the floor.
- **Beyond Candle**: apples-to-apples against llama.cpp (GGUF/quantized), PyTorch, ONNX
  Runtime, and Burn/CubeCL + tch-rs on the parts each does well — per-op where
  meaningful and end-to-end tokens/sec.
- **Instrumentation the program needs (build alongside the harness)**:
  - Per-token cost breakdown on a real anchor (graph build / topo / plan / dispatch /
    kernel wall) on CPU and GPU — the honest test of whether the planner overhead is
    amortized. Capture as a non-`#[ignore]` perf artifact, not folklore on stderr.
  - A roofline floor (model bytes ÷ measured memory bandwidth) so "DRAM-bound" claims
    are checkable rather than asserted.
  - A quantized end-to-end tokens/sec number (Q4_0 path) — on a bandwidth-bound system
    this is the largest single lever and the number every external evaluator asks first.
- **Make a budget constitutional once measured**: e.g. "plan + topo + coverage cost ≤ X%
  of measured decode-step time on the anchor suite," enforced as a perf gate, so the
  lazy-only bet has a numeric falsifier in-repo.

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

### Phases 0–7 + CUDA restructure — ✅ shipped (condensed)

The detailed per-phase blocks for Phases 0–7 and the CUDA stack restructure were condensed on 2026-06-25; they are summarized in the **Shipped ledger** under "Current frontier" above, and the full original text is in git history. The live and deferred work begins at Phase 7.5 below.

### Phase 7.5 — Core simplification: lazy-only execution, graph-rewrite autograd, and crate fissioning

> **Status (2026-06-25):** Shipped — A (fuel-formats), B1 (realize stubs), B2 (graph-owned factories), G (Graph owns Storage), G2 (`Op::Const` unit variant). Deferred (backlog, behind the active frontier) — B3–B6 (remaining op-method → lazy sweep), C (graph-rewrite autograd), D (in-place-as-optimization), E (crate fission), F (layout contracts). The detail below is retained as the backlog spec.

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

> **Status (2026-06-25):** Shipped — steps 1–3 (registry skeleton + `Op::Fused` arm + SoftmaxLastDim), step 6, steps 9a–9c phases A–E.3.0 (binding-table planning-time refactor, KvCache/InferenceContext, Vulkan runtime Device, multi-target realize, pipelined_bridge). Deferred (backlog) — steps 4, 5, 7, 8, 10 (fused-op migration sweep, Op-variant drops, PrecisionGuarantee/cost population, Comparison family), gated on the dispatch-core cleanup landing; step 9c E.3 remainder (`forward_with_cache_on`, `generate_*`, spec decoding) + E.4. The detail below is retained.

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
- ✅ Phase F + H — **executor-unification Session 7 (2026-06-15)**: the `GraphBackend` trait + all surviving impls (Cpu/Cuda/Vulkan/Mkl/Aocl), the `fuel-graph-executor` crate (`GraphExecutor<B>`), and the whole `fuel-graph-cpu` crate (`realize_any`, the typed third evaluator) are deleted. `PipelinedExecutor` (`fuel-dispatch`) is now the sole executor on every realize path. MKL/AOCL retain only their binding-table registration surface; the CUDA FA2 launcher (`fuel-cuda-backend::flash_attn::launch`) is preserved (`#[allow(dead_code)]`) for the queued FA2 eager-wrapper session; legacy-executor diff/oracle tests (`cpu_vulkan_diff.rs`, `conv2d_oracle.rs`, `flash_attn_cuda.rs`, `flash_attn_vulkan.rs`) retired with the trait. `fuel-reference-backend::exec::realize_f32` stays as the correctness oracle.
- ✅ Phase G — `GraphBackend` retain-vs-retire decision: **retired** (above). `fuel-graph-router`'s own crate disposition tracked with Session 6.
- ⏳ Remaining: executor-unification Session 8 (eager `Tensor` + `BackpropOp` tail).

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
