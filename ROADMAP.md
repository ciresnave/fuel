# Fuel — Roadmap

This document describes the current state of this project, the structural and ergonomic
problems it aims to solve, and the planned order of work.

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

Work is organized into nine phases. Later phases depend on earlier ones being
stable but phases within a group can proceed in parallel. Phase 9 is
extension hooks for downstream consumers (specifically: an out-of-tree
agentic library); not gated on the others, just gated on a real consumer
asking for them.

---

### Phase 0 — Ecosystem compatibility

*Immediate. Prerequisite for everything else. Without this, engineers new to the
fork can't get a working build.*

#### What Candlelight revealed

The Candle ecosystem consists of more than a dozen crates that must be kept in
version sync with each other. In practice they are not. Engineers who try to use
more than `candle-core` + `candle-nn` find that:

- `candle-optimisers`, `candle-layer-norm`, `candle-bhop`, `candle-einops`,
  `candle-birnn`, `candle-lstm`, `candle-crf`, and `candle-approx` each require
  separate forks to compile against the current Candle version.
- `candle-layer-norm` does not build on Windows with CUDA 13.0 without a
  patch. The Windows + MSVC path is not tested upstream.
- `candle-cublaslt` (cuBLASLt bindings for fused GEMM) and `candle-cuda-vmm`
  (CUDA Virtual Memory Management for elastic KV cache) have no home in the
  main crate tree at all.
- The result is that every downstream project must maintain its own Candlelight
  fork just to get a building dependency set.

This means the ecosystem is only usable by engineers willing to maintain those
forks themselves. The barrier is too high to attract contributors or users.

#### Work items

- [x] Audit every ecosystem crate (`fuel-optimisers`, `fuel-layer-norm`,
      `fuel-bhop`, `fuel-einops`, `fuel-birnn`, `fuel-lstm`,
      `fuel-crf`, `fuel-approx`) for version compatibility and build
      failures against the current workspace version of `fuel-core`.
      *Findings documented in `COMPATIBILITY.md`.*
- [x] Fix all build failures, including the Windows + CUDA 13.0 / MSVC
      path for `fuel-layer-norm`. Fixed `gen` reserved keyword in
      `fuel-core/src/cuda_backend/device.rs` (Rust edition 2024 reserved
      `gen` as a keyword; replaced with `r#gen` at the call site to
      `fuel_ug::cuda::code_gen::gen`). CUDA + cudnn features both check
      clean; CUDA tests pass on RTX 4070.
- [x] Bring `fuel-cublaslt` and `fuel-cuda-vmm` into this workspace as
      first-class crates rather than external dependencies (or, at minimum,
      ensure they are version-pinned and buildable).
- [x] Extract a new `fuel-vmm` crate from `fuel-cuda-vmm`. The page-tracking
      logic in `VirtualMemoryPool` (page state table, physical-handle map,
      allocation/deallocation math) is already free of any CUDA-specific code —
      CUDA calls appear at exactly three sites that map cleanly to an 8-method
      `VmmBackend` trait. `fuel-vmm` holds the trait and the generic pool
      structs (`VirtualMemoryPool<B>`, `SharedMemoryPool<B>`); `fuel-cuda-vmm`
      becomes `impl VmmBackend for CudaVmmBackend` with type aliases for
      backward compatibility. Benefits: (a) ROCm's HIP VMM API is a near-exact
      mirror of CUDA's, so Phase 5 multi-backend support gets elastic KV cache
      for free; (b) a CPU backend using `mmap`/`VirtualAlloc` with
      `MAP_NORESERVE` semantics becomes implementable for the CPU tier of tiered
      storage without duplicating pool logic; (c) `VirtualMemoryPool` is
      monomorphized per backend so the trait abstraction is zero-cost at runtime.
- [x] Establish a workspace-level `Cargo.toml` version matrix that defines
      the exact dependency set that is known to compile together cleanly.
      *The workspace `Cargo.toml` `[workspace.dependencies]` block is the
      authoritative version matrix. The human-readable summary, known-good
      constraints, and known-bad combinations are documented in
      `COMPATIBILITY.md`.*
- [x] Add CI that validates the full multi-crate build on each platform
      (Linux/CUDA, Windows/CUDA, macOS/Metal) so version drift is caught
      immediately. *`.github/workflows/rust-ci.yml` covers CPU builds on Linux,
      Windows, macOS, AVX2, and ARM; `.github/workflows/ci_cuda.yaml` covers
      CUDA 13.0 on a GPU runner.*
- [x] Write a one-page compatibility guide documenting which crate versions
      are tested together and why the matrix exists.
      *Created `COMPATIBILITY.md` at workspace root.*

**Success criterion**: a developer who clones this fork and runs `cargo build`
gets a working build with all features enabled, on all supported platforms,
without any manual patching or private fork git dependencies.

---

### Phase 1 — Documentation and clarity

*Low risk. Reversible. Highest return per unit of effort.*

- [x] Add `# Example` doc blocks to every public API in `fuel-core`
- [x] Add `# Example` doc blocks to every public API in `fuel-nn`
- [x] Add `# Example` doc blocks to every public API in `fuel-datasets`
- [x] Add `# Example` doc blocks to every public API in `fuel-onnx`
- [x] Add a top-level decision guide (in `README.md` or a dedicated `GUIDE.md`)
      routing users by intent: tensor math, custom layers, pretrained models,
      inference pipelines, ONNX, custom backends
      *Created `GUIDE.md` at workspace root.*
- [x] Write a clear "what this crate is for / what is explicitly not here / what
      to use next" section at the top of each crate's `lib.rs`
- [x] Define and document anti-goals per layer explicitly — what will never
      belong in each crate — so that future drift has a written boundary to push
      against
- [x] Add maturity labels (`stable` / `evolving` / `experimental` /
      `example-only`) to major subsystems and document them
- [x] Write canonical pattern guides (not examples — architecture guides) for:
      - minimal tensor program
      - minimal trainable module with autograd
      - minimal pretrained model load and forward pass
      - minimal inference loop with sampling
      - minimal custom operation extension
      *Created `PATTERNS.md` at workspace root.*

---

### Phase 2 — Structural: use-case crate separation

*Medium complexity. Does not change published crate names.*

The goal is to give inference-specific and training-specific tooling their own
canonical homes without changing anything below them in the dependency graph.
Nothing in `fuel-core`, `fuel-nn`, or `fuel-transformers` will depend on
either of these crates. They are opt-in by definition.

**On naming and scope**
These crates are named for what they orchestrate, not for what they are made of.
`fuel-inference` is the right name because the crate exists to support the act
of running inference — not because it is a generic "runtime." `fuel-training`
exists to support the act of training. Domain-specific applications (a
recommendation engine, a categorization pipeline) are applications composed from
these building blocks, not parts of the framework; they belong in user code or
separate ecosystem projects, not in this repository.

**Create `fuel-inference` as a leaf crate**

Move into `fuel-inference`:

- `fuel-nn/src/kv_cache.rs` — all cache implementations (`Cache`, `KvCache`,
  `RotatingKvCache`, `ConcatKvCache`, `ScatteredKvCache`)
- `fuel-nn/src/sampling.rs` — `gumbel_softmax`
- `fuel-transformers/src/generation/mod.rs` — `LogitsProcessor`, `Sampling`,
  all logit processing strategies
- `fuel-transformers/src/pipelines/` — the planned (currently stub) pipeline
  and session abstractions belong here
- Any future: batching, streaming decode, token generation loops, speculative
  decoding, cancellation, inference session management

**Create `fuel-training` as a leaf crate**

Initially empty beyond its scaffolding. As training-orchestration code accumulates
(whether migrated from examples or written fresh), this is where it lives:

- Training loop abstractions
- Gradient accumulation strategies
- Learning rate schedulers
- Gradient clipping utilities
- Mixed precision training policy
- Run-time checkpoint saving and resumption
- Training session management

**Key property to document explicitly on both crates:**
> Nothing in the Fuel ecosystem depends on `fuel-inference` or
> `fuel-training`. Both are leaf crates. They aggregate; they do not define.

#### Inference capabilities to contribute from Lightbulb

Lightbulb is an inference engine built on top of this Candle fork that was
developed independently because the pieces needed to build a production-quality
inference engine were not available or not usable in Candle as-is. Its
implementations are now the intended source material for `fuel-inference`.
Contributing them back avoids others having to reinvent the same work.

*KV cache management* (from `lightbulb::cache`):

- **Prefix caching**: Hash-based reuse of KV states for shared prompt prefixes
  (system prompts, few-shot examples). Measured 15–50% TTFT reduction on
  repeated prefixes. Stores (SHA256 hash → per-layer KV tensors) with LRU
  eviction of the cache itself.
- **Composable eviction policies**: A `EvictionPolicy` trait with a
  `VotingAggregator` that combines multiple policies with per-policy weights.
  Implemented policies include recency (LRU/sliding window) and H2O
  (Heavy-Hitter Oracle: preserve tokens with highest cumulative attention
  scores, discard the rest).
- **KV cache compression**: Three orthogonal strategies that can be combined:
  - *KIVI*: 2/4-bit per-channel quantization with per-head scales and optional
    residual coding for keys. 2–4× KV memory reduction.
  - *R-KV*: Importance-redundancy scoring that retains a configurable budget
    fraction (e.g., 34%) of tokens ranked by attention importance minus
    redundancy. \u22651.5× throughput on long CPU decodes.
  - *Low-Rank*: Attention approximation at a tunable rank parameter. Trades
    a small perplexity cost for a fixed KV memory ceiling.
- **Segmented eviction**: Per-span tracking with a `SpanRegistry` so spans
  (long sequences, conversation turns, document chunks) can be evicted as
  complete units rather than by individual tokens.
- **Tiered storage**: GPU (VRAM) → CPU (RAM) → Disk (filesystem/RocksDB).
  Demoted segments retain position IDs for correct RoPE re-injection when
  promoted back. Supports `<RETRIEVE:key>` token patterns for model-triggered
  promotion.
- **Streaming policy**: Sink-token + recent-window strategy for attention sinks
  in very long sequences (StreamingLLM pattern).

*Inference scheduling* (from `lightbulb::engine`):

- **Memory-aware scheduler**: Extends slot pool with a memory budget, per-slot
  cost tracking (base + per-token KV cost), and a priority queue
  (Low / Normal / High / Critical). Above a configurable eviction-pressure
  threshold, low-priority requests are queued rather than admitted.
- **Speculative decoding**: Draft-model-generates-K, target-model-verifies
  pattern (Leviathan et al. 2023). Full statistics tracking (acceptance rate,
  draft/target time). Auto-fallback when acceptance rate drops below
  configurable floor. Measured 1.3–2× latency improvement on typical workloads.
- **Chunked prefill**: Breaks long prefill sequences into chunks to bound
  time-to-first-token and allow interleaving with decode steps.
- **MoE routing**: Capacity-aware token routing for Mixture-of-Experts models
  (Mixtral, Qwen-MoE, etc.). Top-K selection with Token Drop and Expanded Drop
  capacity overflow policies. Per-expert batch construction for parallel
  expert execution.
- **Context compression**: For conversations exceeding context length, compress
  or summarize earlier turns without losing coherence.
- **Tool call infrastructure**: Structured tool call parsing, dispatch, and
  result injection for function-calling models.

---

### Phase 3 — Structural: model area organization ✅ (directory reorganization complete)

*Medium complexity. `fuel-transformers` internal only. Published API surface unchanged.*

`fuel-transformers` is approaching the point where its flat structure creates
genuine contributor confusion. Reorganize its internal module hierarchy before
it grows further.

Proposed internal structure (not new crates — internal modules only, for now):

```text
fuel-transformers/src/
  models/
    llm/          LLaMA, Mistral, Falcon, Phi, Gemma, Qwen, DeepSeek, etc.
    vision/       ViT, DINOv2, EfficientNet, ResNet, CLIP, SigLIP, etc.
    audio/        Whisper, EnCodec, Mimi, DAC, Parler TTS, etc.
    diffusion/    Stable Diffusion, Flux, Wuerstchen, etc.
    multimodal/   LLaVA, Moondream, PaliGemma, Pixtral, etc.
    encoders/     BERT, T5 encoder, Nomic BERT, etc.
    common/       with_tracing.rs, shared attention primitives, etc.
  quantized/      All quantized_*.rs variants consolidated here
```

Separate architecture definitions from inference glue within each model file:

- Config structs and forward passes stay in `models/`
- KV-cache handling, decode loops, and sampling hooks move to `fuel-inference`

#### Model-layer capabilities to contribute from Lightbulb

Lightbulb also accumulated implementations at the model/kernel layer that
belong in `fuel-transformers` or `fuel-nn`, not in `fuel-inference`.

*Fused operations* (from `lightbulb::model::fused_kernels`) — **added to `fuel-nn/src/fused_ops.rs`** ✅:

- **`fused_linear_silu`**: Combines linear projection + SiLU activation in a
  single pass, eliminating one intermediate tensor allocation. ~11% bandwidth
  reduction in MLP forward passes.
- **`fused_matmul_residual`**: Combines the output write of matmul with the
  residual addition, avoiding a second memory round-trip.
- **`fused_rmsnorm`**: Portable fallback using fuel-core tensor ops so
  RMSNorm does not materialize a separate squared-norms tensor. Provides a
  stable dispatch point for a future `fuel-layer-norm` CUDA kernel.

*Unified quantized/float linear layer* (from `lightbulb::model`):

- **`QuantizableLinear`**: An enum over `fuel_nn::Linear` (fp32/fp16/bf16
  from safetensors) and `QMatMul` (Q4\_0, Q4\_K, Q8\_0, etc. from GGUF),
  both implementing the `Module` trait identically. Inference code written
  against `QuantizableLinear` works with either weight format without changes.
  This belongs in `fuel-nn` so that every model can adopt it without
  importing an external crate.

*LoRA adapter support* (from `lightbulb::lora`):

- Low-Rank Adaptation weight injection as a `Module`-compatible wrapper.
  Currently in `fuel-examples`; should be a first-class type in
  `fuel-nn` so adapter-enabled models don't need to re-implement it.

*Multi-GPU support* (from `lightbulb::multi_gpu`):

- **Tensor parallelism**: Column-wise and row-wise sharding strategies for
  linear layers. `TensorShard` type that carries rank, world size, and
  original shape metadata. `ShardedLinear` layer that handles all-reduce
  after the local matmul.
- **Pipeline parallelism**: Stage assignment and inter-stage communication
  primitives for models too large to fit on a single device.
- **Device topology**: Enumeration of interconnect types (NVLink, PCIe) and
  bandwidth estimation for the DAG transfer cost model (connects to Phase 5).
- **Distributed cache**: Cache state synchronisation protocol across GPUs for
  paged and prefix caches.

The multi-GPU work belongs in `fuel-transformers` (or a new
`fuel-parallel` crate if it grows large enough) because it is model-topology
infrastructure, not inference policy.

#### Phase 2 work items

- [x] Scaffold `fuel-inference` crate (re-exports `kv_cache`, `generation`,
      `sampling` from their current locations for discoverability; physical code
      migration is the next step).
- [x] Scaffold `fuel-training` crate (empty framework; training-loop
      abstractions will migrate here as they are written or ported).
- [x] Update workspace `Cargo.toml` to include both crates in the `[members]`
      list and `[workspace.dependencies]` so all crates can reference them
      without version drift.
- [x] Add `[x] QuantizableLinear` enum to `fuel-nn` — wraps `Linear` (float)
      and `QMatMul` (quantized/GGUF) behind a single `Module`-compatible
      interface. `dequantized_weight()` helper returns the weight as a plain
      tensor regardless of storage format.
- [x] Add `LoraLinear` type and `lora_linear` / `lora_linear_peft` /
      `lora_linear_with_base` constructors to `fuel-nn`. LoRA adapters are
      now a first-class type — no external crate or per-model reimplementation
      required. `merge_weights()` bakes the adapter into a plain `Linear` for
      zero-overhead export.
- [x] Physically move `fuel-nn/src/kv_cache.rs` into `fuel-core` (better
      than `fuel-inference` — avoids circular deps entirely). `fuel-nn`
      now has a 7-line backward-compat shim (`pub use fuel::kv_cache::*`);
      `fuel-inference` re-exports from `fuel::kv_cache::*`. All existing
      callers are unaffected. Doctest references updated to `fuel_core::`.
- [x] Physically move `fuel-nn/src/sampling.rs` into `fuel-core` (same
      reasoning — `fuel-core` is at the bottom of the dep graph, no cycle
      possible). `fuel-nn` now has a 7-line shim (`pub use fuel::sampling::*`);
      `fuel-inference` re-exports from `fuel::sampling::*`. `fuel-inference`
      no longer depends on `fuel-nn` at all.
- [x] Decouple `fuel-transformers/src/generation/mod.rs` from `fuel-nn`:
      replaced `fuel_nn::sampling::gumbel_softmax` with
      `fuel::sampling::gumbel_softmax` (now in `fuel-core`) and replaced
      `fuel_nn::ops::softmax_last_dim` with an inline numerically stable
      softmax using only `fuel-core` ops (`max_keepdim`, `broadcast_sub`,
      `exp`, `sum_keepdim`, `broadcast_div`). Generation stays in
      `fuel-transformers` — moving it to `fuel-inference` would require
      `fuel-transformers` to depend on `fuel-inference`, violating the
      leaf principle. `fuel-inference` re-exports from
      `fuel_transformers::generation` — the public API is already in the
      right namespace for callers.

**Phase 2 current status**: Scaffolding is complete. Both crates exist in the
workspace and `cargo check` passes. `fuel-inference` is a re-export facade
surfacing `kv_cache`, `sampling`, and `generation` from their current locations
— no physical code migration has occurred yet. `fuel-training` is an empty
scaffold with documentation describing what will live there. The Lightbulb
inference and scheduling capabilities listed above remain to be contributed:

- [x] Contribute prefix caching (hash-based KV reuse for shared prompt prefixes).
      Implemented `PrefixCache` in `fuel-inference/src/prefix_cache.rs` — stores
      per-layer `(K, V)` tensor pairs keyed by token-sequence hash with LRU eviction.
      `lookup()`, `insert()`, `longest_prefix_match()`, `cached_seq_len()`.
      10 unit tests, 1 doctest, 0 failures.
- [x] Contribute composable eviction policies (`EvictionPolicy` trait,
      `VotingAggregator`, LRU, H2O). Implemented in
      `fuel-inference/src/eviction.rs` — `EvictionPolicy` trait with `score()`
      method, `LruPolicy` (recency-based), `H2oPolicy` (attention-importance),
      `VotingAggregator` (weighted combination with `select_keep()`/`select_evict()`).
      10 unit tests, 4 doctests, 0 failures.
- [x] Contribute KV cache compression (KIVI quantization, R-KV importance
      scoring, low-rank approximation). Implemented in
      `fuel-inference/src/kv_compress.rs` — `KvCompressor` trait with
      `CompressedKv` decompress round-trip. Three strategies: `KiviCompressor`
      (2/4-bit per-channel asymmetric quantization), `RkvCompressor`
      (importance-redundancy scoring with budget fraction and redundancy
      weight), `LowRankCompressor` (rank-R mean-centered projection).
      20 unit tests, 1 doctest, 0 failures.
- [x] Contribute segmented eviction (`SpanRegistry`, per-span tracking).
      Implemented `SpanRegistry` in `fuel-inference/src/segmented_eviction.rs` —
      span-level KV cache management where logical segments (system prompts,
      turns, documents, tool outputs) are tracked and evicted as complete units.
      `SpanKind`-based priority, pin/unpin, custom priority, FIFO tie-breaking,
      `plan_eviction()` produces `EvictionPlan` with position ranges.
      13 unit tests, 1 doctest, 0 failures.
- [x] Contribute tiered storage (GPU → CPU → Disk demotion/promotion).
      Implemented `TieredStore` in `fuel-inference/src/tiered_storage.rs` —
      GPU/CPU/Disk tiers with byte-budget tracking, `demote()`/`promote()`
      returning `TierTransfer` descriptors, position range preservation for
      RoPE re-injection, access-count-based demotion candidate selection,
      `touch()`, unbounded disk tier. 17 unit tests, 1 doctest, 0 failures.
- [x] Contribute streaming policy (StreamingLLM sink-token + recent-window).
      Implemented `StreamingPolicy` in `fuel-inference/src/streaming.rs` —
      sink-token + recent-window strategy (Xiao et al., ICLR 2024).
      `select_keep()`, `select_evict()`, `position_ids()` for RoPE correction,
      `needs_eviction()`. 12 unit tests, 1 doctest, 0 failures.
- [x] Contribute memory-aware scheduler (budget tracking, priority queue,
      eviction-pressure admission control). Implemented `MemoryScheduler` in
      `fuel-inference/src/scheduler.rs` — byte-budget tracking, 4-level
      `Priority` (Low/Normal/High/Critical), pressure-threshold gating,
      `try_admit()`/`release()`/`drain_queue()`/`update_usage()`.
      11 unit tests, 1 doctest, 0 failures.
- [x] Contribute speculative decoding (draft/verify pattern, auto-fallback).
      Implemented in `fuel-inference/src/speculative.rs` —
      `verify_draft()` implements the core accept/reject algorithm comparing
      draft vs target log-probabilities. `SpeculativeConfig` (draft_len,
      acceptance thresholds), `SpeculativeStats` (rolling acceptance rate,
      auto-fallback detection). Deterministic `pseudo_uniform()` for
      reproducible verification. 9 unit tests, 0 failures.
- [x] Contribute chunked prefill (bounded TTFT, decode interleaving).
      Implemented `ChunkedPrefill` in `fuel-inference/src/chunked_prefill.rs` —
      splits long prompts into bounded-size chunks with `PrefillChunk` yielding
      tokens, `index_pos`, and `is_last` flag. Supports reset, progress
      tracking, and arbitrary chunk sizes. 11 unit tests, 1 doctest, 0 failures.
- [x] Contribute MoE routing (capacity-aware top-K, Token Drop / Expanded Drop).
      Implemented `MoeRouter` in `fuel-inference/src/moe_routing.rs` —
      top-K softmax gating, `OverflowPolicy` (TokenDrop/NoDrop), per-expert
      capacity control, `ExpertBatch` construction, expert load distribution.
      11 unit tests, 1 doctest, 0 failures.
- [x] Contribute context compression (conversation summarization for long contexts).
      Implemented `ContextCompressor` in `fuel-inference/src/context_compress.rs` —
      turn-level token budgeting with `Role` (System/User/Assistant/Tool),
      recency × importance scoring, `plan_compression()` selecting lowest-scored
      turns, `mark_compressed()` for caller-driven summarisation, pinned turns,
      compressed fraction tracking. 12 unit tests, 1 doctest, 0 failures.
- [x] Contribute tool call infrastructure (structured parsing, dispatch, result injection).
      Implemented in `fuel-inference/src/tool_call.rs` — `ToolRegistry` with
      `ToolDef`/`ParamDef` schema, `ToolCall` parsing/validation (required
      params, unknown params, JSON check), `ToolResult` with
      `format_for_injection()`, `extract_tool_calls()` heuristic JSON extractor,
      `system_prompt()` generation. 20 unit tests, 1 doctest, 0 failures.
- [x] Populate `fuel-training` with training loop abstractions, gradient
      accumulation, LR schedulers, gradient clipping, checkpoint save/resume.
      Implemented 5 modules: `lr_scheduler` (6 schedulers: constant, step decay,
      cosine annealing, linear warmup, cosine-with-warmup, sequential
      composition), `grad_clip` (L2 norm and per-element value clipping),
      `grad_accum` (multi-step accumulation with averaged gradients),
      `checkpoint` (save/load with epoch, step, and named metrics metadata),
      `training_loop` (composable driver wiring clipping + scheduling + logging).
      31 tests (17 unit + 14 doctest), 0 failures. Mixed-precision policy is
      deferred — it requires `DType` autocast hooks in `fuel-core` that do
      not yet exist.

#### Phase 3 work items

- [x] Create category subdirectory structure in `fuel-transformers/src/models/`
      (`llm/`, `vision/`, `audio/`, `diffusion/`, `multimodal/`, `encoders/`,
      `common/`, `quantized/`)
- [x] Move LLM models (LLaMA, Mistral, Falcon, Phi, Gemma, Qwen, DeepSeek,
      etc.) into `models/llm/`
- [x] Move vision models (ViT, DINOv2, EfficientNet, ResNet, CLIP, SigLIP,
      etc.) into `models/vision/`
- [x] Move audio models (Whisper, EnCodec, Mimi, DAC, Parler TTS, etc.) into
      `models/audio/`
- [x] Move diffusion models (Stable Diffusion, Flux, Wuerstchen, etc.) into
      `models/diffusion/`
- [x] Move multimodal models (LLaVA, Moondream, PaliGemma, Pixtral, etc.)
      into `models/multimodal/`
- [x] Move encoder-only models (BERT, T5 encoder, Nomic BERT, etc.) into
      `models/encoders/`
- [x] Consolidate quantized model variants into `quantized/` subdirectory
- [x] Contribute Lightbulb multi-GPU support (tensor parallelism, pipeline
      parallelism, device topology, distributed cache) to a new
      `fuel-parallel` crate. Implemented 5 modules with 58 tests:
      - `topology.rs`: `DeviceTopology` graph with `DeviceInfo`, `DeviceKind`,
        `Interconnect` enum (NvLink/PCIe/InfinityFabric/SharedMemory/Network),
        `Link` with bandwidth/latency, `fastest_peer()`, `transfer_time_us()`.
        11 unit tests, 1 doctest.
      - `comm.rs`: `Communicator` trait (object-safe, Send) with `all_reduce`,
        `all_gather`, `reduce_scatter`, `broadcast`, `barrier`.
        `IdentityComm` single-process mock for testing. `ReduceOp`,
        `CommInfo`. 8 unit tests, 2 doctests.
      - `tensor_parallel.rs`: `TensorShard` metadata, `TensorParallelConfig`
        with `shard_range()`/`make_shard()`, `ColumnParallel` (no comm),
        `RowParallel<C: Communicator>` (all-reduce after local matmul),
        `LayerParallelPlan`. 9 unit tests, 1 doctest.
      - `pipeline_parallel.rs`: GPipe and 1F1B schedules, `PipelineConfig`,
        `Schedule` with `bubble_ratio()`, `StageAssignment::uniform()` for
        layer-to-stage mapping. 13 unit tests, 1 doctest.
      - `distributed_cache.rs`: `CacheShardInfo` (layer-to-rank assignment),
        `CacheSyncProtocol` (track per-rank-per-layer seq positions, prefix
        confirmation, flush protocol), `CacheRoutingHint`. 11 unit tests,
        1 doctest.

---

### Phase 4 — Ergonomics

*Ongoing. Parallel with other phases. Highest impact on adoption.*

**Error messages with shape context**
Fuel's error types already carry shape information in many cases. The goal is
to ensure this information surfaces consistently in a form that immediately
identifies the operation, the shapes involved, and what was expected. An error
that reads "expected `(batch, seq, 768)`, got `(batch, seq, 512)` in layer
`output_proj`" eliminates a class of debugging that currently requires reading
source code.

Shape-context `.with_context()` wrapping status in `fuel-nn`:

- [x] `Linear` — format includes in/out features and input shape
- [x] `Conv1d` — format includes in/out channels, kernel size, input shape
- [x] `Conv2d` — format includes in/out channels, kernel size, input shape
- [x] `LayerNorm` — format includes norm size and input shape
- [x] `RmsNorm` — shares `LayerNorm` implementation, inherits context
- [x] `Embedding` — format includes vocab size, hidden dim, indices shape
- [x] `BatchNorm` — format includes num_features and input shape
- [x] `GroupNorm` — format includes groups, channels, and input shape
- [x] `LSTM` — format includes in/hidden dimensions and input shape
- [x] `GRU` — format includes in/hidden dimensions and input shape
- [x] `ConvTranspose1d` — format includes in/out channels, kernel size, input shape
- [x] `ConvTranspose2d` — format includes in/out channels, kernel size, input shape

**Initialization convenience path** ✅
Currently getting a trainable model running requires understanding `Var`,
`VarBuilder`, `VarMap`, and their relationships before anything produces output.
`TrainingContext` (added to `fuel-nn`) bundles all four into a single struct
with `cpu_f32()` / `cpu_bf16()` shorthands, `vb()` / `vb_pp()` for building,
`vars()` for the optimizer, and `varmap()` for checkpointing.

**Builder pattern for complex configuration** ✅
Fluent builder methods (`.with_lr()`, `.with_stride()`, `.no_bias()`, etc.) added to:

- `ParamsAdamW` — `with_lr`, `with_beta1`, `with_beta2`, `with_eps`, `with_weight_decay`
- `Conv1dConfig` — `with_padding`, `with_stride`, `with_dilation`, `with_groups`
- `Conv2dConfig` — `with_padding`, `with_stride`, `with_dilation`, `with_groups`
- `LayerNormConfig` — `with_eps`, `no_mean_removal`, `no_bias`
- `BatchNormConfig` — `with_eps`, `no_mean_removal`, `no_affine`, `with_momentum`

- [x] **Function and parameter naming audit**: Comprehensive audit of all public
      API names across `fuel-core` and `fuel-nn`. Added non-breaking
      descriptive aliases for the most confusing APIs:
  - `Tensor::transpose_last_two()` → alias for `t()`
  - `Tensor::matvec()` → alias for `mv()`
  - `Tensor::scale_and_shift(scale, shift)` → alias for `affine(mul, add)`
  - `loss::negative_log_likelihood()` → alias for `nll()`
  - `AdamWConfig` type alias → for `ParamsAdamW` (matches `LayerNormConfig` convention)
  - `VarBuilder::push_prefix()` already existed as canonical (no change needed)
  - All aliases include full doc comments with runnable examples.

**IDE-first documentation standard**
All public items should have documentation that is useful when seen only as a
tooltip in a developer's editor: one-line summary, parameter semantics, common
failure modes, and a runnable example. Phase 1 begins this work; Phase 4
completes it by raising quality beyond the minimum bar.

- [x] **`cargo doc` warning elimination**: Fixed all rustdoc warnings across
      `fuel-core` (13 warnings: unresolved cross-crate links, bare URLs),
      `fuel-nn` (15 warnings: unresolved links, empty code blocks, ambiguous
      paths), and `fuel-transformers` (39 warnings: 16 double-semicolons `;;`
      in real code, 5 `[CLS]`/`[GH]`/`[CSM]` bracket-token links, 3 private
      item links, 14 bare URLs). Zero warnings now emitted by any of the three
      packages (the 6 remaining `fuel-core` "unused doc comment" warnings are
      upstream macro-generated items not fixable without changing the macro).

- [x] **`cargo test --doc` full compliance**: Fixed all 167 fuel-transformers
      doctest failures (887 now pass, 0 fail). Seven root causes addressed:
      (1) `fuel_core` → `fuel` import fix across 49 files (130 failures);
      (2) `unimplemented!()` examples changed to `no_run` in 11 files (17 failures);
      (3) pub(crate) field assertions removed from mixtral/stable_lm doctests (4);
      (4) `pub use` added for `VarBuilder` in quantized_rwkv_v5/v6 (4);
      (5) quantized VarBuilder imports corrected in 4 quantized model files (5);
      (6) `from_gguf()` argument counts fixed in quantized_glm4/phi3 doctests (4);
      (7) `pub use` for `Cache`/`Config` and `VisionModel` import in
      quantized_llama2_c/blip (3). Full workspace: 1804 passed, 0 failed.

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
- [ ] **Debuggability requirements**, non-negotiable from day one:
  - `Tensor::realize_eagerly()` — forces immediate execution of any pending
    graph for a single tensor, for use in debug prints and interactive
    development. The lazy model must not force developers to fly blind.
  - Planner "why did I pick this" traces — for any dispatched operation,
    the planner can emit a human-readable explanation of which backend was
    selected, which candidates were considered, and the cost-model inputs
    that drove the decision. Controlled by an environment variable and a
    per-graph flag.
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
- [x] **Performance: Arc-shared weights.** `ConstData` variants hold
      `Arc<[T]>`, `RefTensor` stores `Arc<[T]>`, `LlamaWeights` fields
      are `Arc<[f32]>`. Const-node evaluation in both executors is now a
      refcount bump, not a memcpy. Eliminated ~8 GB/call of allocation
      churn on TinyLlama.
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
- [ ] **Additional `HostStorage` impls** (unblocked by the trait):
  - `MmappedHostStorage` for zero-copy safetensors loading.
  - `PinnedHostStorage` for page-locked GPU DMA memory.
  - `SharedMemHostStorage` for IPC across processes.
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
`fuel-graph-vulkan` (ML-layer + graph) directly with no intermediate
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
fuel-graph-vulkan    (ML-layer + graph integration)
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

- [ ] Add `.realize()` / `.materialize()` to `Tensor`. For tensors
      backed by an eager-mode storage (current state), it's a
      no-op; for graph-built tensors (post-step C), it triggers
      executor dispatch.
- [ ] Migrate every `Tensor::*` op method to build a graph node
      instead of calling `Storage::*` directly. The dispatch path
      becomes the lazy-stack's `realize_*` entry points, with a
      fast-path for one-node graphs to amortise per-op overhead.
- [ ] Update `to_vec*`, `to_scalar`, `Display` impls, and any
      other "force value" entry points to call `.realize()`
      implicitly so users don't have to.
- [ ] Migration pass through `fuel-nn`, `fuel-transformers`,
      `fuel-examples`: most code remains unchanged because op
      methods retain their signatures; only "inspect a value"
      sites need `.realize()`.
- [ ] Document the idiom in `GUIDE.md` and `PATTERNS.md`.
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

Order:

1. **A (`fuel-formats`)** — ships now, parallel-safe with B
   because it touches the byte-decode bodies of loader files,
   not the Tensor construction call sites B is rewriting. No
   `Tensor` / `Storage` / `Device` coupling.
2. **B (drop eager + `.realize()`)** — the largest semantic
   change. Once eager is gone, `Storage` shrinks to a thin enum
   and `Tensor` op methods are pure graph builders. Most
   downstream cleanup is gated on this.
3. **C and E together** — once `Tensor_`'s `op: BackpropOp` and
   `is_variable` come out (C), `Tensor` is small enough that
   extracting it to `fuel-tensor` (E) is the same motion. The
   ROADMAP's original split treated C and E as separate phases;
   in practice C cannot finish without touching every site E
   needs to touch, and doing them together avoids a
   transitional state where `Tensor_` is half-shrunken.
4. **A2 (`fuel-loaders` finalization)** — afternoon of work
   once E lands. File-transport wrappers move from `fuel-core`
   to `fuel-loaders`; `fuel-core` re-exports for back-compat;
   no parse/construct seam to maintain.
5. **D (inplace-rewrite optimizer)** — depends on C+E producing
   the unified forward+backward graph to do liveness on.

B and C/E are tightly coupled but should ship as separate
landings rather than one mega-PR — B first (so the eager-vs-
lazy duality is collapsed before autograd refactor), then CE.

Total estimated scope: A is one week (parser extraction is
self-contained); B is two-to-three weeks including downstream
migration; CE together is six-to-eight weeks (every op
constructor touched, plus mechanical Tensor extraction); A2 is
half a day; D is one-to-two weeks of optimizer-pass work.
Roughly two-to-three months end-to-end.

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

This phase should not be attempted concurrently with Phase 8
(FlashAttention) or Phase 8.5 (sparsity); both add new
kernels/ops and would have to absorb the autograd-rewrite mid-
flight. Phase 9 (agentic hooks) is gated on a real consumer
and not in conflict.

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
