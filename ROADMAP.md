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

Work is organized into six phases. Later phases depend on earlier ones being
stable but phases within a group can proceed in parallel.

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

**Three-layer architecture** (in progress):

```text
fuel-core-types       ← Shape, DType, Layout, Error, CpuStorage (enum def),
                          BackendStorage/BackendDevice traits, WithDType, VecOps,
                          SIMD kernels, conv params, op traits
        ↑
fuel-cpu-backend      ← impl BackendStorage for CpuStorage, impl BackendDevice
        ↑                 for CpuDevice (matmul, binary ops, reductions, etc.)
fuel-core             ← Device/Storage enums, Tensor, Var, backprop,
                          custom_op, quantized, re-exports everything
```

Progress:

- [x] Created `fuel-core-types` crate with 21 source files extracting all
  foundational types, traits, and CPU SIMD infrastructure from `fuel-core`.
  Compiles standalone. Added to workspace members.
- [x] Verified full workspace builds with `fuel-core-types` present
  (`cargo check --workspace` passes).
- [x] Wire `fuel-core` to re-export from `fuel-core-types` — **partial**:
  - ✅ Wired: `shape.rs`, `layout.rs`, `strided_index.rs`, `dummy_dtype.rs`
    (fuel-core re-exports these entirely from fuel-core-types)
  - ❌ Blocked: `dtype.rs` (orphan rule: `TryFrom<safetensors::Dtype> for DType`),
    `backend.rs` (BackendStorage methods return fuel-core-types `Result`, but
    implementations use fuel-core `Result`), `error.rs` (MetalError conflict)
  - ❌ Blocked: `convert.rs` blanket impls `impl<T: WithDType> TryFrom<&Tensor> for Vec<T>`
    cause coherence errors when `WithDType` comes from upstream crate
  - Note: `From<fuel_core_types::Error> for fuel_core::Error` bridge enables
    `?` operator across crate boundary for re-exported shape/layout methods
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
- [ ] Extract cuda/metal backends into separate crates (already behind feature flags with separate kernel crates).

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
- [ ] **CI oracle gate.** The anchor-model suite runs on both the reference
      backend and the optimized pipeline, and CI asserts outputs match
      within tolerance on every PR. This gate applies to every sub-phase
      below. (Not yet wired into CI — the equivalence tests in
      `fuel-graph-cpu::tests` are the precursor, but they only cover the
      per-op level, not full anchor models.)
- [ ] **Debuggability requirements**, non-negotiable from day one:
  - `Tensor::realize_eagerly()` — forces immediate execution of any pending
    graph for a single tensor, for use in debug prints and interactive
    development. The lazy model must not force developers to fly blind.
  - Planner "why did I pick this" traces — for any dispatched operation,
    the planner can emit a human-readable explanation of which backend was
    selected, which candidates were considered, and the cost-model inputs
    that drove the decision. Controlled by an environment variable and a
    per-graph flag.
  - Shape mismatch errors report *where in the graph* the mismatch
    occurred, not just "at realize time," using the source location of the
    op that introduced the conflict. (Build-time shape validation in the
    `Tensor::*` builders already catches most of these with source
    locations via Rust panics; the structured graph-traversal version is
    still owed.)

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
- [ ] Implement **sequence-length bucketing** for dynamic shapes. Not yet
      started.
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
- [ ] Port the **anchor model set** (see below) to the lazy API, one at a
      time, validating each against the reference backend via the oracle
      gate. Partially done:
  - **LLaMA family** (Llama 1/2/3, TinyLlama, Mistral): end-to-end via
    `LlamaConfig`/`LlamaModel`. TinyLlama 1.1B generates coherent text
    at ~0.28 s/token on a modern desktop CPU (DRAM-bandwidth-bound at
    f32). Mistral is architecturally identical and runs without code
    changes.
  - **Qwen2**: end-to-end via the same model code with optional Q/K/V
    attention biases. `Qwen/Qwen2-0.5B-Instruct` generates coherent
    text at ~0.25 s/token. Bias detection is automatic from safetensors.
  - Whisper, ConvNeXt, SD 1.5, YOLOv8, BERT, Qwen2-MoE — not yet
    ported.

**Exit criterion for 6a**: all seven anchor models run end-to-end on the
lazy CPU backend, produce output bit-equivalent to the reference backend
within tolerance, and training works on at least one anchor (suggested: a
small Llama 3 variant). Performance does not yet need to match eager CPU
Candle — correctness is the 6a gate. **Status**: 2 of 7 anchors landed
(Llama family + Qwen2). Training on LLaMA has been validated on a
2-layer variant with full backward-pass gradient-descent loops in tests.

#### Sub-phase 6b — Multi-backend routing on one device

*Introduce backend choice. Prove the planner can pick between alternatives
on a single piece of hardware.*

- Add CUDA and Metal as selectable backends in the planner. Backends
  operate strictly on their native typed storage (`CpuStorage`,
  `CudaStorage`, `MetalStorage`); the internal dispatch boundary maps the
  localized graph onto strongly typed backend engines. No inner `dyn`
  downcasting.
- Implement the **probe-score-init lifecycle**: at startup, each backend
  reports its hardware compatibility score (0–100) for the available
  hardware, and the planner learns which backends are viable.
- Implement the **Judge module**: profile every
  (operation, backend, dtype, size class) tuple for latency and numerical
  precision against the reference backend. Results stored in a persistent
  on-disk dispatch table so the Judge runs once at install time, not per
  execution.
- Implement **ranked dispatch tables**: top-N backends per
  (op, dtype, size class), per criterion (fastest / most accurate /
  balanced). Runtime dispatch is an O(1) table lookup.
- The **cost model** for 6b is simple: one device means no transfer costs,
  just per-op latency from the dispatch table.
- Validate all anchor models on CUDA and (where available) Metal,
  oracle-gated.

**Exit criterion for 6b**: anchor models run on CUDA and Metal with
per-operation backend selection, oracle-equivalent to the reference
backend, and measurably faster than the lazy-CPU baseline on workloads
where the GPU is an improvement.

#### Sub-phase 6c — Multi-device routing on one node

*Introduce cross-device routing. This is where the DAG planner earns its
keep.*

- Extend the cost model with **transfer costs**: H2D, D2H, D2D (where
  supported), and inter-GPU bandwidth sourced from `fuel-parallel`'s
  `DeviceTopology`.
- Implement the **DAG planner**: transform the forward+backward graph into
  a layered DAG where each node is a (step, backend) pair, price each edge
  against the transfer cost model, and find the minimum-cost execution
  path via forward dynamic programming. This is the point at which a
  single computation can span CPU and GPU when the compute savings
  outweigh the transfer penalty.
- Insert **automatic transfer nodes** into the optimized graph where the
  planner decides a cross-device hop is worth it.
- Multi-GPU cases specifically (tensor parallelism, pipeline parallelism)
  use the existing `fuel-parallel` primitives rather than inventing new
  ones.
- Validate anchor models with artificial memory constraints that force
  cross-device execution (for example, a model too large for one GPU's
  VRAM).

**Exit criterion for 6c**: at least one anchor model that does not fit on
a single GPU runs correctly and faster under the DAG planner than under
any hand-tuned placement.

#### Sub-phase 6d — Kernel fusion, symbolic autograd, paged attention

*Optimization phase. Everything here is an improvement on a working 6c
baseline, not a prerequisite for it. Sub-phase 6d may be broken into
parallel tracks; its pieces are independent.*

- **Kernel fusion.** Backend engines compile localized sequences (e.g.
  `MatMul → Add → ReLU`) into single kernel launches where a fused kernel
  exists in the catalog. New fused kernels enter the catalog via the
  oracle acceptance gate — no fused kernel ships without bit-equivalent
  validation against the reference backend on a matrix of
  (dtype × shape × input distribution) tests.
- **Symbolic autograd transform.** Replace 6a's unfused backward with a
  graph-to-graph rewrite pass. Per-op gradient rules become graph
  constructors that emit new nodes representing the gradient computation,
  and the resulting backward graph is fused and scheduled by the planner
  alongside the forward graph. Backward fusion happens for free. Unlocks
  automatic gradient checkpointing (the planner decides which forward
  activations to drop and recompute) and higher-order gradients.
- **Paged attention.** Replace 6a's bucketing with a true paged attention
  kernel that consults a page table to fetch only populated cache blocks.
  Collapses all LLM decode shapes into a single execution path. The
  planner does not need to change — it simply picks the paged kernel when
  available and the bucketing fallback otherwise. Paged attention ships
  only after passing the oracle acceptance gate against sequence lengths
  up to the maximum supported context. Scheduled *after* the rest of Phase
  6 is stable and proven correct.
- **Scheduler integration.** The Lightbulb-contributed inference scheduler
  (`fuel-inference`) already handles speculative decoding and MoE expert
  routing as runtime policy. 6d integrates those with the planner so that,
  for example, expert batches land on backends the planner knows are best
  suited for them.

**Exit criterion for 6d**: Fuel's performance ceiling matches or exceeds
the best hand-tuned execution on each anchor model at the time of writing.
This is the "we built what we set out to build" gate.

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
- [ ] **Fix `fuel-core/tests/custom_op_tests.rs`** — still gated with
      `#![cfg(any())]`. Post-refactor, rewrite against `HostStorage`
      trait methods with `fuel-cpu-backend` as a dev-dep.

#### 7b — Vendor-optimized CPU backend crates

*Each is a self-contained crate implementing the backend trait against
a specific vendor library. Users opt in at their own Cargo.toml level,
so there are no feature-flag conflicts across transitive deps. The
pure-Rust `fuel-cpu-backend` stays the default; these are for users
who want the last 20-40% of performance out of specific hardware.*

- [ ] **`fuel-mkl-ffi` crate.** Thin FFI wrappers around Intel oneMKL's
      cblas interface. Adapted from rstsr-mkl-ffi's approach — copy the
      core wrappers into the Fuel tree so we control the release
      cadence and can adapt the Rust API to Fuel conventions. Minimal
      scope: `sgemm`, `dgemm`, a few common LAPACK routines. Can grow
      as Fuel's op catalog needs more FFI targets.
- [ ] **`fuel-aocl-ffi` crate.** Same pattern for AMD's AOCL. AOCL is
      BLIS under the hood with AMD-specific kernel tunings; on Zen 4
      and Zen 5 it typically wins 10-30% over MKL for large GEMMs.
      Adapted from rstsr-aocl-ffi.
- [ ] **`fuel-mkl-cpu-backend` crate.** Depends on `fuel-core-types`
      (for `HostStorage` and `BackendStorage`) and `fuel-mkl-ffi`.
      Provides `MklCpuBackendStorage` that wraps any `HostStorage`,
      implements `BackendStorage`, dispatches matmul / conv / the
      obvious BLAS targets through MKL, and falls back to `gemm` for
      anything MKL doesn't have a native implementation of. Builds
      require the Intel oneMKL runtime to be installed; the crate's
      README documents the setup.
- [ ] **`fuel-aocl-cpu-backend` crate.** Same for AOCL. Builds require
      AMD AOCL runtime to be installed.
- [ ] **Runtime CPU detection (optional wrapper crate).** A
      `fuel-cpu-auto-backend` that depends on all three and picks at
      startup via `raw-cpuid`: Intel → MKL, AMD → AOCL, unknown →
      pure-Rust. Opt-in; not wired into the default dep chain.

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
                            fuel-cpu-auto-backend   (optional runtime dispatch)
                                      │
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
         fuel-mkl-cpu-backend    fuel-cpu-backend    fuel-aocl-cpu-backend
                     │            (pure Rust;              │
                     ▼             gemm under              ▼
               fuel-mkl-ffi        the hood)         fuel-aocl-ffi
                     │                                     │
                     ▼                                     ▼
              Intel oneMKL runtime                    AMD AOCL runtime
                 (system lib)                         (system lib)

All three implement the same BackendStorage / BackendDevice trait surface
and operate on HostStorage-backed tensors. Consumers pick one (or several
via fuel-cpu-auto-backend) at Cargo.toml level; no feature flags.
```

The backend crate split is the Tier 2 target state from Phase 5. Before that
landmark, the graph is the same but the backend code lives inside `fuel-core`
modules rather than separate crates.

Side dependencies: `fuel-onnx` and `fuel-datasets` depend downward as needed.
`fuel-pyo3` wraps whichever layers it needs without influencing them.
`fuel-dispatch` (Phase 5 Tier 4, long-term) sits between `fuel-core` and
new user-facing op-sequence APIs, with no effect on any layer above or below.
