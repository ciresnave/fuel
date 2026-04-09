# Candle Fork — Roadmap

This document describes the current state of this fork, the structural and ergonomic
problems it aims to solve, and the planned order of work.

---

## Identity

This fork is a **layered Rust ML framework**.

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
│  candle-inference, candle-training  (leaf crates — nothing depends on     │
│  either of these)                                                          │
│                                                                            │
│  candle-inference: sampling, logits processing, KV-cache policy,          │
│  token generation loops, speculative decoding, batching, streaming         │
│  decode, cancellation, inference session abstractions                      │
│                                                                            │
│  candle-training: training loops, gradient accumulation, LR scheduling,   │
│  gradient clipping, mixed precision policy, run-time checkpointing,        │
│  training session abstractions                                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Models                                                                    │
│  candle-transformers  (will be restructured internally)                   │
│  Architecture config structs, layer composition, forward passes,          │
│  weight name mapping. No serving logic, no decode loops, no sessions.     │
├────────────────────────────────────────────────────────────────────────────┤
│  IO                                                                        │
│  candle-core (safetensors, npy, pickle), candle-onnx                      │
│  Bidirectional data exchange across any boundary: files, network,         │
│  memory buffers. Checkpoint load and save, format translation, ONNX       │
│  import/export, HF Hub integration glue, config normalization,            │
│  tokenizer glue. To be consolidated.                                       │
├────────────────────────────────────────────────────────────────────────────┤
│  NN                                                                        │
│  candle-nn                                                                 │
│  Layers, losses, optimizers, parameter utilities, initialization,         │
│  VarBuilder, VarMap. No model-family assumptions. No serving abstractions.│
├────────────────────────────────────────────────────────────────────────────┤
│  Foundation                                                                │
│  candle-core                                                               │
│  Tensors, devices, dtypes, shapes, layouts, base ops, autograd,           │
│  storage backends, error types. No tokenization. No model-level concepts. │
├────────────────────────────────────────────────────────────────────────────┤
│  Backends / Kernels                                                        │
│  candle-kernels, candle-metal-kernels, candle-flash-attn, candle-ug       │
│  Hardware and runtime targets (CPU, CUDA, Metal) plus the concrete        │
│  mathematical kernel implementations for each: matrix multiply, conv,     │
│  flash attention, quantized dot products, SIMD/BLAS dispatch. This layer  │
│  knows tensors as shaped memory regions and operations as mathematical     │
│  functions over those regions. It has no concept of layers, models,       │
│  losses, tokens, or any other ML abstraction.                              │
│                                                                            │
│  Foundation: `BackendDevice` and `BackendStorage` traits already exist    │
│  in candle-core. CUDA and Metal are behind feature flags. Phase 5         │
│  formalizes these as a published plugin contract and opens the type for   │
│  third-party backends.                                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Current State

### What is working well

- Dependency direction between published crates is already mostly correct.
  `candle-core` does not depend on `candle-nn`, which does not depend on
  `candle-transformers`. The early-exit property is structurally present.
- `candle-core` has a reasonable backend abstraction (CPU, CUDA, Metal).
- Quantization has a meaningful home in `candle-core::quantized`, better
  centralized than most frameworks at a comparable stage.
- The breadth of model implementations in `candle-transformers` is genuinely
  impressive and is a key asset.

### Identified problems

**Documentation**
The primary way users currently learn non-trivial usage patterns is by reading
examples. Examples are useful but they are poor architecture documentation. Most
public API items across all crates lack doc comments or runnable examples in the
documentation itself.

**Ergonomics / developer experience**
Using Candle non-trivially requires understanding `Var`, `VarBuilder`, `VarMap`,
device management, and dtype handling simultaneously before anything works. There
is no convenience path for common cases. Error messages often carry the right
information but do not always present it in a form that immediately tells you what
went wrong and how to fix it.

**Inference and training concerns are scattered**
`candle-nn` currently contains `kv_cache.rs` and `sampling.rs`.
`candle-transformers` contains `generation/` with `LogitsProcessor` and the
`Sampling` enum, as well as a `pipelines/` directory intended for orchestration
logic. These are inference-specific tools with no natural home below the
orchestration layer. The consequence is that `candle-nn` carries inference
weight that pure layer-building users never need.

**`candle-transformers` is a flat namespace with no internal structure**
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

- [x] Audit every ecosystem crate (`candle-optimisers`, `candle-layer-norm`,
      `candle-bhop`, `candle-einops`, `candle-birnn`, `candle-lstm`,
      `candle-crf`, `candle-approx`) for version compatibility and build
      failures against the current workspace version of `candle-core`.
      *Findings documented in `COMPATIBILITY.md`.*
- [x] Fix all build failures, including the Windows + CUDA 13.0 / MSVC
      path for `candle-layer-norm`. Fixed `gen` reserved keyword in
      `candle-core/src/cuda_backend/device.rs` (Rust edition 2024 reserved
      `gen` as a keyword; replaced with `r#gen` at the call site to
      `candle_ug::cuda::code_gen::gen`). CUDA + cudnn features both check
      clean; CUDA tests pass on RTX 4070.
- [x] Bring `candle-cublaslt` and `candle-cuda-vmm` into this workspace as
      first-class crates rather than external dependencies (or, at minimum,
      ensure they are version-pinned and buildable).
- [x] Extract a new `candle-vmm` crate from `candle-cuda-vmm`. The page-tracking
      logic in `VirtualMemoryPool` (page state table, physical-handle map,
      allocation/deallocation math) is already free of any CUDA-specific code —
      CUDA calls appear at exactly three sites that map cleanly to an 8-method
      `VmmBackend` trait. `candle-vmm` holds the trait and the generic pool
      structs (`VirtualMemoryPool<B>`, `SharedMemoryPool<B>`); `candle-cuda-vmm`
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

- [x] Add `# Example` doc blocks to every public API in `candle-core`
- [x] Add `# Example` doc blocks to every public API in `candle-nn`
- [x] Add `# Example` doc blocks to every public API in `candle-datasets`
- [x] Add `# Example` doc blocks to every public API in `candle-onnx`
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
Nothing in `candle-core`, `candle-nn`, or `candle-transformers` will depend on
either of these crates. They are opt-in by definition.

**On naming and scope**
These crates are named for what they orchestrate, not for what they are made of.
`candle-inference` is the right name because the crate exists to support the act
of running inference — not because it is a generic "runtime." `candle-training`
exists to support the act of training. Domain-specific applications (a
recommendation engine, a categorization pipeline) are applications composed from
these building blocks, not parts of the framework; they belong in user code or
separate ecosystem projects, not in this repository.

**Create `candle-inference` as a leaf crate**

Move into `candle-inference`:

- `candle-nn/src/kv_cache.rs` — all cache implementations (`Cache`, `KvCache`,
  `RotatingKvCache`, `ConcatKvCache`, `ScatteredKvCache`)
- `candle-nn/src/sampling.rs` — `gumbel_softmax`
- `candle-transformers/src/generation/mod.rs` — `LogitsProcessor`, `Sampling`,
  all logit processing strategies
- `candle-transformers/src/pipelines/` — the planned (currently stub) pipeline
  and session abstractions belong here
- Any future: batching, streaming decode, token generation loops, speculative
  decoding, cancellation, inference session management

**Create `candle-training` as a leaf crate**

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
> Nothing in the Candle ecosystem depends on `candle-inference` or
> `candle-training`. Both are leaf crates. They aggregate; they do not define.

#### Inference capabilities to contribute from Lightbulb

Lightbulb is an inference engine built on top of this Candle fork that was
developed independently because the pieces needed to build a production-quality
inference engine were not available or not usable in Candle as-is. Its
implementations are now the intended source material for `candle-inference`.
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

*Medium complexity. `candle-transformers` internal only. Published API surface unchanged.*

`candle-transformers` is approaching the point where its flat structure creates
genuine contributor confusion. Reorganize its internal module hierarchy before
it grows further.

Proposed internal structure (not new crates — internal modules only, for now):

```text
candle-transformers/src/
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
- KV-cache handling, decode loops, and sampling hooks move to `candle-inference`

#### Model-layer capabilities to contribute from Lightbulb

Lightbulb also accumulated implementations at the model/kernel layer that
belong in `candle-transformers` or `candle-nn`, not in `candle-inference`.

*Fused operations* (from `lightbulb::model::fused_kernels`) — **added to `candle-nn/src/fused_ops.rs`** ✅:

- **`fused_linear_silu`**: Combines linear projection + SiLU activation in a
  single pass, eliminating one intermediate tensor allocation. ~11% bandwidth
  reduction in MLP forward passes.
- **`fused_matmul_residual`**: Combines the output write of matmul with the
  residual addition, avoiding a second memory round-trip.
- **`fused_rmsnorm`**: Portable fallback using candle-core tensor ops so
  RMSNorm does not materialize a separate squared-norms tensor. Provides a
  stable dispatch point for a future `candle-layer-norm` CUDA kernel.

*Unified quantized/float linear layer* (from `lightbulb::model`):

- **`QuantizableLinear`**: An enum over `candle_nn::Linear` (fp32/fp16/bf16
  from safetensors) and `QMatMul` (Q4\_0, Q4\_K, Q8\_0, etc. from GGUF),
  both implementing the `Module` trait identically. Inference code written
  against `QuantizableLinear` works with either weight format without changes.
  This belongs in `candle-nn` so that every model can adopt it without
  importing an external crate.

*LoRA adapter support* (from `lightbulb::lora`):

- Low-Rank Adaptation weight injection as a `Module`-compatible wrapper.
  Currently in `candle-examples`; should be a first-class type in
  `candle-nn` so adapter-enabled models don't need to re-implement it.

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

The multi-GPU work belongs in `candle-transformers` (or a new
`candle-parallel` crate if it grows large enough) because it is model-topology
infrastructure, not inference policy.

#### Phase 2 work items

- [x] Scaffold `candle-inference` crate (re-exports `kv_cache`, `generation`,
      `sampling` from their current locations for discoverability; physical code
      migration is the next step).
- [x] Scaffold `candle-training` crate (empty framework; training-loop
      abstractions will migrate here as they are written or ported).
- [x] Update workspace `Cargo.toml` to include both crates in the `[members]`
      list and `[workspace.dependencies]` so all crates can reference them
      without version drift.
- [x] Add `[x] QuantizableLinear` enum to `candle-nn` — wraps `Linear` (float)
      and `QMatMul` (quantized/GGUF) behind a single `Module`-compatible
      interface. `dequantized_weight()` helper returns the weight as a plain
      tensor regardless of storage format.
- [x] Add `LoraLinear` type and `lora_linear` / `lora_linear_peft` /
      `lora_linear_with_base` constructors to `candle-nn`. LoRA adapters are
      now a first-class type — no external crate or per-model reimplementation
      required. `merge_weights()` bakes the adapter into a plain `Linear` for
      zero-overhead export.
- [x] Physically move `candle-nn/src/kv_cache.rs` into `candle-core` (better
      than `candle-inference` — avoids circular deps entirely). `candle-nn`
      now has a 7-line backward-compat shim (`pub use candle::kv_cache::*`);
      `candle-inference` re-exports from `candle::kv_cache::*`. All existing
      callers are unaffected. Doctest references updated to `candle_core::`.
- [x] Physically move `candle-nn/src/sampling.rs` into `candle-core` (same
      reasoning — `candle-core` is at the bottom of the dep graph, no cycle
      possible). `candle-nn` now has a 7-line shim (`pub use candle::sampling::*`);
      `candle-inference` re-exports from `candle::sampling::*`. `candle-inference`
      no longer depends on `candle-nn` at all.
- [x] Decouple `candle-transformers/src/generation/mod.rs` from `candle-nn`:
      replaced `candle_nn::sampling::gumbel_softmax` with
      `candle::sampling::gumbel_softmax` (now in `candle-core`) and replaced
      `candle_nn::ops::softmax_last_dim` with an inline numerically stable
      softmax using only `candle-core` ops (`max_keepdim`, `broadcast_sub`,
      `exp`, `sum_keepdim`, `broadcast_div`). Generation stays in
      `candle-transformers` — moving it to `candle-inference` would require
      `candle-transformers` to depend on `candle-inference`, violating the
      leaf principle. `candle-inference` re-exports from
      `candle_transformers::generation` — the public API is already in the
      right namespace for callers.

**Phase 2 current status**: Scaffolding is complete. Both crates exist in the
workspace and `cargo check` passes. `candle-inference` is a re-export facade
surfacing `kv_cache`, `sampling`, and `generation` from their current locations
— no physical code migration has occurred yet. `candle-training` is an empty
scaffold with documentation describing what will live there. The Lightbulb
inference and scheduling capabilities listed above remain to be contributed:

- [x] Contribute prefix caching (hash-based KV reuse for shared prompt prefixes).
      Implemented `PrefixCache` in `candle-inference/src/prefix_cache.rs` — stores
      per-layer `(K, V)` tensor pairs keyed by token-sequence hash with LRU eviction.
      `lookup()`, `insert()`, `longest_prefix_match()`, `cached_seq_len()`.
      10 unit tests, 1 doctest, 0 failures.
- [x] Contribute composable eviction policies (`EvictionPolicy` trait,
      `VotingAggregator`, LRU, H2O). Implemented in
      `candle-inference/src/eviction.rs` — `EvictionPolicy` trait with `score()`
      method, `LruPolicy` (recency-based), `H2oPolicy` (attention-importance),
      `VotingAggregator` (weighted combination with `select_keep()`/`select_evict()`).
      10 unit tests, 4 doctests, 0 failures.
- [x] Contribute KV cache compression (KIVI quantization, R-KV importance
      scoring, low-rank approximation). Implemented in
      `candle-inference/src/kv_compress.rs` — `KvCompressor` trait with
      `CompressedKv` decompress round-trip. Three strategies: `KiviCompressor`
      (2/4-bit per-channel asymmetric quantization), `RkvCompressor`
      (importance-redundancy scoring with budget fraction and redundancy
      weight), `LowRankCompressor` (rank-R mean-centered projection).
      20 unit tests, 1 doctest, 0 failures.
- [x] Contribute segmented eviction (`SpanRegistry`, per-span tracking).
      Implemented `SpanRegistry` in `candle-inference/src/segmented_eviction.rs` —
      span-level KV cache management where logical segments (system prompts,
      turns, documents, tool outputs) are tracked and evicted as complete units.
      `SpanKind`-based priority, pin/unpin, custom priority, FIFO tie-breaking,
      `plan_eviction()` produces `EvictionPlan` with position ranges.
      13 unit tests, 1 doctest, 0 failures.
- [x] Contribute tiered storage (GPU → CPU → Disk demotion/promotion).
      Implemented `TieredStore` in `candle-inference/src/tiered_storage.rs` —
      GPU/CPU/Disk tiers with byte-budget tracking, `demote()`/`promote()`
      returning `TierTransfer` descriptors, position range preservation for
      RoPE re-injection, access-count-based demotion candidate selection,
      `touch()`, unbounded disk tier. 17 unit tests, 1 doctest, 0 failures.
- [x] Contribute streaming policy (StreamingLLM sink-token + recent-window).
      Implemented `StreamingPolicy` in `candle-inference/src/streaming.rs` —
      sink-token + recent-window strategy (Xiao et al., ICLR 2024).
      `select_keep()`, `select_evict()`, `position_ids()` for RoPE correction,
      `needs_eviction()`. 12 unit tests, 1 doctest, 0 failures.
- [x] Contribute memory-aware scheduler (budget tracking, priority queue,
      eviction-pressure admission control). Implemented `MemoryScheduler` in
      `candle-inference/src/scheduler.rs` — byte-budget tracking, 4-level
      `Priority` (Low/Normal/High/Critical), pressure-threshold gating,
      `try_admit()`/`release()`/`drain_queue()`/`update_usage()`.
      11 unit tests, 1 doctest, 0 failures.
- [x] Contribute speculative decoding (draft/verify pattern, auto-fallback).
      Implemented in `candle-inference/src/speculative.rs` —
      `verify_draft()` implements the core accept/reject algorithm comparing
      draft vs target log-probabilities. `SpeculativeConfig` (draft_len,
      acceptance thresholds), `SpeculativeStats` (rolling acceptance rate,
      auto-fallback detection). Deterministic `pseudo_uniform()` for
      reproducible verification. 9 unit tests, 0 failures.
- [x] Contribute chunked prefill (bounded TTFT, decode interleaving).
      Implemented `ChunkedPrefill` in `candle-inference/src/chunked_prefill.rs` —
      splits long prompts into bounded-size chunks with `PrefillChunk` yielding
      tokens, `index_pos`, and `is_last` flag. Supports reset, progress
      tracking, and arbitrary chunk sizes. 11 unit tests, 1 doctest, 0 failures.
- [x] Contribute MoE routing (capacity-aware top-K, Token Drop / Expanded Drop).
      Implemented `MoeRouter` in `candle-inference/src/moe_routing.rs` —
      top-K softmax gating, `OverflowPolicy` (TokenDrop/NoDrop), per-expert
      capacity control, `ExpertBatch` construction, expert load distribution.
      11 unit tests, 1 doctest, 0 failures.
- [x] Contribute context compression (conversation summarization for long contexts).
      Implemented `ContextCompressor` in `candle-inference/src/context_compress.rs` —
      turn-level token budgeting with `Role` (System/User/Assistant/Tool),
      recency × importance scoring, `plan_compression()` selecting lowest-scored
      turns, `mark_compressed()` for caller-driven summarisation, pinned turns,
      compressed fraction tracking. 12 unit tests, 1 doctest, 0 failures.
- [x] Contribute tool call infrastructure (structured parsing, dispatch, result injection).
      Implemented in `candle-inference/src/tool_call.rs` — `ToolRegistry` with
      `ToolDef`/`ParamDef` schema, `ToolCall` parsing/validation (required
      params, unknown params, JSON check), `ToolResult` with
      `format_for_injection()`, `extract_tool_calls()` heuristic JSON extractor,
      `system_prompt()` generation. 20 unit tests, 1 doctest, 0 failures.
- [x] Populate `candle-training` with training loop abstractions, gradient
      accumulation, LR schedulers, gradient clipping, checkpoint save/resume.
      Implemented 5 modules: `lr_scheduler` (6 schedulers: constant, step decay,
      cosine annealing, linear warmup, cosine-with-warmup, sequential
      composition), `grad_clip` (L2 norm and per-element value clipping),
      `grad_accum` (multi-step accumulation with averaged gradients),
      `checkpoint` (save/load with epoch, step, and named metrics metadata),
      `training_loop` (composable driver wiring clipping + scheduling + logging).
      31 tests (17 unit + 14 doctest), 0 failures. Mixed-precision policy is
      deferred — it requires `DType` autocast hooks in `candle-core` that do
      not yet exist.

#### Phase 3 work items

- [x] Create category subdirectory structure in `candle-transformers/src/models/`
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
      `candle-parallel` crate. Implemented 5 modules with 58 tests:
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
Candle's error types already carry shape information in many cases. The goal is
to ensure this information surfaces consistently in a form that immediately
identifies the operation, the shapes involved, and what was expected. An error
that reads "expected `(batch, seq, 768)`, got `(batch, seq, 512)` in layer
`output_proj`" eliminates a class of debugging that currently requires reading
source code.

Shape-context `.with_context()` wrapping status in `candle-nn`:

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
`TrainingContext` (added to `candle-nn`) bundles all four into a single struct
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
      API names across `candle-core` and `candle-nn`. Added non-breaking
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
      `candle-core` (13 warnings: unresolved cross-crate links, bare URLs),
      `candle-nn` (15 warnings: unresolved links, empty code blocks, ambiguous
      paths), and `candle-transformers` (39 warnings: 16 double-semicolons `;;`
      in real code, 5 `[CLS]`/`[GH]`/`[CSM]` bracket-token links, 3 private
      item links, 14 bare URLs). Zero warnings now emitted by any of the three
      packages (the 6 remaining `candle-core` "unused doc comment" warnings are
      upstream macro-generated items not fixable without changing the macro).

- [x] **`cargo test --doc` full compliance**: Fixed all 167 candle-transformers
      doctest failures (887 now pass, 0 fail). Seven root causes addressed:
      (1) `candle_core` → `candle` import fix across 49 files (130 failures);
      (2) `unimplemented!()` examples changed to `no_run` in 11 files (17 failures);
      (3) pub(crate) field assertions removed from mixtral/stable_lm doctests (4);
      (4) `pub use` added for `VarBuilder` in quantized_rwkv_v5/v6 (4);
      (5) quantized VarBuilder imports corrected in 4 quantized model files (5);
      (6) `from_gguf()` argument counts fixed in quantized_glm4/phi3 doctests (4);
      (7) `pub use` for `Cache`/`Config` and `VisionModel` import in
      quantized_llama2_c/blip (3). Full workspace: 1804 passed, 0 failed.

---

### Phase 5 — Backend modularity and pluggable dispatch

*Large scope. Affects only the Backends/Kernels layer and candle-core's Device
type. Layers 1–4 are untouched and do not need to wait for this phase.*

#### Starting point — what already exists

The seam is present. `candle-core/src/backend.rs` defines `BackendDevice` and
`BackendStorage` as associated-type traits; CPU, CUDA, and Metal all implement
them. CUDA and Metal are already behind Cargo feature flags, meaning a
CPU-only user never compiles GPU code. The kernel crates (`candle-kernels`,
`candle-metal-kernels`, `candle-flash-attn`, `candle-ug`) are already separate
from `candle-core`.

What is absent:

- The `Device` enum is closed: `Cpu`, `Cuda(CudaDevice)`, `Metal(MetalDevice)`.
  Adding a fourth backend means modifying `candle-core`, which is a breaking
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
dispatch system for linear algebra. Studying it sharpens what Candle's backend
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

This is the far end of what a modular backend system can look like. Candle's
path arrives there through three well-sequenced tiers. None of them require
changing anything above the Backends/Kernels layer.

#### Tier 1 — Expose existing flags (now, cost: negligible)

This is a documentation-only change.

- [x] Document the `cuda` and `metal` Cargo feature flags prominently: in
  `candle-core/src/lib.rs` (feature table added to crate header) and the top-level
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
  contract, preconditions, and expected error conditions. (`candle-core/src/backend.rs`
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
candle-core-types       ← Shape, DType, Layout, Error, CpuStorage (enum def),
                          BackendStorage/BackendDevice traits, WithDType, VecOps,
                          SIMD kernels, conv params, op traits
        ↑
candle-cpu-backend      ← impl BackendStorage for CpuStorage, impl BackendDevice
        ↑                 for CpuDevice (matmul, binary ops, reductions, etc.)
candle-core             ← Device/Storage enums, Tensor, Var, backprop,
                          custom_op, quantized, re-exports everything
```

Progress:

- [x] Created `candle-core-types` crate with 21 source files extracting all
  foundational types, traits, and CPU SIMD infrastructure from `candle-core`.
  Compiles standalone. Added to workspace members.
- [x] Verified full workspace builds with `candle-core-types` present
  (`cargo check --workspace` passes).
- [x] Wire `candle-core` to re-export from `candle-core-types` — **partial**:
  - ✅ Wired: `shape.rs`, `layout.rs`, `strided_index.rs`, `dummy_dtype.rs`
    (candle-core re-exports these entirely from candle-core-types)
  - ❌ Blocked: `dtype.rs` (orphan rule: `TryFrom<safetensors::Dtype> for DType`),
    `backend.rs` (BackendStorage methods return candle-core-types `Result`, but
    implementations use candle-core `Result`), `error.rs` (MetalError conflict)
  - ❌ Blocked: `convert.rs` blanket impls `impl<T: WithDType> TryFrom<&Tensor> for Vec<T>`
    cause coherence errors when `WithDType` comes from upstream crate
  - Note: `From<candle_core_types::Error> for candle_core::Error` bridge enables
    `?` operator across crate boundary for re-exported shape/layout methods
- [x] Create `candle-cpu-backend` crate (extract cpu_backend module from
  candle-core). Created with 6 source files: `lib.rs`, `ops.rs` (~1770 lines of
  CPU computation kernels — MatMul, pooling, convolution, reductions, index ops),
  `utils.rs` (Map traits + vectorised helpers), `conv2d.rs` (tiled/im2col Conv2D),
  `mkl.rs` (Intel MKL FFI), `accelerate.rs` (Apple Accelerate FFI). Compiles
  standalone against `candle-core-types`. Added as workspace member and as
  dependency of `candle-core` with `mkl`/`accelerate` feature forwarding.
- [x] Delegation: candle-core's `cpu_backend/mod.rs` now delegates all major
  operations to `candle-cpu-backend` via 5 macros (`cpu_map1!`, `cpu_map1any!`,
  `cpu_map2!`, `cpu_map2u8!`, `cpu_map2_in_place!`).
  Delegated: affine, avg_pool2d, max_pool2d, upsample_nearest1d/2d,
  upsample_bilinear2d, reduce_op (Sum/Min/Max/ArgMin/ArgMax), index_select,
  gather, where_cond, index_add, matmul, cmp, conv1d, conv2d,
  conv_transpose1d, conv_transpose2d, scatter_set, scatter_add_set.
  Type unification: CmpOp, ReduceOp re-exported from candle-core-types;
  ParamsConv1D/2D/Transpose1D/Transpose2D + CudnnFwdAlgo re-exported from
  candle-core-types (~250 lines of duplicate struct defs removed from conv.rs).
  Dead local helper structs removed (~1370 lines total: MatMul, Conv1D/2D,
  ConvTranspose1D/2D, Im2Col, Im2Col1D, Col2Im1D, Cmp, WCond, ReduceIndex,
  ReduceSum, Affine, AvgPool2D, MaxPool2D, UpsampleNearest1D/2D,
  UpsampleBilinear2D, Gather, IndexSelect, ElemUpdate, Set, Add,
  Scatter, IndexAdd, plus conv2d.rs submodule deleted).
  mod.rs reduced from 3284 → 1917 lines. Zero errors, zero test failures
  (462 candle-core tests pass).
  Trait consolidation: UnaryOpT/BinaryOpT re-exported from candle-core-types
  (eliminated ~65 lines of duplicate trait definitions in op.rs).
  `unary_dispatch<B>`/`binary_dispatch<B>` functions added to candle-cpu-backend
  for standalone use. candle-core's `unary_impl`/`binary_impl` retain thin enum
  dispatch calling candle-cpu-backend's `unary_map`/`binary_map` helpers
  (full delegation blocked until CpuStorage is re-exported from candle-core-types,
  which requires resolving dtype.rs/convert.rs orphan rule issues).
- [ ] Extract cuda/metal backends into separate crates (future, lower priority —
  already behind feature flags with separate kernel crates).

#### Tier 3 — Open Device for third-party backends ✅ COMPLETE

`Device::Custom(Arc<dyn DynBackendDevice>)` and `Storage::Custom(Box<dyn DynBackendStorage>)`
are now fully wired. Object-safe `DynBackendStorage` (31 methods) and `DynBackendDevice`
(11 methods) traits live in `candle-core/src/dyn_backend.rs`.

**Implementation summary:**

- `Device::Custom` arm handled in all 16 match sites in `device.rs`
- `Storage::Custom` arm handled in all match sites across `storage.rs` (~30 methods),
  `tensor.rs` (4 sites), `safetensors.rs` (2 sites), `quantized/mod.rs` (3 sites),
  `quantized/ggml_file.rs` (1 site), and `candle-pyo3` (1 site)
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
use candle_core::dyn_backend::{DynBackendDevice, DynBackendStorage};
use candle_core::Device;
use std::sync::Arc;

struct MyDevice { /* ... */ }
impl DynBackendDevice for MyDevice { /* ... */ }

let device = Device::custom(Arc::new(MyDevice::new()));
```

#### Tier 4 — Operation-level routing (long-term vision)

Tiers 1–3 solve compile-time selection and third-party extensibility. They do
not address cross-backend routing: can operations within the same computation
use different backends?

Today, Candle executes eagerly. A tensor is on a device; an op runs immediately
on that device's backend. There is no mechanism to compile a sequence of
operations first, consult routing tables, and then execute with per-op backend
selection.

A future `candle-dispatch` crate could provide this without changing the eager
programming model for users who do not opt in:

- A **lazy op-sequence builder**: accepts operation descriptors without executing
  them, building an operation DAG equivalent to faster-blaster's `op_chain`.
- A **probe/score mechanism**: at startup, probe all registered backends against
  the available hardware and rate each one per operation type.
- A **judge equivalent**: benchmark candidate backends against a
  `candle-reference-backend` (the correctness oracle; analogous to
  `faster-blaster-reference`) and store latency profiles and precision curves.
- A **ranked dispatch table**: top-N backends per (op, dtype, size class), per
  criterion (fastest / most accurate / balanced). O(1) per-dispatch lookup.
- A **DAG planner**: transforms the op sequence into a layered DAG, prices each
  (step, backend) node against a data transfer cost model for cross-device edges,
  and selects the minimum-cost execution path using dynamic programming.

This tier requires `candle-core` to support at least a thin deferred execution
mode — tensors that record their op sequence before committing to a device — or
alternatively it operates at a higher level, emitting explicit `.to_device()`
transfers between steps as the DAG planner directs. The former is a deeper
change; the latter is composable on top of today's eager model. Either way,
it depends on Tiers 2 and 3 being stable first.

This is the point at which Candle's dispatch story converges with the
architecture that faster-blaster was designed around from the beginning.

---

## Anti-goals by layer

These are explicit rules. When a proposed addition fits one of these descriptions,
the answer is always no for that layer — find the right layer instead.

| Layer                                 | Will never contain                                                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Foundation (`candle-core`)            | Tokenization, model-family assumptions, serving abstractions, HF Hub client code                                      |
| NN (`candle-nn`)                      | Model-architecture implementations, inference session management, decode loops, training loops                        |
| Models (`candle-transformers`)        | Serving infrastructure, batching schedulers, streaming decode loops, session lifecycle, training policy               |
| IO (`candle-core` IO + `candle-onnx`) | Runtime policy, model architecture logic, serving abstractions                                                        |
| Inference (`candle-inference`)        | New tensor primitives, new dtypes, new backend dispatch, training policy, anything that redefines foundation concepts |
| Training (`candle-training`)          | New tensor primitives, new dtypes, inference-specific concerns (KV caches, sampling, decode loops)                    |
| Backends/Kernels                      | ML concepts, model logic, layer abstractions, training or inference policy, anything above shaped memory and math     |

---

## What will not change

- Published crate names will not be renamed speculatively. Renaming happens only
  after the new shape has proven itself, per the sequencing principle: define →
  document → reorganize → extract → rename.
- The early-exit property. A user who only wants tensor math must never be
  required to carry inference infrastructure.
- The breadth of model implementations. `candle-transformers` is a genuine asset.
  The goal is to give it structure, not reduce its scope.
- Minimum viable complexity. Simple programs should stay simple. The framework
  should feel small from the bottom and powerful from the top.

---

## Dependency graph (target state)

```text
candle-inference ─────────────────────────────────────────────────┐
candle-training  ─────────────────────────────────────────────────┤
       │                                                      leaf crates
       │  both depend on                                          │
       ▼                                                          │
candle-transformers ──────────────────────────────────────────────┤
       │                                                          │
       │  depends on                                          IO layer
       ▼                                                          │
candle-nn ────────────────────────────────────────────────────────┘
       │
       │  depends on
       ▼
candle-core
       │
       │  depends on  [feature flags select which backend crates are compiled]
       ▼
candle-cpu-backend          candle-cuda-backend         candle-metal-backend
    (always)                [feature = "cuda"]          [feature = "metal"]
                                   │                           │
                                   ▼                           ▼
                       candle-kernels              candle-metal-kernels
                       candle-flash-attn
```

The backend crate split is the Tier 2 target state from Phase 5. Before that
landmark, the graph is the same but the backend code lives inside `candle-core`
modules rather than separate crates.

Side dependencies: `candle-onnx` and `candle-datasets` depend downward as needed.
`candle-pyo3` wraps whichever layers it needs without influencing them.
`candle-dispatch` (Phase 5 Tier 4, long-term) sits between `candle-core` and
new user-facing op-sequence APIs, with no effect on any layer above or below.
