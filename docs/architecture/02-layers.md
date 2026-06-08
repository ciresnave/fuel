# Layers and crate boundaries

**Status**: v0.3 (draft, 2026-06-08). v0.3 changes: the IO and Models layers gain explicit crate tiers for model interchange (per [13-interchange](13-interchange.md)) — a format tier (`fuel-formats` core + IR-free `fuel-format-*` leaves), an interchange tier (`fuel-interchange-weights` + `fuel-interchange-graph` cores + per-format `fuel-format-interchange-*` binding leaves), and a model tier (`fuel-model-core` registry + `fuel-model-*` leaves). v0.2 changes: fuel-loaders explicitly supports remote sources (HuggingFace Hub, GitHub, HTTPS) with sibling-cache and tolerance-recipe auto-discovery alongside the loaded model file.

How fuel is decomposed into crates, what each crate owns, and which way dependencies flow. Anchored in the existing ROADMAP layer model; this section pins the architectural intent so phase work doesn't drift.

---

## The layer model, restated

Fuel is seven conceptual layers stacked downward. Dependencies flow only downward — no lower layer may depend on a higher one. Users can stop at exactly the layer they need; each layer is independently usable.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│  Use-Case Orchestration                                                    │
│  fuel-inference, fuel-training (leaf crates — nothing depends on either)   │
├────────────────────────────────────────────────────────────────────────────┤
│  Models                                                                    │
│  fuel-model-core (registry, AutoModel), fuel-model-* (one per model;        │
│  common models pre-split, long tail lazy), fuel-transformers (umbrella)     │
├────────────────────────────────────────────────────────────────────────────┤
│  Interchange                                                               │
│  fuel-interchange-weights, fuel-interchange-graph (cores),                  │
│  fuel-format-interchange-* (per-format binding: onnx, gguf, safetensors…)  │
├────────────────────────────────────────────────────────────────────────────┤
│  IO / Formats                                                              │
│  fuel-formats (core), fuel-format-* (one per format, IR-free parse),        │
│  fuel-loaders (transport adapters incl. remote sources)                     │
├────────────────────────────────────────────────────────────────────────────┤
│  NN                                                                        │
│  fuel-nn (layers, losses, optimizers, parameter utilities, initialization) │
├────────────────────────────────────────────────────────────────────────────┤
│  Foundation                                                                │
│  fuel-tensor, fuel-autograd, fuel-graph, fuel-graph-router,               │
│  fuel-graph-executor, fuel-storage, fuel-core-types                        │
├────────────────────────────────────────────────────────────────────────────┤
│  Backends / Kernels                                                        │
│  fuel-cpu-backend, fuel-cuda-backend, fuel-vulkan-backend,                  │
│  fuel-metal-backend, fuel-aocl-cpu-backend, fuel-mkl-cpu-backend,         │
│  fuel-cuda-kernels, fuel-flash-attn-cuda, fuel-conv, fuel-quantized       │
└────────────────────────────────────────────────────────────────────────────┘
```

The Foundation layer is where most of this architecture-doc set lives. The IR (03), the optimizer (04), the runtime (06), the persistence layer (11) are all Foundation concerns. Backends (05) sit beneath. Higher layers consume the Foundation surface but don't shape it.

## Two architectural rules that keep the layering clean

**Rule 1: dependencies flow downward only.** Enforced via Cargo's dep graph. A leaf-layer crate (fuel-inference) can depend on Models, IO, NN, Foundation, Backends. A Foundation crate cannot depend on Models or higher. Violations are caught at build time.

**Rule 2: Foundation is the substrate; backends are extensions.** The Foundation layer defines the architecture's commitments (the IR, the optimizer's surface, the binding-table catalog, the cost-model interface). Backends implement the contract the Foundation specifies. Adding a backend doesn't require Foundation changes; adding a primitive Op or a foundational concept may require backend updates to honor it.

This is what makes "compile fuel without CUDA" or "ship fuel-cpu-only-binary" work cleanly: backends are Cargo features; the Foundation works without any of them (the reference backend is always available).

## Crate boundaries inside Foundation

Foundation has more crates than the diagram shows; the boundaries inside it are load-bearing. The relevant ones for this architecture doc set:

- **fuel-core-types**: dtypes, shapes, layouts, errors, the dispatch-key types, the `BackendCapabilities` shape. Zero backend dependencies. Re-exported through fuel-core for ergonomics.
- **fuel-graph**: the `Op` enum (primitive variants + `Op::Fused` arm), `Node`, `Graph`, `FusedOpRegistry` metadata types, `OptimizationMap` rules, the rule engine. Depends on fuel-core-types only.
- **fuel-storage**: the binding-table catalog, `KernelRef` ABI, `OpParams`, dispatch wrappers that bridge dispatch-erased `Storage` to backend-typed kernels. Depends on fuel-graph + fuel-core-types + per-backend crates.
- **fuel-graph-executor**: walks the optimized form, dispatches pre-resolved KernelRefs, manages slot assignment from current backend telemetry. Where 06-runtime's commitments live.
- **fuel-graph-router**: multi-backend dispatch surface; reads BackendCapabilities and current telemetry to pick devices for ops the optimizer hasn't placed yet.
- **fuel-tensor** + **fuel-autograd** (post-fission per Phase 7.5 work item E): the user-facing handle + autograd story. fuel-tensor wraps fuel-graph; fuel-autograd does graph-rewrite-as-backward.
- **fuel-conv**, **fuel-quantized**: ops that warrant their own crates because they have substantial standalone value (conv has its own kernel ecosystem; quantization has its own dtype family). Both are Foundation-layer despite being "ops" because they define types that Foundation-layer consumers use.

The fission decisions in Phase 7.5 (work item E) are about cleaving consumer boundaries: Lightbulb (inference-only consumer) wants fuel-tensor without fuel-autograd; mlmf (network IPC consumer) wants fuel-formats without fuel-loaders. Each split is justified by a class of consumer that uses one side and not the other.

**fuel-loaders supports remote sources.** In addition to local-filesystem model loading, fuel-loaders supports common remote sources via URI schemes: `hf://owner/repo` (HuggingFace Hub via the `hf-hub` Rust crate), `github://owner/repo/path` (GitHub via raw.githubusercontent.com), `https://...` (any HTTPS-accessible model file). When loading from a remote source, the loader auto-discovers sibling cache files and tolerance recipes at the same location and uses any that match the user's environment fingerprint (per [11-persistence §Cache generation and distribution](11-persistence.md#cache-generation-and-distribution)). On miss, the loader downloads only the model file; fuel falls back to local optimization. The remote-source layer is leaf functionality — it depends on fuel-formats for parsing what it downloaded; it doesn't shape Foundation-layer concerns above it.

## Crate boundaries at the IO and Interchange layers

Model import/export ([13-interchange](13-interchange.md)) splits along the weight-vs-graph axis into three tiers. Each tier is a *core + leaves* family, mirroring `fuel-formats` : `fuel-format-*`.

**Format tier (IR-free byte parsing).** `fuel-formats` holds the shared substrate (the `Read`/`Seek` transport traits, dtype mapping, the common error type). Each format is a leaf — `fuel-format-safetensors`, `fuel-format-gguf`, `fuel-format-pickle`, `fuel-format-onnx`, … — that parses bytes into format-native structs. **No leaf depends on `fuel-graph` or any storage type**; the tier knows only `fuel-core-types`. This is what lets a safetensors-only consumer pull almost nothing. `fuel-format-onnx` is the parse half of the former `fuel-onnx` placeholder; the map half moves up to the interchange tier.

**Interchange tier (translate ↔ the IR).** Two cores:

- `fuel-interchange-weights` — named tensors + dtype + **quantization recipe** → fuel `Storage`. Depended on by *every* interchange leaf, weight-only and graph alike (a graph format still loads its weights). The quant interpreter lives here once.
- `fuel-interchange-graph` — operator DAG ↔ the base map ([03-ir](03-ir.md)): the op-map helpers, the decomposition/fusion-recognition library, the conformance-matrix machinery, and the native-format read/write (which reuses [11-persistence](11-persistence.md)'s base-map serialization — no new DAG format).

Each `fuel-format-interchange-*` leaf depends on its `fuel-format-*` peer plus the relevant core(s), and owns the one thing that can't be hoisted: the **per-format node↔weight binding** (ONNX by name, `.pt2` by blob index, TFLite by buffer index). A weight-only leaf needs only the weights core; a graph leaf needs both.

## Crate boundaries at the Models layer

The model tier is `fuel-model-core` + `fuel-model-*` leaves, with `fuel-transformers` retained as an optional umbrella.

- `fuel-model-core` — the `Model` trait, the `model_type`/`general.architecture` → builder **registry**, and `AutoModel::from_path`. Registration is **link-time distributed** (`inventory`/`linkme`): merely depending on a `fuel-model-*` crate makes it appear in the registry, so "no feature gates" works and the scaffolder can emit a self-registering crate without editing a central dispatch file. The imported-graph → known-architecture *recognizer* ([13-interchange](13-interchange.md)) lives here, since it depends on both the registry and the interchange tier.
- `fuel-model-*` — one architecture per crate. Generic building blocks (RoPE, RMSNorm, GQA attention, SwiGLU MLP) stay in `fuel-nn`, not in a per-family crate; a shared-component crate is extracted only on *real* duplication across ≥2 model crates, never as a speculative family taxonomy (multimodal models span "families," so families aren't a partition).
- `fuel-transformers` — an optional umbrella re-exporting the `fuel-model-*` crates behind features, so a consumer gets *either* granular (`cargo add fuel-model-llama`) *or* batteries-included, and nobody is forced into feature assembly.

**Applying the stopping rule (below) to this tier.** Per-format leaves and the high-demand `fuel-model-*` crates (Llama, Qwen, Mistral, Gemma, Phi, Mamba, Whisper, BERT, CLIP/SigLIP, Stable Diffusion/Flux) are split **now**: their single-format / single-model consumers are near-certain, so the split lands under real pressure. The long tail of architectures stays in the umbrella and is **extracted lazily** — when a real single-model consumer appears, or when the scaffolder emits a new model as its own crate. Big-banging all ~65 existing models into separate crates up front would be the speculative split the stopping rule rejects.

## Crate boundaries at the Backend layer

Each backend is its own crate, each gated by a Cargo feature on whichever consumer wants it. The pattern:

- **Backend crate** (e.g., `fuel-cuda-backend`): typed kernels operating on the backend's concrete storage type. No fuel-storage dependency.
- **Backend-side wrapper module in fuel-storage** (e.g., `fuel_storage::dispatch::cuda`): dispatch wrappers that pattern-match `BackendStorage::Cuda(...)`, extract the typed storage, call the backend's typed kernel. fuel-storage depends on the backend crate; backend doesn't depend on fuel-storage. Cycle avoided.
- **Hardware FFI wrappers** (e.g., `baracuda` for CUDA, `vulkane` for Vulkan, `aocl-blas-rs` for AOCL): live outside fuel entirely. Backend crates depend on these via crates.io. Fuel itself never names raw FFI.

This is how fuel can support CUDA, Vulkan, Metal, AOCL, MKL as independent compile-time-optional backends without coupling them to each other or to the Foundation layer's identity.

## What's not in the layer model

Three concerns that span layers rather than fitting into one:

- **The optimization-cache and tolerance-recipe artifacts ([11-persistence](11-persistence.md))**: produced by Foundation, consumed by Foundation, but their format is a Foundation/Backend concern (cache embeds backend kernel hashes, hardware fingerprints). They're cross-cutting; the persistence section treats them holistically.
- **Empirical Judge profile data**: produced by Foundation (the Judge), consumed by Foundation (the optimizer + route picker), measured against Backends. Cross-cutting.
- **Pattern-harvest telemetry ([08-pattern-harvest](08-pattern-harvest.md))**: produced by Foundation (the optimizer reads the base map for harvest); consumed by the project's server (outside the layer model entirely).

These don't break the layer model — they're consistent with it. They just don't fit cleanly inside any one layer.

## Stopping rule for new crates

A new crate is justified only when there is a class of consumer that uses one side and not the other. Indicators:

- The consumer needs the included surface and not the excluded surface.
- The fission unlocks compile-time leanness, IPC scenarios, or independent ecosystem development.
- The split is in the dependency graph, not just in the file layout.

A speculative split (e.g., "let's separate this just in case someone wants it standalone") is rejected. Every Foundation-layer crate that exists today emerged from a real consumer pressure; new ones land the same way.

---

## See also

- [03-ir](03-ir.md) — the IR types live in fuel-graph (Foundation layer).
- [05-backend-contract](05-backend-contract.md) — what backends advertise to the Foundation layer.
- [11-persistence](11-persistence.md) — cross-cutting artifact concerns.
- [13-interchange](13-interchange.md) — the import/export model the IO / interchange / model tiers implement.
- ROADMAP §"Layer Model" — the original layer-model diagram and dependency-direction commitment.
- ROADMAP §"Phase 7.5 — Core simplification" — work item E (fission) decisions.
