# Model Interchange: Import / Export Plan

> **[2026-06-15 banner]** [`docs/architecture/13-interchange.md` v0.3](../architecture/13-interchange.md) is now **AUTHORITATIVE** for interchange; the sections below predate the 2026-06-14 "plan IS the graph" reconciliation and are retained as the **active migration plan** (crate moves, format dossiers, phased roadmap, caller fixes — 13-interchange.md cites this file by path). The concrete stale claims have been reconciled in place (StableHLO is now import+export; the native `.fuel` is the whole-graph serialization; load-vs-import distinction added; primitive count ~80–90); where this file and 13-interchange.md v0.3 still differ in framing, **v0.3 wins.**
>
> **Reconciled 2026-06-15** against the 2026-06-14 redirection + current git: StableHLO promoted to import+export, native `.fuel` = whole graph (base map + storage + optimized paths) mmap-backed via 11-persistence, load-vs-import distinction added, primitive count set to ~80–90.

**Status:** Research + plan (2026-06-04). Not yet started.
**Goal:** Let Fuel *read* a model architecture (the operation sequence, not just weights) from
as many external sources as practical, and *write* a Fuel model out to formats other engines
can consume. Fuel-IR is the hub; every format is an importer and/or exporter spoke.

---

## 0. The honest reframe (read this first)

The popular framing ("there are ~4 ways models are distributed; support all of them") conflates two
fundamentally different problems and includes one category that doesn't apply to Fuel. Disentangling
them is the whole design:

### Two problems, not one

| | **Weight + architecture-tag formats** | **Graph formats** |
|---|---|---|
| **Examples** | safetensors+`config.json`, GGUF, PyTorch `.pt/.bin`, HF transformers source | ONNX, StableHLO, `torch.export` (`.pt2`), TorchScript, TFLite, CoreML |
| **What's in the file** | tensors + metadata + a **named architecture tag** (`"llama"`) | an actual **DAG of operators** |
| **The "code" is…** | *implicit* — the runtime must already implement that architecture | *explicit* — encoded as nodes |
| **What Fuel must build** | an **arch-tag → model-builder registry** (+ container parsers, mostly done) | a **graph op-mapper** (external opset ↔ Fuel-IR) |
| **Can you "parse the architecture"?** | **No.** `config.json` is hyperparameters; the forward pass is arbitrary Python. You hand-implement each architecture. | **Yes**, for the parseable ones (ONNX). |

The single most important consequence: **"support HuggingFace transformers" is not a parser.** There is
no generic way to translate an arbitrary Python `forward()` into a graph without tracing/executing it.
"Supporting" HF means *maintaining a zoo of hand-written architectures keyed by `model_type`* — which is
exactly what Fuel's eager-retirement program already produces. GGUF is the same problem for the *code*
part: its architecture is an implicit tag that indexes into that same zoo.

### The category that doesn't apply

"Code generation / JIT engines (Triton, TVM)" is a red herring for *model interchange*. Those generate
**kernels**, not distribute **architectures** — nobody ships a model "as a Triton program." Fuel itself
*is* a JIT/dispatch engine; this is something Fuel **is**, not something it ingests. The legitimate modern
members of that bucket are `torch.export`'s ATen graph and StableHLO, which are graph formats (above).

### Net

- The **weight side** is ~60% built. The missing piece is a **registry**, not more parsers.
- The **graph side** is greenfield. **ONNX is the flagship import+export target** (the only multi-producer,
  fixed-versioned, Rust-parseable graph standard). **StableHLO is the high-value second graph target — import
  *and* export** (it is the JAX/TF/XLA convergence point, so one importer covers all three; both directions
  via FFI, which is why it sits behind ONNX). The remaining graph formats are deferred or out of scope.
  *(Reconciled 2026-06-15: StableHLO was "FFI shim / export-only"; v0.3 makes it an import path too.)*
- The **native `.fuel` whole-graph format** reuses [11-persistence](../architecture/11-persistence.md)'s
  serialization (base map + storage + optimized paths, mmap-backed in-file) — **not** Serde-on-the-IR, and
  not a byproduct of export. It is the **load** path; conversion from foreign formats is **import**.

---

## 1. Current state of Fuel (audited 2026-06-04)

**Container parsers — already exist** in [fuel-formats/src/](fuel-formats/src/), transport-agnostic
(`impl Read + Seek` / `&[u8]`, no backend/`Tensor` deps):
- `safetensors.rs`, `gguf.rs` (mmap, v2/v3), `ggml.rs` (legacy), `pickle.rs` (`.pt/.bin`), `imatrix.rs`.

**Weight gateway:** `VarBuilder` in [fuel-nn/src/var_builder.rs](fuel-nn/src/var_builder.rs) — uniform
`.get(shape, name)` over safetensors/`HashMap`/`VarMap`/custom. Architecture constructors take a `VarBuilder`.

**Architecture zoo:** ~65 models in [fuel-transformers/src/models/](fuel-transformers/src/models/)
(44 LLM, 21 vision, 8 encoder, 8 audio, 4 diffusion, 15+ multimodal). Each is a Rust module with a
`Config: Deserialize` struct + a `from_hf_json_str` + a `Model::new(config, vb)` constructor.
**There is no registry** — the user manually imports the right model type. ← key gap.

**Fuel-IR:** [fuel-graph/src/lib.rs](fuel-graph/src/lib.rs). The canonical primitive vocabulary is
**~80–90 primitives** ("comparable to stablehlo's primitive set" per [03-ir](../architecture/03-ir.md);
the as-audited `Op` enum carried more variants at fine granularity, 2026-06-04)
(elementwise, reductions, shape/dtype, indexing, matmul, scalar, cross-device, multi-output `View`/`ViewOwned`)
+ one `Fused(FusedOpId, params)` arm backed by a **23-entry fused-op registry** (RMSNorm, RoPE, FlashAttn,
QMatMul, SelectiveScan, …). `Graph` is a node arena + side-tables (`storage_map`, `layouts`,
`placements`, `target_backends`, `multi_outputs`, `side_effect_roots`). **No Serde derives; no on-disk
graph format; no graph export of any kind.** ← second key gap.

**Tokenizers:** ad-hoc — `hf_hub` downloads `tokenizer.json` in [fuel-core/src/lazy.rs](fuel-core/src/lazy.rs).
GGUF-embedded vocab is parsed by `fuel-formats` but not wrapped in a high-level API. No native codec.

---

## 2. The two missing foundations (do these first — they unlock everything else)

### F1 — Architecture registry (`fuel-transformers`)
A `model_type`/`general.architecture` (lowercased string) → builder map. This is what turns "65 models"
into "load almost any supported model from a path." Unify the GGUF tag space and the HF `model_type` space
onto **one** string key (they already align: `"llama"`, `"falcon"`, `"mamba"`, …).

```
register_arch!("llama", LlamaConfig, LlamaModel);   // config struct + constructor
// AutoModel::from_path(p) -> reads config.json or GGUF KV -> looks up tag -> builds -> loads weights
```
- Honor `architectures[]` to disambiguate the task head (`LlamaForCausalLM` vs `…ForSequenceClassification`).
- Permissive deserialization is mandatory: field names vary across archs (`n_embd` vs `hidden_size`,
  optional `num_key_value_heads`, scattered `rope_scaling` schemas). Use `#[serde(default, alias=...)]`.
- Return `Result` and validate at build time (Fuel principle): unknown tag → a clear error listing
  supported tags, not a panic.

### F2 — Native `.fuel` whole-graph serialization (reuse 11-persistence)
The native format is **not** a new codec bolted onto `fuel-graph` — it **reuses
[11-persistence](../architecture/11-persistence.md)'s serialization** of the graph. A unified `.fuel` holds
the **whole graph**: the base map (canonical primitive DAG) **plus storage plus any optimized paths**
(branch-point alternatives), with storage **mmap-backed in-file** rather than split to an external
safetensors sidecar by default. This gives us:
- a **native `.fuel` whole-graph format** loaded losslessly with no conversion (round-trips our own opset),
- the canonical hub the graph *exporters* walk and the graph *importers* build into (the **base map**).

Design notes:
- This is the **load** path (`map_from_file` mmaps the whole graph straight into memory), distinct from the
  **import** path the rest of this document governs — see the load-vs-import note below.
- Reuse 11-persistence's base-map serialization; do **not** invent a second on-disk DAG format and do **not**
  frame this as "add Serde to `fuel-graph` as the native format." Storage classes (shared / session / transient)
  ride along per 11-persistence.
- For *distribution*, an exported `.fuel` is stripped to the base map (hardware-independent); a receiver
  validates-and-scoped-re-optimizes any included optimized paths, or re-optimizes from the base map on
  different hardware. Embedding vs externalizing storage is a per-export choice, not a fixed sidecar default.
- This is also the clean place to land graph CSE/validation on load.

> **Load vs import (redirection decision #4).** `map_from_file` **loads** the native `.fuel` whole-graph with
> no conversion (mmap, lossless). The `from_*` constructors (`from_gguf`, `from_safetensors`, ONNX, `.pt2`,
> StableHLO) **import** — they convert a foreign format into a base map (lossy, conversion path). Loading is
> fast and lossless; importing is what the per-format dossiers below cover.

---

## 3. Per-format dossiers (graph formats)

### ONNX — **flagship import + export target**
- **Container:** protobuf (`onnx.proto`, `ModelProto→GraphProto→NodeProto[]`). Trivial to parse with `prost`.
- **Opset:** fixed, versioned. Standard `ai.onnx` domain ≈ **~187 ops** (current opset ~26, ONNX 1.21).
  Each op is immutably specified per opset version.
- **Weights:** in-graph `initializer` `TensorProto`s **or** external-data sidecar (required for >2 GB —
  protobuf's hard limit). Importer **must** handle both.
- **Rust crates:** `tract`/`tract-onnx` (pure-Rust parser + engine, most mature; 0.21→0.22, note a
  0.20→0.21 compat break), `candle-onnx` (pure-Rust ONNX→candle graph; partial op coverage — a direct
  template for ONNX→fuel-graph), `ort` (bindings to MS onnxruntime 2.x rewrite — **runs** models but hides
  the graph; not what we want).
- **Recommended approach:** **parse the protobuf ourselves with `prost` on `onnx.proto`** (keeps
  `fuel-formats`' zero-backend-dep philosophy), then map nodes → Fuel-IR. Mirror `candle-onnx`'s op table.
  Do **not** depend on `tract`/`ort` for the graph — only as an oracle for differential testing.
- **Gotchas:** (1) op semantics are **opset-version-dependent** — branch on the model's `opset_import`;
  (2) **shape inference is not in the file** — only graph I/O shapes are guaranteed; you must infer the rest
  (Fuel's build-time validation helps here); (3) external-data resolution.

### `torch.export` / `ExportedProgram` (`.pt2`) — **deferred import (Tier 2)**
- **Container:** ZIP — `/models/*.json` (the graph), `/data/weights/` (one raw blob per tensor + a JSON
  index), `/data/constants/`, `/extra/`. Readable in Rust *in principle* (unzip + JSON + raw reads).
- **Opset:** FX graph in ATen IR. **Two tiers:** full ATen (~2000 ops, version-sensitive) **or** **Core
  ATen (~180 ops)** after `run_decompositions()`. **Only ingest Core ATen** and document it — the same
  `.pt2` can contain wildly different op granularity depending on whether the producer decomposed.
- **Rust crates:** **none.** Python-only today; we'd write the first reader against an **evolving,
  PyTorch-version-tied schema** (not an independently-specified contract). Pin a PyTorch version.
- **Verdict:** strategically correct future PyTorch path (TorchScript is dead), but high-maintenance and
  greenfield. Tier 2, after ONNX proves the op-mapper.

### StableHLO — **import + export via FFI (Tier 3)**
- **Container:** MLIR dialect; portable artifact = **MLIR bytecode** of the versioned **VHLO** dialect.
- **Opset:** ~100 small orthogonal compute ops (~119 in the current spec), fully specified — a near match for
  fuel's ~80–90 primitives. Past v1.0; 5yr-back/2yr-forward compat. See the worked op map
  ([docs/interchange/stablehlo-to-fuel-op-map.md](../interchange/stablehlo-to-fuel-op-map.md)).
- **Rust crates:** **effectively none.** No maintained Rust reader. `melior`/`mlir-sys` bind MLIR generally
  but **don't ship the StableHLO dialect** — you'd link StableHLO's C/C++ API yourself.
- **Verdict:** you **cannot hand-parse it** without an MLIR runtime, so **both directions ride FFI to the
  StableHLO C API** (the portable artifact is MLIR/VHLO bytecode). It is now an **import target as well as
  export**: StableHLO is the convergence point JAX, TensorFlow, and PyTorch/XLA all lower to, so one importer
  transitively covers all three producers — notably the only clean path to **JAX** models. Export is natural
  from the base map (also ~100 primitives). The MLIR-runtime requirement keeps it behind ONNX in priority
  despite the broad reach. Control flow, dynamic shapes, and collectives are handled by import-time lowering /
  other layers, not rejected (see the op map).

### TorchScript — **skip (deprecated)**
Officially deprecated as of PyTorch ~2.9–2.10; PyTorch directs users to `torch.export`. Format embeds
Python pickle + JIT bytecode whose semantics live in the C++ interpreter — **unparseable outside libtorch**.
Do not invest; tell users to re-export as `.pt2` or ONNX.

### TF SavedModel / GraphDef, TFLite — **Tier 3 / FFI**
SavedModel = protobuf `GraphDef` (~1000+ loosely-versioned ops, no tight spec) + checkpoint shards.
TFLite = clean FlatBuffer, ~150 builtin ops (watch custom/Flex ops). No mature pure-Rust graph parser
(`tract` ingests *some*). TFLite's flatbuffer is the more tractable of the two if demand appears.

### CoreML (`.mlmodel`/`.mlpackage`) — **out of scope unless Apple-deploy demand**
Protobuf; two parallel op sets (legacy NeuralNetwork layers vs modern **MIL**). `.mlpackage` splits weights
out of the protobuf. No mature Rust crate — you'd `prost`-gen from Apple's `.proto`. Apple-platform-only.

### TensorRT engines — **explicitly out of scope (not an interchange format)**
A `.plan` is an opaque, kernel-baked artifact tied to exact GPU arch + TRT version + OS. **No graph, no
portability.** The correct flow is ONNX → TensorRT compiles it; never parse `.plan` files.

---

## 4. Weight + tokenizer formats (mostly done; gaps noted)

- **safetensors** — read+write via official `safetensors` crate (v0.7; now a **PyTorch Foundation**
  project). Dtype enum has grown FP8 (E5M2/E4M3/E8M0) + MX sub-byte (F4/F6) — quantized checkpoints now
  ship *as* safetensors with **separate scale tensors** + a `_quantization_metadata`/`quantization_config`
  recipe. Fuel reads tensors today; **gap = interpreting the quant recipe** (a Fuel-side interpreter).
- **GGUF** — read+write; `general.alignment` **defaults to 32 when absent** (don't hardcode); `Q4_K_M` is a
  *filename mix label*, not a ggml type ID (per-tensor types carry the real scheme). Fuel reads; confirm the
  **write** path for export.
- **PyTorch pickle** — **read-only legacy ingest, transcode to safetensors immediately**; pickle is
  arbitrary-code-execution on load. Fuel's parser already interprets only tensor-rebuild opcodes. Never write.
- **`config.json`** — `model_type` (stable lowercase dispatch key) + `architectures[]` (task head) +
  nested `quantization_config`. Feeds F1.
- **Tokenizers (gap → native codec):** the `tokenizers` crate is **Rust-native, first-party** — adopt it as
  the spine (no Python). Add SentencePiece (`rust-tokenizers`/`sentencepiece`) and tiktoken (`tiktoken-rs`)
  coverage; evaluate **`kitoken`** (2025) as a single dep covering HF + SP + tiktoken + Tekken. GGUF-embedded
  vocab needs custom merge/pre-tok reconstruction.

---

## 5. The op-impedance problem (the core engineering risk)

Fuel-IR (~80–90 prims + 23 fused) ≠ ONNX (~187) ≠ Core ATen (~180) ≠ StableHLO (~100/~119 spec). Mapping is
**not** 1:1:
- **Import** = decompose each external high-level op into a Fuel primitive sequence (e.g. ONNX `LSTM`,
  `GRU`, `Softmax(axis)`, `BatchNormalization` → primitive graphs; or recognize a known pattern and emit a
  `Fused(...)` directly when one exists, e.g. ONNX `LayerNormalization` → `Fused(LAYER_NORM_LAST_DIM)`).
- **Export** = the inverse, plus **re-fusion**: a Fuel `Fused(RMS_NORM…)` must lower to the target's
  primitive sequence (most targets lack a native RMSNorm op).
- **Round-trips are lossy.** Define a conformance matrix per format giving every source op a *disposition*:
  1:1 primitive, decomposition, fused-op recognition, **import-time lowering** (control flow / dynamic shapes
  / collectives → graph structure, per *representation ≠ op*), or *another Fuel layer*; only a construct with
  no graph representation at all is a hard `Result` error at build time, naming the offending op — never a
  silent drop. The worked example is the **StableHLO op map**
  ([docs/interchange/stablehlo-to-fuel-op-map.md](../interchange/stablehlo-to-fuel-op-map.md)): of 119 spec
  ops, ≈100 are covered or handled, the hard-reject set is nearly empty, and it names the genuine vocabulary
  gaps (sort/top-k, pooling, FFT, inverse-trig, product-reduce).
- **Strategy:** build a declarative **op-mapping table** crate-side, table-driven both directions, with a
  differential-test harness (run Fuel vs `tract`/`onnxruntime` on the same ONNX, assert tensor parity).
  Quantized weights: preserve the scheme through import (store as const blocks / `Fused(QMATMUL)`), don't
  eagerly dequantize.

---

## 6. Crate layout (FINALIZED — see [architecture §13](../architecture/13-interchange.md) + [§02](../architecture/02-layers.md))

Three core+leaf tiers, split on the weight⊥graph axis. Per-format and high-demand-model leaves are
separate crates (not feature gates); the long model tail stays in an umbrella and is extracted lazily.

**Format tier — IR-free byte parsing** (`fuel-core-types` only):
- `fuel-formats` — shared substrate (transport traits, dtype map, errors). Already exists.
- `fuel-format-safetensors`, `-gguf`, `-pickle`, `-onnx`, … — one per format, bytes → format structs.
  `fuel-format-onnx` is the parse half of the retired `fuel-onnx` placeholder.

**Interchange tier — translate ↔ the base map:**
- `fuel-interchange-weights` — named tensors + dtype + **quant recipe** → `Storage`. Used by *every*
  interchange leaf (graph formats load weights too). The quant interpreter lives here, once.
- `fuel-interchange-graph` — op DAG ↔ base map: op-map helpers, decomposition/fusion-recognition,
  conformance matrix, and **native-format read/write that reuses [11-persistence](../architecture/11-persistence.md)'s
  base-map serialization** (no new DAG format).
- `fuel-format-interchange-{onnx,gguf,safetensors,pickle,…}` — per-format leaves owning the one
  un-hoistable thing: the **node↔weight binding**. Weight-only leaves depend on the weights core; graph
  leaves depend on both cores + their `fuel-format-*` peer.

**Model tier:**
- `fuel-model-core` — `Model` trait, `model_type` → builder **registry** via link-time `inventory`/`linkme`,
  `AutoModel::from_path`, and the imported-graph→known-arch recognizer.
- `fuel-model-*` — one architecture per crate; self-registering. **Pre-split now:** llama, qwen, mistral,
  gemma, phi, mamba, whisper, bert, clip/siglip, stable-diffusion/flux. Generic blocks (RoPE, RMSNorm,
  GQA, SwiGLU) stay in `fuel-nn`; shared-component crates only on real cross-model duplication.
- `fuel-transformers` — optional umbrella re-exporting `fuel-model-*` behind features; home for the
  lazy long tail until each is extracted under real pressure.

**Dev tooling (not runtime):**
- `fuel-codegen` — CLI scaffolder: source AST (+ optional trace oracle) → draft self-registering
  `fuel-model-*` crate. Reuses `fuel-interchange-graph`'s op-map (a traced model is ATen). Heavy parse
  deps stay out of the runtime graph.
- `fuel-tokenizers` — thin wrapper over the Rust-native `tokenizers` crate (+ `kitoken` for SP/tiktoken).

## 6b. Migration tranches (do in order)

**T0 — Tier seam + registry (prerequisite; justified now).** Establish `fuel-interchange-weights` +
`fuel-interchange-graph` cores, split `fuel-formats` → `fuel-format-*` for the existing parsers
(safetensors/gguf/pickle), stand up `fuel-model-core` with `inventory` registration + `AutoModel::from_path`,
move the existing weight-load logic (`VarBuilder`/`LazyTensor::from_safetensors`) behind
`fuel-interchange-weights`. Correct callers (bounded — only format/model-loading sites move). Reconcile the
native format against 11-persistence's base-map serialization. **No new format capability yet — this is the
homes the new work needs.**

**T1 — Validate the seam with one vertical slice.** Native `.fuel` base-map round-trip *or* the first
`fuel-format-onnx` import. Proves the op-map/binding seam before committing to breadth.

**T2 — Pre-split high-demand models.** Extract the ten `fuel-model-*` crates above from `fuel-transformers`
into self-registering crates; umbrella re-exports them behind features. Caller fixes are mechanical.

**T3 — Breadth + scaffolder.** ONNX import/export coverage; `fuel-codegen` (rides T1's op-map); `.pt2`
Core-ATen import; StableHLO export via FFI. The per-model long-tail explosion happens here, *via* the
scaffolder and lazy extraction — never as a big-bang.

**Timing note:** T0/T2 are mechanical, high-churn, low-capability moves — schedule them when they won't
collide head-on with other in-flight branches (storage-unification, etc.). Workspace path-deps + shared
version keep cross-cutting changes to "edit N files, one `cargo build`"; defer any independent crates.io
publishing until a model/format crate stabilizes.

---

## 7. Phased roadmap

**Tier 0 — Foundations (unlock everything; do first)**
1. F1: architecture registry + `AutoModel::from_path` (HF safetensors+config.json and GGUF onto one tag space).
2. F2: Serde on Fuel-IR + native `.fuel` graph format (external-weights sidecar) + round-trip tests.
3. `fuel-tokenizers`: adopt `tokenizers` crate; GGUF-vocab reconstruction.

**Tier 1 — ONNX (highest external value)**
4. ONNX protobuf parse (`prost` on `onnx.proto`) in `fuel-formats`; external-data handling.
5. ONNX **import** op-mapper in `fuel-interchange` (start with the ~40 ops that cover transformer/CNN
   inference; opset-version branching; decomposition + fuse-recognition).
6. ONNX **export** (Fuel-IR → ONNX, with fused-op lowering).
7. Differential-test harness vs `tract`/`onnxruntime`. Conformance matrix.

**Tier 2 — PyTorch-native graph**
8. `torch.export` `.pt2` **import**, **Core ATen only**, pinned PyTorch version. Reuse the op-mapper.
9. GGUF **export** (write path) if not done in Tier 0.

**Tier 3 — Compiler-IR + TF (interop, lower demand)**
10. StableHLO **import + export** via FFI to the StableHLO C API. One importer covers JAX/TF/PyTorch-XLA
    (their shared lowering target — the only clean path to JAX); export is natural from the base map.
11. TFLite import (flatbuffer) if demand appears.

**Out of scope (documented, with rationale):** TorchScript (deprecated), CoreML (Apple-only, no Rust crate),
TensorRT `.plan` (kernel-baked, non-portable), TF SavedModel full-fidelity (FFI-only).

---

## 8. Decisions

**Resolved (2026-06-08, recorded in [architecture §13](../architecture/13-interchange.md) + decisions-log):**

- **Weight⊥graph separation** — interchange splits into weight + graph; format tier stays IR-free.
- **Hub** — the base map; no second neutral IR.
- **Native format** — reuses 11-persistence's serialization; **not** a new format and **not** Serde-on-`fuel-graph`.
  The unified `.fuel` is the **whole graph** (base map + storage + optimized paths), storage **mmap-backed
  in-file** (no external safetensors sidecar by default). `map_from_file` **loads** it with no conversion;
  the `from_*` constructors **import** (convert a foreign format → base map). *(Reconciled 2026-06-15 to v0.3:
  was "base-map serialization" / external-sidecar framing.)*
- **Crate granularity** — per-format + per-model leaves (not feature gates); link-time `inventory`
  registration; optional `fuel-transformers` umbrella; pre-split high-demand models, lazy long tail.
- **Sequencing** — tier seam + registry (T0) → validate with one importer (T1) → pre-split models (T2)
  → breadth + scaffolder (T3). The per-model explosion rides the scaffolder, not a big-bang.

**Still open — defer to when T1/T3 start (recommendations noted):**

- **Priority axis:** import vs export first. *Recommend import-first* (more immediate value); export rides
  the same op-mapper a step later.
- **ONNX dependency stance:** hand-roll the protobuf parse (zero-backend-dep, matches the IR-free format
  tier — *recommended*) vs depend on `candle-onnx`/`tract` for the mapper (faster start, heavier dep).
- **v1 ONNX import coverage:** "transformer + CNN inference ops" (~40 ops — *recommended*) vs "broad opset"
  (the genuinely cold ~187-op tail). Expand on demand.
```
