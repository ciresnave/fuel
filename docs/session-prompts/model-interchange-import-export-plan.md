# Model Interchange: Import / Export Plan

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
- The **graph side** is greenfield. **ONNX is the only flagship-worthy target**; the rest are FFI shims,
  deferred, or out of scope.
- A **native Fuel graph format** (Serde on the IR) is a cheap, high-value byproduct of doing export at all.

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

**Fuel-IR:** [fuel-graph/src/lib.rs](fuel-graph/src/lib.rs). `Op` enum ≈ **110+ primitive variants**
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

### F2 — Fuel-IR serialization (`fuel-graph`)
Add Serde (or a hand-rolled stable codec) to `Op`, `Node`, `Graph`, `OpParams`, side-tables, and the
fused-op IDs. This gives us:
- a **native `.fuel` graph format** (round-trips with zero loss — our own opset),
- the substrate every graph *exporter* writes from and every graph *importer* writes into.

Design notes:
- Version the format header; the fused-op registry IDs are the stable vocabulary — never renumber, only append.
- `storage_map` weights serialize **externally** (safetensors sidecar) by default — keep the graph file small
  and mmap-friendly; embed only on request. (Mirrors ONNX external-data and `.pt2`'s per-tensor blobs.)
- This is also the clean place to land graph CSE/validation on load.

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

### StableHLO — **export-via-FFI only (Tier 3)**
- **Container:** MLIR dialect; portable artifact = **MLIR bytecode** of the versioned **VHLO** dialect.
- **Opset:** ~100 small orthogonal compute ops, fully specified. Past v1.0; 5yr-back/2yr-forward compat.
- **Rust crates:** **effectively none.** No maintained Rust reader. `melior`/`mlir-sys` bind MLIR generally
  but **don't ship the StableHLO dialect** — you'd link StableHLO's C/C++ API yourself.
- **Verdict:** you **cannot hand-parse it** without an MLIR runtime. Treat as an **export** target via FFI
  to the StableHLO C API if/when XLA-ecosystem interop is wanted. Not an import path. Low priority.

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

Fuel-IR (~110 prims + 23 fused) ≠ ONNX (~187) ≠ Core ATen (~180) ≠ StableHLO (~100). Mapping is **not**
1:1:
- **Import** = decompose each external high-level op into a Fuel primitive sequence (e.g. ONNX `LSTM`,
  `GRU`, `Softmax(axis)`, `BatchNormalization` → primitive graphs; or recognize a known pattern and emit a
  `Fused(...)` directly when one exists, e.g. ONNX `LayerNormalization` → `Fused(LAYER_NORM_LAST_DIM)`).
- **Export** = the inverse, plus **re-fusion**: a Fuel `Fused(RMS_NORM…)` must lower to the target's
  primitive sequence (most targets lack a native RMSNorm op).
- **Round-trips are lossy.** Define a conformance matrix per format: which ops are 1:1, which decompose,
  which are unsupported (→ hard `Result` error at build time, listing the offending op — never silently drop).
- **Strategy:** build a declarative **op-mapping table** crate-side, table-driven both directions, with a
  differential-test harness (run Fuel vs `tract`/`onnxruntime` on the same ONNX, assert tensor parity).
  Quantized weights: preserve the scheme through import (store as const blocks / `Fused(QMATMUL)`), don't
  eagerly dequantize.

---

## 6. Proposed crate layout

- `fuel-formats` — stays the transport-agnostic **container** parser home (already is). Add ONNX protobuf
  parse here (pure data → structs), GGUF/safetensors **write** paths.
- **`fuel-interchange`** (new) — depends on `fuel-graph` + `fuel-formats`. Houses the **graph op-mappers**
  (ONNX↔IR, later ATen→IR, StableHLO export-FFI) and the native `.fuel` serde. Keeps op-mapping out of the
  lean `fuel-graph` core.
- `fuel-transformers` — gains the **arch registry** (F1) + `AutoModel::from_path`.
- `fuel-tokenizers` (new, thin) — wraps `tokenizers`/`kitoken`; one place for tokenizer ingest.

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
10. StableHLO **export** via FFI to the StableHLO C API (no import).
11. TFLite import (flatbuffer) if demand appears.

**Out of scope (documented, with rationale):** TorchScript (deprecated), CoreML (Apple-only, no Rust crate),
TensorRT `.plan` (kernel-baked, non-portable), TF SavedModel full-fidelity (FFI-only).

---

## 8. Open decisions for the user

- **Priority axis:** import (consume the world's models) vs export (emit Fuel models elsewhere)? The plan
  front-loads import (more immediate value); export rides the same op-mapper a step later.
- **ONNX dependency stance:** hand-roll the protobuf parse (zero-backend-dep, matches `fuel-formats`
  philosophy, recommended) vs depend on `candle-onnx`/`tract` for the mapper (faster start, heavier dep).
- **Coverage bar for v1 ONNX import:** "transformer + CNN inference ops" (~40 ops, fast) vs "broad opset"
  (long tail of rarely-used ops). Recommend the former, expand on demand (`no_consumer_not_a_reason` cuts
  both ways — but a 187-op spec has a genuinely cold tail).
```
