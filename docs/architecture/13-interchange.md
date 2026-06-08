# Model interchange: import and export

**Status**: v0.2 (draft, 2026-06-08). v0.2 changes: StableHLO promoted from export-only to **import + export** (it is the JAX/TF/XLA convergence point); the op-map model reframed around per-op *dispositions* and the *representation ≠ op* principle (control flow / dynamic shape / collectives are lowered to graph structure or handled at another layer, not rejected); links the worked StableHLO op map. v0.1: first statement of the import/export architecture. Design pass — no code. Anchors to [02-layers](02-layers.md) (crate tiers), [03-ir](03-ir.md) (the base map is the hub), and [11-persistence](11-persistence.md) (the native format reuses the base-map serialization).

How a model defined elsewhere becomes a fuel graph, and how a fuel graph becomes a model another engine can run. The IR — specifically the base map ([03-ir §The base map](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained)) — is the hub; every external format is an importer spoke, an exporter spoke, or both.

---

## The two-axis decomposition

The popular framing ("there are a handful of model-distribution formats; support them") conflates two concerns that vary **independently**. Every real format decomposes into three layers:

- **Container** — how bytes are framed (protobuf, flatbuffer, zip, MLIR bytecode, raw + sibling).
- **Weight payload** — named tensors + dtype + quantization recipe.
- **Graph payload** — the operation DAG, *or absent*.

These recombine in the wild, which is the architectural justification for separating them:

- **Same weights, different graph.** The logical weight payload (name → tensor) is architecture-agnostic. HuggingFace ships the same Llama as `pytorch_model.bin` *and* `model.safetensors`; llama.cpp re-containers it as GGUF. safetensors *cannot* name the architecture — proof the weight axis is independent of the graph/arch axis by construction.
- **Same graph, different weights.** ONNX carries weights as embedded `initializer`s *or* as external-data sidecars — one graph format, two weight backends, in one spec.

So interchange splits into **weight interchange** and **graph interchange**, and they are reused at different rates: weight handling is needed by *every* format (a graph format still has to load its weights); graph handling is needed only by graph formats.

### Two families of format

| | **Weight + architecture-tag formats** | **Graph formats** |
|---|---|---|
| Examples | safetensors+`config.json`, GGUF, PyTorch `.pt/.bin` | ONNX, StableHLO, `torch.export` (`.pt2`), TFLite |
| Contains | tensors + a **named architecture tag** (`"llama"`) | an actual operator DAG |
| The "code" is | *implicit* — the runtime must already implement that architecture | *explicit* — encoded as nodes |
| Fuel needs | the **model registry** ([02-layers](02-layers.md)) — a tag → builder map | a **graph op-mapper** ↔ base map |

The consequence that reorganizes everything: **a weight+tag format has no graph to parse.** `config.json` is hyperparameters; the forward pass is arbitrary host-language code. "Supporting" such a format is maintaining a zoo of hand-written (or scaffolder-generated) architectures keyed by `model_type`/`general.architecture` — *not* writing a parser. This is what the model tier ([02-layers](02-layers.md)) already is.

## The hub is the base map

Import builds a fuel graph; export walks one. The canonical meeting point is the **base map** ([03-ir](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained)): primitive-only, hardware-independent, deterministically derived from the user-facing form. Targeting the base map rather than inventing a second neutral IR is a hard commitment — **fuel's primitive `Op` vocabulary *is* the interchange vocabulary.** Its size (~80–90 primitives, "comparable to stablehlo's primitive set" per [03-ir](03-ir.md)) is what makes this tractable.

- **Import** = map each external operator to a fuel primitive *subgraph* (decompose), recognizing known patterns and emitting `Op::Fused(id, …)` directly where a registry entry exists (e.g. ONNX `LayerNormalization` → `Fused(LAYER_NORM_LAST_DIM)`).
- **Export** = the inverse, plus **re-fusion**: a fuel `Fused(RMS_NORM…)` lowers to the target's primitive sequence when the target lacks a native op. Exporting from the base map is the easy case — it is already primitive-only; StableHLO (also ~100 primitives) is a near-vocabulary-match.

**Round-trips are lossy, and that is stated, not hidden.** Each format carries a conformance matrix giving every source op a *disposition*: a 1:1 primitive, a decomposition, fused-op recognition, **import-time lowering**, or **another Fuel layer**.

***Representation ≠ op.*** A construct can live in the DAG without the op vocabulary naming it. A conditional is the clearest case: a constant predicate folds to the taken branch; a data-dependent, side-effect-free conditional becomes **predication** (compute both branches, `Op::Where`-select); a statically-bounded loop **unrolls** into nodes; a dynamic shape is **specialized** from concrete inputs. None adds a control-flow op; all are representable. So "fuel has no `if`/`while`/region-`reduce` op" is not "fuel can't import control flow" — the importer lowers it to graph structure. Other constructs are handled at a different layer entirely: tuples → multi-output bundles ([12-multi-output](12-multi-output.md)), ordering/async → the scheduler, quantize/dequantize → weight interchange, single-replica collectives → identity. Only a construct with *no graph representation at all* — an unbounded data-dependent side-effecting loop, an unknown `custom_call` — is a genuine hard-reject: a **build-time `Result` error naming the offending op**, never a silent drop ([validate-at-build](01-identity.md)).

The worked example is the **StableHLO op map** (`docs/interchange/stablehlo-to-fuel-op-map.md`): of 119 spec ops, ≈100 are covered (primitive/decompose/fused) or handled (lowering/other-layer); the hard-reject set is nearly empty; and the audit names the genuine vocabulary gaps (sort/top-k, pooling, FFT, inverse-trig, product-reduce) as candidate ops to add only under real consumer pressure.

## The per-format binding is the seam

Weight-decode and graph-op-map are separately reusable, but they meet at the **binding** — *which graph node consumes which weight tensor*. That binding is irreducibly format-specific: ONNX binds initializers **by name**, `.pt2` by **blob index**, TFLite by **buffer index**. So:

- weight *payload decoding* → a shared weight-interchange core,
- graph *op-mapping* → a shared graph-interchange core,
- **node↔weight binding → the per-format crate**, the glue wiring the two cores together.

Hoisting the binding into a core produces a leaky abstraction that special-cases every format anyway. It stays format-local; that is the right and only job of a `fuel-format-interchange-*` leaf, because the two cores do the heavy lifting.

## The native format is the base-map serialization (not a new format)

Fuel's own on-disk graph format is **not introduced here** — it already exists. [11-persistence](11-persistence.md) serializes the base map as part of the optimization cache, in a mmap-friendly, schema-versioned, DAG-format-versioned encoding, and explicitly treats the base map as a *hardware-independent, shippable* artifact ([11-persistence §Distribution](11-persistence.md#distribution-cache-as-a-deployment-artifact)). The model-interchange layer **reuses that encoding**: the native `.fuel` model artifact is the base-map section of the persistence format, shipped as a standalone sibling artifact with weights external (safetensors by convention).

Consequences:

- No second DAG serialization. The DAG-format-version machinery and the "read the previous N versions" policy ([decision #18](10-decisions-log.md)) cover the native interchange format for free.
- A `.fuel` export is hardware-independent (base map, not optimized form); an importing process re-optimizes locally, optionally skipping decomposition.
- Weights ride alongside as safetensors, mirroring how every other format externalizes large tensors.
- This native artifact is **L1** (model: base map + weights) of the three-layer persistence stack — L2 adds the optimization plan (hot-load), L3 adds a runtime snapshot (resume). "Save with/without the plan or runtime state" is which sibling artifacts a caller writes; see [11-persistence §Runtime snapshots](11-persistence.md#runtime-snapshots-resuming-designated-durable-state-l3).

## Where interchange reconnects with the model zoo

Graph import produces a graph regardless of whether it matches a known architecture — an arbitrary ONNX file is just a DAG, and fuel can run it as-is. But an imported graph can additionally be **pattern-matched against the registered architectures** ("this is a Llama") and swapped for the optimized parametric version. That recognizer is a *model-tier* concern (it depends on both interchange and the zoo) and lives in `fuel-model-core`; it does not invert the layering, because the model tier already sits above interchange.

## Source as a graph: tracing, not static parsing

Host-language source (e.g. HuggingFace `transformers`) is *syntactically* parseable, but a model's compute graph is a **runtime** property: depth comes from `config.num_hidden_layers`, branches resolve from config, modules are built in `__init__`. Static AST extraction yields a scaffold, not a correct graph. The ecosystem's answer — and fuel's — is **tracing**: run the model on example inputs and record the ops. The output of a trace *is* `torch.export`/ONNX. So "ingest a source-distributed model and run it" collapses into the **graph-import path**; fuel does not re-implement a tracer.

The distinct, complementary capability is the **scaffolder** (`fuel-codegen`, a dev-time tool, [02-layers](02-layers.md)): from source AST + `config.json` (and optionally a trace as oracle), emit a *draft parametric* `fuel-model-*` crate — `Config` struct, `new()` skeleton, `forward()` stub with recognized ops and `TODO` markers for the ~20% that is genuinely novel (where the architectural novelty — and the absence of an existing fused op — actually lives). The scaffolder shares the graph-interchange op-map (a traced model arrives as ATen, so its "operator → `Op`" table *is* the `.pt2` interchange rule set). The trace then serves as the differential-test oracle for the completed port. Two tools, two outputs: a flattened config-specialized graph to *run now*, vs. parametric Rust to *add to the zoo*.

## Format posture

Priorities and per-format reality (effort and tooling rationale live in the migration plan, `docs/session-prompts/model-interchange-import-export-plan.md`):

- **ONNX** — flagship import *and* export. The only multi-producer, fixed-versioned, Rust-parseable graph standard. Parse the protobuf directly (keeps the format tier dependency-light); map to/from the base map.
- **safetensors / GGUF / pickle** — weight interchange (GGUF/safetensors read+write; pickle read-only, transcode out). The "code" is the architecture tag → model registry.
- **`torch.export` (`.pt2`)** — Core-ATen-only import; the strategic PyTorch path. Greenfield Rust reader, evolving schema — pin a version.
- **StableHLO** — import *and* export. High-value **import** target: it is the convergence point JAX, TensorFlow, and PyTorch/XLA all lower to, so one StableHLO importer transitively covers all three producers — notably the only clean path to **JAX** models. Both directions ride **FFI to the StableHLO C API** (the portable artifact is MLIR/VHLO bytecode — no Rust reader without an MLIR runtime), so it stays behind ONNX in priority despite the broad reach. Export is natural from the base map (also ~100 primitives). Control flow, dynamic shapes, and collectives are handled by import-lowering / other layers, not rejected (see the op map).

## Non-goals

- **No TensorRT `.plan` interchange.** A kernel-baked engine tied to exact GPU/driver/version — not a portable graph. Import the *source* (ONNX); let TensorRT compile.
- **No TorchScript investment.** Deprecated upstream (PyTorch ~2.9–2.10); unparseable outside libtorch. Route users to `.pt2`/ONNX.
- **No full-fidelity TF SavedModel / CoreML without FFI.** No mature pure-Rust graph parser; both need vendor-lib binding for fidelity. Deferred unless consumer pressure appears.
- **No second neutral IR.** The base map is the hub; per-format interchange crates map to it, not to an intermediate.
- **No runtime-extensible importers.** Importers are compiled in, like the rest of the IR vocabulary.

---

## See also

- [02-layers](02-layers.md) — the format / interchange / model crate tiers and the per-format binding seam in crate form.
- [03-ir §The base map](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained) — the canonical hub the op-map targets.
- [11-persistence](11-persistence.md) — the base-map serialization the native format reuses; DAG-format versioning.
- [01-identity](01-identity.md) — validate-at-build (unsupported op → `Result`, never silent drop).
- [12-multi-output](12-multi-output.md) — the bundle mechanism that absorbs `tuple`/`get_tuple_element` and region multi-results.
- `docs/interchange/stablehlo-to-fuel-op-map.md` — the worked op map + Fuel completeness audit (119 StableHLO ops, per-op disposition, named gaps).
- `docs/session-prompts/model-interchange-import-export-plan.md` — the migration plan: crate moves, format dossiers, phased roadmap, caller fixes.
