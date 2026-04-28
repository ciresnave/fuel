# Fuel — Developer Guide

This document routes you to the right crate based on what you are trying to do.
Read the section that matches your goal, then follow the links.

---

## "I just want to do tensor math"

**Start here**: [`fuel-core`](fuel-core/)

`fuel-core` is the entire dependency you need. It gives you `Tensor`, `Device`,
`DType`, and the full set of mathematical operations. There is no mandatory
dependency on any higher crate.

```toml
[dependencies]
fuel-core = "0.10"
```

```rust
use fuel_core::{Device, Tensor};

let x = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
let y = x.sin()?;
```

**Stop here if**: you are writing numerical algorithms, implementing custom ops,
experimenting with tensor shapes, or embedding a minimal compute engine.

---

## "I want to build a trainable neural network from scratch"

**Start here**: [`fuel-nn`](fuel-nn/)

`fuel-nn` adds parameterized layers, activations, optimizers, and the parameter
management utilities (`VarBuilder`, `VarMap`) you need to define and train a model.

```toml
[dependencies]
fuel-core = "0.10"
fuel-nn   = "0.10"
```

```rust
use fuel_core::{DType, Device};
use fuel_nn::{linear, seq, Activation, Module, Sequential, VarBuilder, VarMap};

let vmap = VarMap::new();
let vb = VarBuilder::from_varmap(&vmap, DType::F32, &Device::Cpu);

let mlp = seq()
    .add(linear(784, 256, vb.pp("l1"))?)
    .add(Activation::Relu)
    .add(linear(256, 10, vb.pp("l2"))?);
```

**Stop here if**: you are implementing a novel architecture or a paper that has
no existing implementation in `fuel-transformers`.

---

## "I want to use a pretrained model (LLaMA, Whisper, BERT, …)"

**Start here**: [`fuel-transformers`](fuel-transformers/)

`fuel-transformers` contains ready-to-use implementations of the major published
model families. Load weights from a safetensors checkpoint or a GGUF file, build
the model, and run a forward pass.

```toml
[dependencies]
fuel-core         = "0.10"
fuel-nn           = "0.10"
fuel-transformers = "0.10"
```

See [`fuel-examples/`](fuel-examples/) for complete end-to-end examples for
each model family (download, tokenize, inference loop).

### Model families available

| Domain     | Examples                                                            |
| ---------- | ------------------------------------------------------------------- |
| LLMs       | LLaMA 2/3, Mistral, Mixtral, Falcon, Phi-2/3, Gemma, Qwen, DeepSeek |
| Vision     | ViT, DINOv2, EfficientNet, ResNet, CLIP, SigLIP                     |
| Audio      | Whisper, EnCodec, Mimi, DAC                                         |
| Diffusion  | Stable Diffusion 1.5/2/XL, Flux, Wuerstchen                         |
| Multimodal | LLaVA, Moondream, PaliGemma, Pixtral                                |
| Encoders   | BERT, T5, Nomic BERT                                                |

---

## "I want to run an inference pipeline (sampling, streaming, batching)"

**Start here**: `fuel-inference` *(Phase 2 — in progress)*

`fuel-inference` will be the leaf crate for orchestrating token generation:
sampling strategies, logit processing, KV-cache lifetime management, speculative
decoding, batched decode, and streaming output.

Until that crate is available, the current home for these utilities is:

- `fuel-transformers/src/generation/` — `LogitsProcessor`, `Sampling`
- `fuel-nn/src/kv_cache.rs` — `KvCache`, `RotatingKvCache`, `ConcatKvCache`
- `fuel-nn/src/sampling.rs` — `gumbel_softmax`
- `fuel-examples/` — complete inference loops for each model

See [ROADMAP.md](ROADMAP.md) Phase 2 for the migration plan.

---

## "I want to train a model (data loops, LR scheduling, checkpointing)"

**Start here**: `fuel-training` *(Phase 2 — in progress)*

`fuel-training` will be the leaf crate for training orchestration:
training loops, gradient accumulation, learning rate schedulers, gradient
clipping, mixed precision policy, and checkpoint management.

Until that crate is available, write training loops using `fuel-nn`'s optimizer
and `VarMap` directly. See `fuel-examples/` for working training examples
(MNIST classifier, image training, etc.).

---

## "I want to import or evaluate an ONNX model"

**Start here**: [`fuel-onnx`](fuel-onnx/)

```toml
[dependencies]
fuel-core  = "0.10"
fuel-onnx  = "0.10"
```

```rust
use std::collections::HashMap;
use fuel_onnx::{read_file, simple_eval};

let model = read_file("model.onnx")?;
let outputs = simple_eval(&model, HashMap::new())?;
```

`fuel-onnx` is in the `exclude` list in the workspace `Cargo.toml` because it
requires `protobuf` codegen. Build it separately if needed.

---

## "I want to load standard ML datasets (MNIST, CIFAR, …)"

**Start here**: [`fuel-datasets`](fuel-datasets/)

```toml
[dependencies]
fuel-core     = "0.10"
fuel-datasets = "0.10"
```

```rust
use fuel_datasets::vision::mnist;
let dataset = mnist::load()?; // downloads to ~/.cache/huggingface/datasets
```

---

## "I want my model to run on the fastest available CPU backend"

**Start here**: [`fuel-core::dispatch`](fuel-core/src/dispatch.rs)

Compiling with `--features aocl,onemkl` makes the AOCL and oneMKL backends
*available*. By default they're not used — `LazyTensor::realize_f32()` keeps
running through the portable Rust `gemm` baseline so behaviour stays
predictable for users who never opt in. To switch on per-op empirical
routing across every CPU backend Fuel sees, the app calls
`fuel_core::dispatch::populate_dispatch_table()` once:

```rust
use fuel_core::dispatch;

dispatch::populate_dispatch_table()?;  // first run: ~10–60s. Cached after.
// Every subsequent realize_f32() now consults the dispatch table per op.
```

The Phase 6b judge profiles every loaded backend × every (op, dtype,
size_class) triple, persists the result to the OS cache dir, and the
Router uses it at realize time to pick the empirical winner per op. No
vendor heuristic involved — MKL and AOCL both run on Zen5 in our tests
and the dispatch table picks whichever was actually faster that run.

If a previous profile becomes stale (driver upgrade, BLAS swap), call
`dispatch::invalidate()`. The next `populate_dispatch_table()` re-runs.

See [ROADMAP.md](ROADMAP.md) Phase 7b for the per-vendor backend layer
and Phase 6b for the empirical dispatch infrastructure.

---

## "I want to add a new hardware backend"

**Start here**: [`fuel-core/src/backend.rs`](fuel-core/src/backend.rs)

The `BackendDevice` and `BackendStorage` traits define the contract a new backend
must implement. The CPU, CUDA, and Metal backends are already in `fuel-core`
behind Cargo feature flags. AOCL and oneMKL ship as separate per-vendor crates
(`fuel-aocl-cpu-backend`, `fuel-mkl-cpu-backend`) following the Phase 7b pattern;
mirror them when adding a new CPU vendor (Apple Accelerate, ARM Performance
Libraries, OpenBLAS).

See [ROADMAP.md](ROADMAP.md) Phase 5 for the planned progression:

1. **Tier 1** (now): feature flags already exist — document them clearly.
2. **Tier 2** (near-term): extract each backend into its own crate; publish
   `BackendDevice`/`BackendStorage` as a stable interface.
3. **Tier 3** (medium-term): open `Device` with a `Custom(Arc<dyn BackendDevice>)`
   arm so third-party backends need no fork.
4. **Tier 4** (long-term): per-operation routing DAG across backends.

---

## "I want to use Python bindings"

**Start here**: [`fuel-pyo3`](fuel-pyo3/)

`fuel-pyo3` provides Python access to `fuel-core` and `fuel-nn` via PyO3.
Build with `maturin develop` inside `fuel-pyo3/`.

---

## Architecture in one diagram

```text
┌──────────────────────────────────────────────────────────────────┐
│  fuel-inference          fuel-training     (leaf — Phase 2)  │
├──────────────────────────────────────────────────────────────────┤
│  fuel-transformers        (models layer)                       │
├──────────────────────────────────────────────────────────────────┤
│  fuel-nn                  (NN layer)                           │
├──────────────────────────────────────────────────────────────────┤
│  fuel-core                (foundation)                         │
├──────────────────────────────────────────────────────────────────┤
│  CPU backend  │  CUDA backend  │  Metal backend  (kernels layer) │
└──────────────────────────────────────────────────────────────────┘
```

Dependencies flow **downward only**. A user who needs only tensor math carries
only `fuel-core`. The early-exit property is structural, not aspirational.

The full design rationale, per-layer anti-goals, and phased work plan are in
[ROADMAP.md](ROADMAP.md).
