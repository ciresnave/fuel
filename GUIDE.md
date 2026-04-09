# Candle — Developer Guide

This document routes you to the right crate based on what you are trying to do.
Read the section that matches your goal, then follow the links.

---

## "I just want to do tensor math"

**Start here**: [`candle-core`](candle-core/)

`candle-core` is the entire dependency you need. It gives you `Tensor`, `Device`,
`DType`, and the full set of mathematical operations. There is no mandatory
dependency on any higher crate.

```toml
[dependencies]
candle-core = "0.10"
```

```rust
use candle_core::{Device, Tensor};

let x = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
let y = x.sin()?;
```

**Stop here if**: you are writing numerical algorithms, implementing custom ops,
experimenting with tensor shapes, or embedding a minimal compute engine.

---

## "I want to build a trainable neural network from scratch"

**Start here**: [`candle-nn`](candle-nn/)

`candle-nn` adds parameterized layers, activations, optimizers, and the parameter
management utilities (`VarBuilder`, `VarMap`) you need to define and train a model.

```toml
[dependencies]
candle-core = "0.10"
candle-nn   = "0.10"
```

```rust
use candle_core::{DType, Device};
use candle_nn::{linear, seq, Activation, Module, Sequential, VarBuilder, VarMap};

let vmap = VarMap::new();
let vb = VarBuilder::from_varmap(&vmap, DType::F32, &Device::Cpu);

let mlp = seq()
    .add(linear(784, 256, vb.pp("l1"))?)
    .add(Activation::Relu)
    .add(linear(256, 10, vb.pp("l2"))?);
```

**Stop here if**: you are implementing a novel architecture or a paper that has
no existing implementation in `candle-transformers`.

---

## "I want to use a pretrained model (LLaMA, Whisper, BERT, …)"

**Start here**: [`candle-transformers`](candle-transformers/)

`candle-transformers` contains ready-to-use implementations of the major published
model families. Load weights from a safetensors checkpoint or a GGUF file, build
the model, and run a forward pass.

```toml
[dependencies]
candle-core         = "0.10"
candle-nn           = "0.10"
candle-transformers = "0.10"
```

See [`candle-examples/`](candle-examples/) for complete end-to-end examples for
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

**Start here**: `candle-inference` *(Phase 2 — in progress)*

`candle-inference` will be the leaf crate for orchestrating token generation:
sampling strategies, logit processing, KV-cache lifetime management, speculative
decoding, batched decode, and streaming output.

Until that crate is available, the current home for these utilities is:

- `candle-transformers/src/generation/` — `LogitsProcessor`, `Sampling`
- `candle-nn/src/kv_cache.rs` — `KvCache`, `RotatingKvCache`, `ConcatKvCache`
- `candle-nn/src/sampling.rs` — `gumbel_softmax`
- `candle-examples/` — complete inference loops for each model

See [ROADMAP.md](ROADMAP.md) Phase 2 for the migration plan.

---

## "I want to train a model (data loops, LR scheduling, checkpointing)"

**Start here**: `candle-training` *(Phase 2 — in progress)*

`candle-training` will be the leaf crate for training orchestration:
training loops, gradient accumulation, learning rate schedulers, gradient
clipping, mixed precision policy, and checkpoint management.

Until that crate is available, write training loops using `candle-nn`'s optimizer
and `VarMap` directly. See `candle-examples/` for working training examples
(MNIST classifier, image training, etc.).

---

## "I want to import or evaluate an ONNX model"

**Start here**: [`candle-onnx`](candle-onnx/)

```toml
[dependencies]
candle-core  = "0.10"
candle-onnx  = "0.10"
```

```rust
use std::collections::HashMap;
use candle_onnx::{read_file, simple_eval};

let model = read_file("model.onnx")?;
let outputs = simple_eval(&model, HashMap::new())?;
```

`candle-onnx` is in the `exclude` list in the workspace `Cargo.toml` because it
requires `protobuf` codegen. Build it separately if needed.

---

## "I want to load standard ML datasets (MNIST, CIFAR, …)"

**Start here**: [`candle-datasets`](candle-datasets/)

```toml
[dependencies]
candle-core     = "0.10"
candle-datasets = "0.10"
```

```rust
use candle_datasets::vision::mnist;
let dataset = mnist::load()?; // downloads to ~/.cache/huggingface/datasets
```

---

## "I want to add a new hardware backend"

**Start here**: [`candle-core/src/backend.rs`](candle-core/src/backend.rs)

The `BackendDevice` and `BackendStorage` traits define the contract a new backend
must implement. The CPU, CUDA, and Metal backends are already in `candle-core`
behind Cargo feature flags.

See [ROADMAP.md](ROADMAP.md) Phase 5 for the planned progression:

1. **Tier 1** (now): feature flags already exist — document them clearly.
2. **Tier 2** (near-term): extract each backend into its own crate; publish
   `BackendDevice`/`BackendStorage` as a stable interface.
3. **Tier 3** (medium-term): open `Device` with a `Custom(Arc<dyn BackendDevice>)`
   arm so third-party backends need no fork.
4. **Tier 4** (long-term): per-operation routing DAG across backends.

---

## "I want to use Python bindings"

**Start here**: [`candle-pyo3`](candle-pyo3/)

`candle-pyo3` provides Python access to `candle-core` and `candle-nn` via PyO3.
Build with `maturin develop` inside `candle-pyo3/`.

---

## Architecture in one diagram

```text
┌──────────────────────────────────────────────────────────────────┐
│  candle-inference          candle-training     (leaf — Phase 2)  │
├──────────────────────────────────────────────────────────────────┤
│  candle-transformers        (models layer)                       │
├──────────────────────────────────────────────────────────────────┤
│  candle-nn                  (NN layer)                           │
├──────────────────────────────────────────────────────────────────┤
│  candle-core                (foundation)                         │
├──────────────────────────────────────────────────────────────────┤
│  CPU backend  │  CUDA backend  │  Metal backend  (kernels layer) │
└──────────────────────────────────────────────────────────────────┘
```

Dependencies flow **downward only**. A user who needs only tensor math carries
only `candle-core`. The early-exit property is structural, not aspirational.

The full design rationale, per-layer anti-goals, and phased work plan are in
[ROADMAP.md](ROADMAP.md).
