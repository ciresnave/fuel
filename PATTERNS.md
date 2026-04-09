# Candle Canonical Patterns

Architecture guides — not API demos. Each pattern captures the minimal correct
structure for a common task. Copy, adapt, and extend.

---

## 1. Minimal tensor program

The atoms of Candle: create tensors on a device, perform arithmetic, move
results to the host.

```rust
use candle_core::{Device, DType, Tensor};

fn main() -> candle_core::Result<()> {
    // All computation is tied to a device. CPU is always available.
    let device = Device::Cpu;

    // Create 2-D tensors from nested slices.
    // DType determines storage: F32 for most numerics, F64 for scientific work.
    let a = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
    let b = Tensor::new(&[[5f32, 6.], [7., 8.]], &device)?;

    // Element-wise and matrix operations use the same API on CPU and GPU.
    let sum = (&a + &b)?;
    let prod = a.matmul(&b)?;

    // Pull a single value back to CPU-native Rust types.
    let scalar: f32 = prod.get(0)?.get(0)?.to_scalar()?;
    println!("top-left cell = {scalar}");

    Ok(())
}
```

Key properties:

- All fallible operations return `candle_core::Result<T>`. Use `?` throughout.
- `&device` is borrowed; multiple tensors share one device handle.
- Tensors are reference-counted; cloning is O(1) (no data copy).

---

## 2. Minimal trainable module

How to wire autograd through custom layers. The pattern that every real model
follows regardless of size.

```rust
use candle_core::{Device, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, linear, AdamW, ParamsAdamW, Optimizer};

/// A two-layer MLP with a skip connection.
struct TwoLayerMlp {
    fc1: Linear,
    fc2: Linear,
}

impl TwoLayerMlp {
    /// Construct from a `VarMap`-backed `VarBuilder`.
    ///
    /// `vb` namespaces weight tensors in the checkpoint so that
    /// `fc1.weight`, `fc2.weight`, etc. are stored without collision.
    fn new(in_dim: usize, hidden: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(in_dim, hidden, vb.pp("fc1"))?,
            fc2: linear(hidden, out_dim, vb.pp("fc2"))?,
        })
    }
}

impl Module for TwoLayerMlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // fc1 → ReLU → fc2. Gradients flow automatically.
        let h = self.fc1.forward(x)?.relu()?;
        self.fc2.forward(&h)
    }
}

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // VarMap owns all trainable parameters. Every tensor created through it is
    // tracked; gradients accumulate on backward.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = TwoLayerMlp::new(4, 16, 2, vb)?;

    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;

    // Fake batch: 8 samples, 4 features.
    let xs = Tensor::randn(0f32, 1., (8, 4), &device)?;
    let ys = Tensor::zeros((8, 2), DType::F32, &device)?;

    for _step in 0..100 {
        let logits = model.forward(&xs)?;
        // MSE loss: mean over all elements.
        let loss = logits.sub(&ys)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;   // gradient descent in one call
    }

    Ok(())
}
```

Key properties:

- `VarMap` tracks every parameter created through its `VarBuilder`.
- `backward_step` computes gradients and applies the update atomically.
- `Module::forward` is the only required method; no `backward` override needed.

---

## 3. Minimal pretrained model load and forward pass

How to load weights from a `.safetensors` file and run a forward pass. The
same steps apply to HuggingFace Hub downloads.

```rust
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder};
// Replace `MyModel` with the concrete model type from candle-transformers.
// use candle_transformers::models::llama::{Llama, Config};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;  // swap for Device::new_cuda(0)? on GPU

    // Load weights from a local safetensors file.
    // For multi-shard checkpoints pass a Vec of paths.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &["model.safetensors"],
            candle_core::DType::F32,
            &device,
        )?
    };

    // Build the model from the weight file.
    // The Config struct exposes the same fields as HuggingFace config.json.
    // let config: Config = serde_json::from_reader(std::fs::File::open("config.json")?)?;
    // let model = Llama::load(vb, &config)?;

    // Prepare a token batch: [batch_size, seq_len].
    // Real inference tokenises text; here we use a hard-coded tensor.
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device)?;

    // Forward pass — no gradient tracking unless you explicitly enable it.
    // let logits = model.forward(&input_ids, 0)?;

    Ok(())
}
```

Key properties:

- `from_mmaped_safetensors` avoids loading the entire file into RAM.
  Mark `unsafe` because mmap can observe external writes.
- `VarBuilder` and the model struct are the only two objects needed.
- The model's `forward` signature varies: most transformers accept
  `(input_ids, position_offset)` or `(input_ids, attention_mask, cache)`.

---

## 4. Minimal inference loop with sampling

How to autoregressively decode tokens. This is the inner loop of every LLM
chat server.

```rust
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

/// Decode up to `max_new_tokens` tokens given a prompt.
fn generate(
    // model: &mut impl ForwardWithCache,  // from your concrete model
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    temperature: f64,
    device: &Device,
) -> candle_core::Result<Vec<u32>> {
    let mut tokens = prompt_tokens;
    let mut logits_proc = LogitsProcessor::new(
        /*seed=*/ 42,
        Some(temperature),
        /*top_p=*/ None,
    );

    for pos in 0..max_new_tokens {
        // Build a single-token input from the last generated token.
        let input = Tensor::new(&[*tokens.last().unwrap()], device)?
            .unsqueeze(0)?;  // shape: [1, 1]

        // model.forward(&input, pos)? returns logits of shape [1, 1, vocab_size].
        // let logits = model.forward(&input, pos)?;
        // let next_token_logits = logits.squeeze(0)?.squeeze(0)?; // [vocab_size]
        //
        // Sample the next token.
        // let next_token = logits_proc.sample(&next_token_logits)?;
        //
        // tokens.push(next_token);
        //
        // Stop at EOS.
        // if next_token == EOS_TOKEN_ID { break; }
        let _ = pos; // suppress unused warning in the skeleton
    }

    Ok(tokens)
}

fn main() -> candle_core::Result<()> {
    let _device = Device::Cpu;
    // let tokens = generate(&mut model, prompt_ids, 200, 0.7, &device)?;
    Ok(())
}
```

Key properties:

- `LogitsProcessor` implements temperature scaling and nucleus (top-p) sampling.
  Pass `temperature = None` for greedy decoding.
- Positional offset `pos` tracks where in the KV cache to write. Start at 0
  for the first token after the prompt and increment by 1 each step.
- The EOS check is model-specific; look it up in the model's config.

---

## 5. Minimal custom operation extension

How to register a new differentiable operation and plug it into autograd.
Use this when you need an op that isn't in `candle-core`.

```rust
use candle_core::{Tensor, Result, op::{Op, BackpropOp, UnaryOpT}};

/// A custom element-wise activation: f(x) = x * sigmoid(x)  (Swish / SiLU).
///
/// Candle already has `Tensor::silu()` — this is only for illustration.
#[derive(Debug, Clone)]
struct SwishOp;

impl UnaryOpT for SwishOp {
    const NAME: &'static str = "swish";
    const KERNEL: &'static str = "uswish";    // Metal/CUDA kernel name if applicable
    const V: Self = SwishOp;

    fn f32(v: f32) -> f32 {
        v * (1.0 / (1.0 + (-v).exp()))
    }
    fn f64(v: f64) -> f64 {
        v * (1.0 / (1.0 + (-v).exp()))
    }
}

/// Wrapper that applies the op and registers the backward closure.
fn swish(x: &Tensor) -> Result<Tensor> {
    // apply_op1 dispatches to CPU/CUDA/Metal via the UnaryOpT trait.
    let out = x.apply_op1_no_bwd(&SwishOp)?;

    // For autograd support, record the backward function.
    // The closure receives the output gradient and returns the input gradient.
    // Uncomment and adapt for a real differentiable op:
    //
    // if x.is_tracked() {
    //     let x_clone = x.clone();
    //     let out = out.with_backward(move |grad| {
    //         // d/dx [x * sigma(x)] = sigma(x) + x * sigma(x) * (1 - sigma(x))
    //         // = swish(x) / x + swish(x) * (1 - sigma(x))
    //         let sigma = x_clone.sigmoid()?;
    //         let dsigma = (&sigma * (1. - &sigma))?;
    //         let grad_input = (&grad * (&sigma + (&x_clone * &dsigma)?)?)?;
    //         Ok(grad_input)
    //     });
    // }

    Ok(out)
}

fn main() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let x = Tensor::new(&[-2f32, -1., 0., 1., 2.], &device)?;
    let y = swish(&x)?;
    println!("{y}");
    Ok(())
}
```

Key properties:

- `UnaryOpT` automatically dispatches to CPU, CUDA, and Metal backends if a
  kernel is registered. For CPU-only ops, `apply_op1_no_bwd` suffices.
- The `KERNEL` name must match the entry point in your `.cu` / `.metal` file
  if you want GPU support.
- Autograd is opt-in: wrap in `with_backward` only when you need gradients.

---

## Cross-references

| Goal                    | Where to look                        |
| ----------------------- | ------------------------------------ |
| Routing by use case     | [GUIDE.md](GUIDE.md)                 |
| Ecosystem compatibility | [COMPATIBILITY.md](COMPATIBILITY.md) |
| Architecture vision     | [ROADMAP.md](ROADMAP.md)             |
| Backend plugin system   | [ROADMAP.md Phase 5](ROADMAP.md)     |
