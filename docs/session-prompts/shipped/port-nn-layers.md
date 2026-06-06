# Port: fuel-nn Module wrappers (Linear, Conv, Norm, Embedding, etc.)

## Eager source

The bulk of `fuel-nn/src/*.rs`:

- `linear.rs` — `Linear { weight, bias: Option }` + `linear`,
  `linear_no_bias` constructors.
- `conv.rs` — `Conv1d`, `Conv2d`, `Conv1dConfig`,
  `Conv2dConfig` (+ depthwise / grouped variants).
- `layer_norm.rs` — `LayerNorm`, `RmsNorm`, configs.
- `batch_norm.rs` — `BatchNorm`, `BatchNorm1d/2d/3d`.
- `group_norm.rs` — `GroupNorm`.
- `embedding.rs` — `Embedding`.
- `sequential.rs` — `Sequential` container.
- `activation.rs` — activation function modules.
- `encoding.rs` — `PositionalEncoding`.
- `init.rs` — Xavier / Kaiming / Uniform initializers.
- `kv_cache.rs` — `KvCache` (already partially shipped as
  `fuel-core/src/lazy_kv_cache.rs` — extend, don't duplicate).
- `rotary_emb.rs` — `RotaryEmbedding`, `rope` function.
- `lora.rs` — `LoraLinear` (LazyTensor's WeightStorage::WithLoRA
  already exists; this is the high-level Module wrapper).
- `quantizable_linear.rs` — Q4_0 linear wrapper.
- `rnn.rs` — `LSTM`, `GRU` (LSTM already shipped as
  `lazy_lstm`).
- `fused_ops.rs` — fused-op convenience wrappers (e.g.
  `linear_then_relu`).
- `cpu_flash_attention.rs` — CPU flash-attn fallback (deprecated
  with shipped lazy FlashAttn path?).
- `moe.rs` — Mixture-of-experts router + experts.
- `sampling.rs` — Top-K / top-P / temperature samplers.
- `func.rs` — functional ops (concat, split, etc. — may already
  be on LazyTensor).
- `training_context.rs` — TrainingContext for backprop tape.
- `var_builder.rs` — VarBuilder for weight loading.
- `var_map.rs` — VarMap parameter container.

Total: ~15k LOC across the crate.

## Lazy module name

This is too large for one module. Split into:

- `fuel-core/src/lazy_nn/mod.rs` — directory module entry point.
- `fuel-core/src/lazy_nn/linear.rs`
- `fuel-core/src/lazy_nn/conv.rs`
- `fuel-core/src/lazy_nn/norm.rs` (LayerNorm + RmsNorm + GroupNorm
  + BatchNorm)
- `fuel-core/src/lazy_nn/embedding.rs`
- `fuel-core/src/lazy_nn/sequential.rs`
- `fuel-core/src/lazy_nn/activation.rs`
- `fuel-core/src/lazy_nn/lora.rs`
- `fuel-core/src/lazy_nn/moe.rs`
- `fuel-core/src/lazy_nn/sampling.rs`
- Plus a top-level `Module` trait analogous to eager's
  `fuel::Module`.

## Architecture summary

Each eager module is a thin wrapper that holds weights and
implements `Module::forward(&self, xs: &Tensor) -> Result<Tensor>`.

Lazy equivalent:
- Hold weights as `Arc<[f32]>` or `WeightStorage`.
- `forward(&self, xs: &LazyTensor) -> Result<LazyTensor>` delegates
  to the matching `LazyTensor::*` primitive.

E.g. `Linear`:
```rust
pub struct LazyLinear {
    weight: WeightStorage,
    bias: Option<Arc<[f32]>>,
    in_features: usize,
    out_features: usize,
}

impl LazyLinear {
    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let y = self.weight.apply_linear(xs, self.in_features, self.out_features);
        match &self.bias {
            Some(b) => {
                let b = xs.const_f32_like(Arc::clone(b), Shape::from_dims(&[self.out_features]));
                y.broadcast_add(&b)
            }
            None => Ok(y),
        }
    }
}
```

## Primitives needed

- All shipped. This port is glue + tests, not new graph ops.

## Reusable modules

- All shipped lazy_* modules — same primitives the model ports
  use.
- `lazy_lstm` already exists as the LSTM wrapper.
- `lazy_kv_cache` already exists.

## Splits

Mandatory splits given the size:

1. Sub-port 1: `Module` trait + `Linear` + `Embedding` + tests.
   Smallest viable surface, unblocks downstream.
2. Sub-port 2: `Conv1d` + `Conv2d` + their configs.
3. Sub-port 3: `LayerNorm` + `RmsNorm` + `GroupNorm` + `BatchNorm`.
4. Sub-port 4: `Sequential` + `Activation` (RELU, GELU, SiLU,
   Sigmoid, Tanh as modules).
5. Sub-port 5: `LoRA` + `Quantizable Linear`.
6. Sub-port 6: `MoE` router + experts.
7. Sub-port 7: `Sampling` + `Init`.

Ship each as its own commit.

## Test strategy

Per sub-port:
- Each module: forward output shape + finite check.
- A numerical golden vs the matching LazyTensor primitive (e.g.
  `LazyLinear::forward` output equals
  `weight.apply_linear(xs, in, out) + bias` directly).

## References

- Eager source: `fuel-nn/src/*.rs`.
- Module trait shape: `fuel-core/src/lib.rs` already has a
  `Module` trait — verify it works for LazyTensor or extend.
