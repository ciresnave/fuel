//! # candle-nn
//!
//! **Layer**: NN — sits directly above `candle-core`. Provides parameterized building
//! blocks for neural networks. Nothing in `candle-core`, `candle-transformers`, or
//! the inference/training leaf crates is allowed to depend on `candle-nn`'s internals;
//! the dependency arrow is strictly downward.
//!
//! **Stability**: `stable`
//!
//! ## What this crate is for
//!
//! `candle-nn` provides the standard ML building blocks built on top of
//! [`candle-core`](https://docs.rs/candle-core) tensors:
//!
//! - **Layers**: [`Linear`], [`Conv1d`], [`Conv2d`], [`Embedding`], [`LayerNorm`],
//!   [`RmsNorm`], [`BatchNorm`], [`GroupNorm`], RNNs ([`LSTM`], [`GRU`]).
//! - **Activations**: [`Activation`] (ReLU, SiLU, GELU, and many others), [`PReLU`].
//! - **Optimizers**: [`AdamW`], [`SGD`].
//! - **Loss functions**: cross-entropy, MSE, and others in the `loss` module.
//! - **Parameter management**: [`VarBuilder`] for loading weights from safetensors or
//!   GGUF files; [`VarMap`] for tracking trainable parameters.
//! - **Module composition**: [`Sequential`], [`Func`], the [`Module`] trait.
//! - **Miscellaneous**: [`Dropout`], one-hot encoding ([`encoding::one_hot`]), MoE helpers, rotary embeddings.
//!
//! ## Quick start
//!
//! ```rust
//! use candle::{Device, DType, Tensor};
//! use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
//! # fn main() -> candle::Result<()> {
//! let device = Device::Cpu;
//! let vb = VarBuilder::zeros(DType::F32, &device);
//! let layer = linear(4, 2, vb.pp("fc"))?;
//! let input = Tensor::zeros((1, 4), DType::F32, &device)?;
//! let output = layer.forward(&input)?;
//! assert_eq!(output.dims(), &[1, 2]);
//! # Ok(()) }
//! ```
//!
//! ## What is explicitly NOT here
//!
//! - **No model-architecture implementations.** Full model definitions (LLaMA, Whisper,
//!   etc.) belong in `candle-transformers`.
//! - **No inference session management.** Decode loops, beam search, and KV-cache
//!   policy belong in `candle-inference`.
//! - **No training loops.** Gradient accumulation, LR scheduling, and checkpoint
//!   management belong in `candle-training`.
//! - **No dataset loading.** Use `candle-datasets` for that.
//!
//! ## Ecosystem crates
//!
//! - [`candle-core`](https://docs.rs/candle-core): tensors, devices, dtypes, autograd.
//! - [`candle-transformers`](https://docs.rs/candle-transformers): full model
//!   architectures built from these building blocks.
//! - [`candle-datasets`](https://docs.rs/candle-datasets): standard ML datasets.
//! - [`candle-onnx`](https://docs.rs/candle-onnx): ONNX model import.

pub mod activation;
pub mod batch_norm;
pub mod conv;
pub mod cpu_flash_attention;
pub mod embedding;
pub mod encoding;
pub mod fused_ops;
pub mod func;
pub mod group_norm;
pub mod init;
pub mod kv_cache;
pub mod layer_norm;
pub mod linear;
pub mod lora;
pub mod loss;
pub mod quantizable_linear;
pub mod moe;
pub mod ops;
pub mod optim;
pub mod rnn;
pub mod rotary_emb;
pub mod sampling;
pub mod sequential;
pub mod training_context;
pub mod var_builder;
pub mod var_map;

pub use activation::{prelu, Activation, PReLU};
pub use batch_norm::{batch_norm, BatchNorm, BatchNormConfig};
pub use conv::{
    conv1d, conv1d_no_bias, conv2d, conv2d_no_bias, conv_transpose1d, conv_transpose1d_no_bias,
    conv_transpose2d, conv_transpose2d_no_bias, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
pub use embedding::{embedding, Embedding};
pub use fused_ops::{fused_linear_silu, fused_matmul_residual, fused_rmsnorm};
pub use func::{func, func_t, Func, FuncT};
pub use group_norm::{group_norm, GroupNorm};
pub use init::Init;
pub use layer_norm::{
    layer_norm, layer_norm_no_bias, rms_norm, LayerNorm, LayerNormConfig, RmsNorm,
};
pub use linear::{linear, linear_b, linear_no_bias, Linear};
pub use lora::{lora_linear, lora_linear_peft, lora_linear_with_base, LoraLinear};
pub use quantizable_linear::QuantizableLinear;
pub use ops::Dropout;
pub use optim::{AdamW, AdamWConfig, Optimizer, ParamsAdamW, SGD};
pub use rnn::{gru, lstm, GRUConfig, LSTMConfig, GRU, LSTM, RNN};
pub use sequential::{seq, Sequential};
pub use var_builder::VarBuilder;
pub use training_context::TrainingContext;
pub use var_map::VarMap;

pub use candle::{Module, ModuleT};
