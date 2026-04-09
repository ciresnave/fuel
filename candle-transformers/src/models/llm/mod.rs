//! Large language model implementations.
//!
//! Decoder-only and encoder-decoder transformer models for text generation,
//! code generation, machine translation, and related sequence tasks.

// Re-export shared utilities so that `super::with_tracing` keeps resolving
// from files in this directory (they previously lived one level up).
pub use super::common::with_tracing;

pub mod based;
pub mod bigcode;
pub mod chatglm;
pub mod codegeex4_9b;
pub mod deepseek2;
pub mod falcon;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod gemma4;
pub mod glm4;
pub mod glm4_new;
pub mod granite;
pub mod granitemoehybrid;
pub mod helium;
pub mod llama;
pub mod llama2_c;
pub mod llama2_c_weights;
pub mod mamba;
pub mod mamba2;
pub mod marian;
pub mod mistral;
pub mod mixformer;
pub mod mixtral;
pub mod mpt;
pub mod olmo;
pub mod olmo2;
pub mod persimmon;
pub mod phi;
pub mod phi3;
pub mod qwen2;
pub mod qwen2_moe;
pub mod qwen3;
pub mod qwen3_moe;
pub mod recurrent_gemma;
pub mod rwkv_v5;
pub mod rwkv_v6;
pub mod rwkv_v7;
pub mod smol;
pub mod stable_lm;
pub mod starcoder2;
pub mod t5;
pub mod yi;
