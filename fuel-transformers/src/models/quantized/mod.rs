//! Quantized model variants.
//!
//! GGUF/GGML quantized implementations of models for reduced memory usage
//! and faster inference on CPU.

pub use super::common::with_tracing;
pub use super::llm::llama2_c;
pub use super::llm::mpt;
pub use super::multimodal::blip;
pub use super::multimodal::blip_text;

pub mod quantized_blip;
pub mod quantized_blip_text;
pub mod quantized_gemma3;
pub mod quantized_glm4;
pub mod quantized_lfm2;
pub mod quantized_llama;
pub mod quantized_llama2_c;
pub mod quantized_metavoice;
pub mod quantized_mistral;
pub mod quantized_mixformer;
pub mod quantized_moondream;
pub mod quantized_mpt;
pub mod quantized_phi;
pub mod quantized_phi3;
pub mod quantized_qwen2;
pub mod quantized_qwen3;
pub mod quantized_qwen3_moe;
pub mod quantized_recurrent_gemma;
pub mod quantized_rwkv_v5;
pub mod quantized_rwkv_v6;
pub mod quantized_stable_lm;
pub mod quantized_t5;
