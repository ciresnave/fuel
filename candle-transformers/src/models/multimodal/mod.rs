//! Multimodal model implementations.
//!
//! Vision-language models, CLIP variants, document understanding,
//! and other models that combine multiple modalities.

pub use super::common::with_tracing;
pub use super::vision::fastvit;

pub mod blip;
pub mod blip_text;
pub mod chinese_clip;
pub mod clip;
pub mod colpali;
pub mod llava;
pub mod mobileclip;
pub mod moondream;
pub mod openclip;
pub mod paddleocr_vl;
pub mod paligemma;
pub mod pixtral;
pub mod qwen3_vl;
pub mod siglip;
pub mod voxtral;
