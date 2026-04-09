//! Candle implementations for various deep learning models
//!
//! This crate provides implementations of popular machine learning models and architectures
//! organized by modality:
//!
//! - [`llm`] — Large language models: LLaMA, Mistral, Falcon, Phi, Gemma, Qwen, DeepSeek, …
//! - [`encoders`] — Encoder-only models: BERT, DeBERTa, DistilBERT, ModernBERT, …
//! - [`vision`] — Computer vision: ViT, DINOv2, EfficientNet, ResNet, SegFormer, …
//! - [`audio`] — Audio models: Whisper, EnCodec, Mimi, Parler TTS, DAC, …
//! - [`diffusion`] — Image generation: Stable Diffusion, Flux, Wuerstchen, …
//! - [`multimodal`] — Vision-language: CLIP, LLaVA, PaliGemma, Pixtral, Moondream, …
//! - [`quantized`] — GGUF/GGML quantized variants of the above
//! - [`common`] — Shared utilities (traced wrappers, common primitives)
//!
//! All models are also re-exported at the top level of this module for backward
//! compatibility — `crate::models::llama` and `crate::models::llm::llama` both work.

// ── Category modules ──────────────────────────────────────────────────
pub mod llm;
pub mod vision;
pub mod audio;
pub mod diffusion;
pub mod multimodal;
pub mod encoders;
pub mod common;
pub mod quantized;

// ── Backward-compatible re-exports ────────────────────────────────────
// Every model that previously lived at `crate::models::<name>` is re-exported
// here so that existing `use crate::models::llama` paths continue to compile.

// common
pub use common::with_tracing;

// llm
pub use llm::based;
pub use llm::bigcode;
pub use llm::chatglm;
pub use llm::codegeex4_9b;
pub use llm::deepseek2;
pub use llm::falcon;
pub use llm::gemma;
pub use llm::gemma2;
pub use llm::gemma3;
pub use llm::gemma4;
pub use llm::glm4;
pub use llm::glm4_new;
pub use llm::granite;
pub use llm::granitemoehybrid;
pub use llm::helium;
pub use llm::llama;
pub use llm::llama2_c;
pub use llm::llama2_c_weights;
pub use llm::mamba;
pub use llm::mamba2;
pub use llm::marian;
pub use llm::mistral;
pub use llm::mixformer;
pub use llm::mixtral;
pub use llm::mpt;
pub use llm::olmo;
pub use llm::olmo2;
pub use llm::persimmon;
pub use llm::phi;
pub use llm::phi3;
pub use llm::qwen2;
pub use llm::qwen2_moe;
pub use llm::qwen3;
pub use llm::qwen3_moe;
pub use llm::recurrent_gemma;
pub use llm::rwkv_v5;
pub use llm::rwkv_v6;
pub use llm::rwkv_v7;
pub use llm::smol;
pub use llm::stable_lm;
pub use llm::starcoder2;
pub use llm::t5;
pub use llm::yi;

// vision
pub use vision::beit;
pub use vision::convmixer;
pub use vision::convnext;
pub use vision::depth_anything_v2;
pub use vision::dinov2;
pub use vision::dinov2reg4;
pub use vision::efficientnet;
pub use vision::efficientvit;
pub use vision::eva2;
pub use vision::fastvit;
pub use vision::hiera;
pub use vision::mobilenetv4;
pub use vision::mobileone;
pub use vision::repvgg;
pub use vision::resnet;
pub use vision::segformer;
pub use vision::segment_anything;
pub use vision::trocr;
pub use vision::vgg;
pub use vision::vit;

// audio
pub use audio::csm;
pub use audio::dac;
pub use audio::encodec;
pub use audio::metavoice;
pub use audio::mimi;
pub use audio::parler_tts;
pub use audio::snac;
pub use audio::whisper;

// diffusion
pub use diffusion::flux;
pub use diffusion::mmdit;
pub use diffusion::stable_diffusion;
pub use diffusion::wuerstchen;
pub use diffusion::z_image;

// multimodal
pub use multimodal::blip;
pub use multimodal::blip_text;
pub use multimodal::chinese_clip;
pub use multimodal::clip;
pub use multimodal::colpali;
pub use multimodal::llava;
pub use multimodal::mobileclip;
pub use multimodal::moondream;
pub use multimodal::openclip;
pub use multimodal::paddleocr_vl;
pub use multimodal::paligemma;
pub use multimodal::pixtral;
pub use multimodal::qwen3_vl;
pub use multimodal::siglip;
pub use multimodal::voxtral;

// encoders
pub use encoders::bert;
pub use encoders::debertav2;
pub use encoders::distilbert;
pub use encoders::jina_bert;
pub use encoders::modernbert;
pub use encoders::nomic_bert;
pub use encoders::nvembed_v2;
pub use encoders::stella_en_v5;
pub use encoders::xlm_roberta;

// quantized
pub use quantized::quantized_blip;
pub use quantized::quantized_blip_text;
pub use quantized::quantized_gemma3;
pub use quantized::quantized_glm4;
pub use quantized::quantized_lfm2;
pub use quantized::quantized_llama;
pub use quantized::quantized_llama2_c;
pub use quantized::quantized_metavoice;
pub use quantized::quantized_mistral;
pub use quantized::quantized_mixformer;
pub use quantized::quantized_moondream;
pub use quantized::quantized_mpt;
pub use quantized::quantized_phi;
pub use quantized::quantized_phi3;
pub use quantized::quantized_qwen2;
pub use quantized::quantized_qwen3;
pub use quantized::quantized_qwen3_moe;
pub use quantized::quantized_recurrent_gemma;
pub use quantized::quantized_rwkv_v5;
pub use quantized::quantized_rwkv_v6;
pub use quantized::quantized_stable_lm;
pub use quantized::quantized_t5;
