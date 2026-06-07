//! Moondream — wasm + lazy port.
//!
//! Migrated from the retired eager `fuel_nn` / `fuel_transformers::models::moondream`
//! stack onto the lazy substrate at `fuel::lazy_moondream`.
//!
//! The lazy `MoondreamModel` exposes a single-pass `forward(pixel_values,
//! &[u32] text_tokens)` that returns logits for the concatenated
//! `[image_features; text_embeds]` sequence — there is no separate
//! `vision_encoder()` / `text_model.forward_with_img()` split and no
//! per-step KV cache today. This wasm binary therefore re-runs the full
//! forward pass per generated token (correctness > perf for v1), matching
//! the desktop `fuel-examples/examples/moondream/main.rs` binary.
//!
//! Deferrals vs the eager wasm binary:
//!   - Quantized GGUF (q4_0) variant: no `lazy_quantized_moondream` yet;
//!     `quantized` flag now always rejects.
//!   - KV cache + `clear_kv_cache()`: the lazy module doesn't expose those
//!     hooks yet; full prefix re-run each step.
//!   - `MoondreamWeights::load_from_mmapped` is presently a stub
//!     (see `fuel_core::lazy_moondream`); construction will error at
//!     runtime until that lands. The binary still compiles + the
//!     ergonomic surface is in place for the eventual loader bring-up.

use fuel::lazy::LazyTensor;
use fuel::lazy_moondream::{MoondreamConfig, MoondreamModel, MoondreamWeights};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use fuel_transformers::generation::LogitsProcessor;
use fuel_wasm_example_moondream::console_log;
use js_sys::Date;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    model: MoondreamModel,
    config: MoondreamConfig,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    index: usize,
    pixel_values: Option<LazyTensor>,
}

#[derive(Serialize, Deserialize)]
struct Output {
    token: String,
    token_id: u32,
}
#[derive(Serialize, Deserialize)]
struct InitInput {
    prompt: String,
    seed: u64,
    temp: f64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(weights: Vec<u8>, tokenizer: Vec<u8>, quantized: bool) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");

        if quantized {
            return Err(JsError::new(
                "the lazy moondream wasm binary does not yet support quantized \
                 weights (no lazy_quantized_moondream module). Drop the flag to \
                 use the safetensors weights.",
            ));
        }

        // The lazy moondream config (v2 — 1152-dim vision tower, 2048-dim
        // Phi-1.5 text decoder). Kept here for future use once the loader
        // lands; suppress unused warnings via leading `_`.
        let _config = moondream_v2_lazy_config();
        console_log!("config loaded in {:?}", Date::now());

        let _tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        console_log!("weights len: {:?}", weights.len());

        // NOTE: `MoondreamWeights::load_from_mmapped` is a stub in the lazy
        // module today (see `fuel_core::lazy_moondream`). It also takes
        // `&MmapedSafetensors`, which is filesystem-backed and not directly
        // constructable in the browser from a `Vec<u8>`. The migration keeps
        // the API surface in place so the binary compiles; runtime fails
        // gracefully until the loader lands.
        let _ = weights; // accepted from JS; not consumable until the loader exists.
        // Type-check the stub signature so future API drift is caught at
        // compile time (this prevents the binary from silently rotting).
        let _stub: fn(&MmapedSafetensors, &MoondreamConfig)
            -> fuel::Result<MoondreamWeights> = MoondreamWeights::load_from_mmapped;
        Err(JsError::new(
            "MoondreamWeights::load_from_mmapped is a stub today; the lazy \
             moondream port cannot load HuggingFace safetensors yet. Track \
             progress in fuel_core::lazy_moondream.",
        ))

        // Once the loader lands and a `Vec<u8>` -> Weights pathway exists
        // (e.g. via `BufferedSafetensors`), the construction tail would be:
        //
        // let model = MoondreamModel { config: _config.clone(), weights };
        // let logits_processor = LogitsProcessor::new(299792458, None, None);
        // Ok(Self {
        //     model,
        //     config: _config,
        //     tokenizer: _tokenizer,
        //     tokens: vec![],
        //     logits_processor,
        //     repeat_penalty: 1.,
        //     repeat_last_n: 64,
        //     pixel_values: None,
        //     index: 0,
        // })
    }

    pub fn set_image_embeddings(&mut self, image: Vec<u8>) -> Result<(), JsError> {
        // The lazy `MoondreamModel::forward` consumes pixel_values directly
        // and runs the vision encoder + projection inline; there is no
        // separate `vision_encoder()` extraction point today. We therefore
        // cache the lazy-wrapped pixel tensor instead of a pre-encoded
        // image-embeddings tensor.
        console_log!("loading image as tensor");
        let start = Date::now();
        let pixel_values = self.load_image(image)?;
        console_log!("image loaded in {:?}s", (Date::now() - start) / 1000.);
        self.pixel_values = Some(pixel_values);
        Ok(())
    }

    #[wasm_bindgen]
    pub fn init_with_image_prompt(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let InitInput {
            prompt,
            seed,
            temp,
            top_p,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
        } = serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;

        let prompt = format!("\n\nQuestion: {prompt}\n\nAnswer:");
        // No per-step KV cache in the lazy v1 forward; nothing to clear.

        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();
        self.index = 0;

        // Moondream tokenizer bos_token is "<|endoftext|>"
        // https://huggingface.co/vikhyatk/moondream2/blob/main/special_tokens_map.json
        let special_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => return Err(JsError::new("BOS token not found in the tokenizer.")),
        };

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?;

        if tokens.is_empty() {
            return Err(JsError::new(
                "Empty prompts are not supported in the Moondream model.",
            ));
        }

        if verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('\u{2581}', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        // Seed `self.tokens` with `[bos] ++ prompt_tokens` so subsequent
        // `next_token` calls re-run the full prefix.
        self.tokens.push(special_token);
        self.tokens.extend(tokens.get_ids().iter().copied());
        let text = match self.process() {
            Ok(text) => text,
            Err(_e) => {
                console_log!("error decoding token");
                Output {
                    token: "".to_string(),
                    token_id: 0,
                }
            }
        };
        Ok(serde_wasm_bindgen::to_value(&text)?)
    }
    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<JsValue, JsError> {
        let text = match self.process() {
            Ok(text) => text,
            Err(_e) => {
                console_log!("error decoding token");
                Output {
                    token: "".to_string(),
                    token_id: 0,
                }
            }
        };
        Ok(serde_wasm_bindgen::to_value(&text)?)
    }
}
impl Model {
    fn load_image(&self, image: Vec<u8>) -> Result<LazyTensor, JsError> {
        let img = image::ImageReader::new(std::io::Cursor::new(image))
            .with_guessed_format()
            .map_err(|e| JsError::new(&e.to_string()))?
            .decode()
            .map_err(|e| JsError::new(&e.to_string()))?
            .resize_to_fill(378, 378, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let raw = img.into_raw(); // length = h * w * 3, layout HxWxC
        let mut nchw = vec![0.0_f32; 3 * h * w];
        // mean/std: 0.5 for all channels — pixels are (x / 255 - 0.5) / 0.5
        // so the result lives in [-1, 1].
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let src = (y * w + x) * 3 + c;
                    let dst = c * h * w + y * w + x;
                    let v = raw[src] as f32 / 255.0;
                    nchw[dst] = (v - 0.5) / 0.5;
                }
            }
        }
        Ok(LazyTensor::from_f32(
            Arc::<[f32]>::from(nchw),
            Shape::from_dims(&[1, 3, 378, 378]),
            &Device::cpu(),
        ))
    }
}

impl Model {
    fn process(&mut self) -> Result<Output, JsError> {
        let pixel_values = match &self.pixel_values {
            Some(pv) => pv,
            None => return Err(JsError::new("Pixel values are not set.")),
        };

        // Lazy `forward` always re-runs the full prefix; pass the cumulative
        // token list every step.
        let logits = self
            .model
            .forward(pixel_values, &self.tokens)
            .map_err(|e| JsError::new(&format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();

        // logits shape: (1, num_patches + text_len, vocab) — slice the last
        // text-position row.
        let vocab_size = self.config.text.vocab_size;
        let seq = self.config.vision.num_patches + self.tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();

        if self.repeat_penalty != 1. {
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            fuel_transformers::utils::apply_repeat_penalty(
                &mut last_logits,
                self.repeat_penalty,
                &self.tokens[start_at..],
            );
        }
        let next_token = self
            .logits_processor
            .sample(&last_logits)
            .map_err(|e| JsError::new(&format!("sample: {e}")))?;
        self.tokens.push(next_token);
        let token = match self.tokenizer.decode(&[next_token], true) {
            Ok(token) => token,
            Err(e) => {
                console_log!("error decoding token: {:?}", e);
                "".to_string()
            }
        };
        self.index += 1;
        Ok(Output {
            token,
            token_id: next_token,
        })
    }
}

/// Moondream-v2 lazy config — mirrors the desktop binary at
/// `fuel-examples/examples/moondream/main.rs`. The vision + projection
/// halves come from `MoondreamVisionConfig::v2()` /
/// `MoondreamProjectionConfig::v2()`; the text decoder is the Phi-1.5
/// MixFormer with the same parameters the eager `moondream::Config::v2()`
/// shipped.
fn moondream_v2_lazy_config() -> MoondreamConfig {
    use fuel::lazy_mixformer::{MixFormerActivation, MixFormerConfig};
    use fuel::lazy_moondream::{MoondreamProjectionConfig, MoondreamVisionConfig};
    MoondreamConfig {
        vision: MoondreamVisionConfig::v2(),
        projection: MoondreamProjectionConfig::v2(),
        text: MixFormerConfig {
            vocab_size: 51200,
            hidden_size: 2048,
            n_inner: None, // 4 * 2048
            num_hidden_layers: 24,
            num_attention_heads: 32,
            rotary_dim: 32,
            layer_norm_eps: 1e-5,
            max_position_embeddings: 2048,
            rope_theta: 10_000.0,
            hidden_activation: MixFormerActivation::GeluPytorchTanh,
            tie_word_embeddings: false,
        },
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
