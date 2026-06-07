use fuel::lazy::{LazyTensor, WeightStorage};
use fuel::lazy_blip::{BlipConfig, BlipForConditionalGeneration, BlipWeights};
use fuel::lazy_blip_text::{
    BlipTextAttentionWeights, BlipTextConfig, BlipTextFfnWeights, BlipTextLayerWeights,
    BlipTextWeights, LayerNormWeights as TextLayerNormWeights,
};
use fuel::lazy_blip_vision::{
    BlipMlpWeights, BlipVisionAttentionWeights, BlipVisionConfig, BlipVisionLayerWeights,
    BlipVisionWeights, LayerNormWeights as VisionLayerNormWeights,
};
use fuel::safetensors::BufferedSafetensors;
use fuel::{Device, Shape};
use fuel_transformers::generation::LogitsProcessor;
use fuel_wasm_example_blip::console_log;
use fuel_wasm_example_blip::token_output_stream::TokenOutputStream;
use js_sys::Date;
use std::sync::Arc;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    model: BlipForConditionalGeneration,
    tokenizer: TokenOutputStream,
}
const SEP_TOKEN_ID: u32 = 102;

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        _config: Vec<u8>,
        quantized: bool,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let tokenizer = TokenOutputStream::new(tokenizer);

        if quantized {
            return Err(JsError::new(
                "The lazy blip port does not yet support the quantized GGUF \
                 checkpoint; only the F32 safetensors path is wired. Drop \
                 --quantized or wait for the lazy GGUF port to land.",
            ));
        }

        // The lazy BLIP port currently exposes only the
        // `Salesforce/blip-image-captioning-large` preset (matching the
        // eager wasm binary's original target).
        let config = BlipConfig::image_captioning_large();

        let start = Date::now();
        let st = BufferedSafetensors::new(weights)
            .map_err(|e| JsError::new(&format!("parse safetensors: {e}")))?;
        let weights = load_blip_weights_from_buffered(&st, &config)
            .map_err(|e| JsError::new(&format!("load blip weights: {e}")))?;
        let model = BlipForConditionalGeneration {
            config,
            weights,
        };

        console_log!("model loaded in {:?}s", (Date::now() - start) / 1000.);
        Ok(Self { model, tokenizer })
    }
    #[wasm_bindgen]
    pub fn generate_caption_from_image(&mut self, image: Vec<u8>) -> Result<String, JsError> {
        let device = Device::cpu();
        console_log!("loading image as tensor");
        let start = Date::now();
        let pixel_values = self.load_image(image, &device)?;
        console_log!("image loaded in {:?}s", (Date::now() - start) / 1000.);

        let mut logits_processor = LogitsProcessor::new(299792458, None, None);
        let mut token_ids = vec![30522u32];
        let mut text: String = "".to_string();
        let vocab_size = self.model.config.text_config.vocab_size;

        let start = Date::now();
        for _ in 0..1000 {
            // The lazy text decoder has no KV cache yet — every step
            // re-runs the full sequence through vision + text. This
            // is slow on wasm/CPU but correct.
            let logits = self
                .model
                .forward(&pixel_values, &token_ids, 0)
                .map_err(|e| JsError::new(&e.to_string()))?;
            let data = logits.realize_f32();
            let seq = token_ids.len();
            // logits has shape (1, T, vocab); pick the LAST token's row.
            let off = (seq - 1) * vocab_size;
            let last_logits = &data[off..off + vocab_size];

            let token = logits_processor
                .sample(last_logits)
                .map_err(|e| JsError::new(&e.to_string()))?;
            if token == SEP_TOKEN_ID {
                break;
            }
            token_ids.push(token);
            if let Some(t) = self.tokenizer.next_token(token)? {
                text.push_str(&t);
            }
        }
        if let Some(rest) = self
            .tokenizer
            .decode_rest()
            .map_err(|m| JsError::new(&m.to_string()))?
        {
            text.push_str(&rest);
        }
        console_log!("caption generated in {:?}s", (Date::now() - start) / 1000.);
        Ok(text)
    }
}

impl Model {
    fn load_image(&self, image: Vec<u8>, device: &Device) -> Result<LazyTensor, JsError> {
        let img = image::ImageReader::new(std::io::Cursor::new(image))
            .with_guessed_format()
            .map_err(|e| JsError::new(&e.to_string()))?
            .decode()
            .map_err(|e| JsError::new(&e.to_string()))?
            .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let raw = img.into_raw(); // (H, W, C) row-major, u8

        // OpenAI / BLIP normalization.
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let stdv = [0.26862954f32, 0.261_302_6, 0.275_777_1];

        // Convert HWC u8 → CHW f32 with normalization, then add the
        // leading batch dim so the tensor lands as (1, 3, 384, 384) —
        // matching `BlipForConditionalGeneration::forward`'s expected
        // pixel-values layout.
        let h = 384usize;
        let w = 384usize;
        let mut out = vec![0.0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = raw[(y * w + x) * 3 + c] as f32 / 255.0;
                    let v = (v - mean[c]) / stdv[c];
                    out[(c * h + y) * w + x] = v;
                }
            }
        }
        Ok(LazyTensor::from_f32(
            Arc::<[f32]>::from(out),
            Shape::from_dims(&[1, 3, h, w]),
            device,
        ))
    }
}

fn main() {
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// BufferedSafetensors → BlipWeights adapter.
//
// `BlipWeights::load_from_mmapped` and its siblings are concretely typed
// against `MmapedSafetensors`, which is unavailable on `wasm32`. This
// adapter mirrors the same loader walk but reads tensor views through
// `BufferedSafetensors` (which owns the in-memory `Vec<u8>` we got from
// JS) so the lazy substrate path works in the browser. The shape of the
// loader (every tensor name, every dtype branch, every transpose
// convention) tracks `fuel_core::lazy_blip::BlipWeights::load_from_mmapped`
// and its `lazy_blip_text` / `lazy_blip_vision` callees byte-for-byte.
// ---------------------------------------------------------------------------

fn load_blip_weights_from_buffered(
    st: &BufferedSafetensors,
    cfg: &BlipConfig,
) -> fuel::Result<BlipWeights> {
    let vision =
        load_vision_weights_from_buffered(st, &cfg.vision_config, "vision_model.")?;
    let text = load_text_weights_from_buffered(
        st,
        &cfg.text_config,
        cfg.vision_config.hidden_size,
        "text_decoder.",
    )?;
    Ok(BlipWeights { vision, text })
}

// ---- byte-level decoders (match `fuel_core::lazy::load_tensor_as_f32` /
// `load_transposed_matrix` / `load_transposed_matrix_preserve_dtype`) ----

fn buffered_tensor_as_f32(st: &BufferedSafetensors, name: &str) -> fuel::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st.get(name)?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                out.push(f64::from_le_bytes(arr) as f32);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => fuel::bail!(
            "buffered_tensor_as_f32: unsupported dtype {other:?} for tensor {name:?}"
        ),
    }
}

fn buffered_transposed_matrix_f32(
    st: &BufferedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> fuel::Result<Vec<f32>> {
    let flat = buffered_tensor_as_f32(st, name)?;
    if flat.len() != out_features * in_features {
        fuel::bail!(
            "buffered_transposed_matrix_f32: tensor {name:?} has {} elements, expected {} ({out_features} × {in_features})",
            flat.len(),
            out_features * in_features,
        );
    }
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

fn buffered_transposed_matrix_preserve_dtype(
    st: &BufferedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> fuel::Result<WeightStorage> {
    use safetensors::Dtype;
    let view = st.get(name)?;
    let bytes = view.data();
    let expected = out_features * in_features;
    match view.dtype() {
        Dtype::BF16 => {
            if bytes.len() != expected * 2 {
                fuel::bail!(
                    "buffered_transposed_matrix_preserve_dtype: bf16 tensor {name:?} has {} bytes, expected {}",
                    bytes.len(),
                    expected * 2,
                );
            }
            let mut out = vec![half::bf16::ZERO; expected];
            for i in 0..out_features {
                for j in 0..in_features {
                    let src_off = (i * in_features + j) * 2;
                    let bits = u16::from_le_bytes([bytes[src_off], bytes[src_off + 1]]);
                    out[j * out_features + i] = half::bf16::from_bits(bits);
                }
            }
            Ok(WeightStorage::BF16(Arc::from(out)))
        }
        _ => {
            // F32, F64, F16 all fall through to the f32 upcast path.
            let flat = buffered_transposed_matrix_f32(st, name, out_features, in_features)?;
            Ok(WeightStorage::F32(Arc::from(flat)))
        }
    }
}

// ---- vision (mirrors `BlipVisionWeights::load_from_mmapped`) ----

fn load_vision_ln(
    st: &BufferedSafetensors,
    prefix: &str,
) -> fuel::Result<VisionLayerNormWeights> {
    Ok(VisionLayerNormWeights {
        gain: Arc::from(buffered_tensor_as_f32(st, &format!("{prefix}.weight"))?),
        bias: Arc::from(buffered_tensor_as_f32(st, &format!("{prefix}.bias"))?),
    })
}

fn load_vision_weights_from_buffered(
    st: &BufferedSafetensors,
    cfg: &BlipVisionConfig,
    prefix: &str,
) -> fuel::Result<BlipVisionWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let patch_proj = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}embeddings.patch_embedding.weight"),
    )?);
    let patch_proj_bias = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}embeddings.patch_embedding.bias"),
    )?);
    let class_token = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}embeddings.class_embedding"),
    )?);
    let position_embedding = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}embeddings.position_embedding"),
    )?);

    let mut layers: Vec<BlipVisionLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("{prefix}encoder.layers.{i}");
        let ln1 = load_vision_ln(st, &format!("{lp}.layer_norm1"))?;
        let qkv = buffered_transposed_matrix_preserve_dtype(
            st,
            &format!("{lp}.self_attn.qkv.weight"),
            3 * h,
            h,
        )?;
        let qkv_bias = Arc::from(buffered_tensor_as_f32(
            st,
            &format!("{lp}.self_attn.qkv.bias"),
        )?);
        let projection = buffered_transposed_matrix_preserve_dtype(
            st,
            &format!("{lp}.self_attn.projection.weight"),
            h,
            h,
        )?;
        let projection_bias = Arc::from(buffered_tensor_as_f32(
            st,
            &format!("{lp}.self_attn.projection.bias"),
        )?);
        let attn = BlipVisionAttentionWeights {
            qkv,
            qkv_bias,
            projection,
            projection_bias,
        };
        let ln2 = load_vision_ln(st, &format!("{lp}.layer_norm2"))?;
        let mlp = BlipMlpWeights {
            fc1: buffered_transposed_matrix_preserve_dtype(
                st,
                &format!("{lp}.mlp.fc1.weight"),
                inter,
                h,
            )?,
            fc1_bias: Arc::from(buffered_tensor_as_f32(
                st,
                &format!("{lp}.mlp.fc1.bias"),
            )?),
            fc2: buffered_transposed_matrix_preserve_dtype(
                st,
                &format!("{lp}.mlp.fc2.weight"),
                h,
                inter,
            )?,
            fc2_bias: Arc::from(buffered_tensor_as_f32(
                st,
                &format!("{lp}.mlp.fc2.bias"),
            )?),
        };
        layers.push(BlipVisionLayerWeights { ln1, attn, ln2, mlp });
    }

    let post_layernorm = load_vision_ln(st, &format!("{prefix}post_layernorm"))?;

    Ok(BlipVisionWeights {
        patch_proj,
        patch_proj_bias,
        class_token,
        position_embedding,
        layers,
        post_layernorm,
    })
}

// ---- text decoder (mirrors `BlipTextWeights::load_from_mmapped`) ----

fn load_text_ln(
    st: &BufferedSafetensors,
    prefix: &str,
) -> fuel::Result<TextLayerNormWeights> {
    Ok(TextLayerNormWeights {
        gain: Arc::from(buffered_tensor_as_f32(st, &format!("{prefix}.weight"))?),
        bias: Arc::from(buffered_tensor_as_f32(st, &format!("{prefix}.bias"))?),
    })
}

fn load_text_attn_from_buffered(
    st: &BufferedSafetensors,
    prefix: &str,
    hidden_size: usize,
    kv_in_dim: usize,
) -> fuel::Result<BlipTextAttentionWeights> {
    Ok(BlipTextAttentionWeights {
        query: buffered_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}.self.query.weight"),
            hidden_size,
            hidden_size,
        )?,
        query_bias: Arc::from(buffered_tensor_as_f32(
            st,
            &format!("{prefix}.self.query.bias"),
        )?),
        key: buffered_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}.self.key.weight"),
            hidden_size,
            kv_in_dim,
        )?,
        key_bias: Arc::from(buffered_tensor_as_f32(
            st,
            &format!("{prefix}.self.key.bias"),
        )?),
        value: buffered_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}.self.value.weight"),
            hidden_size,
            kv_in_dim,
        )?,
        value_bias: Arc::from(buffered_tensor_as_f32(
            st,
            &format!("{prefix}.self.value.bias"),
        )?),
        out_dense: buffered_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}.output.dense.weight"),
            hidden_size,
            hidden_size,
        )?,
        out_dense_bias: Arc::from(buffered_tensor_as_f32(
            st,
            &format!("{prefix}.output.dense.bias"),
        )?),
        out_ln: load_text_ln(st, &format!("{prefix}.output.LayerNorm"))?,
    })
}

fn load_text_weights_from_buffered(
    st: &BufferedSafetensors,
    cfg: &BlipTextConfig,
    encoder_hidden_size: usize,
    prefix: &str,
) -> fuel::Result<BlipTextWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let word_embedding = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}bert.embeddings.word_embeddings.weight"),
    )?);
    let position_embedding = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}bert.embeddings.position_embeddings.weight"),
    )?);
    let embed_ln = load_text_ln(st, &format!("{prefix}bert.embeddings.LayerNorm"))?;

    let mut layers: Vec<BlipTextLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("{prefix}bert.encoder.layer.{i}");
        let self_attn = load_text_attn_from_buffered(st, &format!("{lp}.attention"), h, h)?;
        let cross_attn = load_text_attn_from_buffered(
            st,
            &format!("{lp}.crossattention"),
            h,
            encoder_hidden_size,
        )?;
        let ffn = BlipTextFfnWeights {
            intermediate: buffered_transposed_matrix_preserve_dtype(
                st,
                &format!("{lp}.intermediate.dense.weight"),
                inter,
                h,
            )?,
            intermediate_bias: Arc::from(buffered_tensor_as_f32(
                st,
                &format!("{lp}.intermediate.dense.bias"),
            )?),
            output: buffered_transposed_matrix_preserve_dtype(
                st,
                &format!("{lp}.output.dense.weight"),
                h,
                inter,
            )?,
            output_bias: Arc::from(buffered_tensor_as_f32(
                st,
                &format!("{lp}.output.dense.bias"),
            )?),
            output_ln: load_text_ln(st, &format!("{lp}.output.LayerNorm"))?,
        };
        layers.push(BlipTextLayerWeights {
            self_attn,
            cross_attn,
            ffn,
        });
    }

    let pred_dense = buffered_transposed_matrix_preserve_dtype(
        st,
        &format!("{prefix}cls.predictions.transform.dense.weight"),
        h,
        h,
    )?;
    let pred_dense_bias = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}cls.predictions.transform.dense.bias"),
    )?);
    let pred_ln = load_text_ln(st, &format!("{prefix}cls.predictions.transform.LayerNorm"))?;

    let lm_head = buffered_transposed_matrix_preserve_dtype(
        st,
        &format!("{prefix}cls.predictions.decoder.weight"),
        cfg.vocab_size,
        h,
    )?;
    let lm_head_bias = Arc::from(buffered_tensor_as_f32(
        st,
        &format!("{prefix}cls.predictions.bias"),
    )?);

    Ok(BlipTextWeights {
        word_embedding,
        position_embedding,
        embed_ln,
        layers,
        pred_dense,
        pred_dense_bias,
        pred_ln,
        lm_head,
        lm_head_bias,
    })
}
