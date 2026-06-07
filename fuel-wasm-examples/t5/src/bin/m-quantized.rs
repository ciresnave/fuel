use fuel::lazy::WeightStorage;
use fuel::lazy_t5::{
    T5Activation, T5AttentionWeights, T5Config, T5DecoderLayerWeights, T5EncoderLayerWeights,
    T5FfnWeights, T5Model, T5Weights,
};
use fuel::quantized::GgmlDType;
use fuel::quantized::gguf_file::Content;
use fuel_transformers::generation::LogitsProcessor;
use fuel_wasm_example_t5::console_log;
use std::io::Cursor;
use std::sync::Arc;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

/// Parsed HF `config.json` for T5 / Flan-T5. Mirrors the eager
/// `quantized_t5::Config` shape. The lazy [`T5Config`] only needs a strict
/// subset, so the auxiliary token-id fields are kept here.
struct LoadedT5 {
    cfg: T5Config,
    pad_token_id: u32,
    eos_token_id: u32,
    #[allow(dead_code)]
    decoder_start_token_id: Option<u32>,
}

fn parse_t5_config(json: &[u8]) -> Result<LoadedT5, JsError> {
    let v: serde_json::Value = serde_json::from_slice(json)
        .map_err(|e| JsError::new(&format!("parsing T5 config.json: {e}")))?;
    let get_usize = |key: &str| -> Result<usize, JsError> {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| JsError::new(&format!("T5 config.json: missing/invalid field {key:?}")))
    };
    let get_usize_opt = |key: &str| -> Option<usize> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_bool = |key: &str, default: bool| -> bool {
        v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
    };

    let vocab_size = get_usize("vocab_size")?;
    let d_model = get_usize("d_model")?;
    let d_kv = get_usize("d_kv")?;
    let d_ff = get_usize("d_ff")?;
    let num_layers = get_usize("num_layers")?;
    let num_decoder_layers = get_usize_opt("num_decoder_layers");
    let num_heads = get_usize("num_heads")?;
    let relative_attention_num_buckets = get_usize("relative_attention_num_buckets")?;
    let relative_attention_max_distance =
        get_usize_opt("relative_attention_max_distance").unwrap_or(128);
    let layer_norm_epsilon = get_f64("layer_norm_epsilon", 1e-6);
    let tie_word_embeddings = get_bool("tie_word_embeddings", true);

    let ffp: &str = v
        .get("feed_forward_proj")
        .and_then(|x| x.as_str())
        .unwrap_or("relu");
    let (gated_ffn, activation) = match ffp {
        "gated-gelu" => (true, T5Activation::GeluPytorchTanh),
        "gated-silu" => (true, T5Activation::Silu),
        "relu" => (false, T5Activation::Relu),
        "silu" | "swish" => (false, T5Activation::Silu),
        "gelu" => (false, T5Activation::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" => (false, T5Activation::GeluPytorchTanh),
        other => {
            if let Some(inner) = other.strip_prefix("gated-") {
                let act = match inner {
                    "gelu" => T5Activation::GeluPytorchTanh,
                    "silu" | "swish" => T5Activation::Silu,
                    "relu" => T5Activation::Relu,
                    _ => T5Activation::GeluPytorchTanh,
                };
                (true, act)
            } else {
                (false, T5Activation::Relu)
            }
        }
    };

    let pad_token_id = get_usize_opt("pad_token_id").unwrap_or(0) as u32;
    let eos_token_id = get_usize_opt("eos_token_id").unwrap_or(1) as u32;
    let decoder_start_token_id = get_usize_opt("decoder_start_token_id").map(|x| x as u32);

    Ok(LoadedT5 {
        cfg: T5Config {
            vocab_size,
            d_model,
            d_kv,
            d_ff,
            num_layers,
            num_decoder_layers,
            num_heads,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            layer_norm_epsilon,
            activation,
            gated_ffn,
            tie_word_embeddings,
        },
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
    })
}

#[wasm_bindgen]
pub struct ModelEncoder {
    model: T5Model,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
pub struct ModelConditionalGeneration {
    model: T5Model,
    tokenizer: Tokenizer,
    loaded: LoadedT5,
}

#[wasm_bindgen]
impl ModelConditionalGeneration {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
    ) -> Result<ModelConditionalGeneration, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let loaded = parse_t5_config(&config)?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let t5_weights = load_quantized_t5_weights_from_gguf_buffer(&weights, &loaded.cfg)
            .map_err(|e| JsError::new(&format!("load quantized t5 weights: {e}")))?;
        let model = T5Model {
            config: loaded.cfg.clone(),
            weights: t5_weights,
        };
        Ok(Self {
            model,
            tokenizer,
            loaded,
        })
    }

    pub fn decode(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let input: ConditionalGenerationParams =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        // Lazy port has no KV cache; clear_kv_cache is a no-op here.
        let pad_token = self.loaded.pad_token_id;
        let mut output_token_ids: Vec<u32> = vec![pad_token];
        let prompt = input.prompt;
        let repeat_penalty = input.repeat_penalty;
        let repeat_last_n = input.repeat_last_n;
        let seed = input.seed;
        let max_length = usize::clamp(input.max_length.unwrap_or(512), 0, 512);
        let temperature = if input.temperature <= 0. {
            None
        } else {
            Some(input.temperature)
        };
        let top_p = if input.top_p <= 0. || input.top_p >= 1. {
            None
        } else {
            Some(input.top_p)
        };
        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);
        let tokens: Vec<u32> = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?
            .get_ids()
            .to_vec();

        let encoder_output = self
            .model
            .forward_encoder(&tokens)
            .map_err(|e| JsError::new(&format!("encoder forward: {e}")))?;
        let vocab_size = self.loaded.cfg.vocab_size;
        let eos_token_id = self.loaded.eos_token_id;
        let mut decoded = String::new();
        for _ in 0.. {
            if output_token_ids.len() > max_length {
                break;
            }
            let logits = self
                .model
                .forward_decoder(&output_token_ids, &encoder_output)
                .map_err(|e| JsError::new(&format!("decoder forward: {e}")))?;
            let logits_data = logits.realize_f32();
            let tgt_len = output_token_ids.len();
            let last_off = (tgt_len - 1) * vocab_size;
            let mut last_logits: Vec<f32> =
                logits_data[last_off..last_off + vocab_size].to_vec();
            if repeat_penalty != 1.0 {
                let start_at = output_token_ids.len().saturating_sub(repeat_last_n);
                fuel_transformers::utils::apply_repeat_penalty(
                    &mut last_logits,
                    repeat_penalty,
                    &output_token_ids[start_at..],
                );
            }
            let next_token_id = logits_processor
                .sample(&last_logits)
                .map_err(|e| JsError::new(&format!("sample: {e}")))?;
            if next_token_id == eos_token_id {
                break;
            }
            output_token_ids.push(next_token_id);
            if let Some(text) = self.tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                decoded += &text;
            }
        }
        Ok(serde_wasm_bindgen::to_value(
            &ConditionalGenerationOutput {
                generation: decoded,
            },
        )?)
    }
}

#[wasm_bindgen]
impl ModelEncoder {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
    ) -> Result<ModelEncoder, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let loaded = parse_t5_config(&config)?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let t5_weights = load_quantized_t5_weights_from_gguf_buffer(&weights, &loaded.cfg)
            .map_err(|e| JsError::new(&format!("load quantized t5 weights: {e}")))?;
        let model = T5Model {
            config: loaded.cfg,
            weights: t5_weights,
        };
        Ok(Self { model, tokenizer })
    }

    pub fn decode(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let input: DecoderParams =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        let sentences = input.sentences;
        let normalize_embeddings = input.normalize_embeddings;
        let n_sentences = sentences.len();
        let mut all_embeddings = Vec::with_capacity(n_sentences);
        let d_model = self.model.config.d_model;
        for sentence in sentences {
            let tokens: Vec<u32> = self
                .tokenizer
                .encode(sentence, true)
                .map_err(|m| JsError::new(&m.to_string()))?
                .get_ids()
                .to_vec();
            let embeddings = self
                .model
                .forward_encoder(&tokens)
                .map_err(|e| JsError::new(&format!("encoder forward: {e}")))?;
            let dims = embeddings.shape().dims().to_vec();
            console_log!("generated embeddings {:?}", dims);
            let data = embeddings.realize_f32();
            let n_tokens = dims[1];
            let mut pooled = vec![0.0_f32; d_model];
            for t in 0..n_tokens {
                for h in 0..d_model {
                    pooled[h] += data[t * d_model + h];
                }
            }
            let inv = 1.0_f32 / (n_tokens as f32);
            for v in &mut pooled {
                *v *= inv;
            }
            if normalize_embeddings {
                let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    let inv = 1.0 / norm;
                    for x in &mut pooled {
                        *x *= inv;
                    }
                }
            }
            console_log!("pooled {:?}", pooled.len());
            all_embeddings.push(pooled);
        }

        Ok(serde_wasm_bindgen::to_value(&DecoderOutput {
            embeddings: all_embeddings,
        })?)
    }
}

// ---- GGUF buffer loader for T5 -----------------------------------------------
//
// Buffer-based analog of `QuantizedT5Model::from_gguf`. Reads a llama.cpp T5
// GGUF byte buffer and constructs `T5Weights` with Q4_0 Linear matrices and
// F32 norm gains / embedding / relative-attention bias.
//
// Mirrors `fuel-core::lazy_quantized_t5`'s tensor-name conventions:
// `token_embd.weight`, `enc.blk.{i}.attn_*`, `dec.blk.{i}.cross_attn_*`,
// `enc.output_norm.weight`, etc.
fn load_quantized_t5_weights_from_gguf_buffer(
    bytes: &[u8],
    cfg: &T5Config,
) -> Result<T5Weights, fuel::Error> {
    let mut cursor = Cursor::new(bytes);
    let content = Content::read(&mut cursor)
        .map_err(|e| fuel::Error::Msg(format!("gguf parse header: {e}")))?;
    let data_off = content.tensor_data_offset as usize;

    let get_tensor_bytes = |name: &str| -> Result<(&[u8], GgmlDType, Vec<usize>), fuel::Error> {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| fuel::Error::Msg(format!("gguf: missing tensor {name:?}")))?;
        let elems = info.shape.elem_count();
        let block_size = info.ggml_dtype.block_size();
        let bytes_len = elems / block_size * info.ggml_dtype.type_size();
        let start = data_off + info.offset as usize;
        Ok((
            &bytes[start..start + bytes_len],
            info.ggml_dtype,
            info.shape.dims().to_vec(),
        ))
    };

    let load_f32 = |name: &str| -> Result<Vec<f32>, fuel::Error> {
        let (b, dt, _) = get_tensor_bytes(name)?;
        dequant_bytes_to_f32(b, dt, name)
    };

    let load_weight =
        |name: &str, out_features: usize, in_features: usize| -> Result<WeightStorage, fuel::Error> {
            let (b, dt, dims) = get_tensor_bytes(name)?;
            let expected = out_features * in_features;
            let actual: usize = dims.iter().product();
            if actual != expected {
                return Err(fuel::Error::Msg(format!(
                    "gguf: tensor {name:?} has {actual} elements, expected {expected} for [{out_features}, {in_features}]",
                )));
            }
            match dt {
                GgmlDType::Q4_0 => Ok(WeightStorage::Q4_0 {
                    words: bytes_to_u32_arc(b),
                    bytes_len: b.len(),
                    in_features,
                    out_features,
                }),
                _ => {
                    let f32_out_in = dequant_bytes_to_f32(b, dt, name)?;
                    let mut f32_in_out = vec![0.0_f32; expected];
                    for o in 0..out_features {
                        for j in 0..in_features {
                            f32_in_out[j * out_features + o] = f32_out_in[o * in_features + j];
                        }
                    }
                    Ok(WeightStorage::F32(Arc::from(f32_in_out)))
                }
            }
        };

    let d = cfg.d_model;
    let inner = cfg.num_heads * cfg.d_kv;
    let d_ff = cfg.d_ff;
    let n_enc = cfg.num_layers;
    let n_dec = cfg.num_decoder_layers.unwrap_or(cfg.num_layers);
    let gated = cfg.gated_ffn;

    let shared_embedding = load_f32("token_embd.weight")?;
    if shared_embedding.len() != cfg.vocab_size * d {
        return Err(fuel::Error::Msg(format!(
            "gguf token_embd.weight: {} elems, expected {}x{}",
            shared_embedding.len(),
            cfg.vocab_size,
            d,
        )));
    }
    let encoder_rel_bias = load_f32("enc.blk.0.attn_rel_b.weight")?;
    let decoder_rel_bias = load_f32("dec.blk.0.attn_rel_b.weight")?;
    let expected_rel = cfg.relative_attention_num_buckets * cfg.num_heads;
    if encoder_rel_bias.len() != expected_rel {
        return Err(fuel::Error::Msg(format!(
            "gguf enc.blk.0.attn_rel_b.weight: {} elems, expected {}x{}",
            encoder_rel_bias.len(),
            cfg.relative_attention_num_buckets,
            cfg.num_heads,
        )));
    }
    if decoder_rel_bias.len() != expected_rel {
        return Err(fuel::Error::Msg(format!(
            "gguf dec.blk.0.attn_rel_b.weight: {} elems, expected {}x{}",
            decoder_rel_bias.len(),
            cfg.relative_attention_num_buckets,
            cfg.num_heads,
        )));
    }

    let load_attn = |prefix: &str,
                     q_name: &str,
                     k_name: &str,
                     v_name: &str,
                     o_name: &str|
     -> Result<T5AttentionWeights, fuel::Error> {
        let q = load_weight(&format!("{prefix}.{q_name}.weight"), inner, d)?;
        let k = load_weight(&format!("{prefix}.{k_name}.weight"), inner, d)?;
        let v = load_weight(&format!("{prefix}.{v_name}.weight"), inner, d)?;
        let o = load_weight(&format!("{prefix}.{o_name}.weight"), d, inner)?;
        Ok(T5AttentionWeights { q, k, v, o })
    };

    let load_ffn = |prefix: &str| -> Result<T5FfnWeights, fuel::Error> {
        if gated {
            let wi_0 = load_weight(&format!("{prefix}.ffn_gate.weight"), d_ff, d)?;
            let wi_1 = load_weight(&format!("{prefix}.ffn_up.weight"), d_ff, d)?;
            let wo = load_weight(&format!("{prefix}.ffn_down.weight"), d, d_ff)?;
            Ok(T5FfnWeights::Gated { wi_0, wi_1, wo })
        } else {
            let wi = load_weight(&format!("{prefix}.ffn_up.weight"), d_ff, d)?;
            let wo = load_weight(&format!("{prefix}.ffn_down.weight"), d, d_ff)?;
            Ok(T5FfnWeights::Dense { wi, wo })
        }
    };

    let mut encoder_layers: Vec<T5EncoderLayerWeights> = Vec::with_capacity(n_enc);
    for i in 0..n_enc {
        let prefix = format!("enc.blk.{i}");
        let self_attn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
        let self_attn = load_attn(&prefix, "attn_q", "attn_k", "attn_v", "attn_o")?;
        let ffn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
        let ffn = load_ffn(&prefix)?;
        encoder_layers.push(T5EncoderLayerWeights {
            self_attn_norm_gain,
            self_attn,
            ffn_norm_gain,
            ffn,
        });
    }

    let mut decoder_layers: Vec<T5DecoderLayerWeights> = Vec::with_capacity(n_dec);
    for i in 0..n_dec {
        let prefix = format!("dec.blk.{i}");
        let self_attn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(&format!("{prefix}.attn_norm.weight"))?);
        let self_attn = load_attn(&prefix, "attn_q", "attn_k", "attn_v", "attn_o")?;
        let cross_attn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(&format!("{prefix}.cross_attn_norm.weight"))?);
        let cross_attn = load_attn(
            &prefix,
            "cross_attn_q",
            "cross_attn_k",
            "cross_attn_v",
            "cross_attn_o",
        )?;
        let ffn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(&format!("{prefix}.ffn_norm.weight"))?);
        let ffn = load_ffn(&prefix)?;
        decoder_layers.push(T5DecoderLayerWeights {
            self_attn_norm_gain,
            self_attn,
            cross_attn_norm_gain,
            cross_attn,
            ffn_norm_gain,
            ffn,
        });
    }

    let encoder_final_norm_gain: Arc<[f32]> = Arc::from(load_f32("enc.output_norm.weight")?);
    let decoder_final_norm_gain: Arc<[f32]> = Arc::from(load_f32("dec.output_norm.weight")?);

    let lm_head = if content.tensor_infos.contains_key("output.weight") {
        Some(load_weight("output.weight", cfg.vocab_size, d)?)
    } else if cfg.tie_word_embeddings {
        None
    } else {
        return Err(fuel::Error::Msg(
            "gguf: output.weight absent and config has tie_word_embeddings=false".into(),
        ));
    };

    Ok(T5Weights {
        shared_embedding: Arc::from(shared_embedding),
        encoder_rel_bias: Arc::from(encoder_rel_bias),
        decoder_rel_bias: Arc::from(decoder_rel_bias),
        encoder_layers,
        decoder_layers,
        encoder_final_norm_gain,
        decoder_final_norm_gain,
        lm_head,
    })
}

fn bytes_to_u32_arc(bytes: &[u8]) -> Arc<[u32]> {
    let padded_len = bytes.len().div_ceil(4) * 4;
    let mut padded = vec![0_u8; padded_len];
    padded[..bytes.len()].copy_from_slice(bytes);
    let words: Vec<u32> = padded
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Arc::from(words)
}

fn dequant_bytes_to_f32(
    bytes: &[u8],
    dt: GgmlDType,
    name: &str,
) -> Result<Vec<f32>, fuel::Error> {
    use half::{bf16, f16};
    match dt {
        GgmlDType::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(fuel::Error::Msg(format!(
                    "gguf {name}: F32 byte count {} not multiple of 4",
                    bytes.len(),
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        GgmlDType::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(fuel::Error::Msg(format!(
                    "gguf {name}: F16 byte count {} not multiple of 2",
                    bytes.len(),
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        GgmlDType::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(fuel::Error::Msg(format!(
                    "gguf {name}: BF16 byte count {} not multiple of 2",
                    bytes.len(),
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        GgmlDType::Q4_0 => Ok(cpu_dequant_q4_0_bytes(bytes)),
        other => Err(fuel::Error::Msg(format!(
            "gguf {name}: dequant of {other:?} not supported in wasm m-quantized loader",
        ))),
    }
}

fn cpu_dequant_q4_0_bytes(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    let bpb = 18usize;
    let epb = 32usize;
    let n_blocks = bytes.len() / bpb;
    let mut out = vec![0.0_f32; n_blocks * epb];
    for b in 0..n_blocks {
        let off = b * bpb;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let base = b * epb;
        for kk in 0..16 {
            let packed = bytes[off + 2 + kk];
            let lo = (packed & 0x0F) as i32 - 8;
            let hi = ((packed >> 4) & 0x0F) as i32 - 8;
            out[base + kk] = lo as f32 * d;
            out[base + 16 + kk] = hi as f32 * d;
        }
    }
    out
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ConditionalGenerationOutput {
    generation: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DecoderOutput {
    embeddings: Vec<Vec<f32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DecoderParams {
    sentences: Vec<String>,
    normalize_embeddings: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConditionalGenerationParams {
    prompt: String,
    temperature: f64,
    seed: u64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    max_length: Option<usize>,
}

fn main() {
    console_error_panic_hook::set_once();
}
