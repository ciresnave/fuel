use fuel::lazy::WeightStorage;
use fuel::lazy_t5::{T5Activation, T5Config, T5Model, T5Weights};
use fuel::safetensors::BufferedSafetensors;
use fuel_transformers::generation::LogitsProcessor;
use fuel_wasm_example_t5::console_log;
use std::sync::Arc;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

/// Parsed HF `config.json` for T5 / mT5. The lazy [`T5Config`] only needs a
/// strict subset of the fields HuggingFace publishes, so we hold on to the
/// auxiliary fields (`pad_token_id`, `eos_token_id`,
/// `decoder_start_token_id`) here.
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

    // Parse `feed_forward_proj` — HF uses strings like "relu", "gated-gelu",
    // "gated-silu", "gelu_new", etc.
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
        let st = BufferedSafetensors::new(weights)
            .map_err(|e| JsError::new(&format!("safetensors deserialize: {e}")))?;
        let loaded = parse_t5_config(&config)?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let weights = load_t5_weights(&st, &loaded.cfg)
            .map_err(|e| JsError::new(&format!("load t5 weights: {e}")))?;
        let model = T5Model {
            config: loaded.cfg.clone(),
            weights,
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
            // Lazy port has no KV cache — each step re-runs the decoder over
            // the full target prefix.
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
        let st = BufferedSafetensors::new(weights)
            .map_err(|e| JsError::new(&format!("safetensors deserialize: {e}")))?;
        let loaded = parse_t5_config(&config)?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let weights = load_t5_weights(&st, &loaded.cfg)
            .map_err(|e| JsError::new(&format!("load t5 weights: {e}")))?;
        let model = T5Model {
            config: loaded.cfg,
            weights,
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
            // Shape is (1, n_tokens, d_model); mean-pool across token axis.
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

// ---- T5 weight loader against `BufferedSafetensors` --------------------------
//
// Mirrors `T5Weights::load_from_mmapped` in `fuel-core::lazy_t5` but speaks
// `BufferedSafetensors` (owning `Vec<u8>`) so it works under wasm32 where
// `memmap2`-backed `MmapedSafetensors` is not usable.

fn load_t5_weights(
    st: &BufferedSafetensors,
    cfg: &T5Config,
) -> Result<T5Weights, fuel::Error> {
    let d = cfg.d_model;
    let inner = cfg.num_heads * cfg.d_kv;
    let d_ff = cfg.d_ff;
    let n_enc = cfg.num_layers;
    let n_dec = cfg.num_decoder_layers.unwrap_or(cfg.num_layers);
    let gated = cfg.gated_ffn;
    let ffn_name = if gated { "DenseGatedActDense" } else { "DenseReluDense" };

    let shared_embedding: Arc<[f32]> = Arc::from(load_f32(st, "shared.weight")?);
    let encoder_rel_bias: Arc<[f32]> = Arc::from(load_f32(
        st,
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    )?);
    let decoder_rel_bias: Arc<[f32]> = Arc::from(load_f32(
        st,
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    )?);

    let mut encoder_layers = Vec::with_capacity(n_enc);
    for i in 0..n_enc {
        let p = format!("encoder.block.{i}");
        let self_attn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(st, &format!("{p}.layer.0.layer_norm.weight"))?);
        let self_attn =
            load_t5_attention(st, &format!("{p}.layer.0.SelfAttention"), d, inner)?;
        let ffn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(st, &format!("{p}.layer.1.layer_norm.weight"))?);
        let ffn = load_t5_ffn(st, &format!("{p}.layer.1.{ffn_name}"), d, d_ff, gated)?;
        encoder_layers.push(fuel::lazy_t5::T5EncoderLayerWeights {
            self_attn_norm_gain,
            self_attn,
            ffn_norm_gain,
            ffn,
        });
    }

    let mut decoder_layers = Vec::with_capacity(n_dec);
    for i in 0..n_dec {
        let p = format!("decoder.block.{i}");
        let self_attn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(st, &format!("{p}.layer.0.layer_norm.weight"))?);
        let self_attn =
            load_t5_attention(st, &format!("{p}.layer.0.SelfAttention"), d, inner)?;
        let cross_attn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(st, &format!("{p}.layer.1.layer_norm.weight"))?);
        let cross_attn =
            load_t5_attention(st, &format!("{p}.layer.1.EncDecAttention"), d, inner)?;
        let ffn_norm_gain: Arc<[f32]> =
            Arc::from(load_f32(st, &format!("{p}.layer.2.layer_norm.weight"))?);
        let ffn = load_t5_ffn(st, &format!("{p}.layer.2.{ffn_name}"), d, d_ff, gated)?;
        decoder_layers.push(fuel::lazy_t5::T5DecoderLayerWeights {
            self_attn_norm_gain,
            self_attn,
            cross_attn_norm_gain,
            cross_attn,
            ffn_norm_gain,
            ffn,
        });
    }

    let encoder_final_norm_gain: Arc<[f32]> =
        Arc::from(load_f32(st, "encoder.final_layer_norm.weight")?);
    let decoder_final_norm_gain: Arc<[f32]> =
        Arc::from(load_f32(st, "decoder.final_layer_norm.weight")?);

    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load_transposed_preserve_dtype(
            st,
            "lm_head.weight",
            cfg.vocab_size,
            d,
        )?)
    };

    Ok(T5Weights {
        shared_embedding,
        encoder_rel_bias,
        decoder_rel_bias,
        encoder_layers,
        decoder_layers,
        encoder_final_norm_gain,
        decoder_final_norm_gain,
        lm_head,
    })
}

fn load_t5_attention(
    st: &BufferedSafetensors,
    prefix: &str,
    d_model: usize,
    inner_dim: usize,
) -> Result<fuel::lazy_t5::T5AttentionWeights, fuel::Error> {
    let q = load_transposed_preserve_dtype(st, &format!("{prefix}.q.weight"), inner_dim, d_model)?;
    let k = load_transposed_preserve_dtype(st, &format!("{prefix}.k.weight"), inner_dim, d_model)?;
    let v = load_transposed_preserve_dtype(st, &format!("{prefix}.v.weight"), inner_dim, d_model)?;
    let o = load_transposed_preserve_dtype(st, &format!("{prefix}.o.weight"), d_model, inner_dim)?;
    Ok(fuel::lazy_t5::T5AttentionWeights { q, k, v, o })
}

fn load_t5_ffn(
    st: &BufferedSafetensors,
    prefix: &str,
    d_model: usize,
    d_ff: usize,
    gated: bool,
) -> Result<fuel::lazy_t5::T5FfnWeights, fuel::Error> {
    if gated {
        let wi_0 = load_transposed_preserve_dtype(st, &format!("{prefix}.wi_0.weight"), d_ff, d_model)?;
        let wi_1 = load_transposed_preserve_dtype(st, &format!("{prefix}.wi_1.weight"), d_ff, d_model)?;
        let wo = load_transposed_preserve_dtype(st, &format!("{prefix}.wo.weight"), d_model, d_ff)?;
        Ok(fuel::lazy_t5::T5FfnWeights::Gated { wi_0, wi_1, wo })
    } else {
        let wi = load_transposed_preserve_dtype(st, &format!("{prefix}.wi.weight"), d_ff, d_model)?;
        let wo = load_transposed_preserve_dtype(st, &format!("{prefix}.wo.weight"), d_model, d_ff)?;
        Ok(fuel::lazy_t5::T5FfnWeights::Dense { wi, wo })
    }
}

fn load_f32(st: &BufferedSafetensors, name: &str) -> Result<Vec<f32>, fuel::Error> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| fuel::Error::Msg(format!("load_f32 {name:?}: {e}")))?;
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
        other => Err(fuel::Error::Msg(format!(
            "load_f32: unsupported dtype {other:?} for tensor {name:?}"
        ))),
    }
}

/// Buffered analog of `load_transposed_matrix_preserve_dtype`. Reads a
/// `[out_features, in_features]` weight matrix from `st`, transposes to
/// `[in_features, out_features]`, and wraps the result in a
/// [`WeightStorage`]. Preserves BF16 source storage; upcasts F16/F64 to
/// F32 (matching the upstream loader's behavior).
fn load_transposed_preserve_dtype(
    st: &BufferedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<WeightStorage, fuel::Error> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| fuel::Error::Msg(format!("load {name:?}: {e}")))?;
    let bytes = view.data();
    let expected = out_features * in_features;
    match view.dtype() {
        Dtype::BF16 => {
            if bytes.len() != expected * 2 {
                return Err(fuel::Error::Msg(format!(
                    "{name:?}: expected {} BF16 elems, got {} bytes",
                    expected,
                    bytes.len(),
                )));
            }
            let mut out = vec![half::bf16::from_f32(0.0); expected];
            for i in 0..out_features {
                for j in 0..in_features {
                    let src = i * in_features + j;
                    let raw =
                        u16::from_le_bytes([bytes[src * 2], bytes[src * 2 + 1]]);
                    out[j * out_features + i] = half::bf16::from_bits(raw);
                }
            }
            Ok(WeightStorage::BF16(Arc::from(out)))
        }
        _ => {
            let flat = load_f32(st, name)?;
            if flat.len() != expected {
                return Err(fuel::Error::Msg(format!(
                    "{name:?}: {} elems, expected {} ({}x{})",
                    flat.len(),
                    expected,
                    out_features,
                    in_features,
                )));
            }
            let mut out = vec![0.0_f32; expected];
            for i in 0..out_features {
                for j in 0..in_features {
                    out[j * out_features + i] = flat[i * in_features + j];
                }
            }
            Ok(WeightStorage::F32(Arc::from(out)))
        }
    }
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
