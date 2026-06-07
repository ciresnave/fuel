use fuel::lazy_bert::{BertConfig, BertLayerWeights, BertModel, BertWeights};
use fuel::safetensors::BufferedSafetensors;
use fuel_wasm_example_bert::console_log;
use std::sync::Arc;
use tokenizers::{PaddingParams, Tokenizer};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    bert:      BertModel,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(weights: Vec<u8>, tokenizer: Vec<u8>, config: Vec<u8>) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let st = BufferedSafetensors::new(weights).map_err(|e| JsError::new(&e.to_string()))?;
        let config: BertConfig =
            serde_json::from_slice(&config).map_err(|e| JsError::new(&e.to_string()))?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let weights =
            load_bert_weights(&st, &config).map_err(|e| JsError::new(&e.to_string()))?;
        let bert = BertModel::new(config, weights);

        Ok(Self { bert, tokenizer })
    }

    pub fn get_embeddings(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let input: Params =
            serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;
        let sentences = input.sentences;
        let normalize_embeddings = input.normalize_embeddings;

        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(|m| JsError::new(&m.to_string()))?;

        // Lazy BertModel processes one sequence at a time; iterate
        // per-sentence and compute the mean-pooled embedding via the
        // host-side f32 vector returned by `realize_f32()`. Bidirectional
        // attention has no native incremental decode, so a Python-style
        // batched call is just N forward passes here.
        let hidden_size = self.bert.config.hidden_size;
        let mut embeddings_data: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        for sentence_tokens in tokens.iter() {
            let token_ids: Vec<u32> = sentence_tokens.get_ids().to_vec();
            let seq = token_ids.len();
            console_log!("running inference on sequence of {seq} tokens");
            let hidden = self
                .bert
                .forward(&token_ids)
                .map_err(|e| JsError::new(&e.to_string()))?;
            let flat = hidden.realize_f32();
            // `flat` is `[1, seq, hidden_size]` laid out row-major; take
            // a per-feature mean across the seq axis.
            let mut pooled = vec![0.0_f32; hidden_size];
            for t in 0..seq {
                for h in 0..hidden_size {
                    pooled[h] += flat[t * hidden_size + h];
                }
            }
            let denom = seq as f32;
            for v in pooled.iter_mut() {
                *v /= denom;
            }
            if normalize_embeddings {
                let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in pooled.iter_mut() {
                        *v /= norm;
                    }
                }
            }
            embeddings_data.push(pooled);
        }
        console_log!("generated {} embeddings", embeddings_data.len());
        Ok(serde_wasm_bindgen::to_value(&Embeddings {
            data: embeddings_data,
        })?)
    }
}

/// Build a `BertWeights` from a `BufferedSafetensors`. Mirrors the
/// `BertWeights::load_from_mmapped` loader in `fuel-core::lazy_bert`,
/// but speaks `BufferedSafetensors` (owning `Vec<u8>`) so it works
/// under wasm32 where `memmap2`-backed `MmapedSafetensors` doesn't.
fn load_bert_weights(
    st: &BufferedSafetensors,
    cfg: &BertConfig,
) -> Result<BertWeights, fuel::Error> {
    let h = cfg.hidden_size;
    let h_ff = cfg.intermediate_size;

    let prefix = detect_prefix(st);

    let word_embeddings = load_f32(st, &format!("{prefix}embeddings.word_embeddings.weight"))?;
    if word_embeddings.len() != cfg.vocab_size * h {
        return Err(fuel::Error::Msg(format!(
            "word_embeddings: {} elements, expected {} ({}x{})",
            word_embeddings.len(),
            cfg.vocab_size * h,
            cfg.vocab_size,
            h,
        )));
    }
    let position_embeddings =
        load_f32(st, &format!("{prefix}embeddings.position_embeddings.weight"))?;
    let token_type_embeddings = load_f32(
        st,
        &format!("{prefix}embeddings.token_type_embeddings.weight"),
    )?;
    let emb_ln_stem = format!("{prefix}embeddings.LayerNorm");
    let emb_ln_gamma = load_layer_norm_param(st, &emb_ln_stem, true)?;
    let emb_ln_beta = load_layer_norm_param(st, &emb_ln_stem, false)?;

    let mut layers: Vec<BertLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{prefix}encoder.layer.{i}");
        let attn_q_w = load_transposed(st, &format!("{p}.attention.self.query.weight"), h, h)?;
        let attn_q_b = load_f32(st, &format!("{p}.attention.self.query.bias"))?;
        let attn_k_w = load_transposed(st, &format!("{p}.attention.self.key.weight"), h, h)?;
        let attn_k_b = load_f32(st, &format!("{p}.attention.self.key.bias"))?;
        let attn_v_w = load_transposed(st, &format!("{p}.attention.self.value.weight"), h, h)?;
        let attn_v_b = load_f32(st, &format!("{p}.attention.self.value.bias"))?;
        let attn_out_w =
            load_transposed(st, &format!("{p}.attention.output.dense.weight"), h, h)?;
        let attn_out_b = load_f32(st, &format!("{p}.attention.output.dense.bias"))?;
        let attn_ln_stem = format!("{p}.attention.output.LayerNorm");
        let attn_ln_gamma = load_layer_norm_param(st, &attn_ln_stem, true)?;
        let attn_ln_beta = load_layer_norm_param(st, &attn_ln_stem, false)?;
        let ffn_in_w = load_transposed(st, &format!("{p}.intermediate.dense.weight"), h_ff, h)?;
        let ffn_in_b = load_f32(st, &format!("{p}.intermediate.dense.bias"))?;
        let ffn_out_w = load_transposed(st, &format!("{p}.output.dense.weight"), h, h_ff)?;
        let ffn_out_b = load_f32(st, &format!("{p}.output.dense.bias"))?;
        let ffn_ln_stem = format!("{p}.output.LayerNorm");
        let ffn_ln_gamma = load_layer_norm_param(st, &ffn_ln_stem, true)?;
        let ffn_ln_beta = load_layer_norm_param(st, &ffn_ln_stem, false)?;
        layers.push(BertLayerWeights {
            attn_q_w:      Arc::from(attn_q_w),
            attn_q_b:      Arc::from(attn_q_b),
            attn_k_w:      Arc::from(attn_k_w),
            attn_k_b:      Arc::from(attn_k_b),
            attn_v_w:      Arc::from(attn_v_w),
            attn_v_b:      Arc::from(attn_v_b),
            attn_out_w:    Arc::from(attn_out_w),
            attn_out_b:    Arc::from(attn_out_b),
            attn_ln_gamma: Arc::from(attn_ln_gamma),
            attn_ln_beta:  Arc::from(attn_ln_beta),
            ffn_in_w:      Arc::from(ffn_in_w),
            ffn_in_b:      Arc::from(ffn_in_b),
            ffn_out_w:     Arc::from(ffn_out_w),
            ffn_out_b:     Arc::from(ffn_out_b),
            ffn_ln_gamma:  Arc::from(ffn_ln_gamma),
            ffn_ln_beta:   Arc::from(ffn_ln_beta),
        });
    }

    Ok(BertWeights {
        word_embeddings:       Arc::from(word_embeddings),
        position_embeddings:   Arc::from(position_embeddings),
        token_type_embeddings: Arc::from(token_type_embeddings),
        emb_ln_gamma:          Arc::from(emb_ln_gamma),
        emb_ln_beta:           Arc::from(emb_ln_beta),
        layers,
    })
}

fn detect_prefix(st: &BufferedSafetensors) -> String {
    for p in ["bert.", "distilbert."] {
        let probe = format!("{p}embeddings.word_embeddings.weight");
        if st.get(&probe).is_ok() {
            return p.to_string();
        }
    }
    String::new()
}

fn load_layer_norm_param(
    st: &BufferedSafetensors,
    stem: &str,
    is_weight: bool,
) -> Result<Vec<f32>, fuel::Error> {
    let (modern, legacy) = if is_weight {
        (".weight", ".gamma")
    } else {
        (".bias", ".beta")
    };
    let m = format!("{stem}{modern}");
    if st.get(&m).is_ok() {
        return load_f32(st, &m);
    }
    let l = format!("{stem}{legacy}");
    if st.get(&l).is_ok() {
        return load_f32(st, &l);
    }
    Err(fuel::Error::Msg(format!(
        "LayerNorm param not found under {stem:?}: tried {m:?} and {l:?}"
    )))
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

fn load_transposed(
    st: &BufferedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>, fuel::Error> {
    let flat = load_f32(st, name)?;
    if flat.len() != out_features * in_features {
        return Err(fuel::Error::Msg(format!(
            "load_transposed: tensor {name:?} has {} elements, expected {} ({}x{})",
            flat.len(),
            out_features * in_features,
            out_features,
            in_features,
        )));
    }
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Embeddings {
    data: Vec<Vec<f32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Params {
    sentences:            Vec<String>,
    normalize_embeddings: bool,
}
fn main() {
    console_error_panic_hook::set_once();
}
