use fuel::lazy_mixformer::{
    MixFormerActivation, MixFormerConfig, MixFormerModel, MixFormerWeights,
};
use fuel::safetensors::MmapedSafetensors;
use fuel_transformers::generation::LogitsProcessor;
use fuel_transformers::utils::apply_repeat_penalty;
use fuel_wasm_example_phi::console_log;
use js_sys::Date;
use serde::Deserialize;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

/// Wasm-side wrapper around the lazy MixFormer model.
///
/// The quantized (GGUF) branch is currently parked: no
/// `lazy_quantized_mixformer` module exists yet on the lazy substrate,
/// so toggling `quantized: true` returns a clean error rather than
/// silently routing through an architecturally incompatible loader
/// (Phi-3 has split Q/K/V projections, MixFormer has a fused
/// `Wqkv`).
enum SelectedModel {
    MixFormer(MixFormerModel),
}

#[wasm_bindgen]
pub struct Model {
    model: SelectedModel,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    /// Total tokens already fed through the model. The lazy v1
    /// MixFormer recomputes the full prefix every step (no KV
    /// cache), so this is the `start_pos` we pass to `forward`.
    start_pos: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelName {
    pub _name_or_path: String,
}

/// HF-style `config.json` shadow struct so we can `serde_json` it
/// directly. The lazy `MixFormerConfig` deliberately renames most
/// fields (`n_embd` → `hidden_size`, `n_head` → `num_attention_heads`,
/// `layer_norm_epsilon` → `layer_norm_eps`, …); this struct mirrors
/// the upstream HF field names so existing weight repos load
/// unchanged.
#[derive(Debug, Clone, Deserialize)]
struct HfMixFormerConfig {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    #[serde(default)]
    n_inner: Option<usize>,
    n_head: usize,
    rotary_dim: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_epsilon: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
}

fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10_000.0
}

impl From<HfMixFormerConfig> for MixFormerConfig {
    fn from(c: HfMixFormerConfig) -> Self {
        Self {
            vocab_size: c.vocab_size,
            hidden_size: c.n_embd,
            n_inner: c.n_inner,
            num_hidden_layers: c.n_layer,
            num_attention_heads: c.n_head,
            rotary_dim: c.rotary_dim,
            layer_norm_eps: c.layer_norm_epsilon,
            max_position_embeddings: c.n_positions,
            rope_theta: c.rope_theta,
            // HF MixFormer reference uses NewGelu, which maps to
            // PyTorch's tanh-approximate GELU on the lazy side.
            hidden_activation: MixFormerActivation::GeluPytorchTanh,
            tie_word_embeddings: c.tie_word_embeddings,
        }
    }
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let name: ModelName = serde_json::from_slice(&config)?;
        let hf_config: HfMixFormerConfig = serde_json::from_slice(&config)?;
        let cfg: MixFormerConfig = hf_config.into();

        console_log!("config loaded {:?}", name);
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let start = Date::now();
        console_log!("weights len: {:?}", weights.len());

        if quantized {
            // No `lazy_quantized_mixformer` module exists yet on
            // the lazy substrate. Routing Phi-2 GGUF through the
            // Phi-3 quantized model would silently produce garbage
            // (different Wqkv vs split Q/K/V layout), so we fail
            // loudly until the quantized port lands.
            return Err(JsError::new(
                "quantized MixFormer is not yet ported to the lazy substrate \
                 (no fuel::lazy_quantized_mixformer). Pass quantized=false \
                 and load the f32 safetensors checkpoint, or wait for the \
                 follow-up port.",
            ));
        }

        // The lazy `load_from_mmapped` API consumes an mmap-backed
        // safetensors view. wasm32 has no usable mmap, so this is a
        // known structural follow-up — a `BufferedSafetensors`
        // sibling loader on `MixFormerWeights` is the missing piece.
        // Until then, callers must place the buffer at this fixed
        // path before constructing `Model` (the wasm host bundles
        // the weights into the virtual filesystem).
        let _ = weights;
        let st = unsafe { MmapedSafetensors::new("phi.safetensors") }
            .map_err(|e| JsError::new(&format!("mmap safetensors: {e}")))?;
        let mix_weights = MixFormerWeights::load_from_mmapped(&st, &cfg)
            .map_err(|e| JsError::new(&format!("load weights: {e}")))?;
        let model = SelectedModel::MixFormer(MixFormerModel { config: cfg, weights: mix_weights });

        console_log!("model loaded in {:?}s", (Date::now() - start) / 1000.);
        let logits_processor = LogitsProcessor::new(299792458, None, None);
        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            start_pos: 0,
        })
    }

    #[wasm_bindgen]
    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> Result<String, JsError> {
        // No KV cache in the lazy v1 forward — "clearing" the
        // cache reduces to resetting our token history.
        self.start_pos = 0;
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
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?
            .get_ids()
            .to_vec();
        let text = self
            .process(&tokens)
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let last_token = *self.tokens.last().unwrap();
        let text = self
            .process(&[last_token])
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }
}

impl Model {
    fn process(&mut self, tokens: &[u32]) -> fuel::Result<String> {
        // Lazy MixFormer.forward takes a token slice + a start_pos —
        // we feed the new tokens, then advance start_pos by the
        // number of tokens we just consumed.
        let logits = match &self.model {
            SelectedModel::MixFormer(m) => m.forward(tokens, self.start_pos)?,
        };
        // logits shape: (1, seq, vocab) — host-realize and grab the
        // final row.
        let dims_owned = logits.shape().dims().to_vec();
        let logits_data = logits.realize_f32();
        let vocab = dims_owned[dims_owned.len() - 1];
        let seq = dims_owned[dims_owned.len() - 2];
        let last_off = (seq - 1) * vocab;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab].to_vec();

        if self.repeat_penalty != 1.0 {
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            apply_repeat_penalty(
                &mut last_logits,
                self.repeat_penalty,
                &self.tokens[start_at..],
            );
        }

        let next_token = self.logits_processor.sample(&last_logits)?;
        self.tokens.push(next_token);
        self.start_pos += tokens.len();
        let token = match self.tokenizer.decode(&[next_token], false) {
            Ok(token) => token,
            Err(e) => {
                console_log!("error decoding token: {:?}", e);
                "".to_string()
            }
        };
        // console_log!("token: {:?}: {:?}", token, next_token);
        Ok(token)
    }
}

fn main() {
    console_error_panic_hook::set_once();
}
