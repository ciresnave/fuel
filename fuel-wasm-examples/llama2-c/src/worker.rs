use crate::model::{Config, Llama, load_llama2c_bin};
use fuel::inference_context::{InferenceContext, KvCache};
use fuel::{DType, Device, Result};
use fuel_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use yew_agent::{HandlerId, Public, WorkerLink};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transferred via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub tokenizer: Vec<u8>,
    pub model: Vec<u8>,
}

/// Maximum decode horizon for the WASM demo. The lazy `Llama2cModel`
/// has no internal `seq_len` cap, so we pick a host-side ceiling that
/// matches the legacy llama2.c default.
const SEQ_LEN_MAX: usize = 1024;

pub struct Model {
    pub config: Config,
    pub llama: Llama,
    pub tokenizer: Tokenizer,
    /// Per-layer K/V cache. Wrapped in an `Arc<Mutex>` so the
    /// worker can reset it between generation runs (legacy public
    /// API surface; the wasm example relies on this from `bin/m.rs`).
    pub cache: Arc<Mutex<KvCache>>,
}

impl Model {
    fn run(
        &self,
        link: &WorkerLink<Worker>,
        id: HandlerId,
        temp: f64,
        top_p: f64,
        prompt: String,
    ) -> Result<()> {
        let dev = Device::cpu();
        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1.0 {
            None
        } else {
            Some(top_p)
        };
        console_log!("temp: {temp:?} top_p: {top_p:?} prompt: {prompt}");
        let mut logits_processor = LogitsProcessor::new(299792458, temp, top_p);
        let mut tokens = self
            .tokenizer
            .encode(prompt.to_string(), true)
            .map_err(|m| fuel::Error::Msg(m.to_string()))?
            .get_ids()
            .to_vec();
        link.respond(id, Ok(WorkerOutput::Generated(prompt)));

        let mut ctx = InferenceContext::new(dev.clone());
        let mut cache = self.cache.lock().unwrap();
        cache.clear();

        // Prefill: feed the full prompt; subsequent steps feed only the
        // newly sampled token thanks to the persistent KV cache.
        let logits = self
            .llama
            .forward_with_kv_context(&tokens, &mut cache, &mut ctx)?;
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if let Some(text) = self.tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            link.respond(id, Ok(WorkerOutput::Generated(text)));
        }

        loop {
            if tokens.len() >= SEQ_LEN_MAX {
                break;
            }
            let last = *tokens.last().unwrap();
            let logits =
                self.llama
                    .forward_with_kv_context(&[last], &mut cache, &mut ctx)?;
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                link.respond(id, Ok(WorkerOutput::Generated(text)));
            }
        }
        Ok(())
    }
}

impl Model {
    pub fn load(md: ModelData) -> Result<Self> {
        let dev = Device::cpu();
        let mut reader = std::io::Cursor::new(md.model);
        // `load_llama2c_bin` reads the 7-`i32` Karpathy header, the
        // F32 tensor payloads, and (for tied classifiers) ties the
        // lm_head to the embedding table — replacing the old
        // `TransformerWeights::from_reader` + `VarBuilder` dance.
        let llama = load_llama2c_bin(&mut reader)?;
        let config = llama.config.clone();
        let cache = KvCache::with_capacity(
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            SEQ_LEN_MAX,
            DType::F32,
            &dev,
        )?;
        let tokenizer =
            Tokenizer::from_bytes(&md.tokenizer).map_err(|m| fuel::Error::Msg(m.to_string()))?;
        Ok(Self {
            config,
            llama,
            tokenizer,
            cache: Arc::new(Mutex::new(cache)),
        })
    }
}

pub struct Worker {
    link: WorkerLink<Self>,
    model: Option<Model>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    Run(f64, f64, String),
}

#[derive(Serialize, Deserialize)]
pub enum WorkerOutput {
    Generated(String),
    GenerationDone(std::result::Result<(), String>),
    WeightsLoaded,
}

impl yew_agent::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = std::result::Result<WorkerOutput, String>;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Self { link, model: None }
    }

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        let output = match msg {
            WorkerInput::ModelData(md) => match Model::load(md) {
                Ok(model) => {
                    self.model = Some(model);
                    Ok(WorkerOutput::WeightsLoaded)
                }
                Err(err) => Err(format!("model creation error {err:?}")),
            },
            WorkerInput::Run(temp, top_p, prompt) => match &mut self.model {
                None => Err("model has not been set yet".to_string()),
                Some(model) => {
                    let result = model
                        .run(&self.link, id, temp, top_p, prompt)
                        .map_err(|e| e.to_string());
                    Ok(WorkerOutput::GenerationDone(result))
                }
            },
        };
        self.link.respond(id, output);
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}
