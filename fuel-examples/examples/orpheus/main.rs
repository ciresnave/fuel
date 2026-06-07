#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use fuel::lazy::{LazyTensor, LlamaConfig, LlamaModel, LlamaWeights};
use fuel::lazy_llama_full::{build_llama3_model, Llama3Model, LlamaFullConfig};
use fuel::lazy_snac::{SnacConfig, SnacModel, SnacWeights};
use fuel::Shape;
use serde::Deserialize;
use tokenizers::Tokenizer;

// https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/realtime_streaming_example/main.py#L43
const STOP_TOKEN_ID: u32 = 128258;

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long, default_value = "Hey, how are you doing today?")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    /// The output wav file.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    #[arg(long, default_value = "3b-0.1-ft")]
    which: Which,

    #[arg(long, default_value = "tara")]
    voice: Voice,

    #[arg(long)]
    use_flash_attn: bool,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Voice {
    #[value(name = "tara")]
    Tara,
    #[value(name = "leah")]
    Leah,
    #[value(name = "jess")]
    Jess,
    #[value(name = "leo")]
    Leo,
    #[value(name = "dan")]
    Dan,
    #[value(name = "mia")]
    Mia,
    #[value(name = "zac")]
    Zac,
    #[value(name = "zoe")]
    Zoe,
}

impl Voice {
    fn as_str(&self) -> &'static str {
        match self {
            Voice::Tara => "tara",
            Voice::Leah => "leah",
            Voice::Jess => "jess",
            Voice::Leo => "leo",
            Voice::Dan => "dan",
            Voice::Mia => "mia",
            Voice::Zac => "zac",
            Voice::Zoe => "zoe",
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "3b-0.1-ft")]
    ThreeB0_1Ft,
}

#[derive(Debug, Deserialize)]
struct HfSnacConfig {
    #[serde(default = "default_audio_channels")]
    audio_channels: usize,
    encoder_dim: usize,
    decoder_dim: usize,
    decoder_rates: Vec<usize>,
    #[serde(default)]
    attn_window_size: Option<usize>,
    codebook_size: usize,
    codebook_dim: usize,
    vq_strides: Vec<usize>,
    #[serde(default)]
    noise: bool,
    #[serde(default)]
    depthwise: bool,
}

fn default_audio_channels() -> usize {
    1
}

impl From<HfSnacConfig> for SnacConfig {
    fn from(c: HfSnacConfig) -> Self {
        SnacConfig {
            audio_channels: c.audio_channels,
            encoder_dim: c.encoder_dim,
            decoder_dim: c.decoder_dim,
            decoder_rates: c.decoder_rates,
            attn_window_size: c.attn_window_size,
            codebook_size: c.codebook_size,
            codebook_dim: c.codebook_dim,
            vq_strides: c.vq_strides,
            noise: c.noise,
            depthwise: c.depthwise,
        }
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        fuel::utils::with_avx(),
        fuel::utils::with_neon(),
        fuel::utils::with_simd128(),
        fuel::utils::with_f16c()
    );
    let prompt = args.prompt.clone();
    let mut model = Model::load(args)?;
    model.run(&prompt)?;
    Ok(())
}

struct Model {
    llama: Llama3Model,
    llama_cfg: LlamaConfig,
    tokenizer: Tokenizer,
    verbose_prompt: bool,
    snac: SnacModel,
    snac_cfg: SnacConfig,
    out_file: String,
    voice: Voice,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: u64,
}

fn load_snac() -> Result<(SnacModel, SnacConfig)> {
    let api = hf_hub::api::sync::Api::new()?;
    let m = api.model("hubertsiuzdak/snac_24khz".to_string());
    let config_path = m.get("config.json")?;
    let cfg_json: HfSnacConfig =
        serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let cfg: SnacConfig = cfg_json.into();
    let m = api.model("lmz/fuel-snac".to_string());
    let model_path = m.get("snac_24khz.safetensors")?;
    let st = unsafe { fuel::safetensors::MmapedSafetensors::new(&model_path) }
        .map_err(|e| E::msg(format!("mmap snac safetensors: {e}")))?;
    let weights = SnacWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load snac weights: {e}")))?;
    let model = SnacModel {
        config: cfg.clone(),
        weights,
    };
    Ok((model, cfg))
}

impl Model {
    fn load(args: Args) -> Result<Self> {
        let start = std::time::Instant::now();
        let api = hf_hub::api::sync::Api::new()?;
        let model_id = match args.model_id {
            Some(model_id) => model_id.to_string(),
            None => match args.which {
                Which::ThreeB0_1Ft => "canopylabs/orpheus-3b-0.1-ft".to_string(),
            },
        };
        let revision = match args.revision {
            Some(r) => r,
            None => "main".to_string(),
        };
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id,
            hf_hub::RepoType::Model,
            revision,
        ));
        let model_files = match args.model_file {
            Some(m) => vec![m.into()],
            None => match args.which {
                Which::ThreeB0_1Ft => {
                    fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
                }
            },
        };
        let config_path = match args.config_file {
            Some(m) => m.into(),
            None => repo.get("config.json")?,
        };
        let tokenizer_path = match args.tokenizer_file {
            Some(m) => m.into(),
            None => repo.get("tokenizer.json")?,
        };
        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let start = std::time::Instant::now();
        let _ = fuel_examples::device(args.cpu)?;
        let _ = args.use_flash_attn;

        let config_str = std::fs::read_to_string(&config_path)?;
        let full_cfg = LlamaFullConfig::from_hf_json_str(&config_str)
            .map_err(|e| E::msg(format!("parsing orpheus config.json: {e}")))?;
        let llama_cfg: LlamaConfig = full_cfg.to_lazy_config();

        let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&model_files) }
            .map_err(|e| E::msg(format!("mmap orpheus safetensors: {e}")))?;
        let weights: LlamaWeights = LlamaWeights::load_from_mmapped(&st, &llama_cfg)
            .map_err(|e| E::msg(format!("load orpheus weights: {e}")))?;
        let inner = LlamaModel {
            config: llama_cfg.clone(),
            weights,
        };
        let llama = build_llama3_model(&full_cfg, inner.weights.clone());
        let llama = Llama3Model {
            inner: LlamaModel {
                config: llama_cfg.clone(),
                weights: llama.inner.weights,
            },
            rope_scaling: llama.rope_scaling,
            eos_token_id: llama.eos_token_id,
        };

        println!("loaded the model in {:?}", start.elapsed());

        let (snac, snac_cfg) = load_snac()?;
        Ok(Self {
            llama,
            llama_cfg,
            tokenizer,
            verbose_prompt: args.verbose_prompt,
            snac,
            snac_cfg,
            voice: args.voice,
            out_file: args.out_file,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
        })
    }

    fn run(&mut self, prompt: &str) -> Result<()> {
        println!("running the model on '{prompt}'");
        let prompt = format!("{voice}: {prompt}", voice = self.voice.as_str());
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        // https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/orpheus_tts_pypi/orpheus_tts/engine_class.py#L82
        let mut tokens = [
            &[128259],
            tokens.get_ids(),
            &[128009, 128260, 128261, 128257],
        ]
        .concat();
        if self.verbose_prompt {
            println!("{tokens:?}");
        }

        println!("starting the inference loop");
        let mut audio_tokens = vec![];
        let vocab_size = self.llama_cfg.vocab_size;
        for index in 0..2000 {
            // Lazy LLaMA forward re-feeds the full prefix each step (no KV cache yet).
            let logits = self
                .llama
                .forward(&tokens, 0)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            let logits_data = logits.realize_f32();
            let seq = tokens.len();
            let last_off = (seq - 1) * vocab_size;
            let last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();

            let next_token = sample(
                &last_logits,
                self.temperature as f32,
                self.top_k,
                self.top_p.map(|p| p as f32),
                self.seed.wrapping_add(index as u64),
            );
            if let Some(tok) = self.tokenizer.id_to_token(next_token) {
                match tok.strip_prefix("<custom_token_") {
                    Some(tok) => match tok.strip_suffix('>') {
                        Some(tok) => {
                            let tok = tok.parse::<u32>()?;
                            // https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/orpheus_tts_pypi/orpheus_tts/decoder.py#L86C35-L86C63
                            let tok = tok - 10 - ((audio_tokens.len() as u32 % 7) * 4096);
                            audio_tokens.push(tok);
                        }
                        None => {
                            println!("{index}: unexpected custom token {next_token} {tok}");
                        }
                    },
                    None => {
                        println!("{index}: unexpected token {next_token} {tok}");
                    }
                }
            }
            if next_token == STOP_TOKEN_ID {
                println!("reached stop token");
                break;
            }
            tokens.push(next_token);
        }
        println!("generated {} audio tokens", audio_tokens.len());
        let mut codes0 = vec![];
        let mut codes1 = vec![];
        let mut codes2 = vec![];
        for audio_tokens in audio_tokens.chunks_exact(7) {
            codes0.push(audio_tokens[0]);
            for i in [1, 4] {
                codes1.push(audio_tokens[i]);
            }
            for i in [2, 3, 5, 6] {
                codes2.push(audio_tokens[i]);
            }
        }
        // Anchor LazyTensor so const_u32_like has something to hang off of.
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32],
            Shape::from_dims(&[1]),
            &fuel::Device::cpu(),
        );
        let t0 = codes0.len();
        let t1 = codes1.len();
        let t2 = codes2.len();
        let codes0_lt = anchor.const_u32_like(codes0, Shape::from_dims(&[1, t0]));
        let codes1_lt = anchor.const_u32_like(codes1, Shape::from_dims(&[1, t1]));
        let codes2_lt = anchor.const_u32_like(codes2, Shape::from_dims(&[1, t2]));
        let pcm = self
            .snac
            .decode_codes(&[codes0_lt, codes1_lt, codes2_lt])
            .map_err(|e| E::msg(format!("snac decode: {e}")))?;
        println!("decoded to pcm shape: {:?}", pcm.shape().dims());
        let pcm_data = pcm.realize_f32();
        // Output shape is (1, audio_channels, T). Extract first batch, first channel.
        let total = pcm_data.len();
        let t = total / self.snac_cfg.audio_channels.max(1);
        let pcm_ch0: Vec<f32> = pcm_data[..t].to_vec();
        let mut output = std::fs::File::create(&self.out_file)?;
        fuel_examples::wav::write_pcm_as_wav(&mut output, &pcm_ch0, 24000)?;
        Ok(())
    }
}

fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: u64,
) -> u32 {
    if temperature <= 0.0 {
        let mut best_i = 0usize;
        let mut best = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        return best_i as u32;
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_l) * inv_t).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum.max(1e-30);
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut keep_mask: Vec<bool> = vec![true; probs.len()];
    if let Some(k) = top_k {
        for &i in idx.iter().skip(k) {
            keep_mask[i] = false;
        }
    }
    if let Some(p_cut) = top_p {
        let mut cum2 = 0.0;
        let mut allow = true;
        for &i in &idx {
            if !keep_mask[i] {
                continue;
            }
            if !allow {
                keep_mask[i] = false;
                continue;
            }
            cum2 += probs[i];
            if cum2 >= p_cut {
                allow = false;
            }
        }
    }
    let mut filtered: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, p)| if keep_mask[i] { *p } else { 0.0 })
        .collect();
    let s: f32 = filtered.iter().sum();
    if s > 0.0 {
        for v in &mut filtered {
            *v /= s;
        }
    } else {
        return 0;
    }
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    let r = (state as f32) / (u64::MAX as f32);
    let mut cum = 0.0;
    for (i, p) in filtered.iter().enumerate() {
        cum += *p;
        if r <= cum {
            return i as u32;
        }
    }
    (filtered.len() - 1) as u32
}
