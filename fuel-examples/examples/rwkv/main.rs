#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

// The RWKV byte-pair tokenizer is a small custom helper that has no equivalent
// in the lazy-graph module set, so keep importing it from fuel_transformers
// (this is purely a tokenization utility — no eager Tensor ops involved).
use fuel_transformers::models::rwkv_v5::Tokenizer;

use fuel::lazy_rwkv5::{Rwkv5Config, Rwkv5Model, Rwkv5Weights};
use fuel::lazy_rwkv6::{Rwkv6Model, Rwkv6Weights};
use fuel::lazy_rwkv7::{Rwkv7Config, Rwkv7Model, Rwkv7Weights};
use hf_hub::{api::sync::Api, Repo, RepoType};

const EOS_TOKEN_ID: u32 = 261;

enum Model {
    M5(Rwkv5Model),
    M6(Rwkv6Model),
}

impl Model {
    fn forward(&self, tokens: &[u32]) -> fuel::Result<fuel::lazy::LazyTensor> {
        match self {
            Self::M5(m) => m.forward(tokens),
            Self::M6(m) => m.forward(tokens),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::M5(m) => m.config.vocab_size,
            Self::M6(m) => m.config.vocab_size,
        }
    }
}

struct TextGeneration {
    model: Model,
    tokenizer: Tokenizer,
    seed: u64,
    temperature: f32,
    top_p: Option<f32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            seed,
            temperature: temp.unwrap_or(0.0) as f32,
            top_p: top_p.map(|p| p as f32),
            repeat_penalty,
            repeat_last_n,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        let mut tokens = self.tokenizer.encode(prompt)?;
        let vocab_size = self.model.vocab_size();
        print!("{}", self.tokenizer.decode(&tokens)?);
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            // Prefill-only: re-run the whole sequence each step. The lazy
            // RWKV modules unroll the time loop at graph-build time and do
            // not yet expose a resume-from-state API; mirror the mamba
            // binary's per-step rerun.
            let logits = self
                .model
                .forward(&tokens)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            let logits_data = logits.realize_f32();
            let seq = tokens.len();
            let last_off = (seq - 1) * vocab_size;
            let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
            if self.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                apply_repeat_penalty(&mut last_logits, self.repeat_penalty, &tokens[start_at..]);
            }
            let next_token = sample(
                &last_logits,
                self.temperature,
                None,
                self.top_p,
                self.seed.wrapping_add(index as u64),
            );
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == EOS_TOKEN_ID || next_token == 0 {
                break;
            }
            print!("{}", self.tokenizer.decode(&[next_token])?);
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

/// Text generation pipeline for RWKV v7 models.
/// Separate from v5/v6 because v7's eager binary previously had a
/// different State, but in the lazy port both share the prefill-only
/// "re-run the whole sequence" pattern. The v7 flow keeps the
/// presence/frequency penalty + stop-sequence logic from the eager
/// binary since it's binary-side bookkeeping (no Tensor ops).
struct TextGenerationV7 {
    model: Rwkv7Model,
    tokenizer: Tokenizer,
    seed: u64,
    temperature: f32,
    top_p: Option<f32>,
    alpha_presence: f32,
    alpha_frequency: f32,
    alpha_decay: f32,
    stop: Option<String>,
}

impl TextGenerationV7 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Rwkv7Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        alpha_presence: f32,
        alpha_frequency: f32,
        alpha_decay: f32,
        stop: Option<String>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            seed,
            temperature: temp.unwrap_or(1.0) as f32,
            top_p: top_p.map(|p| p as f32),
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            stop,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        // Strip trailing whitespace — RWKV tokenizer produces non-English output otherwise
        let prompt = prompt.trim_end();
        let mut tokens = self.tokenizer.encode(prompt)?;
        let vocab_size = self.model.config.vocab_size;

        // RWKV penalty state: per-token occurrence counts with exponential decay
        let penalties_enabled = self.alpha_presence != 0.0 || self.alpha_frequency != 0.0;
        let mut occurrence: Vec<f32> = vec![0.0; vocab_size];

        // Update penalty counts for prompt tokens up front (matches eager order).
        if penalties_enabled {
            for &t in tokens.iter() {
                for count in occurrence.iter_mut() {
                    *count *= self.alpha_decay;
                }
                if (t as usize) < vocab_size {
                    occurrence[t as usize] += 1.0;
                }
            }
        }

        // Print the prompt
        print!("{}", self.tokenizer.decode(&tokens)?);
        std::io::stdout().flush()?;

        // Track generated text for stop sequence detection
        let mut generated_text = String::new();
        let mut printed_len = 0; // How many chars we've already printed
        let mut generated_tokens = 0usize;

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            // Prefill-only: re-run the whole sequence each step. The lazy
            // RWKV-v7 module unrolls the time loop at graph-build time and
            // does not yet expose a resume-from-state API.
            let logits = self
                .model
                .forward(&tokens)
                .map_err(|e| E::msg(format!("forward: {e}")))?;
            let logits_data = logits.realize_f32();
            let seq = tokens.len();
            let last_off = (seq - 1) * vocab_size;
            let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();

            // Apply RWKV presence + frequency penalty
            if penalties_enabled {
                for (i, logit) in last_logits.iter_mut().enumerate() {
                    if occurrence[i] > 0.0 {
                        *logit -= self.alpha_presence + self.alpha_frequency * occurrence[i];
                    }
                }
            }

            let next_token = sample(
                &last_logits,
                self.temperature,
                None,
                self.top_p,
                self.seed.wrapping_add(index as u64),
            );
            tokens.push(next_token);
            generated_tokens += 1;

            if penalties_enabled {
                for count in occurrence.iter_mut() {
                    *count *= self.alpha_decay;
                }
                if (next_token as usize) < vocab_size {
                    occurrence[next_token as usize] += 1.0;
                }
            }

            if next_token == EOS_TOKEN_ID || next_token == 0 {
                break;
            }

            let token_text = self.tokenizer.decode(&[next_token])?;
            generated_text.push_str(&token_text);

            // Check for stop sequence
            if let Some(stop) = &self.stop {
                if let Some(pos) = generated_text.find(stop.as_str()) {
                    // Print only up to the stop sequence
                    if pos > printed_len {
                        print!("{}", &generated_text[printed_len..pos]);
                        std::io::stdout().flush()?;
                    }
                    break;
                }
                // Only print text that can't be the start of stop sequence
                // Keep the last (stop.chars().count() - 1) chars buffered
                // Use char boundaries to avoid splitting multi-byte UTF-8 characters
                let stop_char_count = stop.chars().count();
                let total_chars = generated_text.chars().count();
                let safe_char_count = total_chars.saturating_sub(stop_char_count - 1);
                // Convert char count back to byte offset at a valid boundary
                let safe_len = generated_text
                    .char_indices()
                    .nth(safe_char_count)
                    .map(|(i, _)| i)
                    .unwrap_or(generated_text.len());
                if safe_len > printed_len {
                    print!("{}", &generated_text[printed_len..safe_len]);
                    std::io::stdout().flush()?;
                    printed_len = safe_len;
                }
            } else {
                print!("{}", token_text);
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    // RWKV v5 models
    Eagle7b,
    World1b5,
    World3b,
    // RWKV v6 models
    World6_1b6,
    // RWKV v7 models: rwkv7-g1d (original v7 architecture, generation 1 dataset d)
    #[value(name = "rwkv7-g1d-0.1b")]
    Rwkv7G1d0_1b,
    #[value(name = "rwkv7-g1d-0.4b")]
    Rwkv7G1d0_4b,
    #[value(name = "rwkv7-g1d-1.5b")]
    Rwkv7G1d1_5b,
    #[value(name = "rwkv7-g1d-2.9b")]
    Rwkv7G1d2_9b,
    #[value(name = "rwkv7-g1d-7.2b")]
    Rwkv7G1d7_2b,
    #[value(name = "rwkv7-g1d-13.3b")]
    Rwkv7G1d13_3b,
    // RWKV v7a models: rwkv7a-g1d (v7a variant, generation 1 dataset d)
    #[value(name = "rwkv7a-g1d-0.1b")]
    Rwkv7aG1d0_1b,
    // RWKV v7b models: rwkv7b-g1b (v7b variant, generation 1 dataset b)
    #[value(name = "rwkv7b-g1b-0.1b")]
    Rwkv7bG1b0_1b,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Which {
    fn is_v7(&self) -> bool {
        matches!(
            self,
            Self::Rwkv7G1d0_1b
                | Self::Rwkv7G1d0_4b
                | Self::Rwkv7G1d1_5b
                | Self::Rwkv7G1d2_9b
                | Self::Rwkv7G1d7_2b
                | Self::Rwkv7G1d13_3b
                | Self::Rwkv7aG1d0_1b
                | Self::Rwkv7bG1b0_1b
        )
    }

    fn model_id(&self) -> &'static str {
        match self {
            Self::Eagle7b => "RWKV/v5-Eagle-7B-HF",
            Self::World1b5 => "RWKV/rwkv-5-world-1b5",
            Self::World3b => "RWKV/rwkv-5-world-3b",
            Self::World6_1b6 => "paperfun/rwkv",
            Self::Rwkv7G1d0_1b
            | Self::Rwkv7G1d0_4b
            | Self::Rwkv7G1d1_5b
            | Self::Rwkv7G1d2_9b
            | Self::Rwkv7G1d7_2b
            | Self::Rwkv7G1d13_3b
            | Self::Rwkv7aG1d0_1b
            | Self::Rwkv7bG1b0_1b => "DanielClough/rwkv7-g1-safetensors",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::Eagle7b => "refs/pr/1",
            Self::World1b5 | Self::World3b => "refs/pr/2",
            Self::World6_1b6 => "main",
            Self::Rwkv7G1d0_1b
            | Self::Rwkv7G1d0_4b
            | Self::Rwkv7G1d1_5b
            | Self::Rwkv7G1d2_9b
            | Self::Rwkv7G1d7_2b
            | Self::Rwkv7G1d13_3b
            | Self::Rwkv7aG1d0_1b
            | Self::Rwkv7bG1b0_1b => "main",
        }
    }

    /// Built-in v7 config skeleton (`hidden_size`, `num_hidden_layers`).
    /// LoRA dims are filled in from the safetensors at load time.
    fn v7_skeleton(&self) -> Option<(usize, usize)> {
        match self {
            Self::Rwkv7G1d0_1b | Self::Rwkv7aG1d0_1b | Self::Rwkv7bG1b0_1b => Some((768, 12)),
            Self::Rwkv7G1d0_4b => Some((1024, 24)),
            Self::Rwkv7G1d1_5b => Some((2048, 24)),
            Self::Rwkv7G1d2_9b => Some((2560, 32)),
            Self::Rwkv7G1d7_2b => Some((4096, 32)),
            Self::Rwkv7G1d13_3b => Some((4096, 61)),
            _ => None,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Preset {
    /// Chat: temp 1.0, top_p 0.5, presence 2.0, frequency 0.1, decay 0.99
    Chat,
    /// Creative (fiction etc.): temp 0.6, top_p 0.7, presence 2.0, frequency 0.2, decay 0.99
    Creative,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum PromptTemplate {
    /// Pass prompt as-is with no formatting.
    Raw,
    /// Chat format: User: {prompt}\n\nA:
    Chat,
    /// Think format: User: {prompt}\n\nA: <think>
    Think,
    /// Fake think (recommended): User: {prompt}\n\nA: <think></think
    FakeThink,
    /// Fill-in-middle for G1c+ models (text, code, everything): ✿prefix✿✿suffix✿{suffix}✿middle✿{prompt}
    Fim,
}

/// Format the user prompt according to the selected template.
fn apply_template(
    template: PromptTemplate,
    prompt: &str,
    system: Option<&str>,
    suffix: Option<&str>,
) -> String {
    match template {
        PromptTemplate::Raw => prompt.to_string(),
        PromptTemplate::Chat => {
            // Replace \n\n in user prompt with \n (double newline is chat round separator)
            let prompt = prompt.replace("\n\n", "\n");
            let mut out = String::new();
            if let Some(sys) = system {
                out.push_str(&format!("System: {sys}\n\n"));
            }
            out.push_str(&format!("User: {prompt}\n\nA:"));
            out
        }
        PromptTemplate::Think => {
            let prompt = prompt.replace("\n\n", "\n");
            let mut out = String::new();
            if let Some(sys) = system {
                out.push_str(&format!("System: {sys}\n\n"));
            }
            out.push_str(&format!("User: {prompt}\n\nA: <think>"));
            out
        }
        PromptTemplate::FakeThink => {
            let prompt = prompt.replace("\n\n", "\n");
            let mut out = String::new();
            if let Some(sys) = system {
                out.push_str(&format!("System: {sys}\n\n"));
            }
            out.push_str(&format!("User: {prompt}\n\nA: <think></think"));
            out
        }
        PromptTemplate::Fim => {
            let suffix = suffix.unwrap_or("");
            // FIM prompt for G1c and newer models (works for text, code, and everything)
            // Recommended format: ✿prefix✿✿suffix✿<suffix>✿middle✿<prompt>
            // The model continues from <prompt> and generates until it reaches <suffix>
            format!("✿prefix✿✿suffix✿{suffix}✿middle✿{prompt}")
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// Prompt template to apply (v7 only).
    #[arg(long, default_value = "raw")]
    template: PromptTemplate,

    /// System prompt for chat/think templates.
    #[arg(long)]
    system: Option<String>,

    /// Suffix text for FIM (fill-in-middle) template.
    #[arg(long)]
    suffix: Option<String>,

    /// Sampling preset (v7 only). Overrides temperature, top_p, and penalty defaults.
    #[arg(long)]
    preset: Option<Preset>,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = "0.5")]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    /// Stop generation when this text is produced (e.g., --stop "User:").
    #[arg(long)]
    stop: Option<String>,

    #[arg(long, default_value = "world1b5")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    /// Quantized GGUF weights are not yet wired into the lazy RWKV port —
    /// passing this flag will error out cleanly.
    #[arg(long)]
    quantized: bool,

    /// Data type for inference: kept for CLI-compat; lazy RWKV runs in F32.
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty (v5/v6 only).
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty (v5/v6 only).
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// RWKV presence penalty (v7 only). Flat additive penalty for any token that has appeared.
    #[arg(long, default_value_t = 2.0)]
    alpha_presence: f32,

    /// RWKV frequency penalty (v7 only). Additive penalty proportional to token count.
    #[arg(long, default_value_t = 0.1)]
    alpha_frequency: f32,

    /// RWKV penalty count decay (v7 only). Exponential decay applied to token counts each step.
    #[arg(long, default_value_t = 0.99)]
    alpha_decay: f32,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let mut args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Apply preset overrides (v7 only)
    if let Some(preset) = args.preset {
        match preset {
            Preset::Chat => {
                args.temperature = 1.0;
                args.top_p = Some(0.5);
                args.alpha_presence = 2.0;
                args.alpha_frequency = 0.1;
                args.alpha_decay = 0.99;
            }
            Preset::Creative => {
                args.temperature = 0.6;
                args.top_p = Some(0.7);
                args.alpha_presence = 2.0;
                args.alpha_frequency = 0.2;
                args.alpha_decay = 0.99;
            }
        }
    }

    // CLI compat: --cpu and --dtype have no effect on the lazy port (executor
    // chooses device + dtype is fixed at F32 in the lazy v1 modules).
    let _ = args.cpu;
    let _ = args.dtype;

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id
            .clone()
            .unwrap_or_else(|| args.which.model_id().to_string()),
        RepoType::Model,
        args.revision
            .clone()
            .unwrap_or_else(|| args.which.revision().to_string()),
    ));
    let tokenizer_path = match &args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("lmz/fuel-rwkv".to_string())
            .get("rwkv_vocab_v20230424.json")?,
    };
    let config_filename = match (&args.config_file, args.which.is_v7()) {
        (Some(file), _) => Some(std::path::PathBuf::from(file)),
        (None, true) => None, // v7 models use built-in config, no config.json needed
        (None, false) => Some(repo.get("config.json")?),
    };
    let filenames = match &args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                anyhow::bail!(
                    "quantized RWKV is not yet supported in the lazy port (no \
                     lazy_quantized_rwkv* module exists); rerun without --quantized."
                );
            }
            vec![match args.which {
                Which::World1b5 | Which::World3b | Which::Eagle7b => repo.get("model.safetensors")?,
                Which::World6_1b6 => repo.get("rwkv-6-world-1b6.safetensors")?,
                Which::Rwkv7G1d0_1b => {
                    repo.get("rwkv7-g1d-0.1b-20260129-ctx8192.safetensors")?
                }
                Which::Rwkv7G1d0_4b => {
                    repo.get("rwkv7-g1d-0.4b-20260210-ctx8192.safetensors")?
                }
                Which::Rwkv7G1d1_5b => {
                    repo.get("rwkv7-g1d-1.5b-20260212-ctx8192.safetensors")?
                }
                Which::Rwkv7G1d2_9b => {
                    repo.get("rwkv7-g1d-2.9b-20260131-ctx8192.safetensors")?
                }
                Which::Rwkv7G1d7_2b => {
                    repo.get("rwkv7-g1d-7.2b-20260131-ctx8192.safetensors")?
                }
                Which::Rwkv7G1d13_3b => {
                    repo.get("rwkv7-g1d-13.3b-20260131-ctx8192.safetensors")?
                }
                Which::Rwkv7aG1d0_1b => {
                    repo.get("rwkv7a-g1d-0.1b-20260212-ctx8192.safetensors")?
                }
                Which::Rwkv7bG1b0_1b => {
                    repo.get("rwkv7b-g1b-0.1b-20250822-ctx4096.safetensors")?
                }
            }]
        }
    };
    let tokenizer = Tokenizer::new(tokenizer_path)?;

    if args.which.is_v7() {
        // RWKV v7 path
        let (hidden_size, num_hidden_layers) = match &config_filename {
            Some(config_file) => {
                let v: serde_json::Value = serde_json::from_slice(&std::fs::read(config_file)?)?;
                let h = v
                    .get("hidden_size")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .ok_or_else(|| E::msg("config.json: missing hidden_size"))?;
                let l = v
                    .get("num_hidden_layers")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .ok_or_else(|| E::msg("config.json: missing num_hidden_layers"))?;
                (h, l)
            }
            None => args
                .which
                .v7_skeleton()
                .ok_or_else(|| E::msg("v7 variant must have built-in config"))?,
        };

        // Mmap safetensors and peek at LoRA dimensions on layer 0 — the upstream
        // RWKV-v7 family uses different LoRA widths per size, so it's safer to
        // infer them than to hardcode.
        let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
            .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
        let infer_lora = |name: &str| -> Result<usize> {
            let view = st
                .get(name)
                .map_err(|e| E::msg(format!("look up {name}: {e}")))?;
            let shape = view.shape();
            shape
                .get(1)
                .copied()
                .ok_or_else(|| E::msg(format!("{name}: expected 2-D shape, got {shape:?}")))
        };
        let d_decay = infer_lora("rwkv.blocks.0.attention.w1")?;
        let d_aaa = infer_lora("rwkv.blocks.0.attention.a1")?;
        let d_gate = infer_lora("rwkv.blocks.0.attention.g1")?;
        // v1/v2 only exist from layer 1 onwards; peek at layer 1.
        let d_mv = infer_lora("rwkv.blocks.1.attention.v1")?;

        let config = Rwkv7Config {
            vocab_size: 65536,
            hidden_size,
            num_hidden_layers,
            head_size: 64,
            intermediate_size: None,
            d_decay,
            d_aaa,
            d_mv,
            d_gate,
            layer_norm_epsilon: 1e-5,
        };

        let weights = Rwkv7Weights::load_from_mmapped(&st, &config)
            .map_err(|e| E::msg(format!("load weights: {e}")))?;
        let model = Rwkv7Model { config: config.clone(), weights };

        // For FIM template, auto-set stop sequence to ✿ (delimiter signals end of middle)
        let stop = match (&args.stop, args.template) {
            (Some(s), _) => Some(s.clone()), // User-specified stop takes precedence
            (None, PromptTemplate::Fim) => Some("✿".to_string()), // FIM auto-stops on delimiter
            (None, _) => None,
        };

        let mut pipeline = TextGenerationV7::new(
            model,
            tokenizer,
            args.seed,
            Some(args.temperature),
            args.top_p,
            args.alpha_presence,
            args.alpha_frequency,
            args.alpha_decay,
            stop,
        );
        let prompt = apply_template(
            args.template,
            &args.prompt,
            args.system.as_deref(),
            args.suffix.as_deref(),
        );
        pipeline.run(&prompt, args.sample_len)?;
    } else {
        // v5/v6 path
        let config_path = config_filename
            .ok_or_else(|| E::msg("config.json is required for v5/v6 RWKV"))?;
        let cfg_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path)?)?;
        let get_usize = |k: &str| -> Result<usize> {
            cfg_json
                .get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| E::msg(format!("config.json: missing {k}")))
        };
        let get_f64 = |k: &str| -> Option<f64> { cfg_json.get(k).and_then(|x| x.as_f64()) };
        let intermediate_size = cfg_json
            .get("intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);

        let rwkv_cfg = Rwkv5Config {
            vocab_size: get_usize("vocab_size")?,
            hidden_size: get_usize("hidden_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            attention_hidden_size: get_usize("attention_hidden_size")?,
            head_size: get_usize("head_size")?,
            num_attention_heads: cfg_json
                .get("num_attention_heads")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(0),
            intermediate_size,
            layer_norm_epsilon: get_f64("layer_norm_epsilon").unwrap_or(1e-5),
            rescale_every: cfg_json
                .get("rescale_every")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(0),
        };

        let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
            .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;

        let model = match args.which {
            Which::World1b5 | Which::World3b | Which::Eagle7b => {
                let weights = Rwkv5Weights::load_from_mmapped(&st, &rwkv_cfg)
                    .map_err(|e| E::msg(format!("load v5 weights: {e}")))?;
                Model::M5(Rwkv5Model {
                    config: rwkv_cfg.clone(),
                    weights,
                })
            }
            Which::World6_1b6 => {
                let weights = Rwkv6Weights::load_from_mmapped(&st, &rwkv_cfg)
                    .map_err(|e| E::msg(format!("load v6 weights: {e}")))?;
                Model::M6(Rwkv6Model {
                    config: rwkv_cfg.clone(),
                    weights,
                })
            }
            _ => unreachable!(),
        };

        let mut pipeline = TextGeneration::new(
            model,
            tokenizer,
            args.seed,
            Some(args.temperature),
            args.top_p,
            args.repeat_penalty,
            args.repeat_last_n,
        );
        pipeline.run(&args.prompt, args.sample_len)?;
    }

    Ok(())
}

// ───── Local sampling helpers (lifted from yi/helium migration) ─────

fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let mut seen = std::collections::HashSet::new();
    for &t in context {
        if !seen.insert(t) {
            continue;
        }
        let idx = t as usize;
        if idx < logits.len() {
            let v = logits[idx];
            logits[idx] = if v >= 0.0 { v / penalty } else { v * penalty };
        }
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
    let mut probs: Vec<f32> = logits.iter().map(|&x| ((x - max_l) * inv_t).exp()).collect();
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
