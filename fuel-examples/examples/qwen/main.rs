#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use fuel::lazy::{LlamaTokenizer, SamplingStrategy};
use fuel::lazy_llama2c::Llama2cModel;
use fuel::{DType, Device};
use std::io::Write;

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "0.5b")]
    W0_5b,
    #[value(name = "1.8b")]
    W1_8b,
    #[value(name = "4b")]
    W4b,
    #[value(name = "7b")]
    W7b,
    #[value(name = "14b")]
    W14b,
    #[value(name = "72b")]
    W72b,
    #[value(name = "moe-a2.7b")]
    MoeA27b,
    #[value(name = "2-0.5b")]
    W2_0_5b,
    #[value(name = "2-1.5b")]
    W2_1_5b,
    #[value(name = "2-7b")]
    W2_7b,
    #[value(name = "2-72b")]
    W2_72b,
    #[value(name = "3-0.6b")]
    W3_0_6b,
    #[value(name = "3-1.7b")]
    W3_1_7b,
    #[value(name = "3-4b")]
    W3_4b,
    #[value(name = "3-8b")]
    W3_8b,
    #[value(name = "3-moe-a3b")]
    W3MoeA3b,
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
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_path: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "0.5b")]
    model: WhichModel,

    /// Skip chat template formatting (use raw prompt, like base model)
    #[arg(long)]
    no_chat_template: bool,

    /// Enable thinking/reasoning mode (allows model to show its reasoning process)
    #[arg(long)]
    thinking: bool,
}

impl Args {
    fn should_use_chat_template(&self) -> bool {
        matches!(
            self.model,
            WhichModel::W3_0_6b
                | WhichModel::W3_1_7b
                | WhichModel::W3_4b
                | WhichModel::W3_8b
                | WhichModel::W3MoeA3b
        ) && !self.no_chat_template
    }
}

fn format_prompt(prompt: &str, use_chat_template: bool, thinking: bool) -> String {
    if !use_chat_template {
        return prompt.to_string();
    }
    let think_tag = if thinking { " /think" } else { " /no_think" };
    format!("<|im_start|>user\n{prompt}{think_tag}<|im_end|>\n<|im_start|>assistant\n")
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
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let use_chat_template = args.should_use_chat_template();
    let thinking = args.thinking;
    let model_id = match args.model_id.clone() {
        Some(model_id) => model_id,
        None => {
            let (version, size) = match args.model {
                WhichModel::W2_0_5b => ("2", "0.5B"),
                WhichModel::W2_1_5b => ("2", "1.5B"),
                WhichModel::W2_7b => ("2", "7B"),
                WhichModel::W2_72b => ("2", "72B"),
                WhichModel::W0_5b => ("1.5", "0.5B"),
                WhichModel::W1_8b => ("1.5", "1.8B"),
                WhichModel::W4b => ("1.5", "4B"),
                WhichModel::W7b => ("1.5", "7B"),
                WhichModel::W14b => ("1.5", "14B"),
                WhichModel::W72b => ("1.5", "72B"),
                WhichModel::MoeA27b => ("1.5", "MoE-A2.7B"),
                WhichModel::W3_0_6b => ("3", "0.6B"),
                WhichModel::W3_1_7b => ("3", "1.7B"),
                WhichModel::W3_4b => ("3", "4B"),
                WhichModel::W3_8b => ("3", "8B"),
                WhichModel::W3MoeA3b => ("3", "30B-A3B"),
            };
            format!("Qwen/Qwen{version}-{size}")
        }
    };

    let start = std::time::Instant::now();
    println!("loading model from {model_id}");
    let model = Llama2cModel::from_hub(&model_id)?;
    println!("loaded the model in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let tokenizer = LlamaTokenizer::from_hub(&model_id).map_err(E::msg)?;
    println!("loaded the tokenizer in {:?}", start.elapsed());

    let prompt = format_prompt(&args.prompt, use_chat_template, thinking);
    let prompt_tokens = tokenizer.encode(&prompt, true).map_err(E::msg)?;
    let mut streamed: Vec<u32> = prompt_tokens.clone();
    let mut printed_text = tokenizer.decode(&streamed, true).map_err(E::msg)?;
    print!("{}", printed_text);
    std::io::stdout().flush()?;

    let strategy = match args.temperature {
        Some(t) if t > 0.0 => SamplingStrategy::Temperature {
            temp: t as f32,
            seed: args.seed,
        },
        _ => SamplingStrategy::Greedy,
    };

    let device = if args.cpu {
        Device::cpu()
    } else {
        Device::cpu()
    };

    let start_gen = std::time::Instant::now();
    let output_tokens = model.generate_streaming_with_kv_context(
        &prompt_tokens,
        args.sample_len,
        strategy,
        tokenizer.eos_id(),
        &device,
        DType::F32,
        |tok| {
            streamed.push(tok);
            if let Ok(full) = tokenizer.decode(&streamed, true) {
                if let Some(delta) = full.strip_prefix(&printed_text) {
                    print!("{delta}");
                    std::io::stdout().flush().ok();
                }
                printed_text = full;
            }
        },
    )?;
    let dt = start_gen.elapsed();
    let generated_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());
    println!();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
