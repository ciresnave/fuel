#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use fuel_examples::token_output_stream::TokenOutputStream;

use fuel::lazy_phi::PhiModel;
use fuel::{DType, Device};
use fuel_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: PhiModel,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: PhiModel,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            // Lazy forward: returns LazyTensor of shape (1, seq, vocab).
            let logits_lazy = self.model.forward(ctxt, pos)?;
            // Realize as f32 and select the LAST position's logits.
            let logits_flat = logits_lazy.realize_f32();
            let vocab = self.model.config.vocab_size;
            let seq = ctxt.len();
            let last_offset = (seq - 1) * vocab;
            let logits_v: Vec<f32> = logits_flat[last_offset..last_offset + vocab].to_vec();
            let logits = fuel::Tensor::new(logits_v, &Device::cpu())?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                fuel_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest()? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "1")]
    V1,
    #[value(name = "1.5")]
    V1_5,
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
    #[value(name = "3-medium")]
    V3Medium,
    #[value(name = "4-mini")]
    V4Mini,
    #[value(name = "2-old")]
    V2Old,
    PuffinPhiV2,
    PhiHermes,
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

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    mmlu_dir: Option<String>,

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
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "2")]
    model: WhichModel,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The dtype to be used for running the model, e.g. f32, bf16, or f16.
    #[arg(long)]
    dtype: Option<String>,
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

    if args.quantized {
        anyhow::bail!(
            "lazy phi binary: quantized Phi (GGUF) path is not yet shipped via the lazy module — \
             use the eager `quantized-phi` binary or wait for a `lazy_quantized_phi` port."
        );
    }
    match args.model {
        WhichModel::V2 => {}
        other => {
            anyhow::bail!(
                "lazy phi binary: only `--model 2` (Phi-2) is supported. Got {other:?}. \
                 Other Phi variants (Phi-1, Phi-1.5, Phi-3, Phi-4, MixFormer, PuffinPhi, PhiHermes) \
                 are not yet shipped via the lazy module."
            )
        }
    }

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "microsoft/phi-2".to_string(),
    };
    let revision = args.revision.unwrap_or_else(|| "main".to_string());
    let repo = api.repo(Repo::with_revision(model_id.clone(), RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let _device = fuel_examples::device(args.cpu)?;
    let _ = args.weight_file; // weight_file override path not yet supported — only --model-id is.
    if let Some(dtype) = args.dtype.as_deref() {
        // The lazy port runs f32 only for now; honor an explicit f32 selection, error on others.
        match dtype {
            "f32" => {}
            other => anyhow::bail!("lazy phi binary: only --dtype f32 is supported (got {other:?})"),
        }
    }
    let _ = DType::F32;
    let model = PhiModel::from_hub(&model_id)?;
    println!("loaded the model in {:?}", start.elapsed());

    match (args.prompt, args.mmlu_dir) {
        (None, None) | (Some(_), Some(_)) => {
            anyhow::bail!("exactly one of --prompt and --mmlu-dir must be specified")
        }
        (Some(prompt), None) => {
            let mut pipeline = TextGeneration::new(
                model,
                tokenizer,
                args.seed,
                args.temperature,
                args.top_p,
                args.repeat_penalty,
                args.repeat_last_n,
                args.verbose_prompt,
            );
            pipeline.run(&prompt, args.sample_len)?;
        }
        (None, Some(mmlu_dir)) => mmlu(model, tokenizer, mmlu_dir)?,
    }
    Ok(())
}

fn mmlu<P: AsRef<std::path::Path>>(
    model: PhiModel,
    tokenizer: Tokenizer,
    mmlu_dir: P,
) -> anyhow::Result<()> {
    for dir_entry in mmlu_dir.as_ref().read_dir()?.flatten() {
        let dir_entry = dir_entry.path();
        let theme = match dir_entry.file_stem().and_then(|v| v.to_str()) {
            None => "".to_string(),
            Some(v) => match v.strip_suffix("_test") {
                None => v.replace('_', " "),
                Some(v) => v.replace('_', " "),
            },
        };
        if dir_entry.extension().as_ref().and_then(|v| v.to_str()) != Some("csv") {
            continue;
        }
        println!("reading {dir_entry:?}");
        let dir_entry = std::fs::File::open(dir_entry)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(dir_entry);
        let token_a = tokenizer.token_to_id("A").unwrap();
        let token_b = tokenizer.token_to_id("B").unwrap();
        let token_c = tokenizer.token_to_id("C").unwrap();
        let token_d = tokenizer.token_to_id("D").unwrap();
        for row in reader.records() {
            let row = match row {
                Err(_) => continue,
                Ok(row) => row,
            };
            if row.len() < 5 {
                continue;
            }
            let question = row.get(0).unwrap();
            let answer_a = row.get(1).unwrap();
            let answer_b = row.get(2).unwrap();
            let answer_c = row.get(3).unwrap();
            let answer_d = row.get(4).unwrap();
            let answer = row.get(5).unwrap();
            let prompt = format!(
                    "{} {theme}.\n{question}\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}\nAnswer:\n",
                    "The following are multiple choice questions (with answers) about"
                );
            let tokens = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
            let token_ids = tokens.get_ids().to_vec();
            // Lazy forward: returns LazyTensor of shape (1, seq, vocab).
            let logits_lazy = model.forward(&token_ids, 0)?;
            let logits_flat = logits_lazy.realize_f32();
            let vocab = model.config.vocab_size;
            let seq = token_ids.len();
            let last_offset = (seq - 1) * vocab;
            let pr_a = logits_flat[last_offset + token_a as usize];
            let pr_b = logits_flat[last_offset + token_b as usize];
            let pr_c = logits_flat[last_offset + token_c as usize];
            let pr_d = logits_flat[last_offset + token_d as usize];
            let model_answer = if pr_a > pr_b && pr_a > pr_c && pr_a > pr_d {
                "A"
            } else if pr_b > pr_c && pr_b > pr_d {
                "B"
            } else if pr_c > pr_d {
                "C"
            } else {
                "D"
            };

            println!("{prompt}\n -> {model_answer} vs {answer}");
        }
    }
    Ok(())
}
