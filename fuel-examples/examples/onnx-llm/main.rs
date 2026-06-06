#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use fuel::lazy::LazyTensor;
use fuel::Device;
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::io::Write;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    SmolLM135M,
}

#[derive(Parser)]
struct Args {
    /// The prompt to be used.
    #[arg(long, default_value = "My favorite theorem is ")]
    prompt: String,

    /// The model to be used.
    #[arg(value_enum, long, default_value_t = Which::SmolLM135M)]
    which: Which,

    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    /// The number of tokens to generate.
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// The temperature used for sampling.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

// Simple LCG for reproducible sampling.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f64>,
    rng: &mut Lcg,
) -> usize {
    if temperature <= 0.0 {
        // Greedy.
        let mut best_i = 0;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        return best_i;
    }
    // Apply temperature.
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    // Softmax (numerically stable).
    let max = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = scaled.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut probs: Vec<(usize, f32)> = exp
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e / sum))
        .collect();
    // Top-k filter.
    if let Some(k) = top_k {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probs.truncate(k);
    }
    // Top-p (nucleus) filter.
    if let Some(p) = top_p {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut acc = 0.0_f64;
        let mut cut = probs.len();
        for (i, (_, pr)) in probs.iter().enumerate() {
            acc += *pr as f64;
            if acc >= p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
    }
    // Renormalize.
    let s: f32 = probs.iter().map(|(_, p)| *p).sum();
    let r = rng.next_f32() * s;
    let mut acc = 0.0_f32;
    for (i, pr) in &probs {
        acc += *pr;
        if acc >= r {
            return *i;
        }
    }
    probs.last().map(|(i, _)| *i).unwrap_or(0)
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let _device = if args.cpu { Device::cpu() } else { Device::cpu() };

    let (model_id, tokenizer_id) = match args.which {
        Which::SmolLM135M => ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-135M"),
    };

    let api = Api::new()?;
    let model_repo = api.model(model_id.to_string());
    let tokenizer_repo = api.model(tokenizer_id.to_string());

    let model_path = model_repo.get("onnx/model.onnx")?;
    let config_file = model_repo.get("config.json")?;
    let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;

    let tokenizer_path = tokenizer_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let tokens_u32 = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let tokens: Vec<i64> = tokens_u32.iter().map(|&t| t as i64).collect();

    println!("Loading ONNX model from {:?}", model_path);
    let evaluator = fuel_onnx::LazyOnnxEval::from_path(&model_path)?;

    let mut generated_tokens = tokens.clone();
    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let mut rng = Lcg::new(args.seed);

    // Past key/value state lives on the host between frames; rebuilt as
    // fresh LazyTensors each iteration so the graph stays small.
    let num_layers = config.num_hidden_layers;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.hidden_size / config.num_attention_heads;

    // (past_kv[layer].0 = key flat host data, key shape ; past_kv[layer].1 = value flat, shape).
    let mut past_kv: Option<Vec<((Vec<f32>, (usize, usize, usize, usize)), (Vec<f32>, (usize, usize, usize, usize)))>> =
        None;

    let device = Device::cpu();

    for _ in 0..args.max_tokens {
        // Build a fresh graph this iteration: anchor on the f32
        // sample-rate-like dummy — but we need a real F32 anchor. Use
        // the first KV tensor (always present) or a fresh zero f32
        // anchor.
        let anchor =
            LazyTensor::from_f32(vec![0.0_f32; 1], (1usize,), &device);

        let mut inputs: std::collections::HashMap<String, LazyTensor> =
            std::collections::HashMap::new();

        if let Some(past) = &past_kv {
            // Single-token continuation.
            let last_token = generated_tokens[generated_tokens.len() - 1];
            let input_ids = anchor.const_i64_like(vec![last_token], (1usize, 1usize));
            inputs.insert("input_ids".to_string(), input_ids);

            let seq_len = generated_tokens.len();
            let attn = anchor.const_i64_like(vec![1_i64; seq_len], (1usize, seq_len));
            inputs.insert("attention_mask".to_string(), attn);

            let pos =
                anchor.const_i64_like(vec![(seq_len - 1) as i64], (1usize, 1usize));
            inputs.insert("position_ids".to_string(), pos);

            for (i, (k_pair, v_pair)) in past.iter().enumerate() {
                let k_lazy = anchor.const_f32_like(
                    Arc::<[f32]>::from(k_pair.0.clone().into_boxed_slice()),
                    k_pair.1,
                );
                let v_lazy = anchor.const_f32_like(
                    Arc::<[f32]>::from(v_pair.0.clone().into_boxed_slice()),
                    v_pair.1,
                );
                inputs.insert(format!("past_key_values.{}.key", i), k_lazy);
                inputs.insert(format!("past_key_values.{}.value", i), v_lazy);
            }
        } else {
            // Prefill: feed full prompt.
            let seq_len = generated_tokens.len();
            let input_ids = anchor.const_i64_like(
                generated_tokens.clone(),
                (1usize, seq_len),
            );
            inputs.insert("input_ids".to_string(), input_ids);

            let attn = anchor.const_i64_like(vec![1_i64; seq_len], (1usize, seq_len));
            inputs.insert("attention_mask".to_string(), attn);

            let pos: Vec<i64> = (0..seq_len as i64).collect();
            let pos_t = anchor.const_i64_like(pos, (1usize, seq_len));
            inputs.insert("position_ids".to_string(), pos_t);

            // Empty key/value tensors (shape ..., 0, head_dim).
            for i in 0..num_layers {
                let empty_shape = (1usize, num_kv_heads, 0usize, head_dim);
                let empty: Vec<f32> = vec![];
                let k = anchor.const_f32_like(
                    Arc::<[f32]>::from(empty.clone().into_boxed_slice()),
                    empty_shape,
                );
                let v = anchor.const_f32_like(
                    Arc::<[f32]>::from(empty.into_boxed_slice()),
                    empty_shape,
                );
                inputs.insert(format!("past_key_values.{}.key", i), k);
                inputs.insert(format!("past_key_values.{}.value", i), v);
            }
        }

        let outputs = evaluator.run(&inputs)?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| anyhow::anyhow!("missing logits output"))?;
        let logits_dims = logits.shape();
        let dims = logits_dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let vocab = dims[2];
        assert_eq!(batch, 1);
        let logits_vec = logits.realize_f32();
        let last_row = &logits_vec[(seq - 1) * vocab..seq * vocab];

        let next_id = sample(
            last_row,
            args.temperature,
            args.top_k,
            args.top_p,
            &mut rng,
        );
        let next_token_id = next_id as u32;
        generated_tokens.push(next_token_id as i64);

        if let Some(token_str) = tokenizer.decode(&[next_token_id], true).ok() {
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }
        if let Some(eos_id) = tokenizer.token_to_id("<|endoftext|>") {
            if next_token_id == eos_id {
                break;
            }
        }

        // Stash present.*.key / value as host f32 vectors for next iter.
        let mut new_past = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let k = outputs
                .get(&format!("present.{}.key", i))
                .ok_or_else(|| anyhow::anyhow!("missing present.{}.key", i))?;
            let v = outputs
                .get(&format!("present.{}.value", i))
                .ok_or_else(|| anyhow::anyhow!("missing present.{}.value", i))?;
            let k_shape = k.shape();
            let k_dims = k_shape.dims();
            let k_tuple = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
            let v_shape = v.shape();
            let v_dims = v_shape.dims();
            let v_tuple = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
            new_past.push(((k.realize_f32(), k_tuple), (v.realize_f32(), v_tuple)));
        }
        past_kv = Some(new_past);
    }

    println!("\nGeneration complete!");
    Ok(())
}
