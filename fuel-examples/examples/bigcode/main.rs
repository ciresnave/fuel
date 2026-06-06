#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;

use fuel::lazy_bigcode::{BigCodeConfig, BigCodeModel, BigCodeWeights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

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
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    #[arg(long, default_value = "bigcode/starcoderbase-1b")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    weight_file: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = fuel_examples::device(args.cpu)?;

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let filenames = match args.weight_file {
        Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
        None => ["model.safetensors"]
            .iter()
            .map(|f| repo.get(f))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = BigCodeConfig::starcoder_1b();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = BigCodeWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = BigCodeModel { config: config.clone(), weights };
    println!("loaded the model in {:?}", start.elapsed());

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let mut tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits = model
            .forward(&tokens, 0)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let vocab_size = config.vocab_size;
        let seq = tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
        let next_token = sample(
            &last_logits,
            args.temperature.map(|t| t as f32).unwrap_or(0.0),
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        let token = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{token}");
        std::io::stdout().flush()?;
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.3} token/s)",
        args.sample_len,
        args.sample_len as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

fn sample(logits: &[f32], temperature: f32, top_p: Option<f32>, seed: u64) -> u32 {
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
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
