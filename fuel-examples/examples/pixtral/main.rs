#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_mistral::MistralConfig;
use fuel::lazy_pixtral::{
    PixtralActivation, PixtralConfig, PixtralModel, PixtralProjectorConfig, PixtralVisionConfig,
    PixtralWeights,
};
use fuel::{Device, Shape};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value = "Describe the image.\n")]
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
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    image: String,

    #[arg(long)]
    vision_only: bool,
}

fn parse_activation(s: Option<&str>) -> PixtralActivation {
    match s {
        Some("gelu") | Some("gelu_new") => PixtralActivation::Gelu,
        Some("gelu_pytorch_tanh") => PixtralActivation::GeluPytorchTanh,
        _ => PixtralActivation::Silu,
    }
}

fn pixtral_config_from_hf_json_str(json: &str) -> Result<PixtralConfig> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| E::msg(format!("parsing config.json: {e}")))?;
    let vc = v
        .get("vision_config")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let tc = v
        .get("text_config")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let get_usize = |o: &serde_json::Value, key: &str, default: usize| -> usize {
        o.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    };
    let get_usize_opt = |o: &serde_json::Value, key: &str| -> Option<usize> {
        o.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let get_f64 = |o: &serde_json::Value, key: &str, default: f64| -> f64 {
        o.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
    };
    let get_str = |o: &serde_json::Value, key: &str| -> Option<String> {
        o.get(key).and_then(|x| x.as_str()).map(|x| x.to_string())
    };

    let vision = PixtralVisionConfig {
        hidden_size: get_usize(&vc, "hidden_size", 1024),
        num_channels: get_usize(&vc, "num_channels", 3),
        image_size: get_usize(&vc, "image_size", 1024),
        patch_size: get_usize(&vc, "patch_size", 16),
        rope_theta: get_f64(&vc, "rope_theta", 10_000.0),
        intermediate_size: get_usize(&vc, "intermediate_size", 4096),
        num_hidden_layers: get_usize(&vc, "num_hidden_layers", 24),
        num_attention_heads: get_usize(&vc, "num_attention_heads", 16),
        head_dim: get_usize_opt(&vc, "head_dim"),
        activation: parse_activation(get_str(&vc, "hidden_act").as_deref()),
        rms_norm_eps: get_f64(&vc, "rms_norm_eps", 1e-5),
    };

    let text = MistralConfig {
        vocab_size: get_usize(&tc, "vocab_size", 131072),
        hidden_size: get_usize(&tc, "hidden_size", 5120),
        intermediate_size: get_usize(&tc, "intermediate_size", 14336),
        num_hidden_layers: get_usize(&tc, "num_hidden_layers", 40),
        num_attention_heads: get_usize(&tc, "num_attention_heads", 32),
        num_key_value_heads: get_usize(&tc, "num_key_value_heads", 8),
        head_dim: get_usize(&tc, "head_dim", 128),
        rms_norm_eps: get_f64(&tc, "rms_norm_eps", 1e-5),
        rope_theta: get_f64(&tc, "rope_theta", 1_000_000.0),
        max_position_embeddings: get_usize(&tc, "max_position_embeddings", 1024_000),
        sliding_window: get_usize_opt(&tc, "sliding_window"),
    };

    let projector_act = parse_activation(
        v.get("projector_hidden_act")
            .and_then(|x| x.as_str()),
    );
    let projector = PixtralProjectorConfig {
        in_dim: vision.hidden_size,
        out_dim: text.hidden_size,
        activation: projector_act,
    };

    Ok(PixtralConfig {
        vision,
        projector,
        text,
    })
}

fn load_image<P: AsRef<std::path::Path>>(path: P, image_size: usize) -> Result<Vec<f32>> {
    let img = image::ImageReader::open(path)?.decode()?;
    let img = img.resize_to_fill(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let raw = img.into_raw();
    // CLIP-style normalization (matches the eager call).
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.261_302_6, 0.275_777_1];
    let mut chw = vec![0f32; 3 * image_size * image_size];
    for h in 0..image_size {
        for w in 0..image_size {
            for c in 0..3 {
                let v = raw[(h * image_size + w) * 3 + c] as f32 / 255.0;
                chw[c * image_size * image_size + h * image_size + w] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok(chw)
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
    let _ = fuel_examples::device(args.cpu)?;

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "mistral-community/pixtral-12b".to_string(),
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    let config_filename = match args.config_file {
        Some(config_file) => std::path::PathBuf::from(config_file),
        None => repo.get("config.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let config_json = std::fs::read_to_string(&config_filename)?;
    let config = pixtral_config_from_hf_json_str(&config_json)?;

    let img_size = config.vision.image_size;
    let pixel_chw = load_image(&args.image, img_size)?;
    println!("loaded image (1, 3, {img_size}, {img_size})");

    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;

    if args.vision_only {
        let weights = PixtralWeights::load_from_mmapped(&st, &config)
            .map_err(|e| E::msg(format!("load weights: {e}")))?;
        let _ = weights;
        println!("vision-only loaded; lazy_pixtral does not expose a standalone vision forward, skipping.");
        return Ok(());
    }

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let start = std::time::Instant::now();
    let weights = PixtralWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load weights: {e}")))?;
    let model = PixtralModel { config: config.clone(), weights };
    println!("loaded the model in {:?}", start.elapsed());

    let pixel_values = LazyTensor::from_f32(
        Arc::<[f32]>::from(pixel_chw),
        Shape::from_dims(&[1, 3, img_size, img_size]),
        &Device::cpu(),
    );

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let mut tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let eos_token_id = tokenizer.token_to_id("</s>");
    let np = config.vision.num_patches();

    let mut generated_tokens = 0usize;
    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits = model
            .forward(&pixel_values, &tokens)
            .map_err(|e| E::msg(format!("forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let vocab_size = config.text.vocab_size;
        let seq = np + tokens.len();
        let last_off = (seq - 1) * vocab_size;
        let mut last_logits: Vec<f32> = logits_data[last_off..last_off + vocab_size].to_vec();
        if args.repeat_penalty != 1.0 {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(&mut last_logits, args.repeat_penalty, &tokens[start_at..]);
        }
        let next_token = sample(
            &last_logits,
            args.temperature.map(|t| t as f32).unwrap_or(0.0),
            args.top_p.map(|p| p as f32),
            args.seed.wrapping_add(index as u64),
        );
        tokens.push(next_token);
        generated_tokens += 1;
        if Some(next_token) == eos_token_id {
            break;
        }
        let tok = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{tok}");
        std::io::stdout().flush()?;
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

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
