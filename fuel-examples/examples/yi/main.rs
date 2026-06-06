#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use fuel::lazy::{LlamaConfig, LlamaWeights};
use fuel::lazy_yi::{YiConfig, YiModel, YiWeights};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "6b")]
    L6b,
    #[value(name = "34b")]
    L34b,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)] cpu: bool,
    #[arg(long)] tracing: bool,
    #[arg(long)] prompt: String,
    #[arg(long)] temperature: Option<f64>,
    #[arg(long)] top_p: Option<f64>,
    #[arg(long)] top_k: Option<usize>,
    #[arg(long, default_value_t = 299792458)] seed: u64,
    #[arg(long, short = 'n', default_value_t = 100)] sample_len: usize,
    #[arg(long, default_value = "01-ai/Yi-6B")] model_id: String,
    #[arg(long, default_value = "main")] revision: String,
    #[arg(long)] tokenizer_file: Option<String>,
    #[arg(long)] config_file: Option<String>,
    #[arg(long)] weight_files: Option<String>,
    #[arg(long, default_value_t = 1.1)] repeat_penalty: f32,
    #[arg(long, default_value_t = 64)] repeat_last_n: usize,
    #[arg(long, default_value = "6b")] which: Which,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = args.tracing;
    let _device = fuel_examples::device(args.cpu)?;
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(args.model_id.clone(), RepoType::Model, args.revision.clone()));
    let tokenizer_filename = match args.tokenizer_file.as_ref() { Some(f) => std::path::PathBuf::from(f), None => repo.get("tokenizer.json")? };
    let filenames: Vec<std::path::PathBuf> = match args.weight_files.as_ref() {
        Some(f) => f.split(',').map(std::path::PathBuf::from).collect(),
        None => fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config_file = match args.config_file.as_ref() { Some(f) => std::path::PathBuf::from(f), None => repo.get("config.json")? };
    let config_json = std::fs::read_to_string(&config_file)?;
    let yi_cfg: YiConfig = yi_config_from_hf_json_str(&config_json, args.which)?;
    let llama_cfg = LlamaConfig {
        vocab_size: yi_cfg.vocab_size, dim: yi_cfg.hidden_size, n_layers: yi_cfg.num_hidden_layers,
        n_heads: yi_cfg.num_attention_heads, n_kv_heads: yi_cfg.num_key_value_heads,
        head_dim: yi_cfg.head_dim, ffn_dim: yi_cfg.intermediate_size,
        norm_eps: yi_cfg.rms_norm_eps, rope_base: yi_cfg.rope_theta,
    };
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }.map_err(|e| E::msg(format!("mmap: {e}")))?;
    let llama_weights: LlamaWeights = LlamaWeights::load_from_mmapped(&st, &llama_cfg).map_err(|e| E::msg(format!("weights: {e}")))?;
    let weights = YiWeights { token_embedding: llama_weights.token_embedding, layers: llama_weights.layers, final_norm_gain: llama_weights.final_norm_gain, output: llama_weights.output };
    let model = YiModel { config: yi_cfg.clone(), weights };
    let eos_token_id = parse_eos_token_id(&config_json).or_else(|| tokenizer.token_to_id("<|endoftext|>"));
    let mut tok_stream = fuel_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    print!("{}", args.prompt); std::io::stdout().flush()?;
    let mut tokens = tok_stream.tokenizer().encode(args.prompt.clone(), true).map_err(E::msg)?.get_ids().to_vec();
    let mut generated = 0usize; let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let logits = model.forward(&tokens, 0).map_err(|e| E::msg(format!("forward: {e}")))?;
        let data = logits.realize_f32(); let v = yi_cfg.vocab_size; let seq = tokens.len();
        let last_off = (seq - 1) * v;
        let mut last: Vec<f32> = data[last_off..last_off + v].to_vec();
        if args.repeat_penalty != 1.0 { let s = tokens.len().saturating_sub(args.repeat_last_n); apply_repeat_penalty(&mut last, args.repeat_penalty, &tokens[s..]); }
        let next = sample(&last, args.temperature.map(|t| t as f32).unwrap_or(0.0), args.top_k, args.top_p.map(|p| p as f32), args.seed.wrapping_add(index as u64));
        tokens.push(next); generated += 1;
        if Some(next) == eos_token_id { break; }
        if let Some(t) = tok_stream.next_token(next)? { print!("{}", t.replace("<|im_end|>", "\n")); std::io::stdout().flush()?; }
    }
    let dt = start_gen.elapsed();
    if let Some(r) = tok_stream.decode_rest().map_err(E::msg)? { print!("{r}"); }
    println!("\n{generated} tokens generated ({:.2} token/s)", generated as f64 / dt.as_secs_f64());
    Ok(())
}

fn yi_config_from_hf_json_str(json: &str, which: Which) -> Result<YiConfig> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| E::msg(format!("parse config: {e}")))?;
    let get_usize = |k: &str| -> Option<usize> { v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize) };
    let get_f64 = |k: &str| -> Option<f64> { v.get(k).and_then(|x| x.as_f64()) };
    let (dh, di, dl, dh2, dkv, dhead) = match which { Which::L6b => (4096, 11_008, 32, 32, 4, 128), Which::L34b => (7168, 20_480, 60, 56, 8, 128) };
    Ok(YiConfig {
        vocab_size: get_usize("vocab_size").unwrap_or(64_000),
        hidden_size: get_usize("hidden_size").unwrap_or(dh),
        intermediate_size: get_usize("intermediate_size").unwrap_or(di),
        num_hidden_layers: get_usize("num_hidden_layers").unwrap_or(dl),
        num_attention_heads: get_usize("num_attention_heads").unwrap_or(dh2),
        num_key_value_heads: get_usize("num_key_value_heads").unwrap_or(dkv),
        head_dim: get_usize("head_dim").unwrap_or(dhead),
        rms_norm_eps: get_f64("rms_norm_eps").unwrap_or(1e-5),
        rope_theta: get_f64("rope_theta").unwrap_or(5_000_000.0),
        max_position_embeddings: get_usize("max_position_embeddings").unwrap_or(4096),
    })
}

fn parse_eos_token_id(j: &str) -> Option<u32> { serde_json::from_str::<serde_json::Value>(j).ok()?.get("eos_token_id")?.as_u64().map(|x| x as u32) }
fn apply_repeat_penalty(logits: &mut [f32], p: f32, ctx: &[u32]) { let mut seen = std::collections::HashSet::new(); for &t in ctx { if !seen.insert(t) { continue; } let i = t as usize; if i < logits.len() { let v = logits[i]; logits[i] = if v >= 0.0 { v / p } else { v * p }; } } }
fn sample(logits: &[f32], temp: f32, top_k: Option<usize>, top_p: Option<f32>, seed: u64) -> u32 {
    if temp <= 0.0 { let mut bi = 0usize; let mut b = logits[0]; for (i, &v) in logits.iter().enumerate().skip(1) { if v > b { b = v; bi = i; } } return bi as u32; }
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max); let inv = 1.0 / temp.max(1e-6);
    let mut probs: Vec<f32> = logits.iter().map(|&x| ((x - m) * inv).exp()).collect();
    let s: f32 = probs.iter().sum(); for p in &mut probs { *p /= s.max(1e-30); }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut keep = vec![true; probs.len()];
    if let Some(k) = top_k { for &i in idx.iter().skip(k) { keep[i] = false; } }
    if let Some(pc) = top_p { let mut c = 0.0; let mut allow = true; for &i in &idx { if !keep[i] { continue; } if !allow { keep[i] = false; continue; } c += probs[i]; if c >= pc { allow = false; } } }
    let mut f: Vec<f32> = probs.iter().enumerate().map(|(i, p)| if keep[i] { *p } else { 0.0 }).collect();
    let ss: f32 = f.iter().sum(); if ss > 0.0 { for v in &mut f { *v /= ss; } } else { return 0; }
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    st ^= st >> 33; st = st.wrapping_mul(0xff51_afd7_ed55_8ccd); st ^= st >> 33;
    let r = (st as f32) / (u64::MAX as f32); let mut c = 0.0;
    for (i, p) in f.iter().enumerate() { c += *p; if r <= c { return i as u32; } }
    (f.len() - 1) as u32
}
