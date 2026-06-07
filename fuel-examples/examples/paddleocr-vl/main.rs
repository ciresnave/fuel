//! PaddleOCR-VL — lazy-graph port.
//!
//! v1 of this revived binary:
//!
//! * Single image, dynamic resolution via the NaViT vision tower
//!   ([`PaddleOcrVlNaVitModel`]). The image is preprocessed host-side
//!   with [`bilinear_resize_to_grid`] (CatmullRom resize + ImageNet
//!   normalize) onto the closest supported `(rows, cols) * factor`
//!   grid where `factor = patch_size * spatial_merge_size = 28`.
//! * Vision features are prepended to the text embeddings before the
//!   text decoder — mirroring the LLaVA / PaliGemma lazy binaries. No
//!   `<image>` placeholder splicing in v1.
//! * Greedy argmax decode; no KV cache (every step re-runs the whole
//!   graph). EOS detection via the tokenizer's `</s>` or
//!   `<|endoftext|>` id.
//! * Task selection via `--task ocr|table|formula|chart` simply picks
//!   a prompt template. Multi-image / batch / video remain
//!   unimplemented in v1.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use std::io::Write;
use std::sync::Arc;

use fuel::lazy::LazyTensor;
use fuel::lazy_paddleocr_vl::bilinear_resize_to_grid;
use fuel::lazy_paddleocr_vl_text::{
    load_paddleocr_vl_text_weights_with_prefix, PaddleOcrVlTextConfig, PaddleOcrVlTextModel,
};
use fuel::lazy_paddleocr_vl_vision::{
    PaddleOcrVlNaVitConfig, PaddleOcrVlNaVitModel, PaddleOcrVlNaVitWeights,
};
use fuel::safetensors::MmapedSafetensors;

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a local document image.
    #[arg(long)]
    image: String,

    /// Task type — selects the prompt template fed to the text decoder.
    #[arg(long, value_parser = ["ocr", "table", "formula", "chart"], default_value = "ocr")]
    task: String,

    /// Hugging Face model id.
    #[arg(long, default_value = "PaddlePaddle/PaddleOCR-VL")]
    model_id: String,

    /// HF revision (branch / commit / tag).
    #[arg(long, default_value = "main")]
    revision: String,

    /// Override `tokenizer.json` path.
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Override the safetensors weight files (comma-separated).
    #[arg(long)]
    weight_files: Option<String>,

    /// Run on CPU (parity flag; lazy realize routes through the
    /// default backend chooser).
    #[arg(long)]
    cpu: bool,

    /// Use bfloat16 (parity flag; v1 is CPU-F32 only — same as the
    /// retired binary).
    #[arg(long)]
    bf16: bool,

    /// Greedy decode length cap (tokens after the prompt).
    #[arg(long, default_value_t = 1024)]
    max_length: usize,

    /// Random seed (parity flag; v1 is greedy argmax — no sampling).
    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,
}

/// Prompt template per task. The published checkpoint expects a
/// short instruction prefix before the model emits the OCR / table /
/// formula / chart payload; the exact phrasings are mirrored from
/// the upstream Python inference scripts.
fn task_prompt(task: &str) -> &'static str {
    match task {
        "ocr" => "OCR:",
        "table" => "Table Recognition:",
        "formula" => "Formula Recognition:",
        "chart" => "Chart Recognition:",
        _ => "OCR:",
    }
}

/// Build a default set of `(rows, cols)` candidate grids in **pixel
/// units**, each a multiple of `factor = patch_size * spatial_merge_size`
/// = 28. Covers square + a few landscape and portrait shapes; the
/// preprocessor picks the closest aspect ratio.
fn default_supported_grids(factor: usize, max_pixels: usize) -> Vec<(usize, usize)> {
    // Anchor sizes (in factor units). 27 × 28 = 756 matches the
    // base 27×27 grid the position embedding is trained against.
    // 12–48 covers the typical document aspect-ratio range.
    let units: &[(usize, usize)] = &[
        (27, 27), // square (matches base grid)
        (20, 36), // 5:9 landscape
        (36, 20), // 9:5 portrait
        (16, 48), // ~1:3 landscape
        (48, 16), // ~3:1 portrait
        (24, 32), // 3:4 landscape
        (32, 24), // 4:3 portrait
        (18, 54), // 1:3 (wide)
        (54, 18), // 3:1 (tall)
    ];
    units
        .iter()
        .map(|&(rh, rw)| (rh * factor, rw * factor))
        .filter(|&(h, w)| h * w <= max_pixels)
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = args.cpu; // parity flag
    let _ = args.bf16; // parity flag
    let _ = args.seed; // parity flag (greedy decode)

    // ---- Resolve hub repo / files ------------------------------------------
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    let tokenizer_path = match args.tokenizer_file {
        Some(p) => std::path::PathBuf::from(p),
        None => repo.get("tokenizer.json")?,
    };
    let weight_files: Vec<std::path::PathBuf> = match &args.weight_files {
        Some(files) => files.split(',').map(std::path::PathBuf::from).collect(),
        None => match fuel_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
            Ok(files) => files,
            // Single-shard fall back: many small checkpoints (incl.
            // PaddleOCR-VL) ship a flat `model.safetensors` without
            // an index file.
            Err(_) => vec![repo.get("model.safetensors")?],
        },
    };

    // ---- Config presets ----------------------------------------------------
    // The lazy port hard-codes the published checkpoint shapes via
    // the `paddleocr_vl_default` / `paddleocr_vl` presets. Hugging-
    // Face config translation lives in `fuel-core` (see the session
    // prompt's Part 3 follow-up); v1 ships with the preset.
    let text_cfg = PaddleOcrVlTextConfig::paddleocr_vl_default();
    let vision_cfg = PaddleOcrVlNaVitConfig::paddleocr_vl();
    let factor = vision_cfg.patch_size * vision_cfg.spatial_merge_size;
    let supported_grids = default_supported_grids(factor, vision_cfg.max_pixels);
    if supported_grids.is_empty() {
        return Err(E::msg(
            "default supported_grids set is empty for the configured max_pixels",
        ));
    }
    println!(
        "config: vision base {}×{} (patch {}, merge {}), text dim {} / {} layers / vocab {}",
        vision_cfg.base_image_size,
        vision_cfg.base_image_size,
        vision_cfg.patch_size,
        vision_cfg.spatial_merge_size,
        text_cfg.hidden_size,
        text_cfg.num_hidden_layers,
        text_cfg.vocab_size,
    );
    println!(
        "{} candidate grids (factor = {factor}, max_pixels = {})",
        supported_grids.len(),
        vision_cfg.max_pixels,
    );

    // ---- Tokenizer ---------------------------------------------------------
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // ---- Image preprocessing via the bilinear NaViT resizer ----------------
    let img = image::ImageReader::open(&args.image)?
        .decode()
        .map_err(E::msg)?;
    println!(
        "loaded {} ({}×{})",
        args.image,
        img.width(),
        img.height()
    );
    let (pixel_values, h_grid, w_grid) =
        bilinear_resize_to_grid(&img, &supported_grids).map_err(|e| E::msg(format!("{e}")))?;
    println!(
        "preprocessed to {}×{} (patch grid {}×{})",
        h_grid,
        w_grid,
        h_grid / vision_cfg.patch_size,
        w_grid / vision_cfg.patch_size,
    );

    // ---- Weights -----------------------------------------------------------
    let load_start = std::time::Instant::now();
    let st = unsafe { MmapedSafetensors::multi(&weight_files) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let vision_weights =
        PaddleOcrVlNaVitWeights::load_from_mmapped(&st, &vision_cfg, text_cfg.hidden_size)
            .map_err(|e| E::msg(format!("load NaViT vision weights: {e}")))?;
    let text_weights = load_paddleocr_vl_text_weights_with_prefix(&st, &text_cfg, "")
        .map_err(|e| E::msg(format!("load text weights: {e}")))?;
    println!("loaded weights in {:?}", load_start.elapsed());

    let vision_model =
        PaddleOcrVlNaVitModel::new(vision_cfg.clone(), text_cfg.hidden_size, vision_weights);
    let text_model = PaddleOcrVlTextModel {
        config: text_cfg.clone(),
        weights: text_weights,
    };

    // ---- Vision tower: pixels → (merged_patches, text_hidden) --------------
    let vision_features = vision_model
        .forward(&pixel_values)
        .map_err(|e| E::msg(format!("NaViT forward: {e}")))?;
    let v_dims = vision_features.shape().dims().to_vec();
    if v_dims.len() != 2 || v_dims[1] != text_cfg.hidden_size {
        return Err(E::msg(format!(
            "NaViT output shape {v_dims:?} does not match (N, text_hidden={})",
            text_cfg.hidden_size,
        )));
    }
    let num_vision_tokens = v_dims[0];
    println!("vision features: {num_vision_tokens} tokens × {} hidden", v_dims[1]);

    // Reshape vision features to (1, N, hidden) so they can be
    // concatenated along the sequence axis with the text embeddings.
    let vision_embeds = vision_features
        .reshape(fuel::Shape::from_dims(&[
            1,
            num_vision_tokens,
            text_cfg.hidden_size,
        ]))
        .map_err(|e| E::msg(format!("reshape vision features: {e}")))?;

    // ---- Prompt → tokens ---------------------------------------------------
    let prompt = task_prompt(&args.task).to_string();
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut tokens: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let vocab_size = text_cfg.vocab_size;
    let eos_token_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"));

    // ---- Greedy decode -----------------------------------------------------
    let gen_start = std::time::Instant::now();
    let mut generated = 0_usize;
    for _ in 0..args.max_length {
        // Embed the current text tokens on the vision graph so they
        // share a graph with `vision_embeds` and concat succeeds.
        let text_embeds = vision_embeds
            .embed_tokens_anchored(
                text_model.weights.token_embedding.clone(),
                text_cfg.vocab_size,
                text_cfg.hidden_size,
                &tokens,
            )
            .map_err(|e| E::msg(format!("embed_tokens_anchored: {e}")))?;

        // Prepend vision features to the text embeddings along the
        // sequence axis. Final shape: (1, num_vision_tokens +
        // tokens.len(), hidden).
        let combined = vision_embeds
            .concat(&text_embeds, 1_usize)
            .map_err(|e| E::msg(format!("concat vision + text: {e}")))?;

        let logits = text_model
            .forward_embeds(&combined, 0)
            .map_err(|e| E::msg(format!("text decoder forward: {e}")))?;
        let data = logits.realize_f32();

        // Pick the last position (vision_len + text_len - 1).
        let seq = num_vision_tokens + tokens.len();
        let off = (seq - 1) * vocab_size;
        let last = &data[off..off + vocab_size];
        let mut best_i = 0_usize;
        let mut best = last[0];
        for (i, &v) in last.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        let next = best_i as u32;
        if Some(next) == eos_token_id {
            break;
        }
        tokens.push(next);
        generated += 1;
        let piece = tokenizer.decode(&[next], true).map_err(E::msg)?;
        print!("{piece}");
        std::io::stdout().flush()?;
    }
    let dt = gen_start.elapsed();
    println!(
        "\n{generated} tokens generated in {:?} ({:.2} tok/s)",
        dt,
        generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
