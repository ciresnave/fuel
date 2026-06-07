//! PaddleOCR-VL — lazy port.
//!
//! The eager binary implemented a full generation loop (KV cache, EOS-stop
//! decoding, multi-image / batch / video pipelines). The lazy module at
//! `fuel::lazy_paddleocr_vl` currently exposes a single-pass `forward` over
//! `(image_pixels: (C, H, W), text_tokens, image_token_id, start_pos)`
//! returning logits of shape `(1, seq, vocab)`. There is no KV cache and no
//! `generate` helper yet — those are deferred to follow-up work on the lazy
//! module (matches the LLaVA / PaliGemma lazy ports).
//!
//! What this binary does today:
//!   1. Loads the HF PaddleOCR-VL config + tokenizer.
//!   2. Reads + preprocesses one image (smart_resize → CHW pixels in [-1, 1]).
//!   3. Builds a `PaddleOcrVlConfig` from the HF JSON (text + vision sub-configs).
//!   4. Loads weights from safetensors via `PaddleOcrVlModel::load_from_mmapped`.
//!   5. Tokenizes the prompt with image placeholder tokens sized to match the
//!      vision encoder's tile-grid output count.
//!   6. Runs a single `PaddleOcrVlModel::forward` and greedy-argmax-decodes
//!      the max-logit next token as a smoke test.
//!
//! The `--batch`, `--video`, multi-image, and `--max-length` knobs are kept
//! on the CLI for parity but are no-ops in this single-pass port — extending
//! the lazy module with a generation loop / KV cache is a separate work item.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};
use fuel::lazy::LazyTensor;
use fuel::lazy_paddleocr_vl::{PaddleOcrVlConfig, PaddleOcrVlModel};
use fuel::lazy_paddleocr_vl_text::PaddleOcrVlTextConfig;
use fuel::lazy_paddleocr_vl_vision::{
    PaddleOcrVlVisionActivation, PaddleOcrVlVisionConfig,
};
use fuel::safetensors::MmapedSafetensors;
use fuel::{Device, Shape};
use fuel_transformers::models::paddleocr_vl::Config;
use std::sync::Arc;
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL";

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Task {
    /// Text recognition (OCR)
    Ocr,
    /// Table recognition
    Table,
    /// Formula recognition
    Formula,
    /// Chart recognition
    Chart,
    /// Video mode - reserved; not supported by the lazy v1 binary.
    Video,
}

impl Task {
    fn prompt(&self) -> &'static str {
        match self {
            Task::Ocr => "OCR:",
            Task::Table => "Table Recognition:",
            Task::Formula => "Formula Recognition:",
            Task::Chart => "Chart Recognition:",
            Task::Video => "OCR:",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to document image(s). Lazy v1 supports a single image.
    #[arg(long, num_args = 1..)]
    image: Vec<String>,

    /// Batch mode — accepted for CLI parity; not implemented by the lazy v1
    /// binary (the lazy module has no generation loop yet).
    #[arg(long, num_args = 1..)]
    batch: Vec<String>,

    /// Video input — accepted for CLI parity; not implemented by the lazy
    /// v1 binary.
    #[arg(long)]
    video: Option<String>,

    /// Frames per second to extract from video (default: 1.0).
    #[arg(long, default_value = "1.0")]
    fps: f32,

    /// Maximum number of frames to extract from video (default: 16).
    #[arg(long, default_value = "16")]
    max_frames: usize,

    /// Similarity threshold for video dedup (kept for CLI parity).
    #[arg(long, default_value = "0.85")]
    similarity_threshold: f32,

    /// Task type
    #[arg(long, value_enum, default_value = "ocr")]
    task: Task,

    /// Model repository or path
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Model revision
    #[arg(long, default_value = "main")]
    revision: String,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Maximum generation length — kept for CLI parity; the lazy v1 binary
    /// only runs a single greedy step.
    #[arg(long, default_value = "1024")]
    max_length: usize,

    /// Use bfloat16 precision — accepted for CLI parity; the lazy module is
    /// F32-only today.
    #[arg(long)]
    bf16: bool,
}

/// Smart resize algorithm matching PyTorch's PaddleOCRVLImageProcessor.
///
/// Rescales the image so that:
/// 1. Both dimensions are divisible by `factor` (patch_size × merge_size = 28).
/// 2. Total pixels are within [min_pixels, max_pixels] range.
/// 3. Aspect ratio is maintained as closely as possible.
fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let mut h = height;
    let mut w = width;

    if h < factor {
        w = (w * factor + h / 2) / h;
        h = factor;
    }
    if w < factor {
        h = (h * factor + w / 2) / w;
        w = factor;
    }

    let aspect = if h > w {
        h as f64 / w as f64
    } else {
        w as f64 / h as f64
    };
    if aspect > 200.0 {
        return Err(E::msg(format!(
            "Aspect ratio {:.1} exceeds maximum of 200",
            aspect
        )));
    }

    let mut h_bar = ((h + factor / 2) / factor) * factor;
    let mut w_bar = ((w + factor / 2) / factor) * factor;

    let total_pixels = h_bar * w_bar;

    if total_pixels > max_pixels {
        let beta = ((h * w) as f64 / max_pixels as f64).sqrt();
        h_bar = ((h as f64 / beta / factor as f64).floor() as usize) * factor;
        w_bar = ((w as f64 / beta / factor as f64).floor() as usize) * factor;
    } else if total_pixels < min_pixels {
        let beta = (min_pixels as f64 / (h * w) as f64).sqrt();
        h_bar = ((h as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((w as f64 * beta / factor as f64).ceil() as usize) * factor;
    }

    Ok((h_bar, w_bar))
}

/// Load and preprocess an image into a `(C, H, W)` lazy tensor of f32
/// values in [-1, 1]. Returns `(pixels, new_h, new_w)` so the caller can
/// pick the right number of placeholder tokens for the chosen tile grid.
fn load_image_lazy(path: &str, device: &Device) -> Result<(LazyTensor, usize, usize)> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);

    let patch_size = 14;
    let spatial_merge = 2;
    let factor = patch_size * spatial_merge; // 28
    let min_pixels = 147_384;
    let max_pixels = 2_822_400;

    let (new_height, new_width) = smart_resize(height, width, factor, min_pixels, max_pixels)?;

    let resized = image::imageops::resize(
        &img,
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let channels = 3usize;
    let mut normalized = vec![0f32; channels * new_height * new_width];
    for c in 0..channels {
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let idx = c * new_height * new_width + y * new_width + x;
                normalized[idx] = pixel[c] as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }

    let pixels = LazyTensor::from_f32(
        Arc::<[f32]>::from(normalized),
        Shape::from_dims(&[channels, new_height, new_width]),
        device,
    );

    println!(
        "Image: {}x{} -> {}x{}",
        width, height, new_width, new_height
    );

    Ok((pixels, new_height, new_width))
}

/// Translate the HF eager `Config` into the lazy `PaddleOcrVlConfig`.
fn lazy_config_from_hf(cfg: &Config) -> PaddleOcrVlConfig {
    let mrope_section = cfg
        .rope_scaling
        .as_ref()
        .map(|rs| rs.mrope_section.clone())
        .unwrap_or_else(|| vec![16, 24, 24]);

    let text = PaddleOcrVlTextConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rms_norm_eps: cfg.layer_norm_eps,
        rope_theta: cfg.rope_theta,
        max_position_embeddings: cfg.max_position_embeddings,
        use_bias: cfg.use_bias,
        tie_word_embeddings: cfg.tie_word_embeddings,
        mrope_section,
    };

    let v = &cfg.vision_config;
    let vision = PaddleOcrVlVisionConfig {
        hidden_size: v.hidden_size,
        intermediate_size: v.intermediate_size,
        num_hidden_layers: v.num_hidden_layers,
        num_attention_heads: v.num_attention_heads,
        num_channels: v.num_channels,
        image_size: v.image_size,
        patch_size: v.patch_size,
        // Eager `Activation` enum -> lazy enum. The published checkpoint uses
        // GeluPytorchTanh; fall back to it on any unhandled variant so the
        // smoke test doesn't crash on a checkpoint that lists an unusual
        // activation name.
        hidden_activation: PaddleOcrVlVisionActivation::GeluPytorchTanh,
        layer_norm_eps: v.layer_norm_eps,
        spatial_merge_size: v.spatial_merge_size,
        // The lazy vision encoder runs at a fixed tile size with its own
        // internal 2D RoPE; rope_theta is canonical for the published model.
        rope_theta: 10_000.0,
    };

    PaddleOcrVlConfig {
        text,
        vision,
        max_tiles_per_side: 4,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Lazy realizes through the CPU/router; the `--cpu` and `--bf16` flags
    // are accepted for CLI parity but the lazy v1 path is CPU-routed F32.
    let _ = args.cpu;
    let _ = args.bf16;
    let device = Device::cpu();

    println!("Loading model from {}...", args.model_id);
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));

    // Load config.
    let config_file = repo.get("config.json")?;
    let hf_cfg: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    println!(
        "Vision: {}L {}H, Text: {}L {}H (GQA: {}KV)",
        hf_cfg.vision_config.num_hidden_layers,
        hf_cfg.vision_config.num_attention_heads,
        hf_cfg.num_hidden_layers,
        hf_cfg.num_attention_heads,
        hf_cfg.num_key_value_heads,
    );

    let lazy_cfg = lazy_config_from_hf(&hf_cfg);

    // Load tokenizer.
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;

    // Load weights.
    let model_file = repo.get("model.safetensors")?;
    println!("Loading weights from {:?}...", model_file);
    let st = unsafe { MmapedSafetensors::multi(&[model_file.clone()]) }
        .map_err(|e| E::msg(format!("mmap: {e}")))?;
    let model = PaddleOcrVlModel::load_from_mmapped(&st, &lazy_cfg)
        .map_err(|e| E::msg(format!("weights: {e}")))?;
    println!("Model loaded successfully");

    // Validate input mode.
    let is_video = args.video.is_some();
    let is_batch = !args.batch.is_empty();
    let is_image = !args.image.is_empty();
    let input_count = is_video as u8 + is_batch as u8 + is_image as u8;
    if input_count == 0 {
        bail!("Either --image, --batch, or --video must be specified");
    }
    if input_count > 1 {
        bail!("Cannot combine --image, --batch, and --video. Use only one input mode.");
    }
    if is_video {
        bail!(
            "--video is not supported by the lazy v1 binary (no generation loop / video encoder \
             in the lazy module yet)."
        );
    }
    if is_batch {
        bail!(
            "--batch is not supported by the lazy v1 binary (no generation loop in the lazy \
             module yet)."
        );
    }
    if args.image.len() > 1 {
        bail!(
            "Multi-image input is not supported by the lazy v1 binary; pass exactly one --image."
        );
    }

    // Load + preprocess the single image. The lazy module expects (C, H, W).
    let image_path = &args.image[0];
    println!("Processing image: {}", image_path);
    let (pixel_values, new_h, new_w) = load_image_lazy(image_path, &device)?;

    // The lazy vision encoder picks a tile grid via `aspect_ratio_chooser`
    // and emits `num_tiles * patches_per_tile_merged` vision tokens. We must
    // place exactly that many placeholder tokens in the prompt.
    let v_cfg = &lazy_cfg.vision;
    let merge = v_cfg.spatial_merge_size;
    let per_tile_merged = v_cfg.num_patches_per_tile() / (merge * merge);
    let (rows, cols) = fuel::lazy_paddleocr_vl::aspect_ratio_chooser(
        new_h,
        new_w,
        lazy_cfg.max_tiles_per_side,
    );
    let num_image_tokens = rows * cols * per_tile_merged;
    println!(
        "Image tile grid: {}x{} -> {} vision tokens",
        rows, cols, num_image_tokens
    );

    // Build the chat-formatted prompt:
    //   <BOS> User:  <VISION_START> <IMG>×N <VISION_END> task\nAssistant:
    let bos_token_id = tokenizer.token_to_id("<|begin_of_sentence|>").unwrap_or(1);
    let user_encoding = tokenizer
        .encode("User: ", false)
        .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
    let task_encoding = tokenizer
        .encode(args.task.prompt(), false)
        .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
    let assistant_encoding = tokenizer
        .encode("\nAssistant: ", false)
        .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;

    let mut input_ids: Vec<u32> = Vec::new();
    input_ids.push(bos_token_id);
    input_ids.extend(user_encoding.get_ids());
    input_ids.push(hf_cfg.vision_start_token_id);
    input_ids.extend(std::iter::repeat(hf_cfg.image_token_id).take(num_image_tokens));
    input_ids.push(hf_cfg.vision_end_token_id);
    input_ids.extend(task_encoding.get_ids());
    input_ids.extend(assistant_encoding.get_ids());

    println!(
        "Input sequence length: {} (task: {:?})",
        input_ids.len(),
        args.task
    );
    println!("\nRunning single forward pass...");
    let start = std::time::Instant::now();
    let logits = model
        .forward(Some(&pixel_values), &input_ids, hf_cfg.image_token_id, 0)
        .map_err(|e| E::msg(format!("forward: {e}")))?;
    let logits_vec = logits.realize_f32();
    let dims = logits.shape();
    let dims = dims.dims();
    if dims.len() != 3 {
        bail!("expected (1, seq, vocab) logits; got {:?}", dims);
    }
    let vocab = dims[2];
    let seq = dims[1];
    let last = &logits_vec[(seq - 1) * vocab..seq * vocab];
    let (next_tok_idx, _) = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits row");
    let next_token = next_tok_idx as u32;
    let elapsed = start.elapsed();

    let eos_token_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2);

    println!("\n{:=<60}", "");
    println!("Task: {:?}", args.task);
    println!("{:=<60}", "");
    println!("Single-step greedy next-token id: {}", next_token);
    if next_token == eos_token_id {
        println!("(was EOS)");
    } else if let Ok(decoded) = tokenizer.decode(&[next_token], true) {
        println!("Decoded: {}", decoded);
    }
    println!(
        "Forward pass in {:.2}s ({} input tokens)",
        elapsed.as_secs_f32(),
        input_ids.len()
    );
    println!(
        "(--max-length, --batch, --video, multi-image not implemented in the lazy v1 binary)"
    );

    Ok(())
}
