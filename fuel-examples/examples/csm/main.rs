//! CSM (Conversational Speech Model) — lazy port demo, v1 surface only.
//!
//! Sesame's CSM is a two-stage speech-token model: a Llama-1B backbone
//! consumes the summed multi-codebook + text embedding of an interleaved
//! `(audio_cb_0..N-1, text)` frame and predicts the codebook-0 audio token,
//! then a smaller Llama-100M decoder auto-regressively predicts codebooks
//! 1..N-1. Each generated audio frame triggers two KV-cache rollouts:
//! one over the backbone (single step) and 31 steps over the decoder.
//!
//! `lazy_csm` intentionally ships only the **embedding + heads** v1 surface
//! (see `fuel-core/src/lazy_csm.rs` for the scope rationale). The full
//! dual-Llama AR generate loop is consumer-driven glue and lives outside
//! this binary today — it requires per-frame KV-cache reuse and a
//! `backbone.forward_embeds(emb, pos)` entry point that the lazy LLaMA
//! port doesn't expose yet. We document that as a TODO and bail out
//! cleanly on `--num-frames > 1`.
//!
//! This binary demonstrates the v1 surface end-to-end against the canonical
//! Sesame checkpoint:
//!   - load `CsmWeights::load_from_mmapped` from a safetensors file,
//!   - read the first interleaved frame from a voice safetensors file
//!     (the same layout as the original eager example —
//!     `tokens: I64 [1, S, num_codebooks+1]`,
//!     `mask:   U8  [1, S, num_codebooks+1]`),
//!   - call `CsmModel::forward_single_frame(audio_codes, text_tokens,
//!     mask, anchor)` which returns `(embed, codebook0_logits)`,
//!   - argmax the codebook-0 logits at the final position.
//!
//! TODO(consumer-driven AR loop):
//!   1. Wire `LlamaModel::forward_embeds(emb, start_pos)` for the backbone
//!      so the codebook-0 token comes from the backbone hidden state, not
//!      from `codebook0_head` applied directly to the embedding.
//!   2. Implement the 31-step decoder loop with a fresh decoder KV cache
//!      per generated audio frame:
//!        h0 = backbone.forward_embeds(emb, pos);
//!        c0 = sample(codebook0_logits(h0[:,-1,:]));
//!        curr = concat(h0[:,-1,:], audio_embed_for_code(c0, 0));
//!        curr = project_to_decoder(curr);  // (1, 2, decoder_dim)
//!        for i in 1..num_codebooks {
//!            di = decoder.forward_embeds(curr, dec_pos);
//!            ci = sample(audio_head_logits(di[:,-1,:], i));
//!            curr = audio_embed_for_code(ci, i);
//!        }
//!   3. Feed the generated audio codes back into a fresh interleaved
//!      frame (with `tokens_mask` set so the new audio columns are active
//!      and text column is zero) and re-enter the backbone for the next
//!      frame.
//!   4. Decode the produced codebook-0..31 stream through Mimi to get
//!      PCM audio.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;

use fuel::lazy::LazyTensor;
use fuel::lazy_csm::{CsmConfig, CsmModel, CsmWeights};
use fuel::Shape;
use hf_hub::{api::sync::Api, Repo, RepoType};

#[derive(Parser, Debug)]
#[command(author, version, about = "CSM lazy demo (v1 embed + codebook0 head)", long_about = None)]
struct Args {
    /// Voice-prompt safetensors file. Layout matches the original Sesame
    /// eager example: `tokens: I64 [1, S, num_codebooks+1]`,
    /// `mask: U8 [1, S, num_codebooks+1]`. We read frame 0 (the first
    /// position along the seq axis) for the v1 demo.
    #[arg(long)]
    voice_safetensors: String,

    /// Conversational text prompt (informational only in the v1 demo —
    /// the AR loop that would tokenize and feed it is the TODO above).
    #[arg(long, default_value = "Hey how are you doing today?")]
    prompt: String,

    /// CSM model weights (safetensors). Defaults to the canonical
    /// `sesame/csm-1b` repo. Pass an explicit path to use a local mirror.
    #[arg(long)]
    weights: Option<String>,

    /// Number of audio frames to generate. v1 only supports a single
    /// frame (no AR loop yet — see the TODO at the top of this file).
    /// The CLI accepts the flag for forward-compatibility; values > 1
    /// bail out with a clear pointer to what's missing.
    #[arg(long, default_value_t = 1)]
    num_frames: usize,

    /// Optional override for the model repo (HF id like `sesame/csm-1b`).
    #[arg(long)]
    model_id: Option<String>,

    /// Repo revision.
    #[arg(long, default_value = "main")]
    revision: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.num_frames == 0 {
        bail!("--num-frames must be >= 1");
    }
    if args.num_frames > 1 {
        bail!(
            "--num-frames {} requested but lazy_csm v1 only ships the \
             embed + codebook-0 head surface. The dual-Llama AR loop \
             that generates additional frames is consumer-side TODO; \
             see the module-level comment in fuel-examples/examples/csm/main.rs \
             for the implementation sketch.",
            args.num_frames,
        );
    }

    println!("prompt (informational, not yet tokenized): {:?}", args.prompt);

    // ---- Locate model weights ------------------------------------------
    let weights_path: std::path::PathBuf = match args.weights {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let model_id = args
                .model_id
                .unwrap_or_else(|| "sesame/csm-1b".to_string());
            let api = Api::new()?;
            let repo = api.repo(Repo::with_revision(
                model_id,
                RepoType::Model,
                args.revision,
            ));
            repo.get("model.safetensors")?
        }
    };
    println!("loading CSM weights from {}", weights_path.display());

    // ---- Build the model ------------------------------------------------
    let cfg = CsmConfig::sesame();
    println!(
        "config: audio_num_codebooks={}, audio_vocab_size={}, \
         text_vocab_size={}, backbone_dim={}, decoder_dim={}",
        cfg.audio_num_codebooks,
        cfg.audio_vocab_size,
        cfg.text_vocab_size,
        cfg.backbone_dim,
        cfg.decoder_dim,
    );

    let t_load = std::time::Instant::now();
    let st = unsafe { fuel::safetensors::MmapedSafetensors::new(&weights_path) }
        .map_err(|e| E::msg(format!("mmap csm weights: {e}")))?;
    let weights = CsmWeights::load_from_mmapped(&st, &cfg)
        .map_err(|e| E::msg(format!("load csm weights: {e}")))?;
    let model = CsmModel { config: cfg.clone(), weights };
    println!("loaded weights in {:?}", t_load.elapsed());

    // ---- Read voice safetensors (tokens + mask, frame 0) ---------------
    println!("loading voice prompt from {}", args.voice_safetensors);
    let voice_st = unsafe {
        fuel::safetensors::MmapedSafetensors::new(&args.voice_safetensors)
    }
    .map_err(|e| E::msg(format!("mmap voice safetensors: {e}")))?;

    let cb = cfg.audio_num_codebooks;
    let expected_cols = cb + 1;

    let tokens_view = voice_st
        .get("tokens")
        .map_err(|e| E::msg(format!("voice file missing `tokens`: {e}")))?;
    let mask_view = voice_st
        .get("mask")
        .map_err(|e| E::msg(format!("voice file missing `mask`: {e}")))?;

    let tokens_shape: Vec<usize> = tokens_view.shape().to_vec();
    let mask_shape: Vec<usize> = mask_view.shape().to_vec();
    println!(
        "voice tokens shape={:?} dtype={:?}, mask shape={:?} dtype={:?}",
        tokens_shape,
        tokens_view.dtype(),
        mask_shape,
        mask_view.dtype(),
    );
    if tokens_shape != mask_shape {
        bail!(
            "tokens shape {:?} does not match mask shape {:?}",
            tokens_shape, mask_shape,
        );
    }
    if tokens_shape.len() != 3 || tokens_shape[0] != 1 || tokens_shape[2] != expected_cols {
        bail!(
            "voice tokens shape {:?} doesn't match expected [1, S, {}] \
             (num_codebooks+1)",
            tokens_shape, expected_cols,
        );
    }
    let seq_total = tokens_shape[1];
    println!("voice seq_len={} (using only frame 0 for v1 demo)", seq_total);

    // Decode tokens (I64 → u32) for frame 0 only.
    let tokens_bytes = tokens_view.data();
    let tokens_all_u32: Vec<u32> = match tokens_view.dtype() {
        safetensors::Dtype::I64 => tokens_bytes
            .chunks_exact(8)
            .map(|b| {
                i64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]) as u32
            })
            .collect(),
        safetensors::Dtype::U32 => tokens_bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => bail!("voice tokens: unexpected dtype {other:?}"),
    };
    // Decode mask (U8) for frame 0 only.
    let mask_all_u8: Vec<u8> = mask_view.data().to_vec();

    // Slice frame 0: take the first row of (seq_total, num_codebooks+1).
    let frame0_row = &tokens_all_u32[..expected_cols];
    let frame0_mask = &mask_all_u8[..expected_cols];

    // Split into (audio_codes[0..cb], text_token[cb]).
    let audio_codes: Vec<u32> = frame0_row[..cb].to_vec();
    let text_tokens: Vec<u32> = vec![frame0_row[cb]];
    // tokens_mask is `seq_len * (cb + 1)`; for seq_len = 1 that's just
    // the frame-0 row.
    let tokens_mask: Vec<u8> = frame0_mask.to_vec();

    // ---- Run the v1 forward pass ---------------------------------------
    // Anchor LazyTensor: every constant table is materialized on its graph.
    let anchor = LazyTensor::from_f32(
        vec![0.0_f32],
        Shape::from_dims(&[1]),
        &fuel::Device::cpu(),
    );

    let t_fwd = std::time::Instant::now();
    let (embed, c0_logits) = model
        .forward_single_frame(&audio_codes, &text_tokens, &tokens_mask, &anchor)
        .map_err(|e| E::msg(format!("forward_single_frame: {e}")))?;
    println!(
        "embed shape={:?}, codebook0_logits shape={:?} ({:?})",
        embed.shape().dims(),
        c0_logits.shape().dims(),
        t_fwd.elapsed(),
    );

    // Realize and pick the codebook-0 token at the last (only) position.
    let logits_data = c0_logits.realize_f32();
    let v = cfg.audio_vocab_size;
    // (1, seq_len=1, v) flat → take the trailing v elements.
    let last = &logits_data[logits_data.len() - v..];
    let (best_idx, best_val) = last
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        });
    println!(
        "argmax codebook-0 token = {} (logit {:.4})",
        best_idx, best_val,
    );
    println!(
        "v1 demo complete. To generate codebooks 1..{} and additional \
         frames, see the TODO at the top of this file.",
        cfg.audio_num_codebooks - 1,
    );

    Ok(())
}
