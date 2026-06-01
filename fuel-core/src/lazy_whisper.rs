//! Whisper speech recognition model ported to the lazy-graph API.
//!
//! Fuel's anchor-model #4 under Phase 6a. Whisper is an encoder-decoder
//! transformer that distinguishes itself from the prior anchors along
//! several axes:
//!
//! - **Conv1d stem** in the encoder: two stride-1/stride-2 kernel-3
//!   convolutions on the log-mel spectrogram compress 3000 frames down
//!   to 1500 before any attention runs. Fuel's lazy graph doesn't
//!   expose a native `Conv1d` op yet, so the stem is built here out of
//!   existing primitives (slice + concat + reshape + matmul). Both
//!   conv layers are kernel-3 with padding 1; stride-2 uses the even/
//!   odd reshape trick to get the downsampling.
//! - **Encoder-decoder cross-attention**: the decoder attends into the
//!   frozen encoder hidden state. Each decoder layer runs three
//!   sublayers (causal self-attn, cross-attn, FFN) instead of BERT's
//!   two or LLaMA's one-attn-then-FFN shape.
//! - **Pre-LayerNorm blocks**: LayerNorm is applied *before* each
//!   sublayer (`sublayer(LN(x)) + x`) — opposite of BERT's post-LN
//!   order. Matches the original Whisper paper and HuggingFace's
//!   `modeling_whisper.py`.
//! - **Sinusoidal positional embedding on the encoder**: fixed, not
//!   learned. The decoder's position embedding is learned, matching
//!   the `max_target_positions = 448` cap in Whisper-tiny's config.
//! - **Tied output projection**: logits come from `hidden @ embed^T`,
//!   reusing the decoder's token embedding matrix as the unembedding
//!   projection. No separate `lm_head` tensor on disk.
//! - **k_proj never has a bias** across all attention blocks — Whisper-
//!   specific quirk that our loader reflects by making the `*_k_b`
//!   fields absent (`None`).
//!
//! # Scope
//!
//! This module covers the full architecture + greedy decoding. Two
//! pieces that a real Whisper runner still needs are deferred:
//!
//! - **Audio preprocessing** (waveform → 80×3000 log-mel spectrogram):
//!   the STFT + mel filterbank + log-scale pipeline. Live outside the
//!   lazy graph. The binary at `fuel-lazy-examples/src/bin/whisper-
//!   lazy.rs` reads a pre-computed mel feature tensor from disk.
//! - **KV cache for decoder self-attention**: a Whisper-specific cache
//!   needs separate self-cache + frozen cross-cache; we re-run the
//!   full decoder forward each step here. Produces correct output; a
//!   KV cache is a constant-factor speedup best layered on the
//!   working greedy decoder rather than before it.
//!
//! # Example
//!
//! ```no_run
//! use fuel_core::lazy_whisper::{WhisperModel, WhisperTokenizer};
//!
//! let model = WhisperModel::from_hub("openai/whisper-tiny")?;
//! let tokenizer = WhisperTokenizer::from_hub("openai/whisper-tiny")?;
//! // Pre-computed mel spectrogram: [1, 80, 3000]
//! let mel = vec![0.0_f32; 80 * 3000];
//! // <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
//! let prompt = vec![50258u32, 50259, 50359, 50363];
//! let tokens = model.generate_greedy(&mel, &prompt, /* max_new = */ 64)?;
//! let text = tokenizer.decode(&tokens, true)?;
//! println!("{text}");
//! # Ok::<(), fuel_core::Error>(())
//! ```

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use serde::Deserialize;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WhisperConfig {
    pub vocab_size:             usize,
    pub num_mel_bins:           usize,
    pub d_model:                usize,
    pub encoder_layers:         usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim:        usize,
    pub decoder_layers:         usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim:        usize,
    pub max_source_positions:   usize,
    pub max_target_positions:   usize,
    #[serde(default = "default_scale_embedding")]
    pub scale_embedding:        bool,
    pub bos_token_id:           u32,
    pub eos_token_id:           u32,
    pub pad_token_id:           u32,
    pub decoder_start_token_id: u32,
}

fn default_scale_embedding() -> bool {
    false
}

impl WhisperConfig {
    pub fn from_hf_json_str(s: &str) -> crate::Result<Self> {
        serde_json::from_str::<Self>(s)
            .map_err(|e| crate::Error::Msg(format!("parsing whisper config.json: {e}")).bt())
    }

    pub fn encoder_head_dim(&self) -> usize {
        assert_eq!(self.d_model % self.encoder_attention_heads, 0);
        self.d_model / self.encoder_attention_heads
    }

    pub fn decoder_head_dim(&self) -> usize {
        assert_eq!(self.d_model % self.decoder_attention_heads, 0);
        self.d_model / self.decoder_attention_heads
    }
}

// ---- Weight storage --------------------------------------------------------

/// Weights for one encoder layer. Whisper's encoder blocks are the
/// standard pre-norm transformer shape: `x + self_attn(LN(x))`,
/// `x + ffn(LN(x))`. The `*_k_b` field on the self-attention is
/// intentionally `None` — HF Whisper never stores a bias for K.
#[derive(Debug, Clone)]
pub struct WhisperEncoderLayerWeights {
    pub self_attn_ln_g: Arc<[f32]>,
    pub self_attn_ln_b: Arc<[f32]>,
    pub q_w: Arc<[f32]>,
    pub q_b: Arc<[f32]>,
    pub k_w: Arc<[f32]>,
    pub v_w: Arc<[f32]>,
    pub v_b: Arc<[f32]>,
    pub out_w: Arc<[f32]>,
    pub out_b: Arc<[f32]>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
    pub fc1_w: Arc<[f32]>,
    pub fc1_b: Arc<[f32]>,
    pub fc2_w: Arc<[f32]>,
    pub fc2_b: Arc<[f32]>,
}

/// Weights for one decoder layer. Adds cross-attention on top of the
/// encoder layer's shape. Same no-k-bias rule applies to both the
/// self-attention and cross-attention blocks.
#[derive(Debug, Clone)]
pub struct WhisperDecoderLayerWeights {
    // --- self-attention ----
    pub self_ln_g: Arc<[f32]>,
    pub self_ln_b: Arc<[f32]>,
    pub self_q_w: Arc<[f32]>,
    pub self_q_b: Arc<[f32]>,
    pub self_k_w: Arc<[f32]>,
    pub self_v_w: Arc<[f32]>,
    pub self_v_b: Arc<[f32]>,
    pub self_out_w: Arc<[f32]>,
    pub self_out_b: Arc<[f32]>,
    // --- cross-attention ----
    pub cross_ln_g: Arc<[f32]>,
    pub cross_ln_b: Arc<[f32]>,
    pub cross_q_w: Arc<[f32]>,
    pub cross_q_b: Arc<[f32]>,
    pub cross_k_w: Arc<[f32]>,
    pub cross_v_w: Arc<[f32]>,
    pub cross_v_b: Arc<[f32]>,
    pub cross_out_w: Arc<[f32]>,
    pub cross_out_b: Arc<[f32]>,
    // --- FFN ----
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
    pub fc1_w: Arc<[f32]>,
    pub fc1_b: Arc<[f32]>,
    pub fc2_w: Arc<[f32]>,
    pub fc2_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct WhisperEncoderWeights {
    /// Shape `[out=d_model, in=num_mel_bins, k=3]`, kept in HF order.
    pub conv1_w: Arc<[f32]>,
    pub conv1_b: Arc<[f32]>,
    /// Shape `[out=d_model, in=d_model, k=3]`.
    pub conv2_w: Arc<[f32]>,
    pub conv2_b: Arc<[f32]>,
    /// Fixed sinusoidal embedding. Shape `[max_source_positions, d_model]`.
    pub positional: Arc<[f32]>,
    pub layers:    Vec<WhisperEncoderLayerWeights>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct WhisperDecoderWeights {
    /// Shape `[vocab_size, d_model]`. Also used as the output projection
    /// via weight tying.
    pub embed_tokens: Arc<[f32]>,
    /// Shape `[max_target_positions, d_model]`.
    pub embed_positions: Arc<[f32]>,
    pub layers: Vec<WhisperDecoderLayerWeights>,
    pub final_ln_g: Arc<[f32]>,
    pub final_ln_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct WhisperWeights {
    pub encoder: WhisperEncoderWeights,
    pub decoder: WhisperDecoderWeights,
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WhisperModel {
    pub config:  WhisperConfig,
    pub weights: WhisperWeights,
}

impl WhisperModel {
    /// Run the encoder to produce the `[1, 1500, d_model]` context the
    /// decoder cross-attends into.
    ///
    /// `mel` is a flat row-major `[1, num_mel_bins, T]` spectrogram —
    /// typically `[1, 80, 3000]` for 30 s of 16 kHz audio. The function
    /// validates shape-vs-config at entry.
    pub fn forward_encoder(&self, mel: &[f32], mel_time: usize) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let n_mel = cfg.num_mel_bins;
        assert_eq!(
            mel.len(),
            n_mel * mel_time,
            "forward_encoder: mel has {} elements, expected {}×{}",
            mel.len(), n_mel, mel_time
        );
        let mel_t = LazyTensor::from_f32(mel.to_vec(), Shape::from_dims(&[1, n_mel, mel_time]), &crate::Device::cpu());

        // --- conv stem (pre-attention downsample) ------------------------
        // conv1: kernel=3, stride=1, padding=1 → [1, d, T]
        let x = conv1d_k3_s1_p1(
            &mel_t,
            &self.weights.encoder.conv1_w,
            &self.weights.encoder.conv1_b,
            n_mel,
            d,
            mel_time,
        )
        .gelu();
        // conv2: kernel=3, stride=2, padding=1 → [1, d, T/2]
        assert!(mel_time.is_multiple_of(2), "mel_time must be even for stride-2 conv");
        let t_half = mel_time / 2;
        let x = conv1d_k3_s2_p1(
            &x,
            &self.weights.encoder.conv2_w,
            &self.weights.encoder.conv2_b,
            d,
            d,
            mel_time,
        )
        .gelu();

        // --- transpose to [1, T/2, d] and add positional ------------------
        let x = x.permute(&[0, 2, 1]);  // [1, T/2, d]
        let pos = x
            .const_f32_like(
                self.weights.encoder.positional.clone(),
                Shape::from_dims(&[cfg.max_source_positions, d]),
            )
            .slice(0, 0, t_half)
            .reshape(Shape::from_dims(&[1, t_half, d]))
            .broadcast_to(Shape::from_dims(&[1, t_half, d])).unwrap();
        let mut x = x.add(&pos);

        // --- encoder layers ---------------------------------------------
        for lw in &self.weights.encoder.layers {
            x = encoder_layer(&x, lw, cfg, t_half);
        }

        // --- final LN ---------------------------------------------------
        Ok(layer_norm_affine(
            &x,
            &self.weights.encoder.final_ln_g,
            &self.weights.encoder.final_ln_b,
            1e-5,
            d,
            t_half,
        ))
    }

    /// Run the decoder for a given token prefix, attending into a
    /// precomputed encoder context. Returns logits of shape
    /// `[1, seq, vocab_size]` — the caller slices the last row to pick
    /// the next token.
    pub fn forward_decoder(&self, tokens: &[u32], encoder_out: &LazyTensor) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        let d = cfg.d_model;
        let seq = tokens.len();
        assert!(seq > 0, "forward_decoder: empty tokens");
        assert!(
            seq <= cfg.max_target_positions,
            "forward_decoder: seq {seq} > max_target_positions {}",
            cfg.max_target_positions,
        );

        // Bootstrap the graph from encoder_out (keeps everything on one graph).
        let input_ids = encoder_out.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let embed = encoder_out
            .const_f32_like(self.weights.decoder.embed_tokens.clone(), Shape::from_dims(&[cfg.vocab_size, d]));
        let position_ids_vec: Vec<u32> = (0..seq as u32).collect();
        let position_ids = encoder_out
            .const_u32_like(position_ids_vec, Shape::from_dims(&[seq]));
        let pos_emb = encoder_out.const_f32_like(
            self.weights.decoder.embed_positions.clone(),
            Shape::from_dims(&[cfg.max_target_positions, d]),
        );

        let tok = embed.index_select(0, &input_ids);  // [seq, d]
        let pos = pos_emb.index_select(0, &position_ids);  // [seq, d]
        let mut x = tok.add(&pos).reshape(Shape::from_dims(&[1, seq, d]));

        for lw in &self.weights.decoder.layers {
            x = decoder_layer(&x, encoder_out, lw, cfg, seq);
        }

        let x = layer_norm_affine(
            &x,
            &self.weights.decoder.final_ln_g,
            &self.weights.decoder.final_ln_b,
            1e-5,
            d,
            seq,
        );

        // Tied output projection: logits = x @ embed^T → [1, seq, vocab].
        // embed is [vocab, d] row-major; transpose to [d, vocab] and matmul.
        let embed_t = embed.transpose().unwrap();  // [d, vocab]
        Ok(x.matmul(&embed_t))
    }

    /// Greedy decode for `max_new_tokens` steps starting from
    /// `prompt_tokens`, conditioning on a pre-computed mel
    /// spectrogram. Returns the prompt tokens concatenated with the
    /// newly generated tokens (stops early if the EOS token is
    /// produced).
    ///
    /// This rebuilds the full decoder forward every step — no KV
    /// cache yet. That keeps the code short enough to read in one
    /// sitting at the cost of quadratic decode time. KV cache is a
    /// follow-up.
    pub fn generate_greedy(
        &self,
        mel: &[f32],
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> crate::Result<Vec<u32>> {
        let mel_time = mel.len() / self.config.num_mel_bins;
        let encoder_out = self.forward_encoder(mel, mel_time)?.realize_f32();
        // Re-materialize the encoder output into a fresh LazyTensor for
        // each decode step. Cheap vs rerunning the encoder.
        let t_half = mel_time / 2;
        let enc_shape = Shape::from_dims(&[1, t_half, self.config.d_model]);
        let eos = self.config.eos_token_id;

        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        for _ in 0..max_new_tokens {
            let encoder_t = LazyTensor::from_f32(encoder_out.clone(), enc_shape.clone(), &crate::Device::cpu());
            let logits = self.forward_decoder(&tokens, &encoder_t)?;
            let flat = logits.realize_f32();
            // logits shape is [1, seq, vocab]. Pick the last row.
            let vocab = self.config.vocab_size;
            let last_row_start = (tokens.len() - 1) * vocab;
            let row = &flat[last_row_start..last_row_start + vocab];
            let (argmax, _) = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
            let next_tok = argmax as u32;
            tokens.push(next_tok);
            if next_tok == eos {
                break;
            }
        }
        Ok(tokens)
    }
}

// ---- Layer primitives ------------------------------------------------------

/// `y = LayerNorm(x) * gamma + beta`. Same shape as BERT's, parked
/// here to keep the Whisper module self-contained.
fn layer_norm_affine(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    eps: f64,
    hidden: usize,
    seq: usize,
) -> LazyTensor {
    let normed = x.layer_norm_last_dim(eps);
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))
        .broadcast_to(Shape::from_dims(&[1, seq, hidden])).unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))
        .broadcast_to(Shape::from_dims(&[1, seq, hidden])).unwrap();
    normed.mul(&g).add(&b)
}

/// `y = x @ W + b`. `x` is `[1, seq, in_f]`, `W` is `[in_f, out_f]` (the
/// transposed-at-load form we store in).
fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t);
    match b {
        Some(b) => {
            let bias = x
                .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f]))
                .broadcast_to(Shape::from_dims(&[1, seq, out_f])).unwrap();
            proj.add(&bias)
        }
        None => proj,
    }
}

/// Zero-pad `x: [1, C, T]` by 1 along the time axis, returning
/// `[1, C, T+2]`. Built via concat with a const zero tensor — no
/// native `Pad` op is needed since we only use this one padding.
fn pad_t_axis_one_each_side(x: &LazyTensor, c: usize, t: usize) -> LazyTensor {
    let zeros = x.const_f32_like(vec![0.0_f32; c], Shape::from_dims(&[1, c, 1]));
    zeros.concat(x, 2).concat(&zeros, 2)  // [1, c, t+2]
}

/// Conv1d with kernel_size=3, stride=1, padding=1, on `x: [1, in_c, T]`
/// with weight `[out_c, in_c, 3]` (HF row-major order) and bias
/// `[out_c]`. Output has shape `[1, out_c, T]`.
///
/// Built as a 3-way im2col + matmul: at each output position `t`, the
/// kernel touches `padded[t .. t+3]`. We slice those three overlapping
/// windows, concat them along the channel axis (so the combined slice
/// has `3*in_c` channels), and matmul with the kernel reshaped to
/// `[3*in_c, out_c]`.
fn conv1d_k3_s1_p1(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    in_c: usize,
    out_c: usize,
    t: usize,
) -> LazyTensor {
    // Pad T axis to T+2.
    let padded = pad_t_axis_one_each_side(x, in_c, t);
    // Three stride-1 windows along the time axis, each of length T.
    let s0 = padded.slice(2, 0, t);
    let s1 = padded.slice(2, 1, t);
    let s2 = padded.slice(2, 2, t);
    // Concat along channel axis: [1, 3*in_c, T].
    let stacked = s0.concat(&s1, 1).concat(&s2, 1);
    // Move channels-last for matmul: [1, T, 3*in_c].
    let stacked_tlast = stacked.permute(&[0, 2, 1]);
    // Kernel in storage order [out_c, in_c, 3]. We want
    // `[3*in_c, out_c]` where the 3*in_c axis is laid out in the same
    // order as the channel-stack above (k=0 block, k=1 block, k=2 block).
    // HF row-major `[out_c, in_c, 3]` indexes as `w[o*in_c*3 + i*3 + k]`.
    // Target row-major `[3*in_c, out_c]` indexes as
    // `w_out[(k*in_c + i)*out_c + o]`.
    let mut w_out = vec![0.0_f32; 3 * in_c * out_c];
    for o in 0..out_c {
        for i in 0..in_c {
            for k in 0..3 {
                w_out[(k * in_c + i) * out_c + o] = w[o * in_c * 3 + i * 3 + k];
            }
        }
    }
    let w_t = x.const_f32_like(w_out, Shape::from_dims(&[3 * in_c, out_c]));
    let y = stacked_tlast.matmul(&w_t);  // [1, T, out_c]
    // Add bias (broadcast [out_c] across [1, T, out_c]).
    let bias = x
        .const_f32_like(b.clone(), Shape::from_dims(&[out_c]))
        .reshape(Shape::from_dims(&[1, 1, out_c]))
        .broadcast_to(Shape::from_dims(&[1, t, out_c])).unwrap();
    y.add(&bias).permute(&[0, 2, 1])  // back to [1, out_c, T]
}

/// Conv1d with kernel_size=3, stride=2, padding=1, on `x: [1, in_c, T]`
/// → `[1, out_c, T/2]`. `T` must be even.
///
/// Strategy: pad to `T+2`, then slice out three `T/2`-length windows
/// covering the stride-2 access pattern (one starting at position 0,
/// one at position 1, one at position 2, each sub-sampling every
/// other element). The even/odd indexing is expressed via reshape to
/// a `[_, T/2, 2]` tile then a dim-3 slice.
fn conv1d_k3_s2_p1(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    in_c: usize,
    out_c: usize,
    t_in: usize,
) -> LazyTensor {
    assert!(t_in.is_multiple_of(2), "conv1d_k3_s2_p1 needs even T, got {t_in}");
    let t_out = t_in / 2;
    // Pad to T_in + 2. Effective T = T_in + 2.
    let padded = pad_t_axis_one_each_side(x, in_c, t_in);
    // For output t ∈ [0..T_out), the 3 kernel taps read padded at:
    //   k=0: 2t  →  positions 0, 2, 4, ..., 2*(T_out-1) = T_in - 2
    //   k=1: 2t+1 → 1, 3, ..., T_in - 1
    //   k=2: 2t+2 → 2, 4, ..., T_in
    //
    // Express each via a contiguous slice + reshape + dim-3 slice:
    //   head_head = padded.slice(2, 0, 2*T_out).reshape([1, C, T_out, 2])
    //   s0 = head_head[:, :, :, 0]  (even positions)
    //   s1 = head_head[:, :, :, 1]  (odd positions)
    //   head_tail = padded.slice(2, 2, 2*T_out).reshape([1, C, T_out, 2])
    //   s2 = head_tail[:, :, :, 0]  (shifted even positions = 2, 4, …, T_in)
    let head_head = padded
        .slice(2, 0, 2 * t_out)
        .reshape(Shape::from_dims(&[1, in_c, t_out, 2]));
    let s0 = head_head
        .slice(3, 0, 1)
        .reshape(Shape::from_dims(&[1, in_c, t_out]));
    let s1 = head_head
        .slice(3, 1, 1)
        .reshape(Shape::from_dims(&[1, in_c, t_out]));
    let head_tail = padded
        .slice(2, 2, 2 * t_out)
        .reshape(Shape::from_dims(&[1, in_c, t_out, 2]));
    let s2 = head_tail
        .slice(3, 0, 1)
        .reshape(Shape::from_dims(&[1, in_c, t_out]));
    let stacked = s0.concat(&s1, 1).concat(&s2, 1);  // [1, 3*in_c, T_out]
    let stacked_tlast = stacked.permute(&[0, 2, 1]);  // [1, T_out, 3*in_c]
    // Same kernel reshuffle as the stride-1 case.
    let mut w_out = vec![0.0_f32; 3 * in_c * out_c];
    for o in 0..out_c {
        for i in 0..in_c {
            for k in 0..3 {
                w_out[(k * in_c + i) * out_c + o] = w[o * in_c * 3 + i * 3 + k];
            }
        }
    }
    let w_t = x.const_f32_like(w_out, Shape::from_dims(&[3 * in_c, out_c]));
    let y = stacked_tlast.matmul(&w_t);
    let bias = x
        .const_f32_like(b.clone(), Shape::from_dims(&[out_c]))
        .reshape(Shape::from_dims(&[1, 1, out_c]))
        .broadcast_to(Shape::from_dims(&[1, t_out, out_c])).unwrap();
    y.add(&bias).permute(&[0, 2, 1])  // [1, out_c, T_out]
}

/// Multi-head self-attention + output projection. Shared between the
/// encoder self-attention and the decoder self/cross-attention; the
/// caller supplies the Q/K/V projections and whether a causal mask
/// should be applied.
#[allow(clippy::too_many_arguments)]
fn multi_head_attn(
    q_src: &LazyTensor,
    k_src: &LazyTensor,
    v_src: &LazyTensor,
    q_w: &Arc<[f32]>,
    q_b: &Arc<[f32]>,
    k_w: &Arc<[f32]>,
    v_w: &Arc<[f32]>,
    v_b: &Arc<[f32]>,
    out_w: &Arc<[f32]>,
    out_b: &Arc<[f32]>,
    d: usize,
    n_heads: usize,
    d_head: usize,
    q_seq: usize,
    kv_seq: usize,
    causal: bool,
) -> LazyTensor {
    let q = linear(q_src, q_w, Some(q_b), d, d, q_seq);
    let k = linear(k_src, k_w, None, d, d, kv_seq);  // no K bias
    let v = linear(v_src, v_w, Some(v_b), d, d, kv_seq);

    // [1, seq, n_heads, d_head] → [1, n_heads, seq, d_head]
    let q = q
        .reshape(Shape::from_dims(&[1, q_seq, n_heads, d_head]))
        .permute(&[0, 2, 1, 3]);
    let k = k
        .reshape(Shape::from_dims(&[1, kv_seq, n_heads, d_head]))
        .permute(&[0, 2, 1, 3]);
    let v = v
        .reshape(Shape::from_dims(&[1, kv_seq, n_heads, d_head]))
        .permute(&[0, 2, 1, 3]);

    let k_t = k.permute(&[0, 1, 3, 2]);  // [1, n_heads, d_head, kv_seq]
    let scale = 1.0_f64 / (d_head as f64).sqrt();
    let mut scores = q.matmul(&k_t).mul_scalar(scale);  // [1, n_heads, q_seq, kv_seq]

    if causal {
        // Additive lower-triangular mask: -inf above diagonal.
        let mut mask = vec![0.0_f32; q_seq * kv_seq];
        for i in 0..q_seq {
            for j in 0..kv_seq {
                if j > i {
                    mask[i * kv_seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask_t = scores
            .const_f32_like(mask, Shape::from_dims(&[q_seq, kv_seq]))
            .reshape(Shape::from_dims(&[1, 1, q_seq, kv_seq]))
            .broadcast_to(Shape::from_dims(&[1, n_heads, q_seq, kv_seq])).unwrap();
        scores = scores.add(&mask_t);
    }

    let probs = scores.softmax_last_dim();
    let ctx = probs
        .matmul(&v)
        .permute(&[0, 2, 1, 3])
        .reshape(Shape::from_dims(&[1, q_seq, d]));
    linear(&ctx, out_w, Some(out_b), d, d, q_seq)
}

/// One pre-LN encoder block: self-attention + FFN, each as
/// `x + sublayer(LN(x))`.
fn encoder_layer(
    x: &LazyTensor,
    lw: &WhisperEncoderLayerWeights,
    cfg: &WhisperConfig,
    seq: usize,
) -> LazyTensor {
    let d = cfg.d_model;
    let n_heads = cfg.encoder_attention_heads;
    let d_head = cfg.encoder_head_dim();

    let x_ln = layer_norm_affine(x, &lw.self_attn_ln_g, &lw.self_attn_ln_b, 1e-5, d, seq);
    let attn = multi_head_attn(
        &x_ln, &x_ln, &x_ln,
        &lw.q_w, &lw.q_b, &lw.k_w, &lw.v_w, &lw.v_b, &lw.out_w, &lw.out_b,
        d, n_heads, d_head, seq, seq, false,
    );
    let x = x.add(&attn);

    let x_ln = layer_norm_affine(&x, &lw.final_ln_g, &lw.final_ln_b, 1e-5, d, seq);
    let h_ff = cfg.encoder_ffn_dim;
    let mid = linear(&x_ln, &lw.fc1_w, Some(&lw.fc1_b), d, h_ff, seq).gelu();
    let ffn = linear(&mid, &lw.fc2_w, Some(&lw.fc2_b), h_ff, d, seq);
    x.add(&ffn)
}

/// One pre-LN decoder block: causal self-attn + cross-attn + FFN.
fn decoder_layer(
    x: &LazyTensor,
    encoder_out: &LazyTensor,
    lw: &WhisperDecoderLayerWeights,
    cfg: &WhisperConfig,
    q_seq: usize,
) -> LazyTensor {
    let d = cfg.d_model;
    let n_heads = cfg.decoder_attention_heads;
    let d_head = cfg.decoder_head_dim();
    let kv_seq_self = q_seq;

    // --- self-attn (causal) -------
    let x_ln = layer_norm_affine(x, &lw.self_ln_g, &lw.self_ln_b, 1e-5, d, q_seq);
    let self_attn = multi_head_attn(
        &x_ln, &x_ln, &x_ln,
        &lw.self_q_w, &lw.self_q_b, &lw.self_k_w, &lw.self_v_w, &lw.self_v_b,
        &lw.self_out_w, &lw.self_out_b,
        d, n_heads, d_head, q_seq, kv_seq_self, true,
    );
    let x = x.add(&self_attn);

    // --- cross-attn ---------------
    let x_ln = layer_norm_affine(&x, &lw.cross_ln_g, &lw.cross_ln_b, 1e-5, d, q_seq);
    // encoder_out has shape [1, T_enc, d]. Use it as the K and V source.
    let enc_dims = encoder_out.dims();
    assert_eq!(
        enc_dims.len(), 3,
        "encoder_out must be [1, T, d]; got {enc_dims:?}"
    );
    let kv_seq_cross = enc_dims[1];
    let cross = multi_head_attn(
        &x_ln, encoder_out, encoder_out,
        &lw.cross_q_w, &lw.cross_q_b, &lw.cross_k_w, &lw.cross_v_w, &lw.cross_v_b,
        &lw.cross_out_w, &lw.cross_out_b,
        d, n_heads, d_head, q_seq, kv_seq_cross, false,
    );
    let x = x.add(&cross);

    // --- FFN ----------------------
    let x_ln = layer_norm_affine(&x, &lw.final_ln_g, &lw.final_ln_b, 1e-5, d, q_seq);
    let h_ff = cfg.decoder_ffn_dim;
    let mid = linear(&x_ln, &lw.fc1_w, Some(&lw.fc1_b), d, h_ff, q_seq).gelu();
    let ffn = linear(&mid, &lw.fc2_w, Some(&lw.fc2_b), h_ff, d, q_seq);
    x.add(&ffn)
}

// ---- Safetensors loader ----------------------------------------------------

impl WhisperWeights {
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &WhisperConfig,
    ) -> crate::Result<Self> {
        let d = cfg.d_model;
        // --- encoder ------------------------------------------------
        let conv1_w = load_f32(st, "model.encoder.conv1.weight")?;  // [d, n_mel, 3]
        let conv1_b = load_f32(st, "model.encoder.conv1.bias")?;
        let conv2_w = load_f32(st, "model.encoder.conv2.weight")?;  // [d, d, 3]
        let conv2_b = load_f32(st, "model.encoder.conv2.bias")?;
        let positional = load_f32(st, "model.encoder.embed_positions.weight")?;  // [max_src, d]

        let mut enc_layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            let p = format!("model.encoder.layers.{i}");
            let self_attn_ln_g = load_f32(st, &format!("{p}.self_attn_layer_norm.weight"))?;
            let self_attn_ln_b = load_f32(st, &format!("{p}.self_attn_layer_norm.bias"))?;
            let q_w = load_transposed(st, &format!("{p}.self_attn.q_proj.weight"), d, d)?;
            let q_b = load_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
            let k_w = load_transposed(st, &format!("{p}.self_attn.k_proj.weight"), d, d)?;
            let v_w = load_transposed(st, &format!("{p}.self_attn.v_proj.weight"), d, d)?;
            let v_b = load_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
            let out_w = load_transposed(st, &format!("{p}.self_attn.out_proj.weight"), d, d)?;
            let out_b = load_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;
            let final_ln_g = load_f32(st, &format!("{p}.final_layer_norm.weight"))?;
            let final_ln_b = load_f32(st, &format!("{p}.final_layer_norm.bias"))?;
            let fc1_w = load_transposed(st, &format!("{p}.fc1.weight"), cfg.encoder_ffn_dim, d)?;
            let fc1_b = load_f32(st, &format!("{p}.fc1.bias"))?;
            let fc2_w = load_transposed(st, &format!("{p}.fc2.weight"), d, cfg.encoder_ffn_dim)?;
            let fc2_b = load_f32(st, &format!("{p}.fc2.bias"))?;
            enc_layers.push(WhisperEncoderLayerWeights {
                self_attn_ln_g: Arc::from(self_attn_ln_g),
                self_attn_ln_b: Arc::from(self_attn_ln_b),
                q_w: Arc::from(q_w), q_b: Arc::from(q_b),
                k_w: Arc::from(k_w),
                v_w: Arc::from(v_w), v_b: Arc::from(v_b),
                out_w: Arc::from(out_w), out_b: Arc::from(out_b),
                final_ln_g: Arc::from(final_ln_g),
                final_ln_b: Arc::from(final_ln_b),
                fc1_w: Arc::from(fc1_w), fc1_b: Arc::from(fc1_b),
                fc2_w: Arc::from(fc2_w), fc2_b: Arc::from(fc2_b),
            });
        }
        let enc_final_ln_g = load_f32(st, "model.encoder.layer_norm.weight")?;
        let enc_final_ln_b = load_f32(st, "model.encoder.layer_norm.bias")?;

        // --- decoder ------------------------------------------------
        let dec_embed_tokens = load_f32(st, "model.decoder.embed_tokens.weight")?;  // [V, d]
        let dec_embed_positions = load_f32(st, "model.decoder.embed_positions.weight")?;

        let mut dec_layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let p = format!("model.decoder.layers.{i}");
            let self_ln_g = load_f32(st, &format!("{p}.self_attn_layer_norm.weight"))?;
            let self_ln_b = load_f32(st, &format!("{p}.self_attn_layer_norm.bias"))?;
            let self_q_w = load_transposed(st, &format!("{p}.self_attn.q_proj.weight"), d, d)?;
            let self_q_b = load_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
            let self_k_w = load_transposed(st, &format!("{p}.self_attn.k_proj.weight"), d, d)?;
            let self_v_w = load_transposed(st, &format!("{p}.self_attn.v_proj.weight"), d, d)?;
            let self_v_b = load_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
            let self_out_w = load_transposed(st, &format!("{p}.self_attn.out_proj.weight"), d, d)?;
            let self_out_b = load_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;

            let cross_ln_g = load_f32(st, &format!("{p}.encoder_attn_layer_norm.weight"))?;
            let cross_ln_b = load_f32(st, &format!("{p}.encoder_attn_layer_norm.bias"))?;
            let cross_q_w = load_transposed(st, &format!("{p}.encoder_attn.q_proj.weight"), d, d)?;
            let cross_q_b = load_f32(st, &format!("{p}.encoder_attn.q_proj.bias"))?;
            let cross_k_w = load_transposed(st, &format!("{p}.encoder_attn.k_proj.weight"), d, d)?;
            let cross_v_w = load_transposed(st, &format!("{p}.encoder_attn.v_proj.weight"), d, d)?;
            let cross_v_b = load_f32(st, &format!("{p}.encoder_attn.v_proj.bias"))?;
            let cross_out_w = load_transposed(st, &format!("{p}.encoder_attn.out_proj.weight"), d, d)?;
            let cross_out_b = load_f32(st, &format!("{p}.encoder_attn.out_proj.bias"))?;

            let final_ln_g = load_f32(st, &format!("{p}.final_layer_norm.weight"))?;
            let final_ln_b = load_f32(st, &format!("{p}.final_layer_norm.bias"))?;
            let fc1_w = load_transposed(st, &format!("{p}.fc1.weight"), cfg.decoder_ffn_dim, d)?;
            let fc1_b = load_f32(st, &format!("{p}.fc1.bias"))?;
            let fc2_w = load_transposed(st, &format!("{p}.fc2.weight"), d, cfg.decoder_ffn_dim)?;
            let fc2_b = load_f32(st, &format!("{p}.fc2.bias"))?;

            dec_layers.push(WhisperDecoderLayerWeights {
                self_ln_g: Arc::from(self_ln_g),
                self_ln_b: Arc::from(self_ln_b),
                self_q_w: Arc::from(self_q_w), self_q_b: Arc::from(self_q_b),
                self_k_w: Arc::from(self_k_w),
                self_v_w: Arc::from(self_v_w), self_v_b: Arc::from(self_v_b),
                self_out_w: Arc::from(self_out_w), self_out_b: Arc::from(self_out_b),
                cross_ln_g: Arc::from(cross_ln_g),
                cross_ln_b: Arc::from(cross_ln_b),
                cross_q_w: Arc::from(cross_q_w), cross_q_b: Arc::from(cross_q_b),
                cross_k_w: Arc::from(cross_k_w),
                cross_v_w: Arc::from(cross_v_w), cross_v_b: Arc::from(cross_v_b),
                cross_out_w: Arc::from(cross_out_w), cross_out_b: Arc::from(cross_out_b),
                final_ln_g: Arc::from(final_ln_g),
                final_ln_b: Arc::from(final_ln_b),
                fc1_w: Arc::from(fc1_w), fc1_b: Arc::from(fc1_b),
                fc2_w: Arc::from(fc2_w), fc2_b: Arc::from(fc2_b),
            });
        }
        let dec_final_ln_g = load_f32(st, "model.decoder.layer_norm.weight")?;
        let dec_final_ln_b = load_f32(st, "model.decoder.layer_norm.bias")?;

        Ok(Self {
            encoder: WhisperEncoderWeights {
                conv1_w: Arc::from(conv1_w),
                conv1_b: Arc::from(conv1_b),
                conv2_w: Arc::from(conv2_w),
                conv2_b: Arc::from(conv2_b),
                positional: Arc::from(positional),
                layers: enc_layers,
                final_ln_g: Arc::from(enc_final_ln_g),
                final_ln_b: Arc::from(enc_final_ln_b),
            },
            decoder: WhisperDecoderWeights {
                embed_tokens: Arc::from(dec_embed_tokens),
                embed_positions: Arc::from(dec_embed_positions),
                layers: dec_layers,
                final_ln_g: Arc::from(dec_final_ln_g),
                final_ln_b: Arc::from(dec_final_ln_b),
            },
        })
    }
}

fn load_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| crate::Error::Msg(format!("whisper load_f32 {name:?}: {e}")).bt())?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => crate::bail!("whisper load_f32: unsupported dtype {other:?} for {name:?}"),
    }
}

fn load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = load_f32(st, name)?;
    if flat.len() != out_features * in_features {
        crate::bail!(
            "whisper load_transposed: {name:?} has {} elements, expected {} ({}×{})",
            flat.len(), out_features * in_features, out_features, in_features,
        );
    }
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

// ---- HuggingFace Hub integration -------------------------------------------

impl WhisperModel {
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub whisper config: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = WhisperConfig::from_hf_json_str(&config_str)?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Msg(format!("hf-hub whisper safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&weights_path) }?;
        let weights = WhisperWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

// ---- Tokenizer -------------------------------------------------------------

/// Thin wrapper over `tokenizers::Tokenizer` for Whisper's BPE. Exposes
/// the special-token IDs a real transcription pipeline needs.
pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
}

impl WhisperTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::Error::Msg(format!("whisper tokenizer: {e}")))?;
        Ok(Self { inner })
    }

    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub whisper tokenizer.json: {e}")))?;
        Self::from_file(path)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> crate::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| crate::Error::Msg(format!("whisper decode: {e}")))
    }

    pub fn token_to_id(&self, s: &str) -> Option<u32> {
        self.inner.token_to_id(s)
    }
}

// ---- Test helpers (public so integration tests can reuse) ------------------

/// Hyperparameters for a tiny synthetic Whisper variant — small
/// enough to forward in milliseconds, structurally identical to the
/// real Whisper-tiny shape (encoder + decoder + cross-attention).
pub fn tiny_cfg() -> WhisperConfig {
    WhisperConfig {
        vocab_size:              128,
        num_mel_bins:             8,
        d_model:                  16,
        encoder_layers:            2,
        encoder_attention_heads:   4,
        encoder_ffn_dim:          32,
        decoder_layers:            2,
        decoder_attention_heads:   4,
        decoder_ffn_dim:          32,
        max_source_positions:     16,  // mel_time/2
        max_target_positions:     32,
        scale_embedding:       false,
        bos_token_id:             1,
        eos_token_id:             2,
        pad_token_id:             0,
        decoder_start_token_id:   1,
    }
}

fn arc(v: Vec<f32>) -> Arc<[f32]> {
    Arc::from(v)
}

/// Synthetic zero-weights (with `1.0` for LayerNorm gain) for the
/// given Whisper config. Public so integration tests across the
/// workspace can reuse the same shape-validated weight constructor
/// the in-module tests use.
pub fn zero_weights(cfg: &WhisperConfig) -> WhisperWeights {
        let d = cfg.d_model;
        let z = |n: usize| arc(vec![0.0_f32; n]);
        let o = |n: usize| arc(vec![1.0_f32; n]);
        WhisperWeights {
            encoder: WhisperEncoderWeights {
                conv1_w: z(d * cfg.num_mel_bins * 3),
                conv1_b: z(d),
                conv2_w: z(d * d * 3),
                conv2_b: z(d),
                positional: z(cfg.max_source_positions * d),
                layers: (0..cfg.encoder_layers)
                    .map(|_| WhisperEncoderLayerWeights {
                        self_attn_ln_g: o(d),
                        self_attn_ln_b: z(d),
                        q_w: z(d * d), q_b: z(d),
                        k_w: z(d * d),
                        v_w: z(d * d), v_b: z(d),
                        out_w: z(d * d), out_b: z(d),
                        final_ln_g: o(d),
                        final_ln_b: z(d),
                        fc1_w: z(d * cfg.encoder_ffn_dim),
                        fc1_b: z(cfg.encoder_ffn_dim),
                        fc2_w: z(cfg.encoder_ffn_dim * d),
                        fc2_b: z(d),
                    }).collect(),
                final_ln_g: o(d),
                final_ln_b: z(d),
            },
            decoder: WhisperDecoderWeights {
                embed_tokens: z(cfg.vocab_size * d),
                embed_positions: z(cfg.max_target_positions * d),
                layers: (0..cfg.decoder_layers)
                    .map(|_| WhisperDecoderLayerWeights {
                        self_ln_g: o(d), self_ln_b: z(d),
                        self_q_w: z(d * d), self_q_b: z(d),
                        self_k_w: z(d * d),
                        self_v_w: z(d * d), self_v_b: z(d),
                        self_out_w: z(d * d), self_out_b: z(d),
                        cross_ln_g: o(d), cross_ln_b: z(d),
                        cross_q_w: z(d * d), cross_q_b: z(d),
                        cross_k_w: z(d * d),
                        cross_v_w: z(d * d), cross_v_b: z(d),
                        cross_out_w: z(d * d), cross_out_b: z(d),
                        final_ln_g: o(d), final_ln_b: z(d),
                        fc1_w: z(d * cfg.decoder_ffn_dim),
                        fc1_b: z(cfg.decoder_ffn_dim),
                        fc2_w: z(cfg.decoder_ffn_dim * d),
                        fc2_b: z(d),
                    }).collect(),
                final_ln_g: o(d),
                final_ln_b: z(d),
            },
        }
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_whisper_tiny_config() {
        // Trimmed actual config.json from openai/whisper-tiny.
        let json = r#"{
          "vocab_size": 51865,
          "num_mel_bins": 80,
          "d_model": 384,
          "encoder_layers": 4,
          "encoder_attention_heads": 6,
          "encoder_ffn_dim": 1536,
          "decoder_layers": 4,
          "decoder_attention_heads": 6,
          "decoder_ffn_dim": 1536,
          "max_source_positions": 1500,
          "max_target_positions": 448,
          "scale_embedding": false,
          "bos_token_id": 50257,
          "eos_token_id": 50257,
          "pad_token_id": 50257,
          "decoder_start_token_id": 50258
        }"#;
        let cfg = WhisperConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 51865);
        assert_eq!(cfg.encoder_head_dim(), 64);
        assert_eq!(cfg.decoder_head_dim(), 64);
    }

    #[test]
    fn encoder_forward_shape() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let model = WhisperModel { config: cfg.clone(), weights };
        // mel_time = 32 → T/2 = 16 = max_source_positions.
        let mel = vec![0.0_f32; cfg.num_mel_bins * 32];
        let enc = model.forward_encoder(&mel, 32).unwrap();
        let flat = enc.realize_f32();
        assert_eq!(flat.len(), 1 * 16 * cfg.d_model);
        assert!(flat.iter().all(|v| v.is_finite()));

        // Phase 6a oracle gate.
        let flat_ref = enc.realize_f32_reference();
        crate::test_utils::assert_allclose_f32(&flat, &flat_ref, 1e-4, 1e-3);
    }

    #[test]
    fn decoder_forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let model = WhisperModel { config: cfg.clone(), weights };
        let mel = vec![0.0_f32; cfg.num_mel_bins * 32];
        let enc = model.forward_encoder(&mel, 32).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let logits = model.forward_decoder(&tokens, &enc).unwrap();
        let flat = logits.realize_f32();
        assert_eq!(flat.len(), 1 * tokens.len() * cfg.vocab_size);
        assert!(flat.iter().all(|v| v.is_finite()));

        // Phase 6a oracle gate.
        let flat_ref = logits.realize_f32_reference();
        crate::test_utils::assert_allclose_f32(&flat, &flat_ref, 1e-4, 1e-3);
    }

    #[test]
    fn generate_greedy_runs() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let model = WhisperModel { config: cfg.clone(), weights };
        let mel = vec![0.0_f32; cfg.num_mel_bins * 32];
        let tokens = model.generate_greedy(&mel, &[1], 4).unwrap();
        // With zero-init weights the logits are zero across the vocab,
        // so argmax picks index 0 (pad_token_id); pad != eos here so
        // the loop runs to max_new.
        assert_eq!(tokens, vec![1, 0, 0, 0, 0]);
    }
}
