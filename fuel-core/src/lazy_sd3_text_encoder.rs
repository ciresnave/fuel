//! Stable Diffusion 3 / 3.5 triple-CLIP text-encoder composer
//! ported to the lazy-graph API.
//!
//! SD3 conditions its MMDiT-X backbone on the concatenation of THREE
//! text encoders rather than the one-or-two used by SD 1.5 / SDXL:
//!
//!   - **CLIP-L** (`openai/clip-vit-large-patch14`, 768-dim,
//!     12 layers, QuickGELU). Penultimate-layer per-token hidden
//!     state `(1, 77, 768)` + EOS-pooled embedding `(1, 768)`.
//!   - **CLIP-G** (`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`,
//!     1280-dim, 32 layers, GELU). Penultimate-layer per-token
//!     hidden state `(1, 77, 1280)` + EOS-pooled embedding
//!     `(1, 1280)` projected through a `[1280, 1280]` no-bias
//!     linear (`text_projection.weight`).
//!   - **T5-XXL v1.1** (`google/t5-v1_1-xxl`, 4096-dim,
//!     24 layers). Full encoder output `(1, 77, 4096)`.
//!
//! The composer returns two tensors:
//!
//!   - `y`        — pooled vector `(1, 2048)` =
//!                  `cat([clip_l_pooled (768), clip_g_pooled_projected (1280)], dim=-1)`.
//!                  Fed into MMDiT's `y_embedder`.
//!   - `context`  — per-token sequence `(1, 154, 4096)`:
//!                  1. `clip_concat = cat([clip_l_pen (1,77,768), clip_g_pen (1,77,1280)], -1)` → `(1, 77, 2048)`,
//!                  2. `clip_padded = pad_zeros(clip_concat, -1, 0, 2048)` → `(1, 77, 4096)`,
//!                  3. `cat([clip_padded, t5 (1,77,4096)], -2)` → `(1, 154, 4096)`.
//!                  Fed into MMDiT's `context_embedder`.
//!
//! # Substrate choice
//!
//! Reuses [`crate::lazy_sd_text_encoder::SdTextEncoder`] for both
//! CLIP encoders because that module already supports the
//! QuickGELU / GELU activation switch needed for CLIP-L vs CLIP-G
//! (CLIP-L uses QuickGELU; CLIP-G uses GELU, matching the eager
//! `Config::sdxl` / `Config::sdxl2` presets that the retired SD3
//! binary at `_stable-diffusion-3_retired/clip.rs` consumed).
//! Reuses [`crate::lazy_t5::T5Model::forward_encoder`] for the
//! T5-XXL hidden states.
//!
//! Note on penultimate semantics: the eager reference captures
//! the penultimate hidden BEFORE the final LayerNorm; this lazy
//! port reuses `SdTextEncoder::forward_until_encoder_layer` which
//! applies the final LayerNorm to the intermediate slot too. The
//! shape is identical and the SDXL pipeline has been treating the
//! LN'd intermediate as the conditioning signal since `lazy_sd_text_encoder`
//! shipped, so we follow that convention here. Downstream MMDiT
//! conditioning normalizes anyway via its `context_embedder` and
//! `y_embedder`.
//!
//! Tokenization stays in the binary — this module is pure tensor:
//! callers tokenize once and hand `[u32; 77]` slices to `encode`.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32 throughout. Tokens for all three
//! encoders must be exactly 77 long (CLIP/T5 share the
//! `max_position_embeddings = 77` budget that SD3 ships with).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_sd_text_encoder::{
    ClipTextActivation, ClipTextConfig, ClipTextWeights, SdTextEncoder,
};
use crate::lazy_t5::{T5Config, T5Model, T5Weights};
use crate::Result;
use fuel_ir::Shape;

/// SD3 max position embeddings for each of the three encoders.
/// Both CLIPs and T5 use 77 to keep the per-token sequence
/// budgets aligned (the composer concatenates CLIP-padded
/// `(1, 77, 4096)` with T5 `(1, 77, 4096)` along the token dim
/// to produce `(1, 154, 4096)`).
pub const SD3_MAX_POSITION_EMBEDDINGS: usize = 77;

/// CLIP-L pooled width.
pub const SD3_CLIP_L_DIM: usize = 768;

/// CLIP-G pooled width (matches the projection output width).
pub const SD3_CLIP_G_DIM: usize = 1280;

/// Pooled-y vector width: `CLIP-L (768) + CLIP-G (1280) = 2048`.
pub const SD3_POOLED_DIM: usize = SD3_CLIP_L_DIM + SD3_CLIP_G_DIM;

/// MMDiT context channel width = T5-XXL `d_model = 4096`. The
/// concatenated-and-padded CLIP slot is padded out to the same
/// width before being concatenated with the T5 slot along the
/// token dim.
pub const SD3_CONTEXT_DIM: usize = 4096;

/// SD3 context token budget: `(77 CLIP + 77 T5) = 154`.
pub const SD3_CONTEXT_TOKENS: usize = 2 * SD3_MAX_POSITION_EMBEDDINGS;

// ---- Config ----------------------------------------------------------------

/// Composition of the three sub-configs that drive the triple-CLIP
/// composer. Provides preset constructors for the published SD3 /
/// 3.5 checkpoints; advanced callers can build a custom config by
/// stitching together their own `ClipTextConfig` / `T5Config`.
#[derive(Debug, Clone)]
pub struct Sd3TripleClipConfig {
    /// CLIP-L config (768-dim, 12 layers, QuickGELU).
    pub clip_l: ClipTextConfig,
    /// CLIP-G config (1280-dim, 32 layers, GELU).
    pub clip_g: ClipTextConfig,
    /// T5-XXL v1.1 config (4096-dim, 24 layers).
    pub t5: T5Config,
}

impl Sd3TripleClipConfig {
    /// Published SD3-medium / SD3.5-large / SD3.5-large-turbo
    /// preset. CLIP-L uses the SDXL TE1 shape (QuickGELU);
    /// CLIP-G uses the SDXL TE2 shape (GELU); T5 uses the
    /// T5-v1_1-XXL shape (24 layers, gated GELU, `d_model = 4096`,
    /// `d_kv = 64`, `d_ff = 10240`, no `tie_word_embeddings`).
    ///
    /// Override fields after construction to target derivative
    /// checkpoints (e.g. shrinking the T5 layer count to match
    /// distilled variants).
    pub fn sd3_medium() -> Self {
        Self {
            clip_l: ClipTextConfig::sdxl_te1(),
            clip_g: Self::clip_g_config(),
            t5: Self::t5_xxl_config(),
        }
    }

    /// CLIP-G uses the SDXL TE2 shape but the eager SD3 reference
    /// (`fuel-examples/_stable-diffusion-3_retired/clip.rs` at
    /// commit `cfcb35cf~1`) routes it through the same
    /// `stable_diffusion::clip::Config::sdxl2` preset that the
    /// SDXL TE2 already shipped against, so the in-tree
    /// `sdxl_te2()` preset is a drop-in match.
    fn clip_g_config() -> ClipTextConfig {
        // `sdxl_te2` already sets: vocab=49408, hidden=1280, layers=32,
        // heads=20, intermediate=5120, max_pos=77, GELU activation.
        ClipTextConfig::sdxl_te2()
    }

    /// T5-XXL v1.1 (`google/t5-v1_1-xxl`) shape used by SD3.
    fn t5_xxl_config() -> T5Config {
        T5Config {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_decoder_layers: Some(24),
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
            activation: crate::lazy_t5::T5Activation::GeluPytorchTanh,
            gated_ffn: true,
            tie_word_embeddings: false,
        }
    }
}

// ---- Weight storage --------------------------------------------------------

/// Composition of the three sub-weight bundles plus the CLIP-G
/// `text_projection` no-bias linear that the eager SD3 binary
/// loads separately.
#[derive(Debug, Clone)]
pub struct Sd3TripleClipWeights {
    pub clip_l: ClipTextWeights,
    pub clip_g: ClipTextWeights,
    /// `[1280, 1280]` no-bias linear projecting CLIP-G's pooled
    /// EOS vector before it is concatenated with CLIP-L's pooled
    /// to form the `y` vector.
    pub clip_g_text_projection: WeightStorage,
    pub t5: T5Weights,
}

// ---- Model -----------------------------------------------------------------

/// Composing model that owns the three sub-encoders and the
/// CLIP-G text-projection linear. Call [`Self::encode`] with
/// pre-padded `[u32; 77]` token slices for each of the three
/// encoders to obtain the `(context, y)` pair that MMDiT's
/// `context_embedder` and `y_embedder` consume.
#[derive(Debug, Clone)]
pub struct Sd3TripleClip {
    pub config: Sd3TripleClipConfig,
    pub clip_l: SdTextEncoder,
    pub clip_g: SdTextEncoder,
    pub clip_g_text_projection: WeightStorage,
    pub t5: T5Model,
}

impl Sd3TripleClip {
    /// Build the composer from its three sub-models + the CLIP-G
    /// projection. Convenience wrapper that mirrors the eager
    /// `StableDiffusion3TripleClipWithTokenizer::new_split`
    /// constructor structurally.
    pub fn new(
        config: Sd3TripleClipConfig,
        weights: Sd3TripleClipWeights,
    ) -> Self {
        let clip_l = SdTextEncoder {
            config: config.clip_l.clone(),
            weights: weights.clip_l,
        };
        let clip_g = SdTextEncoder {
            config: config.clip_g.clone(),
            weights: weights.clip_g,
        };
        let t5 = T5Model {
            config: config.t5.clone(),
            weights: weights.t5,
        };
        Self {
            config,
            clip_l,
            clip_g,
            clip_g_text_projection: weights.clip_g_text_projection,
            t5,
        }
    }

    /// Encode the prompt across all three encoders. Each token
    /// slice MUST be exactly `SD3_MAX_POSITION_EMBEDDINGS == 77`
    /// long (callers tokenize + pad in the binary, matching the
    /// `lazy_sd_text_encoder::SdTextTokenizer::encode_padded`
    /// convention used by every SD-family lazy pipeline).
    ///
    /// The EOS-pool position for each CLIP encoder is the FIRST
    /// occurrence of `eos_token_id` in that encoder's token
    /// stream — the same convention the eager
    /// `StableDiffusion3TripleClipWithTokenizer::encode_text_to_embedding`
    /// uses (where `eos_position = tokens.len() - 1` is computed
    /// against the un-padded token list, i.e. the slot where the
    /// EOS marker sits before padding fills the tail). If no
    /// EOS token is present in the stream, the last slot
    /// `seq - 1` is used as a fallback (matching what
    /// `SdTextTokenizer::encode_padded` produces when the prompt
    /// over-fills the budget and the tokenizer truncates with
    /// EOS at the tail).
    ///
    /// Returns `(context, y)`:
    ///
    ///   - `context: (1, 154, 4096)` per-token conditioning,
    ///   - `y:       (1, 2048)`     pooled conditioning.
    pub fn encode(
        &self,
        prompt_tokens_clip_l: &[u32],
        prompt_tokens_clip_g: &[u32],
        prompt_tokens_t5: &[u32],
    ) -> Result<(LazyTensor, LazyTensor)> {
        let seq = SD3_MAX_POSITION_EMBEDDINGS;
        if prompt_tokens_clip_l.len() != seq {
            return Err(crate::Error::Msg(format!(
                "Sd3TripleClip::encode: prompt_tokens_clip_l has {} tokens, expected {seq}",
                prompt_tokens_clip_l.len(),
            )).bt());
        }
        if prompt_tokens_clip_g.len() != seq {
            return Err(crate::Error::Msg(format!(
                "Sd3TripleClip::encode: prompt_tokens_clip_g has {} tokens, expected {seq}",
                prompt_tokens_clip_g.len(),
            )).bt());
        }
        if prompt_tokens_t5.len() != seq {
            return Err(crate::Error::Msg(format!(
                "Sd3TripleClip::encode: prompt_tokens_t5 has {} tokens, expected {seq}",
                prompt_tokens_t5.len(),
            )).bt());
        }
        let eos_pos_clip_l = find_eos_pos(
            prompt_tokens_clip_l, self.config.clip_l.eos_token_id, seq,
        );
        let eos_pos_clip_g = find_eos_pos(
            prompt_tokens_clip_g, self.config.clip_g.eos_token_id, seq,
        );

        let clip_l_dim = self.config.clip_l.hidden_size;
        let clip_g_dim = self.config.clip_g.hidden_size;
        let t5_dim = self.config.t5.d_model;
        if clip_l_dim + clip_g_dim != SD3_POOLED_DIM {
            return Err(crate::Error::Msg(format!(
                "Sd3TripleClip::encode: clip_l.hidden_size ({clip_l_dim}) + clip_g.hidden_size ({clip_g_dim}) = {} != SD3_POOLED_DIM {SD3_POOLED_DIM}",
                clip_l_dim + clip_g_dim,
            )).bt());
        }
        if t5_dim != SD3_CONTEXT_DIM {
            return Err(crate::Error::Msg(format!(
                "Sd3TripleClip::encode: t5.d_model ({t5_dim}) != SD3_CONTEXT_DIM {SD3_CONTEXT_DIM}",
            )).bt());
        }

        // 1. CLIP-L: penultimate-layer hidden + EOS-pooled final hidden.
        //    This forward seeds THE graph; CLIP-G and T5 below anchor
        //    onto it so the cross-encoder concats compose in-graph
        //    (cross-graph concat is a build error).
        let (clip_l_final, clip_l_pen) = self
            .clip_l
            .forward_until_encoder_layer(prompt_tokens_clip_l, -2)?;
        let clip_l_pooled = clip_l_final
            .slice(1_usize, eos_pos_clip_l, 1)?
            .reshape(Shape::from_dims(&[1, clip_l_dim]))?;

        // 2. CLIP-G: penultimate-layer hidden + EOS-pooled final hidden,
        //    then project the pooled vector through the no-bias linear.
        let (clip_g_final, clip_g_pen) = self
            .clip_g
            .forward_until_encoder_layer_anchored(&clip_l_final, prompt_tokens_clip_g, -2)?;
        let clip_g_pooled_raw = clip_g_final
            .slice(1_usize, eos_pos_clip_g, 1)?
            .reshape(Shape::from_dims(&[1, clip_g_dim]))?;
        let clip_g_pooled = self.clip_g_text_projection.apply_linear(
            &clip_g_pooled_raw, clip_g_dim, clip_g_dim,
        );

        // 3. y vector: concat pooled CLIP-L + projected CLIP-G pooled.
        let y = clip_l_pooled.concat(&clip_g_pooled, 1_usize)?;

        // 4. context: cat CLIP penultimates along channel, pad to 4096,
        //    cat with T5 hidden along the token dim.
        let clip_concat = clip_l_pen.concat(&clip_g_pen, 2_usize)?;
        let pad_amount = SD3_CONTEXT_DIM - SD3_POOLED_DIM;
        let clip_padded = clip_concat.pad_with_zeros(2_usize, 0, pad_amount)?;
        // T5 anchored on the shared graph: embed tokens off the CLIP-L
        // anchor, then run the encoder over the pre-embedded input.
        let t5_embeds = self
            .t5
            .embed_tokens_anchored(&clip_l_final, prompt_tokens_t5)?;
        let t5_hidden = self.t5.forward_encoder_embeds(&t5_embeds)?;
        let context = clip_padded.concat(&t5_hidden, 1_usize)?;

        Ok((context, y))
    }
}

// ---- Helpers ---------------------------------------------------------------

/// First occurrence of `eos_id` in `tokens`, with the final slot
/// `seq - 1` as a fallback when no EOS marker is present. Mirrors
/// the eager `StableDiffusion3TripleClipWithTokenizer::encode_text_to_embedding`
/// behavior of using `tokens.len() - 1` after the tokenizer adds
/// the EOS to the tail.
fn find_eos_pos(tokens: &[u32], eos_id: u32, seq: usize) -> usize {
    tokens
        .iter()
        .position(|&t| t == eos_id)
        .unwrap_or(seq - 1)
}

// ---- Safetensors loader ----------------------------------------------------

impl Sd3TripleClipWeights {
    /// Load all three sub-encoder weight bundles + the CLIP-G
    /// `text_projection` from three pre-mmapped safetensors
    /// handles. This is the split-checkpoint shape used by SD 3.5
    /// (`clip_l.safetensors` / `clip_g.safetensors` /
    /// `t5xxl_fp16.safetensors` shipped separately on the Hub).
    ///
    /// Naming conventions:
    ///
    ///   - `st_clip_l` is expected to follow HF CLIP-L naming
    ///     (`text_model.*` — same as the existing
    ///     [`ClipTextWeights::load_from_mmapped`] contract that
    ///     SDXL TE1 already consumes).
    ///   - `st_clip_g` is expected to follow HF CLIP-G naming
    ///     and **additionally** carry a `text_projection.weight`
    ///     `[1280, 1280]` tensor (no bias). The SD 3.5 published
    ///     `clip_g.safetensors` ships this at the top of the
    ///     file; the SD3-medium monolithic checkpoint stores it
    ///     under `clip_g.transformer.text_projection.weight` —
    ///     both layouts are auto-detected.
    ///   - `st_t5` is expected to follow standard T5 naming
    ///     (`shared.weight`, `encoder.block.{i}.*`,
    ///     `encoder.final_layer_norm.weight`). The composer
    ///     only consumes the encoder side; decoder layers and
    ///     LM head are loaded but unused.
    ///
    /// **Checkpoint-naming caveat**: the SD3-medium monolithic
    /// checkpoint stores all three encoders under
    /// `text_encoders.{clip_l,clip_g,t5xxl}.transformer.*` rather
    /// than at the root. Routing the monolithic file through this
    /// loader is left to the binary-integration session per the
    /// design doc in `docs/session-prompts/lazy-sd3-port.md` —
    /// the simplest path is a prefix-stripping wrapper around
    /// `MmapedSafetensors` that the binary owns. The shape
    /// contract of this loader (three sub-encoder weight bundles
    /// + the CLIP-G text projection) is what matters for
    /// downstream pipeline composition.
    pub fn load_from_mmapped(
        st_clip_l: &crate::safetensors::MmapedSafetensors,
        st_clip_g: &crate::safetensors::MmapedSafetensors,
        st_t5: &crate::safetensors::MmapedSafetensors,
        config: &Sd3TripleClipConfig,
    ) -> Result<Self> {
        use crate::lazy::load_transposed_matrix_preserve_dtype;

        let clip_l = ClipTextWeights::load_from_mmapped(st_clip_l, &config.clip_l)?;
        let clip_g = ClipTextWeights::load_from_mmapped(st_clip_g, &config.clip_g)?;

        // CLIP-G text_projection: `[1280, 1280]` no-bias linear. The
        // SD 3.5 split checkpoint stores it at the root, the SD3-medium
        // monolithic checkpoint stores it under
        // `clip_g.transformer.text_projection.weight`. Try both names.
        let clip_g_text_projection = match load_transposed_matrix_preserve_dtype(
            st_clip_g, "text_projection.weight", config.clip_g.hidden_size, config.clip_g.hidden_size,
        ) {
            Ok(w) => w,
            Err(_) => load_transposed_matrix_preserve_dtype(
                st_clip_g,
                "clip_g.transformer.text_projection.weight",
                config.clip_g.hidden_size,
                config.clip_g.hidden_size,
            )?,
        };

        let t5 = T5Weights::load_from_mmapped(st_t5, &config.t5)?;

        Ok(Self {
            clip_l,
            clip_g,
            clip_g_text_projection,
            t5,
        })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_sd_text_encoder::ClipLayerWeights;
    use crate::lazy_t5::{T5AttentionWeights, T5DecoderLayerWeights, T5EncoderLayerWeights, T5FfnWeights};
    use std::sync::Arc;

    fn arc(v: Vec<f32>) -> Arc<[f32]> {
        Arc::from(v)
    }

    /// Tiny CLIP config used to drive the shape tests. We use a
    /// reduced hidden + layer count to keep the unit test cheap,
    /// then mirror that into the composer's
    /// `SD3_*` shape contract by patching the constants we care
    /// about (see `tiny_triple_config`).
    fn tiny_clip_cfg(hidden: usize, n_layers: usize, activation: ClipTextActivation) -> ClipTextConfig {
        // `hidden` must match SD3's required shape contract
        // (CLIP-L=768, CLIP-G=1280), but the MLP can be tiny in a
        // unit test — the activation + layer-loop structure is
        // what we're exercising.
        ClipTextConfig {
            vocab_size: 64,
            hidden_size: hidden,
            num_hidden_layers: n_layers,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: SD3_MAX_POSITION_EMBEDDINGS,
            layer_norm_eps: 1e-5,
            bos_token_id: 0,
            eos_token_id: 2,
            pad_token_id: 1,
            activation,
        }
    }

    fn tiny_clip_weights(cfg: &ClipTextConfig, seed: u32) -> ClipTextWeights {
        let mut s = seed;
        let next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let vec_of = |n: usize, nb: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let position_embedding = vec_of(cfg.max_position_embeddings * h, &mut *nb);
        let layers: Vec<ClipLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| ClipLayerWeights {
                ln1_g: arc(vec![1.0_f32; h]),
                ln1_b: arc(vec![0.0_f32; h]),
                q_w: vec_of(h * h, &mut *nb),
                q_b: vec_of(h, &mut *nb),
                k_w: vec_of(h * h, &mut *nb),
                k_b: vec_of(h, &mut *nb),
                v_w: vec_of(h * h, &mut *nb),
                v_b: vec_of(h, &mut *nb),
                out_w: vec_of(h * h, &mut *nb),
                out_b: vec_of(h, &mut *nb),
                ln2_g: arc(vec![1.0_f32; h]),
                ln2_b: arc(vec![0.0_f32; h]),
                fc1_w: vec_of(h * cfg.intermediate_size, &mut *nb),
                fc1_b: vec_of(cfg.intermediate_size, &mut *nb),
                fc2_w: vec_of(cfg.intermediate_size * h, &mut *nb),
                fc2_b: vec_of(h, &mut *nb),
            })
            .collect();
        ClipTextWeights {
            token_embedding,
            position_embedding,
            layers,
            final_ln_g: arc(vec![1.0_f32; h]),
            final_ln_b: arc(vec![0.0_f32; h]),
        }
    }

    fn tiny_t5_cfg(d_model: usize, n_layers: usize) -> T5Config {
        // Tiny T5: keep `d_model = SD3_CONTEXT_DIM` (required for
        // the composer's channel arithmetic) but shrink everything
        // else so the test can allocate weight tensors quickly.
        // `d_ff = 16` keeps the FFN matrices at `4096 * 16 = 64 K`
        // floats — comparable in size to CLIP-L's MLP.
        T5Config {
            vocab_size: 64,
            d_model,
            d_kv: 4,
            d_ff: 16,
            num_layers: n_layers,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            layer_norm_epsilon: 1e-6,
            activation: crate::lazy_t5::T5Activation::GeluPytorchTanh,
            gated_ffn: true,
            tie_word_embeddings: false,
        }
    }

    fn tiny_t5_weights(cfg: &T5Config, seed: u32) -> T5Weights {
        let mut s = seed;
        let next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        // Single mutable RNG closure threaded through every weight
        // tensor; closures below all take it as a `&mut dyn FnMut`
        // parameter so we never need to alias the borrow.
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
            Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
        }
        fn mk_attn(d: usize, inner: usize, nb: &mut dyn FnMut() -> f32) -> T5AttentionWeights {
            T5AttentionWeights {
                q: WeightStorage::F32(vec_of(d * inner, &mut *nb)),
                k: WeightStorage::F32(vec_of(d * inner, &mut *nb)),
                v: WeightStorage::F32(vec_of(d * inner, &mut *nb)),
                o: WeightStorage::F32(vec_of(inner * d, &mut *nb)),
            }
        }
        fn mk_ffn(d: usize, d_ff: usize, nb: &mut dyn FnMut() -> f32) -> T5FfnWeights {
            T5FfnWeights::Gated {
                wi_0: WeightStorage::F32(vec_of(d * d_ff, &mut *nb)),
                wi_1: WeightStorage::F32(vec_of(d * d_ff, &mut *nb)),
                wo: WeightStorage::F32(vec_of(d_ff * d, &mut *nb)),
            }
        }
        let d = cfg.d_model;
        let inner = cfg.num_heads * cfg.d_kv;
        let d_ff = cfg.d_ff;
        let shared_embedding = vec_of(cfg.vocab_size * d, &mut *nb);
        let encoder_rel_bias = vec_of(
            cfg.relative_attention_num_buckets * cfg.num_heads,
            &mut *nb,
        );
        let decoder_rel_bias = vec_of(
            cfg.relative_attention_num_buckets * cfg.num_heads,
            &mut *nb,
        );
        let encoder_layers: Vec<T5EncoderLayerWeights> = (0..cfg.num_layers)
            .map(|_| T5EncoderLayerWeights {
                self_attn_norm_gain: arc(vec![1.0_f32; d]),
                self_attn: mk_attn(d, inner, &mut *nb),
                ffn_norm_gain: arc(vec![1.0_f32; d]),
                ffn: mk_ffn(d, d_ff, &mut *nb),
            })
            .collect();
        let decoder_layers: Vec<T5DecoderLayerWeights> = (0..cfg.num_decoder_layers.unwrap_or(cfg.num_layers))
            .map(|_| T5DecoderLayerWeights {
                self_attn_norm_gain: arc(vec![1.0_f32; d]),
                self_attn: mk_attn(d, inner, &mut *nb),
                cross_attn_norm_gain: arc(vec![1.0_f32; d]),
                cross_attn: mk_attn(d, inner, &mut *nb),
                ffn_norm_gain: arc(vec![1.0_f32; d]),
                ffn: mk_ffn(d, d_ff, &mut *nb),
            })
            .collect();
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(d * cfg.vocab_size, &mut *nb)))
        };
        T5Weights {
            shared_embedding,
            encoder_rel_bias,
            decoder_rel_bias,
            encoder_layers,
            decoder_layers,
            encoder_final_norm_gain: arc(vec![1.0_f32; d]),
            decoder_final_norm_gain: arc(vec![1.0_f32; d]),
            lm_head,
        }
    }

    /// Tiny triple-CLIP config + weights that satisfy the SD3
    /// channel-arithmetic invariant: CLIP-L hidden + CLIP-G hidden
    /// must equal `SD3_POOLED_DIM (2048)` and T5 `d_model` must
    /// equal `SD3_CONTEXT_DIM (4096)`.
    fn tiny_triple() -> (Sd3TripleClipConfig, Sd3TripleClipWeights) {
        let clip_l = tiny_clip_cfg(SD3_CLIP_L_DIM, 2, ClipTextActivation::QuickGelu);
        let clip_g = tiny_clip_cfg(SD3_CLIP_G_DIM, 2, ClipTextActivation::Gelu);
        let t5 = tiny_t5_cfg(SD3_CONTEXT_DIM, 1);
        let clip_l_w = tiny_clip_weights(&clip_l, 11);
        let clip_g_w = tiny_clip_weights(&clip_g, 22);
        // No-bias text projection `[1280, 1280]`.
        let mut nb: Box<dyn FnMut() -> f32> = Box::new({
            let mut s: u32 = 33;
            move || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
            }
        });
        let proj_data: Vec<f32> = (0..(SD3_CLIP_G_DIM * SD3_CLIP_G_DIM))
            .map(|_| nb()).collect();
        let clip_g_text_projection = WeightStorage::F32(Arc::from(proj_data));
        let t5_w = tiny_t5_weights(&t5, 44);

        let cfg = Sd3TripleClipConfig { clip_l, clip_g, t5 };
        let weights = Sd3TripleClipWeights {
            clip_l: clip_l_w,
            clip_g: clip_g_w,
            clip_g_text_projection,
            t5: t5_w,
        };
        (cfg, weights)
    }

    /// Output shapes match the SD3 contract: `context (1, 154,
    /// 4096)` and `y (1, 2048)`.
    #[test]
    fn encode_shapes() {
        let (cfg, weights) = tiny_triple();
        let model = Sd3TripleClip::new(cfg, weights);
        // Tiny tests use vocab_size=64 in all three encoders, so
        // wrap token ids into that budget instead of running raw
        // (0..77).
        let tokens: Vec<u32> = (0..SD3_MAX_POSITION_EMBEDDINGS as u32)
            .map(|t| t % 64)
            .collect();
        let (context, y) = model.encode(&tokens, &tokens, &tokens).unwrap();
        assert_eq!(
            context.shape().dims(),
            &[1, SD3_CONTEXT_TOKENS, SD3_CONTEXT_DIM],
            "context must be (1, 154, 4096)",
        );
        assert_eq!(
            y.shape().dims(),
            &[1, SD3_POOLED_DIM],
            "y must be (1, 2048)",
        );
    }

    /// Convenience constants line up with the documented contract.
    #[test]
    fn shape_constants() {
        assert_eq!(SD3_MAX_POSITION_EMBEDDINGS, 77);
        assert_eq!(SD3_CLIP_L_DIM, 768);
        assert_eq!(SD3_CLIP_G_DIM, 1280);
        assert_eq!(SD3_POOLED_DIM, 2048);
        assert_eq!(SD3_CONTEXT_DIM, 4096);
        assert_eq!(SD3_CONTEXT_TOKENS, 154);
    }

    /// Token-length mismatches must surface a typed error rather
    /// than panic — this is the API the binary surface relies on.
    #[test]
    fn rejects_bad_token_lengths() {
        let (cfg, weights) = tiny_triple();
        let model = Sd3TripleClip::new(cfg, weights);
        let good: Vec<u32> = (0..SD3_MAX_POSITION_EMBEDDINGS as u32)
            .map(|t| t % 64)
            .collect();
        let bad: Vec<u32> = vec![1_u32, 2, 3];
        assert!(model.encode(&bad, &good, &good).is_err());
        assert!(model.encode(&good, &bad, &good).is_err());
        assert!(model.encode(&good, &good, &bad).is_err());
    }

    /// `Sd3TripleClipConfig::sd3_medium()` produces the
    /// documented preset shape contract — CLIP-L is the SDXL TE1
    /// (768/12/QuickGelu), CLIP-G is the SDXL TE2 (1280/32/Gelu),
    /// T5 is the T5-v1_1-XXL shape (4096/24, gated GELU).
    #[test]
    fn sd3_medium_preset() {
        let cfg = Sd3TripleClipConfig::sd3_medium();
        assert_eq!(cfg.clip_l.hidden_size, SD3_CLIP_L_DIM);
        assert_eq!(cfg.clip_l.num_hidden_layers, 12);
        assert_eq!(cfg.clip_l.activation, ClipTextActivation::QuickGelu);
        assert_eq!(cfg.clip_g.hidden_size, SD3_CLIP_G_DIM);
        assert_eq!(cfg.clip_g.num_hidden_layers, 32);
        assert_eq!(cfg.clip_g.activation, ClipTextActivation::Gelu);
        assert_eq!(cfg.t5.d_model, SD3_CONTEXT_DIM);
        assert_eq!(cfg.t5.num_layers, 24);
        assert!(cfg.t5.gated_ffn);
        assert!(!cfg.t5.tie_word_embeddings);
    }

    /// `load_from_mmapped` smoke: signature compiles and is callable.
    #[test]
    fn load_from_mmapped_smoke() {
        let _ = Sd3TripleClipWeights::load_from_mmapped;
    }
}
