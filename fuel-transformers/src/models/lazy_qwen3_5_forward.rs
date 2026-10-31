// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Qwen3.5 decoder stack from the composed mixers: ROADMAP row 115, step 4
//! (`docs/design-notes/2026-10-30-qwen35-composed-forward-row-115.md`).
//!
//! Prefill from position 0 only, f32. [`Qwen3_5Model::forward`] (the typed refusal, GAP-368) is
//! unchanged: it needs KV-cached attention layers and the carried Gated DeltaNet state, which are
//! the next step. [`Qwen3_5Model::prefill_logits`] is the whole-sequence forward.

use crate::models::lazy_qwen3_5::{Qwen3_5Config, Qwen3_5Layer, Qwen3_5Mixer, Qwen3_5Model};
use crate::models::lazy_qwen3_5_attn::gated_full_attention;
use crate::models::lazy_qwen3_5_gdn::gated_delta_net_mixer;
use fuel_ir::Shape;
use fuel_ir::error::{Error, Result};
use fuel_tensor::{Device, lazy::Tensor};
use std::sync::Arc;

/// One decoder layer over `h [T, hidden]`:
/// `h + mixer(norm(h))`, then `h + down(silu(gate(norm(h))) * up(norm(h)))`.
pub fn decoder_layer(h: &Tensor, layer: &Qwen3_5Layer, cfg: &Qwen3_5Config) -> Result<Tensor> {
    let x = h.rms_norm_affine(Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;
    let mixed = match &layer.mixer {
        Qwen3_5Mixer::GatedDeltaNet(w) => gated_delta_net_mixer(&x, w, cfg, None)?.0,
        Qwen3_5Mixer::FullAttention(w) => gated_full_attention(&x, w, cfg)?,
    };
    let h = h.add(&mixed)?;
    let x = h.rms_norm_affine(Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;
    let (hid, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let gate = layer.ffn_gate.apply_linear(&x, hid, inter)?;
    let up = layer.ffn_up.apply_linear(&x, hid, inter)?;
    let mlp = layer
        .ffn_down
        .apply_linear(&gate.silu().mul(&up)?, inter, hid)?;
    h.add(&mlp)
}

/// Every stage of a prefill, unrealised: the embedding, the output of each decoder layer, the
/// final norm and the logits.
pub struct PrefillStages {
    /// `[T, hidden]`.
    pub embed: Tensor,
    /// One `[T, hidden]` per layer, in order.
    pub layers: Vec<Tensor>,
    /// `[T, hidden]`.
    pub final_norm: Tensor,
    /// `[T, vocab]`.
    pub logits: Tensor,
}

impl Qwen3_5Model {
    /// Whole-sequence forward from position 0, returning every stage. Nothing is computed until a
    /// stage is realised.
    pub fn prefill_stages(&self, tokens: &[u32]) -> Result<PrefillStages> {
        let cfg = &self.config;
        let w = &self.weights;
        if tokens.is_empty() {
            return Err(Error::Msg("Qwen3_5Model::prefill_stages: no tokens".into()).bt());
        }
        let t = tokens.len();
        let embed = Tensor::embed_tokens(
            Arc::clone(&w.token_embedding),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?
        .reshape(Shape::from_dims(&[t, cfg.hidden_size]))?;
        let mut h = embed.clone();
        let mut layers = Vec::with_capacity(w.layers.len());
        for layer in &w.layers {
            h = decoder_layer(&h, layer, cfg)?;
            layers.push(h.clone());
        }
        let final_norm = h.rms_norm_affine(Arc::clone(&w.final_norm_gain), cfg.rms_norm_eps)?;
        let logits = w
            .output
            .apply_linear(&final_norm, cfg.hidden_size, cfg.vocab_size)?;
        Ok(PrefillStages {
            embed,
            layers,
            final_norm,
            logits,
        })
    }

    /// Logits `[T, vocab]` of a whole-sequence forward from position 0.
    pub fn prefill_logits(&self, tokens: &[u32]) -> Result<Tensor> {
        Ok(self.prefill_stages(tokens)?.logits)
    }
}
