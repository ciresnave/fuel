// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cached decode for Qwen3.5: ROADMAP row 115, step 5. The keys and values of each full-attention
//! layer and the `(S, conv window)` of each Gated DeltaNet layer are carried between calls as host
//! data, so a call over `T` new tokens costs `T` tokens of compute, not the whole prefix.
//!
//! This is the composed, unfused path (see `lazy_qwen3_5_forward`); it is NOT the plan-once decode
//! session of `fuel-inference` (ROADMAP row 117 owns that). Each call builds one graph, realises
//! the logits and every new cache entry in a single walk, and stores the entries.

use crate::models::lazy_qwen3_5::{Qwen3_5Layer, Qwen3_5Mixer, Qwen3_5Model};
use crate::models::lazy_qwen3_5_attn::{AttnPast, gated_full_attention_cached};
use crate::models::lazy_qwen3_5_forward::mlp_block;
use crate::models::lazy_qwen3_5_gdn::{GdnState, gated_delta_net_mixer};
use fuel_ir::Shape;
use fuel_ir::error::{Error, Result};
use fuel_tensor::Device;
use fuel_tensor::lazy::{Tensor, realize_many_f32};
use std::sync::Arc;

/// What one layer carries between calls.
#[derive(Debug, Clone)]
enum LayerCache {
    /// Nothing seen yet.
    Empty,
    /// Gated DeltaNet: `S [nv, dk, dv]` and the conv window `[1, conv_dim, K - 1]`, fixed size.
    Gdn {
        recurrent: Vec<f32>,
        conv_window: Vec<f32>,
    },
    /// Full attention: keys and values `[n_kv_heads, len, head_dim]`, growing by one row per token.
    Attn {
        k: Vec<f32>,
        v: Vec<f32>,
        len: usize,
    },
}

/// The per-sequence state of a [`Qwen3_5Model`] decode.
#[derive(Debug, Clone)]
pub struct Qwen3_5Cache {
    layers: Vec<LayerCache>,
    len: usize,
}

impl Qwen3_5Cache {
    /// Tokens consumed so far (the position of the next token).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True before the first call.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host bytes held, to size a batch of sessions.
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerCache::Empty => 0,
                LayerCache::Gdn {
                    recurrent,
                    conv_window,
                } => 4 * (recurrent.len() + conv_window.len()),
                LayerCache::Attn { k, v, .. } => 4 * (k.len() + v.len()),
            })
            .sum()
    }
}

/// The tensors a layer produced that the cache must keep.
enum NewState {
    Gdn(GdnState),
    Attn { k: Tensor, v: Tensor },
}

fn gdn_past(x: &Tensor, c: &LayerCache, m: &Qwen3_5Model) -> Result<Option<GdnState>> {
    let cfg = &m.config;
    let LayerCache::Gdn {
        recurrent,
        conv_window,
    } = c
    else {
        return Ok(None);
    };
    let s_dims = [
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
    ];
    let w_dims = [1, cfg.linear_conv_dim(), cfg.linear_conv_kernel_dim - 1];
    Ok(Some(GdnState {
        recurrent: x.const_f32_like(recurrent.clone(), Shape::from_dims(&s_dims))?,
        conv_window: x.const_f32_like(conv_window.clone(), Shape::from_dims(&w_dims))?,
    }))
}

fn attn_past(c: &LayerCache) -> Option<AttnPast<'_>> {
    match c {
        LayerCache::Attn { k, v, len } => Some(AttnPast { k, v, len: *len }),
        _ => None,
    }
}

fn mixer_cached(
    x: &Tensor,
    layer: &Qwen3_5Layer,
    cache: &LayerCache,
    m: &Qwen3_5Model,
) -> Result<(Tensor, NewState)> {
    match &layer.mixer {
        Qwen3_5Mixer::GatedDeltaNet(w) => {
            let past = gdn_past(x, cache, m)?;
            let (y, st) = gated_delta_net_mixer(x, w, &m.config, past.as_ref())?;
            Ok((y, NewState::Gdn(st)))
        }
        Qwen3_5Mixer::FullAttention(w) => {
            let (y, k, v) =
                gated_full_attention_cached(x, w, &m.config, attn_past(cache).as_ref())?;
            Ok((y, NewState::Attn { k, v }))
        }
    }
}

fn layer_cached(
    h: &Tensor,
    layer: &Qwen3_5Layer,
    cache: &LayerCache,
    m: &Qwen3_5Model,
) -> Result<(Tensor, NewState)> {
    let cfg = &m.config;
    let x = h.rms_norm_affine(Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;
    let (mixed, new) = mixer_cached(&x, layer, cache, m)?;
    Ok((mlp_block(&h.add(&mixed)?, layer, cfg)?, new))
}

impl Qwen3_5Model {
    /// An empty cache for a new sequence.
    pub fn new_cache(&self) -> Qwen3_5Cache {
        Qwen3_5Cache {
            layers: vec![LayerCache::Empty; self.weights.layers.len()],
            len: 0,
        }
    }

    fn check_call(&self, cache: &Qwen3_5Cache, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::Msg("Qwen3_5Model::decode: no tokens".into()).bt());
        }
        if cache.layers.len() != self.weights.layers.len() {
            return Err(
                Error::Msg("Qwen3_5Model::decode: cache is from another model".into()).bt(),
            );
        }
        Ok(())
    }

    fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let cfg = &self.config;
        Tensor::embed_tokens(
            Arc::clone(&self.weights.token_embedding),
            cfg.vocab_size,
            cfg.hidden_size,
            tokens,
            &Device::cpu(),
        )?
        .reshape(Shape::from_dims(&[tokens.len(), cfg.hidden_size]))
    }

    /// Feed `tokens` after the `cache.len()` tokens already consumed and return their logits
    /// `[tokens.len(), vocab]` (row-major). The cache is only updated when the call succeeds.
    pub fn decode(&self, cache: &mut Qwen3_5Cache, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let w = &self.weights;
        self.check_call(cache, tokens)?;
        let mut h = self.embed(tokens)?;
        let mut news = Vec::with_capacity(w.layers.len());
        for (layer, c) in w.layers.iter().zip(&cache.layers) {
            let (next, new) = layer_cached(&h, layer, c, self)?;
            h = next;
            news.push(new);
        }
        let normed = h.rms_norm_affine(Arc::clone(&w.final_norm_gain), cfg.rms_norm_eps)?;
        let logits = w
            .output
            .apply_linear(&normed, cfg.hidden_size, cfg.vocab_size)?;
        let (logits, stored) = realize_step(&logits, &news, cache.len + tokens.len())?;
        cache.layers = stored;
        cache.len += tokens.len();
        Ok(logits)
    }
}

/// Realise the logits and every new cache entry in one walk; `len` is the position after this call.
fn realize_step(
    logits: &Tensor,
    news: &[NewState],
    len: usize,
) -> Result<(Vec<f32>, Vec<LayerCache>)> {
    let mut targets: Vec<&Tensor> = vec![logits];
    for n in news {
        match n {
            NewState::Gdn(s) => targets.extend([&s.recurrent, &s.conv_window]),
            NewState::Attn { k, v } => targets.extend([k, v]),
        }
    }
    let mut out = realize_many_f32(&targets).into_iter();
    let logits = out
        .next()
        .ok_or_else(|| Error::Msg("Qwen3_5Model::decode: no logits".into()).bt())?;
    let stored = news
        .iter()
        .map(|n| stored_entry(n, len, &mut out))
        .collect::<Result<Vec<_>>>()?;
    Ok((logits, stored))
}

/// Take the realised buffers of one layer's new state off `out`, in the order they were queued.
fn stored_entry(
    n: &NewState,
    len: usize,
    out: &mut impl Iterator<Item = Vec<f32>>,
) -> Result<LayerCache> {
    let mut next = || {
        out.next()
            .ok_or_else(|| Error::Msg("Qwen3_5Model::decode: missing cache entry".into()).bt())
    };
    Ok(match n {
        NewState::Gdn(_) => LayerCache::Gdn {
            recurrent: next()?,
            conv_window: next()?,
        },
        NewState::Attn { .. } => LayerCache::Attn {
            k: next()?,
            v: next()?,
            len,
        },
    })
}
