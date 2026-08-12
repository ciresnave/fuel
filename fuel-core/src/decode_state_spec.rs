//! Per-layer decode-state description (GAP-029 / GAP-166).
//!
//! # Why this exists
//!
//! The decode seam used to describe a model's cache with three scalars —
//! `n_layers`, `n_kv_heads`, `head_dim` — and that vocabulary **asserts** two
//! things it was never entitled to assert: that every layer holds the same
//! state, and that the state is per-head K/V at all.
//!
//! Both are false on `main` today. `DeepSeek2Model` decodes through a
//! [`crate::lazy_latent_cache::LazyLatentCache`] whose per-layer state is a
//! compressed latent trailing `[kv_lora_rank]` plus a post-RoPE `k_pe` trailing
//! `[qk_rope_head_dim]` — no per-head K/V anywhere. The hazard is not that such
//! a model *cannot* implement the scalar vocabulary; it is that it **can, and
//! wrongly**: `num_attention_heads` and `v_head_dim` are both in scope, so the
//! methods are syntactically satisfiable and a scheduler trusting them allocates
//! a standard KV cache for a model that never reads one. It type-checks, it
//! runs, and it allocates the wrong state.
//!
//! # The vocabulary is not invented here
//!
//! Fuel already generalized this, twice, and the decode seam simply did not
//! adopt it: [`crate::lazy_latent_cache::LazyLatentCache::new`] and
//! [`crate::inference_context::LatentKvCache::with_capacity`] both take
//! `slot_trailing: Vec<Vec<usize>>` — a per-slot trailing shape. In that
//! vocabulary a standard KV layer is simply the 2-slot case:
//!
//! ```text
//! KV  layer: [[n_kv_heads, head_dim], [n_kv_heads, head_dim]]   // K, V
//! MLA layer: [[kv_lora_rank],         [qk_rope_head_dim]]       // latent, k_pe
//! ```
//!
//! So `(n_kv_heads, head_dim)` is not a more primitive fact than a slot list —
//! it is one inhabitant of it.
//!
//! # What this module adds that `slot_trailing` does not have
//!
//! `LazyLatentCache` applies **one** `slot_trailing` list to **every** layer.
//! That describes state *kind* generically while still asserting uniformity
//! across *layers* — it buys one of the two dimensions and reads as if it bought
//! both. Gemma3 already varies per-layer behaviour (sliding-window vs full
//! causal, local vs global RoPE base) and LFM2 (GAP-098) varies per-layer state
//! *kind*, interleaving attention with ShortConv blocks whose decode state is a
//! rolling window rather than a growing cache.
//!
//! Hence: a spec is indexed **by layer**. Uniform models return the same spec
//! for every index and are unaffected.
//!
//! # Scope boundary — read this before assuming the assert is gone
//!
//! This module changes the vocabulary the **decode trait** speaks. It does
//! **not** change [`crate::kv_block_pool::KvGeometry`], whose own documentation
//! commits to the vLLM shared-block-table model ("a physical block addresses the
//! SAME slot in *every* layer's K/V buffer"), nor `ModelDims` in `fuel-inference`.
//! [`LayerStateSpec::collapse_uniform`] exists precisely to hand those consumers
//! what they already expect, unchanged.
//!
//! **A fix that relocates an assert while looking like it removed one is worse
//! than no fix**, because it consumes the attention that would have caught it.
//! The allocator-side assumption is tracked separately; nothing here removes it.

use crate::{Error, Result};

/// One per-token state buffer a layer appends to during decode.
///
/// The trailing shape is the per-token extent: a standard K buffer is
/// `[n_kv_heads, head_dim]`, an MLA compressed latent is `[kv_lora_rank]`, and
/// an empty trailing is a legal per-token scalar slot (matching
/// [`crate::lazy_latent_cache::LazyLatentCache::new`]'s documented contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSlot {
    /// Per-token trailing shape, excluding the sequence axis.
    pub trailing: Vec<usize>,
}

impl StateSlot {
    pub fn new(trailing: impl Into<Vec<usize>>) -> Self {
        Self { trailing: trailing.into() }
    }

    /// Elements contributed per token. An empty trailing is a scalar slot, so
    /// the empty product is 1 — not 0.
    pub fn elems_per_token(&self) -> usize {
        self.trailing.iter().product::<usize>().max(1)
    }
}

/// The decode state ONE layer requires.
///
/// Note the contract is deliberately *"the state this layer requires"* and not
/// *"the KV shape of every layer"*. The two spellings have the same size and
/// opposite futures: under the first, a model with a differently-shaped layer
/// adds a **variant**; under the second it changes the **meaning** of the method
/// for every existing implementor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerStateSpec {
    /// Per-head key/value cache — the shape all 8 decoder-only families in
    /// GAP-029 scope use. Two slots, K and V, both `[n_kv_heads, head_dim]`.
    KeyValue { n_kv_heads: usize, head_dim: usize },
    /// A general slot list: MLA's `[latent, k_pe]`, or any future layer whose
    /// per-token state is not per-head K/V.
    Slots(Vec<StateSlot>),
}

impl LayerStateSpec {
    /// The slot list this layer's state decomposes into. `KeyValue` is the
    /// 2-slot case, which is the whole point of keeping it as a named variant
    /// rather than a special case: it is a *spelling*, not a different kind.
    pub fn slots(&self) -> Vec<StateSlot> {
        match self {
            Self::KeyValue { n_kv_heads, head_dim } => {
                let t = vec![*n_kv_heads, *head_dim];
                vec![StateSlot::new(t.clone()), StateSlot::new(t)]
            }
            Self::Slots(s) => s.clone(),
        }
    }

    /// Per-head KV dimensions, or `None` if this layer is not per-head KV.
    ///
    /// **Returns `None` rather than fabricating a plausible pair.** A model
    /// whose layer is not KV-shaped has no honest answer here, and the failure
    /// this whole module exists to prevent is exactly the one where a caller
    /// receives a syntactically valid pair that describes state the model does
    /// not keep.
    pub fn kv_dims(&self) -> Option<(usize, usize)> {
        match self {
            Self::KeyValue { n_kv_heads, head_dim } => Some((*n_kv_heads, *head_dim)),
            Self::Slots(_) => None,
        }
    }

    /// Collapse a per-layer spec list into the single `(n_kv_heads, head_dim)`
    /// pair today's `ModelDims` / [`crate::kv_block_pool::KvGeometry`] consumers
    /// require.
    ///
    /// **This helper is the one place the old assert is still made, so it makes
    /// it LOUDLY.** It returns an error rather than a value when the specs are
    /// not uniform per-head KV — because a helper that silently picked layer 0
    /// and moved on would become the new hiding place for exactly the assumption
    /// this module was written to surface. The uniform case is the *only* case it
    /// can serve honestly, and it says so by failing on every other one.
    pub fn collapse_uniform(specs: &[LayerStateSpec]) -> Result<(usize, usize)> {
        let first = specs.first().ok_or_else(|| Error::Msg(
            "LayerStateSpec::collapse_uniform: no layers — a model with zero \
             layers has no cache geometry to collapse".into(),
        ).bt())?;

        let (n_kv_heads, head_dim) = first.kv_dims().ok_or_else(|| Error::Msg(
            "LayerStateSpec::collapse_uniform: layer 0 is not per-head KV, so it \
             has no (n_kv_heads, head_dim) to collapse to. This model needs a \
             slot-aware allocator, not a collapsed pair — see GAP-166.".into(),
        ).bt())?;

        for (i, s) in specs.iter().enumerate().skip(1) {
            match s.kv_dims() {
                Some(d) if d == (n_kv_heads, head_dim) => {}
                Some((h, d)) => return Err(Error::Msg(format!(
                    "LayerStateSpec::collapse_uniform: layer {i} is per-head KV \
                     ({h}, {d}) but layer 0 is ({n_kv_heads}, {head_dim}). A \
                     single pair cannot describe both; collapsing would silently \
                     allocate layer 0's geometry for every layer.",
                )).bt()),
                None => return Err(Error::Msg(format!(
                    "LayerStateSpec::collapse_uniform: layer {i} is not per-head \
                     KV while layer 0 is. Collapsing would allocate KV for a layer \
                     that keeps a different state kind entirely — the wrong state, \
                     not merely the wrong size.",
                )).bt()),
            }
        }
        Ok((n_kv_heads, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(h: usize, d: usize) -> LayerStateSpec {
        LayerStateSpec::KeyValue { n_kv_heads: h, head_dim: d }
    }
    /// An MLA-shaped layer, spelled exactly as `DeepSeek2Model` validates it in
    /// `forward_with_latent_cache_impl`: slot 0 trailing `[kv_lora_rank]`,
    /// slot 1 trailing `[qk_rope_head_dim]`.
    fn mla(kv_lora_rank: usize, qk_rope_head_dim: usize) -> LayerStateSpec {
        LayerStateSpec::Slots(vec![
            StateSlot::new(vec![kv_lora_rank]),
            StateSlot::new(vec![qk_rope_head_dim]),
        ])
    }

    #[test]
    fn kv_layer_is_the_two_slot_case() {
        // The claim the whole design rests on: (n_kv_heads, head_dim) is not a
        // more primitive fact than a slot list, it is one inhabitant of it.
        let slots = kv(8, 64).slots();
        assert_eq!(slots.len(), 2, "K and V");
        assert_eq!(slots[0].trailing, vec![8, 64]);
        assert_eq!(slots[1].trailing, vec![8, 64]);
        assert_eq!(slots[0].elems_per_token(), 512);
    }

    #[test]
    fn mla_layer_is_expressible_and_is_not_kv() {
        let m = mla(512, 64);
        let slots = m.slots();
        assert_eq!(slots[0].trailing, vec![512], "compressed latent");
        assert_eq!(slots[1].trailing, vec![64], "post-RoPE k_pe");
        // The load-bearing half: it must NOT answer the KV question.
        assert_eq!(
            m.kv_dims(), None,
            "an MLA layer must decline (n_kv_heads, head_dim) rather than \
             fabricate a plausible pair — fabricating it is the exact defect \
             GAP-166 records",
        );
    }

    #[test]
    fn scalar_slot_contributes_one_element_not_zero() {
        // Empty trailing is a legal per-token scalar slot; the empty product is
        // 1. Getting this wrong would size a buffer to zero bytes and the
        // failure would appear far from here.
        assert_eq!(StateSlot::new(vec![]).elems_per_token(), 1);
    }

    #[test]
    fn uniform_kv_collapses_to_the_pair_todays_consumers_expect() {
        let specs = vec![kv(8, 64); 32];
        assert_eq!(LayerStateSpec::collapse_uniform(&specs).unwrap(), (8, 64));
    }

    // ----------------------------------------------------------------------
    // The architect's fourth gate: the collapse helper must not become the new
    // place the assert hides. Each of these asserts a REFUSAL, because a helper
    // that quietly returned layer 0's geometry would pass every uniform test
    // above while reintroducing precisely the defect this module removes.
    // ----------------------------------------------------------------------

    #[test]
    fn non_uniform_kv_dims_refuse_to_collapse() {
        let mut specs = vec![kv(8, 64); 4];
        specs[2] = kv(4, 64); // GQA-style divergence
        let err = LayerStateSpec::collapse_uniform(&specs).unwrap_err().to_string();
        assert!(err.contains("layer 2"), "must name the offending layer: {err}");
    }

    #[test]
    fn mixed_state_kinds_refuse_to_collapse() {
        // The LFM2 / GAP-098 shape: attention layers interleaved with layers
        // holding a different state kind. Collapsing here would allocate KV for
        // a layer that keeps something else — the wrong state, not the wrong size.
        let specs = vec![kv(8, 64), mla(512, 64), kv(8, 64)];
        let err = LayerStateSpec::collapse_uniform(&specs).unwrap_err().to_string();
        assert!(
            err.contains("wrong state"),
            "the error must distinguish wrong-state from wrong-size: {err}",
        );
    }

    #[test]
    fn all_mla_refuses_to_collapse_even_though_it_is_uniform() {
        // Uniformity is NOT sufficient — this model is perfectly uniform and
        // still has no honest (n_kv_heads, head_dim). A helper that keyed only
        // on "are all layers equal?" would pass this and be wrong.
        let specs = vec![mla(512, 64); 8];
        assert!(
            LayerStateSpec::collapse_uniform(&specs).is_err(),
            "uniform-but-not-KV must still refuse; uniformity and KV-shapedness \
             are independent properties and conflating them is the bug",
        );
    }

    #[test]
    fn zero_layers_refuses_rather_than_returning_a_default() {
        assert!(LayerStateSpec::collapse_uniform(&[]).is_err());
    }

    // ----------------------------------------------------------------------
    // SABOTAGE RECORD (2026-08-12) — these tests were born GREEN, so passing
    // proved nothing until the defect they exist to catch was actually
    // introduced and they were watched to fail.
    //
    // `collapse_uniform` was temporarily replaced with the exact defect named
    // in its own doc — pick layer 0, fabricate `(1, 1)` when it is not KV, and
    // move on:
    //
    //     return Ok(first.kv_dims().unwrap_or((1, 1)));
    //
    // Result, in a run whose log carries `Compiling fuel-core` so the binary is
    // known to be rebuilt rather than cached:
    //
    //     mixed_state_kinds_refuse_to_collapse                ... FAILED
    //     all_mla_refuses_to_collapse_even_though_it_is_uniform ... FAILED
    //     non_uniform_kv_dims_refuse_to_collapse              ... FAILED
    //     uniform_kv_collapses_to_the_pair_todays_consumers_expect ... ok
    //     test result: FAILED. 5 passed; 3 failed
    //
    // The last line is the part that makes the first three meaningful: the
    // positive control stayed GREEN, so the three failures are the helper
    // being wrong, not the suite being broken. A refusal test that fails under
    // every implementation certifies nothing.
    //
    // `zero_layers_refuses_rather_than_returning_a_default` also stayed green,
    // and that is recorded rather than hidden: the sabotage left the
    // empty-input branch intact, so that test did NOT discriminate this
    // particular defect. It guards a different branch and should not be counted
    // as evidence for this one.
    // ----------------------------------------------------------------------
}
