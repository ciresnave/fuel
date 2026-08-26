// SPDX-License-Identifier: MIT OR Apache-2.0
//! The shared persistent-decode rebind driver (GAP-029 increment 2b).
//!
//! # Why this module exists
//!
//! `rebind_and_realize_prebuilt` — the per-token hot path that reuses a held,
//! already-optimized decode graph — was **hand-copied per model**. `LlamaModel`
//! and `PhiModel` each carried a 48-line copy, structurally identical, and
//! `DeepSeek2Model` carries a third that has *diverged* (`rebind_..._mla`).
//!
//! **Two hand-maintained copies of a hot path are a reproduction mechanism, not
//! two implementations.** GAP-029 adds up to eight more model families; porting
//! them against the copied shape would have produced ten copies and kept the
//! generator that produces copy eleven. So the driver is collapsed here *before*
//! any new family is added.
//!
//! # Why the two existing impls were ported first
//!
//! This refactor is behaviour-preserving, which means **it has no born-red
//! state**: "the tests still pass" is also exactly what a no-op produces. The
//! only thing that makes it verifiable is an oracle that predates it —
//! `LlamaModel` and `PhiModel` already work, so correctness *is* "their
//! behaviour is unchanged". A newly added family cannot serve that role: it has
//! no prior behaviour to be unchanged from, so its test can only assert what the
//! new code already does.
//!
//! Measured oracle (not assumed): 8 Phi decode tests + the parallel Llama set,
//! including `phi_generate_loop_persistent_byte_exact_and_plans_once`, which
//! asserts byte-exact logits *and* plans-once and therefore already walks the
//! multi-token **rebind** path this driver implements.
//!
//! # The path divergence this driver must PRESERVE, not unify
//!
//! Measured 2026-08-12 on CPU/F32 — and stated nowhere else in the tree:
//!
//! ```text
//! llama: offset_node.is_some() == true   → device-offset path (Op::WriteSliceDoff)
//! phi:   offset_node.is_some() == false  → SymEnv path
//! ```
//!
//! The KV-write offset rides a **device buffer** for one and a **symbol** for the
//! other. Forcing either model onto the other's path would be a behaviour change
//! wearing a refactor's clothes, so the divergence lives behind
//! [`PersistentDecodeModel::build_decode_token_data`]: each model keeps building
//! its own token data exactly as before, and this driver never inspects which
//! path it got.

use crate::inference_context::{DecodeSession, DecodeTokenData, InferenceContext, KvCache, KvSlot};
use crate::lazy::Tensor;
use crate::{Device, Result};
use fuel_ir::{DType, Shape};
use std::sync::Arc;

/// The per-model variation points of one persistent-decode rebind step.
///
/// Deliberately **one** data hook plus a layer count, rather than a hook per
/// difference. The three ways the Llama and Phi copies differed were: an extra
/// `rope_inv_freq` argument, an extra `cache_dtype` + device-offset argument
/// pair, and the `SymEnv` construction. The first two are inputs to token-data
/// construction and collapse into the hook; the third **dissolved** — see
/// [`crate::inference_context::DecodeSession::per_token_sym_env`], whose doc
/// carries the measurement and the do-not-delete warning.
pub trait PersistentDecodeModel {
    /// Transformer layers whose K/V versions are bumped after a step.
    fn decode_n_layers(&self) -> usize;

    /// Recompute this token's device-resident data Consts (token ids, RoPE
    /// cos/sin, mask, and — only on the device-offset path — the KV-write
    /// offset).
    ///
    /// **This is where the per-model path divergence lives and stays.** An impl
    /// reads `session.offset_node().is_some()` if it needs to; the driver does
    /// not, so it cannot accidentally normalise the two models onto one path.
    fn build_decode_token_data(
        &self,
        device: &Device,
        cached_len: usize,
        tokens: &[u32],
        session: &DecodeSession,
        cache: &KvCache,
        rope_inv_freq: Option<&[f64]>,
    ) -> Result<DecodeTokenData>;
}

/// One persistent-decode step on an already-built session: recompute the
/// per-token data, rebind, realize the held graph, advance cache state.
///
/// `ctx` is **not** mutated on this path — the data lands in a clone of the
/// session's held `base_cache`, not in `ctx.persistent`.
///
/// # A panic preserved on purpose
///
/// The `expect` below is carried over verbatim from *both* original copies,
/// where the caller guarantees `Some`. It is a known violation of the
/// never-panic rule and converting it to a typed error would be an improvement —
/// but this increment is behaviour-preserving, and mixing a semantic fix into it
/// would blunt the only oracle available (the pre-existing tests, which can only
/// certify "unchanged"). Tracked as a follow-up rather than smuggled in here.
pub fn rebind_and_realize_prebuilt<M: PersistentDecodeModel + ?Sized>(
    model: &M,
    tokens: &[u32],
    cache: &mut KvCache,
    ctx: &InferenceContext,
    session: &Option<DecodeSession>,
    rope_inv_freq: Option<&[f64]>,
) -> Result<Vec<f32>> {
    let seq = tokens.len();
    let cached_len = cache.cached_len;
    let device = ctx.device().clone();

    // Session guaranteed Some + valid by the caller.
    let s = session.as_ref().expect("session is Some");

    let data =
        model.build_decode_token_data(&device, cached_len, tokens, s, cache, rope_inv_freq)?;

    // Bind BOTH per-token symbols. Inert for a device-offset session, and
    // load-bearing for a SymEnv one — see `per_token_sym_env`'s doc before
    // concluding from one model that this is dead.
    let sym_env = s.per_token_sym_env(cached_len)?;
    let logits_vec = s.realize_token(&device, data, &sym_env)?;

    // Bump cache state (identical to the D1 path).
    cache.cached_len += seq;
    for li in 0..model.decode_n_layers() {
        cache.bump_version(li, KvSlot::K);
        cache.bump_version(li, KvSlot::V);
    }
    Ok(logits_vec)
}

// ===========================================================================
// SABOTAGE RECORD (2026-08-12)
//
// This extraction is behaviour-preserving, so it has NO born-red state: a
// passing suite is also exactly what a no-op produces, and "I moved the code
// and the tests are green" is indistinguishable from "I moved nothing".
//
// The specific risk worth discriminating is not "is the driver correct" — the
// pre-existing suites cover that — but "do BOTH models actually route through
// it, or does one still have a live private copy the compiler happily kept?"
// A sabotage that reddens only one model would prove the other never reached
// this code.
//
// Sabotage applied: drop `cache.cached_len += seq`, so the cache never advances.
// Run carries `Compiling fuel-core`, so the binary was rebuilt, not cached:
//
//   llama : forward_with_kv_context_persistent_plan_once_matches_d1  FAILED
//           generate_loop_persistent_byte_exact_and_plans_once       FAILED
//   llama3: scaled_persistent_decode_matches_full_prefix_forward     FAILED
//   phi   : phi_persistent_plan_once_matches_d1                      FAILED
//           phi_generate_loop_persistent_byte_exact_and_plans_once   FAILED
//   test result: FAILED. 27 passed; 5 failed
//
// Both families went red — plus `Llama3Model`, which reaches this through
// `LlamaModel` — so all three carriers demonstrably execute this driver. The 27
// that stayed green are what keep the 5 meaningful: the suite is not simply
// broken.
//
// The assertions fired at **"persistent decode token 2"**, i.e. the REBIND
// path rather than the first-token BUILD path — which is the step this driver
// actually implements, and the one a 1-token test would have been blind to.
// ===========================================================================

// ===========================================================================
// THE SHARED DECODE **BUILD** PATH (GAP-029 increment 3)
// ===========================================================================

/// # ⚠️ THIS IS SEVEN FAMILIES' BUILD PATH. A CHANGE HERE IS A SEVEN-FAMILY CHANGE.
///
/// [`build_decode_graph`] and its two tails were, until GAP-029 increment 3,
/// two inherent methods on `LlamaModel`: `forward_with_kv_context_impl` (the D1
/// rebuild-per-step path) and `build_and_realize_first_decode_token` (the D2
/// plan-once path). They live here now, and the relocation is the point.
///
/// ## Why relocated rather than doc-commented in place
///
/// The architect's ruling on increment 3 was that the six LLaMA-shaped families
/// (Qwen2, Qwen3, Qwen3Moe, SmolLm3, Glm4, Phi3) **reuse Llama's build path**
/// rather than each getting a copy — increment 3 copies it *zero* times. That
/// creates a hazard one level up from the `per_token_sym_env` trap: an inherent
/// method on `LlamaModel` that six other types call is **a shared implementation
/// wearing a private name**, so the next person to optimise "`LlamaModel`'s
/// decode path" edits seven families' decode path with nothing telling them.
/// Their evidence would be about Llama; the requirement comes from the others.
///
/// A doc comment asks the reader to already be careful. Moving the body out of
/// `LlamaModel` makes it *structurally impossible* to edit it while believing it
/// is Llama's — the same reason increment 2b put the offset-path divergence
/// behind a hook instead of behind a warning.
///
/// ## D1 and D2 are ONE body, and that was a build-order requirement
///
/// The two paths differ only in how the per-token data Consts are minted
/// (concrete vs re-bindable) and in what happens after the graph is built
/// (realize-and-discard vs optimize-and-hold). Everything between — embed,
/// RoPE tables, mask, the per-layer KV-write loop, final norm, logits — is
/// identical. [`DataConsts`] is that one difference, and it is a parameter of
/// the body **from its first line**: built for D2 alone, D1 would have become a
/// second per-family copy, six times over.
///
/// ## Phi is deliberately NOT on this path
///
/// `PhiModel` keeps its own build body. It is the outlier, not the template
/// (parallel attention block, LayerNorm+bias), and folding it in was declined on
/// its own merits. "Phi adopts the shared path" is a recorded follow-on, not
/// part of this increment.
mod shared_build_path_docs {}

/// Which flavour of per-token data Consts [`build_decode_graph`] mints.
///
/// This is the *entire* D1/D2 difference inside the graph-building half.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DataConsts {
    /// **D1** — the rebuild-per-step path. Token ids, RoPE tables and the mask
    /// are baked into the graph as concrete Consts; the graph is realized once
    /// and dropped. The KV-write offset always rides the backend-generic
    /// `SymEnv` here (no capture to make device-resident).
    Baked,
    /// **D2** — the plan-once persistent path. Every per-token datum is a
    /// re-bindable placeholder Const whose Arc is bound through the
    /// `InferenceContext`, and the optimized graph is held on a
    /// [`DecodeSession`].
    Rebindable,
}

impl DataConsts {
    /// The public entry point this mode serves, for error messages.
    fn entry(self) -> &'static str {
        match self {
            Self::Baked => "forward_with_kv_context",
            Self::Rebindable => "forward_with_kv_context_persistent",
        }
    }
}

/// The model-scalar geometry the shared build path needs.
///
/// Deliberately **not** a description of the KV state — that is
/// [`crate::decode_state_spec`]'s job, and conflating the two is the GAP-166
/// mistake. These are the dimensions of the *graph* this path builds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeDims {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Model width (`dim` / `hidden_size`).
    pub hidden: usize,
    pub vocab: usize,
    /// Width of the RoPE cos/sin tables. Equal to `head_dim` for a full-rotary
    /// family (Llama, Qwen2); a partial-rotary family (Glm4) passes its
    /// `rotary_dim`. Carried as a value rather than assumed, because that one
    /// `usize` is the whole of Glm4's divergence from this path.
    pub rope_width: usize,
    /// Multiplier applied to the token embeddings before the layer loop.
    /// `None` = no scaling, and `None` **emits no graph node at all** — a
    /// `Some(1.0)` would still cost a `mul_scalar`, which is why this is an
    /// `Option` rather than a plain `f64` defaulting to 1.
    ///
    /// Gemma-family models pass `Some(sqrt(hidden_size))`; everyone else `None`.
    ///
    /// ⚠️ **It is the SEAM that applies this, deliberately, and the ordering is
    /// the reason.** The scale must land BEFORE the activation dtype cast:
    /// prefill scales in f32, so scaling after a cast would round in bf16 on a
    /// bf16 cache and diverge — **invisible on an f32 gate**. A hook would let
    /// each family put it wherever, making that hazard per-family and silent;
    /// a value applied at one fixed point makes the ordering unforgettable.
    ///
    /// It is a scalar rather than a dimension, which the struct's name does not
    /// quite cover — stated here rather than left to overreach quietly.
    pub embed_scale: Option<f64>,
}

/// **Per-layer attention-mask variation** — the axis GAP-029 measured across
/// increment 3's families and the earlier six-axis check could not see, every
/// one of its axes being model-scalar.
///
/// Note the axis is named for the **behaviour that varies**, not the first
/// mechanism found: three families vary by sliding window, SmolLm3 varies by
/// skipping RoPE entirely, and a checklist keyed on `sliding_window` would have
/// cleared SmolLm3 outright.
///
/// # Shape in the graph
///
/// The mask Const is `[n_variants, 1, seq, max_seq_len]`; the build path hoists
/// `n_variants` width-1 slices **once**, before the layer loop, and layer `i`
/// takes `variant_for_layer(i)`. `DecodeTokenData`, `DecodeSession` and the
/// rebind driver are untouched — still one mask Arc, one `mask_node`.
///
/// **`n_variants == 1` mints `[1, 1, seq, max_seq_len]` and emits no slice node
/// at all**, so a uniform family's graph is byte-identical to the pre-GAP-029
/// one. That is asserted, not claimed: see `mask_variants_are_byte_identical_
/// to_the_dense_builder_when_uniform`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskPlan {
    /// One entry per variant. `None` is dense causal; `Some(w)` attends only to
    /// the most recent `w` positions.
    windows: Vec<Option<usize>>,
    /// Variant index per layer. `len() == n_layers`.
    per_layer: Vec<usize>,
}

impl MaskPlan {
    /// The uniform dense plan — every layer attends causally to the whole
    /// prefix. What `LlamaModel` and every other non-varying family use, and
    /// byte-identical to the pre-GAP-029 single-mask decode path.
    pub fn dense(n_layers: usize) -> Self {
        Self {
            windows: vec![None],
            per_layer: vec![0; n_layers],
        }
    }

    /// Validated at construction, per the build-time-validation rule: an
    /// out-of-range variant index would otherwise surface as a panic deep in
    /// the layer loop, or — worse — as a silently mis-masked layer.
    pub fn new(windows: Vec<Option<usize>>, per_layer: Vec<usize>) -> Result<Self> {
        if windows.is_empty() {
            return Err(
                fuel_ir::Error::Msg("MaskPlan: needs at least one variant".to_string()).bt(),
            );
        }
        if let Some((li, v)) = per_layer
            .iter()
            .enumerate()
            .find(|(_, v)| **v >= windows.len())
        {
            return Err(fuel_ir::Error::Msg(format!(
                "MaskPlan: layer {li} selects mask variant {v} but only {} \
                 variant(s) are defined",
                windows.len(),
            ))
            .bt());
        }
        Ok(Self { windows, per_layer })
    }

    /// The **`layer_idx < split`** plan: the first `split` layers attend
    /// through a `window`, the rest densely. This is exactly the predicate
    /// Qwen2, Qwen3 and Qwen3Moe share (`use_sliding_window && layer_idx <
    /// max_window_layers`), so all three express their variation by calling
    /// this rather than hand-rolling a variant table.
    ///
    /// **Total by construction — no `Result`, no panic.** Both degenerate
    /// splits collapse to a single variant, so a model that is entirely dense
    /// *or* entirely windowed emits no slice node and pays nothing:
    /// `split == 0` is [`Self::dense`], `split >= n_layers` is one windowed
    /// variant.
    pub fn split_window(n_layers: usize, split: usize, window: usize) -> Self {
        let split = split.min(n_layers);
        if split == 0 {
            return Self::dense(n_layers);
        }
        if split == n_layers {
            return Self {
                windows: vec![Some(window)],
                per_layer: vec![0; n_layers],
            };
        }
        Self {
            windows: vec![Some(window), None],
            per_layer: (0..n_layers).map(|i| usize::from(i >= split)).collect(),
        }
    }

    /// **Arbitrary per-layer window selection, TOTAL by construction.**
    ///
    /// [`Self::split_window`] covers the *prefix* predicate Qwen2/Qwen3/Qwen3Moe
    /// share. Gemma3's is **modular** (`(i + 1) % sliding_window_pattern > 0`),
    /// so it needs a free selector — and it cannot use [`Self::new`], because
    /// `decode_mask_plan` is infallible and `expect`-ing a `Result` on a
    /// production path is exactly the never-panic violation this project bans.
    /// A closure returning `bool` over exactly two defined variants cannot
    /// produce an out-of-range index, so this needs no `Result` at all.
    ///
    /// Collapses to a single variant when the selector is constant, so a config
    /// that happens to window every layer (or none) still emits no slice.
    pub fn per_layer_window(
        n_layers: usize,
        window: usize,
        uses_window: impl Fn(usize) -> bool,
    ) -> Self {
        let per_layer: Vec<usize> = (0..n_layers)
            .map(|i| usize::from(!uses_window(i)))
            .collect();
        match (per_layer.contains(&0), per_layer.contains(&1)) {
            (true, false) => Self {
                windows: vec![Some(window)],
                per_layer: vec![0; n_layers],
            },
            (false, true) => Self::dense(n_layers),
            // Mixed (or zero layers): keep both variants, windowed first.
            _ => Self {
                windows: vec![Some(window), None],
                per_layer,
            },
        }
    }

    pub fn n_variants(&self) -> usize {
        self.windows.len()
    }

    pub fn n_layers(&self) -> usize {
        self.per_layer.len()
    }

    pub fn variant_for_layer(&self, layer_idx: usize) -> usize {
        self.per_layer[layer_idx]
    }

    /// This layer's window width, or `None` if it attends densely.
    ///
    /// **The single source shared with the mask bytes** — the CUDA flash arm's
    /// `window_size_*` and the mask Const come from the *same* plan entry, so
    /// they cannot disagree. A family deriving the window separately for the arm
    /// would be free to drift from the mask it actually applies, and *"arm says
    /// dense, mask says windowed"* is precisely the silent defect GAP-194 was
    /// filed about.
    pub fn window_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.windows[self.per_layer[layer_idx]]
    }

    /// Mix the plan's **structural** content into a decode shape key.
    ///
    /// The variant count and the per-layer assignment change the graph's shape
    /// and wiring, so a session built under one plan must not be reused under
    /// another. The window *widths* are deliberately excluded: they are data,
    /// rebound per token like the RoPE tables, and baking them would forfeit
    /// plan reuse across a change that is already handled correctly — the same
    /// argument `decode_shape_key`'s own doc makes for `rope_base`.
    pub fn mix_into(&self, h: &mut crate::decode_shape::ShapeKeyHasher) {
        h.mix_u64(self.windows.len() as u64);
        for v in &self.per_layer {
            h.mix_u64(*v as u64);
        }
    }
}

/// **Per-layer RoPE-BASE variation** — the sibling axis to [`MaskPlan`], and
/// the one Gemma3 forced.
///
/// A family with two RoPE bases (Gemma3: `rope_theta` for global layers,
/// `rope_local_base_freq` for sliding ones) produces **different table bytes per
/// variant**, so — exactly like the mask — the Const must physically carry N of
/// them and the layer loop must select.
///
/// This is the criterion that separates it from SmolLm3's per-layer RoPE
/// *gating*, which needs nothing: **variation in DATA needs a variant axis;
/// variation in WHETHER-TO-APPLY does not.** A skipping layer simply does not
/// consume the shared tables; a differently-based layer needs different bytes.
///
/// # Shape, and an honest asymmetry with `MaskPlan`
///
/// `n_variants == 1` mints `[seq, rope_width]` — **exactly today's shape, with
/// no slice and no reshape** — so every single-base family's graph is unchanged
/// (pinned by the `*_held_decode_graph_has_not_grown` node-count tests).
///
/// `n_variants > 1` mints `[n, seq, rope_width]` and hoists the per-variant
/// views once before the layer loop. Unlike the mask — whose Const is already
/// rank-4, so a width-1 slice is directly usable — RoPE tables are rank-2, so
/// each variant costs a slice **and** a reshape. That cost is real and is named
/// here rather than presented as symmetric.
#[derive(Clone, Debug, PartialEq)]
pub struct RopePlan {
    bases: Vec<f64>,
    per_layer: Vec<usize>,
}

impl RopePlan {
    /// The single-base plan — what every family except Gemma3 uses.
    pub fn single(base: f64, n_layers: usize) -> Self {
        Self {
            bases: vec![base],
            per_layer: vec![0; n_layers],
        }
    }

    /// Validated at construction, like [`MaskPlan::new`]: an out-of-range
    /// variant index would otherwise be a panic in the layer loop or, worse, a
    /// layer silently roped at the wrong frequency.
    pub fn new(bases: Vec<f64>, per_layer: Vec<usize>) -> Result<Self> {
        if bases.is_empty() {
            return Err(fuel_ir::Error::Msg("RopePlan: needs at least one base".to_string()).bt());
        }
        if let Some((li, v)) = per_layer
            .iter()
            .enumerate()
            .find(|(_, v)| **v >= bases.len())
        {
            return Err(fuel_ir::Error::Msg(format!(
                "RopePlan: layer {li} selects RoPE variant {v} but only {} base(s) \
                 are defined",
                bases.len(),
            ))
            .bt());
        }
        Ok(Self { bases, per_layer })
    }

    /// **Two-base per-layer selection, TOTAL by construction** — the sibling of
    /// [`MaskPlan::per_layer_window`], and what Gemma3 uses: `base_when_true` is
    /// its `rope_local_base_freq` (sliding layers), `base_when_false` its
    /// `rope_theta` (full-causal layers).
    ///
    /// Infallible for the same reason: a `bool` selector over exactly two
    /// defined bases cannot index out of range, so `decode_rope_plan` needs no
    /// `Result` and no `expect`.
    ///
    /// ⚠️ **Collapses to a single base when the two are EQUAL** — which is not a
    /// micro-optimisation but a correctness-of-measurement matter: Gemma3's
    /// shipped fixtures set `rope_local_base_freq == rope_theta`, and a plan
    /// that still reported 2 variants there would make the graph *look*
    /// dual-base while every layer read identical bytes. Collapsing makes the
    /// degeneracy visible in `n_variants()`, which is what the non-vacuity
    /// assertions in Gemma3's tests key on.
    pub fn per_layer_base(
        n_layers: usize,
        base_when_true: f64,
        base_when_false: f64,
        uses_first: impl Fn(usize) -> bool,
    ) -> Self {
        if base_when_true == base_when_false {
            return Self::single(base_when_true, n_layers);
        }
        let per_layer: Vec<usize> = (0..n_layers).map(|i| usize::from(!uses_first(i))).collect();
        match (per_layer.contains(&0), per_layer.contains(&1)) {
            (true, false) => Self::single(base_when_true, n_layers),
            (false, true) => Self::single(base_when_false, n_layers),
            _ => Self {
                bases: vec![base_when_true, base_when_false],
                per_layer,
            },
        }
    }

    pub fn n_variants(&self) -> usize {
        self.bases.len()
    }

    pub fn n_layers(&self) -> usize {
        self.per_layer.len()
    }

    pub fn variant_for_layer(&self, layer_idx: usize) -> usize {
        self.per_layer[layer_idx]
    }

    /// Mix the plan's **structural** content into a decode shape key: the
    /// variant count and per-layer assignment wire the graph. The base *values*
    /// are excluded for the same reason `rope_base` always was — they are data,
    /// recomputed and rebound every token, so baking them would forfeit plan
    /// reuse across a change already handled correctly.
    pub fn mix_into(&self, h: &mut crate::decode_shape::ShapeKeyHasher) {
        h.mix_u64(self.bases.len() as u64);
        for v in &self.per_layer {
            h.mix_u64(*v as u64);
        }
    }
}

/// Build the stacked RoPE cos/sin tables, variants concatenated along a leading
/// axis. Single source of table bytes for both the build path and every rebind.
fn build_rope_variants(
    plan: &RopePlan,
    width: usize,
    cached_len: usize,
    seq: usize,
    rope_inv_freq: Option<&[f64]>,
) -> (Vec<f32>, Vec<f32>) {
    // A caller-supplied inverse-frequency vector (scaled RoPE) replaces the
    // base entirely, so it applies to every variant — a scaled multi-base model
    // would need per-variant inv_freq, and no such model exists yet. Single-base
    // is the only shape that reaches here with `Some`.
    if let Some(inv) = rope_inv_freq {
        return fuel_graph::build_rope_tables_with_inv_freq(inv, cached_len, seq, width);
    }
    let mut cos = Vec::with_capacity(plan.n_variants() * seq * width);
    let mut sin = Vec::with_capacity(plan.n_variants() * seq * width);
    for base in &plan.bases {
        let (c, s) = fuel_graph::build_rope_tables(*base, cached_len, seq, width);
        cos.extend_from_slice(&c);
        sin.extend_from_slice(&s);
    }
    (cos, sin)
}

/// Build the `[n_variants, 1, seq, max_seq_len]` decode mask, variants
/// concatenated along the leading axis.
///
/// This is the single source of mask bytes for **both** halves of decode: the
/// build path bakes/binds it here, and each family's per-token rebind hook
/// recomputes it through this same function. Two mask formulas — one for the
/// held graph and one for the rebind — is precisely the divergence that would
/// go unnoticed until a windowed family decoded its second token.
pub(crate) fn build_decode_mask_variants(
    plan: &MaskPlan,
    cached_len: usize,
    seq: usize,
    max_seq_len: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(plan.n_variants() * seq * max_seq_len);
    for window in &plan.windows {
        match window {
            None => out.extend_from_slice(&crate::lazy::build_decode_causal_mask(
                cached_len,
                seq,
                max_seq_len,
            )),
            Some(window) => out.extend_from_slice(&crate::lazy::build_decode_causal_mask_windowed(
                cached_len,
                seq,
                max_seq_len,
                *window,
            )),
        }
    }
    out
}

/// Everything the shared build path needs from one model family.
///
/// Four of increment 3's seven per-family items are served through this trait
/// rather than copied; the two that are genuinely per-family — the attention
/// block and the token-data hook — are the two methods that take real work.
pub trait DecodeBackbone: PersistentDecodeModel {
    /// Family name, threaded into every error this shared path raises.
    ///
    /// **Not cosmetic.** The 23 hunks that made collapsing two build bodies look
    /// hard turned out to be mostly error-message prefixes, so sharing the body
    /// makes diagnostics lie by default: without this, a Qwen3Moe failure
    /// reports `"forward_with_kv_context: …"` and reads as a Llama bug.
    fn decode_family(&self) -> &'static str;

    fn decode_dims(&self) -> DecodeDims;

    /// Identity of the held decode plan — family + the config values that change
    /// graph structure + this weight set. Must fold in
    /// [`MaskPlan::mix_into`].
    fn decode_shape_key(&self) -> u64;

    fn decode_mask_plan(&self) -> MaskPlan;

    /// Per-layer RoPE-base assignment. **Required, not defaulted, on purpose:**
    /// a default would be silently correct for the single-base families and
    /// silently WRONG for a multi-base one that forgot to override it. Every
    /// family stating it is the compiler enumerating them.
    fn decode_rope_plan(&self) -> RopePlan;

    /// The `[vocab, hidden]` embedding table the token lookup reads.
    fn decode_token_embedding(&self) -> Arc<[f32]>;

    /// One transformer layer: attention (writing this step's K/V into the
    /// cache buffers) plus the FFN. **This is the per-family architecture** —
    /// partial rotary, GQA replication, norm placement, bias presence, MoE
    /// routing all live here and nowhere else on this path.
    ///
    /// `layer_idx` is passed because per-layer variation is real (mask variant
    /// selection is resolved by the caller, but Gemma3's RoPE base and LFM2's
    /// state kind are not).
    fn decode_apply_layer(
        &self,
        layer_idx: usize,
        inputs: &DecodeLayerInputs<'_>,
    ) -> Result<Tensor>;

    /// Final norm + LM head → `[batch, seq, vocab]`.
    fn decode_final_norm_and_head(&self, h: &Tensor) -> Result<Tensor>;
}

/// The per-layer arguments of [`DecodeBackbone::decode_apply_layer`], bundled
/// so the signature stays readable as families are added.
pub struct DecodeLayerInputs<'a> {
    pub x: &'a Tensor,
    pub k_cache: &'a Tensor,
    pub v_cache: &'a Tensor,
    pub cached_len_sym: fuel_ir::SymId,
    pub attended_len_sym: fuel_ir::SymId,
    /// `Some` on the device-offset (`Op::WriteSliceDoff`) path, `None` on the
    /// backend-generic `SymEnv` path. The layer must honour both — see the
    /// divergence note at the top of this module.
    pub offset: Option<&'a Tensor>,
    pub rope_cos: &'a Tensor,
    pub rope_sin: &'a Tensor,
    /// **This layer's** mask variant, already sliced out of the stacked Const.
    pub mask: &'a Tensor,
    /// **This layer's** window width, or `None` if it attends densely — the
    /// same plan entry `mask` was built from ([`MaskPlan::window_for_layer`]).
    ///
    /// Carried so a layer offering the CUDA flash-decode arm can state the
    /// truth about its own key range instead of asserting `None`. The arm's
    /// `flash_decoding` kernel does not implement local attention, so a
    /// windowed layer's offer is *declined* — which is correct and is the
    /// point. Asserting `None` there would make the arm attend the whole
    /// prefix and silently drop the window (GAP-194).
    pub attn_window: Option<usize>,
}

/// Per-token host data, before it becomes either a baked Const or an upload.
pub(crate) struct DecodeTokenHost {
    pub token_ids: Vec<u32>,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    /// `[n_variants, 1, seq, max_seq_len]` as f32; converted to the cache dtype
    /// at the Const/upload boundary.
    pub mask: Vec<f32>,
}

/// Compute one decode step's host-side data — the RoPE tables and the stacked
/// mask. Pure host math, no graph and no upload, shared by the build path (which
/// bakes or uploads it) and every family's rebind hook (which uploads it).
pub(crate) fn compute_decode_token_host<M: DecodeBackbone + ?Sized>(
    model: &M,
    cached_len: usize,
    tokens: &[u32],
    max_seq_len: usize,
    rope_inv_freq: Option<&[f64]>,
) -> DecodeTokenHost {
    let dims = model.decode_dims();
    let seq = tokens.len();
    let (rope_cos, rope_sin) = build_rope_variants(
        &model.decode_rope_plan(),
        dims.rope_width,
        cached_len,
        seq,
        rope_inv_freq,
    );
    DecodeTokenHost {
        token_ids: tokens.to_vec(),
        rope_cos,
        rope_sin,
        mask: build_decode_mask_variants(&model.decode_mask_plan(), cached_len, seq, max_seq_len),
    }
}

/// The graph [`build_decode_graph`] produced, plus the handles each tail needs.
pub(crate) struct BuiltDecodeGraph {
    logits_root: Tensor,
    kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)>,
    /// `Some` under [`DataConsts::Rebindable`] only — D1 has no re-bindable
    /// node to hold.
    rebindable: Option<RebindableNodes>,
    cached_len_sym: fuel_ir::SymId,
    attended_len_sym: fuel_ir::SymId,
}

/// The stable NodeIds + first-token Arcs a [`DecodeSession`] is built from.
pub(crate) struct RebindableNodes {
    token_ids_node: fuel_graph::NodeId,
    rope_cos_node: fuel_graph::NodeId,
    rope_sin_node: fuel_graph::NodeId,
    mask_node: fuel_graph::NodeId,
    offset_node: Option<fuel_graph::NodeId>,
    first_token: DecodeTokenData,
}

/// Build one decode-step graph: embed → RoPE tables → stacked mask → per-layer
/// KV-write attention → final norm → logits.
///
/// The only thing `consts` changes inside this body is how the four per-token
/// data Consts are minted. Everything else — including the KV placeholder
/// binding, which both paths do identically — is shared.
#[allow(clippy::too_many_arguments)]
fn build_decode_graph<M: DecodeBackbone + ?Sized>(
    model: &M,
    consts: DataConsts,
    tokens: &[u32],
    cache: &KvCache,
    ctx: &mut InferenceContext,
    return_all_positions: bool,
    rope_inv_freq: Option<&[f64]>,
) -> Result<BuiltDecodeGraph> {
    let dims = model.decode_dims();
    let plan = model.decode_mask_plan();
    let family = model.decode_family();
    let entry = consts.entry();
    let seq = tokens.len();
    let batch = 1;
    let cached_len = cache.cached_len;

    if seq == 0 {
        return Err(fuel_ir::Error::Msg(format!("{family}::{entry}: zero tokens",)).bt());
    }
    if cache.n_layers() != dims.n_layers {
        return Err(fuel_ir::Error::Msg(format!(
            "{family}::{entry}: cache n_layers {} != model n_layers {}",
            cache.n_layers(),
            dims.n_layers,
        ))
        .bt());
    }
    if plan.n_layers() != dims.n_layers {
        return Err(fuel_ir::Error::Msg(format!(
            "{family}::{entry}: mask plan covers {} layer(s) but the model has {}",
            plan.n_layers(),
            dims.n_layers,
        ))
        .bt());
    }
    let max_seq_len = cache.max_seq_len.ok_or_else(|| {
        fuel_ir::Error::Msg(format!(
            "{family}::{entry}: cache was constructed via with_dims (no \
             pre-allocated buffers); call KvCache::with_capacity(...) for the \
             WriteSlice path",
        ))
        .bt()
    })?;
    if cached_len + seq > max_seq_len {
        return Err(fuel_ir::Error::Msg(format!(
            "{family}::{entry}: cached_len ({cached_len}) + seq ({seq}) > \
             max_seq_len ({max_seq_len})",
        ))
        .bt());
    }
    let cache_dtype = cache.dtype.unwrap_or(DType::F32);
    // Behaviour preserved deliberately: the pre-GAP-029 D2 path did NOT check
    // cache-vs-config KV geometry and D1 did. Making D2 stricter here would be
    // a semantic change smuggled into a behaviour-preserving refactor, which
    // blunts the only oracle this refactor has (the pre-existing suites plus
    // the captured golden). Tracked as a follow-up, not fixed in passing.
    if consts == DataConsts::Baked
        && (cache.n_kv_heads != dims.n_kv_heads || cache.head_dim != dims.head_dim)
    {
        return Err(fuel_ir::Error::Msg(format!(
            "{family}::{entry}: cache shape (n_kv_heads={}, head_dim={}) \
             disagrees with model config (n_kv_heads={}, head_dim={})",
            cache.n_kv_heads, cache.head_dim, dims.n_kv_heads, dims.head_dim,
        ))
        .bt());
    }

    let host = compute_decode_token_host(model, cached_len, tokens, max_seq_len, rope_inv_freq);

    // ---- Embed lookup + reshape to [batch, seq, hidden] ----
    let embed = Tensor::from_f32(
        model.decode_token_embedding(),
        Shape::from_dims(&[dims.vocab, dims.hidden]),
        &Device::cpu(),
    );
    let token_ids = match consts {
        DataConsts::Baked => embed.const_u32_like(host.token_ids.clone(), Shape::from_dims(&[seq])),
        DataConsts::Rebindable => {
            embed.const_placeholder_like(Shape::from_dims(&[seq]), DType::U32)
        }
    };
    let token_ids_node = token_ids.inner.id();
    let mut h = embed
        .index_select(0, &token_ids)?
        .reshape(Shape::from_dims(&[batch, seq, dims.hidden]))?;
    // Embedding scale (Gemma family: sqrt(hidden_size)) — applied HERE, before
    // the dtype cast, because prefill scales in f32 and scaling after the cast
    // would round in bf16 on a bf16 cache. `None` emits no node at all.
    if let Some(scale) = dims.embed_scale {
        h = h.mul_scalar(scale);
    }
    // BF16-throughout decode (Phase D increment A): the embedding table stays
    // f32 (CUDA IndexSelect has no BF16 key), but every activation downstream
    // tracks the cache dtype. No-op for f32 caches.
    h = h.to_dtype(cache_dtype)?;

    // ---- RoPE cos/sin tables ----
    // Single-variant families mint `[seq, rope_width]` exactly as before — no
    // stack, no slice, no reshape — so their graphs are unchanged.
    let rope_plan = model.decode_rope_plan();
    if rope_plan.n_layers() != dims.n_layers {
        return Err(fuel_ir::Error::Msg(format!(
            "{family}::{entry}: RoPE plan covers {} layer(s) but the model has {}",
            rope_plan.n_layers(),
            dims.n_layers,
        ))
        .bt());
    }
    // A caller-supplied inv_freq replaces the base for every variant, so a
    // multi-base family cannot express per-variant scaling. Refused rather than
    // silently applying one variant's frequencies to all of them.
    if rope_inv_freq.is_some() && rope_plan.n_variants() > 1 {
        return Err(fuel_ir::Error::Msg(format!(
            "{family}::{entry}: rope_inv_freq override is single-base only, but this \
             model declares {} RoPE variants; per-variant scaled frequencies are not \
             expressible yet",
            rope_plan.n_variants(),
        ))
        .bt());
    }
    let n_rope = rope_plan.n_variants();
    let rope_shape = if n_rope == 1 {
        Shape::from_dims(&[seq, dims.rope_width])
    } else {
        Shape::from_dims(&[n_rope, seq, dims.rope_width])
    };
    let (rope_cos, rope_sin) = match consts {
        DataConsts::Baked => (
            h.const_f32_like(host.rope_cos.clone(), rope_shape.clone()),
            h.const_f32_like(host.rope_sin.clone(), rope_shape),
        ),
        DataConsts::Rebindable => (
            h.const_placeholder_like(rope_shape.clone(), DType::F32),
            h.const_placeholder_like(rope_shape, DType::F32),
        ),
    };
    let rope_cos_node = rope_cos.inner.id();
    let rope_sin_node = rope_sin.inner.id();
    // Hoist the per-variant views ONCE, before the layer loop. The rank-2 table
    // needs a reshape after the slice; the uniform case takes the Const itself
    // and emits neither.
    let rope_2d = Shape::from_dims(&[seq, dims.rope_width]);
    let (rope_cos_v, rope_sin_v): (Vec<Tensor>, Vec<Tensor>) = if n_rope == 1 {
        (vec![rope_cos], vec![rope_sin])
    } else {
        let mut cs = Vec::with_capacity(n_rope);
        let mut ss = Vec::with_capacity(n_rope);
        for v in 0..n_rope {
            cs.push(rope_cos.slice(0, v, 1)?.reshape(rope_2d.clone())?);
            ss.push(rope_sin.slice(0, v, 1)?.reshape(rope_2d.clone())?);
        }
        (cs, ss)
    };

    // ---- Stacked causal mask: [n_variants, 1, seq, max_seq_len] ----
    // Dtype tracks the activation dtype it is broadcast-added onto.
    let n_variants = plan.n_variants();
    let mask_shape = Shape::from_dims(&[n_variants, 1, seq, max_seq_len]);
    let mask = match consts {
        DataConsts::Baked => h.const_like_dtype(&host.mask, mask_shape, cache_dtype)?,
        DataConsts::Rebindable => h.const_placeholder_like(mask_shape, cache_dtype),
    };
    let mask_node = mask.inner.id();
    // Hoist the per-variant width-1 slices ONCE, before the layer loop. The
    // uniform case takes the Const itself: no slice node is emitted at all, so
    // a non-varying family's graph is exactly what it was pre-GAP-029.
    let masks: Vec<Tensor> = if n_variants == 1 {
        vec![mask]
    } else {
        (0..n_variants)
            .map(|v| mask.slice(0, v, 1))
            .collect::<std::result::Result<Vec<_>, _>>()?
    };

    // ---- KV-write offset carrier ----
    // D1 keeps the backend-generic SymEnv `Op::WriteSlice` offset (no capture to
    // serve). D2 uses the device-resident `Op::WriteSliceDoff` offset where the
    // binding exists (CPU/CUDA) so the step is CUDA-graph-capturable, and falls
    // back to SymEnv on Vulkan. The two produce bit-identical KV writes.
    let use_device_offset =
        consts == DataConsts::Rebindable && (ctx.device().is_cpu() || ctx.device().is_cuda());
    let offset_tensor = if use_device_offset {
        Some(h.const_placeholder_like(Shape::from_dims(&[]), DType::I64))
    } else {
        None
    };
    let offset_node = offset_tensor.as_ref().map(|t| t.inner.id());

    let cached_len_sym = fuel_ir::SymId(0);
    // The live attended-prefix length (`cached_len + seq`) — the CUDA flash
    // decode arm's `k_len`. Bound alongside `cached_len_sym` every pass; inert
    // on an f32 decode graph, load-bearing where the flash arm is offered.
    let attended_len_sym = fuel_ir::SymId(1);

    // ---- Per-layer: bind the cache K + V Arcs, run the family's block ----
    let cache_shape = Shape::from_dims(&[batch, dims.n_kv_heads, max_seq_len, dims.head_dim]);
    let mut kv_nodes: Vec<(fuel_graph::NodeId, fuel_graph::NodeId)> =
        Vec::with_capacity(dims.n_layers);
    for li in 0..dims.n_layers {
        let k_arc = cache.slot_storage(li, KvSlot::K).ok_or_else(|| {
            fuel_ir::Error::Msg(format!(
                "{family}::{entry}: cache layer {li} has no K slot \
                 (with_capacity should have populated all layers)",
            ))
            .bt()
        })?;
        let v_arc = cache.slot_storage(li, KvSlot::V).ok_or_else(|| {
            fuel_ir::Error::Msg(format!("{family}::{entry}: cache layer {li} has no V slot",)).bt()
        })?;
        let k_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
        let v_cache_node = h.const_placeholder_like(cache_shape.clone(), cache_dtype);
        let k_id = k_cache_node.inner.id();
        let v_id = v_cache_node.inner.id();
        ctx.insert(k_id, k_arc);
        ctx.insert(v_id, v_arc);
        kv_nodes.push((k_id, v_id));

        h = model.decode_apply_layer(
            li,
            &DecodeLayerInputs {
                x: &h,
                k_cache: &k_cache_node,
                v_cache: &v_cache_node,
                cached_len_sym,
                attended_len_sym,
                offset: offset_tensor.as_ref(),
                rope_cos: &rope_cos_v[rope_plan.variant_for_layer(li)],
                rope_sin: &rope_sin_v[rope_plan.variant_for_layer(li)],
                mask: &masks[plan.variant_for_layer(li)],
                attn_window: plan.window_for_layer(li),
            },
        )?;
    }

    // ---- Final norm + LM head ----
    let logits = model.decode_final_norm_and_head(&h)?;
    let logits_root = if return_all_positions {
        logits.reshape(Shape::from_dims(&[seq * dims.vocab]))?
    } else {
        let last_pos = seq - 1;
        logits
            .slice(1, last_pos, 1)?
            .reshape(Shape::from_dims(&[dims.vocab]))?
    };
    // `realize_one_as_with_env::<f32>` reinterprets the root's raw bytes as
    // `[f32]`; a BF16 root would be UB (half the byte width, silently wrong
    // data). No-op under f32 caches.
    let logits_root = logits_root.to_dtype(DType::F32)?;

    let rebindable = match consts {
        DataConsts::Baked => None,
        DataConsts::Rebindable => Some(RebindableNodes {
            token_ids_node,
            rope_cos_node,
            rope_sin_node,
            mask_node,
            offset_node,
            first_token: upload_decode_token_data(
                ctx.device(),
                &host,
                cache_dtype,
                use_device_offset.then_some(cached_len),
            )?,
        }),
    };

    Ok(BuiltDecodeGraph {
        logits_root,
        kv_nodes,
        rebindable,
        cached_len_sym,
        attended_len_sym,
    })
}

/// Upload one step's host data to device-resident Arcs — the same upload path
/// `KvCache::with_capacity` uses. On CPU the Storage wraps the host bytes; on
/// GPU it performs the (tiny) H2D copy.
pub(crate) fn upload_decode_token_data(
    device: &Device,
    host: &DecodeTokenHost,
    cache_dtype: DType,
    device_offset: Option<usize>,
) -> Result<DecodeTokenData> {
    let upload = crate::pipelined_bridge::upload_host_buffer_to_device;
    let mask = match cache_dtype {
        DType::F32 => fuel_ir::HostBuffer::F32(host.mask.clone()),
        DType::BF16 => {
            fuel_ir::HostBuffer::BF16(host.mask.iter().map(|&v| half::bf16::from_f32(v)).collect())
        }
        other => {
            return Err(fuel_ir::Error::Msg(format!(
                "decode token data: unsupported cache dtype {other:?} (expected F32 or BF16)",
            ))
            .bt());
        }
    };
    Ok(DecodeTokenData {
        token_ids: upload(device, fuel_ir::HostBuffer::U32(host.token_ids.clone()))?,
        rope_cos: upload(device, fuel_ir::HostBuffer::F32(host.rope_cos.clone()))?,
        rope_sin: upload(device, fuel_ir::HostBuffer::F32(host.rope_sin.clone()))?,
        mask: upload(device, mask)?,
        offset: device_offset
            .map(|o| upload(device, fuel_ir::HostBuffer::I64(vec![o as i64])))
            .transpose()?,
    })
}

/// **D1** — build the decode graph fresh, realize it, discard it.
///
/// This is the primitive the persistent path itself falls back to (`seq != 1`,
/// invalidation, `TopologyChanged`), so its rebuild contract is deliberate.
pub fn forward_with_kv_context<M: DecodeBackbone + ?Sized>(
    model: &M,
    tokens: &[u32],
    cache: &mut KvCache,
    ctx: &mut InferenceContext,
    return_all_positions: bool,
    rope_inv_freq: Option<&[f64]>,
) -> Result<Vec<f32>> {
    let seq = tokens.len();
    let built = build_decode_graph(
        model,
        DataConsts::Baked,
        tokens,
        &*cache,
        ctx,
        return_all_positions,
        rope_inv_freq,
    )?;
    let cached_len = cache.cached_len;

    // Planner Stage 4a: populate the plan store for this graph before realizing,
    // so realize's planning half HITs the store. Advisory by design — a warm
    // failure is discarded because the realize below runs the identical planning
    // path and surfaces any genuine error with full realize context.
    let _ = crate::planner::Planner::warm(
        built.logits_root.inner.graph(),
        &[built.logits_root.inner.id()],
        ctx.device(),
    );

    let mut sym_env = fuel_ir::SymEnv::new();
    sym_env.bind(built.cached_len_sym, cached_len)?;
    sym_env.bind(built.attended_len_sym, cached_len + seq)?;
    let logits_vec = ctx.realize_one_as_with_env::<f32>(
        built.logits_root.inner.graph(),
        built.logits_root.inner.id(),
        &sym_env,
    )?;

    // Per-step bindings reference a graph that dies with this call; leaving them
    // in ctx.persistent would leak across decode steps.
    for (k, v) in &built.kv_nodes {
        ctx.remove(*k);
        ctx.remove(*v);
    }

    advance_cache(cache, seq, model.decode_n_layers());
    Ok(logits_vec)
}

/// **D2 build** — mint the held graph with re-bindable data Consts, optimize it
/// ONCE, populate `session`, and return the first token's logits.
pub fn build_and_realize_first_decode_token<M: DecodeBackbone + ?Sized>(
    model: &M,
    tokens: &[u32],
    cache: &mut KvCache,
    ctx: &mut InferenceContext,
    session: &mut Option<DecodeSession>,
    rope_inv_freq: Option<&[f64]>,
) -> Result<Vec<f32>> {
    let seq = tokens.len();
    let cached_len = cache.cached_len;
    let cache_dtype = cache.dtype.unwrap_or(DType::F32);
    let dims = model.decode_dims();
    let built = build_decode_graph(
        model,
        DataConsts::Rebindable,
        tokens,
        &*cache,
        ctx,
        false,
        rope_inv_freq,
    )?;
    let max_seq_len = cache.max_seq_len.expect("checked in build_decode_graph");
    let nodes = built
        .rebindable
        .expect("Rebindable mode yields rebindable nodes");
    let logits_node = built.logits_root.inner.id();
    let graph = built.logits_root.inner.graph().clone();

    // Bind the per-token DATA into ctx so the FIRST realize's const-cache walk
    // resolves them (they are placeholders, absent from graph.storage_map). The
    // KV Arcs went in during the build.
    ctx.insert(
        nodes.token_ids_node,
        Arc::clone(&nodes.first_token.token_ids),
    );
    ctx.insert(nodes.rope_cos_node, Arc::clone(&nodes.first_token.rope_cos));
    ctx.insert(nodes.rope_sin_node, Arc::clone(&nodes.first_token.rope_sin));
    ctx.insert(nodes.mask_node, Arc::clone(&nodes.first_token.mask));
    if let (Some(off_node), Some(off_arc)) = (nodes.offset_node, nodes.first_token.offset.as_ref())
    {
        ctx.insert(off_node, Arc::clone(off_arc));
    }

    let mut sym_env = fuel_ir::SymEnv::new();
    sym_env.bind(built.cached_len_sym, cached_len)?;
    sym_env.bind(built.attended_len_sym, cached_len + seq)?;

    let (effective_target, optimized, base_cache, logits_vec) =
        ctx.prebuild_optimized_capturing_as_with_env::<f32>(&graph, logits_node, &sym_env)?;

    // The held session owns the graph + base_cache now; drop the transient ctx
    // bindings (they live in base_cache from here on, re-bound per token into a
    // clone of it, not into ctx).
    ctx.remove(nodes.token_ids_node);
    ctx.remove(nodes.rope_cos_node);
    ctx.remove(nodes.rope_sin_node);
    ctx.remove(nodes.mask_node);
    if let Some(off_node) = nodes.offset_node {
        ctx.remove(off_node);
    }
    for (k, v) in &built.kv_nodes {
        ctx.remove(*k);
        ctx.remove(*v);
    }

    *session = Some(DecodeSession::new(
        graph,
        optimized,
        effective_target,
        logits_node,
        nodes.token_ids_node,
        nodes.rope_cos_node,
        nodes.rope_sin_node,
        nodes.mask_node,
        built.kv_nodes,
        nodes.offset_node,
        built.cached_len_sym,
        built.attended_len_sym,
        base_cache,
        seq,
        max_seq_len,
        dims.n_layers,
        cache_dtype,
        model.decode_shape_key(),
        // Which ALLOCATION's KV Arcs are baked into `base_cache`, and where they
        // live — both read from the one source (GAP-028).
        cache,
    ));

    advance_cache(cache, seq, dims.n_layers);
    Ok(logits_vec)
}

/// Drop a held session, releasing its stable node bindings from `ctx`.
pub fn drop_decode_session(session: &mut Option<DecodeSession>, ctx: &mut InferenceContext) {
    if let Some(s) = session.take() {
        ctx.remove(s.token_ids_node());
        ctx.remove(s.rope_cos_node());
        ctx.remove(s.rope_sin_node());
        ctx.remove(s.mask_node());
        for (k, v) in s.kv_nodes() {
            ctx.remove(*k);
            ctx.remove(*v);
        }
    }
}

/// **The persistent decode entry point**, shared by every family on this path.
///
/// Three arms, exactly as the pre-GAP-029 `LlamaModel` sibling had them:
/// `seq != 1` falls back to D1 (prefill / spec-decode verification), a missing
/// or stale session takes the D2 build, and a live session takes the rebind
/// driver — with a `TopologyChanged` dropping the session and serving this one
/// token through D1.
pub fn forward_with_kv_context_persistent<M: DecodeBackbone + ?Sized>(
    model: &M,
    tokens: &[u32],
    cache: &mut KvCache,
    ctx: &mut InferenceContext,
    session: &mut Option<DecodeSession>,
    rope_inv_freq: Option<&[f64]>,
) -> Result<Vec<f32>> {
    let seq = tokens.len();
    let max_seq_len = cache.max_seq_len;
    let cache_dtype = cache.dtype.unwrap_or(DType::F32);
    let n_layers = model.decode_dims().n_layers;

    // A non-`seq == 1` step is shape-distinct from the held decode graph — drop
    // any session and rebuild it on the next decode token.
    if seq != 1 {
        drop_decode_session(session, ctx);
        return forward_with_kv_context(model, tokens, cache, ctx, false, rope_inv_freq);
    }

    crate::lazy::refresh_decode_session(
        session,
        ctx,
        seq,
        max_seq_len,
        cache_dtype,
        n_layers,
        model.decode_shape_key(),
        cache,
        || {},
        drop_decode_session,
    );

    match session.as_ref() {
        None => {
            build_and_realize_first_decode_token(model, tokens, cache, ctx, session, rope_inv_freq)
        }
        Some(_) => {
            match rebind_and_realize_prebuilt(model, tokens, cache, &*ctx, &*session, rope_inv_freq)
            {
                Ok(logits) => Ok(logits),
                Err(e) if matches!(e, crate::Error::TopologyChanged { .. }) => {
                    // Stale cached generation — drop the session and serve this
                    // token through D1; the session rebuilds on the next one.
                    drop_decode_session(session, ctx);
                    forward_with_kv_context(model, tokens, cache, ctx, false, rope_inv_freq)
                }
                Err(e) => Err(e),
            }
        }
    }
}

/// Bump `cached_len` and every layer's K/V version after a successful step.
fn advance_cache(cache: &mut KvCache, seq: usize, n_layers: usize) {
    cache.cached_len += seq;
    for li in 0..n_layers {
        cache.bump_version(li, KvSlot::K);
        cache.bump_version(li, KvSlot::V);
    }
}

// ===========================================================================
// SABOTAGE RECORD — THE SHARED BUILD PATH (GAP-029 increment 3, 2026-08-13)
//
// Two sabotages, because the two risks are different questions and one
// sabotage cannot answer both. Both runs carried `Compiling fuel-core`.
//
// ---------------------------------------------------------------------------
// A. SHARED: swap `rope_cos`/`rope_sin` in `build_decode_graph`'s layer loop.
//    Question: do BOTH families actually route through this body, or does one
//    still reach a private copy the compiler happily kept?
//
//      llama : forward_with_kv_context_prefill_matches_non_cached_forward FAILED
//              llama_decode_logits_unchanged_by_the_shared_build_path      FAILED
//      llama3: scaled_persistent_decode_matches_full_prefix_forward        FAILED
//      qwen2 : qwen2_windowed_decode_matches_per_layer_gated_forward       FAILED
//              qwen2_decode_matches_forward_when_no_layer_is_windowed      FAILED
//              qwen2_windowed_multi_token_prefill_..._matches_forward      FAILED
//      test result: FAILED. 41 passed; 6 failed
//
//    Three carriers red — Llama, Llama3 through its delegation, and Qwen2 — so
//    all three demonstrably execute this code. The 41 that stayed green are what
//    make the 6 mean anything.
//
//    ⚠️ AND THE PASSES ARE THE MORE INTERESTING HALF. Three things this run
//    established that a pass/fail count would have hidden:
//
//    (1) `phi_*` and `lazy_deepseek2::*` ALL STAYED GREEN. That is the designed
//        outcome, not luck: Phi keeps its own build body (it is the outlier, not
//        the template) and DeepSeek2 has its own MLA path. Sabotage A doubles as
//        the negative control for "increment 3 did not quietly conscript Phi".
//
//    (2) `forward_with_kv_context_persistent_plan_once_matches_d1` PASSED under
//        a sabotage that corrupts every logit it looks at — because it compares
//        D2 against D1, and both were sabotaged identically. A relative oracle is
//        STRUCTURALLY BLIND to a defect in shared code. It is a good test of the
//        thing it tests and it cannot certify this refactor.
//
//    (3) `forward_with_kv_context_decode_matches_non_cached_forward` PASSED too,
//        and this one is an ABSOLUTE oracle that should have caught it. Its
//        tolerance is `diff < 5e-3 || rel < 1e-2` — loose enough that swapping
//        RoPE's cos and sin does not move a tiny model past it. That is the same
//        weakness GAP-029 recorded from the other side (a real 7.9e-3 masking
//        divergence sitting under the same bound), demonstrated here on a defect
//        nobody could call marginal.
//
//    (2) and (3) together are why the 1e-6 golden was captured BEFORE the
//    refactor rather than trusting the existing suite: of the decode tests that
//    look like they cover this body, the ones that failed did so at prefill
//    width, and the decode-shaped ones did not fail at all.
//
// ---------------------------------------------------------------------------
// B. PER-FAMILY: drop Qwen2's Q bias in `Qwen2Model::apply_layer_with_kv_writes`.
//    Question: does Qwen2's test suite test QWEN2, or is it testing the shared
//    path under a family-shaped name? With seven families riding one body, a
//    family's test passing "because the driver works" is the live hazard.
//
//      qwen2 : all three decode tests                                     FAILED
//      test result: FAILED. 44 passed; 3 failed
//
//    EXACTLY ONE family red; Llama, Llama3, Phi and DeepSeek2 untouched. A
//    sabotage of a family's own code that reddened everything would have been
//    worth much less.
// ===========================================================================

#[cfg(test)]
mod mask_plan_tests {
    use super::*;

    /// **The "uniform families pay literally nothing" claim, asserted rather
    /// than stated.** Every non-varying family — Llama included — routes its
    /// mask through the stacked builder now. If that were not byte-identical to
    /// the dense builder it replaced, every existing decode carrier would have
    /// silently changed numerics under a refactor advertised as
    /// behaviour-preserving.
    #[test]
    fn mask_variants_match_the_dense_builder_when_uniform() {
        for (cached_len, seq, max_seq_len, n_layers) in
            [(0, 1, 8, 2), (3, 1, 8, 4), (0, 5, 8, 1), (2, 3, 16, 3)]
        {
            let plan = MaskPlan::dense(n_layers);
            assert_eq!(
                plan.n_variants(),
                1,
                "the dense plan must be single-variant"
            );
            assert_eq!(
                build_decode_mask_variants(&plan, cached_len, seq, max_seq_len),
                crate::lazy::build_decode_causal_mask(cached_len, seq, max_seq_len),
                "uniform plan must be byte-identical to the dense builder \
                 (cached_len={cached_len}, seq={seq}, max_seq_len={max_seq_len})",
            );
        }
    }

    /// The stacked layout is what the build path slices, so its size and the
    /// ordering of variants are load-bearing, not incidental.
    #[test]
    fn variants_stack_on_the_leading_axis_in_plan_order() {
        let plan = MaskPlan::new(vec![None, Some(2)], vec![0, 1, 1]).expect("valid plan");
        let (cached_len, seq, max_seq_len) = (3, 1, 8);
        let stacked = build_decode_mask_variants(&plan, cached_len, seq, max_seq_len);
        assert_eq!(stacked.len(), 2 * seq * max_seq_len);
        assert_eq!(
            &stacked[..seq * max_seq_len],
            &crate::lazy::build_decode_causal_mask(cached_len, seq, max_seq_len)[..],
            "variant 0 is the dense mask and must come first",
        );
    }

    /// **Both degenerate splits must collapse to ONE variant**, or a family that
    /// windows every layer (or none) silently pays for a slice node and an
    /// N-wide mask Const it cannot use.
    ///
    /// The `split >= n_layers` arm is the one with no family on it yet and is
    /// tested for that reason: SmolLm3's mask is uniform-windowed
    /// (`sliding_window` with no `max_window_layers` gate), so it will land here
    /// rather than on the two-variant path.
    #[test]
    fn both_degenerate_splits_collapse_to_a_single_variant() {
        let all_dense = MaskPlan::split_window(4, 0, 8);
        assert_eq!(all_dense.n_variants(), 1, "split 0 must be the dense plan");
        assert_eq!(all_dense, MaskPlan::dense(4), "and identical to it");

        for split in [4, 9] {
            let all_windowed = MaskPlan::split_window(4, split, 8);
            assert_eq!(
                all_windowed.n_variants(),
                1,
                "split {split} covers every layer — one windowed variant, no slice",
            );
            for li in 0..4 {
                assert_eq!(all_windowed.variant_for_layer(li), 0);
            }
            // ...and it must be the WINDOWED variant, not silently the dense one.
            //
            // `cached_len` must exceed `window` or the two masks coincide
            // legitimately and this assertion is vacuous — the same trap
            // `window_wider_than_capacity_is_byte_identical_to_the_dense_mask`
            // records. (Written first with `cached_len = 5` against a window of
            // 8, which is exactly that mistake; the assertion caught it.)
            let (cached_len, seq, max_seq_len, window) = (10, 1, 12, 8);
            assert!(
                cached_len > window,
                "non-vacuity: the window must bite here"
            );
            assert_ne!(
                build_decode_mask_variants(
                    &MaskPlan::split_window(4, split, window),
                    cached_len,
                    seq,
                    max_seq_len,
                ),
                crate::lazy::build_decode_causal_mask(cached_len, seq, max_seq_len),
                "collapsing to one variant must not collapse to the DENSE one",
            );
        }

        // A genuinely mixed split keeps both.
        assert_eq!(MaskPlan::split_window(4, 2, 8).n_variants(), 2);
    }

    /// Build-time validation, per the no-`try_*`/validate-early rule: an
    /// out-of-range variant index would otherwise land as a panic deep inside
    /// the layer loop — or, worse, as a silently mis-masked layer.
    #[test]
    fn a_layer_selecting_an_undefined_variant_is_refused_at_construction() {
        let err = MaskPlan::new(vec![None], vec![0, 1]).expect_err("must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("layer 1") && msg.contains("variant 1"),
            "the refusal must name the offending layer and variant, got: {msg}",
        );
        assert!(
            MaskPlan::new(vec![], vec![]).is_err(),
            "a plan needs a variant"
        );
    }
}
