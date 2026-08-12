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

use crate::inference_context::{
    DecodeSession, DecodeTokenData, InferenceContext, KvCache, KvSlot,
};
use crate::{Device, Result};

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
