//! The identity a held decode plan is baked against.
//!
//! A [`DecodeSession`](crate::inference_context::DecodeSession) holds a graph
//! that has already been built **and optimized** for one specific model. Reusing
//! it is what makes persistent decode fast — measured at 223× on CUDA against
//! re-planning per token — and reusing it *wrongly* is silent: a baked graph
//! computing the wrong architecture returns plausible logits at full speed, with
//! nothing to report.
//!
//! Until this module existed the reuse predicate
//! ([`DecodeSession::is_valid_for`](crate::inference_context::DecodeSession::is_valid_for))
//! keyed on `(seq, max_seq_len, n_layers, cache_dtype)` — pure geometry. That was
//! sufficient while one model shape drove the path and each caller happened to
//! hold one session per model. It is not sufficient for either of the two things
//! now in flight:
//!
//! - **Different architectures, identical geometry.** Qwen3 (per-head Q/K
//!   RmsNorm, sliding-window mask) and Llama at the same dims produce the same
//!   geometric key, so a plan built for one would be judged valid for the other.
//! - **Same architecture, different weights.** This one needs no new model
//!   family to be wrong: two `Qwen3Model`s with identical config and different
//!   weights collide *today*, and the held graph carries the first one's baked
//!   weight `Const`s.
//!
//! ## What belongs in the key — and what must NOT
//!
//! The rule is **baked, not bound**. A held graph bakes its op structure and its
//! weight `Const`s; per-token data (`token_ids`, `rope_cos`/`rope_sin`, `mask`,
//! the KV-write offset) is `const_placeholder_like` and rebound every step from
//! the *current* call.
//!
//! So RoPE frequencies are deliberately **absent** from the key, and that is not
//! an oversight. LLaMA-3.1's scaled frequencies reach the graph through the
//! per-token rebind, so a session is already correct across a frequency change.
//! The instinct is "rope differs ⇒ must be in the key", and acting on it costs
//! plan reuse for no correctness gain — a pure performance regression that hides
//! behind a fully green correctness suite. Over-keying is the failure mode with
//! no error message; that is why the tests here assert **both** halves — that a
//! differing key invalidates, *and* that a matching one does not.
//!
//! ## Why identity, not content
//!
//! Weights are mixed in by `Arc` **pointer**, not by hashing their bytes. Two
//! consequences worth stating rather than discovering:
//!
//! - A model reloaded from disk into fresh allocations gets a new key and
//!   rebuilds its plan. Safe, mildly wasteful, and correct-by-default.
//! - Mutating a weight buffer in place under a live session does **not**
//!   invalidate it. Decode does not do that; training would, and a training
//!   consumer must not reuse a decode session across an optimizer step.
//!
//! The ABA hazard that normally makes pointer identity unsound does not apply:
//! a session holds `Arc` clones of the storages it baked, so those allocations
//! cannot be freed and re-issued to a different live model while its key is in
//! use. Two distinct live models cannot share a weight address.

use std::sync::Arc;

/// FNV-1a 64 over a canonical encoding of a model's decode identity.
///
/// Same constants as `fuel_dispatch::fkc::revhash` and
/// `fuel_memory::dlpack_view` — offset basis `0xcbf29ce484222325`, prime
/// `0x100000001b3`. Pinned by known-answer vectors below rather than by
/// cross-checking against those, so a consistent drift in all three cannot pass.
#[derive(Debug, Clone)]
pub struct ShapeKeyHasher(u64);

impl Default for ShapeKeyHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeKeyHasher {
    pub fn new() -> Self {
        ShapeKeyHasher(0xcbf2_9ce4_8422_2325)
    }

    fn mix_bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    /// Mix a structural discriminant — the model family, a hook's presence, a
    /// mask policy. Prefer a stable literal (`"qwen3"`, `"sliding_window"`) over
    /// anything derived from a `Debug` impl, which is free to change.
    pub fn mix_str(&mut self, s: &str) -> &mut Self {
        self.mix_bytes(s.as_bytes());
        // Length-delimit so `mix_str("ab") + mix_str("c")` cannot collide with
        // `mix_str("a") + mix_str("bc")`.
        self.mix_bytes(&(s.len() as u64).to_le_bytes());
        self
    }

    /// Mix a numeric structural parameter (`n_layers`, `head_dim`, a window
    /// size, an enum discriminant).
    pub fn mix_u64(&mut self, v: u64) -> &mut Self {
        self.mix_bytes(&v.to_le_bytes());
        self
    }

    /// Mix a float structural parameter (`rms_norm_eps`). By bit pattern, so
    /// `NaN` and `-0.0` are handled without a comparison.
    pub fn mix_f64(&mut self, v: f64) -> &mut Self {
        self.mix_bytes(&v.to_bits().to_le_bytes());
        self
    }

    /// Mix a weight buffer's **identity** (its allocation address), which is
    /// what distinguishes two models of the same architecture. See the module
    /// docs on why identity rather than content, and why ABA does not apply.
    pub fn mix_weight<T: ?Sized>(&mut self, w: &Arc<T>) -> &mut Self {
        self.mix_u64(Arc::as_ptr(w) as *const u8 as usize as u64)
    }

    /// Mix an optional weight, distinguishing "absent" from "present" so a model
    /// with a hook disabled cannot key the same as one with it enabled.
    pub fn mix_opt_weight<T: ?Sized>(&mut self, w: Option<&Arc<T>>) -> &mut Self {
        match w {
            Some(a) => {
                self.mix_str("some");
                self.mix_weight(a)
            }
            None => self.mix_str("none"),
        }
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the hash to the FNV-1a **spec**, not to Fuel's other two
    /// implementations of it. A cross-check between implementations proves only
    /// that they agree with each other; a refactor changing all of them
    /// consistently would keep them equal while silently invalidating every
    /// key. Vectors are the published FNV-1a 64 test vectors.
    #[test]
    fn hasher_matches_the_fnv1a_spec() {
        // The empty input must be the offset basis — and note this vector alone
        // is NOT sufficient: FNV-1 and FNV-1a agree on empty input, so it
        // cannot catch the classic xor/multiply transposition.
        assert_eq!(ShapeKeyHasher::new().finish(), 0xcbf2_9ce4_8422_2325);

        // Raw byte folds (no length delimiter) against the published vectors.
        let fold = |s: &str| {
            let mut h = ShapeKeyHasher::new();
            h.mix_bytes(s.as_bytes());
            h.finish()
        };
        assert_eq!(fold("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fold("foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn distinct_structure_gives_distinct_keys() {
        let mut a = ShapeKeyHasher::new();
        a.mix_str("llama").mix_u64(32);
        let mut b = ShapeKeyHasher::new();
        b.mix_str("qwen3").mix_u64(32);
        assert_ne!(a.finish(), b.finish(), "model family must discriminate");
    }

    /// The length delimiter is load-bearing: without it, adjacent string mixes
    /// concatenate and `("ab", "c")` collides with `("a", "bc")`.
    #[test]
    fn adjacent_string_mixes_cannot_be_confused() {
        let mut a = ShapeKeyHasher::new();
        a.mix_str("ab").mix_str("c");
        let mut b = ShapeKeyHasher::new();
        b.mix_str("a").mix_str("bc");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn weight_identity_discriminates_same_shaped_models() {
        let w1: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0]);
        let w2: Arc<[f32]> = Arc::from(vec![1.0_f32, 2.0, 3.0]); // equal CONTENT
        let key = |w: &Arc<[f32]>| {
            let mut h = ShapeKeyHasher::new();
            h.mix_str("llama").mix_weight(w);
            h.finish()
        };
        assert_ne!(
            key(&w1),
            key(&w2),
            "two models with identical config and identical weight VALUES still \
             bake different Consts — the key must separate them",
        );
        // …and a clone of the same Arc is the same model, so it must NOT
        // invalidate. Over-keying here would forfeit plan reuse entirely while
        // every correctness test stayed green.
        assert_eq!(key(&w1), key(&Arc::clone(&w1)));
    }

    #[test]
    fn absent_and_present_hooks_differ() {
        let g: Arc<[f32]> = Arc::from(vec![1.0_f32]);
        let mut with = ShapeKeyHasher::new();
        with.mix_opt_weight(Some(&g));
        let mut without = ShapeKeyHasher::new();
        without.mix_opt_weight(None::<&Arc<[f32]>>);
        assert_ne!(
            with.finish(),
            without.finish(),
            "a disabled hook must not key the same as an enabled one",
        );
    }
}
