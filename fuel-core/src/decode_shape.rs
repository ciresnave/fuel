// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! ## Why a counter, and not weight pointers
//!
//! Model identity is a [`ModelInstanceId`] — a value from a process-global
//! monotonic counter, minted once per constructed model. It is **never
//! recycled**, so two distinct models cannot share one, ever, under any
//! ownership arrangement.
//!
//! The obvious cheaper thing is to fold the weights' `Arc` addresses, and this
//! module did that for exactly one commit. It is **unsound**, and the way it
//! was unsound is the reason the counter is worth its cost:
//!
//! > Pointer identity is safe only while something pins the allocation. A
//! > `DecodeSession` holds `StorageCache = HashMap<NodeId, Arc<RwLock<Storage>>>`
//! > — it pins the *Storage* it baked, **not** the model's `Arc<[f32]>` weight
//! > buffers, which are different allocations (and on GPU the host buffer is not
//! > retained at all). So: hold a session for model A, drop A, construct B — the
//! > allocator may hand back A's addresses, B's key equals A's, and the stale
//! > plan baked with **A's** weight `Const`s is judged valid for B.
//!
//! That is reachable, not exotic: a driver serving many models with cached
//! sessions is precisely where a session outlives its model.
//!
//! The lesson generalizes past this bug. The pointer scheme was justified by a
//! *proof about lifetimes*, and the proof was true of the design it was written
//! for (folding `base_cache`'s own Arcs) and silently stopped being true when
//! the input moved to model-owned weights — a change made for unrelated and
//! good reasons. **A counter that is never reused needs no proof**, so it cannot
//! be invalidated by a refactor somewhere else. Prefer that to a correct
//! argument whose premises live in another file.
//!
//! One consequence worth stating rather than discovering: a model reloaded from
//! disk gets a fresh id and rebuilds its plan. Safe, mildly wasteful, and
//! correct by default. Mutating a weight buffer in place under a live session
//! does **not** invalidate it — decode does not do that; a training consumer
//! must not reuse a decode session across an optimizer step.

use std::sync::atomic::{AtomicU64, Ordering};

/// A process-unique, never-recycled identity for one constructed model.
///
/// Mint one per model (lazily is fine) and mix it into that model's decode
/// key. Because ids are monotonic and never reused, two distinct models cannot
/// collide regardless of allocation, drop order, or who outlives whom — see the
/// module docs for the pointer-identity scheme this replaced and why it failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelInstanceId(u64);

/// Starts at 1 so a `Default`/zero-initialized field can never be mistaken for
/// a real id.
static NEXT_MODEL_INSTANCE: AtomicU64 = AtomicU64::new(1);

impl ModelInstanceId {
    /// Mint the next id. Monotonic and never recycled; `u64` exhaustion would
    /// need 2^64 model constructions in one process.
    pub fn next() -> Self {
        ModelInstanceId(NEXT_MODEL_INSTANCE.fetch_add(1, Ordering::Relaxed))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// A process-unique, never-recycled identity for one KV **allocation** — the
/// set of storage buffers a held decode plan is welded to.
///
/// ## Why a held plan needs this at all
///
/// [`DecodeSession`](crate::inference_context::DecodeSession) keeps a
/// `base_cache` holding the KV storage `Arc`s bound on the first decode token,
/// and the per-token rebind overwrites *only* the data `Const`s — `token_ids`,
/// RoPE tables, mask, offset. It never rebinds `kv_nodes`. So a held plan does
/// not read the `&mut KvCache` it is handed; it reads **the buffers it was
/// built against**, forever.
///
/// Geometry cannot see that. Two same-shaped caches key identically, so the
/// slot-pooled serving happy path — retire request A, admit B on a fresh cache
/// of the same shape, reuse the plan for speed — silently decodes B over A's
/// KV. Full speed, plausible distribution, nothing to report. That is why this
/// id exists and why it is in the validity key.
///
/// ## The id names the ALLOCATION, not the conversation
///
/// This distinction is the entire design, and getting it backwards breaks one
/// of the two directions that matter:
///
/// - **Re-allocate ⇒ new id.** Constructing a cache, and any operation that
///   replaces storage under it ([`KvCache::clear`] / [`KvCache::set_layer`]),
///   mints a fresh id. A plan welded to the old buffers must be rebuilt.
/// - **Rewind ⇒ SAME id.** [`KvCache::truncate_to`] — speculative decoding's
///   reject path — moves `cached_len` and touches no storage. The plan is still
///   welded to exactly the right buffers, so invalidating there would forfeit
///   plan reuse (223× on CUDA) on every rejected draft batch, for no
///   correctness gain.
///
/// [`KvCache::clear`]: crate::inference_context::KvCache::clear
/// [`KvCache::set_layer`]: crate::inference_context::KvCache::set_layer
/// [`KvCache::truncate_to`]: crate::inference_context::KvCache::truncate_to
///
/// Over-keying is the failure mode with no error message: an "always stale"
/// predicate passes every correctness test while quietly disabling the
/// optimization. The tests assert both halves for that reason.
///
/// Never recycled, for the same reason [`ModelInstanceId`] is: pointer identity
/// would be sound only while something pins the allocation, and a session
/// outliving its cache is exactly the case this guards. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvAllocId(u64);

/// Starts at 1 so a `Default`/zero-initialized field can never be mistaken for
/// a real id.
static NEXT_KV_ALLOC: AtomicU64 = AtomicU64::new(1);

impl KvAllocId {
    /// Mint the next id. Monotonic and never recycled.
    pub fn next() -> Self {
        KvAllocId(NEXT_KV_ALLOC.fetch_add(1, Ordering::Relaxed))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

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
    /// `NaN` and `-0.0` need no comparison.
    ///
    /// Note this gives `+0.0` and `-0.0` different keys. That is deliberate and
    /// conservative — the cost is a redundant plan rebuild in a case that does
    /// not arise, and the alternative (normalizing) is a comparison that has to
    /// stay correct forever. Do not "fix" it.
    pub fn mix_f64(&mut self, v: f64) -> &mut Self {
        self.mix_bytes(&v.to_bits().to_le_bytes());
        self
    }

    /// Mix the model's identity — the component that separates two models of
    /// the *same* architecture, whose baked weight `Const`s differ.
    ///
    /// There is deliberately **no** `mix_weight(&Arc<T>)` here. Folding weight
    /// addresses is the obvious cheaper implementation and it is unsound: a
    /// session does not pin the model's weight allocations, so a dropped model
    /// frees addresses a later model can be handed. The module docs carry the
    /// full sequence. A method that looks like identity and silently isn't is
    /// worse than no method, so it is absent rather than deprecated.
    pub fn mix_instance(&mut self, id: ModelInstanceId) -> &mut Self {
        self.mix_str("instance").mix_u64(id.get())
    }

    /// Mix whether an optional hook is present, so a model with it disabled
    /// cannot key the same as one with it enabled. The obvious implementation —
    /// skipping the mix when absent — has exactly that bug.
    pub fn mix_present(&mut self, present: bool) -> &mut Self {
        self.mix_str(if present { "some" } else { "none" })
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

    /// [`KvAllocId`]'s whole contract, stated as a test on the real carrier:
    /// **re-allocating re-mints, rewinding does not.** Both halves matter and
    /// they fail differently — a missing re-mint is a silent wrong answer
    /// (a held plan reused across a cache swap), while a spurious re-mint is a
    /// silent performance loss (speculative decoding's reject path rebuilding
    /// the plan on every rejected batch) that no correctness test would catch.
    #[test]
    fn alloc_id_tracks_the_allocation_not_the_conversation() {
        use crate::inference_context::KvCache;

        let dev = crate::Device::cpu();
        let mut cache =
            KvCache::with_capacity(2, 2, 4, 8, fuel_ir::DType::F32, &dev).expect("with_capacity");
        let first = cache.alloc_id();

        // A second cache of IDENTICAL geometry is a DIFFERENT allocation.
        let other =
            KvCache::with_capacity(2, 2, 4, 8, fuel_ir::DType::F32, &dev).expect("with_capacity");
        assert_ne!(
            other.alloc_id(),
            first,
            "same-geometry caches must not share an id — that collision IS the bug",
        );

        // Rewind the conversation: storage untouched ⇒ id preserved.
        cache.cached_len = 5;
        cache.truncate_to(2);
        assert_eq!(
            cache.alloc_id(),
            first,
            "truncate_to rewinds cached_len and touches no storage — re-minting \
             here would forfeit plan reuse on every rejected draft batch",
        );

        // Release the storage: id must move.
        cache.clear();
        let after_clear = cache.alloc_id();
        assert_ne!(
            after_clear, first,
            "clear() drops every layer, so a plan welded to them is stale",
        );

        // Replace one layer's storage: id must move again.
        let replacement =
            KvCache::with_capacity(2, 2, 4, 8, fuel_ir::DType::F32, &dev).expect("with_capacity");
        let src = replacement.layer(0).expect("layer 0");
        cache.set_layer(
            0,
            crate::inference_context::KvLayer {
                k: std::sync::Arc::clone(&src.k),
                v: std::sync::Arc::clone(&src.v),
                k_layout: src.k_layout.clone(),
                v_layout: src.v_layout.clone(),
                k_version: 0,
                v_version: 0,
                k_authority: src.k_authority.clone(),
                v_authority: src.v_authority.clone(),
            },
        );
        assert_ne!(
            cache.alloc_id(),
            after_clear,
            "set_layer swaps the Arc a held plan baked in — that is a new allocation",
        );
    }

    /// Two models of the same architecture must key differently — this is the
    /// half that needs no new model family to be wrong, since it fires on two
    /// same-config models with different weights.
    #[test]
    fn distinct_model_instances_key_differently() {
        let a = ModelInstanceId::next();
        let b = ModelInstanceId::next();
        assert_ne!(a, b, "ids must never repeat");
        let key = |id| {
            let mut h = ShapeKeyHasher::new();
            h.mix_str("llama").mix_instance(id);
            h.finish()
        };
        assert_ne!(key(a), key(b));

        // …and the SAME instance must key identically. This is the half that
        // guards plan reuse: "always stale" satisfies every correctness test
        // ever written while silently forfeiting the 223x persistent-decode win.
        assert_eq!(key(a), key(a));
    }

    /// The property the pointer scheme could not offer: ids are never recycled,
    /// so a model constructed after another is dropped cannot inherit its key.
    /// No lifetime reasoning, and nothing for a later refactor to invalidate.
    #[test]
    fn ids_are_not_recycled_across_drops() {
        let first = {
            let id = ModelInstanceId::next();
            id.get()
        };
        let second = ModelInstanceId::next().get();
        assert!(
            second > first,
            "id {second} did not advance past the dropped {first} — a recycled id would let a stale plan be judged valid for a different model",
        );
    }

    #[test]
    fn absent_and_present_hooks_differ() {
        let mut with = ShapeKeyHasher::new();
        with.mix_present(true);
        let mut without = ShapeKeyHasher::new();
        without.mix_present(false);
        assert_ne!(
            with.finish(),
            without.finish(),
            "a disabled hook must not key the same as an enabled one",
        );
    }

    /// The end-to-end property, through the real predicate rather than the
    /// hasher: two models at IDENTICAL geometry must not share a held plan.
    ///
    /// Before the key existed `is_valid_for` compared only
    /// `(seq, max_seq_len, n_layers, cache_dtype)`, so these two — same dims,
    /// same layer count, same dtype, different weights — were judged
    /// interchangeable and one would execute the other's baked weight `Const`s.
    #[test]
    fn same_geometry_different_weights_do_not_share_a_plan() {
        use crate::lazy::{LlamaConfig, LlamaModel};

        let cfg = LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        // Two weight sets. Layer contents are irrelevant to the key — what
        // matters is that each construction mints its own instance id, which is
        // exactly the property under test.
        let weights = || crate::lazy::LlamaWeights {
            instance: ModelInstanceId::next(),
            token_embedding: std::sync::Arc::from(vec![0.0_f32; 16 * 8]),
            layers: Vec::new(),
            final_norm_gain: std::sync::Arc::from(vec![1.0_f32; 8]),
            output: crate::lazy::WeightStorage::F32(std::sync::Arc::from(vec![0.0_f32; 8 * 16])),
        };
        let a = LlamaModel {
            config: cfg.clone(),
            weights: weights(),
        };
        let b = LlamaModel {
            config: cfg.clone(),
            weights: weights(),
        };

        assert_ne!(
            a.decode_shape_key(),
            b.decode_shape_key(),
            "identical geometry, different weights — a held plan for one must not be judged valid for the other",
        );

        // The half that guards the 223x: a model must still match ITSELF, and a
        // clone sharing the same weights must too. An "always stale" key passes
        // every correctness test while silently disabling plan reuse.
        assert_eq!(a.decode_shape_key(), a.decode_shape_key());
        let a_clone = LlamaModel {
            config: cfg,
            weights: a.weights.clone(),
        };
        assert_eq!(
            a.decode_shape_key(),
            a_clone.decode_shape_key(),
            "two models sharing one weight set may share a plan — over-keying here costs reuse for no correctness gain",
        );
    }
}
