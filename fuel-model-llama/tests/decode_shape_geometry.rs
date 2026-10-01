// SPDX-License-Identifier: MIT OR Apache-2.0
//! Moved from `fuel-core/src/decode_shape.rs`'s own test module (fuel-core
//! dissolution): it constructs a real `LlamaModel`, which now lives in this
//! crate. Keeping it in `fuel-core` as a `[dev-dependencies]` consumer would
//! hit Cargo's diamond-dependency limitation (two non-unifying copies of
//! `fuel_core` in one test binary's dependency graph: the one being tested,
//! and the one this crate depends on). Here `fuel_core` is an ordinary single
//! dependency, so the diamond does not arise.

use fuel_core::decode_shape::ModelInstanceId;
use fuel_core::lazy::WeightStorage;
use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};

/// The end-to-end property, through the real predicate rather than the
/// hasher: two models at IDENTICAL geometry must not share a held plan.
///
/// Before the key existed `is_valid_for` compared only
/// `(seq, max_seq_len, n_layers, cache_dtype)`, so these two — same dims,
/// same layer count, same dtype, different weights — were judged
/// interchangeable and one would execute the other's baked weight `Const`s.
#[test]
fn same_geometry_different_weights_do_not_share_a_plan() {
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
    let weights = || LlamaWeights {
        instance: ModelInstanceId::next(),
        token_embedding: std::sync::Arc::from(vec![0.0_f32; 16 * 8]),
        layers: Vec::new(),
        final_norm_gain: std::sync::Arc::from(vec![1.0_f32; 8]),
        output: WeightStorage::F32(std::sync::Arc::from(vec![0.0_f32; 8 * 16])),
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
