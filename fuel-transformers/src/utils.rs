//! Shared inference-loop utilities.
//!
//! After the eager-`Tensor` retirement, the only remaining helper is the
//! host-side repetition-penalty applied to the realized vocab-sized logit
//! vector. Causal-mask construction, GQA `repeat_kv`, and tensor-shaped
//! `masked_fill` now live inside each model's lazy implementation (or are
//! handled by graph rewrites) and have been removed from this crate.

/// Applies a repetition penalty in-place to a vocab-sized `f32` logit slice.
///
/// For every distinct token id present in `context` that maps to a valid
/// vocab index, the corresponding logit is rescaled following the GPT-style
/// rule:
///
/// - positive logits are divided by `penalty`
/// - negative logits are multiplied by `penalty`
///
/// Logits for tokens not appearing in `context` are left untouched. The
/// caller owns the buffer; this function does not allocate beyond a small
/// `HashSet` used for context deduplication.
pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let vocab_size = logits.len();
    let mut already_seen = std::collections::HashSet::new();
    for &token_id in context {
        if !already_seen.insert(token_id) {
            continue;
        }
        let idx = token_id as usize;
        if idx >= vocab_size {
            continue;
        }
        let l = logits[idx];
        logits[idx] = if l >= 0.0 { l / penalty } else { l * penalty };
    }
}
