// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared resolution rules for HuggingFace `config.json` parsing.
//!
//! # Why this module is rules and not a struct
//!
//! The obvious reading of "collapse the hand-rolled config parsers" is that
//! they duplicate a *field list*. Measured, they do not — `serde` would have
//! handled the field list, which is exactly why six other configs in this
//! crate already parse with a four-line `#[derive(Deserialize)]` wrapper and
//! needed nothing else.
//!
//! What those six have in common is that they carry **zero** cross-field
//! defaults. The seven hand-rolled parsers all carry at least one, and
//! `#[serde(default = "...")]` cannot reference a sibling field. That is the
//! structural reason they resisted `serde`, and it is the duplication:
//!
//! ```text
//!   6 occurrences   head_dim            <- hidden_size / num_attention_heads
//!   4 occurrences   num_key_value_heads <- num_attention_heads
//! ```
//!
//! Ten hand-written copies of two rules, across six files. This module is
//! those two rules.
//!
//! # Take-if-present, else derive — and why the order matters
//!
//! Every one of the six parsers that derives `head_dim` reads `"head_dim"`
//! from the JSON *first* and only derives when it is absent. That is not
//! incidental: architectures with grouped-query or multi-query attention, and
//! models with padded head dimensions, ship a `head_dim` that is deliberately
//! **not** `hidden_size / num_attention_heads`. A resolver that always
//! computed the quotient would produce a wrong-but-plausible number with no
//! error — so these functions take the parsed `Option` and fall back, never
//! the reverse.
//!
//! The same holds for `num_key_value_heads`: absent conventionally means MHA,
//! but a config that states `1` means true MQA and must survive intact.

/// Per-head feature width: the value from `config.json` when present,
/// otherwise `hidden_size / num_attention_heads`.
///
/// `explicit` is whatever `"head_dim"` deserialized to — `None` when the key
/// is absent. It is returned unchanged when present, **including when it
/// disagrees with the quotient**, which is the case this exists to protect.
use crate::{Error, Result};

#[inline]
pub fn head_dim(explicit: Option<usize>, hidden_size: usize, num_attention_heads: usize) -> usize {
    explicit.unwrap_or(hidden_size / num_attention_heads)
}

/// Key/value head count: the value from `config.json` when present, otherwise
/// `num_attention_heads` (i.e. absent means multi-head attention, one KV head
/// per query head).
///
/// A config that states `1` is expressing multi-query attention and is
/// returned unchanged.
///
/// # GAP-282: the divisibility precondition is checked HERE
///
/// Grouped-query attention repeats each KV head `num_attention_heads /
/// num_key_value_heads` times. That quotient is integer division, so a config
/// whose head count is NOT a multiple of its kv-head count TRUNCATES: the
/// repeat factor silently covers fewer query heads than exist, and nothing
/// downstream reports it. Seventeen models were exposed to that and none
/// guarded it.
///
/// This is the one site that receives BOTH operands, which is why the check
/// lives here rather than being written out per model. It runs at PARSE time,
/// per the constitution: every check that CAN run at build time MUST.
///
/// ⚠️ It does NOT cover models that do not route through this function.
/// Measured at `4dd37b9a`: of the 18 GQA-capable models with no check, THIRTEEN
/// route here and are covered; FIVE do not -- `gemma4_text`, `gemma4_vision`,
/// `llava`, `qwen2_moe`, `z_image`. GAP-282 stays OPEN for those five, and a
/// green gate here is not coverage of all eighteen.
///
/// ⚠️ That split moved under a measurement already taken: it was 11/7 at
/// `be6d368c`, and #103 added `qwen3_vl_text` and `voxtral` as callers while
/// this change was in flight. The caller set is DATA, and a merge adds members
/// to it with zero conflicts -- so the number is stamped with the ref it was
/// measured at, and re-deriving it is cheap.
#[inline]
pub fn num_key_value_heads(explicit: Option<usize>, num_attention_heads: usize) -> Result<usize> {
    let kv = explicit.unwrap_or(num_attention_heads);
    // `is_multiple_of(0)` is false for any non-zero left operand, so a stated
    // kv count of 0 is rejected here rather than dividing by zero downstream.
    if !num_attention_heads.is_multiple_of(kv) {
        return Err(Error::Msg(format!(
            "num_attention_heads ({num_attention_heads}) must be a multiple of \
             num_key_value_heads ({kv}); a non-dividing kv-head count truncates \
             silently when GQA groups query heads"
        )));
    }
    Ok(kv)
}

#[cfg(test)]
mod tests {
    use super::*;

    // EACH ARM BELOW WAS SABOTAGED INDIVIDUALLY, and each sabotage failed
    // exactly one arm while the other three stayed green. That is the claim
    // worth recording: not "the suite went red once", but that every arm is
    // sensitive to a DISTINCT regression and none is decoration.
    //
    //   arm                                        sabotage that fails it alone
    //   head_dim_prefers_the_explicit_value...     ignore `explicit`, always derive
    //   head_dim_derives_only_when_absent          derive as `hidden_size` (drop /heads)
    //   kv_heads_preserves_true_mqa                ignore `explicit`, always use heads
    //   kv_heads_absent_means_mha                  fall back to 1 instead of heads
    //   kv_heads_must_divide_the_head_count       drop the divisibility check (GAP-282)
    //
    // Done while the harness was warm. An arm whose failure cannot be
    // provoked alone is redundant with another or asserts something the code
    // cannot violate -- either way a finding, and one that is nearly
    // impossible to reconstruct later because nobody remembers what each arm
    // was aimed at.

    #[test]
    fn head_dim_prefers_the_explicit_value_over_the_quotient() {
        // The case the corpus cannot exercise: every in-tree fixture that
        // specifies head_dim happens to specify the quotient (phi-2 ships
        // 80 against 2560/32 = 80), so a resolver that ignored the explicit
        // value would pass every existing test. This is that discrimination.
        assert_eq!(
            head_dim(Some(96), 2560, 32),
            96,
            "explicit head_dim must win"
        );
        assert_ne!(head_dim(Some(96), 2560, 32), 2560 / 32);
    }

    #[test]
    fn head_dim_derives_only_when_absent() {
        assert_eq!(head_dim(None, 1024, 16), 64);
    }

    #[test]
    fn kv_heads_preserves_true_mqa() {
        // 1 is a statement, not a missing value.
        assert_eq!(num_key_value_heads(Some(1), 32).unwrap(), 1);
    }

    #[test]
    fn kv_heads_absent_means_mha() {
        assert_eq!(num_key_value_heads(None, 32).unwrap(), 32);
    }

    /// GAP-282: a kv-head count that does not DIVIDE the head count truncates
    /// silently when GQA groups query heads. It is rejected here, at the one site
    /// that receives both operands.
    ///
    /// Provoked ALONE by dropping the divisibility check: the other four arms stay
    /// green, which is this module's standing requirement for a new arm.
    #[test]
    fn kv_heads_must_divide_the_head_count() {
        // Conforming cases still resolve -- the control, without which "it errored"
        // could come from a function that rejects everything.
        assert_eq!(num_key_value_heads(Some(4), 32).unwrap(), 4);
        assert_eq!(num_key_value_heads(Some(32), 32).unwrap(), 32);

        // 32 % 5 == 2: the repeat factor 32/5 == 6 would cover 30 of 32 query
        // heads and the shortfall is silent.
        let err = num_key_value_heads(Some(5), 32)
            .expect_err("a non-dividing kv-head count must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("32") && msg.contains("5"),
            "the error must name BOTH operands, got: {msg}"
        );

        // A stated 0 would divide by zero downstream; is_multiple_of(0) is false
        // for a non-zero left operand, so it is rejected rather than panicking.
        assert!(num_key_value_heads(Some(0), 32).is_err());
    }
}
