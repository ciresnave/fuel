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
#[inline]
pub fn num_key_value_heads(explicit: Option<usize>, num_attention_heads: usize) -> usize {
    explicit.unwrap_or(num_attention_heads)
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
        assert_eq!(num_key_value_heads(Some(1), 32), 1);
    }

    #[test]
    fn kv_heads_absent_means_mha() {
        assert_eq!(num_key_value_heads(None, 32), 32);
    }
}
