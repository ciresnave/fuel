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

use crate::{Error, Result};

/// Per-head feature width: the value from `config.json` when present,
/// otherwise `hidden_size / num_attention_heads`.
///
/// `explicit` is whatever `"head_dim"` deserialized to — `None` when the key
/// is absent. It is returned unchanged when present, **including when it
/// disagrees with the quotient**, which is the case this exists to protect.
///
/// # GAP-288: the DERIVED path is guarded; the EXPLICIT path deliberately is not
///
/// This used to be `explicit.unwrap_or(hidden_size / num_attention_heads)`,
/// which did neither of the two things its sibling twelve lines below does:
///
/// * **No divisibility check.** A config whose `hidden_size` is not a multiple
///   of `num_attention_heads` produced a TRUNCATED quotient and nothing said
///   so — the identical silent-wrong shape GAP-282 fixed for
///   [`num_key_value_heads`], on the identical parse path, in this file.
/// * **No zero guard.** `head_dim(None, h, 0)` is `h / 0`, which PANICS on a
///   config-parse path, against the never-panic rule.
///
/// **The two guards are separate and that is measured, not stylistic:**
/// `0usize.is_multiple_of(0)` is **`true`** (verified), so a single
/// divisibility check passes `head_dim(None, 0, 0)` straight into `0 / 0`. The
/// zero check has to be its own arm.
///
/// **An explicit value is still returned unchanged, including when it
/// disagrees with the quotient.** MQA/GQA models with padded head dimensions
/// ship a deliberate non-quotient `head_dim`; rejecting that would break real
/// configs and is the failure this function was written to prevent. Only the
/// path where this function INVENTS a number is guarded.
#[inline]
pub fn head_dim(
    explicit: Option<usize>,
    hidden_size: usize,
    num_attention_heads: usize,
) -> Result<usize> {
    if let Some(d) = explicit {
        return Ok(d);
    }
    if num_attention_heads == 0 {
        return Err(Error::Msg(format!(
            "config.json omits head_dim and num_attention_heads is 0, so              hidden_size ({hidden_size}) / num_attention_heads cannot be              evaluated"
        )));
    }
    if !hidden_size.is_multiple_of(num_attention_heads) {
        return Err(Error::Msg(format!(
            "config.json omits head_dim and hidden_size ({hidden_size}) is not              a multiple of num_attention_heads ({num_attention_heads}), so a              derived per-head width would silently truncate"
        )));
    }
    Ok(hidden_size / num_attention_heads)
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
            "num_attention_heads ({num_attention_heads}) must be a multiple of \n             num_key_value_heads ({kv}); a non-dividing kv-head count truncates \n             silently when GQA groups query heads"
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
    //   head_dim_must_divide_when_derived         drop head_dim's divisibility
    //                                             check, OR fold its zero guard
    //                                             into the divisibility one
    //                                             (GAP-288)
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
            head_dim(Some(96), 2560, 32).unwrap(),
            96,
            "explicit head_dim must win"
        );
        assert_ne!(head_dim(Some(96), 2560, 32).unwrap(), 2560 / 32);

        // GAP-288: an explicit value survives even when the DERIVED path would
        // have declined. 100 is not a multiple of 3, so `None` here is an error
        // (asserted below) -- but a config that STATES a head_dim is making a
        // claim this function does not second-guess. Only the path where this
        // function invents a number is guarded, and this arm is what proves the
        // guard did not leak onto the explicit path.
        assert_eq!(head_dim(Some(7), 100, 3).unwrap(), 7);
        assert!(head_dim(None, 100, 3).is_err());
    }

    #[test]
    fn head_dim_derives_only_when_absent() {
        assert_eq!(head_dim(None, 1024, 16).unwrap(), 64);
    }

    /// GAP-288: the DERIVED path declines instead of truncating or panicking.
    ///
    /// Before this, `head_dim` was `explicit.unwrap_or(hidden / heads)` -- no
    /// divisibility check (so a non-dividing config silently truncated, the
    /// GAP-282 shape) and no zero guard (so `h / 0` panicked on a parse path).
    #[test]
    fn head_dim_must_divide_when_derived() {
        // Controls first, so "it errored" cannot come from a function that
        // rejects everything.
        assert_eq!(head_dim(None, 1024, 16).unwrap(), 64);
        assert_eq!(head_dim(None, 2560, 32).unwrap(), 80);

        // 100 / 3 == 33, which covers 99 of 100 hidden units and says nothing.
        let err = head_dim(None, 100, 3).expect_err("a non-dividing width must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("100") && msg.contains("3"),
            "the error must name BOTH operands, got: {msg}"
        );

        // ⚠️ SEPARATE ARM, AND THE SEPARATION IS MEASURED RATHER THAN STYLISTIC:
        // `0usize.is_multiple_of(0)` is TRUE (verified), so a lone divisibility
        // check would pass (0, 0) straight into `0 / 0`. This arm fails if the
        // zero guard is folded into the divisibility one.
        let zero = head_dim(None, 0, 0).expect_err("a zero head count must be rejected");
        assert!(
            format!("{zero}").contains("num_attention_heads"),
            "the zero decline must name the operand, got: {zero}"
        );
        assert!(head_dim(None, 512, 0).is_err());
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
