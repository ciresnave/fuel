//! Where a dtype token sits in the KISS-CLASSIFY §6.1 closed vocabulary.
//!
//! This is the **KIND** axis. It is a fact about the *standard* and about
//! whether Fuel has a `DType` at all — never about a backend. Sibling to
//! `storage_status` (lane B, landing separately in this crate)'s **WHY** axis (does a decline expire), which is a
//! fact about one backend's storage.
//!
//! # Why these are two types and not one
//!
//! They cannot merge, and the reason is a type-level impossibility rather than
//! a preference: **they operate on different domains.** KIND classifies a
//! `&str` token — its whole purpose is handling strings that are *not* `DType`
//! variants, like a reserved `f8e4m3fnuz` or junk. WHY classifies a `DType`
//! that is already in the enum. A `StorageStatus` literally cannot be computed
//! for `"f8e4m3fnuz"`, because there is no `DType` to pass it.
//!
//! Pair them at a call site as `{ kind, why: Option<StorageStatus> }` when a
//! consumer wants full classification. Never as a merged enum.
//!
//! # Scope — this is NOT §6.1-0001 conformance
//!
//! §6.1-0001 binds a **reader of a `structure_key`**, and Fuel is not one: it
//! derives and emits keys and treats foreign ones as opaque bytes (K1
//! opacity). See [`crate::RESERVED_DTYPE_TOKENS`] for the measurement. This
//! module exists to classify tokens on the surfaces Fuel *does* ingest, and to
//! give the emit path one place to name the seam spelling.

use crate::dtype::{is_reserved_dtype_token, DType, RESERVED_DTYPE_TOKENS};

/// The KISS-CLASSIFY **sk4** closed dtype vocabulary — all 24 tokens, verbatim
/// from `conformance/src/structure_key.rs` `DTYPES: [&str; 24]` at KISS
/// `origin/main` `19c3ad7`.
///
/// The length is part of the contract: [`TokenKind`]'s three in-vocabulary
/// kinds must partition this array **exactly**, which is what makes a new sk4
/// token a failing test rather than a silent `Unknown`.
///
/// Read this from KISS with `git show origin/main:<path>` on a schema bump — a
/// KISS working tree can sit behind its own `origin/main`, and sk3→sk4 already
/// respelled tokens once (`e4m3fnuz` → `f8e4m3fnuz`).
pub const SK4_DTYPE_TOKENS: [&str; 24] = [
    "f16", "bf16", "f32", "f64", "i8", "i16", "u8", "u16", "i32", "i64", "u32", "u64", "bool",
    "f8e4m3fn", "f8e4m3fnuz", "f8e5m2", "f8e5m2fnuz", "f8e8m0", "f8e6m2", "i4", "u4", "b1", "c64",
    "c128",
];

/// sk4 tokens that are **active in the standard** but for which Fuel has no
/// [`DType`] at all.
///
/// Distinct from [`RESERVED_DTYPE_TOKENS`]: these have real computation
/// semantics and Fuel simply has not implemented them, so they are a Fuel
/// omission (GAP-097). Reserved tokens have *no* semantics at this schema
/// version, so implementing them would be non-conformant rather than helpful.
///
/// **Global, not per-backend.** A token lands here because no Fuel `DType`
/// exists — a fact no backend can change. "This backend cannot do a dtype Fuel
/// *does* have" is the other axis entirely (`storage_status` (lane B, landing separately in this crate)).
pub const RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS: &[&str] =
    &["u16", "u64", "bool", "i4", "u4", "b1", "c64", "c128"];

/// The **sk4 seam spelling** of a Fuel [`DType`], or `None` for a dtype outside
/// the closed vocabulary.
///
/// This is NOT [`DType::as_str`]. Fuel's internal token for the OCP FP8 type is
/// `f8e4m3`; the seam spells it **`f8e4m3fn`** (mandatory variant suffix where
/// more than one variant exists — `f8e5m2` stays unsuffixed, since only the
/// `fnuz` layouts deviate from IEEE E5M2). Conflating the two is how a
/// retired or internal spelling reaches a wire format.
///
/// `None` is the block-scoped sub-byte **element** formats, which are outside
/// §6.1 at sk4 (GAP-153). They are declined, never given a guessed token.
///
/// Exhaustive by design: a new `DType` is a compile error here, forcing a
/// decision instead of a silent omission.
pub fn sk4_token(dt: DType) -> Option<&'static str> {
    Some(match dt {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I8 => "i8",
        DType::I16 => "i16",
        DType::U8 => "u8",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F8E4M3 => "f8e4m3fn",
        DType::F8E5M2 => "f8e5m2",
        DType::F8E8M0 => "f8e8m0",
        DType::F8E6M2 => "f8e6m2",
        // Block-scoped sub-byte ELEMENT formats — outside the sk4 closed set.
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 => return None,
    })
}

/// Where a dtype token sits in the sk4 vocabulary.
///
/// Four values, and the first three **partition** [`SK4_DTYPE_TOKENS`] exactly
/// — see `the_three_kinds_partition_the_sk4_vocabulary_exactly`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// In the vocabulary, active, and Fuel has a [`DType`] for it.
    ///
    /// Says nothing about any backend: a `Supported` token can still be
    /// declined by a particular backend's storage. That is the WHY axis.
    Supported,
    /// In the vocabulary and active, but Fuel has no [`DType`] for it.
    RecognizedUnsupported,
    /// In the vocabulary but **reserved** at this schema version: no
    /// computation semantics until a future additive schema event.
    Reserved,
    /// Not in the sk4 vocabulary at all.
    Unknown,
}

/// Classify a dtype token.
///
/// Case-insensitive, deliberately: the FKC contract surface spells dtype
/// tokens uppercase (`F8E5M2`) while the standard's canonical form is
/// lowercase, and one classifier should serve both rather than growing a
/// second list that drifts.
///
/// **Reserved is tested BEFORE the supported set**, matching KISS's own
/// reference implementation: the reserved tokens are a *subset* of the
/// vocabulary, so a membership-first check would swallow them.
pub fn classify_dtype_token(token: &str) -> TokenKind {
    if is_reserved_dtype_token(token) {
        return TokenKind::Reserved;
    }
    if RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS
        .iter()
        .any(|t| token.eq_ignore_ascii_case(t))
    {
        return TokenKind::RecognizedUnsupported;
    }
    if DType::ALL
        .iter()
        .filter_map(|d| sk4_token(*d))
        .any(|t| token.eq_ignore_ascii_case(t))
    {
        return TokenKind::Supported;
    }
    TokenKind::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn supported_set() -> BTreeSet<&'static str> {
        DType::ALL.iter().filter_map(|d| sk4_token(*d)).collect()
    }

    /// **The completeness gate.** The three in-vocabulary kinds partition the
    /// sk4 vocabulary exactly — 14 + 8 + 2 = 24.
    ///
    /// This is what makes the taxonomy checkable rather than merely tidy: a new
    /// sk4 token, or a new Fuel `DType`, breaks the arithmetic and fails HERE
    /// instead of landing silently in `Unknown` — which is the one kind that
    /// absorbs mistakes without complaining.
    ///
    /// Asserts COUNT **and** MEMBERSHIP. Counts alone would pass if two tokens
    /// swapped sets; membership alone gives a worse failure message.
    #[test]
    fn the_three_kinds_partition_the_sk4_vocabulary_exactly() {
        let supported = supported_set();
        let recognized: BTreeSet<&str> =
            RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS.iter().copied().collect();
        let reserved: BTreeSet<&str> = RESERVED_DTYPE_TOKENS.iter().copied().collect();

        assert_eq!(supported.len(), 14, "supported: {supported:?}");
        assert_eq!(recognized.len(), 8, "recognized-unsupported: {recognized:?}");
        assert_eq!(reserved.len(), 2, "reserved: {reserved:?}");
        assert_eq!(
            supported.len() + recognized.len() + reserved.len(),
            SK4_DTYPE_TOKENS.len(),
            "the three kinds must sum to the vocabulary — a mismatch means a \
             token is unclassified or double-counted",
        );

        // Pairwise disjoint: the counts above only add up to a partition if
        // nothing is in two sets.
        assert!(supported.is_disjoint(&recognized));
        assert!(supported.is_disjoint(&reserved));
        assert!(recognized.is_disjoint(&reserved));

        // MEMBERSHIP, not just arithmetic.
        let union: BTreeSet<&str> = supported
            .union(&recognized)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&reserved)
            .copied()
            .collect();
        let vocabulary: BTreeSet<&str> = SK4_DTYPE_TOKENS.iter().copied().collect();
        assert_eq!(
            union, vocabulary,
            "the three kinds must cover the sk4 vocabulary exactly",
        );

        // Non-vacuity: the vocabulary really is the 24 distinct tokens, so a
        // duplicated entry cannot quietly shrink it into agreement.
        assert_eq!(vocabulary.len(), 24, "sk4 tokens must be distinct");
    }

    /// Every Fuel `DType` is accounted for: 14 carry a seam token, 3 are
    /// block-scoped elements outside the vocabulary. Guards the other
    /// direction from the partition test — a new `DType` that returned a token
    /// already in the vocabulary would pass this but break the partition, and
    /// one that returned `None` breaks this.
    #[test]
    fn every_fuel_dtype_either_has_a_seam_token_or_is_outside_the_vocabulary() {
        let with = DType::ALL.iter().filter(|d| sk4_token(**d).is_some()).count();
        let without = DType::ALL.iter().filter(|d| sk4_token(**d).is_none()).count();
        assert_eq!(with, 14);
        assert_eq!(without, 3, "the block-scoped sub-byte element formats");
        assert_eq!(with + without, DType::ALL.len());

        // Every emitted token must actually be IN the vocabulary — otherwise
        // Fuel would be emitting a spelling the standard does not define.
        for dt in DType::ALL {
            if let Some(tok) = sk4_token(*dt) {
                assert!(
                    SK4_DTYPE_TOKENS.contains(&tok),
                    "{dt:?} emits {tok:?}, which is not an sk4 token",
                );
            }
        }
    }

    /// The seam spelling is NOT the internal spelling, and the one case where
    /// they differ is the one that matters.
    #[test]
    fn the_seam_spelling_differs_from_the_internal_one_for_fp8_e4m3() {
        assert_eq!(sk4_token(DType::F8E4M3), Some("f8e4m3fn"));
        assert_eq!(DType::F8E4M3.as_str(), "f8e4m3");
        // ...and does NOT differ for E5M2, because only fnuz deviates there.
        assert_eq!(sk4_token(DType::F8E5M2), Some("f8e5m2"));
        assert_eq!(DType::F8E5M2.as_str(), "f8e5m2");
    }

    /// Classification over all four kinds, including the prefix trap: `f8e5m2`
    /// is a PREFIX of the reserved `f8e5m2fnuz`, so a `starts_with` classifier
    /// would call the reserved token `Supported`.
    #[test]
    fn classify_covers_all_four_kinds_and_resists_the_prefix_trap() {
        assert_eq!(classify_dtype_token("f32"), TokenKind::Supported);
        assert_eq!(classify_dtype_token("f8e4m3fn"), TokenKind::Supported);
        assert_eq!(classify_dtype_token("bool"), TokenKind::RecognizedUnsupported);
        assert_eq!(classify_dtype_token("c128"), TokenKind::RecognizedUnsupported);
        assert_eq!(classify_dtype_token("f8e5m2fnuz"), TokenKind::Reserved);
        assert_eq!(classify_dtype_token("asdf"), TokenKind::Unknown);

        // The trap, both ends.
        assert_eq!(classify_dtype_token("f8e5m2"), TokenKind::Supported);
        assert_eq!(classify_dtype_token("f8e5m2fnuz"), TokenKind::Reserved);

        // Fuel's INTERNAL spelling is not a seam token — it must not classify
        // as Supported, or the two vocabularies have been conflated.
        assert_eq!(classify_dtype_token("f8e4m3"), TokenKind::Unknown);

        // Case-insensitive, so one classifier serves the uppercase FKC surface.
        assert_eq!(classify_dtype_token("F32"), TokenKind::Supported);
        assert_eq!(classify_dtype_token("F8E5M2FNUZ"), TokenKind::Reserved);
    }

    /// Every token in the vocabulary classifies as something other than
    /// `Unknown`, and nothing outside it classifies as anything else.
    #[test]
    fn classification_is_total_over_the_vocabulary() {
        for tok in SK4_DTYPE_TOKENS {
            assert_ne!(
                classify_dtype_token(tok),
                TokenKind::Unknown,
                "{tok} is in the sk4 vocabulary and must not classify as Unknown",
            );
        }
        // Non-vacuity: the classifier CAN return Unknown, so the loop above is
        // asserting something rather than being satisfied by a stuck function.
        assert_eq!(classify_dtype_token("definitely-not-a-dtype"), TokenKind::Unknown);
    }
}
