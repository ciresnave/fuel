// SPDX-License-Identifier: MIT OR Apache-2.0
//! Types for elements that can be stored and manipulated using tensors.
//!
//! The [`DType`] enum represents the element type of a tensor, and the [`WithDType`] trait
//! allows Rust native types to be used with tensors.
#![allow(clippy::redundant_closure_call)]
use crate::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The different types of elements allowed in tensors.
// EXHAUSTIVE-BY-DESIGN: `DType` is a closed set. The consumers that map a dtype
// to a concrete per-variant fact — `size_in_bytes`, `as_str`, the host-buffer
// allocation tables (fuel-cpu-backend), the safetensors/FDX/DLPack tag maps
// (`--features dlpack`), and scalar `zero`/`one`/`from_f64` — match it
// exhaustively, and several are CROSS-CRATE, so a wildcard there cannot produce a
// correct answer: it would fabricate a byte size / tag and silently corrupt.
// Adding a variant is a breaking change: gate it workspace-wide AND under every
// feature that has a consumer (`--features cuda`/`vulkan`/`dlpack`), never just
// `-p fuel-ir`. Do NOT add `#[non_exhaustive]` here — it would force wildcards
// onto those cross-crate exhaustive tables and route a new dtype to the wrong
// buffer. (The `DType::ALL` + `all_variants_witness` pair below is the existing
// compile-time tripwire this marker names. Consumers that *decline* an unknown
// dtype loudly — `Err`/`bail!`/`None` — are fine and expected; only the
// exhaustive concrete-fact maps are load-bearing.) See GAP-049.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer. Added 2026-05-19 for int8 GEMM (W8A8
    /// inference) support — baracuda's `gemm_s8_*` family accepts
    /// `i8` operands and `mma.sync` `.s8.s8.s32` instructions.
    I8,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Brain floating-point using half precision (16 bits).
    BF16,
    /// Floating-point using half precision (16 bits).
    F16,
    /// Floating-point using single precision (32 bits).
    F32,
    /// Floating-point using double precision (64 bits).
    F64,
    /// 8-bit floating point, 4-bit exponent + 3-bit mantissa — specifically
    /// **OCP finite E4M3FN**: bias 7, max-finite ±448, no infinities, single
    /// NaN. This is the only E4M3 format Fuel represents, so the bare name is
    /// unambiguous today (enforced by
    /// `reserved_token_tests::exactly_one_e4m3_family_dtype_or_revisit_gap_169`).
    ///
    /// The identity `F8E4M3 == OCP E4M3FN` is committed and tested at three
    /// independent sites: the structure-key path (`fuel-dispatch`
    /// `telemetry::baracuda_provider::map_element_kind`), the JIT seam
    /// (`fuel-dispatch` `jit_adopt::fp8_maps_to_baracuda_ocp_element_kinds`),
    /// and the external-contract parser (`fuel-dispatch` `fkc::lower`'s
    /// "F8E4M3FNUZ must not lower to F8E4M3" prefix rejection).
    ///
    /// **The name is deliberately NOT `F8E4M3FN`** (GAP-169). A rename would
    /// touch ~110 sites across every backend, cascade through the
    /// `cuda_dtype!` / `ctor!` macros into `CudaStorageSlice` / `PinnedBuffer`,
    /// and break 3 `fuel-metal-backend` sites that cannot be compiled on the
    /// primary dev box — all for reader clarity only. The interop token is
    /// already `f8e4m3fn` on the sk4 path (`token_kind`), and this dtype's own
    /// token stays `f8e4m3` for the checked-in `.fkc.md` contracts that author
    /// it (see the `RESERVED_DTYPE_TOKENS` note below). Revisit only when a
    /// second E4M3-family variant (e.g. `F8E4M3FNUZ`) actually lands — the
    /// witness above fires then.
    F8E4M3,
    /// 8-bit floating point with 5-bit exponent and 2-bit mantissa — the
    /// OCP-standard FP8 pair's other half.
    ///
    /// E4M3 (above) carries more mantissa and is the forward-pass format;
    /// E5M2 trades mantissa for exponent range and is the gradient format.
    /// Shipping one without the other leaves FP8 training unrepresentable,
    /// and `fuel-cuda-backend`'s cast table already documents six baracuda
    /// cast pairs it skips for want of this variant. Distinct from
    /// [`F8E6M2`](Self::F8E6M2), which is a non-OCP block-scale type.
    F8E5M2,
    /// 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format).
    F6E2M3,
    /// 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format).
    F6E3M2,
    /// 4-bit float (MX4 format).
    F4,
    /// 8-bit float with 8 exponent bits and 0 mantissa bits.
    F8E8M0,
    /// Unsigned 8-bit block-scale: 6 exponent bits, 2 mantissa bits, no sign
    /// (0+6+2=8). A finer-granularity sibling of [`F8E8M0`](Self::F8E8M0) —
    /// trades 2 exponent bits for 2 mantissa — so a scale type, not a signed
    /// value dtype. (The "8-bit float, 6 exp, 2 mant" framing it shipped with in
    /// e275538c was a misread: 1+6+2=9 doesn't fit a signed byte.)
    F8E6M2,
    /// Boolean truth value, one byte per element (`0` = false, `1` = true).
    ///
    /// The output dtype of the element-wise comparisons
    /// (`eq`/`ne`/`lt`/`le`/`gt`/`ge`) — GAP-168(c) / CireSnave's 2026-08-12
    /// ruling that Fuel "uses `Bool` where possible and casts to `U8` when
    /// necessary". Storage width equals `u8` (KISS-CLASSIFY `bool`: "1-byte
    /// truth value; storage width equals `u8`"), so a backend may store it in
    /// a `U8` buffer — but it is a DISTINCT logical dtype: neither an integer
    /// nor a floating-point value (`is_int` and `is_float` are both `false`),
    /// so using a mask AS A NUMBER (multiply/sum/accumulate) is rejected at
    /// build time and extracting a number requires an explicit `.cast(U8)`.
    Bool,
}

impl DType {
    /// Every [`DType`] variant, in declaration order.
    ///
    /// Completeness is REMINDED, not compiler-derived (GAP-248). [`all_variants_witness`]
    /// below is a wildcard-free `match`, so adding a `DType` fails to compile THERE — a
    /// variant cannot be added without being dragged to this file. But extending THIS
    /// list in the same edit is then convention: nothing ties `ALL` to the witness, so a
    /// variant added to the match yet forgotten here would still compile. A hand-written
    /// `const` (not a derived iterator) because the anchor is a compile-time tripwire —
    /// the residual gap is that it reminds rather than derives. Measured complete at head
    /// (`ALL` = witness = enum = 18). The enforced version is a single-source macro
    /// declaring the enum and `ALL` together; see GAP-248.
    pub const ALL: &'static [DType] = &[
        DType::U8,
        DType::I8,
        DType::U32,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::F6E2M3,
        DType::F6E3M2,
        DType::F4,
        DType::F8E8M0,
        DType::F8E6M2,
        DType::Bool,
    ];
}

#[cfg(test)]
mod reserved_token_tests {
    use super::*;
    use std::str::FromStr;

    /// A reserved spelling (KISS-CLASSIFY §6.1's reserved vocabulary) must be
    /// recognized and declined **distinguishably from an unknown token**.
    ///
    /// The whole claim is in one assertion pair: both reject, and they reject
    /// DIFFERENTLY — matchably, not merely with different prose. (Scope: this
    /// is the §6.1 vocabulary applied to Fuel's own parser, not §6.1-0001
    /// conformance — see [`RESERVED_DTYPE_TOKENS`].)
    #[test]
    fn reserved_fnuz_declines_distinctly_from_an_unknown_token() {
        for tok in RESERVED_DTYPE_TOKENS {
            let err = DType::from_str(tok).expect_err("a reserved token must not parse");
            assert!(
                err.is_reserved(),
                "{tok} is reserved by §6.1 and must decline AS reserved, not as \
                 unknown — a caller cannot otherwise tell 'wait for a schema bump' \
                 from 'fix your input'",
            );
        }

        // The other side of the distinction. Without this, a deriver that
        // reported EVERYTHING as reserved would pass the loop above.
        let unknown = DType::from_str("asdf").expect_err("garbage must not parse");
        assert!(
            !unknown.is_reserved(),
            "an unknown token must NOT report as reserved"
        );
        assert_eq!(unknown, DTypeParseError::Unknown("asdf".into()));
    }

    /// The FKC contract spelling is uppercase; the reserved list is canonical
    /// lowercase. One list serves both, so the comparison must be
    /// case-insensitive — otherwise the FKC site silently loses the
    /// distinction while this one keeps it.
    #[test]
    fn reserved_lookup_is_case_insensitive_so_one_list_serves_both_sites() {
        assert!(is_reserved_dtype_token("F8E4M3FNUZ"));
        assert!(is_reserved_dtype_token("f8e5m2fnuz"));
        assert!(
            !is_reserved_dtype_token("f8e4m3fn"),
            "the OCP form is NOT reserved"
        );
        assert!(!is_reserved_dtype_token("asdf"));
    }

    /// **The delimiter trap, named so the next reader sees it.** Every
    /// supported token here is a PREFIX of a reserved one: `f8e4m3` prefixes
    /// `f8e4m3fnuz`, `f8e5m2` prefixes `f8e5m2fnuz`. Any `starts_with` /
    /// `contains` / prefix-strip classifier therefore reads the RESERVED token
    /// as the SUPPORTED one — and the caller then computes on `fnuz` bytes as
    /// if they were OCP, which is silent numerical corruption rather than a
    /// loud failure.
    ///
    /// This site is safe *by construction* (`FromStr` is an exact-literal
    /// `match`, `is_reserved_dtype_token` is whole-token `eq_ignore_ascii_case`)
    /// — but "safe by construction" is a property of today's implementation,
    /// so it is asserted rather than assumed. Both directions are checked: a
    /// reserved token must not resolve to its prefix's dtype, and a supported
    /// token must not be swept up as reserved.
    #[test]
    fn a_reserved_token_is_never_confused_with_the_supported_token_it_extends() {
        // Direction 1 — the dangerous one. A prefix matcher would return
        // Ok(F8E4M3) / Ok(F8E5M2) here.
        assert!(
            !matches!(DType::from_str("f8e4m3fnuz"), Ok(DType::F8E4M3)),
            "f8e4m3fnuz must NOT parse as its OCP prefix f8e4m3 — different \
             exponent bias, so the bytes decode to different numbers",
        );
        assert!(
            !matches!(DType::from_str("f8e5m2fnuz"), Ok(DType::F8E5M2)),
            "f8e5m2fnuz must NOT parse as its OCP prefix f8e5m2",
        );

        // Direction 2 — the mirror. A `contains`-style reserved check would
        // sweep the supported tokens in, declining dtypes Fuel really has.
        assert!(!is_reserved_dtype_token("f8e4m3"));
        assert!(!is_reserved_dtype_token("f8e5m2"));

        // Non-vacuity: these ARE genuine prefix pairs, so the assertions above
        // are exercising the trap and not passing on unrelated strings.
        for tok in RESERVED_DTYPE_TOKENS {
            assert!(
                DType::ALL.iter().any(|d| tok.starts_with(d.as_str())),
                "{tok} was expected to EXTEND a supported token; if that is no \
                 longer true this test has stopped testing the prefix trap",
            );
        }
    }

    /// A reserved token must never also be parseable as a real dtype — i.e. the
    /// reserved list and the accepted set must not overlap. Guards the mirror
    /// mistake: adding `fnuz` to `FromStr`'s accept arms later would make the
    /// reserved arm unreachable and silently activate an unimplemented layout.
    #[test]
    fn reserved_tokens_are_disjoint_from_the_accepted_set() {
        for tok in RESERVED_DTYPE_TOKENS {
            assert!(
                DType::from_str(tok).is_err(),
                "{tok} is reserved and must not resolve to a DType",
            );
        }
        // Non-vacuity: the accept path genuinely works, so the loop above is
        // asserting rejection rather than a broken parser rejecting everything.
        assert_eq!(DType::from_str("f8e4m3").unwrap(), DType::F8E4M3);
        assert_eq!(DType::from_str("f8e5m2").unwrap(), DType::F8E5M2);
    }

    /// **GAP-169 deferral expiry witness.** The bare name `F8E4M3` is
    /// unambiguous only while there is exactly ONE E4M3-family dtype — today,
    /// the OCP-finite `F8E4M3`. If a second ever lands (e.g. an AMD
    /// `F8E4M3FNUZ`: different bias, no −0), the bare name becomes ambiguous and
    /// the rename deferred in GAP-169 (see `DType::F8E4M3`'s doc) must be
    /// revisited. This asserts the **premise** (exactly one E4M3 today), not the
    /// conclusion (the name is fine).
    ///
    /// **Completeness is inherited, not assumed:** the population is
    /// [`DType::ALL`], which [`all_variants_witness`] REMINDS the author to keep
    /// exhaustive (a variant cannot be added without a compile error at the witness,
    /// though nothing forces it into `ALL` — GAP-248; measured complete at head), and
    /// the count is derived through
    /// the same `is_e4m3_family` predicate that feeds the assertion — so the
    /// perturbation control (widen the predicate, watch it fire at two) actually
    /// exercises the detection path. Keep the filter on `DType::ALL`: this file
    /// also holds the raw literals `"f8e4m3fn"` / `"f8e4m3fnuz"` (reserved-token
    /// rejection), which are not any variant's [`as_str`](DType::as_str), so
    /// pointing the predicate at raw tokens would inflate the count with no
    /// variant added.
    ///
    /// **Bound:** this detects a sibling *spelled* `f8e4m3*`, not the E4M3
    /// *format* in the abstract — a sibling spelled otherwise (e.g. a bare
    /// `e4m3fnuz`, the pre-respell sk4 form) would slip past. The prefix match
    /// is deliberate and is NOT the delimiter trap: that trap is a guard firing
    /// on a longer string it did not mean to catch, whereas this witness WANTS
    /// every longer `f8e4m3*` spelling — so `starts_with` is the intended
    /// semantics; do not "fix" it into `==`, which silently destroys the
    /// detection. The token it keys on cannot drift: GAP-169 ruled `as_str()`
    /// must stay `f8e4m3` because external `.fkc.md` contracts parse it, so the
    /// deferral protects its own expiry witness.
    #[test]
    fn exactly_one_e4m3_family_dtype_or_revisit_gap_169() {
        fn is_e4m3_family(dt: DType) -> bool {
            // Prefix, not equality: every `f8e4m3*` spelling (e.g. a future
            // `f8e4m3fnuz`) is E4M3-family and must count. See the bound above.
            dt.as_str().starts_with("f8e4m3")
        }
        let e4m3: Vec<DType> = DType::ALL
            .iter()
            .copied()
            .filter(|&dt| is_e4m3_family(dt))
            .collect();
        assert_eq!(
            e4m3.as_slice(),
            [DType::F8E4M3].as_slice(),
            "GAP-169 deferral premise broken: the E4M3 family now has {} members \
             ({e4m3:?}), not one. A sibling E4M3 variant exists, so the bare \
             `F8E4M3` name is now ambiguous — revisit GAP-169 (rename to \
             F8E4M3FN vs the sibling; see DType::F8E4M3's doc).",
            e4m3.len(),
        );
    }

    /// Every variant in [`DType::ALL`] must return a consistent byte count
    /// from [`size_in_bytes`](DType::size_in_bytes). An incorrect value
    /// silently corrupts memory allocation, stride computation, and buffer
    /// sizing — so this is a regression gate: a new variant that forgets its
    /// size gets caught here rather than in a downstream OOM.
    ///
    /// Sub-byte types (F6E2M3, F6E3M2, F4) legitimately return 0 — they are
    /// packed in hardware and have no natural byte-aligned storage width.
    #[test]
    fn size_in_bytes_is_consistent_for_every_variant() {
        let expected: &[(DType, usize)] = &[
            (DType::U8, 1),
            (DType::I8, 1),
            (DType::U32, 4),
            (DType::I16, 2),
            (DType::I32, 4),
            (DType::I64, 8),
            (DType::BF16, 2),
            (DType::F16, 2),
            (DType::F32, 4),
            (DType::F64, 8),
            (DType::F8E4M3, 1),
            (DType::F8E5M2, 1),
            (DType::F6E2M3, 0),
            (DType::F6E3M2, 0),
            (DType::F4, 0),
            (DType::F8E8M0, 1),
            (DType::F8E6M2, 1),
            (DType::Bool, 1),
        ];
        assert_eq!(
            DType::ALL.len(),
            expected.len(),
            "DType::ALL has {} variants but expected list has {} — \
             add the new variant to the expected list above",
            DType::ALL.len(),
            expected.len(),
        );
        let map: std::collections::HashMap<DType, usize> = expected.iter().copied().collect();
        for &dt in DType::ALL {
            let size = map
                .get(&dt)
                .copied()
                .unwrap_or_else(|| panic!("DType::{dt:?} missing from expected list"));
            assert_eq!(
                dt.size_in_bytes(),
                size,
                "{dt:?}.size_in_bytes() returned {} but expected {size}",
                dt.size_in_bytes(),
            );
        }
    }
}

/// Compile-time completeness tripwire for [`DType::ALL`]. This wildcard-free
/// `match` names every variant, so adding a `DType` fails to compile HERE
/// (non-exhaustive `match`) until the new variant is handled — and
/// [`DType::ALL`] is to be extended in the same edit. Never called; it exists
/// only so the compiler forces acknowledgement of every variant.
#[allow(dead_code)]
fn all_variants_witness(dt: DType) {
    match dt {
        DType::U8
        | DType::I8
        | DType::U32
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::BF16
        | DType::F16
        | DType::F32
        | DType::F64
        | DType::F8E4M3
        | DType::F8E5M2
        | DType::F6E2M3
        | DType::F6E3M2
        | DType::F4
        | DType::F8E8M0
        | DType::F8E6M2
        | DType::Bool => {}
    }
}

/// The **single source of truth** for KISS-CLASSIFY §6.1 tokens that are
/// RESERVED at this schema version.
///
/// Canonical lowercase; compare case-insensitively so the FKC contract
/// spelling (`F8E4M3FNUZ`) resolves against the same list. **One list, two
/// call sites** — `FromStr` here and the FKC token table in
/// `fuel-dispatch/src/fkc/lower.rs` — because two hand-copied lists is the
/// drift pattern this project keeps re-finding.
///
/// These are the AMD `fnuz` FP8 layouts. They are **byte-incompatible** with
/// their OCP counterparts (different exponent bias, no −0, no infinities), so
/// treating one as the other silently corrupts every element rather than
/// failing — which is why the two rejections are worth telling apart.
///
/// # Provenance, and the spelling that changed
///
/// Verbatim from KISS-CLASSIFY at **sk4**: `conformance/src/structure_key.rs`
/// `RESERVED_DTYPES: [&str; 2]`. sk3 spelled these `e4m3fnuz` / `e5m2fnuz`;
/// sk4 respells them with the `f8` prefix. **A stale list here would be wrong
/// in the most expensive way** — tokens that never occur decline nothing, and
/// every test still passes. Re-check this list against KISS on any schema
/// bump; read it with `git show origin/main:<path>`, since a KISS working tree
/// may sit behind its own `origin/main`.
///
/// # Scope — what these do and do NOT discharge
///
/// KISS-CLASSIFY §6.1-0001 places its obligation on a **reader of a
/// `structure_key`**. Fuel has no such reader: it DERIVES and emits keys, and
/// treats foreign ones as opaque bytes by design (K1 opacity —
/// `telemetry::structure_key::StructureKeyToken`). Measured, not assumed: no
/// split/decode path over a structure_key exists outside test code, which
/// parses only Fuel's own just-emitted token to assert its shape.
///
/// So this list does **not** make Fuel §6.1-0001-conformant; there is
/// currently no site in Fuel where that clause applies. What it does is
/// (a) harden the two dtype-token surfaces Fuel really does ingest — its own
/// `FromStr` and FKC contract files, which are external, hand-authored input —
/// so a correctly-spelled `f8e5m2fnuz` is told it is reserved rather than sent
/// hunting for a typo, and (b) put the vocabulary and the decline shape in one
/// place for the day Fuel gains a key reader.
///
/// Note the two vocabularies differ elsewhere even though they agree here:
/// Fuel's internal token for the OCP type is `f8e4m3`, while the KISS seam
/// spells it `f8e4m3fn` (see `telemetry::structure_key_derive`). The reserved
/// spellings coincide only because Fuel has no `fnuz` type at all.
///
/// # This decline EXPIRES (GAP-161)
///
/// sk4 §6.1-0001 gives reserved spellings "no computation semantics at this
/// schema version" and calls activation "a future additive schema event". So
/// unlike a decline that is permanently correct, this one has a **scheduled
/// trigger** — it becomes wrong when KISS activates the tokens. What
/// distinguishes it from a missing-kernel decline is not that it never expires
/// but that its trigger is **external and announced**: it cannot quietly
/// become true through a Fuel-side change. The maintenance that follows is to
/// watch KISS's `RESERVED_DTYPES` on a schema bump, and this const is the
/// single grep target to flip when that happens.
pub const RESERVED_DTYPE_TOKENS: &[&str] = &["f8e4m3fnuz", "f8e5m2fnuz"];

/// Is `s` a §6.1 token that is RESERVED at this schema version?
pub fn is_reserved_dtype_token(s: &str) -> bool {
    RESERVED_DTYPE_TOKENS
        .iter()
        .any(|t| s.eq_ignore_ascii_case(t))
}

/// Error returned when a string cannot be parsed as a [`DType`].
///
/// # Why this distinguishes two rejections
///
/// `f8e5m2fnuz` is a real, specified layout this schema version declines to
/// compute on; `asdf` is not a dtype at all. Those are different facts about
/// the world, and they imply different actions — *wait for a schema bump* vs
/// *fix your input*. Collapsing them tells a caller "no" without telling it
/// *which* no.
///
/// **The distinction is a VARIANT, not a message.** A differing `Display`
/// string is prose a caller cannot act on; the obligation is to a caller of
/// this API, so it has to be matchable.
///
/// For what this does and does not discharge — in particular that it is NOT
/// KISS-CLASSIFY §6.1-0001 conformance, because that clause binds a reader of
/// a `structure_key` and Fuel has none — and for the scheduled external expiry
/// this decline carries, see [`RESERVED_DTYPE_TOKENS`].
#[derive(Debug, PartialEq, Eq)]
pub enum DTypeParseError {
    /// Not a §6.1 dtype token at all.
    Unknown(String),
    /// A §6.1 token that is **reserved** at this schema version: recognized,
    /// deliberately not computed on.
    Reserved(String),
}

impl DTypeParseError {
    /// The offending token, whichever rejection this is.
    pub fn token(&self) -> &str {
        match self {
            DTypeParseError::Unknown(t) | DTypeParseError::Reserved(t) => t,
        }
    }

    /// Was this a recognized-but-reserved spelling?
    pub fn is_reserved(&self) -> bool {
        matches!(self, DTypeParseError::Reserved(_))
    }
}

impl std::fmt::Display for DTypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DTypeParseError::Unknown(t) => write!(f, "cannot parse '{t}' as a dtype"),
            DTypeParseError::Reserved(t) => write!(
                f,
                "dtype token '{t}' is RESERVED at this schema version \
                 (KISS-CLASSIFY §6.1-0001): it names a real layout that is \
                 byte-incompatible with its OCP counterpart, and is deliberately \
                 not computed on — this is not an unknown token"
            ),
        }
    }
}

impl std::error::Error for DTypeParseError {}

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "i8" => Ok(Self::I8),
            "u32" => Ok(Self::U32),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "f8e4m3" => Ok(Self::F8E4M3),
            "f8e5m2" => Ok(Self::F8E5M2),
            "f6e2m3" => Ok(Self::F6E2M3),
            "f6e3m2" => Ok(Self::F6E3M2),
            "f4" => Ok(Self::F4),
            "f8e8m0" => Ok(Self::F8E8M0),
            "f8e6m2" => Ok(Self::F8E6M2),
            "bool" => Ok(Self::Bool),
            // §6.1-0001: a RESERVED spelling is recognized and declined
            // DISTINCTLY from an unknown one. Deleting this arm makes
            // `reserved_fnuz_declines_distinctly_from_an_unknown_token` fail —
            // that is the born-red evidence for this change, and it was
            // observed (3 passed / 1 failed, `Compiling fuel-ir` in the run).
            _ if is_reserved_dtype_token(s) => Err(DTypeParseError::Reserved(s.to_string())),
            _ => Err(DTypeParseError::Unknown(s.to_string())),
        }
    }
}

impl DType {
    /// Returns the string representation of this dtype.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U32 => "u32",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::F6E2M3 => "f6e2m3",
            Self::F6E3M2 => "f6e3m2",
            Self::F4 => "f4",
            Self::F8E8M0 => "f8e8m0",
            Self::F8E6M2 => "f8e6m2",
            Self::Bool => "bool",
        }
    }

    /// Returns the size used by each element in bytes.
    ///
    /// Returns 0 for sub-byte types (F6E2M3, F6E3M2, F4).
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I8 => 1,
            Self::U32 => 4,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::F6E2M3 => 0,
            Self::F6E3M2 => 0,
            Self::F4 => 0,
            Self::F8E8M0 => 1,
            Self::F8E6M2 => 1,
            // Boolean: one byte per element (storage width equals u8).
            Self::Bool => 1,
        }
    }

    /// Returns `true` if this is an integer type (U8, I8, U32, I16, I32, I64).
    pub fn is_int(&self) -> bool {
        match self {
            Self::U8 | Self::I8 | Self::U32 | Self::I16 | Self::I32 | Self::I64 => true,
            Self::BF16
            | Self::F16
            | Self::F32
            | Self::F64
            | Self::F8E4M3
            | Self::F8E5M2
            | Self::F6E2M3
            | Self::F6E3M2
            | Self::F4
            | Self::F8E8M0
            | Self::F8E6M2
            // Bool is a truth value, not an integer.
            | Self::Bool => false,
        }
    }

    /// Returns `true` if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        match self {
            // Bool is a truth value, neither integer nor float.
            Self::U8 | Self::I8 | Self::U32 | Self::I16 | Self::I32 | Self::I64 | Self::Bool => {
                false
            }
            Self::BF16
            | Self::F16
            | Self::F32
            | Self::F64
            | Self::F8E4M3
            | Self::F8E5M2
            | Self::F6E2M3
            | Self::F6E3M2
            | Self::F4
            | Self::F8E8M0
            | Self::F8E6M2 => true,
        }
    }
}

/// Trait for Rust types that can be stored as tensor elements.
///
/// This maps Rust native types (e.g., `f32`, `u8`) to their corresponding [`DType`].
/// It provides conversion routines between Rust values and CPU tensor storage.
pub trait WithDType:
    Sized
    + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd
    + std::fmt::Display
    + 'static
    + Send
    + Sync
    + std::any::Any
{
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn to_scalar(self) -> crate::scalar::Scalar;
}

// B0.4 (WithDType weld break, part 1): the host-buffer conversion methods
// (`cpu_storage_ref`/`to_cpu_storage_owned`/`to_cpu_storage`/
// `cpu_storage_as_slice`/`cpu_storage_data`) moved off `WithDType` onto the
// `HostDType: WithDType` extension trait in `cpu_storage.rs` (where `HostBuffer`
// lives) — this breaks the `dtype <-> cpu_storage` cycle.
//
// B0.5 (WithDType weld break, part 2): the `VecOps` supertrait (the `dtype -> cpu/`
// edge) is GONE — `cpu/` moved to the dedicated `fuel-cpu-kernels` crate, and the
// ~5 fuel-cpu-backend call sites that need `T::vec_dot`/`T::vec_reduce_*` now carry
// an explicit `T: fuel_cpu_kernels::VecOps` bound. `WithDType` is now pure vocabulary
// with no impl edges out of fuel-ir.

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

            fn to_scalar(self) -> crate::scalar::Scalar {
                crate::scalar::Scalar::$dtype(self)
            }
        }
    };
}
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};

with_dtype!(u8, U8, |v: f64| v as u8, |v: u8| v as f64);
with_dtype!(i8, I8, |v: f64| v as i8, |v: i8| v as f64);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
with_dtype!(i16, I16, |v: f64| v as i16, |v: i16| v as f64);
with_dtype!(i32, I32, |v: f64| v as i32, |v: i32| v as f64);
with_dtype!(i64, I64, |v: f64| v as i64, |v: i64| v as f64);
with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);
with_dtype!(f8e4m3, F8E4M3, f8e4m3::from_f64, |v: f8e4m3| v.to_f64());

/// Trait for integer element types that can be used with tensor indexing and masking.
///
/// Implemented for `u8`, `u32`, `i16`, `i32`, and `i64`.
pub trait IntDType: WithDType + num_traits::Bounded {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

impl IntDType for i64 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u8 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for i8 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for i16 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for i32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

/// Trait for floating-point element types.
pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}
impl FloatDType for f8e4m3 {}

// Safetensors interop: DType <-> safetensors::Dtype conversions
use safetensors::tensor as st;

impl From<DType> for st::Dtype {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => st::Dtype::U8,
            DType::I8 => st::Dtype::I8,
            DType::U32 => st::Dtype::U32,
            DType::I16 => st::Dtype::I16,
            DType::I32 => st::Dtype::I32,
            DType::I64 => st::Dtype::I64,
            DType::BF16 => st::Dtype::BF16,
            DType::F16 => st::Dtype::F16,
            DType::F32 => st::Dtype::F32,
            DType::F64 => st::Dtype::F64,
            DType::F8E4M3 => st::Dtype::F8_E4M3,
            DType::F8E5M2 => st::Dtype::F8_E5M2,
            DType::F6E2M3 => st::Dtype::F6_E2M3,
            DType::F6E3M2 => st::Dtype::F6_E3M2,
            DType::F4 => st::Dtype::F4,
            DType::F8E8M0 => st::Dtype::F8_E8M0,
            DType::Bool => st::Dtype::BOOL,
            // safetensors 0.7.0 tracks the OCP MX + standard FP8 set
            // (E4M3/E5M2/E8M0/F6/F4). F8E6M2 is a non-OCP-standard token with no
            // safetensors representation, so it cannot be serialized there.
            DType::F8E6M2 => panic!(
                "F8E6M2 has no safetensors representation (0.7.0 supports \
                 E4M3/E5M2/E8M0/F6/F4, not E6M2)"
            ),
        }
    }
}

impl TryFrom<st::Dtype> for DType {
    type Error = Error;
    fn try_from(value: st::Dtype) -> Result<Self> {
        match value {
            st::Dtype::U8 => Ok(DType::U8),
            st::Dtype::I8 => Ok(DType::I8),
            st::Dtype::U32 => Ok(DType::U32),
            st::Dtype::I16 => Ok(DType::I16),
            st::Dtype::I32 => Ok(DType::I32),
            st::Dtype::I64 => Ok(DType::I64),
            st::Dtype::BF16 => Ok(DType::BF16),
            st::Dtype::F16 => Ok(DType::F16),
            st::Dtype::F32 => Ok(DType::F32),
            st::Dtype::F64 => Ok(DType::F64),
            st::Dtype::F8_E4M3 => Ok(DType::F8E4M3),
            st::Dtype::F8_E5M2 => Ok(DType::F8E5M2),
            st::Dtype::F6_E2M3 => Ok(DType::F6E2M3),
            st::Dtype::F6_E3M2 => Ok(DType::F6E3M2),
            st::Dtype::F4 => Ok(DType::F4),
            st::Dtype::F8_E8M0 => Ok(DType::F8E8M0),
            st::Dtype::BOOL => Ok(DType::Bool),
            dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
        }
    }
}
