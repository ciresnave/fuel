// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel × KISS `structure_key` **byte-match** — the GAP-168 *token-tier* leg.
//!
//! Binds KISS's published, codec-generated reference vectors and asserts that
//! Fuel's independent `sk4` deriver reproduces them **byte for byte**.
//!
//! # Provenance — three commits, because a number is a claim about a tree on BOTH sides
//!
//! ⚠️ **THE sha256 AND BLOB IDS BELOW ARE A RECORDED STAMP, NOT AN ENFORCED
//! CHECK. Nothing in this file hashes the corpus.** They document which
//! artifact was read; they cannot detect a later edit to it. The binding that
//! IS enforced is field-wise, in `corpus_is_the_artifact_this_leg_was_bound_to`
//! (schema, `source_commit`, counts, per-namespace vocabulary versions) plus
//! the per-vector token assertions in `positive_vectors_byte_match` — which
//! together cover strictly more than a digest would for everything this leg
//! reads. A digest was deliberately NOT added: it would duplicate that
//! coverage and require a `sha2` dev-dep. **Recorded here because a hash
//! sitting in a comment READS as a control, and an unlabelled one is a false
//! guard** — the reader must not conclude currency is verified.
//!
//! - **Artifact read:** KISS `f4952b4c` — `conformance/corpus/structure_key_vectors.json`,
//!   blob `fd08f7f2dc6a6f3ee441447ac50e84d01cfd1d0a`,
//!   sha256 `51ea109cf62955645242f6822efcbf652f9b87fa2bd5b38c8c9f2cbf3ff7d178`,
//!   20653 bytes. ⚠️ **RE-MEASURED 2026-09-02 (GAP-273). This leg now reads
//!   `fixtures/kiss-corpus/`, vendored by #39 (`f04bab77`); the stamp above
//!   describes THAT file. It previously read KISS `a43a96f` / blob `f4ec8d44...`
//!   / sha256 `619c834e...` / 16458 bytes, which was CORRECT for the
//!   `tests/corpus/` copy this change DELETES.** ⚠⚠ **AND THE STAMP WAS THE
//!   ONLY INSTRUMENT THAT COULD HAVE CAUGHT THE DRIFT: the six asserted
//!   provenance fields — `schema`, `source_commit`, `token_prefix`,
//!   `structure_key_schema_version`, `recognition_count`, `usable_count` — ARE
//!   SATISFIED BY BOTH FILES, because `source_commit` is the SPEC commit
//!   (`19c3ad7`, `generated_from: spec/classify.md`) and the spec did not move
//!   between the two vendorings. A correct guard aimed at a different axis
//!   passes, and its green reads as *provenance verified*. THAT is why the
//!   caps instruction below is load-bearing rather than hygiene.** ⚠️ **CORRECTED 2026-08-19: this stamp read `a31c624c…` / 12283
//!   bytes, which described the PRE-`f600d870` corpus — the re-vendor (+7
//!   declines, 10 -> 17) did not update it. An unenforced stamp went stale
//!   silently and then caused a real misread: a portfolio-level review
//!   concluded Fuel was seven declines behind KISS on the strength of it.
//!   RE-MEASURE THIS ON EVERY RE-VENDOR.** Read with `git show origin/main:<path>` in `C:\Projects\KISS`
//!   (never the working tree, which is checked out stale); `origin/main` was
//!   confirmed equal to `git ls-remote origin refs/heads/main` at read time, so
//!   this is the live tip and **no invariance claim is needed**. ⚠️ **CORRECTED
//!   2026-09-02: this read "the file has no `coverage_note` key, i.e. it is the
//!   pre-#169 artifact as published". FALSE — it had one (1206 chars), and the
//!   corpus this leg now reads has one of 4677. The sha256/blob/byte stamp in
//!   this same block was CORRECT and current; only the prose conclusion drawn
//!   from it was wrong, which is the harder half to notice.**
//! - **Spec provenance:** the corpus's own `source_commit` = `19c3ad7`
//!   (asserted below, so swapping the vendored file for one of different
//!   provenance fails rather than passing quietly).
//! - **Fuel side:** whatever commit this file lands on. The Fuel-side numbers
//!   below are NOT stable across commits — the derivable set was 10 one hour
//!   before it was 11, on a one-line arm.
//!
//! The vendored copy under `fixtures/kiss-corpus/` is byte-identical to the
//! It is **vendored, not retyped**: the corpus is codec-generated with nothing
//! hand-typed, and retyping a token would destroy exactly that property.
//!
//! # What this leg does and does not measure — three instruments, not two
//!
//! 1. **Vocabulary** — the dtype manifest: **24** tokens a reader must recognize.
//! 2. **Token-grammar** — what Fuel can *spell*: **14** (`fuel_ir::sk4_token`,
//!    asserted in [`fuel_emits_only_recognized_sk4_dtype_spellings`]).
//! 3. **Capability** — what Fuel can actually *derive a key for* on the
//!    production path: **11**.
//!
//! **This file exercises instrument 2.** The **binding stage** on the
//! production path is neither of the stages this file touches: it is the
//! **operand-descriptor path**, `telemetry::baracuda_provider::map_element_kind`,
//! whose `None` aborts the whole derivation through a `?` at the call site.
//! Three dtypes — `i16`, `f8e8m0`, `f8e6m2` — have perfectly legal sk4
//! spellings, serialize correctly, and still cannot produce a key, because
//! Baracuda's `ElementKind` (18 variants at the locked
//! `baracuda-kernel-vocab 0.0.1-alpha.78`) has no counterpart. **A green run
//! here therefore does NOT license "Fuel covers 14 dtypes."** That gap is
//! invisible to both artifact-side instruments by construction, which is why
//! the report format requires the binding stage to be named rather than
//! inferred.
//!
//! `map_element_kind` is not asserted here because it compiles only under
//! `baracuda-types`, a strictly narrower gate than this file's `telemetry`.
//!
//! # Exclusion is not mismatch, and the two are never summed
//!
//! A **mismatch** is Fuel deriving *different bytes* for a cell it can express.
//! An **exclusion** is a cell Fuel cannot express at all — conformant under
//! KISS-CLASSIFY §6.8, with a named Fuel owner. Two of the 20 positive vectors
//! are op-family exclusions (`une`, `scn`); see [`OP_FAMILY_EXCLUSIONS`]. They
//! are reported as a list, never folded into the match count.
//!
//! # Why this is an agreement and not a fit
//!
//! Every cell below is Fuel's **pre-existing** cell, lifted from the unit tests
//! in `telemetry::structure_key_derive`, whose construction idiom dates to
//! `fdc1e987` — months before KISS published this corpus at `958a4ab`
//! (2026-08-12). The inputs were derived from the spec clauses, not fitted to
//! these bytes. The five genuinely new cells (the Vulkan-target twin, the mixed
//! `f8e5m2` weight, and the three `(acc + mp)` cells) are built from the
//! *semantics* in each vector's `note`, reusing the same shapes as their
//! nearest existing sibling, with only the coordinate under test changed.

#![cfg(feature = "telemetry")]

use fuel_dispatch::telemetry::structure_key::FdxOperandDesc;
use fuel_dispatch::telemetry::structure_key_derive::{
    AccMp, FuelOpCategory, GemCell, GemMathPrecision, ReduceAxes,
    derive_structure_key_token_with_acc_mp,
};
use fuel_ir::{DType, Layout, Shape, StrideVec};
use std::collections::{BTreeMap, BTreeSet};

/// Byte-identical vendored copy of the KISS artifact — see the module header.
const CORPUS: &str = include_str!("../fixtures/kiss-corpus/structure_key_vectors.json");

/// The corpus's declared spec provenance. Distinct from the artifact commit
/// (`a43a96f`); conflating the two leaves the report unfalsifiable.
const KISS_SOURCE_COMMIT: &str = "19c3ad7";

/// The sk4 dtype spellings Fuel can emit — instrument 2, the *token-grammar*
/// tier. Pinned as a list rather than a count so a change is a decision with a
/// diff, not a number that silently moves.
///
/// **Not** the set Fuel can key: see the module header on the binding stage.
const FUEL_SK4_SPELLINGS: &[&str] = &[
    "f16", "bf16", "f32", "f64", "i8", "i16", "u8", "u32", "i32", "i64", "f8e4m3fn", "f8e5m2",
    "f8e8m0", "f8e6m2",
    // GAP-168(c): the Bool cut added `DType::Bool`, so Fuel's token-grammar tier
    // (instrument 2) moved 14 -> 15. `bool` is in the corpus recognition AND
    // usable sets and is not reserved. THIS TEST CAUGHT THE OMISSION: the Bool
    // cut was gated with `--lib`, which does not build `tests/`, so main went
    // red here. The list is pinned precisely so this is a decision with a diff.
    "bool",
];

/// Positive vectors Fuel cannot express, with the **field** that excludes them.
///
/// **EMPTY as of GAP-168's op-family increment.** It previously held the only two
/// exclusions — `unary_f16_v8` (`une`) and `noncontraction_scan_mp_only` (`scn`)
/// — both field 2, the §6.5-0006 op-family code, because `FuelOpCategory` had
/// neither variant. Both are now spelled, so **all 20 positives are constructed
/// and byte-matched**; the partition test below asserts the list is empty rather
/// than deleting it, so a future exclusion has to be added deliberately.
///
/// The old note is worth keeping: for `une` Fuel already derived the *identical*
/// operand sub-keys for the same cell shape and differed ONLY in field 2 — which
/// is why closing this was a spelling gap, not a semantic disagreement.
const OP_FAMILY_EXCLUSIONS: &[(&str, &str)] = &[(
    "gem_weight_role_discriminator",
    "KISS-CLASSIFY-6.6-0019 pins <wdt> to the caller's weight-role hint with      operands [weight=i4, weight_scale=f8e8m0, activation=bf16]. Fuel CAN express      the role hint -- `GemCell::weight_dtype` is exactly that field, and      `mixed_fp8_e4m3_x_e4m3_f16` already carries distinct wdt/acc/out. What Fuel      cannot express is the DTYPE: there is no `DType::I4`. `i4` sits in      `fuel_ir::token_kind::RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS`, documented there      as active in the standard with no Fuel DType at all -- a Fuel omission      tracked as GAP-097. No token for this cell can exist until GAP-097 closes,      so this is an absent surface and not an unbuilt cell.",
)];

// ---- cell construction ---------------------------------------------------

fn co(dims: &[usize], dtype: DType) -> FdxOperandDesc {
    FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(dims)), dtype)
}

fn f32c(dims: &[usize]) -> FdxOperandDesc {
    co(dims, DType::F32)
}

fn f16c(dims: &[usize]) -> FdxOperandDesc {
    co(dims, DType::F16)
}

/// An FP8 operand at a 4-byte-aligned base (offset 4 elems × 1 byte), the
/// existing `sk4_gem_mixed_fp8` idiom — which is what derives `v4` rather than
/// `v8` for a 1-byte element.
fn f8(dims: &[usize]) -> FdxOperandDesc {
    FdxOperandDesc::from_layout(
        &Layout::new(
            Shape::from_dims(dims),
            Layout::contiguous(Shape::from_dims(dims))
                .stride()
                .iter()
                .copied()
                .collect::<StrideVec>(),
            4,
        ),
        DType::F8E4M3,
    )
}

/// A bit-stable non-batched f32 gem cell with the given role extents.
fn gem_f32(m: i64, n: i64, k: i64) -> GemCell {
    GemCell {
        m,
        n,
        k,
        batch: None,
        weight_dtype: DType::F32,
        acc_dtype: DType::F32,
        out_dtype: DType::F32,
        math_precision: GemMathPrecision::BitStable,
    }
}

struct Cell {
    op: FuelOpCategory,
    operands: Vec<FdxOperandDesc>,
    target: &'static str,
    acc_mp: Option<AccMp>,
}

impl Cell {
    fn derive(&self) -> Option<String> {
        derive_structure_key_token_with_acc_mp(self.op, &self.operands, self.target, self.acc_mp)
    }
}

fn cell(op: FuelOpCategory, operands: Vec<FdxOperandDesc>, target: &'static str) -> Cell {
    Cell {
        op,
        operands,
        target,
        acc_mp: None,
    }
}

fn cell_acc_mp(
    op: FuelOpCategory,
    operands: Vec<FdxOperandDesc>,
    target: &'static str,
    acc_dtype: DType,
    math_precision: GemMathPrecision,
) -> Cell {
    Cell {
        op,
        operands,
        target,
        acc_mp: Some(AccMp {
            acc_dtype,
            math_precision,
        }),
    }
}

/// Fuel's cell for each expressible positive vector, keyed by the corpus's own
/// vector `name`.
fn cells() -> BTreeMap<&'static str, Cell> {
    use FuelOpCategory::*;

    // The gem operand triple shared by the sm89/sm90/vulkan f32 contractions.
    let gem_ops = || vec![f32c(&[8, 4096]), f32c(&[4096, 4096]), f32c(&[8, 4096])];
    // The rank-2 reduction operand pair (`[4,8] -> [4,1]`).
    let red_ops = || vec![f32c(&[4, 8]), f32c(&[4, 1])];

    let mut m: BTreeMap<&'static str, Cell> = BTreeMap::new();

    // -- elementwise ------------------------------------------------------
    m.insert(
        "elementwise_binary_canonical",
        cell(
            BinaryElementwise,
            vec![f32c(&[128, 256]), f32c(&[128, 256]), f32c(&[128, 256])],
            "cuda:sm89",
        ),
    );
    m.insert(
        "binary_two_operands",
        cell(
            BinaryElementwise,
            vec![f32c(&[128, 256]), f32c(&[128, 256])],
            "cuda:sm89",
        ),
    );
    // GAP-168 op-family increment: the two formerly-excluded positives.
    // `une` — two f16 v8 operands, work class `grid` (>1024 frame elements).
    m.insert(
        "unary_f16_v8",
        cell(
            UnaryElementwise,
            vec![f16c(&[8, 4096]), f16c(&[8, 4096])],
            "cuda:sm89",
        ),
    );
    // `scn` — two f32 v4 operands, work class `warp` (<=32 frame elements), and
    // the §6.7-0013 non-contraction precision coordinate `f32/rm`: accumulator
    // EQUALS compute (f32) while the math precision deviates. That trailing
    // field is why this vector is not merely a family-code change.
    m.insert(
        "noncontraction_scan_mp_only",
        cell_acc_mp(
            Scan,
            vec![f32c(&[2, 16]), f32c(&[2, 16])],
            "cuda:sm89",
            DType::F32,
            GemMathPrecision::ReducedMantissa,
        ),
    );
    m.insert(
        "relu_add_generated_r1",
        cell(
            BinaryElementwise,
            vec![f32c(&[4096]), f32c(&[4096]), f32c(&[4096])],
            "cuda:sm89",
        ),
    );
    // Middle operand broadcasts on axis 0: extent-128 axis at stride 0.
    let bcast = FdxOperandDesc::from_layout(
        &Layout::new(
            Shape::from(vec![128usize, 256]),
            [0isize, 1].into_iter().collect::<StrideVec>(),
            0,
        ),
        DType::F32,
    );
    m.insert(
        "elementwise_broadcast_operand",
        cell(
            BinaryElementwise,
            vec![f32c(&[128, 256]), bcast, f32c(&[128, 256])],
            "cuda:sm89",
        ),
    );

    // -- reductions -------------------------------------------------------
    m.insert(
        "reduction_trailing_axis",
        cell(Reduction(ReduceAxes::TrailingAxis), red_ops(), "cuda:sm89"),
    );
    m.insert(
        "reduction_all_axes",
        cell(Reduction(ReduceAxes::All), red_ops(), "cuda:sm89"),
    );
    m.insert(
        "reduction_rank1_all_axes",
        cell(
            Reduction(ReduceAxes::All),
            vec![f32c(&[8]), f32c(&[1])],
            "cuda:sm89",
        ),
    );
    m.insert(
        "reduction_subset_mask",
        cell(
            Reduction(ReduceAxes::Keepdim(0x0a)),
            vec![f32c(&[2, 4, 3, 5]), f32c(&[2, 1, 3, 1])],
            "cuda:sm89",
        ),
    );

    // -- contractions -----------------------------------------------------
    m.insert(
        "dense_contraction_cuda",
        cell(Contraction(gem_f32(8, 4096, 4096)), gem_ops(), "cuda:sm89"),
    );
    // Same cell, Vulkan capability-set target. §6.8-0002 makes the target a
    // byte-exact passthrough coordinate, so this vector tests that Fuel does
    // not normalize, reorder or otherwise touch a namespaced target string.
    m.insert(
        "dense_contraction_vulkan_target",
        cell(
            Contraction(gem_f32(8, 4096, 4096)),
            gem_ops(),
            "vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none",
        ),
    );
    m.insert(
        "simt_f32",
        cell(Contraction(gem_f32(8, 4096, 4096)), gem_ops(), "cuda:sm90"),
    );
    m.insert(
        "tf32",
        cell(
            Contraction(GemCell {
                math_precision: GemMathPrecision::ReducedMantissa,
                ..gem_f32(8, 4096, 4096)
            }),
            gem_ops(),
            "cuda:sm90",
        ),
    );
    m.insert(
        "gem_batched_cell",
        cell(
            Contraction(GemCell {
                batch: Some(256),
                ..gem_f32(256, 4096, 4096)
            }),
            vec![f32c(&[256, 4096]), f32c(&[4096, 4096]), f32c(&[256, 4096])],
            "cuda:sm90",
        ),
    );
    m.insert(
        "mixed_fp8_e4m3_x_e4m3_f16",
        cell(
            Contraction(GemCell {
                weight_dtype: DType::F8E4M3,
                acc_dtype: DType::F32,
                out_dtype: DType::F16,
                ..gem_f32(8, 4096, 4096)
            }),
            vec![f8(&[8, 4096]), f8(&[4096, 4096]), f8(&[8, 4096])],
            "cuda:sm90",
        ),
    );
    m.insert(
        "mixed_fp8_e4m3_x_e5m2_f32",
        cell(
            Contraction(GemCell {
                weight_dtype: DType::F8E5M2,
                acc_dtype: DType::F32,
                out_dtype: DType::F32,
                ..gem_f32(8, 4096, 4096)
            }),
            vec![f8(&[8, 4096]), f8(&[4096, 4096]), f8(&[8, 4096])],
            "cuda:sm90",
        ),
    );

    // -- non-contraction (acc + mp), §6.7-0013 ----------------------------
    // Accumulator deviates from an f16 compute dtype.
    m.insert(
        "noncontraction_acc_mp_field",
        cell_acc_mp(
            Reduction(ReduceAxes::TrailingAxis),
            vec![f16c(&[4, 8]), f16c(&[4, 1])],
            "cuda:sm89",
            DType::F32,
            GemMathPrecision::BitStable,
        ),
    );
    // Accumulator deviates (f64) from an f32 compute dtype, mp at default.
    m.insert(
        "noncontraction_acc_deviating_f64",
        cell_acc_mp(
            Reduction(ReduceAxes::All),
            red_ops(),
            "cuda:sm89",
            DType::F64,
            GemMathPrecision::BitStable,
        ),
    );
    // Accumulator EQUALS compute (f32); only `<mp>` deviates. Both slots must
    // still be spelled explicitly (§6.7-0013(b)).
    m.insert(
        "noncontraction_reduction_mp_only",
        cell_acc_mp(
            Reduction(ReduceAxes::All),
            red_ops(),
            "cuda:sm89",
            DType::F32,
            GemMathPrecision::ReducedMantissa,
        ),
    );

    m
}

// ---- corpus access -------------------------------------------------------

fn corpus() -> serde_json::Value {
    serde_json::from_str(CORPUS).expect("vendored KISS corpus must parse")
}

fn strs(v: &serde_json::Value, key: &str) -> Vec<String> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("corpus key `{key}` must be an array"))
        .iter()
        .map(|s| s.as_str().expect("string").to_string())
        .collect()
}

fn vectors<'a>(v: &'a serde_json::Value, key: &str) -> &'a Vec<serde_json::Value> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("corpus key `{key}` must be an array"))
}

fn field<'a>(v: &'a serde_json::Value, key: &str) -> &'a str {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("vector is missing string field `{key}`"))
}

// ---- tests ---------------------------------------------------------------

/// The vendored artifact is the one this leg claims to have been bound to.
///
/// Asserting the corpus's *own* declared provenance is what makes the report's
/// commit citations falsifiable: swapping the file for a different schema
/// version or spec commit fails here rather than silently re-baselining every
/// byte-match below.
#[test]
fn corpus_is_the_artifact_this_leg_was_bound_to() {
    let c = corpus();
    assert_eq!(field(&c, "schema"), "kiss-structure-key-vectors-v1");
    assert_eq!(field(&c, "source_commit"), KISS_SOURCE_COMMIT);
    assert_eq!(field(&c, "token_prefix"), "sk4");
    assert_eq!(c["structure_key_schema_version"].as_u64(), Some(4));
    assert_eq!(c["recognition_count"].as_u64(), Some(24));
    assert_eq!(c["usable_count"].as_u64(), Some(22));
    assert_eq!(strs(&c, "dtype_recognition_set").len(), 24);
    assert_eq!(strs(&c, "dtype_usable_set").len(), 22);
    assert_eq!(strs(&c, "target_namespaces"), vec!["cuda", "vulkan"]);
    // The per-namespace vocabulary versions (KISS #200). ASSERTED, not read:
    // a field a consumer only reads is documentation. This is the field whose
    // absence let Fuel and KISS agree on a four-field `vulkan:` token that the
    // vulkan maintainer's own doc had declared malformed four weeks earlier —
    // the byte-match did not fail, it AGREED, because neither side was pointed
    // at the namespace doc. Pinning it here means a vocabulary bump reddens
    // this leg instead of passing through it.
    assert_eq!(c["namespace_vocabulary_versions"]["cuda"].as_u64(), Some(1));
    assert_eq!(
        c["namespace_vocabulary_versions"]["vulkan"].as_u64(),
        Some(5)
    );
    assert_eq!(vectors(&c, "positive_vectors").len(), 21);
    assert_eq!(vectors(&c, "decline_vectors").len(), 17);
}

/// Every positive vector is either **constructed** or **explicitly excluded** —
/// and the two sets partition the corpus exactly.
///
/// This is the anti-drift assertion. Without it a vector added upstream would
/// simply not be looked up, and the byte-match would keep reporting a clean
/// N/N over a silently shrinking denominator — a false negative, the direction
/// that gets filed rather than investigated.
#[test]
fn constructed_and_excluded_partition_the_positive_vectors() {
    let c = corpus();
    let published: BTreeSet<String> = vectors(&c, "positive_vectors")
        .iter()
        .map(|v| field(v, "name").to_string())
        .collect();
    // Uniqueness asserted AS uniqueness: the deduped set must be the same size
    // as the raw list. Written as `== 20` this silently doubled as a corpus-size
    // assertion, so a re-vendor changed the meaning of a test about names.
    assert_eq!(
        published.len(),
        vectors(&c, "positive_vectors").len(),
        "two positive vectors share a name"
    );

    let constructed: BTreeSet<String> = cells().keys().map(|s| s.to_string()).collect();
    let excluded: BTreeSet<String> = OP_FAMILY_EXCLUSIONS
        .iter()
        .map(|(n, _)| n.to_string())
        .collect();

    assert!(
        constructed.is_disjoint(&excluded),
        "a vector cannot be both constructed and excluded: {:?}",
        constructed.intersection(&excluded).collect::<Vec<_>>()
    );
    let covered: BTreeSet<String> = constructed.union(&excluded).cloned().collect();
    assert_eq!(
        covered,
        published,
        "unhandled upstream vectors: {:?} / stale local names: {:?}",
        published.difference(&covered).collect::<Vec<_>>(),
        covered.difference(&published).collect::<Vec<_>>()
    );
    // GAP-168 op-family increment: was 18 constructed / 2 excluded. Both
    // exclusions were field 2 (`une`, `scn`); both are now spelled, so every
    // published positive is constructed and byte-matched.
    //
    // `constructed.len() == 20` USED to sit here and has been REMOVED as
    // redundant: `covered == published` above is a SET equality that already
    // pins every name on both sides, and `is_disjoint` pins the split. The
    // count added no coverage and made a corpus re-vendor look like a defect in
    // this test.
    // SHRINK-ONLY. Was 0 until the f4952b4c re-vendor published
    // `gem_weight_role_discriminator`, whose `i4` weight dtype Fuel has no
    // `DType` for (GAP-097). Raising this number is a deliberate act; lowering
    // it is what closing GAP-097 should do.
    assert_eq!(
        excluded.len(),
        1,
        "the op-family exclusion count moved. Adding one is deliberate and needs \
         a reason naming the ABSENT SURFACE, not merely the unbuilt cell; \
         removing one means a Fuel gap closed and the cell is now constructible."
    );
}

/// **The exclusion's own precondition, so it cannot outlive its cause.**
///
/// `gem_weight_role_discriminator` is excluded because Fuel has no `DType` for
/// `i4`. That is a fact about `fuel_ir`, not about this test, and it is tracked
/// as GAP-097. **If someone implements `DType::I4`, `i4` leaves
/// `RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS` and this fires** -- at which point the
/// cell is CONSTRUCTIBLE and the exclusion must go.
///
/// Without this, closing GAP-097 would leave a stale exclusion quietly costing
/// one vector of coverage, and the byte-match would stay green while covering
/// less than it could. An exclusion that outlives its reason is indistinguishable
/// from one that still has it.
#[test]
fn the_i4_exclusion_still_has_its_reason() {
    assert!(
        fuel_ir::token_kind::RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS.contains(&"i4"),
        "`i4` is no longer in RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS, so Fuel now \
         has a DType for it and `gem_weight_role_discriminator` CAN be \
         constructed. Remove it from OP_FAMILY_EXCLUSIONS, build the cell, and \
         drop the exclusion count to 0."
    );
}

/// **The leg.** Every expressible positive vector, byte for byte.
#[test]
fn positive_vectors_byte_match() {
    let c = corpus();
    let cells = cells();
    let mut matched = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for v in vectors(&c, "positive_vectors") {
        let name = field(v, "name");
        let Some(cell) = cells.get(name) else {
            continue;
        };
        let expected = field(v, "token");
        match cell.derive() {
            // A `None` here is NOT an exclusion — it is a cell Fuel claims to
            // express and then declined to key, which is a defect.
            None => mismatches.push(format!(
                "{name} [{}]: Fuel DECLINED a cell it claims to express\n  expected: {expected}",
                field(v, "clause")
            )),
            Some(got) if got == expected => matched += 1,
            Some(got) => mismatches.push(format!(
                "{name} [{}]:\n  expected: {expected}\n  got:      {got}",
                field(v, "clause")
            )),
        }
    }

    assert!(
        mismatches.is_empty(),
        "byte-match failures:\n{}",
        mismatches.join("\n")
    );
    // Non-vacuity: a lookup bug that matched nothing would otherwise pass.
    // GAP-168 op-family increment: 18 -> 20. `une` and `scn` were the only two
    // inexpressible positives and both are now spelled, so EVERY published
    // positive is byte-matched — the leg no longer carries an op-family
    // exclusion.
    // THE INVARIANT, stated directly: every published positive is either
    // byte-matched or a NAMED op-family exclusion.
    //
    // This used to be `matched == 20`, which enforced the invariant only in
    // combination with `positive_vectors.len() == 20` in the binding test —
    // two magic numbers in two tests, whose CONJUNCTION carried a guarantee
    // neither stated. The hole that opens if either is relaxed: `cells.get`
    // above `continue`s on an unknown name, so an unhandled vector would be
    // skipped and `matched` would still hit its number. Counting against the
    // published total plus the named exclusions closes that by construction and
    // needs no edit when the corpus grows.
    assert_eq!(
        matched + OP_FAMILY_EXCLUSIONS.len(),
        vectors(&c, "positive_vectors").len(),
        "a published positive was neither byte-matched nor a named op-family exclusion"
    );
}

/// Fuel's emitter never produces a token the corpus publishes as a **decline**.
///
/// Fuel is not a *reader* — it has no `structure_key` parse path — so KISS's
/// decline kinds have no Fuel site to map onto, and their `mapping_guard_note`
/// does not bite us on the decline-kind axis. What Fuel *can* be held to is the
/// emit-side dual: an invalid token must be **unrepresentable**, which is
/// strictly stronger than declining it on parse.
///
/// Each is excluded by a **field**, never by a count:
/// - field 1 `<version>` — `sk9`/`sk3`/`sk04`: the prefix is a `sk4|` literal.
/// - field 2 `<op family>` — `zzz`: `FuelOpCategory::code()` is wildcard-free.
/// - field 3 `<dtype>` — `f99`, `f8e4m3fnuz`: sourced from `fuel_ir::sk4_token`,
///   wildcard-free over a closed `DType` with no `fnuz` variant.
/// - field 10 `<contraction>` / `<acc+mp>` — a reserved `<wdt>`, and the `zz`
///   `<mp>` codes: `GemMathPrecision` has exactly two variants, and the dtype
///   coordinates are `DType`s spelled at emission, not free strings.
#[test]
fn fuel_never_emits_a_published_decline_token() {
    let c = corpus();
    let declines: BTreeMap<&str, &str> = vectors(&c, "decline_vectors")
        .iter()
        .map(|v| (field(v, "token"), field(v, "name")))
        .collect();
    // ⚠️ `declines` is keyed by TOKEN, so this length is the DISTINCT-token
    // count — NOT the vector count. Bumping it alone would pass even if two
    // decline vectors collided on one token, which is exactly the
    // non-injectivity failure that makes a false agreement look like a real
    // one. Assert BOTH, so the pair is a real injectivity check on the
    // published decline set rather than an accident of two numbers matching.
    let published = vectors(&c, "decline_vectors").len();
    assert_eq!(published, 17, "published decline vectors");
    assert_eq!(
        declines.len(),
        published,
        "two decline vectors share a token — the decline set is not injective, \
         so a Fuel emission matching one of them would be attributed ambiguously",
    );

    for (name, cell) in cells() {
        let Some(token) = cell.derive() else { continue };
        assert!(
            !declines.contains_key(token.as_str()),
            "cell `{name}` emitted the token published as decline `{}`",
            declines[token.as_str()]
        );
        assert!(
            token.starts_with("sk4|"),
            "cell `{name}` emitted a non-sk4 prefix: {token}"
        );
    }
}

/// §6.7-0013(d) — the all-default `(acc + mp)` spelling is a **forbidden
/// redundant emission**, and Fuel makes it unreachable rather than checking for
/// it.
///
/// Directly against the published decline vector: hand the `rall` f32 cell an
/// `(acc + mp)` that is entirely at its defaults (`f32/st` on an `f32` compute
/// dtype) and Fuel emits the *field-absent* form — which is the corpus's own
/// `reduction_all_axes` **positive** vector — not the redundant form it
/// publishes as `RedundantAccMpField`.
///
/// The two vectors differ by exactly the trailing field, so this asserts the
/// §6.7-0013(c)/(e) omitted-when-absent rule at the same time: the trap here is
/// that the mandatory reduce field one slot earlier emits `-` when inapplicable,
/// and copying that convention would append a spurious `|-`.
#[test]
fn redundant_acc_mp_is_unrepresentable_on_the_emit_path() {
    let c = corpus();
    let redundant = vectors(&c, "decline_vectors")
        .iter()
        .find(|v| field(v, "name") == "redundant_acc_mp_all_default")
        .expect("corpus must publish the rule-(d) decline vector");
    let field_absent = vectors(&c, "positive_vectors")
        .iter()
        .find(|v| field(v, "name") == "reduction_all_axes")
        .expect("corpus must publish the field-absent twin");

    let token = cell_acc_mp(
        FuelOpCategory::Reduction(ReduceAxes::All),
        vec![f32c(&[4, 8]), f32c(&[4, 1])],
        "cuda:sm89",
        DType::F32,
        GemMathPrecision::BitStable,
    )
    .derive()
    .expect("the all-default (acc + mp) cell must still derive a token");

    assert_ne!(
        token,
        field(redundant, "token"),
        "emitted the rule-(d) redundant form"
    );
    assert_eq!(token, field(field_absent, "token"));
    // The two vectors really do differ only by the trailing field — otherwise
    // the assertion above would be passing for an unrelated reason.
    assert_eq!(
        field(redundant, "token"),
        format!("{}|f32/st", field(field_absent, "token")),
        "corpus invariant this test leans on has changed"
    );
}

/// Every dtype spelling Fuel can emit is in the corpus's **recognition set**,
/// and none is a **reserved** token.
///
/// Instrument 2, and the guard against the one drift that produces a
/// well-formed key nothing downstream can catch: a *retired* spelling under a
/// *current* version prefix (§6.1-0004).
#[test]
fn fuel_emits_only_recognized_sk4_dtype_spellings() {
    let c = corpus();
    let recognized: BTreeSet<String> = strs(&c, "dtype_recognition_set").into_iter().collect();
    let reserved: BTreeSet<String> = strs(&c, "reserved_dtypes").into_iter().collect();

    let emitted: BTreeSet<String> = DType::ALL
        .iter()
        .filter_map(|&dt| fuel_ir::sk4_token(dt))
        .map(String::from)
        .collect();

    let expected: BTreeSet<String> = FUEL_SK4_SPELLINGS.iter().map(|s| s.to_string()).collect();
    assert_eq!(emitted, expected, "Fuel's sk4 spelling set moved");

    for tok in &emitted {
        assert!(
            recognized.contains(tok),
            "`{tok}` is outside the closed sk4 vocabulary"
        );
        assert!(
            !reserved.contains(tok),
            "`{tok}` is RESERVED at sk4 and must never be emitted"
        );
    }
    // Non-vacuity, and the discrimination check: the set is neither empty nor
    // everything, so `is-a-subset` is a real constraint here.
    // GAP-168(c): 14 -> 15 with `bool`. Instrument 2 (token-grammar tier) is now
    // 15 of the corpus's 24 recognized spellings — a cross-project number, so it
    // moves only with a diff like this one.
    assert_eq!(emitted.len(), 15);
    assert!(emitted.len() < recognized.len());
}
