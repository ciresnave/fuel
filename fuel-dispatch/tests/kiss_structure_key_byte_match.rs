// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel × KISS `structure_key` **byte-match** — the GAP-168 *token-tier* leg.
//!
//! Binds KISS's published, codec-generated reference vectors and asserts that
//! Fuel's independent `sk5` deriver reproduces them **byte for byte**.
//!
//! # Provenance — three commits, because a number is a claim about a tree on BOTH sides
//!
//! ⚠️ **THE sha256 AND BLOB IDS BELOW ARE A RECORDED STAMP, NOT AN ENFORCED
//! CHECK. Nothing in this file hashes the corpus.** They document which
//! artifact was read; they cannot detect a later edit to it. The binding that
//! IS enforced is field-wise, in `corpus_is_the_artifact_this_leg_was_bound_to`
//! (schema, `source_commit`, prefix, schema version, counts, reserved set,
//! per-namespace vocabulary versions) plus the per-vector token assertions in
//! `positive_vectors_byte_match` — which together cover strictly more than a
//! digest would for everything this leg reads. **Recorded here because a hash
//! sitting in a comment READS as a control, and an unlabelled one is a false
//! guard** — the reader must not conclude currency is verified.
//!
//! - **Artifact read:** KISS `5b28bf51` (sk5, `contract_version` 3, conformance
//!   0.2.1) — `conformance/corpus/structure_key_vectors.json`, blob
//!   `e5eec85c82754f030f1ee2f0ab4bf7829fa7e5d2`, sha256
//!   `8368d9911a2a4f0895157dda2950ecd23d633d9781ce5ce116b670fad25b634e`,
//!   30368 bytes, vendored 2026-10-11 (the bytes were read with
//!   `git show 5b28bf51:<path>`, a commit object, never a working tree; the
//!   blob id was matched against `git rev-parse 5b28bf51:<path>`). It
//!   previously read KISS `f4952b4c` / blob `fd08f7f2…` / 20653 bytes (sk4).
//!   ⚠️ **The six provenance fields this leg asserts are NOT all independent of
//!   the file:** `source_commit` is the *spec* commit KISS hand-writes into the
//!   generator (`19c3ad7`, KISS #218, unchanged across the sk4→sk5 event), so
//!   two different vendorings both satisfy it. It is asserted, but it is NOT
//!   evidence of WHICH vendoring this is — the prefix, schema version and counts
//!   distinguish sk4 from sk5, and **nothing in this file distinguishes
//!   `a33282c` from `5b28bf51`** (KISS #536 changed seven operand masks and two
//!   decline-vector tokens, none of which a provenance field records; the
//!   byte-match itself is what notices). An earlier version of this stamp had
//!   to be corrected more than once for exactly that confusion; **RE-MEASURE IT
//!   ON EVERY RE-VENDOR.**
//! - **Spec provenance:** the corpus's own `source_commit` = `19c3ad7`
//!   (asserted below, so swapping the vendored file for one of different
//!   provenance fails rather than passing quietly).
//! - **Fuel side:** whatever commit this file lands on. The Fuel-side numbers
//!   below are NOT stable across commits.
//!
//! The vendored copy under `fixtures/kiss-corpus/` is **vendored, not
//! retyped**: the corpus is codec-generated with nothing hand-typed, and
//! retyping a token would destroy exactly that property.
//!
//! # What changed at sk5, and what this leg now measures
//!
//! KISS `a33282c` replaced the 24 caller-chosen op-family codes in field 2 by a
//! **derived** seven-class `cell_class` (§6.5-0015), keyed `dequant_form` as an
//! optional sixth sub-key component, fixed the extent-1 broadcast mask (#520),
//! and made `f8e6m2` reserved (#517). So:
//!
//! - The cells are built from **facts** (`DeclaredFacts`: M/N/K roles, index
//!   slot, reduced set) and **roles** (`CellOperand`), never from a category —
//!   the class in each expected token is DERIVED by Fuel and compared, not
//!   supplied. A cell that derived a different class would fail here.
//! - ⚠️ **This leg, like any token-level byte-match, cannot see derivation**:
//!   the class code is IN each token, but a deriver that reads a different
//!   declared fact or a different pointwise test and *happens* to agree on every
//!   sampled cell passes. KISS says so itself (corpus `coverage_note` (d)); the
//!   derivation is exercised separately by the in-module truth-table tests in
//!   `telemetry::structure_key_derive`.
//!
//! # History: a vendoring that briefly could not be byte-matched
//!
//! KISS `a33282c` (the first sk5 artifact) published seven reduction vectors whose
//! keepdim OUTPUT operand was spelled with broadcast mask `00`, although KISS's
//! own `derive_bcast_mask_in_frame` returned the bit for the same operands
//! (§6.6-0009 keepdim form + §6.6-0008 as amended by #520). The vectors are
//! hand-constructed `StructureKey` literals, so the sk5 rule change had not
//! reached them. Fuel followed the clause, reported the seven as a named
//! divergence, and told the KISS lane; KISS #536 (`5b28bf51`) regenerated them
//! with the derived masks (`02`/`03`/`01`/`0a`) — exactly the masks Fuel had
//! derived independently. The divergence machinery is gone; this note stays so
//! a reader who finds `a33282c` cited elsewhere knows why it was not the pin.
//!
//! # What this leg does and does not measure — three instruments, not two
//!
//! 1. **Vocabulary** — the dtype manifest: **24** tokens a reader must recognize.
//! 2. **Token-grammar** — what Fuel can *spell* at sk5: **14**
//!    (`sk6_dtype_token`, asserted in
//!    [`fuel_emits_only_recognized_sk5_dtype_spellings`]); 15 at sk4 before
//!    `f8e6m2` became reserved.
//! 3. **Capability** — what Fuel can actually *derive a key for* on the
//!    production path (Baracuda's `ElementKind` binding stage), which this file
//!    does NOT touch and which is still an `sk4` codec. **A green run here
//!    therefore does NOT license "Fuel covers 14 dtypes" or "production derives
//!    sk5".**
//!
//! # Exclusion is not mismatch, and the two are never summed
//!
//! A **mismatch** is Fuel deriving *different bytes* for a cell it can express.
//! An **exclusion** is a cell Fuel cannot express at all — conformant under
//! KISS-CLASSIFY §6.8, with a named Fuel owner. The three excluded vectors are
//! all `i4` cells (GAP-097); see [`ABSENT_SURFACE_EXCLUSIONS`]. They are
//! reported as a list, never folded into the match count.
//!
//! # Why this is an agreement and not a fit
//!
//! The pre-existing cells are Fuel's own, lifted from the unit tests in
//! `telemetry::structure_key_derive` (construction idiom dates to `fdc1e987`,
//! months before KISS published this corpus), and their inputs were derived
//! from the spec clauses, not fitted to these bytes. The five new sk5 cells are
//! built from the *semantics* in each vector's `note`, reusing the shape of the
//! nearest existing sibling with only the coordinate under test changed.

#![cfg(feature = "telemetry")]

use fuel_dispatch::telemetry::structure_key::FdxOperandDesc;
use fuel_dispatch::telemetry::structure_key_derive::{
    AccMp, CellClass, CellOperand, DeclaredFacts, DequantForm, GemCell, GemMathPrecision,
    ReduceAxes, derive_structure_key_token, sk6_dtype_token,
};
use fuel_ir::{DType, Layout, Shape, StrideVec};
use std::collections::{BTreeMap, BTreeSet};

/// Byte-identical vendored copy of the KISS artifact — see the module header.
const CORPUS: &str = include_str!("../fixtures/kiss-corpus/structure_key_vectors.json");

/// The corpus's declared spec provenance. Distinct from the artifact commit
/// (`5b28bf51`); conflating the two leaves the report unfalsifiable.
const KISS_SOURCE_COMMIT: &str = "19c3ad7";

/// The sk5 dtype spellings Fuel can emit — instrument 2, the *token-grammar*
/// tier. Pinned as a list rather than a count so a change is a decision with a
/// diff, not a number that silently moves.
///
/// `f8e6m2` is **absent on purpose**: reserved at sk5 (KISS #517), so the
/// deriver does not spell it (it did at sk4, when this list had 15 entries).
///
/// **Not** the set Fuel can key: see the module header on the binding stage.
const FUEL_SK5_SPELLINGS: &[&str] = &[
    "f16", "bf16", "f32", "f64", "i8", "i16", "u8", "u32", "i32", "i64", "f8e4m3fn", "f8e5m2",
    "f8e8m0", "bool",
];

/// Positive vectors Fuel cannot express, with the **absent surface** that
/// excludes them. Every entry is an `i4` cell: `i4` sits in
/// `fuel_ir::token_kind::RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS`, documented
/// there as active in the standard with no Fuel `DType` at all (GAP-097). No
/// token for these cells can exist until GAP-097 closes, so each is an absent
/// SURFACE and not an unbuilt cell.
///
/// The two `dequant_form_*` entries are new at sk5. The `dl`/`dc` PLACEMENT
/// they test is still measured against KISS's own bytes, modulo the one
/// unrepresentable dtype coordinate, by
/// [`dequant_form_vectors_match_modulo_the_i4_coordinate`].
const ABSENT_SURFACE_EXCLUSIONS: &[(&str, &str)] = &[
    (
        "gem_weight_role_discriminator",
        "KISS-CLASSIFY-6.6-0019 pins <wdt> to the caller's weight-role hint with operands [weight=i4, weight_scale=f8e8m0, activation=bf16]. Fuel CAN express the role hint -- `GemCell::weight_dtype` is exactly that field. What Fuel cannot express is the DTYPE: there is no `DType::I4` (GAP-097).",
    ),
    (
        "dequant_form_linear_weight",
        "int4 weight operand carrying a linear dequant (`/dl`). Fuel CAN express the dequant_form (`CellOperand::with_dequant_form`); it cannot express the `i4` primary/weight dtype (no `DType::I4`, GAP-097).",
    ),
    (
        "dequant_form_codebook_weight",
        "int4 weight operand carrying a codebook dequant (`/dc`). Same absent surface as `dequant_form_linear_weight`: no `DType::I4` (GAP-097).",
    ),
    (
        "quantized_ctr_scale_f8e8m0",
        "int4 weight with an `f8e8m0` scale operand keyed as the trailing `/qf8e8m0` (sk6, §6.6-0022). Fuel CAN express the scale-role hint (`GemCell::scale_dtype`); it cannot express the `i4` weight/primary dtype (no `DType::I4`, GAP-097). Measured modulo the i4 coordinate by [`quantized_scale_vectors_match_modulo_the_i4_coordinate`].",
    ),
    (
        "quantized_ctr_scale_f16",
        "Same cell as `quantized_ctr_scale_f8e8m0` with an `f16` scale (`/qf16`); same absent surface (no `DType::I4`, GAP-097).",
    ),
];

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
/// existing mixed-fp8 idiom — which is what derives `v4` rather than `v8` for a
/// 1-byte element.
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
        scale_dtype: None,
    }
}

/// Canonical order (§6.6-0014) for the common `inputs…, output` shape: every
/// descriptor but the last is an input and the last is the output. Cells with a
/// different role layout (in-place, several outputs) are built explicitly.
fn io(descs: Vec<FdxOperandDesc>) -> Vec<CellOperand> {
    let n = descs.len();
    descs
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            if i + 1 == n {
                CellOperand::output(d)
            } else {
                CellOperand::input(d)
            }
        })
        .collect()
}

struct Cell {
    facts: DeclaredFacts,
    operands: Vec<CellOperand>,
    target: &'static str,
    acc_mp: Option<AccMp>,
}

impl Cell {
    fn derive(&self) -> Option<String> {
        derive_structure_key_token(&self.facts, &self.operands, self.target, self.acc_mp)
    }
}

fn cell(facts: DeclaredFacts, operands: Vec<CellOperand>, target: &'static str) -> Cell {
    Cell {
        facts,
        operands,
        target,
        acc_mp: None,
    }
}

fn cell_acc_mp(
    facts: DeclaredFacts,
    operands: Vec<CellOperand>,
    target: &'static str,
    acc_dtype: DType,
    math_precision: GemMathPrecision,
) -> Cell {
    Cell {
        facts,
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
    // The gem operand triple shared by the sm89/sm90/vulkan f32 contractions.
    let gem_ops = || {
        io(vec![
            f32c(&[8, 4096]),
            f32c(&[4096, 4096]),
            f32c(&[8, 4096]),
        ])
    };
    // The rank-2 trailing-axis keepdim pair (`[4,8] -> [4,1]`).
    let red_ops = || io(vec![f32c(&[4, 8]), f32c(&[4, 1])]);
    // The rank-2 all-axes keepdim pair (`[4,8] -> [1,1]`): every reduced axis is
    // folded to a size-1 output axis (§6.6-0009). The sk4 byte-match reused
    // `[4,1]` here, which was not a keepdim of an ALL-axes reduction and only
    // went unnoticed because both spelled the same sk4 sub-keys; at sk5 the
    // output mask (`03` vs `02`) tells them apart.
    let all_ops = || io(vec![f32c(&[4, 8]), f32c(&[1, 1])]);
    let none = DeclaredFacts::none;

    let mut m: BTreeMap<&'static str, Cell> = BTreeMap::new();

    // -- elementwise ------------------------------------------------------
    // Two inputs + one output, pointwise by shape: `ew2`.
    m.insert(
        "elementwise_binary_canonical",
        cell(
            none(),
            io(vec![
                f32c(&[128, 256]),
                f32c(&[128, 256]),
                f32c(&[128, 256]),
            ]),
            "cuda:sm89",
        ),
    );
    // Two operands, the read-modify-write operand appearing ONCE as an input
    // (§6.6-0014): `acc op= x`. Two input operands, pointwise: `ew2`. The same
    // two descriptors as input + OUTPUT would be `ew1` — the in-place role is
    // what makes this vector `ew2`, and the vector's note says "binary".
    m.insert(
        "binary_two_operands",
        cell(
            none(),
            vec![
                CellOperand::input(f32c(&[128, 256])),
                CellOperand::in_place(f32c(&[128, 256])),
            ],
            "cuda:sm89",
        ),
    );
    // One input + one output: `ew1`.
    m.insert(
        "unary_f16_v8",
        cell(
            none(),
            io(vec![f16c(&[8, 4096]), f16c(&[8, 4096])]),
            "cuda:sm89",
        ),
    );
    // A scan is the same cell as a unary activation at the key level (`ew1`) —
    // the sk5 design's own example. The trailing field is the §6.7-0013
    // non-contraction precision coordinate `f32/rm`: accumulator EQUALS compute
    // (f32) while the math precision deviates.
    m.insert(
        "noncontraction_scan_mp_only",
        cell_acc_mp(
            none(),
            io(vec![f32c(&[2, 16]), f32c(&[2, 16])]),
            "cuda:sm89",
            DType::F32,
            GemMathPrecision::ReducedMantissa,
        ),
    );
    m.insert(
        "relu_add_generated_r1",
        cell(
            none(),
            io(vec![f32c(&[4096]), f32c(&[4096]), f32c(&[4096])]),
            "cuda:sm89",
        ),
    );
    // Middle operand broadcasts on axis 0: extent-128 axis at stride 0. A
    // broadcast INPUT is still pointwise.
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
            none(),
            io(vec![f32c(&[128, 256]), bcast, f32c(&[128, 256])]),
            "cuda:sm89",
        ),
    );

    // -- reductions: the reduced-axis set is the declared fact -------------
    m.insert(
        "reduction_trailing_axis",
        cell(
            DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
            red_ops(),
            "cuda:sm89",
        ),
    );
    m.insert(
        "reduction_all_axes",
        cell(
            DeclaredFacts::reduction(ReduceAxes::All),
            all_ops(),
            "cuda:sm89",
        ),
    );
    m.insert(
        "reduction_rank1_all_axes",
        cell(
            DeclaredFacts::reduction(ReduceAxes::All),
            io(vec![f32c(&[8]), f32c(&[1])]),
            "cuda:sm89",
        ),
    );
    m.insert(
        "reduction_subset_mask",
        cell(
            DeclaredFacts::reduction(ReduceAxes::Keepdim(0x0a)),
            io(vec![f32c(&[2, 4, 3, 5]), f32c(&[2, 1, 3, 1])]),
            "cuda:sm89",
        ),
    );

    // -- contractions: the M/N/K roles are the declared fact ----------------
    m.insert(
        "dense_contraction_cuda",
        cell(
            DeclaredFacts::contraction(gem_f32(8, 4096, 4096)),
            gem_ops(),
            "cuda:sm89",
        ),
    );
    // Same cell, Vulkan capability-set target. §6.8-0002 makes the target a
    // byte-exact passthrough coordinate, so this vector tests that Fuel does
    // not normalize, reorder or otherwise touch a namespaced target string.
    m.insert(
        "dense_contraction_vulkan_target",
        cell(
            DeclaredFacts::contraction(gem_f32(8, 4096, 4096)),
            gem_ops(),
            "vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none",
        ),
    );
    m.insert(
        "simt_f32",
        cell(
            DeclaredFacts::contraction(gem_f32(8, 4096, 4096)),
            gem_ops(),
            "cuda:sm90",
        ),
    );
    m.insert(
        "tf32",
        cell(
            DeclaredFacts::contraction(GemCell {
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
            DeclaredFacts::contraction(GemCell {
                batch: Some(256),
                ..gem_f32(256, 4096, 4096)
            }),
            io(vec![
                f32c(&[256, 4096]),
                f32c(&[4096, 4096]),
                f32c(&[256, 4096]),
            ]),
            "cuda:sm90",
        ),
    );
    m.insert(
        "mixed_fp8_e4m3_x_e4m3_f16",
        cell(
            DeclaredFacts::contraction(GemCell {
                weight_dtype: DType::F8E4M3,
                acc_dtype: DType::F32,
                out_dtype: DType::F16,
                ..gem_f32(8, 4096, 4096)
            }),
            io(vec![f8(&[8, 4096]), f8(&[4096, 4096]), f8(&[8, 4096])]),
            "cuda:sm90",
        ),
    );
    m.insert(
        "mixed_fp8_e4m3_x_e5m2_f32",
        cell(
            DeclaredFacts::contraction(GemCell {
                weight_dtype: DType::F8E5M2,
                acc_dtype: DType::F32,
                out_dtype: DType::F32,
                ..gem_f32(8, 4096, 4096)
            }),
            io(vec![f8(&[8, 4096]), f8(&[4096, 4096]), f8(&[8, 4096])]),
            "cuda:sm90",
        ),
    );

    // -- non-contraction (acc + mp), §6.7-0013 ----------------------------
    // Accumulator deviates from an f16 compute dtype.
    m.insert(
        "noncontraction_acc_mp_field",
        cell_acc_mp(
            DeclaredFacts::reduction(ReduceAxes::TrailingAxis),
            io(vec![f16c(&[4, 8]), f16c(&[4, 1])]),
            "cuda:sm89",
            DType::F32,
            GemMathPrecision::BitStable,
        ),
    );
    // Accumulator deviates (f64) from an f32 compute dtype, mp at default.
    m.insert(
        "noncontraction_acc_deviating_f64",
        cell_acc_mp(
            DeclaredFacts::reduction(ReduceAxes::All),
            all_ops(),
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
            DeclaredFacts::reduction(ReduceAxes::All),
            all_ops(),
            "cuda:sm89",
            DType::F32,
            GemMathPrecision::ReducedMantissa,
        ),
    );

    // -- the derived cell class (§6.5-0015), new at sk5 ---------------------
    // Three inputs + one output, pointwise: `ew3`.
    m.insert(
        "cell_class_ew3_three_inputs",
        cell(
            none(),
            io(vec![
                f32c(&[128, 256]),
                f32c(&[128, 256]),
                f32c(&[128, 256]),
                f32c(&[128, 256]),
            ]),
            "cuda:sm89",
        ),
    );
    // Table + index + output, the index-operand ROLE declared (slot 1): `ndx`,
    // rule 2, ahead of the pointwise rule. The index is an i32 operand, which
    // is the realistic case AND the discriminating one: §6.6-0012 says the
    // index is identified by its declared role and never by dtype, and the cell
    // would derive `ew2` if the declared slot were ignored.
    m.insert(
        "cell_class_ndx_index_role_declared",
        cell(
            DeclaredFacts::indexed(1),
            io(vec![
                f32c(&[128, 256]),
                co(&[128, 256], DType::I32),
                f32c(&[128, 256]),
            ]),
            "cuda:sm89",
        ),
    );
    // No declared fact, and the written operand does not have the frame
    // extents ([64,256] against a [128,256] frame): `oth`.
    m.insert(
        "cell_class_oth_not_pointwise",
        cell(
            none(),
            io(vec![f32c(&[128, 256]), f32c(&[64, 256])]),
            "cuda:sm89",
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

fn positive<'a>(c: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    vectors(c, "positive_vectors")
        .iter()
        .find(|v| field(v, "name") == name)
        .unwrap_or_else(|| panic!("corpus must publish positive vector `{name}`"))
}

fn decline<'a>(c: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    vectors(c, "decline_vectors")
        .iter()
        .find(|v| field(v, "name") == name)
        .unwrap_or_else(|| panic!("corpus must publish decline vector `{name}`"))
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
    assert_eq!(field(&c, "token_prefix"), "sk6");
    assert_eq!(c["structure_key_schema_version"].as_u64(), Some(6));
    assert_eq!(c["recognition_count"].as_u64(), Some(24));
    assert_eq!(c["usable_count"].as_u64(), Some(21));
    assert_eq!(strs(&c, "dtype_recognition_set").len(), 24);
    assert_eq!(strs(&c, "dtype_usable_set").len(), 21);
    assert_eq!(
        strs(&c, "reserved_dtypes"),
        vec!["f8e4m3fnuz", "f8e5m2fnuz", "f8e6m2"],
        "the reserved set moved — f8e6m2 became reserved at sk5 (KISS #517)"
    );
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
    assert_eq!(vectors(&c, "positive_vectors").len(), 28);
    assert_eq!(vectors(&c, "decline_vectors").len(), 25);
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
    // as the raw list.
    assert_eq!(
        published.len(),
        vectors(&c, "positive_vectors").len(),
        "two positive vectors share a name"
    );

    let constructed: BTreeSet<String> = cells().keys().map(|s| s.to_string()).collect();
    let excluded: BTreeSet<String> = ABSENT_SURFACE_EXCLUSIONS
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
    // SHRINK-ONLY in spirit. Was 0, then 1 (`gem_weight_role_discriminator`,
    // GAP-097), 3 at sk5, now 5 because sk6 published two quantized `i4` scale cells. Raising this
    // number is a deliberate act; lowering it is what closing GAP-097 should do.
    assert_eq!(
        excluded.len(),
        5,
        "the exclusion count moved. Adding one is deliberate and needs a reason \
         naming the ABSENT SURFACE, not merely the unbuilt cell; removing one \
         means a Fuel gap closed and the cell is now constructible."
    );
}

/// **The exclusions' own precondition, so none can outlive its cause.**
///
/// All five exclusions are `i4` cells, excluded because Fuel has no `DType`
/// for `i4` — a fact about `fuel_ir`, tracked as GAP-097. **If someone
/// implements `DType::I4`, `i4` leaves `RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS`
/// and this fires** — at which point the cells are CONSTRUCTIBLE and the
/// exclusions must go. Without this, closing GAP-097 would leave stale
/// exclusions quietly costing five vectors of coverage while the byte-match
/// stayed green.
///
/// It also asserts every exclusion really IS an `i4` cell, by reading the
/// vector's own token: an exclusion added for some other reason, under this
/// precondition, would otherwise inherit a reason it does not have.
#[test]
fn the_i4_exclusion_still_has_its_reason() {
    assert!(
        fuel_ir::token_kind::RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS.contains(&"i4"),
        "`i4` is no longer in RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS, so Fuel now \
         has a DType for it and the excluded cells CAN be constructed. Remove \
         them from ABSENT_SURFACE_EXCLUSIONS, build the cells, and drop the \
         exclusion count to 0."
    );
    let c = corpus();
    for (name, _) in ABSENT_SURFACE_EXCLUSIONS {
        let token = field(positive(&c, name), "token");
        let primary = token.split('|').nth(2).unwrap();
        assert_eq!(
            primary, "i4",
            "exclusion `{name}` is not an i4 cell (primary dtype `{primary}`): \
             the i4/GAP-097 reason does not apply to it: {token}"
        );
    }
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
    // THE INVARIANT, stated directly: every published positive is byte-matched
    // or a NAMED absent-surface exclusion. Counting against the
    // published total plus the named exclusions closes the hole where
    // `cells.get` above `continue`s on an unknown name and `matched` would
    // still hit a fixed number; it needs no edit when the corpus grows.
    assert_eq!(
        matched + ABSENT_SURFACE_EXCLUSIONS.len(),
        vectors(&c, "positive_vectors").len(),
        "a published positive was neither byte-matched nor a named exclusion"
    );
    // Non-vacuity: a lookup bug that matched nothing must not pass.
    assert!(matched > 0);
}

/// The derived class in each constructed cell is what the vector's token
/// carries — read back as a CLASS, not as a string, so a failure names the
/// derivation. Also a coverage census: the corpus must exercise all seven
/// classes at least once between the constructed cells, or "the class is
/// derived" is untested for the missing one.
#[test]
fn every_class_is_exercised_by_a_byte_matched_cell() {
    let want: BTreeSet<&str> = ["ctr", "ndx", "rdc", "ew1", "ew2", "ew3", "oth"]
        .into_iter()
        .collect();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (name, cell) in cells() {
        let token = cell
            .derive()
            .unwrap_or_else(|| panic!("`{name}` must derive"));
        seen.insert(token.split('|').nth(1).unwrap().to_string());
    }
    let seen_refs: BTreeSet<&str> = seen.iter().map(String::as_str).collect();
    assert_eq!(
        seen_refs, want,
        "the constructed cells do not cover all seven classes"
    );
    // The class enum Fuel emits is the same closed set (wildcard-free `code`).
    let enum_codes: BTreeSet<&str> = [
        CellClass::DenseContraction,
        CellClass::IndexedAccess,
        CellClass::Reduction,
        CellClass::ElementwiseOne,
        CellClass::ElementwiseTwo,
        CellClass::ElementwiseThree,
        CellClass::Other,
    ]
    .into_iter()
    .map(CellClass::code)
    .collect();
    assert_eq!(enum_codes, want);
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
/// - field 1 `<version>` — `sk9`/`sk3`/`sk04`: the prefix is a `sk6|` literal.
/// - field 2 `<cell class>` — `zzz` and the retired `gem`: `CellClass` is a
///   closed seven-variant enum and `code()` is wildcard-free.
/// - field 3 `<dtype>` — `f99`, `f8e4m3fnuz`, `f8e6m2`: sourced from
///   `sk6_dtype_token`, which has no `fnuz` variant to spell and declines the
///   reserved `f8e6m2`.
/// - field 7 operand sub-keys — an unknown `dequant_form` spelling and a
///   seven-part sub-key: `DequantForm` has exactly two variants and the
///   sub-key is built with at most six parts.
/// - field 9 `<reduce>` — `rdc` with `-`: the class is `rdc` only when the
///   reduced set is non-empty in the frame.
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
    assert_eq!(published, 25, "published decline vectors");
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
            token.starts_with("sk6|"),
            "cell `{name}` emitted a non-sk6 prefix: {token}"
        );
    }
}

/// The four decline vectors that are NEW at sk5 each have an emit-side dual,
/// and this asserts each one against the vector's own bytes rather than only
/// against "Fuel did not happen to emit it":
///
/// - `unknown_cell_class` / `retired_sk4_op_family_code`: the class codes `zzz`
///   and `gem` are not among the seven Fuel can emit.
/// - `rdc_cell_with_empty_reduce_set`: the vector is the canonical `ew2` cell
///   with its class code respelled `rdc`. Declaring an EMPTY reduced set on that
///   cell must derive the `ew2` positive, not the decline — and the two tokens
///   must differ by exactly the class code, or this test is passing for some
///   unrelated reason.
/// - `dequant_form_unknown_spelling` / `operand_sub_key_seven_parts`: the
///   sixth component is `dl`/`dc` and nothing else, in at most six parts.
#[test]
fn new_sk5_declines_are_unrepresentable_on_the_emit_path() {
    let c = corpus();
    let emittable: BTreeSet<&str> = [
        CellClass::DenseContraction,
        CellClass::IndexedAccess,
        CellClass::Reduction,
        CellClass::ElementwiseOne,
        CellClass::ElementwiseTwo,
        CellClass::ElementwiseThree,
        CellClass::Other,
    ]
    .into_iter()
    .map(CellClass::code)
    .collect();

    for name in ["unknown_cell_class", "retired_sk4_op_family_code"] {
        let v = decline(&c, name);
        assert_eq!(field(v, "decline"), "UnknownCellClass");
        let class = field(v, "token").split('|').nth(1).unwrap();
        assert!(
            !emittable.contains(class),
            "`{name}`: class code `{class}` is one Fuel can emit"
        );
    }

    // rdc + empty reduce set.
    let v = decline(&c, "rdc_cell_with_empty_reduce_set");
    assert_eq!(field(v, "decline"), "RdcWithoutReduce");
    let canonical = field(positive(&c, "elementwise_binary_canonical"), "token");
    assert_eq!(
        field(v, "token"),
        canonical.replacen("|ew2|", "|rdc|", 1),
        "corpus invariant this test leans on has changed"
    );
    let ops = io(vec![
        f32c(&[128, 256]),
        f32c(&[128, 256]),
        f32c(&[128, 256]),
    ]);
    let emitted = derive_structure_key_token(
        &DeclaredFacts::reduction(ReduceAxes::Keepdim(0)),
        &ops,
        "cuda:sm89",
        None,
    )
    .expect("an empty declared reduced set must still derive a token");
    assert_ne!(
        emitted,
        field(v, "token"),
        "emitted rdc with a `-` reduce field"
    );
    assert_eq!(
        emitted, canonical,
        "an empty reduced set derives by shape (ew2)"
    );

    // dequant_form spelling and sub-key arity.
    for name in [
        "dequant_form_unknown_spelling",
        "operand_sub_key_seven_parts",
    ] {
        let v = decline(&c, name);
        let token = field(v, "token");
        let first_sub_key = token.split('|').nth(7).unwrap().split(';').next().unwrap();
        let parts: Vec<&str> = first_sub_key.split('/').collect();
        let bad_spelling = parts.get(5).is_some_and(|p| *p != "dl" && *p != "dc");
        let too_many = parts.len() > 6;
        assert!(
            bad_spelling || too_many,
            "`{name}`: the vector's first sub-key `{first_sub_key}` is not the bad form"
        );
    }
    for form in [DequantForm::Linear, DequantForm::Codebook] {
        let mut ops = io(vec![
            f32c(&[128, 256]),
            f32c(&[128, 256]),
            f32c(&[128, 256]),
        ]);
        ops[0] = ops[0].clone().with_dequant_form(form);
        let token = derive_structure_key_token(&DeclaredFacts::none(), &ops, "cuda:sm89", None)
            .expect("derives");
        let sub_key = token.split('|').nth(7).unwrap().split(';').next().unwrap();
        let parts: Vec<&str> = sub_key.split('/').collect();
        assert_eq!(parts.len(), 6, "{sub_key}");
        assert!(matches!(parts[5], "dl" | "dc"), "{sub_key}");
    }
}

/// The two `dequant_form_*` vectors are `i4` cells Fuel cannot construct. The
/// PLACEMENT of the `dl`/`dc` component is nonetheless measurable against KISS's
/// own bytes: build the **f32-weight analog** of each cell — identical operands,
/// roles and dequant form — and assert its token equals the vector's token with
/// exactly the two `i4` dtype coordinates respelled `f32`.
///
/// ⚠️ What this does and does not show. It shows the sixth component, its
/// position (after `flip`, on operand 0 only), its spelling and the class/
/// contraction fields around it agree with KISS byte for byte. It does NOT show
/// Fuel can key an `i4` cell, and the respelling is a substitution THIS TEST
/// performs on KISS's bytes — so a divergence that is specific to `i4`
/// (e.g. a sub-byte vector-width rule) is invisible here. Each substitution is
/// asserted to apply exactly once, so a vector whose shape drifted fails
/// instead of "matching" a no-op.
#[test]
fn dequant_form_vectors_match_modulo_the_i4_coordinate() {
    let c = corpus();
    for (name, form) in [
        ("dequant_form_linear_weight", DequantForm::Linear),
        ("dequant_form_codebook_weight", DequantForm::Codebook),
    ] {
        let vector = field(positive(&c, name), "token");
        // The vector token with the two i4 coordinates respelled f32.
        assert_eq!(vector.matches("|i4|").count(), 1, "{name}: {vector}");
        assert_eq!(vector.matches("/i4/").count(), 1, "{name}: {vector}");
        let analog_expected = vector
            .replacen("|i4|", "|f32|", 1)
            .replacen("/i4/", "/f32/", 1);

        let mut ops = io(vec![
            f32c(&[8, 4096]),
            f32c(&[4096, 4096]),
            f32c(&[8, 4096]),
        ]);
        ops[0] = ops[0].clone().with_dequant_form(form);
        let got = derive_structure_key_token(
            &DeclaredFacts::contraction(GemCell {
                weight_dtype: DType::F32,
                acc_dtype: DType::F32,
                out_dtype: DType::BF16,
                ..gem_f32(8, 4096, 4096)
            }),
            &ops,
            "cuda:sm90",
            None,
        )
        .expect("the f32 analog must derive");
        assert_eq!(got, analog_expected, "{name}");
    }
    // Control: the two forms differ from each other in the vector itself, so
    // the loop above is not comparing one string to itself.
    assert_ne!(
        field(positive(&c, "dequant_form_linear_weight"), "token"),
        field(positive(&c, "dequant_form_codebook_weight"), "token"),
    );
}

/// sk6 (§6.6-0022): the optional trailing `/q<sdt>` scale-dtype coordinate of a
/// quantized `ctr` cell. The two published positive vectors are `i4` cells and so
/// are excluded (GAP-097); this measures them against KISS's own bytes **modulo
/// the i4 coordinate**, the same way the `dequant_form` vectors are, and asserts
/// each substitution applied exactly once.
#[test]
fn quantized_scale_vectors_match_modulo_the_i4_coordinate() {
    let c = corpus();
    for (name, scale) in [
        ("quantized_ctr_scale_f8e8m0", DType::F8E8M0),
        ("quantized_ctr_scale_f16", DType::F16),
    ] {
        let vector = field(positive(&c, name), "token");
        assert_eq!(vector.matches("|i4|").count(), 1, "{name}: {vector}");
        assert_eq!(vector.matches("/i4/").count(), 1, "{name}: {vector}");
        let analog_expected = vector
            .replacen("|i4|", "|f32|", 1)
            .replacen("/i4/", "/f32/", 1);
        let ops = io(vec![
            f32c(&[8, 4096]),
            f32c(&[4096, 4096]),
            f32c(&[8, 4096]),
        ]);
        let got = derive_structure_key_token(
            &DeclaredFacts::contraction(GemCell {
                weight_dtype: DType::F32,
                acc_dtype: DType::F32,
                out_dtype: DType::BF16,
                scale_dtype: Some(scale),
                ..gem_f32(8, 4096, 4096)
            }),
            &ops,
            "cuda:sm90",
            None,
        )
        .expect("the f32 analog must derive");
        assert_eq!(got, analog_expected, "{name}");
    }
    // Control: the two scale dtypes key DIFFERENT tokens (the point of sk6: an
    // `f8e8m0` and an `f16` scale decode the same stored values to different
    // numbers), and a cell with no designated scale operand omits the
    // coordinate entirely.
    let derive = |scale: Option<DType>| {
        derive_structure_key_token(
            &DeclaredFacts::contraction(GemCell {
                scale_dtype: scale,
                ..gem_f32(8, 4096, 4096)
            }),
            &io(vec![
                f32c(&[8, 4096]),
                f32c(&[4096, 4096]),
                f32c(&[8, 4096]),
            ]),
            "cuda:sm90",
            None,
        )
    };
    let none = derive(None).unwrap();
    let e8 = derive(Some(DType::F8E8M0)).unwrap();
    let f16 = derive(Some(DType::F16)).unwrap();
    assert!(!none.contains("/q"), "{none}");
    assert_eq!(e8, format!("{none}/qf8e8m0"));
    assert_eq!(f16, format!("{none}/qf16"));
    assert_ne!(e8, f16);
    // The reserved `f8e6m2` scale is the published `scale_dtype_reserved`
    // decline: Fuel declines the whole derivation rather than emit it.
    assert_eq!(
        field(decline(&c, "scale_dtype_reserved"), "decline"),
        "ReservedDtype"
    );
    assert!(derive(Some(DType::F8E6M2)).is_none());
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
    let redundant = decline(&c, "redundant_acc_mp_all_default");
    let field_absent = positive(&c, "reduction_all_axes");

    let ops = || io(vec![f32c(&[4, 8]), f32c(&[1, 1])]);
    let token = cell_acc_mp(
        DeclaredFacts::reduction(ReduceAxes::All),
        ops(),
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
/// *current* version prefix (§6.1-0004) — and, new at sk5, a **reserved**
/// spelling (`f8e6m2`) that `fuel_ir`'s own table still spells.
#[test]
fn fuel_emits_only_recognized_sk5_dtype_spellings() {
    let c = corpus();
    let recognized: BTreeSet<String> = strs(&c, "dtype_recognition_set").into_iter().collect();
    let reserved: BTreeSet<String> = strs(&c, "reserved_dtypes").into_iter().collect();

    let emitted: BTreeSet<String> = DType::ALL
        .iter()
        .filter_map(|&dt| sk6_dtype_token(dt))
        .map(String::from)
        .collect();

    let expected: BTreeSet<String> = FUEL_SK5_SPELLINGS.iter().map(|s| s.to_string()).collect();
    assert_eq!(emitted, expected, "Fuel's sk5 spelling set moved");

    for tok in &emitted {
        assert!(
            recognized.contains(tok),
            "`{tok}` is outside the closed sk5 vocabulary"
        );
        assert!(
            !reserved.contains(tok),
            "`{tok}` is RESERVED at sk5 and must never be emitted"
        );
    }
    // Non-vacuity, and the discrimination check: the set is neither empty nor
    // everything, so `is-a-subset` is a real constraint here. 15 -> 14 at sk5:
    // `f8e6m2` left the emitted set.
    assert_eq!(emitted.len(), 14);
    assert!(emitted.len() < recognized.len());
}

/// **The corpus's reserved set, applied to Fuel's dtype table.** For every
/// reserved token KISS publishes that Fuel has a `DType` for, the deriver must
/// not spell it — even where `fuel_ir`'s own table does.
///
/// This is the cross-check that ties the vendored DATA to the deriver's
/// wrapper: the hard-coded `F8E6M2` arm in `sk6_dtype_token` is a local fact,
/// and without this test a future reserved token added upstream would be
/// silently emitted through `fuel_ir::sk4_token`'s `Some`. The `fuel_ir`
/// disagreement is asserted too, so the day `fuel_ir` itself reserves the
/// token (closing the GAP row that records the split) this fails and says so.
#[test]
fn every_reserved_token_with_a_fuel_dtype_is_declined_by_the_deriver() {
    let c = corpus();
    let mut applied = 0usize;
    for tok in strs(&c, "reserved_dtypes") {
        let fuel_dtype = DType::ALL
            .iter()
            .copied()
            .find(|&dt| fuel_ir::sk4_token(dt) == Some(tok.as_str()));
        match fuel_dtype {
            Some(dt) => {
                applied += 1;
                assert_eq!(
                    sk6_dtype_token(dt),
                    None,
                    "reserved token `{tok}` ({dt:?}) is still spelled by the deriver"
                );
            }
            None => {
                // No Fuel dtype spells it (the `fnuz` variants): unrepresentable.
                assert!(
                    DType::ALL
                        .iter()
                        .all(|&dt| sk6_dtype_token(dt) != Some(tok.as_str())),
                    "reserved token `{tok}` is spelled by some Fuel dtype"
                );
            }
        }
    }
    // Non-vacuity: at least one reserved token really is a Fuel dtype
    // (`f8e6m2`), so the `Some` arm — the one that bites — was exercised.
    assert!(applied >= 1, "no reserved token mapped to a Fuel dtype");
}

/// KISS's `target_match_vectors` (§6.8-0002, new in this artifact) test a
/// capability-target MATCHER. This leg derives keys and treats the target as an
/// opaque passthrough coordinate (the vulkan/cuda vectors above pin that), so
/// it does not exercise a matcher. Recorded as a count so growth is noticed and
/// so "not exercised" is a stated fact, not an absence nobody wrote down.
#[test]
fn target_match_vectors_are_recorded_as_not_exercised_by_this_leg() {
    let c = corpus();
    assert_eq!(
        vectors(&c, "target_match_vectors").len(),
        8,
        "KISS's target-match axis changed size; decide whether this leg now exercises it"
    );
}
