// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel × KISS `structure_key` **derivation** vectors: Fuel's sk5 deriver
//! against the 36 cell-carrying vectors in
//! `fixtures/kiss-corpus/structure_key_derivation_vectors.json`.
//!
//! **Provenance (a recorded stamp, not an enforced check):** KISS `60c375c`
//! (conformance 0.2.30, KISS#566), `conformance/corpus/structure_key_derivation_vectors.json`,
//! blob `697aa543` (36 vectors; `git rev-parse origin/main:<path>` at KISS `0ffd6af`),
//! vendored 2026-10-15 with `git show origin/main:<path>` (a commit object, never a
//! working tree). It REPLACES the `ae16d2a7` copy (blob `299811b1`, 33 vectors):
//! the only change is three vectors for the innermost active axis
//! (`innermost_axis_extent_one_is_axis_rank_minus_1`,
//! `innermost_axis_extent_eight_after_unit_axis`,
//! `innermost_axis_extent_zero_is_axis_rank_minus_1`; KISS-CLASSIFY-6.3-0011 and
//! 6.5-0012), all of which Fuel's deriver passes under both feeding disciplines
//! (including `[8,0]`). The schema version and the vector count are asserted
//! below; the bytes are not hashed.
//!
//! Unlike `kiss_structure_key_byte_match.rs` (whose cells are hand-built Rust
//! literals keyed by vector name), every vector here CARRIES its cell
//! (operands in call order, index slot, reduced axes, accumulator, math
//! fidelity), so the cell is built mechanically from the JSON and the deriver
//! is asked what it produces. Nothing in this file names a vector except the
//! pinned outcome tables, so a vector added upstream with no pin fails loudly.
//!
//! # What is measured
//!
//! Each vector gets an outcome under TWO feeding disciplines, because KISS says
//! canonicalization is "the deriver's job" while Fuel's
//! `derive_structure_key_token` documents that it TAKES canonical order and
//! declines anything else:
//!
//! - **as-given** — operands passed in the vector's CALL order and the index
//!   slot unmapped. This is the literal reading of the vectors.
//! - **canonicalized** — the harness does the §6.6-0014 stable partition
//!   (inputs and in-place operands in call order, then outputs) and maps the
//!   index slot, which is what a Fuel caller must do today.
//!
//! Outcomes: `PASS` (bytes equal), `FAIL` (different bytes), `NOT-EXPRESSIBLE`
//! (Fuel has no way to state the cell, with the reason), `DECLINE-DIFFERS`
//! (a decline was expected and a token came out, or the reverse). A decline's
//! KISS reason code (`ReservedDtype`, ...) IS compared, through
//! `derive_structure_key_token_typed` (GAP-354); `REASON-DIFFERS` means both
//! declined with different reasons.
//!
//! # Honest limits
//!
//! - Only the 10 vectors carrying `spec_golden` reproduce an Appendix A.1 token
//!   whose bytes were hand-derived in the spec text; the other 17 token vectors
//!   are KISS's own deriver's output, so a PASS there is agreement with another
//!   deriver, not with the spec.
//! - `decline_contraction_out_of_scope` is not derived: the batch role is
//!   undefined upstream (KISS#543), and a `GemCell` needs M/N/K the vector does
//!   not carry.
//! - A token-level match cannot see derivation: a deriver reading a different
//!   fact that agrees on every sampled cell passes.

#![cfg(feature = "telemetry")]

use fuel_dispatch::telemetry::structure_key::FdxOperandDesc;
use fuel_dispatch::telemetry::structure_key_derive::{
    AccMp, CellOperand, DeclaredFacts, DequantForm, GemMathPrecision, ReduceAxes,
    derive_structure_key_token_typed,
};
use fuel_ir::{DType, Layout, Shape};
use std::collections::BTreeMap;

const VECTORS: &str = include_str!("../fixtures/kiss-corpus/structure_key_derivation_vectors.json");

/// Fuel's `DType` for a KISS dtype spelling, or `None` if Fuel has no such
/// dtype (`i4`, `u4`, `b1`, `u16`, `u64`, the complex tokens, the `fnuz` pair).
fn dtype_of(token: &str) -> Option<DType> {
    Some(match token {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        "i8" => DType::I8,
        "i16" => DType::I16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "u8" => DType::U8,
        "u32" => DType::U32,
        "bool" => DType::Bool,
        "f8e4m3fn" => DType::F8E4M3,
        "f8e5m2" => DType::F8E5M2,
        "f8e8m0" => DType::F8E8M0,
        "f8e6m2" => DType::F8E6M2,
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Pass,
    /// Different bytes: `(got, want)`.
    Fail(String, String),
    /// A decline was expected and a token came out.
    DeclineDiffers(String),
    /// A token was expected and the deriver declined.
    UnexpectedDecline(String),
    /// Both declined, with different reasons: `(got, want)`.
    ReasonDiffers(String, String),
    NotExpressible(String),
}

impl Outcome {
    fn tag(&self) -> &'static str {
        match self {
            Outcome::Pass => "PASS",
            Outcome::Fail(..) => "FAIL",
            Outcome::DeclineDiffers(_) => "DECLINE-DIFFERS",
            Outcome::UnexpectedDecline(_) => "UNEXPECTED-DECLINE",
            Outcome::ReasonDiffers(..) => "REASON-DIFFERS",
            Outcome::NotExpressible(_) => "NOT-EXPRESSIBLE",
        }
    }
}

struct Built {
    facts: DeclaredFacts,
    operands: Vec<CellOperand>,
    target: String,
    acc_mp: Option<AccMp>,
}

/// The `shape`/`strides` of one operand, from the vector's JSON arrays.
fn layout_of(o: &serde_json::Value) -> Result<Layout, String> {
    let extents: Vec<usize> = o["extents"]
        .as_array()
        .ok_or("extents")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let strides: Vec<isize> = o["strides"]
        .as_array()
        .ok_or("strides")?
        .iter()
        .map(|x| x.as_i64().unwrap() as isize)
        .collect();
    Ok(Layout::new(
        Shape::from_dims(&extents),
        strides.into_iter().collect(),
        0,
    ))
}

/// An operand's `use` as a `CellOperand` role.
fn with_role(use_: &str, desc: FdxOperandDesc) -> Result<CellOperand, String> {
    match use_ {
        "in" => Ok(CellOperand::input(desc)),
        "out" => Ok(CellOperand::output(desc)),
        "inout" => Ok(CellOperand::in_place(desc)),
        other => Err(format!("unknown operand use `{other}`")),
    }
}

/// An operand's optional `dequant_form`.
fn with_dequant(cell: CellOperand, form: Option<&str>) -> Result<CellOperand, String> {
    match form {
        None => Ok(cell),
        Some("dl" | "linear") => Ok(cell.with_dequant_form(DequantForm::Linear)),
        Some("dc" | "codebook") => Ok(cell.with_dequant_form(DequantForm::Codebook)),
        Some(other) => Err(format!("unknown dequant_form `{other}`")),
    }
}

/// The descriptor of one operand: layout, dtype, and the stated alignment.
fn desc_of(o: &serde_json::Value) -> Result<FdxOperandDesc, String> {
    let dtok = o["dtype"].as_str().ok_or("operand dtype")?;
    let dtype = dtype_of(dtok)
        .ok_or_else(|| format!("no Fuel DType for `{dtok}` (GAP-097 absent surface)"))?;
    let mut desc = FdxOperandDesc::from_layout(&layout_of(o)?, dtype);
    // The vector states the base alignment outright; `from_layout` would model
    // 256 for offset 0, which is the only thing it cannot be told.
    desc.align_bytes = o["alignment"].as_u64().ok_or("alignment")? as u32;
    Ok(desc)
}

/// One operand of a vector, as Fuel's `CellOperand`.
fn operand_of(o: &serde_json::Value) -> Result<CellOperand, String> {
    let cell = with_role(o["use"].as_str().ok_or("use")?, desc_of(o)?)?;
    with_dequant(cell, o["dequant_form"].as_str())
}

/// The declared reduced axes as Fuel's `Keepdim` bitmask (bit `d` = frame axis `d`).
fn reduce_axes_of(v: &serde_json::Value) -> Result<Option<ReduceAxes>, String> {
    let axes = v["reduced_axes"].as_array().ok_or("reduced_axes")?;
    if axes.is_empty() {
        return Ok(None);
    }
    let mut mask: u32 = 0;
    for a in axes {
        let bit = a.as_u64().ok_or("axis")? as u32;
        // An axis past the mask width cannot be stated at all.
        if bit >= 8 {
            return Err(format!("reduced axis {bit} does not fit Fuel's u8 mask"));
        }
        mask |= 1 << bit;
    }
    Ok(Some(ReduceAxes::Keepdim(mask as u8)))
}

/// §6.6-0014 stable partition (everything that is not an output, then outputs),
/// returning the new order and the index slot mapped through it.
fn canonicalized(
    cells: Vec<(usize, CellOperand)>,
    index: Option<usize>,
) -> (Vec<(usize, CellOperand)>, Option<usize>) {
    use fuel_dispatch::telemetry::structure_key_derive::OperandRole;
    let (mut front, back): (Vec<_>, Vec<_>) = cells
        .into_iter()
        .partition(|(_, c)| c.role != OperandRole::Output);
    front.extend(back);
    let index = index.and_then(|slot| front.iter().position(|(call, _)| *call == slot));
    (front, index)
}

/// The `(acc + mp)` coordinate: accumulator null = the compute dtype.
fn acc_mp_of(v: &serde_json::Value, compute: DType) -> Result<AccMp, String> {
    let math_precision = match v["math_fidelity"].as_str().ok_or("math_fidelity")? {
        "st" => GemMathPrecision::BitStable,
        "rm" => GemMathPrecision::ReducedMantissa,
        other => return Err(format!("unknown math_fidelity `{other}`")),
    };
    let acc_dtype = match v["accumulator"].as_str() {
        None => compute,
        Some(t) => dtype_of(t).ok_or_else(|| format!("no Fuel DType for accumulator `{t}`"))?,
    };
    Ok(AccMp {
        acc_dtype,
        math_precision,
    })
}

/// A contraction is out of scope for a derivation vector (no M/N/K roles).
fn check_in_scope(v: &serde_json::Value) -> Result<(), String> {
    if v["contraction_declared"].as_bool() == Some(true) {
        return Err(
            "contraction declared, but the vector carries no M/N/K/batch roles \
             (batch role undefined upstream, KISS#543)"
                .to_string(),
        );
    }
    Ok(())
}

/// The operands in CALL order, each with its call-order slot.
fn cells_of(v: &serde_json::Value) -> Result<Vec<(usize, CellOperand)>, String> {
    let ops = v["operands"].as_array().ok_or("no operands")?;
    ops.iter()
        .enumerate()
        .map(|(i, o)| operand_of(o).map(|c| (i, c)))
        .collect()
}

/// The declared facts: index slot and reduced axes.
fn facts_of(v: &serde_json::Value, index: Option<usize>) -> Result<DeclaredFacts, String> {
    let mut facts = index.map_or_else(DeclaredFacts::none, DeclaredFacts::indexed);
    facts.reduced_axes = reduce_axes_of(v)?;
    Ok(facts)
}

/// Build Fuel's cell from one vector. `canonicalize` applies the §6.6-0014
/// stable partition and maps the call-order index slot through it.
fn build(v: &serde_json::Value, canonicalize: bool) -> Result<Built, String> {
    check_in_scope(v)?;
    let mut cells = cells_of(v)?;
    let mut index = v["index_operand"].as_u64().map(|x| x as usize);
    if canonicalize {
        (cells, index) = canonicalized(cells, index);
    }
    let facts = facts_of(v, index)?;
    let operands: Vec<CellOperand> = cells.into_iter().map(|(_, c)| c).collect();
    let compute = operands
        .first()
        .map(|o| o.desc.dtype)
        .ok_or("no operands")?;
    Ok(Built {
        facts,
        operands,
        target: v["target"].as_str().ok_or("target")?.to_string(),
        acc_mp: Some(acc_mp_of(v, compute)?),
    })
}

fn evaluate(v: &serde_json::Value, canonicalize: bool) -> Outcome {
    let built = match build(v, canonicalize) {
        Ok(b) => b,
        Err(why) => return Outcome::NotExpressible(why),
    };
    // KISS (cell_note, `call_order_is_not_canonical_order`): a deriver whose input
    // contract is canonical order reports NOT-EXPRESSIBLE on call-order input,
    // "which is not a defect". Fuel's deriver takes canonical order (caller's
    // contract), so as-given call order is not a cell it can be asked.
    if !canonicalize {
        let mut seen_output = false;
        for o in &built.operands {
            use fuel_dispatch::telemetry::structure_key_derive::OperandRole;
            if o.role == OperandRole::Output {
                seen_output = true;
            } else if seen_output {
                return Outcome::NotExpressible(
                    "input contract is canonical order (KISS: not a defect)".to_string(),
                );
            }
        }
    }
    let got = derive_structure_key_token_typed(
        &built.facts,
        &built.operands,
        &built.target,
        built.acc_mp,
    );
    let want = v["value"].as_str().unwrap().to_string();
    match (v["expect"].as_str(), got) {
        (Some("token"), Ok(t)) if t == want => Outcome::Pass,
        (Some("token"), Ok(t)) => Outcome::Fail(t, want),
        (Some("token"), Err(why)) => {
            Outcome::UnexpectedDecline(format!("{}; want {want}", why.wire()))
        }
        (Some("decline"), Err(why)) if why.wire() == want => Outcome::Pass,
        (Some("decline"), Err(why)) => Outcome::ReasonDiffers(why.wire().to_string(), want),
        (Some("decline"), Ok(t)) => Outcome::DeclineDiffers(t),
        (other, _) => panic!("unknown `expect` {other:?}"),
    }
}

fn load() -> Vec<serde_json::Value> {
    let doc: serde_json::Value = serde_json::from_str(VECTORS).expect("vectors parse");
    assert_eq!(
        doc["structure_key_schema_version"].as_u64(),
        Some(5),
        "schema version moved: re-measure"
    );
    doc["vectors"].as_array().expect("vectors").clone()
}

/// Vectors where a DECLINE is required and Fuel's deriver EMITS A TOKEN instead.
/// Pinned as a SET, both directions (a repair removes a name and must edit this
/// list in the same change; a regression adds one and fails here).
///
/// These are deriver-vs-spec divergences, the dangerous direction: the deriver's
/// own contract is "a typed decline, never a wrong token".
///
/// (An earlier revision also listed `decline_reserved_dtype`: it put a reserved
/// dtype at operand 1, which KISS then ruled is NOT a key position, so Fuel's
/// keying it was right and the vector was corrected, see the provenance note.)
const EMITS_WHERE_KISS_DECLINES: &[&str] = &[];

/// Vectors Fuel cannot state at all (an absent surface, not a mismatch).
const NOT_EXPRESSIBLE: &[&str] = &[
    // The batch role is undefined upstream (KISS#543) and the vector carries no
    // M/N/K extents for a `GemCell`.
    "decline_contraction_out_of_scope",
    // `i4` has no Fuel `DType` (GAP-097), so both the `/dl` dequant placement
    // and the bare sub-byte cell are unstatable.
    "quantized_operand_dequant_form",
    "sub_byte_dtype_i4",
    // The reserved dtype is `f8e4m3fnuz`: Fuel has no `DType` for the `fnuz`
    // layouts at all, so the cell cannot be built (the decline is by construction).
    "decline_reserved_dtype",
];

/// Fed in CALL order, Fuel's deriver declines because it documents that it takes
/// canonical order (`is_canonical_order`); KISS says canonicalization is the
/// deriver's job. Passes once the caller canonicalizes.
const NEEDS_CANONICAL_INPUT: &[&str] = &["call_order_is_not_canonical_order"];

fn rows() -> BTreeMap<String, (Outcome, Outcome, bool)> {
    let mut rows = BTreeMap::new();
    for v in &load() {
        let name = v["name"].as_str().unwrap().to_string();
        let golden = v.get("spec_golden").is_some_and(|g| !g.is_null());
        rows.insert(name, (evaluate(v, false), evaluate(v, true), golden));
    }
    rows
}

/// The measured report, printed so a human can read it.
#[test]
fn report_every_vector_under_both_feeding_disciplines() {
    let rows = rows();
    assert_eq!(rows.len(), 36, "vector count changed: re-pin the tables");
    for (name, (given, canon, golden)) in &rows {
        println!(
            "[kiss-deriv] {:<44} golden={:<5} as-given={:<18} canonicalized={}",
            name,
            golden,
            given.tag(),
            canon.tag()
        );
        for (label, o) in [("as-given", given), ("canonicalized", canon)] {
            match o {
                Outcome::Fail(got, want) => {
                    println!(
                        "[kiss-deriv]    {label}: got  {got}
[kiss-deriv]    {label}: want {want}"
                    )
                }
                Outcome::DeclineDiffers(t) => {
                    println!("[kiss-deriv]    {label}: expected a decline, got token {t}")
                }
                Outcome::UnexpectedDecline(want) => {
                    println!("[kiss-deriv]    {label}: declined, want {want}")
                }
                Outcome::ReasonDiffers(got, want) => {
                    println!("[kiss-deriv]    {label}: declined {got}, want {want}")
                }
                Outcome::NotExpressible(why) => println!("[kiss-deriv]    {label}: {why}"),
                Outcome::Pass => {}
            }
        }
    }
    let count = |f: &dyn Fn(&(Outcome, Outcome, bool)) -> &Outcome| {
        let mut m: BTreeMap<&'static str, usize> = BTreeMap::new();
        for r in rows.values() {
            *m.entry(f(r).tag()).or_default() += 1;
        }
        m
    };
    println!("[kiss-deriv] as-given:      {:?}", count(&|r| &r.0));
    println!("[kiss-deriv] canonicalized: {:?}", count(&|r| &r.1));
}

/// Every vector whose bytes the spec text itself pins (`spec_golden`) reproduces,
/// when fed canonically. A FAIL here is a deriver-vs-spec defect and not a
/// disagreement between two derivers.
#[test]
fn all_ten_spec_golden_vectors_byte_match_when_fed_canonically() {
    let rows = rows();
    let goldens: Vec<_> = rows.iter().filter(|(_, r)| r.2).collect();
    assert_eq!(goldens.len(), 10, "spec_golden count changed: re-measure");
    let bad: Vec<_> = goldens
        .iter()
        .filter(|(_, r)| r.1 != Outcome::Pass)
        .map(|(n, r)| format!("{n}: {:?}", r.1))
        .collect();
    assert!(bad.is_empty(), "golden vectors failed: {bad:?}");
}

/// The three outcome classes are pinned as SETS: a correct count can hide a wrong
/// distribution, so names (not numbers) are compared, in both directions.
#[test]
fn divergences_and_absent_surfaces_are_exactly_the_pinned_sets() {
    let rows = rows();
    let names = |f: &dyn Fn(&(Outcome, Outcome, bool)) -> bool| -> Vec<String> {
        rows.iter()
            .filter(|(_, r)| f(r))
            .map(|(n, _)| n.clone())
            .collect()
    };
    let sorted = |xs: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = xs.iter().map(|s| s.to_string()).collect();
        v.sort();
        v
    };

    // CONTROL: the harness can see a mismatch at all. The corpus and Fuel agree
    // on every decline now, so an empty set is a finding and not evidence of an
    // inert harness only if a deliberately WRONG expectation is reported. Take a
    // passing token vector and corrupt its expected token; and take a passing
    // decline vector and corrupt its expected reason.
    let vs = load();
    let tok = vs.iter().find(|v| v["expect"] == "token").unwrap();
    let mut wrong = tok.clone();
    wrong["value"] = serde_json::Value::String("sk5|not-the-token".to_string());
    assert!(matches!(evaluate(&wrong, true), Outcome::Fail(..)));
    let dec = vs
        .iter()
        .find(|v| v["name"] == "decline_collapsed_reduction_output")
        .unwrap();
    let mut wrong = dec.clone();
    wrong["value"] = serde_json::Value::String("ReservedDtype".to_string());
    assert!(matches!(evaluate(&wrong, true), Outcome::ReasonDiffers(..)));

    assert_eq!(
        names(&|r| matches!(r.1, Outcome::DeclineDiffers(_))),
        sorted(EMITS_WHERE_KISS_DECLINES),
        "the set of vectors where Fuel EMITS a token and KISS requires a decline changed"
    );
    assert_eq!(
        names(&|r| matches!(r.1, Outcome::NotExpressible(_))),
        sorted(NOT_EXPRESSIBLE),
        "the set of vectors Fuel cannot state changed"
    );
    assert_eq!(
        names(&|r| r.0 != r.1 && r.1 == Outcome::Pass),
        sorted(NEEDS_CANONICAL_INPUT),
        "the set of vectors that pass only when the caller canonicalizes changed"
    );
    // Everything else passes in BOTH disciplines.
    let accounted =
        EMITS_WHERE_KISS_DECLINES.len() + NOT_EXPRESSIBLE.len() + NEEDS_CANONICAL_INPUT.len();
    let both_pass = rows
        .values()
        .filter(|r| r.0 == Outcome::Pass && r.1 == Outcome::Pass)
        .count();
    assert_eq!(both_pass + accounted, 36, "an unpinned outcome exists");
}
