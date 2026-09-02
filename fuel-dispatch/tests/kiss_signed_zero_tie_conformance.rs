// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Runs the vendored KISS §6.13 signed-zero tie vectors against Fuel's
//! PRODUCTION min/max, raw-bit compared.**
//!
//! The 48 vectors in `fixtures/kiss-corpus/ops-minmax-signed-zero.json` have
//! been in this repository, parsed on every build, wired to nothing. The reader
//! (`kiss_corpus.rs`) is deliberately dormant because `corpus_verdict`'s seam is
//! a JIT-*candidate-adoption* seam — no candidate output crosses it and its
//! `seed` selects a random probe disjoint from these fixed inputs. That
//! dormancy is correct and this file does not disturb it.
//!
//! **Testing our own ops against the vectors needs no seam at all.** Read the
//! JSON, run the op, compare bits.
//!
//! # Why this is a live ratchet and not `#[ignore]` or a pinned expectation
//!
//! An `#[ignore]`d test is a dormant instrument, and a dormant instrument is
//! precisely why 48 executable vectors sat unfired — fixing that with a second
//! dormant instrument repeats the shape one level up. It also never fails, so
//! nothing forces its retirement.
//!
//! Asserting *current* behaviour with a FIXME is worse: it would encode
//! `max(+0,-0) = -0` as EXPECTED, which is a false sentence, and when the
//! a-bias fix lands the test goes red — inviting the next person to "fix" it
//! back to the defect. **A trap laid for whoever does the right thing.**
//!
//! So: a KNOWN-FAILING SET, checked on every run, in three arms. It states a
//! true sentence (*these vectors fail today*), it cannot survive the fix, and
//! it is a SET rather than a COUNT because a correct total can hide a wrong
//! distribution — eight repaired and eight broken keeps a count at sixteen and
//! reports clean.
//!
//! # The corpus contains its own positive control
//!
//! Measured: 24 of the 48 vectors have `a` and `b` of DIFFERING sign, where an
//! a-biased and a b-biased implementation disagree. The other 24 are `(+0,+0)`
//! and `(-0,-0)` pairs which **must pass under either bias**. A failure there is
//! a different defect, not the tie bias, and this file refuses to absorb it into
//! the known-failing set.
//!
//! # Why four corpus ops map onto two Fuel ops
//!
//! §6.13: the four minmax decompositions "share the identical innermost select
//! — `cmp_ge(a,b) -> a` for `max_prop`/`fmax_ieee`, `cmp_le(a,b) -> a` for
//! `min_prop`/`fmin_ieee` — and differ only in their NaN arms". **Measured on
//! this corpus: 0 NaN inputs, and 96 of 96 inputs are ±0.** So the NaN arms
//! never differentiate here and both spellings test the same tie rule.

use fuel_cpu_backend::chassis::binary::{BinaryOp, Maximum, Minimum};
use half::bf16;

/// tcIds known to fail today, per **GAP-271**: Fuel is b-biased on a ±0 tie,
/// KISS §6.13 specifies a-bias.
///
/// **MEASURED, never predicted** — a tcId listed here that actually passes
/// fires the `PASSES + in set` arm below. Run first, list second.
///
/// This list may only SHRINK. Fixing the tie bias empties it, and the arm goes
/// red until it is emptied — the exemption cannot outlive the defect.
const KNOWN_FAILING_GAP271: &[u64] = &[
    1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34, 37, 38, 41, 42, 45, 46,
];

#[derive(Debug)]
struct Vector {
    tc_id: u64,
    op: String,
    dtype: String,
    a: String,
    b: String,
    expected: String,
    signs_differ: bool,
}

fn bits_of(hex: &str) -> Vec<u8> {
    hex.split_whitespace()
        .map(|b| u8::from_str_radix(b, 16).expect("hex byte"))
        .collect()
}

fn is_negative(hex: &str) -> bool {
    bits_of(hex)[0] & 0x80 != 0
}

fn load() -> Vec<Vector> {
    let json = include_str!("../fixtures/kiss-corpus/ops-minmax-signed-zero.json");
    let doc: serde_json::Value = serde_json::from_str(json).expect("corpus parses");
    doc["vectors"]
        .as_array()
        .expect("vectors array")
        .iter()
        .map(|v| {
            let a = v["inputs"][0]["bits"].as_str().expect("a bits").to_string();
            let b = v["inputs"][1]["bits"].as_str().expect("b bits").to_string();
            Vector {
                tc_id: v["tcId"].as_u64().expect("tcId"),
                op: v["op"].as_str().expect("op").to_string(),
                dtype: v["dtype"].as_str().expect("dtype").to_string(),
                signs_differ: is_negative(&a) != is_negative(&b),
                a,
                b,
                expected: v["expected"]["bits"]
                    .as_str()
                    .expect("expected")
                    .to_string(),
            }
        })
        .collect()
}

/// Run Fuel's production op on the vector's inputs, returning the result's raw
/// bytes in the corpus's big-endian spelling.
fn run(v: &Vector) -> Vec<u8> {
    let (a, b) = (bits_of(&v.a), bits_of(&v.b));
    let is_max = v.op.contains("max");
    match v.dtype.as_str() {
        "f32" => {
            let (x, y) = (
                f32::from_be_bytes(a.try_into().expect("4 bytes")),
                f32::from_be_bytes(b.try_into().expect("4 bytes")),
            );
            let r = if is_max {
                <Maximum as BinaryOp<f32>>::apply(x, y)
            } else {
                <Minimum as BinaryOp<f32>>::apply(x, y)
            };
            r.to_be_bytes().to_vec()
        }
        "f64" => {
            let (x, y) = (
                f64::from_be_bytes(a.try_into().expect("8 bytes")),
                f64::from_be_bytes(b.try_into().expect("8 bytes")),
            );
            let r = if is_max {
                <Maximum as BinaryOp<f64>>::apply(x, y)
            } else {
                <Minimum as BinaryOp<f64>>::apply(x, y)
            };
            r.to_be_bytes().to_vec()
        }
        "bf16" => {
            let (x, y) = (
                bf16::from_bits(u16::from_be_bytes(a.try_into().expect("2 bytes"))),
                bf16::from_bits(u16::from_be_bytes(b.try_into().expect("2 bytes"))),
            );
            let r = if is_max {
                <Maximum as BinaryOp<bf16>>::apply(x, y)
            } else {
                <Minimum as BinaryOp<bf16>>::apply(x, y)
            };
            r.to_bits().to_be_bytes().to_vec()
        }
        other => panic!("unhandled dtype {other} -- the corpus grew and this test did not"),
    }
}

#[test]
fn foundation_the_corpus_is_present_and_has_both_classes() {
    let vs = load();
    let disc = vs.iter().filter(|v| v.signs_differ).count();
    let same = vs.len() - disc;
    println!(
        "[kiss-6.13] {} vectors: {disc} discriminating, {same} same-sign",
        vs.len()
    );
    assert_eq!(vs.len(), 48, "corpus size changed -- re-derive the sets");
    assert!(
        disc > 0 && same > 0,
        "one class is empty ({disc} discriminating, {same} same-sign) -- with no \
         discriminating vectors this suite cannot see the tie bias, and with no \
         same-sign vectors it has no built-in control"
    );
}

/// The corpus's own positive control. `(+0,+0)` and `(-0,-0)` pairs return the
/// same answer under a-bias and b-bias alike, so these **must pass whatever the
/// tie convention is**. A failure here is a different defect and must not be
/// absorbed into the known-failing set.
#[test]
fn same_sign_vectors_pass_under_either_tie_convention() {
    let failures: Vec<String> = load()
        .iter()
        .filter(|v| !v.signs_differ)
        .filter(|v| run(v) != bits_of(&v.expected))
        .map(|v| format!("tcId {} {} {}", v.tc_id, v.op, v.dtype))
        .collect();
    assert!(
        failures.is_empty(),
        "SAME-SIGN vectors failed: {failures:?}\n\
         These return the same answer under either tie bias, so this is NOT the \
         GAP-271 signed-zero bias -- it is a different defect in min/max, and the \
         tie-bias story is incomplete. Stop and investigate rather than listing \
         these as known-failing."
    );
}

/// **Every entry in the set must name a real, DISCRIMINATING vector.**
///
/// Found by sabotage: listing a same-sign tcId was silently ignored, because the
/// ratchet below only iterates discriminating vectors — so an entry that names
/// nothing, or names a vector that cannot exhibit the tie bias, never fired any
/// arm. An exemption list whose entries nothing validates is the defect this
/// whole file exists to argue against, reproduced inside it.
#[test]
fn every_known_failing_entry_names_a_real_discriminating_vector() {
    let vs = load();
    let bogus: Vec<String> = KNOWN_FAILING_GAP271
        .iter()
        .filter_map(|id| match vs.iter().find(|v| v.tc_id == *id) {
            None => Some(format!("tcId {id}: not in the corpus at all")),
            Some(v) if !v.signs_differ => Some(format!(
                "tcId {id}: SAME-SIGN, so it cannot exhibit the tie bias and this                  entry exempts nothing"
            )),
            Some(_) => None,
        })
        .collect();
    assert!(
        bogus.is_empty(),
        "KNOWN_FAILING_GAP271 has entries that name no discriminating vector:
  {}",
        bogus.join(
            "
  "
        )
    );
}

/// The ratchet. Three arms, so the set can neither grow silently nor outlive
/// the defect.
#[test]
fn discriminating_vectors_match_the_known_failing_set() {
    let vs = load();
    let mut unexpected_fail = Vec::new();
    let mut unexpected_pass = Vec::new();
    let mut recorded = 0usize;

    for v in vs.iter().filter(|v| v.signs_differ) {
        let got = run(v);
        let want = bits_of(&v.expected);
        let listed = KNOWN_FAILING_GAP271.contains(&v.tc_id);
        match (got == want, listed) {
            (false, true) => recorded += 1,
            (true, true) => unexpected_pass.push(v.tc_id),
            (false, false) => unexpected_fail.push(format!(
                "tcId {} {} {} -- got {:02x?} want {:02x?}",
                v.tc_id, v.op, v.dtype, got, want
            )),
            (true, false) => {}
        }
    }

    println!(
        "[kiss-6.13] discriminating: {recorded} fail-as-recorded, {} unexpected fails, {} unexpected passes",
        unexpected_fail.len(),
        unexpected_pass.len()
    );
    if !unexpected_fail.is_empty() {
        println!("[kiss-6.13] tcIds to record if this is GAP-271:");
        for f in &unexpected_fail {
            println!("    {f}");
        }
    }

    assert!(
        unexpected_pass.is_empty(),
        "listed as known-failing but PASSES: {unexpected_pass:?}\n\
         Delete these tcIds from KNOWN_FAILING_GAP271. The list may only shrink, \
         and an entry that outlives its defect is an exemption with no subject."
    );
    assert!(
        unexpected_fail.is_empty(),
        "{} discriminating vector(s) fail and are NOT recorded:\n  {}\n\n\
         Either this is the GAP-271 b-bias and the tcIds belong in \
         KNOWN_FAILING_GAP271 with that reason, or it is a regression. Do not \
         add them without deciding which.",
        unexpected_fail.len(),
        unexpected_fail.join("\n  ")
    );
}
