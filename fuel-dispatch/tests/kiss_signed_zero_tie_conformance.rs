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

/// **§6.13 conformance for the ±0 ties. This ASSERTS, and #67 is why it can.**
///
/// ⚠️ **HISTORY, because a future reader will otherwise re-litigate it.** This
/// arm REPORTED rather than asserted until #67 (`4ffd3a34`), because the old
/// code DELEGATED the tie to `f32::min`, whose ±0 behaviour Rust disclaims.
/// Measured then: the answer varied WITHIN ONE BINARY at one optimisation level
/// — `#[inline(never)]` gave one operand, `black_box` the other, and literal
/// operands were constant-folded to a third answer that never executed. Three
/// people measured it and got three results, all correct in their own context.
///
/// **#67 replaced that with an explicit `a >= b` select — §6.13's
/// `cmp_ge(a,b) -> a`.** ⚠️ **THAT is the licence for asserting here, and "it
/// comes out right today" is not: an assertion on an explicit select cannot be
/// re-decided by a compiler upgrade, and one on a delegated intrinsic can. The
/// two justifications are indistinguishable in a green test and diverge the
/// first time someone bumps the toolchain.**
#[test]
fn discriminating_vectors_conform_to_kiss_613() {
    let vs = load();
    let mut evaluated = 0usize;
    let mut nonconforming = Vec::new();

    for v in vs.iter().filter(|v| v.signs_differ) {
        evaluated += 1;
        let got = run(v);
        if got != bits_of(&v.expected) {
            nonconforming.push(format!(
                "tcId {} {} {} -- got {:02x?} want {:02x?}",
                v.tc_id,
                v.op,
                v.dtype,
                got,
                bits_of(&v.expected)
            ));
        }
    }

    println!(
        "[kiss-6.13] {evaluated} discriminating - {} nonconforming",
        nonconforming.len()
    );
    assert_eq!(
        evaluated, 24,
        "evaluated {evaluated} discriminating vectors, expected 24 -- the corpus \
         or the classifier changed, and a clean result would mean nothing"
    );
    assert!(
        nonconforming.is_empty(),
        "{} vector(s) do NOT match §6.13:\n  {}\n\n\
         Fuel regressed on the ±0 tie. #67 made it an explicit `a >= b` select; \
         if this fires, either that select changed or something now reaches the \
         old delegated path.",
        nonconforming.len(),
        nonconforming.join("\n  ")
    );
}

/// **A check on the FIXTURE, not on Fuel — and it is deliberately NOT a second
/// opinion about the implementation.**
///
/// ⚠️ Measured: for all 24 discriminating vectors the corpus's `expected` IS
/// operand `a`. So *"Fuel matches `expected`"* and *"Fuel returns operand a"*
/// are THE SAME FACT, and asserting both against Fuel would be one check wearing
/// two hats — the shape where two agreeing artifacts read as two pieces of
/// evidence.
///
/// What is genuinely independent is asserting the RULE against the CORPUS: if
/// the vendored `expected` values ever stopped encoding a-bias, the arm above
/// would faithfully verify Fuel against a wrong oracle and pass.
#[test]
fn the_corpus_expected_values_encode_a_bias() {
    let vs = load();
    let disagreeing: Vec<u64> = vs
        .iter()
        .filter(|v| v.signs_differ)
        .filter(|v| v.expected != v.a)
        .map(|v| v.tc_id)
        .collect();
    assert!(
        disagreeing.is_empty(),
        "corpus `expected` is not operand `a` for tcIds {disagreeing:?} -- §6.13 \
         specifies a-bias on a ±0 tie, so either the vendored corpus changed or \
         the spec did. The conformance arm above trusts these values; if they \
         are wrong it verifies Fuel against a wrong oracle and passes."
    );
}
