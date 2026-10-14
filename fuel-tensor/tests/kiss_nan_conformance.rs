// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Runs the vendored KISS NaN-operand vectors through Fuel's lazy `Tensor`
//! API on CPU, raw-bit compared (computed NaNs by NaN-ness and quietness,
//! KISS-CONFORM-6.8-0010).**
//!
//! `fuel-dispatch` vendors and parses `ops-narrow-move-nan`,
//! `ops-narrow-select-nan` and `ops-transcendental-nan` (KISS `5b28bf51`) but
//! nothing executed them: `kiss_corpus` only pins their shape. This file is the
//! evaluator. It goes through `Tensor` -> optimizer -> CPU dispatch, i.e. the
//! PRODUCTION path, not a chassis function called by hand, so a kernel that is
//! right in isolation but bypassed (or rewritten) by the graph still fails here.
//!
//! # What is and is not exercised (counted, not asserted away)
//!
//! Every vector lands in exactly one of `pass`, `fail`, `not_exercised`, and the
//! three sum to the corpus size. `not_exercised` names WHY per vector:
//!
//! - `copysign`: Fuel has no `copysign` op (absent from `Op` and the builder).
//! - the two 8-bit float dtypes (`f8e5m2`, `f8e4m3fn`): the lazy `Tensor` has no
//!   constructor/realize pair for them on this path.
//!
//! A `not_exercised` vector is NOT a pass. The summary line prints the count.
//!
//! # Why a known-failing SET and not a count
//!
//! A correct total can hide a wrong distribution: one vector repaired and one
//! regressed keeps a count unchanged. The failing set is pinned by `(file,
//! tcId)`, in both directions: a vector that now FAILS and is not listed fails
//! the test, and a listed vector that now PASSES fails it too, so the list
//! cannot outlive the defect.

use fuel_ir::Shape;
use fuel_tensor::Device;
use fuel_tensor::lazy::Tensor;
use half::{bf16, f16};

const MOVE: &str =
    include_str!("../../fuel-dispatch/fixtures/kiss-corpus/ops-narrow-move-nan.json");
const SELECT: &str =
    include_str!("../../fuel-dispatch/fixtures/kiss-corpus/ops-narrow-select-nan.json");
const TRANSC: &str =
    include_str!("../../fuel-dispatch/fixtures/kiss-corpus/ops-transcendental-nan.json");
const MINMAX_NAN: &str =
    include_str!("../../fuel-dispatch/fixtures/kiss-corpus/ops-minmax-nan.json");
const ROUNDING: &str = include_str!("../../fuel-dispatch/fixtures/kiss-corpus/ops-rounding.json");

#[derive(Debug, Clone)]
struct Vector {
    file: &'static str,
    tc_id: u64,
    op: String,
    dtype: String,
    inputs: Vec<Vec<u8>>,
    expected: Vec<u8>,
    class: String,
    ulp_bound: u64,
    /// `"moved"` (exact-byte) or `"computed"` (KISS-CONFORM-6.8-0010: compared by
    /// NaN-ness and quietness only, never by sign or payload).
    nan_provenance: String,
}

fn bits_of(hex: &str) -> Vec<u8> {
    hex.split(|c: char| c.is_whitespace() || c == '\u{b7}')
        .filter(|s| !s.is_empty())
        .map(|b| u8::from_str_radix(b, 16).expect("hex byte"))
        .collect()
}

fn load(file: &'static str, json: &str) -> Vec<Vector> {
    let doc: serde_json::Value = serde_json::from_str(json).expect("corpus parses");
    doc["vectors"]
        .as_array()
        .expect("vectors array")
        .iter()
        .map(|v| Vector {
            file,
            tc_id: v["tcId"].as_u64().expect("tcId"),
            op: v["op"].as_str().expect("op").to_string(),
            dtype: v["dtype"].as_str().expect("dtype").to_string(),
            inputs: v["inputs"]
                .as_array()
                .expect("inputs")
                .iter()
                .map(|i| bits_of(i["bits"].as_str().expect("bits")))
                .collect(),
            expected: bits_of(v["expected"]["bits"].as_str().expect("expected bits")),
            class: v["class"].as_str().expect("class").to_string(),
            ulp_bound: v["ulp_bound"].as_u64().unwrap_or(0),
            nan_provenance: v["nan_provenance"].as_str().unwrap_or("").to_string(),
        })
        .collect()
}

fn all_vectors() -> Vec<Vector> {
    let mut v = load("ops-narrow-move-nan", MOVE);
    v.extend(load("ops-narrow-select-nan", SELECT));
    v.extend(load("ops-transcendental-nan", TRANSC));
    v.extend(load("ops-minmax-nan", MINMAX_NAN));
    v.extend(load("ops-rounding", ROUNDING));
    v
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail { got: Vec<u8>, want: Vec<u8> },
    NotExercised(&'static str),
}

/// Integer total-order distance between two float bit patterns (the corpus's
/// `ulp_metric`): map sign-magnitude to a monotone unsigned line, subtract.
fn ulp_distance(a: &[u8], b: &[u8]) -> u64 {
    fn key(bytes: &[u8]) -> u128 {
        let n = bytes.len() * 8;
        let mut v: u128 = 0;
        for byte in bytes {
            v = (v << 8) | u128::from(*byte);
        }
        let sign = 1u128 << (n - 1);
        if v & sign != 0 {
            (!v) & ((sign << 1) - 1)
        } else {
            v | sign
        }
    }
    let (x, y) = (key(a), key(b));
    x.abs_diff(y) as u64
}

/// `Some(is_quiet)` if `bytes` is a NaN of `dtype`, else `None`. Only the dtypes
/// the computed-NaN vectors use (`f32`, `f64`) are decoded.
fn nan_quiet(dtype: &str, bytes: &[u8]) -> Option<bool> {
    match (dtype, bytes) {
        ("f32", [a, b, c, d]) => {
            let v = u32::from_be_bytes([*a, *b, *c, *d]);
            (v & 0x7F80_0000 == 0x7F80_0000 && v & 0x007F_FFFF != 0).then_some(v & 0x0040_0000 != 0)
        }
        ("f64", b) if b.len() == 8 => {
            let v = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            (v & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && v & 0x000F_FFFF_FFFF_FFFF != 0)
                .then_some(v & 0x0008_0000_0000_0000 != 0)
        }
        _ => None,
    }
}

fn one() -> Shape {
    Shape::from_dims(&[1])
}

/// Big-endian operand bytes -> a fixed-size array.
fn bytes<const N: usize>(v: &Vector, i: usize) -> [u8; N] {
    v.inputs[i].clone().try_into().expect("operand width")
}

/// `neg` / `abs` on the 16-bit floats: sign-bit edits that must keep a NaN's payload.
fn run_sign_edit(v: &Vector) -> Option<Vec<u8>> {
    let dev = Device::cpu();
    let neg = v.op == "neg";
    match v.dtype.as_str() {
        "bf16" => {
            let x = bf16::from_bits(u16::from_be_bytes(bytes::<2>(v, 0)));
            let t = Tensor::from_bf16(vec![x], one(), &dev).unwrap();
            let r = if neg { t.neg() } else { t.abs() };
            Some(r.realize_bf16()[0].to_bits().to_be_bytes().to_vec())
        }
        "f16" => {
            let x = f16::from_bits(u16::from_be_bytes(bytes::<2>(v, 0)));
            let t = Tensor::from_f16(vec![x], one(), &dev).unwrap();
            let r = if neg { t.neg() } else { t.abs() };
            Some(r.realize_f16()[0].to_bits().to_be_bytes().to_vec())
        }
        _ => None,
    }
}

/// The four minmax ops, applied to a pair of same-graph tensors.
fn apply_minmax(op: &str, a: &Tensor, b: &Tensor) -> Tensor {
    match op {
        "max_prop" => a.maximum(b),
        "min_prop" => a.minimum(b),
        "fmax_ieee" => a.fmax_ieee(b),
        _ => a.fmin_ieee(b),
    }
    .unwrap()
}

/// `max_prop` / `min_prop` / `fmax_ieee` / `fmin_ieee` on bf16, f16, f32, f64.
fn run_minmax(v: &Vector) -> Option<Vec<u8>> {
    let dev = Device::cpu();
    let op = v.op.as_str();
    match v.dtype.as_str() {
        "bf16" => {
            let a = bf16::from_bits(u16::from_be_bytes(bytes::<2>(v, 0)));
            let b = bf16::from_bits(u16::from_be_bytes(bytes::<2>(v, 1)));
            let ta = Tensor::from_bf16(vec![a], one(), &dev).unwrap();
            let tb = ta.const_bf16_like(vec![b], one()).unwrap();
            Some(
                apply_minmax(op, &ta, &tb).realize_bf16()[0]
                    .to_bits()
                    .to_be_bytes()
                    .to_vec(),
            )
        }
        "f16" => {
            let a = f16::from_bits(u16::from_be_bytes(bytes::<2>(v, 0)));
            let b = f16::from_bits(u16::from_be_bytes(bytes::<2>(v, 1)));
            let ta = Tensor::from_f16(vec![a], one(), &dev).unwrap();
            let tb = ta.const_f16_like(vec![b], one()).unwrap();
            Some(
                apply_minmax(op, &ta, &tb).realize_f16()[0]
                    .to_bits()
                    .to_be_bytes()
                    .to_vec(),
            )
        }
        "f32" => {
            let a = f32::from_be_bytes(bytes::<4>(v, 0));
            let b = f32::from_be_bytes(bytes::<4>(v, 1));
            let ta = Tensor::from_f32(vec![a], one(), &dev).unwrap();
            let tb = ta.const_f32_like(vec![b], one()).unwrap();
            Some(
                apply_minmax(op, &ta, &tb).realize_f32()[0]
                    .to_bits()
                    .to_be_bytes()
                    .to_vec(),
            )
        }
        "f64" => {
            let a = f64::from_be_bytes(bytes::<8>(v, 0));
            let b = f64::from_be_bytes(bytes::<8>(v, 1));
            let ta = Tensor::from_f64(vec![a], one(), &dev).unwrap();
            let tb = ta.const_f64_like(vec![b], one()).unwrap();
            Some(
                apply_minmax(op, &ta, &tb).realize_f64()[0]
                    .to_bits()
                    .to_be_bytes()
                    .to_vec(),
            )
        }
        _ => None,
    }
}

/// `floor` / `ceil` / `trunc` / `round_even` on f32. Fuel's `round` is
/// round-half-to-even (IEEE roundeven), which is KISS's `round_even`; `trunc` is
/// Fuel's own graph expansion (GAP-300), not a primitive.
fn run_rounding(v: &Vector) -> Option<Vec<u8>> {
    if v.dtype != "f32" {
        return None;
    }
    let x = f32::from_be_bytes(bytes::<4>(v, 0));
    let t = Tensor::from_f32(vec![x], one(), &Device::cpu()).unwrap();
    let r = match v.op.as_str() {
        "floor" => t.floor(),
        "ceil" => t.ceil(),
        "trunc" => t.trunc(),
        _ => t.round(),
    };
    Some(r.realize_f32()[0].to_bits().to_be_bytes().to_vec())
}

/// `exp` / `log` / `sin` on f32 and f64.
fn run_transcendental(v: &Vector) -> Option<Vec<u8>> {
    let dev = Device::cpu();
    let op = v.op.as_str();
    match v.dtype.as_str() {
        "f32" => {
            let t =
                Tensor::from_f32(vec![f32::from_be_bytes(bytes::<4>(v, 0))], one(), &dev).unwrap();
            let r = match op {
                "exp" => t.exp(),
                "log" => t.log(),
                _ => t.sin(),
            };
            Some(r.realize_f32()[0].to_bits().to_be_bytes().to_vec())
        }
        "f64" => {
            let t =
                Tensor::from_f64(vec![f64::from_be_bytes(bytes::<8>(v, 0))], one(), &dev).unwrap();
            let r = match op {
                "exp" => t.exp(),
                "log" => t.log(),
                _ => t.sin(),
            };
            Some(r.realize_f64()[0].to_bits().to_be_bytes().to_vec())
        }
        _ => None,
    }
}

/// What Fuel produces for one vector, or why it cannot be asked.
fn produce(v: &Vector) -> Result<Vec<u8>, &'static str> {
    if v.op == "copysign" {
        return Err("no `copysign` op in Fuel");
    }
    if matches!(v.dtype.as_str(), "f8e5m2" | "f8e4m3fn") {
        return Err("no 8-bit float Tensor path on CPU");
    }
    let got = match v.op.as_str() {
        "neg" | "abs" => run_sign_edit(v),
        "max_prop" | "min_prop" | "fmax_ieee" | "fmin_ieee" => run_minmax(v),
        "floor" | "ceil" | "trunc" | "round_even" => run_rounding(v),
        "exp" | "log" | "sin" => run_transcendental(v),
        _ => None,
    };
    got.ok_or("no evaluator arm for this (op, dtype)")
}

/// Does `got` conform to the vector's expectation under its comparator?
fn conforms(v: &Vector, got: &[u8]) -> bool {
    if v.nan_provenance == "computed" {
        // KISS-CONFORM-6.8-0010: a COMPUTED NaN matches iff the observed value is
        // also NaN and agrees in quietness. Sign and payload MUST NOT be compared.
        // (x86 SSE delivers the negative "real indefinite" `FFC00000` for
        // log(-1); a harness that compared it to `7FC00000` would report a defect
        // the spec says does not exist.)
        return match nan_quiet(&v.dtype, got) {
            Some(q) => Some(q) == nan_quiet(&v.dtype, &v.expected),
            None => false,
        };
    }
    if v.class == "ULP" {
        return ulp_distance(got, &v.expected) <= v.ulp_bound;
    }
    got == v.expected.as_slice()
}

fn run(v: &Vector) -> Outcome {
    match produce(v) {
        Err(why) => Outcome::NotExercised(why),
        Ok(got) if conforms(v, &got) => Outcome::Pass,
        Ok(got) => Outcome::Fail {
            got,
            want: v.expected.clone(),
        },
    }
}

/// `(file, tcId)` pairs that FAIL today. Pinned as a set, both directions.
const KNOWN_FAILING: &[(&str, u64)] = &[];

#[test]
fn the_evaluator_can_fail_and_measures_distance() {
    // CONTROL for the harness itself: a wrong expectation must be reported as a
    // failure, and the distance function must agree with hand-computed values.
    assert_eq!(ulp_distance(&[0x3F, 0x80, 0, 0], &[0x3F, 0x80, 0, 0]), 0);
    assert_eq!(ulp_distance(&[0x3F, 0x80, 0, 0], &[0x3F, 0x80, 0, 1]), 1);
    // +0 and -0 are ADJACENT under totalOrder, not equal.
    assert_eq!(ulp_distance(&[0, 0, 0, 0], &[0x80, 0, 0, 0]), 1);
    let v = Vector {
        file: "control",
        tc_id: 0,
        op: "neg".into(),
        dtype: "bf16".into(),
        inputs: vec![vec![0x3F, 0x80]],
        expected: vec![0x3F, 0x80], // deliberately WRONG: neg(1.0) is BF80
        class: "exact-byte".into(),
        ulp_bound: 0,
        nan_provenance: "moved".into(),
    };
    assert!(
        matches!(run(&v), Outcome::Fail { .. }),
        "a wrong expectation must be reported as a failure, or this harness is inert"
    );
    // CONTROL for the computed-NaN arm: a quiet-vs-signalling disagreement and a
    // finite-where-NaN-expected result MUST fail, while a sign-only difference
    // MUST NOT (6.8-0010). Without this the arm could be vacuously permissive.
    assert_eq!(nan_quiet("f32", &[0x7F, 0xC0, 0, 0]), Some(true));
    assert_eq!(nan_quiet("f32", &[0xFF, 0xC0, 0, 0]), Some(true));
    assert_eq!(nan_quiet("f32", &[0x7F, 0x80, 0, 1]), Some(false));
    assert_eq!(nan_quiet("f32", &[0x3F, 0x80, 0, 0]), None);
    let neg_nan = Vector {
        file: "control",
        tc_id: 1,
        op: "log".into(),
        dtype: "f32".into(),
        inputs: vec![vec![0xBF, 0x80, 0, 0]], // log(-1.0): a computed NaN
        expected: vec![0x7F, 0xC0, 0, 0],
        class: "ULP".into(),
        ulp_bound: 2,
        nan_provenance: "computed".into(),
    };
    assert_eq!(
        run(&neg_nan),
        Outcome::Pass,
        "sign of a computed NaN must not be compared"
    );
    let finite_expected_nan = Vector {
        inputs: vec![vec![0x3F, 0x80, 0, 0]], // log(1.0) = 0.0, finite, expected NaN
        ..neg_nan
    };
    assert!(
        matches!(run(&finite_expected_nan), Outcome::Fail { .. }),
        "finite where NaN is expected MUST be a mismatch"
    );
}

#[test]
fn nan_vectors_conform_except_the_pinned_known_failing_set() {
    let vs = all_vectors();
    assert_eq!(vs.len(), 198, "corpus size changed: 30 + 40 + 12 + 44 + 72");

    let (mut pass, mut not_ex) = (0usize, 0usize);
    let mut failing: Vec<(&'static str, u64)> = Vec::new();
    let mut report = Vec::new();
    for v in &vs {
        match run(v) {
            Outcome::Pass => pass += 1,
            Outcome::NotExercised(_) => not_ex += 1,
            Outcome::Fail { got, want } => {
                failing.push((v.file, v.tc_id));
                report.push(format!(
                    "{} tcId {} {} {}: inputs {:02x?} got {:02x?} want {:02x?}",
                    v.file, v.tc_id, v.op, v.dtype, v.inputs, got, want
                ));
            }
        }
    }
    println!(
        "[kiss-nan] {} vectors: {pass} pass, {} fail, {not_ex} not exercised",
        vs.len(),
        failing.len()
    );
    for r in &report {
        println!("[kiss-nan] FAIL {r}");
    }
    assert_eq!(
        pass + failing.len() + not_ex,
        vs.len(),
        "accounting does not sum"
    );
    // DETECTOR for GAP-352 / GAP-353 (the vectors Fuel cannot exercise): 34 are the
    // 8-bit float dtypes and 2 are `copysign`. When Fuel gains either path this
    // count drops and this reddens, so those rows are re-measured and closed in
    // the same change. 36 is the 82-vector corpus minus the 46 that pass.
    assert_eq!(
        not_ex, 36,
        "the not-exercised count changed: a new Fuel path (8-bit floats or copysign) \
         now runs vectors that used to be skipped. Re-measure and update GAP-352 / GAP-353"
    );
    // Control: the evaluator exercised something, so zero failures is a finding
    // about Fuel and not about an evaluator that ran nothing.
    assert!(pass > 0, "no vector was exercised: the evaluator is inert");

    let mut want: Vec<(&str, u64)> = KNOWN_FAILING.to_vec();
    want.sort_unstable();
    failing.sort_unstable();
    assert_eq!(
        failing,
        want,
        "the failing set changed.\n  newly failing / fixed vectors are listed by diff of the two sets above;\n  {}",
        report.join("\n  ")
    );
}
