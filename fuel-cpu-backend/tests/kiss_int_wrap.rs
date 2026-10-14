// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Runs the vendored KISS integer-wrap vectors (`ops-int-wrap.json`, 440
//! exact-byte cells) through Fuel's CPU binary op, raw-bit compared.**
//!
//! The vectors pin `add` / `sub` / `mul` on the eight integer dtypes at the wrap
//! boundaries (KISS-OPS-6.2-0002): the result is the exact mathematical value
//! reduced modulo 2^N, so a SATURATING implementation differs on at least one
//! cell per `(dtype, op)`. Expected bytes are big-endian raw two's-complement
//! patterns.
//!
//! # What is exercised
//!
//! The op goes through `DynBackendStorage::binary_op_dyn`, the CPU backend's
//! dispatch entry (`cpu_binary_op`), on a one-element `HostBuffer`. Fuel's
//! `HostBuffer` has `U8`, `I8`, `U32`, `I16`, `I32`, `I64` and NO `u16`/`u64`
//! variant, so those two dtypes (110 of the 440 cells) are NOT-EXPRESSIBLE: an
//! absent surface, counted and pinned, never folded into a pass.
//!
//! The lazy `Tensor` is not used: it has no constructor for most integer dtypes,
//! so it could not state the cells at all.

use fuel_backend_contract::dyn_backend::DynBackendStorage;
use fuel_cpu_backend::dyn_impl::CpuStorage;
use fuel_ir::op::BinaryOp;
use fuel_ir::{HostBuffer, Layout, Shape};

const VECTORS: &str = include_str!("../../fuel-dispatch/fixtures/kiss-corpus/ops-int-wrap.json");

#[derive(Debug)]
struct Vector {
    tc_id: u64,
    op: String,
    dtype: String,
    a: Vec<u8>,
    b: Vec<u8>,
    expected: Vec<u8>,
}

fn bits_of(hex: &str) -> Vec<u8> {
    hex.split(|c: char| c.is_whitespace() || c == '\u{b7}')
        .filter(|s| !s.is_empty())
        .map(|b| u8::from_str_radix(b, 16).expect("hex byte"))
        .collect()
}

fn load() -> Vec<Vector> {
    let doc: serde_json::Value = serde_json::from_str(VECTORS).expect("corpus parses");
    doc["vectors"]
        .as_array()
        .expect("vectors")
        .iter()
        .map(|v| Vector {
            tc_id: v["tcId"].as_u64().expect("tcId"),
            op: v["op"].as_str().expect("op").to_string(),
            dtype: v["dtype"].as_str().expect("dtype").to_string(),
            a: bits_of(v["inputs"][0]["bits"].as_str().expect("a")),
            b: bits_of(v["inputs"][1]["bits"].as_str().expect("b")),
            expected: bits_of(v["expected"]["bits"].as_str().expect("expected")),
        })
        .collect()
}

/// A one-element `HostBuffer` of `dtype` from big-endian bytes, or `None` if Fuel
/// has no such storage variant.
fn buffer(dtype: &str, be: &[u8]) -> Option<HostBuffer> {
    Some(match dtype {
        "u8" => HostBuffer::U8(vec![be[0]]),
        "i8" => HostBuffer::I8(vec![be[0] as i8]),
        "u32" => HostBuffer::U32(vec![u32::from_be_bytes(be.try_into().ok()?)]),
        "i16" => HostBuffer::I16(vec![i16::from_be_bytes(be.try_into().ok()?)]),
        "i32" => HostBuffer::I32(vec![i32::from_be_bytes(be.try_into().ok()?)]),
        "i64" => HostBuffer::I64(vec![i64::from_be_bytes(be.try_into().ok()?)]),
        _ => return None,
    })
}

/// The element's big-endian bytes.
fn be_bytes(buf: &HostBuffer) -> Vec<u8> {
    match buf {
        HostBuffer::U8(v) => v[0].to_be_bytes().to_vec(),
        HostBuffer::I8(v) => v[0].to_be_bytes().to_vec(),
        HostBuffer::U32(v) => v[0].to_be_bytes().to_vec(),
        HostBuffer::I16(v) => v[0].to_be_bytes().to_vec(),
        HostBuffer::I32(v) => v[0].to_be_bytes().to_vec(),
        HostBuffer::I64(v) => v[0].to_be_bytes().to_vec(),
        other => panic!("unexpected result buffer {other:?}"),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail { got: Vec<u8> },
    NotExpressible,
    Error(String),
}

fn run(v: &Vector) -> Outcome {
    let (Some(a), Some(b)) = (buffer(&v.dtype, &v.a), buffer(&v.dtype, &v.b)) else {
        return Outcome::NotExpressible;
    };
    let op = match v.op.as_str() {
        "add" => BinaryOp::Add,
        "sub" => BinaryOp::Sub,
        "mul" => BinaryOp::Mul,
        other => panic!("unknown op {other}"),
    };
    let l = Layout::contiguous(Shape::from_dims(&[1]));
    let r = match CpuStorage(a).binary_op_dyn(&CpuStorage(b), &l, &l, op) {
        Ok(r) => r,
        Err(e) => return Outcome::Error(format!("{e}")),
    };
    let got = be_bytes(
        &r.as_any()
            .downcast_ref::<CpuStorage>()
            .expect("cpu result")
            .0,
    );
    if got == v.expected {
        Outcome::Pass
    } else {
        Outcome::Fail { got }
    }
}

/// Cells on which the backend returns a typed error today: all of `i8`.
const ERRORING_CELLS: usize = 55;

/// `(op, dtype)` cells that FAIL today, pinned as a set in both directions.
const KNOWN_FAILING: &[(&str, &str)] = &[];

#[test]
fn the_harness_can_fail() {
    // CONTROL: a deliberately wrong expectation (a SATURATING result) is a failure.
    let v = Vector {
        tc_id: 0,
        op: "add".into(),
        dtype: "u8".into(),
        a: vec![200],
        b: vec![100],
        expected: vec![255], // saturating; the wrapping answer is 44
    };
    assert!(matches!(run(&v), Outcome::Fail { .. }));
    let ok = Vector {
        expected: vec![44],
        ..v
    };
    assert_eq!(run(&ok), Outcome::Pass);
}

/// Outcome counts over the whole corpus.
struct Tally {
    pass: usize,
    not_expr: usize,
    failing: Vec<(String, String)>,
    errors: Vec<String>,
}

/// Runs every vector, printing each failure and error and a one-line summary.
fn tally(vs: &[Vector]) -> Tally {
    let mut t = Tally {
        pass: 0,
        not_expr: 0,
        failing: Vec::new(),
        errors: Vec::new(),
    };
    for v in vs {
        match run(v) {
            Outcome::Pass => t.pass += 1,
            Outcome::NotExpressible => t.not_expr += 1,
            Outcome::Fail { got } => {
                println!(
                    "[kiss-int-wrap] FAIL tcId {} {} {} a={:02x?} b={:02x?} got {:02x?} want {:02x?}",
                    v.tc_id, v.op, v.dtype, v.a, v.b, got, v.expected
                );
                t.failing.push((v.op.clone(), v.dtype.clone()));
            }
            Outcome::Error(e) => t
                .errors
                .push(format!("tcId {} {} {}: {e}", v.tc_id, v.op, v.dtype)),
        }
    }
    println!(
        "[kiss-int-wrap] {} vectors: {} pass, {} fail, {} not expressible, {} errors",
        vs.len(),
        t.pass,
        t.failing.len(),
        t.not_expr,
        t.errors.len()
    );
    for e in &t.errors {
        println!("[kiss-int-wrap] ERROR {e}");
    }
    t
}

#[test]
fn int_wrap_vectors_conform_except_the_pinned_sets() {
    let vs = load();
    assert_eq!(vs.len(), 440, "corpus size changed");
    let Tally {
        pass,
        not_expr,
        mut failing,
        errors,
    } = tally(&vs);
    // `i8` add/sub/mul: the CPU backend has a `HostBuffer::I8` variant but its
    // binary op has no `(I8, I8)` arm, so the call returns a typed error
    // (`dtype mismatch in add, lhs: I8, rhs: I8`: the message is misleading, the
    // dtypes DO match). A typed error is not a wrong result, but it is not a
    // conforming one either: 55 cells (15 add + 25 sub + 15 mul), pinned by dtype.
    assert_eq!(
        errors.len(),
        ERRORING_CELLS,
        "the number of cells the backend ERRORS on changed: {errors:?}"
    );
    assert!(
        errors.iter().all(|e| e.contains(" i8:")),
        "an error outside the pinned i8 set: {errors:?}"
    );
    assert!(pass > 0, "nothing was exercised: the harness is inert");
    assert_eq!(
        pass + failing.len() + not_expr + errors.len(),
        vs.len(),
        "accounting does not sum"
    );
    // `u16` and `u64` have no `HostBuffer` variant: 110 cells.
    assert_eq!(
        not_expr, 110,
        "the not-expressible count changed: Fuel gained (or lost) an integer storage variant"
    );
    failing.sort();
    failing.dedup();
    let mut want: Vec<(String, String)> = KNOWN_FAILING
        .iter()
        .map(|(o, d)| (o.to_string(), d.to_string()))
        .collect();
    want.sort();
    assert_eq!(failing, want, "the failing (op, dtype) set changed");
}
