# Convergence-C · Increment C-1 — Shape-Expression Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Fuel's FKC layer a typed shape-expression AST with a §6.19 canonical wire codec (byte-matching the KISS goldens) and a gap-propagating evaluator, then wire it into `eval_shape_rule` so a contract's declared `DimExpr` output shape is evaluated + cross-checked.

**Architecture:** A new `fuel-dispatch/src/fkc/shape_expr.rs` ports the KISS reference (`conformance/src/shape_expr.rs`) — the `Dim`/`ShapeExpr` AST, `encode`/`decode`, and `eval_dim`/`eval_shape` — as Fuel's own independent implementation, verified byte-for-byte against the KISS conformance vectors vendored as Fuel-side fixtures. `eval_shape_rule` gains a string parser (role names → positional AST) that extends its existing `same_as` fast path.

**Tech Stack:** Rust (edition 2024), `fuel-dispatch` crate, `fuel_ir::Shape`.

## Global Constraints

- **Build discipline (CLAUDE.md, hard rules):** NEVER workspace-wide cargo — always `-p fuel-dispatch`. ONE cargo invocation at a time. Never panic on production paths — every malformed input is a typed `Result` decline.
- **Byte-contract (the conformance gate):** the wire bytes MUST be byte-identical to the KISS reference. The golden anchor `Div(Extent(0,last),Const(2))` MUST encode to exactly `08 03 00 02 00 FF 09 00 03 02 00 00 00 00 00 00 00`. Tags: `SameAs=0x01 Extent=0x02 Const=0x03 Param=0x04 Add=0x05 Sub=0x06 Mul=0x07 Div=0x08`; reserved-reject `0x09/0x0A/0x0B`; `0x00`→ZeroTag. `LAST=0xFF` (u8), `SYMBOLIC=i64::MIN`. `0xFF` (u8 single-axis) is DISTINCT from §6.19-0020's `0xFFFE` (u16 set-mask) — never unify.
- **Never-panic:** `shape_expr.rs` returns `Result<_, ShapeExprError>` on every path; `eval_shape_rule` maps a decline to an `ImportWarning` + `Ok(None)` (skip; never a false reject), a `Gap` to `Ok(None)`, and a concrete result to `Ok(Some(shape))`.
- **Scope:** C-1 is the EXPRESSION kind only (`SameAs` + `DimExpr`). The role/index-woven kind (`reduce_shape`/`gather_shape`/`matmul_shape`/`shape_consistent`) is C-2. Do NOT add those here.

## File Structure

- **Create** `fuel-dispatch/src/fkc/shape_expr.rs` — the typed AST + wire codec + evaluator + the 5 EXPRESSION conformance tests.
- **Modify** `fuel-dispatch/src/fkc/mod.rs` — add `mod shape_expr;` (between `mod shape_constraint;` line 55 and `mod validate;` line 56).
- **Modify** `fuel-dispatch/src/fkc/return_check.rs` — extend `eval_shape_rule` (line 29) with a DimExpr parser + evaluation; add unit tests.
- **Modify** `ROADMAP.md` — correct the stale line ~128 ("OutputDesc.shape_rule §5, parsed-but-unevaluated").
- **Modify** `docs/outreach/baracuda-shape-oracle-rfc-ask.md` — add a superseded/resolution banner (asks resolved by KISS merge `3bd6d2d`).

---

### Task 1: shape_expr.rs — AST + `encode` + golden serialization

**Files:**
- Create: `fuel-dispatch/src/fkc/shape_expr.rs`
- Modify: `fuel-dispatch/src/fkc/mod.rs` (add `mod shape_expr;`)

**Interfaces:**
- Produces: `Dim` (`Extent{operand:u8,axis:u8} | Const(i64) | Param(u8) | Add|Sub|Mul|Div(Box<Dim>,Box<Dim>)`), `ShapeExpr::SameAs{operand:u8}`, consts `TAG_*`, `LAST: u8 = 0xFF`, `SYMBOLIC: i64 = i64::MIN`, `Dim::encode(&self)->Vec<u8>`, `ShapeExpr::encode(&self)->Vec<u8>`.

- [ ] **Step 1: Write the failing test** (append to `shape_expr.rs` under `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // §6.20-0005 canonical serialization — byte-identical to the KISS goldens.
    #[test]
    fn serialization_golden() {
        assert_eq!(ShapeExpr::SameAs { operand: 0 }.encode(), vec![0x01, 0x00]);
        assert_eq!(Dim::Extent { operand: 0, axis: LAST }.encode(), vec![0x02, 0x00, 0xFF]);
        assert_eq!(Dim::Const(2).encode(), vec![0x03, 0x02, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(Dim::Param(0).encode(), vec![0x04, 0x00]);
        // The rope-half anchor — the byte contract of record.
        let half = Dim::Div(
            Box::new(Dim::Extent { operand: 0, axis: LAST }),
            Box::new(Dim::Const(2)),
        );
        assert_eq!(
            half.encode(),
            vec![0x08, 0x03, 0x00, 0x02, 0x00, 0xFF, 0x09, 0x00, 0x03, 0x02, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fuel-dispatch --lib fkc::shape_expr::tests::serialization_golden`
Expected: FAIL to compile — `shape_expr` module / `Dim` / `ShapeExpr` not found.

- [ ] **Step 3: Write minimal implementation** (top of `fuel-dispatch/src/fkc/shape_expr.rs`, above the test module)

```rust
//! Fuel's independent typed shape-expression AST + §6.19 canonical wire codec +
//! evaluator (KISS-Ops §6.20). Byte-matches the KISS reference
//! (`conformance/src/shape_expr.rs`), verified against the vendored golden vectors —
//! the shape-side companion to the value oracle. EXPRESSION kind only (`SameAs` +
//! `DimExpr`); the role/index-woven kind (reduce/gather/matmul) is a separate variant
//! (Convergence-C C-2). Every malformed input is a typed decline, never a panic.

// §6.20-0005 tag space (one byte; 0x00 reserved per §6.19-0006).
pub const TAG_SAME_AS: u8 = 0x01;
pub const TAG_EXTENT: u8 = 0x02;
pub const TAG_CONST: u8 = 0x03;
pub const TAG_PARAM: u8 = 0x04;
pub const TAG_ADD: u8 = 0x05;
pub const TAG_SUB: u8 = 0x06;
pub const TAG_MUL: u8 = 0x07;
pub const TAG_DIV: u8 = 0x08;
pub const TAG_REDUCE: u8 = 0x09; // reserved (extension-registry) — reader rejects
pub const TAG_WITH_DIM: u8 = 0x0A; // reserved
pub const TAG_DIMS: u8 = 0x0B; // reserved

/// Sentinel extent for a symbolic / data-dependent axis length → surfaced Gap (§6.20-0004).
pub const SYMBOLIC: i64 = i64::MIN;

/// Reserved `axis` sentinel = the trailing axis, resolved to `rank-1` at eval (§6.20-0002/-0003).
/// Concrete axes are `0..MAX_RANK-1` (MAX_RANK=8), so `0xFF` is unambiguously `last`. DISTINCT
/// from §6.19-0020's `0xFFFE` (u16 axis-set mask) — a different field, different width.
pub const LAST: u8 = 0xFF;

/// A single-dimension expression (`DimExpr`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// The size of `operand`'s `axis` (non-negative index, or [`LAST`] = trailing).
    Extent { operand: u8, axis: u8 },
    Const(i64),
    /// The op's `field`-th declared param.
    Param(u8),
    Add(Box<Dim>, Box<Dim>),
    Sub(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    /// Floor division (toward −∞).
    Div(Box<Dim>, Box<Dim>),
}

/// A whole-shape expression (`ShapeExpr`). The closed core is `SameAs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExpr {
    SameAs { operand: u8 },
}

impl ShapeExpr {
    /// Canonical wire bytes (§6.20-0005).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            ShapeExpr::SameAs { operand } => vec![TAG_SAME_AS, *operand],
        }
    }
}

impl Dim {
    /// Canonical wire bytes (§6.20-0005).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Dim::Extent { operand, axis } => vec![TAG_EXTENT, *operand, *axis],
            Dim::Const(c) => {
                let mut v = vec![TAG_CONST];
                v.extend_from_slice(&c.to_le_bytes());
                v
            }
            Dim::Param(f) => vec![TAG_PARAM, *f],
            Dim::Add(a, b) => encode_binary(TAG_ADD, a, b),
            Dim::Sub(a, b) => encode_binary(TAG_SUB, a, b),
            Dim::Mul(a, b) => encode_binary(TAG_MUL, a, b),
            Dim::Div(a, b) => encode_binary(TAG_DIV, a, b),
        }
    }
}

fn encode_binary(tag: u8, a: &Dim, b: &Dim) -> Vec<u8> {
    let (ca, cb) = (a.encode(), b.encode());
    let mut v = vec![tag];
    v.extend_from_slice(&(ca.len() as u16).to_le_bytes());
    v.extend_from_slice(&ca);
    v.extend_from_slice(&(cb.len() as u16).to_le_bytes());
    v.extend_from_slice(&cb);
    v
}
```

Then add to `fuel-dispatch/src/fkc/mod.rs` after line 55 (`mod shape_constraint;`):

```rust
mod shape_expr;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fuel-dispatch --lib fkc::shape_expr::tests::serialization_golden`
Expected: PASS (1 passed). A `dead_code` warning on unused consts/eval is expected until later tasks.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_expr.rs fuel-dispatch/src/fkc/mod.rs
git commit -m "feat(fkc): shape-expr AST + §6.19 wire encoder (byte-matches KISS golden)"
```

---

### Task 2: `decode` — typed-decline reader + round-trip

**Files:**
- Modify: `fuel-dispatch/src/fkc/shape_expr.rs`

**Interfaces:**
- Produces: `ShapeExprError` (enum, `PartialEq`), `decode_dim(&[u8]) -> Result<Dim, ShapeExprError>` such that `decode_dim(&d.encode()) == Ok(d)`.

- [ ] **Step 1: Write the failing test** (add to `mod tests`)

```rust
#[test]
fn decode_round_trip_and_declines() {
    let half = Dim::Div(
        Box::new(Dim::Extent { operand: 0, axis: LAST }),
        Box::new(Dim::Const(2)),
    );
    assert_eq!(decode_dim(&half.encode()).unwrap(), half); // round-trip
    // §6.20-0006 typed declines — never a panic.
    assert_eq!(decode_dim(&[0x00]), Err(ShapeExprError::ZeroTag));
    assert_eq!(decode_dim(&[0x09, 0x00]), Err(ShapeExprError::ReservedTag { tag: 0x09 }));
    assert_eq!(decode_dim(&[0x03, 0x02, 0x00]), Err(ShapeExprError::TruncatedBlob { need: 9, got: 3 }));
    assert_eq!(decode_dim(&[0x04, 0x00, 0xAB]), Err(ShapeExprError::TrailingBytes { extra: 1 }));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fuel-dispatch --lib fkc::shape_expr::tests::decode_round_trip_and_declines`
Expected: FAIL to compile — `ShapeExprError` / `decode_dim` not found.

- [ ] **Step 3: Write minimal implementation** (add to `shape_expr.rs`, after the `Dim`/`encode` code)

```rust
/// A typed decline. A reader MUST refuse malformed input with one of these, never a panic
/// (§6.20-0003/0006).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExprError {
    ZeroTag,
    ReservedTag { tag: u8 },
    TruncatedBlob { need: usize, got: usize },
    TrailingBytes { extra: usize },
    AxisOutOfRange { axis: u8, rank: usize },
    OperandOutOfRange { operand: u8, operands: usize },
    ParamOutOfRange { field: u8, params: usize },
    DivideByZero,
}

/// Decode one `DimExpr` blob, rejecting a malformed one with a typed decline. Round-trips:
/// `decode_dim(&d.encode()) == Ok(d)`.
pub fn decode_dim(blob: &[u8]) -> Result<Dim, ShapeExprError> {
    let (d, consumed) = decode_dim_at(blob, 0)?;
    if consumed != blob.len() {
        return Err(ShapeExprError::TrailingBytes { extra: blob.len() - consumed });
    }
    Ok(d)
}

fn decode_dim_at(blob: &[u8], pos: usize) -> Result<(Dim, usize), ShapeExprError> {
    let tag = *blob
        .get(pos)
        .ok_or(ShapeExprError::TruncatedBlob { need: pos + 1, got: blob.len() })?;
    match tag {
        0x00 => Err(ShapeExprError::ZeroTag),
        TAG_EXTENT => {
            need(blob, pos, 3)?;
            Ok((Dim::Extent { operand: blob[pos + 1], axis: blob[pos + 2] }, pos + 3))
        }
        TAG_CONST => {
            need(blob, pos, 9)?;
            let mut a = [0u8; 8];
            a.copy_from_slice(&blob[pos + 1..pos + 9]);
            Ok((Dim::Const(i64::from_le_bytes(a)), pos + 9))
        }
        TAG_PARAM => {
            need(blob, pos, 2)?;
            Ok((Dim::Param(blob[pos + 1]), pos + 2))
        }
        TAG_ADD | TAG_SUB | TAG_MUL | TAG_DIV => {
            let (c1, p1) = read_child(blob, pos + 1)?;
            let (c2, p2) = read_child(blob, p1)?;
            let (a, b) = (Box::new(c1), Box::new(c2));
            let d = match tag {
                TAG_ADD => Dim::Add(a, b),
                TAG_SUB => Dim::Sub(a, b),
                TAG_MUL => Dim::Mul(a, b),
                _ => Dim::Div(a, b),
            };
            Ok((d, p2))
        }
        other => Err(ShapeExprError::ReservedTag { tag: other }),
    }
}

/// A `u16`-LE length-prefixed child expression at `pos`.
fn read_child(blob: &[u8], pos: usize) -> Result<(Dim, usize), ShapeExprError> {
    if blob.len() < pos + 2 {
        return Err(ShapeExprError::TruncatedBlob { need: 2, got: blob.len().saturating_sub(pos) });
    }
    let len = u16::from_le_bytes([blob[pos], blob[pos + 1]]) as usize;
    let start = pos + 2;
    if blob.len() < start + len {
        return Err(ShapeExprError::TruncatedBlob { need: len, got: blob.len().saturating_sub(start) });
    }
    let child = decode_dim(&blob[start..start + len])?; // child consumes its declared length exactly
    Ok((child, start + len))
}

fn need(blob: &[u8], pos: usize, n: usize) -> Result<(), ShapeExprError> {
    if blob.len() < pos + n {
        Err(ShapeExprError::TruncatedBlob { need: n, got: blob.len().saturating_sub(pos) })
    } else {
        Ok(())
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fuel-dispatch --lib fkc::shape_expr::tests::decode_round_trip_and_declines`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_expr.rs
git commit -m "feat(fkc): shape-expr typed-decline decoder + round-trip"
```

---

### Task 3: `eval` — axis resolution, floor-div, gap propagation

**Files:**
- Modify: `fuel-dispatch/src/fkc/shape_expr.rs`

**Interfaces:**
- Produces: `DimValue::{Concrete(i64), Gap}`, `ShapeValue::{Concrete(Vec<i64>), Gap}`,
  `eval_dim(&Dim, &[Vec<i64>], &[i64]) -> Result<DimValue, ShapeExprError>`,
  `eval_shape(&ShapeExpr, &[Vec<i64>], &[i64]) -> Result<ShapeValue, ShapeExprError>`.

- [ ] **Step 1: Write the failing test** (add to `mod tests` — the vocabulary, axis/floordiv, and symbolic-gap vectors)

```rust
#[test]
fn eval_vocabulary_axis_floordiv_gap() {
    // §6.20-0002 vocabulary.
    let ops = vec![vec![2i64, 3, 4]];
    let params = vec![7i64];
    assert_eq!(eval_shape(&ShapeExpr::SameAs { operand: 0 }, &ops, &params).unwrap(),
               ShapeValue::Concrete(vec![2, 3, 4]));
    assert_eq!(eval_dim(&Dim::Extent { operand: 0, axis: 1 }, &ops, &params).unwrap(),
               DimValue::Concrete(3));
    assert_eq!(eval_dim(&Dim::Const(5), &ops, &params).unwrap(), DimValue::Concrete(5));
    assert_eq!(eval_dim(&Dim::Param(0), &ops, &params).unwrap(), DimValue::Concrete(7));
    // (extent(op0,axis0=2) * 3) + param0(7) = 13
    let e = Dim::Add(
        Box::new(Dim::Mul(Box::new(Dim::Extent { operand: 0, axis: 0 }), Box::new(Dim::Const(3)))),
        Box::new(Dim::Param(0)),
    );
    assert_eq!(eval_dim(&e, &ops, &params).unwrap(), DimValue::Concrete(13));

    // §6.20-0003 axis + floor-div.
    let r3 = vec![vec![2i64, 3, 5]];
    assert_eq!(eval_dim(&Dim::Extent { operand: 0, axis: LAST }, &r3, &[]).unwrap(), DimValue::Concrete(5));
    assert_eq!(eval_dim(&Dim::Extent { operand: 0, axis: 2 }, &r3, &[]).unwrap(), DimValue::Concrete(5));
    assert_eq!(eval_dim(&Dim::Extent { operand: 0, axis: 3 }, &r3, &[]),
               Err(ShapeExprError::AxisOutOfRange { axis: 3, rank: 3 }));
    let fd = |a, b| eval_dim(&Dim::Div(Box::new(Dim::Const(a)), Box::new(Dim::Const(b))), &r3, &[]);
    assert_eq!(fd(7, 2).unwrap(), DimValue::Concrete(3));
    assert_eq!(fd(-7, 2).unwrap(), DimValue::Concrete(-4)); // floor(−3.5) = −4
    assert_eq!(fd(1, 0), Err(ShapeExprError::DivideByZero));

    // §6.20-0004 symbolic → Gap, propagates.
    let sym = vec![vec![4i64, SYMBOLIC]];
    assert_eq!(eval_dim(&Dim::Extent { operand: 0, axis: LAST }, &sym, &[]).unwrap(), DimValue::Gap);
    let half = Dim::Div(Box::new(Dim::Extent { operand: 0, axis: LAST }), Box::new(Dim::Const(2)));
    assert_eq!(eval_dim(&half, &sym, &[]).unwrap(), DimValue::Gap);
    assert_eq!(eval_shape(&ShapeExpr::SameAs { operand: 0 }, &sym, &[]).unwrap(), ShapeValue::Gap);
    assert_eq!(eval_dim(&Dim::Extent { operand: 0, axis: 0 }, &sym, &[]).unwrap(), DimValue::Concrete(4));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fuel-dispatch --lib fkc::shape_expr::tests::eval_vocabulary_axis_floordiv_gap`
Expected: FAIL to compile — `DimValue`/`ShapeValue`/`eval_dim`/`eval_shape` not found.

- [ ] **Step 3: Write minimal implementation** (add to `shape_expr.rs`)

```rust
/// Evaluating a `DimExpr`: a concrete dim, or a surfaced gap (§6.20-0004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimValue { Concrete(i64), Gap }

/// Evaluating a `ShapeExpr`: a concrete shape, or a surfaced gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeValue { Concrete(Vec<i64>), Gap }

/// Resolve a non-negative `axis` (or [`LAST`]) against `rank` (§6.20-0003).
fn resolve_axis(axis: u8, rank: usize) -> Result<usize, ShapeExprError> {
    if axis == LAST {
        return rank.checked_sub(1).ok_or(ShapeExprError::AxisOutOfRange { axis, rank });
    }
    let a = axis as usize;
    if a >= rank { return Err(ShapeExprError::AxisOutOfRange { axis, rank }); }
    Ok(a)
}

/// Floor division (toward −∞), unlike Rust's truncating `/`.
fn floordiv(a: i64, b: i64) -> i64 {
    let (q, r) = (a / b, a % b);
    if r != 0 && ((r < 0) != (b < 0)) { q - 1 } else { q }
}

fn eval_binary(
    a: DimValue, b: DimValue, f: impl Fn(i64, i64) -> Result<i64, ShapeExprError>,
) -> Result<DimValue, ShapeExprError> {
    match (a, b) {
        (DimValue::Concrete(x), DimValue::Concrete(y)) => Ok(DimValue::Concrete(f(x, y)?)),
        _ => Ok(DimValue::Gap), // a gap in either operand propagates (§6.20-0004)
    }
}

/// Evaluate a `DimExpr` against operand shapes + param values.
pub fn eval_dim(d: &Dim, operands: &[Vec<i64>], params: &[i64]) -> Result<DimValue, ShapeExprError> {
    match d {
        Dim::Extent { operand, axis } => {
            let op = *operand as usize;
            let shape = operands.get(op).ok_or(ShapeExprError::OperandOutOfRange {
                operand: *operand, operands: operands.len() })?;
            let idx = resolve_axis(*axis, shape.len())?;
            let ext = shape[idx];
            Ok(if ext == SYMBOLIC { DimValue::Gap } else { DimValue::Concrete(ext) })
        }
        Dim::Const(c) => Ok(DimValue::Concrete(*c)),
        Dim::Param(f) => {
            let fi = *f as usize;
            let v = params.get(fi).ok_or(ShapeExprError::ParamOutOfRange {
                field: *f, params: params.len() })?;
            Ok(DimValue::Concrete(*v))
        }
        Dim::Add(a, b) => eval_binary(eval_dim(a, operands, params)?, eval_dim(b, operands, params)?, |x, y| Ok(x + y)),
        Dim::Sub(a, b) => eval_binary(eval_dim(a, operands, params)?, eval_dim(b, operands, params)?, |x, y| Ok(x - y)),
        Dim::Mul(a, b) => eval_binary(eval_dim(a, operands, params)?, eval_dim(b, operands, params)?, |x, y| Ok(x * y)),
        Dim::Div(a, b) => eval_binary(eval_dim(a, operands, params)?, eval_dim(b, operands, params)?, |x, y| {
            if y == 0 { Err(ShapeExprError::DivideByZero) } else { Ok(floordiv(x, y)) }
        }),
    }
}

/// Evaluate a `ShapeExpr` to a concrete shape (or a surfaced gap).
pub fn eval_shape(s: &ShapeExpr, operands: &[Vec<i64>], _params: &[i64]) -> Result<ShapeValue, ShapeExprError> {
    match s {
        ShapeExpr::SameAs { operand } => {
            let op = *operand as usize;
            let shape = operands.get(op).ok_or(ShapeExprError::OperandOutOfRange {
                operand: *operand, operands: operands.len() })?;
            if shape.iter().any(|&e| e == SYMBOLIC) {
                Ok(ShapeValue::Gap)
            } else {
                Ok(ShapeValue::Concrete(shape.clone()))
            }
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fuel-dispatch --lib fkc::shape_expr::tests::eval_vocabulary_axis_floordiv_gap`
Expected: PASS. Confirm all three shape_expr tests: `cargo test -p fuel-dispatch --lib fkc::shape_expr` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_expr.rs
git commit -m "feat(fkc): shape-expr evaluator — axis/floordiv/gap-propagation"
```

---

### Task 4: wire `eval_shape_rule` — DimExpr string parser + evaluation

**Files:**
- Modify: `fuel-dispatch/src/fkc/return_check.rs` (extend `eval_shape_rule` at line 29; add tests)

**Interfaces:**
- Consumes: `shape_expr::{Dim, ShapeExpr, DimValue, ShapeValue, eval_dim, eval_shape, ShapeExprError, LAST}`.
- The existing `eval_shape_rule(rule: &str, combo: ProbeComboRef, section: &str) -> Result<Option<Shape>, FkcError>` signature is UNCHANGED. New behavior: a `same_as(role)` string keeps its fast path; a DimExpr-form string (`extent(role,axis)`, `const(N)`, `param(N)`, `add/sub/mul/div(a,b)`) parses to a positional AST (role→position via the combo's order) and evaluates to a single-dim `Shape` `[dim]`; a `Gap` or an unrecognized form → `Ok(None)`; a parse/eval decline on a *recognized* DimExpr form → `Ok(None)` (skip; the cross-check never false-rejects). Unrecognized non-DimExpr strings still return `Ok(None)` exactly as today.

**Note (scope):** A DimExpr shape rule evaluates to a **single dimension** → a rank-1 `Shape [dim]` (matches the KISS `DimExpr` = one dimension). Whole-shape `same_as(role)` stays the `ShapeExpr` path. `param(N)` evaluation requires threaded param values; `eval_shape_rule` has none in the combo, so a `param`-bearing rule surfaces a `ParamOutOfRange` decline → `Ok(None)` (a documented C-1 limitation, same skip behavior as today's `from_params`).

- [ ] **Step 1: Write the failing test** (add to `return_check.rs`'s `#[cfg(test)] mod tests`, beside the existing `eval_shape_rule` tests ~line 319)

```rust
#[test]
fn eval_shape_rule_evaluates_dimexpr() {
    // combo: role "x" shape [4, 8]. div(extent(x, last), const(2)) = [4].
    let c: ProbeComboRef = &[("x".into(), Shape::from_dims(&[4, 8]), DType::F32)];
    assert_eq!(
        eval_shape_rule("div(extent(x, last), const(2))", c, "k").unwrap(),
        Some(Shape::from_dims(&[4]))
    );
    // extent(x, 0) = [4]; const(9) = [9].
    assert_eq!(eval_shape_rule("extent(x, 0)", c, "k").unwrap(), Some(Shape::from_dims(&[4])));
    assert_eq!(eval_shape_rule("const(9)", c, "k").unwrap(), Some(Shape::from_dims(&[9])));
    // same_as still works (fast path preserved).
    assert_eq!(eval_shape_rule("same_as(x)", c, "k").unwrap(), Some(Shape::from_dims(&[4, 8])));
    // An unknown role or a param rule → Ok(None) (skip).
    assert_eq!(eval_shape_rule("extent(nope, 0)", c, "k").unwrap(), None);
    assert_eq!(eval_shape_rule("param(0)", c, "k").unwrap(), None);
    // A decline on a recognized DimExpr (÷0) → Ok(None) (skip), never a panic/error.
    assert_eq!(eval_shape_rule("div(extent(x, last), const(0))", c, "k").unwrap(), None);
    // A DimExpr that evaluates negative is not a valid shape dim → Ok(None) (skip).
    assert_eq!(eval_shape_rule("sub(const(2), extent(x, 1))", c, "k").unwrap(), None); // 2 − 8 = −6
}
```

**Note (concrete probe shapes; no `SYMBOLIC` bridge needed):** `solve_probe_shapes` yields **concrete** probe shapes (`fuel_ir::Shape` dims are `usize`), so a `Gap` cannot arise through `eval_shape_rule` — the symbolic→`Gap` behavior is fully covered at the `shape_expr` layer (Task 3). `shape_to_i64` therefore just widens `usize`→`i64`. A `DimExpr` rule denotes a **single dimension** → a rank-1 output `Shape [d]` for `d ≥ 0`; a negative eval result is not a valid shape dim → `Ok(None)`. (A whole multi-dim non-`SameAs` output shape needs the reserved `Dims`/`WithDim` tags, out of C-1's core scope — such rules stay `Ok(None)`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fuel-dispatch --lib fkc::return_check::tests::eval_shape_rule_evaluates_dimexpr`
Expected: FAIL — DimExpr strings currently return `Ok(None)`, so the first `assert_eq!` (expects `Some([4])`) fails.

- [ ] **Step 3: Write minimal implementation** — replace the body of `eval_shape_rule` (return_check.rs:29-32) with a parser + evaluator. Keep the `same_as` fast path; add DimExpr parsing.

```rust
/// §5.2: `same_as(role)` yields the operand's whole shape; a `DimExpr` string
/// (`extent(role,axis)` / `const(N)` / `param(N)` / `add|sub|mul|div(a,b)`) evaluates to a
/// single-dim shape via the shape-expr oracle (§6.20). Every other token, a surfaced gap, or
/// a decline on a recognized DimExpr → `Ok(None)` (not-evaluable; never a false reject).
pub fn eval_shape_rule(rule: &str, combo: ProbeComboRef, _section: &str) -> Result<Option<Shape>, FkcError> {
    let rule = rule.trim();
    if let Some(r) = inner(rule, "same_as(") {
        return Ok(role(combo, r).map(|(_, s, _)| s.clone()));
    }
    // A DimExpr form: parse (role names → positional AST), then evaluate over the combo.
    if is_dimexpr_head(rule) {
        let Some(dim) = crate::fkc::shape_expr_parse::parse_dim(rule, combo) else { return Ok(None) };
        let operands: Vec<Vec<i64>> = combo.iter().map(|(_, s, _)| shape_to_i64(s)).collect();
        return match crate::fkc::shape_expr::eval_dim(&dim, &operands, &[]) {
            // A DimExpr denotes a single dimension → a rank-1 output shape (d ≥ 0).
            Ok(crate::fkc::shape_expr::DimValue::Concrete(d)) if d >= 0 => {
                Ok(Some(Shape::from_dims(&[d as usize])))
            }
            // Negative dim (not a valid shape), a surfaced Gap, or a decline → skip; never a false reject.
            Ok(_) | Err(_) => Ok(None),
        };
    }
    Ok(None)
}

/// True iff `rule` starts with a recognized `DimExpr` constructor head.
fn is_dimexpr_head(rule: &str) -> bool {
    const HEADS: &[&str] = &["extent(", "const(", "param(", "add(", "sub(", "mul(", "div("];
    HEADS.iter().any(|h| rule.starts_with(h))
}

/// Widen a `fuel_ir::Shape`'s concrete `usize` extents to `i64` for the evaluator. Probe shapes
/// are concrete (no symbolic sentinel arises on this path — see the Task-4 note).
fn shape_to_i64(s: &Shape) -> Vec<i64> {
    s.dims().iter().map(|&d| d as i64).collect()
}
```

Create a small parser module `fuel-dispatch/src/fkc/shape_expr_parse.rs` (role-name → positional `Dim`), and add `mod shape_expr_parse;` to `fkc/mod.rs` after `mod shape_expr;`:

```rust
//! Parse the FKC authoring DSL (`extent(role, axis)`, `const(N)`, `param(N)`,
//! `add|sub|mul|div(a, b)`) into the positional `shape_expr::Dim` AST. Role names are
//! resolved to positional operand indices via the probe combo's canonical order
//! (§6.4-0009 wire form is positional). Returns `None` on any malformed / unknown-role input
//! (the caller maps `None` → skip; never a false reject).

use crate::fkc::return_check::ProbeComboRef;
use crate::fkc::shape_expr::{Dim, LAST};

/// Position of `role` in the combo's canonical order.
fn role_pos(combo: ProbeComboRef, role: &str) -> Option<u8> {
    combo.iter().position(|(r, _, _)| r == role).and_then(|p| u8::try_from(p).ok())
}

/// Split `"a, b"` (the two args of a binary node) at the top-level comma (depth 0).
fn split_top_comma(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return Some((s[..i].trim(), s[i + 1..].trim())),
            _ => {}
        }
    }
    None
}

/// Strip a `head( ... )` wrapper, returning the inner text.
fn inner<'a>(s: &'a str, head: &str) -> Option<&'a str> {
    s.trim().strip_prefix(head)?.strip_suffix(')').map(str::trim)
}

pub fn parse_dim(rule: &str, combo: ProbeComboRef) -> Option<Dim> {
    let rule = rule.trim();
    if let Some(args) = inner(rule, "extent(") {
        let (role, axis) = split_top_comma(args)?;
        let operand = role_pos(combo, role)?;
        let axis = if axis == "last" { LAST } else { axis.parse::<u8>().ok()? };
        return Some(Dim::Extent { operand, axis });
    }
    if let Some(n) = inner(rule, "const(") {
        return Some(Dim::Const(n.parse::<i64>().ok()?));
    }
    if let Some(f) = inner(rule, "param(") {
        return Some(Dim::Param(f.parse::<u8>().ok()?));
    }
    for (head, ctor) in [
        ("add(", 0u8), ("sub(", 1), ("mul(", 2), ("div(", 3),
    ] {
        if let Some(args) = inner(rule, head) {
            let (a, b) = split_top_comma(args)?;
            let (da, db) = (Box::new(parse_dim(a, combo)?), Box::new(parse_dim(b, combo)?));
            return Some(match ctor {
                0 => Dim::Add(da, db),
                1 => Dim::Sub(da, db),
                2 => Dim::Mul(da, db),
                _ => Dim::Div(da, db),
            });
        }
    }
    None
}
```

Make `shape_expr` items reachable from `return_check`/`shape_expr_parse`: in `fkc/mod.rs` change `mod shape_expr;` to `pub(crate) mod shape_expr;` and add `pub(crate) mod shape_expr_parse;`. Ensure `ProbeComboRef` is `pub(crate)` (it is `pub type` in return_check.rs:11 — confirm it is reachable as `crate::fkc::return_check::ProbeComboRef`; if `return_check` is a private `mod`, add `pub(crate)` where needed).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fuel-dispatch --lib fkc::return_check::tests::eval_shape_rule_evaluates_dimexpr`
Expected: PASS. Then confirm no regression in the existing eval_shape_rule tests: `cargo test -p fuel-dispatch --lib fkc::return_check` → all pass (the `same_as`/`from_params`/`matmul` cases at ~319-322 still return their prior values — `matmul(a,b)` is not a DimExpr head, still `Ok(None)`).

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs fuel-dispatch/src/fkc/shape_expr_parse.rs fuel-dispatch/src/fkc/mod.rs
git commit -m "feat(fkc): eval_shape_rule evaluates the DimExpr vocab (role-name authoring)"
```

---

### Task 5: docs — ROADMAP:128 correction + superseded banner

**Files:**
- Modify: `ROADMAP.md` (~line 128)
- Modify: `docs/outreach/baracuda-shape-oracle-rfc-ask.md`

- [ ] **Step 1: Correct ROADMAP:128**

Locate the stale line describing `OutputDesc.shape_rule` as a KISS §5 field "parsed-but-unevaluated". Replace with the accurate statement: `OutputDesc.shape_rule` is a **Fuel FKC field** (`fkc/schema.rs:220`), evaluated by `eval_shape_rule` (`fkc/return_check.rs`, since `b1c33f91`); Convergence-C C-1 extends that evaluator to the full KISS DimExpr vocab (§6.20) via `fkc/shape_expr.rs`. (Read the surrounding ROADMAP context first; match its formatting.)

- [ ] **Step 2: Add the superseded banner to the outreach doc**

At the top of `docs/outreach/baracuda-shape-oracle-rfc-ask.md`, add:

```markdown
> **⭑ SUPERSEDED / RESOLVED (2026-07-20).** The open asks (a)/(b) below are resolved by the
> KISS shape-oracle RFC merge at `3bd6d2d` (KISS-Ops §6.20 + KISS-Contract §6.4-0011). Axis
> encoding = option A (non-negative index | `last`=0xFF, distinct from `0xFFFE`); vocabulary =
> `SameAs` + `DimExpr{Extent,Const,Param,+−×÷floor}`; `reduce_extent`→`reduced_count` +
> shape-side `extent(axis)`. Fuel implements this vocab in `fkc/shape_expr.rs` (Convergence-C).
> Retained for the historical record.
```

- [ ] **Step 3: Commit**

```bash
git add ROADMAP.md docs/outreach/baracuda-shape-oracle-rfc-ask.md
git commit -m "docs(convergence-c): correct ROADMAP shape_rule line + supersede shape-oracle ask"
```

---

## Verification (whole increment)

- [ ] `cargo test -p fuel-dispatch --lib fkc::shape_expr` → 3 passed (golden, decode, eval).
- [ ] `cargo test -p fuel-dispatch --lib fkc::return_check` → all pass (new DimExpr test + no regression).
- [ ] `cargo check -p fuel-dispatch` clean (warnings only).
- [ ] The golden anchor byte-vector matches exactly (asserted in `serialization_golden`).
- [ ] Ping Baracuda (`3s56q9w4`) — C-1 DimExpr §6.19 wire codec is on the branch for the offered byte-level pre-check.

## Follow-ons (NOT this increment)

- **C-2** — role/index-woven kind: `reduce_shape`/`gather_shape`/`matmul_shape` + `shape_consistent` (§6.20-0007/0008, §6.4-0011); the KISS tests 6/8/9 as fixtures.
- **C-3** — migrate the 22 registry decomposes onto the vocab; read `docs/outreach/kiss-conformance-architecture-fuel-ratify.md` §4 first (recipe=§6.13 independence).
- **Param threading** — thread contract param values into `eval_shape_rule` so `param(N)` rules evaluate (currently skip).
