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

/// Reserved `axis` sentinel = the trailing axis, resolved to `rank-1` at eval
/// (§6.20-0002/-0003). Concrete axes are `0..MAX_RANK-1` (MAX_RANK=8), so `0xFF` is
/// unambiguously `last`. DISTINCT from §6.19-0020's `0xFFFE` (u16 axis-set mask) — a
/// different field, different width.
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

/// A typed decline. A reader MUST refuse malformed input with one of these, never a
/// panic (§6.20-0003/0006).
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
}
