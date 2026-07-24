//! Fuel's independent typed shape-expression AST + §6.19 canonical wire codec +
//! evaluator (KISS-Ops §6.20). Byte-matches the KISS reference
//! (`conformance/src/shape_expr.rs`), verified against the vendored golden vectors —
//! the shape-side companion to the value oracle. The closed EXPRESSION core is
//! `SameAs` + `DimExpr`; `WithDim`/`Dims` are the experimental umbrella-§6.4
//! extension constructors (§6.20-0009/-0010, issue #80) — a whole-shape decoder
//! ([`decode_shape`]) reads them while the `DimExpr` decoder ([`decode_dim`]) keeps
//! declining them (a `Dim` is never a whole shape). The role/index-woven kind
//! (reduce/gather/matmul) is a separate variant (Convergence-C C-2). Every
//! malformed input is a typed decline, never a panic.

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

/// A whole-shape expression (`ShapeExpr`). The closed core is `SameAs`; `WithDim`
/// and `Dims` are the experimental-range extension constructors registered via the
/// KISS umbrella §6.4 registry (Ops §6.20-0009/-0010, issue #80), activating the
/// §6.20-0005 tags `0x0A`/`0x0B`. `Reduce` (`0x09`) stays reserved (no consumer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExpr {
    /// The operand's whole shape.
    SameAs { operand: u8 },
    /// EXPERIMENTAL (umbrella §6.4; tag `0x0A`, §6.20-0009). The `operand`'s shape
    /// with the resolved `axis` (non-negative, or [`LAST`] = trailing) replaced by
    /// `dim`. Functional spelling `with_dim(operand, axis, dim)`.
    WithDim { operand: u8, axis: u8, dim: Box<Dim> },
    /// EXPERIMENTAL (umbrella §6.4; tag `0x0B`, §6.20-0010). A whole shape built
    /// from `N >= 0` ordered `DimExpr`s (`N = 0` = the rank-0 scalar shape).
    /// Functional spelling `dims([dim, …])`.
    Dims(Vec<Dim>),
}

impl ShapeExpr {
    /// Canonical wire bytes (§6.20-0005; the experimental extension tags
    /// `0x0A`/`0x0B` per §6.20-0009/-0010, u16-LE length-prefixed children).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            ShapeExpr::SameAs { operand } => vec![TAG_SAME_AS, *operand],
            // 0x0A, u8 operand, u8 axis (0xFF = last), one u16-LE length-prefixed
            // child DimExpr blob (§6.20-0009).
            ShapeExpr::WithDim { operand, axis, dim } => {
                let child = dim.encode();
                let mut v = vec![TAG_WITH_DIM, *operand, *axis];
                v.extend_from_slice(&(child.len() as u16).to_le_bytes());
                v.extend_from_slice(&child);
                v
            }
            // 0x0B, u8 count, then count × (u16-LE length-prefixed child DimExpr
            // blob) (§6.20-0010).
            ShapeExpr::Dims(dims) => {
                let mut v = vec![TAG_DIMS, dims.len() as u8];
                for d in dims {
                    let child = d.encode();
                    v.extend_from_slice(&(child.len() as u16).to_le_bytes());
                    v.extend_from_slice(&child);
                }
                v
            }
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
        // Whole-shape / reserved tags declined at the DimExpr grammar level: WithDim
        // (0x0A) / Dims (0x0B) are WHOLE-SHAPE constructors — now active, but only via
        // [`decode_shape`]; a `DimExpr` is never a whole shape, so they decline here
        // (§6.20-0009/-0010). Reduce (0x09) stays reserved at both levels — no
        // consumer (§6.20-0007 derives reduce-family shapes from op attrs).
        TAG_REDUCE | TAG_WITH_DIM | TAG_DIMS => Err(ShapeExprError::ReservedTag { tag }),
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

/// Decode one whole-shape [`ShapeExpr`] blob, rejecting a malformed one with a typed
/// decline (§6.20-0006). Reads `SameAs` (core) plus the registered experimental
/// extension constructors `WithDim`/`Dims` (§6.20-0009/-0010); the still-reserved
/// `Reduce` (`0x09`) and the `DimExpr`-only tags decline typed. A well-formed blob
/// round-trips: `decode_shape(&s.encode()) == Ok(s)`. Distinct from [`decode_dim`],
/// which reads the single-dimension grammar (a `Dim` is never a whole shape, so a
/// whole-shape tag declines there, and vice-versa).
pub fn decode_shape(blob: &[u8]) -> Result<ShapeExpr, ShapeExprError> {
    let (s, consumed) = decode_shape_at(blob, 0)?;
    if consumed != blob.len() {
        return Err(ShapeExprError::TrailingBytes { extra: blob.len() - consumed });
    }
    Ok(s)
}

fn decode_shape_at(blob: &[u8], pos: usize) -> Result<(ShapeExpr, usize), ShapeExprError> {
    let tag = *blob
        .get(pos)
        .ok_or(ShapeExprError::TruncatedBlob { need: pos + 1, got: blob.len() })?;
    match tag {
        0x00 => Err(ShapeExprError::ZeroTag),
        TAG_SAME_AS => {
            need(blob, pos, 2)?;
            Ok((ShapeExpr::SameAs { operand: blob[pos + 1] }, pos + 2))
        }
        // 0x0A, u8 operand, u8 axis, one u16-LE length-prefixed child (§6.20-0009).
        TAG_WITH_DIM => {
            need(blob, pos, 3)?;
            let operand = blob[pos + 1];
            let axis = blob[pos + 2];
            let (child, next) = read_child(blob, pos + 3)?;
            Ok((ShapeExpr::WithDim { operand, axis, dim: Box::new(child) }, next))
        }
        // 0x0B, u8 count, then count × u16-LE length-prefixed children (§6.20-0010).
        TAG_DIMS => {
            need(blob, pos, 2)?;
            let count = blob[pos + 1] as usize;
            let mut dims = Vec::with_capacity(count);
            let mut p = pos + 2;
            for _ in 0..count {
                let (child, next) = read_child(blob, p)?;
                dims.push(child);
                p = next;
            }
            Ok((ShapeExpr::Dims(dims), p))
        }
        // `Reduce` (0x09), the `DimExpr`-only tags, and any unallocated tag decline
        // typed at the whole-shape grammar level (§6.20-0006).
        other => Err(ShapeExprError::ReservedTag { tag: other }),
    }
}

/// Evaluating a `DimExpr`: a concrete dim, or a surfaced gap (§6.20-0004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimValue { Concrete(i64), Gap }

/// Evaluating a `ShapeExpr`: a concrete shape, or a surfaced gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeValue { Concrete(Vec<i64>), Gap }

/// Resolve a non-negative `axis` (or [`LAST`]) against `rank` (§6.20-0003).
/// `pub` since Increment C slice 1 (T2): the recipe-interior `axis_last`
/// resolver (`fuel_graph::runtime_fused::resolve_rel_attrs`) reuses this as
/// the single LAST→`rank−1` evaluator instead of growing a second one.
pub fn resolve_axis(axis: u8, rank: usize) -> Result<usize, ShapeExprError> {
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
pub fn eval_shape(s: &ShapeExpr, operands: &[Vec<i64>], params: &[i64]) -> Result<ShapeValue, ShapeExprError> {
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
        // §6.20-0009 `with_dim`: the operand's shape with the resolved axis replaced
        // by `dim`. A Gap replacement, or a symbolic extent in any KEPT axis,
        // surfaces the whole shape as a Gap (§6.20-0004); an out-of-range axis is a
        // typed decline (§6.20-0003), never a panic.
        ShapeExpr::WithDim { operand, axis, dim } => {
            let op = *operand as usize;
            let shape = operands.get(op).ok_or(ShapeExprError::OperandOutOfRange {
                operand: *operand, operands: operands.len() })?;
            let idx = resolve_axis(*axis, shape.len())?;
            let replacement = eval_dim(dim, operands, params)?;
            let mut out = shape.clone();
            let mut gap = false;
            for (i, e) in out.iter_mut().enumerate() {
                if i == idx {
                    match &replacement {
                        DimValue::Concrete(v) => *e = *v,
                        DimValue::Gap => gap = true,
                    }
                } else if *e == SYMBOLIC {
                    gap = true;
                }
            }
            if gap { Ok(ShapeValue::Gap) } else { Ok(ShapeValue::Concrete(out)) }
        }
        // §6.20-0010 `dims`: build the whole shape from the ordered DimExprs. A Gap
        // in ANY element surfaces the whole shape as a Gap (§6.20-0004); an empty
        // list is the rank-0 scalar shape.
        ShapeExpr::Dims(dims) => {
            let mut out = Vec::with_capacity(dims.len());
            let mut gap = false;
            for d in dims {
                match eval_dim(d, operands, params)? {
                    DimValue::Concrete(v) => out.push(v),
                    DimValue::Gap => gap = true,
                }
            }
            if gap { Ok(ShapeValue::Gap) } else { Ok(ShapeValue::Concrete(out)) }
        }
    }
}

// ---- §6.20-0007/0008 the ROLE/INDEX-WOVEN kind (Convergence-C C-2) ----------------
// These ride the op's role/index structure, NOT the SameAs+DimExpr expression core —
// a distinct shape-rule kind (§6.20-0008). Not expressible as a wire ShapeExpr.

/// §6.20-0007 `reduce`-family shape rule: the input shape with `reduce_axes` removed
/// (`keepdim=false`) or set to `1` (`keepdim=true`) — derived from op semantics.
pub fn reduce_shape(input: &[i64], reduce_axes: &[usize], keepdim: bool) -> Vec<i64> {
    let set: std::collections::BTreeSet<usize> = reduce_axes.iter().copied().collect();
    let mut out = Vec::new();
    for (i, &e) in input.iter().enumerate() {
        if set.contains(&i) {
            if keepdim {
                out.push(1);
            }
        } else {
            out.push(e);
        }
    }
    out
}

/// §6.20-0008 `gather`/`index_select`/`embedding` shape rule: the data shape with the
/// gathered `axis` replaced by the index shape (`data[..axis] ++ index ++ data[axis+1..]`).
/// In general the output equals NO operand's shape — which is why advertising
/// `same_as(data)` for a gather is a bug the shape oracle catches.
pub fn gather_shape(data: &[i64], index: &[i64], axis: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(data.len() - 1 + index.len());
    out.extend_from_slice(&data[..axis]);
    out.extend_from_slice(index);
    out.extend_from_slice(&data[axis + 1..]);
    out
}

/// §6.20-0008 `matmul` (contraction) shape rule: role-vector-derived (KISS-Classify
/// §6.6-0016 M/N/K axis roles, carried as roles not a ShapeExpr). For the canonical
/// same-rank ≥ 2 cell `lhs[..batch, M, K] · rhs[..batch, K, N] -> [..batch, M, N]`; the
/// output equals neither operand (§6.20-0008).
pub fn matmul_shape(lhs: &[i64], rhs: &[i64]) -> Vec<i64> {
    let r = lhs.len();
    let mut out = lhs[..r - 2].to_vec(); // aligned leading batch dims
    out.push(lhs[r - 2]); // M (lhs second-last)
    out.push(rhs[r - 1]); // N (rhs last)
    out
}

/// KISS-CONTRACT-6.4-0011: the Interface `declared` output shape is consistent iff it
/// equals the op's shape rule `computed` over the operand shapes. A surfaced `Gap`
/// (symbolic/data-dependent output) is never a hard inconsistency — a consumer cannot
/// assert a mismatch it cannot compute. The shape-side companion to the §6.4-0006 value
/// oracle.
pub fn shape_consistent(declared: &[i64], computed: &ShapeValue) -> bool {
    match computed {
        ShapeValue::Concrete(c) => declared == c.as_slice(),
        ShapeValue::Gap => true,
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

    // §6.20-0002/-0005 the DimExpr decoder declines the whole-shape / reserved tags
    // BY NAME: WithDim (0x0A) / Dims (0x0B) are now active but only via
    // `decode_shape` — a `DimExpr` is never a whole shape, so `decode_dim` still
    // declines them; Reduce (0x09) stays reserved (no consumer, §6.20-0007 derives
    // reduce shapes from attrs).
    #[test]
    fn reserved_extension_tags_decline_by_name() {
        // Pin the allocated-reserved wire values (KISS §6.20-0005).
        assert_eq!(TAG_REDUCE, 0x09);
        assert_eq!(TAG_WITH_DIM, 0x0A);
        assert_eq!(TAG_DIMS, 0x0B);
        for tag in [TAG_REDUCE, TAG_WITH_DIM, TAG_DIMS] {
            // Bare reserved tag: the typed decline names the tag, never a panic.
            assert_eq!(decode_dim(&[tag]), Err(ShapeExprError::ReservedTag { tag }));
            // A payload after a reserved tag does not rescue it.
            assert_eq!(
                decode_dim(&[tag, 0x00, 0x01]),
                Err(ShapeExprError::ReservedTag { tag })
            );
        }
        // A reserved tag in CHILD position (inside a binary node) declines identically.
        assert_eq!(
            decode_dim(&[TAG_ADD, 1, 0, TAG_WITH_DIM, 1, 0, TAG_DIMS]),
            Err(ShapeExprError::ReservedTag { tag: TAG_WITH_DIM })
        );
        // The first UNALLOCATED tag past the reserved block also declines (the
        // §6.20-0005 closed-vocabulary guard is open-ended, not just these three).
        assert_eq!(decode_dim(&[0x0C]), Err(ShapeExprError::ReservedTag { tag: 0x0C }));
    }

    // §6.20-0009 WithDim + §6.20-0010 Dims — the experimental umbrella-§6.4
    // extension constructors, byte-matched against the KISS reference goldens
    // (`conformance/tests/shape_expr.rs` @ 469abd5, PR #86). The exact golden hex
    // from each KISS `assert_golden(...)` call is transcribed here as a LITERAL
    // byte vector — Fuel is the second dissimilar implementation on the promotion
    // path, so an equal byte sequence is the confirmation gate.
    #[test]
    fn with_dim_extension_golden_roundtrip_eval() {
        // Golden — with_dim(operand=0, axis=1, Const(7)):
        // KISS "0A 00 01 09 00 03 07 00 00 00 00 00 00 00".
        let wd = ShapeExpr::WithDim { operand: 0, axis: 1, dim: Box::new(Dim::Const(7)) };
        assert_eq!(
            wd.encode(),
            vec![0x0A, 0x00, 0x01, 0x09, 0x00, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        // Golden — with_dim(operand=1, axis=last, Param(0)): KISS "0A 01 FF 02 00 04 00".
        let wd_last =
            ShapeExpr::WithDim { operand: 1, axis: LAST, dim: Box::new(Dim::Param(0)) };
        assert_eq!(wd_last.encode(), vec![0x0A, 0x01, 0xFF, 0x02, 0x00, 0x04, 0x00]);

        // Round-trip (the whole-shape decoder).
        assert_eq!(decode_shape(&wd.encode()).unwrap(), wd);
        assert_eq!(decode_shape(&wd_last.encode()).unwrap(), wd_last);

        // §6.20-0006 typed declines — never a panic. Truncated before the axis
        // byte; a child-length prefix that over-runs the buffer.
        assert_eq!(
            decode_shape(&[0x0A, 0x00]),
            Err(ShapeExprError::TruncatedBlob { need: 3, got: 2 })
        );
        assert_eq!(
            decode_shape(&[0x0A, 0x00, 0x01, 0x09, 0x00]),
            Err(ShapeExprError::TruncatedBlob { need: 9, got: 0 })
        );

        // §6.20-0003 eval — axis-replacement.
        let ops = vec![vec![2i64, 3, 5]];
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim { operand: 0, axis: 1, dim: Box::new(Dim::Const(9)) },
                &ops, &[]
            ).unwrap(),
            ShapeValue::Concrete(vec![2, 9, 5])
        );
        // `last` resolves to the trailing axis.
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim { operand: 0, axis: LAST, dim: Box::new(Dim::Const(9)) },
                &ops, &[]
            ).unwrap(),
            ShapeValue::Concrete(vec![2, 3, 9])
        );
        // §6.20-0004 gap propagation: a symbolic extent in a KEPT axis surfaces the
        // whole shape as a Gap; REPLACING the symbolic axis clears it.
        let sym = vec![vec![4i64, SYMBOLIC]];
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim { operand: 0, axis: 0, dim: Box::new(Dim::Const(7)) },
                &sym, &[]
            ).unwrap(),
            ShapeValue::Gap
        );
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim { operand: 0, axis: LAST, dim: Box::new(Dim::Const(7)) },
                &sym, &[]
            ).unwrap(),
            ShapeValue::Concrete(vec![4, 7])
        );
        // A Gap REPLACEMENT expression also surfaces a whole-shape Gap.
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim {
                    operand: 0,
                    axis: 0,
                    dim: Box::new(Dim::Extent { operand: 0, axis: LAST }),
                },
                &sym, &[]
            ).unwrap(),
            ShapeValue::Gap
        );
        // An out-of-range axis is a typed decline, never a panic (§6.20-0003).
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim { operand: 0, axis: 5, dim: Box::new(Dim::Const(1)) },
                &vec![vec![2i64, 3]], &[]
            ),
            Err(ShapeExprError::AxisOutOfRange { axis: 5, rank: 2 })
        );
        // Param threads through the replacement dim (whole-shape param plumbing).
        assert_eq!(
            eval_shape(
                &ShapeExpr::WithDim { operand: 0, axis: 1, dim: Box::new(Dim::Param(0)) },
                &ops, &[42]
            ).unwrap(),
            ShapeValue::Concrete(vec![2, 42, 5])
        );
    }

    #[test]
    fn dims_extension_golden_roundtrip_eval() {
        // Golden — dims([Extent(0,0), Const(2)]):
        // KISS "0B 02 03 00 02 00 00 09 00 03 02 00 00 00 00 00 00 00".
        let dims = ShapeExpr::Dims(vec![
            Dim::Extent { operand: 0, axis: 0 },
            Dim::Const(2),
        ]);
        assert_eq!(
            dims.encode(),
            vec![
                0x0B, 0x02, 0x03, 0x00, 0x02, 0x00, 0x00, 0x09, 0x00, 0x03, 0x02, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00
            ]
        );
        // Golden — the empty Dims (N=0) = the rank-0 scalar shape: KISS "0B 00".
        let scalar = ShapeExpr::Dims(vec![]);
        assert_eq!(scalar.encode(), vec![0x0B, 0x00]);

        // Round-trip.
        assert_eq!(decode_shape(&dims.encode()).unwrap(), dims);
        assert_eq!(decode_shape(&scalar.encode()).unwrap(), scalar);

        // §6.20-0006 typed declines — never a panic: a count promising children the
        // blob lacks; the still-reserved Reduce (0x09, no consumer); trailing bytes.
        assert_eq!(
            decode_shape(&[0x0B, 0x02]),
            Err(ShapeExprError::TruncatedBlob { need: 2, got: 0 })
        );
        assert_eq!(
            decode_shape(&[0x09, 0x00]),
            Err(ShapeExprError::ReservedTag { tag: 0x09 })
        );
        assert_eq!(
            decode_shape(&[0x0B, 0x00, 0xAB]),
            Err(ShapeExprError::TrailingBytes { extra: 1 })
        );

        // Eval — a qmatmul/scan-style whole-shape reweave across two operands.
        let ops = vec![vec![8i64, 16, 32, 64], vec![1i64, 1, 1, 128]];
        assert_eq!(
            eval_shape(
                &ShapeExpr::Dims(vec![
                    Dim::Extent { operand: 0, axis: 0 },
                    Dim::Extent { operand: 0, axis: 2 },
                    Dim::Extent { operand: 0, axis: 3 },
                    Dim::Extent { operand: 1, axis: 3 },
                ]),
                &ops, &[]
            ).unwrap(),
            ShapeValue::Concrete(vec![8, 32, 64, 128])
        );
        // The empty Dims evaluates to the rank-0 scalar shape.
        assert_eq!(
            eval_shape(&ShapeExpr::Dims(vec![]), &ops, &[]).unwrap(),
            ShapeValue::Concrete(vec![])
        );
        // §6.20-0004 gap propagation: a symbolic extent in ANY element surfaces the
        // whole shape as a Gap.
        let sym = vec![vec![4i64, SYMBOLIC]];
        assert_eq!(
            eval_shape(
                &ShapeExpr::Dims(vec![
                    Dim::Extent { operand: 0, axis: 0 },
                    Dim::Extent { operand: 0, axis: LAST },
                ]),
                &sym, &[]
            ).unwrap(),
            ShapeValue::Gap
        );
    }

    // The whole-shape decoder reads the core `SameAs` too, and declines a
    // DimExpr-only tag; the DimExpr decoder in turn declines the whole-shape
    // WithDim/Dims tags (a `Dim` is never a whole shape) — the two grammar levels
    // stay disjoint (§6.20-0006).
    #[test]
    fn decode_shape_core_and_grammar_level_declines() {
        assert_eq!(decode_shape(&[0x01, 0x00]).unwrap(), ShapeExpr::SameAs { operand: 0 });
        // A DimExpr-only tag (Const 0x03) is not a whole-shape constructor.
        assert_eq!(
            decode_shape(&[0x03, 0x02, 0, 0, 0, 0, 0, 0, 0]),
            Err(ShapeExprError::ReservedTag { tag: 0x03 })
        );
        // WithDim/Dims are whole-shape tags → the DimExpr reader declines them even
        // with a well-formed whole-shape payload.
        assert_eq!(
            decode_dim(&[TAG_WITH_DIM, 0x00, 0x01, 0x02, 0x00, 0x04, 0x00]),
            Err(ShapeExprError::ReservedTag { tag: TAG_WITH_DIM })
        );
        assert_eq!(
            decode_dim(&[TAG_DIMS, 0x00]),
            Err(ShapeExprError::ReservedTag { tag: TAG_DIMS })
        );
        // Reduce (0x09) stays reserved at BOTH levels — no consumer.
        assert_eq!(decode_shape(&[0x09, 0x00]), Err(ShapeExprError::ReservedTag { tag: 0x09 }));
        assert_eq!(decode_dim(&[0x09, 0x00]), Err(ShapeExprError::ReservedTag { tag: 0x09 }));
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

    // §6.20-0007 reduce-family shape rule.
    #[test]
    fn reduce_shape_rule() {
        assert_eq!(reduce_shape(&[2, 3, 5], &[2], false), vec![2, 3]);    // drop last
        assert_eq!(reduce_shape(&[2, 3, 5], &[2], true), vec![2, 3, 1]);  // keepdim
        assert_eq!(reduce_shape(&[2, 3, 5], &[0, 2], false), vec![3]);    // multi-axis
        assert_eq!(reduce_shape(&[8, 4096], &[1], false), vec![8]);       // reduce(sum, last)
    }

    // §6.20-0008 gather / matmul: the output equals no operand's shape.
    #[test]
    fn out_differs_from_operands() {
        // gather: data[..axis] ++ index ++ data[axis+1..].
        assert_eq!(gather_shape(&[8, 4096], &[16], 0), vec![16, 4096]);
        assert_eq!(gather_shape(&[1000, 64], &[2, 5], 0), vec![2, 5, 64]); // embedding
        // matmul: [M,K]·[K,N] -> [M,N], batched too.
        assert_eq!(matmul_shape(&[8, 4096], &[4096, 1024]), vec![8, 1024]);
        assert_eq!(matmul_shape(&[4, 8, 16], &[4, 16, 32]), vec![4, 8, 32]);
        // The oracle catches a false same_as(operand) claim: output ≠ either operand.
        let g = gather_shape(&[8, 4096], &[16], 0);
        assert!(!shape_consistent(&[8, 4096], &ShapeValue::Concrete(g)));
        let m = matmul_shape(&[8, 4096], &[4096, 1024]);
        assert!(!shape_consistent(&[8, 4096], &ShapeValue::Concrete(m.clone())));
        assert!(!shape_consistent(&[4096, 1024], &ShapeValue::Concrete(m)));
    }

    // KISS-CONTRACT-6.4-0011 declared ⟷ computed consistency (Gap is never a mismatch).
    #[test]
    fn contract_output_shape_consistency() {
        let computed = ShapeValue::Concrete(reduce_shape(&[8, 4096], &[1], false));
        assert!(shape_consistent(&[8], &computed));           // [8] matches reduce → [8]
        assert!(!shape_consistent(&[8, 4096], &computed));    // declaring rank-2 is the caught error
        assert!(shape_consistent(&[8], &ShapeValue::Gap));    // a Gap is never a hard mismatch
    }
}
