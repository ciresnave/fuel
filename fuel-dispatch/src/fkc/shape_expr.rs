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
