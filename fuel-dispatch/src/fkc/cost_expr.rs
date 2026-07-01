//! Cost-expression mini-parser + evaluator (FKC §4.4 / adoption plan §2.3,
//! strategy **(A)** — the AST half only).
//!
//! A cost expression (`flops`, `bytes_moved`, `overhead_ns`,
//! `memory.device_bytes`) is a small arithmetic formula over shape /
//! parameter symbols. This module parses such a string into a
//! [`CompiledCostExpr`] AST and evaluates it against a `symbol_bindings`
//! map ([`eval`]). It does **not** produce a `CostFn` fn-pointer — that
//! is the next slice (the trampoline / global side-table /
//! `register_into`). This slice stops at the parsed AST + a capacity-only
//! evaluator.
//!
//! ## Grammar (§2.3)
//!
//! ```text
//! expr    := term  (('+' | '-') term)*
//! term    := factor (('*' | '/' | '%') factor)*
//! factor  := number | symbol | '(' expr ')' | ('-' factor)
//! number  := integer | float        (decimal; `1`, `4`, `2.5`)
//! symbol  := identifier             (`m`, `n`, `k`, `dtype_bytes`, …)
//! ```
//!
//! Accepted shape/param symbols are open: any identifier resolves at
//! eval time from the supplied bindings (a missing symbol is an eval
//! error, not a parse error — the §2.3 vocabulary `m`, `n`, `k`,
//! `n`(=elem count), `dtype_bytes` plus any param name the kernel uses).
//! Anything outside the grammar (a stray `(`, `==`, an empty expression,
//! a bad character) is a parse error surfaced to the caller as
//! [`FkcError::CostExprParse`].
//!
//! ## The `Unknown` sentinel
//!
//! A cost block that is absent, `class`-only, or `judge_measured` with no
//! coefficient expressions compiles to [`CompiledCostExpr::Unknown`].
//! The register slice maps that to the existing `unknown_cost` fn-pointer
//! sentinel; this slice only records it.
//!
//! ## Eval is capacity-only (v1)
//!
//! [`eval`] evaluates over `u64`-equivalent `f64` symbol values
//! (capacity / `Extent::bound()`), matching what today's `CostFn`
//! receives. The per-tier `memory` beyond `device_bytes` and any
//! `SymEnv`-resolved live-extent term are parsed-but-not-evaluated (held
//! by the caller, documented at the construction site).

use std::collections::HashMap;

use crate::fused::CostEstimate;
use crate::kernel::OpParams;
use fuel_ir::dispatch::OpKind;
use fuel_ir::{DType, Shape};

/// A compiled cost-expression AST (FKC §4.4 / §2.3 strategy A).
///
/// Cloneable + comparable so it can ride on a resolved record and be
/// asserted in tests. Held as data, never as a fn-pointer (this slice).
#[derive(Debug, Clone, PartialEq)]
pub enum CompiledCostExpr {
    /// Sentinel: no cost claim (absent block / class-only /
    /// judge_measured with no expressions). Maps to `unknown_cost` in the
    /// register slice.
    Unknown,
    /// A parsed expression tree.
    Expr(CostNode),
}

/// One node of a cost-expression AST.
#[derive(Debug, Clone, PartialEq)]
pub enum CostNode {
    /// A numeric literal (integer or float; held as `f64`).
    Lit(f64),
    /// A shape / param symbol resolved at eval time (`m`, `n`, `k`,
    /// `dtype_bytes`, …). A dotted member path (`lhs.dim`, `x.dim`,
    /// FKC §4.4 "shape dims by role") is carried as one dotted `Sym`.
    Sym(String),
    /// A binary arithmetic operation.
    Bin {
        op: BinOp,
        lhs: Box<CostNode>,
        rhs: Box<CostNode>,
    },
    /// Unary negation.
    Neg(Box<CostNode>),
    /// A shape-axis index: `base[index]` (FKC §3.5 `dim[i]`, §4.4
    /// `lhs.dim[0]` / `out_shape[2]`). `base` is the symbol path, `index`
    /// the (usually literal) axis expression.
    Index {
        base: Box<CostNode>,
        index: Box<CostNode>,
    },
    /// A function-style term: `f(arg, ...)` (e.g. `block_bytes(quant_type)`
    /// in the quant-matmul cost hints). Resolved at eval time from the
    /// bindings by its canonical string spelling.
    Call {
        name: String,
        args: Vec<CostNode>,
    },
}

/// The four supported binary operators (§2.3: `+ - * / %`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// A parse failure (kept local; the lower layer wraps it into
/// [`crate::fkc::FkcError::CostExprParse`] with the field/section context).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostParseError(pub String);

impl std::fmt::Display for CostParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An eval failure (an undefined symbol, or division by zero).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostEvalError(pub String);

impl std::fmt::Display for CostEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ===========================================================================
// Tokenizer
// ===========================================================================

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, CostParseError> {
    let mut toks = Vec::new();
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                toks.push(Tok::Star);
                i += 1;
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '%' => {
                toks.push(Tok::Percent);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '[' => {
                toks.push(Tok::LBracket);
                i += 1;
            }
            ']' => {
                toks.push(Tok::RBracket);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            c if c.is_ascii_digit() => {
                let start = i;
                let mut seen_dot = false;
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit() || (bytes[i] == '.' && !seen_dot))
                {
                    if bytes[i] == '.' {
                        seen_dot = true;
                    }
                    i += 1;
                }
                let lit: String = bytes[start..i].iter().collect();
                let val: f64 = lit
                    .parse()
                    .map_err(|_| CostParseError(format!("bad numeric literal `{lit}`")))?;
                toks.push(Tok::Num(val));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                // An identifier may be a DOTTED member path (`lhs.dim`,
                // `x.dim`, `out_shape`) per FKC §4.4 "shape dims by role". A
                // `.` is consumed into the identifier only when followed by an
                // alphabetic/underscore (a member name), so it is never
                // confused with a decimal point.
                loop {
                    while i < bytes.len()
                        && (bytes[i].is_ascii_alphanumeric() || bytes[i] == '_')
                    {
                        i += 1;
                    }
                    if i + 1 < bytes.len()
                        && bytes[i] == '.'
                        && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == '_')
                    {
                        i += 1; // consume the '.'
                        continue;
                    }
                    break;
                }
                let ident: String = bytes[start..i].iter().collect();
                toks.push(Tok::Ident(ident));
            }
            other => {
                return Err(CostParseError(format!(
                    "unexpected character `{other}` in cost expression"
                )));
            }
        }
    }
    Ok(toks)
}

// ===========================================================================
// Recursive-descent parser
// ===========================================================================

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_expr(&mut self) -> Result<CostNode, CostParseError> {
        let mut lhs = self.parse_term()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_term()?;
            lhs = CostNode::Bin {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<CostNode, CostParseError> {
        let mut lhs = self.parse_factor()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Rem,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_factor()?;
            lhs = CostNode::Bin {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<CostNode, CostParseError> {
        let atom = match self.bump() {
            Some(Tok::Num(v)) => CostNode::Lit(v),
            Some(Tok::Ident(name)) => {
                // A `(` immediately after an identifier is a function call
                // (`block_bytes(quant_type)`, FKC §4.4 cost hints).
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump(); // consume '('
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            match self.peek() {
                                Some(Tok::Comma) => {
                                    self.bump();
                                }
                                _ => break,
                            }
                        }
                    }
                    match self.bump() {
                        Some(Tok::RParen) => {}
                        _ => return Err(CostParseError(
                            "unbalanced `(` in function call — expected `)`".into(),
                        )),
                    }
                    CostNode::Call { name, args }
                } else {
                    CostNode::Sym(name)
                }
            }
            Some(Tok::Minus) => {
                let inner = self.parse_factor()?;
                return Ok(CostNode::Neg(Box::new(inner)));
            }
            Some(Tok::LParen) => {
                let inner = self.parse_expr()?;
                match self.bump() {
                    Some(Tok::RParen) => inner,
                    _ => return Err(CostParseError("unbalanced `(` — expected `)`".into())),
                }
            }
            Some(other) => {
                return Err(CostParseError(format!(
                    "unexpected token `{other:?}` where a value was expected"
                )))
            }
            None => {
                return Err(CostParseError(
                    "unexpected end of cost expression (expected a value)".into(),
                ))
            }
        };
        // Postfix `[index]` runs (shape-axis index: `dim[i]`, `out_shape[2]`,
        // `lhs.dim[0]`; FKC §3.5 / §4.4). Chained indexing is allowed.
        let mut node = atom;
        while matches!(self.peek(), Some(Tok::LBracket)) {
            self.bump(); // consume '['
            let index = self.parse_expr()?;
            match self.bump() {
                Some(Tok::RBracket) => {}
                _ => return Err(CostParseError("unbalanced `[` — expected `]`".into())),
            }
            node = CostNode::Index {
                base: Box::new(node),
                index: Box::new(index),
            };
        }
        Ok(node)
    }
}

/// Parse a single cost-expression string into a [`CostNode`] AST.
///
/// Returns [`CostParseError`] on anything outside the §2.3 grammar; the
/// lower layer wraps it into [`crate::fkc::FkcError::CostExprParse`].
pub fn parse_expr(src: &str) -> Result<CostNode, CostParseError> {
    let toks = tokenize(src)?;
    if toks.is_empty() {
        return Err(CostParseError("empty cost expression".into()));
    }
    let mut parser = Parser { toks, pos: 0 };
    let node = parser.parse_expr()?;
    if parser.pos != parser.toks.len() {
        return Err(CostParseError(format!(
            "trailing tokens after a complete expression (at token {})",
            parser.pos
        )));
    }
    Ok(node)
}

/// Compile a single cost-field expression into a [`CompiledCostExpr`].
/// `None` (an absent field, i.e. YAML `~`) compiles to
/// [`CompiledCostExpr::Unknown`].
pub fn compile_field(src: Option<&str>) -> Result<CompiledCostExpr, CostParseError> {
    match src {
        None => Ok(CompiledCostExpr::Unknown),
        Some(s) if s.trim().is_empty() => Ok(CompiledCostExpr::Unknown),
        Some(s) => Ok(CompiledCostExpr::Expr(parse_expr(s)?)),
    }
}

// ===========================================================================
// Evaluator (capacity-only)
// ===========================================================================

/// Evaluate a [`CompiledCostExpr`] against `bindings` (symbol → capacity
/// value). The [`CompiledCostExpr::Unknown`] sentinel evaluates to `0.0`.
///
/// Symbols absent from `bindings` are an [`CostEvalError`] — the caller
/// (the register slice) supplies the §2.3 vocabulary (`m`, `n`, `k`,
/// element count, `dtype_bytes`, plus op-param names).
pub fn eval(
    expr: &CompiledCostExpr,
    bindings: &HashMap<String, f64>,
) -> Result<f64, CostEvalError> {
    match expr {
        CompiledCostExpr::Unknown => Ok(0.0),
        CompiledCostExpr::Expr(node) => eval_node(node, bindings),
    }
}

fn eval_node(node: &CostNode, bindings: &HashMap<String, f64>) -> Result<f64, CostEvalError> {
    match node {
        CostNode::Lit(v) => Ok(*v),
        CostNode::Sym(name) => bindings
            .get(name)
            .copied()
            .ok_or_else(|| CostEvalError(format!("undefined cost symbol `{name}`"))),
        CostNode::Index { base, index } => {
            // Resolve by the canonical spelling `base[index]` from bindings
            // (capacity-eval; the importer supplies the resolved axis values).
            let key = canonical_key(node);
            if let Some(v) = bindings.get(&key) {
                return Ok(*v);
            }
            // Fall through: a numeric index against a bound `base` symbol is
            // not statically resolvable here without a shape vector, so an
            // unbound shape-index is an eval error (capacity-eval supplies the
            // full key). Touch the children so a malformed index still errors.
            let _ = eval_node(base, bindings);
            let _ = eval_node(index, bindings);
            Err(CostEvalError(format!("undefined cost shape-index `{key}`")))
        }
        CostNode::Call { .. } => {
            let key = canonical_key(node);
            bindings
                .get(&key)
                .copied()
                .ok_or_else(|| CostEvalError(format!("undefined cost term `{key}`")))
        }
        CostNode::Neg(inner) => Ok(-eval_node(inner, bindings)?),
        CostNode::Bin { op, lhs, rhs } => {
            let l = eval_node(lhs, bindings)?;
            let r = eval_node(rhs, bindings)?;
            match op {
                BinOp::Add => Ok(l + r),
                BinOp::Sub => Ok(l - r),
                BinOp::Mul => Ok(l * r),
                BinOp::Div => {
                    if r == 0.0 {
                        Err(CostEvalError("division by zero in cost expression".into()))
                    } else {
                        Ok(l / r)
                    }
                }
                BinOp::Rem => {
                    if r == 0.0 {
                        Err(CostEvalError("modulo by zero in cost expression".into()))
                    } else {
                        Ok(l % r)
                    }
                }
            }
        }
    }
}

/// Bind the §2.3 cost-expression symbol vocabulary (`m`/`n`/`k`/`batch`/
/// `dtype_bytes`, plus a generic element count) from a dispatch context, so a
/// contract's *declared* cost expression can be evaluated to a concrete FLOP
/// count. Op-specific symbols come from [`OpParams`]; the vocabulary extends as
/// declared-cost contracts for more op families land.
pub fn bind_cost_symbols(
    _op: OpKind,
    shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
) -> HashMap<String, f64> {
    let mut b = HashMap::new();
    if let Some(out) = dtypes.last() {
        b.insert("dtype_bytes".to_string(), out.size_in_bytes() as f64);
    }
    match params {
        OpParams::Matmul {
            lhs_batch_dims, m, n, k, ..
        } => {
            let batch = lhs_batch_dims.iter().copied().product::<usize>().max(1);
            b.insert("batch".to_string(), batch as f64);
            b.insert("m".to_string(), *m as f64);
            b.insert("n".to_string(), *n as f64);
            b.insert("k".to_string(), *k as f64);
        }
        _ => {
            // Generic fallback: the total output element count as `n`.
            if let Some(out_shape) = shapes.last() {
                b.insert("n".to_string(), out_shape.elem_count() as f64);
            }
        }
    }
    b
}

/// The cost trampoline: evaluate a contract's *declared* cost expression against
/// a dispatch context, turning a parsed [`CompiledCostExpr`] into a concrete
/// [`CostEstimate`] — the value FKC import previously dropped in favor of the
/// `unknown_cost` sentinel. An undefined symbol is a typed [`CostEvalError`]
/// (never a panic, never a silent zero). `bytes_moved` stays 0 until contracts
/// carry a separate `bytes:` expression.
pub fn cost_estimate(
    expr: &CompiledCostExpr,
    op: OpKind,
    shapes: &[Shape],
    dtypes: &[DType],
    params: &OpParams,
) -> Result<CostEstimate, CostEvalError> {
    let symbols = bind_cost_symbols(op, shapes, dtypes, params);
    let flops = eval(expr, &symbols)?;
    Ok(CostEstimate {
        flops: flops.max(0.0) as u64,
        ..CostEstimate::default()
    })
}

/// Reconstruct a canonical string spelling of a shape-index / call node so it
/// can be looked up in the `bindings` map (the importer keys resolved
/// shape-axis / term values by this spelling at capacity-eval time).
fn canonical_key(node: &CostNode) -> String {
    match node {
        CostNode::Lit(v) => {
            if v.fract() == 0.0 {
                format!("{}", *v as i64)
            } else {
                format!("{v}")
            }
        }
        CostNode::Sym(s) => s.clone(),
        CostNode::Index { base, index } => {
            format!("{}[{}]", canonical_key(base), canonical_key(index))
        }
        CostNode::Call { name, args } => {
            let inner: Vec<String> = args.iter().map(canonical_key).collect();
            format!("{name}({})", inner.join(","))
        }
        CostNode::Neg(inner) => format!("-{}", canonical_key(inner)),
        CostNode::Bin { .. } => "<expr>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn parses_and_evals_3_times_n_times_4() {
        // "3 * n * 4" with n=8 → 96 (the prompt's canonical case).
        let expr = compile_field(Some("3 * n * 4")).expect("parses");
        let v = eval(&expr, &b(&[("n", 8.0)])).expect("evals");
        assert_eq!(v, 96.0);
    }

    #[test]
    fn precedence_mul_binds_tighter_than_add() {
        // 2 + 3 * 4 = 14, not 20.
        let expr = compile_field(Some("2 + 3 * 4")).unwrap();
        assert_eq!(eval(&expr, &b(&[])).unwrap(), 14.0);
    }

    #[test]
    fn parens_override_precedence() {
        let expr = compile_field(Some("(2 + 3) * 4")).unwrap();
        assert_eq!(eval(&expr, &b(&[])).unwrap(), 20.0);
    }

    #[test]
    fn matmul_flops_2_m_n_k() {
        let expr = compile_field(Some("2 * m * n * k")).unwrap();
        let v = eval(&expr, &b(&[("m", 4.0), ("n", 8.0), ("k", 16.0)])).unwrap();
        assert_eq!(v, 2.0 * 4.0 * 8.0 * 16.0);
    }

    #[test]
    fn rem_and_div_and_float_literal() {
        assert_eq!(eval(&compile_field(Some("10 % 3")).unwrap(), &b(&[])).unwrap(), 1.0);
        assert_eq!(eval(&compile_field(Some("12 / 4")).unwrap(), &b(&[])).unwrap(), 3.0);
        assert_eq!(eval(&compile_field(Some("2.5 * 2")).unwrap(), &b(&[])).unwrap(), 5.0);
    }

    #[test]
    fn unary_negation() {
        assert_eq!(eval(&compile_field(Some("-3 + 5")).unwrap(), &b(&[])).unwrap(), 2.0);
    }

    #[test]
    fn dtype_bytes_symbol() {
        let expr = compile_field(Some("3 * n * dtype_bytes")).unwrap();
        let v = eval(&expr, &b(&[("n", 10.0), ("dtype_bytes", 4.0)])).unwrap();
        assert_eq!(v, 120.0);
    }

    #[test]
    fn none_and_empty_are_unknown() {
        assert_eq!(compile_field(None).unwrap(), CompiledCostExpr::Unknown);
        assert_eq!(compile_field(Some("   ")).unwrap(), CompiledCostExpr::Unknown);
        // Unknown evaluates to 0.
        assert_eq!(eval(&CompiledCostExpr::Unknown, &b(&[])).unwrap(), 0.0);
    }

    #[test]
    fn malformed_expressions_are_parse_errors() {
        // The prompt's garbage case (after a comparator that the grammar
        // doesn't have) and other malformed forms.
        assert!(compile_field(Some("k_len <= sk garbage(")).is_err());
        assert!(compile_field(Some("2 *")).is_err());
        assert!(compile_field(Some("(2 + 3")).is_err());
        assert!(compile_field(Some("2 2")).is_err());
        assert!(compile_field(Some("@")).is_err());
        assert!(compile_field(Some("==")).is_err());
    }

    #[test]
    fn undefined_symbol_is_eval_error() {
        let expr = compile_field(Some("n * missing")).unwrap();
        assert!(eval(&expr, &b(&[("n", 4.0)])).is_err());
    }

    // --- Extended grammar (FKC §3.5 / §4.4): shape-index, member path, call ---

    #[test]
    fn shape_index_and_member_path_parse() {
        // The corpus conv cost expressions.
        compile_field(Some("2 * out_shape[0] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"))
            .expect("out_shape[i] parses");
        compile_field(Some("2 * out_elems * (x.dim[1] / groups) * weight.dim[2] * weight.dim[3]"))
            .expect("role.dim[i] parses");
        compile_field(Some("2 * x.dim[0] * x.dim[1] * (x.dim[2] - weight.dim[2] + 1) * weight.dim[2]"))
            .expect("causal-conv expr parses");
    }

    #[test]
    fn function_call_term_parses() {
        // The quant-matmul cost hint with a function-style term.
        compile_field(Some(
            "(batch_count*m*k*4 + n*k*block_bytes(quant_type) + batch_count*m*n*4)",
        ))
        .expect("block_bytes(quant_type) parses");
    }

    #[test]
    fn shape_index_evaluates_from_canonical_binding() {
        let expr = compile_field(Some("2 * out_shape[0]")).unwrap();
        let v = eval(&expr, &b(&[("out_shape[0]", 8.0)])).unwrap();
        assert_eq!(v, 16.0);
    }

    #[test]
    fn member_path_is_one_dotted_symbol() {
        // `x.dim` is a single dotted symbol, NOT `x` `.` `dim`.
        let node = parse_expr("x.dim").unwrap();
        assert_eq!(node, CostNode::Sym("x.dim".to_string()));
    }

    #[test]
    fn unbalanced_bracket_is_parse_error() {
        assert!(compile_field(Some("out_shape[0")).is_err());
        assert!(compile_field(Some("x.dim 0]")).is_err());
    }

    #[test]
    fn lone_dot_is_still_a_parse_error() {
        // A bare `.` (not inside a number, not a member path) is rejected.
        assert!(compile_field(Some("2 . 3")).is_err());
    }
}
