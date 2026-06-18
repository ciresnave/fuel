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
    /// `dtype_bytes`, …).
    Sym(String),
    /// A binary arithmetic operation.
    Bin {
        op: BinOp,
        lhs: Box<CostNode>,
        rhs: Box<CostNode>,
    },
    /// Unary negation.
    Neg(Box<CostNode>),
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
            c if c.is_ascii_digit() || c == '.' => {
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
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == '_')
                {
                    i += 1;
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
        match self.bump() {
            Some(Tok::Num(v)) => Ok(CostNode::Lit(v)),
            Some(Tok::Ident(name)) => Ok(CostNode::Sym(name)),
            Some(Tok::Minus) => {
                let inner = self.parse_factor()?;
                Ok(CostNode::Neg(Box::new(inner)))
            }
            Some(Tok::LParen) => {
                let inner = self.parse_expr()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(inner),
                    _ => Err(CostParseError("unbalanced `(` — expected `)`".into())),
                }
            }
            Some(other) => Err(CostParseError(format!(
                "unexpected token `{other:?}` where a value was expected"
            ))),
            None => Err(CostParseError(
                "unexpected end of cost expression (expected a value)".into(),
            )),
        }
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
}
