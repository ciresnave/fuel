//! Compile a contract's parsed cost AST into the live ranking cost path (§2.3).
//! The parser/evaluator (`cost_expr.rs`) is complete; this only adds wiring.
use std::sync::{Mutex, OnceLock};
use crate::fkc::cost_expr::{eval, CompiledCostExpr, CostEvalError};
use crate::fkc::lower::{ResolvedPrimitive, ResolvedFused};
use crate::fused::CostEstimate;
use fuel_graph::registry::FusedOpParams;
use fuel_ir::{DType, Shape};

/// The closed classification of a contract's `cost:` block (§2.3 / V-FKC-9).
/// Every non-placeholder cost must map to exactly one of these; `None` from
/// [`classify_cost`] means the block is NOT load-bearing (a placeholder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostClassKind {
    /// `class: free` — an honest metadata-only op (no coefficients needed).
    Free,
    /// `provenance: judge_measured` — the Judge populates the real cost;
    /// shape-hint expressions or all-`~` are both legitimate.
    JudgeMeasured,
    /// `provenance: declared` with a usable `flops`/`bytes_moved` AST.
    DeclaredFormula,
    /// `provenance: declared` with a pinned `cost.cost_fn` (a real fn wins
    /// outright over any declared AST).
    VendorSpec,
}

/// Classify a contract's cost block. A cost block is load-bearing iff it maps
/// to `Some(kind)`. `class: free` is the only no-expression license for a
/// `declared` block; otherwise `declared` needs a pinned `cost_fn` OR a usable
/// `flops`/`bytes_moved` expression to not be a bare placeholder.
pub fn classify_cost(provenance: &str, class: &str, has_any_expr: bool, has_cost_fn: bool) -> Option<CostClassKind> {
    if class == "free" { return Some(CostClassKind::Free); }
    match provenance {
        "judge_measured" => Some(CostClassKind::JudgeMeasured),
        "declared" if has_cost_fn => Some(CostClassKind::VendorSpec),
        "declared" if has_any_expr => Some(CostClassKind::DeclaredFormula),
        _ => None,
    }
}

/// Bounded, dedup'd process-lifetime leak (mirrors `register::intern`). Unknown → None.
pub fn intern_cost_expr(expr: &CompiledCostExpr) -> Option<&'static CompiledCostExpr> {
    if matches!(expr, CompiledCostExpr::Unknown) { return None; }
    static POOL: OnceLock<Mutex<Vec<&'static CompiledCostExpr>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = pool.lock().expect("cost_expr interner poisoned");
    if let Some(&e) = g.iter().find(|&&x| x == expr) { return Some(e); }
    let leaked: &'static CompiledCostExpr = Box::leak(Box::new(expr.clone()));
    g.push(leaked);
    Some(leaked)
}

/// A contract-pinned cost_fn wins outright (stays on entry.cost); the declared
/// AST does not compete with it, so return None when a fn is pinned.
pub fn stamp_primitive_cost_expr(p: &ResolvedPrimitive) -> Option<&'static CompiledCostExpr> {
    if p.cost_fn.is_some() { return None; }
    intern_cost_expr(&p.cost)
}
/// Intern a fused op's declared cost AST. Consumed by the fused registration
/// path (Task 2.4: `BackendImpl.cost_expr` wiring).
pub fn stamp_fused_cost_expr(f: &ResolvedFused) -> Option<&'static CompiledCostExpr> {
    intern_cost_expr(&f.cost)
}

/// Minimal fused-cost symbol binder: `n` (last input elem_count) + `dtype_bytes`.
/// A fused (m,n,k) formula would under-bind and eval-error → the caller falls
/// back to the compose-from-decompose estimate (already a non-zero cost).
pub fn fused_cost_estimate(expr: &CompiledCostExpr, input_shapes: &[Shape], input_dtypes: &[DType], _params: &FusedOpParams)
    -> Result<CostEstimate, CostEvalError> {
    let mut b = std::collections::HashMap::new();
    if let Some(s) = input_shapes.last() { b.insert("n".to_string(), s.elem_count() as f64); }
    if let Some(d) = input_dtypes.last() { b.insert("dtype_bytes".to_string(), d.size_in_bytes() as f64); }
    Ok(CostEstimate { flops: eval(expr, &b)?.max(0.0) as u64, ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::cost_expr::compile_field;

    #[test]
    fn unknown_cost_expr_is_never_interned() {
        assert!(
            intern_cost_expr(&CompiledCostExpr::Unknown).is_none(),
            "Unknown must map to None, never a leaked handle"
        );
    }

    #[test]
    fn equal_expressions_dedup_to_the_same_leaked_pointer() {
        // Two independently-parsed but textually-equal expressions must
        // intern to the SAME `'static` pointer (PartialEq-keyed dedup), not
        // grow the pool unboundedly on repeated imports of the same
        // contract shape.
        let a = compile_field(Some("n")).expect("parses");
        let b = compile_field(Some("n")).expect("parses");
        let pa = intern_cost_expr(&a).expect("interned");
        let pb = intern_cost_expr(&b).expect("interned");
        assert!(
            std::ptr::eq(pa, pb),
            "equal cost expressions must dedup to the same leaked pointer"
        );

        // A distinct expression gets a distinct pointer.
        let c = compile_field(Some("2 * n")).expect("parses");
        let pc = intern_cost_expr(&c).expect("interned");
        assert!(
            !std::ptr::eq(pa, pc),
            "a distinct cost expression must not collide with an unrelated one"
        );
    }
}
