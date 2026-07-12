//! Compile a contract's parsed cost AST into the live ranking cost path (§2.3).
//! The parser/evaluator (`cost_expr.rs`) is complete; this only adds wiring.
use std::sync::{Mutex, OnceLock};
use crate::fkc::cost_expr::CompiledCostExpr;
use crate::fkc::lower::{ResolvedPrimitive, ResolvedFused};

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
/// path starting in Task 2.4 (BackendImpl.cost_expr wiring).
#[allow(dead_code)] // wired in Task 2.4
pub fn stamp_fused_cost_expr(f: &ResolvedFused) -> Option<&'static CompiledCostExpr> {
    intern_cost_expr(&f.cost)
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
