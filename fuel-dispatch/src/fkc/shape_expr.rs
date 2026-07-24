//! Fuel's independent typed shape-expression AST + §6.19 canonical wire codec +
//! evaluator (KISS-Ops §6.20). Byte-matches the KISS reference
//! (`conformance/src/shape_expr.rs`), verified against the vendored golden vectors —
//! the shape-side companion to the value oracle. EXPRESSION kind only (`SameAs` +
//! `DimExpr`); the role/index-woven kind (reduce/gather/matmul) is a separate variant
//! (Convergence-C C-2). Every malformed input is a typed decline, never a panic.
//!
//! MOVED (Increment C slice 1, T1): the implementation now lives in
//! `fuel_kernel_seam_types::shape_expr` — its permanent, dependency-free home, so
//! `fuel-graph` can carry `Dim`/`ShapeExpr` in `OpAttrs` without depending on this
//! crate. This module is a re-export shim: every existing
//! `crate::fkc::shape_expr::…` path keeps compiling unchanged. The codec goldens
//! moved with it (see the seam-types test module). The text-surface parser
//! (`shape_expr_parse`) stays here — contract-import machinery, not grammar.

pub use fuel_kernel_seam_types::shape_expr::*;
