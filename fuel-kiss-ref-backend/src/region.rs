//! Composed-**region** reference / differential over the mapped floor.
//!
//! Translates a Fuel recipe region ([`PatternNode`]) into a kiss-ref §6.13
//! expression tree ([`kiss_ops_vocab::decomp::Expr`]) and hands it to kiss-ref's
//! **first-class composed-expression seam** —
//! [`kiss_ref_core::reference_expr`] / [`kiss_ref_core::diff_expr`] and their
//! `_f32`/`_f16`/`_bf16` mirrors (the narrow mirrors were minted for this
//! consumer). That seam is the same [`kiss_ref_core::eval_expr`] engine the
//! previous hand-rolled per-node composition drove, so the delegation is
//! numerically inert — pinned by the migration-equivalence tests below, which
//! keep a verbatim copy of the old loop as an oracle. This makes multi-node
//! elementwise regions (a fused op's `decompose`) diffable against kiss-ref,
//! not just single primitives.
//!
//! The `PatternNode` → `Expr` translation stays Fuel's (it is Fuel's mapping),
//! as does the advisory ULP band and every typed decline.
//!
//! Scope: elementwise, default-attrs, mapped ops only. `SeeThrough`/`Any` are
//! matcher-only and decline ([`KissRefError::UnsupportedNode`]); a node carrying
//! non-default [`OpAttrs`] declines ([`KissRefError::UnsupportedAttrs`]) because
//! the kiss `Expr` grammar has no attribute channel. Declines are typed
//! coverage gaps, never panics — per KISS-CONFORM §6.6-0007 this whole path is
//! advisory (kiss-ref flags, never verdicts).

use crate::mapping::{op_to_kiss, supports};
use crate::reference::to_rows;
use crate::KissRefError;
use fuel_ir::DType;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use kiss_ops_vocab::decomp::Expr;
use kiss_ref_core::{DiffReport, Tolerance};

/// Whether the region evaluator covers `(region, dtype)`: at least one op node,
/// and **every** op node is mapped into kiss-ref's vocabulary with
/// `Support::Done` on `dtype` and default (empty) attrs. Matcher-only nodes
/// (`SeeThrough`/`Any`) decline. This is the gate the advisory cross-check
/// consults before calling `reference_region_*`/`diff_region_*`.
pub fn region_supported(region: &PatternNode, dtype: DType) -> bool {
    region_op_count(region) >= 1 && node_supported(region, dtype)
}

fn node_supported(node: &PatternNode, dtype: DType) -> bool {
    match node {
        PatternNode::Op { op, operands, attrs } => {
            *attrs == OpAttrs::default()
                && supports(*op, dtype)
                && operands.iter().all(|o| node_supported(o, dtype))
        }
        PatternNode::Bind { .. } => true,
        PatternNode::SeeThrough { .. } | PatternNode::Any => false,
    }
}

/// The number of `Op` nodes in a region (`Bind`/`Any` are leaves and count 0;
/// `SeeThrough` is counted *through* — structural metadata, even though
/// [`region_supported`] declines it for evaluation).
pub fn region_op_count(region: &PatternNode) -> usize {
    match region {
        PatternNode::Op { operands, .. } => {
            1 + operands.iter().map(region_op_count).sum::<usize>()
        }
        PatternNode::SeeThrough { then } => region_op_count(then),
        PatternNode::Bind { .. } | PatternNode::Any => 0,
    }
}

// ---- advisory tolerance band (kiss-ref refinement, 2026-07-23) --------------
//
// REFERENCE-ONLY. The functions below (`op_ulp_ceiling`, `region_ulp_ceilings`,
// `region_advisory_tolerance`) are the canonical statement of the §6.8 advisory
// band, but they are NOT on the live ingestion path: this whole adapter crate
// is pulled only under `fuel-dispatch`'s `cuda` feature, whereas the live band
// (`fuel_dispatch::jit_ingest::advisory_ulp_band`) must compute under
// `--features jit` alone, without this cuda-gated adapter. So the live path
// carries its own copy of this same formula. The two are hand-maintained and
// cannot be co-compiled on a CPU build; they are kept from drifting by a shared
// fixture — `fuel_kernel_seam_types::advisory_band_reference_cases()` — that
// BOTH sides assert against (adapter side: `advisory_band_matches_shared_cases`
// below; live side: `advisory_ulp_band_matches_shared_cases` in jit_ingest.rs).
// If you change the formula here, change it there and update that one fixture.

/// A transcendental op tag — one whose hardware value can differ from the
/// wide-precision truth by more than a correctly-rounded op. **Mirrors
/// fuel-dispatch `fkc/verify/ulp.rs::is_transcendental` exactly** so the two
/// classifications never drift: IEEE requires `Sqrt`/`Recip` to be
/// correctly-rounded, so they are NOT here (they count as exact ops in the
/// band).
fn is_transcendental_tag(op: OpTag) -> bool {
    use OpTag as T;
    matches!(
        op,
        T::Exp
            | T::Log
            | T::Sin
            | T::Cos
            | T::Tanh
            | T::Sigmoid
            | T::Silu
            | T::Gelu
            | T::GeluErf
            | T::Erf
            | T::Rsqrt
    )
}

/// The fallback per-op ceiling for a transcendental whose kiss `Op` exposes no
/// declared §6.8 ceiling (kiss non-primitives — `Tanh`, `Sigmoid`, `Silu`,
/// `GeluTanh`, `Gelu`, `Rsqrt` — inherit their decomposition's tolerance and
/// return `None` from `Op::ulp_ceiling`). Per the kiss-ref band refinement
/// (2026-07-23): treat a mapped op with no exposed ceiling as 4 ULP.
const FALLBACK_TRANSCENDENTAL_ULP_CEILING: u64 = 4;

/// The per-op §6.8 ULP ceiling contribution for the advisory band:
/// `Some(ceiling)` for a transcendental op tag (read from kiss-ref's
/// `Op::ulp_ceiling` where declared, else the 4-ULP fallback), `None` for an
/// exact op (contributes to the exact-rounding term instead).
pub fn op_ulp_ceiling(op: OpTag) -> Option<u64> {
    if !is_transcendental_tag(op) {
        return None;
    }
    let declared = op_to_kiss(op).and_then(|k| k.ulp_ceiling());
    // `ceil() as u64` is never-panic (saturating cast); declared ceilings are
    // small integers (2/4/8) today.
    Some(declared.map_or(FALLBACK_TRANSCENDENTAL_ULP_CEILING, |c| c.ceil() as u64))
}

/// Region ceiling metadata: every `Op` node in pre-order (root first, operands
/// left-to-right; `SeeThrough` traversed through) with its per-op ceiling
/// contribution per [`op_ulp_ceiling`]. This is the input the advisory-band
/// computation consumes; exposed so callers can ledger the per-op breakdown.
pub fn region_ulp_ceilings(region: &PatternNode) -> Vec<(OpTag, Option<u64>)> {
    let mut out = Vec::new();
    collect_ceilings(region, &mut out);
    out
}

fn collect_ceilings(node: &PatternNode, out: &mut Vec<(OpTag, Option<u64>)>) {
    match node {
        PatternNode::Op { op, operands, .. } => {
            out.push((*op, op_ulp_ceiling(*op)));
            for o in operands {
                collect_ceilings(o, out);
            }
        }
        PatternNode::SeeThrough { then } => collect_ceilings(then, out),
        PatternNode::Bind { .. } | PatternNode::Any => {}
    }
}

/// The advisory comparison band for a region, per the kiss-ref tolerance
/// refinement (2026-07-23):
///
/// * single exact op → [`Tolerance::Exact`];
/// * multi-node exact-only region → `Ulp(n_ops - 1)` (each intermediate
///   rounding contributes at most ~1 ULP; the final rounding matches the
///   reference's);
/// * transcendental-containing region → `Ulp(Σ per-op §6.8 ceilings over the
///   region's transcendental ops + (n_exact_ops - 1))`, where the exact term
///   saturates at 0 (an all-transcendental region adds no exact-rounding
///   term, so a lone transcendental keeps exactly its own ceiling).
///
/// **Cancellation caveat (pinned):** linear ULP addition is a first-order
/// model — cancellation-heavy regions (e.g. subtractions of nearby
/// intermediates) can exceed the band and flag spuriously. The label stays
/// advisory-only per KISS-CONFORM §6.6-0007 (kiss-ref flags, never verdicts),
/// and the raw `max_ulp` is always recorded alongside the flag.
///
/// `None` iff the region contains no op nodes (nothing to band).
pub fn region_advisory_tolerance(region: &PatternNode) -> Option<Tolerance> {
    let ceilings = region_ulp_ceilings(region);
    let n_ops = ceilings.len();
    if n_ops == 0 {
        return None;
    }
    let trans_sum: u64 = ceilings
        .iter()
        .filter_map(|&(_, c)| c)
        .fold(0u64, |acc, c| acc.saturating_add(c));
    let n_trans = ceilings.iter().filter(|(_, c)| c.is_some()).count();
    let n_exact = n_ops - n_trans;
    Some(if n_trans == 0 {
        if n_ops == 1 {
            Tolerance::Exact
        } else {
            Tolerance::Ulp((n_ops - 1) as u64)
        }
    } else {
        Tolerance::Ulp(trans_sum.saturating_add(n_exact.saturating_sub(1) as u64))
    })
}

// ---- PatternNode -> Expr translation + row-wise evaluation ------------------

/// Translate a recipe region into a kiss §6.13 expression tree. `Bind{i}` →
/// `Input(i)`; a default-attrs mapped `Op` → `Apply`; everything else declines
/// with a typed error (never panics).
fn region_to_expr(region: &PatternNode) -> Result<Expr, KissRefError> {
    match region {
        PatternNode::Bind { index } => Ok(Expr::Input(*index)),
        PatternNode::Op { op, operands, attrs } => {
            if *attrs != OpAttrs::default() {
                return Err(KissRefError::UnsupportedAttrs(*op));
            }
            let kiss_op = op_to_kiss(*op).ok_or(KissRefError::UnsupportedOp(*op))?;
            let args = operands
                .iter()
                .map(region_to_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::Apply(kiss_op, args))
        }
        PatternNode::SeeThrough { .. } | PatternNode::Any => Err(KissRefError::UnsupportedNode),
    }
}

/// Validate + prepare a region evaluation: the region must be op-rooted, fully
/// translatable, and the operand column count must match the region's bind
/// arity (`max bind index + 1`). Returns the translated tree plus the
/// transposed per-element rows.
fn region_rows<T: Copy>(
    region: &PatternNode,
    operands: &[&[T]],
) -> Result<(Expr, Vec<Vec<T>>), KissRefError> {
    let root = match region {
        PatternNode::Op { op, .. } => *op,
        // A region that is not op-rooted (bare Bind / matcher-only root)
        // computes nothing — decline.
        _ => return Err(KissRefError::UnsupportedNode),
    };
    let expr = region_to_expr(region)?;
    // bind_indices() is sorted+deduped, so last() is the max index.
    let expected = region.bind_indices().last().map_or(0, |&m| m as usize + 1);
    if expected != 0 && operands.len() != expected {
        return Err(KissRefError::Arity { op: root, expected, got: operands.len() });
    }
    let rows = to_rows(root, operands)?;
    Ok((expr, rows))
}

/// Borrow the transposed rows in the `&[&[T]]` shape kiss-ref's composed seam
/// takes.
fn row_refs<T>(rows: &[Vec<T>]) -> Vec<&[T]> {
    rows.iter().map(|r| r.as_slice()).collect()
}

/// Map a kiss-ref evaluation error onto Fuel's never-panic decline surface.
///
/// One arm is load-bearing: kiss's `diff_expr_*` raises
/// [`kiss_ref_core::Error::LengthMismatch`] for a candidate whose length does
/// not match the reference's, and that is *Fuel's own* typed decline
/// ([`KissRefError::LengthMismatch`]) — the adapter promised it before the
/// migration onto kiss's seam and keeps promising it after, rather than burying
/// it inside an opaque `Eval`. Everything else wraps unchanged.
fn map_kiss_error(e: kiss_ref_core::Error) -> KissRefError {
    match e {
        kiss_ref_core::Error::LengthMismatch { expected, got } => {
            KissRefError::LengthMismatch { expected, got }
        }
        other => KissRefError::Eval(other),
    }
}

// The region lanes delegate to kiss-ref's composed-`Expr` seam
// (`reference_expr_*` / `diff_expr_*`), which owns the row-wise `eval_expr`
// walk AND the `DiffReport` loop — per-row ULP distance in the dtype's own
// lattice (both-NaN = 0, exactly-one-NaN = the dtype lattice MAX: u32::MAX for
// f32, u16::MAX for the narrow floats, widened to u64), running `max_ulp`, and
// `first_mismatch` carrying the dtype value widened to f64. Fuel keeps only
// what is Fuel's: the `PatternNode` -> `Expr` translation, the arity/attrs/
// matcher-node guards, and the decline typing.
macro_rules! region_float {
    ($refr:ident, $diff:ident, $t:ty, $kref:path, $kdiff:path) => {
        /// kiss-ref's reference output for a composed `region` over column-major
        /// `operands` (one slice per region input, matched by `Bind` index).
        pub fn $refr(
            region: &PatternNode,
            operands: &[&[$t]],
        ) -> Result<Vec<$t>, KissRefError> {
            let (expr, rows) = region_rows(region, operands)?;
            $kref(&expr, &row_refs(&rows)).map_err(map_kiss_error)
        }

        /// Differential of `candidate` vs kiss-ref's composed-region reference
        /// over `operands`, under `tol`. `candidate` holds one value per
        /// element.
        pub fn $diff(
            region: &PatternNode,
            candidate: &[$t],
            operands: &[&[$t]],
            tol: Tolerance,
        ) -> Result<DiffReport, KissRefError> {
            let (expr, rows) = region_rows(region, operands)?;
            $kdiff(&expr, &row_refs(&rows), candidate, tol).map_err(map_kiss_error)
        }
    };
}

region_float!(
    reference_region_f32,
    diff_region_f32,
    f32,
    kiss_ref_core::reference_expr_f32,
    kiss_ref_core::diff_expr_f32
);
region_float!(
    reference_region_f64,
    diff_region_f64,
    f64,
    kiss_ref_core::reference_expr,
    kiss_ref_core::diff_expr
);
region_float!(
    reference_region_f16,
    diff_region_f16,
    half::f16,
    kiss_ref_core::reference_expr_f16,
    kiss_ref_core::diff_expr_f16
);
region_float!(
    reference_region_bf16,
    diff_region_bf16,
    half::bf16,
    kiss_ref_core::reference_expr_bf16,
    kiss_ref_core::diff_expr_bf16
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KissRefError;
    use fuel_ir::DType;
    use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
    use kiss_ref_core::Tolerance;

    fn bind(i: u8) -> PatternNode {
        PatternNode::Bind { index: i }
    }

    fn node(tag: OpTag, operands: Vec<PatternNode>) -> PatternNode {
        PatternNode::Op { op: tag, operands, attrs: OpAttrs::default() }
    }

    /// relu(a + b) — the canonical 2-op, 2-input region.
    fn relu_add() -> PatternNode {
        node(OpTag::Relu, vec![node(OpTag::Add, vec![bind(0), bind(1)])])
    }

    // ---- reference evaluation ------------------------------------------------

    #[test]
    fn region_relu_add_matches_hand_math() {
        let a = [1.0f32, -5.0, 2.5, 0.0];
        let b = [2.0f32, 3.0, -4.0, -0.0];
        let out = reference_region_f32(&relu_add(), &[&a, &b]).unwrap();
        assert_eq!(out, vec![3.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn region_shared_bind_is_node_identity() {
        // mul(x, x) — one input read twice.
        let region = node(OpTag::Mul, vec![bind(0), bind(0)]);
        let x = [3.0f32, -2.0];
        let out = reference_region_f32(&region, &[&x]).unwrap();
        assert_eq!(out, vec![9.0, 4.0]);
    }

    #[test]
    fn region_composition_matches_per_op_reference_chain() {
        // exp(a + b) through the region evaluator == Add then Exp through the
        // per-op adapter (same kiss kernels => byte-identical).
        let a = [0.5f32, -1.25, 3.0];
        let b = [0.25f32, 1.0, -2.5];
        let region = node(OpTag::Exp, vec![node(OpTag::Add, vec![bind(0), bind(1)])]);
        let got = reference_region_f32(&region, &[&a, &b]).unwrap();
        let sum = crate::reference::reference_f32(OpTag::Add, &[&a, &b]).unwrap();
        let expect = crate::reference::reference_f32(OpTag::Exp, &[&sum]).unwrap();
        assert_eq!(got, expect);
    }

    // ---- support / metadata --------------------------------------------------

    #[test]
    fn region_supported_accepts_mapped_default_attr_region() {
        assert!(region_supported(&relu_add(), DType::F32));
        assert!(region_supported(&relu_add(), DType::F64));
        assert!(region_supported(&relu_add(), DType::F16));
        assert!(region_supported(&relu_add(), DType::BF16));
    }

    #[test]
    fn region_supported_requires_at_least_one_op_node() {
        assert!(!region_supported(&bind(0), DType::F32));
    }

    #[test]
    fn region_supported_declines_unmapped_op() {
        let region = node(OpTag::MatMul, vec![bind(0), bind(1)]);
        assert!(!region_supported(&region, DType::F32));
    }

    #[test]
    fn region_supported_declines_nondefault_attrs() {
        let region = PatternNode::Op {
            op: OpTag::AddScalar,
            operands: vec![bind(0)],
            attrs: OpAttrs { scalars: vec![1.0], ..OpAttrs::default() },
        };
        assert!(!region_supported(&region, DType::F32));
    }

    #[test]
    fn region_supported_declines_matcher_only_nodes() {
        let st = PatternNode::SeeThrough { then: Box::new(relu_add()) };
        assert!(!region_supported(&st, DType::F32));
        let any_operand = node(OpTag::Relu, vec![PatternNode::Any]);
        assert!(!region_supported(&any_operand, DType::F32));
    }

    #[test]
    fn region_supported_declines_unmapped_dtype() {
        assert!(!region_supported(&relu_add(), DType::F6E2M3));
    }

    #[test]
    fn region_op_count_counts_op_nodes_only() {
        assert_eq!(region_op_count(&relu_add()), 2);
        assert_eq!(region_op_count(&bind(0)), 0);
        // SeeThrough is counted *through* (structural metadata), though
        // region_supported declines it.
        let st = PatternNode::SeeThrough { then: Box::new(relu_add()) };
        assert_eq!(region_op_count(&st), 2);
    }

    // ---- decline paths (typed, never panic) ----------------------------------

    #[test]
    fn matcher_only_nodes_err_unsupported_node() {
        let a = [1.0f32];
        let st = PatternNode::SeeThrough { then: Box::new(relu_add()) };
        assert!(matches!(
            reference_region_f32(&st, &[&a, &a]),
            Err(KissRefError::UnsupportedNode)
        ));
        let any_operand = node(OpTag::Relu, vec![PatternNode::Any]);
        assert!(matches!(
            reference_region_f32(&any_operand, &[&a]),
            Err(KissRefError::UnsupportedNode)
        ));
    }

    #[test]
    fn nondefault_attrs_err_unsupported_attrs() {
        let region = PatternNode::Op {
            op: OpTag::AddScalar,
            operands: vec![bind(0)],
            attrs: OpAttrs { scalars: vec![1.0], ..OpAttrs::default() },
        };
        let x = [1.0f32];
        assert!(matches!(
            reference_region_f32(&region, &[&x]),
            Err(KissRefError::UnsupportedAttrs(OpTag::AddScalar))
        ));
    }

    #[test]
    fn unmapped_op_errs_unsupported_op() {
        let region = node(OpTag::MatMul, vec![bind(0), bind(1)]);
        let x = [1.0f32];
        assert!(matches!(
            reference_region_f32(&region, &[&x, &x]),
            Err(KissRefError::UnsupportedOp(OpTag::MatMul))
        ));
    }

    #[test]
    fn region_arity_mismatch_errs() {
        let a = [1.0f32, 2.0];
        assert!(matches!(
            reference_region_f32(&relu_add(), &[&a]),
            Err(KissRefError::Arity { op: OpTag::Relu, expected: 2, got: 1 })
        ));
    }

    #[test]
    fn ragged_region_operands_err_not_panic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32];
        assert!(matches!(
            reference_region_f32(&relu_add(), &[&a, &b]),
            Err(KissRefError::LengthMismatch { .. })
        ));
    }

    // ---- differential --------------------------------------------------------

    #[test]
    fn diff_region_matching_candidate_conforms() {
        let a = [1.0f32, -5.0, 2.5];
        let b = [2.0f32, 3.0, -4.0];
        let cand = [3.0f32, 0.0, 0.0];
        let rep = diff_region_f32(&relu_add(), &cand, &[&a, &b], Tolerance::Exact).unwrap();
        assert!(rep.conforms());
        assert_eq!(rep.n, 3);
        assert_eq!(rep.max_ulp, 0);
    }

    #[test]
    fn diff_region_flags_planted_error() {
        let a = [1.0f32, -5.0, 2.5];
        let b = [2.0f32, 3.0, -4.0];
        let cand = [3.0f32, 999.0, 0.0]; // row 1 corrupted
        let rep = diff_region_f32(&relu_add(), &cand, &[&a, &b], Tolerance::Exact).unwrap();
        assert!(!rep.conforms());
        assert_eq!(rep.mismatches, 1);
        let (idx, reference, got) = rep.first_mismatch.unwrap();
        assert_eq!(idx, 1);
        assert_eq!(reference, 0.0);
        assert_eq!(got, 999.0);
        assert!(rep.max_ulp > 0);
    }

    #[test]
    fn diff_region_candidate_length_mismatch_errs() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let cand = [4.0f32];
        assert!(matches!(
            diff_region_f32(&relu_add(), &cand, &[&a, &b], Tolerance::Exact),
            Err(KissRefError::LengthMismatch { expected: 2, got: 1 })
        ));
    }

    #[test]
    fn diff_region_nan_semantics_mirror_kiss() {
        // add(NaN, 1) -> NaN reference. Both-NaN = 0 ULP (conforms); one-NaN =
        // the dtype lattice's MAX (u32::MAX for f32, NOT u64::MAX) — replicating
        // kiss diff_f32's loop semantics exactly.
        let region = node(OpTag::Add, vec![bind(0), bind(1)]);
        let a = [f32::NAN];
        let b = [1.0f32];
        let both =
            diff_region_f32(&region, &[f32::NAN], &[&a, &b], Tolerance::Exact).unwrap();
        assert!(both.conforms());
        let one = diff_region_f32(&region, &[0.0], &[&a, &b], Tolerance::Exact).unwrap();
        assert!(!one.conforms());
        assert_eq!(one.max_ulp, u32::MAX as u64);
    }

    #[test]
    fn region_wide_and_narrow_dtypes_evaluate() {
        use half::{bf16, f16};
        // f64.
        let a64 = [1.5f64, -2.0];
        let b64 = [2.5f64, 0.5];
        assert_eq!(
            reference_region_f64(&relu_add(), &[&a64, &b64]).unwrap(),
            vec![4.0, 0.0]
        );
        // f16: a planted 1-ULP error fails Exact, conforms at Ulp(1).
        let a16: Vec<f16> = [1.0f32, -5.0].iter().map(|&v| f16::from_f32(v)).collect();
        let b16: Vec<f16> = [2.0f32, 3.0].iter().map(|&v| f16::from_f32(v)).collect();
        let reference = reference_region_f16(&relu_add(), &[&a16, &b16]).unwrap();
        assert_eq!(reference[0].to_f32(), 3.0);
        let cand: Vec<f16> =
            reference.iter().map(|&x| f16::from_bits(x.to_bits() + 1)).collect();
        assert!(!diff_region_f16(&relu_add(), &cand, &[&a16, &b16], Tolerance::Exact)
            .unwrap()
            .conforms());
        assert!(diff_region_f16(&relu_add(), &cand, &[&a16, &b16], Tolerance::Ulp(1))
            .unwrap()
            .conforms());
        // bf16.
        let ab: Vec<bf16> = [4.0f32, -1.0].iter().map(|&v| bf16::from_f32(v)).collect();
        let bb: Vec<bf16> = [0.5f32, 0.25].iter().map(|&v| bf16::from_f32(v)).collect();
        let out = reference_region_bf16(&relu_add(), &[&ab, &bb]).unwrap();
        assert_eq!(out[0].to_f32(), 4.5);
        assert_eq!(out[1].to_f32(), 0.0);
    }

    // ---- advisory tolerance band (kiss-ref refinement, 2026-07-23) -----------

    #[test]
    fn op_ulp_ceiling_reads_kiss_declared_and_defaults() {
        // §6.8-declared atoms read kiss's Op::ulp_ceiling.
        assert_eq!(op_ulp_ceiling(OpTag::Exp), Some(4));
        assert_eq!(op_ulp_ceiling(OpTag::Erf), Some(4));
        // Transcendentals that are kiss non-primitives (no declared ceiling)
        // default to 4.
        assert_eq!(op_ulp_ceiling(OpTag::Tanh), Some(4));
        assert_eq!(op_ulp_ceiling(OpTag::Rsqrt), Some(4));
        // Exact ops carry no ceiling; Sqrt is IEEE correctly-rounded => exact
        // class (mirrors fuel-dispatch fkc/verify/ulp.rs).
        assert_eq!(op_ulp_ceiling(OpTag::Add), None);
        assert_eq!(op_ulp_ceiling(OpTag::Sqrt), None);
    }

    #[test]
    fn region_ulp_ceilings_lists_op_nodes_preorder() {
        assert_eq!(
            region_ulp_ceilings(&relu_add()),
            vec![(OpTag::Relu, None), (OpTag::Add, None)]
        );
        let region = node(OpTag::Exp, vec![bind(0)]);
        assert_eq!(region_ulp_ceilings(&region), vec![(OpTag::Exp, Some(4))]);
    }

    #[test]
    fn advisory_tolerance_single_exact_op_is_exact() {
        let region = node(OpTag::Add, vec![bind(0), bind(1)]);
        assert_eq!(region_advisory_tolerance(&region), Some(Tolerance::Exact));
    }

    #[test]
    fn advisory_tolerance_multi_node_exact_region_is_n_minus_one() {
        assert_eq!(region_advisory_tolerance(&relu_add()), Some(Tolerance::Ulp(1)));
    }

    #[test]
    fn advisory_tolerance_transcendental_region_sums_ceilings() {
        // exp(a+b): 1 transcendental (4) + 1 exact -> 4 + (1-1) = 4.
        let exp_add = node(OpTag::Exp, vec![node(OpTag::Add, vec![bind(0), bind(1)])]);
        assert_eq!(region_advisory_tolerance(&exp_add), Some(Tolerance::Ulp(4)));
        // silu(add(a, exp(b))): transcendentals {Silu: 4 default, Exp: 4} +
        // 1 exact -> 8 + (1-1) = 8.
        let region = node(
            OpTag::Silu,
            vec![node(OpTag::Add, vec![bind(0), node(OpTag::Exp, vec![bind(1)])])],
        );
        assert_eq!(region_advisory_tolerance(&region), Some(Tolerance::Ulp(8)));
        // All-transcendental exp(tanh(x)): 4 + 4; the exact term saturates at 0
        // (never -1).
        let et = node(OpTag::Exp, vec![node(OpTag::Tanh, vec![bind(0)])]);
        assert_eq!(region_advisory_tolerance(&et), Some(Tolerance::Ulp(8)));
        // A lone transcendental keeps its own ceiling.
        let e = node(OpTag::Exp, vec![bind(0)]);
        assert_eq!(region_advisory_tolerance(&e), Some(Tolerance::Ulp(4)));
    }

    #[test]
    fn advisory_tolerance_none_for_op_free_region() {
        assert_eq!(region_advisory_tolerance(&bind(0)), None);
    }

    /// DRIFT GUARD (this crate's half). Pins the reference band formula
    /// (`region_advisory_tolerance`) to the shared fixture that the live copy
    /// in `fuel-dispatch::jit_ingest::advisory_ulp_band` is pinned to from the
    /// other side. The two formulas cannot be co-compiled on a CPU build (this
    /// adapter is cuda-gated), so the shared
    /// `advisory_band_reference_cases()` table is the only thing keeping them in
    /// lockstep — if either drifts, its side fails against this table. The
    /// adapter's richer `Option<Tolerance>` is normalized to the fixture's
    /// `Option<u64>` shape: `None`(op-free) and `Some(Exact)`(single exact op)
    /// both collapse to `None` (an exact comparison), `Some(Ulp(n))` to
    /// `Some(n)`.
    #[test]
    fn advisory_band_matches_shared_cases() {
        fn normalize(t: Option<Tolerance>) -> Option<u64> {
            match t {
                None => None,                    // op-free region
                Some(Tolerance::Exact) => None,  // single exact op
                Some(Tolerance::Ulp(n)) => Some(n),
            }
        }
        for (region, expected) in
            fuel_kernel_seam_types::advisory_band_reference_cases()
        {
            assert_eq!(
                normalize(region_advisory_tolerance(&region)),
                expected,
                "reference band drifted from the shared fixture for {region:?}"
            );
        }
    }

    // ---- kiss-ref composed-expression seam (rev 1f3981f) ---------------------

    /// Guards the kiss-ref rev pin. `diff_expr_f32`/`diff_expr_f16` are the
    /// composed-`Expr` mirrors kiss-ref minted **for this consumer** (see the
    /// `narrow_expr!` note in kiss-ref-core `diff.rs`); they do not exist at the
    /// previous pin `b75a748`, so this test fails to COMPILE against the old
    /// rev — which is the only way to prove the bump took effect rather than a
    /// cached checkout silently resolving the old tree.
    ///
    /// It also pinned the migration's premise: kiss's own seam is byte-identical
    /// to the hand-rolled `region_float!` loop, wide lane and narrow lane. Now
    /// that `region_float!` *is* that seam, this half is near-tautological — the
    /// rev-pin compile guard is what it still buys; the equivalence claim is
    /// carried by the `migration_*` tests below (which keep the old loop as an
    /// explicit oracle).
    #[test]
    fn kiss_expr_seam_matches_hand_rolled_region_diff() {
        let expr = region_to_expr(&relu_add()).unwrap();

        // Wide lane (f32), with a planted mismatch so every DiffReport field is
        // exercised — not just the all-conforming happy path.
        let a = [1.0f32, -5.0, 2.5];
        let b = [2.0f32, 3.0, -4.0];
        let cand = [3.0f32, 999.0, 0.0]; // row 1 corrupted
        let ours = diff_region_f32(&relu_add(), &cand, &[&a, &b], Tolerance::Exact).unwrap();
        let rows = crate::reference::to_rows(OpTag::Relu, &[&a, &b]).unwrap();
        let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let theirs =
            kiss_ref_core::diff_expr_f32(&expr, &row_refs, &cand, Tolerance::Exact).unwrap();
        assert_eq!(ours.n, theirs.n);
        assert_eq!(ours.mismatches, theirs.mismatches);
        assert_eq!(ours.max_ulp, theirs.max_ulp);
        assert_eq!(ours.first_mismatch, theirs.first_mismatch);

        // Narrow lane (f16) — the mirror kiss-ref added specifically because
        // Fuel's advisory covers f16/bf16.
        use half::f16;
        let a16: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
        let b16: Vec<f16> = b.iter().map(|&v| f16::from_f32(v)).collect();
        let ours16 = reference_region_f16(&relu_add(), &[&a16, &b16]).unwrap();
        let rows16 = crate::reference::to_rows(OpTag::Relu, &[&a16, &b16]).unwrap();
        let row_refs16: Vec<&[f16]> = rows16.iter().map(|r| r.as_slice()).collect();
        let theirs16 = kiss_ref_core::reference_expr_f16(&expr, &row_refs16).unwrap();
        assert_eq!(ours16, theirs16);
    }

    // ---- K2 migration equivalence: hand-rolled loop -> kiss-ref's seam -------
    //
    // `reference_region_*`/`diff_region_*` used to compose the reference by hand:
    // translate the region to an `Expr`, then run kiss's `eval_expr` row-wise and
    // build the `DiffReport` in a local loop. They now delegate to kiss-ref's
    // first-class `reference_expr_*`/`diff_expr_*`. That is the SAME engine, so
    // the swap must be numerically inert — these tests pin it by keeping a
    // verbatim copy of the pre-migration composition (`legacy_region_float!`)
    // and asserting new == old, field for field, on every lane.

    /// Verbatim copy of the pre-migration `region_float!` body — the hand-rolled
    /// per-node composition the migration replaces. Test-only oracle.
    macro_rules! legacy_region_float {
        ($refr:ident, $diff:ident, $t:ty, $ulp:path, $wide:expr) => {
            fn $refr(region: &PatternNode, operands: &[&[$t]]) -> Result<Vec<$t>, KissRefError> {
                let (expr, rows) = region_rows(region, operands)?;
                rows.iter()
                    .map(|r| kiss_ref_core::eval_expr(&expr, r).map_err(KissRefError::Eval))
                    .collect()
            }

            fn $diff(
                region: &PatternNode,
                candidate: &[$t],
                operands: &[&[$t]],
                tol: Tolerance,
            ) -> Result<DiffReport, KissRefError> {
                let (expr, rows) = region_rows(region, operands)?;
                let reference: Vec<$t> = rows
                    .iter()
                    .map(|r| kiss_ref_core::eval_expr(&expr, r).map_err(KissRefError::Eval))
                    .collect::<Result<Vec<$t>, KissRefError>>()?;
                if candidate.len() != reference.len() {
                    return Err(KissRefError::LengthMismatch {
                        expected: reference.len(),
                        got: candidate.len(),
                    });
                }
                let mut report = DiffReport {
                    n: reference.len(),
                    mismatches: 0,
                    max_ulp: 0,
                    first_mismatch: None,
                };
                for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
                    let d = $ulp(e, g) as u64;
                    if d > report.max_ulp {
                        report.max_ulp = d;
                    }
                    let ok = match tol {
                        Tolerance::Exact => d == 0,
                        Tolerance::Ulp(n) => d <= n,
                    };
                    if !ok {
                        report.mismatches += 1;
                        if report.first_mismatch.is_none() {
                            report.first_mismatch = Some((i, $wide(e), $wide(g)));
                        }
                    }
                }
                Ok(report)
            }
        };
    }

    legacy_region_float!(
        legacy_reference_region_f32,
        legacy_diff_region_f32,
        f32,
        kiss_ref_core::ulp_distance_f32,
        |x: f32| x as f64
    );
    legacy_region_float!(
        legacy_reference_region_f64,
        legacy_diff_region_f64,
        f64,
        kiss_ref_core::ulp_distance_f64,
        |x: f64| x
    );
    legacy_region_float!(
        legacy_reference_region_f16,
        legacy_diff_region_f16,
        half::f16,
        kiss_ref_core::ulp_distance_f16,
        |x: half::f16| x.to_f32() as f64
    );
    legacy_region_float!(
        legacy_reference_region_bf16,
        legacy_diff_region_bf16,
        half::bf16,
        kiss_ref_core::ulp_distance_bf16,
        |x: half::bf16| x.to_f32() as f64
    );

    /// `a*a - b*b` — a 3-op, 2-input, exact-only region with a shared bind on
    /// each side. Mirrors the coverage shape kiss-ref's own `diff_expr` tests
    /// use, and exercises the translator's operand recursion + bind reuse.
    fn sq_diff() -> PatternNode {
        node(
            OpTag::Sub,
            vec![
                node(OpTag::Mul, vec![bind(0), bind(0)]),
                node(OpTag::Mul, vec![bind(1), bind(1)]),
            ],
        )
    }

    #[test]
    fn migration_sq_diff_byte_exact_on_both_narrow_lanes() {
        use half::{bf16, f16};
        let av = [1.5f32, -3.0, 0.5, 7.0];
        let bv = [0.5f32, 2.0, -1.25, 0.25];

        // f16 lane.
        let a16: Vec<f16> = av.iter().map(|&v| f16::from_f32(v)).collect();
        let b16: Vec<f16> = bv.iter().map(|&v| f16::from_f32(v)).collect();
        let new16 = reference_region_f16(&sq_diff(), &[&a16, &b16]).unwrap();
        let old16 = legacy_reference_region_f16(&sq_diff(), &[&a16, &b16]).unwrap();
        assert_eq!(new16, old16, "f16 region reference moved under the migration");
        // Byte-exact, not merely equal: compare the bit patterns.
        let new_bits: Vec<u16> = new16.iter().map(|x| x.to_bits()).collect();
        let old_bits: Vec<u16> = old16.iter().map(|x| x.to_bits()).collect();
        assert_eq!(new_bits, old_bits);
        // And it is the value hand-math predicts.
        let want16: Vec<f16> = av
            .iter()
            .zip(&bv)
            .map(|(&a, &b)| {
                let (a, b) = (f16::from_f32(a), f16::from_f32(b));
                f16::from_f32(f16::from_f32(a.to_f32() * a.to_f32()).to_f32()
                    - f16::from_f32(b.to_f32() * b.to_f32()).to_f32())
            })
            .collect();
        assert_eq!(new16, want16);

        // bf16 lane.
        let abf: Vec<bf16> = av.iter().map(|&v| bf16::from_f32(v)).collect();
        let bbf: Vec<bf16> = bv.iter().map(|&v| bf16::from_f32(v)).collect();
        let newb = reference_region_bf16(&sq_diff(), &[&abf, &bbf]).unwrap();
        let oldb = legacy_reference_region_bf16(&sq_diff(), &[&abf, &bbf]).unwrap();
        assert_eq!(newb, oldb, "bf16 region reference moved under the migration");
        let newb_bits: Vec<u16> = newb.iter().map(|x| x.to_bits()).collect();
        let oldb_bits: Vec<u16> = oldb.iter().map(|x| x.to_bits()).collect();
        assert_eq!(newb_bits, oldb_bits);

        // Wide lanes too — the whole `region_float!` family migrated, not just
        // the narrow mirrors.
        let new32 = reference_region_f32(&sq_diff(), &[&av, &bv]).unwrap();
        assert_eq!(new32, legacy_reference_region_f32(&sq_diff(), &[&av, &bv]).unwrap());
        let a64: Vec<f64> = av.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = bv.iter().map(|&v| v as f64).collect();
        let new64 = reference_region_f64(&sq_diff(), &[&a64, &b64]).unwrap();
        assert_eq!(new64, legacy_reference_region_f64(&sq_diff(), &[&a64, &b64]).unwrap());
    }

    #[test]
    fn migration_planted_one_ulp_caught_exact_tolerated_at_ulp1() {
        use half::{bf16, f16};
        let av = [1.5f32, -3.0, 0.5, 7.0];
        let bv = [0.5f32, 2.0, -1.25, 0.25];

        // f32: bump the LAST element by one ULP so `first_mismatch`'s index is a
        // real observation, not index 0 by construction.
        let reference = reference_region_f32(&sq_diff(), &[&av, &bv]).unwrap();
        let mut cand = reference.clone();
        let last = cand.len() - 1;
        cand[last] = f32::from_bits(reference[last].to_bits() + 1);
        let strict = diff_region_f32(&sq_diff(), &cand, &[&av, &bv], Tolerance::Exact).unwrap();
        assert!(!strict.conforms(), "a 1-ULP error must fail Tolerance::Exact");
        assert_eq!(strict.mismatches, 1);
        assert_eq!(strict.max_ulp, 1);
        assert_eq!(strict.first_mismatch.map(|(i, _, _)| i), Some(last));
        let loose = diff_region_f32(&sq_diff(), &cand, &[&av, &bv], Tolerance::Ulp(1)).unwrap();
        assert!(loose.conforms(), "a 1-ULP error must be tolerated at Ulp(1)");
        assert_eq!(loose.max_ulp, 1); // raw distance still recorded
        // Field-for-field against the pre-migration loop.
        assert_eq!(
            strict,
            legacy_diff_region_f32(&sq_diff(), &cand, &[&av, &bv], Tolerance::Exact).unwrap()
        );
        assert_eq!(
            loose,
            legacy_diff_region_f32(&sq_diff(), &cand, &[&av, &bv], Tolerance::Ulp(1)).unwrap()
        );

        // f64 lane (the one whose kiss seam is the un-suffixed `diff_expr`).
        let a64: Vec<f64> = av.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = bv.iter().map(|&v| v as f64).collect();
        let r64 = reference_region_f64(&sq_diff(), &[&a64, &b64]).unwrap();
        let mut c64 = r64.clone();
        c64[last] = f64::from_bits(r64[last].to_bits() + 1);
        let s64 = diff_region_f64(&sq_diff(), &c64, &[&a64, &b64], Tolerance::Exact).unwrap();
        assert!(!s64.conforms());
        assert_eq!(s64.max_ulp, 1);
        assert!(diff_region_f64(&sq_diff(), &c64, &[&a64, &b64], Tolerance::Ulp(1))
            .unwrap()
            .conforms());
        assert_eq!(
            s64,
            legacy_diff_region_f64(&sq_diff(), &c64, &[&a64, &b64], Tolerance::Exact).unwrap()
        );

        // f16 lane.
        let a16: Vec<f16> = av.iter().map(|&v| f16::from_f32(v)).collect();
        let b16: Vec<f16> = bv.iter().map(|&v| f16::from_f32(v)).collect();
        let r16 = reference_region_f16(&sq_diff(), &[&a16, &b16]).unwrap();
        let c16: Vec<f16> = r16.iter().map(|&x| f16::from_bits(x.to_bits() + 1)).collect();
        let s16 = diff_region_f16(&sq_diff(), &c16, &[&a16, &b16], Tolerance::Exact).unwrap();
        assert!(!s16.conforms());
        assert_eq!(s16.max_ulp, 1);
        assert!(diff_region_f16(&sq_diff(), &c16, &[&a16, &b16], Tolerance::Ulp(1))
            .unwrap()
            .conforms());
        assert_eq!(
            s16,
            legacy_diff_region_f16(&sq_diff(), &c16, &[&a16, &b16], Tolerance::Exact).unwrap()
        );

        // bf16 lane.
        let abf: Vec<bf16> = av.iter().map(|&v| bf16::from_f32(v)).collect();
        let bbf: Vec<bf16> = bv.iter().map(|&v| bf16::from_f32(v)).collect();
        let rb = reference_region_bf16(&sq_diff(), &[&abf, &bbf]).unwrap();
        let cb: Vec<bf16> = rb.iter().map(|&x| bf16::from_bits(x.to_bits() + 1)).collect();
        let sb = diff_region_bf16(&sq_diff(), &cb, &[&abf, &bbf], Tolerance::Exact).unwrap();
        assert!(!sb.conforms());
        assert_eq!(sb.max_ulp, 1);
        assert!(diff_region_bf16(&sq_diff(), &cb, &[&abf, &bbf], Tolerance::Ulp(1))
            .unwrap()
            .conforms());
        assert_eq!(
            sb,
            legacy_diff_region_bf16(&sq_diff(), &cb, &[&abf, &bbf], Tolerance::Exact).unwrap()
        );
    }

    #[test]
    fn migration_kiss_errors_stay_typed_declines() {
        // `MissingInput` is what kiss's engine raises when an `Expr` reads an
        // `input(i)` with no column supplied. Fuel's own arity guard fires FIRST
        // on the region path (a typed decline that the migration must not lose)…
        let a = [1.5f32, -3.0];
        assert!(matches!(
            reference_region_f32(&sq_diff(), &[&a]),
            Err(KissRefError::Arity { op: OpTag::Sub, expected: 2, got: 1 })
        ));
        // …and if kiss's seam does raise it, the adapter surfaces it as a typed
        // `Eval` decline — never a panic.
        let expr = region_to_expr(&sq_diff()).unwrap();
        let short: Vec<&[f32]> = vec![&a[..1]]; // one row, only column 0
        let raised = kiss_ref_core::reference_expr_f32(&expr, &short).unwrap_err();
        assert_eq!(raised, kiss_ref_core::Error::MissingInput(1));
        assert_eq!(
            map_kiss_error(raised),
            KissRefError::Eval(kiss_ref_core::Error::MissingInput(1))
        );
        // The one kiss error the adapter does NOT wrap: `LengthMismatch` is
        // Fuel's own typed decline (`diff_region_*` promised it before the
        // migration and must keep promising it after).
        assert_eq!(
            map_kiss_error(kiss_ref_core::Error::LengthMismatch { expected: 2, got: 1 }),
            KissRefError::LengthMismatch { expected: 2, got: 1 }
        );
        let cand = [0.0f32];
        let b = [0.5f32, 2.0];
        assert!(matches!(
            diff_region_f32(&sq_diff(), &cand, &[&a, &b], Tolerance::Exact),
            Err(KissRefError::LengthMismatch { expected: 2, got: 1 })
        ));
    }
}
