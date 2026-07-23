//! Empirical precision-bound verification (`V-FKC-9`, Task 4.4).
//!
//! [`verify_precision_bound`] diffs a *candidate* kernel invocation against a
//! *reference*-tagged invocation of the same probe inputs, checking the
//! declared precision [`Bound`] (`max_ulp` / `max_relative` / `max_absolute`
//! from the FKC precision block). Hardware-free: both `cand` and `refr` are
//! [`KernelInvoker`]s, so unit tests here use fake in-process invokers; the
//! real CPU-reference-vs-CUDA-candidate wiring is Task 4.5.

use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, ProbeInputs, VerifyError, VerifyOutcome};
use crate::kernel::BindingEntry;
use fuel_graph::jit::{OpTag, PatternNode};

/// ULP (units-in-the-last-place) distance between two `f32` values.
///
/// Uses an IEEE-754 **total-order** mapping (the same sign-magnitude →
/// monotonic transform `f32::total_cmp` uses) before differencing, so the
/// distance is correct across the sign/zero boundary. A naive
/// `bits_x - bits_y` on the raw sign-magnitude patterns is right only for
/// same-sign operands: it reports `2^31` ULP between `+0.0` and `-0.0` (which
/// are adjacent, distance 1) and is meaningless for any candidate/reference
/// pair that straddles zero.
///
/// Shared by [`verify_precision_bound`] here and the CUDA seed harness so the
/// two never drift.
pub(crate) fn ulp_distance(x: f32, y: f32) -> u64 {
    fn total_order_key(f: f32) -> u32 {
        let b = f.to_bits();
        // Negative: flip every bit (reverses the descending magnitude order).
        // Non-negative: set the sign bit (lifts the positives above the
        // negatives). Result is a u32 that increases monotonically with the
        // real value, with `-0.0` immediately below `+0.0`.
        if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
    }
    u64::from(total_order_key(x).abs_diff(total_order_key(y)))
}

/// A declared precision bound to check a candidate against a reference.
/// Mirrors the FKC precision block's machine-checkable claims
/// (`max_ulp` / `max_relative` / `max_absolute`).
#[derive(Debug, Clone, Copy)]
pub enum Bound {
    /// Maximum allowed ULP (units-in-last-place) distance between candidate
    /// and reference bit patterns.
    MaxUlp(u32),
    /// Maximum allowed `|cand - ref| / |ref|` (reference-relative error).
    MaxRelative(f64),
    /// Maximum allowed `|cand - ref|` (absolute error).
    MaxAbsolute(f64),
}

/// Empirically checks a precision [`Bound`] by invoking `cand` and `refr` on
/// the same probes and comparing their `f32` outputs elementwise. Returns the
/// FIRST out-of-bound element as `Fail { detail }` (mirrors
/// `verify_bit_stability`'s "report the first divergence" posture) rather
/// than accumulating every mismatch.
///
/// Never panics: a probe whose byte length isn't a multiple of 4 would panic
/// inside `bytemuck::cast_slice`, so this reinterprets defensively — any
/// non-`f32`-aligned output is reported as a `Fail` rather than allowed to
/// panic the process. Mismatched candidate/reference output lengths only
/// compare the overlapping prefix (`zip` stops at the shorter side); that is
/// a conservative pass they'd otherwise need a separate shape-check for, and
/// is out of scope for this Phase-1 numeric-bound verifier.
pub fn verify_precision_bound(
    cand: &dyn KernelInvoker,
    refr: &dyn KernelInvoker,
    entry: &BindingEntry,
    probes: &[ProbeInputs],
    bound: Bound,
) -> Result<VerifyOutcome, VerifyError> {
    for (probe_idx, probe) in probes.iter().enumerate() {
        let a: HostTensor = cand.invoke(entry, probe)?;
        let b: HostTensor = refr.invoke(entry, probe)?;

        if a.bytes.len() % 4 != 0 || b.bytes.len() % 4 != 0 {
            return Ok(VerifyOutcome::Fail {
                detail: format!(
                    "probe {probe_idx}: output byte length not a multiple of 4 (cand {}, ref {}) — cannot reinterpret as f32",
                    a.bytes.len(),
                    b.bytes.len()
                ),
            });
        }
        let af: &[f32] = bytemuck::cast_slice(&a.bytes);
        let bf: &[f32] = bytemuck::cast_slice(&b.bytes);

        for (elem_idx, (x, y)) in af.iter().zip(bf.iter()).enumerate() {
            let ok = match bound {
                Bound::MaxAbsolute(m) => (*x as f64 - *y as f64).abs() <= m,
                Bound::MaxRelative(m) => {
                    let denom = (*y as f64).abs().max(f64::from(f32::EPSILON));
                    ((*x as f64 - *y as f64).abs() / denom) <= m
                }
                Bound::MaxUlp(m) => ulp_distance(*x, *y) <= m as u64,
            };
            if !ok {
                return Ok(VerifyOutcome::Fail {
                    detail: format!(
                        "probe {probe_idx} elem {elem_idx}: candidate {x} vs reference {y} exceeds bound {bound:?}"
                    ),
                });
            }
        }
    }
    Ok(VerifyOutcome::Pass)
}

/// A transcendental unary atom — one whose hardware value can differ from the
/// wide-precision (§6.5-0007) truth by more than a correctly-rounded op. IEEE
/// requires `Sqrt`/`Recip` to be correctly-rounded, so they are NOT here;
/// `Exp`/`Log`/`Sin`/`Cos`/`Tanh`/`Sigmoid`/`Silu`/`Gelu`/`GeluErf`/`Erf`/
/// `Rsqrt` are. Mirrors `cost.rs`'s `cost_elementwise_unary_transcendental_cpu`
/// classification so the two never drift.
pub(crate) fn is_transcendental(tag: OpTag) -> bool {
    use OpTag::*;
    matches!(tag, Exp | Log | Sin | Cos | Tanh | Sigmoid | Silu | Gelu | GeluErf | Erf | Rsqrt)
}

/// Whether a recipe region contains any transcendental atom. A
/// transcendental-containing region gets the widened comparison band on the
/// live kiss-ref / CPU-oracle path (see [`widen_bound_for_transcendental`]).
/// `Bind`/`Any` are leaves (no op); `SeeThrough` recurses.
pub fn region_contains_transcendental(region: &PatternNode) -> bool {
    match region {
        PatternNode::Op { op, operands, .. } => {
            is_transcendental(*op) || operands.iter().any(region_contains_transcendental)
        }
        PatternNode::SeeThrough { then } => region_contains_transcendental(then),
        PatternNode::Bind { .. } | PatternNode::Any => false,
    }
}

/// Widen a precision [`Bound`] to 2× for a live comparison of a
/// transcendental-containing region. Two implementations each within the ULP
/// ceiling `C` of the wide-precision truth can differ from EACH OTHER by up to
/// `2C` (triangle inequality); kiss-ref and Fuel's CPU oracle are both
/// hardware-precision (neither is the wide-precision truth), so a live
/// candidate-vs-reference check on a transcendental region must allow `2C`.
/// Tight transcendental verification defers to the frozen wide-precision
/// corpus, not to this live path (KISS, 2026-07-18). `MaxUlp` saturates so a
/// huge declared ceiling can never overflow (never-panic).
pub fn widen_bound_for_transcendental(bound: Bound) -> Bound {
    match bound {
        Bound::MaxUlp(m) => Bound::MaxUlp(m.saturating_mul(2)),
        Bound::MaxRelative(m) => Bound::MaxRelative(m * 2.0),
        Bound::MaxAbsolute(m) => Bound::MaxAbsolute(m * 2.0),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        is_transcendental, region_contains_transcendental, ulp_distance,
        widen_bound_for_transcendental, Bound,
    };
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};

    #[test]
    fn widen_doubles_each_bound() {
        assert!(matches!(widen_bound_for_transcendental(Bound::MaxUlp(4)), Bound::MaxUlp(8)));
        match widen_bound_for_transcendental(Bound::MaxRelative(1e-6)) {
            Bound::MaxRelative(m) => assert!((m - 2e-6).abs() < 1e-18),
            other => panic!("expected MaxRelative, got {other:?}"),
        }
        match widen_bound_for_transcendental(Bound::MaxAbsolute(0.25)) {
            Bound::MaxAbsolute(m) => assert_eq!(m, 0.5),
            other => panic!("expected MaxAbsolute, got {other:?}"),
        }
        // Saturates rather than overflowing (never-panic).
        assert!(matches!(
            widen_bound_for_transcendental(Bound::MaxUlp(u32::MAX)),
            Bound::MaxUlp(u32::MAX)
        ));
    }

    #[test]
    fn is_transcendental_classifies_exactly() {
        for t in [
            OpTag::Exp, OpTag::Log, OpTag::Sin, OpTag::Cos, OpTag::Tanh, OpTag::Sigmoid,
            OpTag::Silu, OpTag::Gelu, OpTag::GeluErf, OpTag::Erf, OpTag::Rsqrt,
        ] {
            assert!(is_transcendental(t), "{t:?} should be transcendental");
        }
        // Sqrt/Recip are IEEE correctly-rounded — NOT band-widened.
        for t in [OpTag::Sqrt, OpTag::Recip, OpTag::Relu, OpTag::Neg, OpTag::Abs, OpTag::Sqr] {
            assert!(!is_transcendental(t), "{t:?} should NOT be transcendental");
        }
    }

    #[test]
    fn region_transcendental_detection_walks_nested() {
        // Op{Neg, [Op{Exp, [Bind0]}]} — a nested transcendental atom.
        let inner = PatternNode::Op {
            op: OpTag::Exp,
            operands: vec![PatternNode::Bind { index: 0 }],
            attrs: OpAttrs::default(),
        };
        let outer = PatternNode::Op {
            op: OpTag::Neg,
            operands: vec![inner],
            attrs: OpAttrs::default(),
        };
        assert!(region_contains_transcendental(&outer), "nested Exp must be found");

        // Op{Neg, [Op{Sqr, [Bind0]}]} — no transcendental atom.
        let inner2 = PatternNode::Op {
            op: OpTag::Sqr,
            operands: vec![PatternNode::Bind { index: 0 }],
            attrs: OpAttrs::default(),
        };
        let outer2 = PatternNode::Op {
            op: OpTag::Neg,
            operands: vec![inner2],
            attrs: OpAttrs::default(),
        };
        assert!(!region_contains_transcendental(&outer2), "no transcendental atom present");
    }

    #[test]
    fn ulp_distance_signed_zero_is_one() {
        // -0.0 and +0.0 are adjacent in IEEE-754 total order: 1 ULP apart,
        // NOT 2^31 (the raw sign-magnitude subtraction bug).
        assert_eq!(ulp_distance(-0.0, 0.0), 1);
        assert_eq!(ulp_distance(0.0, -0.0), 1);
    }

    #[test]
    fn ulp_distance_same_value_is_zero() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(-3.5, -3.5), 0);
    }

    #[test]
    fn ulp_distance_adjacent_same_sign_is_one() {
        let a = 1.0_f32;
        let b = f32::from_bits(a.to_bits() + 1); // next representable above 1.0
        assert_eq!(ulp_distance(a, b), 1);
        let c = -1.0_f32;
        let d = f32::from_bits(c.to_bits() + 1); // next-toward-zero below -1.0
        assert_eq!(ulp_distance(c, d), 1);
    }

    #[test]
    fn ulp_distance_straddling_zero_is_small() {
        // smallest +subnormal -> +0 -> -0 -> smallest -subnormal = 3 steps.
        let pos_min = f32::from_bits(1); // +2^-149
        let neg_min = f32::from_bits(0x8000_0001); // -2^-149
        assert_eq!(ulp_distance(pos_min, neg_min), 3);
    }
}
