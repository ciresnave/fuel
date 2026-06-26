//! Unary-elementwise chassis — one shape/loop pass shared by every
//! per-(op, dtype) unary kernel (Relu / Neg / Sqr / … / GeluTanh,
//! across f32 / f64 / bf16 / f16).
//!
//! ## Design
//!
//! The chassis splits into three layers so per-op authors write the
//! math once and get four dtype impls for free:
//!
//! 1. [`UnaryOp<T>`] — what the chassis function consumes. One
//!    `apply(T) -> T` method.
//! 2. [`UnaryOpCore`] — what op authors implement. Two methods
//!    (`f32` + `f64`) carrying the per-precision math. Op markers
//!    are zero-sized structs implementing this trait.
//! 3. Four blanket impls — every `O: UnaryOpCore` automatically
//!    gets `UnaryOp<f32>`, `UnaryOp<f64>`, `UnaryOp<bf16>` (via
//!    f32 round-trip), and `UnaryOp<f16>` (via f32 round-trip).
//!
//! The half-float round-trip preserves the pre-refactor behavior
//! bit-for-bit: every existing `unary_kernel!(*_bf16, ..., |x|
//! half::bf16::from_f32(<f32 op>(x.to_f32())))` invocation becomes
//! a single `<UnaryOpCore>::f32` definition that flows through the
//! blanket bf16 impl.
//!
//! Adding a new op is "implement two methods on a new marker."
//! Adding a new dtype to the family is "add one more blanket impl
//! in this file." Adding a new dtype to a single op is impossible
//! by design — the blanket impls cover every op uniformly.

use bytemuck::Pod;

use crate::byte_storage::CpuStorageBytes;
use fuel_ir::{Error, Result};

// =============================================================================
// Traits
// =============================================================================

/// Per-(op, dtype) unary operation. The chassis function
/// [`unary`] consumes one of these implementations to walk a
/// byte-shaped tensor elementwise.
///
/// Implementations are auto-derived from [`UnaryOpCore`] via four
/// blanket impls — don't implement this directly.
pub trait UnaryOp<T: Copy> {
    fn apply(x: T) -> T;
}

/// What op authors actually implement. Two methods carry the f32
/// and f64 math respectively; the blanket [`UnaryOp`] impls in this
/// module derive the four dtype-specific implementations (f32 / f64
/// direct, bf16 / f16 via f32 round-trip).
///
/// Splitting `f32` and `f64` lets ops that want extra f64 precision
/// (e.g. higher-precision constants) opt in without forcing every
/// op to be generic over a numeric trait we'd have to define.
pub trait UnaryOpCore {
    fn f32(x: f32) -> f32;
    fn f64(x: f64) -> f64;
}

// Blanket impls.

impl<O: UnaryOpCore> UnaryOp<f32> for O {
    fn apply(x: f32) -> f32 { <O as UnaryOpCore>::f32(x) }
}

impl<O: UnaryOpCore> UnaryOp<f64> for O {
    fn apply(x: f64) -> f64 { <O as UnaryOpCore>::f64(x) }
}

impl<O: UnaryOpCore> UnaryOp<half::bf16> for O {
    fn apply(x: half::bf16) -> half::bf16 {
        half::bf16::from_f32(<O as UnaryOpCore>::f32(x.to_f32()))
    }
}

impl<O: UnaryOpCore> UnaryOp<half::f16> for O {
    fn apply(x: half::f16) -> half::f16 {
        half::f16::from_f32(<O as UnaryOpCore>::f32(x.to_f32()))
    }
}

// =============================================================================
// Chassis function
// =============================================================================

/// Elementwise `out[i] = U::apply(input[i])`. Validates byte
/// lengths match (input and output must hold the same element
/// count of type `T`), then walks the typed views.
///
/// `name` appears in size-mismatch error messages so the
/// diagnostic points at the entry the caller invoked.
pub fn unary<T, U>(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
) -> Result<()>
where
    T: Copy + Pod,
    U: UnaryOp<T>,
{
    let in_bytes = input.len_bytes();
    let out_bytes = output.len_bytes();
    if in_bytes != out_bytes {
        return Err(Error::Msg(format!(
            "{name}: input bytes={in_bytes} != output bytes={out_bytes}",
        ))
        .bt());
    }
    let in_view: &[T] = input.as_slice()?;
    let out_view: &mut [T] = output.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = U::apply(in_view[i]);
    }
    Ok(())
}

// =============================================================================
// Op markers
// =============================================================================
//
// Each op is a zero-sized struct implementing `UnaryOpCore`. The
// four `UnaryOp<T>` impls fall out of the blanket impls above.
// Pre-refactor `unary_kernel!(*_bf16, ..., |x| half::bf16::from_f32(f32_op(x.to_f32())))`
// invocations all collapse onto the bf16 blanket impl.

/// ReLU: `max(0, x)`.
pub struct Relu;
impl UnaryOpCore for Relu {
    fn f32(x: f32) -> f32 { x.max(0.0) }
    fn f64(x: f64) -> f64 { x.max(0.0) }
}

/// Negation: `-x`.
pub struct Neg;
impl UnaryOpCore for Neg {
    fn f32(x: f32) -> f32 { -x }
    fn f64(x: f64) -> f64 { -x }
}

/// Square: `x * x`.
pub struct Sqr;
impl UnaryOpCore for Sqr {
    fn f32(x: f32) -> f32 { x * x }
    fn f64(x: f64) -> f64 { x * x }
}

/// Square root. Negative inputs yield NaN per IEEE-754.
pub struct Sqrt;
impl UnaryOpCore for Sqrt {
    fn f32(x: f32) -> f32 { x.sqrt() }
    fn f64(x: f64) -> f64 { x.sqrt() }
}

/// Reciprocal: `1 / x`. Zero input yields IEEE-754 inf/NaN.
pub struct Recip;
impl UnaryOpCore for Recip {
    fn f32(x: f32) -> f32 { 1.0 / x }
    fn f64(x: f64) -> f64 { 1.0 / x }
}

/// Absolute value: `|x|`.
pub struct Abs;
impl UnaryOpCore for Abs {
    fn f32(x: f32) -> f32 { x.abs() }
    fn f64(x: f64) -> f64 { x.abs() }
}

/// Hyperbolic tangent.
pub struct Tanh;
impl UnaryOpCore for Tanh {
    fn f32(x: f32) -> f32 { x.tanh() }
    fn f64(x: f64) -> f64 { x.tanh() }
}

/// Exponential: `e^x`.
pub struct Exp;
impl UnaryOpCore for Exp {
    fn f32(x: f32) -> f32 { x.exp() }
    fn f64(x: f64) -> f64 { x.exp() }
}

/// Natural log. Negative inputs yield NaN per IEEE-754.
pub struct Log;
impl UnaryOpCore for Log {
    fn f32(x: f32) -> f32 { x.ln() }
    fn f64(x: f64) -> f64 { x.ln() }
}

/// Sine.
pub struct Sin;
impl UnaryOpCore for Sin {
    fn f32(x: f32) -> f32 { x.sin() }
    fn f64(x: f64) -> f64 { x.sin() }
}

/// Cosine.
pub struct Cos;
impl UnaryOpCore for Cos {
    fn f32(x: f32) -> f32 { x.cos() }
    fn f64(x: f64) -> f64 { x.cos() }
}

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
pub struct Sigmoid;
impl UnaryOpCore for Sigmoid {
    fn f32(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    fn f64(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
}

/// SiLU / Swish: `x * sigmoid(x)`.
pub struct Silu;
impl UnaryOpCore for Silu {
    fn f32(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
    fn f64(x: f64) -> f64 { x / (1.0 + (-x).exp()) }
}

/// Heaviside step: `1 where x > 0, else 0`.
pub struct Step;
impl UnaryOpCore for Step {
    fn f32(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }
    fn f64(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }
}

/// Floor: `⌊x⌋`.
pub struct Floor;
impl UnaryOpCore for Floor {
    fn f32(x: f32) -> f32 { x.floor() }
    fn f64(x: f64) -> f64 { x.floor() }
}

/// Ceiling: `⌈x⌉`.
pub struct Ceil;
impl UnaryOpCore for Ceil {
    fn f32(x: f32) -> f32 { x.ceil() }
    fn f64(x: f64) -> f64 { x.ceil() }
}

/// Round-half-to-even (banker's rounding, IEEE-754 roundeven).
pub struct Round;
impl UnaryOpCore for Round {
    fn f32(x: f32) -> f32 { x.round_ties_even() }
    fn f64(x: f64) -> f64 { x.round_ties_even() }
}

/// Sign: `-1 / 0 / 1`. `sign(0) = 0` by subgradient convention.
pub struct Sign;
impl UnaryOpCore for Sign {
    fn f32(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
    }
    fn f64(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
    }
}

/// Gauss error function (`erf` via libm).
pub struct Erf;
impl UnaryOpCore for Erf {
    fn f32(x: f32) -> f32 { libm::erff(x) }
    fn f64(x: f64) -> f64 { libm::erf(x) }
}

/// GELU (exact erf form): `0.5 * x * (1 + erf(x/√2))`.
pub struct GeluErf;
impl UnaryOpCore for GeluErf {
    fn f32(x: f32) -> f32 {
        0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
    }
    fn f64(x: f64) -> f64 {
        0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2))
    }
}

/// Reciprocal square root: `1 / sqrt(x)`.
pub struct Rsqrt;
impl UnaryOpCore for Rsqrt {
    fn f32(x: f32) -> f32 { 1.0 / x.sqrt() }
    fn f64(x: f64) -> f64 { 1.0 / x.sqrt() }
}

/// GELU (tanh approximation): `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
/// Matches `Op::Gelu`'s tanh-approximation semantics in fuel-graph.
/// f32 uses a 7-digit constant for √(2/π); f64 uses a 16-digit
/// constant — both match the pre-chassis `gelu_*` functions
/// bit-for-bit. bf16 / f16 route through the f32 path via the
/// blanket impl.
pub struct GeluTanh;
impl UnaryOpCore for GeluTanh {
    fn f32(x: f32) -> f32 {
        const COEFF: f32 = 0.797_884_56;
        let inner = COEFF * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }
    fn f64(x: f64) -> f64 {
        const COEFF: f64 = 0.797_884_560_802_865_4;
        let inner = COEFF * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }
}

// =============================================================================
// Structural tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_op_neg_f32_negates() {
        assert_eq!(<Neg as UnaryOp<f32>>::apply(2.5), -2.5);
        assert_eq!(<Neg as UnaryOp<f32>>::apply(-3.0), 3.0);
        assert_eq!(<Neg as UnaryOp<f32>>::apply(0.0), 0.0);
    }

    #[test]
    fn unary_op_relu_f32_clamps_negative_to_zero() {
        assert_eq!(<Relu as UnaryOp<f32>>::apply(2.5), 2.5);
        assert_eq!(<Relu as UnaryOp<f32>>::apply(-3.0), 0.0);
        assert_eq!(<Relu as UnaryOp<f32>>::apply(0.0), 0.0);
    }

    #[test]
    fn unary_op_bf16_blanket_routes_through_f32() {
        // Sqr: a value too large to represent precisely in bf16
        // after squaring should round-trip through f32 first. The
        // pre-chassis kernel did the same; bytes must match.
        let x = half::bf16::from_f32(1.5);
        let got = <Sqr as UnaryOp<half::bf16>>::apply(x).to_f32();
        let expect = half::bf16::from_f32(1.5 * 1.5).to_f32();
        assert_eq!(got, expect);
    }

    #[test]
    fn unary_op_gelu_tanh_f32_at_zero_is_zero() {
        assert!(<GeluTanh as UnaryOp<f32>>::apply(0.0).abs() < 1e-6);
    }

    #[test]
    fn unary_op_gelu_tanh_f32_at_one() {
        // gelu(1) ≈ 0.8412 (tanh approx) per the existing
        // `gelu_at_known_points` test.
        let got = <GeluTanh as UnaryOp<f32>>::apply(1.0);
        assert!((got - 0.841_192).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn unary_op_sign_f32_handles_zero_as_zero() {
        assert_eq!(<Sign as UnaryOp<f32>>::apply(0.0), 0.0);
        assert_eq!(<Sign as UnaryOp<f32>>::apply(5.0), 1.0);
        assert_eq!(<Sign as UnaryOp<f32>>::apply(-5.0), -1.0);
    }

    #[test]
    fn unary_chassis_size_mismatch_errors() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let mut output = CpuStorageBytes::from_zero_bytes(4); // 1 f32, not 2
        let r = unary::<f32, Neg>("test", &input, &mut output);
        assert!(r.is_err());
    }

    #[test]
    fn unary_chassis_walks_all_elements() {
        let input = CpuStorageBytes::from_slice(&[1.0_f32, -2.0, 3.0, -4.0]);
        let mut output = CpuStorageBytes::from_zero_bytes(input.len_bytes());
        unary::<f32, Neg>("test", &input, &mut output).expect("unary neg_f32");
        let r: &[f32] = output.as_slice().unwrap();
        assert_eq!(r, &[-1.0, 2.0, -3.0, 4.0]);
    }
}
