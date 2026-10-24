// SPDX-License-Identifier: MIT OR Apache-2.0
//! The elementwise half of the accuracy-claim seeders (ROADMAP row 39): the unary and binary
//! f32 ops, each driven through the REGISTERED kernel and compared per element with an `f64`
//! host formula of the same `f32` inputs. Backend-free like [`super::accuracy_common`], whose
//! claim, case and record types it reuses, so a backend's seeder passes its own [`Runner`].
//!
//! Each op declares the input domain it is measured over (the formula is only meaningful
//! there: `log` of a negative is NaN) and a handful of EDGE values prepended to the random
//! inputs, so a tie in `round`, a zero in `step`/`sign` and a negative zero are exercised
//! instead of left to chance.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;

use super::accuracy_common::{Claim, OpAccuracy, Runner, case, f32_tensor, scaled};
use super::bit_stability::VerifyError;
use crate::kernel::{BindingEntry, OpParams};

/// `(rows, cols)` of the measured tensors: a scalar-sized one, a length that is not a multiple
/// of any vector width, a hidden-width row, a prefill-sized block and a wide row.
pub(crate) const ELEMENTWISE_SHAPES: &[(usize, usize)] =
    &[(1, 1), (1, 1000), (1, 3072), (448, 1024), (4, 8192)];

/// One unary op's measurement recipe.
pub(crate) struct UnarySpec {
    pub op: OpKind,
    /// Maps a uniform `u` in `[-1, 1)` to an input in the op's measured domain.
    pub domain: fn(f32) -> f32,
    /// Inputs written over the front of every tensor with room for them.
    pub edges: &'static [f32],
    pub reference: fn(f64) -> f64,
    pub reference_label: &'static str,
    pub input_range: &'static str,
}

/// One binary op's measurement recipe. `domain_b` is separate because the divisor of `div` and
/// `rem` and the exponent of `pow` have their own domains.
pub(crate) struct BinarySpec {
    pub op: OpKind,
    pub domain_a: fn(f32) -> f32,
    pub domain_b: fn(f32) -> f32,
    /// `(a, b)` pairs written over the front of every tensor with room for them.
    pub edges: &'static [(f32, f32)],
    pub reference: fn(f64, f64) -> f64,
    pub reference_label: &'static str,
    pub input_range: &'static str,
}

/// `erf` in `f64` by the non-alternating series
/// `2/sqrt(pi) * exp(-x^2) * sum 2^n x^(2n+1) / (1 * 3 * ... * (2n+1))`, which has no
/// cancellation, so it is accurate to about 1e-15 on the measured domain `|x| <= 6`.
pub(crate) fn erf_f64(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..200 {
        term *= 2.0 * x2 / (2 * n + 1) as f64;
        sum += term;
        if term.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    2.0 / std::f64::consts::PI.sqrt() * (-x2).exp() * sum
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Round half to even (`rint`), which is what the CUDA `RoundElementwise` kernel does.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Python-style modulo (the result takes the divisor's sign), which is what `RemElementwise`
/// computes.
fn rem_python(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

fn wide(u: f32) -> f32 {
    u * 8.0
}
fn unit(u: f32) -> f32 {
    u
}
/// `[0.01, 100)`: strictly positive, spanning four decades.
fn positive(u: f32) -> f32 {
    0.01 + (u + 1.0) * 0.5 * 99.99
}
/// `|x|` in `[0.25, 8)` with both signs.
fn away_from_zero(u: f32) -> f32 {
    let m = 0.25 + u.abs() * 7.75;
    if u < 0.0 { -m } else { m }
}
fn trig(u: f32) -> f32 {
    u * 20.0
}
/// `[-10, 5)`: the absolute metric would measure `exp` near `e^10 = 22026` in thousands of
/// ulp-of-1, so the domain stops where the largest result is 148.
fn exp_domain(u: f32) -> f32 {
    -10.0 + (u + 1.0) * 7.5
}
fn tanh_domain(u: f32) -> f32 {
    u * 5.0
}
fn activation(u: f32) -> f32 {
    u * 6.0
}
fn power_exponent(u: f32) -> f32 {
    u * 3.0
}
/// `[0.5, 4)`: a base for `pow` that keeps the result real and at most `4^3 = 64`.
fn pow_base(u: f32) -> f32 {
    0.5 + (u + 1.0) * 0.5 * 3.5
}

const SIGN_EDGES: &[f32] = &[0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-30, -1e-30];
const ROUND_EDGES: &[f32] = &[
    0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999997, 0.0, -0.0, 8388608.0, 1e-30,
];

/// The unary ops that carry a claim in this slice.
pub(crate) const UNARY_SPECS: &[UnarySpec] = &[
    UnarySpec {
        op: OpKind::NegElementwise,
        domain: wide,
        edges: SIGN_EDGES,
        reference: |x| -x,
        reference_label: "f64 -x",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::AbsElementwise,
        domain: wide,
        edges: SIGN_EDGES,
        reference: f64::abs,
        reference_label: "f64 |x|",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::SqrElementwise,
        domain: wide,
        edges: SIGN_EDGES,
        reference: |x| x * x,
        reference_label: "f64 x * x",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::SqrtElementwise,
        domain: positive,
        edges: &[0.0, 1.0, 4.0],
        reference: f64::sqrt,
        reference_label: "f64 sqrt",
        input_range: "uniform [0.01, 100) plus edges",
    },
    UnarySpec {
        op: OpKind::RecipElementwise,
        domain: away_from_zero,
        edges: &[1.0, -1.0, 0.25, -0.25, 8.0],
        reference: |x| 1.0 / x,
        reference_label: "f64 1 / x",
        input_range: "|x| uniform [0.25, 8), both signs, plus edges",
    },
    UnarySpec {
        op: OpKind::ExpElementwise,
        domain: exp_domain,
        edges: &[0.0, 1.0, -1.0],
        reference: f64::exp,
        reference_label: "f64 exp",
        input_range: "uniform [-10, 5) plus edges",
    },
    UnarySpec {
        op: OpKind::LogElementwise,
        domain: positive,
        edges: &[1.0, std::f32::consts::E, 0.01],
        reference: f64::ln,
        reference_label: "f64 ln",
        input_range: "uniform [0.01, 100) plus edges",
    },
    UnarySpec {
        op: OpKind::SinElementwise,
        domain: trig,
        edges: &[0.0, 1.0, -1.0],
        reference: f64::sin,
        reference_label: "f64 sin",
        input_range: "uniform [-20, 20) plus edges",
    },
    UnarySpec {
        op: OpKind::CosElementwise,
        domain: trig,
        edges: &[0.0, 1.0, -1.0],
        reference: f64::cos,
        reference_label: "f64 cos",
        input_range: "uniform [-20, 20) plus edges",
    },
    UnarySpec {
        op: OpKind::TanhElementwise,
        domain: tanh_domain,
        edges: &[0.0, 1.0, -1.0],
        reference: f64::tanh,
        reference_label: "f64 tanh",
        input_range: "uniform [-5, 5) plus edges",
    },
    UnarySpec {
        op: OpKind::ReluElementwise,
        domain: wide,
        edges: SIGN_EDGES,
        reference: |x| if x < 0.0 { 0.0 } else { x },
        reference_label: "f64 x < 0 ? 0 : x",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::GeluTanhElementwise,
        domain: activation,
        edges: &[0.0, 1.0, -1.0],
        reference: |x| {
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        },
        reference_label: "f64 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))",
        input_range: "uniform [-6, 6) plus edges",
    },
    UnarySpec {
        op: OpKind::GeluElementwise,
        domain: activation,
        edges: &[0.0, 1.0, -1.0],
        reference: |x| 0.5 * x * (1.0 + erf_f64(x / std::f64::consts::SQRT_2)),
        reference_label: "f64 0.5 x (1 + erf(x / sqrt 2))",
        input_range: "uniform [-6, 6) plus edges",
    },
    UnarySpec {
        op: OpKind::StepElementwise,
        domain: wide,
        edges: SIGN_EDGES,
        reference: |x| if x > 0.0 { 1.0 } else { 0.0 },
        reference_label: "f64 x > 0 ? 1 : 0",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::SiluElementwise,
        domain: activation,
        edges: &[0.0, 1.0, -1.0],
        reference: |x| x * sigmoid_f64(x),
        reference_label: "f64 x / (1 + exp(-x))",
        input_range: "uniform [-6, 6) plus edges",
    },
    UnarySpec {
        op: OpKind::SigmoidElementwise,
        domain: activation,
        edges: &[0.0, 1.0, -1.0],
        reference: sigmoid_f64,
        reference_label: "f64 1 / (1 + exp(-x))",
        input_range: "uniform [-6, 6) plus edges",
    },
    UnarySpec {
        op: OpKind::RsqrtElementwise,
        domain: positive,
        edges: &[1.0, 4.0, 0.01],
        reference: |x| 1.0 / x.sqrt(),
        reference_label: "f64 1 / sqrt(x)",
        input_range: "uniform [0.01, 100) plus edges",
    },
    UnarySpec {
        op: OpKind::FloorElementwise,
        domain: wide,
        edges: ROUND_EDGES,
        reference: f64::floor,
        reference_label: "f64 floor",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::CeilElementwise,
        domain: wide,
        edges: ROUND_EDGES,
        reference: f64::ceil,
        reference_label: "f64 ceil",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::RoundElementwise,
        domain: wide,
        edges: ROUND_EDGES,
        reference: round_half_even,
        reference_label: "f64 round half to even",
        input_range: "uniform [-8, 8) plus edges",
    },
    UnarySpec {
        op: OpKind::ErfElementwise,
        domain: activation,
        edges: &[0.0, 1.0, -1.0],
        reference: erf_f64,
        reference_label: "f64 erf (non-alternating series)",
        input_range: "uniform [-6, 6) plus edges",
    },
    UnarySpec {
        op: OpKind::SignElementwise,
        domain: wide,
        edges: SIGN_EDGES,
        reference: |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        reference_label: "f64 x > 0 ? 1 : x < 0 ? -1 : 0",
        input_range: "uniform [-8, 8) plus edges",
    },
];

/// The binary ops that carry a claim in this slice.
pub(crate) const BINARY_SPECS: &[BinarySpec] = &[
    BinarySpec {
        op: OpKind::AddElementwise,
        domain_a: wide,
        domain_b: wide,
        edges: &[(0.0, 0.0), (1.0, -1.0)],
        reference: |a, b| a + b,
        reference_label: "f64 a + b",
        input_range: "a, b uniform [-8, 8) plus edges",
    },
    BinarySpec {
        op: OpKind::SubElementwise,
        domain_a: wide,
        domain_b: wide,
        edges: &[(0.0, 0.0), (1.0, 1.0)],
        reference: |a, b| a - b,
        reference_label: "f64 a - b",
        input_range: "a, b uniform [-8, 8) plus edges",
    },
    BinarySpec {
        op: OpKind::MulElementwise,
        domain_a: wide,
        domain_b: wide,
        edges: &[(0.0, 5.0), (-1.0, 3.0)],
        reference: |a, b| a * b,
        reference_label: "f64 a * b",
        input_range: "a, b uniform [-8, 8) plus edges",
    },
    BinarySpec {
        op: OpKind::DivElementwise,
        domain_a: wide,
        domain_b: away_from_zero,
        edges: &[(1.0, 1.0), (0.0, 3.0), (-1.0, 4.0)],
        reference: |a, b| a / b,
        reference_label: "f64 a / b",
        input_range: "a uniform [-8, 8), |b| uniform [0.25, 8) both signs, plus edges",
    },
    BinarySpec {
        op: OpKind::MaximumElementwise,
        domain_a: wide,
        domain_b: wide,
        edges: &[(1.0, 1.0), (0.0, -0.0), (-1.0, 2.0)],
        reference: f64::max,
        reference_label: "f64 max",
        input_range: "a, b uniform [-8, 8) plus edges",
    },
    BinarySpec {
        op: OpKind::MinimumElementwise,
        domain_a: wide,
        domain_b: wide,
        edges: &[(1.0, 1.0), (0.0, -0.0), (-1.0, 2.0)],
        reference: f64::min,
        reference_label: "f64 min",
        input_range: "a, b uniform [-8, 8) plus edges",
    },
    BinarySpec {
        op: OpKind::PowElementwise,
        domain_a: pow_base,
        domain_b: power_exponent,
        edges: &[(2.0, 3.0), (1.0, 3.0), (4.0, -2.0)],
        reference: f64::powf,
        reference_label: "f64 powf",
        input_range: "a uniform [0.5, 4), b uniform [-3, 3) plus edges",
    },
    BinarySpec {
        op: OpKind::RemElementwise,
        domain_a: wide,
        domain_b: away_from_zero,
        edges: &[(5.0, 3.0), (-5.0, 3.0), (5.0, -3.0), (-5.0, -3.0)],
        reference: rem_python,
        reference_label: "f64 Python-style modulo",
        input_range: "a uniform [-8, 8), |b| uniform [0.25, 8) both signs, plus edges",
    },
];

/// Inputs for one tensor: `len` uniform values mapped through `domain`, with `edges` over the
/// front when there is room (a one-element tensor takes only the first).
fn inputs(len: usize, seed: u64, domain: fn(f32) -> f32, edges: &[f32]) -> Vec<f32> {
    let mut v: Vec<f32> = scaled(len, seed, 1.0).into_iter().map(domain).collect();
    for (slot, &e) in v.iter_mut().zip(edges) {
        *slot = e;
    }
    v
}

/// Measure one unary op: the f32 kernel in `entry` against `spec.reference` over
/// [`ELEMENTWISE_SHAPES`].
pub(crate) fn measure_unary(
    run: &Runner,
    entry: &BindingEntry,
    spec: &UnarySpec,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32];
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in ELEMENTWISE_SHAPES.iter().enumerate() {
        let x = inputs(rows * cols, 0x6000 + idx as u64, spec.domain, spec.edges);
        let got = run(
            entry,
            OpParams::None,
            &[f32_tensor(vec![rows, cols], &x)],
            vec![rows, cols],
        )?;
        let want: Vec<f64> = x.iter().map(|&v| (spec.reference)(v as f64)).collect();
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        spec.op,
        &dt,
        claim,
        entry,
        cases,
        spec.reference_label,
        spec.input_range,
    ))
}

/// Measure one binary op: the f32 kernel in `entry` against `spec.reference` over
/// [`ELEMENTWISE_SHAPES`], both operands the same shape.
pub(crate) fn measure_binary(
    run: &Runner,
    entry: &BindingEntry,
    spec: &BinarySpec,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let dt = [DType::F32, DType::F32, DType::F32];
    let edges_a: Vec<f32> = spec.edges.iter().map(|e| e.0).collect();
    let edges_b: Vec<f32> = spec.edges.iter().map(|e| e.1).collect();
    let mut cases = Vec::new();
    for (idx, &(rows, cols)) in ELEMENTWISE_SHAPES.iter().enumerate() {
        let n = rows * cols;
        let a = inputs(n, 0x7000 + idx as u64, spec.domain_a, &edges_a);
        let b = inputs(n, 0x7800 + idx as u64, spec.domain_b, &edges_b);
        let got = run(
            entry,
            OpParams::None,
            &[
                f32_tensor(vec![rows, cols], &a),
                f32_tensor(vec![rows, cols], &b),
            ],
            vec![rows, cols],
        )?;
        let want: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (spec.reference)(x as f64, y as f64))
            .collect();
        cases.push(case(format!("rows={rows} cols={cols}"), &got, &want));
    }
    Ok(OpAccuracy::new(
        spec.op,
        &dt,
        claim,
        entry,
        cases,
        spec.reference_label,
        spec.input_range,
    ))
}

#[cfg(test)]
#[path = "accuracy_elementwise_tests.rs"]
mod tests;
