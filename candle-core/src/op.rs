//! Enums and traits representing the operations in candle's computation graph.
//!
//! This module defines the core operation types that make up the autograd graph. Each time a
//! tensor operation is performed (add, matmul, reshape, etc.), an [`Op`] variant is recorded
//! so that gradients can be computed during backpropagation. The leaf enums ([`UnaryOp`],
//! [`BinaryOp`], [`ReduceOp`], [`CmpOp`]) categorize the primitive operations, while the
//! [`BackpropOp`] wrapper manages optional graph tracking.
//!
#![allow(clippy::redundant_closure_call)]
use crate::Tensor;
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};
use num_traits::float::Float;

pub use candle_core_types::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};

/// Element-wise binary operations on two tensors of the same shape.
///
/// These operations preserve the input dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Element-wise addition.
    Add,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise division.
    Div,
    /// Element-wise maximum of two tensors.
    Maximum,
    /// Element-wise minimum of two tensors.
    Minimum,
}

/// Element-wise unary operations applied to a single tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Exponential function (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Log,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Absolute value.
    Abs,
    /// Negation (`-x`).
    Neg,
    /// Reciprocal (`1/x`).
    Recip,
    /// Square (`x^2`).
    Sqr,
    /// Square root.
    Sqrt,
    /// GELU activation using the tanh approximation.
    Gelu,
    /// GELU activation using the exact erf formulation.
    GeluErf,
    /// Gauss error function.
    Erf,
    /// Rectified linear unit (`max(0, x)`).
    Relu,
    /// SiLU (Swish) activation (`x * sigmoid(x)`).
    Silu,
    /// Hyperbolic tangent.
    Tanh,
    /// Floor rounding.
    Floor,
    /// Ceiling rounding.
    Ceil,
    /// Round to nearest integer.
    Round,
    /// Sign function (-1, 0, or 1).
    Sign,
}

impl BinaryOp {
    /// Look up a [`BinaryOp`] variant by its `BinaryOpT::NAME` string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "add" => Some(Self::Add),
            "sub" => Some(Self::Sub),
            "mul" => Some(Self::Mul),
            "div" => Some(Self::Div),
            "maximum" => Some(Self::Maximum),
            "minimum" => Some(Self::Minimum),
            _ => None,
        }
    }
}

impl UnaryOp {
    /// Look up a [`UnaryOp`] variant by its `UnaryOpT::NAME` string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "exp" => Some(Self::Exp),
            "log" => Some(Self::Log),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "abs" => Some(Self::Abs),
            "neg" => Some(Self::Neg),
            "recip" => Some(Self::Recip),
            "sqr" => Some(Self::Sqr),
            "sqrt" => Some(Self::Sqrt),
            "gelu" => Some(Self::Gelu),
            "gelu_erf" => Some(Self::GeluErf),
            "erf" => Some(Self::Erf),
            "relu" => Some(Self::Relu),
            "silu" => Some(Self::Silu),
            "tanh" => Some(Self::Tanh),
            "floor" => Some(Self::Floor),
            "ceil" => Some(Self::Ceil),
            "round" => Some(Self::Round),
            "sign" => Some(Self::Sign),
            _ => None,
        }
    }
}

/// A node in the autograd computation graph.
///
/// Each `Op` variant records the operation that produced a tensor along with references to its
/// input tensors. During backpropagation, the graph is walked in reverse topological order and
/// each `Op` is used to compute gradients for its inputs. Variants marked `#[allow(dead_code)]`
/// are tracked in the graph but their backward pass is not yet implemented.
#[derive(Clone)]
pub enum Op {
    /// Element-wise binary operation on two tensors.
    Binary(Tensor, Tensor, BinaryOp),
    /// Element-wise unary operation on one tensor.
    Unary(Tensor, UnaryOp),
    /// Element-wise comparison, producing a `u8` mask tensor.
    Cmp(Tensor, CmpOp),
    /// Reduction along specified dimensions. The `Vec<usize>` stores the reduced shape with
    /// `keepdim=true`, used during backpropagation to broadcast the gradient.
    Reduce(Tensor, ReduceOp, Vec<usize>),
    /// Matrix multiplication of two tensors.
    Matmul(Tensor, Tensor),
    /// Gather elements along an axis using an index tensor.
    Gather(Tensor, Tensor, usize),
    /// Scatter values into a tensor at positions given by an index tensor.
    Scatter(Tensor, Tensor, Tensor, usize),
    /// Scatter-add: like scatter but accumulates (adds) into the destination.
    ScatterAdd(Tensor, Tensor, Tensor, usize),
    /// Select elements along an axis using an index tensor.
    IndexSelect(Tensor, Tensor, usize),
    /// Accumulate (add) values into a tensor at positions given by an index tensor.
    IndexAdd(Tensor, Tensor, Tensor, usize),
    /// Conditional selection: `where(cond, on_true, on_false)`.
    WhereCond(Tensor, Tensor, Tensor),

    #[allow(dead_code)]
    Conv1D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
    },

    #[allow(dead_code)]
    ConvTranspose1D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    },

    #[allow(dead_code)]
    Conv2D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
    },

    #[allow(dead_code)]
    ConvTranspose2D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    },

    AvgPool2D {
        arg: Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    MaxPool2D {
        arg: Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    UpsampleNearest1D {
        arg: Tensor,
        target_size: usize,
    },
    UpsampleNearest2D {
        arg: Tensor,
        target_h: usize,
        target_w: usize,
    },
    UpsampleBilinear2D {
        arg: Tensor,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
    },

    /// Concatenation of tensors along a given axis.
    Cat(Vec<Tensor>, usize),

    /// Affine transformation: `x * mul + add`.
    #[allow(dead_code)] // add is currently unused.
    Affine {
        arg: Tensor,
        mul: f64,
        add: f64,
    },
    /// Dtype cast.
    ToDType(Tensor),
    /// Tensor copy (identity in the graph, used for gradient routing).
    Copy(Tensor),
    /// Broadcasting a tensor to a larger shape.
    Broadcast(Tensor),
    /// Slicing a contiguous range along one dimension: `(tensor, dim, start, len)`.
    Narrow(Tensor, usize, usize, usize),
    /// Writing a slice into dimension 0 at a given offset.
    SliceScatter0(Tensor, Tensor, usize),
    /// Reshaping a tensor (changes layout, not data).
    Reshape(Tensor),
    /// Moving a tensor to a different device.
    ToDevice(Tensor),
    /// Transposing two dimensions.
    Transpose(Tensor, usize, usize),
    /// Arbitrary permutation of dimensions.
    Permute(Tensor, Vec<usize>),
    /// ELU activation with the given alpha parameter.
    Elu(Tensor, f64),
    /// Element-wise power: `x^exponent`.
    Powf(Tensor, f64),
    /// A user-defined unary operation (see [`crate::CustomOp1`]).
    CustomOp1(
        Tensor,
        std::sync::Arc<dyn crate::CustomOp1 + Send + Sync>,
    ),
    /// A user-defined binary operation (see [`crate::CustomOp2`]).
    CustomOp2(
        Tensor,
        Tensor,
        std::sync::Arc<dyn crate::CustomOp2 + Send + Sync>,
    ),
    /// A user-defined ternary operation (see [`crate::CustomOp3`]).
    CustomOp3(
        Tensor,
        Tensor,
        Tensor,
        std::sync::Arc<dyn crate::CustomOp3 + Send + Sync>,
    ),
}

/// Element-wise addition operator. Implements [`BinaryOpT`].
pub struct Add;
/// Element-wise division operator. Implements [`BinaryOpT`].
pub struct Div;
/// Element-wise multiplication operator. Implements [`BinaryOpT`].
pub struct Mul;
/// Element-wise subtraction operator. Implements [`BinaryOpT`].
pub struct Sub;
/// Element-wise maximum operator. Implements [`BinaryOpT`].
pub struct Maximum;
/// Element-wise minimum operator. Implements [`BinaryOpT`].
pub struct Minimum;
/// Element-wise exponential (`e^x`) operator. Implements [`UnaryOpT`].
pub struct Exp;
/// Element-wise natural logarithm operator. Implements [`UnaryOpT`].
pub struct Log;
/// Element-wise sine operator. Implements [`UnaryOpT`].
pub struct Sin;
/// Element-wise cosine operator. Implements [`UnaryOpT`].
pub struct Cos;
/// Element-wise absolute value operator. Implements [`UnaryOpT`].
pub struct Abs;
/// Element-wise negation (`-x`) operator. Implements [`UnaryOpT`].
pub struct Neg;
/// Element-wise reciprocal (`1/x`) operator. Implements [`UnaryOpT`].
pub struct Recip;
/// Element-wise square (`x^2`) operator. Implements [`UnaryOpT`].
pub struct Sqr;
/// Element-wise square root operator. Implements [`UnaryOpT`].
pub struct Sqrt;
/// GELU activation (tanh approximation) operator. Implements [`UnaryOpT`].
pub struct Gelu;
/// GELU activation (exact erf) operator. Implements [`UnaryOpT`].
pub struct GeluErf;
/// Gauss error function operator. Implements [`UnaryOpT`].
pub struct Erf;
/// Rectified linear unit (`max(0, x)`) operator. Implements [`UnaryOpT`].
pub struct Relu;
/// SiLU (Swish) activation (`x * sigmoid(x)`) operator. Implements [`UnaryOpT`].
pub struct Silu;
/// Hyperbolic tangent operator. Implements [`UnaryOpT`].
pub struct Tanh;
/// Floor rounding operator. Implements [`UnaryOpT`].
pub struct Floor;
/// Ceiling rounding operator. Implements [`UnaryOpT`].
pub struct Ceil;
/// Round-to-nearest-integer operator. Implements [`UnaryOpT`].
pub struct Round;
/// Sign function (-1, 0, or 1) operator. Implements [`UnaryOpT`].
pub struct Sign;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr, $f32_vec: ident, $f64_vec: ident) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bf16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> f16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i16(v1: i16, v2: i16) -> i16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i32(v1: i32, v2: i32) -> i32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i64(v1: i64, v2: i64) -> i64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f8e4m3(v1: f8e4m3, v2: f8e4m3) -> f8e4m3 {
                $e(v1, v2)
            }

            #[cfg(feature = "mkl")]
            const F32_VEC: bool = true;
            #[cfg(feature = "mkl")]
            const F64_VEC: bool = true;
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
                crate::mkl::$f32_vec(xs1, xs2, ys)
            }
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
                crate::mkl::$f64_vec(xs1, xs2, ys)
            }

            #[cfg(feature = "accelerate")]
            const F32_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            const F64_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
                crate::accelerate::$f32_vec(xs1, xs2, ys)
            }
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
                crate::accelerate::$f64_vec(xs1, xs2, ys)
            }
        }
    };
}

bin_op!(Add, "add", |v1, v2| v1 + v2, vs_add, vd_add);
bin_op!(Sub, "sub", |v1, v2| v1 - v2, vs_sub, vd_sub);
bin_op!(Mul, "mul", |v1, v2| v1 * v2, vs_mul, vd_mul);
bin_op!(Div, "div", |v1, v2| v1 / v2, vs_div, vd_div);
bin_op!(
    Minimum,
    "minimum",
    |v1, v2| if v1 > v2 { v2 } else { v1 },
    vs_min,
    vd_min
);
bin_op!(
    Maximum,
    "maximum",
    |v1, v2| if v1 < v2 { v2 } else { v1 },
    vs_max,
    vd_max
);

#[allow(clippy::redundant_closure_call)]
macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline(always)]
            fn i16(_: i16) -> i16 {
                todo!("no unary function for i16")
            }
            #[inline(always)]
            fn i32(_: i32) -> i32 {
                todo!("no unary function for i32")
            }
            #[inline(always)]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }
            #[inline(always)]
            fn f8e4m3($a: f8e4m3) -> f8e4m3 {
                $e
            }
        }
    };

    ($op: ident, $name: literal, $a: ident, $e: expr, $f32_vec:ident, $f64_vec:ident) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline(always)]
            fn i16(_: i16) -> i16 {
                todo!("no unary function for i16")
            }
            #[inline(always)]
            fn i32(_: i32) -> i32 {
                todo!("no unary function for i32")
            }
            #[inline(always)]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }
            #[inline(always)]
            fn f8e4m3($a: f8e4m3) -> f8e4m3 {
                $e
            }

            #[cfg(feature = "mkl")]
            const F32_VEC: bool = true;
            #[cfg(feature = "mkl")]
            const F64_VEC: bool = true;
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f32_vec(xs: &[f32], ys: &mut [f32]) {
                crate::mkl::$f32_vec(xs, ys)
            }
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f64_vec(xs: &[f64], ys: &mut [f64]) {
                crate::mkl::$f64_vec(xs, ys)
            }

            #[cfg(feature = "accelerate")]
            const F32_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            const F64_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f32_vec(xs: &[f32], ys: &mut [f32]) {
                crate::accelerate::$f32_vec(xs, ys)
            }
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f64_vec(xs: &[f64], ys: &mut [f64]) {
                crate::accelerate::$f64_vec(xs, ys)
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp(), vs_exp, vd_exp);
unary_op!(Log, "log", v, v.ln(), vs_ln, vd_ln);
unary_op!(Sin, "sin", v, v.sin(), vs_sin, vd_sin);
unary_op!(Cos, "cos", v, v.cos(), vs_cos, vd_cos);
unary_op!(Tanh, "tanh", v, v.tanh(), vs_tanh, vd_tanh);
unary_op!(Neg, "neg", v, -v);
unary_op!(Recip, "recip", v, v.recip());
unary_op!(Sqr, "sqr", v, v * v, vs_sqr, vd_sqr);
unary_op!(Sqrt, "sqrt", v, v.sqrt(), vs_sqrt, vd_sqrt);

// Hardcode the value for sqrt(2/pi)
// https://github.com/huggingface/candle/issues/1982
#[allow(clippy::excessive_precision)]
const SQRT_TWO_OVER_PI_F32: f32 = 0.79788456080286535587989211986876373;
#[allow(clippy::excessive_precision)]
const SQRT_TWO_OVER_PI_F64: f64 = 0.79788456080286535587989211986876373;

/// Tanh based approximation of the `gelu` operation
/// GeluErf is the more precise one.
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";
    const V: Self = Gelu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f32_const(0.5)
            * v
            * (bf16::ONE
                + bf16::tanh(
                    bf16::from_f32_const(SQRT_TWO_OVER_PI_F32)
                        * v
                        * (bf16::ONE + bf16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f32_const(0.5)
            * v
            * (f16::ONE
                + f16::tanh(
                    f16::from_f32_const(SQRT_TWO_OVER_PI_F32)
                        * v
                        * (f16::ONE + f16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        0.5 * v * (1.0 + f32::tanh(SQRT_TWO_OVER_PI_F32 * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        0.5 * v * (1.0 + f64::tanh(SQRT_TWO_OVER_PI_F64 * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f32(0.5)
            * v
            * (f8e4m3::ONE
                + f8e4m3::tanh(
                    f8e4m3::from_f32(SQRT_TWO_OVER_PI_F32)
                        * v
                        * (f8e4m3::ONE + f8e4m3::from_f32(0.044715) * v * v),
                ))
    }
    const KERNEL: &'static str = "ugelu";

    #[cfg(feature = "mkl")]
    const F32_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::mkl::vs_gelu(xs, ys)
    }

    #[cfg(feature = "mkl")]
    const F64_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::mkl::vd_gelu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F32_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::accelerate::vs_gelu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F64_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::accelerate::vd_gelu(xs, ys)
    }
}

/// `erf` operation
/// <https://en.wikipedia.org/wiki/Error_function>
impl UnaryOpT for Erf {
    const NAME: &'static str = "erf";
    const KERNEL: &'static str = "uerf";
    const V: Self = Erf;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        crate::cpu::erf::erf_f32(v)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        crate::cpu::erf::erf_f64(v)
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f64(Self::f64(v.to_f64()))
    }
}

/// Silu operation
impl UnaryOpT for Silu {
    const NAME: &'static str = "silu";
    const V: Self = Silu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v / (bf16::ONE + (-v).exp())
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v / (f16::ONE + (-v).exp())
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v / (f8e4m3::ONE + (-v).exp())
    }
    const KERNEL: &'static str = "usilu";

    #[cfg(feature = "mkl")]
    const F32_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::mkl::vs_silu(xs, ys)
    }

    #[cfg(feature = "mkl")]
    const F64_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::mkl::vd_silu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F32_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::accelerate::vs_silu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F64_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::accelerate::vd_silu(xs, ys)
    }
}

impl UnaryOpT for Abs {
    const NAME: &'static str = "abs";
    const KERNEL: &'static str = "uabs";
    const V: Self = Abs;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.abs()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.abs()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.abs()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.abs()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v.abs()
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v.abs()
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v.abs()
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.abs()
    }
}

impl UnaryOpT for Ceil {
    const NAME: &'static str = "ceil";
    const KERNEL: &'static str = "uceil";
    const V: Self = Ceil;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.ceil()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.ceil()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.ceil()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.ceil()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.ceil()
    }
}

impl UnaryOpT for Floor {
    const NAME: &'static str = "floor";
    const KERNEL: &'static str = "ufloor";
    const V: Self = Floor;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.floor()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.floor()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.floor()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.floor()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.floor()
    }
}

impl UnaryOpT for Round {
    const NAME: &'static str = "round";
    const KERNEL: &'static str = "uround";
    const V: Self = Round;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.round()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.round()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.round()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.round()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.round()
    }
}

impl UnaryOpT for GeluErf {
    const NAME: &'static str = "gelu_erf";
    const KERNEL: &'static str = "ugelu_erf";
    const V: Self = GeluErf;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        (crate::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        (crate::cpu::erf::erf_f64(v * std::f64::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f32(Self::f32(v.to_f32()))
    }
}

impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";
    const KERNEL: &'static str = "urelu";
    const V: Self = Relu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.max(bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.max(f16::ZERO)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.max(0f32)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.max(0f64)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v.max(0)
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v.max(0)
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v.max(0)
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.max(f8e4m3::ZERO)
    }
}

/// `BackpropOp` is a wrapper around `Option<Op>`. The main goal is to ensure that dependencies are
/// properly checked when creating a new value
#[derive(Clone)]
pub struct BackpropOp(Option<Op>);

impl BackpropOp {
    /// Creates a no-op backpropagation node (no gradient tracking).
    ///
    /// Use this when creating tensors that should not participate in the autograd graph.
    pub fn none() -> Self {
        BackpropOp(None)
    }

    pub(crate) fn new1(arg: &Tensor, f: impl Fn(Tensor) -> Op) -> Self {
        let op = if arg.track_op() {
            Some(f(arg.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new2(arg1: &Tensor, arg2: &Tensor, f: impl Fn(Tensor, Tensor) -> Op) -> Self {
        let op = if arg1.track_op() || arg2.track_op() {
            Some(f(arg1.clone(), arg2.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new3(
        arg1: &Tensor,
        arg2: &Tensor,
        arg3: &Tensor,
        f: impl Fn(Tensor, Tensor, Tensor) -> Op,
    ) -> Self {
        let op = if arg1.track_op() || arg2.track_op() || arg3.track_op() {
            Some(f(arg1.clone(), arg2.clone(), arg3.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new<A: AsRef<Tensor>>(args: &[A], f: impl Fn(Vec<Tensor>) -> Op) -> Self {
        let op = if args.iter().any(|arg| arg.as_ref().track_op()) {
            let args: Vec<Tensor> = args.iter().map(|arg| arg.as_ref().clone()).collect();
            Some(f(args))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

impl std::ops::Deref for BackpropOp {
    type Target = Option<Op>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl UnaryOpT for Sign {
    const NAME: &'static str = "sign";
    const KERNEL: &'static str = "usign";
    const V: Self = Sign;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from((v > bf16::ZERO) as i8) - bf16::from((v < bf16::ZERO) as i8)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from((v > f16::ZERO) as i8) - f16::from((v < f16::ZERO) as i8)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        f32::from(v > 0.) - f32::from(v < 0.)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        f64::from(v > 0.) - f64::from(v < 0.)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        u8::min(1, v)
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        u32::min(1, v)
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        (v > 0) as i16 - (v < 0) as i16
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        (v > 0) as i32 - (v < 0) as i32
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        (v > 0) as i64 - (v < 0) as i64
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        if v > f8e4m3::ZERO {
            f8e4m3::ONE
        } else if v < f8e4m3::ZERO {
            -f8e4m3::ONE
        } else {
            f8e4m3::ZERO
        }
    }
}
