//! Enums and traits representing tensor operations.
//!
//! This module defines the core operation enums and traits needed by backend implementations.
//! The `Op` enum (computation graph) and concrete operation types (`Add`, `Mul`, etc.) remain
//! in `candle-core`.
#![allow(clippy::redundant_closure_call)]
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};

/// Element-wise comparison operations.
///
/// These produce a `u8` tensor where each element is `0` (false) or `1` (true).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal to (`==`).
    Eq,
    /// Not equal to (`!=`).
    Ne,
    /// Less than or equal to (`<=`).
    Le,
    /// Greater than or equal to (`>=`).
    Ge,
    /// Strictly less than (`<`).
    Lt,
    /// Strictly greater than (`>`).
    Gt,
}

/// Reduction operations that collapse one or more dimensions of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum of elements along the reduced dimensions.
    Sum,
    /// Minimum value along the reduced dimension.
    Min,
    /// Maximum value along the reduced dimension.
    Max,
    /// Index of the minimum value along the reduced dimension (returns a `u32` tensor).
    ArgMin,
    /// Index of the maximum value along the reduced dimension (returns a `u32` tensor).
    ArgMax,
}

impl ReduceOp {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ArgMax => "argmax",
            Self::ArgMin => "argmin",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }
}

/// Trait for implementing an element-wise unary operation across all supported dtypes.
pub trait UnaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16) -> bf16;
    fn f16(v1: f16) -> f16;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u8(v1: u8) -> u8;
    fn u32(v1: u32) -> u32;
    fn i16(v1: i16) -> i16;
    fn i32(v1: i32) -> i32;
    fn i64(v1: i64) -> i64;
    fn f8e4m3(v1: f8e4m3) -> f8e4m3;

    const BF16_VEC: bool = false;
    fn bf16_vec(_xs: &[bf16], _ys: &mut [bf16]) {}
    const F16_VEC: bool = false;
    fn f16_vec(_xs: &[f16], _ys: &mut [f16]) {}
    const F32_VEC: bool = false;
    fn f32_vec(_xs: &[f32], _ys: &mut [f32]) {}
    const F64_VEC: bool = false;
    fn f64_vec(_xs: &[f64], _ys: &mut [f64]) {}
}

/// Trait for implementing an element-wise binary operation across all supported dtypes.
pub trait BinaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16, v2: bf16) -> bf16;
    fn f16(v1: f16, v2: f16) -> f16;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
    fn u32(v1: u32, v2: u32) -> u32;
    fn i16(v1: i16, v2: i16) -> i16;
    fn i32(v1: i32, v2: i32) -> i32;
    fn i64(v1: i64, v2: i64) -> i64;
    fn f8e4m3(v1: f8e4m3, v2: f8e4m3) -> f8e4m3;

    const BF16_VEC: bool = false;
    fn bf16_vec(_xs1: &[bf16], _xs2: &[bf16], _ys: &mut [bf16]) {}
    const F16_VEC: bool = false;
    fn f16_vec(_xs1: &[f16], _xs2: &[f16], _ys: &mut [f16]) {}
    const F32_VEC: bool = false;
    fn f32_vec(_xs1: &[f32], _xs2: &[f32], _ys: &mut [f32]) {}
    const F64_VEC: bool = false;
    fn f64_vec(_xs1: &[f64], _xs2: &[f64], _ys: &mut [f64]) {}
    const U8_VEC: bool = false;
    fn u8_vec(_xs1: &[u8], _xs2: &[u8], _ys: &mut [u8]) {}
    const U32_VEC: bool = false;
    fn u32_vec(_xs1: &[u32], _xs2: &[u32], _ys: &mut [u32]) {}
    const I64_VEC: bool = false;
    fn i64_vec(_xs1: &[i64], _xs2: &[i64], _ys: &mut [i64]) {}
}
