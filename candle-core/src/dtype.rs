//! Types for elements that can be stored and manipulated using tensors.
//!
//! The [`DType`] enum and its inherent methods, Display, FromStr, DTypeParseError,
//! and safetensors interop are all defined in `candle-core-types` and re-exported here.
//!
//! [`WithDType`] is a local marker subtrait of `candle_core_types::dtype::WithDType`.
//! Keeping it local lets Rust's coherence checker prove disjointness of generic impls
//! (e.g. `NdArray for S` vs `NdArray for Vec<S>`).  All methods are inherited from
//! the upstream trait.
#![allow(clippy::redundant_closure_call)]

// Re-export DType and related types from candle-core-types.
pub use candle_core_types::dtype::{DType, DTypeParseError};

/// Local marker subtrait — inherits every method from
/// `candle_core_types::dtype::WithDType` but is defined in this crate so that
/// the orphan/coherence rules treat it as a local trait.
pub trait WithDType: candle_core_types::dtype::WithDType {}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {}
    };
}
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};

with_dtype!(u8, U8, |v: f64| v as u8, |v: u8| v as f64);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
with_dtype!(i16, I16, |v: f64| v as i16, |v: i16| v as f64);
with_dtype!(i32, I32, |v: f64| v as i32, |v: i32| v as f64);
with_dtype!(i64, I64, |v: f64| v as i64, |v: i64| v as f64);
with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);
with_dtype!(f8e4m3, F8E4M3, f8e4m3::from_f64, |v: f8e4m3| v.to_f64());

pub trait IntDType: WithDType + num_traits::Bounded {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

impl IntDType for i64 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u8 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for i16 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for i32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}
impl FloatDType for f8e4m3 {}
