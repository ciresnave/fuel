//! Types for elements that can be stored and manipulated using tensors.
//!
//! The [`DType`] enum represents the element type of a tensor, and the [`WithDType`] trait
//! allows Rust native types to be used with tensors.
#![allow(clippy::redundant_closure_call)]
use crate::{CpuStorage, CpuStorageRef, Error, Result};

/// The different types of elements allowed in tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Brain floating-point using half precision (16 bits).
    BF16,
    /// Floating-point using half precision (16 bits).
    F16,
    /// Floating-point using single precision (32 bits).
    F32,
    /// Floating-point using double precision (64 bits).
    F64,
    /// 8-bit floating point with 4-bit exponent and 3-bit mantissa.
    F8E4M3,
    /// 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format).
    F6E2M3,
    /// 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format).
    F6E3M2,
    /// 4-bit float (MX4 format).
    F4,
    /// 8-bit float with 8 exponent bits and 0 mantissa bits.
    F8E8M0,
}

/// Error returned when a string cannot be parsed as a [`DType`].
#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError(String);

impl std::fmt::Display for DTypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot parse '{}' as a dtype", self.0)
    }
}

impl std::error::Error for DTypeParseError {}

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "f8e4m3" => Ok(Self::F8E4M3),
            "f6e2m3" => Ok(Self::F6E2M3),
            "f6e3m2" => Ok(Self::F6E3M2),
            "f4" => Ok(Self::F4),
            "f8e8m0" => Ok(Self::F8E8M0),
            _ => Err(DTypeParseError(s.to_string())),
        }
    }
}

impl DType {
    /// Returns the string representation of this dtype.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F8E4M3 => "f8e4m3",
            Self::F6E2M3 => "f6e2m3",
            Self::F6E3M2 => "f6e3m2",
            Self::F4 => "f4",
            Self::F8E8M0 => "f8e8m0",
        }
    }

    /// Returns the size used by each element in bytes.
    ///
    /// Returns 0 for sub-byte types (F6E2M3, F6E3M2, F4).
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::F8E4M3 => 1,
            Self::F6E2M3 => 0,
            Self::F6E3M2 => 0,
            Self::F4 => 0,
            Self::F8E8M0 => 1,
        }
    }

    /// Returns `true` if this is an integer type (U8, U32, I16, I32, I64).
    pub fn is_int(&self) -> bool {
        match self {
            Self::U8 | Self::U32 | Self::I16 | Self::I32 | Self::I64 => true,
            Self::BF16
            | Self::F16
            | Self::F32
            | Self::F64
            | Self::F8E4M3
            | Self::F6E2M3
            | Self::F6E3M2
            | Self::F4
            | Self::F8E8M0 => false,
        }
    }

    /// Returns `true` if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        match self {
            Self::U8 | Self::U32 | Self::I16 | Self::I32 | Self::I64 => false,
            Self::BF16
            | Self::F16
            | Self::F32
            | Self::F64
            | Self::F8E4M3
            | Self::F6E2M3
            | Self::F6E3M2
            | Self::F4
            | Self::F8E8M0 => true,
        }
    }
}

/// Trait for Rust types that can be stored as tensor elements.
///
/// This maps Rust native types (e.g., `f32`, `u8`) to their corresponding [`DType`].
/// It provides conversion routines between Rust values and CPU tensor storage.
pub trait WithDType:
    Sized
    + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd
    + std::fmt::Display
    + 'static
    + Send
    + Sync
    + std::any::Any
    + crate::cpu::kernels::VecOps
{
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn to_scalar(self) -> crate::scalar::Scalar;
    fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_>;
    fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
    fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

            fn to_scalar(self) -> crate::scalar::Scalar {
                crate::scalar::Scalar::$dtype(self)
            }

            fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_> {
                CpuStorageRef::$dtype(data)
            }

            fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage {
                CpuStorage::$dtype(data)
            }

            fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }
        }
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

/// Trait for integer element types that can be used with tensor indexing and masking.
///
/// Implemented for `u8`, `u32`, `i16`, `i32`, and `i64`.
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

/// Trait for floating-point element types.
pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}
impl FloatDType for f8e4m3 {}

// Safetensors interop: DType <-> safetensors::Dtype conversions
use safetensors::tensor as st;

impl From<DType> for st::Dtype {
    fn from(value: DType) -> Self {
        match value {
            DType::U8 => st::Dtype::U8,
            DType::U32 => st::Dtype::U32,
            DType::I16 => st::Dtype::I16,
            DType::I32 => st::Dtype::I32,
            DType::I64 => st::Dtype::I64,
            DType::BF16 => st::Dtype::BF16,
            DType::F16 => st::Dtype::F16,
            DType::F32 => st::Dtype::F32,
            DType::F64 => st::Dtype::F64,
            DType::F8E4M3 => st::Dtype::F8_E4M3,
            DType::F6E2M3 => st::Dtype::F6_E2M3,
            DType::F6E3M2 => st::Dtype::F6_E3M2,
            DType::F4 => st::Dtype::F4,
            DType::F8E8M0 => st::Dtype::F8_E8M0,
        }
    }
}

impl TryFrom<st::Dtype> for DType {
    type Error = Error;
    fn try_from(value: st::Dtype) -> Result<Self> {
        match value {
            st::Dtype::U8 => Ok(DType::U8),
            st::Dtype::U32 => Ok(DType::U32),
            st::Dtype::I16 => Ok(DType::I16),
            st::Dtype::I32 => Ok(DType::I32),
            st::Dtype::I64 => Ok(DType::I64),
            st::Dtype::BF16 => Ok(DType::BF16),
            st::Dtype::F16 => Ok(DType::F16),
            st::Dtype::F32 => Ok(DType::F32),
            st::Dtype::F64 => Ok(DType::F64),
            st::Dtype::F8_E4M3 => Ok(DType::F8E4M3),
            st::Dtype::F6_E2M3 => Ok(DType::F6E2M3),
            st::Dtype::F6_E3M2 => Ok(DType::F6E3M2),
            st::Dtype::F4 => Ok(DType::F4),
            st::Dtype::F8_E8M0 => Ok(DType::F8E8M0),
            dtype => Err(Error::UnsupportedSafeTensorDtype(dtype)),
        }
    }
}
